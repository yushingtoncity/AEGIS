"""Migration 0004 and the Phase 6 store functions: claimed orders, broker updates, fills.

``claim_order`` is the last check and the first record before an order is
sent; ``apply_broker_update`` moves a claimed order on as the broker reports.
Both are pinned here through the repository, and migration 0004's triggers
are pinned again with raw SQL, because they are the store's own guard under
whatever the Python code does. Temporary databases only; no network.
"""

import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from aegis.store import db as store_db
from aegis.store.db import connect, list_migrations, migrate, open_store
from aegis.store.errors import ClaimRefused, StoreError
from aegis.store.models import (
    TERMINAL_ORDER_STATUSES,
    Approval,
    ApprovalResponse,
    Broker,
    DecisionPurpose,
    Event,
    EventLevel,
    Fill,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PolicyDecision,
    PositionIntent,
    Proposal,
    TimeInForce,
    Verdict,
)
from aegis.store.repo import (
    apply_broker_update,
    claim_order,
    count_orders_submitted_between,
    get_decision,
    get_decision_approval,
    get_execution_orders,
    get_latest_verdict,
    get_live_approvals,
    get_verdicts_since,
    get_order,
    get_order_by_broker_id,
    get_proposal_order,
    get_proposal_trace,
    insert_proposal,
    record_approval,
    record_decision,
    record_fill,
    set_halt_until,
    set_kill_switch,
    upsert_order,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
DAY_START = datetime(2026, 7, 30, 4, 0, tzinfo=UTC)  # midnight in New York
DAY_END = DAY_START + timedelta(days=1)
CAP = 10


# --- helpers --------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "execution.db")
    yield connection
    connection.close()


def _proposal(proposal_id="prop-0001", *, instrument=Instrument.EQUITY, symbol="AAPL"):
    return Proposal(
        id=proposal_id,
        created_at=NOW - timedelta(minutes=2),
        cycle_id="cycle-0001",
        symbol=symbol,
        instrument=instrument,
        side=OrderSide.BUY,
        quantity=2.0,
        order_type=OrderType.LIMIT,
        limit_price=200.0,
        thesis="SYNTHETIC TEST DATA.",
        confidence=0.8,
        invalidation="SYNTHETIC TEST DATA.",
        raw_model_output="{}",
        model_name="synthetic-model-v0",
        prompt_version="test-prompt-1",
    )


def _decision(proposal_id, decision_id, verdict, purpose=DecisionPurpose.EVALUATE, minutes_ago=1):
    return PolicyDecision(
        id=decision_id,
        proposal_id=proposal_id,
        decided_at=NOW - timedelta(minutes=minutes_ago),
        verdict=verdict,
        rules_evaluated=[],
        purpose=purpose,
    )


def _seed(conn, proposal_id="prop-0001", verdict=Verdict.AUTO_EXECUTE, **proposal_kwargs):
    """A proposal, its verdict and the pre_submit re-evaluation. Returns the two decision ids."""
    insert_proposal(conn, _proposal(proposal_id, **proposal_kwargs))
    decision = record_decision(conn, _decision(proposal_id, f"dec-{proposal_id}", verdict))
    regate = record_decision(
        conn,
        _decision(proposal_id, f"pre-{proposal_id}", verdict, DecisionPurpose.PRE_SUBMIT, 0),
    )
    return decision.id, regate.id


def _approve(conn, decision_id, proposal_id="prop-0001", *, response=ApprovalResponse.APPROVED,
             expires_at=NOW + timedelta(minutes=15), approval_id=None):
    return record_approval(conn, Approval(
        id=approval_id or f"appr-{decision_id}",
        proposal_id=proposal_id,
        requested_at=NOW - timedelta(seconds=30),
        responded_at=NOW - timedelta(seconds=30),
        response=response,
        channel="cli",
        responder="operator",
        decision_id=decision_id,
        expires_at=expires_at,
    ))


def _order(proposal_id="prop-0001", *, decision_id=None, regate_decision_id=None, **overrides):
    fields = {
        "id": f"ord-{proposal_id}",
        "proposal_id": proposal_id,
        "client_order_id": f"aegis-{proposal_id}",
        "broker": Broker.PAPER,
        "status": OrderStatus.APPROVED,
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": 10.0,
        "limit_price": 200.0,
        "decision_id": decision_id or f"dec-{proposal_id}",
        "regate_decision_id": regate_decision_id or f"pre-{proposal_id}",
        "instrument": Instrument.EQUITY,
        "order_type": OrderType.LIMIT,
        "time_in_force": TimeInForce.DAY,
    }
    fields.update(overrides)
    return Order(**fields)


def _claim(conn, order=None, *, now=NOW, cap=CAP, event=None):
    return claim_order(
        conn, order or _order(), now=now, day_start=DAY_START, day_end=DAY_END,
        max_daily_trades=cap, event=event,
    )


def _counts(conn):
    tables = ("orders", "fills", "events", "approvals", "policy_decisions")
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}


def _legacy_order(proposal_id="prop-0001", n=1, **overrides):
    fields = {
        "id": f"legacy-{n}",
        "proposal_id": proposal_id,
        "client_order_id": f"legacy-{n}",
        "broker": Broker.PAPER,
        "status": OrderStatus.SUBMITTED,
        "submitted_at": NOW - timedelta(hours=1),
        "updated_at": NOW - timedelta(hours=1),
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": 1.0,
        "limit_price": 200.0,
    }
    fields.update(overrides)
    return Order(**fields)


def _refused(conn, order=None, **kwargs):
    before = _counts(conn)
    with pytest.raises(ClaimRefused) as info:
        _claim(conn, order, **kwargs)
    assert _counts(conn) == before  # nothing written
    assert not conn.in_transaction
    return info.value.reason


# --- migration 0004 on a store with history -----------------------------------


class TestMigration:
    def test_0004_on_a_populated_version_3_store_keeps_every_row(self, tmp_path):
        path = tmp_path / "v3.db"
        conn = connect(path)
        conn.execute(store_db._SCHEMA_VERSION_DDL)
        for version, name, file in list_migrations()[:3]:
            store_db._apply_migration(conn, version, name, file)
        insert_proposal(conn, _proposal())
        conn.execute(
            "INSERT INTO policy_decisions (id, proposal_id, decided_at, verdict, rules_evaluated)"
            " VALUES ('dec-old', 'prop-0001', ?, 'REJECT', '[]')", (NOW.isoformat(),),
        )
        conn.execute(
            "INSERT INTO approvals (id, proposal_id, requested_at, response, channel)"
            " VALUES ('appr-old', 'prop-0001', ?, 'approved', 'cli')", (NOW.isoformat(),),
        )
        for n in (1, 2):  # two pre-Phase 6 orders sharing a broker id: allowed then, allowed now
            conn.execute(
                "INSERT INTO orders (id, proposal_id, client_order_id, broker, broker_order_id, status,"
                " submitted_at, updated_at, symbol, side, quantity, limit_price)"
                " VALUES (?, 'prop-0001', ?, 'paper', 'same-broker-id', 'filled', ?, ?, 'AAPL', 'buy', 1, 200)",
                (f"legacy-{n}", f"legacy-{n}", NOW.isoformat(), NOW.isoformat()),
            )
        conn.execute(
            "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees)"
            " VALUES ('fill-old', 'legacy-1', ?, 0, 1, 0)", (NOW.isoformat(),),  # a zero price, once allowed
        )
        assert migrate(conn) == 4
        trace = get_proposal_trace(conn, "prop-0001")
        assert trace.decisions[0].purpose is DecisionPurpose.EVALUATE
        assert trace.approvals[0].decision_id is None and trace.approvals[0].expires_at is None
        assert [o.decision_id for o in trace.orders] == [None, None]
        assert {o.filled_quantity for o in trace.orders} == {0.0}
        assert trace.fills[0].broker_fill_id is None and trace.fills[0].fill_price == 0.0
        # a pre-Phase 6 order still moves through upsert_order exactly as before
        legacy = get_order(conn, "legacy-1")
        moved = upsert_order(conn, legacy.model_copy(update={"status": OrderStatus.CANCELLED}))
        assert moved.status is OrderStatus.CANCELLED
        conn.close()

    def test_0004_lands_whole_or_not_at_all(self, tmp_path, monkeypatch):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        for _, _, real in list_migrations():
            (migrations / real.name).write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        broken = migrations / "0004_execution.sql"
        broken.write_text(broken.read_text(encoding="utf-8") + "INSERT INTO no_such_table VALUES (1);\n")
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(tmp_path / "s.db")
        with pytest.raises(StoreError, match="apply migration for 0004_execution"):
            migrate(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(orders)")]
        assert "decision_id" not in columns and store_db.schema_version(conn) == 3
        triggers = conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall()
        assert [row[0] for row in triggers] == ["controls_keep_rows"]
        conn.close()

    def test_the_triggers_0004_adds(self, conn):
        rows = conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger' AND name != 'controls_keep_rows'"
            " ORDER BY name"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("approvals_answer_once", "approvals"),
            ("approvals_decision_link_fixed", "approvals"),
            ("approvals_decision_same_proposal", "approvals"),
            ("fills_execution_fixed", "fills"),
            ("fills_execution_kept", "fills"),
            ("fills_execution_price_positive", "fills"),
            ("orders_decision_link_fixed", "orders"),
            ("orders_execution_born_approved", "orders"),
            ("orders_execution_failed_unfilled", "orders"),
            ("orders_execution_filled_quantity", "orders"),
            ("orders_execution_identity_fixed", "orders"),
            ("orders_execution_kept", "orders"),
            ("orders_execution_status_forward", "orders"),
            ("orders_execution_terminal_frozen", "orders"),
        ]


# --- claim_order ----------------------------------------------------------------


class TestClaim:
    def test_an_auto_execute_order_is_claimed_approved_and_counted(self, conn):
        _seed(conn)
        event = Event(level=EventLevel.INFO, kind="order_claimed", message="claimed", occurred_at=NOW)
        stored = _claim(conn, event=event)
        assert stored.status is OrderStatus.APPROVED
        assert stored.submitted_at == NOW and stored.updated_at == NOW
        assert stored.broker_order_id is None and stored.filled_quantity == 0.0
        assert (stored.decision_id, stored.regate_decision_id, stored.approval_id) == (
            "dec-prop-0001", "pre-prop-0001", None,
        )
        assert stored.instrument is Instrument.EQUITY and stored.time_in_force is TimeInForce.DAY
        assert get_order(conn, "aegis-prop-0001") == stored == get_proposal_order(conn, "prop-0001")
        # it counts toward the daily cap from this instant, before any broker answer
        assert count_orders_submitted_between(conn, DAY_START, DAY_END) == 1
        kinds = [row[0] for row in conn.execute("SELECT kind FROM events")]
        assert kinds == ["order_claimed"]

    def test_the_argument_cannot_backdate_or_prefill_the_claim(self, conn):
        _seed(conn)
        sneaky = _order(submitted_at=NOW - timedelta(days=3), updated_at=NOW - timedelta(days=3))
        stored = _claim(conn, sneaky)
        assert stored.submitted_at == NOW and stored.updated_at == NOW

    def test_a_needs_approval_order_rests_on_an_unexpired_approval_of_that_decision(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        approval = _approve(conn, decision_id)
        stored = _claim(conn, _order(approval_id=approval.id))
        assert stored.approval_id == approval.id

    @pytest.mark.parametrize(
        ("setup", "expected"),
        [
            ("none", "needs approval, and the order names none"),
            ("rejected", "is rejected"),
            ("unanswered", "is unanswered"),
            ("expired", "has expired"),
            ("expires now", "has expired"),
            ("no expiry", "has no expiry"),
            ("other decision", "is not the approval of decision"),
            ("earlier decision, same proposal", "is not the approval of decision"),
            ("missing", "does not exist"),
        ],
    )
    def test_without_a_valid_approval_a_needs_approval_order_is_refused(self, conn, setup, expected):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        approval_id = None
        if setup == "rejected":
            approval_id = _approve(conn, decision_id, response=ApprovalResponse.REJECTED).id
        elif setup == "unanswered":
            approval_id = record_approval(conn, Approval(
                id="appr-pending", proposal_id="prop-0001", channel="cli", decision_id=decision_id,
                expires_at=NOW + timedelta(minutes=5),
            )).id
        elif setup == "expired":
            approval_id = _approve(conn, decision_id, expires_at=NOW - timedelta(seconds=1)).id
        elif setup == "expires now":
            approval_id = _approve(conn, decision_id, expires_at=NOW).id
        elif setup == "no expiry":
            approval_id = _approve(conn, decision_id, expires_at=None).id
        elif setup == "other decision":
            other, _ = _seed(conn, "prop-0002", verdict=Verdict.NEEDS_APPROVAL)
            approval_id = _approve(conn, other, "prop-0002").id
        elif setup == "earlier decision, same proposal":
            earlier = record_decision(
                conn, _decision("prop-0001", "dec-earlier", Verdict.NEEDS_APPROVAL, minutes_ago=5)
            )
            approval_id = _approve(conn, earlier.id).id
        elif setup == "missing":
            approval_id = "appr-nowhere"
        assert expected in _refused(conn, _order(approval_id=approval_id))

    @pytest.mark.parametrize("verdict", [Verdict.REJECT, Verdict.FLAG_ONLY])
    def test_no_order_follows_a_reject_or_a_flag(self, conn, verdict):
        _seed(conn, verdict=verdict)
        assert f"is {verdict.value}: no order may follow it" in _refused(conn)

    def test_the_links_must_be_what_they_say(self, conn):
        _seed(conn)
        _seed(conn, "prop-0002")
        cases = {
            "does not exist": _order(decision_id="dec-nowhere"),
            "is not a decision on proposal": _order(decision_id="dec-prop-0002"),
            "is a pre_submit decision, not a verdict": _order(decision_id="pre-prop-0001"),
            "pre_submit decision pre-nowhere does not exist": _order(regate_decision_id="pre-nowhere"),
            "is on another proposal": _order(regate_decision_id="pre-prop-0002"),
            "is not a pre_submit decision": _order(regate_decision_id="dec-prop-0001"),
            "rests on no approval": _order(approval_id="appr-x"),
        }
        for expected, order in cases.items():
            assert expected in _refused(conn, order), expected

    def test_the_kill_switch_refuses_the_claim(self, conn):
        _seed(conn)
        set_kill_switch(conn, True, now=NOW)
        assert _refused(conn) == "the kill switch is on"

    def test_a_halt_in_force_refuses_and_an_expired_one_does_not(self, conn):
        _seed(conn)
        set_halt_until(conn, NOW + timedelta(seconds=1), now=NOW)
        assert _refused(conn).startswith("trading is halted until 2026-07-30T15:00:01")
        set_halt_until(conn, NOW, now=NOW)  # it ends at this very instant
        assert _claim(conn).status is OrderStatus.APPROVED

    def test_a_halt_that_cannot_be_read_refuses(self, conn):
        _seed(conn)
        # a value with the timestamp's shape that is no timestamp: the CHECK lets it in
        conn.execute("UPDATE controls SET value = '2026-13-99T99:99:99' WHERE key = 'halt_until'")
        assert _refused(conn) == "cannot prove trading is not halted"

    def test_the_daily_cap_counts_claimed_submitted_and_legacy_orders_not_failed_ones(self, conn):
        for n in range(1, 4):
            _seed(conn, f"prop-000{n}")
        upsert_order(conn, _legacy_order())  # submitted an hour ago: a trade today
        upsert_order(conn, _legacy_order(n=2, status=OrderStatus.FAILED))  # never a trade
        upsert_order(conn, _legacy_order(n=3, submitted_at=DAY_START - timedelta(seconds=1)))  # yesterday
        upsert_order(conn, _legacy_order(n=4, submitted_at=DAY_END))  # tomorrow: the window is half-open
        _claim(conn, _order("prop-0001"), cap=3)
        _claim(conn, _order("prop-0002"), cap=3)  # 2 sent before it: under the cap of 3
        assert "3 orders sent today: at or above the cap of 3" in _refused(conn, _order("prop-0003"), cap=3)
        assert _claim(conn, _order("prop-0003"), cap=4).status is OrderStatus.APPROVED

    def test_a_cap_of_zero_admits_nothing(self, conn):
        _seed(conn)
        assert "at or above the cap of 0" in _refused(conn, cap=0)

    @pytest.mark.parametrize(
        ("now", "start", "end"),
        [
            (NOW, DAY_END, DAY_START),  # swapped: the count would always be 0
            (NOW + timedelta(days=1), DAY_START, DAY_END),  # the claim lands outside the window
            (DAY_START - timedelta(seconds=1), DAY_START, DAY_END),
            (DAY_END, DAY_START, DAY_END),  # half-open
        ],
    )
    def test_a_window_without_the_claim_in_it_is_a_caller_bug(self, conn, now, start, end):
        # Found in review: an inconsistent window counted nothing, so the cap admitted everything.
        _seed(conn)
        before = _counts(conn)
        with pytest.raises(StoreError, match="must contain now") as info:
            claim_order(conn, _order(), now=now, day_start=start, day_end=end, max_daily_trades=1)
        assert not isinstance(info.value, ClaimRefused)
        assert _counts(conn) == before
        assert _claim(conn, now=DAY_START).status is OrderStatus.APPROVED  # the start is inside

    @pytest.mark.parametrize(
        ("verdict", "recheck", "expected"),
        [
            (Verdict.AUTO_EXECUTE, Verdict.REJECT, "re-check pre-2 is REJECT"),
            (Verdict.AUTO_EXECUTE, Verdict.FLAG_ONLY, "re-check pre-2 is FLAG_ONLY"),
            (Verdict.AUTO_EXECUTE, Verdict.NEEDS_APPROVAL, "no longer goes without approval"),
            (Verdict.NEEDS_APPROVAL, Verdict.REJECT, "re-check pre-2 is REJECT"),
            (Verdict.NEEDS_APPROVAL, Verdict.FLAG_ONLY, "re-check pre-2 is FLAG_ONLY"),
        ],
    )
    def test_the_pre_submit_re_check_must_still_allow_the_order(self, conn, verdict, recheck, expected):
        decision_id, _ = _seed(conn, verdict=verdict)
        record_decision(conn, _decision("prop-0001", "pre-2", recheck, DecisionPurpose.PRE_SUBMIT, 0))
        approval_id = _approve(conn, decision_id).id if verdict is Verdict.NEEDS_APPROVAL else None
        assert expected in _refused(conn, _order(regate_decision_id="pre-2", approval_id=approval_id))

    @pytest.mark.parametrize("recheck", [Verdict.AUTO_EXECUTE, Verdict.NEEDS_APPROVAL])
    def test_an_approved_order_passes_a_re_check_that_allows_it(self, conn, recheck):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        record_decision(conn, _decision("prop-0001", "pre-2", recheck, DecisionPurpose.PRE_SUBMIT, 0))
        approval = _approve(conn, decision_id)
        stored = _claim(conn, _order(regate_decision_id="pre-2", approval_id=approval.id))
        assert stored.regate_decision_id == "pre-2"

    def test_an_approval_filed_under_another_proposal_authorises_nothing(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        _seed(conn, "prop-0002", verdict=Verdict.NEEDS_APPROVAL)
        with pytest.raises(StoreError, match="answers a decision on its own proposal"):
            _approve(conn, decision_id, "prop-0002")
        # and were such a row there anyway, the claim would not rest on it
        conn.execute("DROP TRIGGER approvals_decision_same_proposal")
        stray = _approve(conn, decision_id, "prop-0002")
        assert "is not the approval of decision" in _refused(conn, _order(approval_id=stray.id))

    def test_one_order_per_proposal_ever(self, conn):
        _seed(conn)
        first = _claim(conn)
        apply_broker_update(
            conn, first.client_order_id, status=OrderStatus.CANCELLED, synced_at=NOW,
            status_reason="test",
        )
        again = _order(client_order_id="aegis-prop-0001-again", id="ord-again")
        assert "already has order aegis-prop-0001" in _refused(conn, again)
        # and the store's own index says the same, under any writer
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute(
                "INSERT INTO orders (id, proposal_id, client_order_id, broker, status, submitted_at,"
                " updated_at, symbol, side, quantity, limit_price, decision_id)"
                " VALUES ('x', 'prop-0001', 'x', 'paper', 'approved', ?, ?, 'AAPL', 'buy', 1, 1,"
                " 'dec-prop-0001')", (NOW.isoformat(), NOW.isoformat()),
            )

    def test_an_option_order_names_its_intent(self, conn):
        _seed(conn, instrument=Instrument.OPTION, symbol="SPY261016C00640000")
        option = _order(
            instrument=Instrument.OPTION, symbol="SPY261016C00640000", quantity=1.0, limit_price=8.0,
            position_intent=PositionIntent.BUY_TO_OPEN,
        )
        assert _claim(conn, option).position_intent is PositionIntent.BUY_TO_OPEN

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"decision_id": None, "regate_decision_id": None}, "names its decision"),
            ({"status": OrderStatus.SUBMITTED}, "born approved"),
            ({"broker_order_id": "early"}, "no broker id"),
            ({"filled_quantity": 1.0}, "nothing filled"),
            ({"order_type": OrderType.MARKET, "limit_price": None}, "a limit order"),
            ({"instrument": None}, "names its instrument"),
            ({"time_in_force": None}, "time in force"),
            ({"limit_price": None}, "positive limit price"),
            ({"limit_price": 0.0}, "positive limit price"),
            ({"limit_price": -1.0}, "positive limit price"),
            ({"position_intent": PositionIntent.BUY_TO_OPEN}, "position intent"),
        ],
    )
    def test_a_malformed_claim_is_a_caller_bug_not_a_refusal(self, conn, overrides, message):
        _seed(conn)
        order = _order().model_copy(update=overrides)
        if overrides.get("decision_id", "x") is None:
            order = order.model_copy(update={"decision_id": None, "regate_decision_id": None})
        before = _counts(conn)
        with pytest.raises(StoreError, match=message) as info:
            _claim(conn, order)
        assert not isinstance(info.value, ClaimRefused)
        assert _counts(conn) == before

    def test_max_daily_trades_must_be_an_int(self, conn):
        _seed(conn)
        for bad in (True, 3.0, "3", None):
            with pytest.raises(StoreError, match="max_daily_trades must be an int"):
                _claim(conn, cap=bad)

    def test_a_claim_refused_is_still_a_store_error_one_line(self, conn):
        _seed(conn)
        set_kill_switch(conn, True, now=NOW)
        with pytest.raises(StoreError) as info:
            _claim(conn)
        assert str(info.value) == (
            "store operation failed: claim order (refused: the kill switch is on) for aegis-prop-0001"
        )

    def test_an_event_that_cannot_be_written_rolls_the_claim_back(self, conn):
        _seed(conn)
        bad = Event(level=EventLevel.INFO, kind="k", message="m", payload={"x": {1, 2}})
        before = _counts(conn)
        with pytest.raises(StoreError, match="claim order"):
            _claim(conn, event=bad)
        assert _counts(conn) == before


# --- apply_broker_update ------------------------------------------------------


def _claimed(conn, **overrides):
    _seed(conn)
    return _claim(conn, _order(**overrides))


def _update(conn, status, *, at=1, **kwargs):
    return apply_broker_update(
        conn, "aegis-prop-0001", status=status, synced_at=NOW + timedelta(seconds=at), **kwargs,
    )


def _fills(conn):
    return [
        tuple(row) for row in conn.execute(
            "SELECT fill_quantity, fill_price, broker_fill_id FROM fills ORDER BY rowid"
        )
    ]


class TestBrokerUpdate:
    def test_accepted_then_two_fills_then_replayed(self, conn):
        _claimed(conn)
        order, fill = _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1", broker_status="accepted")
        assert fill is None and order.status is OrderStatus.SUBMITTED and order.broker_order_id == "b-1"
        assert order.last_synced_at == NOW + timedelta(seconds=1) and order.submitted_at == NOW
        order, fill = _update(
            conn, OrderStatus.PARTIALLY_FILLED, at=2, filled_quantity=3.0, avg_fill_price=200.0,
            broker_status="partially_filled",
        )
        assert (fill.fill_quantity, fill.fill_price, fill.broker_fill_id) == (3.0, 200.0, "b-1:3.0")
        assert order.filled_quantity == 3.0 and order.avg_fill_price == 200.0
        order, fill = _update(
            conn, OrderStatus.FILLED, at=3, filled_quantity=10.0, avg_fill_price=200.7,
            broker_status="filled",
        )
        # the second fill is priced so the fills add up to the broker's average: (2007 - 600) / 7
        assert fill.fill_quantity == 7.0 and fill.fill_price == pytest.approx(201.0)
        assert fill.broker_fill_id == "b-1:10.0" and order.status is OrderStatus.FILLED
        total = sum(q * p for q, p, _ in _fills(conn))
        assert total == pytest.approx(10.0 * 200.7)
        # a replayed sync records nothing twice and writes nothing
        before = _counts(conn)
        again, fill = _update(
            conn, OrderStatus.FILLED, at=9, filled_quantity=10.0, avg_fill_price=200.7, broker_status="filled",
        )
        assert fill is None and again == order and _counts(conn) == before
        assert get_order_by_broker_id(conn, "b-1") == order

    def test_a_fill_in_one_step_from_approved(self, conn):
        _claimed(conn)
        order, fill = _update(
            conn, OrderStatus.FILLED, broker_order_id="b-1", filled_quantity=10.0, avg_fill_price=199.5,
        )
        assert order.status is OrderStatus.FILLED and fill.fill_price == 199.5 and fill.fill_quantity == 10

    def test_a_partial_fill_then_a_cancel_keeps_the_fill(self, conn):
        _claimed(conn)
        _update(conn, OrderStatus.PARTIALLY_FILLED, broker_order_id="b-1", filled_quantity=4.0, avg_fill_price=5.0)
        order, fill = _update(conn, OrderStatus.CANCELLED, at=2, status_reason="day order expired")
        assert fill is None and order.status is OrderStatus.CANCELLED and order.filled_quantity == 4.0
        assert order.status_reason == "day order expired"

    def test_an_average_without_a_new_fill_leaves_the_stored_average(self, conn):
        # Found by the sweep: the average must always match the fills.
        _claimed(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1", avg_fill_price=7.0)
        assert get_order(conn, "aegis-prop-0001").avg_fill_price is None
        _update(conn, OrderStatus.PARTIALLY_FILLED, at=2, filled_quantity=4.0, avg_fill_price=5.0)
        order, fill = _update(conn, OrderStatus.CANCELLED, at=3, avg_fill_price=1.0)
        assert fill is None and order.avg_fill_price == 5.0
        assert _fills(conn) == [(4.0, 5.0, "b-1:4.0")]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"filled_quantity": 11.0, "avg_fill_price": 1.0}, "above the order's 10"),
            ({"filled_quantity": -1.0, "avg_fill_price": 1.0}, "is not a quantity"),
            ({"filled_quantity": float("nan"), "avg_fill_price": 1.0}, "is not a quantity"),
            ({"filled_quantity": 2.0}, "positive average fill price"),
            ({"filled_quantity": 2.0, "avg_fill_price": 0.0}, "positive average fill price"),
            ({"filled_quantity": 2.0, "avg_fill_price": float("inf")}, "positive average fill price"),
        ],
    )
    def test_figures_that_do_not_add_up_are_refused_and_nothing_lands(self, conn, kwargs, message):
        _claimed(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1")
        before = (_counts(conn), get_order(conn, "aegis-prop-0001"))
        with pytest.raises(StoreError, match=message):
            _update(conn, OrderStatus.PARTIALLY_FILLED, at=2, **kwargs)
        assert (_counts(conn), get_order(conn, "aegis-prop-0001")) == before

    def test_a_shrinking_total_or_a_new_average_that_cannot_be_right_is_refused(self, conn):
        _claimed(conn)
        _update(conn, OrderStatus.PARTIALLY_FILLED, broker_order_id="b-1", filled_quantity=5.0, avg_fill_price=10.0)
        with pytest.raises(StoreError, match="below the 5 recorded"):
            _update(conn, OrderStatus.PARTIALLY_FILLED, at=2, filled_quantity=4.0, avg_fill_price=10.0)
        # 5 more at an average of 4 would mean the new five cost -2 each
        with pytest.raises(StoreError, match="does not account for the new fill"):
            _update(conn, OrderStatus.FILLED, at=2, filled_quantity=10.0, avg_fill_price=4.0)
        assert len(_fills(conn)) == 1

    def test_filled_means_filled_in_full(self, conn):
        _claimed(conn)
        with pytest.raises(StoreError, match="a filled order is filled in full"):
            _update(conn, OrderStatus.FILLED, broker_order_id="b-1", filled_quantity=9.0, avg_fill_price=1.0)

    def test_a_fill_needs_the_brokers_order_id(self, conn):
        _claimed(conn)
        with pytest.raises(StoreError, match="needs the broker's order id"):
            _update(conn, OrderStatus.PARTIALLY_FILLED, filled_quantity=1.0, avg_fill_price=1.0)

    def test_status_only_moves_forward(self, conn):
        _claimed(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1")
        for backward in (OrderStatus.APPROVED, OrderStatus.PROPOSED, OrderStatus.GATED):
            with pytest.raises(StoreError, match="only moves forward"):
                _update(conn, backward, at=2)
        _update(conn, OrderStatus.PARTIALLY_FILLED, at=3, filled_quantity=1.0, avg_fill_price=1.0)
        with pytest.raises(StoreError, match="only moves forward"):
            _update(conn, OrderStatus.SUBMITTED, at=4)
        with pytest.raises(StoreError, match="filled in part did not fail"):
            _update(conn, OrderStatus.FAILED, at=4)

    @pytest.mark.parametrize("first", [OrderStatus.APPROVED, OrderStatus.SUBMITTED])
    def test_an_order_with_anything_filled_never_fails(self, conn, first):
        # Found in review: from approved or submitted, a failed status would
        # have kept the fills and dropped the order from the daily trade cap.
        _claimed(conn)
        if first is OrderStatus.SUBMITTED:
            _update(conn, first, broker_order_id="b-1", filled_quantity=4.0, avg_fill_price=200.0)
        before = (get_order(conn, "aegis-prop-0001"), _fills(conn))
        with pytest.raises(StoreError, match="filled in part did not fail"):
            _update(conn, OrderStatus.FAILED, at=2, broker_order_id="b-1", filled_quantity=4.0,
                    avg_fill_price=200.0)
        assert (get_order(conn, "aegis-prop-0001"), _fills(conn)) == before
        with pytest.raises(sqlite3.IntegrityError, match="anything filled never fails"):
            _raw(conn, "UPDATE orders SET status = 'failed', filled_quantity = 4"
                       " WHERE client_order_id = 'aegis-prop-0001'")

    def test_an_order_with_nothing_filled_may_fail(self, conn):
        _claimed(conn)
        failed, _ = _update(conn, OrderStatus.FAILED, status_reason="rejected: insufficient buying power")
        assert failed.status is OrderStatus.FAILED and failed.filled_quantity == 0

    def test_a_fill_id_is_exact_however_close_the_totals(self, conn):
        # Found in review: with .10g, 12.123456788 and 12.123456789 shared a
        # name and the final fill could never be recorded.
        _claimed(conn, quantity=12.123456789)
        _update(conn, OrderStatus.PARTIALLY_FILLED, broker_order_id="b-1", filled_quantity=12.123456788,
                avg_fill_price=10.0)
        order, fill = _update(conn, OrderStatus.FILLED, at=2, filled_quantity=12.123456789,
                              avg_fill_price=10.0)
        assert order.status is OrderStatus.FILLED and fill.broker_fill_id == "b-1:12.123456789"
        assert [row[2] for row in _fills(conn)] == ["b-1:12.123456788", "b-1:12.123456789"]

    @pytest.mark.parametrize("terminal", sorted(TERMINAL_ORDER_STATUSES, key=lambda s: s.value))
    def test_a_terminal_order_is_frozen_but_a_replay_is_a_no_op(self, conn, terminal):
        _claimed(conn)
        filled = {"filled_quantity": 10.0, "avg_fill_price": 2.0} if terminal is OrderStatus.FILLED else {}
        order, _ = _update(conn, terminal, broker_order_id="b-1", **filled)
        event = Event(level=EventLevel.CRITICAL, kind="k", message="m", occurred_at=NOW)
        assert _update(conn, terminal, at=5, event=event, **filled) == (order, None)
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0  # a replay writes nothing
        for change in ({"broker_status": "something new"}, {"status_reason": "late news"}):
            with pytest.raises(StoreError, match="a filled, cancelled or failed order never changes"):
                _update(conn, terminal, at=6, **change, **filled)
        assert get_order(conn, "aegis-prop-0001") == order

    def test_the_broker_id_is_set_once(self, conn):
        _claimed(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1")
        with pytest.raises(StoreError, match="broker id is set once"):
            _update(conn, OrderStatus.SUBMITTED, at=2, broker_order_id="b-2")
        assert _update(conn, OrderStatus.SUBMITTED, at=3)[0].broker_order_id == "b-1"

    def test_only_execution_era_orders_take_broker_updates(self, conn):
        _seed(conn)
        upsert_order(conn, _legacy_order())
        with pytest.raises(StoreError, match="predates Phase 6"):
            apply_broker_update(conn, "legacy-1", status=OrderStatus.FILLED, synced_at=NOW)
        with pytest.raises(StoreError, match="no such order"):
            apply_broker_update(conn, "nope", status=OrderStatus.FILLED, synced_at=NOW)
        with pytest.raises(StoreError, match="status must be an OrderStatus"):
            apply_broker_update(conn, "legacy-1", status="filled", synced_at=NOW)

    def test_the_event_lands_with_the_update(self, conn):
        _claimed(conn)
        event = Event(level=EventLevel.INFO, kind="order_filled", message="m", occurred_at=NOW)
        _update(conn, OrderStatus.FILLED, broker_order_id="b-1", filled_quantity=10.0, avg_fill_price=1.0,
                event=event)
        assert [row[0] for row in conn.execute("SELECT kind FROM events")] == ["order_filled"]

    def test_upsert_order_refuses_execution_era_orders(self, conn):
        claimed = _claimed(conn)
        with pytest.raises(StoreError, match="written by claim_order only"):
            upsert_order(conn, claimed)
        bare = claimed.model_copy(update={"decision_id": None, "regate_decision_id": None})
        with pytest.raises(StoreError, match="changes through apply_broker_update only"):
            upsert_order(conn, bare)
        assert get_order(conn, claimed.client_order_id) == claimed

    def test_record_fill_refuses_execution_era_orders(self, conn):
        claimed = _claimed(conn)
        fill = Fill(order_id=claimed.id, filled_at=NOW, fill_price=200.0, fill_quantity=3.0)
        with pytest.raises(StoreError, match="written by apply_broker_update only"):
            record_fill(conn, fill)
        assert _fills(conn) == []
        legacy = upsert_order(conn, _legacy_order(proposal_id="prop-0001"))
        record_fill(conn, fill.model_copy(update={"order_id": legacy.id}))  # as before
        assert len(_fills(conn)) == 1


# --- reads ----------------------------------------------------------------------


class TestReads:
    def test_decisions_carry_their_purpose(self, conn):
        decision_id, regate_id = _seed(conn)
        assert get_decision(conn, decision_id).purpose is DecisionPurpose.EVALUATE
        assert get_decision(conn, regate_id).purpose is DecisionPurpose.PRE_SUBMIT
        assert get_decision(conn, "nope") is None

    def test_a_decision_has_at_most_one_approval(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        assert get_decision_approval(conn, decision_id) is None
        approval = _approve(conn, decision_id)
        assert get_decision_approval(conn, decision_id) == approval
        with pytest.raises(StoreError, match="UNIQUE"):
            _approve(conn, decision_id, approval_id="appr-second")

    def test_no_order_reads_none(self, conn):
        assert get_proposal_order(conn, "prop-0001") is None
        assert get_order_by_broker_id(conn, "b-1") is None

    def test_the_broker_id_lookup_finds_the_claimed_order_not_a_legacy_one(self, conn):
        _seed(conn)
        upsert_order(conn, _legacy_order(broker_order_id="b-1"))  # written first
        claimed = _claim(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1")
        assert get_order_by_broker_id(conn, "b-1").id == claimed.id


# --- the triggers, under any writer ---------------------------------------------


def _raw(conn, sql, *params):
    conn.execute(sql, params)


class TestTriggers:
    def test_a_claimed_order_is_born_approved_with_submitted_at(self, conn):
        _seed(conn)
        base = (
            "INSERT INTO orders (id, proposal_id, client_order_id, broker, status, submitted_at, updated_at,"
            " symbol, side, quantity, limit_price, decision_id, broker_order_id, filled_quantity)"
            " VALUES ('x', 'prop-0001', 'x', 'paper', ?, ?, ?, 'AAPL', 'buy', 1, 1, 'dec-prop-0001', ?, ?)"
        )
        ts = NOW.isoformat()
        for status, submitted, broker_id, filled in (
            ("submitted", ts, None, 0), ("approved", None, None, 0), ("approved", ts, "b", 0),
            ("approved", ts, None, 1),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="born approved"):
                _raw(conn, base, status, submitted, ts, broker_id, filled)

    def test_status_forward_terminal_frozen_identity_fixed(self, conn):
        claimed = _claimed(conn)
        key = claimed.client_order_id
        with pytest.raises(sqlite3.IntegrityError, match="only moves forward"):
            _raw(conn, "UPDATE orders SET status = 'proposed' WHERE client_order_id = ?", key)
        for column, value in (
            ("quantity", 99), ("limit_price", 1), ("symbol", "TSLA"), ("side", "sell"),
            ("approval_id", "apr-other"), ("regate_decision_id", "dec-prop-0001"), ("submitted_at", "x"),
            ("instrument", "option"), ("time_in_force", None),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="identity never changes"):
                conn.execute(f"UPDATE orders SET {column} = ? WHERE client_order_id = ?", (value, key))
        with pytest.raises(sqlite3.IntegrityError, match="decision_id never changes"):
            _raw(conn, "UPDATE orders SET decision_id = NULL WHERE client_order_id = ?", key)
        with pytest.raises(sqlite3.IntegrityError, match="filled quantity only grows"):
            _raw(conn, "UPDATE orders SET filled_quantity = 11 WHERE client_order_id = ?", key)
        _raw(conn, "UPDATE orders SET status = 'cancelled' WHERE client_order_id = ?", key)
        with pytest.raises(sqlite3.IntegrityError, match="never changes"):
            _raw(conn, "UPDATE orders SET status_reason = 'x' WHERE client_order_id = ?", key)

    def test_a_legacy_order_cannot_be_given_a_decision(self, conn):
        _seed(conn)
        upsert_order(conn, _legacy_order())
        with pytest.raises(sqlite3.IntegrityError, match="decision_id never changes"):
            _raw(conn, "UPDATE orders SET decision_id = 'dec-prop-0001' WHERE id = 'legacy-1'")
        # and a legacy row still moves any way it always could
        _raw(conn, "UPDATE orders SET status = 'proposed' WHERE id = 'legacy-1'")

    def test_a_claimed_orders_fill_has_a_positive_price(self, conn):
        claimed = _claimed(conn)
        with pytest.raises(sqlite3.IntegrityError, match="positive price"):
            _raw(conn, "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees)"
                       " VALUES ('f', ?, ?, 0, 1, 0)", claimed.id, NOW.isoformat())

    def test_a_claimed_order_and_its_fills_are_never_deleted_or_rewritten(self, conn):
        claimed = _claimed(conn)
        _update(conn, OrderStatus.PARTIALLY_FILLED, broker_order_id="b-1", filled_quantity=4.0,
                avg_fill_price=5.0)
        _update(conn, OrderStatus.CANCELLED, at=2)
        for sql, message in (
            ("UPDATE fills SET fill_price = 6 WHERE order_id = ?", "fill of a claimed order never changes"),
            ("UPDATE fills SET fill_quantity = 1 WHERE order_id = ?", "fill of a claimed order never changes"),
            ("DELETE FROM fills WHERE order_id = ?", "fill of a claimed order never changes"),
            ("DELETE FROM orders WHERE id = ?", "a claimed order is never deleted"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match=message):
                _raw(conn, sql, claimed.id)
        assert _fills(conn) == [(4.0, 5.0, "b-1:4.0")]
        # rows written before Phase 6 can still be removed as before
        legacy = upsert_order(conn, _legacy_order(proposal_id="prop-0001"))
        record_fill(conn, Fill(order_id=legacy.id, filled_at=NOW, fill_price=1.0, fill_quantity=1.0))
        _raw(conn, "DELETE FROM fills WHERE order_id = ?", legacy.id)
        _raw(conn, "DELETE FROM orders WHERE id = ?", legacy.id)

    def test_a_broker_fill_id_names_one_fill(self, conn):
        claimed = _claimed(conn)
        sql = ("INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees, broker_fill_id)"
               " VALUES (?, ?, ?, 1, 1, 0, 'b-1:1')")
        _raw(conn, sql, "f1", claimed.id, NOW.isoformat())
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            _raw(conn, sql, "f2", claimed.id, NOW.isoformat())

    def test_an_approval_is_answered_once_and_its_decision_and_expiry_are_fixed(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        approval = _approve(conn, decision_id)
        for sql in (
            "UPDATE approvals SET response = 'rejected' WHERE id = ?",
            "UPDATE approvals SET responder = 'someone else' WHERE id = ?",
            "UPDATE approvals SET note = 'changed my mind' WHERE id = ?",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="answered once"):
                _raw(conn, sql, approval.id)
        for sql in (
            "UPDATE approvals SET decision_id = NULL WHERE id = ?",
            "UPDATE approvals SET expires_at = '2099-01-01T00:00:00+00:00' WHERE id = ?",
            "UPDATE approvals SET proposal_id = 'prop-0002' WHERE id = ?",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="decision, proposal and expiry never change"):
                _raw(conn, sql, approval.id)
        # the same answer recorded again is not a change
        assert record_approval(conn, approval) == approval
        with pytest.raises(StoreError, match="answered once"):
            record_approval(conn, approval.model_copy(update={"response": ApprovalResponse.REJECTED}))

    def test_a_pending_approval_is_answered_through_record_approval(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        pending = record_approval(conn, Approval(
            id="appr-1", proposal_id="prop-0001", channel="cli", decision_id=decision_id,
            requested_at=NOW, expires_at=NOW + timedelta(minutes=15),
        ))
        answered = record_approval(conn, pending.model_copy(update={
            "response": ApprovalResponse.APPROVED, "responded_at": NOW, "responder": "op", "note": "ok",
        }))
        assert answered.response is ApprovalResponse.APPROVED and answered.note == "ok"
        assert answered.expires_at == pending.expires_at

    def test_the_broker_order_id_is_unique_among_claimed_orders(self, conn):
        _claimed(conn)
        _update(conn, OrderStatus.SUBMITTED, broker_order_id="b-1")
        _seed(conn, "prop-0002")
        _claim(conn, _order("prop-0002"))
        with pytest.raises(StoreError, match="UNIQUE"):
            apply_broker_update(conn, "aegis-prop-0002", status=OrderStatus.SUBMITTED, synced_at=NOW,
                                broker_order_id="b-1")


# --- a seeded sweep of broker histories -----------------------------------------

SWEEP_SEED = 20261006
SWEEP_CASES = 500
_STATUSES = [s for s in OrderStatus]


def _random_step(rng, order_quantity, filled):
    """One broker report, as often nonsense as sense: any status, a cumulative total that may
    grow, stall, shrink or overshoot, and an average price that may be missing or absurd."""
    status = rng.choice(_STATUSES)
    total = rng.choice((
        filled, filled, min(order_quantity, filled + rng.choice((1.0, 2.0, 3.0))), order_quantity,
        max(0.0, filled - 1.0), order_quantity + 1.0,
    ))
    price = rng.choice((None, 0.0, -1.0, float("inf"), 1.0, 5.0, 10.0, 10.0, 12.5))
    broker_id = rng.choice(("b-1", "b-1", "b-1", None, "b-2"))
    return status, total, price, broker_id


class TestSweep:
    def test_no_broker_history_breaks_the_stores_rules(self, tmp_path):
        """Whatever the broker reports, in whatever order: every update lands whole or not at all;
        the status only moves forward and a terminal order never changes; the fills add up to the
        filled quantity, never past the order's; their value is the broker's average times the
        filled quantity; and the broker id, once set, stays."""
        rng = random.Random(SWEEP_SEED)
        conn = open_store(tmp_path / "sweep.db")
        order_of = {s: i for i, s in enumerate(
            [OrderStatus.APPROVED, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        )}
        landed = refused = terminal_seen = 0
        try:
            for case in range(SWEEP_CASES):
                proposal_id = f"prop-{case:04d}"
                _seed(conn, proposal_id)
                quantity = float(rng.choice((1, 3, 10)))
                claimed = _claim(conn, _order(
                    proposal_id, quantity=quantity, client_order_id=f"c-{case}", id=f"o-{case}",
                ), cap=SWEEP_CASES + 1)
                key = claimed.client_order_id
                for step in range(rng.randint(1, 8)):
                    before = get_order(conn, key)
                    fills_before = conn.execute(
                        "SELECT COUNT(*) FROM fills WHERE order_id = ?", (claimed.id,)
                    ).fetchone()[0]
                    status, total, price, broker_id = _random_step(rng, quantity, before.filled_quantity)
                    broker_id = None if broker_id is None else f"{broker_id}-{case}"
                    try:
                        after, fill = apply_broker_update(
                            conn, key, status=status, synced_at=NOW + timedelta(seconds=step),
                            broker_order_id=broker_id, filled_quantity=total, avg_fill_price=price,
                        )
                    except StoreError:
                        refused += 1
                        assert get_order(conn, key) == before  # nothing landed
                        assert conn.execute(
                            "SELECT COUNT(*) FROM fills WHERE order_id = ?", (claimed.id,)
                        ).fetchone()[0] == fills_before
                        assert not conn.in_transaction
                        continue
                    landed += 1
                    if before.status in TERMINAL_ORDER_STATUSES:
                        terminal_seen += 1
                        assert after == before and fill is None
                    if before.status in order_of and after.status in order_of:
                        assert order_of[after.status] >= order_of[before.status]
                    assert after.status not in (OrderStatus.PROPOSED, OrderStatus.GATED)
                    if before.broker_order_id is not None:
                        assert after.broker_order_id == before.broker_order_id
                    assert before.filled_quantity <= after.filled_quantity <= quantity
                    if after.status is OrderStatus.FILLED:
                        assert after.filled_quantity == quantity
                rows = conn.execute(
                    "SELECT fill_quantity, fill_price FROM fills WHERE order_id = ?", (claimed.id,)
                ).fetchall()
                final = get_order(conn, key)
                assert sum(q for q, _ in rows) == pytest.approx(final.filled_quantity)
                if final.filled_quantity:
                    value = sum(q * p for q, p in rows)
                    assert value == pytest.approx(final.avg_fill_price * final.filled_quantity)
                    assert all(p > 0 for _, p in rows)
        finally:
            conn.close()
        assert landed >= 300 and refused >= 300 and terminal_seen >= 20, (landed, refused, terminal_seen)


# --- the reads Phase 6's dispatcher uses ------------------------------------------


class TestDispatchReads:
    def test_execution_orders_lists_claimed_orders_only(self, conn):
        _seed(conn)
        _seed(conn, "prop-0002")
        upsert_order(conn, _legacy_order())
        first = _claim(conn)
        second = _claim(conn, _order("prop-0002"))
        assert [o.id for o in get_execution_orders(conn)] == [first.id, second.id]
        apply_broker_update(conn, second.client_order_id, status=OrderStatus.CANCELLED,
                            synced_at=NOW, status_reason="test")
        assert [o.id for o in get_execution_orders(conn)] == [first.id]
        assert [o.id for o in get_execution_orders(conn, open_only=False)] == [first.id, second.id]

    def test_the_latest_verdict_ignores_pre_submit_checks(self, conn):
        _seed(conn)
        assert get_latest_verdict(conn, "prop-0001").id == "dec-prop-0001"
        record_decision(conn, _decision("prop-0001", "dec-later", Verdict.REJECT, minutes_ago=0))
        record_decision(conn, _decision("prop-0001", "pre-later", Verdict.AUTO_EXECUTE,
                                        DecisionPurpose.PRE_SUBMIT, minutes_ago=-1))
        assert get_latest_verdict(conn, "prop-0001").id == "dec-later"
        assert get_latest_verdict(conn, "prop-nowhere") is None

    def test_verdicts_since_by_verdict_and_bound(self, conn):
        _seed(conn)  # dec-prop-0001: AUTO_EXECUTE, a minute ago
        _seed(conn, "prop-0002", verdict=Verdict.NEEDS_APPROVAL)
        record_decision(conn, _decision("prop-0002", "dec-old", Verdict.NEEDS_APPROVAL, minutes_ago=90))
        since = NOW - timedelta(minutes=1)
        assert [d.id for d in get_verdicts_since(conn, since, Verdict.AUTO_EXECUTE)] == ["dec-prop-0001"]
        assert [d.id for d in get_verdicts_since(conn, since, Verdict.NEEDS_APPROVAL)] == ["dec-prop-0002"]
        assert [d.id for d in get_verdicts_since(conn, since + timedelta(seconds=1), Verdict.AUTO_EXECUTE)] == []
        hours = [d.id for d in get_verdicts_since(conn, NOW - timedelta(hours=2), Verdict.NEEDS_APPROVAL)]
        assert hours == ["dec-prop-0002", "dec-old"]  # newest first; pre_submit rows never listed

    def test_live_approvals_are_approved_and_unexpired(self, conn):
        decision_id, _ = _seed(conn, verdict=Verdict.NEEDS_APPROVAL)
        other, _ = _seed(conn, "prop-0002", verdict=Verdict.NEEDS_APPROVAL)
        third, _ = _seed(conn, "prop-0003", verdict=Verdict.NEEDS_APPROVAL)
        live = _approve(conn, decision_id)  # expires NOW + 15 min
        _approve(conn, other, "prop-0002", response=ApprovalResponse.REJECTED)
        _approve(conn, third, "prop-0003", expires_at=NOW)
        assert [a.id for a in get_live_approvals(conn, NOW)] == [live.id]
        assert get_live_approvals(conn, NOW + timedelta(minutes=15)) == []

    @pytest.mark.parametrize("bad", ["2026-07-30", None, 5])
    def test_bounds_must_be_instants(self, conn, bad):
        with pytest.raises(StoreError):
            get_verdicts_since(conn, bad, Verdict.AUTO_EXECUTE)
        with pytest.raises(StoreError):
            get_live_approvals(conn, bad)
