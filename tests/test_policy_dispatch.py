"""The dispatcher: from a recorded verdict to one order at the broker, and
the store following it there. docs/phase6/SPEC_PHASE6.md is the design.

Everything runs against a real temporary store and the real engine; the
contexts come from ``policy_factories`` (no network) and the broker is a
scripted stand-in that implements the Executor interface. One test drives
the real ``PaperExecutor`` over a scripted client, end to end, so the 6b
guard is proven to accept what this module claims.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from policy_factories import (
    NOW,
    OPTIONS_BUYING_POWER,
    WATCHLIST,
    call_debit_spread,
    context_for,
    long_call,
    make_context,
    make_option,
    make_position,
    make_proposal,
)

from aegis.config import AegisConfig, BrokerConfig, RiskLimits
from aegis.execution.base import Executor
from aegis.execution.models import ExecutionError, ExecutionOutcome, OrderReceipt
from aegis.policy import dispatch
from aegis.policy.context import trading_day, wall_clock
from aegis.policy.engine import evaluate
from aegis.policy.errors import PlacementError
from aegis.store import (
    ApprovalResponse,
    Controls,
    DecisionPurpose,
    EventLevel,
    Instrument,
    OrderSide,
    OrderStatus,
    OrderType,
    PolicyDecision,
    PositionIntent,
    Verdict,
    count_orders_submitted_between,
    get_controls,
    get_decision,
    get_decision_approval,
    get_execution_orders,
    get_order,
    get_proposal_order,
    insert_proposal,
    open_store,
    record_decision,
    set_halt_until,
    set_kill_switch,
)

UTC = timezone.utc
LATER = NOW + timedelta(seconds=10)
CONFIG = AegisConfig(watchlist=list(WATCHLIST), broker=BrokerConfig(enabled=True))
OFF = AegisConfig(watchlist=list(WATCHLIST))  # broker.enabled false, as shipped


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "dispatch.db")
    yield connection
    connection.close()


def _snapshot(conn):
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {t: conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in tables}


def _events(conn):
    return [row[0] for row in conn.execute("SELECT kind FROM events ORDER BY rowid")]


def _judged(conn, proposal=None, context=None):
    """The proposal stored and judged by the real engine at NOW."""
    proposal = proposal or make_proposal()
    insert_proposal(conn, proposal.proposal, proposal.legs)
    decision = evaluate(proposal, context or context_for(proposal), conn)
    return proposal, decision


def _builder(**overrides):
    """A context builder for the re-check: the factory's market at LATER."""

    def build(conn, proposal, *, config=None):
        return context_for(proposal, **{"now": LATER, **overrides})

    return build


# --- the broker stand-in --------------------------------------------------------


class FakeBroker(Executor):
    """A broker that keeps a book of orders and answers from it, unless a
    call is scripted to fail. Records every call."""

    def __init__(self, clock=lambda: LATER):
        self.clock = clock
        self.book: dict[str, OrderReceipt] = {}
        self.calls: list[tuple] = []
        self.send: list = []  # per submit: an ExecutionError (raised) or ("land", error)
        self.lookups: list = []  # per get_order: an ExecutionError, or a value to return
        self.cancels: list = []  # per cancel: an ExecutionError
        self.listing_error: ExecutionError | None = None
        self.foreign: list[OrderReceipt] = []

    def receipt(self, client_order_id, symbol="AAPL", side=OrderSide.BUY, quantity="2", **changes):
        fields = {
            "broker_order_id": str(uuid.uuid4()), "client_order_id": client_order_id,
            "symbol": symbol, "side": side, "quantity": Decimal(quantity),
            "broker_status": "accepted", "status": OrderStatus.SUBMITTED,
            "filled_quantity": Decimal("0"), "submitted_at": self.clock(),
            "fetched_at": self.clock(),
        }
        fields.update(changes)
        return OrderReceipt(**fields)

    def fill(self, client_order_id, filled, average, word="partially_filled"):
        status = OrderStatus.FILLED if word == "filled" else OrderStatus.PARTIALLY_FILLED
        self.book[client_order_id] = self.book[client_order_id].model_copy(update={
            "filled_quantity": Decimal(filled), "avg_fill_price": Decimal(average),
            "broker_status": word, "status": status, "fetched_at": self.clock(),
        })

    def submit_order(self, order):
        self.calls.append(("submit", order))
        script = self.send.pop(0) if self.send else None
        if isinstance(script, BaseException):
            raise script
        landed = self.receipt(order.client_order_id, order.symbol, order.side, str(order.quantity))
        self.book[order.client_order_id] = landed
        if isinstance(script, tuple):  # it landed, but the answer was lost
            raise script[1]
        return landed

    def get_order(self, client_order_id):
        self.calls.append(("get", client_order_id))
        if self.lookups:
            script = self.lookups.pop(0)
            if isinstance(script, BaseException):
                raise script
            return script
        found = self.book.get(client_order_id)
        return None if found is None else found.model_copy(update={"fetched_at": self.clock()})

    def get_open_orders(self):
        self.calls.append(("list",))
        if self.listing_error is not None:
            raise self.listing_error
        open_words = (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
        return [r for r in self.book.values() if r.status in open_words] + self.foreign

    def cancel_order(self, broker_order_id):
        self.calls.append(("cancel", broker_order_id))
        if self.cancels:
            raise self.cancels.pop(0)
        for key, receipt in self.book.items():
            if receipt.broker_order_id == broker_order_id:
                self.book[key] = receipt.model_copy(update={
                    "broker_status": "canceled", "status": OrderStatus.CANCELLED,
                    "fetched_at": self.clock(),
                })

    def close_position(self, symbol):  # D5: nothing calls it
        raise AssertionError("close_position must never be called by the dispatcher")


def _failure(outcome, text="refused"):
    return ExecutionError("submit order", "aegis-prop-0001", RuntimeError(text), outcome=outcome)


def _factory(broker):
    built = []

    def factory(conn, config):
        built.append(config)
        return broker

    factory.built = built
    return factory


def _place(conn, decision_id, broker=None, *, config=CONFIG, builder=None, clock=lambda: LATER,
           dry_run=False):
    return dispatch.place(
        conn, decision_id, config=config,
        broker_factory=_factory(broker or FakeBroker()), context_builder=builder or _builder(),
        clock=clock, dry_run=dry_run,
    )


def _refused(*args, **kwargs):
    with pytest.raises(PlacementError) as info:
        _place(*args, **kwargs)
    return info.value


def _approve(conn, decision, clock=lambda: LATER, **kwargs):
    return dispatch.answer(conn, decision.id, approved=True, by="operator", config=CONFIG,
                           clock=clock, **kwargs)


# --- the happy paths -------------------------------------------------------------


class TestPlace:
    def test_an_auto_execute_verdict_becomes_one_order_at_the_broker(self, conn):
        proposal, verdict = _judged(conn)
        assert verdict.verdict is Verdict.AUTO_EXECUTE
        broker = FakeBroker()
        placement = _place(conn, verdict.id, broker)
        [(kind, sent)] = broker.calls
        assert kind == "submit"
        assert (sent.symbol, sent.side, sent.quantity, sent.limit_price) == (
            "AAPL", OrderSide.BUY, 2, Decimal("200.0"),
        )
        assert sent.client_order_id == "aegis-prop-0001" and sent.approval_id is None
        assert sent.decision_id == verdict.id and sent.regate_decision_id == placement.regate.id
        stored = get_order(conn, "aegis-prop-0001")
        assert stored == placement.order
        assert stored.status is OrderStatus.SUBMITTED and stored.broker_status == "accepted"
        assert stored.broker_order_id == broker.book["aegis-prop-0001"].broker_order_id
        regate = get_decision(conn, placement.regate.id)
        assert regate.purpose is DecisionPurpose.PRE_SUBMIT and regate.verdict is Verdict.AUTO_EXECUTE
        assert _events(conn) == ["order_claimed", "order_submitted"]
        assert placement.lines[0] == "claimed aegis-prop-0001: buy 2 AAPL limit 200 day"

    def test_an_approved_single_option_is_bought_to_open(self, conn):
        proposal, verdict = _judged(conn, long_call())
        assert verdict.verdict is Verdict.NEEDS_APPROVAL
        approval = _approve(conn, verdict)
        broker = FakeBroker()
        placement = _place(conn, verdict.id, broker)
        sent = broker.calls[0][1]
        assert sent.instrument is Instrument.OPTION and sent.symbol == proposal.legs[0].symbol
        assert sent.position_intent is PositionIntent.BUY_TO_OPEN and sent.quantity == 1
        assert sent.approval_id == approval.id and sent.limit_price == Decimal("8.0")
        assert placement.order.status is OrderStatus.SUBMITTED
        assert placement.order.approval_id == approval.id

    def test_the_real_paper_executor_accepts_what_the_dispatcher_claims(self, conn, fixture):
        """End to end with 6b's PaperExecutor over a scripted client: its own
        check of the claimed row passes, one request goes out, the receipt is
        recorded."""
        from test_execution_paper import FakeClient, _body

        from aegis.execution.paper import PaperExecutor

        _, verdict = _judged(conn)
        client = FakeClient(_body(fixture, qty="2", limit_price="200"))
        factory = lambda c, config: PaperExecutor(c, client, clock=lambda: LATER)  # noqa: E731
        placement = dispatch.place(conn, verdict.id, config=CONFIG, broker_factory=factory,
                                   context_builder=_builder(), clock=lambda: LATER)
        assert [call[0] for call in client.calls] == ["submit_order"]
        assert placement.order.status is OrderStatus.SUBMITTED
        assert placement.order.broker_order_id == "61e69015-8549-4bfd-b9c3-01e75843f47d"


class TestNothingIsSent:
    def test_with_placing_off_nothing_is_read_built_or_written(self, conn):
        _, verdict = _judged(conn)
        before = _snapshot(conn)
        broker = FakeBroker()
        factory = _factory(broker)
        with pytest.raises(PlacementError, match="broker.enabled is false"):
            dispatch.place(conn, verdict.id, config=OFF, broker_factory=factory,
                           context_builder=_builder(), clock=lambda: LATER)
        assert _snapshot(conn) == before and broker.calls == [] and factory.built == []

    def test_a_dry_run_rechecks_and_writes_nothing_even_with_placing_off(self, conn):
        _, verdict = _judged(conn)
        before = _snapshot(conn)
        broker = FakeBroker()
        placement = _place(conn, verdict.id, broker, config=OFF, dry_run=True)
        assert placement.order is None and broker.calls == []
        assert placement.lines[0] == "would place aegis-prop-0001: buy 2 AAPL limit 200 day"
        assert placement.regate.verdict is Verdict.AUTO_EXECUTE
        assert _snapshot(conn) == before  # not even the pre_submit decision

    @pytest.mark.parametrize(("seconds", "allowed"), [(299, True), (300, False), (301, False)])
    def test_an_auto_execute_verdict_goes_stale(self, conn, seconds, allowed):
        _, verdict = _judged(conn)
        clock = lambda: NOW + timedelta(seconds=seconds)  # noqa: E731
        if allowed:
            assert _place(conn, verdict.id, clock=clock).order is not None
        else:
            assert "not younger than broker.max_decision_age_seconds" in str(
                _refused(conn, verdict.id, clock=clock)
            )

    def test_a_verdict_from_the_future_is_refused(self, conn):
        _, verdict = _judged(conn)
        assert "dated after now" in str(_refused(conn, verdict.id, clock=lambda: NOW - timedelta(seconds=1)))

    def test_only_the_latest_verdict_on_a_proposal_may_go(self, conn):
        proposal, first = _judged(conn)
        later = evaluate(proposal, context_for(proposal, now=NOW + timedelta(seconds=5)), conn)
        assert f"supersedes it: {later.id}" in str(_refused(conn, first.id))
        assert _place(conn, later.id).order is not None

    def test_a_pre_submit_check_or_an_unknown_id_is_no_verdict(self, conn):
        _, verdict = _judged(conn)
        placement = _place(conn, verdict.id)
        assert "pre_submit re-check is not a verdict" in str(_refused(conn, placement.regate.id))
        assert "no such decision" in str(_refused(conn, "dec-nowhere"))

    @pytest.mark.parametrize("verdict", [Verdict.REJECT, Verdict.FLAG_ONLY])
    def test_no_order_follows_a_reject_or_a_flag(self, conn, verdict):
        proposal = make_proposal()
        insert_proposal(conn, proposal.proposal)
        decision = record_decision(conn, PolicyDecision(
            proposal_id=proposal.id, decided_at=NOW, verdict=verdict, rules_evaluated=[],
        ))
        assert f"the verdict is {verdict.value}: no order follows it" in str(_refused(conn, decision.id))

    def test_one_order_per_proposal_ever(self, conn):
        _, verdict = _judged(conn)
        _place(conn, verdict.id)
        error = _refused(conn, verdict.id)
        assert "already has order aegis-prop-0001, submitted: one order per proposal" in str(error)

    def test_a_needs_approval_verdict_waits_for_a_live_yes(self, conn):
        _, verdict = _judged(conn, long_call())
        assert "awaits a human answer" in str(_refused(conn, verdict.id))
        _approve(conn, verdict)
        expiry = LATER + timedelta(seconds=CONFIG.broker.approval_ttl_seconds)
        assert "the approval expired" in str(_refused(conn, verdict.id, clock=lambda: expiry))
        assert _place(conn, verdict.id, clock=lambda: expiry - timedelta(seconds=1)).order

    def test_a_rejected_verdict_never_goes(self, conn):
        _, verdict = _judged(conn, long_call())
        dispatch.answer(conn, verdict.id, approved=False, by="operator", config=CONFIG,
                        clock=lambda: LATER)
        assert "the answer was rejected" in str(_refused(conn, verdict.id))

    @pytest.mark.parametrize(
        ("proposal", "message"),
        [
            (make_proposal(order_type=OrderType.MARKET, limit_price=None), "only limit orders"),
            (make_proposal(quantity=2.5), "is not a whole number of shares"),
            (call_debit_spread(), "a 2-leg option order: only single-leg options"),
        ],
    )
    def test_shapes_out_of_phase_6_are_refused_before_the_recheck(self, conn, proposal, message):
        insert_proposal(conn, proposal.proposal, proposal.legs)
        decision = record_decision(conn, PolicyDecision(
            proposal_id=proposal.id, decided_at=NOW, verdict=Verdict.AUTO_EXECUTE, rules_evaluated=[],
        ))
        before = _snapshot(conn)
        assert message in str(_refused(conn, decision.id))
        assert _snapshot(conn) == before  # no re-check was run


class TestTheRecheck:
    def test_an_auto_execute_order_whose_recheck_needs_approval_does_not_go(self, conn):
        _, verdict = _judged(conn)
        broker = FakeBroker()
        # the account shrank: the same order now breaks auto_tier's notional cap
        builder = _builder(limits=RiskLimits(auto_execute={"enabled": True, "max_notional": 100}))
        error = _refused(conn, verdict.id, broker, builder=builder)
        assert "the re-check is NEEDS_APPROVAL" in str(error) and broker.calls == []
        assert _events(conn) == ["policy_needs_approval", "order_refused"]
        regate = [r for r in conn.execute("SELECT purpose, verdict FROM policy_decisions")]
        assert tuple(regate[-1]) == ("pre_submit", "NEEDS_APPROVAL")
        assert get_proposal_order(conn, verdict.proposal_id) is None

    def test_a_kill_switch_set_after_the_verdict_stops_it_at_the_recheck(self, conn):
        _, verdict = _judged(conn)
        set_kill_switch(conn, True, now=NOW)
        broker = FakeBroker()
        assert "the re-check is REJECT" in str(_refused(conn, verdict.id, broker))
        assert broker.calls == []

    def _decision(self, verdict, escalated=(), rejected=()):
        rows = [{"rule": name, "outcome": "ESCALATE", "detail": "x"} for name in escalated]
        rows += [{"rule": name, "outcome": "REJECT", "detail": "x"} for name in rejected]
        return PolicyDecision(proposal_id="p", decided_at=NOW, verdict=verdict, rules_evaluated=rows,
                              notes="n")

    @pytest.mark.parametrize(
        ("verdict", "regate", "allowed"),
        [
            (("AUTO_EXECUTE", ()), ("AUTO_EXECUTE", ()), True),
            (("AUTO_EXECUTE", ()), ("NEEDS_APPROVAL", ("auto_tier",)), False),
            (("AUTO_EXECUTE", ()), ("REJECT", ()), False),
            (("NEEDS_APPROVAL", ("options_escalate", "auto_tier")),
             ("NEEDS_APPROVAL", ("auto_tier",)), True),
            (("NEEDS_APPROVAL", ("options_escalate", "auto_tier")),
             ("NEEDS_APPROVAL", ("options_escalate", "auto_tier")), True),
            (("NEEDS_APPROVAL", ("auto_tier",)),
             ("NEEDS_APPROVAL", ("auto_tier", "short_sale")), False),
            (("NEEDS_APPROVAL", ("auto_tier",)), ("AUTO_EXECUTE", ()), True),
            (("NEEDS_APPROVAL", ("auto_tier",)), ("FLAG_ONLY", ()), False),
            (("NEEDS_APPROVAL", ("auto_tier",)), ("REJECT", ()), False),
        ],
    )
    def test_what_the_recheck_must_say(self, verdict, regate, allowed):
        """D3: an approved order goes only if every escalation now was one the
        human saw; an AUTO_EXECUTE one only if the re-check is AUTO_EXECUTE."""
        original = self._decision(Verdict(verdict[0]), verdict[1])
        check = self._decision(Verdict(regate[0]), regate[1])
        why = dispatch.regate_refusal(original, check)
        assert (why is None) is allowed, why
        if regate == ("NEEDS_APPROVAL", ("auto_tier", "short_sale")):
            assert why.endswith("did not cover: short_sale")


class TestMint:
    def _pair(self, proposal, **context):
        verdict = PolicyDecision(proposal_id=proposal.id, decided_at=NOW,
                                 verdict=Verdict.NEEDS_APPROVAL, rules_evaluated=[])
        regate = verdict.model_copy(update={"id": "pre-1", "purpose": DecisionPurpose.PRE_SUBMIT})
        approval = type("A", (), {"id": "appr-1"})()
        return verdict, regate, approval, context_for(proposal, **context)

    def test_selling_a_held_option_closes_it(self):
        proposal = make_option([("sell", "call", 640.0)], side="sell", limit_price=8.0)
        symbol = proposal.legs[0].symbol
        verdict, regate, approval, context = self._pair(
            proposal, positions=(make_position(symbol, qty=3, side="long"),)
        )
        order = dispatch.mint(proposal, verdict, regate, approval, context)
        assert order.position_intent is PositionIntent.SELL_TO_CLOSE and order.side is OrderSide.SELL

    @pytest.mark.parametrize(
        ("proposal_side", "positions", "message"),
        [
            ("sell", (), "would open a short"),
            ("sell", ((1, "long"),), "would open a short: the account holds 1"),
            ("buy", ((1, "short"),), "buying to close is out of Phase 6"),
            ("buy", None, "positions are unknown"),
        ],
    )
    def test_writing_or_covering_options_is_out_of_scope(self, proposal_side, positions, message):
        proposal = make_option([(proposal_side, "call", 640.0, 2)], side=proposal_side,
                               limit_price=8.0)
        symbol = proposal.legs[0].symbol
        held = None if positions is None else tuple(
            make_position(symbol, qty=qty, side=side) for qty, side in positions
        )
        verdict, regate, approval, context = self._pair(proposal, positions=held)
        with pytest.raises(PlacementError, match=message):
            dispatch.mint(proposal, verdict, regate, approval, context)

    def test_a_price_the_broker_cannot_take_is_refused(self):
        proposal = make_proposal(limit_price=200.123456)
        verdict, regate, _, context = self._pair(proposal)
        with pytest.raises(PlacementError, match="cannot be built"):
            dispatch.mint(proposal, verdict, regate, None, context)


# --- what the send left behind ---------------------------------------------------


class TestOutcomes:
    @pytest.mark.parametrize(
        ("outcome", "status", "reason", "kind"),
        [
            (ExecutionOutcome.REJECTED, OrderStatus.FAILED, "rejected: ", "order_rejected"),
            (ExecutionOutcome.NOT_SENT, OrderStatus.FAILED, "not_sent: ", "order_not_sent"),
        ],
    )
    def test_a_refusal_or_nothing_sent_is_failed_and_on_record(self, conn, outcome, status, reason, kind):
        _, verdict = _judged(conn)
        broker = FakeBroker()
        broker.send = [_failure(outcome, "insufficient buying power")]
        error = _refused(conn, verdict.id, broker)
        assert error.outcome == outcome.value
        stored = get_order(conn, "aegis-prop-0001")
        assert stored.status is status and stored.status_reason.startswith(reason)
        assert "insufficient buying power" in stored.status_reason
        assert _events(conn)[-1] == kind
        assert [c[0] for c in broker.calls] == ["submit"]  # never retried

    def test_an_unknown_outcome_is_looked_up_and_adopted_when_the_order_landed(self, conn):
        _, verdict = _judged(conn)
        broker = FakeBroker()
        broker.send = [("land", _failure(ExecutionOutcome.UNKNOWN, "read timed out"))]
        placement = _place(conn, verdict.id, broker)
        assert [c[0] for c in broker.calls] == ["submit", "get"]  # one lookup, no resend
        assert placement.order.status is OrderStatus.SUBMITTED
        assert placement.order.status_reason.startswith("unknown_outcome: ")
        assert _events(conn)[-2:] == ["order_unknown_outcome", "order_adopted"]
        assert "the send ended unclear (unknown)" in placement.lines[-1]

    @pytest.mark.parametrize("lookup", [None, _failure(ExecutionOutcome.UNKNOWN, "timeout")])
    def test_an_unknown_outcome_not_found_stays_claimed_and_counted(self, conn, lookup):
        _, verdict = _judged(conn)
        broker = FakeBroker()
        broker.send = [_failure(ExecutionOutcome.UNKNOWN, "read timed out")]
        broker.lookups = [lookup]
        error = _refused(conn, verdict.id, broker)
        assert error.outcome == "unknown" and "never sent again" in str(error)
        stored = get_order(conn, "aegis-prop-0001")
        assert stored.status is OrderStatus.APPROVED and stored.status_reason.startswith("unknown_outcome")
        start, end = trading_day(LATER, CONFIG)
        assert count_orders_submitted_between(conn, start, end) == 1
        level = conn.execute("SELECT level FROM events WHERE kind = 'order_unknown_outcome'").fetchone()
        assert level[0] == EventLevel.CRITICAL.value

    def test_an_interrupted_send_goes_on_the_record_and_is_reraised(self, conn):
        _, verdict = _judged(conn)
        broker = FakeBroker()
        broker.send = [KeyboardInterrupt()]
        with pytest.raises(KeyboardInterrupt):
            _place(conn, verdict.id, broker)
        stored = get_order(conn, "aegis-prop-0001")
        assert stored.status is OrderStatus.APPROVED
        assert stored.status_reason == "unknown_outcome: the send was interrupted (KeyboardInterrupt)"

    def test_the_claim_counts_the_daily_cap_on_the_store(self, conn):
        _, verdict = _judged(conn)
        capped = CONFIG.model_copy(update={"risk_limits": RiskLimits(max_daily_trades=0)})
        broker = FakeBroker()
        error = _refused(conn, verdict.id, broker, config=capped)
        assert "the claim was refused: 0 orders sent today: at or above the cap of 0" in str(error)
        assert broker.calls == [] and _events(conn)[-1] == "order_refused"


# --- approve / reject ---------------------------------------------------------------


class TestAnswer:
    def test_an_approval_is_written_once_tied_to_its_decision(self, conn):
        _, verdict = _judged(conn, long_call())
        approval = _approve(conn, verdict, note="  looks right\n ")
        assert approval.response is ApprovalResponse.APPROVED and approval.responder == "operator"
        assert approval.expires_at == LATER + timedelta(seconds=900)
        assert approval.decision_id == verdict.id and approval.channel == "cli"
        assert get_decision_approval(conn, verdict.id) == approval
        assert _events(conn)[-1] == "approval_answered"
        with pytest.raises(PlacementError, match="already answered: approved by operator"):
            dispatch.answer(conn, verdict.id, approved=False, by="someone", config=CONFIG,
                            clock=lambda: LATER)

    def test_a_rejection_has_no_expiry(self, conn):
        _, verdict = _judged(conn, long_call())
        answer = dispatch.answer(conn, verdict.id, approved=False, by="op", config=CONFIG,
                                 clock=lambda: LATER)
        assert answer.response is ApprovalResponse.REJECTED and answer.expires_at is None

    def test_only_needs_approval_awaits_an_answer(self, conn):
        _, verdict = _judged(conn)
        with pytest.raises(PlacementError, match="only NEEDS_APPROVAL awaits an answer"):
            _approve(conn, verdict)

    @pytest.mark.parametrize("by", ["", "   ", "\n\t"])
    def test_someone_must_answer(self, conn, by):
        _, verdict = _judged(conn, long_call())
        with pytest.raises(PlacementError, match="say who answers"):
            dispatch.answer(conn, verdict.id, approved=True, by=by, config=CONFIG, clock=lambda: LATER)

    def test_a_naive_clock_is_refused(self, conn):
        _, verdict = _judged(conn, long_call())
        with pytest.raises(PlacementError, match="timezone-aware"):
            _approve(conn, verdict, clock=lambda: datetime(2026, 7, 30, 15, 0))


# --- sync -------------------------------------------------------------------------


def _placed(conn, broker, proposal=None):
    _, verdict = _judged(conn, proposal)
    if verdict.verdict is Verdict.NEEDS_APPROVAL:
        _approve(conn, verdict)
    return _place(conn, verdict.id, broker).order


def _sync(conn, broker, clock=lambda: LATER, config=CONFIG):
    return dispatch.sync(conn, config=config, broker_factory=_factory(broker), clock=clock)


class TestSync:
    def test_fills_come_in_as_the_broker_reports_them_and_a_replay_adds_nothing(self, conn):
        broker = FakeBroker()
        order = _placed(conn, broker)
        broker.fill(order.client_order_id, "1", "199.5")
        report = _sync(conn, broker)
        assert report.failures == 0
        assert report.lines == ("aegis-prop-0001: submitted -> partially_filled (1 of 2 filled)",)
        broker.fill(order.client_order_id, "2", "199.75", "filled")
        _sync(conn, broker)
        assert _sync(conn, broker).lines == ()  # nothing open any more
        stored = get_order(conn, order.client_order_id)
        assert stored.status is OrderStatus.FILLED and stored.avg_fill_price == 199.75
        fills = conn.execute("SELECT fill_quantity, fill_price FROM fills ORDER BY rowid").fetchall()
        assert [tuple(f) for f in fills] == [(1.0, 199.5), (1.0, 200.0)]

    def test_an_unchanged_order_is_said_so(self, conn):
        broker = FakeBroker()
        _placed(conn, broker)
        assert _sync(conn, broker).lines == ("aegis-prop-0001: submitted, unchanged",)

    def test_an_unacknowledged_order_the_broker_never_got_is_cancelled_after_the_grace(self, conn):
        broker = FakeBroker()
        _, verdict = _judged(conn)
        broker.send = [_failure(ExecutionOutcome.UNKNOWN, "timeout")]
        broker.lookups = [None]
        _refused(conn, verdict.id, broker)
        early = _sync(conn, broker, clock=lambda: LATER + timedelta(seconds=119))
        assert "not at the broker yet (119 s since the claim, grace 120 s)" in early.lines[0]
        assert get_order(conn, "aegis-prop-0001").status is OrderStatus.APPROVED
        late = _sync(conn, broker, clock=lambda: LATER + timedelta(seconds=120))
        assert late.lines[0] == "aegis-prop-0001: approved -> cancelled (the broker never had it)"
        stored = get_order(conn, "aegis-prop-0001")
        assert stored.status is OrderStatus.CANCELLED and stored.status_reason == "not_found_at_broker"
        start, end = trading_day(LATER, CONFIG)
        assert count_orders_submitted_between(conn, start, end) == 1  # D9: still counted

    def test_an_order_lost_at_the_broker_is_a_failure(self, conn):
        broker = FakeBroker()
        order = _placed(conn, broker)
        del broker.book[order.client_order_id]
        report = _sync(conn, broker)
        assert report.failures == 1 and "unknown at the broker" in report.lines[0]
        assert _events(conn)[-1] == "order_sync_failed"

    def test_an_orphan_turns_the_kill_switch_on_once(self, conn):
        broker = FakeBroker()
        _placed(conn, broker)
        broker.foreign = [broker.receipt("dashboard-1", "MSFT")]
        report = _sync(conn, broker)
        assert report.failures == 1
        assert report.lines[-2:] == (
            f"ORPHAN dashboard-1 ({broker.foreign[0].broker_order_id}) MSFT",
            "kill switch ON (an order the store does not hold is working)",
        )
        assert get_controls(conn).kill_switch
        assert _events(conn)[-2:] == ["orphan_broker_order", "kill_switch"]
        again = _sync(conn, broker)
        assert again.failures == 1 and _events(conn)[-1] == "orphan_broker_order"

    def test_a_broker_that_cannot_be_read_is_a_failure(self, conn):
        broker = FakeBroker()
        _placed(conn, broker)
        broker.lookups = [_failure(ExecutionOutcome.UNKNOWN, "503")]
        broker.listing_error = _failure(ExecutionOutcome.UNKNOWN, "503")
        report = _sync(conn, broker)
        assert report.failures == 2

    def test_a_receipt_the_store_refuses_is_a_failure_on_record(self, conn):
        broker = FakeBroker()
        order = _placed(conn, broker)
        broker.book[order.client_order_id] = broker.book[order.client_order_id].model_copy(update={
            "status": OrderStatus.FAILED, "broker_status": "rejected",
            "filled_quantity": Decimal("1"), "avg_fill_price": Decimal("1"),
        })
        report = _sync(conn, broker)
        assert report.failures == 1 and _events(conn)[-1] == "order_sync_failed"


# --- cancel and stand down --------------------------------------------------------


class TestCancel:
    def test_a_working_order_is_cancelled_and_read_back(self, conn):
        broker = FakeBroker()
        order = _placed(conn, broker)
        report = dispatch.cancel(conn, order.client_order_id, config=CONFIG,
                                 broker_factory=_factory(broker), clock=lambda: LATER)
        assert ("cancel", order.broker_order_id) in broker.calls
        assert get_order(conn, order.client_order_id).status is OrderStatus.CANCELLED
        assert report.lines[0] == "aegis-prop-0001: cancel requested"

    def test_a_finished_order_is_left_alone(self, conn):
        broker = FakeBroker()
        order = _placed(conn, broker)
        broker.fill(order.client_order_id, "2", "200", "filled")
        _sync(conn, broker)
        report = dispatch.cancel(conn, order.client_order_id, config=CONFIG,
                                 broker_factory=_factory(broker), clock=lambda: LATER)
        assert report.lines == ("aegis-prop-0001: already filled, nothing to cancel",)

    def test_an_unknown_name_is_refused(self, conn):
        with pytest.raises(PlacementError, match="no order placed under this name"):
            dispatch.cancel(conn, "aegis-nowhere", config=CONFIG,
                            broker_factory=_factory(FakeBroker()), clock=lambda: LATER)


class TestStandDown:
    def test_with_the_controls_clear_nothing_happens(self, conn):
        broker = FakeBroker()
        factory = _factory(broker)
        report = dispatch.stand_down(conn, config=CONFIG, broker_factory=factory, clock=lambda: LATER)
        assert report.lines == ("controls clear: nothing to stand down",) and factory.built == []

    @pytest.mark.parametrize("stop", ["kill", "halt", "unreadable"])
    def test_every_working_order_is_cancelled_when_trading_must_stop(self, conn, stop):
        broker = FakeBroker()
        order = _placed(conn, broker)
        if stop == "kill":
            set_kill_switch(conn, True, now=LATER)
        elif stop == "halt":
            set_halt_until(conn, LATER + timedelta(hours=1), now=LATER)
        else:
            conn.execute("UPDATE controls SET value = '2026-13-99T99:99:99' WHERE key = 'halt_until'")
        report = dispatch.stand_down(conn, config=CONFIG, broker_factory=_factory(broker),
                                     clock=lambda: LATER)
        assert report.failures == 0 and report.lines[0].startswith("standing down: ")
        assert get_order(conn, order.client_order_id).status is OrderStatus.CANCELLED
        assert _events(conn)[-1] == "stand_down"

    def test_a_cancel_that_fails_is_a_failure(self, conn):
        broker = FakeBroker()
        _placed(conn, broker)
        set_kill_switch(conn, True, now=LATER)
        broker.cancels = [_failure(ExecutionOutcome.UNKNOWN, "502")]
        report = dispatch.stand_down(conn, config=CONFIG, broker_factory=_factory(broker),
                                     clock=lambda: LATER)
        assert report.failures == 1
        level = conn.execute("SELECT level FROM events WHERE kind = 'stand_down'").fetchone()[0]
        assert level == EventLevel.CRITICAL.value


# --- tick ---------------------------------------------------------------------------


class TestTick:
    def _tick(self, conn, broker, config=CONFIG, builder=None):
        return dispatch.tick(conn, config=config, broker_factory=_factory(broker),
                             context_builder=builder or _builder(), clock=lambda: LATER)

    def test_it_places_what_may_go_and_nothing_twice(self, conn):
        broker = FakeBroker()
        _judged(conn)
        option, verdict = _judged(conn, long_call(id="prop-0002", cycle_id="c2"))
        _approve(conn, verdict)
        report = self._tick(conn, broker)
        assert {o.client_order_id for o in report.placed} == {"aegis-prop-0001", "aegis-prop-0002"}
        assert report.failures == 0
        again = self._tick(conn, broker)
        assert again.placed == () and [c for c in broker.calls if c[0] == "submit"].__len__() == 2

    def test_with_placing_off_it_follows_and_stands_down_but_places_nothing(self, conn):
        broker = FakeBroker()
        _judged(conn)
        report = self._tick(conn, broker, config=OFF)
        assert report.lines[-1] == "placing nothing: broker.enabled is false"
        assert report.placed == ()

    def test_under_the_kill_switch_it_places_nothing(self, conn):
        broker = FakeBroker()
        _judged(conn)
        set_kill_switch(conn, True, now=LATER)
        report = self._tick(conn, broker)
        assert report.lines[-1] == "placing nothing: the controls say stop"
        assert [c for c in broker.calls if c[0] == "submit"] == []

    def test_a_refused_placement_is_a_line_not_a_failure(self, conn):
        broker = FakeBroker()
        _judged(conn)
        builder = _builder(limits=RiskLimits(auto_execute={"enabled": True, "max_notional": 1}))
        report = self._tick(conn, broker, builder=builder)
        assert report.failures == 0 and "the re-check is NEEDS_APPROVAL" in report.lines[-1]


# --- the engine and context pieces this phase added -----------------------------------


def test_a_pre_submit_decision_is_recorded_as_one(conn):
    proposal = long_call()
    insert_proposal(conn, proposal.proposal, proposal.legs)
    decision = evaluate(proposal, context_for(proposal), conn, purpose=DecisionPurpose.PRE_SUBMIT)
    assert get_decision(conn, decision.id).purpose is DecisionPurpose.PRE_SUBMIT
    payload = conn.execute("SELECT payload FROM events").fetchone()[0]
    assert '"purpose": "pre_submit"' in payload
    plain = evaluate(proposal, context_for(proposal), conn)
    assert plain.purpose is DecisionPurpose.EVALUATE
    assert '"purpose"' not in conn.execute("SELECT payload FROM events ORDER BY rowid DESC").fetchone()[0]


def test_the_trading_day_is_new_yorks_calendar_day():
    start, end = trading_day(datetime(2026, 7, 31, 2, 0, tzinfo=UTC), CONFIG)  # 22:00 NY, Jul 30
    assert (start, end) == (datetime(2026, 7, 30, 4, 0, tzinfo=UTC), datetime(2026, 7, 31, 4, 0, tzinfo=UTC))


def test_the_wall_clock_is_aware():
    assert wall_clock().utcoffset() == timedelta(0)


def test_the_paper_broker_needs_keys(monkeypatch, conn):
    from aegis.config import ConfigError
    from aegis.execution import paper

    monkeypatch.setattr(paper, "load_env", lambda: None)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ALPACA_API_KEY"):
        dispatch.paper_broker(conn, CONFIG)


def test_the_paper_broker_is_hardened(monkeypatch, conn):
    from aegis.execution import paper

    monkeypatch.setattr(paper, "load_env", lambda: None)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key-id")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    venue = dispatch.paper_broker(conn, CONFIG)
    assert isinstance(venue, paper.PaperExecutor)
    assert venue._client._session.timeout_seconds == CONFIG.broker.http_timeout_seconds
