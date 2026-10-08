"""PaperExecutor and its hardened client, against stand-ins only.

Two kinds of stand-in, no network either way (CLAUDE.md):

- ``FakeClient`` answers like a hardened TradingClient (raw JSON) from a
  script and records every call: what the executor sends, what it refuses
  to send, and how it reads each answer and each failure.
- A real alpaca-py ``TradingClient`` built by ``paper_trading_client``,
  with a recording transport adapter mounted on its HTTP session: the
  hardening itself (one request on a 429 or a 504, the timeout, the paper
  host) proven through alpaca-py's own request path.

The store is a real temporary one: orders are claimed through
``claim_order``, as Phase 6 does, before anything may be sent.
"""

import copy
import importlib.metadata
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import requests

# the executor before alpaca-py's trading package: it imports aegis.data first,
# which silences alpaca-py's websockets deprecation warning (see paper.py)
from aegis.execution import paper  # isort: skip

from alpaca.common import rest as alpaca_rest
from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderStatus as AlpacaStatus
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
from requests.adapters import BaseAdapter

from aegis.execution.models import ApprovedOrder, ExecutionError, ExecutionOutcome
from aegis.execution.paper import (
    OPEN_ORDERS_PAGE,
    PAPER_URL,
    PaperExecutor,
    outcome_of,
    paper_trading_client,
    receipt_from_broker,
    store_status,
)
from aegis.store import (
    Approval,
    ApprovalResponse,
    Broker,
    DecisionPurpose,
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
    apply_broker_update,
    claim_order,
    get_order,
    insert_proposal,
    open_store,
    record_approval,
    record_decision,
    set_halt_until,
    set_kill_switch,
    upsert_order,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
DAY_START = datetime(2026, 7, 30, 4, 0, tzinfo=UTC)
DAY_END = DAY_START + timedelta(days=1)
BROKER_ID = "61e69015-8549-4bfd-b9c3-01e75843f47d"
OPTION = "SPY261016C00640000"
TIMEOUT = 7.5


# --- the store: orders claimed as Phase 6 claims them -----------------------------


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "paper.db")
    yield connection
    connection.close()


def _equity(proposal_id="prop-0001", **overrides):
    fields = {
        "proposal_id": proposal_id,
        "decision_id": f"dec-{proposal_id}",
        "regate_decision_id": f"pre-{proposal_id}",
        "client_order_id": f"aegis-{proposal_id}",
        "instrument": Instrument.EQUITY,
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": 10,
        "limit_price": Decimal("200.5"),
    }
    fields.update(overrides)
    return ApprovedOrder(**fields)


def _option(proposal_id="prop-0001", **overrides):
    fields = {
        "instrument": Instrument.OPTION,
        "symbol": OPTION,
        "quantity": 2,
        "limit_price": Decimal("8.25"),
        "position_intent": PositionIntent.BUY_TO_OPEN,
        "approval_id": f"appr-{proposal_id}",
    }
    fields.update(overrides)
    return _equity(proposal_id, **fields)


def _claim(conn, approved, *, broker=Broker.PAPER):
    """Seed what Phase 6 writes before a send, through the repository:
    the proposal, its verdict, the pre_submit re-check, the approval (when
    the order names one), and the claimed order row."""
    pid = approved.proposal_id
    verdict = Verdict.NEEDS_APPROVAL if approved.approval_id else Verdict.AUTO_EXECUTE
    insert_proposal(conn, Proposal(
        id=pid, created_at=NOW - timedelta(minutes=3), cycle_id="cycle-0001",
        symbol=approved.symbol, instrument=approved.instrument, side=approved.side,
        quantity=float(approved.quantity), order_type=OrderType.LIMIT,
        limit_price=float(approved.limit_price), thesis="SYNTHETIC TEST DATA.", confidence=0.8,
        invalidation="SYNTHETIC TEST DATA.", raw_model_output="{}",
        model_name="synthetic-model-v0", prompt_version="test-prompt-1",
    ))
    for decision_id, purpose in (
        (approved.decision_id, DecisionPurpose.EVALUATE),
        (approved.regate_decision_id, DecisionPurpose.PRE_SUBMIT),
    ):
        record_decision(conn, PolicyDecision(
            id=decision_id, proposal_id=pid, decided_at=NOW - timedelta(minutes=1),
            verdict=verdict, rules_evaluated=[], purpose=purpose,
        ))
    if approved.approval_id:
        record_approval(conn, Approval(
            id=approved.approval_id, proposal_id=pid, requested_at=NOW - timedelta(minutes=1),
            responded_at=NOW - timedelta(seconds=30), response=ApprovalResponse.APPROVED,
            channel="cli", responder="operator", decision_id=approved.decision_id,
            expires_at=NOW + timedelta(minutes=15),
        ))
    return claim_order(conn, Order(
        id=f"ord-{pid}", proposal_id=pid, client_order_id=approved.client_order_id,
        broker=broker, status=OrderStatus.APPROVED, symbol=approved.symbol, side=approved.side,
        quantity=float(approved.quantity), limit_price=float(approved.limit_price),
        decision_id=approved.decision_id, regate_decision_id=approved.regate_decision_id,
        approval_id=approved.approval_id, instrument=approved.instrument,
        order_type=OrderType.LIMIT, time_in_force=TimeInForce.DAY,
        position_intent=approved.position_intent,
    ), now=NOW, day_start=DAY_START, day_end=DAY_END, max_daily_trades=10)


def _snapshot(conn):
    """Every row of every table: the executor reads the store, never writes it."""
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall() for table in tables}


# --- the broker's answers -----------------------------------------------------------


def _body(fixture, **changes):
    body = copy.deepcopy(fixture("alpaca_order_accepted.json"))
    body.update(changes)
    return body


def _mleg_parent(fixture):
    """A multi-leg parent as Alpaca lists it: symbol, side and asset blank,
    the legs underneath (alpaca-py's own Order model reads '' as None)."""
    return _body(
        fixture, id="0b7c2f43-5d1e-4c39-9d0f-6f0a7f1b2c3d", client_order_id="dashboard-spread-1",
        symbol="", side="", asset_id="", asset_class="", position_intent="", order_class="mleg",
        qty="1", limit_price="1.25", status="new",
    )


def _option_body(fixture, **changes):
    return _body(fixture, symbol=OPTION, asset_class="us_option", qty="2", limit_price="8.25",
                 **changes)


def _response(status, body=""):
    response = requests.Response()
    response.status_code = status
    response._content = (body if isinstance(body, str) else json.dumps(body)).encode()
    response.reason = "test"
    response.url = PAPER_URL + "/v2/orders"
    return response


def _api_error(status, body='{"code": 40010001, "message": "refused"}'):
    """What alpaca-py raises for an HTTP error answer."""
    response = _response(status, body)
    return APIError(response.text, requests.HTTPError(response=response))


class FakeClient:
    """Answers like a hardened TradingClient from a script, and records
    every call. An unscripted call fails the test (IndexError)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def _answer(self, name, *args):
        self.calls.append((name, *args))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def submit_order(self, order_data):
        return self._answer("submit_order", order_data)

    def get_order_by_client_id(self, client_id):
        return self._answer("get_order_by_client_id", client_id)

    def get_order_by_id(self, order_id):
        return self._answer("get_order_by_id", order_id)

    def get_orders(self, filter=None):
        return self._answer("get_orders", filter)

    def cancel_order_by_id(self, order_id):
        return self._answer("cancel_order_by_id", order_id)

    def close_position(self, symbol_or_asset_id, close_options=None):
        return self._answer("close_position", symbol_or_asset_id)


def _venue(conn, *answers, clock=lambda: NOW):
    client = FakeClient(*answers)
    return PaperExecutor(conn, client, clock=clock), client


def _refused(venue, order):
    with pytest.raises(ExecutionError) as info:
        venue.submit_order(order)
    return info.value


# --- the status map -------------------------------------------------------------------


class TestStatusMap:
    ALPACA_STATUSES = {
        "new", "partially_filled", "filled", "done_for_day", "canceled", "expired", "replaced",
        "pending_cancel", "pending_replace", "pending_review", "accepted", "pending_new",
        "accepted_for_bidding", "stopped", "rejected", "suspended", "calculated", "held",
    }

    def test_the_pin_is_alpacas_whole_list(self):
        # a status alpaca-py adds shows up here, to be mapped on purpose
        assert {status.value for status in AlpacaStatus} == self.ALPACA_STATUSES

    @pytest.mark.parametrize("status", sorted(ALPACA_STATUSES))
    def test_every_alpaca_status(self, status):
        ended = {"filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELLED,
                 "expired": OrderStatus.CANCELLED, "rejected": OrderStatus.FAILED}
        nothing, some = store_status(status, Decimal("0")), store_status(status, Decimal("3"))
        if status in ended:
            assert nothing is some is ended[status]
        else:  # still open, and counted: partially filled once anything filled
            assert (nothing, some) == (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)

    @pytest.mark.parametrize("status", ["brand_new_word", "FILLED", "Canceled", " filled"])
    def test_a_word_it_does_not_know_reads_as_open(self, status):
        assert store_status(status, Decimal("0")) is OrderStatus.SUBMITTED


# --- reading the broker's JSON ----------------------------------------------------------


class TestReceiptFromBroker:
    def test_an_accepted_order(self, fixture):
        receipt = receipt_from_broker(_body(fixture), fetched_at=NOW)
        assert receipt.broker_order_id == BROKER_ID and receipt.client_order_id == "aegis-prop-0001"
        assert (receipt.symbol, receipt.side, receipt.quantity) == ("AAPL", OrderSide.BUY, Decimal("10"))
        assert (receipt.broker_status, receipt.status) == ("accepted", OrderStatus.SUBMITTED)
        assert receipt.filled_quantity == 0 and receipt.avg_fill_price is None
        # nanoseconds, read to the microsecond
        assert receipt.submitted_at == datetime(2026, 7, 30, 15, 0, 1, 120876, tzinfo=UTC)
        assert receipt.fetched_at == NOW

    def test_a_partial_fill_and_a_full_one(self, fixture):
        partial = receipt_from_broker(
            _body(fixture, status="partially_filled", filled_qty="4", filled_avg_price="200.25"),
            fetched_at=NOW,
        )
        assert partial.status is OrderStatus.PARTIALLY_FILLED
        assert (partial.filled_quantity, partial.avg_fill_price) == (Decimal("4"), Decimal("200.25"))
        filled = receipt_from_broker(
            _body(fixture, status="filled", filled_qty="10", filled_avg_price="200.4"), fetched_at=NOW
        )
        assert filled.status is OrderStatus.FILLED and filled.filled_quantity == 10

    def test_numbers_sent_as_numbers_read_the_same(self, fixture):
        receipt = receipt_from_broker(
            _body(fixture, qty=10, filled_qty=2.5, filled_avg_price=199.75, status="partially_filled"),
            fetched_at=NOW,
        )
        assert (receipt.quantity, receipt.filled_quantity, receipt.avg_fill_price) == (
            Decimal("10"), Decimal("2.5"), Decimal("199.75"),
        )

    @pytest.mark.parametrize("average", [None, "", "0", "0.0"])
    def test_nothing_filled_has_no_average(self, fixture, average):
        receipt = receipt_from_broker(_body(fixture, filled_avg_price=average), fetched_at=NOW)
        assert receipt.avg_fill_price is None

    def test_an_order_with_no_side_or_quantity(self, fixture):
        receipt = receipt_from_broker(_body(fixture, side="", qty=None), fetched_at=NOW)
        assert receipt.side is None and receipt.quantity is None

    def test_a_multi_leg_parent_reads_with_no_symbol(self, fixture):
        # found in review: Alpaca sends an mleg parent's symbol, side and asset blank
        receipt = receipt_from_broker(_mleg_parent(fixture), fetched_at=NOW)
        assert (receipt.symbol, receipt.side, receipt.quantity) == (None, None, Decimal("1"))
        assert receipt.client_order_id == "dashboard-spread-1"

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"id": None}, "has no id"),
            ({"id": 12}, "has no id"),
            ({"id": "not-a-uuid"}, "badly formed"),
            ({"status": None}, "has no status"),
            ({"status": ""}, "has no status"),
            ({"filled_qty": None}, "filled_qty is not a number"),
            ({"filled_qty": "abc"}, "filled_qty is not a number"),
            ({"filled_qty": True}, "filled_qty is not a number"),
            ({"filled_qty": "NaN"}, "filled_qty is not finite"),
            ({"filled_qty": "Infinity"}, "filled_qty is not finite"),
            ({"filled_qty": "-1"}, "greater than or equal to 0"),
            ({"filled_qty": "3", "filled_avg_price": None}, "has an average fill price"),
            ({"filled_qty": "3", "filled_avg_price": "-2"}, "greater than 0"),
            ({"filled_qty": "11", "filled_avg_price": "1"}, "more filled than ordered"),
            ({"qty": "zero"}, "qty is not a number"),
            ({"side": "short"}, "short"),
            ({"client_order_id": None}, "client_order_id"),
        ],
    )
    def test_what_it_will_not_read_as_an_order(self, fixture, changes, message):
        with pytest.raises(ValueError, match=message):
            receipt_from_broker(_body(fixture, **changes), fetched_at=NOW)

    @pytest.mark.parametrize("body", [None, [], "accepted", 3])
    def test_an_answer_that_is_not_an_order(self, body):
        with pytest.raises(ValueError, match="not an order"):
            receipt_from_broker(body, fetched_at=NOW)


# --- how a failure is read ------------------------------------------------------------


class TestOutcomeOf:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_a_definite_refusal(self, status):
        assert outcome_of(_api_error(status)) == (ExecutionOutcome.REJECTED, status)

    @pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
    def test_an_answer_that_proves_nothing(self, status):
        assert outcome_of(_api_error(status)) == (ExecutionOutcome.UNKNOWN, status)

    @pytest.mark.parametrize(
        "body",
        [
            '{"code": 40010001, "message": "client_order_id must be unique"}',
            '{"code": 40010001, "message": "Client_Order_ID already exists"}',
            "client_order_id: duplicate",
        ],
    )
    def test_a_refused_name_means_an_order_by_that_name_exists(self, body):
        assert outcome_of(_api_error(422, body)) == (ExecutionOutcome.UNKNOWN, 422)

    def test_an_api_error_with_no_status(self):
        assert outcome_of(APIError("no response")) == (ExecutionOutcome.UNKNOWN, None)

    def test_a_connection_that_was_never_made(self):
        assert outcome_of(requests.exceptions.ConnectTimeout()) == (ExecutionOutcome.NOT_SENT, None)

    @pytest.mark.parametrize(
        "error",
        [
            requests.exceptions.ReadTimeout(),
            requests.exceptions.ConnectionError(),
            requests.exceptions.ChunkedEncodingError(),
            requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0),
            ValueError("bad"),
            RuntimeError("boom"),
        ],
    )
    def test_anything_else_is_unknown(self, error):
        assert outcome_of(error) == (ExecutionOutcome.UNKNOWN, None)


# --- submit ----------------------------------------------------------------------------


class TestSubmit:
    def test_an_equity_order_is_sent_once_exactly_as_claimed(self, conn, fixture):
        approved = _equity()
        _claim(conn, approved)
        before = _snapshot(conn)
        venue, client = _venue(conn, _body(fixture))
        receipt = venue.submit_order(approved)
        assert [call[0] for call in client.calls] == ["submit_order"]
        request = client.calls[0][1]
        assert isinstance(request, LimitOrderRequest)
        assert json.loads(json.dumps(request.to_request_fields())) == {
            "symbol": "AAPL", "qty": 10.0, "side": "buy", "type": "limit",
            "time_in_force": "day", "limit_price": 200.5, "client_order_id": "aegis-prop-0001",
        }
        assert receipt.broker_order_id == BROKER_ID and receipt.status is OrderStatus.SUBMITTED
        assert receipt.fetched_at == NOW
        assert _snapshot(conn) == before  # read, never written

    def test_an_option_order_carries_its_intent(self, conn, fixture):
        approved = _option()
        _claim(conn, approved)
        venue, client = _venue(conn, _option_body(fixture))
        venue.submit_order(approved)
        fields = json.loads(json.dumps(client.calls[0][1].to_request_fields()))
        assert fields == {
            "symbol": OPTION, "qty": 2.0, "side": "buy", "type": "limit", "time_in_force": "day",
            "limit_price": 8.25, "client_order_id": "aegis-prop-0001",
            "position_intent": "buy_to_open",
        }

    @pytest.mark.parametrize("thing", [None, {"client_order_id": "aegis-prop-0001"}, "order"])
    def test_anything_but_an_approved_order_is_not_sent(self, conn, thing):
        venue, client = _venue(conn)
        error = _refused(venue, thing)
        assert error.outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    def test_a_store_order_is_not_an_approved_order(self, conn):
        claimed = _claim(conn, _equity())
        venue, client = _venue(conn)
        assert _refused(venue, claimed).outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    def test_an_order_nobody_claimed_is_not_sent(self, conn):
        venue, client = _venue(conn)
        error = _refused(venue, _equity())
        assert "no order was claimed under this name" in str(error)
        assert error.outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    def test_a_row_written_before_phase_6_is_not_a_claim(self, conn):
        approved = _equity()
        _claim(conn, _equity("prop-0002"))  # the proposal row exists through this one
        insert_proposal(conn, Proposal(
            id="prop-0001", created_at=NOW, cycle_id="c", symbol="AAPL",
            instrument=Instrument.EQUITY, side=OrderSide.BUY, quantity=10.0,
            order_type=OrderType.LIMIT, limit_price=200.5, thesis="SYNTHETIC TEST DATA.",
            confidence=0.8, invalidation="SYNTHETIC TEST DATA.", raw_model_output="{}",
            model_name="m", prompt_version="p",
        ))
        upsert_order(conn, Order(
            id="legacy", proposal_id="prop-0001", client_order_id="aegis-prop-0001",
            broker=Broker.PAPER, status=OrderStatus.APPROVED, symbol="AAPL", side=OrderSide.BUY,
            quantity=10.0, limit_price=200.5,
        ))
        venue, client = _venue(conn)
        assert "not claimed through claim_order" in str(_refused(venue, approved))
        assert client.calls == []

    @pytest.mark.parametrize(
        ("overrides", "name"),
        [
            ({"symbol": "MSFT"}, "symbol"),
            ({"side": OrderSide.SELL}, "side"),
            ({"quantity": 11}, "quantity"),
            ({"limit_price": Decimal("200.51")}, "limit price"),
            ({"decision_id": "dec-other"}, "decision"),
            ({"regate_decision_id": "pre-other"}, "pre_submit decision"),
            ({"approval_id": "appr-forged"}, "approval"),
        ],
    )
    def test_an_order_that_is_not_the_claimed_one_is_not_sent(self, conn, overrides, name):
        _claim(conn, _equity())
        venue, client = _venue(conn)
        error = _refused(venue, _equity(**overrides))
        assert f"its {name} is not the claimed order's" in str(error)
        assert error.outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    @pytest.mark.parametrize(
        ("overrides", "name"),
        [
            ({"position_intent": PositionIntent.BUY_TO_CLOSE}, "position intent"),
            ({"approval_id": "appr-forged"}, "approval"),
            ({"symbol": "SPY261016P00640000"}, "symbol"),
        ],
    )
    def test_an_option_order_that_is_not_the_claimed_one_is_not_sent(self, conn, overrides, name):
        _claim(conn, _option())
        venue, client = _venue(conn)
        assert f"its {name} is not the claimed order's" in str(_refused(venue, _option(**overrides)))
        assert client.calls == []

    def test_an_equity_order_claimed_as_an_option_is_not_sent(self, conn):
        _claim(conn, _option())
        venue, client = _venue(conn)
        forged = _equity(approval_id="appr-prop-0001")  # every link right, the shape wrong
        assert "its instrument is not the claimed order's" in str(_refused(venue, forged))
        assert client.calls == []

    def test_an_order_claimed_for_another_broker_is_not_sent(self, conn):
        _claim(conn, _equity(), broker=Broker.LIVE)
        venue, client = _venue(conn)
        assert "its broker is not the claimed order's" in str(_refused(venue, _equity()))
        assert client.calls == []

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"status": OrderStatus.SUBMITTED, "broker_order_id": BROKER_ID}, "is submitted, not approved"),
            ({"status": OrderStatus.FAILED, "status_reason": "rejected"}, "is failed, not approved"),
            ({"status": OrderStatus.CANCELLED, "status_reason": "not_found"}, "is cancelled, not approved"),
            ({"status": OrderStatus.APPROVED, "status_reason": "unknown_outcome"}, "never sent twice"),
            ({"status": OrderStatus.APPROVED}, "never sent twice"),  # synced: looked up once
        ],
    )
    def test_an_order_sent_or_looked_up_before_is_never_sent_again(self, conn, update, message):
        approved = _equity()
        _claim(conn, approved)
        apply_broker_update(conn, approved.client_order_id, synced_at=NOW, **update)
        venue, client = _venue(conn)
        error = _refused(venue, approved)
        assert message in str(error) and client.calls == []

    def test_a_reason_on_record_alone_means_an_attempt_was_made(self, conn):
        # apply_broker_update always stamps last_synced_at too; this reads the reason by itself
        approved = _equity()
        _claim(conn, approved)
        conn.execute(
            "UPDATE orders SET status_reason = 'unknown_outcome' WHERE client_order_id = ?",
            (approved.client_order_id,),
        )
        venue, client = _venue(conn)
        assert "never sent twice" in str(_refused(venue, approved)) and client.calls == []

    def test_the_kill_switch_turned_on_after_the_claim_stops_the_send(self, conn):
        approved = _equity()
        _claim(conn, approved)
        set_kill_switch(conn, True, now=NOW)
        venue, client = _venue(conn)
        assert "refused: the kill switch is on" in str(_refused(venue, approved))
        assert client.calls == []

    def test_a_halt_in_force_stops_the_send_and_an_ended_one_does_not(self, conn, fixture):
        approved = _equity()
        _claim(conn, approved)
        set_halt_until(conn, NOW + timedelta(seconds=1), now=NOW)
        venue, client = _venue(conn, _body(fixture))
        assert "trading is halted until 2026-07-30T15:00:01" in str(_refused(venue, approved))
        assert client.calls == []
        set_halt_until(conn, NOW, now=NOW)  # it ends at this very instant
        assert venue.submit_order(approved).status is OrderStatus.SUBMITTED

    def test_a_halt_that_cannot_be_read_stops_the_send(self, conn):
        approved = _equity()
        _claim(conn, approved)
        conn.execute("UPDATE controls SET value = '2026-13-99T99:99:99' WHERE key = 'halt_until'")
        venue, client = _venue(conn)
        assert "cannot prove trading is not halted" in str(_refused(venue, approved))
        assert client.calls == []

    def test_a_store_that_cannot_be_read_stops_the_send(self, tmp_path):
        broken = open_store(tmp_path / "gone.db")
        broken.close()
        venue, client = _venue(broken)
        error = _refused(venue, _equity())
        assert "the store cannot be read" in str(error) and client.calls == []

    def test_a_naive_clock_stops_the_send(self, conn):
        approved = _equity()
        _claim(conn, approved)
        venue, client = _venue(conn, clock=lambda: datetime(2026, 7, 30, 15, 0))
        error = _refused(venue, approved)
        assert error.outcome is ExecutionOutcome.NOT_SENT and "naive" in str(error)
        assert client.calls == []

    @pytest.mark.parametrize(
        ("error", "outcome", "status"),
        [
            (_api_error(422, '{"code": 40310000, "message": "insufficient buying power"}'),
             ExecutionOutcome.REJECTED, 422),
            (_api_error(403), ExecutionOutcome.REJECTED, 403),
            (_api_error(422, '{"code": 40010001, "message": "client_order_id must be unique"}'),
             ExecutionOutcome.UNKNOWN, 422),
            (_api_error(429), ExecutionOutcome.UNKNOWN, 429),
            (_api_error(504, "upstream timed out"), ExecutionOutcome.UNKNOWN, 504),
            (_api_error(500), ExecutionOutcome.UNKNOWN, 500),
            (requests.exceptions.ConnectTimeout(), ExecutionOutcome.NOT_SENT, None),
            (requests.exceptions.ReadTimeout(), ExecutionOutcome.UNKNOWN, None),
            (requests.exceptions.ConnectionError(), ExecutionOutcome.UNKNOWN, None),
            (requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0),
             ExecutionOutcome.UNKNOWN, None),
        ],
    )
    def test_a_failed_send_says_what_it_left_behind_and_is_tried_once(
        self, conn, error, outcome, status
    ):
        approved = _equity()
        _claim(conn, approved)
        before = _snapshot(conn)
        venue, client = _venue(conn, error)
        refused = _refused(venue, approved)
        assert (refused.outcome, refused.status_code) == (outcome, status)
        assert refused.cause is error and refused.key == approved.client_order_id
        assert len(client.calls) == 1
        assert _snapshot(conn) == before

    def test_an_interrupt_during_the_send_is_not_swallowed(self, conn):
        approved = _equity()
        _claim(conn, approved)
        venue, _ = _venue(conn, KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            venue.submit_order(approved)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"client_order_id": "aegis-prop-9999"}, "names client_order_id 'aegis-prop-9999'"),
            ({"symbol": "MSFT"}, "names symbol 'MSFT'"),
            ({"side": "sell"}, "names side"),
            ({"qty": "100"}, "names quantity"),
            ({"id": "not-a-uuid"}, "badly formed"),
            ({"status": None}, "has no status"),
        ],
    )
    def test_an_answer_that_is_not_this_order_is_an_unknown_outcome(
        self, conn, fixture, changes, message
    ):
        approved = _equity()
        _claim(conn, approved)
        venue, _ = _venue(conn, _body(fixture, **changes))
        error = _refused(venue, approved)
        assert error.outcome is ExecutionOutcome.UNKNOWN and message in str(error)

    def test_a_clock_that_fails_after_the_send_is_an_unknown_outcome(self, conn, fixture):
        # found in review: after a request is made nothing may read as not_sent
        approved = _equity()
        _claim(conn, approved)
        readings = iter([NOW, datetime(2026, 7, 30, 15, 0)])  # aware for the check, naive after
        venue, client = _venue(conn, _body(fixture), clock=lambda: next(readings))
        error = _refused(venue, approved)
        assert error.outcome is ExecutionOutcome.UNKNOWN and len(client.calls) == 1

    @pytest.mark.parametrize(
        ("update", "message"),
        [
            ({"approval_id": None}, "only with a human approval"),
            ({"quantity": 2.5}, "valid integer"),
            ({"position_intent": PositionIntent.SELL_TO_CLOSE}, "names its position intent"),
        ],
    )
    def test_a_copy_that_skipped_the_rules_is_not_sent(self, conn, update, message):
        # found in review: model_copy(update=...) builds a copy without validation
        approved = _option()
        _claim(conn, approved)
        venue, client = _venue(conn)
        error = _refused(venue, approved.model_copy(update=update))
        assert error.outcome is ExecutionOutcome.NOT_SENT and message in str(error)
        assert client.calls == []

    @pytest.mark.parametrize("body", [None, [], "ok"])
    def test_an_answer_that_is_no_order_is_an_unknown_outcome(self, conn, body):
        approved = _equity()
        _claim(conn, approved)
        venue, _ = _venue(conn, body)
        assert _refused(venue, approved).outcome is ExecutionOutcome.UNKNOWN


# --- cancel ----------------------------------------------------------------------------


class TestCancel:
    def test_a_cancel_the_broker_accepts(self, conn):
        venue, client = _venue(conn, None)
        assert venue.cancel_order(BROKER_ID) is None
        assert client.calls == [("cancel_order_by_id", BROKER_ID)]

    @pytest.mark.parametrize("bad", ["", "aegis-prop-0001", "61e69015", None, 12])
    def test_a_name_that_is_no_broker_id_is_not_sent(self, conn, bad):
        venue, client = _venue(conn)
        with pytest.raises(ExecutionError) as info:
            venue.cancel_order(bad)
        assert info.value.outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    @pytest.mark.parametrize("ended", ["canceled", "expired"])
    def test_an_order_the_broker_already_ended_is_a_no_op(self, conn, fixture, ended):
        venue, client = _venue(conn, _api_error(422), _body(fixture, status=ended))
        assert venue.cancel_order(BROKER_ID) is None
        assert [call[0] for call in client.calls] == ["cancel_order_by_id", "get_order_by_id"]

    @pytest.mark.parametrize("status", ["filled", "pending_cancel", "partially_filled"])
    def test_an_order_the_broker_will_not_cancel_is_refused(self, conn, fixture, status):
        filled = {"filled_qty": "10", "filled_avg_price": "200.4"} if status != "pending_cancel" else {}
        if status == "partially_filled":
            filled = {"filled_qty": "4", "filled_avg_price": "200.4"}
        venue, _ = _venue(conn, _api_error(422), _body(fixture, status=status, **filled))
        with pytest.raises(ExecutionError, match=f"the order is {status}") as info:
            venue.cancel_order(BROKER_ID)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.REJECTED, 422)

    def test_an_unknown_order_is_refused(self, conn):
        venue, client = _venue(conn, _api_error(404), _api_error(404))
        with pytest.raises(ExecutionError) as info:
            venue.cancel_order(BROKER_ID)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.REJECTED, 404)
        assert len(client.calls) == 2

    def test_a_read_back_about_another_order_proves_nothing(self, conn, fixture):
        other = _body(fixture, id="7f1c1d2e-1111-4bfd-b9c3-01e75843f47d", status="canceled")
        venue, _ = _venue(conn, _api_error(422), other)
        with pytest.raises(ExecutionError) as info:
            venue.cancel_order(BROKER_ID)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.REJECTED, 422)

    def test_the_read_back_matches_the_id_in_any_case(self, conn, fixture):
        venue, _ = _venue(conn, _api_error(422), _body(fixture, status="canceled"))
        assert venue.cancel_order(BROKER_ID.upper()) is None

    def test_a_refusal_that_cannot_be_read_back_stays_a_refusal(self, conn):
        venue, _ = _venue(conn, _api_error(422), requests.exceptions.ReadTimeout())
        with pytest.raises(ExecutionError) as info:
            venue.cancel_order(BROKER_ID)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.REJECTED, 422)

    @pytest.mark.parametrize(
        "error", [_api_error(500), _api_error(429), requests.exceptions.ReadTimeout()]
    )
    def test_a_cancel_that_may_have_landed_is_unknown_and_not_read_back(self, conn, error):
        venue, client = _venue(conn, error)
        with pytest.raises(ExecutionError) as info:
            venue.cancel_order(BROKER_ID)
        assert info.value.outcome is ExecutionOutcome.UNKNOWN and len(client.calls) == 1


# --- reads -------------------------------------------------------------------------------


class TestReads:
    def test_the_open_orders_in_one_page(self, conn, fixture):
        orders = [_body(fixture), _body(fixture, id="7f1c1d2e-1111-4bfd-b9c3-01e75843f47d",
                                        client_order_id="manual-1", symbol="MSFT")]
        venue, client = _venue(conn, orders)
        receipts = venue.get_open_orders()
        assert [r.client_order_id for r in receipts] == ["aegis-prop-0001", "manual-1"]
        request = client.calls[0][1]
        assert isinstance(request, GetOrdersRequest)
        assert (request.status, request.limit, request.nested) == (QueryOrderStatus.OPEN, 500, False)

    def test_no_open_orders(self, conn):
        venue, _ = _venue(conn, [])
        assert venue.get_open_orders() == []

    def test_a_full_page_is_refused_not_read_as_all_of_them(self, conn, fixture):
        venue, _ = _venue(conn, [_body(fixture)] * OPEN_ORDERS_PAGE)
        with pytest.raises(ExecutionError, match="more may be open") as info:
            venue.get_open_orders()
        assert info.value.outcome is ExecutionOutcome.UNKNOWN
        venue, _ = _venue(conn, [_body(fixture)] * (OPEN_ORDERS_PAGE - 1))
        assert len(venue.get_open_orders()) == OPEN_ORDERS_PAGE - 1

    @pytest.mark.parametrize("answer", [{"orders": []}, None, "x"])
    def test_an_answer_that_is_no_list_is_refused(self, conn, answer):
        venue, _ = _venue(conn, answer)
        with pytest.raises(ExecutionError, match="not a list"):
            venue.get_open_orders()

    def test_a_multi_leg_order_on_the_account_does_not_blind_the_listing(self, conn, fixture):
        # found in review: one hand-placed spread made the whole listing unreadable
        venue, _ = _venue(conn, [_body(fixture), _mleg_parent(fixture)])
        receipts = venue.get_open_orders()
        assert [(r.client_order_id, r.symbol) for r in receipts] == [
            ("aegis-prop-0001", "AAPL"), ("dashboard-spread-1", None),
        ]

    def test_one_order_it_cannot_read_fails_the_listing_and_is_named(self, conn, fixture):
        venue, _ = _venue(conn, [_body(fixture), _body(fixture, id="nope")])
        with pytest.raises(ExecutionError, match="get open orders for nope") as info:
            venue.get_open_orders()
        assert info.value.key == "nope" and info.value.outcome is ExecutionOutcome.UNKNOWN

    def test_a_failed_listing(self, conn):
        venue, _ = _venue(conn, _api_error(503))
        with pytest.raises(ExecutionError) as info:
            venue.get_open_orders()
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.UNKNOWN, 503)

    def test_an_order_by_its_name(self, conn, fixture):
        venue, client = _venue(conn, _body(fixture, status="filled", filled_qty="10",
                                           filled_avg_price="200.4"))
        receipt = venue.get_order("aegis-prop-0001")
        assert receipt.status is OrderStatus.FILLED
        assert client.calls == [("get_order_by_client_id", "aegis-prop-0001")]

    def test_a_name_the_broker_does_not_know_reads_none(self, conn):
        venue, _ = _venue(conn, _api_error(404, '{"code": 40410000, "message": "order not found"}'))
        assert venue.get_order("aegis-prop-0001") is None

    def test_an_answer_for_another_name_is_unknown(self, conn, fixture):
        venue, _ = _venue(conn, _body(fixture, client_order_id="aegis-prop-0002"))
        with pytest.raises(ExecutionError, match="answered for 'aegis-prop-0002'") as info:
            venue.get_order("aegis-prop-0001")
        assert info.value.outcome is ExecutionOutcome.UNKNOWN

    @pytest.mark.parametrize("error", [_api_error(500), _api_error(403), requests.exceptions.ReadTimeout()])
    def test_a_failed_read_is_an_error_not_none(self, conn, error):
        venue, _ = _venue(conn, error)
        with pytest.raises(ExecutionError, match="get order"):
            venue.get_order("aegis-prop-0001")


class TestClosePosition:
    def test_it_returns_the_closing_orders_receipt(self, conn, fixture):
        venue, client = _venue(conn, _body(fixture, client_order_id="alpaca-made", side="sell"))
        receipt = venue.close_position("AAPL")
        assert receipt.side is OrderSide.SELL and client.calls == [("close_position", "AAPL")]

    @pytest.mark.parametrize("bad", ["", "aapl", "AAPL ", None, "SPY   261016C00640000"])
    def test_a_symbol_that_is_not_one_is_not_sent(self, conn, bad):
        venue, client = _venue(conn)
        with pytest.raises(ExecutionError) as info:
            venue.close_position(bad)
        assert info.value.outcome is ExecutionOutcome.NOT_SENT and client.calls == []

    def test_a_clock_that_fails_after_the_close_is_an_unknown_outcome(self, conn, fixture):
        venue, client = _venue(conn, _body(fixture), clock=lambda: datetime(2026, 7, 30, 15, 0))
        with pytest.raises(ExecutionError) as info:
            venue.close_position("AAPL")
        assert info.value.outcome is ExecutionOutcome.UNKNOWN and len(client.calls) == 1

    def test_a_failure(self, conn):
        venue, _ = _venue(conn, _api_error(504))
        with pytest.raises(ExecutionError) as info:
            venue.close_position(OPTION)
        assert info.value.outcome is ExecutionOutcome.UNKNOWN


# --- the hardened client, through alpaca-py's own request path --------------------------


class Transport(BaseAdapter):
    """Mounted on the client's HTTP session: records each request that
    reaches the wire, and answers it from a script."""

    def __init__(self, *answers):
        super().__init__()
        self.answers = list(answers)
        self.sent = []

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        self.sent.append((request, timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        status, body = answer
        response = _response(status, body)
        response.url, response.request = request.url, request
        return response

    def close(self):
        pass


@pytest.fixture
def no_sleep(monkeypatch):
    """alpaca-py sleeps between retries: a sleep here means it retried."""

    def refuse(seconds):
        raise AssertionError(f"alpaca-py tried to retry (slept {seconds}s)")

    monkeypatch.setattr(alpaca_rest.time, "sleep", refuse)


def _hardened(*answers):
    client = paper_trading_client(timeout_seconds=TIMEOUT, api_key="test-key-id", secret_key="test-secret")
    transport = Transport(*answers)
    client._session.mount("https://", transport)
    return client, transport


class TestHardenedClient:
    def test_the_private_attributes_it_hardens_are_alpaca_pys_own(self):
        # pinned to the version they were read from: an upgrade must re-check them
        assert importlib.metadata.version("alpaca-py") == "0.43.5"
        plain = TradingClient(api_key="test-key-id", secret_key="test-secret", paper=True)
        assert plain._base_url == BaseURL.TRADING_PAPER == PAPER_URL
        assert (plain._retry, plain._retry_codes) == (3, [429, 504])
        assert type(plain._session) is requests.Session and plain._use_raw_data is False

    def test_paper_trading_client_hardens_all_four(self):
        client = paper_trading_client(timeout_seconds=TIMEOUT, api_key="test-key-id", secret_key="test-secret")
        assert client._base_url == BaseURL.TRADING_PAPER
        assert (client._retry, client._retry_wait, client._retry_codes) == (0, 0, [])
        assert client._session.timeout_seconds == TIMEOUT and client._use_raw_data is True
        PaperExecutor(None, client)  # accepted

    @pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "10", None])
    def test_a_timeout_that_is_not_one_is_refused(self, timeout):
        with pytest.raises(ValueError, match="positive number of seconds"):
            paper_trading_client(timeout_seconds=timeout, api_key="k", secret_key="s")

    @pytest.mark.parametrize(
        ("api_key", "secret_key"),
        [
            ("test-key-id", "test-secret\n"),
            (" test-key-id", "test-secret"),
            ("test key", "test-secret"),
            ("test-key-id", ""),
            ("test-key-id", "test\x00secret"),
            ("test-key-id", 12345),
        ],
    )
    def test_a_key_that_is_not_one_word_is_refused_without_showing_it(self, api_key, secret_key):
        # found in review: a stray newline put the header value, key and all, in the error
        with pytest.raises(ValueError, match="its value is not shown") as info:
            paper_trading_client(timeout_seconds=TIMEOUT, api_key=api_key, secret_key=secret_key)
        assert "test-secret" not in str(info.value) and "test-key-id" not in str(info.value)

    def test_one_key_without_the_other_is_refused(self):
        with pytest.raises(ValueError, match="both keys"):
            paper_trading_client(timeout_seconds=TIMEOUT, api_key="k")

    def test_with_no_keys_passed_it_reads_them_from_the_environment(self, monkeypatch):
        monkeypatch.setattr(paper, "load_env", lambda: None)
        monkeypatch.setenv("ALPACA_API_KEY", "env-key-id")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
        assert paper_trading_client(timeout_seconds=TIMEOUT)._api_key == "env-key-id"
        monkeypatch.delenv("ALPACA_SECRET_KEY")
        with pytest.raises(Exception, match="ALPACA_SECRET_KEY"):
            paper_trading_client(timeout_seconds=TIMEOUT)

    @pytest.mark.parametrize(
        "build",
        [
            # the data layer's kind of client: retries, no timeout, parsed models
            lambda: TradingClient(api_key="k", secret_key="s", paper=True),
            # a live account
            lambda: TradingClient(api_key="k", secret_key="s", paper=False, raw_data=True),
        ],
    )
    def test_a_client_that_is_not_hardened_is_refused(self, build):
        with pytest.raises(ExecutionError, match="build it with paper_trading_client") as info:
            PaperExecutor(None, build())
        assert info.value.outcome is ExecutionOutcome.NOT_SENT

    @pytest.mark.parametrize(
        ("attribute", "value", "problem"),
        [
            ("_base_url", BaseURL.TRADING_LIVE, "not the paper host"),
            ("_base_url", "https://paper-api.alpaca.markets.example.com", "not the paper host"),
            ("_retry", 3, "retries on its own"),
            ("_retry_codes", [429, 504], "retries on its own"),
            ("_session", requests.Session(), "no paper-host check and no timeout"),
            ("_use_raw_data", False, "raw JSON"),
        ],
    )
    def test_a_hardened_client_undone_is_refused(self, attribute, value, problem):
        client = paper_trading_client(timeout_seconds=TIMEOUT, api_key="k", secret_key="s")
        setattr(client, attribute, value)
        with pytest.raises(ExecutionError, match=problem):
            PaperExecutor(None, client)

    def test_one_post_to_the_paper_host_with_the_timeout(self, conn, fixture, no_sleep):
        approved = _equity()
        _claim(conn, approved)
        client, transport = _hardened((200, _body(fixture)))
        receipt = PaperExecutor(conn, client, clock=lambda: NOW).submit_order(approved)
        assert receipt.broker_order_id == BROKER_ID
        [(request, timeout)] = transport.sent
        assert (request.method, request.url, timeout) == ("POST", PAPER_URL + "/v2/orders", TIMEOUT)
        assert json.loads(request.body) == {
            "symbol": "AAPL", "qty": 10.0, "side": "buy", "type": "limit", "time_in_force": "day",
            "limit_price": 200.5, "client_order_id": "aegis-prop-0001",
        }
        assert request.headers["APCA-API-KEY-ID"] == "test-key-id"

    @pytest.mark.parametrize("status", [429, 503, 504])
    def test_a_rate_limit_or_a_gateway_timeout_is_one_request_and_unknown(
        self, conn, no_sleep, status
    ):
        approved = _equity()
        _claim(conn, approved)
        client, transport = _hardened((status, "try again later"))
        with pytest.raises(ExecutionError) as info:
            PaperExecutor(conn, client, clock=lambda: NOW).submit_order(approved)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.UNKNOWN, status)
        assert len(transport.sent) == 1

    def test_a_refusal_with_a_json_body(self, conn, no_sleep):
        approved = _equity()
        _claim(conn, approved)
        client, transport = _hardened(
            (422, {"code": 40310000, "message": "insufficient buying power"})
        )
        with pytest.raises(ExecutionError, match="insufficient buying power") as info:
            PaperExecutor(conn, client, clock=lambda: NOW).submit_order(approved)
        assert (info.value.outcome, info.value.status_code) == (ExecutionOutcome.REJECTED, 422)
        assert len(transport.sent) == 1

    @pytest.mark.parametrize(
        ("error", "outcome"),
        [
            (requests.exceptions.ReadTimeout(), ExecutionOutcome.UNKNOWN),
            (requests.exceptions.ConnectionError(), ExecutionOutcome.UNKNOWN),
            (requests.exceptions.ConnectTimeout(), ExecutionOutcome.NOT_SENT),
        ],
    )
    def test_a_transport_failure_is_one_attempt(self, conn, no_sleep, error, outcome):
        approved = _equity()
        _claim(conn, approved)
        client, transport = _hardened(error)
        with pytest.raises(ExecutionError) as info:
            PaperExecutor(conn, client, clock=lambda: NOW).submit_order(approved)
        assert info.value.outcome is outcome and len(transport.sent) == 1

    def test_a_200_with_a_body_that_is_not_json_is_unknown(self, conn, no_sleep):
        approved = _equity()
        _claim(conn, approved)
        client, _ = _hardened((200, "<html>ok</html>"))
        with pytest.raises(ExecutionError) as info:
            PaperExecutor(conn, client, clock=lambda: NOW).submit_order(approved)
        assert info.value.outcome is ExecutionOutcome.UNKNOWN

    @pytest.mark.parametrize(
        "host", [BaseURL.TRADING_LIVE, "https://paper-api.alpaca.markets.example.com"]
    )
    def test_a_request_to_any_other_host_never_leaves(self, conn, no_sleep, host):
        approved = _equity()
        _claim(conn, approved)
        client, transport = _hardened((200, "{}"))
        venue = PaperExecutor(conn, client, clock=lambda: NOW)
        client._base_url = host  # pointed elsewhere after the check
        with pytest.raises(ExecutionError, match="only https://paper-api.alpaca.markets") as info:
            venue.submit_order(approved)
        assert info.value.outcome is ExecutionOutcome.NOT_SENT and transport.sent == []

    def test_a_timeout_the_caller_brings_is_kept(self):
        client, transport = _hardened((200, "[]"))
        client._session.request("GET", PAPER_URL + "/v2/orders", timeout=1)
        assert transport.sent[0][1] == 1

    def test_cancel_and_reads_through_the_real_path(self, conn, fixture, no_sleep):
        client, transport = _hardened(
            (204, ""),
            (404, {"code": 40410000, "message": "order not found"}),
            (200, [_body(fixture)]),
        )
        venue = PaperExecutor(conn, client, clock=lambda: NOW)
        assert venue.cancel_order(BROKER_ID) is None
        assert venue.get_order("aegis-prop-0001") is None
        assert [r.broker_order_id for r in venue.get_open_orders()] == [BROKER_ID]
        methods = [(request.method, request.url.split("?")[0]) for request, _ in transport.sent]
        assert methods == [
            ("DELETE", f"{PAPER_URL}/v2/orders/{BROKER_ID}"),
            ("GET", f"{PAPER_URL}/v2/orders:by_client_order_id"),
            ("GET", f"{PAPER_URL}/v2/orders"),
        ]
        assert all(timeout == TIMEOUT for _, timeout in transport.sent)


def test_the_store_connection_is_only_read(conn, fixture):
    """Across every call that touches the store, nothing is written: the
    executor's word goes back through the store's own writers, in 6c."""
    approved = _equity()
    _claim(conn, approved)
    before = _snapshot(conn)
    venue, _ = _venue(conn, _body(fixture), _api_error(403))
    venue.submit_order(approved)
    with pytest.raises(ExecutionError):
        venue.submit_order(approved)  # still unsent in the store, so it may try: the broker refuses
    assert _snapshot(conn) == before and not conn.in_transaction
    assert get_order(conn, approved.client_order_id).status is OrderStatus.APPROVED
    assert isinstance(conn, sqlite3.Connection)
