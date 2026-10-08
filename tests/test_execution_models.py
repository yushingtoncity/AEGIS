"""The execution boundary's models: what an ApprovedOrder may be (and what it
can never be built as), what a receipt holds, and how a failure reads.

Fixtures only, no network (CLAUDE.md)."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from aegis.execution.models import (
    CLIENT_ORDER_ID_PREFIX,
    ApprovedOrder,
    ExecutionError,
    ExecutionOutcome,
    OrderReceipt,
    is_tradable_symbol,
)
from aegis.store.models import (
    Instrument,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
OPTION = "SPY261016C00640000"


def _equity(**overrides):
    fields = {
        "proposal_id": "prop-0001",
        "decision_id": "dec-prop-0001",
        "regate_decision_id": "pre-prop-0001",
        "client_order_id": "aegis-prop-0001",
        "instrument": Instrument.EQUITY,
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": 10,
        "limit_price": Decimal("200.50"),
    }
    fields.update(overrides)
    return ApprovedOrder(**fields)


def _option(**overrides):
    fields = {
        "instrument": Instrument.OPTION,
        "symbol": OPTION,
        "quantity": 1,
        "limit_price": Decimal("8.25"),
        "position_intent": PositionIntent.BUY_TO_OPEN,
        "approval_id": "appr-1",
    }
    fields.update(overrides)
    return _equity(**fields)


class TestApprovedOrder:
    def test_an_equity_limit_day_order(self):
        order = _equity()
        assert order.order_type is OrderType.LIMIT and order.time_in_force is TimeInForce.DAY
        assert order.position_intent is None and order.approval_id is None
        assert order.client_order_id == CLIENT_ORDER_ID_PREFIX + order.proposal_id

    @pytest.mark.parametrize(
        ("side", "intent"),
        [
            (OrderSide.BUY, PositionIntent.BUY_TO_OPEN),
            (OrderSide.BUY, PositionIntent.BUY_TO_CLOSE),
            (OrderSide.SELL, PositionIntent.SELL_TO_OPEN),
            (OrderSide.SELL, PositionIntent.SELL_TO_CLOSE),
        ],
    )
    def test_a_single_leg_option_order_with_its_intent_and_an_approval(self, side, intent):
        order = _option(side=side, position_intent=intent)
        assert order.instrument is Instrument.OPTION and order.position_intent is intent

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"client_order_id": "aegis-prop-0002"}, "one name per proposal"),
            ({"client_order_id": "prop-0001"}, "one name per proposal"),
            ({"symbol": "aapl"}, "is not a ticker"),
            ({"symbol": "AA PL"}, "is not a ticker"),
            ({"symbol": ""}, "is not a ticker"),
            ({"symbol": OPTION}, "is not a ticker"),
            ({"symbol": "ＡＡＰＬ"}, "is not a ticker"),
            ({"position_intent": PositionIntent.BUY_TO_OPEN}, "names no position intent"),
            ({"quantity": 0}, "greater than 0"),
            ({"quantity": -1}, "greater than 0"),
            ({"quantity": 1.5}, "valid integer"),
            ({"quantity": 2.0}, "valid integer"),  # strict: a float is not a share count
            ({"quantity": True}, "valid integer"),
            ({"quantity": "3"}, "valid integer"),
            ({"limit_price": Decimal("0")}, "greater than 0"),
            ({"limit_price": Decimal("-1")}, "greater than 0"),
            ({"limit_price": Decimal("NaN")}, "finite"),
            ({"limit_price": Decimal("Infinity")}, "finite"),
            ({"limit_price": Decimal("1.00001")}, "decimal places"),
            ({"order_type": OrderType.MARKET}, "order_type"),
            ({"time_in_force": "gtc"}, "time_in_force"),
            ({"proposal_id": "", "client_order_id": "aegis-"}, "at least 1 character"),
            ({"decision_id": ""}, "at least 1 character"),
            ({"regate_decision_id": ""}, "at least 1 character"),
            ({"approval_id": ""}, "at least 1 character"),
            ({"thesis": "buy it"}, "Extra inputs are not permitted"),  # no free text crosses
            ({"legs": []}, "Extra inputs are not permitted"),  # no multi-leg order
            ({"order_class": "mleg"}, "Extra inputs are not permitted"),
            ({"extended_hours": True}, "Extra inputs are not permitted"),
        ],
    )
    def test_what_an_equity_order_can_never_be(self, overrides, message):
        with pytest.raises(ValidationError, match=message):
            _equity(**overrides)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"approval_id": None}, "only with a human approval"),
            ({"position_intent": None}, "names its position intent"),
            ({"side": OrderSide.SELL, "position_intent": PositionIntent.BUY_TO_OPEN},
             "sell_to_close or sell_to_open"),
            ({"side": OrderSide.BUY, "position_intent": PositionIntent.SELL_TO_CLOSE},
             "buy_to_close or buy_to_open"),
            ({"symbol": "SPY"}, "not a compact OCC option symbol"),
            ({"symbol": "SPY   261016C00640000"}, "not a compact OCC option symbol"),
            ({"symbol": "SPY261316C00640000"}, "not a compact OCC option symbol"),  # month 13
            ({"symbol": "SPY261016X00640000"}, "not a compact OCC option symbol"),
            ({"symbol": "spy261016c00640000"}, "not a compact OCC option symbol"),
        ],
    )
    def test_what_an_option_order_can_never_be(self, overrides, message):
        with pytest.raises(ValidationError, match=message):
            _option(**overrides)

    def test_it_is_frozen(self):
        order = _equity()
        with pytest.raises(ValidationError, match="frozen"):
            order.quantity = 1000

    def test_a_price_given_as_text_or_a_float_reads_as_its_decimal(self):
        assert _equity(limit_price="2.25").limit_price == Decimal("2.25")
        assert _equity(limit_price=0.1).limit_price == Decimal("0.1")
        assert _equity(limit_price="0.1234").limit_price == Decimal("0.1234")


class TestSymbols:
    @pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "F", "SPY", OPTION, "SPXW261016P05800000"])
    def test_tradable(self, symbol):
        assert is_tradable_symbol(symbol)

    @pytest.mark.parametrize(
        "symbol", ["", "aapl", "AAPL ", " AAPL", "SPY   261016C00640000", "TOOLONGTICKER", "1ABC", "SPY/USD"]
    )
    def test_not_tradable(self, symbol):
        assert not is_tradable_symbol(symbol)


def _receipt(**overrides):
    fields = {
        "broker_order_id": "61e69015-8549-4bfd-b9c3-01e75843f47d",
        "client_order_id": "aegis-prop-0001",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
        "broker_status": "accepted",
        "status": OrderStatus.SUBMITTED,
        "filled_quantity": Decimal("0"),
        "fetched_at": NOW,
    }
    fields.update(overrides)
    return OrderReceipt(**fields)


class TestOrderReceipt:
    def test_an_accepted_order(self):
        receipt = _receipt()
        assert receipt.avg_fill_price is None and receipt.submitted_at is None

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"filled_quantity": Decimal("3")}, "has an average fill price"),
            ({"filled_quantity": Decimal("11"), "avg_fill_price": Decimal("1")}, "more filled than ordered"),
            ({"filled_quantity": Decimal("-1")}, "greater than or equal to 0"),
            ({"avg_fill_price": Decimal("0")}, "greater than 0"),
            ({"fetched_at": datetime(2026, 7, 30, 15, 0)}, "timezone-aware"),
            ({"broker_status": ""}, "at least 1 character"),
            ({"broker_order_id": ""}, "at least 1 character"),
            ({"symbol": ""}, "at least 1 character"),
            ({"quantity": Decimal("0")}, "greater than 0"),
            ({"note": "x"}, "Extra inputs are not permitted"),
        ],
    )
    def test_what_a_receipt_can_never_hold(self, overrides, message):
        with pytest.raises(ValidationError, match=message):
            _receipt(**overrides)

    def test_an_order_without_symbol_side_or_quantity(self):
        receipt = _receipt(
            symbol=None, side=None, quantity=None, filled_quantity=Decimal("2"),
            avg_fill_price=Decimal("1.5"),
        )
        assert receipt.symbol is None and receipt.side is None and receipt.quantity is None


class TestExecutionError:
    def test_its_message_names_the_call_the_key_the_outcome_and_the_cause(self):
        error = ExecutionError(
            "submit order", "aegis-prop-0001", TimeoutError("read timed out"),
            outcome=ExecutionOutcome.UNKNOWN, status_code=None,
        )
        assert str(error) == (
            "broker call failed: submit order for aegis-prop-0001 [unknown]"
            " (TimeoutError: read timed out)"
        )
        assert error.outcome is ExecutionOutcome.UNKNOWN and error.key == "aegis-prop-0001"

    def test_without_a_key_or_a_cause(self):
        error = ExecutionError("get open orders", outcome=ExecutionOutcome.REJECTED, status_code=403)
        assert str(error) == "broker call failed: get open orders [rejected]"
        assert error.status_code == 403 and error.cause is None

    def test_the_outcome_is_required(self):
        with pytest.raises(TypeError):
            ExecutionError("submit order")  # type: ignore[call-arg]

    def test_the_three_outcomes(self):
        assert {outcome.value for outcome in ExecutionOutcome} == {"rejected", "not_sent", "unknown"}


def test_the_interface_file_runs_on_its_own():
    """base.py names its models for the annotations only, so it runs with no
    package around it: the brain's architecture harness loads it by path to
    prove it would catch such a load (tests/test_brain_architecture.py)."""
    import aegis.execution.base as base

    scope = {}
    exec(compile(Path(base.__file__).read_text(encoding="utf-8"), "base.py", "exec"), scope)
    assert scope["Executor"].__abstractmethods__ == {
        "submit_order", "cancel_order", "get_open_orders", "get_order", "close_position",
    }
