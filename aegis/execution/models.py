"""The typed models of the broker boundary: the order that may cross it, the
receipt that comes back, and how a call can fail.

``ApprovedOrder`` is the only thing an Executor will send. It names the
decisions and the approval that authorised it, and its shape is the Phase 6
scope (docs/phase6/SPEC_PHASE6.md): one limit order, good for the day, on
an equity or on a single option contract. Anything else (a market order,
a multi-leg order, GTC, a fractional quantity, an option without a human
approval) cannot be built, so it can never be sent. No free text from the
proposal travels with it (CLAUDE.md invariant 7).

``OrderReceipt`` is the broker's word on one order, read off its own JSON
and stamped with ``fetched_at`` like every other external read.

``ExecutionError`` carries an ``outcome``, because after a failed call the
caller has to know what happened at the broker: ``rejected`` (the broker
answered and refused: nothing changed), ``not_sent`` (nothing left this
process: nothing changed), or ``unknown`` (the request may have taken
effect). An unknown outcome is never retried on its own; the caller looks
the order up instead.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aegis.data.models import parse_occ_symbol
from aegis.store.models import (
    Instrument,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    TimeInForce,
)

CLIENT_ORDER_ID_PREFIX = "aegis-"
"""Every order AEGIS sends is named ``aegis-<proposal_id>``: one proposal,
one name, so the broker can never hold two orders for the same proposal
under AEGIS's name."""

_TICKER = re.compile(r"[A-Z][A-Z0-9.]{0,9}")
_COMPACT_OCC = re.compile(r"[A-Z0-9]{1,6}\d{6}[CP]\d{8}")

_INTENTS_BY_SIDE = {
    OrderSide.BUY: frozenset({PositionIntent.BUY_TO_OPEN, PositionIntent.BUY_TO_CLOSE}),
    OrderSide.SELL: frozenset({PositionIntent.SELL_TO_OPEN, PositionIntent.SELL_TO_CLOSE}),
}


def _is_compact_occ(symbol: str) -> bool:
    if not _COMPACT_OCC.fullmatch(symbol):
        return False
    try:
        parse_occ_symbol(symbol)
    except ValueError:
        return False
    return True


def is_tradable_symbol(symbol: str) -> bool:
    """Whether ``symbol`` is a ticker or a compact OCC option symbol: the two
    shapes a symbol AEGIS sends to the broker may take."""
    return bool(_TICKER.fullmatch(symbol)) or _is_compact_occ(symbol)


class ApprovedOrder(BaseModel):
    """One order the policy engine has authorised, and nothing else.

    ``decision_id`` is the verdict (AUTO_EXECUTE, or NEEDS_APPROVAL with
    ``approval_id`` the human's answer to it), ``regate_decision_id`` the
    pre_submit re-check it passed. The Executor still verifies all of it
    against the store before it sends anything: holding one of these is not
    enough on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    regate_decision_id: str = Field(min_length=1)
    approval_id: str | None = Field(default=None, min_length=1)
    client_order_id: str
    instrument: Instrument
    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0, strict=True)
    # Alpaca takes at most 4 decimal places (sub-penny prices below $1 only).
    limit_price: Decimal = Field(gt=0, allow_inf_nan=False, decimal_places=4)
    order_type: Literal[OrderType.LIMIT] = OrderType.LIMIT
    time_in_force: Literal[TimeInForce.DAY] = TimeInForce.DAY
    position_intent: PositionIntent | None = None

    @model_validator(mode="after")
    def _shape(self) -> ApprovedOrder:
        expected = CLIENT_ORDER_ID_PREFIX + self.proposal_id
        if self.client_order_id != expected:
            raise ValueError(f"client_order_id must be {expected!r}, one name per proposal")
        if self.instrument is Instrument.EQUITY:
            if not _TICKER.fullmatch(self.symbol):  # an OCC symbol is longer than any ticker
                raise ValueError(f"symbol {self.symbol!a} is not a ticker")
            if self.position_intent is not None:
                raise ValueError("an equity order names no position intent")
            return self
        if not _is_compact_occ(self.symbol):
            raise ValueError(f"symbol {self.symbol!a} is not a compact OCC option symbol")
        if self.position_intent not in _INTENTS_BY_SIDE[self.side]:
            raise ValueError(
                f"an option {self.side.value} names its position intent"
                f" ({' or '.join(sorted(i.value for i in _INTENTS_BY_SIDE[self.side]))})"
            )
        if self.approval_id is None:
            raise ValueError("an option order goes only with a human approval (spec D6)")
        return self


class OrderReceipt(BaseModel):
    """The broker's word on one order, as of ``fetched_at``.

    ``broker_status`` is the broker's own status word, kept verbatim;
    ``status`` is what it means in the store's terms. ``filled_quantity`` and
    ``avg_fill_price`` are the broker's cumulative figures (the store turns
    their growth into fills). ``avg_fill_price`` is None while nothing has
    filled. ``symbol``, ``side`` and ``quantity`` are None only for an order
    that has none: a multi-leg parent (Alpaca sends them blank; its legs
    carry them) or a notional order, neither of which AEGIS ever sends, but
    either of which may sit on the account and must not make a listing of
    it unreadable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_order_id: str = Field(min_length=1)
    client_order_id: str = Field(min_length=1)
    symbol: str | None = Field(default=None, min_length=1)
    side: OrderSide | None = None
    quantity: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    broker_status: str = Field(min_length=1)
    status: OrderStatus
    filled_quantity: Decimal = Field(ge=0, allow_inf_nan=False)
    avg_fill_price: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)
    submitted_at: datetime | None = None
    fetched_at: datetime

    @model_validator(mode="after")
    def _figures(self) -> OrderReceipt:
        if self.filled_quantity > 0 and self.avg_fill_price is None:
            raise ValueError("a receipt with anything filled has an average fill price")
        if self.quantity is not None and self.filled_quantity > self.quantity:
            raise ValueError("a receipt never shows more filled than ordered")
        if self.fetched_at.utcoffset() is None:
            raise ValueError("fetched_at is timezone-aware")
        return self


class ExecutionOutcome(str, Enum):
    """What a failed broker call did at the broker."""

    REJECTED = "rejected"
    """The broker answered and refused. Nothing changed there."""
    NOT_SENT = "not_sent"
    """Nothing left this process (a check refused, or the connection was
    never made). Nothing changed at the broker."""
    UNKNOWN = "unknown"
    """The request may have taken effect: a timeout, a 5xx, a duplicate
    name, an answer that would not parse. Look the order up; never resend."""


class ExecutionError(Exception):
    """A broker call failed, and ``outcome`` says what that left behind.

    Shaped like ``StoreError`` and ``DataError``: what was being done, for
    which key (a client order id, a broker order id, a symbol), and the
    cause, so a CLI prints one clean line. ``status_code`` is the HTTP
    status when the broker answered one.
    """

    def __init__(
        self,
        what: str,
        key: str | None = None,
        cause: BaseException | None = None,
        *,
        outcome: ExecutionOutcome,
        status_code: int | None = None,
    ) -> None:
        self.what = what
        self.key = key
        self.cause = cause
        self.outcome = outcome
        self.status_code = status_code
        target = f"{what} for {key}" if key else what
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        super().__init__(f"broker call failed: {target} [{outcome.value}]{detail}")
