"""PaperExecutor: the Executor over the Alpaca paper account.

This is the one place in AEGIS where a trading client is used to place,
cancel or close anything. The data layer's cached client
(``aegis.data.clients``) stays a reader; this module builds its own, with
``paper_trading_client``, and hardens it before any request:

- the paper host only: the client's base URL must be Alpaca's paper URL,
  and its HTTP session refuses a request to any other host, so nothing
  configured, overridden or passed in can point it at a live account;
- no retries: alpaca-py retries a 429 or a 504 three times by default,
  which could send one order twice, so retries are set to zero and a retry
  is always the caller's decision, on the record;
- a timeout: alpaca-py sets none, so a hung request would block forever;
- raw JSON answers, read here into ``OrderReceipt``, so an answer
  alpaca-py's own models would not accept still says what happened.

The hardening reaches into four private attributes of alpaca-py's REST
client (``_base_url``, ``_retry``, ``_retry_codes``, ``_session``), pinned
by tests/test_execution_paper.py against alpaca-py 0.43.5: an upgrade that
moves them fails there, not in production.

``submit_order`` sends nothing it has not first checked against the store:
the order row ``claim_order`` wrote must exist, match the ApprovedOrder
field for field, be unsent and unsynced, and the controls must still be
clear (the kill switch off, no halt in force, the halt readable). Every
failure says what it left behind (``ExecutionOutcome``); nothing is ever
retried here.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

# aegis.data first, out of the usual order on purpose: importing it silences
# alpaca-py's own websockets deprecation warning (aegis/data/__init__.py says
# why), which alpaca.trading raises on import. Nothing new is filtered.
from aegis.data.models import utcnow  # isort: skip

from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaSide
from alpaca.trading.enums import PositionIntent as AlpacaIntent
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
from pydantic import ValidationError
from requests import Session
from requests.exceptions import ConnectTimeout

from aegis.config import load_env, require_env
from aegis.store.errors import StoreError
from aegis.store.models import Broker, OrderSide, OrderStatus
from aegis.store.repo import get_controls
from aegis.store.repo import get_order as get_stored_order

from .base import Executor
from .models import (
    ApprovedOrder,
    ExecutionError,
    ExecutionOutcome,
    OrderReceipt,
    is_tradable_symbol,
)

PAPER_URL = BaseURL.TRADING_PAPER.value
"""Alpaca's paper trading host: the only one this module talks to."""

OPEN_ORDERS_PAGE = 500
"""The most orders Alpaca's ``GET /orders`` returns in one answer (its own
maximum, not a tunable). A full page may not be all of them, so it is
refused rather than read as complete."""

_UNCLEAR_STATUSES = frozenset({408, 409, 429})
"""4xx answers that do not prove the broker refused the request: a timeout,
a conflict, a rate limit. Read as an unknown outcome."""


# --- the broker's words, in the store's terms ------------------------------


def store_status(broker_status: str, filled_quantity: Decimal) -> OrderStatus:
    """What Alpaca's order status means in the store's terms.

    ``filled`` is filled; ``canceled`` and ``expired`` are cancelled (a DAY
    order that did not fill expires at the close); ``rejected`` is failed.
    Every other word, one Alpaca adds later included, reads as still open
    (partially filled when anything has filled): an order the store cannot
    place is kept counted and watched, never dropped. ``replaced`` stays
    open too: AEGIS never replaces an order, so a replaced one is an anomaly
    for the caller to report, and ``broker_status`` keeps the word.
    """
    if broker_status == "filled":
        return OrderStatus.FILLED
    if broker_status in ("canceled", "expired"):
        return OrderStatus.CANCELLED
    if broker_status == "rejected":
        return OrderStatus.FAILED
    return OrderStatus.PARTIALLY_FILLED if filled_quantity > 0 else OrderStatus.SUBMITTED


def _decimal(value: object, name: str) -> Decimal:
    """A number from the broker's JSON (Alpaca sends them as strings)."""
    if not isinstance(value, (str, int, float)):  # a bool is an int, and reads as no number
        raise ValueError(f"{name} is not a number: {value!r}")
    try:
        number = Decimal(value) if isinstance(value, str) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} is not a number: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def _blank(value: object) -> bool:
    return value is None or value == ""


def receipt_from_broker(body: object, *, fetched_at: datetime) -> OrderReceipt:
    """Read one order out of Alpaca's JSON. A ValueError (or pydantic's
    ValidationError) when it is not an order this module can account for."""
    if not isinstance(body, Mapping):
        raise ValueError(f"the broker's answer is not an order: {type(body).__name__}")
    broker_order_id = body.get("id")
    if not isinstance(broker_order_id, str):
        raise ValueError("the order has no id")
    UUID(broker_order_id)  # Alpaca's ids are UUIDs: anything else is not its answer
    status = body.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("the order has no status")
    filled = _decimal(body.get("filled_qty"), "filled_qty")
    average = body.get("filled_avg_price")
    side = body.get("side")
    quantity = body.get("qty")
    return OrderReceipt(
        broker_order_id=broker_order_id,
        client_order_id=body.get("client_order_id"),
        symbol=body.get("symbol"),
        side=None if _blank(side) else OrderSide(side),
        quantity=None if _blank(quantity) else _decimal(quantity, "qty"),
        broker_status=status,
        status=store_status(status, filled),
        filled_quantity=filled,
        avg_fill_price=None if filled == 0 or _blank(average) else _decimal(average, "filled_avg_price"),
        submitted_at=body.get("submitted_at"),
        fetched_at=fetched_at,
    )


# --- how a failed call ended ----------------------------------------------------


def outcome_of(error: BaseException) -> tuple[ExecutionOutcome, int | None]:
    """What a failed request left behind at the broker, and the HTTP status
    if the broker answered one.

    Only two things prove nothing was sent: a request this module refused to
    send (to a host other than the paper one) and a connection that timed
    out before it was made. Only a definite 4xx proves the broker refused:
    not a timeout, conflict or rate limit (408, 409, 429), and not an answer
    about the ``client_order_id``, which is how a duplicate name is refused
    (an order by that name already exists). Everything else (a 5xx, a read
    timeout, a dropped connection, any other error) is unknown.
    """
    if isinstance(error, _OffPaper | ConnectTimeout):
        return ExecutionOutcome.NOT_SENT, None
    if isinstance(error, APIError):
        code = error.status_code
        if not isinstance(code, int):
            return ExecutionOutcome.UNKNOWN, None
        definite = 400 <= code < 500 and code not in _UNCLEAR_STATUSES
        if definite and "client_order_id" not in str(error).lower():
            return ExecutionOutcome.REJECTED, code
        return ExecutionOutcome.UNKNOWN, code
    return ExecutionOutcome.UNKNOWN, None


def _failed(what: str, key: str | None, error: BaseException) -> ExecutionError:
    outcome, code = outcome_of(error)
    return ExecutionError(what, key, error, outcome=outcome, status_code=code)


# --- the hardened client ------------------------------------------------------


class _OffPaper(Exception):
    """A request aimed at a host other than the paper one. Never sent."""


class _PaperSession(Session):
    """The HTTP session of a hardened client: paper host only, and a
    timeout on every request that does not bring its own."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__()
        self.timeout_seconds = timeout_seconds

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if not str(url).startswith(PAPER_URL + "/"):
            raise _OffPaper(f"refused a request to {url!r}: only {PAPER_URL} is allowed")
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout_seconds
        return super().request(method, url, *args, **kwargs)


def _hardening_problems(client: TradingClient) -> list[str]:
    problems = []
    if getattr(client, "_base_url", None) != BaseURL.TRADING_PAPER:
        problems.append("its base URL is not the paper host")
    if getattr(client, "_retry", None) != 0 or getattr(client, "_retry_codes", None) != []:
        problems.append("it retries on its own")
    if not isinstance(getattr(client, "_session", None), _PaperSession):
        problems.append("its HTTP session has no paper-host check and no timeout")
    if getattr(client, "_use_raw_data", None) is not True:
        problems.append("it does not answer in raw JSON")
    return problems


def _require_hardened(client: TradingClient) -> None:
    problems = _hardening_problems(client)
    if problems:
        raise ExecutionError(
            f"use a trading client ({'; '.join(problems)}: build it with paper_trading_client)",
            outcome=ExecutionOutcome.NOT_SENT,
        )


def paper_trading_client(
    *,
    timeout_seconds: float,
    api_key: str | None = None,
    secret_key: str | None = None,
) -> TradingClient:
    """A TradingClient for the paper account, hardened as the module
    docstring says. With no keys passed, they are read from ``.env``
    (``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``), never printed or stored."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive number of seconds")
    if (api_key is None) != (secret_key is None):
        raise ValueError("pass both keys, or neither to read them from .env")
    if api_key is None:
        load_env()
        api_key, secret_key = require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY")
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True, raw_data=True)
    client._retry = 0
    client._retry_wait = 0
    client._retry_codes = []
    client._session = _PaperSession(timeout_seconds)
    _require_hardened(client)
    return client


# --- the store's word on an order ----------------------------------------------


def _stored_limit(price: float | None) -> Decimal | None:
    return None if price is None else Decimal(repr(price))


def _unsendable(conn: sqlite3.Connection, order: ApprovedOrder, now: datetime) -> str | None:
    """Why ``order`` may not be sent, in words; None when it may.

    The store is the authority: ``claim_order`` wrote the row only after the
    controls, the links and the daily cap allowed it. Here the row must
    still say exactly what the order says, must not have been sent or looked
    up before (an order is sent once, ever), and the controls are read again
    in case they changed since the claim.
    """
    try:
        row = get_stored_order(conn, order.client_order_id)
        controls = get_controls(conn)
    except StoreError as exc:
        return f"the store cannot be read ({exc})"
    if row is None:
        return "no order was claimed under this name"
    if row.decision_id is None:
        return "the stored order was not claimed through claim_order"
    pairs = (
        ("proposal", row.proposal_id, order.proposal_id),
        ("broker", row.broker, Broker.PAPER),
        ("decision", row.decision_id, order.decision_id),
        ("pre_submit decision", row.regate_decision_id, order.regate_decision_id),
        ("approval", row.approval_id, order.approval_id),
        ("instrument", row.instrument, order.instrument),
        ("symbol", row.symbol, order.symbol),
        ("side", row.side, order.side),
        ("quantity", row.quantity, order.quantity),
        ("limit price", _stored_limit(row.limit_price), order.limit_price),
        ("order type", row.order_type, order.order_type),
        ("time in force", row.time_in_force, order.time_in_force),
        ("position intent", row.position_intent, order.position_intent),
    )
    for name, stored, given in pairs:
        if stored != given:
            return f"its {name} is not the claimed order's"
    if row.status is not OrderStatus.APPROVED:
        return f"the claimed order is {row.status.value}, not approved"
    if (
        row.broker_order_id is not None
        or row.filled_quantity != 0
        or row.status_reason is not None
        or row.last_synced_at is not None
    ):
        return "the claimed order was sent or looked up before: it is never sent twice"
    if controls.kill_switch:
        return "the kill switch is on"
    if controls.halt_unknown:
        return "cannot prove trading is not halted"
    if controls.halt_until is not None and controls.halt_until > now:
        return f"trading is halted until {controls.halt_until.isoformat()}"
    return None


def _limit_request(order: ApprovedOrder) -> LimitOrderRequest:
    intent = order.position_intent
    return LimitOrderRequest(
        symbol=order.symbol,
        qty=order.quantity,
        side=AlpacaSide(order.side.value),
        time_in_force=AlpacaTimeInForce.DAY,
        limit_price=float(order.limit_price),
        client_order_id=order.client_order_id,
        position_intent=None if intent is None else AlpacaIntent(intent.value),
    )


def _receipt_mismatch(receipt: OrderReceipt, order: ApprovedOrder) -> str | None:
    pairs = (
        ("client_order_id", receipt.client_order_id, order.client_order_id),
        ("symbol", receipt.symbol, order.symbol),
        ("side", receipt.side, order.side),
        ("quantity", receipt.quantity, order.quantity),
    )
    for name, said, sent in pairs:
        if said != sent:
            return f"names {name} {said!r}, not {sent!r}"
    return None


# --- the Executor ---------------------------------------------------------------


class PaperExecutor(Executor):
    """The Executor over the Alpaca paper account.

    ``conn`` is the store the orders were claimed in, read (never written)
    before every send. ``client`` is a TradingClient from
    ``paper_trading_client``; a TradingClient that is not hardened is
    refused. Tests pass a stand-in that answers like one. ``clock`` reads
    the time for the halt check and for ``fetched_at``.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: Any,
        *,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        if isinstance(client, TradingClient):
            _require_hardened(client)
        self._conn = conn
        self._client = client
        self._clock = clock

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ExecutionError(
                "read the clock", cause=ValueError("the clock is naive"),
                outcome=ExecutionOutcome.NOT_SENT,
            )
        return now

    def _receipt(self, body: object, what: str, key: str | None) -> OrderReceipt:
        try:
            return receipt_from_broker(body, fetched_at=self._now())
        except (ValueError, ValidationError) as exc:
            raise ExecutionError(what, key, exc, outcome=ExecutionOutcome.UNKNOWN) from exc

    def submit_order(self, order: ApprovedOrder) -> OrderReceipt:
        if not isinstance(order, ApprovedOrder):
            raise ExecutionError(
                "submit order", cause=TypeError(f"not an ApprovedOrder: {type(order).__name__}"),
                outcome=ExecutionOutcome.NOT_SENT,
            )
        key = order.client_order_id
        why_not = _unsendable(self._conn, order, self._now())
        if why_not is not None:
            raise ExecutionError(
                f"submit order (refused: {why_not})", key, outcome=ExecutionOutcome.NOT_SENT
            )
        try:
            request = _limit_request(order)
        except (ValueError, ValidationError) as exc:
            raise ExecutionError(
                "submit order", key, exc, outcome=ExecutionOutcome.NOT_SENT
            ) from exc
        try:
            body = self._client.submit_order(request)
        except Exception as exc:
            raise _failed("submit order", key, exc) from exc
        receipt = self._receipt(body, "submit order", key)
        mismatch = _receipt_mismatch(receipt, order)
        if mismatch is not None:
            raise ExecutionError(
                f"submit order (the broker's receipt {mismatch})", key,
                outcome=ExecutionOutcome.UNKNOWN,
            )
        return receipt

    def cancel_order(self, broker_order_id: str) -> None:
        try:
            UUID(broker_order_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ExecutionError(
                "cancel order", str(broker_order_id), exc, outcome=ExecutionOutcome.NOT_SENT
            ) from exc
        try:
            self._client.cancel_order_by_id(broker_order_id)
            return
        except Exception as exc:
            refusal = _failed("cancel order", broker_order_id, exc)
            if refusal.outcome is not ExecutionOutcome.REJECTED:
                raise refusal from exc
        # The broker refused. If it had already ended the order, there was
        # nothing left to cancel: that is no failure.
        try:
            body = self._client.get_order_by_id(broker_order_id)
        except Exception:
            raise refusal from refusal.cause
        receipt = self._receipt(body, "cancel order", broker_order_id)
        if receipt.status is OrderStatus.CANCELLED:
            return
        raise ExecutionError(
            f"cancel order (the broker refused: the order is {receipt.broker_status})",
            broker_order_id, refusal.cause,
            outcome=ExecutionOutcome.REJECTED, status_code=refusal.status_code,
        )

    def get_open_orders(self) -> list[OrderReceipt]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=OPEN_ORDERS_PAGE, nested=False)
        try:
            body = self._client.get_orders(request)
        except Exception as exc:
            raise _failed("get open orders", None, exc) from exc
        if not isinstance(body, list):
            raise ExecutionError(
                "get open orders", cause=ValueError(f"not a list: {type(body).__name__}"),
                outcome=ExecutionOutcome.UNKNOWN,
            )
        if len(body) >= OPEN_ORDERS_PAGE:
            raise ExecutionError(
                "get open orders",
                cause=ValueError(f"{len(body)} orders fill the broker's page: more may be open"),
                outcome=ExecutionOutcome.UNKNOWN,
            )
        return [self._receipt(item, "get open orders", None) for item in body]

    def get_order(self, client_order_id: str) -> OrderReceipt | None:
        try:
            body = self._client.get_order_by_client_id(client_order_id)
        except Exception as exc:
            if isinstance(exc, APIError) and exc.status_code == 404:
                return None
            raise _failed("get order", client_order_id, exc) from exc
        receipt = self._receipt(body, "get order", client_order_id)
        if receipt.client_order_id != client_order_id:
            raise ExecutionError(
                f"get order (the broker answered for {receipt.client_order_id!r})",
                client_order_id, outcome=ExecutionOutcome.UNKNOWN,
            )
        return receipt

    def close_position(self, symbol: str) -> OrderReceipt:
        if not isinstance(symbol, str) or not is_tradable_symbol(symbol):
            raise ExecutionError(
                "close position", str(symbol), ValueError("not a ticker or an OCC option symbol"),
                outcome=ExecutionOutcome.NOT_SENT,
            )
        try:
            body = self._client.close_position(symbol)
        except Exception as exc:
            raise _failed("close position", symbol, exc) from exc
        return self._receipt(body, "close position", symbol)
