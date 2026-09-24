"""Typed repository functions over the Phase 3 store.

Every function takes an open connection (from ``open_store``) first and
speaks the frozen records in ``aegis.store.models`` in and out. SQL is
parameterized only: table and column names are literal in this file, values
travel as ``?`` parameters, and the one assembled query — the ``IN`` list of
open order statuses — is built from ``?`` marks alone. Rows are read by
column name (``sqlite3.Row``), which is what makes ``SELECT *`` safe here.

Each write runs inside ``transaction`` (``BEGIN IMMEDIATE … COMMIT``), and
every ``sqlite3.Error`` — the busy timeout expiring while another process
holds the write lock included — is re-raised as ``StoreError`` naming the
operation and the record id, so callers never see a raw ``IntegrityError``
— a foreign-key violation reads "store operation failed: add reasoning for
<id> (IntegrityError: FOREIGN KEY constraint failed)". ``transaction`` is
not re-entrant, so never call a write from inside your own transaction.

Writes return the record as stored. Plain inserts hand back the record they
were given: the models already normalise symbols and timestamps, so the row
is exactly the record. The two upserts (``record_approval``,
``upsert_order``) read the row back, because the stored row can differ from
the argument. Reads return ``None`` or an empty list for nothing found;
``get_proposal_trace`` is the one exception and raises, because asking for
the lineage of a proposal that does not exist is a caller bug worth a
clean error.

Timestamps are stored as ``datetime.isoformat()`` of an aware UTC value —
one fixed text shape (``…T14:05:00+00:00``), so text comparison against
ISO bounds (``get_daily_pnl``) is chronological. ``rules_evaluated`` and
``payload`` are ``json.dumps`` / ``json.loads``; ``raw_model_output`` is
stored verbatim and never parsed.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from aegis.data.models import utcnow
from aegis.store.db import transaction
from aegis.store.errors import StoreError
from aegis.store.models import (
    OPEN_ORDER_STATUSES,
    Approval,
    ApprovalResponse,
    Broker,
    Event,
    EventLevel,
    Fill,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PnlSnapshot,
    PolicyDecision,
    PositionSnapshot,
    Proposal,
    ProposalTrace,
    Reasoning,
    ReasoningStage,
    Verdict,
)

# What a repository call can fail with: the database itself (sqlite3.Error),
# a row that will not convert back into a record (malformed timestamp or
# JSON text, or a pydantic ValidationError — both ValueErrors), a payload
# that will not serialise to JSON (TypeError), or an int that does not fit a
# SQLite INTEGER (OverflowError: Python ints are unbounded, the column is 64
# bits). All become StoreError with the operation and id.
_REPO_ERRORS = (sqlite3.Error, ValueError, TypeError, OverflowError)

_OPEN_STATUS_VALUES: tuple[str, ...] = tuple(
    sorted(status.value for status in OPEN_ORDER_STATUSES)
)

# The one assembled query in the store: the IN list is ``?`` marks only, one
# per open status; the status values are bound as parameters like every
# other value in this file. ``test_store.py`` pins this shape.
_OPEN_ORDERS_SQL = (
    "SELECT * FROM orders WHERE status IN ("
    + ", ".join("?" for _ in _OPEN_STATUS_VALUES)
    + ") ORDER BY updated_at, rowid"
)


# --- column <-> field conversion --------------------------------------------


def _iso(value: datetime | None) -> str | None:
    """A timestamp as stored: ISO-8601 text with offset (the models make it UTC)."""
    return value.isoformat() if value is not None else None


def _from_iso(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text is not None else None


def _json(value: Any) -> str | None:
    """Strict JSON text: NaN/Infinity would be stored as tokens no non-Python reader accepts."""
    return json.dumps(value, allow_nan=False) if value is not None else None


def _from_json(text: str | None) -> Any:
    return json.loads(text) if text is not None else None


def _row_to_proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(
        id=row["id"],
        created_at=_from_iso(row["created_at"]),
        cycle_id=row["cycle_id"],
        symbol=row["symbol"],
        instrument=Instrument(row["instrument"]),
        side=OrderSide(row["side"]),
        quantity=row["quantity"],
        order_type=OrderType(row["order_type"]),
        limit_price=row["limit_price"],
        thesis=row["thesis"],
        confidence=row["confidence"],
        invalidation=row["invalidation"],
        raw_model_output=row["raw_model_output"],
        model_name=row["model_name"],
        prompt_version=row["prompt_version"],
    )


def _row_to_reasoning(row: sqlite3.Row) -> Reasoning:
    return Reasoning(
        id=row["id"],
        proposal_id=row["proposal_id"],
        stage=ReasoningStage(row["stage"]),
        created_at=_from_iso(row["created_at"]),
        content=row["content"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
    )


def _row_to_decision(row: sqlite3.Row) -> PolicyDecision:
    return PolicyDecision(
        id=row["id"],
        proposal_id=row["proposal_id"],
        decided_at=_from_iso(row["decided_at"]),
        verdict=Verdict(row["verdict"]),
        rules_evaluated=_from_json(row["rules_evaluated"]),
        failing_rule=row["failing_rule"],
        notes=row["notes"],
    )


def _row_to_approval(row: sqlite3.Row) -> Approval:
    response = row["response"]
    return Approval(
        id=row["id"],
        proposal_id=row["proposal_id"],
        requested_at=_from_iso(row["requested_at"]),
        responded_at=_from_iso(row["responded_at"]),
        response=ApprovalResponse(response) if response is not None else None,
        channel=row["channel"],
        responder=row["responder"],
    )


def _row_to_order(row: sqlite3.Row) -> Order:
    return Order(
        id=row["id"],
        proposal_id=row["proposal_id"],
        client_order_id=row["client_order_id"],
        broker=Broker(row["broker"]),
        broker_order_id=row["broker_order_id"],
        status=OrderStatus(row["status"]),
        submitted_at=_from_iso(row["submitted_at"]),
        updated_at=_from_iso(row["updated_at"]),
        symbol=row["symbol"],
        side=OrderSide(row["side"]),
        quantity=row["quantity"],
        limit_price=row["limit_price"],
    )


def _row_to_fill(row: sqlite3.Row) -> Fill:
    return Fill(
        id=row["id"],
        order_id=row["order_id"],
        filled_at=_from_iso(row["filled_at"]),
        fill_price=row["fill_price"],
        fill_quantity=row["fill_quantity"],
        fees=row["fees"],
    )


def _row_to_pnl_snapshot(row: sqlite3.Row) -> PnlSnapshot:
    return PnlSnapshot(
        id=row["id"],
        taken_at=_from_iso(row["taken_at"]),
        equity=row["equity"],
        cash=row["cash"],
        buying_power=row["buying_power"],
        daily_pnl=row["daily_pnl"],
        realized_pnl=row["realized_pnl"],
        unrealized_pnl=row["unrealized_pnl"],
    )


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        occurred_at=_from_iso(row["occurred_at"]),
        level=EventLevel(row["level"]),
        kind=row["kind"],
        message=row["message"],
        payload=_from_json(row["payload"]),
    )


# --- writes -----------------------------------------------------------------


def insert_proposal(conn: sqlite3.Connection, proposal: Proposal) -> Proposal:
    """Insert a new proposal (its id must be unused)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO proposals (id, created_at, cycle_id, symbol, instrument, side,"
                " quantity, order_type, limit_price, thesis, confidence, invalidation,"
                " raw_model_output, model_name, prompt_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal.id,
                    _iso(proposal.created_at),
                    proposal.cycle_id,
                    proposal.symbol,
                    proposal.instrument.value,
                    proposal.side.value,
                    proposal.quantity,
                    proposal.order_type.value,
                    proposal.limit_price,
                    proposal.thesis,
                    proposal.confidence,
                    proposal.invalidation,
                    proposal.raw_model_output,
                    proposal.model_name,
                    proposal.prompt_version,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("insert proposal", proposal.id, exc) from exc
    return proposal


def add_reasoning(conn: sqlite3.Connection, reasoning: Reasoning) -> Reasoning:
    """Append one agent stage's output to its proposal (which must exist)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO reasoning (id, proposal_id, stage, created_at, content,"
                " tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    reasoning.id,
                    reasoning.proposal_id,
                    reasoning.stage.value,
                    _iso(reasoning.created_at),
                    reasoning.content,
                    reasoning.tokens_in,
                    reasoning.tokens_out,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("add reasoning", reasoning.id, exc) from exc
    return reasoning


def record_decision(conn: sqlite3.Connection, decision: PolicyDecision) -> PolicyDecision:
    """Record the policy engine's verdict on a proposal (which must exist)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO policy_decisions (id, proposal_id, decided_at, verdict,"
                " rules_evaluated, failing_rule, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.id,
                    decision.proposal_id,
                    _iso(decision.decided_at),
                    decision.verdict.value,
                    _json(decision.rules_evaluated),
                    decision.failing_rule,
                    decision.notes,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("record decision", decision.id, exc) from exc
    return decision


def record_approval(conn: sqlite3.Connection, approval: Approval) -> Approval:
    """Insert an approval request, or fill in the response of one already recorded.

    Call it when the request goes out and again, with the same id, once it is
    answered: the second call updates only ``responded_at``, ``response`` and
    ``responder`` — the request itself (proposal, requested_at, channel) is
    immutable. Exactly one row per approval id. Returns the row as stored.
    """
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO approvals (id, proposal_id, requested_at, responded_at, response,"
                " channel, responder) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET"
                " responded_at = excluded.responded_at,"
                " response = excluded.response,"
                " responder = excluded.responder",
                (
                    approval.id,
                    approval.proposal_id,
                    _iso(approval.requested_at),
                    _iso(approval.responded_at),
                    approval.response.value if approval.response is not None else None,
                    approval.channel,
                    approval.responder,
                ),
            )
            row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval.id,)).fetchone()
            stored = _row_to_approval(row)
    except _REPO_ERRORS as exc:
        raise StoreError("record approval", approval.id, exc) from exc
    return stored


def upsert_order(conn: sqlite3.Connection, order: Order) -> Order:
    """Insert an order, or update the one that already has its ``client_order_id``.

    ``client_order_id`` is the idempotency key: calling this twice for the
    same one produces exactly one row, always. On a repeat the row takes the
    new ``status``, ``updated_at``, ``quantity`` and ``limit_price``, and
    ``broker_order_id`` / ``submitted_at`` only when the new value is not
    None (so a later status update cannot erase them). The returned Order is
    the row as stored, so it keeps the ORIGINAL ``id``, ``proposal_id``,
    ``broker``, ``symbol`` and ``side`` of the first insert — not the ones on
    the argument.
    """
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO orders (id, proposal_id, client_order_id, broker, broker_order_id,"
                " status, submitted_at, updated_at, symbol, side, quantity, limit_price)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (client_order_id) DO UPDATE SET"
                " broker_order_id = COALESCE(excluded.broker_order_id, orders.broker_order_id),"
                " status = excluded.status,"
                " submitted_at = COALESCE(excluded.submitted_at, orders.submitted_at),"
                " updated_at = excluded.updated_at,"
                " limit_price = excluded.limit_price,"
                " quantity = excluded.quantity",
                (
                    order.id,
                    order.proposal_id,
                    order.client_order_id,
                    order.broker.value,
                    order.broker_order_id,
                    order.status.value,
                    _iso(order.submitted_at),
                    _iso(order.updated_at),
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.limit_price,
                ),
            )
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (order.client_order_id,)
            ).fetchone()
            stored = _row_to_order(row)
    except _REPO_ERRORS as exc:
        raise StoreError("upsert order", order.client_order_id, exc) from exc
    return stored


def record_fill(conn: sqlite3.Connection, fill: Fill) -> Fill:
    """Record one execution of an order (which must exist)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fill.id,
                    fill.order_id,
                    _iso(fill.filled_at),
                    fill.fill_price,
                    fill.fill_quantity,
                    fill.fees,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("record fill", fill.id, exc) from exc
    return fill


def snapshot_positions(
    conn: sqlite3.Connection, snapshots: Sequence[PositionSnapshot]
) -> list[PositionSnapshot]:
    """Record one cycle's position snapshots in a single transaction: all or none."""
    stored = list(snapshots)
    if not stored:
        return stored
    snapshot = stored[0]  # the row being written when a failure hits
    try:
        with transaction(conn):
            for snapshot in stored:
                conn.execute(
                    "INSERT INTO position_snapshots (id, taken_at, symbol, quantity, avg_cost,"
                    " market_value, unrealized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.id,
                        _iso(snapshot.taken_at),
                        snapshot.symbol,
                        snapshot.quantity,
                        snapshot.avg_cost,
                        snapshot.market_value,
                        snapshot.unrealized_pnl,
                    ),
                )
    except _REPO_ERRORS as exc:
        raise StoreError("snapshot positions", snapshot.id, exc) from exc
    return stored


def snapshot_pnl(conn: sqlite3.Connection, snapshot: PnlSnapshot) -> PnlSnapshot:
    """Record the account-level P&L as of ``snapshot.taken_at``."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO pnl_snapshots (id, taken_at, equity, cash, buying_power, daily_pnl,"
                " realized_pnl, unrealized_pnl) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot.id,
                    _iso(snapshot.taken_at),
                    snapshot.equity,
                    snapshot.cash,
                    snapshot.buying_power,
                    snapshot.daily_pnl,
                    snapshot.realized_pnl,
                    snapshot.unrealized_pnl,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("snapshot pnl", snapshot.id, exc) from exc
    return snapshot


def log_event(conn: sqlite3.Connection, event: Event) -> Event:
    """Append an operational event (risk-limit trip, kill switch, heartbeat, error)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO events (id, occurred_at, level, kind, message, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.id,
                    _iso(event.occurred_at),
                    event.level.value,
                    event.kind,
                    event.message,
                    _json(event.payload),
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("log event", event.id, exc) from exc
    return event


# --- reads ------------------------------------------------------------------


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> Proposal | None:
    try:
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row_to_proposal(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get proposal", proposal_id, exc) from exc


def get_order(conn: sqlite3.Connection, client_order_id: str) -> Order | None:
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return _row_to_order(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get order", client_order_id, exc) from exc


def get_open_orders(conn: sqlite3.Connection) -> list[Order]:
    """Every order in an open status (``OPEN_ORDER_STATUSES``), oldest update first."""
    try:
        rows = conn.execute(_OPEN_ORDERS_SQL, _OPEN_STATUS_VALUES).fetchall()
        return [_row_to_order(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get open orders", cause=exc) from exc


def get_daily_pnl(conn: sqlite3.Connection, day: date | None = None) -> PnlSnapshot | None:
    """The latest P&L snapshot taken on ``day`` (a UTC calendar day; default today), or None.

    Bounds are ``[day 00:00Z, next day 00:00Z)`` as ISO text; every stored
    timestamp has the same UTC text shape, so the text comparison is
    chronological. A ``datetime`` (a ``date`` subclass) means the UTC
    calendar day of that instant — naive taken as UTC, any other offset
    converted, as the records themselves are — never its local date.
    """
    if day is None:
        day = utcnow().date()
    elif isinstance(day, datetime):
        aware = day if day.tzinfo is not None else day.replace(tzinfo=timezone.utc)
        day = aware.astimezone(timezone.utc).date()
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    try:
        row = conn.execute(
            "SELECT * FROM pnl_snapshots WHERE taken_at >= ? AND taken_at < ?"
            " ORDER BY taken_at DESC, rowid DESC LIMIT 1",
            (start.isoformat(), end.isoformat()),
        ).fetchone()
        return _row_to_pnl_snapshot(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get daily pnl", day.isoformat(), exc) from exc


def get_recent_events(
    conn: sqlite3.Connection, limit: int = 50, level: EventLevel | None = None
) -> list[Event]:
    """The ``limit`` newest events, newest first, optionally only one level."""
    try:
        if level is None:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY occurred_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events WHERE level = ?"
                " ORDER BY occurred_at DESC, rowid DESC LIMIT ?",
                (level.value, limit),
            ).fetchall()
        return [_row_to_event(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get recent events", level.value if level else None, exc) from exc


def get_proposal_trace(conn: sqlite3.Connection, proposal_id: str) -> ProposalTrace:
    """The full lineage of one proposal in one call; StoreError if it does not exist.

    Reasoning is ordered by created_at (then rowid), decisions by decided_at,
    approvals by requested_at, orders by updated_at and fills — every fill
    of every order of the proposal — by filled_at, all oldest first.
    """
    try:
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise StoreError("proposal trace (no such proposal)", proposal_id)
        key = (proposal_id,)
        reasoning = conn.execute(
            "SELECT * FROM reasoning WHERE proposal_id = ? ORDER BY created_at, rowid", key
        ).fetchall()
        decisions = conn.execute(
            "SELECT * FROM policy_decisions WHERE proposal_id = ? ORDER BY decided_at, rowid", key
        ).fetchall()
        approvals = conn.execute(
            "SELECT * FROM approvals WHERE proposal_id = ? ORDER BY requested_at, rowid", key
        ).fetchall()
        orders = conn.execute(
            "SELECT * FROM orders WHERE proposal_id = ? ORDER BY updated_at, rowid", key
        ).fetchall()
        fills = conn.execute(
            "SELECT * FROM fills WHERE order_id IN (SELECT id FROM orders WHERE proposal_id = ?)"
            " ORDER BY filled_at, rowid",
            key,
        ).fetchall()
        return ProposalTrace(
            proposal=_row_to_proposal(row),
            reasoning=tuple(_row_to_reasoning(r) for r in reasoning),
            decisions=tuple(_row_to_decision(r) for r in decisions),
            approvals=tuple(_row_to_approval(r) for r in approvals),
            orders=tuple(_row_to_order(r) for r in orders),
            fills=tuple(_row_to_fill(r) for r in fills),
        )
    except _REPO_ERRORS as exc:
        raise StoreError("proposal trace", proposal_id, exc) from exc
