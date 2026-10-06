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
the argument. ``insert_proposal`` writes the proposal and its option legs in
one transaction — and, asked to, links its cycle's reasoning rows in that
same transaction; ``link_reasoning_to_proposal`` is that UPDATE on its own
(a cycle's reasoning rows are written before its proposal exists) and
returns a row count. Reads return ``None`` or an empty list for nothing found;
``get_proposal_trace`` is the one exception and raises, because asking for
the lineage of a proposal that does not exist is a caller bug worth a
clean error. The token reads (``get_token_usage`` and friends) are what the
brain's budget guard measures against.

The ``controls`` table (migration 0003) is the one thing here that is not a
journal: two fixed rows, the kill switch and the daily-loss halt.
``set_kill_switch`` / ``set_halt_until`` upsert a row and return all the
controls as read back in that transaction (``set_halt_until(...,
only_extend=True)`` is a compare-and-set: it never shortens a stored
halt); ``get_controls`` reads them and fails closed — a row that is missing
or will not parse never raises, it reads as "kill switch on" / "halt
unknown" with the reason in ``problems``, because a gate that cannot read
its own stop flags must stay shut. Those two writers and
``record_decision`` take an optional ``event``: the audit event is inserted
in the same transaction as the write it describes, so a control change (or
a verdict) and its event land together or not at all.

Timestamps are stored as ``datetime.isoformat()`` of an aware UTC value —
one fixed text shape (``…T14:05:00+00:00``), so text comparison against
ISO bounds (``get_daily_pnl``, ``get_token_usage``) is chronological; a leg's
``expiration`` is ``date.isoformat()``. ``rules_evaluated`` and ``payload``
are ``json.dumps`` / ``json.loads``; ``raw_model_output`` is stored verbatim
and never parsed.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from aegis.data.models import OptionType, utcnow
from aegis.store.db import transaction
from aegis.store.errors import ClaimRefused, StoreError
from aegis.store.models import (
    OPEN_ORDER_STATUSES,
    TERMINAL_ORDER_STATUSES,
    Approval,
    ApprovalResponse,
    Broker,
    Controls,
    DecisionPurpose,
    Event,
    EventLevel,
    Fill,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionIntent,
    PnlSnapshot,
    PolicyDecision,
    PositionSnapshot,
    Proposal,
    ProposalLeg,
    ProposalTrace,
    Reasoning,
    ReasoningStage,
    TimeInForce,
    TokenUsage,
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

# The two rows of the ``controls`` table (migration 0003) and the kill
# switch's two values; ``halt_until`` holds '' or an ISO-8601 instant.
_KILL_SWITCH = "kill_switch"
_HALT_UNTIL = "halt_until"
_SWITCH_ON = "on"
_SWITCH_OFF = "off"

# How much of an unreadable stored value a ``Controls.problems`` line quotes.
_SHOWN_MAX = 60


# --- column <-> field conversion --------------------------------------------


def _iso(value: datetime | None) -> str | None:
    """A timestamp as stored: ISO-8601 text with offset (the models make it UTC)."""
    return value.isoformat() if value is not None else None


def _from_iso(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text is not None else None


def _utc(value: datetime, name: str = "timestamp") -> datetime:
    """A caller's instant as the store compares it: naive is taken as UTC, any
    other offset converted — what the models do to a record's own timestamps.

    Anything but a ``datetime`` is a TypeError, which every caller turns
    into a StoreError like the rest of ``_REPO_ERRORS``: a bare ``date`` or
    ISO text must not reach a query as a bound it was never meant to be.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, not {type(value).__name__}")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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


def _row_to_leg(row: sqlite3.Row) -> ProposalLeg:
    return ProposalLeg(
        id=row["id"],
        proposal_id=row["proposal_id"],
        leg_index=row["leg_index"],
        symbol=row["symbol"],
        option_type=OptionType(row["option_type"]),
        side=OrderSide(row["side"]),
        quantity=row["quantity"],
        strike=row["strike"],
        expiration=date.fromisoformat(row["expiration"]),
    )


def _row_to_reasoning(row: sqlite3.Row) -> Reasoning:
    return Reasoning(
        id=row["id"],
        cycle_id=row["cycle_id"],
        proposal_id=row["proposal_id"],
        stage=ReasoningStage(row["stage"]),
        created_at=_from_iso(row["created_at"]),
        content=row["content"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        model_name=row["model_name"],
        latency_ms=row["latency_ms"],
    )


def _usage(tokens_in: Any, tokens_out: Any, calls: Any) -> TokenUsage:
    """One ``SUM(tokens_in), SUM(tokens_out), COUNT(*)`` row (the sums COALESCEd to 0)."""
    return TokenUsage(tokens_in=int(tokens_in), tokens_out=int(tokens_out), calls=int(calls))


def _row_to_decision(row: sqlite3.Row) -> PolicyDecision:
    return PolicyDecision(
        id=row["id"],
        proposal_id=row["proposal_id"],
        decided_at=_from_iso(row["decided_at"]),
        verdict=Verdict(row["verdict"]),
        rules_evaluated=_from_json(row["rules_evaluated"]),
        failing_rule=row["failing_rule"],
        notes=row["notes"],
        purpose=DecisionPurpose(row["purpose"]),
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
        decision_id=row["decision_id"],
        expires_at=_from_iso(row["expires_at"]),
        note=row["note"],
    )


def _enum_or_none(enum: type, value: Any) -> Any:
    return enum(value) if value is not None else None


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
        decision_id=row["decision_id"],
        regate_decision_id=row["regate_decision_id"],
        approval_id=row["approval_id"],
        instrument=_enum_or_none(Instrument, row["instrument"]),
        order_type=_enum_or_none(OrderType, row["order_type"]),
        time_in_force=_enum_or_none(TimeInForce, row["time_in_force"]),
        position_intent=_enum_or_none(PositionIntent, row["position_intent"]),
        filled_quantity=row["filled_quantity"],
        avg_fill_price=row["avg_fill_price"],
        broker_status=row["broker_status"],
        status_reason=row["status_reason"],
        last_synced_at=_from_iso(row["last_synced_at"]),
    )


def _row_to_fill(row: sqlite3.Row) -> Fill:
    return Fill(
        id=row["id"],
        order_id=row["order_id"],
        filled_at=_from_iso(row["filled_at"]),
        fill_price=row["fill_price"],
        fill_quantity=row["fill_quantity"],
        fees=row["fees"],
        broker_fill_id=row["broker_fill_id"],
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


def insert_proposal(
    conn: sqlite3.Connection,
    proposal: Proposal,
    legs: Sequence[ProposalLeg] = (),
    *,
    link_cycle_reasoning: bool = False,
) -> Proposal:
    """Insert a new proposal (its id must be unused) with its option legs: all or none.

    Every leg must carry ``proposal.id`` as its ``proposal_id`` — a leg of
    another proposal is a StoreError before anything is written. An equity
    proposal passes no legs. With ``link_cycle_reasoning`` the proposal's
    cycle's not-yet-linked reasoning rows are attached to it in the same
    transaction (``link_reasoning_to_proposal``'s UPDATE), so a proposal
    never lands without its reasoning, nor reasoning linked to a proposal
    that failed to land.
    """
    for leg in legs:
        if leg.proposal_id != proposal.id:
            raise StoreError(
                f"insert proposal (leg {leg.leg_index} belongs to proposal {leg.proposal_id})",
                proposal.id,
            )
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
            for leg in legs:
                conn.execute(
                    "INSERT INTO proposal_legs (id, proposal_id, leg_index, symbol, option_type,"
                    " side, quantity, strike, expiration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        leg.id,
                        leg.proposal_id,
                        leg.leg_index,
                        leg.symbol,
                        leg.option_type.value,
                        leg.side.value,
                        leg.quantity,
                        leg.strike,
                        leg.expiration.isoformat(),
                    ),
                )
            if link_cycle_reasoning:  # link_reasoning_to_proposal's UPDATE, in this transaction
                conn.execute(
                    "UPDATE reasoning SET proposal_id = ?"
                    " WHERE cycle_id = ? AND proposal_id IS NULL",
                    (proposal.id, proposal.cycle_id),
                )
    except _REPO_ERRORS as exc:
        raise StoreError("insert proposal", proposal.id, exc) from exc
    return proposal


def add_reasoning(conn: sqlite3.Connection, reasoning: Reasoning) -> Reasoning:
    """Append one agent stage's output under its cycle and/or its proposal (which must exist)."""
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO reasoning (id, cycle_id, proposal_id, stage, created_at, content,"
                " tokens_in, tokens_out, model_name, latency_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    reasoning.id,
                    reasoning.cycle_id,
                    reasoning.proposal_id,
                    reasoning.stage.value,
                    _iso(reasoning.created_at),
                    reasoning.content,
                    reasoning.tokens_in,
                    reasoning.tokens_out,
                    reasoning.model_name,
                    reasoning.latency_ms,
                ),
            )
    except _REPO_ERRORS as exc:
        raise StoreError("add reasoning", reasoning.id, exc) from exc
    return reasoning


def link_reasoning_to_proposal(conn: sqlite3.Connection, cycle_id: str, proposal_id: str) -> int:
    """Attach a cycle's not-yet-linked reasoning rows to the proposal it produced.

    Sets ``proposal_id`` on every reasoning row of ``cycle_id`` whose
    ``proposal_id`` is still NULL — rows already linked (to this proposal or
    another) are left alone — and returns how many rows changed (0 when the
    cycle has none). The proposal must exist: the foreign key fails
    otherwise and the StoreError names both ids.
    """
    try:
        with transaction(conn):
            cursor = conn.execute(
                "UPDATE reasoning SET proposal_id = ? WHERE cycle_id = ? AND proposal_id IS NULL",
                (proposal_id, cycle_id),
            )
            linked = cursor.rowcount
    except _REPO_ERRORS as exc:
        raise StoreError(
            "link reasoning to proposal", f"cycle {cycle_id} -> proposal {proposal_id}", exc
        ) from exc
    return linked


def _insert_event(conn: sqlite3.Connection, event: Event) -> None:
    """The ``events`` INSERT on its own, for a caller that already holds the transaction.

    ``log_event`` is this inside a transaction of its own. The writers that
    take an ``event`` (``record_decision``, ``set_kill_switch``,
    ``set_halt_until``) run it inside theirs, so the event and the write it
    describes commit together or roll back together. Raises what SQLite and
    the JSON encoder raise; the caller wraps it.
    """
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


def record_decision(
    conn: sqlite3.Connection, decision: PolicyDecision, *, event: Event | None = None
) -> PolicyDecision:
    """Record the policy engine's verdict on a proposal (which must exist).

    ``event``, when given, is inserted in the same transaction: the decision
    and the event announcing it land together or not at all. An event that
    cannot be written (a reused id, a payload JSON cannot carry) rolls the
    decision back, and the StoreError names the decision.
    """
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO policy_decisions (id, proposal_id, decided_at, verdict,"
                " rules_evaluated, failing_rule, notes, purpose) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.id,
                    decision.proposal_id,
                    _iso(decision.decided_at),
                    decision.verdict.value,
                    _json(decision.rules_evaluated),
                    decision.failing_rule,
                    decision.notes,
                    decision.purpose.value,
                ),
            )
            if event is not None:
                _insert_event(conn, event)
    except _REPO_ERRORS as exc:
        raise StoreError("record decision", decision.id, exc) from exc
    return decision


def record_approval(conn: sqlite3.Connection, approval: Approval) -> Approval:
    """Insert an approval request, or fill in the response of one already recorded.

    Call it when the request goes out and again, with the same id, once it is
    answered: the second call updates only ``responded_at``, ``response``,
    ``responder`` and ``note`` — the request itself (proposal, requested_at,
    channel, and since Phase 6 its ``decision_id`` and ``expires_at``) is
    immutable. Exactly one row per approval id. Returns the row as stored.

    An approval tied to a decision (``decision_id``) is answered once:
    migration 0004 refuses to change an answer already recorded, and allows
    one approval per decision.
    """
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO approvals (id, proposal_id, requested_at, responded_at, response,"
                " channel, responder, decision_id, expires_at, note)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET"
                " responded_at = excluded.responded_at,"
                " response = excluded.response,"
                " responder = excluded.responder,"
                " note = excluded.note",
                (
                    approval.id,
                    approval.proposal_id,
                    _iso(approval.requested_at),
                    _iso(approval.responded_at),
                    approval.response.value if approval.response is not None else None,
                    approval.channel,
                    approval.responder,
                    approval.decision_id,
                    _iso(approval.expires_at),
                    approval.note,
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

    It writes orders as Phase 3 defined them, and nothing else. An
    execution-era order (one with a ``decision_id``, written by
    ``claim_order``) is refused, both as the argument and as the row the
    ``client_order_id`` already names: those change only through
    ``claim_order`` and ``apply_broker_update``. The Phase 6 columns of an
    argument are not written.
    """
    try:
        if _execution_links(order):
            raise ValueError("an order with decision links is written by claim_order only")
        with transaction(conn):
            existing = conn.execute(
                "SELECT decision_id FROM orders WHERE client_order_id = ?",
                (order.client_order_id,),
            ).fetchone()
            if existing is not None and existing["decision_id"] is not None:
                raise ValueError(
                    "the order is execution-era: it changes through apply_broker_update only"
                )
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
    """Record one execution of an order (which must exist).

    For orders as Phase 3 defined them. The fills of an execution-era order
    (one with a ``decision_id``) are written by ``apply_broker_update`` only,
    so they always add up to the order's filled quantity; such an order is
    refused here.
    """
    try:
        with transaction(conn):
            owner = conn.execute(
                "SELECT decision_id FROM orders WHERE id = ?", (fill.order_id,)
            ).fetchone()
            if owner is not None and owner["decision_id"] is not None:
                raise ValueError(
                    "the order is execution-era: its fills are written by apply_broker_update only"
                )
            conn.execute(
                "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees,"
                " broker_fill_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fill.id,
                    fill.order_id,
                    _iso(fill.filled_at),
                    fill.fill_price,
                    fill.fill_quantity,
                    fill.fees,
                    fill.broker_fill_id,
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
            _insert_event(conn, event)
    except _REPO_ERRORS as exc:
        raise StoreError("log event", event.id, exc) from exc
    return event


def _write_control(
    conn: sqlite3.Connection,
    key: str,
    value: str,
    now: datetime | None,
    event: Event | None,
    *,
    not_before: datetime | None = None,
) -> Controls:
    """Upsert one ``controls`` row, with its audit event, and read all the controls back.

    One transaction: the row, the event (when given) and the read-back. The
    row normally exists (the migration seeds it and nothing may delete it),
    so this is an UPDATE in practice; the INSERT half restores a row that
    was removed by hand. Raises raw — the two public writers wrap it.

    ``not_before`` makes the write a compare-and-set, decided inside the
    transaction (``BEGIN IMMEDIATE`` holds the write lock, so no other
    writer can slip in between the read and the write): when the stored
    halt is readable and already at or after that instant, neither the row
    nor the event is written, and the controls come back as they stand.
    """
    updated_at = _iso(_utc(now if now is not None else utcnow(), "now"))
    with transaction(conn):
        if not_before is not None:
            current = _read_controls(conn)
            in_force = current.halt_until
            if not current.halt_unknown and in_force is not None and in_force >= not_before:
                return current
        conn.execute(
            "INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT (key) DO UPDATE SET"
            " value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, updated_at),
        )
        if event is not None:
            _insert_event(conn, event)
        stored = _read_controls(conn)
    return stored


def set_kill_switch(
    conn: sqlite3.Connection,
    on: bool,
    *,
    now: datetime | None = None,
    event: Event | None = None,
) -> Controls:
    """Turn the kill switch on or off; returns the controls as read back.

    ``now`` (default ``utcnow()``; naive taken as UTC) is recorded as the
    row's ``updated_at``. ``event``, when given, is inserted in the same
    transaction, so the change and its audit event land together or not at
    all. ``on`` must be a real bool: the truthiness of anything else (the
    string "off" is truthy) is not a decision this function will take.
    """
    try:
        if not isinstance(on, bool):
            raise TypeError(f"on must be a bool, not {type(on).__name__}")
        return _write_control(conn, _KILL_SWITCH, _SWITCH_ON if on else _SWITCH_OFF, now, event)
    except _REPO_ERRORS as exc:
        raise StoreError("set kill switch", _KILL_SWITCH, exc) from exc


def set_halt_until(
    conn: sqlite3.Connection,
    until: datetime | None,
    *,
    now: datetime | None = None,
    event: Event | None = None,
    only_extend: bool = False,
) -> Controls:
    """Halt trading until ``until``, or clear the halt with None; returns the
    controls as read back.

    ``until`` is stored as the ISO text of an aware UTC instant (naive taken
    as UTC, any other offset converted); None is stored as ''. By default a
    plain upsert: the operator may shorten or clear a halt. ``now`` and
    ``event`` as in ``set_kill_switch``.

    ``only_extend`` is how the policy engine writes: a halt is extended,
    never shortened or re-announced. The stored halt is read inside the
    same transaction as the write, and when it is already at or after
    ``until`` nothing is written — not the row, not ``event`` — whatever
    the caller's own snapshot of the controls said. No halt on record, an
    earlier or expired one, and one that cannot be read are all replaced.
    It needs an instant: ``only_extend`` with ``until=None`` is refused.
    """
    try:
        if not isinstance(only_extend, bool):
            raise TypeError(f"only_extend must be a bool, not {type(only_extend).__name__}")
        if only_extend and until is None:
            raise ValueError("only_extend needs an instant to extend the halt to, not None")
        instant = None if until is None else _utc(until, "until")
        value = "" if instant is None else instant.isoformat()
        return _write_control(
            conn, _HALT_UNTIL, value, now, event, not_before=instant if only_extend else None
        )
    except _REPO_ERRORS as exc:
        raise StoreError("set halt until", _HALT_UNTIL, exc) from exc


# --- execution (Phase 6) ----------------------------------------------------
#
# An execution-era order is born in claim_order and moved on only by
# apply_broker_update; migration 0004's triggers hold the same rules under
# both (status only forward, terminal rows frozen, filled quantity only
# growing, identity fixed). See docs/phase6/SPEC_PHASE6.md.

_CLAIMABLE_VERDICTS = (Verdict.AUTO_EXECUTE, Verdict.NEEDS_APPROVAL)


def _execution_links(order: Order) -> bool:
    return any(
        link is not None for link in (order.decision_id, order.regate_decision_id, order.approval_id)
    )


def _check_claimable(order: Order) -> None:
    """What ``claim_order`` needs of its argument; a caller bug is a ValueError."""
    if order.decision_id is None or order.regate_decision_id is None:
        raise ValueError("a claimed order names its decision and its pre_submit decision")
    if order.status is not OrderStatus.APPROVED:
        raise ValueError(f"a claimed order is born approved, not {order.status.value}")
    if order.broker_order_id is not None or order.filled_quantity != 0:
        raise ValueError("a claimed order has no broker id and nothing filled yet")
    if order.instrument is None or order.order_type is not OrderType.LIMIT:
        raise ValueError("a claimed order names its instrument and is a limit order")
    if order.time_in_force is None:
        raise ValueError("a claimed order names its time in force")
    if order.limit_price is None or not order.limit_price > 0:
        raise ValueError("a claimed limit order has a positive limit price")
    if (order.instrument is Instrument.OPTION) != (order.position_intent is not None):
        raise ValueError("an option order names its position intent, and an equity order none")


def _decision_row(conn: sqlite3.Connection, decision_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM policy_decisions WHERE id = ?", (decision_id,)
    ).fetchone()


def _claim_refusal(conn: sqlite3.Connection, order: Order, now: datetime) -> str | None:
    """Why the order's own links do not authorise it, in words; None when they do.

    Read inside the claim's transaction. The decision must be a verdict
    (``evaluate``) on this proposal that may lead to an order; the
    ``pre_submit`` decision a re-evaluation of the same proposal that still
    allows one (AUTO_EXECUTE or NEEDS_APPROVAL); and a NEEDS_APPROVAL
    verdict needs the approval of that very decision, on this proposal,
    answered ``approved`` and not expired at ``now``. An AUTO_EXECUTE verdict
    rests on no approval, so its re-check must be AUTO_EXECUTE too. Whether
    an approved order's re-check escalates only on what the human saw is
    the dispatcher's call (spec D3), not the store's.
    """
    decision = _decision_row(conn, order.decision_id)
    if decision is None:
        return f"decision {order.decision_id} does not exist"
    if decision["proposal_id"] != order.proposal_id:
        return f"decision {order.decision_id} is not a decision on proposal {order.proposal_id}"
    if decision["purpose"] != DecisionPurpose.EVALUATE.value:
        return f"decision {order.decision_id} is a {decision['purpose']} decision, not a verdict"
    verdict = Verdict(decision["verdict"])
    if verdict not in _CLAIMABLE_VERDICTS:
        return f"decision {order.decision_id} is {verdict.value}: no order may follow it"
    regate = _decision_row(conn, order.regate_decision_id)
    if regate is None:
        return f"pre_submit decision {order.regate_decision_id} does not exist"
    if regate["proposal_id"] != order.proposal_id:
        return f"pre_submit decision {order.regate_decision_id} is on another proposal"
    if regate["purpose"] != DecisionPurpose.PRE_SUBMIT.value:
        return f"decision {order.regate_decision_id} is not a pre_submit decision"
    recheck = Verdict(regate["verdict"])
    if recheck not in _CLAIMABLE_VERDICTS:
        return f"the pre_submit re-check {order.regate_decision_id} is {recheck.value}"
    if verdict is Verdict.AUTO_EXECUTE:
        if recheck is not Verdict.AUTO_EXECUTE:
            return (
                f"the pre_submit re-check {order.regate_decision_id} is {recheck.value}:"
                " the order no longer goes without approval"
            )
        if order.approval_id is not None:
            return "an AUTO_EXECUTE order rests on no approval"
        return None
    if order.approval_id is None:
        return f"decision {order.decision_id} needs approval, and the order names none"
    approval = conn.execute(
        "SELECT * FROM approvals WHERE id = ?", (order.approval_id,)
    ).fetchone()
    if approval is None:
        return f"approval {order.approval_id} does not exist"
    if approval["decision_id"] != order.decision_id or approval["proposal_id"] != order.proposal_id:
        return f"approval {order.approval_id} is not the approval of decision {order.decision_id}"
    if approval["response"] != ApprovalResponse.APPROVED.value:
        return f"approval {order.approval_id} is {approval['response'] or 'unanswered'}"
    expires_at = _from_iso(approval["expires_at"])
    if expires_at is None:
        return f"approval {order.approval_id} has no expiry"
    if not now < expires_at:
        return f"approval {order.approval_id} has expired"
    return None


def claim_order(
    conn: sqlite3.Connection,
    order: Order,
    *,
    now: datetime,
    day_start: datetime,
    day_end: datetime,
    max_daily_trades: int,
    event: Event | None = None,
) -> Order:
    """Claim one order before it is sent: the last check and the first record, in one transaction.

    Inside one ``BEGIN IMMEDIATE`` (so no other writer moves in between):

    1. the controls, read fresh, must be clear: the kill switch off, no halt
       in force at ``now``, and the halt readable (an unreadable one cannot
       prove trading is not halted);
    2. the order's links must authorise it (``_claim_refusal``);
    3. the proposal has no execution-era order yet (at most one, ever);
    4. the orders submitted in ``[day_start, day_end)`` (naive taken as
       UTC; the window must contain ``now``), failed ones aside, must be
       fewer than ``max_daily_trades``;
    5. the row is inserted ``approved``, with ``submitted_at`` and
       ``updated_at`` set to ``now``, so it counts toward the cap from this
       instant whatever happens next; and ``event``, when given, with it.

    Any of 1-4 failing raises ``ClaimRefused`` with the reason and writes
    nothing. A malformed argument (no decision links, a market order, a
    status other than approved, ...) is a caller bug: ``StoreError``.
    Returns the row as stored.
    """
    try:
        _check_claimable(order)
        if isinstance(max_daily_trades, bool) or not isinstance(max_daily_trades, int):
            raise TypeError("max_daily_trades must be an int")
        instant = _utc(now, "now")
        start, end = _utc(day_start, "day_start"), _utc(day_end, "day_end")
        # A window that does not hold the claim would count nothing, and a
        # cap that counts nothing admits everything: a caller bug, refused.
        if not start <= instant < end:
            raise ValueError("the trading day [day_start, day_end) must contain now")
        with transaction(conn):
            controls = _read_controls(conn)
            if controls.kill_switch:
                raise ClaimRefused(order.client_order_id, "the kill switch is on")
            if controls.halt_unknown:
                raise ClaimRefused(order.client_order_id, "cannot prove trading is not halted")
            if controls.halt_until is not None and controls.halt_until > instant:
                raise ClaimRefused(
                    order.client_order_id,
                    f"trading is halted until {controls.halt_until.isoformat()}",
                )
            why_not = _claim_refusal(conn, order, instant)
            if why_not is not None:
                raise ClaimRefused(order.client_order_id, why_not)
            existing = conn.execute(
                "SELECT client_order_id FROM orders WHERE proposal_id = ? AND decision_id IS NOT NULL",
                (order.proposal_id,),
            ).fetchone()
            if existing is not None:
                raise ClaimRefused(
                    order.client_order_id,
                    f"proposal {order.proposal_id} already has order {existing['client_order_id']}",
                )
            # The rule count_orders_submitted_between counts by: an order claimed
            # but not yet acknowledged counts (the claim sets submitted_at), a
            # definite rejection (failed) does not.
            sent = int(
                conn.execute(
                    "SELECT COUNT(*) FROM orders"
                    " WHERE submitted_at >= ? AND submitted_at < ? AND status != ?",
                    (_iso(start), _iso(end), OrderStatus.FAILED.value),
                ).fetchone()[0]
            )
            if sent >= max_daily_trades:
                raise ClaimRefused(
                    order.client_order_id,
                    f"{sent} orders sent today: at or above the cap of {max_daily_trades}",
                )
            conn.execute(
                "INSERT INTO orders (id, proposal_id, client_order_id, broker, broker_order_id,"
                " status, submitted_at, updated_at, symbol, side, quantity, limit_price,"
                " decision_id, regate_decision_id, approval_id, instrument, order_type,"
                " time_in_force, position_intent, filled_quantity, avg_fill_price,"
                " broker_status, status_reason, last_synced_at)"
                " VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL,"
                " NULL, NULL, NULL)",
                (
                    order.id,
                    order.proposal_id,
                    order.client_order_id,
                    order.broker.value,
                    OrderStatus.APPROVED.value,
                    _iso(instant),
                    _iso(instant),
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.limit_price,
                    order.decision_id,
                    order.regate_decision_id,
                    order.approval_id,
                    order.instrument.value,
                    order.order_type.value,
                    order.time_in_force.value,
                    order.position_intent.value if order.position_intent is not None else None,
                ),
            )
            if event is not None:
                _insert_event(conn, event)
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (order.client_order_id,)
            ).fetchone()
            stored = _row_to_order(row)
    except _REPO_ERRORS as exc:
        raise StoreError("claim order", order.client_order_id, exc) from exc
    return stored


def _fill_id(broker_order_id: str, cumulative: float) -> str:
    """The broker fill id of the execution that took an order to ``cumulative``
    filled: the same order and cumulative quantity always give the same id,
    so replaying a sync records nothing twice. ``repr`` is exact: two
    different totals never share a name, however close."""
    return f"{broker_order_id}:{cumulative!r}"


def apply_broker_update(
    conn: sqlite3.Connection,
    client_order_id: str,
    *,
    status: OrderStatus,
    synced_at: datetime,
    broker_order_id: str | None = None,
    broker_status: str | None = None,
    filled_quantity: float | None = None,
    avg_fill_price: float | None = None,
    status_reason: str | None = None,
    event: Event | None = None,
) -> tuple[Order, Fill | None]:
    """Record what the broker says about an execution-era order; returns the
    order as stored and the fill this update added, if any. One transaction:
    the order, the fill and ``event`` land together or not at all.

    The broker reports cumulative figures. ``filled_quantity`` (None: as
    stored) is the total filled so far and ``avg_fill_price`` its average
    price; when the total grows, the difference is one new fill, priced so
    that the fills' value adds up to the broker's (``Δ(avg × qty) / Δqty``)
    and named ``<broker_order_id>:<cumulative>``, so a replayed update
    finds that name taken and records nothing twice. When nothing more
    filled, the stored average stays as it is, whatever ``avg_fill_price``
    says, so the fills always add up to ``avg_fill_price × filled_quantity``.
    A total that shrinks,
    or passes the order's quantity, or a growth with no positive price to
    account for it, is refused: the broker's figures do not add up, and the
    store will not guess.

    ``broker_order_id`` is recorded the first time it is given and must not
    change after. ``status`` follows the forward-only rule of migration
    0004, which refuses anything else, and an order with anything filled
    never becomes ``failed`` (failed means nothing was traded, and a failed
    order does not count toward the daily trade cap). An update to a
    terminal order that changes nothing is a no-op (a replayed sync) and
    writes nothing, ``event`` included; one that changes anything is
    refused. ``status_reason`` and ``broker_status`` (None: as stored) say
    why, in words.
    """
    try:
        if not isinstance(status, OrderStatus):
            raise TypeError(f"status must be an OrderStatus, not {type(status).__name__}")
        instant = _utc(synced_at, "synced_at")
        with transaction(conn):
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
            ).fetchone()
            if row is None:
                raise ValueError("no such order")
            current = _row_to_order(row)
            if current.decision_id is None:
                raise ValueError("the order predates Phase 6: it changes through upsert_order")
            total = current.filled_quantity if filled_quantity is None else filled_quantity
            broker_id = broker_order_id if broker_order_id is not None else current.broker_order_id
            if current.status in TERMINAL_ORDER_STATUSES and (
                status is current.status
                and total == current.filled_quantity
                and broker_id == current.broker_order_id
                and broker_status in (None, current.broker_status)
                and status_reason in (None, current.status_reason)
            ):
                return current, None
            if status is OrderStatus.FAILED and total > 0:
                raise ValueError(
                    f"failed with {total:.10g} filled: an order that filled in part did not fail"
                )
            if status is OrderStatus.FILLED and total != current.quantity:
                raise ValueError(
                    f"filled with {total:.10g} of {current.quantity:.10g}: a filled order is"
                    " filled in full"
                )
            fill = _new_fill(current, total, avg_fill_price, broker_id, instant)
            conn.execute(
                "UPDATE orders SET status = ?, broker_order_id = ?, broker_status = ?,"
                " filled_quantity = ?, avg_fill_price = ?, status_reason = ?,"
                " last_synced_at = ?, updated_at = ? WHERE client_order_id = ?",
                (
                    status.value,
                    broker_id,
                    broker_status if broker_status is not None else current.broker_status,
                    total,
                    avg_fill_price if fill is not None else current.avg_fill_price,
                    status_reason if status_reason is not None else current.status_reason,
                    _iso(instant),
                    _iso(instant),
                    client_order_id,
                ),
            )
            if fill is not None:
                conn.execute(
                    "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees,"
                    " broker_fill_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        fill.id,
                        fill.order_id,
                        _iso(fill.filled_at),
                        fill.fill_price,
                        fill.fill_quantity,
                        fill.fees,
                        fill.broker_fill_id,
                    ),
                )
            if event is not None:
                _insert_event(conn, event)
            stored = _row_to_order(
                conn.execute(
                    "SELECT * FROM orders WHERE client_order_id = ?", (client_order_id,)
                ).fetchone()
            )
    except _REPO_ERRORS as exc:
        raise StoreError("apply broker update", client_order_id, exc) from exc
    return stored, fill


def _new_fill(
    current: Order,
    total: float,
    avg_fill_price: float | None,
    broker_order_id: str | None,
    filled_at: datetime,
) -> Fill | None:
    """The fill that takes ``current`` from its stored filled quantity to
    ``total``; None when nothing more filled. A ValueError when the figures
    do not add up."""
    if not math.isfinite(total) or total < 0:
        raise ValueError(f"filled quantity {total!r} is not a quantity")
    if total < current.filled_quantity:
        raise ValueError(
            f"filled quantity {total:.10g} is below the {current.filled_quantity:.10g} recorded"
        )
    if total > current.quantity:
        raise ValueError(
            f"filled quantity {total:.10g} is above the order's {current.quantity:.10g}"
        )
    delta = total - current.filled_quantity
    if delta == 0:
        return None
    if broker_order_id is None:
        raise ValueError("a fill needs the broker's order id")
    if avg_fill_price is None or not math.isfinite(avg_fill_price) or avg_fill_price <= 0:
        raise ValueError("a fill needs a positive average fill price")
    before = (current.avg_fill_price or 0.0) * current.filled_quantity
    price = (avg_fill_price * total - before) / delta
    if not math.isfinite(price) or price <= 0:
        raise ValueError(
            f"the broker's average price {avg_fill_price:.10g} over {total:.10g} filled does"
            " not account for the new fill with a positive price"
        )
    return Fill(
        order_id=current.id,
        filled_at=filled_at,
        fill_price=price,
        fill_quantity=delta,
        broker_fill_id=_fill_id(broker_order_id, total),
    )


def get_order_by_broker_id(conn: sqlite3.Connection, broker_order_id: str) -> Order | None:
    """The execution-era order the broker knows as ``broker_order_id``, or None.

    Only orders written through ``claim_order`` are looked up: among them the
    id is unique (migration 0004), while rows written before Phase 6 may
    share one.
    """
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE broker_order_id = ? AND decision_id IS NOT NULL",
            (broker_order_id,),
        ).fetchone()
        return _row_to_order(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get order by broker id", broker_order_id, exc) from exc


def get_proposal_order(conn: sqlite3.Connection, proposal_id: str) -> Order | None:
    """The execution-era order placed for ``proposal_id`` (there is at most one), or None."""
    try:
        row = conn.execute(
            "SELECT * FROM orders WHERE proposal_id = ? AND decision_id IS NOT NULL",
            (proposal_id,),
        ).fetchone()
        return _row_to_order(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get proposal order", proposal_id, exc) from exc


def get_decision(conn: sqlite3.Connection, decision_id: str) -> PolicyDecision | None:
    """One policy decision by id, or None."""
    try:
        row = _decision_row(conn, decision_id)
        return _row_to_decision(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get decision", decision_id, exc) from exc


def get_decision_approval(conn: sqlite3.Connection, decision_id: str) -> Approval | None:
    """The approval tied to ``decision_id`` (there is at most one), or None."""
    try:
        row = conn.execute(
            "SELECT * FROM approvals WHERE decision_id = ?", (decision_id,)
        ).fetchone()
        return _row_to_approval(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get decision approval", decision_id, exc) from exc


# --- reads ------------------------------------------------------------------


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> Proposal | None:
    try:
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        return _row_to_proposal(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get proposal", proposal_id, exc) from exc


def get_recent_proposals(conn: sqlite3.Connection, limit: int = 10) -> list[Proposal]:
    """The ``limit`` newest proposals, newest first (shown to the thesis stage so
    it does not repeat itself)."""
    try:
        rows = conn.execute(
            "SELECT * FROM proposals ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_proposal(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get recent proposals", cause=exc) from exc


def get_proposal_legs(conn: sqlite3.Connection, proposal_id: str) -> list[ProposalLeg]:
    """The proposal's option legs by leg_index; empty for an equity proposal (or an unknown id)."""
    try:
        rows = conn.execute(
            "SELECT * FROM proposal_legs WHERE proposal_id = ? ORDER BY leg_index", (proposal_id,)
        ).fetchall()
        return [_row_to_leg(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get proposal legs", proposal_id, exc) from exc


def get_cycle_reasoning(conn: sqlite3.Connection, cycle_id: str) -> list[Reasoning]:
    """Every reasoning row written under ``cycle_id``, oldest first (created_at, then rowid)."""
    try:
        rows = conn.execute(
            "SELECT * FROM reasoning WHERE cycle_id = ? ORDER BY created_at, rowid", (cycle_id,)
        ).fetchall()
        return [_row_to_reasoning(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get cycle reasoning", cycle_id, exc) from exc


def get_token_usage(conn: sqlite3.Connection, day: date | None = None) -> TokenUsage:
    """Tokens in/out and rows written to ``reasoning`` on ``day`` (a UTC calendar
    day; default today).

    What the budget guard measures the daily budget against. The bounds are
    ``_utc_day_bounds`` (as ``get_daily_pnl``); a NULL token count — a row
    written without usage — adds 0, and ``calls`` counts every row.
    """
    day, start, end = _utc_day_bounds(day)
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), COUNT(*)"
            " FROM reasoning WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchone()
        return _usage(row[0], row[1], row[2])
    except _REPO_ERRORS as exc:
        raise StoreError("get token usage", day.isoformat(), exc) from exc


def get_cycle_token_usage(conn: sqlite3.Connection, cycle_id: str) -> TokenUsage:
    """Tokens in/out and rows of one cycle — what the per-cycle cap is measured against."""
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), COUNT(*)"
            " FROM reasoning WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        return _usage(row[0], row[1], row[2])
    except _REPO_ERRORS as exc:
        raise StoreError("get cycle token usage", cycle_id, exc) from exc


def get_token_usage_by_model(
    conn: sqlite3.Connection, day: date | None = None
) -> dict[str, TokenUsage]:
    """``get_token_usage`` split by ``model_name``, sorted by name; rows without
    one fall under "unknown"."""
    day, start, end = _utc_day_bounds(day)
    try:
        rows = conn.execute(
            "SELECT COALESCE(model_name, 'unknown'), COALESCE(SUM(tokens_in), 0),"
            " COALESCE(SUM(tokens_out), 0), COUNT(*)"
            " FROM reasoning WHERE created_at >= ? AND created_at < ?"
            " GROUP BY COALESCE(model_name, 'unknown') ORDER BY COALESCE(model_name, 'unknown')",
            (start, end),
        ).fetchall()
        return {str(row[0]): _usage(row[1], row[2], row[3]) for row in rows}
    except _REPO_ERRORS as exc:
        raise StoreError("get token usage by model", day.isoformat(), exc) from exc


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


def _utc_day_bounds(day: date | None) -> tuple[date, str, str]:
    """``(day, start, end)``: a UTC calendar day (default today) and its ISO-text bounds.

    Bounds are ``[day 00:00Z, next day 00:00Z)`` as ISO text; every stored
    timestamp has the same UTC text shape, so text comparison against them
    is chronological. A ``datetime`` (a ``date`` subclass) means the UTC
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
    return day, start.isoformat(), end.isoformat()


def get_daily_pnl(conn: sqlite3.Connection, day: date | None = None) -> PnlSnapshot | None:
    """The latest P&L snapshot taken on ``day`` (a UTC calendar day; default today), or None.

    The day's bounds are ``_utc_day_bounds``: a ``datetime`` means the UTC
    calendar day of that instant, never its local date.
    """
    day, start, end = _utc_day_bounds(day)
    try:
        row = conn.execute(
            "SELECT * FROM pnl_snapshots WHERE taken_at >= ? AND taken_at < ?"
            " ORDER BY taken_at DESC, rowid DESC LIMIT 1",
            (start, end),
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


def _shown(value: object) -> str:
    """A stored value as a ``problems`` line quotes it: its repr (so control
    characters arrive escaped), cut to ``_SHOWN_MAX`` characters."""
    text = repr(value)
    return text if len(text) <= _SHOWN_MAX else text[: _SHOWN_MAX - 3] + "..."


def _control_time(text: object) -> datetime | None:
    """A ``controls`` timestamp as an aware UTC datetime; None when it will not parse.

    Naive text is taken as UTC, like every timestamp in the store. Never
    raises: the columns are TEXT NOT NULL in the migration, but a table
    rebuilt by hand can hold anything.
    """
    if not isinstance(text, str):
        return None
    try:
        return _utc(datetime.fromisoformat(text))
    except (ValueError, OverflowError):
        return None


def _read_controls(conn: sqlite3.Connection) -> Controls:
    """The ``controls`` rows as one ``Controls``. A bad row fails closed and
    is explained in ``problems``; only SQLite itself raises (the table missing)."""
    rows = conn.execute("SELECT key, value, updated_at FROM controls ORDER BY key").fetchall()
    kill_rows = [row for row in rows if row["key"] == _KILL_SWITCH]
    halt_rows = [row for row in rows if row["key"] == _HALT_UNTIL]
    other_keys = sorted(
        _shown(row["key"]) for row in rows if row["key"] not in (_KILL_SWITCH, _HALT_UNTIL)
    )
    problems: list[str] = []

    kill_switch = True  # until one readable row says it is off
    kill_updated_at: datetime | None = None
    if len(kill_rows) != 1:
        found = "is missing from" if not kill_rows else f"has {len(kill_rows)} rows in"
        problems.append(f"kill_switch {found} the controls table: reading the kill switch as ON")
    else:
        value = kill_rows[0]["value"]
        kill_updated_at = _control_time(kill_rows[0]["updated_at"])
        if isinstance(value, str) and value in (_SWITCH_ON, _SWITCH_OFF):
            kill_switch = value == _SWITCH_ON
        else:
            problems.append(
                f"kill_switch value {_shown(value)} is neither 'on' nor 'off':"
                " reading the kill switch as ON"
            )

    halt_until: datetime | None = None
    halt_unknown = True  # until one readable row says there is no halt, or until when
    halt_updated_at: datetime | None = None
    if len(halt_rows) != 1:
        found = "is missing from" if not halt_rows else f"has {len(halt_rows)} rows in"
        problems.append(
            f"halt_until {found} the controls table: cannot prove trading is not halted"
        )
    else:
        value = halt_rows[0]["value"]
        halt_updated_at = _control_time(halt_rows[0]["updated_at"])
        if value == "":
            halt_unknown = False
        else:
            halt_until = _control_time(value)
            halt_unknown = halt_until is None
            if halt_unknown:
                problems.append(
                    f"halt_until value {_shown(value)} is neither empty nor an ISO-8601"
                    " timestamp: cannot prove trading is not halted"
                )

    # A row under any other key is a control this code cannot interpret (the
    # key CHECK was bypassed, or newer code wrote it) — it may well be a stop.
    for key in other_keys:
        kill_switch = True
        problems.append(f"unrecognised control key {key}: reading the kill switch as ON")

    return Controls(
        kill_switch=kill_switch,
        halt_until=halt_until,
        halt_unknown=halt_unknown,
        kill_switch_updated_at=kill_updated_at,
        halt_until_updated_at=halt_updated_at,
        problems=tuple(problems),
    )


def get_controls(conn: sqlite3.Connection) -> Controls:
    """The operator's control flags: the kill switch and the daily-loss halt.

    Fails closed and never raises for a bad row — a gate that cannot read
    its own stop flags must stay shut, not crash open:

    - ``kill_switch`` row missing, duplicated, or neither 'on' nor 'off' →
      ``kill_switch=True``;
    - ``halt_until`` row missing, duplicated, or neither '' nor text
      ``datetime.fromisoformat`` parses → ``halt_unknown=True`` and
      ``halt_until=None`` ('' is "no halt": ``halt_until=None``,
      ``halt_unknown=False``; a naive timestamp is taken as UTC);
    - a row under any other key → ``kill_switch=True``.

    Each finding is one line in ``problems``. None of them can happen to a
    store written through this module: the migration seeds both rows,
    CHECKs key and value, and refuses DELETE. ``*_updated_at`` is the row's
    ``updated_at`` (None when that text will not parse — not a problem).

    The table missing altogether — a database not migrated to 0003 — is a
    StoreError: there is nothing to read, and ``db init`` fixes it.
    """
    try:
        return _read_controls(conn)
    except _REPO_ERRORS as exc:
        raise StoreError("get controls", cause=exc) from exc


def get_proposals_since(conn: sqlite3.Connection, since: datetime) -> list[Proposal]:
    """Every proposal created at or after ``since`` (naive taken as UTC), newest first.

    What the policy engine's duplicate rule looks through. The bound is
    inclusive. Ties on ``created_at`` break on rowid, the later insert
    first, as in ``get_recent_proposals``.
    """
    try:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE created_at >= ?"
            " ORDER BY created_at DESC, rowid DESC",
            (_iso(_utc(since, "since")),),
        ).fetchall()
        return [_row_to_proposal(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get proposals since", str(since), exc) from exc


def get_latest_undecided_proposal(conn: sqlite3.Connection) -> Proposal | None:
    """The newest proposal with no row in ``policy_decisions``, or None.

    Newest by ``created_at`` (then rowid). A proposal with any decision at
    all — one or several, whatever the verdict — is decided and is skipped,
    so an older undecided proposal can be the answer.
    """
    try:
        row = conn.execute(
            "SELECT * FROM proposals WHERE NOT EXISTS"
            " (SELECT 1 FROM policy_decisions WHERE policy_decisions.proposal_id = proposals.id)"
            " ORDER BY proposals.created_at DESC, proposals.rowid DESC LIMIT 1"
        ).fetchone()
        return _row_to_proposal(row) if row is not None else None
    except _REPO_ERRORS as exc:
        raise StoreError("get latest undecided proposal", cause=exc) from exc


def count_orders_submitted_between(
    conn: sqlite3.Connection, start: datetime, end: datetime
) -> int:
    """How many orders were sent to the broker in ``[start, end)`` (naive taken as UTC).

    Counts by ``submitted_at`` — an order never submitted (NULL) is not a
    trade — whatever became of it since (filled, cancelled, still open),
    except ``failed``: an order the broker never accepted is not a trade
    either. What the policy engine's daily trade cap measures against; the
    caller supplies the bounds of its trading day.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM orders"
            " WHERE submitted_at >= ? AND submitted_at < ? AND status != ?",
            (_iso(_utc(start, "start")), _iso(_utc(end, "end")), OrderStatus.FAILED.value),
        ).fetchone()
        return int(row[0])
    except _REPO_ERRORS as exc:
        raise StoreError("count orders submitted between", f"{start} .. {end}", exc) from exc


def get_proposal_trace(conn: sqlite3.Connection, proposal_id: str) -> ProposalTrace:
    """The full lineage of one proposal in one call; StoreError if it does not exist.

    Legs are ordered by leg_index; reasoning by created_at (then rowid),
    decisions by decided_at, approvals by requested_at, orders by updated_at
    and fills — every fill of every order of the proposal — by filled_at,
    all oldest first. Reasoning rows reach the trace once
    ``link_reasoning_to_proposal`` has attached them.
    """
    try:
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise StoreError("proposal trace (no such proposal)", proposal_id)
        key = (proposal_id,)
        legs = conn.execute(
            "SELECT * FROM proposal_legs WHERE proposal_id = ? ORDER BY leg_index", key
        ).fetchall()
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
            legs=tuple(_row_to_leg(r) for r in legs),
            reasoning=tuple(_row_to_reasoning(r) for r in reasoning),
            decisions=tuple(_row_to_decision(r) for r in decisions),
            approvals=tuple(_row_to_approval(r) for r in approvals),
            orders=tuple(_row_to_order(r) for r in orders),
            fills=tuple(_row_to_fill(r) for r in fills),
        )
    except _REPO_ERRORS as exc:
        raise StoreError("proposal trace", proposal_id, exc) from exc
