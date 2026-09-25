"""Records persisted by the Phase 3 store.

One frozen pydantic model per table, plus ``ProposalTrace`` (the reassembled
lineage of one proposal) and ``StoreStatus`` (what ``aegis.cli.db status``
reports). The repository writes these verbatim and reads them back, so the
column names in ``migrations/`` (``0001_initial.sql``, with ``reasoning``
rebuilt and ``proposal_legs`` added by ``0002_reasoning_cycles_and_legs.sql``)
are exactly these field names.

Conventions shared by every record:

- ``id`` is a uuid4 string from ``new_id``; callers may omit it.
- Timestamps are timezone-aware UTC. Naive datetimes are taken as UTC (the
  data layer's ``_to_utc`` convention) and any other offset is converted.
  They are stored as ISO-8601 text with offset (``datetime.isoformat()``)
  and parsed back with ``datetime.fromisoformat``.
- Enum columns store the enum's string value verbatim; the SQL CHECK
  constraints list the same values.
- ``rules_evaluated`` and ``payload`` are JSON columns (``json.dumps`` /
  ``json.loads``). ``raw_model_output`` is the model's verbatim text and is
  stored as-is, never parsed.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.data.models import OptionType, utcnow


def new_id() -> str:
    """A fresh uuid4 in its canonical hyphenated form (readable in sqlite3)."""
    return str(uuid.uuid4())


def _to_utc(value: datetime) -> datetime:
    """Naive datetimes are taken as UTC; aware ones are converted to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _upper_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol must not be blank")
    return symbol


class Instrument(str, Enum):
    EQUITY = "equity"
    OPTION = "option"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class ReasoningStage(str, Enum):
    SCAN = "scan"
    THESIS = "thesis"
    PROPOSAL = "proposal"


class Verdict(str, Enum):
    REJECT = "REJECT"
    FLAG_ONLY = "FLAG_ONLY"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    AUTO_EXECUTE = "AUTO_EXECUTE"


class ApprovalResponse(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Broker(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class OrderStatus(str, Enum):
    PROPOSED = "proposed"
    GATED = "gated"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


OPEN_ORDER_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.PROPOSED,
        OrderStatus.GATED,
        OrderStatus.APPROVED,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
    }
)
"""Statuses an order can still move on from; filled/cancelled/failed are terminal."""


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StoreRecord(BaseModel):
    """Base for every persisted record: frozen, uuid id, UTC timestamps.

    The wildcard validator coerces every ``datetime`` field on every record
    to an aware UTC value, so a naive timestamp (or one in another zone)
    never reaches the database. NaN and infinity are rejected on every
    float field: SQLite has no NaN and would store one as NULL, so a NaN
    price would silently read back as None.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    id: str = Field(default_factory=new_id)

    @field_validator("*", mode="after")
    @classmethod
    def _utc_timestamps(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return _to_utc(value)
        return value


class Proposal(StoreRecord):
    """One trade the brain proposed, with its verbatim model output."""

    created_at: datetime = Field(default_factory=utcnow)
    cycle_id: str
    symbol: str
    instrument: Instrument
    side: OrderSide
    quantity: float = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = None
    thesis: str
    confidence: float = Field(ge=0, le=1)
    invalidation: str
    raw_model_output: str
    """The model's verbatim output (JSON text by convention) — never parsed here."""
    model_name: str
    prompt_version: str

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _upper_symbol(value)


class ProposalLeg(StoreRecord):
    """One leg of an option proposal: the contract, the side and the size.

    A single-leg option proposal has one leg; a spread or condor lists its
    legs by ``leg_index`` (0-based, in the order the structure was resolved).
    An equity proposal has none. ``expiration`` is stored as ISO text
    (``date.isoformat()``).
    """

    proposal_id: str
    leg_index: int = Field(ge=0)
    symbol: str
    option_type: OptionType
    side: OrderSide
    quantity: float = Field(gt=0)
    strike: float = Field(gt=0)
    expiration: date

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _upper_symbol(value)


class Reasoning(StoreRecord):
    """One agent stage's output (scan → thesis → proposal) under its cycle.

    ``cycle_id`` is set as the row is written, before any proposal exists;
    ``proposal_id`` is filled in by ``repo.link_reasoning_to_proposal`` once
    the cycle produces one and stays None for a NO_TRADE cycle. A row must
    belong to at least one of the two — the table's CHECK says the same.
    ``model_name`` and ``latency_ms`` describe the LLM call behind ``content``.
    """

    cycle_id: str | None = None
    proposal_id: str | None = None
    stage: ReasoningStage
    created_at: datetime = Field(default_factory=utcnow)
    content: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    model_name: str | None = None
    latency_ms: float | None = None

    @model_validator(mode="after")
    def _under_a_cycle_or_proposal(self) -> Reasoning:
        if self.cycle_id is None and self.proposal_id is None:
            raise ValueError("a reasoning row needs a cycle_id or a proposal_id (or both)")
        return self


class PolicyDecision(StoreRecord):
    """The policy engine's verdict on a proposal and the rules it checked."""

    proposal_id: str
    decided_at: datetime = Field(default_factory=utcnow)
    verdict: Verdict
    rules_evaluated: list[dict[str, Any]]
    """One entry per rule, e.g. {"rule": ..., "limit": ..., "observed": ..., "passed": ...}."""
    failing_rule: str | None = None
    notes: str | None = None


class Approval(StoreRecord):
    """A human approval request and, once answered, its response."""

    proposal_id: str
    requested_at: datetime = Field(default_factory=utcnow)
    responded_at: datetime | None = None
    response: ApprovalResponse | None = None
    channel: str
    responder: str | None = None


class Order(StoreRecord):
    """An order's current state; ``client_order_id`` is the idempotency key."""

    proposal_id: str
    client_order_id: str
    broker: Broker
    broker_order_id: str | None = None
    status: OrderStatus
    submitted_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    limit_price: float | None = None

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _upper_symbol(value)


class Fill(StoreRecord):
    """One (possibly partial) execution of an order."""

    order_id: str
    filled_at: datetime = Field(default_factory=utcnow)
    fill_price: float
    fill_quantity: float = Field(gt=0)
    fees: float = 0.0


class PositionSnapshot(StoreRecord):
    """One position as seen at ``taken_at``."""

    taken_at: datetime = Field(default_factory=utcnow)
    symbol: str
    quantity: float
    avg_cost: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None


class PnlSnapshot(StoreRecord):
    """Account-level P&L as seen at ``taken_at``."""

    taken_at: datetime = Field(default_factory=utcnow)
    equity: float
    cash: float
    buying_power: float
    daily_pnl: float
    realized_pnl: float
    unrealized_pnl: float


class Event(StoreRecord):
    """An operational event: risk-limit trips, kill-switch toggles, heartbeats, errors."""

    occurred_at: datetime = Field(default_factory=utcnow)
    level: EventLevel
    kind: str
    message: str
    payload: dict[str, Any] | None = None


class TokenUsage(BaseModel):
    """Token totals read back from the reasoning table: ``repo.get_token_usage``
    (a UTC day), ``get_cycle_token_usage`` (one cycle) and
    ``get_token_usage_by_model``. ``calls`` is the number of rows summed; a
    NULL token count adds 0."""

    model_config = ConfigDict(frozen=True)

    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


class ProposalTrace(BaseModel):
    """The full lineage of one proposal, as ``repo.get_proposal_trace`` returns it.

    Every tuple is ordered oldest → newest (reasoning by created_at then
    rowid, decisions by decided_at, approvals by requested_at, orders by
    updated_at, fills by filled_at), so the latest item is always the last.
    ``legs`` is by leg_index (empty for an equity proposal) and ``fills``
    holds every fill of every order in ``orders``.
    """

    model_config = ConfigDict(frozen=True)

    proposal: Proposal
    legs: tuple[ProposalLeg, ...] = ()
    reasoning: tuple[Reasoning, ...] = ()
    decisions: tuple[PolicyDecision, ...] = ()
    approvals: tuple[Approval, ...] = ()
    orders: tuple[Order, ...] = ()
    fills: tuple[Fill, ...] = ()

    @property
    def decision(self) -> PolicyDecision | None:
        """The latest policy decision, or None if the proposal was never judged."""
        return self.decisions[-1] if self.decisions else None

    @property
    def approval(self) -> Approval | None:
        """The latest approval request, or None if none was ever raised."""
        return self.approvals[-1] if self.approvals else None


class StoreStatus(BaseModel):
    """What ``aegis.cli.db status`` reports about one database file."""

    model_config = ConfigDict(frozen=True)

    path: str
    exists: bool
    schema_version: int
    journal_mode: str
    applied_migrations: tuple[str, ...]
    pending_migrations: tuple[str, ...]
    row_counts: dict[str, int]
    """Table name → row count for all ten tables (0 for a table that does not exist)."""
    missing_tables: tuple[str, ...] = ()
    """Tables a fully migrated database must have but lacks — one was dropped by hand.

    Empty while migrations are pending (those tables are simply not created
    yet). ``migrate`` never re-runs a recorded migration, so a missing table
    comes back from its migration file, not from ``db init``.
    """
