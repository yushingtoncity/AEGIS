"""Phase 3 — persistence and audit log.

A SQLite journal (stdlib ``sqlite3``, no ORM, so the file stays inspectable)
of every proposal, reasoning stage, policy decision, approval, order, fill,
and account snapshot, so each action AEGIS takes is auditable and replayable
after the fact. WAL journaling lets a dashboard read while the loop writes.
The schema lives in versioned SQL files under ``migrations/``, applied by
``open_store`` and tracked in ``schema_version``; failures are wrapped in
``StoreError`` with context, like ``DataError`` in the data layer.
"""

from aegis.store.db import connect, migrate, open_store, schema_version
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
    StoreStatus,
    Verdict,
    new_id,
)
from aegis.store.repo import (
    add_reasoning,
    get_daily_pnl,
    get_open_orders,
    get_order,
    get_proposal,
    get_proposal_trace,
    get_recent_events,
    insert_proposal,
    log_event,
    record_approval,
    record_decision,
    record_fill,
    snapshot_pnl,
    snapshot_positions,
    upsert_order,
)

__all__ = [
    "OPEN_ORDER_STATUSES",
    "Approval",
    "ApprovalResponse",
    "Broker",
    "Event",
    "EventLevel",
    "Fill",
    "Instrument",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PnlSnapshot",
    "PolicyDecision",
    "PositionSnapshot",
    "Proposal",
    "ProposalTrace",
    "Reasoning",
    "ReasoningStage",
    "StoreError",
    "StoreStatus",
    "Verdict",
    "add_reasoning",
    "connect",
    "get_daily_pnl",
    "get_open_orders",
    "get_order",
    "get_proposal",
    "get_proposal_trace",
    "get_recent_events",
    "insert_proposal",
    "log_event",
    "migrate",
    "new_id",
    "open_store",
    "record_approval",
    "record_decision",
    "record_fill",
    "schema_version",
    "snapshot_pnl",
    "snapshot_positions",
    "upsert_order",
]
