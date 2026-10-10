"""Phase 6: the dispatcher, the one caller of an Executor.

The LLM proposes; the policy engine disposes; this module carries out what
the engine decided, and nothing else. Every order it places rests on a
recorded verdict, passes a fresh re-check, is claimed in the store before a
byte leaves the process, and is followed at the broker through the store's
own writers, so every step stays reconstructible (CLAUDE.md invariant 5).
docs/phase6/SPEC_PHASE6.md is the design; the numbered decisions below are
its.

What it does:

- ``approve`` / ``reject``: a human's answer to one NEEDS_APPROVAL verdict,
  written once, tied to that decision, good for
  ``broker.approval_ttl_seconds`` (D7).
- ``place``: one decision to one order, exactly once.

  1. The decision must be the latest verdict on its proposal, the proposal
     must have no order yet (one order per proposal, ever: D11), and the
     verdict must still stand: AUTO_EXECUTE younger than
     ``broker.max_decision_age_seconds``, or NEEDS_APPROVAL with an
     unexpired "approved" answer.
  2. A full re-evaluation on fresh data, recorded as a ``pre_submit``
     decision (D3). An AUTO_EXECUTE order goes only if the re-check is
     AUTO_EXECUTE too; an approved order only if every rule that escalates
     now already escalated when the human said yes, and nothing rejects or
     flags.
  3. The order is minted from the proposal (equity, or a single option
     contract bought to open or sold to close: D6), then claimed through
     ``repo.claim_order``, which re-reads the controls and counts the daily
     cap in one transaction.
  4. One send. Whatever comes back is recorded: a receipt, a refusal
     (``failed``), nothing sent (``failed``), or an unknown outcome
     (``approved`` with ``status_reason`` set, a CRITICAL event, then one
     lookup by name). Nothing ever retries.

- ``sync``: the store follows the broker. Fills come in as the growth of
  the broker's cumulative figures (the store names each one, so a replay
  records nothing twice). An order the broker never acknowledged and still
  does not know after ``broker.not_found_grace_seconds`` is ``cancelled``
  (and still counts toward the cap: D9). An order open at the broker that
  the store does not hold open is an orphan: a CRITICAL event, and the kill
  switch goes on (D10).
- ``cancel``: one order, by its client order id.
- ``stand_down``: with the kill switch on, a halt in force, or a halt that
  cannot be read, every working order is cancelled (D4). Nothing is ever
  flattened (D5).
- ``tick``: ``sync``, then ``stand_down``, then ``place`` every decision
  that may still go.

Placing needs ``broker.enabled`` (shipped false: D2); reading the broker and
cancelling do not, so turning placing off never strands a working order.

Every broker failure becomes a ``PlacementError`` here: nothing outside the
policy engine handles the execution package's own types, and what this
module hands back is the store's models only. The clock is
``context.wall_clock`` unless a caller passes one (this module may read no
clock of its own).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from aegis.config import AegisConfig
from aegis.execution.base import Executor
from aegis.execution.models import (
    CLIENT_ORDER_ID_PREFIX,
    ApprovedOrder,
    ExecutionError,
    ExecutionOutcome,
    OrderReceipt,
)
from aegis.execution.paper import PaperExecutor, paper_trading_client
from aegis.policy.context import build_context, load_proposal, trading_day, wall_clock
from aegis.policy.engine import evaluate
from aegis.policy.errors import PlacementError
from aegis.policy.models import PolicyContext, ProposalUnderReview, RuleOutcome
from aegis.store.errors import ClaimRefused, StoreError
from aegis.store.models import (
    TERMINAL_ORDER_STATUSES,
    Approval,
    ApprovalResponse,
    Broker,
    DecisionPurpose,
    Event,
    EventLevel,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PolicyDecision,
    PositionIntent,
    TimeInForce,
    Verdict,
    new_id,
)
from aegis.store.repo import (
    apply_broker_update,
    claim_order,
    get_controls,
    get_decision,
    get_decision_approval,
    get_execution_orders,
    get_latest_verdict,
    get_live_approvals,
    get_order,
    get_proposal_order,
    get_verdicts_since,
    log_event,
    record_approval,
    set_kill_switch,
)

Clock = Callable[[], datetime]
BrokerFactory = Callable[[sqlite3.Connection, AegisConfig], Executor]
ContextBuilder = Callable[..., PolicyContext]

APPROVAL_CHANNEL = "cli"
"""The ``channel`` of an answer given through ``orders approve`` / ``reject``."""

# Event kinds, one per thing that can happen to an order.
APPROVAL_ANSWERED = "approval_answered"
ORDER_REFUSED = "order_refused"
ORDER_CLAIMED = "order_claimed"
ORDER_SUBMITTED = "order_submitted"
ORDER_REJECTED = "order_rejected"
ORDER_NOT_SENT = "order_not_sent"
ORDER_UNKNOWN_OUTCOME = "order_unknown_outcome"
ORDER_ADOPTED = "order_adopted"
ORDER_UPDATED = "order_updated"
ORDER_NOT_FOUND = "order_not_found"
ORDER_UNTRACKED = "order_sync_failed"
ORDER_CANCEL_REQUESTED = "order_cancel_requested"
ORPHAN_BROKER_ORDER = "orphan_broker_order"
STAND_DOWN = "stand_down"
KILL_SWITCH_EVENT = "kill_switch"  # the kind ``policy kill`` logs, for the same switch

UNKNOWN_OUTCOME = "unknown_outcome"
"""The ``status_reason`` prefix of an order whose send ended unclear."""


# --- results ------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """What ``place`` did. ``order`` is the row as stored after the send (None
    for a dry run, which claims and sends nothing); ``regate`` the
    ``pre_submit`` decision (unsaved on a dry run); ``lines`` say what
    happened, in words, for the operator."""

    regate: PolicyDecision
    order: Order | None
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    """What ``sync``, ``stand_down`` or ``tick`` did: one line per order it
    touched or could not, and how many of those it could not resolve
    (``failures``; a caller exits non-zero when there are any)."""

    lines: tuple[str, ...] = ()
    failures: int = 0
    placed: tuple[Order, ...] = field(default=())

    def __add__(self, other: Report) -> Report:
        return Report(
            self.lines + other.lines, self.failures + other.failures, self.placed + other.placed
        )


# --- small helpers ------------------------------------------------------------


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _exact(value: float | int | Decimal) -> str:
    """A number as it is, every digit and no exponent: ``200``, ``1234.5678``."""
    number = value if isinstance(value, Decimal) else Decimal(repr(value))
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _now(clock: Clock) -> datetime:
    now = clock()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise PlacementError("read the clock (it gave no timezone-aware instant)")
    return now


def _event(now: datetime, level: EventLevel, kind: str, message: str, **payload: Any) -> Event:
    return Event(
        occurred_at=now, level=level, kind=kind, message=_one_line(message), payload=payload or None
    )


def _seconds(delta: timedelta) -> float:
    return delta.total_seconds()


def _broker_error(what: str, key: str | None, error: ExecutionError) -> PlacementError:
    return PlacementError(what, key, error, outcome=error.outcome.value)


def paper_broker(conn: sqlite3.Connection, config: AegisConfig) -> Executor:
    """The Executor for ``config``'s broker: Phase 6 has the paper account
    only. Its client reads the keys from ``.env``; a missing key is the
    ``ConfigError`` the data layer raises too."""
    if config.broker.kind != "paper":  # the config cannot hold another; refuse it all the same
        raise PlacementError(f"build the broker (kind {config.broker.kind!r} is not paper)")
    try:
        client = paper_trading_client(timeout_seconds=config.broker.http_timeout_seconds)
        return PaperExecutor(conn, client)
    except ExecutionError as exc:
        raise _broker_error("build the broker", None, exc) from exc


def _broker(conn: sqlite3.Connection, config: AegisConfig, factory: BrokerFactory | None) -> Executor:
    return (factory or paper_broker)(conn, config)


# --- the verdict an order rests on --------------------------------------------


def _verdict(conn: sqlite3.Connection, decision_id: str, what: str) -> PolicyDecision:
    """``decision_id``, which must be a verdict and the latest on its proposal."""
    decision = get_decision(conn, decision_id)
    if decision is None:
        raise PlacementError(f"{what} (no such decision)", decision_id)
    if decision.purpose is not DecisionPurpose.EVALUATE:
        raise PlacementError(f"{what} (a {decision.purpose.value} re-check is not a verdict)", decision_id)
    latest = get_latest_verdict(conn, decision.proposal_id)
    if latest is None or latest.id != decision.id:
        later = "none" if latest is None else latest.id
        raise PlacementError(
            f"{what} (a later verdict on proposal {decision.proposal_id} supersedes it: {later})",
            decision_id,
        )
    existing = get_proposal_order(conn, decision.proposal_id)
    if existing is not None:
        raise PlacementError(
            f"{what} (proposal {decision.proposal_id} already has order {existing.client_order_id},"
            f" {existing.status.value}: one order per proposal; run `orders sync` to follow it)",
            decision_id,
        )
    return decision


def _standing(
    conn: sqlite3.Connection, decision: PolicyDecision, config: AegisConfig, now: datetime
) -> Approval | None:
    """Why the verdict no longer stands, raised; else the approval it rests
    on (None for AUTO_EXECUTE)."""
    if decision.verdict is Verdict.AUTO_EXECUTE:
        age = _seconds(now - decision.decided_at)
        limit = config.broker.max_decision_age_seconds
        if age < 0:
            raise PlacementError("place order (the verdict is dated after now)", decision.id)
        if age >= limit:
            raise PlacementError(
                f"place order (the verdict is {age:.0f} s old: not younger than"
                f" broker.max_decision_age_seconds, {limit:g} s)",
                decision.id,
            )
        return None
    if decision.verdict is not Verdict.NEEDS_APPROVAL:
        raise PlacementError(
            f"place order (the verdict is {decision.verdict.value}: no order follows it)", decision.id
        )
    approval = get_decision_approval(conn, decision.id)
    if approval is None or approval.response is None:
        raise PlacementError(
            "place order (it awaits a human answer: run `orders approve` first)", decision.id
        )
    if approval.response is not ApprovalResponse.APPROVED:
        raise PlacementError(f"place order (the answer was {approval.response.value})", decision.id)
    if approval.expires_at is None or not now < approval.expires_at:
        when = "never set" if approval.expires_at is None else approval.expires_at.isoformat()
        raise PlacementError(f"place order (the approval expired: {when})", decision.id)
    return approval


def _escalations(decision: PolicyDecision) -> frozenset[str]:
    return frozenset(
        row["rule"] for row in decision.rules_evaluated if row.get("outcome") == RuleOutcome.ESCALATE.value
    )


def regate_refusal(verdict: PolicyDecision, regate: PolicyDecision) -> str | None:
    """Why the ``pre_submit`` re-check stops the order, in words; None when it
    may go (D3). An AUTO_EXECUTE verdict needs an AUTO_EXECUTE re-check. A
    NEEDS_APPROVAL verdict, approved by a human, needs a re-check that
    rejects and flags nothing, and escalates on no rule the human did not
    see escalating."""
    if verdict.verdict is Verdict.AUTO_EXECUTE:
        if regate.verdict is Verdict.AUTO_EXECUTE:
            return None
        return f"the re-check is {regate.verdict.value}: {regate.notes}"
    if regate.verdict is Verdict.AUTO_EXECUTE:
        return None
    if regate.verdict is not Verdict.NEEDS_APPROVAL:
        return f"the re-check is {regate.verdict.value}: {regate.notes}"
    unseen = sorted(_escalations(regate) - _escalations(verdict))
    if unseen:
        return f"the re-check escalates on rules the approval did not cover: {', '.join(unseen)}"
    return None


# --- minting the order --------------------------------------------------------


def _whole(quantity: float, what: str, key: str) -> int:
    if quantity != int(quantity):
        raise PlacementError(f"place order ({what} {_exact(quantity)} is not a whole number)", key)
    return int(quantity)


def _intent(symbol: str, side: OrderSide, quantity: int, context: PolicyContext, key: str) -> PositionIntent:
    """Buy to open, or sell to close a long the account holds (D6). Selling
    an option to open (writing it uncovered) is out of Phase 6."""
    if context.positions is None:
        raise PlacementError("place order (the account's positions are unknown)", key)
    held = [p for p in context.positions if p.symbol == symbol]
    long_held = sum(abs(p.qty) for p in held if p.side != "short")
    short_held = sum(abs(p.qty) for p in held if p.side == "short")
    if side is OrderSide.BUY:
        if short_held:
            raise PlacementError(
                f"place order (the account is short {short_held:g} {symbol}: buying to close"
                " is out of Phase 6)",
                key,
            )
        return PositionIntent.BUY_TO_OPEN
    if long_held >= quantity:
        return PositionIntent.SELL_TO_CLOSE
    raise PlacementError(
        f"place order (selling {quantity} {symbol} would open a short: the account holds"
        f" {long_held:g}; writing options is out of Phase 6)",
        key,
    )


def shape_problem(proposal: ProposalUnderReview) -> str | None:
    """Why the proposal is not an order Phase 6 places, in words; None when
    it is one (spec: equity or single-leg option, LIMIT, DAY, whole
    quantity). Checked before the re-check, which costs network calls."""
    order = proposal.proposal
    if order.order_type is not OrderType.LIMIT or order.limit_price is None:
        return "only limit orders are placed in Phase 6"
    if proposal.is_equity:
        if order.quantity != int(order.quantity):
            return f"quantity {_exact(order.quantity)} is not a whole number of shares"
        price = Decimal(repr(order.limit_price))
        places = -price.as_tuple().exponent if price.as_tuple().exponent < 0 else 0
        if price >= 1 and places > 2:
            return (
                f"limit {_exact(price)} is finer than a cent: a stock at $1 or more trades in"
                " whole cents, and the broker would refuse it"
            )
        return None
    if len(proposal.legs) != 1:
        return (
            f"a {len(proposal.legs)}-leg option order: only single-leg options are placed until"
            " the multi-leg debit/credit sign is checked live (D6)"
        )
    if proposal.legs[0].quantity != int(proposal.legs[0].quantity):
        return f"quantity {_exact(proposal.legs[0].quantity)} is not a whole number of contracts"
    return None


def mint(
    proposal: ProposalUnderReview,
    verdict: PolicyDecision,
    regate: PolicyDecision,
    approval: Approval | None,
    context: PolicyContext,
) -> ApprovedOrder:
    """The one order ``verdict`` authorises, built from the proposal.

    An equity order is the proposal's own symbol, side and quantity. A
    single-leg option order is its leg's contract, side and contract count,
    with the position intent read off the account's positions in the
    re-check's context. The limit is the proposal's limit price.
    """
    key = verdict.id
    problem = shape_problem(proposal)
    if problem is not None:
        raise PlacementError(f"place order ({problem})", key)
    order = proposal.proposal
    fields: dict[str, Any] = {
        "proposal_id": order.id,
        "decision_id": verdict.id,
        "regate_decision_id": regate.id,
        "approval_id": None if approval is None else approval.id,
        "client_order_id": CLIENT_ORDER_ID_PREFIX + order.id,
        "instrument": order.instrument,
        "limit_price": Decimal(repr(order.limit_price)),
    }
    if order.instrument is Instrument.EQUITY:
        fields.update(symbol=order.symbol, side=order.side, quantity=_whole(order.quantity, "quantity", key))
    else:
        leg = proposal.legs[0]
        quantity = _whole(leg.quantity, "quantity", key)
        fields.update(
            symbol=leg.symbol,
            side=leg.side,
            quantity=quantity,
            position_intent=_intent(leg.symbol, leg.side, quantity, context, key),
        )
    try:
        return ApprovedOrder(**fields)
    except ValidationError as exc:
        raise PlacementError("place order (the order cannot be built)", key, exc) from exc


def _claimable(approved: ApprovedOrder) -> Order:
    """The store row ``claim_order`` writes for ``approved``: the same order,
    in the store's terms."""
    return Order(
        id=new_id(),
        proposal_id=approved.proposal_id,
        client_order_id=approved.client_order_id,
        broker=Broker.PAPER,
        status=OrderStatus.APPROVED,
        symbol=approved.symbol,
        side=approved.side,
        quantity=float(approved.quantity),
        limit_price=float(approved.limit_price),
        decision_id=approved.decision_id,
        regate_decision_id=approved.regate_decision_id,
        approval_id=approved.approval_id,
        instrument=approved.instrument,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        position_intent=approved.position_intent,
    )


def describe(order: ApprovedOrder | Order) -> str:
    """One line: ``buy 10 AAPL limit 200.5 day``."""
    intent = f" ({order.position_intent.value})" if order.position_intent is not None else ""
    return (
        f"{order.side.value} {_exact(order.quantity)} {order.symbol} limit"
        f" {_exact(order.limit_price)} day{intent}"
    )


# --- recording what the broker said -------------------------------------------


def _figures(receipt: OrderReceipt) -> dict[str, Any]:
    return {
        "status": receipt.status,
        "broker_order_id": receipt.broker_order_id,
        "broker_status": receipt.broker_status,
        "filled_quantity": float(receipt.filled_quantity),
        "avg_fill_price": None if receipt.avg_fill_price is None else float(receipt.avg_fill_price),
    }


def _record_receipt(
    conn: sqlite3.Connection, order: Order, receipt: OrderReceipt, kind: str, level: EventLevel
) -> Order:
    """The broker's receipt, through ``apply_broker_update``, with an event."""
    figures = _figures(receipt)
    event = _event(
        receipt.fetched_at, level, kind,
        f"{order.client_order_id}: {receipt.broker_status} ({figures['filled_quantity']:g}"
        f" of {order.quantity:g} filled)",
        client_order_id=order.client_order_id, broker_order_id=receipt.broker_order_id,
        broker_status=receipt.broker_status, filled_quantity=figures["filled_quantity"],
        avg_fill_price=figures["avg_fill_price"],
    )
    stored, _ = apply_broker_update(
        conn, order.client_order_id, synced_at=receipt.fetched_at, event=event, **figures
    )
    return stored


def _record_failure(
    conn: sqlite3.Connection, order: Order, error: ExecutionError, now: datetime
) -> Order:
    """A send that did not end in a receipt, on the record: ``failed`` when
    the broker refused or nothing was sent, ``approved`` with an unknown
    outcome otherwise. Never resent: the reason is written either way."""
    detail = _one_line(error)
    if error.outcome is ExecutionOutcome.UNKNOWN:
        status, reason, kind, level = (
            OrderStatus.APPROVED, f"{UNKNOWN_OUTCOME}: {detail}", ORDER_UNKNOWN_OUTCOME,
            EventLevel.CRITICAL,
        )
    elif error.outcome is ExecutionOutcome.REJECTED:
        status, reason, kind, level = (
            OrderStatus.FAILED, f"rejected: {detail}", ORDER_REJECTED, EventLevel.WARNING,
        )
    else:
        status, reason, kind, level = (
            OrderStatus.FAILED, f"not_sent: {detail}", ORDER_NOT_SENT, EventLevel.WARNING,
        )
    event = _event(
        now, level, kind, f"{order.client_order_id}: {reason}",
        client_order_id=order.client_order_id, outcome=error.outcome.value,
        status_code=error.status_code,
    )
    stored, _ = apply_broker_update(
        conn, order.client_order_id, status=status, synced_at=now, status_reason=reason, event=event
    )
    return stored


def _mark_interrupted(conn: sqlite3.Connection, order: Order, error: BaseException, now: datetime) -> None:
    """Best effort, on the way out of an interrupted send: the attempt goes
    on the record so the order is looked up, never sent blind again."""
    reason = f"{UNKNOWN_OUTCOME}: the send was interrupted ({type(error).__name__})"
    try:
        apply_broker_update(
            conn, order.client_order_id, status=OrderStatus.APPROVED, synced_at=now,
            status_reason=reason,
            event=_event(now, EventLevel.CRITICAL, ORDER_UNKNOWN_OUTCOME,
                         f"{order.client_order_id}: {reason}", client_order_id=order.client_order_id),
        )
    except Exception:  # the original interruption is what the caller must see
        pass


# --- approve / reject ---------------------------------------------------------


def answer(
    conn: sqlite3.Connection,
    decision_id: str,
    *,
    approved: bool,
    by: str,
    config: AegisConfig,
    note: str | None = None,
    clock: Clock = wall_clock,
) -> Approval:
    """A human's answer to one NEEDS_APPROVAL verdict (D7): written once,
    tied to that decision; an approval is good for
    ``broker.approval_ttl_seconds``. It places nothing: ``place`` does, and
    re-checks the proposal first either way."""
    what = "approve" if approved else "reject"
    responder = _one_line(by)
    if not responder:
        raise PlacementError(f"{what} (say who answers: --by NAME)", decision_id)
    decision = _verdict(conn, decision_id, what)
    if decision.verdict is not Verdict.NEEDS_APPROVAL:
        raise PlacementError(
            f"{what} (the verdict is {decision.verdict.value}: only NEEDS_APPROVAL awaits an answer)",
            decision_id,
        )
    existing = get_decision_approval(conn, decision_id)
    if existing is not None and existing.response is not None:
        raise PlacementError(
            f"{what} (already answered: {existing.response.value} by {existing.responder}"
            f" at {existing.responded_at.isoformat() if existing.responded_at else 'unknown'})",
            decision_id,
        )
    now = _now(clock)
    expires_at = now + timedelta(seconds=config.broker.approval_ttl_seconds) if approved else None
    if existing is not None:
        # A request row recorded before the answer fixes its own expiry (the
        # store never changes it), so a "yes" written onto it is only good
        # until then. One with no expiry, or one already past, could never be
        # placed: refuse rather than write an answer that cannot be used.
        expires_at = existing.expires_at
        if approved and (expires_at is None or not now < expires_at):
            when = "none set" if expires_at is None else f"it expired {expires_at.isoformat()}"
            raise PlacementError(
                f"{what} (the approval request on record cannot carry a yes: {when})", decision_id
            )
    response = ApprovalResponse.APPROVED if approved else ApprovalResponse.REJECTED
    clean_note = _one_line(note) if note else None
    return record_approval(
        conn,
        Approval(
            id=existing.id if existing is not None else new_id(),
            proposal_id=decision.proposal_id,
            requested_at=decision.decided_at,
            responded_at=now,
            response=response,
            channel=APPROVAL_CHANNEL,
            responder=responder,
            decision_id=decision.id,
            expires_at=expires_at,
            note=clean_note,
        ),
        event=_event(
            now, EventLevel.INFO, APPROVAL_ANSWERED,
            f"decision {decision.id} ({decision.proposal_id}): {response.value} by {responder}",
            decision_id=decision.id, proposal_id=decision.proposal_id, response=response.value,
            responder=responder,
            expires_at=None if expires_at is None else expires_at.isoformat(),
        ),
    )


# --- place --------------------------------------------------------------------


def place(
    conn: sqlite3.Connection,
    decision_id: str,
    *,
    config: AegisConfig,
    broker_factory: BrokerFactory | None = None,
    context_builder: ContextBuilder | None = None,
    clock: Clock = wall_clock,
    dry_run: bool = False,
    broker: Executor | None = None,
) -> Placement:
    """Place the one order ``decision_id`` authorises, exactly once.

    The steps are the module docstring's. A ``dry_run`` runs the checks and
    the re-check (unsaved) and stops before the claim: nothing is written
    and nothing is sent. Without ``broker.enabled`` only a dry run is
    possible. Any refusal is a ``PlacementError``; one after the claim
    leaves the order on the record with what happened.
    """
    if not dry_run and not config.broker.enabled:
        raise PlacementError(
            "place order (broker.enabled is false in config.yaml: nothing is sent;"
            " --dry-run shows what would be)",
            decision_id,
        )
    now = _now(clock)
    verdict = _verdict(conn, decision_id, "place order")
    approval = _standing(conn, verdict, config, now)
    proposal = load_proposal(conn, verdict.proposal_id)
    problem = shape_problem(proposal)
    if problem is not None:
        raise PlacementError(f"place order ({problem})", decision_id)
    context = (context_builder or build_context)(conn, proposal, config=config)
    regate = evaluate(
        proposal, context, conn, dry_run=dry_run, purpose=DecisionPurpose.PRE_SUBMIT
    )
    why_not = regate_refusal(verdict, regate)
    if why_not is not None:
        if not dry_run:
            log_event(conn, _event(
                _now(clock), EventLevel.WARNING, ORDER_REFUSED,
                f"decision {verdict.id}: {why_not}", decision_id=verdict.id,
                regate_decision_id=regate.id,
            ))
        raise PlacementError(f"place order ({why_not})", decision_id)
    try:
        approved = mint(proposal, verdict, regate, approval, context)
    except PlacementError as exc:
        if not dry_run:  # the re-check is on record; so is why nothing followed it
            log_event(conn, _event(
                _now(clock), EventLevel.WARNING, ORDER_REFUSED,
                f"decision {verdict.id}: {exc.what}", decision_id=verdict.id,
                regate_decision_id=regate.id,
            ))
        raise
    if dry_run:
        return Placement(regate, None, (
            f"would place {approved.client_order_id}: {describe(approved)}",
            f"re-check: {regate.verdict.value} ({regate.notes})",
        ))
    venue = broker if broker is not None else _broker(conn, config, broker_factory)
    return _claim_and_send(conn, venue, approved, regate, config, clock)


def _claim_and_send(
    conn: sqlite3.Connection,
    venue: Executor,
    approved: ApprovedOrder,
    regate: PolicyDecision,
    config: AegisConfig,
    clock: Clock,
) -> Placement:
    key = approved.client_order_id
    now = _now(clock)
    start, end = trading_day(now, config)
    try:
        claimed = claim_order(
            conn, _claimable(approved), now=now, day_start=start, day_end=end,
            max_daily_trades=config.risk_limits.max_daily_trades,
            event=_event(
                now, EventLevel.INFO, ORDER_CLAIMED, f"{key}: claimed, {describe(approved)}",
                client_order_id=key, decision_id=approved.decision_id,
                regate_decision_id=approved.regate_decision_id, approval_id=approved.approval_id,
                symbol=approved.symbol, side=approved.side.value, quantity=approved.quantity,
                limit_price=str(approved.limit_price),
            ),
        )
    except ClaimRefused as exc:
        log_event(conn, _event(now, EventLevel.WARNING, ORDER_REFUSED, f"{key}: {exc.reason}",
                               client_order_id=key, decision_id=approved.decision_id))
        raise PlacementError(f"place order (the claim was refused: {exc.reason})", key) from exc
    lines = [f"claimed {key}: {describe(approved)}", f"re-check: {regate.verdict.value}"]
    try:
        receipt = venue.submit_order(approved)
    except ExecutionError as exc:
        recorded = _record_failure(conn, claimed, exc, _now(clock))
        if exc.outcome is not ExecutionOutcome.UNKNOWN:
            raise _broker_error(f"place order ({recorded.status_reason})", key, exc) from exc
        return _resolve_unknown(conn, venue, recorded, regate, exc, lines, clock)
    except BaseException as exc:
        _mark_interrupted(conn, claimed, exc, _now(clock))
        raise
    try:
        stored = _record_receipt(conn, claimed, receipt, ORDER_SUBMITTED, EventLevel.INFO)
    except StoreError as exc:
        _mark_interrupted(conn, claimed, exc, _now(clock))
        raise PlacementError(
            f"place order (the broker has it as {receipt.broker_order_id}, {receipt.broker_status},"
            " but the store could not record its answer: run `orders sync`)",
            key, exc, outcome=ExecutionOutcome.UNKNOWN.value,
        ) from exc
    lines.append(f"sent: {stored.broker_status} at the broker as {stored.broker_order_id}")
    return Placement(regate, stored, tuple(lines))


def _resolve_unknown(
    conn: sqlite3.Connection,
    venue: Executor,
    order: Order,
    regate: PolicyDecision,
    error: ExecutionError,
    lines: list[str],
    clock: Clock,
) -> Placement:
    """After an unknown outcome: one lookup by name, never a resend. Found,
    the order is adopted; not found or not readable, it stays claimed with
    the reason on record for ``sync`` to settle."""
    key = order.client_order_id
    try:
        receipt = venue.get_order(key)
    except ExecutionError:
        receipt = None
    if receipt is None:
        raise _broker_error(
            "place order (the send ended unclear and the broker does not show the order yet:"
            " it is never sent again; `orders sync` settles it)",
            key, error,
        ) from error
    stored = _record_receipt(conn, order, receipt, ORDER_ADOPTED, EventLevel.WARNING)
    lines.append(
        f"the send ended unclear ({error.outcome.value}); the broker has it:"
        f" {stored.broker_status} as {stored.broker_order_id}"
    )
    return Placement(regate, stored, tuple(lines))


# --- sync ---------------------------------------------------------------------


def _changed(order: Order, receipt: OrderReceipt) -> bool:
    return (
        order.status is not receipt.status
        or order.broker_order_id != receipt.broker_order_id
        or order.broker_status != receipt.broker_status
        or order.filled_quantity != float(receipt.filled_quantity)
    )


def _follow(
    conn: sqlite3.Connection, venue: Executor, order: Order, config: AegisConfig, now: datetime
) -> Report:
    """One open order brought up to date with the broker."""
    key = order.client_order_id
    try:
        receipt = venue.get_order(key)
    except ExecutionError as exc:
        return Report((f"{key}: could not be read at the broker ({_one_line(exc)})",), 1)
    if receipt is None:
        return _not_found(conn, order, config, now)
    if not _changed(order, receipt):
        return Report((f"{key}: {order.status.value}, unchanged",))
    adopted = order.status is OrderStatus.APPROVED
    try:
        stored = _record_receipt(
            conn, order, receipt, ORDER_ADOPTED if adopted else ORDER_UPDATED,
            EventLevel.WARNING if adopted else EventLevel.INFO,
        )
    except StoreError as exc:
        log_event(conn, _event(
            now, EventLevel.CRITICAL, ORDER_UNTRACKED,
            f"{key}: the broker says {receipt.broker_status} ({receipt.filled_quantity} filled)"
            f" and the store refused it: {_one_line(exc)}",
            client_order_id=key, broker_status=receipt.broker_status,
        ))
        return Report((f"{key}: the broker's answer could not be recorded ({_one_line(exc)})",), 1)
    return Report((
        f"{key}: {order.status.value} -> {stored.status.value}"
        f" ({stored.filled_quantity:g} of {stored.quantity:g} filled)",
    ))


def _not_found(conn: sqlite3.Connection, order: Order, config: AegisConfig, now: datetime) -> Report:
    """The broker does not know an order the store holds open."""
    key = order.client_order_id
    if order.status is not OrderStatus.APPROVED:
        log_event(conn, _event(
            now, EventLevel.CRITICAL, ORDER_UNTRACKED,
            f"{key}: {order.status.value} in the store, unknown at the broker",
            client_order_id=key, broker_order_id=order.broker_order_id,
        ))
        return Report((f"{key}: {order.status.value} in the store but unknown at the broker",), 1)
    age = _seconds(now - order.submitted_at) if order.submitted_at is not None else 0.0
    grace = config.broker.not_found_grace_seconds
    if age < grace:
        return Report((f"{key}: not at the broker yet ({age:.0f} s since the claim, grace {grace:g} s)",))
    reason = "not_found_at_broker"
    apply_broker_update(
        conn, key, status=OrderStatus.CANCELLED, synced_at=now, status_reason=reason,
        event=_event(now, EventLevel.WARNING, ORDER_NOT_FOUND,
                     f"{key}: unknown at the broker {age:.0f} s after the claim: cancelled"
                     " (it still counts toward the daily cap)",
                     client_order_id=key),
    )
    return Report((f"{key}: approved -> cancelled (the broker never had it)",))


def _orphans(conn: sqlite3.Connection, venue: Executor, now: datetime) -> Report:
    """Every order open at the broker that the store does not hold open (D10):
    a CRITICAL event each, and the kill switch on."""
    try:
        at_broker = venue.get_open_orders()
    except ExecutionError as exc:
        return Report((f"open orders at the broker could not be read ({_one_line(exc)})",), 1)
    known = {order.client_order_id for order in get_execution_orders(conn)}
    orphans = [receipt for receipt in at_broker if receipt.client_order_id not in known]
    if not orphans:
        return Report()
    lines = []
    for receipt in orphans:
        log_event(conn, _event(
            now, EventLevel.CRITICAL, ORPHAN_BROKER_ORDER,
            f"{receipt.client_order_id} ({receipt.broker_order_id}): open at the broker,"
            f" not held open in the store ({receipt.broker_status} {receipt.symbol})",
            client_order_id=receipt.client_order_id, broker_order_id=receipt.broker_order_id,
            symbol=receipt.symbol, broker_status=receipt.broker_status,
        ))
        lines.append(f"ORPHAN {receipt.client_order_id} ({receipt.broker_order_id}) {receipt.symbol}")
    controls = get_controls(conn)
    if not controls.kill_switch:
        set_kill_switch(conn, True, now=now, event=_event(
            now, EventLevel.CRITICAL, KILL_SWITCH_EVENT,
            f"kill switch on: {len(orphans)} order(s) open at the broker that the store does not hold",
            previous="off", source="orders sync", orphans=[r.client_order_id for r in orphans],
        ))
        lines.append("kill switch ON (an order the store does not hold is working)")
    return Report(tuple(lines), len(orphans))


def _sync(conn: sqlite3.Connection, venue: Executor, config: AegisConfig, clock: Clock) -> Report:
    report = Report()
    for order in get_execution_orders(conn):
        report = report + _follow(conn, venue, order, config, _now(clock))
    return report + _orphans(conn, venue, _now(clock))


def sync(
    conn: sqlite3.Connection,
    *,
    config: AegisConfig,
    broker_factory: BrokerFactory | None = None,
    clock: Clock = wall_clock,
) -> Report:
    """Bring every open order up to date with the broker, and look for orders
    the store does not hold. Works whether or not placing is enabled."""
    return _sync(conn, _broker(conn, config, broker_factory), config, clock)


# --- cancel and stand down ----------------------------------------------------


def _cancel_one(
    conn: sqlite3.Connection, venue: Executor, order: Order, config: AegisConfig, clock: Clock,
    why: str,
) -> Report:
    """Cancel one open order at the broker and read it back."""
    key = order.client_order_id
    now = _now(clock)
    if order.broker_order_id is None:  # never acknowledged: learn whether the broker has it
        settled = _follow(conn, venue, order, config, now)
        order = get_order(conn, key) or order
        if order.status in TERMINAL_ORDER_STATUSES:
            return settled
        if order.broker_order_id is None:
            return settled + Report((f"{key}: not cancelled, the broker does not show it yet",), 1)
    try:
        venue.cancel_order(order.broker_order_id)
    except ExecutionError as exc:
        log_event(conn, _event(now, EventLevel.CRITICAL, ORDER_CANCEL_REQUESTED,
                               f"{key}: cancel failed ({_one_line(exc)})", client_order_id=key))
        return Report((f"{key}: cancel failed ({_one_line(exc)})",), 1)
    log_event(conn, _event(now, EventLevel.INFO, ORDER_CANCEL_REQUESTED,
                           f"{key}: cancel requested ({why})", client_order_id=key,
                           broker_order_id=order.broker_order_id))
    return Report((f"{key}: cancel requested",)) + _follow(conn, venue, order, config, _now(clock))


def cancel(
    conn: sqlite3.Connection,
    client_order_id: str,
    *,
    config: AegisConfig,
    broker_factory: BrokerFactory | None = None,
    clock: Clock = wall_clock,
) -> Report:
    """Cancel one open order (by its client order id) and read it back. An
    order already filled, cancelled or failed is reported as it is."""
    order = get_order(conn, client_order_id)
    if order is None or order.decision_id is None:
        raise PlacementError("cancel order (no order placed under this name)", client_order_id)
    if order.status in TERMINAL_ORDER_STATUSES:
        return Report((f"{client_order_id}: already {order.status.value}, nothing to cancel",))
    venue = _broker(conn, config, broker_factory)
    return _cancel_one(conn, venue, order, config, clock, "operator")


def _stop_reason(conn: sqlite3.Connection, now: datetime) -> str | None:
    controls = get_controls(conn)
    if controls.kill_switch:
        return "the kill switch is on"
    if controls.halt_unknown:
        return "the halt cannot be read"
    if controls.halt_until is not None and controls.halt_until > now:
        return f"trading is halted until {controls.halt_until.isoformat()}"
    return None


def _stand_down(conn: sqlite3.Connection, venue: Executor | None, config: AegisConfig,
                clock: Clock, factory: BrokerFactory | None) -> Report:
    now = _now(clock)
    why = _stop_reason(conn, now)
    if why is None:
        return Report(("controls clear: nothing to stand down",))
    working = get_execution_orders(conn)
    if not working:
        return Report((f"{why}: no working orders",))
    venue = venue if venue is not None else _broker(conn, config, factory)
    report = Report()
    for order in working:
        report = report + _cancel_one(conn, venue, order, config, clock, f"stand down: {why}")
    log_event(conn, _event(
        _now(clock), EventLevel.WARNING if not report.failures else EventLevel.CRITICAL, STAND_DOWN,
        f"stand down ({why}): {len(working)} working order(s), {report.failures} not cancelled",
        reason=why, orders=[order.client_order_id for order in working], failures=report.failures,
    ))
    return Report((f"standing down: {why}",)) + report


def stand_down(
    conn: sqlite3.Connection,
    *,
    config: AegisConfig,
    broker_factory: BrokerFactory | None = None,
    clock: Clock = wall_clock,
) -> Report:
    """With the kill switch on, a halt in force or a halt that cannot be read,
    cancel every working order (D4). With the controls clear, do nothing.
    Never flattens a position (D5)."""
    return _stand_down(conn, None, config, clock, broker_factory)


# --- tick -----------------------------------------------------------------------


def _open_verdict(conn: sqlite3.Connection, decision: PolicyDecision) -> bool:
    """Whether ``decision`` is the latest verdict on a proposal with no order."""
    latest = get_latest_verdict(conn, decision.proposal_id)
    if latest is None or latest.id != decision.id:
        return False
    return get_proposal_order(conn, decision.proposal_id) is None


def awaiting_approval(
    conn: sqlite3.Connection, config: AegisConfig, now: datetime
) -> list[PolicyDecision]:
    """Today's NEEDS_APPROVAL verdicts that no one has answered: the latest
    verdict on their proposal, with no order yet, oldest first. A read."""
    start, _ = trading_day(now, config)
    waiting = [
        decision
        for decision in get_verdicts_since(conn, start, Verdict.NEEDS_APPROVAL)
        if _open_verdict(conn, decision)
        and ((answer := get_decision_approval(conn, decision.id)) is None or answer.response is None)
    ]
    return sorted(waiting, key=lambda d: d.decided_at)


def ready_decisions(
    conn: sqlite3.Connection, config: AegisConfig, now: datetime
) -> list[PolicyDecision]:
    """The decisions that may still go (what ``tick`` places): AUTO_EXECUTE
    verdicts younger than ``broker.max_decision_age_seconds``, and
    NEEDS_APPROVAL verdicts with a live approval; the latest verdict on
    their proposal, with no order yet, one per proposal, oldest first. A
    read: the re-check and the claim still decide."""
    return [get_decision(conn, decision_id) for decision_id in _candidates(conn, config, now)]


def _candidates(conn: sqlite3.Connection, config: AegisConfig, now: datetime) -> list[str]:
    """The ids ``ready_decisions`` lists."""
    since = now - timedelta(seconds=config.broker.max_decision_age_seconds)
    found = [d for d in get_verdicts_since(conn, since, Verdict.AUTO_EXECUTE)]
    found += [
        decision
        for approval in get_live_approvals(conn, now)
        if (decision := get_decision(conn, approval.decision_id)) is not None
    ]
    chosen: dict[str, PolicyDecision] = {}
    for decision in sorted(found, key=lambda d: d.decided_at):
        if _open_verdict(conn, decision):
            chosen.setdefault(decision.proposal_id, decision)
    return [decision.id for decision in chosen.values()]


def tick(
    conn: sqlite3.Connection,
    *,
    config: AegisConfig,
    broker_factory: BrokerFactory | None = None,
    context_builder: ContextBuilder | None = None,
    clock: Clock = wall_clock,
) -> Report:
    """One pass of the loop: ``sync``, then ``stand_down``, then ``place``
    every decision that may still go, while the controls are clear,
    ``broker.enabled`` is on, and sync and stand-down resolved everything
    (an account that could not be fully read is no account to add an order
    to). A placement refused before the broker (by the verdict, the
    re-check or the claim) is a line, not a failure; one the broker refused,
    never received or may have received is a failure."""
    venue = _broker(conn, config, broker_factory)
    report = _sync(conn, venue, config, clock)
    report = report + _stand_down(conn, venue, config, clock, broker_factory)
    if _stop_reason(conn, _now(clock)) is not None:
        return report + Report(("placing nothing: the controls say stop",))
    if not config.broker.enabled:
        return report + Report(("placing nothing: broker.enabled is false",))
    if report.failures:
        return report + Report(("placing nothing: the broker could not be fully read",))
    for decision_id in _candidates(conn, config, _now(clock)):
        try:
            placement = place(
                conn, decision_id, config=config, context_builder=context_builder, clock=clock,
                broker=venue,
            )
        except PlacementError as exc:
            failed = exc.outcome is not None  # it reached, or may have reached, the broker
            report = report + Report((f"decision {decision_id}: {_one_line(exc)}",), int(failed))
            continue
        report = report + Report(placement.lines, 0, (placement.order,) if placement.order else ())
    return report


__all__ = [
    "Placement", "Report", "answer", "awaiting_approval", "cancel", "describe", "mint",
    "paper_broker", "place", "ready_decisions", "regate_refusal", "shape_problem", "stand_down",
    "sync", "tick",
]
