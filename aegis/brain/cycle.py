"""The cycle orchestrator: scan → thesis → proposal, journaled as it runs.

``run_cycle`` is the brain's entry point (the ``once`` CLI and, later, the
scheduler call it). It mints a cycle id, takes or builds the market
snapshot, and runs the three stages through a ``GuardedLLM`` — the
``BudgetGuard`` in front of whichever ``LLMClient`` the caller passes, the
real ``AnthropicLLM`` or a ``FakeLLM`` — writing the audit trail as it goes:

- one ``reasoning`` row per completed stage (the model's raw text, its
  tokens, the model that answered and the latency) under the cycle id,
  linked to the proposal once there is one — in the transaction that
  writes the proposal (``insert_proposal(..., link_cycle_reasoning=True)``).
  No billed token may escape the table the budget guard sums: when
  anything cuts the cycle short, the guard's in-memory tally is compared
  with what the table holds for the cycle and the difference is written as
  one accounting row for the stage in progress (``journal_unrecorded_spend``)
  — its content the stage's last output when there was one
  (``StageOutputInvalid.raw_output``), else a ``(no usable output: …)``
  line, never the prompt;
- the ``proposals`` row and its ``proposal_legs`` when the proposal stage
  trades, written together with that link, all or none;
- events: ``cycle_start``, ``market_snapshot`` (the scan stage's whole
  input — the audit copy of what the model was shown), then ``proposal``
  or ``no_trade``, ``brain_error`` on a failure, and always ``cycle_end``
  with the outcome, the tokens (a failed stage's included) and the cost
  estimate. The guard logs its own ``budget_halt``. Every payload carries
  ``cycle_id``.

Outcomes (``CycleResult.outcome``):

- ``proposal`` — a proposals row exists; ``proposal_id`` names it.
- ``no_trade`` — the thesis stage had no candidate, no candidate could be
  resolved to an offer (a contract missing from the chain, an equity
  without a two-sided quote), or the proposal stage declined;
  ``no_trade_reason`` says which. Declining is a first-class result, not a
  failure.
- ``halted`` — the guard refused a call. ``BudgetExceeded`` becomes the
  outcome rather than propagating; rows written before the halt stay.
- ``failed`` — any other error: a ``BrainError`` (a call that failed after
  retries, a refused or truncated answer, output that never validated, a
  snapshot that could not be built), one that is not a ``BrainError`` at
  all (the store failing mid-cycle, a client that broke its contract), or
  an interrupt (``KeyboardInterrupt``, ``SystemExit``) that lands while a
  stage is retrying. A ``brain_error`` event, the ``cycle_end``, then the
  error is re-raised as it is. The accounting row and the two events are
  each written on a best-effort basis, so a broken store cannot hide the
  original error.

Candidates are tried in the thesis stage's order and the first that
resolves is offered; one that cannot be resolved is skipped with its
reason, and the ``no_trade`` event lists those reasons when none resolves.

The brain's authority ends at the proposals row: nothing here imports
``aegis.execution`` (``tests/test_brain_architecture.py`` pins it).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from aegis.brain.errors import BrainError, BudgetExceeded
from aegis.brain.llm import BudgetGuard, GuardedLLM, LLMClient, LLMResponseError, estimate_cost
from aegis.brain.models import (
    CycleOutcome,
    CycleResult,
    InstrumentOffer,
    MarketSnapshot,
    ProposalResult,
    ScanBrief,
    StageUsage,
    ThesisCandidate,
    TradeProposal,
)
from aegis.brain.proposal import resolve_offer, run_proposal, trade_records
from aegis.brain.scan import run_scan
from aegis.brain.snapshot import build_market_snapshot
from aegis.brain.stage import StageOutputInvalid
from aegis.brain.thesis import run_thesis
from aegis.config import AegisConfig, get_config
from aegis.data.models import utcnow
from aegis.store.errors import StoreError
from aegis.store.models import (
    Event,
    EventLevel,
    Instrument,
    Proposal,
    ProposalLeg,
    Reasoning,
    ReasoningStage,
    new_id,
)
from aegis.store.repo import (
    add_reasoning,
    get_cycle_token_usage,
    get_recent_proposals,
    insert_proposal,
    log_event,
)

MARKET_SNAPSHOT_EVENT = "market_snapshot"
"""Event kind carrying a cycle's snapshot; ``snapshot_from_event`` reads it back."""


# --- store records and events (pure; the stage CLI reuses them) -----------------


def _describe(exc: BaseException) -> str:
    """``<type>: <message>``, or just ``<type>`` for an error without a
    message (a bare ``KeyboardInterrupt``)."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def reasoning_row(cycle_id: str, usage: StageUsage, raw_output: str) -> Reasoning:
    """The reasoning row for one finished stage: the raw text and the usage,
    under ``cycle_id`` and not yet linked to a proposal. ``model_name`` is the
    model that answered (``StageUsage.response_model``), falling back to the
    requested one when no response model is known."""
    return Reasoning(
        cycle_id=cycle_id,
        proposal_id=None,
        stage=usage.stage,
        content=raw_output,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        model_name=usage.response_model or usage.model,
        latency_ms=usage.latency_ms,
    )


def unrecorded_usage(
    guard: BudgetGuard, recorded: Sequence[StageUsage], stage: ReasoningStage, model: str
) -> StageUsage | None:
    """What ``guard`` tallied beyond the ``recorded`` usage — the responses a
    stage received before an error cut it short — as that stage's usage
    (``model`` the requested one, ``response_model`` the last that
    answered); None when every billed response is already counted."""
    tokens_in = guard.tokens_in - sum(usage.tokens_in for usage in recorded)
    tokens_out = guard.tokens_out - sum(usage.tokens_out for usage in recorded)
    calls = guard.calls - sum(usage.attempts for usage in recorded)
    if tokens_in <= 0 and tokens_out <= 0 and calls <= 0:
        return None
    return StageUsage(
        stage=stage,
        model=model,
        response_model=guard.last_model,
        tokens_in=max(0, tokens_in),
        tokens_out=max(0, tokens_out),
        latency_ms=max(0.0, guard.latency_ms - sum(usage.latency_ms for usage in recorded)),
        attempts=max(0, calls),
    )


def journal_unrecorded_spend(
    conn: sqlite3.Connection,
    guard: BudgetGuard,
    cycle_id: str,
    stage: ReasoningStage | None,
    stage_model: str,
    recorded: Sequence[StageUsage],
    exc: BaseException,
) -> StageUsage | None:
    """Account for the spend ``exc`` cut short, so no billed token escapes.

    When the guard's tally for the cycle exceeds what the reasoning table
    holds for it, one accounting row is written for ``stage`` (the stage in
    progress) carrying the unjournaled tokens, the model that answered last
    (else ``stage_model``) and, as content, the stage's last output when it
    is known — ``StageOutputInvalid.raw_output``, or the billed text of a
    truncated or refused answer (``LLMResponseError.response.text``) — else
    ``(no usable output: <type>: <message>)`` (just ``<type>`` when there is
    no message, as for a Ctrl-C) — never the prompt. The error itself is on
    the ``brain_error`` event. The write is best effort: a failure is
    swallowed so it cannot mask ``exc``, and nothing extra is logged.
    Returns the usage ``recorded`` lacks (``unrecorded_usage``) for the
    caller's totals; None before any stage ran.
    """
    if stage is None:
        return None
    pending = unrecorded_usage(guard, recorded, stage, stage_model)
    if isinstance(exc, StageOutputInvalid):
        content = exc.raw_output
    elif isinstance(exc, LLMResponseError) and exc.response.text.strip():
        content = exc.response.text  # billed, so it is part of the record
    else:
        content = f"(no usable output: {_describe(exc)})"
    try:
        journaled = get_cycle_token_usage(conn, cycle_id)
        if guard.recorded_tokens > journaled.total:
            add_reasoning(
                conn,
                Reasoning(
                    cycle_id=cycle_id,
                    proposal_id=None,
                    stage=stage,
                    content=content,
                    tokens_in=max(0, guard.tokens_in - journaled.tokens_in),
                    tokens_out=max(0, guard.tokens_out - journaled.tokens_out),
                    model_name=guard.last_model or stage_model,
                    latency_ms=pending.latency_ms if pending is not None else None,
                ),
            )
    except Exception:  # best effort: never mask the error being handled
        pass
    return pending


def proposal_records(cycle_id: str, result: ProposalResult) -> tuple[Proposal, list[ProposalLeg]]:
    """The store rows for a proposal-stage trade: the proposal (verbatim
    model output, the requested model — the stage's reasoning row names the
    one that answered — and the prompt version) and one leg per ``LegSpec``
    in order, built by ``proposal.trade_records`` — the same rows the
    stage's validation already proved storable. ``ValueError`` when the
    result is a no-trade."""
    trade = result.output.proposal
    if trade is None:
        raise ValueError("the proposal stage did not trade; there is nothing to store")
    return trade_records(
        cycle_id,
        trade,
        raw_output=result.raw_output,
        model_name=result.usage.model,
        prompt_version=result.prompt_version,
    )


def _market_label(snapshot: MarketSnapshot) -> str:
    market_open = snapshot.data_age.market_open
    return "OPEN" if market_open is True else "CLOSED" if market_open is False else "UNKNOWN"


def market_snapshot_event(
    cycle_id: str, snapshot: MarketSnapshot, *, source_cycle: str | None = None
) -> Event:
    """The audit copy of the scan stage's input: the whole snapshot as JSON
    plus a top-level ``cycle_id`` key, so ``stage --cycle`` can find a
    cycle's snapshot among the events (the snapshot itself carries no cycle
    id). The extra key is an accepted deviation from the spec's literal
    ``payload=snapshot.model_dump(mode="json")``: every other cycle payload
    carries ``cycle_id`` too, and ``snapshot_from_event`` drops it again.
    ``source_cycle`` names, in the message only, the cycle a reused
    snapshot was reloaded from (``stage --cycle``)."""
    reloaded = f" (reloaded from cycle {source_cycle})" if source_cycle is not None else ""
    return Event(
        level=EventLevel.INFO,
        kind=MARKET_SNAPSHOT_EVENT,
        message=(
            f"market snapshot for cycle {cycle_id}: {len(snapshot.symbols)} symbol(s), "
            f"market {_market_label(snapshot)}{reloaded}"
        ),
        payload={"cycle_id": cycle_id, **snapshot.model_dump(mode="json")},
    )


def snapshot_from_event(event: Event) -> MarketSnapshot:
    """The snapshot a ``market_snapshot`` event carries; ``ValueError`` for
    another kind of event or a payload that no longer validates."""
    if event.kind != MARKET_SNAPSHOT_EVENT or event.payload is None:
        raise ValueError(f"event {event.id} is a {event.kind!r} event, not a market snapshot")
    fields = {key: value for key, value in event.payload.items() if key != "cycle_id"}
    return MarketSnapshot.model_validate(fields)


def _cycle_event(
    kind: str, message: str, payload: dict[str, Any], level: EventLevel = EventLevel.INFO
) -> Event:
    return Event(level=level, kind=kind, message=message, payload=payload)


def brain_error_event(cycle_id: str, stage: ReasoningStage | None, exc: BaseException) -> Event:
    """The ERROR ``brain_error`` event for a cycle (or a ``stage`` CLI run)
    that ``exc`` cut short: the stage in progress and the error — a
    ``BrainError``'s own message, anything else as ``<type>: <message>``."""
    error = str(exc) if isinstance(exc, BrainError) else _describe(exc)
    return _cycle_event(
        "brain_error",
        f"cycle {cycle_id} failed: {error}",
        {"cycle_id": cycle_id, "stage": stage.value if stage is not None else None, "error": error},
        level=EventLevel.ERROR,
    )


# --- candidates -----------------------------------------------------------------


def describe_candidate(candidate: ThesisCandidate) -> str:
    """``SPY call_debit_spread 640/650 exp 2026-08-21`` or ``QQQ equity``."""
    if candidate.instrument is not Instrument.OPTION or candidate.structure is None:
        return f"{candidate.symbol} equity"
    strikes = "/".join(f"{strike:g}" for strike in candidate.strikes)
    expiration = candidate.expiration.isoformat() if candidate.expiration else "unknown"
    return f"{candidate.symbol} {candidate.structure.value} {strikes} exp {expiration}"


def first_offer(
    candidates: Sequence[ThesisCandidate],
    snapshot: MarketSnapshot,
    *,
    contract_multiplier: int,
) -> tuple[InstrumentOffer | None, tuple[str, ...]]:
    """The first candidate that resolves to an offer, and why each candidate
    before it did not (``"<candidate>: <reason>"``). ``(None, reasons)``
    when none resolves."""
    skipped: list[str] = []
    for candidate in candidates:
        try:
            offer = resolve_offer(candidate, snapshot, contract_multiplier=contract_multiplier)
        except BrainError as exc:
            reason = exc.cause if exc.cause is not None else exc
            skipped.append(f"{describe_candidate(candidate)}: {reason}")
            continue
        return offer, tuple(skipped)
    return None, tuple(skipped)


# --- the cycle ------------------------------------------------------------------


class _Ledger:
    """What the cycle has produced so far — mutable, private to ``run_cycle``,
    so the outcome handlers can report a partial cycle."""

    def __init__(self) -> None:
        self.stage: ReasoningStage | None = None
        self.brief: ScanBrief | None = None
        self.candidates: tuple[ThesisCandidate, ...] = ()
        self.usage: list[StageUsage] = []
        self.proposal: TradeProposal | None = None
        self.proposal_id: str | None = None
        self.no_trade_reason: str | None = None
        self.halt_reason: str | None = None


def _record_stage(
    conn: sqlite3.Connection, ledger: _Ledger, cycle_id: str, usage: StageUsage, raw_output: str
) -> None:
    """Keep the stage's usage for the totals (spent whether or not the row
    lands) and write its reasoning row."""
    ledger.usage.append(usage)
    add_reasoning(conn, reasoning_row(cycle_id, usage, raw_output))


def _record_spend(
    conn: sqlite3.Connection,
    ledger: _Ledger,
    guard: BudgetGuard,
    cycle_id: str,
    config: AegisConfig,
    exc: BaseException,
) -> None:
    """Journal what ``exc`` cut short (``journal_unrecorded_spend``) and add
    the stage's unrecorded usage to the totals ``cycle_end`` reports."""
    stage = ledger.stage
    model = config.brain.stage(stage.value).model if stage is not None else ""
    pending = journal_unrecorded_spend(conn, guard, cycle_id, stage, model, ledger.usage, exc)
    if pending is not None:
        ledger.usage.append(pending)


def _no_trade(
    conn: sqlite3.Connection, ledger: _Ledger, cycle_id: str, what: str, reason: str
) -> CycleOutcome:
    """Log the ``no_trade`` event (``what`` says which stage declined) and settle the outcome."""
    ledger.no_trade_reason = reason
    log_event(
        conn,
        _cycle_event(
            "no_trade",
            f"no trade for cycle {cycle_id}: {what}",
            {"cycle_id": cycle_id, "reason": reason},
        ),
    )
    return "no_trade"


def _run_stages(
    conn: sqlite3.Connection,
    llm: LLMClient,
    config: AegisConfig,
    cycle_id: str,
    snapshot: MarketSnapshot | None,
    now: datetime | None,
    ledger: _Ledger,
) -> CycleOutcome:
    """Steps 3-8 of the cycle; ``BrainError``/``BudgetExceeded`` propagate to ``run_cycle``."""
    brain = config.brain
    if snapshot is None:
        snapshot = build_market_snapshot(config=config, now=now)
    log_event(conn, market_snapshot_event(cycle_id, snapshot))

    ledger.stage = ReasoningStage.SCAN
    scan = run_scan(llm, snapshot, brain, cycle_id=cycle_id)
    _record_stage(conn, ledger, cycle_id, scan.usage, scan.raw_output)
    ledger.brief = scan.brief

    ledger.stage = ReasoningStage.THESIS
    recent = get_recent_proposals(conn, brain.snapshot.recent_proposals_limit)
    thesis = run_thesis(
        llm,
        scan.brief,
        snapshot,
        brain,
        risk_limits=config.risk_limits,
        recent_proposals=recent,
        cycle_id=cycle_id,
    )
    _record_stage(conn, ledger, cycle_id, thesis.usage, thesis.raw_output)
    ledger.candidates = thesis.output.candidates
    if not thesis.output.candidates:
        reason = thesis.output.no_idea_reason or "(no reason given)"
        return _no_trade(
            conn, ledger, cycle_id, "the thesis stage found no compelling idea", reason
        )

    ledger.stage = ReasoningStage.PROPOSAL
    offer, skipped = first_offer(
        thesis.output.candidates, snapshot, contract_multiplier=config.pricing.contract_multiplier
    )
    if offer is None:
        return _no_trade(
            conn, ledger, cycle_id, "no candidate could be resolved to an offer", "; ".join(skipped)
        )
    proposal = run_proposal(llm, offer, snapshot, brain, cycle_id=cycle_id)
    _record_stage(conn, ledger, cycle_id, proposal.usage, proposal.raw_output)
    trade = proposal.output.proposal
    if proposal.output.outcome != "trade" or trade is None:
        reason = proposal.output.no_trade_reason or "(no reason given)"
        return _no_trade(conn, ledger, cycle_id, "the proposal stage declined to trade", reason)

    record, legs = proposal_records(cycle_id, proposal)
    # The proposal, its legs and the link of this cycle's reasoning rows land
    # together or not at all; once they have, cycle_end names the proposal
    # whatever happens next.
    insert_proposal(conn, record, legs, link_cycle_reasoning=True)
    ledger.proposal = trade
    ledger.proposal_id = record.id
    log_event(
        conn,
        _cycle_event(
            "proposal",
            f"cycle {cycle_id} proposed {trade.side.value} {trade.quantity} "
            f"{trade.instrument.value} {trade.symbol} ({record.id})",
            {"cycle_id": cycle_id, "proposal_id": record.id, "symbol": trade.symbol},
        ),
    )
    return "proposal"


def _log_cycle_end(
    conn: sqlite3.Connection,
    cycle_id: str,
    outcome: CycleOutcome,
    ledger: _Ledger,
    cost: float | None,
) -> None:
    tokens_in = sum(usage.tokens_in for usage in ledger.usage)
    tokens_out = sum(usage.tokens_out for usage in ledger.usage)
    log_event(
        conn,
        _cycle_event(
            "cycle_end",
            f"cycle {cycle_id} ended: {outcome} ({tokens_in} tokens in / {tokens_out} out)",
            {
                "cycle_id": cycle_id,
                "outcome": outcome,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "estimated_cost_usd": cost,
                "proposal_id": ledger.proposal_id,
            },
        ),
    )


def run_cycle(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    config: AegisConfig | None = None,
    snapshot: MarketSnapshot | None = None,
    now: datetime | None = None,
) -> CycleResult:
    """Run one brain cycle against the store: see the module docstring.

    ``config`` defaults to ``get_config()``. ``snapshot`` skips the live
    fetch — tests pass one in, and so does the ``once`` CLI, which builds it
    with its own snapshot builder so ``--symbols`` applies; otherwise
    ``build_market_snapshot`` runs for the config watchlist with ``now`` as
    its clock. ``started_at``/``finished_at`` are the wall clock, like the
    rows. ``BudgetExceeded`` is returned as the ``halted`` outcome; any
    other exception — a ``BrainError`` or not, a ``KeyboardInterrupt``
    included — is re-raised after the ``brain_error`` and ``cycle_end``
    events, logged as far as the store allows. Either way the spend the
    stage in progress received is journaled first and counted in
    ``cycle_end`` and ``CycleResult.usage``.
    """
    config = config if config is not None else get_config()
    cycle_id = new_id()
    started_at = utcnow()
    symbols = (
        [s.symbol for s in snapshot.symbols] if snapshot is not None else list(config.watchlist)
    )
    log_event(
        conn,
        _cycle_event(
            "cycle_start",
            f"cycle {cycle_id} started for {', '.join(symbols)}",
            {"cycle_id": cycle_id, "symbols": symbols},
        ),
    )

    ledger = _Ledger()
    guard = BudgetGuard(conn, config.brain, cycle_id)
    guarded = GuardedLLM(llm, guard)
    prices = config.brain.prices_per_mtok
    try:
        outcome = _run_stages(conn, guarded, config, cycle_id, snapshot, now, ledger)
    except BudgetExceeded as exc:
        _record_spend(conn, ledger, guard, cycle_id, config, exc)
        outcome = "halted"  # the guard has logged budget_halt; the rows so far stay
        ledger.halt_reason = str(exc)
    except BaseException as exc:
        # A BrainError, or not one at all: the store failing mid-cycle, a
        # client that broke its contract, a Ctrl-C (or SystemExit) while a
        # stage retries — its billed attempts count all the same. The spend
        # is journaled first; the audit trail then gets both events, each on
        # its own so a broken store cannot hide what actually went wrong.
        _record_spend(conn, ledger, guard, cycle_id, config, exc)
        try:
            log_event(conn, brain_error_event(cycle_id, ledger.stage, exc))
        except StoreError:
            pass
        try:
            _log_cycle_end(conn, cycle_id, "failed", ledger, estimate_cost(ledger.usage, prices))
        except StoreError:
            pass
        raise
    finished_at = utcnow()
    cost = estimate_cost(ledger.usage, prices)
    _log_cycle_end(conn, cycle_id, outcome, ledger, cost)
    return CycleResult(
        cycle_id=cycle_id,
        started_at=started_at,
        finished_at=finished_at,
        outcome=outcome,
        proposal_id=ledger.proposal_id,
        proposal=ledger.proposal,
        no_trade_reason=ledger.no_trade_reason,
        halt_reason=ledger.halt_reason,
        brief=ledger.brief,
        candidates=ledger.candidates,
        usage=tuple(ledger.usage),
        estimated_cost_usd=cost,
    )
