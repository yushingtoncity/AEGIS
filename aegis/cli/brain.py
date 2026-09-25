"""Run the agent brain: one full cycle, one stage for debugging, or a day's token usage.

Run:  python -m aegis.cli.brain once [--db PATH] [--symbols SPY,QQQ]
      python -m aegis.cli.brain stage {scan,thesis,proposal} [--db PATH] [--cycle CYCLE_ID]
      python -m aegis.cli.brain usage [--db PATH] [--day YYYY-MM-DD]

``once`` runs a real cycle — live Alpaca data through the snapshot builder
and the real Anthropic API through ``AnthropicLLM`` — and prints the brief,
the candidates, the proposal (or ``NO_TRADE`` / ``HALTED`` with the reason),
the tokens per stage and an estimated cost. Exit 0 for a proposal or a
no-trade, 2 when the budget guard halted the cycle, 1 on failure — one line
on stderr, never a traceback. A bad command line (an unknown flag, an empty
``--symbols``, a malformed ``--day``) is a failure too: one line and exit 1,
never argparse's exit 2, which here means only a budget halt. Everything
the model wrote is printed one field per line with control characters
removed, so its text can neither restyle the terminal nor forge a line of
the report; a character stdout cannot encode prints as an escape. ``stage``
runs a single stage and prints its raw JSON output and usage: with
``--cycle`` the prior stages' outputs come from that cycle's reasoning rows
and its ``market_snapshot`` event, without it the prior stages run first;
either way the rows are written under a fresh cycle id that logs its own
``market_snapshot`` event (so its rows carry their input and can feed a
later ``--cycle``), and a stage cut short by a halt, an error or a Ctrl-C
still journals the spend it received and, unless the guard halted it,
logs a ``brain_error`` event, as the cycle does — so the budget accounting
and the audit trail stay honest. A prior row that does not validate (that
cycle's stage failed) is a ``BrainError`` naming the cycle. ``usage``
reads the reasoning table: tokens in/out and reasoning rows (one per stage,
however many API calls its retries took) for a UTC day, by model, the
estimated spend and the daily budget used. The rows name the model that
answered, so each is priced through ``model_price`` — a dated snapshot of
a configured alias (``claude-opus-5-5-20260301``) at the alias's
placeholder, as ``once`` priced it; a model with no placeholder either
way (``FakeLLM``'s ``fake-model``) makes the spend n/a. It never creates
or migrates a database. Every cost printed is an ESTIMATE from the
``brain.prices_per_mtok`` placeholders in config.yaml, never a bill.

``main(argv, *, llm=None, snapshot_builder=None)``: tests inject a
``FakeLLM`` and a canned snapshot; the defaults are
``AnthropicLLM.from_config()`` and ``build_market_snapshot``. ``--db``
overrides ``store.db_path`` from config.yaml and is taken relative to the
current directory; ``--db :memory:`` is sqlite's in-memory sentinel.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import unicodedata
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import NoReturn, TypeVar

from aegis.brain.cycle import (
    MARKET_SNAPSHOT_EVENT,
    brain_error_event,
    describe_candidate,
    first_offer,
    journal_unrecorded_spend,
    market_snapshot_event,
    reasoning_row,
    run_cycle,
    snapshot_from_event,
)
from aegis.brain.errors import BrainError, BudgetExceeded
from aegis.brain.llm import (
    AnthropicLLM,
    BudgetGuard,
    GuardedLLM,
    LLMClient,
    estimate_cost,
    model_price,
)
from aegis.brain.models import (
    CycleResult,
    InstrumentOffer,
    MarketSnapshot,
    ProposalResult,
    ScanBrief,
    ScanResult,
    StageUsage,
    ThesisCandidate,
    ThesisOutput,
    ThesisResult,
    TradeProposal,
)
from aegis.brain.proposal import run_proposal
from aegis.brain.scan import merge_staleness_warnings, run_scan
from aegis.brain.schemas import OutputInvalid, validate_output
from aegis.brain.snapshot import build_market_snapshot
from aegis.brain.thesis import run_thesis, validate_candidates
from aegis.config import AegisConfig, ConfigError, ModelPrice, get_config
from aegis.data.models import utcnow
from aegis.store.db import MEMORY_DB, connect, open_store, resolve_db_path, status
from aegis.store.errors import StoreError
from aegis.store.models import ReasoningStage, TokenUsage, new_id
from aegis.store.repo import (
    add_reasoning,
    get_cycle_reasoning,
    get_recent_events,
    get_recent_proposals,
    get_token_usage,
    get_token_usage_by_model,
    log_event,
)

SnapshotBuilder = Callable[..., MarketSnapshot]
"""``build_market_snapshot``'s shape: ``(symbols, *, config=...) -> MarketSnapshot``."""

_OutputT = TypeVar("_OutputT", ScanBrief, ThesisOutput)

INDENT = "  "
ESTIMATE_NOTE = "ESTIMATE from config brain.prices_per_mtok placeholders — not a bill"

# How far back ``stage --cycle`` looks for a cycle's market_snapshot event
# (a cycle logs about five events, so this is weeks of cycles).
_EVENT_SCAN_LIMIT = 2000


def _db_path(arg: str | None) -> Path:
    """An explicit --db resolves against the CWD; otherwise config.yaml decides."""
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)  # resolving it would name a file ":memory:" in the CWD
    return Path(arg).expanduser().resolve()


# --- formatting -----------------------------------------------------------------


def _ts(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown"


def _price(value: float) -> str:
    """A price with every digit that matters and at least two decimals: 4.55, 4.50, 561.275."""
    whole, _, fraction = f"{value:.4f}".rstrip("0").partition(".")
    return f"{whole}.{fraction.ljust(2, '0')}"


def _strike(value: float) -> str:
    """A strike with at least one decimal: 640.0, 642.5 (the trace CLI's shape)."""
    whole, _, fraction = f"{value:.4f}".rstrip("0").partition(".")
    return f"{whole}.{fraction.ljust(1, '0')}"


def _clean(text: object) -> str:
    """Text for the terminal, on one line: control and format characters
    become spaces (an ESC from the model's output could restyle or hide the
    rest of the report), and whitespace runs — newlines included — collapse
    to one space, so no model-authored field can print a line of its own
    and pose as part of the report."""
    kept = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in str(text)
    )
    return " ".join(kept.split())


def _one_line(exc: BaseException) -> str:
    """An error message on one line: a schema failure lists one problem per
    line, and the failure report is a single line on stderr."""
    return _clean(exc)


def _cost(value: float | None) -> str:
    return f"${value:.4f}" if value is not None else "n/a"


def _cost_line(cost: float | None, usage: Sequence[StageUsage], config: AegisConfig) -> str:
    if cost is None:
        prices = config.brain.prices_per_mtok
        unpriced = sorted({u.model for u in usage if model_price(u.model, prices) is None})
        return f"estimated cost: n/a (no price in brain.prices_per_mtok for {', '.join(unpriced)})"
    return f"estimated cost: {_cost(cost)} ({ESTIMATE_NOTE})"


def _market_line(snapshot: MarketSnapshot) -> str:
    age = snapshot.data_age
    if age.market_open is True:
        return f"OPEN (next close {_ts(age.next_close)})"
    if age.market_open is False:
        return f"CLOSED (next open {_ts(age.next_open)}) — all data is the last available, STALE"
    return "UNKNOWN (the market clock could not be fetched) — treating all data as STALE"


def _bullets(items: Sequence[str], depth: int) -> None:
    pad = INDENT * depth
    if not items:
        print(f"{pad}(none)")
    for item in items:
        print(f"{pad}- {_clean(item)}")


def _block(label: str, text: str, depth: int) -> None:
    """``label:`` then the text in full on one line, one indent level deeper."""
    print(f"{INDENT * depth}{label}:")
    print(f"{INDENT * (depth + 1)}{_clean(text)}")


# --- once -----------------------------------------------------------------------


def _print_brief(brief: ScanBrief | None, note: str) -> None:
    print("brief")
    if brief is None:
        print(f"{INDENT}(none — {note})")
        return
    # every field below is the scan model's own text: printed through _clean
    for item in brief.symbols:
        print(f"{INDENT}{_clean(item.symbol)}{' [STALE]' if item.stale_data else ''}")
        print(f"{INDENT * 2}{_clean(item.summary)}")
        print(f"{INDENT * 2}IV: {_clean(item.iv_observation or '(no observation)')}")
        print(f"{INDENT * 2}skew: {_clean(item.skew_observation or '(no observation)')}")
        print(f"{INDENT * 2}catalysts:")
        _bullets(item.catalysts, 3)
    print(f"{INDENT}notable observations")
    _bullets(brief.notable_observations, 2)
    print(f"{INDENT}news catalysts")
    _bullets(brief.news_catalysts, 2)
    print(f"{INDENT}staleness warnings ({len(brief.staleness_warnings)})")
    _bullets(brief.staleness_warnings, 2)


def _print_candidates(candidates: Sequence[ThesisCandidate]) -> None:
    print(f"candidates ({len(candidates)})")
    if not candidates:
        print(f"{INDENT}(none)")
    for index, candidate in enumerate(candidates):
        print(
            f"{INDENT}[{index}] {candidate.direction.value} {_clean(describe_candidate(candidate))}"
            f"   confidence {candidate.confidence:.2f}"
        )
        pad = INDENT * 3
        print(f"{pad}rationale: {_clean(candidate.rationale)}")
        print(f"{pad}key risk: {_clean(candidate.key_risk)}")
        print(f"{pad}invalidation: {_clean(candidate.invalidation)}")


def _print_legs(proposal: TradeProposal) -> None:
    print(f"{INDENT}legs ({len(proposal.legs)})")
    if not proposal.legs:
        print(f"{INDENT * 2}(none)")
    for index, leg in enumerate(proposal.legs):
        print(
            f"{INDENT * 2}[{index}] {leg.side.value} {leg.quantity} {leg.option_type.value}"
            f" {_strike(leg.strike)} exp {leg.expiration.isoformat()}  {_clean(leg.symbol)}"
        )


def _print_proposal(result: CycleResult) -> None:
    proposal = result.proposal
    if result.outcome == "halted":
        print("proposal")
        print(f"{INDENT}HALTED: {_clean(result.halt_reason)}")
        return
    if proposal is None or result.proposal_id is None:
        print("proposal")
        print(f"{INDENT}NO_TRADE: {_clean(result.no_trade_reason)}")
        return
    print(f"proposal {result.proposal_id}")
    limit = f" @ {_price(proposal.limit_price)}" if proposal.limit_price is not None else ""
    terms = f"{proposal.side.value} {proposal.quantity} {proposal.order_type.value}{limit}"
    print(
        f"{INDENT}{_clean(proposal.symbol)} ({proposal.instrument.value})   {terms}"
        f"   confidence {proposal.confidence:.2f}"
    )
    _print_legs(proposal)
    _block("thesis", proposal.thesis, depth=1)
    _block("invalidation", proposal.invalidation, depth=1)


def _print_tokens(usage: Sequence[StageUsage], cost: float | None, config: AegisConfig) -> None:
    print("tokens")
    if not usage:
        print(f"{INDENT}(no calls made)")
    for stage in usage:
        print(
            f"{INDENT}{stage.stage.value:<9} in {stage.tokens_in:,} / out {stage.tokens_out:,}"
            f"   attempts {stage.attempts}   latency {stage.latency_ms:.0f} ms   {stage.model}"
        )
    total_in = sum(u.tokens_in for u in usage)
    total_out = sum(u.tokens_out for u in usage)
    print(f"{INDENT}{'total':<9} in {total_in:,} / out {total_out:,} = {total_in + total_out:,}")
    print(_cost_line(cost, usage, config))


def _print_cycle(result: CycleResult, snapshot: MarketSnapshot, config: AegisConfig) -> None:
    print(f"cycle {result.cycle_id}")
    print(f"{INDENT}started   {_ts(result.started_at)}")
    print(f"{INDENT}finished  {_ts(result.finished_at)}")
    print(f"{INDENT}market    {_market_line(snapshot)}")
    print(f"{INDENT}symbols   {', '.join(s.symbol for s in snapshot.symbols)}")
    print(f"{INDENT}outcome   {result.outcome}")
    print()
    _print_brief(result.brief, "the cycle halted before the scan stage")
    print()
    _print_candidates(result.candidates)
    print()
    _print_proposal(result)
    print()
    _print_tokens(result.usage, result.estimated_cost_usd, config)
    if result.proposal_id is not None:
        print()
        print(f"next: python -m aegis.cli.trace {result.proposal_id}")


def _once(
    db_path: Path,
    config: AegisConfig,
    llm: LLMClient,
    builder: SnapshotBuilder,
    symbols: list[str] | None,
) -> int:
    conn = open_store(db_path)
    try:
        snapshot = builder(symbols, config=config)
        result = run_cycle(conn, llm, config=config, snapshot=snapshot)
    finally:
        conn.close()
    _print_cycle(result, snapshot, config)
    return 2 if result.outcome == "halted" else 0


# --- stage ----------------------------------------------------------------------


def _cycle_snapshot(conn: sqlite3.Connection, cycle_id: str) -> MarketSnapshot:
    """The snapshot logged for ``cycle_id`` (its ``market_snapshot`` event)."""
    for event in get_recent_events(conn, limit=_EVENT_SCAN_LIMIT):
        payload = event.payload or {}
        if event.kind == MARKET_SNAPSHOT_EVENT and payload.get("cycle_id") == cycle_id:
            try:
                return snapshot_from_event(event)
            except ValueError as exc:
                raise BrainError("load market snapshot", cycle_id, exc) from exc
    raise BrainError(
        "load market snapshot",
        cycle_id,
        ValueError(
            f"no market_snapshot event for this cycle among the last {_EVENT_SCAN_LIMIT} events"
        ),
    )


def _prior_output(conn: sqlite3.Connection, cycle_id: str, stage: ReasoningStage) -> str:
    """The raw text of ``cycle_id``'s last ``stage`` reasoning row."""
    rows = [row for row in get_cycle_reasoning(conn, cycle_id) if row.stage is stage]
    if not rows:
        raise BrainError(
            "load prior stage output",
            cycle_id,
            ValueError(f"the cycle has no {stage.value} reasoning row"),
        )
    return rows[-1].content


class _StageRun:
    """One ``stage`` invocation: a fresh cycle id, the guarded client, the
    snapshot (built live, or the prior cycle's) and the rows it writes.

    Each ``*_input`` method hands the next stage what it needs — from the
    prior cycle's reasoning rows with ``--cycle``, else by running the
    earlier stage here — and every stage that runs writes its reasoning
    row under the fresh cycle id.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: AegisConfig,
        llm: LLMClient,
        builder: SnapshotBuilder,
        prior_cycle: str | None,
    ) -> None:
        self.conn = conn
        self.config = config
        self.prior_cycle = prior_cycle
        self.cycle_id = new_id()
        self.guard = BudgetGuard(conn, config.brain, self.cycle_id)
        self.llm = GuardedLLM(llm, self.guard)
        self.usage: list[StageUsage] = []
        self.stage: ReasoningStage | None = None  # the stage in progress, for record_spend
        if prior_cycle is None:
            self.snapshot = builder(None, config=config)
            self.source = "prior stages run live"
        else:
            self.snapshot = _cycle_snapshot(conn, prior_cycle)
            self.source = f"prior outputs from cycle {prior_cycle}"
        # Either way the fresh cycle carries its input as its own
        # market_snapshot event: the audit copy of what its stage was shown,
        # and what lets a later ``stage --cycle`` build on this cycle's rows.
        log_event(
            conn, market_snapshot_event(self.cycle_id, self.snapshot, source_cycle=prior_cycle)
        )

    def _record(self, usage: StageUsage, raw_output: str) -> None:
        self.usage.append(usage)  # spent whether or not the row lands, as in the cycle
        add_reasoning(self.conn, reasoning_row(self.cycle_id, usage, raw_output))

    def record_spend(self, exc: BaseException) -> None:
        """Journal what ``exc`` cut short, as the cycle does (``journal_unrecorded_spend``)."""
        stage = self.stage
        model = self.config.brain.stage(stage.value).model if stage is not None else ""
        pending = journal_unrecorded_spend(
            self.conn, self.guard, self.cycle_id, stage, model, self.usage, exc
        )
        if pending is not None:
            self.usage.append(pending)

    def log_error(self, exc: BaseException) -> None:
        """Log the ``brain_error`` event for ``exc``, as the cycle does — best
        effort: a store that cannot log must not hide the error itself."""
        try:
            log_event(self.conn, brain_error_event(self.cycle_id, self.stage, exc))
        except StoreError:
            pass

    def scan(self) -> ScanResult:
        self.stage = ReasoningStage.SCAN
        result = run_scan(self.llm, self.snapshot, self.config.brain, cycle_id=self.cycle_id)
        self._record(result.usage, result.raw_output)
        return result

    def thesis(self, brief: ScanBrief) -> ThesisResult:
        self.stage = ReasoningStage.THESIS
        brain = self.config.brain
        result = run_thesis(
            self.llm,
            brief,
            self.snapshot,
            brain,
            risk_limits=self.config.risk_limits,
            recent_proposals=get_recent_proposals(self.conn, brain.snapshot.recent_proposals_limit),
            cycle_id=self.cycle_id,
        )
        self._record(result.usage, result.raw_output)
        return result

    def proposal(self, offer: InstrumentOffer) -> ProposalResult:
        self.stage = ReasoningStage.PROPOSAL
        result = run_proposal(
            self.llm, offer, self.snapshot, self.config.brain, cycle_id=self.cycle_id
        )
        self._record(result.usage, result.raw_output)
        return result

    def _prior(
        self,
        stage: ReasoningStage,
        needed_by: ReasoningStage,
        model: type[_OutputT],
        post_validate: Callable[[_OutputT], None] | None = None,
    ) -> _OutputT:
        """The prior cycle's ``stage`` output, validated into ``model`` (and by
        the stage's own ``post_validate``, as the loop did). A row that does
        not validate — that cycle's stage failed and its row holds the last
        invalid text, or a ``(no usable output …)`` line — is a ``BrainError``
        naming the cycle, not an unexpected error."""
        assert self.prior_cycle is not None  # only called with --cycle
        text = _prior_output(self.conn, self.prior_cycle, stage)
        try:
            output = validate_output(model, text)
            if post_validate is not None:
                post_validate(output)
            return output
        except OutputInvalid as exc:
            raise BrainError(
                f"stage {needed_by.value}: cycle {self.prior_cycle} has no valid "
                f"{stage.value} output",
                cause=exc,
            ) from exc

    def thesis_input(self) -> ScanBrief:
        if self.prior_cycle is None:
            return self.scan().brief
        # the row holds the model's text as validated, before the warnings merge
        brief = self._prior(ReasoningStage.SCAN, ReasoningStage.THESIS, ScanBrief)
        return merge_staleness_warnings(brief, self.snapshot)

    def proposal_input(self) -> ThesisOutput:
        if self.prior_cycle is None:
            return self.thesis(self.thesis_input()).output
        return self._prior(
            ReasoningStage.THESIS,
            ReasoningStage.PROPOSAL,
            ThesisOutput,
            lambda output: validate_candidates(output, self.snapshot),
        )


def _stage(
    db_path: Path,
    config: AegisConfig,
    llm: LLMClient,
    builder: SnapshotBuilder,
    stage: ReasoningStage,
    prior_cycle: str | None,
) -> int:
    conn = open_store(db_path)
    try:
        run = _StageRun(conn, config, llm, builder, prior_cycle)
        print(f"cycle {run.cycle_id}   stage {stage.value}   {run.source}")
        print(
            f"snapshot: {', '.join(s.symbol for s in run.snapshot.symbols)}, taken "
            f"{_ts(run.snapshot.taken_at)}, market {_market_line(run.snapshot)}"
        )
        result: ScanResult | ThesisResult | ProposalResult | None = None
        code = 0
        try:
            if stage is ReasoningStage.SCAN:
                result = run.scan()
            elif stage is ReasoningStage.THESIS:
                result = run.thesis(run.thesis_input())
            else:
                output = run.proposal_input()
                if not output.candidates:
                    print(
                        "NO_TRADE: the thesis stage found no compelling idea: "
                        f"{_clean(output.no_idea_reason)}"
                    )
                else:
                    offer, skipped = first_offer(
                        output.candidates,
                        run.snapshot,
                        contract_multiplier=config.pricing.contract_multiplier,
                    )
                    for reason in skipped:
                        print(f"skipped: {_clean(reason)}")
                    if offer is None:
                        print("NO_TRADE: no candidate could be resolved to an offer")
                    else:
                        print(f"offer: {_clean(describe_candidate(offer.candidate))}")
                        result = run.proposal(offer)
        except BudgetExceeded as exc:
            run.record_spend(exc)  # the guard has logged budget_halt
            print(f"HALTED: {_clean(exc)}")
            code = 2
        except BaseException as exc:  # a Ctrl-C mid-stage too: what was billed is journaled
            run.record_spend(exc)
            run.log_error(exc)
            raise
    finally:
        conn.close()
    if result is not None:
        print(f"output ({result.prompt_version}):")
        print(result.raw_output.rstrip("\n"))
    print()
    _print_tokens(run.usage, estimate_cost(run.usage, config.brain.prices_per_mtok), config)
    return code


# --- usage ----------------------------------------------------------------------


def _model_cost(usage: TokenUsage, price: ModelPrice | None) -> float | None:
    """One model's spend at its list price (``estimate_cost``'s arithmetic); None when unpriced."""
    if price is None:
        return None
    return (usage.tokens_in * price.input + usage.tokens_out * price.output) / 1_000_000


def _usage(db_path: Path, config: AegisConfig, day: date | None) -> int:
    if str(db_path) == MEMORY_DB:
        print(
            "brain usage failed: --db :memory: is an in-memory database, not a file:"
            " there is no usage in it to report",
            file=sys.stderr,
        )
        return 1
    if not db_path.exists():
        print(
            f"brain usage failed: database {db_path} does not exist"
            " (run `python -m aegis.cli.db init` to create it)",
            file=sys.stderr,
        )
        return 1
    day = day if day is not None else utcnow().date()
    conn = connect(db_path)  # never open_store: a report applies no migration
    try:
        pending = status(conn, db_path).pending_migrations
        if pending:
            print(
                f"brain usage failed: database {db_path} has pending migrations"
                f" ({', '.join(pending)}): run `python -m aegis.cli.db init` to apply them",
                file=sys.stderr,
            )
            return 1
        total = get_token_usage(conn, day)
        by_model = get_token_usage_by_model(conn, day)
    finally:
        conn.close()

    prices = config.brain.prices_per_mtok
    # keyed by the answering model: model_price maps a dated snapshot id back to its alias
    costs = {
        name: _model_cost(usage, model_price(name, prices)) for name, usage in by_model.items()
    }
    unpriced = [name for name, cost in costs.items() if cost is None]
    # as estimate_cost: a partial sum would mislead, so any unpriced model makes it n/a
    spend = None if unpriced else sum(cost for cost in costs.values() if cost is not None)
    budget = config.brain.daily_token_budget

    print(f"token usage for {day.isoformat()} (UTC)")
    # ``calls`` counts reasoning rows — one per stage, whatever its retries
    # cost — so it is labelled as rows, never as API calls
    print(
        f"{INDENT}tokens    in {total.tokens_in:,} / out {total.tokens_out:,}"
        f" = {total.total:,} over {total.calls} reasoning row(s)"
    )
    print(f"{INDENT}by model")
    if not by_model:
        print(f"{INDENT * 2}(none)")
    for name, usage in by_model.items():
        print(
            f"{INDENT * 2}{name:<30} in {usage.tokens_in:,} / out {usage.tokens_out:,}"
            f"   rows {usage.calls}   est. {_cost(costs[name])}"
        )
    if spend is None:
        missing = ", ".join(unpriced)
        print(f"{INDENT}estimated spend: n/a (no price in brain.prices_per_mtok for {missing})")
    else:
        print(f"{INDENT}estimated spend: {_cost(spend)} ({ESTIMATE_NOTE})")
    print(
        f"{INDENT}daily budget: {total.total:,} / {budget:,} tokens"
        f" ({total.total / budget * 100:.1f}% used)"
    )
    return 0


# --- main -----------------------------------------------------------------------


class _UsageError(Exception):
    """A bad command line, raised in place of argparse's exit so ``main`` can
    report it as one line and exit 1 — exit 2 is reserved for a budget halt."""


class _Parser(argparse.ArgumentParser):
    """``ArgumentParser`` whose errors raise ``_UsageError`` instead of
    printing the usage block and exiting 2 (the subcommand parsers are made
    from the same class, so theirs do too). ``--help`` still exits 0."""

    def error(self, message: str) -> NoReturn:
        raise _UsageError(f"{self.prog}: error: {message}")


def _symbols(arg: str | None, parser: argparse.ArgumentParser) -> list[str] | None:
    if arg is None:
        return None
    symbols = [s.strip().upper() for s in arg.split(",") if s.strip()]
    if not symbols:
        parser.error(f"--symbols needs at least one ticker, got {arg!r}")
    return symbols


def main(
    argv: list[str] | None = None,
    *,
    llm: LLMClient | None = None,
    snapshot_builder: SnapshotBuilder | None = None,
) -> int:
    # a report must not abort on the content it exists to show (the model's
    # own text, the em dash in the ESTIMATE note): a character the stdout
    # encoding lacks prints as a \uXXXX escape, as in the trace CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = _Parser(
        prog="python -m aegis.cli.brain",
        description="Run the AEGIS agent brain: a full cycle, one stage, or a day's token usage.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    once = commands.add_parser(
        "once", parents=[common], help="run one scan -> thesis -> proposal cycle and print it"
    )
    once.add_argument("--symbols", help="comma-separated tickers (default: the config watchlist)")
    stage = commands.add_parser(
        "stage", parents=[common], help="run a single stage for debugging and print its raw output"
    )
    stage.add_argument("stage", choices=[s.value for s in ReasoningStage])
    stage.add_argument(
        "--cycle",
        metavar="CYCLE_ID",
        help="take the prior stages' outputs and the snapshot from this cycle, not a new run",
    )
    usage = commands.add_parser(
        "usage", parents=[common], help="print a UTC day's tokens and estimated spend"
    )
    usage.add_argument("--day", help="UTC day YYYY-MM-DD (default: today)")
    try:
        args = parser.parse_args(argv)
        day: date | None = None
        if args.command == "usage" and args.day:
            try:
                day = date.fromisoformat(args.day)
            except ValueError:
                usage.error(f"--day must be YYYY-MM-DD, got {args.day!r}")
        symbols = _symbols(args.symbols, once) if args.command == "once" else None
    except _UsageError as exc:
        print(f"{exc} (see --help)", file=sys.stderr)
        return 1

    try:
        db_path = _db_path(args.db)
        config = get_config()
        if args.command == "usage":
            return _usage(db_path, config, day)
        client = llm if llm is not None else AnthropicLLM.from_config(config.brain)
        builder = snapshot_builder if snapshot_builder is not None else build_market_snapshot
        if args.command == "once":
            return _once(db_path, config, client, builder, symbols)
        return _stage(db_path, config, client, builder, ReasoningStage(args.stage), args.cycle)
    except (ConfigError, StoreError, BrainError) as exc:
        print(f"brain {args.command} failed: {_one_line(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(
            f"brain {args.command} failed (unexpected): {type(exc).__name__}: {_one_line(exc)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
