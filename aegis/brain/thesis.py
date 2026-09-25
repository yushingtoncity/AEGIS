"""Stage 2 — thesis: the brief in, zero or more ``ThesisCandidate`` out.

The model is shown the scan brief, the open positions, a read-only summary
of the risk limits, the recent proposals (so it does not repeat itself)
and, per symbol, the one expiration and the strikes the snapshot chain
actually holds (``build_thesis_prompt``). It answers with a
``ThesisOutput``: candidates best first, or an empty list with a
``no_idea_reason`` — the prompt says that is a respectable answer.

The pydantic model already checks each candidate's shape (a structure's
strike count, strikes low to high, an invalidation condition). What only
this stage can check is consistency with the snapshot, and
``validate_candidates`` does that as the loop's ``post_validate``: every
symbol must be in the snapshot, an option candidate must name that
symbol's chain expiration and, at each strike, the option type its
structure's leg needs there (a call strike is not a put strike), and there
are at most ``MAX_THESIS_CANDIDATES``. A candidate's direction must also
agree with what it proposes — a bearish call debit spread, or a neutral
equity idea, contradicts itself — since the stored proposal keeps no
direction to show the contradiction later. The rejection message lists
what *is* available so the model can correct itself instead of guessing
again.
"""

from __future__ import annotations

from collections.abc import Sequence

from aegis.brain.llm import LLMClient
from aegis.brain.models import (
    Direction,
    MarketSnapshot,
    OptionStructure,
    ScanBrief,
    ThesisCandidate,
    ThesisOutput,
    ThesisResult,
)
from aegis.brain.prompts import (
    MAX_THESIS_CANDIDATES,
    build_thesis_prompt,
    chain_strikes,
    neutralise_untrusted,
    prompt_version,
    system_prompt,
)
from aegis.brain.proposal import STRUCTURE_LEG_PLAN
from aegis.brain.schemas import OutputInvalid
from aegis.brain.stage import run_json_stage
from aegis.config import BrainConfig, RiskLimits
from aegis.store.models import Instrument, Proposal, ReasoningStage

STRIKE_TOLERANCE = 1e-6
"""Two strikes closer than this are the same strike (float noise from JSON)."""

STRUCTURE_DIRECTIONS: dict[OptionStructure, frozenset[Direction]] = {
    OptionStructure.LONG_CALL: frozenset({Direction.BULLISH}),
    OptionStructure.LONG_PUT: frozenset({Direction.BEARISH}),
    OptionStructure.CALL_DEBIT_SPREAD: frozenset({Direction.BULLISH}),
    OptionStructure.PUT_DEBIT_SPREAD: frozenset({Direction.BEARISH}),
    OptionStructure.CALL_CREDIT_SPREAD: frozenset({Direction.BEARISH, Direction.NEUTRAL}),
    OptionStructure.PUT_CREDIT_SPREAD: frozenset({Direction.BULLISH, Direction.NEUTRAL}),
    OptionStructure.IRON_CONDOR: frozenset({Direction.NEUTRAL}),
}
"""The views each structure expresses, as the thesis prompt's ``STRUCTURE_LEGS``
states them (a credit spread is "bearish to neutral" / "bullish to neutral")."""

EQUITY_DIRECTIONS = frozenset({Direction.BULLISH, Direction.BEARISH})
"""An equity is bought on a bullish view and sold on a bearish one; a neutral
view has no equity expression."""


def _plain(value: float) -> str:
    """A strike without trailing zeros: 640, 642.5."""
    return f"{value:g}"


def _listed(strikes: list[float]) -> str:
    return ", ".join(_plain(k) for k in strikes) or "(none)"


def _fits(strike: float, strikes: list[float]) -> bool:
    return any(abs(strike - k) <= STRIKE_TOLERANCE for k in strikes)


def _direction_problem(loc: str, candidate: ThesisCandidate) -> str | None:
    """Why the candidate's direction contradicts its instrument, or None."""
    direction = candidate.direction
    if candidate.instrument is not Instrument.OPTION:
        if direction in EQUITY_DIRECTIONS:
            return None
        return (
            f"{loc}.direction: an equity candidate is bullish (buy shares) or bearish (sell "
            f"shares), got {direction.value} — a neutral view needs an option structure "
            "(iron_condor, or a credit spread)"
        )
    if candidate.structure is None or direction in STRUCTURE_DIRECTIONS[candidate.structure]:
        return None
    fitting = [s.value for s in OptionStructure if direction in STRUCTURE_DIRECTIONS[s]]
    views = " or ".join(sorted(d.value for d in STRUCTURE_DIRECTIONS[candidate.structure]))
    return (
        f"{loc}.direction: {candidate.structure.value} expresses a {views} view, not "
        f"{direction.value}; structures for a {direction.value} view: {', '.join(fitting)}"
    )


def validate_candidates(output: ThesisOutput, snapshot: MarketSnapshot) -> None:
    """Reject candidates the snapshot cannot back; ``OutputInvalid`` lists
    every problem as ``candidates.<i>.<field>: <what is available>``.

    Each option strike is checked against the option type its leg needs
    (``proposal.STRUCTURE_LEG_PLAN``): a put structure at a strike the chain
    lists only as a call would otherwise pass here and then be skipped as
    unresolvable, and the model would never learn which contracts exist. A
    candidate's direction must agree with its structure's view
    (``STRUCTURE_DIRECTIONS``), and an equity candidate is bullish or
    bearish. The candidate's symbol is model-authored, so it is neutralised
    wherever the message quotes it."""
    problems: list[str] = []
    if len(output.candidates) > MAX_THESIS_CANDIDATES:
        problems.append(
            f"candidates: at most {MAX_THESIS_CANDIDATES} candidates, got "
            f"{len(output.candidates)} — keep the best {MAX_THESIS_CANDIDATES}"
        )
    symbols = {s.symbol: s for s in snapshot.symbols}
    for index, candidate in enumerate(output.candidates):
        loc = f"candidates.{index}"
        name = neutralise_untrusted(candidate.symbol)
        symbol = symbols.get(candidate.symbol)
        if symbol is None:
            problems.append(
                f"{loc}.symbol: {name} is not in the snapshot; "
                f"watchlist symbols: {', '.join(symbols)}"
            )
            continue
        direction = _direction_problem(loc, candidate)
        if direction is not None:
            problems.append(direction)
        if candidate.instrument is not Instrument.OPTION:
            continue
        chain = symbol.chain
        if chain is None or not chain.contracts:
            problems.append(
                f"{loc}.instrument: {name} has no option chain in this snapshot — "
                "equity candidates only"
            )
            continue
        strikes = chain_strikes(chain)
        expiry = chain.expiration.isoformat()
        if candidate.expiration != chain.expiration:
            problems.append(
                f"{loc}.expiration: {candidate.expiration} is not available for "
                f"{name}; the only expiration in the snapshot is {expiry}"
            )
        assert candidate.structure is not None  # the model validator guarantees it for options
        for strike_index, option_type, _side in STRUCTURE_LEG_PLAN[candidate.structure]:
            strike = candidate.strikes[strike_index]
            if not _fits(strike, strikes):
                problems.append(
                    f"{loc}.strikes: strike {_plain(strike)} is not in the {name} "
                    f"chain for {expiry}; available strikes: {_listed(strikes)}"
                )
                continue
            typed = chain_strikes(chain, option_type)
            if not _fits(strike, typed):
                problems.append(
                    f"{loc}.strikes: {candidate.structure.value} needs a {option_type.value} "
                    f"at strike {_plain(strike)}, and the {name} chain for {expiry} has no "
                    f"{option_type.value} there; {option_type.value} strikes: {_listed(typed)}"
                )
    if problems:
        raise OutputInvalid("output did not match the market snapshot:\n" + "\n".join(problems))


def run_thesis(
    llm: LLMClient,
    brief: ScanBrief,
    snapshot: MarketSnapshot,
    config: BrainConfig,
    *,
    risk_limits: RiskLimits,
    recent_proposals: Sequence[Proposal],
    cycle_id: str | None = None,
) -> ThesisResult:
    """Run the thesis stage with ``config.stages.thesis`` and ``config.max_retries``.

    Positions come from ``snapshot.account`` (none when the account fetch
    failed); ``risk_limits`` and ``recent_proposals`` are rendered read-only.
    """
    positions = snapshot.account.positions if snapshot.account is not None else ()
    output, usage, raw = run_json_stage(
        llm,
        stage=ReasoningStage.THESIS,
        stage_config=config.stages.thesis,
        system=system_prompt(),
        user_prompt=build_thesis_prompt(brief, snapshot, positions, risk_limits, recent_proposals),
        output_model=ThesisOutput,
        max_retries=config.max_retries,
        cycle_id=cycle_id,
        post_validate=lambda parsed: validate_candidates(parsed, snapshot),
    )
    return ThesisResult(
        output=output,
        usage=usage,
        raw_output=raw,
        prompt_version=prompt_version(ReasoningStage.THESIS),
    )
