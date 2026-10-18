"""Grade proposals against what the underlying did afterwards. Pure: no I/O.

Each proposal gets a direction. An equity buy is bullish and an equity sell
bearish. An option structure's direction is the sign of its net delta per
share, summed over its legs from the Greeks the cycle's snapshot stored
(a bought call or sold put adds, a bought put or sold call subtracts); below
``neutral_delta`` it is neutral and left out of hit rates. A structure whose
legs lack stored Greeks falls back to a single leg's obvious sign, and is
``unknown`` when it has several.

The entry price is the underlying's spot in the cycle's stored snapshot (what
the brain saw), else the close of the last session that ended before the
proposal. Horizon ``k`` is the close of the ``k``-th session that ended after
the proposal: a proposal at 10:00 on a Monday is graded at 1 day on Monday's
close and at 5 on Friday's. A session counts as ended at 16:00 New York, so
an early-close day is treated as a full one. A horizon whose session has not
closed yet is pending.

Options are graded on the underlying's move, not on the option's own price:
the free data plan has no option price history, and the question the
scorecard answers is whether the call on direction was right.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from aegis.data.models import Bar, OptionType
from aegis.scorecard.models import (
    Direction,
    EntrySource,
    GroupSummary,
    HorizonResult,
    ProposalInputs,
    Scorecard,
    ScoredProposal,
)
from aegis.store.models import Instrument, OrderSide

NEW_YORK = ZoneInfo("America/New_York")
SESSION_CLOSE = time(16, 0)

GROUP_KEYS: dict[str, Callable[[ScoredProposal], str]] = {
    "proposal": lambda p: p.proposal_model,
    "thesis": lambda p: p.thesis_model or "unknown",
    "pair": lambda p: f"{p.thesis_model or 'unknown'} -> {p.proposal_model}",
    "symbol": lambda p: p.underlying,
}


def session_close(bar: Bar) -> datetime:
    """When the daily bar's session ended (its New York date at 16:00)."""
    day = bar.timestamp.astimezone(NEW_YORK).date()
    return datetime.combine(day, SESSION_CLOSE, tzinfo=NEW_YORK)


def _leg_sign(option_type: OptionType, side: OrderSide) -> int:
    bullish = (option_type is OptionType.CALL) == (side is OrderSide.BUY)
    return 1 if bullish else -1


def direction_of(inputs: ProposalInputs, neutral_delta: float) -> tuple[Direction, str]:
    """The proposal's call on the underlying, and a short reason for it."""
    if inputs.instrument is Instrument.EQUITY:
        if inputs.side is OrderSide.BUY:
            return "bullish", "equity buy"
        return "bearish", "equity sell"
    legs = inputs.legs
    if not legs:
        return "unknown", "option proposal with no legs on record"
    if all(leg.symbol in inputs.leg_deltas for leg in legs):
        net = sum(
            inputs.leg_deltas[leg.symbol] * leg.quantity * (1 if leg.side is OrderSide.BUY else -1)
            for leg in legs
        ) / min(leg.quantity for leg in legs)
        basis = f"net delta {net:+.2f}"
        if abs(net) < neutral_delta:
            return "neutral", basis
        return ("bullish" if net > 0 else "bearish"), basis
    if len(legs) == 1:
        leg = legs[0]
        sign = _leg_sign(leg.option_type, leg.side)
        basis = f"{leg.side.value} {leg.option_type.value} (no stored delta)"
        return ("bullish" if sign > 0 else "bearish"), basis
    return "unknown", f"{len(legs)} legs without stored deltas"


def _entry(
    inputs: ProposalInputs, bars: Sequence[Bar]
) -> tuple[float | None, EntrySource | None]:
    if inputs.snapshot_spot is not None and inputs.snapshot_spot > 0:
        return inputs.snapshot_spot, "snapshot"
    before = [bar for bar in bars if session_close(bar) <= inputs.created_at]
    if before and before[-1].close > 0:
        return before[-1].close, "prior_close"
    return None, None


def _horizon(
    days: int,
    after: Sequence[Bar],
    entry: float | None,
    sign: int,
    now: datetime,
) -> HorizonResult:
    if len(after) < days or session_close(after[days - 1]) > now:
        return HorizonResult(days=days, pending=True)
    bar = after[days - 1]
    session = bar.timestamp.astimezone(NEW_YORK).date()
    if entry is None:
        return HorizonResult(days=days, session=session, close=bar.close)
    move = bar.close / entry - 1
    if sign == 0:
        return HorizonResult(days=days, session=session, close=bar.close, underlying_return=move)
    signed = sign * move
    return HorizonResult(
        days=days,
        session=session,
        close=bar.close,
        underlying_return=move,
        directional_return=signed,
        hit=signed > 0,
    )


def score_proposal(
    inputs: ProposalInputs,
    bars: Sequence[Bar],
    horizons_days: Sequence[int],
    *,
    now: datetime,
    neutral_delta: float,
) -> ScoredProposal:
    """Grade one proposal on ``bars`` (the underlying's daily bars, any order)."""
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    direction, basis = direction_of(inputs, neutral_delta)
    entry, source = _entry(inputs, ordered)
    sign = {"bullish": 1, "bearish": -1}.get(direction, 0)
    after = [bar for bar in ordered if session_close(bar) > inputs.created_at]
    note = None
    if entry is None:
        note = "no entry price: no stored snapshot spot and no bar before the proposal"
    elif not ordered:
        note = f"no price bars for {inputs.underlying}"
    return ScoredProposal(
        proposal_id=inputs.proposal_id,
        created_at=inputs.created_at,
        symbol=inputs.symbol,
        underlying=inputs.underlying,
        instrument=inputs.instrument,
        side=inputs.side,
        confidence=inputs.confidence,
        proposal_model=inputs.proposal_model,
        thesis_model=inputs.thesis_model,
        model_source=inputs.model_source,
        verdict=inputs.verdict,
        direction=direction,
        direction_basis=basis,
        entry_price=entry,
        entry_source=source,
        horizons=tuple(_horizon(days, after, entry, sign, now) for days in horizons_days),
        note=note,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    scored: Iterable[ScoredProposal], horizons_days: Sequence[int], group_by: str
) -> tuple[GroupSummary, ...]:
    """Per group and horizon: how many calls were graded, hit rate, average
    signed return, average confidence and Brier score. Groups sort by name."""
    key = GROUP_KEYS[group_by]
    groups: dict[str, list[ScoredProposal]] = {}
    for proposal in scored:
        groups.setdefault(key(proposal), []).append(proposal)
    summaries = []
    for name in sorted(groups):
        members = groups[name]
        for index, days in enumerate(horizons_days):
            graded = [
                (p.confidence, p.horizons[index])
                for p in members
                if p.horizons[index].hit is not None
            ]
            hits = sum(1 for _, result in graded if result.hit)
            summaries.append(
                GroupSummary(
                    group=name,
                    horizon_days=days,
                    proposals=len(members),
                    graded=len(graded),
                    hits=hits,
                    hit_rate=hits / len(graded) if graded else None,
                    avg_directional_return=_mean(
                        [result.directional_return for _, result in graded]
                    ),
                    avg_confidence=_mean([confidence for confidence, _ in graded]),
                    brier=_mean(
                        [(confidence - (1.0 if result.hit else 0.0)) ** 2 for confidence, result in graded]
                    ),
                )
            )
    return tuple(summaries)


def build_scorecard(
    inputs: Iterable[ProposalInputs],
    bars_by_symbol: Mapping[str, Sequence[Bar]],
    horizons_days: Sequence[int],
    *,
    now: datetime,
    neutral_delta: float,
    group_by: str = "proposal",
    since: datetime | None = None,
    until: datetime | None = None,
) -> Scorecard:
    if group_by not in GROUP_KEYS:
        raise ValueError(f"unknown group_by {group_by!r}: expected one of {', '.join(GROUP_KEYS)}")
    horizons = tuple(horizons_days)
    scored = tuple(
        score_proposal(
            item,
            bars_by_symbol.get(item.underlying, ()),
            horizons,
            now=now,
            neutral_delta=neutral_delta,
        )
        for item in inputs
    )
    return Scorecard(
        generated_at=now,
        since=since,
        until=until,
        group_by=group_by,
        horizons_days=horizons,
        proposals=scored,
        summaries=summarize(scored, horizons, group_by),
    )


def first_session_day(inputs: Iterable[ProposalInputs]) -> date | None:
    """The New York date of the earliest proposal, for sizing the bar fetch."""
    days = [item.created_at.astimezone(NEW_YORK).date() for item in inputs]
    return min(days) if days else None
