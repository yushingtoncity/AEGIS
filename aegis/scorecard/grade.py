"""Grade proposals against what the underlying did afterwards. Pure: no I/O.

Each proposal gets a direction. An equity buy is bullish and an equity sell
bearish. An option structure's direction is the sign of its net delta per
share, summed over its legs from the Greeks the cycle's snapshot stored
(a bought call or sold put adds, a bought put or sold call subtracts); below
``neutral_delta`` it is neutral and left out of hit rates. A structure whose
legs lack stored Greeks falls back to a single leg's obvious sign, and is
``unknown`` when it has several.

Prices are split- and dividend-adjusted daily bars, so a split is not a
crash and an ex-dividend drop does not count against a bullish call. The
entry price is the price the proposal could actually have traded at:

- ``snapshot``: the underlying's spot in the cycle's stored snapshot, when
  the snapshot marked it fresh. It is a raw price, so it is moved onto the
  adjusted basis with the adjusted/raw ratio of the session it traded in.
- ``next_open``: when the spot was stale and the proposal came while the
  market was shut, the open of the next session. The stale spot is the last
  trade before the news the brain read, so using it would credit Claude
  with a move it could not have caught.
- none: a stale spot during the session has no clean replacement in daily
  bars, so the proposal is left ungraded and counted in ``no_entry``.

Horizon ``k`` is the close of the ``k``-th session that ended after the
proposal: a proposal at 10:00 on a Monday is graded at 1 day on Monday's
close and at 5 on Friday's. A session counts as ended at 16:00 New York, so
an early-close day is treated as a full one. A horizon whose session has not
closed yet is pending.

Options are graded twice. The hit rate grades the call on the underlying's
move, the question "was the direction right". Beside it, ``option_pnl``
estimates what the structure itself would have made after time decay and
the bid-ask spread (see ``aegis.scorecard.options``), because a right call
can still be a losing option trade.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, time
from functools import partial
from zoneinfo import ZoneInfo

from aegis.brain.models import CONFIDENCE_HORIZON_SESSIONS, confidence_is_defined
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
from aegis.scorecard.options import estimate
from aegis.scorecard.stats import mean_interval, shuffle_p_value, wilson_interval
from aegis.store.models import Instrument, OrderSide

NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

CORPORATE_ACTION_TOLERANCE = 0.05
"""How far the adjusted/raw price ratio may drift between entry and a horizon
before an option estimate is skipped. A dividend moves it by its size, well
under 5% for anything AEGIS trades; a split moves it by 50% or more and
changes the contract's deliverable, which the estimate cannot follow. Fixed,
not tunable: it only separates those two cases."""

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


def _by_day(bars: Sequence[Bar]) -> dict[date, Bar]:
    return {bar.timestamp.astimezone(NEW_YORK).date(): bar for bar in bars}


def _basis_ratio(adjusted: Bar, raw: Bar | None) -> float:
    """adjusted / raw for one session; 1 when the raw bar is missing."""
    if raw is None or raw.close <= 0 or adjusted.close <= 0:
        return 1.0
    return adjusted.close / raw.close


def _ratio_at(when: datetime, bars: Sequence[Bar], raw: Mapping[date, Bar]) -> float:
    """The adjusted/raw ratio of the session that was trading at ``when``: the
    first one that closes at or after it, else (today's bar not out yet) the
    last one before. The session itself, not the day before, because a spot
    taken on a split or ex-dividend day is already on that day's raw basis."""
    if not bars:
        return 1.0
    during = next((bar for bar in bars if session_close(bar) >= when), bars[-1])
    return _basis_ratio(during, raw.get(during.timestamp.astimezone(NEW_YORK).date()))


def session_open(bar: Bar) -> datetime:
    """When the daily bar's session opened (its New York date at 09:30)."""
    day = bar.timestamp.astimezone(NEW_YORK).date()
    return datetime.combine(day, SESSION_OPEN, tzinfo=NEW_YORK)


def _entry(
    inputs: ProposalInputs, bars: Sequence[Bar], after: Sequence[Bar], raw: Mapping[date, Bar]
) -> tuple[float | None, EntrySource | None, str | None]:
    """The entry price on the adjusted basis, where it came from, and why
    there is none when there is none."""
    spot, spot_time = inputs.snapshot_spot, inputs.spot_time
    if spot is not None and spot > 0 and spot_time is not None and not inputs.spot_stale:
        return spot * _ratio_at(spot_time, bars, raw), "snapshot", None
    if not bars:
        return None, None, f"no price bars for {inputs.underlying}"
    if after and session_open(after[0]) >= inputs.created_at and after[0].open > 0:
        return after[0].open, "next_open", None
    if spot is None:
        return None, None, "no entry price: no snapshot spot, and the proposal came during a session"
    return None, None, "no entry price: the snapshot spot was stale and the proposal came during a session"


def _option_fields(
    inputs: ProposalInputs, bar: Bar, raw: Mapping[date, Bar], entry_ratio: float
) -> dict[str, float | bool | None]:
    raw_bar = raw.get(bar.timestamp.astimezone(NEW_YORK).date(), bar)
    if abs(_basis_ratio(bar, raw_bar) / entry_ratio - 1) > CORPORATE_ACTION_TOLERANCE:
        return {}
    result = estimate(inputs, raw_bar.close, session_close(bar))
    if result is None:
        return {}
    return {
        "option_pnl": result.pnl,
        "option_return": result.return_on_debit,
        "option_hit": result.pnl > 0,
    }


def _horizon(
    days: int,
    after: Sequence[Bar],
    entry: float | None,
    sign: int,
    now: datetime,
    option: Callable[[Bar], dict[str, float | bool | None]] | None = None,
) -> HorizonResult:
    if len(after) < days or session_close(after[days - 1]) > now:
        return HorizonResult(days=days, pending=True)
    bar = after[days - 1]
    session = bar.timestamp.astimezone(NEW_YORK).date()
    if entry is None:
        return HorizonResult(days=days, session=session, close=bar.close)
    move = bar.close / entry - 1
    extra = option(bar) if option is not None else {}
    if sign == 0:
        return HorizonResult(
            days=days, session=session, close=bar.close, underlying_return=move, **extra
        )
    signed = sign * move
    return HorizonResult(
        days=days,
        session=session,
        close=bar.close,
        underlying_return=move,
        directional_return=signed,
        hit=signed > 0,
        **extra,
    )


def score_proposal(
    inputs: ProposalInputs,
    bars: Sequence[Bar],
    horizons_days: Sequence[int],
    *,
    now: datetime,
    neutral_delta: float,
    raw_bars: Sequence[Bar] | None = None,
) -> ScoredProposal:
    """Grade one proposal on ``bars``, the underlying's adjusted daily bars
    (any order). ``raw_bars`` are the same sessions unadjusted, used to move
    the snapshot spot onto the adjusted basis and to price option legs; None
    treats ``bars`` as both."""
    ordered = sorted(bars, key=lambda bar: bar.timestamp)
    raw = _by_day(ordered if raw_bars is None else raw_bars)
    direction, basis = direction_of(inputs, neutral_delta)
    after = [bar for bar in ordered if session_close(bar) > inputs.created_at]
    entry, source, note = _entry(inputs, ordered, after, raw)
    if note is None and not ordered:
        note = f"no price bars for {inputs.underlying}"
    sign = {"bullish": 1, "bearish": -1}.get(direction, 0)
    option = None
    if inputs.instrument is Instrument.OPTION and source == "snapshot" and inputs.spot_time:
        entry_ratio = _ratio_at(inputs.spot_time, ordered, raw)
        option = partial(_option_fields, inputs, raw=raw, entry_ratio=entry_ratio)
    return ScoredProposal(
        proposal_id=inputs.proposal_id,
        created_at=inputs.created_at,
        symbol=inputs.symbol,
        underlying=inputs.underlying,
        instrument=inputs.instrument,
        side=inputs.side,
        confidence=inputs.confidence,
        proposal_model=inputs.proposal_model,
        prompt_version=inputs.prompt_version,
        thesis_model=inputs.thesis_model,
        model_source=inputs.model_source,
        verdict=inputs.verdict,
        direction=direction,
        direction_basis=basis,
        entry_price=entry,
        entry_source=source,
        horizons=tuple(
            _horizon(days, after, entry, sign, now, option) for days in horizons_days
        ),
        note=note,
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _confidence_scored(
    graded: Sequence[ScoredProposal], results: Sequence[HorizonResult], days: int
) -> list[tuple[ScoredProposal, HorizonResult]]:
    """The graded calls whose confidence is a probability of this horizon's
    hit: the ``CONFIDENCE_HORIZON_SESSIONS`` horizon only, and only proposals
    written under the defined meaning. An older proposal's confidence named
    no event, so scoring it against any horizon would be meaningless."""
    if days != CONFIDENCE_HORIZON_SESSIONS:
        return []
    return [(p, r) for p, r in zip(graded, results) if confidence_is_defined(p.prompt_version)]


def _group_summary(
    name: str, members: Sequence[ScoredProposal], index: int, days: int
) -> GroupSummary:
    graded = [p for p in members if p.horizons[index].hit is not None]
    results = [p.horizons[index] for p in graded]
    hits = sum(1 for result in results if result.hit)
    hit_rate = hits / len(graded) if graded else None
    signed = [result.directional_return or 0.0 for result in results]
    moves = [result.underlying_return or 0.0 for result in results]
    signs = [1 if p.direction == "bullish" else -1 for p in graded]
    hit_interval = wilson_interval(hits, len(graded))
    confident = _confidence_scored(graded, results, days)
    confident_hit_rate = _mean([1.0 if r.hit else 0.0 for _, r in confident])
    return_interval = mean_interval(signed)
    options = [p.horizons[index].option_hit for p in members]
    option_results = [hit for hit in options if hit is not None]
    option_hits = sum(1 for hit in option_results if hit)
    return GroupSummary(
        group=name,
        horizon_days=days,
        proposals=len(members),
        graded=len(graded),
        hits=hits,
        hit_rate=hit_rate,
        avg_directional_return=_mean(signed),
        confidence_graded=len(confident),
        avg_confidence=_mean([p.confidence for p, _ in confident]),
        brier=_mean([(p.confidence - (1.0 if r.hit else 0.0)) ** 2 for p, r in confident]),
        brier_reference=(
            confident_hit_rate * (1 - confident_hit_rate) if confident_hit_rate is not None else None
        ),
        no_entry=sum(1 for p in members if p.entry_price is None),
        hit_rate_low=hit_interval[0] if hit_interval else None,
        hit_rate_high=hit_interval[1] if hit_interval else None,
        avg_return_low=return_interval[0] if return_interval else None,
        avg_return_high=return_interval[1] if return_interval else None,
        long_hit_rate=(
            sum(1 for move in moves if move > 0) / len(moves) if moves else None
        ),
        long_avg_return=_mean(moves),
        shuffle_p=shuffle_p_value(signs, moves),
        option_graded=len(option_results),
        option_hits=option_hits,
        option_hit_rate=option_hits / len(option_results) if option_results else None,
    )


def summarize(
    scored: Iterable[ScoredProposal], horizons_days: Sequence[int], group_by: str
) -> tuple[GroupSummary, ...]:
    """Per group and horizon: how many calls were graded, hit rate and
    average signed return with 95% intervals, the always-long and
    shuffled-direction baselines, average confidence, Brier score and its
    base-rate reference, and the option P&L win rate. Groups sort by name."""
    key = GROUP_KEYS[group_by]
    groups: dict[str, list[ScoredProposal]] = {}
    for proposal in scored:
        groups.setdefault(key(proposal), []).append(proposal)
    return tuple(
        _group_summary(name, groups[name], index, days)
        for name in sorted(groups)
        for index, days in enumerate(horizons_days)
    )


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
    raw_bars_by_symbol: Mapping[str, Sequence[Bar]] | None = None,
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
            raw_bars=None if raw_bars_by_symbol is None else raw_bars_by_symbol.get(item.underlying, ()),
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
