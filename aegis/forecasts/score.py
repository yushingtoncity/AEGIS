"""Grade stored forecasts against what happened. Pure: no I/O.

Every forecast is scored with log loss, -ln(p) when its event happened and
-ln(1 - p) when it did not, and reported as its **gain** over always saying
0.5: ln 2 minus its log loss. Above zero is better than a coin flip.

The pre-registered primary test asks whether the primary arm's next-session
relative forecasts carry information at all: the mean gain against the same
probabilities shuffled across tickers within each date (``PrimaryTest``).
Shuffling within a date keeps that day's market and the model's overall
lean, and breaks only the link between which ticker got which number.

Intervals come from a moving-block bootstrap over whole dates, blocks as
long as the horizon, because forecasts made on one day share one market and
overlapping five-session windows share most of their days. The effective
sample size is the count the bootstrap's spread is worth if the forecasts
were independent.

Each graded forecast is also scored against no-skill baselines, all built
point in time from adjusted closes whose outcomes were known before the
forecast's session began:

- ``base_rate``: how often this ticker's event happened over the last
  ``baseline_lookback_sessions`` sessions;
- ``signal_5`` and ``signal_20``: how often the event happened, across the
  whole universe, after the same sign of the stock's 5- (or 20-) session
  return relative to the benchmark. This is short-term reversal and
  momentum, learned from the data rather than assumed.

A baseline with fewer than ``baseline_min_samples`` past outcomes says 0.5.
Every baseline probability is held inside the same [0.05, 0.95] as the
model's. The model has to beat these, not just 0.5, before its gain means
skill.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from aegis.config import ForecastScoreConfig
from aegis.data.models import Bar
from aegis.forecasts.events import closes_by_day, outcome
from aegis.forecasts.models import (
    ArmComparison,
    BaselineScore,
    CalibrationBin,
    ForecastScorecard,
    GroupScore,
    Interval,
    PrimaryTest,
    ScoredForecast,
)
from aegis.scorecard.stats import CONFIDENCE_LEVEL, SHUFFLE_SAMPLES, SHUFFLE_SEED, wilson_interval
from aegis.store.models import (
    PRIMARY_HORIZON,
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    ForecastArm,
    ForecastEvent,
    ForecastRecord,
    ForecastRun,
)

COIN_LOG_LOSS = math.log(2.0)

CALIBRATION_WIDTH = 0.05
"""Width of each calibration bin: 18 bins across [0.05, 0.95]. Fixed so every
report bins the same way; it is a reading aid, not a test."""

SIGNAL_SESSIONS = (5, 20)
"""The relative-return lookbacks of the two signal baselines: a trading week
(short-term reversal) and a trading month (momentum)."""

BOOTSTRAP_SEED = 20261022
"""Fixed so the same data always prints the same intervals."""


def log_loss(probability: float, happened: bool) -> float:
    return -math.log(probability if happened else 1.0 - probability)


def _clamp(probability: float) -> float:
    return min(PROBABILITY_CEILING, max(PROBABILITY_FLOOR, probability))


# --- run selection ------------------------------------------------------------


def select_runs(runs: Iterable[ForecastRun]) -> tuple[list[ForecastRun], int]:
    """The first-completed run of each arm per session; the rest are counted
    and left out, so one day never counts twice."""
    kept: dict[tuple[str, date], ForecastRun] = {}
    duplicates = 0
    for run in sorted(runs, key=lambda r: (r.completed_at, r.id)):
        key = (run.arm.value, run.anchor_session)
        if key in kept:
            duplicates += 1
            continue
        kept[key] = run
    return sorted(kept.values(), key=lambda r: (r.anchor_session, r.arm.value)), duplicates


# --- baselines ----------------------------------------------------------------


@dataclass(frozen=True)
class _Past:
    """One past window of one ticker: its start and end session, whether the
    event happened, and the relative-return signs known at its start."""

    start: date
    end: date
    happened: bool
    signs: tuple[int, ...]


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _signs_before(
    index: int, sessions: Sequence[date], closes: Mapping[date, float], bench: Mapping[date, float]
) -> tuple[int, ...]:
    """The sign of the relative return over each ``SIGNAL_SESSIONS`` lookback,
    ending at the session before ``sessions[index]``; 0 when there is not
    enough history."""
    last = index - 1
    signs: list[int] = []
    for k in SIGNAL_SESSIONS:
        if last - k < 0:
            signs.append(0)
            continue
        a, b = sessions[last - k], sessions[last]
        relative = (closes[b] / closes[a]) - (bench[b] / bench[a])
        signs.append(_sign(relative))
    return tuple(signs)


class Baselines:
    """Point-in-time baseline probabilities from adjusted closes."""

    def __init__(
        self,
        closes: Mapping[str, Mapping[date, float]],
        benchmark: str,
        config: ForecastScoreConfig,
    ) -> None:
        self._closes = closes
        self._benchmark = benchmark
        self._config = config
        bench = closes.get(benchmark, {})
        self._bench_sessions = sorted(bench)
        self._history: dict[tuple[str, int], list[_Past]] = {}
        self._pooled: dict[tuple[date, int, int, int], float] = {}

    def _past(self, symbol: str, horizon: int) -> list[_Past]:
        key = (symbol, horizon)
        if key not in self._history:
            self._history[key] = self._build(symbol, horizon)
        return self._history[key]

    def _build(self, symbol: str, horizon: int) -> list[_Past]:
        bench = self._closes.get(self._benchmark, {})
        own = self._closes.get(symbol, {})
        sessions = sorted(set(own) & set(bench))
        position = {day: index for index, day in enumerate(self._bench_sessions)}
        past: list[_Past] = []
        for i, start in enumerate(sessions):
            # the window follows the benchmark's sessions, exactly as a forecast's does
            j = position[start] + horizon
            if j >= len(self._bench_sessions):
                break
            end = self._bench_sessions[j]
            if end not in own:
                continue
            bench_return = bench[end] / bench[start] - 1.0
            if symbol == self._benchmark:
                happened = bench_return > 0
                signs: tuple[int, ...] = ()
            else:
                happened = own[end] / own[start] - 1.0 > bench_return
                signs = _signs_before(i, sessions, own, bench)
            past.append(_Past(start=start, end=end, happened=happened, signs=signs))
        return past

    def _window_start(self, anchor: date) -> date | None:
        before = [day for day in self._bench_sessions if day < anchor]
        if not before:
            return None
        lookback = self._config.baseline_lookback_sessions
        return before[-lookback] if len(before) >= lookback else before[0]

    def _known(self, past: Iterable[_Past], anchor: date) -> list[_Past]:
        start = self._window_start(anchor)
        if start is None:
            return []
        return [p for p in past if p.end < anchor and p.start >= start]

    def _rate(self, entries: Sequence[_Past]) -> float:
        if len(entries) < self._config.baseline_min_samples:
            return 0.5
        return _clamp(sum(p.happened for p in entries) / len(entries))

    def for_forecast(self, symbol: str, event: ForecastEvent, horizon: int, anchor: date) -> dict[str, float]:
        result = {"base_rate": self._rate(self._known(self._past(symbol, horizon), anchor))}
        if event is not ForecastEvent.BEATS_BENCHMARK:
            return result
        own = self._closes.get(symbol, {})
        bench = self._closes.get(self._benchmark, {})
        sessions = sorted(day for day in set(own) & set(bench) if day < anchor)
        signs = (
            _signs_before(len(sessions), sessions, own, bench) if sessions else (0,) * len(SIGNAL_SESSIONS)
        )
        for position, k in enumerate(SIGNAL_SESSIONS):
            result[f"signal_{k}"] = self._pooled_rate(anchor, horizon, position, signs[position])
        return result

    def _pooled_rate(self, anchor: date, horizon: int, position: int, sign: int) -> float:
        if sign == 0:
            return 0.5
        key = (anchor, horizon, position, sign)
        if key not in self._pooled:
            entries = [
                p
                for symbol in self._closes
                if symbol != self._benchmark
                for p in self._known(self._past(symbol, horizon), anchor)
                if p.signs and p.signs[position] == sign
            ]
            self._pooled[key] = self._rate(entries)
        return self._pooled[key]


# --- statistics ---------------------------------------------------------------


def _by_date(items: Sequence[tuple[date, float]]) -> dict[date, list[float]]:
    grouped: dict[date, list[float]] = defaultdict(list)
    for day, value in items:
        grouped[day].append(value)
    return dict(sorted(grouped.items()))


def block_bootstrap(
    values_by_date: Mapping[date, Sequence[float]], block: int, samples: int, seed: int = BOOTSTRAP_SEED
) -> tuple[Interval, float] | None:
    """A percentile interval for the mean over all values, resampling whole
    dates in moving blocks of ``block`` consecutive dates; and the
    bootstrap standard error. None with fewer than two dates."""
    dates = list(values_by_date)
    count = len(dates)
    if count < 2:
        return None
    block = max(1, min(block, count))
    sums = np.array([sum(values_by_date[d]) for d in dates], dtype=float)
    sizes = np.array([len(values_by_date[d]) for d in dates], dtype=float)
    blocks_needed = math.ceil(count / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, count - block + 1, size=(samples, blocks_needed))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(samples, -1)[:, :count]
    means = sums[index].sum(axis=1) / sizes[index].sum(axis=1)
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Interval(low=float(low), high=float(high)), float(np.std(means, ddof=1))


def effective_n(values: Sequence[float], bootstrap_se: float | None) -> float | None:
    """How many independent forecasts the bootstrap's spread is worth (at
    most the count itself)."""
    n = len(values)
    if n < 2 or not bootstrap_se:
        return None
    variance = statistics.variance(values)
    return min(float(n), variance / (bootstrap_se**2))


def shuffle_test(
    groups: Sequence[tuple[Sequence[float], Sequence[bool]]],
    samples: int = SHUFFLE_SAMPLES,
    seed: int = SHUFFLE_SEED,
) -> float | None:
    """One-sided p-value of the total gain against the probabilities
    shuffled within each group (one group per date). None with no forecasts."""
    if not any(probs for probs, _ in groups):
        return None
    rng = np.random.default_rng(seed)
    observed = 0.0
    shuffled = np.zeros(samples)
    for probs, happened in groups:
        p = np.asarray(probs, dtype=float)
        y = np.asarray(happened, dtype=bool)
        observed += float(np.sum(COIN_LOG_LOSS + np.log(np.where(y, p, 1.0 - p))))
        if len(p) < 2:
            shuffled += float(np.sum(COIN_LOG_LOSS + np.log(np.where(y, p, 1.0 - p))))
            continue
        order = rng.random((samples, len(p))).argsort(axis=1)
        q = p[order]
        shuffled += np.sum(COIN_LOG_LOSS + np.log(np.where(y[None, :], q, 1.0 - q)), axis=1)
    as_good = int(np.sum(shuffled >= observed - 1e-12))
    return (as_good + 1) / (samples + 1)


def calibration(pairs: Sequence[tuple[float, bool]]) -> tuple[CalibrationBin, ...]:
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    top = round((PROBABILITY_CEILING - PROBABILITY_FLOOR) / CALIBRATION_WIDTH) - 1
    for probability, happened in pairs:
        index = min(top, int((probability - PROBABILITY_FLOOR) / CALIBRATION_WIDTH + 1e-9))
        bins[index].append((probability, happened))
    result: list[CalibrationBin] = []
    for index in sorted(bins):
        members = bins[index]
        hits = sum(h for _, h in members)
        ci = wilson_interval(hits, len(members))
        low = PROBABILITY_FLOOR + index * CALIBRATION_WIDTH
        result.append(
            CalibrationBin(
                low=round(low, 4),
                high=round(low + CALIBRATION_WIDTH, 4),
                n=len(members),
                mean_forecast=sum(p for p, _ in members) / len(members),
                frequency=hits / len(members),
                interval=Interval(low=ci[0], high=ci[1]) if ci else None,
            )
        )
    return tuple(result)


# --- scoring ------------------------------------------------------------------


def score_forecasts(
    runs: Sequence[ForecastRun],
    forecasts: Mapping[str, Sequence[ForecastRecord]],
    bars: Mapping[str, Sequence[Bar]],
    benchmark: str,
    *,
    now: datetime,
    config: ForecastScoreConfig,
) -> list[ScoredForecast]:
    """Every forecast of ``runs`` with its outcome and baselines."""
    closes = {symbol: closes_by_day(series) for symbol, series in bars.items()}
    bench = closes.get(benchmark, {})
    baselines = Baselines(closes, benchmark, config)
    scored: list[ScoredForecast] = []
    for run in runs:
        for record in forecasts.get(run.id, ()):
            result = outcome(
                event=record.event,
                symbol_closes=closes.get(record.symbol, {}),
                benchmark_closes=bench,
                anchor_session=run.anchor_session,
                horizon=record.horizon_sessions,
                completed_at=run.completed_at,
                now=now,
            )
            base = (
                baselines.for_forecast(record.symbol, record.event, record.horizon_sessions, run.anchor_session)
                if result.status == "graded"
                else {}
            )
            scored.append(
                ScoredForecast(
                    run_id=run.id,
                    arm=run.arm.value,
                    model_name=run.model_name,
                    anchor_session=run.anchor_session,
                    symbol=record.symbol,
                    event=record.event.value,
                    horizon=record.horizon_sessions,
                    probability=record.probability,
                    status=result.status,
                    happened=result.happened,
                    target_session=result.target_session,
                    reason=result.reason,
                    baselines=base,
                )
            )
    return scored


def _gains(items: Sequence[ScoredForecast]) -> list[tuple[date, float]]:
    return [(f.anchor_session, COIN_LOG_LOSS - log_loss(f.probability, bool(f.happened))) for f in items]


def _mean_with_interval(
    pairs: Sequence[tuple[date, float]], block: int, samples: int
) -> tuple[float | None, Interval | None, float | None]:
    if not pairs:
        return None, None, None
    values = [v for _, v in pairs]
    boot = block_bootstrap(_by_date(pairs), block, samples)
    interval, se = (boot if boot is not None else (None, None))
    return sum(values) / len(values), interval, effective_n(values, se)


def group_score(items: Sequence[ScoredForecast], arm: str, event: str, horizon: int, samples: int) -> GroupScore:
    graded = [f for f in items if f.status == "graded"]
    gains = _gains(graded)
    gain, interval, n_eff = _mean_with_interval(gains, horizon, samples)
    leaning = [f for f in graded if f.probability != 0.5]
    hits = sum((f.probability > 0.5) == f.happened for f in leaning)
    hit_ci = wilson_interval(hits, len(leaning))
    baseline_names = sorted({name for f in graded for name in f.baselines})
    baselines: list[BaselineScore] = []
    for name in baseline_names:
        usable = [f for f in graded if name in f.baselines]
        diffs = [
            (f.anchor_session, log_loss(f.baselines[name], bool(f.happened)) - log_loss(f.probability, bool(f.happened)))
            for f in usable
        ]
        mean_diff, diff_ci, _ = _mean_with_interval(diffs, horizon, samples)
        baselines.append(
            BaselineScore(
                name=name,
                log_loss=sum(log_loss(f.baselines[name], bool(f.happened)) for f in usable) / len(usable),
                model_gain=mean_diff if mean_diff is not None else 0.0,
                model_gain_interval=diff_ci,
            )
        )
    return GroupScore(
        arm=arm,
        models=tuple(sorted({f.model_name for f in items})),
        event=event,
        horizon=horizon,
        graded=len(graded),
        pending=sum(f.status == "pending" for f in items),
        ungradable=sum(f.status == "ungradable" for f in items),
        dates=len({f.anchor_session for f in graded}),
        log_loss=(sum(log_loss(f.probability, bool(f.happened)) for f in graded) / len(graded)) if graded else None,
        coin_log_loss=COIN_LOG_LOSS,
        gain=gain,
        gain_interval=interval,
        effective_n=n_eff,
        brier=(sum((f.probability - float(bool(f.happened))) ** 2 for f in graded) / len(graded)) if graded else None,
        lean_hits=hits,
        leaning=len(leaning),
        lean_hit_interval=Interval(low=hit_ci[0], high=hit_ci[1]) if hit_ci else None,
        baselines=tuple(baselines),
        calibration=calibration([(f.probability, bool(f.happened)) for f in graded]),
    )


def primary_test(scored: Sequence[ScoredForecast], samples: int) -> PrimaryTest:
    arm, event, horizon = ForecastArm.PRIMARY.value, ForecastEvent.BEATS_BENCHMARK.value, PRIMARY_HORIZON
    graded = [
        f for f in scored
        if f.arm == arm and f.event == event and f.horizon == horizon and f.status == "graded"
    ]
    gain, interval, n_eff = _mean_with_interval(_gains(graded), horizon, samples)
    by_day: dict[date, list[ScoredForecast]] = defaultdict(list)
    for f in graded:
        by_day[f.anchor_session].append(f)
    groups = [
        ([f.probability for f in day], [bool(f.happened) for f in day]) for _, day in sorted(by_day.items())
    ]
    return PrimaryTest(
        arm=arm,
        event=event,
        horizon=horizon,
        graded=len(graded),
        dates=len(by_day),
        gain=gain,
        gain_interval=interval,
        p_value=shuffle_test(groups),
        shuffles=SHUFFLE_SAMPLES,
        effective_n=n_eff,
    )


def compare_arms(scored: Sequence[ScoredForecast], samples: int) -> tuple[ArmComparison, ...]:
    graded = [f for f in scored if f.status == "graded"]
    keyed: dict[tuple[str, date, str, str, int], ScoredForecast] = {
        (f.arm, f.anchor_session, f.symbol, f.event, f.horizon): f for f in graded
    }
    result: list[ArmComparison] = []
    events = sorted({(f.event, f.horizon) for f in graded})
    for event, horizon in events:
        pairs: list[tuple[date, float]] = []
        for (arm, day, symbol, ev, h), primary in keyed.items():
            if arm != ForecastArm.PRIMARY.value or ev != event or h != horizon:
                continue
            shadow = keyed.get((ForecastArm.SHADOW.value, day, symbol, ev, h))
            if shadow is None:
                continue
            outcome_ = bool(primary.happened)
            pairs.append((day, log_loss(shadow.probability, outcome_) - log_loss(primary.probability, outcome_)))
        if not pairs:
            continue
        mean, interval, _ = _mean_with_interval(pairs, horizon, samples)
        result.append(
            ArmComparison(
                event=event, horizon=horizon, paired=len(pairs), primary_gain_over_shadow=mean, interval=interval
            )
        )
    return tuple(result)


def build_forecast_scorecard(
    runs: Sequence[ForecastRun],
    forecasts: Mapping[str, Sequence[ForecastRecord]],
    bars: Mapping[str, Sequence[Bar]],
    benchmark: str,
    *,
    now: datetime,
    config: ForecastScoreConfig,
    since: date | None = None,
    until: date | None = None,
    detail: bool = False,
) -> ForecastScorecard:
    kept, duplicates = select_runs(runs)
    scored = score_forecasts(kept, forecasts, bars, benchmark, now=now, config=config)
    samples = config.bootstrap_samples
    groups: list[GroupScore] = []
    keys = sorted({(f.arm, f.event, f.horizon) for f in scored})
    for arm, event, horizon in keys:
        members = [f for f in scored if (f.arm, f.event, f.horizon) == (arm, event, horizon)]
        groups.append(group_score(members, arm, event, horizon, samples))
    reasons = Counter(f.reason for f in scored if f.status == "ungradable" and f.reason)
    return ForecastScorecard(
        generated_at=now,
        since=since,
        until=until,
        runs=len(kept),
        duplicate_runs=duplicates,
        primary=primary_test(scored, samples),
        groups=tuple(groups),
        comparisons=compare_arms(scored, samples),
        ungradable_reasons=dict(reasons),
        forecasts=tuple(scored) if detail else (),
    )


__all__ = [
    "COIN_LOG_LOSS",
    "Baselines",
    "block_bootstrap",
    "build_forecast_scorecard",
    "calibration",
    "log_loss",
    "score_forecasts",
    "select_runs",
    "shuffle_test",
]
