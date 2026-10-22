"""Outcome grading and the forecast scorer, on hand-built closes.

Hand-checked numbers: log loss of 0.8 on a hit is -ln 0.8 = 0.22314 and of
0.3 on a miss is -ln 0.7 = 0.35667, so their mean is 0.28991, the gain over
a coin is ln 2 - 0.28991 = 0.40324, and the Brier score is
(0.2^2 + 0.3^2) / 2 = 0.065. A planted-skill fixture must pass the primary
test and a no-skill fixture must not."""

import math
import random
from datetime import date, datetime, time, timedelta, timezone

import pytest

from aegis.config import ForecastScoreConfig
from aegis.forecasts.events import close_of, outcome
from aegis.forecasts.score import (
    COIN_LOG_LOSS,
    Baselines,
    build_forecast_scorecard,
    calibration,
    log_loss,
    select_runs,
    shuffle_test,
)
from aegis.scorecard.grade import NEW_YORK
from aegis.store.models import ForecastArm, ForecastEvent
from forecast_factories import bars_from, make_run, record, sessions

UP, BEATS = ForecastEvent.UP, ForecastEvent.BEATS_BENCHMARK
D0, D1, D2, D3 = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21), date(2026, 10, 22)
DURING_D0 = datetime.combine(D0, time(14, 0), tzinfo=NEW_YORK)
LATER = datetime(2026, 12, 1, tzinfo=timezone.utc)
CONFIG = ForecastScoreConfig(baseline_lookback_sessions=20, baseline_min_samples=10, bootstrap_samples=200)


def _grade(symbol, bench, *, event=BEATS, horizon=1, completed_at=DURING_D0, now=LATER, anchor=D0):
    return outcome(
        event=event,
        symbol_closes=symbol,
        benchmark_closes=bench,
        anchor_session=anchor,
        horizon=horizon,
        completed_at=completed_at,
        now=now,
    )


BENCH = {D0: 100.0, D1: 101.0, D2: 100.0, D3: 102.0}


class TestOutcome:
    def test_beats_and_misses(self):
        assert _grade({D0: 50.0, D1: 51.0}, BENCH).happened is True  # +2% vs +1%
        assert _grade({D0: 50.0, D1: 50.4}, BENCH).happened is False  # +0.8% vs +1%
        assert _grade({D0: 50.0, D1: 50.5}, BENCH).happened is False  # a tie is "no"

    def test_the_five_session_target_and_the_up_event(self):
        result = _grade(BENCH, BENCH, event=UP, horizon=3)
        assert result.status == "graded" and result.happened is True and result.target_session == D3
        assert _grade(BENCH, BENCH, event=UP, horizon=2).happened is False  # 100 -> 100 is not up

    def test_pending_until_the_target_closes(self):
        before = close_of(D1) - timedelta(minutes=1)
        result = _grade({D0: 50.0, D1: 51.0}, BENCH, now=before)
        assert result.status == "pending" and result.target_session == D1
        assert _grade({D0: 50.0}, {D0: 100.0}, now=LATER).status == "pending"  # no later bars yet

    def test_written_after_the_anchor_close_is_ungradable(self):
        result = _grade({D0: 50.0, D1: 51.0}, BENCH, completed_at=close_of(D0))
        assert result.status == "ungradable" and "after the anchor" in result.reason

    def test_a_missing_anchor_bar_is_ungradable_once_the_market_moved_on(self):
        result = _grade({D1: 51.0}, BENCH)
        assert result.status == "ungradable" and "anchor session" in result.reason
        assert _grade({}, {}, now=close_of(D0) - timedelta(hours=1)).status == "pending"

    def test_a_gap_in_the_stock_never_stretches_the_window(self):
        # no D1 bar for the stock: the 1-session forecast is NOT graded on D2
        result = _grade({D0: 50.0, D2: 60.0}, BENCH)
        assert result.status == "ungradable" and result.target_session == D1


class TestLogLossAndCalibration:
    def test_hand_checked_values(self):
        assert log_loss(0.8, True) == pytest.approx(0.22314, abs=1e-5)
        assert log_loss(0.3, False) == pytest.approx(0.35667, abs=1e-5)
        assert COIN_LOG_LOSS == pytest.approx(0.69315, abs=1e-5)

    def test_calibration_bins_cover_the_bounds(self):
        bins = calibration([(0.05, False), (0.5, True), (0.52, False), (0.95, True)])
        assert [(b.low, b.high, b.n) for b in bins] == [(0.05, 0.1, 1), (0.5, 0.55, 2), (0.9, 0.95, 1)]
        assert bins[1].frequency == 0.5 and bins[1].mean_forecast == pytest.approx(0.51)


class TestSelectRuns:
    def test_the_first_completed_run_per_arm_and_session_counts(self):
        first = make_run("a", completed_at=datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc))
        again = make_run("b", completed_at=datetime(2026, 10, 21, 19, 30, tzinfo=timezone.utc))
        shadow = make_run("c", arm=ForecastArm.SHADOW)
        kept, duplicates = select_runs([again, shadow, first])
        assert [r.id for r in kept] == ["a", "c"] and duplicates == 1


def _scorecard(runs, forecasts, closes, now=LATER, **kwargs):
    bars = {symbol: bars_from(symbol, series) for symbol, series in closes.items()}
    return build_forecast_scorecard(runs, forecasts, bars, "SPY", now=now, config=CONFIG, **kwargs)


class TestHandChecked:
    def test_log_loss_brier_and_gain_on_two_forecasts(self):
        closes = {"SPY": BENCH, "AAPL": {D0: 50.0, D1: 51.0}, "MSFT": {D0: 50.0, D1: 50.0}}
        run = make_run("r", anchor=D0)
        card = _scorecard([run], {"r": [record("r", "AAPL", 0.8), record("r", "MSFT", 0.3)]}, closes)
        [group] = card.groups
        assert (group.graded, group.pending, group.ungradable) == (2, 0, 0)
        assert group.log_loss == pytest.approx(0.28991, abs=1e-5)
        assert group.gain == pytest.approx(0.40324, abs=1e-5)
        assert group.brier == pytest.approx(0.065)
        assert (group.lean_hits, group.leaning) == (2, 2)
        assert card.primary.graded == 2 and card.primary.dates == 1


def _market(seed: int, days: int = 60, names: int = 20):
    """Random-walk closes for ``names`` stocks and SPY over ``days`` sessions."""
    rng = random.Random(seed)
    calendar = sessions(date(2026, 1, 5), days + 1)
    closes = {"SPY": {}}
    level = 100.0
    for day in calendar:
        level *= 1 + rng.gauss(0, 0.01)
        closes["SPY"][day] = level
    for n in range(names):
        price, series = 50.0, {}
        for day in calendar:
            price *= 1 + rng.gauss(0, 0.02)
            series[day] = price
        closes[f"S{n:02d}"] = series
    return calendar, closes


def _runs_with(calendar, closes, probability_for):
    runs, forecasts = [], {}
    for i, day in enumerate(calendar[:-1]):
        run = make_run(f"run-{i}", anchor=day, universe=[s for s in closes if s != "SPY"])
        nxt = calendar[i + 1]
        bench = closes["SPY"][nxt] / closes["SPY"][day]
        rows = []
        for symbol in run.universe:
            beat = closes[symbol][nxt] / closes[symbol][day] > bench
            rows.append(record(run.id, symbol, probability_for(beat)))
        runs.append(run)
        forecasts[run.id] = rows
    return runs, forecasts


class TestSkillDetection:
    def test_planted_skill_passes_the_primary_test(self):
        calendar, closes = _market(seed=1)
        runs, forecasts = _runs_with(calendar, closes, lambda beat: 0.6 if beat else 0.4)
        test = _scorecard(runs, forecasts, closes).primary
        assert test.graded == 60 * 20 and test.dates == 60
        assert test.gain == pytest.approx(COIN_LOG_LOSS + math.log(0.6))
        assert test.p_value < 0.01 and test.gain_interval.low > 0

    def test_no_skill_does_not(self):
        calendar, closes = _market(seed=2)
        noise = random.Random(3)
        runs, forecasts = _runs_with(calendar, closes, lambda beat: noise.choice((0.4, 0.6)))
        test = _scorecard(runs, forecasts, closes).primary
        # random 0.4/0.6 guesses cost about ln 2 + (ln 0.6 + ln 0.4) / 2 = -0.020 a forecast
        assert test.p_value > 0.05 and test.gain_interval.low < 0
        assert test.gain == pytest.approx(COIN_LOG_LOSS + (math.log(0.6) + math.log(0.4)) / 2, abs=0.01)

    def test_one_ticker_a_day_has_nothing_to_shuffle(self):
        assert shuffle_test([([0.9], [True])], samples=50) == 1.0
        assert shuffle_test([]) is None


class TestBaselines:
    def test_base_rate_uses_only_outcomes_known_before_the_anchor(self):
        calendar = sessions(date(2026, 3, 2), 40)
        spy = {day: 100.0 for day in calendar}
        # the stock beats every day before day 30, and loses every day after
        stock, price = {}, 50.0
        for i, day in enumerate(calendar):
            price *= 1.01 if i <= 30 else 0.99
            stock[day] = price
        baselines = Baselines({"SPY": spy, "AAA": stock}, "SPY", CONFIG)
        assert baselines.for_forecast("AAA", BEATS, 1, calendar[30])["base_rate"] == 0.95  # clamped
        assert baselines.for_forecast("AAA", BEATS, 1, calendar[5])["base_rate"] == 0.5  # too few yet

    def test_the_signal_baselines_learn_reversal_from_the_pooled_universe(self):
        calendar = sessions(date(2026, 3, 2), 80)
        spy = {day: 100.0 for day in calendar}
        # every stock alternates five sessions up, five down: after a 5-session gain it then loses
        closes = {"SPY": spy}
        for n in range(4):
            price, series = 50.0, {}
            for i, day in enumerate(calendar):
                price *= 1.01 if ((i + n) // 5) % 2 == 0 else 0.99
                series[day] = price
            closes[f"S{n}"] = series
        baselines = Baselines(closes, "SPY", CONFIG)
        probabilities = baselines.for_forecast("S0", BEATS, 1, calendar[70])
        assert set(probabilities) == {"base_rate", "signal_5", "signal_20"}
        assert all(0.05 <= p <= 0.95 for p in probabilities.values())

    def test_the_up_event_has_only_a_base_rate(self):
        calendar = sessions(date(2026, 3, 2), 40)
        spy = {day: 100.0 + i for i, day in enumerate(calendar)}
        assert Baselines({"SPY": spy}, "SPY", CONFIG).for_forecast("SPY", UP, 1, calendar[30]) == {
            "base_rate": 0.95
        }


class TestArms:
    def test_primary_and_shadow_are_compared_on_the_same_forecasts(self):
        closes = {"SPY": BENCH, "AAPL": {D0: 50.0, D1: 51.0}}
        primary, shadow = make_run("p", anchor=D0), make_run("s", anchor=D0, arm=ForecastArm.SHADOW)
        card = _scorecard(
            [primary, shadow], {"p": [record("p", "AAPL", 0.8)], "s": [record("s", "AAPL", 0.5)]}, closes
        )
        [comparison] = card.comparisons
        assert comparison.paired == 1
        assert comparison.primary_gain_over_shadow == pytest.approx(COIN_LOG_LOSS + math.log(0.8))
        assert {g.arm for g in card.groups} == {"primary", "shadow"}

    def test_detail_lists_every_forecast_and_ungradable_reasons_are_counted(self):
        closes = {"SPY": BENCH, "AAPL": {D0: 50.0, D1: 51.0}}
        late = make_run("late", anchor=D0, completed_at=close_of(D0) + timedelta(minutes=5))
        card = _scorecard([late], {"late": [record("late", "AAPL", 0.6)]}, closes, detail=True)
        assert card.ungradable_reasons == {"written after the anchor session's close": 1}
        assert [f.status for f in card.forecasts] == ["ungradable"]
