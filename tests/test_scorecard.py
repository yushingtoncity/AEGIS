"""The read-only proposal scorecard: grading rules, the store join, the CLI.

Prices are fixture bars; nothing here calls a live API. The last class holds
the scorecard to its read-only promise statically: no execution, broker,
policy or store-write imports anywhere in it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from aegis.brain.cycle import market_snapshot_event
from aegis.brain.models import MarketSnapshot
from aegis.cli import scorecard as cli
from aegis.config import REPO_ROOT, ScorecardConfig, get_config
from aegis.data.models import Bar, OptionType
from aegis.scorecard.gather import gather, underlying_of
from aegis.pricing.black_scholes import price as bs_price
from aegis.pricing.time_to_expiry import time_to_expiry
from aegis.scorecard.grade import build_scorecard, direction_of, score_proposal
from aegis.scorecard.models import LegQuote, ProposalInputs
from aegis.scorecard.options import why_not_estimable
from aegis.scorecard.stats import mean_interval, shuffle_p_value, wilson_interval
from aegis.store.db import open_store
from aegis.store.models import (
    Event,
    EventLevel,
    Instrument,
    OrderSide,
    OrderType,
    PolicyDecision,
    Proposal,
    ProposalLeg,
    Reasoning,
    ReasoningStage,
    Verdict,
)
from aegis.store.repo import add_reasoning, insert_proposal, log_event, record_decision

FIXTURES = Path(__file__).parent / "fixtures"
UTC = timezone.utc

# Mon 2026-08-03 10:00 New York (14:00 UTC); the fixture snapshot's SPY spot is 638.75
MONDAY_10AM = datetime(2026, 8, 3, 14, 0, tzinfo=UTC)


def bar(day: date, close: float, symbol: str = "SPY") -> Bar:
    # Alpaca stamps a daily bar at midnight New York (04:00 UTC in summer)
    stamp = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC)
    return Bar(
        symbol=symbol, timestamp=stamp, open=close, high=close, low=close, close=close,
        volume=1000, fetched_at=stamp,
    )


def week_of_bars(symbol: str = "SPY") -> list[Bar]:
    """Fri 07-31 close 630, then Mon..Fri 08-03..08-07 at 640, 645, 635, 650, 660."""
    closes = [(date(2026, 7, 31), 630.0), (date(2026, 8, 3), 640.0), (date(2026, 8, 4), 645.0),
              (date(2026, 8, 5), 635.0), (date(2026, 8, 6), 650.0), (date(2026, 8, 7), 660.0)]
    return [bar(day, close, symbol) for day, close in closes]


def inputs(**overrides) -> ProposalInputs:
    fields = dict(
        proposal_id="p1", created_at=MONDAY_10AM, cycle_id="c1", symbol="SPY",
        underlying="SPY", instrument=Instrument.EQUITY, side=OrderSide.BUY, quantity=1,
        confidence=0.7, proposal_model="claude-opus-5-5", snapshot_spot=638.75,
        spot_time=MONDAY_10AM - timedelta(seconds=2), spot_stale=False,
    )
    fields.update(overrides)
    return ProposalInputs(**fields)


def leg(index: int, option_type: OptionType, side: OrderSide, strike: float,
        symbol: str | None = None, proposal_id: str = "p1") -> ProposalLeg:
    code = "C" if option_type is OptionType.CALL else "P"
    return ProposalLeg(
        proposal_id=proposal_id, leg_index=index,
        symbol=symbol or f"SPY260821{code}{int(strike * 1000):08d}",
        option_type=option_type, side=side, quantity=1, strike=strike,
        expiration=date(2026, 8, 21),
    )


NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)  # Saturday after the fixture week


class TestGrading:
    def test_equity_buy_is_graded_on_later_closes_from_the_snapshot_spot(self):
        scored = score_proposal(inputs(), week_of_bars(), (1, 5), now=NOW, neutral_delta=0.1)
        assert (scored.direction, scored.entry_source, scored.entry_price) == ("bullish", "snapshot", 638.75)
        one, five = scored.horizons
        assert one.session == date(2026, 8, 3) and one.hit is True
        assert one.directional_return == pytest.approx(640 / 638.75 - 1)
        assert five.session == date(2026, 8, 7) and five.directional_return == pytest.approx(660 / 638.75 - 1)

    def test_equity_sell_is_bearish(self):
        scored = score_proposal(inputs(side=OrderSide.SELL), week_of_bars(), (1,), now=NOW, neutral_delta=0.1)
        assert scored.direction == "bearish"
        assert scored.horizons[0].hit is False
        assert scored.horizons[0].directional_return == pytest.approx(-(640 / 638.75 - 1))

    def test_without_a_snapshot_a_proposal_in_session_is_not_graded(self):
        # the last close predates whatever news the brain read that morning
        scored = score_proposal(inputs(snapshot_spot=None), week_of_bars(), (1,), now=NOW, neutral_delta=0.1)
        assert (scored.entry_price, scored.entry_source) == (None, None)
        assert scored.horizons[0].hit is None
        assert "during a session" in scored.note

    def test_a_proposal_after_the_close_enters_at_the_next_open(self):
        after_close = datetime(2026, 8, 3, 21, 0, tzinfo=UTC)  # 17:00 New York
        bars = [bar(d, c) for d, c in [(date(2026, 8, 3), 640.0)]] + [
            Bar(symbol="SPY", timestamp=datetime(2026, 8, 4, 4, 0, tzinfo=UTC), open=642.0,
                high=646, low=641, close=645.0, volume=1, fetched_at=NOW)]
        scored = score_proposal(
            inputs(created_at=after_close, spot_stale=True), bars, (1,), now=NOW,
            neutral_delta=0.1,
        )
        assert (scored.entry_price, scored.entry_source) == (642.0, "next_open")
        assert scored.horizons[0].session == date(2026, 8, 4)
        assert scored.horizons[0].directional_return == pytest.approx(645 / 642 - 1)

    def test_a_stale_spot_during_the_session_is_not_graded(self):
        # IEX froze at 10:00, a headline moved the stock by 11:00: grading from
        # the frozen price would credit the call with a move it could not catch
        scored = score_proposal(inputs(spot_stale=True), week_of_bars(), (1, 5), now=NOW, neutral_delta=0.1)
        assert scored.entry_price is None and "stale" in scored.note
        assert all(h.hit is None for h in scored.horizons)

    def test_a_session_that_has_not_closed_is_pending(self):
        tuesday_noon = datetime(2026, 8, 4, 16, 0, tzinfo=UTC)
        scored = score_proposal(inputs(), week_of_bars(), (1, 2, 10), now=tuesday_noon, neutral_delta=0.1)
        one, two, ten = scored.horizons
        assert one.hit is True and not one.pending
        assert two.pending and two.hit is None  # Tuesday's bar exists but its session is still open
        assert ten.pending

    def test_no_bars_and_no_snapshot_leaves_it_ungraded_with_a_note(self):
        scored = score_proposal(inputs(snapshot_spot=None), [], (1,), now=NOW, neutral_delta=0.1)
        assert scored.entry_price is None and scored.horizons[0].pending
        assert "no price bars" in scored.note

    def test_bull_call_spread_is_bullish_by_net_delta(self):
        legs = (leg(0, OptionType.CALL, OrderSide.BUY, 635), leg(1, OptionType.CALL, OrderSide.SELL, 645))
        deltas = {legs[0].symbol: 0.63, legs[1].symbol: 0.43}
        item = inputs(instrument=Instrument.OPTION, legs=legs, leg_deltas=deltas)
        assert direction_of(item, 0.1) == ("bullish", "net delta +0.20")

    def test_iron_condor_is_neutral_and_never_counts_as_a_hit(self):
        legs = (leg(0, OptionType.PUT, OrderSide.BUY, 620), leg(1, OptionType.PUT, OrderSide.SELL, 630),
                leg(2, OptionType.CALL, OrderSide.SELL, 650), leg(3, OptionType.CALL, OrderSide.BUY, 660))
        deltas = dict(zip((l.symbol for l in legs), (-0.15, -0.28, 0.33, 0.18)))
        item = inputs(instrument=Instrument.OPTION, legs=legs, leg_deltas=deltas)
        scored = score_proposal(item, week_of_bars(), (1,), now=NOW, neutral_delta=0.1)
        assert scored.direction == "neutral"
        assert scored.horizons[0].hit is None
        assert scored.horizons[0].underlying_return == pytest.approx(640 / 638.75 - 1)

    def test_a_single_long_put_without_stored_deltas_is_bearish(self):
        item = inputs(instrument=Instrument.OPTION, legs=(leg(0, OptionType.PUT, OrderSide.BUY, 630),))
        assert direction_of(item, 0.1)[0] == "bearish"

    def test_a_spread_without_stored_deltas_is_unknown(self):
        legs = (leg(0, OptionType.CALL, OrderSide.BUY, 635), leg(1, OptionType.CALL, OrderSide.SELL, 645))
        assert direction_of(inputs(instrument=Instrument.OPTION, legs=legs), 0.1)[0] == "unknown"

    def test_summary_hit_rate_return_and_brier_per_model(self):
        items = [
            inputs(proposal_id="a", confidence=0.8),  # buy, right at 1d
            inputs(proposal_id="b", confidence=0.6, side=OrderSide.SELL),  # wrong at 1d
            inputs(proposal_id="c", confidence=0.5, proposal_model="claude-fable-5-1"),
        ]
        card = build_scorecard(items, {"SPY": week_of_bars()}, (1,), now=NOW, neutral_delta=0.1)
        by_group = {s.group: s for s in card.summaries}
        opus, fable = by_group["claude-opus-5-5"], by_group["claude-fable-5-1"]
        assert (opus.proposals, opus.graded, opus.hits, opus.hit_rate) == (2, 2, 1, 0.5)
        assert opus.avg_directional_return == pytest.approx(0.0)
        assert opus.brier == pytest.approx(((0.8 - 1) ** 2 + (0.6 - 0) ** 2) / 2)
        assert (fable.graded, fable.hit_rate) == (1, 1.0)

    def test_group_by_pair_and_unknown_group_by(self):
        items = [inputs(thesis_model="claude-opus-5-5")]
        card = build_scorecard(items, {}, (1,), now=NOW, neutral_delta=0.1, group_by="pair")
        assert card.summaries[0].group == "claude-opus-5-5 -> claude-opus-5-5"
        assert card.summaries[0].hit_rate is None
        with pytest.raises(ValueError):
            build_scorecard(items, {}, (1,), now=NOW, neutral_delta=0.1, group_by="vibes")


def raw_and_adjusted(rows: list[tuple[date, float, float]], symbol: str = "SPY") -> tuple[list[Bar], list[Bar]]:
    """(day, raw close, adjusted close) rows as raw and adjusted bar lists."""
    return [bar(d, r, symbol) for d, r, _ in rows], [bar(d, a, symbol) for d, _, a in rows]


FRIDAY_10AM = datetime(2026, 7, 31, 14, 0, tzinfo=UTC)


class TestAdjustedBasis:
    def test_a_split_inside_the_horizon_is_not_a_crash(self):
        # 2-for-1 split effective Monday 08-03: raw halves, adjusted is smooth
        raw, adjusted = raw_and_adjusted([
            (date(2026, 7, 30), 1250.0, 625.0),
            (date(2026, 7, 31), 1260.0, 630.0),
            (date(2026, 8, 3), 640.0, 640.0),
        ])
        item = inputs(created_at=FRIDAY_10AM, snapshot_spot=1277.5,
                      spot_time=FRIDAY_10AM - timedelta(seconds=2))
        scored = score_proposal(item, adjusted, (1, 2), now=NOW, neutral_delta=0.1, raw_bars=raw)
        # the raw spot is moved onto the adjusted basis with Friday's ratio
        assert scored.entry_price == pytest.approx(638.75)
        one, two = scored.horizons
        assert one.directional_return == pytest.approx(630 / 638.75 - 1)
        assert two.directional_return == pytest.approx(640 / 638.75 - 1) and two.hit is True
        # graded on raw closes it would have been a 50% loss
        unadjusted = score_proposal(item, raw, (2,), now=NOW, neutral_delta=0.1)
        assert unadjusted.horizons[0].directional_return == pytest.approx(640 / 1277.5 - 1)

    def test_a_spot_taken_on_the_split_day_is_already_post_split(self):
        raw, adjusted = raw_and_adjusted([(date(2026, 7, 31), 1260.0, 630.0), (date(2026, 8, 3), 640.0, 640.0)])
        scored = score_proposal(inputs(), adjusted, (1,), now=NOW, neutral_delta=0.1, raw_bars=raw)
        assert scored.entry_price == pytest.approx(638.75)

    def test_a_dividend_adjusts_the_entry_by_its_sessions_ratio(self):
        # Monday's bar predates a later ex-date, so its adjusted close is 1% lower
        raw, adjusted = raw_and_adjusted([(date(2026, 7, 31), 100.0, 99.0), (date(2026, 8, 3), 101.0, 99.99),
                                          (date(2026, 8, 4), 100.0, 100.0)])
        scored = score_proposal(inputs(snapshot_spot=101.0), adjusted, (1, 2), now=NOW,
                                neutral_delta=0.1, raw_bars=raw)
        assert scored.entry_price == pytest.approx(101.0 * 0.99)
        # the 1.00 dividend paid out is no longer a loss for the bullish call
        assert scored.horizons[1].directional_return == pytest.approx(100.0 / 99.99 - 1)

    def test_without_raw_bars_the_bars_are_taken_as_both_bases(self):
        scored = score_proposal(inputs(), week_of_bars(), (1,), now=NOW, neutral_delta=0.1, raw_bars=None)
        assert scored.entry_price == 638.75


class TestStats:
    def test_wilson_interval(self):
        low, high = wilson_interval(7, 10)
        assert (low, high) == (pytest.approx(0.3968, abs=1e-4), pytest.approx(0.8922, abs=1e-4))
        assert wilson_interval(0, 0) is None
        assert wilson_interval(10, 10)[1] == pytest.approx(1.0) and wilson_interval(0, 10)[0] == pytest.approx(0.0)

    def test_a_small_sample_says_almost_nothing(self):
        # 14 of 20 right sounds good; the interval runs from a coin flip to great
        low, high = wilson_interval(14, 20)
        assert low < 0.5 < 0.85 < high

    def test_mean_interval_is_a_student_t_interval(self):
        low, high = mean_interval([1.0, 2.0, 3.0])
        assert (low, high) == (pytest.approx(2 - 2.4841, abs=1e-3), pytest.approx(2 + 2.4841, abs=1e-3))
        assert mean_interval([1.0]) is None

    def test_shuffle_p_is_exact_for_small_samples(self):
        # the calls got the best of the 6 possible labelings
        assert shuffle_p_value([1, 1, -1, -1], [0.02, 0.01, -0.01, -0.03]) == pytest.approx(1 / 6)
        # the worst one: every labeling does at least as well
        assert shuffle_p_value([-1, -1, 1, 1], [0.02, 0.01, -0.01, -0.03]) == 1.0
        assert shuffle_p_value([], []) is None

    def test_a_model_that_only_ever_says_up_has_no_direction_to_test(self):
        assert shuffle_p_value([1, 1, 1], [0.05, 0.04, 0.03]) == 1.0

    def test_shuffle_p_falls_back_to_a_seeded_sample(self):
        moves = [0.01 * (i + 1) * (1 if i % 2 else -1) for i in range(20)]
        signs = [1 if move > 0 else -1 for move in moves]  # every call right; C(20,10) > 10,000
        first = shuffle_p_value(signs, moves)
        assert first == shuffle_p_value(signs, moves) == pytest.approx(1 / 10_001)
        assert shuffle_p_value(signs, moves, samples=200, seed=1) == pytest.approx(1 / 201)


class TestSummaryBaselines:
    def test_a_bull_and_a_bear_call_on_a_rising_day(self):
        items = [inputs(proposal_id="a", confidence=0.8), inputs(proposal_id="b", side=OrderSide.SELL),
                 inputs(proposal_id="c", spot_stale=True)]
        (summary,) = build_scorecard(items, {"SPY": week_of_bars()}, (1,), now=NOW, neutral_delta=0.1).summaries
        assert (summary.proposals, summary.graded, summary.hits, summary.no_entry) == (3, 2, 1, 1)
        assert (summary.hit_rate_low, summary.hit_rate_high) == pytest.approx(wilson_interval(1, 2))
        assert summary.avg_return_low < 0 < summary.avg_return_high
        assert summary.long_hit_rate == 1.0  # SPY rose, so always-long was right both times
        assert summary.long_avg_return == pytest.approx(640 / 638.75 - 1)
        assert summary.shuffle_p == 1.0  # the same move twice: swapping labels changes nothing
        assert summary.brier_reference == pytest.approx(0.25)
        assert summary.option_graded == 0 and summary.option_hit_rate is None

    def test_an_always_bullish_model_matches_always_long_exactly(self):
        items = [inputs(proposal_id=str(i)) for i in range(3)]
        (summary,) = build_scorecard(items, {"SPY": week_of_bars()}, (5,), now=NOW, neutral_delta=0.1).summaries
        assert summary.hit_rate == summary.long_hit_rate == 1.0
        assert summary.avg_directional_return == pytest.approx(summary.long_avg_return)
        assert summary.shuffle_p == 1.0


def option_inputs(legs, quotes, **overrides) -> ProposalInputs:
    return inputs(instrument=Instrument.OPTION, legs=tuple(legs), leg_quotes=quotes,
                  risk_free_rate=0.04, **overrides)


def expected_leg_pnl(item: ProposalInputs, leg_: ProposalLeg, close: float, at: datetime) -> float:
    quote = item.leg_quotes[leg_.symbol]
    then = bs_price(leg_.option_type, item.snapshot_spot, leg_.strike,
                    time_to_expiry(leg_.expiration, item.spot_time), 0.04, quote.implied_vol)
    later = bs_price(leg_.option_type, close, leg_.strike, time_to_expiry(leg_.expiration, at),
                     0.04, quote.implied_vol)
    side = 1 if leg_.side is OrderSide.BUY else -1
    return 100 * leg_.quantity * (side * (later - then) - (quote.ask - quote.bid))


class TestOptionEstimate:
    CALL = leg(0, OptionType.CALL, OrderSide.BUY, 640)
    QUOTES = {CALL.symbol: LegQuote(bid=12.05, ask=12.35, implied_vol=0.1845)}

    def test_a_right_call_can_still_lose_money_on_the_option(self):
        # SPY ticks up 0.04% by the close: the direction was right, but the
        # spread alone costs more than the delta earned
        bars = [bar(date(2026, 7, 31), 630.0), bar(date(2026, 8, 3), 639.0)]
        item = option_inputs([self.CALL], self.QUOTES)
        result = score_proposal(item, bars, (1,), now=NOW, neutral_delta=0.1).horizons[0]
        assert result.hit is True and result.option_hit is False
        at = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
        assert result.option_pnl == pytest.approx(expected_leg_pnl(item, self.CALL, 639.0, at))
        assert result.option_return == pytest.approx(result.option_pnl / 1235.0)

    def test_a_big_move_pays_after_theta_and_the_spread(self):
        item = option_inputs([self.CALL], self.QUOTES)
        five = score_proposal(item, week_of_bars(), (5,), now=NOW, neutral_delta=0.1).horizons[0]
        at = datetime(2026, 8, 7, 20, 0, tzinfo=UTC)
        assert five.option_pnl == pytest.approx(expected_leg_pnl(item, self.CALL, 660.0, at))
        assert five.option_hit is True

    def test_a_flat_week_shows_the_time_decay(self):
        bars = [bar(date(2026, 7, 31), 630.0)] + [bar(date(2026, 8, d), 638.75) for d in range(3, 8)]
        item = option_inputs([self.CALL], {self.CALL.symbol: LegQuote(bid=12.2, ask=12.2, implied_vol=0.1845)})
        five = score_proposal(item, bars, (5,), now=NOW, neutral_delta=0.1).horizons[0]
        assert five.underlying_return == 0 and five.option_pnl < 0  # no spread: all theta

    def test_a_credit_structure_has_no_return_on_debit(self):
        put = leg(0, OptionType.PUT, OrderSide.SELL, 635)
        item = option_inputs([put], {put.symbol: LegQuote(bid=9.7, ask=9.95, implied_vol=0.199)})
        five = score_proposal(item, week_of_bars(), (5,), now=NOW, neutral_delta=0.1).horizons[0]
        assert five.option_pnl is not None and five.option_return is None
        assert five.option_hit is True  # SPY ran away from the short put

    def test_a_condor_gets_an_option_grade_though_no_direction(self):
        legs = (leg(0, OptionType.PUT, OrderSide.BUY, 630), leg(1, OptionType.PUT, OrderSide.SELL, 635),
                leg(2, OptionType.CALL, OrderSide.SELL, 645), leg(3, OptionType.CALL, OrderSide.BUY, 650))
        quotes = {legs[0].symbol: LegQuote(bid=8.1, ask=8.35, implied_vol=0.2088),
                  legs[1].symbol: LegQuote(bid=9.7, ask=9.95, implied_vol=0.199),
                  legs[2].symbol: LegQuote(bid=9.65, ask=9.95, implied_vol=0.179),
                  legs[3].symbol: LegQuote(bid=7.5, ask=7.8, implied_vol=0.1748)}
        deltas = dict(zip((l.symbol for l in legs), (-0.15, -0.28, 0.33, 0.18)))
        item = option_inputs(legs, quotes, leg_deltas=deltas)
        card = build_scorecard([item], {"SPY": week_of_bars()}, (1,), now=NOW, neutral_delta=0.1)
        (scored,), (summary,) = card.proposals, card.summaries
        assert scored.direction == "neutral" and scored.horizons[0].hit is None
        at = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
        assert scored.horizons[0].option_pnl == pytest.approx(
            sum(expected_leg_pnl(item, l, 640.0, at) for l in legs))
        assert summary.graded == 0 and summary.option_graded == 1

    def test_no_estimate_from_a_stale_entry_or_missing_quotes(self):
        stale = option_inputs([self.CALL], self.QUOTES, spot_stale=True)
        assert score_proposal(stale, week_of_bars(), (1,), now=NOW, neutral_delta=0.1).horizons[0].option_pnl is None
        bare = option_inputs([self.CALL], {})
        assert why_not_estimable(bare) == f"no stored quote and implied vol for {self.CALL.symbol}"
        assert score_proposal(bare, week_of_bars(), (1,), now=NOW, neutral_delta=0.1).horizons[0].option_pnl is None

    def test_no_estimate_across_a_split(self):
        # proposal Friday, 2-for-1 split effective Monday: the contract's deliverable changes
        raw, adjusted = raw_and_adjusted([(date(2026, 7, 30), 630.0, 315.0), (date(2026, 7, 31), 640.0, 320.0),
                                          (date(2026, 8, 3), 330.0, 330.0)])
        item = option_inputs([self.CALL], self.QUOTES, created_at=FRIDAY_10AM,
                             spot_time=FRIDAY_10AM - timedelta(seconds=2))
        friday, monday = score_proposal(item, adjusted, (1, 2), now=NOW, neutral_delta=0.1,
                                        raw_bars=raw).horizons
        assert friday.option_pnl is not None
        assert monday.hit is True and monday.option_pnl is None


class TestGetBarsAdjustment:
    class FakeStockClient:
        def __init__(self):
            self.requests = []

        def get_stock_bars(self, request):
            self.requests.append(request)
            return type("BarSet", (), {"data": {}})()

    def test_adjustment_reaches_the_request_and_the_cache_key(self, monkeypatch):
        from alpaca.data.enums import Adjustment

        from aegis.data import market
        from aegis.data.cache import default_cache

        client = self.FakeStockClient()
        monkeypatch.setattr(market, "stock_client", lambda: client)
        default_cache.clear()
        try:
            market.get_bars("SPY", "1Day", lookback_days=9, adjustment="all")
            market.get_bars("SPY", "1Day", lookback_days=9, adjustment="raw")
            market.get_bars("SPY", "1Day", lookback_days=9)
        finally:
            default_cache.clear()
        assert [r.adjustment for r in client.requests] == [Adjustment.ALL, Adjustment.RAW, None]
        with pytest.raises(ValueError, match="unknown bar adjustment"):
            market.get_bars("SPY", "1Day", adjustment="splits")


class TestConfig:
    def test_repo_config_has_a_scorecard_block(self):
        config = get_config().scorecard
        assert config.horizons_days == (1, 5, 10)
        assert config.neutral_delta == 0.10

    def test_horizons_are_sorted_and_must_be_positive_and_distinct(self):
        assert ScorecardConfig(horizons_days=(10, 1)).horizons_days == (1, 10)
        for bad in ((0,), (1, 1), ()):
            with pytest.raises(ValidationError):
                ScorecardConfig(horizons_days=bad)


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot.model_validate(
        json.loads((FIXTURES / "brain" / "market_snapshot.json").read_text())
    )


def _proposal(pid: str, cycle: str, symbol: str, instrument: Instrument, created_at: datetime,
              model: str = "claude-opus-5-5") -> Proposal:
    return Proposal(
        id=pid, created_at=created_at, cycle_id=cycle, symbol=symbol, instrument=instrument,
        side=OrderSide.BUY, quantity=1, order_type=OrderType.LIMIT, limit_price=1.0,
        thesis="t", confidence=0.65, invalidation="i", raw_model_output="{}",
        model_name=model, prompt_version="v1",
    )


@pytest.fixture
def db(tmp_path: Path, snapshot: MarketSnapshot) -> Path:
    """Two cycles: an option spread with a snapshot, a verdict and model events,
    and an older equity buy from before model_selection existed."""
    path = tmp_path / "aegis.db"
    conn = open_store(path)
    try:
        log_event(conn, market_snapshot_event("c1", snapshot))
        log_event(conn, Event(level=EventLevel.INFO, kind="model_selection", message="m", payload={
            "cycle_id": "c1",
            "stages": [{"stage": "thesis", "model": "claude-opus-5-5", "source": "latest"},
                       {"stage": "proposal", "model": "claude-opus-5-5", "source": "latest"}],
        }))
        add_reasoning(conn, Reasoning(cycle_id="c1", stage=ReasoningStage.THESIS, content="{}",
                                      model_name="claude-opus-5-5"))
        spread = _proposal("p-spread", "c1", "SPY", Instrument.OPTION, MONDAY_10AM)
        legs = [leg(0, OptionType.CALL, OrderSide.BUY, 635, proposal_id="p-spread"),
                leg(1, OptionType.CALL, OrderSide.SELL, 645, proposal_id="p-spread")]
        insert_proposal(conn, spread, legs)
        record_decision(conn, PolicyDecision(proposal_id="p-spread", verdict=Verdict.REJECT,
                                             rules_evaluated=[]))
        insert_proposal(conn, _proposal("p-old", "c0", "QQQ", Instrument.EQUITY,
                                        MONDAY_10AM - timedelta(days=3), model="claude-fable-5-1"))
    finally:
        conn.close()
    return path


class TestGather:
    def test_joins_legs_verdict_models_and_the_cycle_snapshot(self, db: Path):
        conn = cli.open_read_only(db)
        try:
            old, spread = gather(conn)
        finally:
            conn.close()
        assert spread.underlying == "SPY" and spread.snapshot_spot == 638.75
        assert spread.verdict == "REJECT"
        assert (spread.thesis_model, spread.model_source) == ("claude-opus-5-5", "latest")
        assert spread.leg_deltas == {"SPY260821C00635000": 0.6302, "SPY260821C00645000": 0.4305}
        assert spread.spot_stale is False
        assert spread.spot_time == datetime(2026, 7, 30, 19, 44, 58, tzinfo=UTC)
        assert spread.risk_free_rate == 0.04
        assert spread.leg_quotes["SPY260821C00635000"] == LegQuote(bid=14.95, ask=15.25, implied_vol=0.1898)
        assert set(spread.leg_quotes) == {"SPY260821C00635000", "SPY260821C00645000"}
        assert old.proposal_id == "p-old"
        assert (old.snapshot_spot, old.model_source, old.thesis_model, old.verdict) == (None, None, None, None)
        assert old.spot_stale is True and old.leg_quotes == {} and old.risk_free_rate is None

    def test_since_and_until_bound_the_range(self, db: Path):
        conn = cli.open_read_only(db)
        try:
            assert [i.proposal_id for i in gather(conn, since=MONDAY_10AM)] == ["p-spread"]
            assert [i.proposal_id for i in gather(conn, until=MONDAY_10AM)] == ["p-old"]
        finally:
            conn.close()

    def test_underlying_of_an_occ_proposal_symbol(self):
        proposal = _proposal("x", "c", "SPY260821C00640000", Instrument.OPTION, MONDAY_10AM)
        assert underlying_of(proposal, ()) == "SPY"

    def test_the_read_only_connection_refuses_writes(self, db: Path):
        conn = cli.open_read_only(db)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("DELETE FROM proposals")
        finally:
            conn.close()


def _fake_fetcher(calls: list[tuple[str, int, str]]):
    def fetch(symbol: str, lookback_days: int, adjustment: str) -> list[Bar]:
        calls.append((symbol, lookback_days, adjustment))
        if symbol == "QQQ":
            from aegis.data.errors import DataError
            raise DataError("1Day bars", symbol)
        return week_of_bars(symbol)
    return fetch


class TestCli:
    def _main(self, db: Path, *args: str, calls: list | None = None, monkeypatch=None) -> int:
        monkeypatch.setattr(cli, "utcnow", lambda: NOW)
        return cli.main(["--db", str(db), *args], fetcher=_fake_fetcher(calls if calls is not None else []))

    def test_summary_and_detail(self, db: Path, capsys, monkeypatch):
        calls: list[tuple[str, int, str]] = []
        assert self._main(db, "--detail", calls=calls, monkeypatch=monkeypatch) == 0
        out = capsys.readouterr().out
        assert "2 proposal(s) from the start to now, grouped by proposal-stage model" in out
        assert "price fetch failed for QQQ" in out
        assert "claude-opus-5-5" in out and "1/1" in out
        assert "bullish (net delta +0.20)" in out and "entry 638.75 (snapshot)" in out
        assert "models claude-opus-5-5 -> claude-opus-5-5 (latest)" in out
        # adjusted bars for every underlying, raw bars only where adjusted ones came back
        assert [(s, adj) for s, _, adj in calls] == [("QQQ", "all"), ("SPY", "all"), ("SPY", "raw")]
        assert all(days == (date(2026, 8, 8) - date(2026, 7, 31)).days + 7 for _, days, _ in calls)
        assert "split- and dividend-adjusted closes" in out
        assert "baselines on the same graded calls:" in out and "shuffle p" in out

    def test_json(self, db: Path, capsys, monkeypatch):
        assert self._main(db, "--json", "--group-by", "pair", monkeypatch=monkeypatch) == 0
        card = json.loads(capsys.readouterr().out)
        assert card["group_by"] == "pair" and len(card["proposals"]) == 2
        assert card["price_errors"] and card["price_errors"][0].startswith("QQQ")

    def test_the_database_is_byte_for_byte_unchanged(self, db: Path, capsys, monkeypatch):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in db.parent.iterdir()}
        conn = sqlite3.connect(db)
        changes = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        conn.close()
        assert self._main(db, monkeypatch=monkeypatch) == 0
        conn = sqlite3.connect(db)
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone() == changes
        conn.close()
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in db.parent.iterdir()}
        assert after["aegis.db"] == before["aegis.db"]

    def test_missing_database_fails_cleanly(self, tmp_path: Path, capsys):
        assert cli.main(["--db", str(tmp_path / "nope.db")], fetcher=_fake_fetcher([])) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_bad_date_is_an_argument_error(self, db: Path):
        with pytest.raises(SystemExit):
            cli.main(["--db", str(db), "--since", "yesterday"], fetcher=_fake_fetcher([]))


SCORECARD_FILES = sorted((REPO_ROOT / "aegis" / "scorecard").glob("*.py")) + [
    REPO_ROOT / "aegis" / "cli" / "scorecard.py"
]
FORBIDDEN_MODULES = ("aegis.execution", "aegis.policy", "aegis.data.clients", "alpaca", "aegis.notify")


class TestReadOnlyByConstruction:
    @pytest.mark.parametrize("path", SCORECARD_FILES, ids=lambda p: p.name)
    def test_no_execution_broker_policy_or_store_write_imports(self, path: Path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module == "aegis.store.repo":
                    writes = [a.name for a in node.names if not a.name.startswith("get_")]
                    assert not writes, f"{path.name} imports store writers {writes}"
                if node.module == "aegis.store.db":
                    assert not {a.name for a in node.names} & {"transaction", "migrate", "open_store"}
            else:
                continue
            for name in names:
                assert not any(
                    name == bad or name.startswith(bad + ".") for bad in FORBIDDEN_MODULES
                ), f"{path.name} imports {name}"
