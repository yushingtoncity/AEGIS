"""Multi-leg position analysis — pure math over OptionLeg inputs, no fixtures
or network. Each named strategy is checked against hand-worked expiry
arithmetic; the grid method must reproduce it without knowing the name."""

import random
from datetime import date

import pytest

from aegis.data.models import OptionType
from aegis.pricing.errors import PricingError
from aegis.pricing.models import Greeks, OptionLeg, PremiumType, Side
from aegis.pricing.position import (
    analyze_position,
    net_greeks,
    net_premium,
    payoff_at_expiry,
)

EXPIRY = date(2026, 8, 21)
LATER = date(2026, 9, 18)


def _leg(option_type, side, strike, premium, **kwargs):
    kwargs.setdefault("expiry", EXPIRY)
    kwargs.setdefault("quantity", 1)
    return OptionLeg(
        option_type=option_type, side=side, strike=strike, premium=premium, **kwargs
    )


def _call(side, strike, premium, **kwargs):
    return _leg(OptionType.CALL, side, strike, premium, **kwargs)


def _put(side, strike, premium, **kwargs):
    return _leg(OptionType.PUT, side, strike, premium, **kwargs)


def _greeks(delta, gamma=0.02, theta=-0.05, vega=0.10, rho=0.03):
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def _vertical(quantity=1):
    """Long 100C @ 5.00 / short 110C @ 2.00: a 3.00 debit for 10 of width."""
    return [
        _call(Side.LONG, 100.0, 5.0, quantity=quantity),
        _call(Side.SHORT, 110.0, 2.0, quantity=quantity),
    ]


class TestNetPremium:
    def test_debit_vertical(self):
        assert net_premium(_vertical()) == pytest.approx(300.0)

    def test_credit_is_negative(self):
        assert net_premium([_call(Side.SHORT, 100.0, 3.0)]) == pytest.approx(-300.0)

    def test_quantity_and_multiplier(self):
        legs = [_call(Side.LONG, 100.0, 2.5, quantity=4)]
        assert net_premium(legs) == pytest.approx(1000.0)
        assert net_premium(legs, contract_multiplier=10) == pytest.approx(100.0)

    def test_mixed_expiries_are_fine_here(self):
        # A calendar spread has a perfectly good net premium; only its expiry
        # payoff is undefined.
        legs = [
            _call(Side.SHORT, 100.0, 2.0),
            _call(Side.LONG, 100.0, 3.5, expiry=LATER),
        ]
        assert net_premium(legs) == pytest.approx(150.0)


class TestNetGreeks:
    def test_sign_quantity_and_multiplier_applied(self):
        legs = [
            _call(Side.LONG, 100.0, 5.0, quantity=2, greeks=_greeks(0.5)),
            _call(Side.SHORT, 110.0, 2.0, greeks=_greeks(0.3, 0.01, -0.03, 0.08, 0.02)),
        ]
        total = net_greeks(legs)
        assert total is not None
        assert total.delta == pytest.approx(70.0)  # 100 * (2 * 0.5 - 0.3)
        assert total.gamma == pytest.approx(3.0)
        assert total.theta == pytest.approx(-7.0)
        assert total.vega == pytest.approx(12.0)
        assert total.rho == pytest.approx(4.0)

    def test_contract_multiplier(self):
        legs = [_call(Side.LONG, 100.0, 5.0, greeks=_greeks(0.5))]
        assert net_greeks(legs, contract_multiplier=10).delta == pytest.approx(5.0)

    def test_any_leg_without_greeks_gives_none(self):
        legs = [
            _call(Side.LONG, 100.0, 5.0, greeks=_greeks(0.5)),
            _call(Side.SHORT, 110.0, 2.0),
        ]
        assert net_greeks(legs) is None


class TestPayoffAtExpiry:
    @pytest.mark.parametrize(
        ("underlying", "expected"),
        [
            (0.0, -300.0),
            (100.0, -300.0),
            (103.0, 0.0),
            (105.0, 200.0),
            (110.0, 700.0),
            (500.0, 700.0),
        ],
    )
    def test_long_call_vertical(self, underlying, expected):
        assert payoff_at_expiry(_vertical(), underlying) == pytest.approx(expected)

    def test_zero_underlying_is_a_valid_settlement(self):
        # A long put is worth its full strike when the stock goes to zero.
        assert payoff_at_expiry([_put(Side.LONG, 90.0, 1.0)], 0.0) == pytest.approx(8900.0)

    def test_contract_multiplier(self):
        assert payoff_at_expiry(_vertical(), 110.0, contract_multiplier=10) == pytest.approx(70.0)

    def test_accepts_a_generator(self):
        # The expiry check walks the legs once; the sum must still see them.
        legs = _vertical()
        assert payoff_at_expiry((leg for leg in legs), 110.0) == pytest.approx(700.0)
        assert payoff_at_expiry(iter(legs), 0.0) == pytest.approx(-300.0)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_bad_underlying_raises(self, bad):
        with pytest.raises(PricingError):
            payoff_at_expiry(_vertical(), bad)


class TestAnalyzePosition:
    def test_long_call_vertical(self):
        legs = _vertical()
        summary = analyze_position(legs)
        assert summary.net_premium == pytest.approx(300.0)
        assert summary.premium_type is PremiumType.DEBIT
        assert summary.max_loss == pytest.approx(300.0)
        assert summary.max_profit == pytest.approx(700.0)  # (10 - 3) * 100
        assert summary.breakevens == pytest.approx((103.0,))
        assert not summary.unlimited_risk and not summary.unlimited_profit
        assert summary.legs == tuple(legs)
        assert summary.contract_multiplier == 100
        assert summary.net_greeks is None  # legs carry no greeks

    def test_naked_short_call(self):
        summary = analyze_position([_call(Side.SHORT, 100.0, 3.0)])
        assert summary.premium_type is PremiumType.CREDIT
        assert summary.unlimited_risk and summary.max_loss is None
        assert not summary.unlimited_profit
        assert summary.max_profit == pytest.approx(300.0)  # the credit
        assert summary.breakevens == pytest.approx((103.0,))  # K + premium

    def test_naked_long_call(self):
        summary = analyze_position([_call(Side.LONG, 100.0, 3.0)])
        assert summary.unlimited_profit and summary.max_profit is None
        assert not summary.unlimited_risk
        assert summary.max_loss == pytest.approx(300.0)  # the debit
        assert summary.breakevens == pytest.approx((103.0,))

    def test_iron_condor(self):
        legs = [
            _put(Side.LONG, 90.0, 1.0),
            _put(Side.SHORT, 95.0, 2.0),
            _call(Side.SHORT, 105.0, 2.0),
            _call(Side.LONG, 110.0, 1.0),
        ]
        summary = analyze_position(legs)
        assert summary.net_premium == pytest.approx(-200.0)
        assert summary.premium_type is PremiumType.CREDIT
        assert summary.max_loss == pytest.approx(300.0)  # (5 - 2) * 100
        assert summary.max_profit == pytest.approx(200.0)
        assert summary.breakevens == pytest.approx((93.0, 107.0))
        assert not summary.unlimited_risk and not summary.unlimited_profit

    def test_short_put_risk_is_bounded_by_zero(self):
        summary = analyze_position([_put(Side.SHORT, 100.0, 4.0)])
        assert not summary.unlimited_risk
        assert summary.max_loss == pytest.approx(9600.0)  # (K - premium) * 100
        assert summary.max_profit == pytest.approx(400.0)
        assert summary.breakevens == pytest.approx((96.0,))

    def test_long_straddle(self):
        legs = [_call(Side.LONG, 100.0, 5.0), _put(Side.LONG, 100.0, 4.0)]
        summary = analyze_position(legs)
        assert summary.unlimited_profit and summary.max_profit is None
        assert not summary.unlimited_risk
        assert summary.max_loss == pytest.approx(900.0)
        assert summary.breakevens == pytest.approx((91.0, 109.0))

    def test_ratio_spread_has_unlimited_risk(self):
        # 1x2: the extra short call dominates above 110. A "sum of verticals"
        # shortcut would call this bounded.
        legs = [
            _call(Side.LONG, 100.0, 4.0),
            _call(Side.SHORT, 110.0, 1.5, quantity=2),
        ]
        summary = analyze_position(legs)
        assert summary.unlimited_risk and summary.max_loss is None
        assert not summary.unlimited_profit
        assert summary.net_premium == pytest.approx(100.0)
        assert summary.max_profit == pytest.approx(900.0)  # at 110: (10 - 4) + 2 * 1.5
        assert summary.breakevens == pytest.approx((101.0, 119.0))

    def test_call_backspread_has_unlimited_profit(self):
        legs = [
            _call(Side.SHORT, 100.0, 4.0),
            _call(Side.LONG, 110.0, 1.5, quantity=2),
        ]
        summary = analyze_position(legs)
        assert summary.unlimited_profit and summary.max_profit is None
        assert not summary.unlimited_risk
        assert summary.premium_type is PremiumType.CREDIT
        assert summary.max_loss == pytest.approx(900.0)  # worst case at 110
        assert summary.breakevens == pytest.approx((101.0, 119.0))

    def test_breakeven_beyond_the_probe_point(self):
        # K + premium = 250 lies past the probe at 2 * 100 + 1, so no grid
        # segment brackets it; the far-slope rule must still find it.
        summary = analyze_position([_call(Side.SHORT, 100.0, 150.0)])
        assert summary.breakevens == pytest.approx((250.0,))

    def test_quantity_scales_everything(self):
        one = analyze_position(_vertical())
        three = analyze_position(_vertical(quantity=3))
        assert three.net_premium == pytest.approx(3 * one.net_premium)
        assert three.max_loss == pytest.approx(3 * one.max_loss)
        assert three.max_profit == pytest.approx(3 * one.max_profit)
        assert three.breakevens == pytest.approx(one.breakevens)

    def test_contract_multiplier_scales_dollars(self):
        summary = analyze_position(_vertical(), contract_multiplier=10)
        assert summary.contract_multiplier == 10
        assert summary.net_premium == pytest.approx(30.0)
        assert summary.max_loss == pytest.approx(30.0)
        assert summary.max_profit == pytest.approx(70.0)
        assert summary.breakevens == pytest.approx((103.0,))
        with_greeks = [_call(Side.LONG, 100.0, 5.0, greeks=_greeks(0.5))]
        assert analyze_position(with_greeks, contract_multiplier=10).net_greeks.delta == (
            pytest.approx(5.0)
        )

    def test_accepts_a_generator(self):
        legs = _vertical()
        summary = analyze_position(leg for leg in legs)
        assert summary.legs == tuple(legs)
        assert summary.max_loss == pytest.approx(300.0)
        assert summary.max_profit == pytest.approx(700.0)

    def test_fully_offset_position_is_flat(self):
        legs = [_call(Side.LONG, 100.0, 5.0), _call(Side.SHORT, 100.0, 5.0)]
        summary = analyze_position(legs)
        assert summary.max_profit == 0.0
        assert summary.max_loss == 0.0
        assert not summary.unlimited_risk and not summary.unlimited_profit
        assert summary.net_premium == 0.0
        # The P&L touches zero on every real grid point (S = 0 and the
        # strike) but the internal probe at 2 * 100 + 1 must not leak out.
        assert summary.breakevens == (0.0, 100.0)
        assert all(payoff_at_expiry(legs, b) == 0.0 for b in summary.breakevens)

    def test_probe_point_is_never_a_breakeven(self):
        # A far segment sitting exactly at zero touches at the last strike;
        # the probe beyond it is an implementation detail.
        at_width = [_call(Side.LONG, 100.0, 12.0), _call(Side.SHORT, 110.0, 2.0)]
        assert analyze_position(at_width).breakevens == (110.0,)
        free_put = [_put(Side.SHORT, 100.0, 0.0)]
        assert analyze_position(free_put).breakevens == (100.0,)
        odd_strike = [_put(Side.LONG, 473.23, 0.0, quantity=3)]
        assert analyze_position(odd_strike).breakevens == (473.23,)
        for legs in (at_width, free_put, odd_strike):
            probe = 2 * max(leg.strike for leg in legs) + 1
            assert probe not in analyze_position(legs).breakevens
        # ...unless the P&L genuinely crosses zero exactly there.
        assert analyze_position([_call(Side.SHORT, 100.0, 101.0)]).breakevens == (201.0,)

    def test_magnitudes_are_clamped_at_zero(self):
        # A vertical bought for more than its width cannot profit; sold for
        # more than its width it cannot lose. Neither side is ever negative.
        overpaid = [_call(Side.LONG, 100.0, 12.5), _call(Side.SHORT, 110.0, 2.0)]
        summary = analyze_position(overpaid)
        assert summary.max_profit == 0.0
        assert summary.max_loss == pytest.approx(1050.0)
        assert summary.breakevens == ()
        overcredited = [_put(Side.SHORT, 95.0, 6.0), _put(Side.LONG, 90.0, 0.5)]
        summary = analyze_position(overcredited)
        assert summary.max_loss == 0.0
        assert summary.max_profit == pytest.approx(550.0)
        assert not summary.unlimited_risk
        assert analyze_position([_put(Side.SHORT, 100.0, 101.0)]).max_loss == 0.0
        # Same strike, different premiums: always +100 / always -100.
        always_up = [_call(Side.LONG, 100.0, 1.0), _call(Side.SHORT, 100.0, 2.0)]
        summary = analyze_position(always_up)
        assert summary.max_loss == 0.0 and summary.max_profit == pytest.approx(100.0)
        always_down = [_call(Side.LONG, 100.0, 2.0), _call(Side.SHORT, 100.0, 1.0)]
        summary = analyze_position(always_down)
        assert summary.max_profit == 0.0 and summary.max_loss == pytest.approx(100.0)

    def test_pnl_touching_zero_at_a_strike_is_a_breakeven(self):
        # Zero net premium: P&L sits exactly at zero from 90 to 100 with no
        # strict sign change on either side, so only the grid rule finds it.
        legs = [
            _call(Side.LONG, 100.0, 6.0),
            _call(Side.SHORT, 110.0, 2.0),
            _put(Side.SHORT, 90.0, 4.0),
        ]
        assert analyze_position(legs).breakevens == pytest.approx((90.0, 100.0))

    def test_breakevens_are_sorted_when_a_touch_sits_above_a_crossing(self):
        # The zero-kink at 95 is found before the crossing at 85; an unsorted
        # list would let de-duplication swallow the crossing.
        legs = [
            _put(Side.SHORT, 90.0, 2.0),
            _call(Side.SHORT, 90.0, 4.0),
            _call(Side.LONG, 95.0, 1.0),
        ]
        breakevens = analyze_position(legs).breakevens
        assert breakevens[:2] == pytest.approx((85.0, 95.0))
        assert list(breakevens) == sorted(breakevens)

    def test_breakevens_stay_unique_at_large_strikes(self):
        # The far root is found twice (interpolation and the analytic slope
        # rule); above ~1e6 the two differ by an ULP and must still merge.
        legs = [
            _call(Side.LONG, 5840771.47, 842117.95),
            _call(Side.SHORT, 13272545.5, 106268.54, quantity=2),
        ]
        breakevens = analyze_position(legs).breakevens
        assert len(set(breakevens)) == len(breakevens) == 2
        assert breakevens == pytest.approx((6470352.34, 20074738.66))

    def test_zero_cost_reports_debit(self):
        legs = [_call(Side.LONG, 100.0, 2.0), _put(Side.SHORT, 90.0, 2.0)]
        summary = analyze_position(legs)
        assert summary.net_premium == 0.0
        assert summary.premium_type is PremiumType.DEBIT

    def test_net_greeks_flow_through(self):
        legs = [
            _call(Side.LONG, 100.0, 5.0, quantity=2, greeks=_greeks(0.5)),
            _call(Side.SHORT, 110.0, 2.0, greeks=_greeks(0.3)),
        ]
        summary = analyze_position(legs)
        assert summary.net_greeks is not None
        assert summary.net_greeks.delta == pytest.approx(70.0)
        without = [legs[0], legs[1].model_copy(update={"greeks": None})]
        assert analyze_position(without).net_greeks is None


class TestValidation:
    def test_empty_legs(self):
        with pytest.raises(PricingError):
            analyze_position([])
        with pytest.raises(PricingError):
            payoff_at_expiry([], 100.0)

    def test_mixed_expiries(self):
        legs = [
            _call(Side.SHORT, 100.0, 2.0),
            _call(Side.LONG, 100.0, 3.5, expiry=LATER),
        ]
        with pytest.raises(PricingError, match="mixed expiries"):
            analyze_position(legs)
        with pytest.raises(PricingError, match="2026-08-21, 2026-09-18"):
            payoff_at_expiry(legs, 100.0)

    @pytest.mark.parametrize("multiplier", [0, -100])
    def test_non_positive_multiplier(self, multiplier):
        legs = _vertical()
        for call in (
            lambda: net_premium(legs, contract_multiplier=multiplier),
            lambda: net_greeks(legs, contract_multiplier=multiplier),
            lambda: payoff_at_expiry(legs, 100.0, contract_multiplier=multiplier),
            lambda: analyze_position(legs, contract_multiplier=multiplier),
        ):
            with pytest.raises(PricingError, match="contract_multiplier must be positive"):
                call()

    def test_infinite_strike_or_premium(self):
        # OptionLeg's ge/gt bounds let inf through (NaN already fails them);
        # here it would turn into inf dollar figures rather than a clean error.
        bad = float("inf")
        for legs in (
            [_call(Side.LONG, 100.0, bad, symbol="XYZ")],
            [_call(Side.LONG, bad, 1.0)],
        ):
            with pytest.raises(PricingError, match="strike and premium must be finite"):
                analyze_position(legs)
            with pytest.raises(PricingError, match="strike and premium must be finite"):
                payoff_at_expiry(legs, 100.0)
        with pytest.raises(PricingError, match="expiry payoff for XYZ"):
            analyze_position([_call(Side.LONG, 100.0, bad, symbol="XYZ")])

    def test_overflowing_pnl_raises(self):
        # Finite inputs whose dollar P&L still exceeds a float: no inf in the
        # summary, and no complaint about an underlying price nobody passed.
        with pytest.raises(PricingError, match="no room to probe beyond strike 1e\\+308"):
            analyze_position([_call(Side.LONG, 1e308, 1.0)])
        legs = [_call(Side.LONG, 1e307, 1.0), _put(Side.SHORT, 5e306, 1.0)]
        with pytest.raises(PricingError, match="exceeds the float range"):
            analyze_position(legs)
        with pytest.raises(PricingError, match="exceeds the float range"):
            payoff_at_expiry(legs, 0.0)


class TestRandomPositions:
    """The grid method must agree with a brute-force sweep on positions
    nobody hand-coded."""

    @staticmethod
    def _random_legs(rng):
        return [
            _leg(
                rng.choice(list(OptionType)),
                rng.choice(list(Side)),
                strike=rng.choice([80.0, 90.0, 100.0, 110.0, 120.0]),
                premium=round(rng.uniform(0.25, 15.0), 2),
                quantity=rng.randint(1, 3),
            )
            for _ in range(rng.randint(1, 4))
        ]

    def test_grid_agrees_with_dense_sweep(self):
        rng = random.Random(1234)
        for _ in range(40):
            legs = self._random_legs(rng)
            summary = analyze_position(legs)
            strikes = {leg.strike for leg in legs}
            top = 3 * max(strikes)
            # Include the strikes so the sweep's extremes sit on the kinks.
            sweep = sorted(strikes | {top * i / 300 for i in range(301)})
            pnl = [payoff_at_expiry(legs, s) for s in sweep]
            assert list(summary.breakevens) == sorted(summary.breakevens)
            assert all(
                b1 - b0 > 1e-9 for b0, b1 in zip(summary.breakevens, summary.breakevens[1:])
            )
            for breakeven in summary.breakevens:
                assert payoff_at_expiry(legs, breakeven) == pytest.approx(0.0, abs=1e-6)
            for s0, s1, p0, p1 in zip(sweep, sweep[1:], pnl, pnl[1:]):
                if p0 * p1 < 0:  # every sign change is bracketed by a breakeven
                    assert any(s0 <= b <= s1 for b in summary.breakevens)
            # The internal probe point never leaks: a breakeven past the last
            # strike exists only when the far segment crosses zero out there.
            if any(b > max(strikes) for b in summary.breakevens):
                assert payoff_at_expiry(legs, max(strikes)) != 0.0
            # Positive magnitudes, clamped: a position that only wins has
            # max_loss 0.0 and one that only loses has max_profit 0.0.
            if summary.unlimited_profit:
                assert summary.max_profit is None and pnl[-1] > pnl[-2]
            else:
                assert summary.max_profit == pytest.approx(max(0.0, max(pnl)))
                assert summary.max_profit >= 0.0
            if summary.unlimited_risk:
                assert summary.max_loss is None and pnl[-1] < pnl[-2]
            else:
                assert summary.max_loss == pytest.approx(max(0.0, -min(pnl)))
                assert summary.max_loss >= 0.0
