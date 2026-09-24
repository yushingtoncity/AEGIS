"""Core option math: textbook values, identities, finite differences and
edge cases — pure functions, no live API calls."""

import math
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from aegis.data.models import ChainSnapshot, OptionSnapshot, OptionType
from aegis.pricing.black_scholes import (
    call_price,
    discounted_intrinsic,
    intrinsic_value,
    price,
    put_price,
)
from aegis.pricing.enrich import enrich_chain, enrich_option
from aegis.pricing.errors import PricingError
from aegis.pricing.greeks import Greeks, call_greeks, greeks, put_greeks
from aegis.pricing.implied_vol import implied_vol
from aegis.pricing.models import Greeks as ModelGreeks
from aegis.pricing.time_to_expiry import (
    DEFAULT_DAY_COUNT_BASIS,
    DEFAULT_EXPIRY_TIME,
    DEFAULT_EXPIRY_TIMEZONE,
    calendar_days_to_expiry,
    expiry_instant,
    time_to_expiry,
)

CALL, PUT = OptionType.CALL, OptionType.PUT

# Hull's worked example: S=42, K=40, r=10%, sigma=20%, T=6 months.
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)


class TestBlackScholes:
    def test_hull_textbook_values(self):
        assert call_price(**HULL) == pytest.approx(4.76, abs=0.01)
        assert put_price(**HULL) == pytest.approx(0.81, abs=0.01)
        assert price(CALL, q=0.0, **HULL) == pytest.approx(4.76, abs=0.01)
        assert price(PUT, q=0.0, **HULL) == pytest.approx(0.81, abs=0.01)

    def test_call_put_helpers_dispatch_through_price(self):
        args = (100, 95, 0.3, 0.02, 0.4, 0.01)
        assert call_price(*args) == price(CALL, *args)
        assert put_price(*args) == price(PUT, *args)

    def test_put_call_parity_on_seeded_grid(self):
        rng = np.random.default_rng(1234)
        for _ in range(60):
            S = rng.uniform(10, 500)
            K = rng.uniform(0.5 * S, 1.5 * S)
            T = rng.uniform(0.01, 3)
            r = rng.uniform(-0.02, 0.10)
            sigma = rng.uniform(0.05, 1.5)
            q = rng.uniform(0, 0.05)
            parity = call_price(S, K, T, r, sigma, q) - put_price(S, K, T, r, sigma, q)
            forward_gap = S * math.exp(-q * T) - K * math.exp(-r * T)
            assert parity == pytest.approx(forward_gap, abs=1e-8)

    @pytest.mark.parametrize(
        ("option_type", "S", "K", "expected"),
        [
            (CALL, 110.0, 100.0, 10.0),
            (CALL, 90.0, 100.0, 0.0),
            (PUT, 90.0, 100.0, 10.0),
            (PUT, 110.0, 100.0, 0.0),
        ],
    )
    @pytest.mark.parametrize("T", [0.0, -0.01])
    def test_expired_returns_intrinsic(self, option_type, S, K, expected, T):
        value = price(option_type, S, K, T, 0.05, 0.3)
        assert value == expected
        assert isinstance(value, float)
        assert value == intrinsic_value(option_type, S, K)
        # expiry wins even when sigma is also degenerate
        assert price(option_type, S, K, T, 0.05, 0.0) == expected

    @pytest.mark.parametrize("option_type", [CALL, PUT])
    @pytest.mark.parametrize("sigma", [0.0, -0.2])
    def test_zero_vol_returns_discounted_intrinsic(self, option_type, sigma):
        S, K, T, r, q = 110.0, 100.0, 0.5, 0.03, 0.01
        forward_spot, forward_strike = S * math.exp(-q * T), K * math.exp(-r * T)
        expected = max(forward_spot - forward_strike, 0.0)
        if option_type is PUT:
            expected = max(forward_strike - forward_spot, 0.0)
        assert price(option_type, S, K, T, r, sigma, q) == pytest.approx(expected)
        assert discounted_intrinsic(option_type, S, K, T, r, q) == pytest.approx(expected)

    def test_intrinsic_helpers_accept_a_worthless_underlying(self):
        # S = 0 is a legitimate expiry scenario for payoff diagrams.
        assert intrinsic_value(PUT, 0.0, 100.0) == 100.0
        assert intrinsic_value(CALL, 0.0, 100.0) == 0.0
        worthless_put = discounted_intrinsic(PUT, 0.0, 100.0, 1.0, 0.05)
        assert worthless_put == pytest.approx(100 * math.exp(-0.05))

    def test_price_is_finite_and_non_negative(self):
        deep_otm_call = price(CALL, 10.0, 500.0, 0.01, 0.03, 0.05)
        deep_otm_put = price(PUT, 500.0, 10.0, 0.01, 0.03, 0.05)
        assert deep_otm_call == 0.0 and deep_otm_put == 0.0
        assert math.isfinite(price(CALL, 100.0, 100.0, 1e-9, 0.03, 0.3))
        assert math.isfinite(price(PUT, 100.0, 100.0, 3.0, -0.02, 4.0))

    def test_option_type_accepts_its_string_value(self):
        assert price("call", **HULL) == call_price(**HULL)
        with pytest.raises(PricingError):
            price("straddle", **HULL)

    @pytest.mark.parametrize(
        ("S", "K"),
        [
            (0.0, 100.0),
            (-1.0, 100.0),
            (100.0, -5.0),
            (100.0, 0.0),
            (math.nan, 100.0),
            (math.inf, 100.0),
            (100.0, math.nan),
        ],
    )
    def test_bad_spot_or_strike_raises(self, S, K):
        with pytest.raises(PricingError, match="Black-Scholes price"):
            price(CALL, S, K, 0.5, 0.03, 0.3)
        with pytest.raises(PricingError):
            put_price(S, K, 0.5, 0.03, 0.3)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_parameters_raise_instead_of_nan(self, bad):
        for kwargs in ({"T": bad}, {"sigma": bad}, {"r": bad}, {"q": bad}):
            params = {"T": 0.5, "r": 0.03, "sigma": 0.3, "q": 0.0} | kwargs
            with pytest.raises(PricingError):
                price(CALL, 100.0, 100.0, **params)

    def test_extreme_magnitudes_take_the_limits_or_raise_pricing_error(self):
        # S / K underflows to 0.0: the call is worthless, the put is the strike.
        assert price(CALL, 1e-200, 1e200, 0.5, 0.05, 0.2) == 0.0
        assert price(PUT, 1e-200, 1e200, 0.5, 0.05, 0.2) == pytest.approx(1e200 * math.exp(-0.025))
        # sigma sqrt(T) underflows to 0.0 although both are positive.
        assert price(CALL, 100.0, 100.0, 1e-250, 0.05, 1e-200) == discounted_intrinsic(
            CALL, 100.0, 100.0, 1e-250, 0.05
        )
        # A discount factor beyond float range is malformed input, not a crash.
        with pytest.raises(PricingError, match="Black-Scholes price"):
            price(PUT, 100.0, 100.0, 1e300, -0.1, 0.2)
        with pytest.raises(PricingError, match="discounted intrinsic"):
            discounted_intrinsic(PUT, 100.0, 100.0, 1e300, -0.1)


class TestImpliedVol:
    @pytest.mark.parametrize(
        ("option_type", "S", "K", "T", "r", "q"),
        [
            (CALL, 100.0, 100.0, 0.5, 0.03, 0.0),
            (PUT, 100.0, 110.0, 0.5, 0.03, 0.0),
            (CALL, 100.0, 95.0, 1.0, 0.03, 0.02),
            (PUT, 250.0, 240.0, 0.08, 0.04, 0.015),
        ],
    )
    def test_round_trip_recovers_sigma(self, option_type, S, K, T, r, q):
        target = price(option_type, S, K, T, r, 0.25, q)
        assert implied_vol(option_type, target, S, K, T, r, q) == pytest.approx(0.25, abs=1e-6)

    def test_none_when_no_solution(self):
        S, K, T, r = 110.0, 100.0, 0.5, 0.03
        floor = discounted_intrinsic(CALL, S, K, T, r)
        assert implied_vol(CALL, floor - 1.0, S, K, T, r) is None  # below intrinsic
        assert implied_vol(CALL, S + 10.0, S, K, T, r) is None  # above the upper bound
        assert implied_vol(CALL, 12.0, S, K, 0.0, r) is None  # expired
        assert implied_vol(CALL, 12.0, S, K, -0.5, r) is None
        # Expired AND quoted exactly at intrinsic: the bracket sign test alone
        # would pass this through as sigma = lo.
        assert implied_vol(CALL, intrinsic_value(CALL, S, K), S, K, 0.0, r) is None
        assert implied_vol(CALL, intrinsic_value(CALL, S, K), S, K, -0.5, r) is None
        assert implied_vol(PUT, 0.0, S, K, 0.0, r) is None
        assert implied_vol(CALL, 1.0, 1e-200, 1e200, T, r) is None  # S / K underflows
        assert implied_vol(CALL, -1.0, S, K, T, r) is None  # not a price
        assert implied_vol(CALL, None, S, K, T, r) is None
        assert implied_vol(CALL, math.nan, S, K, T, r) is None
        assert implied_vol(PUT, math.inf, S, K, T, r) is None

    @pytest.mark.parametrize(
        ("option_type", "S", "K", "T", "r"),
        [
            (CALL, 100.0, 150.0, 0.01, 0.03),
            (PUT, 638.9, 500.0, (2 + 15 / 1440) / 365, 0.04),  # the no-greeks fixture
            (CALL, 90.0, 100.0, 0.5, 0.03),
        ],
    )
    def test_zero_quote_on_a_deep_otm_contract_has_no_iv(self, option_type, S, K, T, r):
        # price(sigma=lo) underflows to exactly 0.0 here, so a 0.00 quote
        # would satisfy the bracket's lo endpoint; it carries no volatility
        # information (every sigma up to some threshold prices to 0.0).
        assert price(option_type, S, K, T, r, 1e-4) == 0.0
        assert implied_vol(option_type, 0.0, S, K, T, r) is None

    def test_bracket_endpoints_are_returned_exactly(self):
        S, K, T, r = 100.0, 100.0, 0.5, 0.03
        at_lo = price(CALL, S, K, T, r, 1e-4)
        at_hi = price(PUT, S, K, T, r, 5.0)
        assert implied_vol(CALL, at_lo, S, K, T, r) == 1e-4
        assert implied_vol(PUT, at_hi, S, K, T, r) == 5.0
        # a custom bracket changes what counts as "out of range"
        assert implied_vol(PUT, at_hi, S, K, T, r, hi=1.0) is None
        low_vol = price(CALL, S, K, T, r, 0.05)
        assert implied_vol(CALL, low_vol, S, K, T, r, lo=0.1) is None  # below a raised floor
        # An int bracket must not leak an int out of a float-typed API.
        at_one = price(CALL, S, K, T, r, 1.0)
        for iv in (
            implied_vol(CALL, at_one, S, K, T, r, hi=1),
            implied_vol(CALL, at_one, S, K, T, r, lo=1, hi=5),
        ):
            assert iv == 1.0 and type(iv) is float

    def test_result_is_a_plain_float(self):
        target = price(CALL, 100.0, 100.0, 0.5, 0.03, 0.4)
        iv = implied_vol(CALL, target, 100.0, 100.0, 0.5, 0.03)
        assert type(iv) is float

    def test_tol_reaches_the_solver(self, monkeypatch):
        from scipy.optimize import brentq

        seen = {}

        def spy(f, a, b, **kwargs):
            seen.update(kwargs)
            return brentq(f, a, b, **kwargs)

        monkeypatch.setattr("aegis.pricing.implied_vol.brentq", spy)
        target = price(CALL, 100.0, 100.0, 0.5, 0.03, 0.25)
        implied_vol(CALL, target, 100.0, 100.0, 0.5, 0.03, tol=1e-3)
        assert seen["xtol"] == 1e-3
        # A loose tolerance visibly stops early; a tight one recovers sigma.
        loose = implied_vol(CALL, target, 100.0, 100.0, 0.5, 0.03, tol=0.5)
        assert 1e-6 < abs(loose - 0.25) <= 0.5
        tight = implied_vol(CALL, target, 100.0, 100.0, 0.5, 0.03, tol=1e-12)
        assert tight == pytest.approx(0.25, abs=1e-9)

    @pytest.mark.parametrize(("quote", "T"), [(5.0, 0.0), (5.0, 0.5), (-1.0, 0.5)])
    def test_bad_option_type_raises_its_own_error(self, quote, T):
        # T = 0 and a negative quote would otherwise short-circuit to None
        # before the pricer ever sees the type; with T > 0 the error must
        # name implied volatility, not the Black-Scholes price underneath.
        with pytest.raises(PricingError, match="implied volatility"):
            implied_vol("straddle", quote, 100.0, 100.0, T, 0.03)

    @pytest.mark.parametrize(("S", "K"), [(0.0, 100.0), (100.0, -5.0), (math.nan, 100.0)])
    def test_bad_spot_or_strike_raises(self, S, K):
        with pytest.raises(PricingError, match="implied volatility"):
            implied_vol(CALL, 5.0, S, K, 0.5, 0.03)


class TestGreeks:
    ATM = dict(S=100.0, K=100.0, T=0.5, r=0.03, sigma=0.3)

    def test_greeks_model_is_reexported(self):
        assert Greeks is ModelGreeks

    def test_sanity_at_the_money(self):
        call, put = call_greeks(**self.ATM), put_greeks(**self.ATM)
        assert 0.0 < call.delta < 1.0
        assert -1.0 < put.delta < 0.0
        assert call.gamma > 0 and call.vega > 0
        assert call.gamma == pytest.approx(put.gamma, abs=1e-12)
        assert call.vega == pytest.approx(put.vega, abs=1e-12)
        assert call.theta < 0 and put.theta < 0  # long options bleed
        assert call.rho > 0 and put.rho < 0
        assert call.delta - put.delta == pytest.approx(1.0, abs=1e-12)  # e^(-qT) with q=0

    def test_delta_difference_is_the_dividend_discount(self):
        q = 0.02
        call = greeks(CALL, q=q, **self.ATM)
        put = greeks(PUT, q=q, **self.ATM)
        assert call.delta - put.delta == pytest.approx(math.exp(-q * self.ATM["T"]), abs=1e-12)

    def test_helpers_match_generic_entry_point(self):
        assert call_greeks(**self.ATM) == greeks(CALL, **self.ATM)
        assert put_greeks(**self.ATM) == greeks(PUT, **self.ATM)
        assert call_greeks(day_count_basis=360, **self.ATM) == greeks(
            CALL, day_count_basis=360, **self.ATM
        )
        assert put_greeks(day_count_basis=360, **self.ATM) == greeks(
            PUT, day_count_basis=360, **self.ATM
        )

    def test_units_are_per_day_per_point(self):
        # theta per calendar day scales with the basis; vega/rho are per 0.01.
        g365 = greeks(CALL, **self.ATM)
        g360 = greeks(CALL, day_count_basis=360, **self.ATM)
        assert g360.theta == pytest.approx(g365.theta * 365 / 360)
        assert g360.delta == g365.delta and g360.vega == g365.vega and g360.rho == g365.rho
        S, K, T, r, sigma = (self.ATM[k] for k in ("S", "K", "T", "r", "sigma"))
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        raw_vega = S * math.exp(-(d1**2) / 2) / math.sqrt(2 * math.pi) * math.sqrt(T)
        assert g365.vega == pytest.approx(raw_vega / 100)

    @pytest.mark.parametrize(
        ("option_type", "S", "K", "delta"),
        [
            (CALL, 110.0, 100.0, 1.0),
            (CALL, 90.0, 100.0, 0.0),
            (CALL, 100.0, 100.0, 0.5),
            (PUT, 90.0, 100.0, -1.0),
            (PUT, 110.0, 100.0, 0.0),
            (PUT, 100.0, 100.0, -0.5),
        ],
    )
    def test_expired_limits(self, option_type, S, K, delta):
        for T in (0.0, -1.0):
            g = greeks(option_type, S, K, T, 0.03, 0.3)
            assert g == Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    def test_zero_vol_forward_in_the_money(self):
        S, K, T, r, q = 110.0, 100.0, 0.5, 0.03, 0.01
        disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
        call = greeks(CALL, S, K, T, r, 0.0, q)
        assert call.delta == pytest.approx(disc_q)
        assert call.gamma == 0.0 and call.vega == 0.0
        assert call.theta == pytest.approx((q * S * disc_q - r * K * disc_r) / 365)
        assert call.rho == pytest.approx(K * T * disc_r / 100)
        put = greeks(PUT, K, S, T, r, 0.0, q)  # mirror: S=100 against K=110
        assert put.delta == pytest.approx(-disc_q)
        assert put.theta == pytest.approx((r * S * disc_r - q * K * disc_q) / 365)
        assert put.rho == pytest.approx(-S * T * disc_r / 100)
        # The per-day scaling honours the basis in this branch too.
        g360 = greeks(CALL, S, K, T, r, 0.0, q, day_count_basis=360)
        assert g360.theta == pytest.approx((q * S * disc_q - r * K * disc_r) / 360)
        assert g360.rho == call.rho and g360.delta == call.delta

    def test_zero_vol_forward_out_of_the_money_is_flat(self):
        flat = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        assert greeks(CALL, 90.0, 100.0, 0.5, 0.03, 0.0) == flat
        assert greeks(PUT, 110.0, 100.0, 0.5, 0.03, 0.0) == flat

    def test_zero_vol_branch_compares_the_forward_not_spot(self):
        # Spot is below the strike but the forward is above it:
        # 100 > 100.5 e^(-0.05), so the call is in the money.
        call = greeks(CALL, 100.0, 100.5, 1.0, 0.05, 0.0)
        assert call.delta == 1.0
        assert call.rho == pytest.approx(100.5 * math.exp(-0.05) / 100)
        flat = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        assert greeks(PUT, 100.0, 100.5, 1.0, 0.05, 0.0) == flat
        # A dividend yield pulls the forward below the strike although S > K.
        put = greeks(PUT, 100.0, 99.5, 1.0, 0.0, 0.0, 0.05)
        assert put.delta == pytest.approx(-math.exp(-0.05))
        assert greeks(CALL, 100.0, 99.5, 1.0, 0.0, 0.0, 0.05) == flat

    @pytest.mark.parametrize(
        ("S", "K", "T", "sigma"),
        [
            (100.0, 100.0, 1e-9, 0.3),
            (100.0, 150.0, 1e-6, 0.3),
            (100.0, 100.0, 3.0, 5.0),
            (1.0, 1000.0, 0.5, 0.01),
            (1e-200, 1e200, 0.5, 0.2),  # S / K underflows to 0.0
            (100.0, 100.0, 1e-250, 1e-200),  # sigma sqrt(T) underflows to 0.0
            (1e-305, 100.0, 1e-20, 1e-10),  # S sigma sqrt(T) underflows, sigma sqrt(T) does not
        ],
    )
    def test_never_nan_at_extremes(self, S, K, T, sigma):
        for option_type in (CALL, PUT):
            g = greeks(option_type, S, K, T, 0.03, sigma)
            assert all(math.isfinite(v) for v in (g.delta, g.gamma, g.theta, g.vega, g.rho))

    @pytest.mark.parametrize("sigma", [0.2, 0.0])
    def test_overflowing_discount_factor_raises_pricing_error(self, sigma):
        with pytest.raises(PricingError, match="Black-Scholes greeks"):
            greeks(PUT, 100.0, 100.0, 1e300, -0.1, sigma)

    @pytest.mark.parametrize("sigma", [0.2, 0.0])
    def test_overflowing_forward_raises_pricing_error(self, sigma):
        # The discount factors are modest but S e^(-qT) / K e^(-rT) is not:
        # vega and theta would come out as inf * 0 = NaN (or -inf at zero
        # vol), so there is no finite answer to return.
        for option_type in (CALL, PUT):
            with pytest.raises(PricingError, match="Black-Scholes greeks"):
                greeks(option_type, 1e308, 100.0, 1.0, 0.0, sigma, -1.0)
            with pytest.raises(PricingError, match="Black-Scholes greeks"):
                greeks(option_type, 100.0, 1e308, 1.0, -1.0, sigma)

    @pytest.mark.parametrize("sigma", [0.3, 0.0])
    def test_rho_survives_k_times_t_overflowing(self, sigma):
        # With r = 0 the discount factor is exactly 1, so it cannot flag the
        # overflow: K T alone exceeds a float and must not become inf * 0.
        for T in (1.8e306, 1e308):
            g = greeks(CALL, 100.0, 100.0, T, 0.0, sigma)
            assert all(math.isfinite(v) for v in (g.delta, g.gamma, g.theta, g.vega, g.rho))
            assert g.rho == 0.0  # the call's N(d2) tail is exactly 0 out there
        # The put's tail is 1, so the product truly overflows: no finite answer.
        with pytest.raises(PricingError, match="Black-Scholes greeks"):
            greeks(PUT, 1.0, 1e308, 10.0, 0.0, sigma)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_parameters_raise_instead_of_nan(self, bad):
        for kwargs in ({"T": bad}, {"sigma": bad}, {"r": bad}, {"q": bad}):
            params = {"T": 0.5, "r": 0.03, "sigma": 0.3, "q": 0.0} | kwargs
            with pytest.raises(PricingError, match="Black-Scholes greeks"):
                greeks(CALL, 100.0, 100.0, **params)

    @pytest.mark.parametrize("basis", [0, -365])
    @pytest.mark.parametrize("sigma", [0.3, 0.0])
    def test_non_positive_day_count_basis_raises(self, basis, sigma):
        with pytest.raises(PricingError, match="Black-Scholes greeks"):
            greeks(CALL, 100.0, 100.0, 0.5, 0.03, sigma, day_count_basis=basis)
        with pytest.raises(PricingError, match="Black-Scholes greeks"):
            greeks(CALL, 110.0, 100.0, 0.0, 0.03, sigma, day_count_basis=basis)  # expired too

    @pytest.mark.parametrize(("S", "K"), [(0.0, 100.0), (100.0, -5.0), (math.nan, 100.0)])
    def test_bad_spot_or_strike_raises(self, S, K):
        with pytest.raises(PricingError, match="Black-Scholes greeks"):
            greeks(CALL, S, K, 0.5, 0.03, 0.3)


def _numeric_greeks(
    option_type, S, K, T, r, sigma, q, *, h_s=1e-2, h_v=1e-4, h_t=1e-4, h_r=1e-4
):
    """Central differences of price() in the Greeks model's units."""

    def p(S=S, T=T, r=r, sigma=sigma):
        return price(option_type, S, K, T, r, sigma, q)

    return Greeks(
        delta=(p(S=S + h_s) - p(S=S - h_s)) / (2 * h_s),
        gamma=(p(S=S + h_s) - 2 * p() + p(S=S - h_s)) / h_s**2,
        theta=-(p(T=T + h_t) - p(T=T - h_t)) / (2 * h_t) / 365,
        vega=(p(sigma=sigma + h_v) - p(sigma=sigma - h_v)) / (2 * h_v) / 100,
        rho=(p(r=r + h_r) - p(r=r - h_r)) / (2 * h_r) / 100,
    )


class TestFiniteDifferences:
    @pytest.mark.parametrize(
        ("S", "K", "T", "q"),
        [
            (100.0, 100.0, 0.5, 0.0),  # at the money
            (120.0, 100.0, 0.25, 0.0),  # ITM call / OTM put
            (80.0, 100.0, 1.0, 0.02),  # OTM call / ITM put, dividend payer
            (100.0, 90.0, 0.1, 0.03),  # short-dated with yield
            (50.0, 55.0, 2.0, 0.01),  # long-dated
        ],
    )
    @pytest.mark.parametrize("option_type", [CALL, PUT])
    def test_analytic_matches_central_differences(self, option_type, S, K, T, q):
        r, sigma = 0.03, 0.3
        analytic = greeks(option_type, S, K, T, r, sigma, q)
        numeric = _numeric_greeks(option_type, S, K, T, r, sigma, q)
        for name in ("delta", "gamma", "theta", "vega", "rho"):
            expected = pytest.approx(getattr(analytic, name), rel=1e-4, abs=1e-6)
            assert getattr(numeric, name) == expected, name


class TestTimeToExpiry:
    EXPIRY = date(2026, 8, 21)
    WEEK_BEFORE = datetime(2026, 8, 14, 16, 0, tzinfo=ZoneInfo("America/New_York"))

    def test_defaults(self):
        assert (DEFAULT_EXPIRY_TIME, DEFAULT_EXPIRY_TIMEZONE, DEFAULT_DAY_COUNT_BASIS) == (
            "16:00", "America/New_York", 365,
        )

    def test_known_pair_is_exactly_seven_days(self):
        assert time_to_expiry(self.EXPIRY, self.WEEK_BEFORE) == 7 / 365
        assert calendar_days_to_expiry(self.EXPIRY, self.WEEK_BEFORE) == 7.0

    def test_after_expiry_clamps_to_zero(self):
        later = datetime(2026, 8, 21, 20, 0, 1, tzinfo=timezone.utc)
        assert time_to_expiry(self.EXPIRY, later) == 0.0
        assert calendar_days_to_expiry(self.EXPIRY, later) == 0.0
        assert time_to_expiry(self.EXPIRY, datetime(2026, 9, 1, tzinfo=timezone.utc)) == 0.0

    def test_day_count_basis_scales_proportionally(self):
        on_360 = time_to_expiry(self.EXPIRY, self.WEEK_BEFORE, day_count_basis=360)
        on_365 = time_to_expiry(self.EXPIRY, self.WEEK_BEFORE)
        assert on_360 == 7 / 360
        assert on_360 / on_365 == pytest.approx(365 / 360)

    @pytest.mark.parametrize("basis", [0, -365])
    def test_non_positive_day_count_basis_raises(self, basis):
        with pytest.raises(PricingError, match="time to expiry"):
            time_to_expiry(self.EXPIRY, self.WEEK_BEFORE, day_count_basis=basis)

    def test_expiry_instant_honours_dst(self):
        summer = expiry_instant(date(2026, 8, 21))  # EDT, UTC-4
        winter = expiry_instant(date(2026, 12, 18))  # EST, UTC-5
        assert summer == datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)
        assert winter == datetime(2026, 12, 18, 21, 0, tzinfo=timezone.utc)
        assert summer.tzinfo is timezone.utc

    def test_naive_now_is_treated_as_utc(self, monkeypatch):
        # Pin a non-UTC local zone so "naive means UTC" and "naive means
        # local time" differ wherever the suite runs (CI boxes are UTC).
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        time.tzset()
        try:
            naive = datetime(2026, 8, 21, 19, 0)  # one hour before the 20:00 UTC close
            assert time_to_expiry(self.EXPIRY, naive) == 3600 / (86400 * 365)
            assert time_to_expiry(self.EXPIRY, naive) == time_to_expiry(
                self.EXPIRY, naive.replace(tzinfo=timezone.utc)
            )
        finally:
            monkeypatch.undo()
            time.tzset()

    def test_fractional_days_survive(self):
        # 0-3 DTE contracts keep their hours instead of rounding to zero.
        morning_of = datetime(2026, 8, 21, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert calendar_days_to_expiry(self.EXPIRY, morning_of) == pytest.approx(6.5 / 24)
        assert time_to_expiry(self.EXPIRY, morning_of) > 0.0

    def test_custom_close_time_and_zone(self):
        chicago_close = expiry_instant(
            self.EXPIRY, expiry_time="15:00", expiry_timezone="America/Chicago"
        )
        assert chicago_close == expiry_instant(self.EXPIRY)
        early = expiry_instant(self.EXPIRY, expiry_time="13:00")
        assert early == datetime(2026, 8, 21, 17, 0, tzinfo=timezone.utc)
        assert time_to_expiry(self.EXPIRY, self.WEEK_BEFORE, expiry_time="13:00") < 7 / 365
        assert calendar_days_to_expiry(self.EXPIRY, self.WEEK_BEFORE, expiry_time="13:00") == 6.875
        # 16:00 UTC is four hours before 16:00 EDT.
        assert time_to_expiry(self.EXPIRY, self.WEEK_BEFORE, expiry_timezone="UTC") == (
            (7 * 86400 - 4 * 3600) / (86400 * 365)
        )
        assert calendar_days_to_expiry(
            self.EXPIRY, self.WEEK_BEFORE, expiry_timezone="UTC"
        ) == pytest.approx(7 - 4 / 24)

    def test_now_defaults_to_the_current_time(self):
        assert time_to_expiry(date(2099, 1, 15)) > 0.0
        assert time_to_expiry(date(2000, 1, 21)) == 0.0

    def test_default_clock_is_utc(self, monkeypatch):
        # A fake clock whose tz-less reading is a different wall time, so
        # "local time stamped as UTC" gives a wrong number, not a crash.
        fixed, reads = datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc), []

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                reads.append(tz)
                if tz is None:
                    return fixed.astimezone(ZoneInfo("Asia/Tokyo")).replace(tzinfo=None)
                return fixed.astimezone(tz)

        monkeypatch.setattr("aegis.pricing.time_to_expiry.datetime", Clock)
        # 20:00 UTC on 8/14 -> the 20:00 UTC close on 8/21: exactly a week.
        assert time_to_expiry(self.EXPIRY) == 7 / 365
        assert calendar_days_to_expiry(self.EXPIRY) == 7.0
        assert reads and all(tz is timezone.utc for tz in reads)


class TestEnrich:
    # Fixture quotes are as of 2026-07-30 19:45 UTC with SPY around 638.9.
    NOW = datetime(2026, 7, 30, 19, 45, tzinfo=timezone.utc)
    SPOT = 638.9
    RATE = 0.04
    # 16:00 ET on 2026-08-21 is 20:00 UTC: 22 days and 15 minutes after NOW.
    T_FULL = (22 + 15 / 1440) / 365

    @pytest.fixture
    def full(self, fixture):
        data = fixture("option_snapshot_full.json")
        return OptionSnapshot.from_alpaca(data["symbol"], "SPY", data)

    @pytest.fixture
    def no_greeks(self, fixture):
        data = fixture("option_snapshot_no_greeks.json")
        return OptionSnapshot.from_alpaca(data["symbol"], "SPY", data)

    def enrich(self, snapshot, **kwargs):
        return enrich_option(snapshot, self.SPOT, self.RATE, now=self.NOW, **kwargs)

    def test_vendor_values_pass_through_unchanged_and_win(self, full):
        e = self.enrich(full)
        assert e.snapshot == full
        assert (e.spot, e.rate, e.dividend_yield) == (self.SPOT, self.RATE, 0.0)
        assert e.vendor_implied_vol == 0.1845
        assert e.vendor_greeks == Greeks(
            delta=0.5321, gamma=0.0123, theta=-0.0891, vega=0.7421, rho=0.21
        )
        assert e.implied_vol == 0.1845 and e.iv_source == "vendor"
        assert e.greeks == e.vendor_greeks and e.greeks_source == "vendor"

    def test_put_vendor_greeks_pass_through_unchanged(self, no_greeks):
        # The only full-Greeks fixture is a call; a put's negative delta must
        # come through untouched too — nothing "normalises" the vendor path.
        vendor = dict(delta=-0.0312, gamma=0.0021, theta=-0.0177, vega=0.0402, rho=-0.0055)
        put = no_greeks.model_copy(update={"implied_vol": 0.45, **vendor})
        e = self.enrich(put)
        assert e.vendor_greeks == Greeks(**vendor)
        assert e.greeks == e.vendor_greeks and e.greeks_source == "vendor"
        assert e.vendor_implied_vol == 0.45 and e.iv_source == "vendor"

    def test_time_to_expiry_follows_the_calendar_day_convention(self, full):
        e = self.enrich(full)
        assert e.time_to_expiry == pytest.approx(self.T_FULL)
        assert e.time_to_expiry == time_to_expiry(date(2026, 8, 21), self.NOW)
        on_360 = self.enrich(full, day_count_basis=360)
        assert on_360.time_to_expiry == pytest.approx(self.T_FULL * 365 / 360)
        # The basis reaches the model Greeks' per-day theta, not just T.
        assert on_360.model_greeks.theta == pytest.approx(
            greeks(
                CALL, self.SPOT, 640.0, on_360.time_to_expiry, self.RATE,
                on_360.model_implied_vol, day_count_basis=360,
            ).theta
        )
        assert on_360.model_greeks.theta != pytest.approx(e.model_greeks.theta)
        early_close = self.enrich(full, expiry_time="13:00")
        assert early_close.time_to_expiry < e.time_to_expiry
        utc_close = self.enrich(full, expiry_timezone="UTC")
        assert utc_close.time_to_expiry == pytest.approx(
            time_to_expiry(date(2026, 8, 21), self.NOW, expiry_timezone="UTC")
        )
        assert utc_close.time_to_expiry < e.time_to_expiry  # 16:00 UTC is 4h before 16:00 EDT

    def test_model_iv_is_solved_from_the_mid(self, full):
        e = self.enrich(full)
        assert e.price_used == pytest.approx(12.20)  # mid, not last (12.2 by coincidence)
        stale = full.model_copy(update={"last": 11.0})
        assert self.enrich(stale).price_used == pytest.approx(12.20)  # mid wins over a different last
        assert e.model_implied_vol is not None and e.model_implied_vol > 0
        # The fixture quote is synthetic, so only loose agreement with the
        # vendor IV is meaningful; the round trip through price() is exact.
        assert e.model_implied_vol == pytest.approx(0.1845, abs=0.05)
        reproduced = price(CALL, self.SPOT, 640.0, e.time_to_expiry, self.RATE, e.model_implied_vol)
        assert reproduced == pytest.approx(12.20, abs=1e-6)

    def test_model_price_at_vendor_iv_is_near_the_quote(self, full):
        e = self.enrich(full)
        at_vendor_iv = price(CALL, self.SPOT, 640.0, e.time_to_expiry, self.RATE, 0.1845)
        assert e.model_price == pytest.approx(at_vendor_iv)
        assert abs(e.model_price - full.mid) < 2.0

    def test_model_greeks_sit_beside_vendor_greeks(self, full):
        e = self.enrich(full)
        expected = greeks(CALL, self.SPOT, 640.0, e.time_to_expiry, self.RATE, e.model_implied_vol)
        assert e.model_greeks == expected
        assert e.model_greeks.delta == pytest.approx(full.delta, abs=0.05)
        assert e.model_greeks.gamma > 0 and e.model_greeks.vega > 0
        assert e.model_greeks.theta < 0 and e.model_greeks.rho > 0
        assert e.model_greeks != e.vendor_greeks  # kept apart, never merged

    def test_explicit_price_overrides_the_mid(self, full):
        base = self.enrich(full)
        e = self.enrich(full, price=13.0)
        assert e.price_used == 13.0
        assert e.model_implied_vol > base.model_implied_vol  # dearer option, higher IV
        assert e.model_price == base.model_price  # the vendor-IV price ignores it
        # Given is given, even when it is 0.00: no falling through to the mid.
        zero = self.enrich(full, price=0.0)
        assert zero.price_used == 0.0
        assert zero.model_implied_vol is None and zero.iv_source == "vendor"
        assert zero.model_greeks == greeks(
            CALL, self.SPOT, 640.0, zero.time_to_expiry, self.RATE, 0.1845
        )

    def test_last_trade_is_the_weaker_fallback_price(self, full):
        no_quote = full.model_copy(update={"bid": None, "ask": None})
        e = self.enrich(no_quote)
        assert e.price_used == 12.2  # last
        assert e.model_implied_vol is not None

        no_price = no_quote.model_copy(update={"last": None})
        e = self.enrich(no_price)
        assert e.price_used is None and e.model_implied_vol is None
        # Greeks still come out, at the vendor IV.
        assert e.model_greeks == greeks(CALL, self.SPOT, 640.0, e.time_to_expiry, self.RATE, 0.1845)
        assert e.model_price is not None

    def test_dividend_yield_flows_into_every_model_value(self, full):
        plain, with_yield = self.enrich(full), self.enrich(full, dividend_yield=0.012)
        assert with_yield.dividend_yield == 0.012
        assert with_yield.model_price < plain.model_price  # a yield cheapens a call
        assert with_yield.model_greeks.delta < plain.model_greeks.delta
        assert with_yield.model_implied_vol != plain.model_implied_vol

    def test_no_greeks_fixture_falls_back_to_the_model(self, no_greeks):
        e = self.enrich(no_greeks)
        assert e.vendor_implied_vol is None and e.vendor_greeks is None
        assert e.model_price is None  # nothing to price at without a vendor IV
        assert e.price_used == pytest.approx(0.025)
        assert e.time_to_expiry == pytest.approx((2 + 15 / 1440) / 365)
        # A 2-DTE put 140 points OTM quoted at 2.5 cents solves, at a high IV.
        assert e.model_implied_vol is not None
        assert 0.5 < e.model_implied_vol < 5.0
        assert e.iv_source == "model" and e.implied_vol == e.model_implied_vol
        assert e.model_greeks is not None and e.greeks_source == "model"
        assert -0.05 < e.model_greeks.delta < 0.0
        assert e.model_greeks.theta < 0

    def test_zero_quote_never_fabricates_an_iv(self, no_greeks):
        # bid 0 / ask 0 is the common deep-OTM quote: no IV, no Greeks, no
        # sources — never sigma = 1e-4 with all-zero Greeks.
        e = self.enrich(no_greeks.model_copy(update={"ask": 0.0}))
        assert e.price_used == 0.0
        assert e.model_implied_vol is None and e.iv_source is None
        assert e.model_greeks is None and e.greeks_source is None
        # With a vendor IV the Greeks come out at THAT vol, not at the floor.
        with_iv = no_greeks.model_copy(update={"ask": 0.0, "implied_vol": 0.45})
        e = self.enrich(with_iv)
        assert e.model_implied_vol is None and e.iv_source == "vendor"
        assert e.greeks_source == "model"
        assert e.model_greeks == greeks(PUT, self.SPOT, 500.0, e.time_to_expiry, self.RATE, 0.45)
        assert e.model_greeks.delta < 0 and e.model_greeks.vega > 0

    @pytest.mark.parametrize("basis", [0, -365])
    def test_non_positive_day_count_basis_raises(self, full, basis):
        with pytest.raises(PricingError, match="enrich for SPY260821C00640000"):
            self.enrich(full, day_count_basis=basis)
        # Also when nothing would be priced: the check is up front, not sporadic.
        with pytest.raises(PricingError, match="enrich"):
            self.enrich(full.model_copy(update={"expiration": None}), day_count_basis=basis)
        chain = ChainSnapshot(
            underlying="SPY", expiration=date(2026, 8, 21), spot=self.SPOT, contracts=[]
        )
        with pytest.raises(PricingError, match="enrich for SPY"):
            enrich_chain(chain, self.RATE, now=self.NOW, day_count_basis=basis)

    def test_vendor_iv_without_greeks_still_gets_model_greeks(self, full):
        iv_only = full.model_copy(
            update={"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
        )
        e = self.enrich(iv_only)
        assert e.vendor_greeks is None
        assert e.iv_source == "vendor"
        assert e.greeks_source == "model" and e.model_greeks is not None

    def test_partial_vendor_greeks_count_as_missing(self, full):
        e = self.enrich(full.model_copy(update={"rho": None}))
        assert e.vendor_greeks is None
        assert e.greeks_source == "model"

    def test_non_finite_vendor_iv_passes_through_but_is_never_priced(self, full):
        # A NaN in the feed must not abort the whole chain: it is kept as the
        # raw vendor value, and the model side simply has no vendor IV.
        e = self.enrich(full.model_copy(update={"implied_vol": math.nan}))
        assert math.isnan(e.vendor_implied_vol)
        assert e.model_price is None
        assert e.model_implied_vol is not None
        assert e.model_greeks == greeks(
            CALL, self.SPOT, 640.0, e.time_to_expiry, self.RATE, e.model_implied_vol
        )
        no_quote = full.model_copy(
            update={"implied_vol": math.inf, "bid": None, "ask": None, "last": None}
        )
        e = self.enrich(no_quote)
        assert (e.model_implied_vol, e.model_greeks, e.model_price) == (None, None, None)

    def test_missing_contract_fields_propagate_as_none(self, full, fixture):
        # An unparseable symbol leaves strike, type and expiration all None.
        data = fixture("option_snapshot_full.json")
        e = self.enrich(OptionSnapshot.from_alpaca("WEIRD", "SPY", data))
        assert e.time_to_expiry is None
        assert e.model_implied_vol is None and e.model_greeks is None and e.model_price is None
        assert e.price_used == pytest.approx(12.20)
        assert e.vendor_implied_vol == 0.1845 and e.iv_source == "vendor"
        assert e.greeks_source == "vendor"

        for missing in ("strike", "option_type", "expiration"):
            e = self.enrich(full.model_copy(update={missing: None}))
            assert (e.model_implied_vol, e.model_greeks, e.model_price) == (None, None, None), missing
            assert e.iv_source == "vendor"

    def test_expired_contract_takes_the_expiry_limits(self, full):
        e = enrich_option(full, self.SPOT, self.RATE, now=datetime(2026, 9, 1, tzinfo=timezone.utc))
        assert e.time_to_expiry == 0.0
        assert e.model_implied_vol is None  # an expired option has no volatility
        # 638.9 < 640: the call expires out of the money
        assert e.model_greeks == Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        assert e.model_price == 0.0

    def test_now_defaults_to_the_clock(self, full):
        long_gone = full.model_copy(update={"expiration": date(2000, 1, 21)})
        assert enrich_option(long_gone, self.SPOT, self.RATE).time_to_expiry == 0.0

    def test_enrich_chain_reads_one_utc_clock(self, full, monkeypatch):
        reads, base = [], self.NOW

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                reads.append(tz)
                return base + timedelta(hours=len(reads))  # a second read would move T

        monkeypatch.setattr("aegis.pricing.enrich.datetime", Clock)
        chain = ChainSnapshot(
            underlying="SPY", expiration=date(2026, 8, 21), spot=self.SPOT, contracts=[full] * 3
        )
        enriched = enrich_chain(chain, self.RATE)
        assert reads == [timezone.utc]  # one read, and an aware UTC one
        assert len({e.time_to_expiry for e in enriched}) == 1
        assert enriched[0].time_to_expiry == time_to_expiry(
            date(2026, 8, 21), base + timedelta(hours=1)
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_bad_spot_raises(self, full, bad):
        with pytest.raises(PricingError, match="enrich for SPY260821C00640000"):
            enrich_option(full, bad, self.RATE, now=self.NOW)

    def test_bad_rate_or_strike_raises(self, full):
        with pytest.raises(PricingError, match="enrich"):
            enrich_option(full, self.SPOT, math.nan, now=self.NOW)
        with pytest.raises(PricingError, match="enrich"):
            self.enrich(full, dividend_yield=math.inf)
        with pytest.raises(PricingError, match="enrich"):
            self.enrich(full.model_copy(update={"strike": 0.0}))

    def test_enrich_chain_uses_chain_spot(self, full, no_greeks):
        chain = ChainSnapshot(
            underlying="SPY",
            expiration=date(2026, 8, 21),
            spot=self.SPOT,
            contracts=[full, no_greeks],
        )
        enriched = enrich_chain(chain, self.RATE, now=self.NOW)
        assert len(enriched) == 2
        assert [e.snapshot for e in enriched] == [full, no_greeks]
        assert all(e.spot == self.SPOT for e in enriched)
        assert [e.iv_source for e in enriched] == ["vendor", "model"]
        assert enriched[0] == self.enrich(full)
        assert enriched[1] == self.enrich(no_greeks)

    def test_enrich_chain_forwards_every_keyword(self, full):
        chain = ChainSnapshot(
            underlying="SPY", expiration=date(2026, 8, 21), spot=self.SPOT, contracts=[full]
        )
        kw = dict(
            now=self.NOW,
            dividend_yield=0.012,
            price=13.0,
            day_count_basis=360,
            expiry_time="13:00",
            expiry_timezone="UTC",
        )
        [e] = enrich_chain(chain, self.RATE, **kw)
        assert e == enrich_option(full, self.SPOT, self.RATE, **kw)
        assert e.dividend_yield == 0.012
        assert e.price_used == 13.0
        assert e.time_to_expiry == pytest.approx(
            time_to_expiry(
                date(2026, 8, 21), self.NOW,
                day_count_basis=360, expiry_time="13:00", expiry_timezone="UTC",
            )
        )

    def test_enrich_chain_spot_override_and_missing_spot(self, full):
        chain = ChainSnapshot(
            underlying="SPY", expiration=date(2026, 8, 21), spot=None, contracts=[full]
        )
        with pytest.raises(PricingError, match="enrich for SPY"):
            enrich_chain(chain, self.RATE, now=self.NOW)
        assert enrich_chain(chain, self.RATE, spot=640.0, now=self.NOW)[0].spot == 640.0
        with_spot = chain.model_copy(update={"spot": self.SPOT})
        assert enrich_chain(with_spot, self.RATE, spot=640.0, now=self.NOW)[0].spot == 640.0
        with pytest.raises(PricingError, match="enrich for SPY"):
            enrich_chain(with_spot, self.RATE, spot=0.0, now=self.NOW)
        empty = ChainSnapshot(underlying="SPY", expiration=date(2026, 8, 21), spot=self.SPOT)
        assert enrich_chain(empty, self.RATE, now=self.NOW) == []
