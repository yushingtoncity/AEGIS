"""Black-Scholes option prices with a continuous dividend yield.

Model assumptions: European exercise, constant volatility and rate, no
dividends unless ``q`` (a continuous yield) is given, log-normal spot.
US equity options are AMERICAN style, so every number here is an
approximation — worst for deep in-the-money puts (early exercise is worth
something) and for dividend payers (a discrete dividend is not a smooth
yield). Treat the output as a cross-check on vendor values, not as truth.

Units: S, K and the returned price are per share; T is in years (see
``time_to_expiry`` for the calendar-day convention); r, sigma and q are
annualised decimals (0.04, not 4).

Edge cases are explicit branches, never exceptions:
  T <= 0                → intrinsic value (the option has expired)
  sigma <= 0, T > 0     → discounted intrinsic (a deterministic forward);
                          sigma sqrt(T) underflowing to 0 lands here too
  non-finite / non-positive S or K → PricingError (malformed input)
Any finite T, sigma, r, q is accepted — negative rates are fine. Non-finite
values of those raise PricingError too, so the output is always a finite
float >= 0 and never NaN. The one other PricingError is a discount factor
e^(-rT) or e^(-qT) too large for a float (|r| T > ~709): no finite answer
exists, so it is treated as malformed input rather than returned as inf.

Formulas (q continuous yield):
  d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)
  call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
  put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
Textbook check (Hull): S=42, K=40, r=0.10, sigma=0.20, T=0.5 → call 4.76, put 0.81.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from aegis.data.models import OptionType
from aegis.pricing.errors import PricingError


def _check_inputs(
    what: str,
    S: float,
    K: float,
    *,
    zero_spot_ok: bool = False,
    **params: float,
) -> None:
    """Raise PricingError for malformed input — the only reason this package raises.

    S and K must be finite and positive (S may be exactly 0 where
    ``zero_spot_ok``: an underlying can expire worthless); every other
    parameter must be finite.
    """
    if not (math.isfinite(S) and (S > 0 or (zero_spot_ok and S == 0))):
        raise PricingError(what, cause=ValueError(f"S must be finite and positive, got {S!r}"))
    if not (math.isfinite(K) and K > 0):
        raise PricingError(what, cause=ValueError(f"K must be finite and positive, got {K!r}"))
    for name, value in params.items():
        if not math.isfinite(value):
            raise PricingError(what, cause=ValueError(f"{name} must be finite, got {value!r}"))


def _is_call(what: str, option_type: OptionType) -> bool:
    # OptionType(...) accepts the member or its string value; anything else
    # is malformed input rather than silently "a put".
    try:
        return OptionType(option_type) is OptionType.CALL
    except ValueError as exc:
        raise PricingError(what, cause=exc) from exc


def _discount(what: str, rate: float, T: float) -> float:
    """e^(-rate T); a factor too large for a float has no finite answer."""
    try:
        return math.exp(-rate * T)
    except OverflowError as exc:
        raise PricingError(what, cause=exc) from exc


def _vol_sqrt_t(sigma: float, T: float) -> float:
    """sigma sqrt(T), the deterministic-forward test: 0.0 when sigma <= 0 or
    when the product underflows for a positive but absurdly small sigma or T."""
    return sigma * math.sqrt(T) if sigma > 0 else 0.0


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    """The two Black-Scholes quantiles; requires T > 0 and sigma sqrt(T) > 0."""
    vol_sqrt_t = _vol_sqrt_t(sigma, T)
    # log(S) - log(K) rather than log(S / K): the ratio can underflow to 0.
    d1 = (math.log(S) - math.log(K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _intrinsic(is_call: bool, S: float, K: float) -> float:
    # float() so integer S/K cannot leak an int out of a float-typed API.
    return max(float(S - K), 0.0) if is_call else max(float(K - S), 0.0)


def _discounted_intrinsic(
    what: str, is_call: bool, S: float, K: float, T: float, r: float, q: float
) -> float:
    forward_spot = S * _discount(what, q, T)
    forward_strike = K * _discount(what, r, T)
    if is_call:
        return max(forward_spot - forward_strike, 0.0)
    return max(forward_strike - forward_spot, 0.0)


def intrinsic_value(option_type: OptionType, S: float, K: float) -> float:
    """Payoff per share if exercised now: max(S - K, 0) for calls, max(K - S, 0) for puts.

    S may be 0 (the underlying expired worthless) so expiry payoffs can be
    evaluated across the whole price axis.
    """
    what = "intrinsic value"
    _check_inputs(what, S, K, zero_spot_ok=True)
    return _intrinsic(_is_call(what, option_type), S, K)


def discounted_intrinsic(
    option_type: OptionType, S: float, K: float, T: float, r: float, q: float = 0.0
) -> float:
    """The zero-volatility price: max(S e^(-qT) - K e^(-rT), 0) for calls and
    max(K e^(-rT) - S e^(-qT), 0) for puts — the present value of a certain
    forward payoff, and the floor every Black-Scholes price sits on.
    """
    what = "discounted intrinsic"
    _check_inputs(what, S, K, zero_spot_ok=True, T=T, r=r, q=q)
    return _discounted_intrinsic(what, _is_call(what, option_type), S, K, T, r, q)


def price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Black-Scholes price per share; see the module docstring for units and edge cases."""
    what = "Black-Scholes price"
    _check_inputs(what, S, K, T=T, r=r, sigma=sigma, q=q)
    is_call = _is_call(what, option_type)
    if T <= 0:
        return _intrinsic(is_call, S, K)
    if _vol_sqrt_t(sigma, T) == 0.0:
        return _discounted_intrinsic(what, is_call, S, K, T, r, q)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    forward_spot = S * _discount(what, q, T)
    forward_strike = K * _discount(what, r, T)
    if is_call:
        value = forward_spot * norm.cdf(d1) - forward_strike * norm.cdf(d2)
    else:
        value = forward_strike * norm.cdf(-d2) - forward_spot * norm.cdf(-d1)
    # Deep out of the money the two terms cancel to rounding noise that can
    # dip a hair below zero; the true price is never negative.
    return max(float(value), 0.0)


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    return price(OptionType.CALL, S, K, T, r, sigma, q)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    return price(OptionType.PUT, S, K, T, r, sigma, q)
