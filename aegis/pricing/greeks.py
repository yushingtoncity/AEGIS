"""Analytic Black-Scholes Greeks (continuous dividend yield q).

Same model and the same caveats as ``black_scholes``: European exercise,
constant vol and rate, so these are approximations for American-style US
equity options. Units follow the ``Greeks`` model exactly — per share, with
theta per CALENDAR DAY (per-year theta / ``day_count_basis``), vega per
1 vol point (0.01 sigma) and rho per 1 percentage point (0.01 rate); delta
and gamma are unscaled.

Edge cases (explicit branches, never NaN; only malformed input raises):
  T <= 0             → the expired option: gamma = theta = vega = rho = 0,
                       delta = +1 (call, S > K) / -1 (put, S < K) / 0 out of
                       the money / +0.5 or -0.5 exactly at S == K
  sigma <= 0, T > 0  → the deterministic-forward limit: when the forward is
                       in the money (S e^(-qT) > K e^(-rT)) delta = +-e^(-qT)
                       and theta/rho are the derivatives of the discounted
                       intrinsic, otherwise everything is 0; gamma and vega
                       are 0 either way (sigma sqrt(T) underflowing to 0
                       lands here too)
  bad S or K         → PricingError, as in black_scholes; so is a discount
                       factor too large for a float (|r| T > ~709), a forward
                       S e^(-qT) or K e^(-rT) beyond the float range (it would
                       leave vega and theta as inf * 0 = NaN), a rho
                       (K T e^(-rT) N(d2)) beyond the float range, and a
                       non-positive day_count_basis

Formulas (per year / per 1.0 vol / per 1.0 rate, before scaling):
  delta_call = e^(-qT) N(d1)                     delta_put = -e^(-qT) N(-d1)
  gamma      = e^(-qT) n(d1) / (S sigma sqrt(T))
  vega       = S e^(-qT) n(d1) sqrt(T)
  theta_call = -S e^(-qT) n(d1) sigma / (2 sqrt(T)) - r K e^(-rT) N(d2) + q S e^(-qT) N(d1)
  theta_put  = -S e^(-qT) n(d1) sigma / (2 sqrt(T)) + r K e^(-rT) N(-d2) - q S e^(-qT) N(-d1)
  rho_call   = K T e^(-rT) N(d2)                 rho_put   = -K T e^(-rT) N(-d2)
"""

from __future__ import annotations

import math

from scipy.stats import norm

from aegis.data.models import OptionType
from aegis.pricing.black_scholes import (
    _check_inputs,
    _d1_d2,
    _discount,
    _is_call,
    _vol_sqrt_t,
)
from aegis.pricing.errors import PricingError
from aegis.pricing.models import Greeks

__all__ = ["Greeks", "call_greeks", "greeks", "put_greeks"]

_WHAT = "Black-Scholes greeks"


def _scaled(
    delta: float,
    gamma: float,
    theta_per_year: float,
    vega: float,
    rho: float,
    day_count_basis: int,
) -> Greeks:
    """Apply the trader-quoted units of the Greeks model."""
    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        theta=float(theta_per_year) / day_count_basis,
        vega=float(vega) / 100.0,
        rho=float(rho) / 100.0,
    )


def _check_basis(day_count_basis: int) -> None:
    if day_count_basis <= 0:
        raise PricingError(
            _WHAT,
            cause=ValueError(f"day_count_basis must be positive, got {day_count_basis!r}"),
        )


def _rho(K: float, T: float, disc_r: float, tail: float) -> float:
    """K T e^(-rT) tail, per 1.0 rate, where tail is N(d2) (call) or N(-d2)
    (put). A zero tail short-circuits so K T overflowing to inf cannot turn
    into inf * 0 = NaN (r = 0 leaves the discount factor at exactly 1, so
    ``_discount`` never flags it); a product that still overflows has no
    finite answer and is malformed input, like an overflowing discount."""
    if tail == 0.0:
        return 0.0
    # float(tail): norm.cdf hands back a numpy scalar, whose overflow is a
    # RuntimeWarning rather than a plain inf we can test for.
    rho = K * disc_r * float(tail) * T
    if not math.isfinite(rho):
        raise PricingError(
            _WHAT, cause=OverflowError(f"rho exceeds the float range for K={K!r}, T={T!r}")
        )
    return rho


def _forward(label: str, value: float, disc: float) -> float:
    """S e^(-qT) or K e^(-rT) (``label`` says which). One beyond the float
    range has no finite answer, and would poison vega and theta with
    inf * 0 = NaN, so like an overflowing discount factor it is malformed
    input."""
    forward = value * disc
    if not math.isfinite(forward):
        raise PricingError(
            _WHAT,
            cause=OverflowError(f"{label} exceeds the float range ({value!r} * {disc!r})"),
        )
    return forward


def _expired_greeks(is_call: bool, S: float, K: float) -> Greeks:
    if S == K:
        delta = 0.5 if is_call else -0.5
    elif is_call:
        delta = 1.0 if S > K else 0.0
    else:
        delta = -1.0 if S < K else 0.0
    return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)


def _zero_vol_greeks(
    is_call: bool, S: float, K: float, T: float, r: float, q: float, day_count_basis: int
) -> Greeks:
    disc_q = _discount(_WHAT, q, T)
    disc_r = _discount(_WHAT, r, T)
    forward_spot = _forward("S e^(-qT)", S, disc_q)
    forward_strike = _forward("K e^(-rT)", K, disc_r)
    forward_itm = forward_spot > forward_strike if is_call else forward_strike > forward_spot
    if not forward_itm:
        return Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    # Differentiate the discounted intrinsic S e^(-qT) - K e^(-rT) (call) or
    # its negative (put) with respect to S, T and r.
    if is_call:
        delta = disc_q
        theta = q * forward_spot - r * forward_strike
        rho = _rho(K, T, disc_r, 1.0)
    else:
        delta = -disc_q
        theta = r * forward_strike - q * forward_spot
        rho = -_rho(K, T, disc_r, 1.0)
    return _scaled(delta, 0.0, theta, 0.0, rho, day_count_basis)


def greeks(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    day_count_basis: int = 365,
) -> Greeks:
    """Analytic Greeks per share; see the module docstring for units and edge cases."""
    _check_inputs(_WHAT, S, K, T=T, r=r, sigma=sigma, q=q)
    is_call = _is_call(_WHAT, option_type)
    _check_basis(day_count_basis)
    if T <= 0:
        return _expired_greeks(is_call, S, K)
    if _vol_sqrt_t(sigma, T) == 0.0:
        return _zero_vol_greeks(is_call, S, K, T, r, q, day_count_basis)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc_q = _discount(_WHAT, q, T)
    disc_r = _discount(_WHAT, r, T)
    forward_spot = _forward("S e^(-qT)", S, disc_q)
    forward_strike = _forward("K e^(-rT)", K, disc_r)
    pdf_d1 = norm.pdf(d1)

    # Divide by S and by sigma sqrt(T) separately: their product can
    # underflow to 0.0 for a tiny spot even though neither factor is 0.
    gamma = disc_q * pdf_d1 / S / (sigma * sqrt_t)
    vega = forward_spot * pdf_d1 * sqrt_t
    time_decay = -forward_spot * pdf_d1 * sigma / (2.0 * sqrt_t)  # shared by call and put
    if is_call:
        delta = disc_q * norm.cdf(d1)
        theta = time_decay - r * forward_strike * norm.cdf(d2) + q * forward_spot * norm.cdf(d1)
        rho = _rho(K, T, disc_r, norm.cdf(d2))
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = time_decay + r * forward_strike * norm.cdf(-d2) - q * forward_spot * norm.cdf(-d1)
        rho = -_rho(K, T, disc_r, norm.cdf(-d2))
    return _scaled(delta, gamma, theta, vega, rho, day_count_basis)


def call_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    day_count_basis: int = 365,
) -> Greeks:
    return greeks(OptionType.CALL, S, K, T, r, sigma, q, day_count_basis=day_count_basis)


def put_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    *,
    day_count_basis: int = 365,
) -> Greeks:
    return greeks(OptionType.PUT, S, K, T, r, sigma, q, day_count_basis=day_count_basis)
