"""Black-Scholes implied volatility by bracketed root finding.

Inverts ``black_scholes.price`` in sigma with ``scipy.optimize.brentq`` on
``[lo, hi]``. The result is a Black-Scholes (European) IV, so for
American-style US equity options it is an approximation with the same
caveats as the pricer (deep-ITM puts, dividend payers).

What to feed it: the MID price (bid/ask midpoint), never the bid or the
ask — a one-sided quote embeds half the spread and biases the IV. Near
expiry (a few days or less) vega tends to zero, so a tick of price moves
the IV enormously and the number is unstable and close to meaningless;
callers should treat short-dated IVs with suspicion.

Returns None — never raises — when there is no answer inside the bracket:
  T <= 0                        an expired option has no volatility
  price None / non-finite / < 0 not a price
  price == 0                    a zero quote carries no volatility
                                information: far enough out of the money
                                every sigma up to some threshold prices to
                                exactly 0.0, so it is the floor, not a root
  price < BS price at sigma=lo  at or below the discounted-intrinsic floor
  price > BS price at sigma=hi  above what any vol in the bracket can explain
  brentq fails                  (ValueError; the sign test above should
                                catch every such case, this is belt and braces)
The endpoints are handled first: a (positive) price exactly at the sigma=lo
or sigma=hi value returns lo or hi. Malformed S or K raises PricingError,
as in ``black_scholes``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from aegis.data.models import OptionType
from aegis.pricing import black_scholes as bs

_WHAT = "implied volatility"


def implied_vol(
    option_type: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-10,
) -> float | None:
    """Sigma such that the Black-Scholes price equals ``price``, or None.

    ``price`` should be the quote mid. ``lo``/``hi`` bracket the search
    (annualised decimals) and ``tol`` is brentq's absolute xtol on sigma.
    """
    bs._check_inputs(_WHAT, S, K, T=T, r=r, q=q)
    bs._is_call(_WHAT, option_type)
    # price <= 0 rather than < 0: a 0.00 quote on a deep-OTM contract would
    # otherwise match price(lo) == 0.0 and come back as sigma = lo.
    if T <= 0 or price is None or not math.isfinite(price) or price <= 0:
        return None

    def objective(sigma: float) -> float:
        return bs.price(option_type, S, K, T, r, sigma, q) - price

    f_lo = objective(lo)
    f_hi = objective(hi)
    # float(): an int bracket must not leak an int out of a float-typed API.
    if f_lo == 0:
        return float(lo)
    if f_hi == 0:
        return float(hi)
    if f_lo > 0 or f_hi < 0:
        return None
    try:
        root = brentq(objective, lo, hi, xtol=tol)
    except ValueError:
        return None
    return float(root)
