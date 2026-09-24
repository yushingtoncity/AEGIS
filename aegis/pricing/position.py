"""Multi-leg position analysis: net premium, net Greeks and the expiry P&L
picture (max profit, max loss, breakevens, unlimited flags).

Everything here is evaluated AT EXPIRY, so all legs must share one expiry;
calendar and diagonal spreads have no single expiry payoff and are out of
scope (``PricingError``). The expiry P&L of any option position is a
piecewise-linear function of the underlying price whose kinks sit exactly
at the strikes, so it is evaluated numerically on the grid {0, every
strike, one probe beyond the highest strike} instead of by recognising
named strategies: the grid extremes give the bounded max/min, the slope
past the last strike (calls only; puts are flat up there) decides the
unlimited flags, and linear interpolation between grid points is exact for
the breakevens. Ratio spreads, backspreads and any other shape fall out for
free, which is exactly where per-strategy formulas go wrong.

Assumptions: the position is held to expiry (early assignment on short
American-style legs is ignored), the underlying cannot trade below zero
(which is what bounds put risk), and every dollar figure includes the
contract multiplier. Per-share inputs come from ``OptionLeg``.

Malformed input raises ``PricingError``: an empty position, mixed expiries,
a non-positive contract multiplier, a leg whose strike or premium is not
finite (``OptionLeg``'s bounds let inf through), or a P&L that overflows a
float. Every other input produces a finite summary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from aegis.data.models import OptionType
from aegis.pricing.black_scholes import intrinsic_value
from aegis.pricing.errors import PricingError
from aegis.pricing.models import Greeks, OptionLeg, PositionSummary, PremiumType

# Two breakevens closer than this, relative to their size, are the same root
# found twice (the far segment's analytic root also shows up by interpolation
# when it lies before the probe point). Relative because the two formulas
# differ by an ULP, which outgrows any absolute tolerance at large strikes.
_BREAKEVEN_TOLERANCE = 1e-9


def net_premium(legs: Sequence[OptionLeg], *, contract_multiplier: int = 100) -> float:
    """Signed dollars paid for the position: positive = debit, negative = credit."""
    _check_multiplier(contract_multiplier)
    return sum(
        (leg.sign * leg.quantity * leg.premium * contract_multiplier for leg in legs),
        0.0,
    )


def net_greeks(
    legs: Sequence[OptionLeg], *, contract_multiplier: int = 100
) -> Greeks | None:
    """Position Greeks in dollar terms: sum of sign * quantity * multiplier *
    per-share greeks. None if any leg lacks greeks — a partial sum would be
    misleading."""
    _check_multiplier(contract_multiplier)
    total = Greeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    for leg in legs:
        if leg.greeks is None:
            return None
        total = total + leg.greeks.scaled(leg.sign * leg.quantity * contract_multiplier)
    return total


def payoff_at_expiry(
    legs: Sequence[OptionLeg], underlying_price: float, *, contract_multiplier: int = 100
) -> float:
    """Total P&L in dollars if the underlying settles at ``underlying_price``
    at expiry, premium paid/received included. ``underlying_price`` may be 0
    (a stock can go to zero) but not negative or non-finite."""
    _check_multiplier(contract_multiplier)
    legs = _check_legs(legs)
    if not math.isfinite(underlying_price) or underlying_price < 0:
        raise PricingError(f"expiry payoff at underlying price {underlying_price!r}")
    total = sum(
        (
            leg.sign
            * leg.quantity
            * contract_multiplier
            * (intrinsic_value(leg.option_type, underlying_price, leg.strike) - leg.premium)
            for leg in legs
        ),
        0.0,
    )
    if not math.isfinite(total):
        raise PricingError(
            f"expiry payoff at underlying price {underlying_price!r}",
            cause=OverflowError("P&L exceeds the float range"),
        )
    return total


def analyze_position(
    legs: Sequence[OptionLeg], *, contract_multiplier: int = 100
) -> PositionSummary:
    """Risk picture of a position held to expiry (see the module docstring
    for the grid method).

    ``max_profit`` / ``max_loss`` are positive magnitudes and ``None`` when
    the matching ``unlimited_*`` flag is set; a position that cannot profit
    (a vertical bought for more than its width) reports max_profit 0.0, and
    one that cannot lose (sold for more than its width) reports max_loss
    0.0 — never a negative magnitude. ``premium_type`` is CREDIT when net
    premium is negative and DEBIT otherwise, so a zero-cost position reports
    DEBIT with net_premium 0.0. A fully offset position (long and short the
    same contract) is a flat zero payoff: max_profit and max_loss are both
    0.0 and no flag is set.
    """
    _check_multiplier(contract_multiplier)
    legs = _check_legs(legs)
    strikes = sorted({leg.strike for leg in legs})
    # S = 0 bounds the downside, every strike is a kink, and one probe past
    # the highest strike pins the final linear segment.
    probe = 2 * strikes[-1] + 1
    if not math.isfinite(probe):
        raise PricingError(
            "expiry payoff",
            cause=OverflowError(f"no room to probe beyond strike {strikes[-1]!r}"),
        )
    grid = [0.0, *strikes, probe]
    pnl = [
        payoff_at_expiry(legs, s, contract_multiplier=contract_multiplier) for s in grid
    ]
    # Past the highest strike puts are flat and each call moves 1:1 with S.
    far_slope = contract_multiplier * sum(
        leg.sign * leg.quantity for leg in legs if leg.option_type is OptionType.CALL
    )
    premium = net_premium(legs, contract_multiplier=contract_multiplier)
    return PositionSummary(
        legs=legs,
        contract_multiplier=contract_multiplier,
        net_premium=premium,
        premium_type=PremiumType.CREDIT if premium < 0 else PremiumType.DEBIT,
        net_greeks=net_greeks(legs, contract_multiplier=contract_multiplier),
        # Clamped at 0.0 (listed first so a flat position reports 0.0, not
        # -0.0): a P&L that never crosses zero has no profit, or no loss.
        max_profit=None if far_slope > 0 else max(0.0, max(pnl)),
        max_loss=None if far_slope < 0 else max(0.0, -min(pnl)),
        breakevens=_breakevens(grid, pnl, far_slope),
        unlimited_risk=far_slope < 0,
        unlimited_profit=far_slope > 0,
    )


def _check_multiplier(contract_multiplier: int) -> None:
    if contract_multiplier <= 0:
        raise PricingError(
            "position analysis",
            cause=ValueError(
                f"contract_multiplier must be positive, got {contract_multiplier!r}"
            ),
        )


def _check_legs(legs: Sequence[OptionLeg]) -> tuple[OptionLeg, ...]:
    """Materialise ``legs`` (an iterator must survive being walked more than
    once) and reject what has no expiry P&L: an empty position, a leg whose
    strike or premium is not finite, and mixed expiries — calendar and
    diagonal spreads are out of scope for this analysis."""
    legs = tuple(legs)
    if not legs:
        raise PricingError("expiry payoff of an empty position")
    for leg in legs:
        if not (math.isfinite(leg.strike) and math.isfinite(leg.premium)):
            raise PricingError(
                "expiry payoff",
                leg.symbol,
                cause=ValueError(
                    "strike and premium must be finite, "
                    f"got strike={leg.strike!r} premium={leg.premium!r}"
                ),
            )
    expiries = sorted({leg.expiry for leg in legs})
    if len(expiries) > 1:
        raise PricingError(
            "expiry payoff across mixed expiries "
            f"({', '.join(e.isoformat() for e in expiries)})"
        )
    return legs


def _breakevens(
    grid: Sequence[float], pnl: Sequence[float], far_slope: float
) -> tuple[float, ...]:
    """Every underlying price where the expiry P&L crosses or touches zero,
    ascending and de-duplicated."""
    # The probe point is not a real grid point: a far segment sitting at
    # zero already touches at the last strike, and a crossing exactly at
    # the probe is found by the far-slope rule below.
    roots = [s for s, p in zip(grid[:-1], pnl[:-1]) if p == 0]
    for (s0, p0), (s1, p1) in pairwise(zip(grid, pnl)):
        if p0 * p1 < 0:
            # Exact on a linear segment; one division keeps round inputs round.
            roots.append((s0 * p1 - s1 * p0) / (p1 - p0))
    # The last segment continues at far_slope forever, so its root can lie
    # past the probe point where no interpolation bracket exists.
    s_last, p_last = grid[-2], pnl[-2]
    if far_slope != 0 and p_last * far_slope < 0:
        roots.append(s_last - p_last / far_slope)
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or root - unique[-1] > _BREAKEVEN_TOLERANCE * max(1.0, root):
            unique.append(root)
    return tuple(unique)
