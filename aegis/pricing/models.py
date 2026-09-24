"""Typed inputs and outputs of the pricing engine.

Everything the pricing modules return is one of these frozen models, so
callers (the Phase 4 brain, the Phase 5 policy engine, CLIs) never see bare
tuples. Per-share values are stated per share; position-level values carry
the contract multiplier and are stated in dollars — each field's docstring
says which.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from aegis.data.models import OptionSnapshot, OptionType


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class PremiumType(str, Enum):
    DEBIT = "debit"    # net premium paid (positive net_premium)
    CREDIT = "credit"  # net premium received (negative net_premium)


class Greeks(BaseModel):
    """Analytic Black-Scholes sensitivities, per share of the underlying.

    Units (chosen so values read the way traders quote them):
      delta  dPrice / dSpot, in (-1, 1)
      gamma  dDelta / dSpot, per $1 move in spot
      theta  dPrice / dTime, per CALENDAR DAY (negative for long options)
      vega   dPrice per 1 vol point (a 0.01 change in sigma)
      rho    dPrice per 1 percentage point (0.01) change in the rate
    Multiply by the contract multiplier (100) for per-contract dollars.
    """

    model_config = ConfigDict(frozen=True)

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    def scaled(self, factor: float) -> "Greeks":
        """Every sensitivity multiplied by ``factor`` (sign, quantity, multiplier)."""
        return Greeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            theta=self.theta * factor,
            vega=self.vega * factor,
            rho=self.rho * factor,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
        )


class OptionLeg(BaseModel):
    """One leg of a position: a single contract series, long or short.

    ``premium`` is the per-share price paid (long) or received (short) —
    an option quoted at 2.35 is premium=2.35, not 235. ``greeks`` are
    per-share, for one long unit; ``position.analyze_position`` applies
    sign, quantity and the contract multiplier.
    """

    model_config = ConfigDict(frozen=True)

    option_type: OptionType
    side: Side
    quantity: int = Field(gt=0)
    strike: float = Field(gt=0)
    expiry: date
    premium: float = Field(ge=0)
    greeks: Greeks | None = None
    symbol: str | None = None

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self.side is Side.LONG else -1


class PositionSummary(BaseModel):
    """Aggregate risk picture of a multi-leg option position.

    Dollar fields include the contract multiplier. ``max_profit`` /
    ``max_loss`` are stated as positive magnitudes and are ``None`` when
    unbounded; the ``unlimited_*`` flags say so explicitly so callers never
    have to guess what ``None`` means. ``max_loss`` counts premium: a long
    call vertical bought for a 1.50 debit has max_loss 150.0.
    """

    model_config = ConfigDict(frozen=True)

    legs: tuple[OptionLeg, ...]
    contract_multiplier: int
    net_premium: float
    """Signed dollars: positive = debit paid, negative = credit received."""
    premium_type: PremiumType
    net_greeks: Greeks | None
    """Sum of sign * quantity * multiplier * per-share greeks; None if any
    leg lacks greeks (a partial sum would be misleading)."""
    max_profit: float | None
    max_loss: float | None
    breakevens: tuple[float, ...]
    """Underlying prices at expiry where P&L crosses zero, ascending."""
    unlimited_risk: bool
    unlimited_profit: bool


class EnrichedOption(BaseModel):
    """An OptionSnapshot with vendor and model values side by side.

    ``vendor_*`` are exactly what the data layer received (None where
    Alpaca omitted them). ``model_*`` are computed here from the quote mid,
    spot and rate. ``implied_vol`` / ``greeks`` prefer the vendor value and
    fall back to the model value, and ``iv_source`` / ``greeks_source``
    record which one won so a cross-check is always possible.
    """

    model_config = ConfigDict(frozen=True)

    snapshot: OptionSnapshot
    spot: float
    rate: float
    dividend_yield: float
    time_to_expiry: float | None
    """Years to expiry under the calendar-day convention; None if the
    snapshot has no expiration date."""
    price_used: float | None
    """The market price the model IV was solved from (quote mid), if any."""
    vendor_implied_vol: float | None
    vendor_greeks: Greeks | None
    model_implied_vol: float | None
    model_greeks: Greeks | None
    model_price: float | None
    """Black-Scholes price at the vendor IV (for cross-checking the quote);
    None without a vendor IV."""

    @property
    def implied_vol(self) -> float | None:
        return (
            self.vendor_implied_vol
            if self.vendor_implied_vol is not None
            else self.model_implied_vol
        )

    @property
    def iv_source(self) -> str | None:
        if self.vendor_implied_vol is not None:
            return "vendor"
        return "model" if self.model_implied_vol is not None else None

    @property
    def greeks(self) -> Greeks | None:
        return self.vendor_greeks if self.vendor_greeks is not None else self.model_greeks

    @property
    def greeks_source(self) -> str | None:
        if self.vendor_greeks is not None:
            return "vendor"
        return "model" if self.model_greeks is not None else None
