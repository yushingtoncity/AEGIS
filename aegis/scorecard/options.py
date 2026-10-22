"""Estimate what an option structure would have made, not just its direction.

Grading an option on the underlying's move ignores what an option holder
actually pays: time decay and the bid-ask spread. A bought call can lose
money on a small move in the right direction. This module marks each leg
with Black-Scholes at the implied vol the cycle's snapshot stored:

    leg P&L per share = side * (BS(S_h, T_h) - BS(S_0, T_0)) - (ask - bid)

``side`` is +1 bought and -1 sold. ``S_0, T_0`` are the underlying's raw
spot and the time to expiry when the snapshot was taken; ``S_h, T_h`` the
raw close and the time left at the horizon's close. The model change is
added to the quoted mid, so a model-versus-market gap at entry does not leak
into the P&L. A bought leg pays half the spread going in and half coming
out (and a sold leg the same), so one full spread per leg in total, with the
exit spread assumed equal to the entry one.

What it leaves out: changes in implied vol (no IV crush after earnings),
early exercise, dividends, and a fill worse than the quote. It is an
estimate to set beside the direction grade, never a P&L record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from aegis.pricing.black_scholes import price
from aegis.pricing.errors import PricingError
from aegis.pricing.time_to_expiry import time_to_expiry
from aegis.scorecard.models import ProposalInputs
from aegis.store.models import OrderSide

CONTRACT_MULTIPLIER = 100
"""Shares per standard US equity option contract; fixed by the exchange, not tunable."""


@dataclass(frozen=True)
class OptionEstimate:
    """A structure's estimated P&L in dollars and its net entry debit
    (negative for a credit structure)."""

    pnl: float
    debit: float

    @property
    def return_on_debit(self) -> float | None:
        return self.pnl / self.debit if self.debit > 0 else None


def why_not_estimable(inputs: ProposalInputs) -> str | None:
    """Why this proposal's option P&L cannot be estimated, or None if it can."""
    if not inputs.legs:
        return "no legs on record"
    if inputs.snapshot_spot is None or inputs.spot_time is None or inputs.risk_free_rate is None:
        return "no snapshot spot, time or rate"
    for leg in inputs.legs:
        quote = inputs.leg_quotes.get(leg.symbol)
        if quote is None or quote.bid is None or quote.ask is None or quote.implied_vol is None:
            return f"no stored quote and implied vol for {leg.symbol}"
        if quote.ask < quote.bid or quote.bid < 0 or quote.implied_vol <= 0:
            return f"unusable stored quote for {leg.symbol}"
    return None


def estimate(inputs: ProposalInputs, raw_close: float, at: datetime) -> OptionEstimate | None:
    """The structure's estimated P&L if closed at ``raw_close`` at time ``at``.

    None when ``why_not_estimable`` says so or the pricing model refuses the
    inputs. ``raw_close`` must be on the same (unadjusted) basis as the
    strikes and the snapshot spot.
    """
    spot, spot_time = inputs.snapshot_spot, inputs.spot_time
    if why_not_estimable(inputs) is not None or spot is None or spot_time is None:
        return None
    rate = inputs.risk_free_rate or 0.0
    pnl = debit = 0.0
    try:
        for leg in inputs.legs:
            quote = inputs.leg_quotes[leg.symbol]
            bid, ask, vol = quote.bid or 0.0, quote.ask or 0.0, quote.implied_vol or 0.0
            then = price(
                leg.option_type,
                spot,
                leg.strike,
                time_to_expiry(leg.expiration, spot_time),
                rate,
                vol,
            )
            later = price(
                leg.option_type,
                raw_close,
                leg.strike,
                time_to_expiry(leg.expiration, at),
                rate,
                vol,
            )
            side = 1 if leg.side is OrderSide.BUY else -1
            contracts = leg.quantity * CONTRACT_MULTIPLIER
            pnl += contracts * (side * (later - then) - (ask - bid))
            debit += contracts * (ask if side > 0 else -bid)
    except PricingError:
        return None
    return OptionEstimate(pnl=pnl, debit=debit)
