"""Put vendor and model values side by side for one option or a whole chain.

Alpaca's option snapshots carry an implied volatility and Greeks for most
contracts, but omit them for near-dated, deep-OTM or unquoted ones. This
module never replaces what the vendor sent: ``EnrichedOption`` keeps the
vendor values untouched (``vendor_*``) and computes our own next to them
(``model_*``), so the two can always be cross-checked. The preference order
is vendor first, model as the fallback, and it is exposed rather than
hidden — ``EnrichedOption.implied_vol`` / ``.greeks`` apply it, and
``iv_source`` / ``greeks_source`` say which side won.

What gets computed, given spot, rate and (optionally) a dividend yield:
  time_to_expiry     from the snapshot's expiration under the calendar-day
                     convention of ``time_to_expiry`` (None if no expiration)
  price_used         the price the model IV is solved from: an explicit
                     ``price`` argument, else the quote MID, else the last
                     trade. Mid is the right input; last is a weaker
                     fallback because it may be stale and one-sided.
  model_implied_vol  Black-Scholes IV solved from ``price_used`` (None when
                     the solver finds no answer, e.g. a quote at intrinsic)
  model_greeks       analytic Greeks at the model IV, else at the vendor IV
                     (so a vendor that supplies IV but omits Greeks still
                     gets Greeks), else None
  model_price        Black-Scholes price at the VENDOR IV, for checking the
                     quote against the vendor's own volatility

Any missing snapshot field (strike, type, expiration, quotes) makes the
dependent ``model_*`` fields None; nothing here raises for missing data, so
a whole chain can be enriched in one pass. ``PricingError`` is raised only
for malformed input: a non-finite or non-positive spot, a non-finite rate
or yield, a non-positive strike, or a non-positive day-count basis. The
model values are European Black-Scholes numbers, so for American-style US
equity options they are approximations (see ``black_scholes``).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from aegis.data.models import ChainSnapshot, OptionSnapshot
from aegis.pricing import black_scholes as bs
from aegis.pricing.errors import PricingError
from aegis.pricing.greeks import greeks as bs_greeks
from aegis.pricing.implied_vol import implied_vol
from aegis.pricing.models import EnrichedOption, Greeks
from aegis.pricing.time_to_expiry import (
    DEFAULT_DAY_COUNT_BASIS,
    DEFAULT_EXPIRY_TIME,
    DEFAULT_EXPIRY_TIMEZONE,
    time_to_expiry,
)

_WHAT = "enrich"


def _check_market_inputs(
    symbol: str, spot: float, rate: float, dividend_yield: float, day_count_basis: int
) -> None:
    """Raise PricingError for malformed market inputs — the only reason enrich raises."""
    if not (math.isfinite(spot) and spot > 0):
        raise PricingError(
            _WHAT, symbol, cause=ValueError(f"spot must be finite and positive, got {spot!r}")
        )
    for name, value in (("rate", rate), ("dividend_yield", dividend_yield)):
        if not math.isfinite(value):
            raise PricingError(
                _WHAT, symbol, cause=ValueError(f"{name} must be finite, got {value!r}")
            )
    # Checked here rather than left to time_to_expiry / greeks: those only
    # run for snapshots with an expiration, so the error would be sporadic.
    if day_count_basis <= 0:
        raise PricingError(
            _WHAT,
            symbol,
            cause=ValueError(f"day_count_basis must be positive, got {day_count_basis!r}"),
        )


def _vendor_greeks(snapshot: OptionSnapshot) -> Greeks | None:
    """The vendor's Greeks as a model, only when all five are present — a
    partial set would silently mix vendor and model numbers downstream."""
    values = (snapshot.delta, snapshot.gamma, snapshot.theta, snapshot.vega, snapshot.rho)
    if any(value is None for value in values):
        return None
    delta, gamma, theta, vega, rho = values
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def enrich_option(
    snapshot: OptionSnapshot,
    spot: float,
    rate: float,
    *,
    now: datetime | None = None,
    dividend_yield: float = 0.0,
    price: float | None = None,
    day_count_basis: int = DEFAULT_DAY_COUNT_BASIS,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> EnrichedOption:
    """Compute the model side of one snapshot; see the module docstring.

    ``price`` overrides the quote mid as the input to the IV solve. ``now``
    defaults to the current UTC time; pass it explicitly for reproducible
    results. The day-count and expiry-instant keywords mirror config.yaml's
    ``pricing`` block — this package never reads config.
    """
    _check_market_inputs(snapshot.symbol, spot, rate, dividend_yield, day_count_basis)
    strike = snapshot.strike
    if strike is not None and not (math.isfinite(strike) and strike > 0):
        raise PricingError(
            _WHAT,
            snapshot.symbol,
            cause=ValueError(f"strike must be finite and positive, got {strike!r}"),
        )

    T = (
        None
        if snapshot.expiration is None
        else time_to_expiry(
            snapshot.expiration,
            now,
            day_count_basis=day_count_basis,
            expiry_time=expiry_time,
            expiry_timezone=expiry_timezone,
        )
    )
    if price is not None:
        price_used: float | None = price
    elif snapshot.mid is not None:
        price_used = snapshot.mid
    else:
        price_used = snapshot.last

    vendor_iv = snapshot.implied_vol
    # A feed can carry a NaN/inf IV; it passes through untouched as vendor_*
    # but is not a volatility the model can price at.
    usable_vendor_iv = (
        vendor_iv if vendor_iv is not None and math.isfinite(vendor_iv) else None
    )
    option_type = snapshot.option_type
    # Every model value needs the contract to be fully identified and alive.
    priceable = strike is not None and option_type is not None and T is not None

    model_iv = None
    if priceable and price_used is not None:
        model_iv = implied_vol(option_type, price_used, spot, strike, T, rate, dividend_yield)

    # Prefer the IV we solved ourselves (it matches the quote we hold); fall
    # back to the vendor's so IV-only snapshots still get Greeks.
    sigma = model_iv if model_iv is not None else usable_vendor_iv
    model_greeks = None
    if priceable and sigma is not None:
        model_greeks = bs_greeks(
            option_type, spot, strike, T, rate, sigma, dividend_yield,
            day_count_basis=day_count_basis,
        )

    model_price = None
    if priceable and usable_vendor_iv is not None:
        model_price = bs.price(
            option_type, spot, strike, T, rate, usable_vendor_iv, dividend_yield
        )

    return EnrichedOption(
        snapshot=snapshot,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        time_to_expiry=T,
        price_used=price_used,
        vendor_implied_vol=vendor_iv,
        vendor_greeks=_vendor_greeks(snapshot),
        model_implied_vol=model_iv,
        model_greeks=model_greeks,
        model_price=model_price,
    )


def enrich_chain(
    chain: ChainSnapshot,
    rate: float,
    *,
    spot: float | None = None,
    now: datetime | None = None,
    dividend_yield: float = 0.0,
    price: float | None = None,
    day_count_basis: int = DEFAULT_DAY_COUNT_BASIS,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> list[EnrichedOption]:
    """Enrich every contract in a chain, one EnrichedOption per contract in
    chain order.

    ``spot`` defaults to ``chain.spot``; PricingError if neither is
    available. Every other keyword is forwarded to ``enrich_option``
    unchanged, ``price`` included — it overrides the quote mid for EVERY
    contract, which is rarely what a caller wants, so leave it None and let
    each IV be solved from its own mid.
    """
    if spot is None:
        spot = chain.spot
    if spot is None:
        raise PricingError(
            _WHAT,
            chain.underlying,
            cause=ValueError("no spot price: pass spot= or set chain.spot"),
        )
    _check_market_inputs(chain.underlying, spot, rate, dividend_yield, day_count_basis)
    # Fix one clock for the whole chain so every contract shares the same T
    # instead of drifting by microseconds down the list.
    if now is None:
        now = datetime.now(timezone.utc)
    return [
        enrich_option(
            contract,
            spot,
            rate,
            now=now,
            dividend_yield=dividend_yield,
            price=price,
            day_count_basis=day_count_basis,
            expiry_time=expiry_time,
            expiry_timezone=expiry_timezone,
        )
        for contract in chain.contracts
    ]
