"""Build the scan stage's input: a ``MarketSnapshot`` of the watchlist.

This is the ONE place in ``aegis.brain`` that talks to the live data
layer (``aegis.data.market`` / ``account`` / ``news``); every other brain
module consumes the typed ``MarketSnapshot`` it returns. The live entry
point, ``build_market_snapshot``, only fetches; everything it does with
the fetched models is done by pure assembly helpers that tests drive from
the canned fixtures without a network:

  symbol_snapshot_from   Quote + Bars + ChainSnapshot/EnrichedOptions + NewsItems
                         -> SymbolSnapshot
  chain_summary_from     ChainSnapshot + EnrichedOptions -> ChainSummary (ATM ± N)
  account_summary_from   AccountState -> AccountSummary
  data_age_from          MarketClock + SymbolSnapshots -> DataAge

The fetchers are called through this module's globals (``get_spot`` etc.),
so a test replaces them with ``monkeypatch.setattr(aegis.brain.snapshot,
"get_spot", fake)`` and never reaches Alpaca.

Which expiration: each symbol's chain is the NEAREST listed expiration
whose DTE is at least ``risk_limits.min_dte`` and at most
``brain.snapshot.max_dte`` (``eligible_expiration``). The floor is read
from ``risk_limits`` itself, so the brain and the policy engine's
``options_min_dte`` share one number. DTE here is the pricing engine's
calendar-day convention — ``calendar_days_to_expiry``, fractional days to
the exchange close on expiration day — in whole days, rounded down
(``whole_days_to_expiry``): during the session that is the policy's count
of calendar days from the trading date, and it is never more than that
count, so a chain the brain picks is never one ``options_min_dte`` rejects.
The data layer lists the expirations and fetches the chain
(``get_option_chain(symbol, eligible=...)``): ``in_dte_window`` decides
which dates qualify and the data layer takes the nearest of them, which is
``eligible_expiration`` over the listing. This module then checks the chain
it gets back, and keeps only the contracts of that expiration. With no expiration in the
window the symbol gets a ``no eligible expiration`` line in its errors and
no chain — never a nearer expiration — and the cycle goes on.

Failure policy: a fetch that fails for one symbol (``DataError``,
``ConfigError``, or a ``PricingError`` from enrichment) becomes an entry in
that ``SymbolSnapshot.errors`` and the symbol still appears, so one broken
feed never blanks the whole brief. A market-clock failure leaves
``DataAge.market_open`` None; an account failure leaves ``account`` None;
both add a warning.

Staleness (the same rule for a quote and a chain): stale when the market
is not known to be open (closed, or the clock is unknown), when the venue
timestamp is missing, or when it is older than ``stale_after_seconds``. A
quote with no usable price at all (no quote, or a timestamp but neither a
trade nor a two-sided quote, so ``spot`` is None) is stale too, however
fresh its timestamp, and warned about as missing. ``DataAge.any_stale`` is
set by any stale spot *or* chain. ``DataAge.warnings`` are deterministic
system facts — the scan stage merges them into the brief so the model can
never drop them.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, timedelta, timezone

from aegis.brain.errors import BrainError
from aegis.brain.models import (
    AccountSummary,
    ChainSummary,
    ContractSummary,
    DataAge,
    HeadlineItem,
    MarketSnapshot,
    OpenPosition,
    SymbolSnapshot,
)
from aegis.config import AegisConfig, ConfigError, get_config
from aegis.data.errors import DataError, NoEligibleExpiration
from aegis.data.models import (
    AccountState,
    Bar,
    ChainSnapshot,
    MarketClock,
    NewsItem,
    OptionSnapshot,
    OptionType,
    Quote,
    format_age,
    utcnow,
)
from aegis.pricing.enrich import enrich_chain
from aegis.pricing.errors import PricingError
from aegis.pricing.models import EnrichedOption, Greeks
from aegis.pricing.time_to_expiry import (
    DEFAULT_EXPIRY_TIME,
    DEFAULT_EXPIRY_TIMEZONE,
    calendar_days_to_expiry,
)

# The live fetchers pull in alpaca-py, which still imports ``websockets.legacy``
# — deprecated at import time since websockets 14. That warning is the
# dependency's to act on, not this package's, so the one import site keeps it
# from failing a ``python -W error`` test run. Nothing else is filtered.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="websockets.legacy is deprecated", category=DeprecationWarning
    )
    from aegis.data.account import get_account_state
    from aegis.data.market import get_bars, get_market_clock, get_option_chain, get_spot
    from aegis.data.news import get_news

OPTIONS_FEED_NOTE = "free-plan options quotes run ~15 minutes behind (indicative feed)"
"""Standing warning whenever any option chain is in the snapshot."""

MARKET_CLOSED_WARNING = (
    "market is CLOSED — all quotes and chains are the last available and STALE"
)
CLOCK_UNKNOWN_WARNING = (
    "market status UNKNOWN — the market clock could not be fetched; "
    "treating all data as potentially STALE"
)
NO_ELIGIBLE_EXPIRATION = "no eligible expiration"
"""How a symbol's error line starts when no expiration fell in the DTE window."""

_DTE_KEYS = "risk_limits.min_dte, brain.snapshot.max_dte"

_FETCH_ERRORS = (DataError, ConfigError)


# --- small shared helpers -----------------------------------------------------


def _utc(now: datetime | None) -> datetime:
    """An aware UTC clock; a naive ``now`` is treated as UTC (as the pricing package does)."""
    if now is None:
        return utcnow()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _age(timestamp: datetime | None, now: datetime) -> float | None:
    """Seconds between a venue timestamp and ``now``, clamped at zero (clock skew)."""
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds())


def _is_stale(
    timestamp: datetime | None,
    now: datetime,
    *,
    stale_after_seconds: float,
    market_open: bool | None,
) -> bool:
    """The staleness rule from the module docstring."""
    if market_open is not True or timestamp is None:
        return True
    return (now - timestamp).total_seconds() > stale_after_seconds


def _vendor_greeks(snapshot: OptionSnapshot) -> Greeks | None:
    """The vendor's Greeks, only when all five are present — the same rule as
    ``aegis.pricing.enrich``; used when a contract has no enriched counterpart."""
    values = (snapshot.delta, snapshot.gamma, snapshot.theta, snapshot.vega, snapshot.rho)
    if any(value is None for value in values):
        return None
    delta, gamma, theta, vega, rho = values
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


# --- which expiration ---------------------------------------------------------


def whole_days_to_expiry(
    expiration: date,
    now: datetime,
    *,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> int:
    """DTE as the snapshot counts it: the pricing engine's
    ``calendar_days_to_expiry`` (fractional calendar days to the exchange
    close on expiration day, 0 once it has passed) rounded down to whole days.

    With the expiry instant at the 16:00 close, this is the policy engine's
    DTE (calendar days from the trading date to expiration) through the
    session, and one less after the close — or in the last hour before it,
    across a daylight-saving change. For a contract that has not expired it
    is never more, so a floor this count clears is one ``options_min_dte``
    clears too (``in_dte_window`` admits no expired contract)."""
    return math.floor(
        calendar_days_to_expiry(
            expiration, _utc(now), expiry_time=expiry_time, expiry_timezone=expiry_timezone
        )
    )


def in_dte_window(
    expiration: date,
    *,
    now: datetime,
    min_dte: int,
    max_dte: int,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> bool:
    """Whether ``expiration`` is ``min_dte`` to ``max_dte`` days out, both
    ends included (``whole_days_to_expiry``). An expiration whose close has
    passed is never in the window, whatever the floor: its DTE reads 0 here
    (the count is clamped), but the contract is gone."""
    remaining = calendar_days_to_expiry(
        expiration, _utc(now), expiry_time=expiry_time, expiry_timezone=expiry_timezone
    )
    if remaining <= 0:
        return False
    return min_dte <= math.floor(remaining) <= max_dte


def eligible_expiration(
    expirations: Iterable[date],
    *,
    now: datetime,
    min_dte: int,
    max_dte: int,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> date | None:
    """The nearest of ``expirations`` inside the DTE window, or None when
    none is — never a nearer one outside it."""
    for expiration in sorted(set(expirations)):
        if in_dte_window(
            expiration,
            now=now,
            min_dte=min_dte,
            max_dte=max_dte,
            expiry_time=expiry_time,
            expiry_timezone=expiry_timezone,
        ):
            return expiration
    return None


def _dte_window(config: AegisConfig, now: datetime) -> Callable[[date], bool]:
    """The window as the data layer's ``eligible`` test: the floor straight
    from ``risk_limits.min_dte``, the cap from ``brain.snapshot.max_dte``."""
    pricing = config.pricing

    def eligible(expiration: date) -> bool:
        return in_dte_window(
            expiration,
            now=now,
            min_dte=config.risk_limits.min_dte,
            max_dte=config.brain.snapshot.max_dte,
            expiry_time=pricing.expiry_time,
            expiry_timezone=pricing.expiry_timezone,
        )

    return eligible


def _no_eligible_expiration(config: AegisConfig, why: str) -> str:
    window = f"{config.risk_limits.min_dte}-{config.brain.snapshot.max_dte} DTE"
    return f"{NO_ELIGIBLE_EXPIRATION} ({window}, {_DTE_KEYS}): {why}; option chain omitted"


# --- pure assembly helpers ----------------------------------------------------


def _contract_summary(
    snapshot: OptionSnapshot,
    enriched: EnrichedOption | None,
    *,
    option_type: OptionType,
    strike: float,
    chain: ChainSnapshot,
    now: datetime,
) -> ContractSummary:
    """One contract row. With an ``EnrichedOption`` the IV/Greeks follow its
    vendor-first preference and carry their source; without one (enrichment
    failed, e.g. no spot) only the vendor values are shown. ``option_type``
    and ``strike`` are the selection keys (the snapshot's own, already
    known to be present)."""
    if enriched is not None:
        implied_vol, iv_source = enriched.implied_vol, enriched.iv_source
        greeks, greeks_source = enriched.greeks, enriched.greeks_source
    else:
        implied_vol = snapshot.implied_vol
        iv_source = None if implied_vol is None else "vendor"
        greeks = _vendor_greeks(snapshot)
        greeks_source = None if greeks is None else "vendor"
    return ContractSummary(
        symbol=snapshot.symbol,
        option_type=option_type,
        strike=strike,
        expiration=snapshot.expiration or chain.expiration,
        bid=snapshot.bid,
        ask=snapshot.ask,
        mid=snapshot.mid,
        last=snapshot.last,
        volume=snapshot.volume,
        open_interest=snapshot.open_interest,
        implied_vol=implied_vol,
        iv_source=iv_source,
        greeks=greeks,
        greeks_source=greeks_source,
        quote_age_seconds=_age(snapshot.quote_time, now),
    )


def chain_summary_from(
    chain: ChainSnapshot,
    enriched: Sequence[EnrichedOption],
    *,
    now: datetime,
    stale_after_seconds: float,
    market_open: bool | None,
    strikes_each_side: int,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> ChainSummary:
    """The chain reduced to ATM ± ``strikes_each_side`` strikes, calls and
    puts, ordered by strike (call before put at each strike).

    ``enriched`` is ``enrich_chain(chain, ...)`` — matched to contracts by
    (type, strike); contracts without a match fall back to vendor values.
    ATM is measured against ``chain.spot``, else the enriched spot; with
    neither, the lowest strikes are shown (``ChainSnapshot.atm_strikes``).
    ``days_to_expiry`` is fractional calendar days to the exchange close on
    expiration day, the same convention as the pricing engine's T.
    """
    now = _utc(now)
    spot = chain.spot
    if spot is None and enriched:
        spot = enriched[0].spot
    if spot != chain.spot:
        chain = chain.model_copy(update={"spot": spot})

    by_key = {
        (e.snapshot.option_type, e.snapshot.strike): e
        for e in enriched
        if e.snapshot.option_type is not None and e.snapshot.strike is not None
    }
    contracts: list[ContractSummary] = []
    for strike in chain.atm_strikes(strikes_each_side):
        for option_type in (OptionType.CALL, OptionType.PUT):
            snapshot = chain.contract_at(strike, option_type)
            if snapshot is None or snapshot.strike is None:
                continue
            contracts.append(
                _contract_summary(
                    snapshot,
                    by_key.get((option_type, snapshot.strike)),
                    option_type=option_type,
                    strike=snapshot.strike,
                    chain=chain,
                    now=now,
                )
            )

    latest = chain.latest_quote_time
    return ChainSummary(
        underlying=chain.underlying,
        expiration=chain.expiration,
        days_to_expiry=calendar_days_to_expiry(
            chain.expiration, now, expiry_time=expiry_time, expiry_timezone=expiry_timezone
        ),
        spot=spot,
        atm_strike=chain.atm_strike(),
        contracts=tuple(contracts),
        quote_age_seconds=_age(latest, now),
        stale=_is_stale(
            latest, now, stale_after_seconds=stale_after_seconds, market_open=market_open
        ),
    )


def _bars_summary(
    bars: Sequence[Bar],
) -> tuple[float | None, float | None, float | None, float | None]:
    """(last_close, change_5d_pct, high_20d, low_20d), newest bar last.

    change_5d_pct compares the last close with the close five bars earlier;
    the 20-bar high/low need a full 20 bars — anything less is None rather
    than a misleading partial range.
    """
    ordered = sorted(bars, key=lambda b: b.timestamp)
    if not ordered:
        return None, None, None, None
    last_close = ordered[-1].close
    change_5d_pct = None
    if len(ordered) >= 6 and ordered[-6].close:
        change_5d_pct = (last_close / ordered[-6].close - 1.0) * 100.0
    high_20d = low_20d = None
    if len(ordered) >= 20:
        window = ordered[-20:]
        high_20d = max(b.high for b in window)
        low_20d = min(b.low for b in window)
    return last_close, change_5d_pct, high_20d, low_20d


def _headline(item: NewsItem, now: datetime) -> HeadlineItem:
    return HeadlineItem(
        headline=item.headline,
        summary=item.summary,
        source=item.source,
        published_at=item.published_at,
        age_seconds=_age(item.published_at, now),
        symbols=tuple(item.symbols),
    )


def symbol_snapshot_from(
    symbol: str,
    quote: Quote | None,
    bars: Sequence[Bar],
    chain: ChainSnapshot | None,
    enriched: Sequence[EnrichedOption],
    news: Sequence[NewsItem],
    *,
    now: datetime,
    stale_after_seconds: float,
    market_open: bool | None,
    strikes_each_side: int,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
    errors: Sequence[str] = (),
) -> SymbolSnapshot:
    """Assemble one symbol from already-fetched models. Any of quote/chain
    may be None (the fetch failed — say why in ``errors``); a missing quote
    counts as stale, since there is no fresh spot at all — and so does a
    quote whose timestamp is fresh but which carries no usable price
    (``spot`` None: no trade and no two-sided quote)."""
    now = _utc(now)
    last_close, change_5d_pct, high_20d, low_20d = _bars_summary(bars)
    spot = quote.spot if quote is not None else None
    spot_time = quote.spot_time if quote is not None else None
    return SymbolSnapshot(
        symbol=symbol.strip().upper(),
        spot=spot,
        bid=quote.bid if quote is not None else None,
        ask=quote.ask if quote is not None else None,
        spot_time=spot_time,
        spot_age_seconds=_age(spot_time, now),
        stale=spot is None
        or _is_stale(
            spot_time, now, stale_after_seconds=stale_after_seconds, market_open=market_open
        ),
        last_close=last_close,
        change_5d_pct=change_5d_pct,
        high_20d=high_20d,
        low_20d=low_20d,
        chain=(
            None
            if chain is None
            else chain_summary_from(
                chain,
                enriched,
                now=now,
                stale_after_seconds=stale_after_seconds,
                market_open=market_open,
                strikes_each_side=strikes_each_side,
                expiry_time=expiry_time,
                expiry_timezone=expiry_timezone,
            )
        ),
        headlines=tuple(_headline(item, now) for item in news),
        errors=tuple(errors),
    )


def account_summary_from(state: AccountState) -> AccountSummary:
    """The account as the brain sees it: balances and open positions, no
    account number."""
    return AccountSummary(
        equity=state.equity,
        cash=state.cash,
        buying_power=state.buying_power,
        positions=tuple(
            OpenPosition(
                symbol=p.symbol,
                qty=p.qty,
                side=p.side,
                avg_entry_price=p.avg_entry_price,
                market_value=p.market_value,
                unrealized_pl=p.unrealized_pl,
                current_price=p.current_price,
            )
            for p in state.positions
        ),
        fetched_at=state.fetched_at,
    )


def _symbol_warnings(snapshot: SymbolSnapshot, *, market_open: bool | None) -> list[str]:
    """Per-symbol lines: fetch errors, missing data, and — only while the
    market is open, when it is specific to the symbol rather than implied
    by the market-wide line — stale data."""
    lines = [f"{snapshot.symbol}: {error}" for error in snapshot.errors]
    if snapshot.spot is None:
        lines.append(f"{snapshot.symbol}: spot price missing")
    elif snapshot.stale and market_open is True:
        lines.append(
            f"{snapshot.symbol}: spot quote is STALE ("
            f"{_age_label_seconds(snapshot.spot_age_seconds)})"
        )
    if snapshot.chain is None:
        # A symbol with no expiration in the DTE window already says why
        # above; "missing" would read as a failed fetch.
        if not any(error.startswith(NO_ELIGIBLE_EXPIRATION) for error in snapshot.errors):
            lines.append(f"{snapshot.symbol}: option chain missing")
    elif snapshot.chain.stale and market_open is True:
        lines.append(
            f"{snapshot.symbol}: option chain is STALE ("
            f"{_age_label_seconds(snapshot.chain.quote_age_seconds)})"
        )
    return lines


def _age_label_seconds(age_seconds: float | None) -> str:
    """'age 3m 12s' / 'unknown age' from an age already measured in seconds."""
    if age_seconds is None:
        return "unknown age"
    anchor = datetime(2000, 1, 1, tzinfo=timezone.utc)
    return f"age {format_age(anchor, anchor + timedelta(seconds=age_seconds))}"


def data_age_from(
    clock: MarketClock | None,
    symbols: Sequence[SymbolSnapshot],
    *,
    stale_after_seconds: float,
    errors: Sequence[str] = (),
) -> DataAge:
    """Market status plus the deterministic warning list, in a fixed order:
    market status (closed / clock unknown), snapshot-level fetch ``errors``
    (clock, account) verbatim, per-symbol lines, then the standing
    options-feed note whenever any chain is present."""
    market_open = clock.is_open if clock is not None else None
    lines: list[str] = []
    if clock is None:
        lines.append(CLOCK_UNKNOWN_WARNING)
    elif not clock.is_open:
        next_open = (
            clock.next_open.strftime("%Y-%m-%d %H:%M UTC") if clock.next_open else "unknown"
        )
        lines.append(f"{MARKET_CLOSED_WARNING}; next open {next_open}")
    lines.extend(errors)
    for snapshot in symbols:
        lines.extend(_symbol_warnings(snapshot, market_open=market_open))
    if any(s.chain is not None for s in symbols):
        lines.append(OPTIONS_FEED_NOTE)

    # A stale chain counts even when the spot is fresh (and vice versa).
    any_stale = market_open is not True or any(
        s.stale or (s.chain is not None and s.chain.stale) for s in symbols
    )
    return DataAge(
        market_open=market_open,
        next_open=clock.next_open if clock is not None else None,
        next_close=clock.next_close if clock is not None else None,
        stale_after_seconds=stale_after_seconds,
        any_stale=any_stale,
        warnings=tuple(lines),
    )


# --- live path ----------------------------------------------------------------


def _fetch_symbol(
    symbol: str,
    config: AegisConfig,
    *,
    now: datetime,
    market_open: bool | None,
    stale_after_seconds: float,
) -> SymbolSnapshot:
    """Fetch everything for one symbol, recording each failure instead of raising."""
    errors: list[str] = []
    quote: Quote | None = None
    bars: list[Bar] = []
    chain: ChainSnapshot | None = None
    enriched: list[EnrichedOption] = []
    news: list[NewsItem] = []
    pricing = config.pricing
    snapshot_config = config.brain.snapshot

    try:
        quote = get_spot(symbol)
    except _FETCH_ERRORS as exc:
        errors.append(str(exc))
    try:
        bars = get_bars(symbol)
    except _FETCH_ERRORS as exc:
        errors.append(str(exc))
    eligible = _dte_window(config, now)
    try:
        chain = get_option_chain(symbol, eligible=eligible)
    except NoEligibleExpiration as exc:
        listed = len(exc.listed)
        errors.append(
            _no_eligible_expiration(
                config, f"none of the {listed} listed expiration(s) is in the window"
            )
        )
    except _FETCH_ERRORS as exc:
        errors.append(str(exc))
    if chain is not None and not eligible(chain.expiration):
        # Whatever answered, a chain outside the window is not used: no
        # nearer (or later) expiration stands in for an eligible one.
        days = whole_days_to_expiry(
            chain.expiration,
            now,
            expiry_time=pricing.expiry_time,
            expiry_timezone=pricing.expiry_timezone,
        )
        errors.append(
            _no_eligible_expiration(
                config,
                f"the chain fetched expires {chain.expiration.isoformat()}, at {days} DTE",
            )
        )
        chain = None
    if chain is not None:
        # Only that expiration's contracts go on: a contract the vendor
        # listed under another date (or none) never reaches the thesis stage.
        kept = [c for c in chain.contracts if c.expiration == chain.expiration]
        if len(kept) < len(chain.contracts):
            dropped = len(chain.contracts) - len(kept)
            errors.append(
                f"{dropped} contract(s) in the {chain.expiration.isoformat()} chain expire "
                "on another date or none: left out"
            )
            chain = chain.model_copy(update={"contracts": kept})
    if chain is not None:
        # The chain fetch tolerates a missing underlying quote (chain.spot
        # None); our own spot quote is the fallback for the model IV/Greeks.
        spot = chain.spot if chain.spot is not None else (quote.spot if quote else None)
        try:
            enriched = enrich_chain(
                chain,
                pricing.risk_free_rate,
                spot=spot,
                now=now,
                day_count_basis=pricing.day_count_basis,
                expiry_time=pricing.expiry_time,
                expiry_timezone=pricing.expiry_timezone,
            )
        except PricingError as exc:
            errors.append(str(exc))
    if snapshot_config.headlines_per_symbol > 0:
        try:
            news = get_news(symbol, limit=snapshot_config.headlines_per_symbol)
        except _FETCH_ERRORS as exc:
            errors.append(str(exc))

    return symbol_snapshot_from(
        symbol,
        quote,
        bars,
        chain,
        enriched,
        news,
        now=now,
        stale_after_seconds=stale_after_seconds,
        market_open=market_open,
        strikes_each_side=snapshot_config.strikes_each_side,
        expiry_time=pricing.expiry_time,
        expiry_timezone=pricing.expiry_timezone,
        errors=errors,
    )


def build_market_snapshot(
    symbols: Sequence[str] | None = None,
    *,
    config: AegisConfig | None = None,
    now: datetime | None = None,
) -> MarketSnapshot:
    """Fetch the watchlist (or ``symbols``) through the data layer and
    assemble the scan stage's input.

    Tunables come from ``config`` (default ``get_config()``): the pricing
    block drives enrichment and the DTE count, ``risk_limits.min_dte`` and
    ``brain.snapshot.max_dte`` the expiration window, ``brain.snapshot`` the
    ATM window, headline count and staleness threshold. ``now`` fixes the
    clock for ages, time-to-expiry and the window (default: the current UTC
    time). Raises ``BrainError``
    only when there is nothing to snapshot (an empty symbol list); every
    fetch failure is recorded in the result instead.
    """
    config = config if config is not None else get_config()
    now = _utc(now)
    wanted = config.watchlist if symbols is None else symbols
    cleaned = list(dict.fromkeys(s.strip().upper() for s in wanted if s and s.strip()))
    if not cleaned:
        raise BrainError("build market snapshot", cause=ValueError("no symbols to snapshot"))
    stale_after_seconds = config.brain.snapshot.stale_after_minutes * 60.0

    errors: list[str] = []
    clock: MarketClock | None = None
    try:
        clock = get_market_clock()
    except _FETCH_ERRORS as exc:
        errors.append(f"market clock unavailable: {exc}")
    market_open = clock.is_open if clock is not None else None

    account: AccountSummary | None = None
    try:
        account = account_summary_from(get_account_state())
    except _FETCH_ERRORS as exc:
        errors.append(f"account state unavailable: {exc}")

    symbol_snapshots = tuple(
        _fetch_symbol(
            symbol,
            config,
            now=now,
            market_open=market_open,
            stale_after_seconds=stale_after_seconds,
        )
        for symbol in cleaned
    )
    return MarketSnapshot(
        taken_at=now,
        symbols=symbol_snapshots,
        account=account,
        data_age=data_age_from(
            clock, symbol_snapshots, stale_after_seconds=stale_after_seconds, errors=errors
        ),
        risk_free_rate=config.pricing.risk_free_rate,
    )
