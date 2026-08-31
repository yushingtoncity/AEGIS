"""Market data reads: spot quotes, bars, option expirations/chains, market clock.

All fetches go through the shared TTL cache (TTLs from config.yaml) and all
failures surface as DataError with context. Free-plan behavior baked in:

- Stock endpoints default to the SIP feed, which rejects the most recent
  15 minutes without a paid subscription — on that error we retry on the
  free IEX feed.
- Options endpoints silently fall back to the ~15-minute-delayed
  'indicative' feed on free accounts, which is why every model carries
  venue timestamps.
- alpaca-py 0.43.x chain snapshots carry no volume and no open interest;
  open interest is joined in from the trading API's contract listing and
  volume from batched daily option bars (best-effort).
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Sequence

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.requests import (
    OptionBarsRequest,
    OptionChainRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetStatus
from alpaca.trading.requests import GetOptionContractsRequest
from requests.exceptions import RequestException

from aegis.config import get_config
from aegis.data.cache import default_cache
from aegis.data.clients import option_client, stock_client, trading_client
from aegis.data.errors import DataError
from aegis.data.models import (
    Bar,
    ChainSnapshot,
    MarketClock,
    OptionSnapshot,
    Quote,
    as_mapping,
    utcnow,
)

_FETCH_ERRORS = (APIError, RequestException)

_TIMEFRAME_UNITS = {
    "Min": TimeFrameUnit.Minute,
    "Hour": TimeFrameUnit.Hour,
    "Day": TimeFrameUnit.Day,
    "Week": TimeFrameUnit.Week,
    "Month": TimeFrameUnit.Month,
}

_VOLUME_CHUNK = 100  # OCC symbols per option-bars request, to keep URLs sane
_MAX_CONTRACT_PAGES = 10  # 10k contracts/page; hard stop, not a silent cap


def _timeframe(spec: str) -> TimeFrame:
    match = re.fullmatch(r"(\d+)(Min|Hour|Day|Week|Month)", spec)
    if not match:
        raise DataError(f"bars with invalid timeframe {spec!r}")
    try:
        return TimeFrame(int(match.group(1)), _TIMEFRAME_UNITS[match.group(2)])
    except ValueError as exc:  # e.g. the SDK only allows 1Day/1Week
        raise DataError(f"bars with unsupported timeframe {spec!r}", cause=exc) from exc


def _is_subscription_error(error: Exception) -> bool:
    return "subscription" in str(error).lower()


def _latest(request_cls: type, method: Any, symbol: str) -> dict[str, Any] | None:
    """Latest-quote/-trade fetch with a free-plan (IEX) retry."""
    try:
        result = method(request_cls(symbol_or_symbols=symbol))
    except APIError as exc:
        if not _is_subscription_error(exc):
            raise
        result = method(request_cls(symbol_or_symbols=symbol, feed=DataFeed.IEX))
    entry = result.get(symbol)
    return None if entry is None else dict(as_mapping(entry))


def get_spot(symbol: str) -> Quote:
    """Latest quote + last trade for a stock/ETF symbol (cached, quotes TTL)."""
    symbol = symbol.upper()
    ttl = get_config().cache.ttl_seconds.quotes

    def fetch() -> Quote:
        try:
            quote = _latest(
                StockLatestQuoteRequest, stock_client().get_stock_latest_quote, symbol
            )
            trade = _latest(
                StockLatestTradeRequest, stock_client().get_stock_latest_trade, symbol
            )
        except _FETCH_ERRORS as exc:
            raise DataError("spot quote", symbol, exc) from exc
        return Quote.from_alpaca(symbol, quote=quote, trade=trade)

    return default_cache.get_or_fetch(("spot", symbol), ttl, fetch)


def get_bars(
    symbol: str,
    timeframe: str | None = None,
    lookback_days: int | None = None,
) -> list[Bar]:
    """Historical bars, newest last. Defaults come from config.yaml's bars block."""
    symbol = symbol.upper()
    config = get_config()
    tf_spec = timeframe or config.bars.timeframe
    days = lookback_days or config.bars.lookback_days
    ttl = config.cache.ttl_seconds.quotes

    def fetch() -> list[Bar]:
        tf = _timeframe(tf_spec)
        start = utcnow() - timedelta(days=days)

        def request(feed: DataFeed | None) -> Any:
            return stock_client().get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start,
                    **({} if feed is None else {"feed": feed}),
                )
            )

        try:
            try:
                bar_set = request(None)
            except APIError as exc:
                if not _is_subscription_error(exc):
                    raise
                bar_set = request(DataFeed.IEX)
        except _FETCH_ERRORS as exc:
            raise DataError(f"{tf_spec} bars", symbol, exc) from exc
        fetched_at = utcnow()
        return [
            Bar.from_alpaca(symbol, as_mapping(bar), fetched_at=fetched_at)
            for bar in bar_set.data.get(symbol, [])
        ]

    return default_cache.get_or_fetch(("bars", symbol, tf_spec, days), ttl, fetch)


def _fetch_contracts(underlying: str, expiration: date | None) -> list[dict[str, Any]]:
    """Active option contracts from the trading API (paginated)."""
    request_kwargs: dict[str, Any] = {
        "underlying_symbols": [underlying],
        "status": AssetStatus.ACTIVE,
        "limit": 10_000,
    }
    if expiration is None:
        request_kwargs["expiration_date_gte"] = utcnow().date()
    else:
        request_kwargs["expiration_date"] = expiration

    contracts: list[dict[str, Any]] = []
    page_token: str | None = None
    try:
        for _ in range(_MAX_CONTRACT_PAGES):
            response = trading_client().get_option_contracts(
                GetOptionContractsRequest(page_token=page_token, **request_kwargs)
            )
            contracts.extend(dict(as_mapping(c)) for c in response.option_contracts or [])
            page_token = response.next_page_token
            if not page_token:
                break
        else:
            raise DataError(
                f"option contracts (pagination exceeded {_MAX_CONTRACT_PAGES} pages; "
                "refusing to return a truncated listing)",
                underlying,
            )
    except _FETCH_ERRORS as exc:
        raise DataError("option contracts", underlying, exc) from exc
    return contracts


def list_expirations(underlying: str) -> list[date]:
    """Active option expiration dates for an underlying, nearest first."""
    underlying = underlying.upper()
    ttl = get_config().cache.ttl_seconds.chains

    def fetch() -> list[date]:
        contracts = _fetch_contracts(underlying, expiration=None)
        return sorted({c["expiration_date"] for c in contracts})

    return default_cache.get_or_fetch(("expirations", underlying), ttl, fetch)


def _daily_option_volumes(symbols: Sequence[str]) -> dict[str, float]:
    """Most recent daily volume per contract — best-effort, never raises.

    Chain snapshots in alpaca-py 0.43.x have no volume field, so this
    batches daily option bars separately; on failure contracts simply keep
    volume=None.
    """
    volumes: dict[str, float] = {}
    start = utcnow() - timedelta(days=5)
    try:
        for chunk_start in range(0, len(symbols), _VOLUME_CHUNK):
            chunk = list(symbols[chunk_start : chunk_start + _VOLUME_CHUNK])
            bar_set = option_client().get_option_bars(
                OptionBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start
                )
            )
            for occ_symbol, bars in bar_set.data.items():
                if bars:
                    volumes[occ_symbol] = float(bars[-1].volume)
    except _FETCH_ERRORS:
        pass
    return volumes


def get_option_chain(
    underlying: str, expiration: date | str | None = None
) -> ChainSnapshot:
    """Full option chain for one expiration (the nearest available if None).

    Merges three Alpaca sources: chain snapshots (quotes, IV, Greeks),
    the contract listing (open interest, authoritative strike/type), and
    daily option bars (volume, best-effort). Greeks/IV stay None wherever
    Alpaca omits them — that must never crash a caller.
    """
    underlying = underlying.upper()
    if isinstance(expiration, str):
        expiration = date.fromisoformat(expiration)
    ttl = get_config().cache.ttl_seconds.chains

    def fetch() -> ChainSnapshot:
        target = expiration
        if target is None:
            expirations = list_expirations(underlying)
            if not expirations:
                raise DataError("option chain (no active contracts)", underlying)
            target = expirations[0]

        contract_meta = {
            c["symbol"]: c for c in _fetch_contracts(underlying, target)
        }

        try:
            snapshots = option_client().get_option_chain(
                OptionChainRequest(underlying_symbol=underlying, expiration_date=target)
            )
        except _FETCH_ERRORS as exc:
            raise DataError("option chain", underlying, exc) from exc

        volumes = _daily_option_volumes(sorted(snapshots.keys()))

        try:
            spot = get_spot(underlying).spot
        except DataError:
            spot = None  # a chain without the underlying quote is still useful

        fetched_at = utcnow()
        contracts = []
        for occ_symbol, snapshot in snapshots.items():
            meta = contract_meta.get(occ_symbol, {})
            contracts.append(
                OptionSnapshot.from_alpaca(
                    occ_symbol,
                    underlying,
                    as_mapping(snapshot),
                    strike=meta.get("strike_price"),
                    option_type=meta.get("type"),
                    expiration=meta.get("expiration_date"),
                    open_interest=meta.get("open_interest"),
                    volume=volumes.get(occ_symbol),
                    fetched_at=fetched_at,
                )
            )
        contracts.sort(key=lambda c: (c.strike is None, c.strike or 0.0, c.option_type or ""))
        return ChainSnapshot(
            underlying=underlying,
            expiration=target,
            spot=spot,
            contracts=contracts,
            fetched_at=fetched_at,
        )

    cache_key = ("chain", underlying, None if expiration is None else expiration.isoformat())
    return default_cache.get_or_fetch(cache_key, ttl, fetch)


def get_market_clock() -> MarketClock:
    """Whether US equity markets are open, plus the next open/close times."""
    ttl = get_config().cache.ttl_seconds.quotes

    def fetch() -> MarketClock:
        try:
            clock = trading_client().get_clock()
        except _FETCH_ERRORS as exc:
            raise DataError("market clock", None, exc) from exc
        return MarketClock.from_alpaca(as_mapping(clock))

    return default_cache.get_or_fetch(("clock",), ttl, fetch)
