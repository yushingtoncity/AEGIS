"""Typed models returned by the data layer.

Everything the rest of AEGIS sees about the outside world is one of these
models — raw API JSON never leaves ``aegis.data``. Every model carries a
``fetched_at`` UTC timestamp, plus venue timestamps where the feed provides
them, because free-plan Alpaca options data is ~15 minutes delayed and
staleness must always be visible.

The ``from_alpaca`` classmethods accept plain mappings shaped like alpaca-py
0.43.x SDK models (long field names: ``bid_price``, ``ask_price``, ...).
The live path feeds them SDK objects dumped via ``as_mapping``; tests feed
them canned JSON fixtures. Optional data that Alpaca omits (Greeks and IV
for near-dated or unquoted contracts, especially) maps to ``None`` — parsing
absent optional fields never raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_mapping(obj: Any) -> Mapping[str, Any]:
    """Dump an alpaca-py SDK model (pydantic) to a plain mapping."""
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"cannot convert {type(obj).__name__} to a mapping")


def format_age(timestamp: datetime | None, now: datetime | None = None) -> str:
    """Human-readable age of a timestamp: '4s', '3m 12s', '2h 05m', '3d 02h'."""
    if timestamp is None:
        return "unknown age"
    seconds = max(0, int(((now or utcnow()) - timestamp).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def _to_utc(value: Any) -> datetime | None:
    """Coerce an ISO string or datetime to an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError(f"cannot interpret {value!r} as a timestamp")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _enum_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _get(payload: Mapping[str, Any] | None, key: str) -> Any:
    return payload.get(key) if payload else None


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


def parse_occ_symbol(symbol: str) -> tuple[str, date, OptionType, float]:
    """Split an OCC option symbol (e.g. SPY260821C00640000) into
    (root, expiration, type, strike).

    The trailing 15 characters are fixed-width: YYMMDD + C/P + strike*1000;
    everything before them is the root symbol.
    """
    symbol = symbol.strip().upper()
    if len(symbol) < 16:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, tail = symbol[:-15], symbol[-15:]
    type_char = tail[6]
    if type_char not in ("C", "P") or not tail[:6].isdigit() or not tail[7:].isdigit():
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    try:
        expiration = datetime.strptime(tail[:6], "%y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"bad expiration in OCC symbol {symbol!r}") from exc
    option_type = OptionType.CALL if type_char == "C" else OptionType.PUT
    return root, expiration, option_type, int(tail[7:]) / 1000.0


class FetchedModel(BaseModel):
    """Base for all data-layer models: records when WE fetched the data.

    Venue timestamps (quote_time etc.) say how fresh the data is at the
    source; fetched_at says how fresh our copy is.
    """

    model_config = ConfigDict(frozen=True)

    fetched_at: datetime = Field(default_factory=utcnow)

    def age_seconds(self, now: datetime | None = None) -> float:
        return ((now or utcnow()) - self.fetched_at).total_seconds()


class Quote(FetchedModel):
    """Latest quote and last trade for an equity symbol."""

    symbol: str
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    last: float | None = None
    quote_time: datetime | None = None
    last_time: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @property
    def spot(self) -> float | None:
        """Best available price: last trade, else quote midpoint."""
        return self.last if self.last is not None else self.mid

    @property
    def spot_time(self) -> datetime | None:
        return self.last_time if self.last is not None else self.quote_time

    @classmethod
    def from_alpaca(
        cls,
        symbol: str,
        quote: Mapping[str, Any] | None = None,
        trade: Mapping[str, Any] | None = None,
        fetched_at: datetime | None = None,
    ) -> "Quote":
        """Build from alpaca-py Quote/Trade shaped mappings.

        IEX quotes outside market hours report 0.0 bid/ask; a real equity
        never quotes at exactly 0, so zeros map to None here. (Option quotes
        keep zero bids — see OptionSnapshot.)
        """

        def _price(value: Any) -> float | None:
            price = _to_float(value)
            return None if price == 0 else price

        return cls(
            symbol=symbol.upper(),
            bid=_price(_get(quote, "bid_price")),
            ask=_price(_get(quote, "ask_price")),
            bid_size=_to_float(_get(quote, "bid_size")),
            ask_size=_to_float(_get(quote, "ask_size")),
            last=_to_float(_get(trade, "price")),
            quote_time=_to_utc(_get(quote, "timestamp")),
            last_time=_to_utc(_get(trade, "timestamp")),
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )


class Bar(FetchedModel):
    """One OHLCV bar."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    @classmethod
    def from_alpaca(
        cls,
        symbol: str,
        payload: Mapping[str, Any],
        fetched_at: datetime | None = None,
    ) -> "Bar":
        trade_count = payload.get("trade_count")
        return cls(
            symbol=symbol.upper(),
            timestamp=_to_utc(payload["timestamp"]),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload["volume"]),
            vwap=_to_float(payload.get("vwap")),
            trade_count=None if trade_count is None else int(trade_count),
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )


class OptionSnapshot(FetchedModel):
    """One option contract's market state.

    Greeks and IV are None whenever Alpaca omits them — common for
    near-dated, deep-OTM, or unquoted contracts (Phase 2 adds our own
    computed fallback). volume and open_interest come from separate Alpaca
    endpoints and may likewise be None. A 0.0 bid is meaningful for options
    and is preserved, unlike equity quotes.
    """

    symbol: str
    underlying: str
    expiration: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    implied_vol: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    quote_time: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    @classmethod
    def from_alpaca(
        cls,
        symbol: str,
        underlying: str,
        payload: Mapping[str, Any] | None = None,
        *,
        strike: Any = None,
        option_type: Any = None,
        expiration: Any = None,
        open_interest: Any = None,
        volume: Any = None,
        fetched_at: datetime | None = None,
    ) -> "OptionSnapshot":
        """Build from an alpaca-py OptionsSnapshot-shaped mapping.

        Contract metadata (strike/type/expiration/open interest, from the
        trading API's contract listing) is passed via keywords and wins over
        parsing the OCC symbol, which serves as the fallback.
        """
        symbol = symbol.strip().upper()
        if strike is None or option_type is None or expiration is None:
            try:
                _, occ_exp, occ_type, occ_strike = parse_occ_symbol(symbol)
            except ValueError:
                occ_exp = occ_type = occ_strike = None
            strike = strike if strike is not None else occ_strike
            option_type = option_type if option_type is not None else occ_type
            expiration = expiration if expiration is not None else occ_exp
        type_str = _enum_str(option_type)
        quote = _get(payload, "latest_quote")
        trade = _get(payload, "latest_trade")
        greeks = _get(payload, "greeks")
        return cls(
            symbol=symbol,
            underlying=underlying.upper(),
            expiration=_to_date(expiration),
            strike=_to_float(strike),
            option_type=None if type_str is None else OptionType(type_str.lower()),
            bid=_to_float(_get(quote, "bid_price")),
            ask=_to_float(_get(quote, "ask_price")),
            last=_to_float(_get(trade, "price")),
            volume=_to_float(volume),
            open_interest=_to_float(open_interest),
            implied_vol=_to_float(_get(payload, "implied_volatility")),
            delta=_to_float(_get(greeks, "delta")),
            gamma=_to_float(_get(greeks, "gamma")),
            theta=_to_float(_get(greeks, "theta")),
            vega=_to_float(_get(greeks, "vega")),
            rho=_to_float(_get(greeks, "rho")),
            quote_time=_to_utc(_get(quote, "timestamp")),
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )


class ChainSnapshot(FetchedModel):
    """A full option chain for one underlying and one expiration."""

    underlying: str
    expiration: date
    spot: float | None = None
    contracts: list[OptionSnapshot] = Field(default_factory=list)

    def _side(self, option_type: OptionType) -> list[OptionSnapshot]:
        side = [
            c
            for c in self.contracts
            if c.option_type is option_type and c.strike is not None
        ]
        return sorted(side, key=lambda c: c.strike)

    @property
    def calls(self) -> list[OptionSnapshot]:
        return self._side(OptionType.CALL)

    @property
    def puts(self) -> list[OptionSnapshot]:
        return self._side(OptionType.PUT)

    @property
    def strikes(self) -> list[float]:
        return sorted({c.strike for c in self.contracts if c.strike is not None})

    @property
    def latest_quote_time(self) -> datetime | None:
        """The newest venue quote timestamp in the chain — the best measure
        of how delayed the feed is."""
        times = [c.quote_time for c in self.contracts if c.quote_time is not None]
        return max(times) if times else None

    def atm_strike(self) -> float | None:
        if self.spot is None or not self.strikes:
            return None
        return min(self.strikes, key=lambda s: abs(s - self.spot))

    def atm_strikes(self, each_side: int = 5) -> list[float]:
        """Strikes centered on the one nearest spot (ATM ± each_side).

        Without a spot price, falls back to the lowest strikes so callers
        still get something to display.
        """
        strikes = self.strikes
        atm = self.atm_strike()
        if atm is None:
            return strikes[: 2 * each_side + 1]
        center = strikes.index(atm)
        return strikes[max(0, center - each_side) : center + each_side + 1]

    def contract_at(
        self, strike: float, option_type: OptionType
    ) -> OptionSnapshot | None:
        for contract in self.contracts:
            if (
                contract.option_type is option_type
                and contract.strike is not None
                and abs(contract.strike - strike) < 1e-6
            ):
                return contract
        return None


class Position(BaseModel):
    """One open position (numeric fields converted from Alpaca's strings)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    qty: float
    side: str
    avg_entry_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None
    current_price: float | None = None

    @classmethod
    def from_alpaca(cls, payload: Mapping[str, Any]) -> "Position":
        return cls(
            symbol=str(payload["symbol"]).upper(),
            qty=_to_float(payload.get("qty")) or 0.0,
            side=(_enum_str(payload.get("side")) or "").lower(),
            avg_entry_price=_to_float(payload.get("avg_entry_price")),
            market_value=_to_float(payload.get("market_value")),
            unrealized_pl=_to_float(payload.get("unrealized_pl")),
            current_price=_to_float(payload.get("current_price")),
        )


class AccountState(FetchedModel):
    """Paper account status. Alpaca sends monetary fields as strings; they
    are converted to floats here (None when absent)."""

    account_number: str
    status: str
    currency: str | None = None
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    portfolio_value: float | None = None
    positions: list[Position] = Field(default_factory=list)

    @property
    def masked_account_number(self) -> str:
        """All but the last four characters hidden — safe to print/log."""
        return "****" + self.account_number[-4:] if self.account_number else "****"

    @classmethod
    def from_alpaca(
        cls,
        payload: Mapping[str, Any],
        positions: list[Mapping[str, Any]] | None = None,
        fetched_at: datetime | None = None,
    ) -> "AccountState":
        return cls(
            account_number=str(payload.get("account_number") or ""),
            status=_enum_str(payload.get("status")) or "UNKNOWN",
            currency=payload.get("currency"),
            equity=_to_float(payload.get("equity")),
            cash=_to_float(payload.get("cash")),
            buying_power=_to_float(payload.get("buying_power")),
            portfolio_value=_to_float(payload.get("portfolio_value")),
            positions=[Position.from_alpaca(p) for p in positions or []],
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )


class NewsItem(FetchedModel):
    """One news article."""

    id: int | str
    headline: str
    source: str | None = None
    url: str | None = None
    summary: str | None = None
    symbols: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_alpaca(
        cls,
        payload: Mapping[str, Any],
        fetched_at: datetime | None = None,
    ) -> "NewsItem":
        return cls(
            id=payload.get("id") or "",
            headline=str(payload.get("headline") or "").strip(),
            source=payload.get("source") or None,
            url=payload.get("url") or None,
            summary=str(payload.get("summary") or "").strip() or None,
            symbols=[str(s).upper() for s in payload.get("symbols") or []],
            published_at=_to_utc(payload.get("created_at")),
            updated_at=_to_utc(payload.get("updated_at")),
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )


class MarketClock(FetchedModel):
    """US equity market clock (from the trading API)."""

    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None

    @classmethod
    def from_alpaca(
        cls,
        payload: Mapping[str, Any],
        fetched_at: datetime | None = None,
    ) -> "MarketClock":
        return cls(
            is_open=bool(payload.get("is_open")),
            next_open=_to_utc(payload.get("next_open")),
            next_close=_to_utc(payload.get("next_close")),
            **({} if fetched_at is None else {"fetched_at": fetched_at}),
        )
