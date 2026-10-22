"""What the forecast stage is shown: live data in, a ``ForecastInputs`` out.

Read once per day and shared by both arms. The market clock is read first
and fails closed: a closed market or a clock that cannot be read means no
forecast run (``MarketNotOpen``), so a forecast is only ever made during a
regular session, before the close that starts its clock.

Every return is computed from split- and dividend-adjusted daily closes of
sessions that ended before today, so a partial bar for the session in
progress never enters them. The live price (the latest IEX trade, else the
quote mid) is shown with its timestamp and age, and today's move is that
price against the last completed close. A fetch that fails is recorded on
the ticker as an error line and the ticker is still asked about; the model
sees "n/a" where data is missing. Headlines published after the inputs were
taken are dropped.

``build_forecast_inputs`` takes its fetchers as arguments (``LIVE``
by default) so tests drive it with fixtures.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from aegis.brain.models import ForecastInputs, ForecastTicker, HeadlineItem
from aegis.config import AegisConfig
from aegis.data.errors import DataError
from aegis.data.market import get_bars, get_market_clock, get_spot
from aegis.data.models import Bar, MarketClock, NewsItem, Quote
from aegis.data.news import get_news
from aegis.scorecard.grade import NEW_YORK, session_close

ADJUSTED = "all"  # split- and dividend-adjusted, the basis every forecast is graded on

RANGE_SESSIONS = 20
"""Sessions behind the range position, the 20-session return and the
volatility: about a trading month, the window the scan stage uses too."""

SHORT_SESSIONS = 5
"""Sessions behind the short-term return: about a trading week, the
forecast's own secondary horizon."""


class MarketNotOpen(Exception):
    """The regular session is not in progress (or the clock could not be
    read): no forecast may be made now."""


@dataclass(frozen=True)
class Fetchers:
    """The data reads ``build_forecast_inputs`` needs, swappable in tests."""

    clock: Callable[[], MarketClock]
    spot: Callable[[str], Quote]
    bars: Callable[[str, int], list[Bar]]
    news: Callable[[str, int], list[NewsItem]]


def _live_bars(symbol: str, lookback_days: int) -> list[Bar]:
    return get_bars(symbol, "1Day", lookback_days=lookback_days, adjustment=ADJUSTED)


def _live_news(symbol: str, limit: int) -> list[NewsItem]:
    return get_news(symbol, limit=limit)


LIVE = Fetchers(clock=get_market_clock, spot=get_spot, bars=_live_bars, news=_live_news)


def completed_sessions(bars: Sequence[Bar], anchor_session: date, now: datetime) -> list[Bar]:
    """Daily bars of sessions before ``anchor_session`` that had closed by
    ``now``, oldest first."""
    return sorted(
        (
            bar
            for bar in bars
            if bar.timestamp.astimezone(NEW_YORK).date() < anchor_session and session_close(bar) <= now
        ),
        key=lambda bar: bar.timestamp,
    )


def _return_pct(closes: Sequence[float], sessions: int) -> float | None:
    """Percent return over the last ``sessions`` sessions, or None without enough closes."""
    if len(closes) <= sessions or closes[-1 - sessions] <= 0:
        return None
    return (closes[-1] / closes[-1 - sessions] - 1.0) * 100.0


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _range_position(bars: Sequence[Bar]) -> float | None:
    if len(bars) < RANGE_SESSIONS:
        return None
    window = bars[-RANGE_SESSIONS:]
    high, low = max(bar.high for bar in window), min(bar.low for bar in window)
    if high <= low:
        return None
    return min(1.0, max(0.0, (window[-1].close - low) / (high - low)))


def _volatility_pct(closes: Sequence[float]) -> float | None:
    if len(closes) <= RANGE_SESSIONS:
        return None
    window = closes[-RANGE_SESSIONS - 1 :]
    returns = [(b / a - 1.0) * 100.0 for a, b in zip(window, window[1:]) if a > 0]
    return statistics.stdev(returns) if len(returns) >= 2 else None


def _age_seconds(when: datetime | None, now: datetime) -> float | None:
    return None if when is None else max(0.0, (now - when).total_seconds())


def _headline(item: NewsItem, now: datetime) -> HeadlineItem:
    return HeadlineItem(
        headline=item.headline,
        summary=item.summary,
        source=item.source,
        published_at=item.published_at,
        age_seconds=_age_seconds(item.published_at, now),
        symbols=tuple(item.symbols),
    )


@dataclass(frozen=True)
class _Raw:
    """One ticker's fetched data before it is summarised."""

    quote: Quote | None
    bars: list[Bar]
    news: list[NewsItem]
    errors: tuple[str, ...]


def _fetch(symbol: str, config: AegisConfig, fetch: Fetchers) -> _Raw:
    errors: list[str] = []
    quote: Quote | None = None
    bars: list[Bar] = []
    news: list[NewsItem] = []
    try:
        quote = fetch.spot(symbol)
    except DataError as exc:
        errors.append(str(exc))
    try:
        bars = fetch.bars(symbol, config.forecast.bars_lookback_days)
    except DataError as exc:
        errors.append(str(exc))
    limit = config.forecast.headlines_per_symbol
    if limit > 0:
        try:
            news = fetch.news(symbol, limit)
        except DataError as exc:
            errors.append(str(exc))
    return _Raw(quote=quote, bars=bars, news=news, errors=tuple(errors))


def summarise_ticker(
    symbol: str,
    role: str,
    raw: _Raw,
    benchmark: _Raw | None,
    *,
    anchor_session: date,
    now: datetime,
    stale_after_seconds: float,
) -> ForecastTicker:
    """One ticker's block, from its fetched data and the benchmark's (None
    when the ticker is the benchmark). Pure."""
    bars = completed_sessions(raw.bars, anchor_session, now)
    closes = [bar.close for bar in bars]
    last_close = closes[-1] if closes else None
    price = raw.quote.spot if raw.quote is not None else None
    price_time = raw.quote.spot_time if raw.quote is not None else None
    age = _age_seconds(price_time, now)
    today = None
    if price is not None and last_close:
        today = (price / last_close - 1.0) * 100.0

    bench_closes: list[float] = []
    bench_today: float | None = None
    if benchmark is not None:
        bench_bars = completed_sessions(benchmark.bars, anchor_session, now)
        bench_closes = [bar.close for bar in bench_bars]
        bench_price = benchmark.quote.spot if benchmark.quote is not None else None
        if bench_price is not None and bench_closes:
            bench_today = (bench_price / bench_closes[-1] - 1.0) * 100.0

    def rel(sessions: int) -> float | None:
        if benchmark is None:
            return None
        return _minus(_return_pct(closes, sessions), _return_pct(bench_closes, sessions))

    headlines = tuple(
        _headline(item, now)
        for item in raw.news
        if item.published_at is None or item.published_at <= now
    )
    return ForecastTicker(
        symbol=symbol,
        role=role,  # type: ignore[arg-type]
        price=price,
        price_time=price_time,
        price_age_seconds=age,
        stale=age is None or age > stale_after_seconds,
        last_close=last_close,
        last_close_session=bars[-1].timestamp.astimezone(NEW_YORK).date() if bars else None,
        today_pct=today,
        rel_today_pct=_minus(today, bench_today) if benchmark is not None else None,
        return_1d_pct=_return_pct(closes, 1),
        return_5d_pct=_return_pct(closes, SHORT_SESSIONS),
        return_20d_pct=_return_pct(closes, RANGE_SESSIONS),
        rel_5d_pct=rel(SHORT_SESSIONS),
        rel_20d_pct=rel(RANGE_SESSIONS),
        range_20d_position=_range_position(bars),
        volatility_20d_pct=_volatility_pct(closes),
        headlines=headlines,
        errors=raw.errors,
    )


def build_forecast_inputs(
    config: AegisConfig, *, now: datetime, fetch: Fetchers = LIVE
) -> ForecastInputs:
    """Read the clock, then every ticker; ``MarketNotOpen`` unless the
    regular session is in progress (a clock that cannot be read counts as
    closed)."""
    try:
        clock = fetch.clock()
    except DataError as exc:
        raise MarketNotOpen(f"market clock unreadable, so no forecast: {exc}") from exc
    if not clock.is_open:
        raise MarketNotOpen("the market is closed: forecasts are made only during a regular session")
    if clock.next_close is not None and clock.next_close <= now:
        raise MarketNotOpen("the session's close has passed: no forecast after the close")
    anchor_session = now.astimezone(NEW_YORK).date()
    forecast = config.forecast
    stale_after = forecast.stale_after_minutes * 60.0
    benchmark_raw = _fetch(forecast.benchmark, config, fetch)
    tickers = [
        summarise_ticker(
            symbol,
            "stock",
            _fetch(symbol, config, fetch),
            benchmark_raw,
            anchor_session=anchor_session,
            now=now,
            stale_after_seconds=stale_after,
        )
        for symbol in forecast.universe
    ]
    tickers.append(
        summarise_ticker(
            forecast.benchmark,
            "benchmark",
            benchmark_raw,
            None,
            anchor_session=anchor_session,
            now=now,
            stale_after_seconds=stale_after,
        )
    )
    return ForecastInputs(
        taken_at=now,
        anchor_session=anchor_session,
        benchmark=forecast.benchmark,
        tickers=tuple(tickers),
    )


__all__ = [
    "LIVE",
    "Fetchers",
    "MarketNotOpen",
    "build_forecast_inputs",
    "completed_sessions",
    "summarise_ticker",
]
