"""``build_forecast_inputs`` with fixture fetchers: the market clock fails
closed, only completed sessions feed the returns, late headlines are dropped,
and a failed fetch is recorded on the ticker instead of sinking the run."""

from datetime import datetime, timedelta, timezone

import pytest

from aegis.data.errors import DataError
from aegis.data.models import MarketClock, NewsItem, Quote
from aegis.forecasts.inputs import Fetchers, MarketNotOpen, build_forecast_inputs, completed_sessions
from forecast_factories import ANCHOR, NOW, bar, forecast_config, sessions

CLOSE_TODAY = datetime(2026, 10, 21, 20, 0, tzinfo=timezone.utc)
HISTORY = [day for day in sessions(ANCHOR - timedelta(days=60), 60) if day < ANCHOR][-30:]


def _closes(symbol: str, start: float, step: float) -> list:
    """One bar per history session, rising by ``step`` a session, plus a
    partial bar for today that must never be used."""
    bars = [bar(symbol, day, start + step * i) for i, day in enumerate(HISTORY)]
    bars.append(bar(symbol, ANCHOR, 9999.0))
    return bars


def _fetchers(*, clock=None, spot=None, bars=None, news=None) -> Fetchers:
    prices = {"AAPL": 130.0, "MSFT": 101.0, "SPY": 515.0}

    def default_spot(symbol):
        return Quote(symbol=symbol, last=prices[symbol], last_time=NOW - timedelta(seconds=30))

    series = {"AAPL": _closes("AAPL", 100.0, 1.0), "MSFT": _closes("MSFT", 100.0, 0.0),
              "SPY": _closes("SPY", 500.0, 0.5)}

    return Fetchers(
        clock=clock or (lambda: MarketClock(is_open=True, next_close=CLOSE_TODAY)),
        spot=spot or default_spot,
        bars=bars or (lambda symbol, lookback: series[symbol]),
        news=news or (lambda symbol, limit: []),
    )


class TestTheClockFailsClosed:
    def test_a_closed_market_makes_nothing(self):
        fetch = _fetchers(clock=lambda: MarketClock(is_open=False))
        with pytest.raises(MarketNotOpen, match="closed"):
            build_forecast_inputs(forecast_config(), now=NOW, fetch=fetch)

    def test_an_unreadable_clock_counts_as_closed(self):
        def broken():
            raise DataError("market clock")

        with pytest.raises(MarketNotOpen, match="unreadable"):
            build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers(clock=broken))

    def test_nothing_after_the_close_even_if_the_clock_lags(self):
        fetch = _fetchers(clock=lambda: MarketClock(is_open=True, next_close=NOW - timedelta(minutes=1)))
        with pytest.raises(MarketNotOpen, match="close has passed"):
            build_forecast_inputs(forecast_config(), now=NOW, fetch=fetch)


class TestInputs:
    def test_universe_in_order_then_the_benchmark(self):
        inputs = build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers())
        assert inputs.symbols == ("AAPL", "MSFT", "SPY")
        assert [t.role for t in inputs.tickers] == ["stock", "stock", "benchmark"]
        assert inputs.anchor_session == ANCHOR and inputs.taken_at == NOW

    def test_returns_use_completed_sessions_only(self):
        aapl, msft, spy = build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers()).tickers
        assert aapl.last_close == 129.0 and aapl.last_close_session == HISTORY[-1]  # never today's 9999
        assert aapl.return_1d_pct == pytest.approx((129 / 128 - 1) * 100)
        assert aapl.return_5d_pct == pytest.approx((129 / 124 - 1) * 100)
        assert aapl.today_pct == pytest.approx((130 / 129 - 1) * 100)
        spy_5d = (514.5 / 512.0 - 1) * 100
        assert msft.return_5d_pct == 0.0 and msft.rel_5d_pct == pytest.approx(-spy_5d)
        assert aapl.range_20d_position == 1.0 and msft.volatility_20d_pct == 0.0
        assert spy.rel_5d_pct is None and spy.rel_today_pct is None

    def test_a_bar_whose_session_has_not_closed_is_left_out(self):
        late = datetime(2026, 10, 20, 19, 0, tzinfo=timezone.utc)  # 15:00 New York on the 20th
        bars = [bar("AAPL", day, 1.0) for day in HISTORY]
        assert completed_sessions(bars, ANCHOR, late)[-1].timestamp < bars[-1].timestamp

    def test_fresh_and_stale_prices_are_marked(self):
        def spot(symbol):
            age = timedelta(minutes=40) if symbol == "MSFT" else timedelta(seconds=10)
            return Quote(symbol=symbol, last=100.0, last_time=NOW - age)

        aapl, msft, _ = build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers(spot=spot)).tickers
        assert not aapl.stale and msft.stale and msft.price_age_seconds == 2400

    def test_headlines_after_the_inputs_were_taken_are_dropped(self):
        def news(symbol, limit):
            return [
                NewsItem(id=1, headline="before", published_at=NOW - timedelta(hours=1)),
                NewsItem(id=2, headline="after", published_at=NOW + timedelta(minutes=5)),
            ]

        aapl = build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers(news=news)).tickers[0]
        assert [h.headline for h in aapl.headlines] == ["before"]

    def test_a_failed_fetch_is_an_error_line_and_the_ticker_is_still_asked(self):
        def spot(symbol):
            if symbol == "MSFT":
                raise DataError("latest quote", "MSFT")
            return Quote(symbol=symbol, last=100.0, last_time=NOW)

        inputs = build_forecast_inputs(forecast_config(), now=NOW, fetch=_fetchers(spot=spot))
        msft = inputs.tickers[1]
        assert msft.price is None and msft.stale and msft.errors == ("failed to fetch latest quote for MSFT",)
        assert msft.return_5d_pct == 0.0  # the bars still came through

    def test_no_headline_fetch_when_the_limit_is_zero(self):
        def news(symbol, limit):
            raise AssertionError("news must not be fetched")

        build_forecast_inputs(forecast_config(headlines_per_symbol=0), now=NOW, fetch=_fetchers(news=news))
