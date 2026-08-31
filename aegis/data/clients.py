"""Lazy singleton constructors for Alpaca API clients.

Clients are created on first use — importing aegis.data never requires
credentials — and cached for the life of the process. Phase 1 is paper-only:
``paper=True`` is hardcoded in the trading client. Live trading, if it ever
exists, enters through a separate ``aegis.execution`` implementation gated
by the Phase 5 policy engine — never by flipping this flag.
"""

from __future__ import annotations

from functools import lru_cache

from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from aegis.config import load_env, require_env


def _credentials() -> tuple[str, str]:
    load_env()
    return require_env("ALPACA_API_KEY"), require_env("ALPACA_SECRET_KEY")


@lru_cache(maxsize=1)
def trading_client() -> TradingClient:
    """Paper trading API: account, clock, option contract listings."""
    api_key, secret_key = _credentials()
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


@lru_cache(maxsize=1)
def stock_client() -> StockHistoricalDataClient:
    """Stock market data: latest quotes/trades, historical bars."""
    api_key, secret_key = _credentials()
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


@lru_cache(maxsize=1)
def option_client() -> OptionHistoricalDataClient:
    """Options market data: chain snapshots, option bars."""
    api_key, secret_key = _credentials()
    return OptionHistoricalDataClient(api_key=api_key, secret_key=secret_key)


@lru_cache(maxsize=1)
def news_client() -> NewsClient:
    """News API."""
    api_key, secret_key = _credentials()
    return NewsClient(api_key=api_key, secret_key=secret_key)
