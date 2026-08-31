"""News reads."""

from __future__ import annotations

from typing import Sequence

from alpaca.common.exceptions import APIError
from alpaca.data.requests import NewsRequest
from requests.exceptions import RequestException

from aegis.config import get_config
from aegis.data.cache import default_cache
from aegis.data.clients import news_client
from aegis.data.errors import DataError
from aegis.data.models import NewsItem, as_mapping, utcnow


def get_news(
    symbols: str | Sequence[str] | None = None, limit: int = 5
) -> list[NewsItem]:
    """Latest headlines, newest first (cached, news TTL).

    symbols may be one symbol, a sequence of symbols, or None for
    market-wide news. Alpaca's news API takes a comma-separated string and
    caps pages at 50 items; limit is clamped to 1..50.
    """
    if symbols is None:
        symbols_arg = None
    elif isinstance(symbols, str):
        symbols_arg = symbols.upper()
    else:
        symbols_arg = ",".join(s.upper() for s in symbols)
    limit = max(1, min(int(limit), 50))
    ttl = get_config().cache.ttl_seconds.news

    def fetch() -> list[NewsItem]:
        try:
            news_set = news_client().get_news(
                NewsRequest(symbols=symbols_arg, limit=limit, sort="desc")
            )
        except (APIError, RequestException) as exc:
            raise DataError("news", symbols_arg, exc) from exc
        fetched_at = utcnow()
        return [
            NewsItem.from_alpaca(as_mapping(item), fetched_at=fetched_at)
            for item in news_set.data.get("news", [])
        ]

    return default_cache.get_or_fetch(("news", symbols_arg, limit), ttl, fetch)
