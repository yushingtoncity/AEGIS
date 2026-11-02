"""The ``Source`` interface and one adapter per provider.

A source names the subjects it fetches for a set of tickers and topics, and
fetches one subject at a time. ``fetch`` returns normalized ``Signal``
models or raises ``SourceError``; anything else it raises is caught by the
collector and recorded as an error too. Sources never touch the store.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from aegis.config import SignalsConfig
from aegis.data.apewisdom import APEWISDOM_BASE, APEWISDOM_FILTER, apewisdom_signals
from aegis.data.news import NEWS_ENDPOINT, news_signals
from aegis.data.signals import Signal, SignalSource

MARKET_WIDE_NEWS_LIMIT = 50
"""Alpaca's page cap; a topic is matched against the latest 50 market-wide headlines."""


class Source(Protocol):
    name: SignalSource
    endpoint: str

    def subjects(self, tickers: Sequence[str], topics: Sequence[str]) -> list[str]: ...

    def fetch(self, subject: str) -> list[Signal]: ...


class NewsSource:
    """Alpaca News: a ticker's own headlines; a topic is a keyword matched
    against the market-wide feed (Alpaca has no topic search)."""

    name = SignalSource.NEWS
    endpoint = NEWS_ENDPOINT

    def __init__(self, config: SignalsConfig) -> None:
        self._source = config.sources.news
        self._topics = set(config.topics)

    def subjects(self, tickers: Sequence[str], topics: Sequence[str]) -> list[str]:
        return [*tickers, *topics]

    def fetch(self, subject: str) -> list[Signal]:
        if subject not in self._topics:
            return news_signals(
                subject,
                limit=self._source.per_subject,
                confidence=self._source.confidence,
                market_wide=False,
            )
        feed = news_signals(
            subject, limit=MARKET_WIDE_NEWS_LIMIT, confidence=self._source.confidence, market_wide=True
        )
        needle = subject.casefold()
        matched = [s for s in feed if needle in f"{s.headline} {s.summary or ''}".casefold()]
        return matched[: self._source.per_subject]


class ApeWisdomSource:
    """ApeWisdom: one request covers every ticker, so its one subject is the
    filter name and its signals carry each ticker as their own subject."""

    name = SignalSource.APEWISDOM
    endpoint = f"{APEWISDOM_BASE}/{APEWISDOM_FILTER}"

    def __init__(self, config: SignalsConfig, tickers: Sequence[str]) -> None:
        self._source = config.sources.apewisdom
        self._tickers = tuple(tickers)

    def subjects(self, tickers: Sequence[str], topics: Sequence[str]) -> list[str]:
        return [APEWISDOM_FILTER] if tickers else []

    def fetch(self, subject: str) -> list[Signal]:
        return apewisdom_signals(
            self._tickers,
            pages=self._source.pages,
            confidence=self._source.confidence,
            timeout=self._source.timeout_seconds,
        )


def build_sources(config: SignalsConfig, tickers: Sequence[str]) -> list[Source]:
    """The enabled sources, in config order."""
    sources: list[Source] = []
    if config.sources.news.enabled:
        sources.append(NewsSource(config))
    if config.sources.apewisdom.enabled:
        sources.append(ApeWisdomSource(config, tickers))
    return sources
