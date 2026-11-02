"""Client-side rate limiting per source.

Two caps from ``signals.sources.<name>``: ``requests_per_minute`` is counted
in this process (a cycle is one process), ``requests_per_day`` from the
store's ``signal_fetches`` rows since the UTC midnight, so it holds across
the separate launchd runs. ``None`` means the provider states no limit.
Either cap reached is a ``rate_limited`` SourceError before anything is sent.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta

from aegis.config import SignalSourceConfig
from aegis.data.models import utcnow
from aegis.data.signals import SourceError
from aegis.store.signals import requests_sent

_MINUTE = timedelta(minutes=1)


class RateLimiter:
    def __init__(
        self,
        conn: sqlite3.Connection,
        source: str,
        config: SignalSourceConfig,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._conn = conn
        self._source = source
        self._config = config
        self._clock = clock
        self._sent: deque[datetime] = deque()

    def acquire(self) -> None:
        """Allow one request now, or raise ``SourceError('rate_limited')``."""
        now = self._clock()
        per_minute = self._config.requests_per_minute
        while self._sent and now - self._sent[0] >= _MINUTE:
            self._sent.popleft()
        if per_minute is not None and len(self._sent) >= per_minute:
            raise SourceError("rate_limited", f"limiter: {per_minute} requests a minute reached")
        per_day = self._config.requests_per_day
        if per_day is not None:
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if requests_sent(self._conn, self._source, midnight) >= per_day:
                raise SourceError("rate_limited", f"limiter: {per_day} requests a day reached")
        self._sent.append(now)
