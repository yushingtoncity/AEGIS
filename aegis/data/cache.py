"""Simple in-memory TTL cache.

Sits in front of every external fetch in the data layer so repeated reads
inside a TTL window (TTLs come from config.yaml's cache block) cost zero
API calls — free-tier Alpaca allows ~200 requests/minute. Cached values are
frozen pydantic models, so sharing them between callers is safe.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

_MISSING = object()


class TTLCache:
    """Thread-safe key -> value store where each entry carries its own TTL.

    ``clock`` is injectable (defaults to time.monotonic) so tests can drive
    expiry without sleeping.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return default
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._entries[key]
                return default
            return value

    def set(self, key: Any, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        with self._lock:
            self._entries[key] = (self._clock() + ttl_seconds, value)

    def get_or_fetch(
        self, key: Any, ttl_seconds: float, fetch: Callable[[], Any]
    ) -> Any:
        """Return the cached value, or call ``fetch`` and cache its result.

        ``fetch`` runs outside the lock — a slow network call must not block
        readers of other keys. A cached None is a real value, not a miss.
        """
        value = self.get(key, _MISSING)
        if value is _MISSING:
            value = fetch()
            self.set(key, value, ttl_seconds)
        return value

    def invalidate(self, key: Any) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


default_cache = TTLCache()
"""Shared instance used by the market/account/news modules."""
