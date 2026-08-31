"""TTL cache expiry behavior, driven by a fake clock (no sleeping)."""

import pytest

from aegis.data.cache import TTLCache


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    return TTLCache(clock=clock)


def test_get_returns_default_on_miss(cache):
    assert cache.get("missing") is None
    assert cache.get("missing", 42) == 42


def test_set_then_get_within_ttl(cache, clock):
    cache.set("k", "v", ttl_seconds=5)
    clock.advance(4.9)
    assert cache.get("k") == "v"


def test_entry_expires_after_ttl(cache, clock):
    cache.set("k", "v", ttl_seconds=5)
    clock.advance(5)
    assert cache.get("k") is None
    assert len(cache) == 0  # expired entry evicted on read


def test_entries_have_independent_ttls(cache, clock):
    cache.set("fast", 1, ttl_seconds=5)
    cache.set("slow", 2, ttl_seconds=60)
    clock.advance(10)
    assert cache.get("fast") is None
    assert cache.get("slow") == 2


def test_set_refreshes_ttl(cache, clock):
    cache.set("k", "old", ttl_seconds=5)
    clock.advance(4)
    cache.set("k", "new", ttl_seconds=5)
    clock.advance(4)
    assert cache.get("k") == "new"


def test_get_or_fetch_fetches_once(cache, clock):
    calls = []

    def fetch():
        calls.append(1)
        return "value"

    assert cache.get_or_fetch("k", 5, fetch) == "value"
    assert cache.get_or_fetch("k", 5, fetch) == "value"
    assert len(calls) == 1


def test_get_or_fetch_refetches_after_expiry(cache, clock):
    calls = []

    def fetch():
        calls.append(1)
        return len(calls)

    assert cache.get_or_fetch("k", 5, fetch) == 1
    clock.advance(6)
    assert cache.get_or_fetch("k", 5, fetch) == 2
    assert len(calls) == 2


def test_cached_none_is_a_value_not_a_miss(cache):
    calls = []

    def fetch():
        calls.append(1)
        return None

    assert cache.get_or_fetch("k", 5, fetch) is None
    assert cache.get_or_fetch("k", 5, fetch) is None
    assert len(calls) == 1


def test_fetch_exception_propagates_and_caches_nothing(cache):
    def fetch():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cache.get_or_fetch("k", 5, fetch)
    assert cache.get("k") is None


def test_invalidate_and_clear(cache):
    cache.set("a", 1, ttl_seconds=60)
    cache.set("b", 2, ttl_seconds=60)
    cache.invalidate("a")
    cache.invalidate("never-existed")  # no-op, no error
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.clear()
    assert len(cache) == 0


@pytest.mark.parametrize("ttl", [0, -1])
def test_nonpositive_ttl_rejected(cache, ttl):
    with pytest.raises(ValueError):
        cache.set("k", "v", ttl_seconds=ttl)
