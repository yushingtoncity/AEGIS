"""The market clock: parsing Alpaca's clock payload and ``get_market_clock``
over a fake trading client — canned fixtures, no live API calls."""

from datetime import datetime, timezone

import pytest
from alpaca.common.exceptions import APIError
from requests.exceptions import ConnectionError as RequestsConnectionError

from aegis.data import market
from aegis.data.cache import default_cache
from aegis.data.errors import DataError
from aegis.data.models import MarketClock, utcnow

CLOCK_KEY = ("clock",)


class FakeTradingClient:
    """Stands in for alpaca-py's TradingClient: ``get_clock`` only."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def get_clock(self):
        self.calls += 1
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def fresh_clock_cache():
    """The data layer caches the clock for the quotes TTL; no test may see another's."""
    default_cache.invalidate(CLOCK_KEY)
    yield
    default_cache.invalidate(CLOCK_KEY)


@pytest.fixture
def use_client(monkeypatch):
    def install(*outcomes):
        client = FakeTradingClient(*outcomes)
        monkeypatch.setattr(market, "trading_client", lambda: client)
        return client

    return install


class TestMarketClockModel:
    def test_open_market(self, fixture):
        before = utcnow()
        clock = MarketClock.from_alpaca(fixture("clock.json"))
        assert clock.is_open is True
        # Alpaca reports exchange-local offsets; the model normalises to UTC
        assert clock.next_close == datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
        assert clock.next_open == datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)
        assert clock.next_open.tzinfo is not None and clock.next_close.tzinfo is not None
        assert before <= clock.fetched_at <= utcnow()  # like every other data model

    def test_closed_market(self, fixture):
        clock = MarketClock.from_alpaca(fixture("clock_closed.json"))
        assert clock.is_open is False
        assert clock.next_open == datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)
        assert clock.next_close == datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        assert clock.next_open < clock.next_close  # closed: the open comes first

    def test_explicit_fetched_at_is_kept(self, fixture):
        stamp = datetime(2026, 7, 30, 19, 0, 5, tzinfo=timezone.utc)
        clock = MarketClock.from_alpaca(fixture("clock.json"), fetched_at=stamp)
        assert clock.fetched_at == stamp
        assert clock.age_seconds(now=datetime(2026, 7, 30, 19, 0, 35, tzinfo=timezone.utc)) == 30

    def test_missing_fields_degrade_to_closed_and_none(self):
        clock = MarketClock.from_alpaca({})
        assert clock.is_open is False  # absent is never "open"
        assert clock.next_open is None and clock.next_close is None

    def test_is_frozen(self, fixture):
        clock = MarketClock.from_alpaca(fixture("clock.json"))
        with pytest.raises(Exception):
            clock.is_open = False


class TestGetMarketClock:
    def test_returns_a_typed_clock(self, fixture, use_client):
        client = use_client(fixture("clock.json"))
        clock = market.get_market_clock()
        assert isinstance(clock, MarketClock)
        assert clock.is_open is True
        assert clock.next_close == datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
        assert client.calls == 1

    def test_cached_within_the_quotes_ttl(self, fixture, use_client):
        client = use_client(fixture("clock.json"), fixture("clock_closed.json"))
        first = market.get_market_clock()
        second = market.get_market_clock()
        assert second is first  # one API call, the same frozen model
        assert client.calls == 1
        default_cache.invalidate(CLOCK_KEY)
        assert market.get_market_clock().is_open is False  # the next fetch sees the new state
        assert client.calls == 2

    @pytest.mark.parametrize(
        "error",
        [
            APIError('{"code": 40110000, "message": "request is not authorized"}'),
            RequestsConnectionError("down"),
        ],
        ids=["api-error", "transport-error"],
    )
    def test_failures_are_data_errors_with_context(self, use_client, error):
        use_client(error)
        with pytest.raises(DataError, match="failed to fetch market clock") as info:
            market.get_market_clock()
        assert info.value.what == "market clock"
        assert info.value.symbol is None
        assert info.value.cause is error

    def test_a_failure_is_not_cached(self, fixture, use_client):
        client = use_client(APIError('{"message": "boom"}'), fixture("clock.json"))
        with pytest.raises(DataError):
            market.get_market_clock()
        assert market.get_market_clock().is_open is True
        assert client.calls == 2
