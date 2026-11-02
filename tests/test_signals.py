"""The signals module: sources on recorded fixtures, the store, rate limiting,
dedupe, decay, the digest, collection with failing sources, config, the scan
prompt and the cycle. No test opens a socket: HTTP goes through a fake
session and Alpaca News through a monkeypatched ``get_news``.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
import requests

from aegis.brain.cycle import SIGNALS_DIGEST_EVENT, run_cycle
from aegis.brain.models import MarketSnapshot
from aegis.brain.prompts import (
    INJECTION_GUARD,
    NEWS_DELIM_CLOSE,
    NEWS_DELIM_OPEN,
    SIGNALS_HEADER,
    build_scan_prompt,
    scan_prompt_version,
)
from aegis.brain.scan import run_scan
from aegis.brain.testing import FakeLLM
from aegis.config import AegisConfig, SignalSourceConfig, SignalsConfig, load_config
from aegis.data import news as news_module
from aegis.data.apewisdom import APEWISDOM_FILTER, apewisdom_signals, rows_to_signals
from aegis.data.models import NewsItem
from aegis.data.news import news_signals, news_to_signals
from aegis.data.signals import Signal, SignalKind, SignalSource, SourceError, fetch_json
from aegis.signals.collect import collect_signals, signals_active, to_records
from aegis.signals.digest import build_digest, clean_text
from aegis.signals.rank import canonical_url, dedupe, rank_signals, recency, title_key
from aegis.signals.ratelimit import RateLimiter
from aegis.signals.source import ApeWisdomSource, NewsSource, build_sources
from aegis.store.db import open_store
from aegis.store.models import SignalFetch, SignalFetchStatus, SignalRecord
from aegis.store.repo import get_recent_events
from aegis.store.signals import (
    get_cycle_fetches,
    get_fetch_signals,
    get_signals,
    latest_ok_fetch,
    record_fetch,
    requests_sent,
)

NOW = datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc)
SIGNALS_ON = SignalsConfig(enabled=True)


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "aegis.db")
    yield connection
    connection.close()


@pytest.fixture
def snapshot(fixture) -> MarketSnapshot:
    return MarketSnapshot.model_validate(fixture("brain/market_snapshot.json"))


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body
        self._text = text

    def json(self):
        if self._text is not None:
            return json.loads(self._text)
        return self._body


class FakeSession:
    """Serves canned responses (or raises canned errors) in order and records each call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def record(**overrides) -> SignalRecord:
    fields = {
        "fetch_id": "f-1",
        "source": "news",
        "kind": "news",
        "subject": "NVDA",
        "source_item_id": "item",
        "endpoint": "e",
        "published_at": NOW - timedelta(hours=1),
        "fetched_at": NOW,
        "headline": "Chipmakers rally on data-center demand",
        "engagement": 0.0,
        "confidence": 0.6,
    }
    fields.update(overrides)
    return SignalRecord(**fields)


class FakeSource:
    """A ``Source`` serving canned signals per subject (or raising), counting calls."""

    def __init__(self, name=SignalSource.NEWS, by_subject=None, error=None, subjects=None):
        self.name = name
        self.endpoint = f"fake:{name.value}"
        self.by_subject = by_subject or {}
        self.error = error
        self._subjects = subjects
        self.calls = []

    def subjects(self, tickers, topics):
        return list(self._subjects) if self._subjects is not None else [*tickers, *topics]

    def fetch(self, subject):
        self.calls.append(subject)
        if self.error is not None:
            raise self.error
        return list(self.by_subject.get(subject, []))


def signal(subject="NVDA", item="n-1", headline="Nvidia beats estimates", **overrides) -> Signal:
    fields = {
        "source": SignalSource.NEWS,
        "kind": SignalKind.NEWS,
        "subject": subject,
        "source_item_id": item,
        "endpoint": "fake:news",
        "published_at": NOW - timedelta(hours=1),
        "headline": headline,
        "confidence": 0.6,
        "fetched_at": NOW,
    }
    fields.update(overrides)
    return Signal(**fields)


# --- sources --------------------------------------------------------------------


class TestApeWisdom:
    def test_rows_become_unscored_mention_signals_for_watched_tickers_only(self, fixture):
        rows = fixture("signals/apewisdom_page1.json")["results"]
        got = rows_to_signals(rows, ["nvda", "AAPL", "MSFT", "SPY"], confidence=0.3, fetched_at=NOW, endpoint="e")
        assert [s.subject for s in got] == ["NVDA", "AAPL"]  # TSLA unwatched; MSFT row has no count
        nvda, aapl = got
        assert nvda.kind is SignalKind.MENTIONS and nvda.source is SignalSource.APEWISDOM
        assert nvda.sentiment is None and nvda.engagement == 812.0 and nvda.confidence == 0.3
        assert nvda.headline == "812 Reddit mentions in 24h (a day earlier 540), rank 1 (a day earlier 2)"
        assert aapl.engagement == 233.0  # numeric strings are read as numbers
        assert nvda.raw["upvotes"] == 9450 and nvda.published_at == NOW
        assert nvda.source_item_id == f"{APEWISDOM_FILTER}:NVDA:20261030T1500"

    def test_client_reads_pages_until_the_last_and_finds_a_ticker_below_page_one(self, fixture):
        session = FakeSession(
            FakeResponse(body=fixture("signals/apewisdom_page1.json")),
            FakeResponse(body={**fixture("signals/apewisdom_page2.json"), "pages": 2}),
        )
        got = apewisdom_signals(["SPY", "NVDA"], pages=3, confidence=0.3, timeout=5, session=session)
        assert sorted(s.subject for s in got) == ["NVDA", "SPY"]
        assert [url for url, _ in session.calls] == [
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1",
            "https://apewisdom.io/api/v1.0/filter/all-stocks/page/2",
        ]
        assert all(kwargs["timeout"] == 5 and kwargs["allow_redirects"] is False for _, kwargs in session.calls)

    def test_a_body_without_results_is_an_error(self):
        session = FakeSession(FakeResponse(body={"message": "maintenance"}))
        with pytest.raises(SourceError) as info:
            apewisdom_signals(["NVDA"], pages=1, confidence=0.3, timeout=5, session=session)
        assert info.value.status == "error"


class TestFetchJson:
    SECRET = "sk-test-0000-not-a-real-key"

    @pytest.mark.parametrize(
        "response, status, http_status",
        [
            (FakeResponse(429), "rate_limited", 429),
            (FakeResponse(503), "error", 503),
            (FakeResponse(301), "error", 301),
            (FakeResponse(200, text="<html>not json"), "error", 200),
            (requests.Timeout("read timed out"), "error", None),
            (requests.ConnectionError("refused"), "error", None),
        ],
    )
    def test_failures_map_to_a_status_and_never_echo_the_headers(self, response, status, http_status):
        session = FakeSession(response)
        with pytest.raises(SourceError) as info:
            fetch_json("https://example.test/x", headers={"Authorization": self.SECRET}, timeout=1, session=session)
        assert info.value.status == status and info.value.http_status == http_status
        assert self.SECRET not in str(info.value)
        assert info.value.__cause__ is None  # the original exception (and its message) is not chained
        assert info.value.__context__ is None or info.value.__suppress_context__

    def test_ok_returns_the_body(self):
        assert fetch_json("https://example.test/x", timeout=1, session=FakeSession(FakeResponse(body=[1]))) == [1]


class TestNews:
    def test_items_become_unscored_signals_with_provenance(self, fixture):
        items = [NewsItem.from_alpaca(n, fetched_at=NOW) for n in fixture("news.json")["news"]]
        got = news_to_signals(items, "NVDA", confidence=0.6)
        assert [s.source_item_id for s in got] == ["43107442", "43107981"]
        assert all(s.sentiment is None and s.engagement == 0 and s.kind is SignalKind.NEWS for s in got)
        assert got[0].url == "https://www.benzinga.com/example/fed-holds" and got[1].url is None
        assert got[0].raw["headline"] == "Fed Holds Rates Steady, Signals Patience On Cuts"
        assert got[0].endpoint == "alpaca:/v1beta1/news"

    def test_a_failed_fetch_is_a_source_error(self, monkeypatch):
        from aegis.data.errors import DataError

        def broken(symbols=None, limit=5):
            raise DataError("news", symbols, ConnectionError("down"))

        monkeypatch.setattr(news_module, "get_news", broken)
        with pytest.raises(SourceError) as info:
            news_signals("NVDA", limit=5, confidence=0.6, market_wide=False)
        assert info.value.status == "error" and "ConnectionError" in str(info.value)

    def test_a_topic_is_matched_in_the_market_wide_feed(self, monkeypatch, fixture):
        items = [NewsItem.from_alpaca(n, fetched_at=NOW) for n in fixture("news.json")["news"]]
        asked = []

        def fake(symbols=None, limit=5):
            asked.append((symbols, limit))
            return items

        monkeypatch.setattr(news_module, "get_news", fake)
        source = NewsSource(SignalsConfig(topics=("chipmakers",)))
        assert source.subjects(["NVDA"], ["chipmakers"]) == ["NVDA", "chipmakers"]
        got = source.fetch("chipmakers")
        assert [s.headline for s in got] == ["Chipmakers Rally On Data-Center Demand"]
        assert asked == [(None, 50)]
        source.fetch("NVDA")
        assert asked[-1] == ("NVDA", 10)

    def test_build_sources_follows_the_switches(self):
        assert [s.name for s in build_sources(SignalsConfig(), ["NVDA"])] == [SignalSource.NEWS]
        both = SignalsConfig(sources={"apewisdom": {"enabled": True}})
        sources = build_sources(both, ["NVDA"])
        assert [s.name for s in sources] == [SignalSource.NEWS, SignalSource.APEWISDOM]
        assert isinstance(sources[1], ApeWisdomSource) and sources[1].subjects(["NVDA"], []) == [APEWISDOM_FILTER]


# --- store --------------------------------------------------------------------------


class TestStore:
    def test_a_fetch_and_its_signals_round_trip(self, conn):
        fetch = SignalFetch(
            source="news", subject="NVDA", endpoint="e", started_at=NOW, finished_at=NOW,
            status=SignalFetchStatus.OK, item_count=1, cycle_id="c-1",
        )
        stored = record(fetch_id=fetch.id, raw_json={"id": 7, "nested": {"a": [1, 2]}}, url="https://x.test/a")
        record_fetch(conn, fetch, [stored])
        assert get_fetch_signals(conn, fetch.id) == [stored]
        assert get_signals(conn, [stored.id, "nope"]) == [stored]
        assert get_cycle_fetches(conn, "c-1") == [fetch]
        assert latest_ok_fetch(conn, "news", "NVDA", NOW - timedelta(minutes=1)) == fetch
        assert latest_ok_fetch(conn, "news", "NVDA", NOW + timedelta(seconds=1)) is None

    def test_signals_must_belong_to_the_fetch(self, conn):
        from aegis.store.errors import StoreError

        fetch = SignalFetch(source="news", subject="X", endpoint="e", started_at=NOW, finished_at=NOW,
                            status=SignalFetchStatus.OK)
        with pytest.raises(StoreError):
            record_fetch(conn, fetch, [record(fetch_id="other")])

    def test_requests_sent_skips_cache_hits_and_our_own_limiter(self, conn):
        def add(status, http_status=None):
            record_fetch(conn, SignalFetch(source="news", subject="X", endpoint="e", started_at=NOW,
                                           finished_at=NOW, status=status, http_status=http_status))

        add(SignalFetchStatus.OK)
        add(SignalFetchStatus.ERROR)
        add(SignalFetchStatus.RATE_LIMITED, 429)  # the provider answered: a request was sent
        add(SignalFetchStatus.RATE_LIMITED)  # our limiter refused: nothing was sent
        add(SignalFetchStatus.CACHED)
        add(SignalFetchStatus.DISABLED)
        assert requests_sent(conn, "news", NOW - timedelta(hours=1)) == 3
        assert requests_sent(conn, "apewisdom", NOW - timedelta(hours=1)) == 0


# --- rate limiting --------------------------------------------------------------------


class TestRateLimiter:
    def test_per_minute_cap_resets_after_a_minute(self, conn):
        clock = [NOW]
        limiter = RateLimiter(conn, "news", SignalSourceConfig(requests_per_minute=2), clock=lambda: clock[0])
        limiter.acquire()
        limiter.acquire()
        with pytest.raises(SourceError) as info:
            limiter.acquire()
        assert info.value.status == "rate_limited" and info.value.http_status is None
        clock[0] = NOW + timedelta(seconds=60)
        limiter.acquire()

    def test_per_day_cap_counts_the_store_since_utc_midnight(self, conn):
        for hours in (1, 2, 20):  # two today, one yesterday
            at = NOW.replace(hour=0) + timedelta(hours=hours) - timedelta(days=1 if hours == 20 else 0)
            record_fetch(conn, SignalFetch(source="news", subject="X", endpoint="e", started_at=at,
                                           finished_at=at, status=SignalFetchStatus.OK))
        RateLimiter(conn, "news", SignalSourceConfig(requests_per_day=3), clock=lambda: NOW).acquire()
        with pytest.raises(SourceError, match="2 requests a day"):
            RateLimiter(conn, "news", SignalSourceConfig(requests_per_day=2), clock=lambda: NOW).acquire()

    def test_no_caps_never_refuses(self, conn):
        limiter = RateLimiter(conn, "news", SignalSourceConfig(), clock=lambda: NOW)
        for _ in range(500):
            limiter.acquire()


# --- dedupe and decay ----------------------------------------------------------------------


class TestDedupe:
    def test_canonical_url_ignores_case_www_fragments_tracking_and_slashes(self):
        a = canonical_url("https://WWW.Benzinga.com/news/nvda/?utm_source=x&id=7#top")
        b = canonical_url("https://benzinga.com/news/nvda?id=7&fbclid=abc")
        assert a == b == "https://benzinga.com/news/nvda?id=7"
        assert canonical_url(None) is None and canonical_url("  ") is None

    def test_title_key_ignores_case_and_punctuation(self):
        assert title_key("Nvidia: Beats   Estimates!") == title_key("nvidia beats estimates") == "nvidia beats estimates"

    def test_same_url_across_sources_keeps_the_more_engaged_and_merges_subjects(self):
        a = record(id="a", source="news", url="https://x.test/story?utm_medium=s", engagement=0, subject="NVDA")
        b = record(id="b", source="apewisdom", kind="news", url="https://x.test/story", engagement=50,
                   subject="AMD", headline="Something else entirely")
        [(kept, subjects, absorbed)] = dedupe([a, b])
        assert kept.id == "b" and set(subjects) == {"NVDA", "AMD"} and absorbed == ("a",)

    def test_same_title_without_urls_is_a_duplicate_and_groups_join_transitively(self):
        a = record(id="a", url="https://x.test/1", headline="Alpha story")
        b = record(id="b", url="https://x.test/1", headline="Beta story")
        c = record(id="c", url=None, headline="BETA story.")
        d = record(id="d", url=None, headline="Unrelated")
        groups = dedupe([d, c, b, a])
        assert sorted(sorted([k.id, *ab]) for k, _, ab in groups) == [["a", "b", "c"], ["d"]]

    def test_ties_are_broken_stably_whatever_the_input_order(self):
        a = record(id="a", headline="Same", published_at=NOW - timedelta(hours=2))
        b = record(id="b", headline="Same", published_at=NOW - timedelta(hours=1))
        assert dedupe([a, b])[0][0].id == dedupe([b, a])[0][0].id == "a"  # the earlier copy

    def test_only_the_newest_mention_count_per_ticker_survives(self):
        old = record(id="old", source="apewisdom", kind="mentions", headline="10 mentions", engagement=900,
                     published_at=NOW - timedelta(hours=5))
        new = record(id="new", source="apewisdom", kind="mentions", headline="12 mentions", engagement=900,
                     published_at=NOW - timedelta(hours=1))
        [(kept, _, absorbed)] = dedupe([old, new])
        assert kept.id == "new" and absorbed == ("old",)


class TestDecay:
    def test_recency_halves_each_half_life(self):
        assert recency(0, 6) == 1.0 and recency(6, 6) == 0.5 and recency(12, 6) == 0.25

    def test_stale_faded_and_future_signals_are_dropped(self):
        records = [
            record(id="fresh", headline="fresh", published_at=NOW - timedelta(hours=1)),
            record(id="old", headline="old", published_at=NOW - timedelta(hours=49)),
            record(id="faded", headline="faded", published_at=NOW - timedelta(hours=30)),  # 0.6 * 0.5**5 < 0.05
            record(id="future", headline="future", published_at=NOW + timedelta(minutes=5)),
        ]
        ranked = rank_signals(records, NOW, half_life_hours=6, max_age_hours=48, min_weight=0.05)
        assert [r.record.id for r in ranked] == ["fresh"]
        assert ranked[0].weight == pytest.approx(0.6 * 0.5 ** (1 / 6))

    def test_newer_outranks_older_and_engagement_orders_the_survivors(self):
        older = record(id="older", headline="older", published_at=NOW - timedelta(hours=6))
        newer = record(id="newer", headline="newer", published_at=NOW - timedelta(hours=1))
        loud = record(id="loud", headline="loud", source="apewisdom", kind="mentions", engagement=999,
                      published_at=NOW - timedelta(hours=1), confidence=0.3)
        ranked = rank_signals([older, newer, loud], NOW, half_life_hours=6, max_age_hours=48, min_weight=0.01)
        assert [r.record.id for r in ranked] == ["loud", "newer", "older"]
        assert ranked[0].score == pytest.approx(ranked[0].weight * (1 + math.log10(1000)))

    def test_a_record_without_a_publish_time_ages_from_its_fetch(self):
        r = record(published_at=None, fetched_at=NOW - timedelta(hours=6))
        [ranked] = rank_signals([r], NOW, half_life_hours=6, max_age_hours=48, min_weight=0.0)
        assert ranked.age_hours == pytest.approx(6)


# --- the digest -------------------------------------------------------------------------------


class TestDigest:
    def test_clean_text_strips_links_markup_and_control_characters(self):
        dirty = (
            "Buy <b>NOW</b> &amp; [click here](https://evil.test/x) ![img](http://i.test/p.png) "
            "see www.evil.test/a or https://evil.test/b?c=1 `rm -rf`​\nnext\tline\x07"
        )
        assert clean_text(dirty, 500) == "Buy NOW & click here img see or rm -rf next line"

    def test_clean_text_caps_length(self):
        assert clean_text("word " * 100, 40) == ("word " * 8).strip()[:37].rstrip() + "..."
        assert len(clean_text("x" * 1000, 40)) == 40

    def _ranked(self, n, headline="Headline number {i} about chips"):
        records = [
            record(id=f"r{i:02d}", headline=headline.format(i=i), published_at=NOW - timedelta(minutes=i))
            for i in range(n)
        ]
        return rank_signals(records, NOW, half_life_hours=6, max_age_hours=48, min_weight=0.0)

    def test_lines_carry_subject_kind_time_and_unscored(self):
        digest = build_digest(self._ranked(1), max_items=5, max_item_chars=200, max_prompt_tokens=500)
        assert digest.text == "- NVDA | news | 2026-10-30 15:00 UTC | unscored | Headline number 0 about chips"
        assert digest.lines[0].rank == 1 and digest.dropped == 0

    def test_a_scored_line_shows_its_sign(self):
        [r] = rank_signals([record(sentiment=-0.5)], NOW, half_life_hours=6, max_age_hours=48, min_weight=0)
        digest = build_digest([r], max_items=5, max_item_chars=200, max_prompt_tokens=500)
        assert "sentiment -0.50" in digest.text

    def test_max_items_and_the_token_cap_both_bind(self):
        assert len(build_digest(self._ranked(20), max_items=3, max_item_chars=200, max_prompt_tokens=5000).lines) == 3
        capped = build_digest(self._ranked(20), max_items=20, max_item_chars=200, max_prompt_tokens=100)
        assert capped.token_estimate <= 100 * 0.9
        assert len(capped.text) <= 100 * 4 and capped.dropped == 20 - len(capped.lines) > 0
        assert [line.rank for line in capped.lines] == list(range(1, len(capped.lines) + 1))


# --- collection ---------------------------------------------------------------------------------


class TestCollect:
    def test_fetches_stores_and_ranks(self, conn):
        source = FakeSource(by_subject={"NVDA": [signal()], "AAPL": [signal("AAPL", "n-2", "Apple event")]})
        result = collect_signals(conn, SIGNALS_ON, tickers=["NVDA", "AAPL"], cycle_id="c-1",
                                 sources=[source], now=NOW)
        assert source.calls == ["NVDA", "AAPL"]
        assert [s.status for s in result.statuses] == ["ok", "ok"]
        assert len(result.digest.lines) == 2 and result.considered == 2
        fetches = get_cycle_fetches(conn, "c-1")
        assert [(f.subject, f.status, f.item_count) for f in fetches] == [
            ("NVDA", SignalFetchStatus.OK, 1), ("AAPL", SignalFetchStatus.OK, 1)
        ]
        shown = get_signals(conn, [line.signal_id for line in result.digest.lines])
        assert {s.source_item_id for s in shown} == {"n-1", "n-2"}  # every shown line is a stored row

    def test_a_recent_fetch_is_reused_without_a_request(self, conn):
        source = FakeSource(by_subject={"NVDA": [signal()]})
        collect_signals(conn, SIGNALS_ON, tickers=["NVDA"], sources=[source], now=NOW)
        again = collect_signals(conn, SIGNALS_ON, tickers=["NVDA"], cycle_id="c-2", sources=[source], now=NOW)
        assert source.calls == ["NVDA"]  # one request in all
        assert [s.status for s in again.statuses] == ["cached"] and len(again.digest.lines) == 1
        assert [f.status for f in get_cycle_fetches(conn, "c-2")] == [SignalFetchStatus.CACHED]

    def test_configured_tickers_replace_the_watchlist(self, conn):
        source = FakeSource()
        collect_signals(conn, SignalsConfig(enabled=True, tickers=("TTWO",)), tickers=["SPY"],
                        sources=[source], now=NOW)
        assert source.calls == ["TTWO"]

    @pytest.mark.parametrize(
        "error, status",
        [
            (SourceError("rate_limited", "provider answered 429 Too Many Requests", 429), "rate_limited"),
            (SourceError("disabled", "no key"), "disabled"),
            (SourceError("error", "provider answered HTTP 500", 500), "error"),
            (KeyError("schema changed"), "error"),  # not a SourceError: a bug or a new payload shape
        ],
    )
    def test_a_failing_source_is_recorded_logged_and_skipped(self, conn, error, status):
        broken = FakeSource(name=SignalSource.APEWISDOM, error=error, subjects=["all-stocks"])
        healthy = FakeSource(by_subject={"NVDA": [signal()]})
        result = collect_signals(conn, SIGNALS_ON, tickers=["NVDA"], cycle_id="c-1",
                                 sources=[broken, healthy], now=NOW)
        assert [(s.source, s.status) for s in result.statuses] == [("apewisdom", status), ("news", "ok")]
        assert len(result.digest.lines) == 1
        [fetch, _] = get_cycle_fetches(conn, "c-1")
        assert fetch.status.value == status and fetch.error
        [event] = [e for e in get_recent_events(conn) if e.kind == "signal_source_degraded"]
        assert event.level.value == "warning" and event.payload["status"] == status
        assert event.payload["cycle_id"] == "c-1" and event.payload["fallback_items"] == 0

    def test_a_failed_fetch_falls_back_to_the_last_good_one(self, conn):
        # fetch reuse runs on the wall clock, so "two hours ago" is in real time
        older = datetime.now(timezone.utc) - timedelta(hours=2)
        fetch = SignalFetch(source="news", subject="NVDA", endpoint="e", started_at=older, finished_at=older,
                            status=SignalFetchStatus.OK)
        record_fetch(conn, fetch, to_records(fetch.id, [signal(published_at=older, fetched_at=older)]))
        failing = FakeSource(error=SourceError("error", "provider answered HTTP 502", 502))
        config = SignalsConfig(enabled=True, fetch_interval_minutes=30)
        result = collect_signals(conn, config, tickers=["NVDA"], sources=[failing], now=older + timedelta(hours=3))
        assert failing.calls == ["NVDA"] and result.statuses[0].items == 1
        assert len(result.digest.lines) == 1

    def test_a_long_error_is_cut(self, conn):
        failing = FakeSource(error=RuntimeError("x" * 5000))
        result = collect_signals(conn, SIGNALS_ON, tickers=["NVDA"], sources=[failing], now=NOW)
        assert len(result.statuses[0].detail) == 300

    def test_repeated_provider_ids_in_one_fetch_are_stored_once(self):
        records = to_records("f", [signal(item="same"), signal(item="same", headline="dup")])
        assert [r.headline for r in records] == ["Nvidia beats estimates"]

    def test_assignment(self):
        assert not signals_active(SignalsConfig(), NOW)
        assert signals_active(SIGNALS_ON, NOW)
        alternate = SignalsConfig(enabled=True, assignment="alternate_days")
        days = [signals_active(alternate, NOW + timedelta(days=d)) for d in range(4)]
        assert days in ([True, False, True, False], [False, True, False, True])
        late = datetime(2026, 10, 31, 2, 0, tzinfo=timezone.utc)  # still Oct 30 in New York
        assert signals_active(alternate, late) == signals_active(alternate, NOW)


# --- config ------------------------------------------------------------------------------------


class TestConfig:
    def test_the_repo_ships_signals_off_with_free_sources_only(self):
        config = load_config()
        assert config.signals.enabled is False
        assert config.signals.sources.apewisdom.enabled is False
        assert set(SignalsConfig.model_fields["sources"].annotation.model_fields) == {"news", "apewisdom"}

    def test_enabled_must_be_a_real_bool(self):
        with pytest.raises(ValueError):
            SignalsConfig(enabled="yes")

    def test_tickers_and_topics_are_cleaned(self):
        config = SignalsConfig(tickers=(" nvda", "NVDA", "aapl"), topics=(" Chips ",))
        assert config.tickers == ("NVDA", "AAPL") and config.topics == ("chips",)
        with pytest.raises(ValueError):
            SignalsConfig(tickers=("  ",))

    def test_the_digest_cap_must_fit_under_the_cycle_cap_when_on(self):
        brain = {"per_cycle_token_cap": 12000, "daily_token_budget": 500000}
        AegisConfig(watchlist=["SPY"], brain=brain, signals={"enabled": False, "max_prompt_tokens": 9000})
        with pytest.raises(ValueError, match="per_cycle_token_cap"):
            AegisConfig(watchlist=["SPY"], brain=brain, signals={"enabled": True, "max_prompt_tokens": 9000})


# --- the scan prompt ----------------------------------------------------------------------------


class TestScanPrompt:
    def test_no_signals_is_byte_for_byte_the_old_prompt(self, snapshot):
        assert build_scan_prompt(snapshot) == build_scan_prompt(snapshot, ()) == build_scan_prompt(snapshot, None)
        assert SIGNALS_HEADER not in build_scan_prompt(snapshot)
        assert scan_prompt_version(False) == "scan-v2+system-v1"
        assert scan_prompt_version(True) == "scan-v2+system-v1+signals-v1"

    def test_the_digest_sits_inside_the_untrusted_block_after_the_guard(self, snapshot):
        prompt = build_scan_prompt(snapshot, ["- NVDA | news | 2026-10-30 14:00 UTC | unscored | Chips rally"])
        start, end = prompt.index(NEWS_DELIM_OPEN), prompt.index(NEWS_DELIM_CLOSE)
        assert prompt.index(INJECTION_GUARD) < start
        assert start < prompt.index(SIGNALS_HEADER) < prompt.index("Chips rally") < end
        assert prompt.count(NEWS_DELIM_OPEN) == prompt.count(NEWS_DELIM_CLOSE) == 1

    def test_run_scan_records_the_signals_version(self, snapshot):
        llm = FakeLLM.from_fixtures({"scan": ["scan_ok.json", "scan_ok.json"]})
        config = AegisConfig(watchlist=["SPY"]).brain
        assert run_scan(llm, snapshot, config).prompt_version == "scan-v2+system-v1"
        result = run_scan(llm, snapshot, config, signals=["- NVDA | news | x | unscored | y"])
        assert result.prompt_version == "scan-v2+system-v1+signals-v1"
        assert "- NVDA | news | x | unscored | y" in llm.calls[1].messages[0].content


# --- the cycle ------------------------------------------------------------------------------------

OK = {"scan": ["scan_ok.json"], "thesis": ["thesis_ok.json"], "proposal": ["proposal_ok.json"]}


def _config(**signals) -> AegisConfig:
    return AegisConfig(watchlist=["SPY", "QQQ"], signals={"enabled": True, **signals})


class TestCycle:
    def test_signals_off_logs_nothing_and_fetches_nothing(self, conn, snapshot):
        source = FakeSource(by_subject={"SPY": [signal("SPY")]})
        result = run_cycle(conn, FakeLLM.from_fixtures(OK), config=AegisConfig(watchlist=["SPY", "QQQ"]),
                           snapshot=snapshot, signal_sources=[source])
        assert result.outcome == "proposal" and source.calls == []
        kinds = {e.kind for e in get_recent_events(conn, limit=100)}
        assert not kinds & {SIGNALS_DIGEST_EVENT, "signals_error", "signal_source_degraded"}

    def test_signals_on_shows_the_digest_and_logs_which_signals(self, conn, snapshot):
        # the cycle ranks on the wall clock, so the item is an hour old in real time
        fresh = datetime.now(timezone.utc) - timedelta(hours=1)
        source = FakeSource(by_subject={"SPY": [signal("SPY", headline="SPY breaks out", published_at=fresh)]})
        llm = FakeLLM.from_fixtures(OK)
        result = run_cycle(conn, llm, config=_config(), snapshot=snapshot, signal_sources=[source])
        assert result.outcome == "proposal"
        assert "SPY breaks out" in llm.calls[0].messages[0].content
        assert all("SPY breaks out" not in call.messages[0].content for call in llm.calls[1:])  # scan only
        [event] = [e for e in get_recent_events(conn, limit=100) if e.kind == SIGNALS_DIGEST_EVENT]
        assert event.payload["cycle_id"] == result.cycle_id and event.payload["active"] is True
        [shown] = event.payload["shown"]
        assert get_signals(conn, [shown["signal_id"]])[0].headline == "SPY breaks out"
        assert [s["status"] for s in event.payload["sources"]] == ["ok", "ok"]

    def test_every_source_failing_still_runs_the_whole_cycle(self, conn, snapshot):
        broken = FakeSource(error=SourceError("error", "provider answered HTTP 500", 500))
        result = run_cycle(conn, FakeLLM.from_fixtures(OK), config=_config(), snapshot=snapshot,
                           signal_sources=[broken])
        assert result.outcome == "proposal"
        kinds = [e.kind for e in get_recent_events(conn, limit=100)]
        assert SIGNALS_DIGEST_EVENT in kinds and kinds.count("signal_source_degraded") == 2

    def test_a_collector_crash_is_a_warning_and_the_scan_runs_without_signals(self, conn, snapshot, monkeypatch):
        import aegis.brain.cycle as cycle_module

        def crash(*args, **kwargs):
            raise RuntimeError("store went away")

        monkeypatch.setattr(cycle_module, "collect_signals", crash)
        llm = FakeLLM.from_fixtures(OK)
        result = run_cycle(conn, llm, config=_config(), snapshot=snapshot, signal_sources=[])
        assert result.outcome == "proposal" and SIGNALS_HEADER not in llm.calls[0].messages[0].content
        [event] = [e for e in get_recent_events(conn, limit=100) if e.kind == "signals_error"]
        assert event.level.value == "warning" and "store went away" in event.message

    def test_an_off_day_under_alternate_days_is_logged_inactive(self, conn, snapshot):
        source = FakeSource(by_subject={"SPY": [signal("SPY")]})
        config = _config(assignment="alternate_days")
        if signals_active(config.signals, snapshot.taken_at):
            snapshot = snapshot.model_copy(update={"taken_at": snapshot.taken_at + timedelta(days=1)})
        run_cycle(conn, FakeLLM.from_fixtures(OK), config=config, snapshot=snapshot, signal_sources=[source])
        assert source.calls == []
        [event] = [e for e in get_recent_events(conn, limit=100) if e.kind == SIGNALS_DIGEST_EVENT]
        assert event.payload == {"cycle_id": event.payload["cycle_id"], "active": False}
