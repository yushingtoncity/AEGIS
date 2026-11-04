"""The recap chart (``aegis.notify.chart``), the recap's benchmark line,
the 2,000 character guard and the image upload in ``aegis.notify.discord``.

Stores are tmp_path files; the channel, the HTTP session and the bar reader
are stand-ins with made-up figures. Nothing reaches Discord or a broker.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone

import pytest

from aegis.cli import notify as cli
from aegis.config import AegisConfig, FeedConfig
from aegis.data.models import AccountState, Bar
from aegis.notify import chart, discord, feed
from aegis.notify.feed import FEED_RECAP, feed_outbox, recap_days, recap_event, send_feed
from aegis.notify.models import BODY_MAX, Notice
from aegis.notify.summary import NEW_YORK
from aegis.store import EventLevel, get_events_since, log_event, open_store

DAY = date(2026, 10, 28)
NOW = datetime.combine(DAY, time(16, 21), tzinfo=NEW_YORK).astimezone(timezone.utc)
ON = FeedConfig(enabled=True)
WEBHOOK = "https://discord.com/api/webhooks/123456/s3cret-token"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "chart.db")
    yield connection
    connection.close()


def _account(pct: float) -> AccountState:
    return AccountState(account_number="PA1", status="ACTIVE", equity=100_000 * (1 + pct / 100),
                        last_equity=100_000.0, positions=[])


def _recaps(conn, changes, bench=None):
    """One recap per weekday ending on DAY, oldest first."""
    days = []
    day = DAY
    while len(days) < len(changes):
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()
    for i, (d, pct) in enumerate(zip(days, changes)):
        at = datetime.combine(d, time(16, 21), tzinfo=NEW_YORK).astimezone(timezone.utc)
        account = None if pct is None else _account(pct)
        log_event(conn, recap_event(conn, d, account, at, benchmark="SPY",
                                    benchmark_pct=None if bench is None else bench[i]))
    return days


class FakeChannel:
    name = discord.CHANNEL_NAME

    def __init__(self):
        self.sent: list[tuple[Notice, bytes | None]] = []

    def send(self, notice, image=None):
        self.sent.append((notice, image))


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self):
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(204)


def _notice(body="x", title="AEGIS day recap"):
    return Notice(event_id="e1", occurred_at=NOW, level=EventLevel.INFO, kind=FEED_RECAP,
                  title=title, body=body)


# --- the chart's numbers --------------------------------------------------------


class TestCumulative:
    def test_compounds_each_day(self):
        assert chart.cumulative([1.0, 1.0]) == pytest.approx([1.0, 2.01])

    def test_a_missing_day_is_a_gap_and_the_level_carries_on(self):
        assert chart.cumulative([1.0, None, -1.0]) == pytest.approx([1.0, None, -0.01])

    def test_drawable_needs_two_account_points(self):
        one = [chart.DayChange(DAY, 1.0, 0.5), chart.DayChange(DAY, None, 0.5)]
        assert not chart.drawable(one)
        assert chart.drawable([chart.DayChange(DAY, 1.0, None), chart.DayChange(DAY, -1.0, None)])

    def test_render_is_a_png(self):
        days = [chart.DayChange(DAY - timedelta(days=i), pct, 0.1) for i, pct in enumerate((0.5, None, -0.2))]
        png = chart.render_png(list(reversed(days)), "SPY", feed.DISCLAIMER_TEXT)
        assert png.startswith(PNG_MAGIC) and len(png) < discord.IMAGE_MAX_BYTES

    def test_the_chart_disclaimer_is_the_posts(self):
        assert feed.DISCLAIMER_TEXT == discord.DISCLAIMER


# --- the recap post -------------------------------------------------------------


class TestRecapPost:
    def test_recap_days_reads_the_window_one_per_day(self, conn):
        days = _recaps(conn, [0.5, -0.25, 1.0], bench=[0.1, 0.2, None])
        log_event(conn, recap_event(conn, days[-1], _account(9.0), NOW))  # a second one that day
        got = recap_days(conn, DAY, 30)
        assert [d.day for d in got] == days
        assert got[-1].account_pct == pytest.approx(1.0) and got[-1].benchmark_pct is None
        assert [d.day for d in recap_days(conn, DAY, 1)] == [DAY]

    def test_the_recap_carries_a_chart_once_there_are_two_days(self, conn):
        _recaps(conn, [0.5, -0.25], bench=[0.1, 0.4])
        (post,) = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts
        assert post.key == f"recap:{DAY.isoformat()}"
        assert post.image is not None and post.image.startswith(PNG_MAGIC)
        assert "SPY for comparison: up 0.40% today." in post.notice.body.splitlines()

    def test_one_day_is_text_only(self, conn):
        _recaps(conn, [0.5])
        (post,) = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts
        assert post.image is None
        assert "for comparison" not in post.notice.body

    def test_chart_days_zero_turns_it_off(self, conn):
        _recaps(conn, [0.5, -0.25])
        (post,) = feed_outbox(conn, FeedConfig(enabled=True, chart_days=0), discord.CHANNEL_NAME, NOW).posts
        assert post.image is None

    def test_a_chart_that_fails_to_draw_leaves_the_text(self, conn, monkeypatch):
        _recaps(conn, [0.5, -0.25])

        def broken(*args):
            raise RuntimeError("no matplotlib")

        monkeypatch.setattr(chart, "render_png", broken)
        (post,) = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts
        assert post.image is None and post.notice.title.startswith("AEGIS day recap")

    def test_the_chart_is_percent_even_with_amounts_shown(self, conn):
        _recaps(conn, [0.5, -0.25])
        (post,) = feed_outbox(conn, FeedConfig(enabled=True, show_amounts=True), discord.CHANNEL_NAME, NOW).posts
        assert post.image is not None and "$" in post.notice.body  # dollars in the text only

    def test_send_attaches_the_image_and_others_go_without(self, conn):
        _recaps(conn, [0.5, -0.25])
        channel = FakeChannel()
        report = send_feed(conn, ON, channel, lambda: NOW)
        assert report.sent == 1
        ((notice, image),) = channel.sent
        assert image is not None and image.startswith(PNG_MAGIC)

    def test_a_recap_already_posted_is_not_drawn_again(self, conn, monkeypatch):
        _recaps(conn, [0.5, -0.25])
        send_feed(conn, ON, FakeChannel(), lambda: NOW)
        calls = []
        monkeypatch.setattr(chart, "render_png", lambda *a: calls.append(a) or PNG_MAGIC)
        assert feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts == ()
        assert calls == []


# --- Discord ----------------------------------------------------------------------


class TestDiscordLimits:
    def test_an_escaped_body_never_passes_2000(self):
        body = ("*_" * (BODY_MAX // 2))[:BODY_MAX]  # every character doubles when escaped
        content = discord.content_for(_notice(body=body))
        assert len(content) <= discord.CONTENT_MAX
        assert content.endswith(f"...\n-# {discord.DISCLAIMER}")
        before = content.split("...\n-# ")[0]
        assert (len(before) - len(before.rstrip("\\"))) % 2 == 0  # no lone escape before the "..."

    def test_a_short_body_is_untouched(self):
        assert discord.content_for(_notice(body="up 0.5%")) == (
            f"**AEGIS day recap**\nup 0.5%\n-# {discord.DISCLAIMER}"
        )

    def test_a_post_with_an_image_is_one_multipart_request(self):
        session = FakeSession()
        discord.DiscordChannel(WEBHOOK, timeout=7, session=session).send(_notice(), image=PNG_MAGIC + b"x")
        (call,) = session.calls
        assert "json" not in call and call["allow_redirects"] is False and call["timeout"] == 7
        name, data, kind = call["files"]["files[0]"]
        assert (name, kind) == (discord.IMAGE_FILENAME, "image/png") and data.startswith(PNG_MAGIC)
        payload = json.loads(call["data"]["payload_json"])
        assert payload["allowed_mentions"] == {"parse": []}
        assert payload["attachments"] == [{"id": 0, "filename": discord.IMAGE_FILENAME}]
        assert payload["content"].endswith(discord.DISCLAIMER)

    @pytest.mark.parametrize("image", [b"", b"x" * (8 * 1024 * 1024 + 1)])
    def test_an_empty_or_oversized_image_is_left_off(self, image):
        session = FakeSession()
        discord.DiscordChannel(WEBHOOK, timeout=1, session=session).send(_notice(), image=image)
        (call,) = session.calls
        assert "files" not in call and "attachments" not in call["json"]


# --- the CLI's benchmark read -------------------------------------------------------


def _bar(day: date, close: float) -> Bar:
    at = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK).astimezone(timezone.utc)
    return Bar(symbol="SPY", timestamp=at, open=close, high=close, low=close, close=close,
               volume=1.0, fetched_at=NOW)


class TestBenchmark:
    def test_change_on_the_day_from_daily_bars(self, monkeypatch):
        import aegis.data.market as market

        bars = [_bar(DAY - timedelta(days=1), 500.0), _bar(DAY, 502.0)]
        monkeypatch.setattr(market, "get_bars", lambda *a, **k: bars)
        assert cli.benchmark_change("SPY", DAY) == pytest.approx(0.4)
        assert cli.benchmark_change("SPY", DAY + timedelta(days=1)) is None

    @pytest.mark.parametrize("reader, expected", [
        (lambda s, d: 0.4, 0.4), (lambda s, d: (_ for _ in ()).throw(RuntimeError("down")), None),
    ])
    def test_recap_writes_the_benchmark_or_none(self, tmp_path, monkeypatch, reader, expected):
        db = tmp_path / "cli.db"
        open_store(db).close()
        monkeypatch.setattr(cli, "get_config", lambda: AegisConfig(watchlist=["SPY"], feed=FeedConfig()))
        argv = ["feed-recap", "--db", str(db), "--day", DAY.isoformat()]
        assert cli.main(argv, clock=lambda: NOW, account_reader=lambda: _account(0.5),
                        benchmark_reader=reader) == 0
        conn = open_store(db)
        (event,) = get_events_since(conn, NOW - timedelta(days=2), kind=FEED_RECAP)
        conn.close()
        assert event.payload["benchmark"] == "SPY"
        assert event.payload["benchmark_change_pct"] == expected
