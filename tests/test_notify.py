"""Phase 7, the notifier: which events go out, that each goes out once, the
ntfy channel's request, and ``python -m aegis.cli.notify``. Stores are
tmp_path files, the channel and the HTTP session are stand-ins and the
clock is fixed; nothing reaches the network."""

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from aegis.cli import notify as cli
from aegis.config import AegisConfig, ConfigError, NotifyConfig, load_config, DEFAULT_CONFIG_PATH
from aegis.notify import channels, imessage, ntfy, slack
from aegis.notify.errors import NotifyError
from aegis.notify.models import BODY_MAX, TITLE_MAX, Notice
from aegis.notify.outbox import (
    NOTIFY_FAILED,
    NOTIFY_SENT,
    notice_for,
    outbox,
    send_due,
    wanted,
)
from aegis.store import Event, EventLevel, get_events_since, log_event, open_store

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
ON = NotifyConfig(enabled=True)
OFF = NotifyConfig()
TOPIC = "https://ntfy.example/aegis-s3cret-topic"
SECRET = "s3cret"


def _event(minutes_ago: float = 1, *, level=EventLevel.WARNING, kind="order_refused",
           message="aegis-p1: refused", payload=None, id=None) -> Event:
    fields = dict(occurred_at=NOW - timedelta(minutes=minutes_ago), level=level, kind=kind,
                  message=message, payload=payload)
    return Event(**fields) if id is None else Event(id=id, **fields)


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "notify.db")
    yield connection
    connection.close()


class FakeChannel:
    """Records what it was asked to send; fails for the event ids in ``failing``."""

    def __init__(self, name="imessage", failing=()):
        self.name = name
        self.failing = set(failing)
        self.sent: list[Notice] = []

    def send(self, notice):
        if notice.event_id in self.failing:
            raise NotifyError("post to ntfy", notice.event_id, detail="HTTP 503")
        self.sent.append(notice)


def _clock():
    return NOW


# --- the store read -------------------------------------------------------------


class TestEventsSince:
    def test_oldest_first_bound_inclusive(self, conn):
        late = log_event(conn, _event(1, id="late"))
        early = log_event(conn, _event(5, id="early"))
        edge = log_event(conn, _event(10, id="edge"))
        log_event(conn, _event(11, id="before"))
        assert get_events_since(conn, NOW - timedelta(minutes=10)) == [edge, early, late]

    def test_one_kind(self, conn):
        log_event(conn, _event(2, kind="order_refused"))
        sent = log_event(conn, _event(1, level=EventLevel.INFO, kind=NOTIFY_SENT))
        assert get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_SENT) == [sent]
        assert get_events_since(conn, NOW - timedelta(hours=1), kind="nothing") == []

    def test_naive_bound_is_utc(self, conn):
        kept = log_event(conn, _event(1))
        naive = (NOW - timedelta(minutes=2)).replace(tzinfo=None)
        assert get_events_since(conn, naive) == [kept]

    def test_re_exported(self):
        import aegis.store
        from aegis.store import repo

        assert aegis.store.get_events_since is repo.get_events_since
        assert "get_events_since" in aegis.store.__all__


# --- what is wanted -------------------------------------------------------------


class TestWanted:
    @pytest.mark.parametrize("level, expected", [
        (EventLevel.DEBUG, False), (EventLevel.INFO, False), (EventLevel.WARNING, True),
        (EventLevel.ERROR, True), (EventLevel.CRITICAL, True),
    ])
    def test_by_level(self, level, expected):
        assert wanted(_event(level=level, kind="something"), OFF) is expected

    def test_listed_kinds_whatever_their_level(self):
        for kind in OFF.kinds:
            assert wanted(_event(level=EventLevel.INFO, kind=kind), OFF)
        assert not wanted(_event(level=EventLevel.INFO, kind="market_snapshot"), OFF)

    def test_never_its_own_kinds(self):
        for kind in (NOTIFY_SENT, NOTIFY_FAILED, "notify_anything"):
            assert not wanted(_event(level=EventLevel.CRITICAL, kind=kind),
                              NotifyConfig(kinds=(kind,), min_level="debug"))

    def test_ignored_kinds_never_go_out(self):
        assert not wanted(_event(level=EventLevel.WARNING, kind="policy_reject"), OFF)
        assert wanted(_event(level=EventLevel.WARNING, kind="policy_reject"), NotifyConfig(ignore_kinds=()))
        assert not wanted(_event(level=EventLevel.INFO, kind="order_updated"),
                          NotifyConfig(ignore_kinds=("order_updated",)))

    def test_min_level_moves_the_line(self):
        assert not wanted(_event(level=EventLevel.WARNING, kind="x"), NotifyConfig(min_level="critical"))
        assert wanted(_event(level=EventLevel.INFO, kind="x"), NotifyConfig(min_level="info"))


# --- the notice -----------------------------------------------------------------


class TestNotice:
    def test_approval_needed_carries_the_command(self):
        event = _event(level=EventLevel.INFO, kind="policy_needs_approval",
                       message="proposal p1 (buy SPY): needs approval",
                       payload={"decision_id": "dec-1", "symbol": "SPY"})
        notice = notice_for(event)
        assert notice.title == "AEGIS Approval needed (SPY)"
        assert notice.body == ("proposal p1 (buy SPY): needs approval\n\n"
                               "Approve: python -m aegis.cli.orders approve dec-1 --by NAME")
        assert (notice.priority, notice.tags) == (4, ("bell",))
        assert (notice.event_id, notice.kind, notice.occurred_at) == (event.id, event.kind, event.occurred_at)

    def test_levels_set_priority_and_tag(self):
        critical = notice_for(_event(level=EventLevel.CRITICAL, kind="orphan_broker_order"))
        assert critical.title == "AEGIS CRITICAL Unknown order at the broker"
        assert (critical.priority, critical.tags) == (5, ("rotating_light",))
        info = notice_for(_event(level=EventLevel.INFO, kind="order_updated"))
        assert info.title == "AEGIS Order update" and (info.priority, info.tags) == (3, ())
        unnamed = notice_for(_event(level=EventLevel.ERROR, kind="brain_cycle_failed"))
        assert unnamed.title == "AEGIS ERROR brain_cycle_failed" and unnamed.priority == 4

    def test_stored_text_cannot_break_lines_or_overflow(self):
        notice = notice_for(_event(message="line one\nline\x1b[31m two " + "x" * 5000,
                                   payload={"symbol": "SP\nY" + "Z" * 500}))
        assert "\n" not in notice.body and "\x1b" not in notice.body
        assert len(notice.body) == BODY_MAX and notice.body.endswith("...")
        assert "\n" not in notice.title and len(notice.title) == TITLE_MAX

    def test_hint_survives_a_long_message(self):
        notice = notice_for(_event(kind="policy_needs_approval", message="m" * 5000,
                                   payload={"decision_id": "dec-9"}))
        assert len(notice.body) <= BODY_MAX and notice.body.endswith("approve dec-9 --by NAME")

    def test_an_oversized_decision_id_drops_the_hint_and_still_builds(self):
        notice = notice_for(_event(kind="policy_needs_approval", message="m",
                                   payload={"decision_id": "d" * 5000}))
        assert notice.body == "m"

    def test_empty_message_falls_back_to_the_kind(self):
        assert notice_for(_event(message=" \n ")).body == "order_refused"


# --- the outbox -----------------------------------------------------------------


def _rows(path):
    raw = sqlite3.connect(path)
    try:
        return raw.execute("SELECT * FROM events ORDER BY rowid").fetchall()
    finally:
        raw.close()


class TestOutbox:
    def test_due_is_wanted_recent_and_oldest_first(self, conn):
        log_event(conn, _event(61, id="too-old"))
        second = log_event(conn, _event(2, id="second"))
        first = log_event(conn, _event(30, id="first"))
        log_event(conn, _event(1, level=EventLevel.INFO, kind="market_snapshot"))
        log_event(conn, _event(-1, id="from-the-future"))
        due = outbox(conn, OFF, "ntfy", NOW)
        assert [n.event_id for n in due.notices] == [first.id, second.id]
        assert (due.waiting, due.given_up) == (0, 0)

    def test_sent_on_this_channel_is_not_due_again(self, conn):
        done = log_event(conn, _event(5, id="done"))
        log_event(conn, _event(4, level=EventLevel.INFO, kind=NOTIFY_SENT,
                               payload={"event_id": done.id, "channel": "ntfy"}))
        assert outbox(conn, OFF, "ntfy", NOW).notices == ()
        assert [n.event_id for n in outbox(conn, OFF, "other", NOW).notices] == [done.id]

    def test_max_per_run_holds_the_rest(self, conn):
        for i in range(5):
            log_event(conn, _event(10 - i, id=f"e{i}"))
        due = outbox(conn, NotifyConfig(max_per_run=2), "ntfy", NOW)
        assert [n.event_id for n in due.notices] == ["e0", "e1"] and due.waiting == 3

    def test_given_up_after_max_attempts(self, conn):
        log_event(conn, _event(10, id="flaky"))
        for _ in range(3):
            log_event(conn, _event(5, kind=NOTIFY_FAILED, payload={"event_id": "flaky", "channel": "ntfy"}))
        due = outbox(conn, OFF, "ntfy", NOW)
        assert due.notices == () and due.given_up == 1 and due.failures == {"flaky": 3}
        assert [n.event_id for n in outbox(conn, NotifyConfig(max_attempts=4), "ntfy", NOW).notices] == ["flaky"]

    def test_a_naive_now_is_utc(self, conn):
        event = log_event(conn, _event(2))
        due = outbox(conn, OFF, "ntfy", NOW.replace(tzinfo=None))
        assert [n.event_id for n in due.notices] == [event.id]

    def test_writes_nothing(self, tmp_path):
        path = tmp_path / "notify.db"
        conn = open_store(path)
        try:
            log_event(conn, _event(1))
            before = _rows(path)
            outbox(conn, OFF, "ntfy", NOW)
        finally:
            conn.close()
        assert _rows(path) == before


class TestSendDue:
    def test_refuses_while_disabled(self, conn):
        log_event(conn, _event(1))
        with pytest.raises(NotifyError, match="notify.enabled is false"):
            send_due(conn, OFF, FakeChannel(), _clock)
        assert get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_SENT) == []

    def test_sends_once_and_records_it(self, conn):
        event = log_event(conn, _event(3))
        channel = FakeChannel()
        report = send_due(conn, ON, channel, _clock)
        assert [n.event_id for n in channel.sent] == [event.id]
        assert (report.sent, report.failed) == (1, 0)
        [record] = get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_SENT)
        assert record.level is EventLevel.INFO and record.occurred_at == NOW
        assert record.payload == {"event_id": event.id, "channel": "imessage", "attempt": 1}
        again = FakeChannel()
        assert send_due(conn, ON, again, _clock).sent == 0 and again.sent == []

    def test_a_failure_is_recorded_and_the_run_goes_on(self, conn):
        bad = log_event(conn, _event(5, id="bad"))
        good = log_event(conn, _event(4, id="good"))
        channel = FakeChannel(failing={"bad"})
        report = send_due(conn, ON, channel, _clock)
        assert [n.event_id for n in channel.sent] == [good.id]
        assert (report.sent, report.failed) == (1, 1)
        [failure] = get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_FAILED)
        assert failure.level is EventLevel.WARNING
        assert failure.payload == {"event_id": bad.id, "channel": "imessage", "attempt": 1}
        assert "attempt 1 of 3" in failure.message and "HTTP 503" in failure.message

    def test_retries_then_gives_up_and_never_sends_its_own_failures(self, conn):
        log_event(conn, _event(5, id="bad"))
        channel = FakeChannel(failing={"bad"})
        attempts = [send_due(conn, ON, channel, _clock) for _ in range(4)]
        assert [r.failed for r in attempts] == [1, 1, 1, 0]
        assert attempts[-1].given_up == 1
        failures = get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_FAILED)
        assert [f.payload["attempt"] for f in failures] == [1, 2, 3]
        assert channel.sent == []  # the notify_failed warnings were never sent themselves


# --- the ntfy channel -----------------------------------------------------------


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self, status=200, raises=None):
        self.status, self.raises = status, raises
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.raises is not None:
            raise self.raises
        return _Response(self.status)


def _notice(**overrides):
    fields = dict(event_id="ev-1", occurred_at=NOW, level=EventLevel.CRITICAL, kind="orphan_broker_order",
                  title="AEGIS CRITICAL Unknown order é at the broker", body="aegis-p1: ünknown",
                  priority=5, tags=("rotating_light",))
    return Notice(**{**fields, **overrides})


class TestNtfyChannel:
    def test_the_request(self):
        session = FakeSession()
        ntfy.NtfyChannel(TOPIC, timeout=7, session=session).send(_notice())
        [(url, kwargs)] = session.calls
        assert url == TOPIC
        assert kwargs["data"] == "aegis-p1: ünknown".encode("utf-8")
        assert kwargs["timeout"] == 7 and kwargs["allow_redirects"] is False
        assert kwargs["headers"] == {
            "Title": "AEGIS CRITICAL Unknown order ? at the broker",
            "Priority": "5",
            "Tags": "rotating_light",
            "Content-Type": "text/plain; charset=utf-8",
        }

    def test_token_is_a_bearer_header(self):
        session = FakeSession()
        ntfy.NtfyChannel(TOPIC, timeout=1, token=" tk_abc ", session=session).send(_notice(tags=()))
        headers = session.calls[0][1]["headers"]
        assert headers["Authorization"] == "Bearer tk_abc" and "Tags" not in headers

    @pytest.mark.parametrize("status", [301, 400, 403, 429, 500])
    def test_a_refusal_is_a_notify_error(self, status):
        with pytest.raises(NotifyError, match=f"HTTP {status}") as caught:
            ntfy.NtfyChannel(TOPIC, timeout=1, session=FakeSession(status=status)).send(_notice())
        assert SECRET not in str(caught.value)

    def test_a_transport_error_never_quotes_the_url(self):
        leak = requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {TOPIC}")
        with pytest.raises(NotifyError) as caught:
            ntfy.NtfyChannel(TOPIC, timeout=1, session=FakeSession(raises=leak)).send(_notice())
        assert str(caught.value) == "notify failed: post to ntfy for ev-1 (ConnectionError)"
        assert caught.value.__cause__ is None and caught.value.__suppress_context__

    def test_repr_hides_the_address_and_token(self):
        text = repr(ntfy.NtfyChannel(TOPIC, timeout=1, token=SECRET, session=FakeSession()))
        assert SECRET not in text and "ntfy.example" not in text

    @pytest.mark.parametrize("url, why", [
        ("http://ntfy.example/topic", "https"),
        ("https:///topic", "no host"),
        ("https://ntfy.example/", "no topic"),
        ("https://ntfy.example", "no topic"),
        ("https://me:pw@ntfy.example/topic", "user or password"),
        ("https://ntfy.example/topic?auth=x", "query"),
    ])
    def test_bad_topic_urls_are_refused_without_echoing_them(self, url, why):
        with pytest.raises(ConfigError, match=why) as caught:
            ntfy.check_topic_url(url)
        assert url not in str(caught.value)

    def test_from_env(self, monkeypatch):
        monkeypatch.setattr(ntfy, "load_env", lambda: None)
        monkeypatch.delenv(ntfy.TOPIC_URL_ENV, raising=False)
        with pytest.raises(ConfigError, match="NTFY_TOPIC_URL"):
            ntfy.from_env(ON)
        monkeypatch.setenv(ntfy.TOPIC_URL_ENV, TOPIC)
        monkeypatch.setenv(ntfy.TOKEN_ENV, "tk")
        channel = ntfy.from_env(NotifyConfig(timeout_seconds=3))
        assert channel.name == "ntfy" and (channel._url, channel._token, channel._timeout) == (TOPIC, "tk", 3)


class FakeRunner:
    def __init__(self, returncode=0, raises=None):
        self.returncode, self.raises = returncode, raises
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(args, self.returncode, b"", b"execution error: +15555550123")


PHONE = "+15555550123"


class TestIMessageChannel:
    def test_the_command_keeps_text_out_of_the_script(self):
        runner = FakeRunner()
        hostile = _notice(title='AEGIS x" & do shell script "rm -rf ~', body="end tell\nline")
        imessage.IMessageChannel(PHONE, timeout=9, runner=runner).send(hostile)
        [(args, kwargs)] = runner.calls
        script = [args[i + 1] for i, a in enumerate(args[:-2]) if a == "-e"]
        assert args[0] == "/usr/bin/osascript" and tuple(script) == imessage.SCRIPT
        assert args[-2:] == [PHONE, f"{hostile.title}\n{hostile.body}"]
        assert kwargs == {"capture_output": True, "timeout": 9, "check": False}

    def test_a_nonzero_exit_names_only_the_code(self):
        with pytest.raises(NotifyError) as caught:
            imessage.IMessageChannel(PHONE, timeout=1, runner=FakeRunner(returncode=1)).send(_notice())
        assert str(caught.value) == "notify failed: send an iMessage for ev-1 (osascript exit 1)"

    @pytest.mark.parametrize("error", [subprocess.TimeoutExpired(["osascript", PHONE], 1),
                                       FileNotFoundError(f"no such file: {PHONE}")])
    def test_a_timeout_or_missing_osascript(self, error):
        with pytest.raises(NotifyError) as caught:
            imessage.IMessageChannel(PHONE, timeout=1, runner=FakeRunner(raises=error)).send(_notice())
        assert PHONE not in str(caught.value) and type(error).__name__ in str(caught.value)

    @pytest.mark.parametrize("good", [PHONE, "+1 (555) 555-0123", "me@icloud.com"])
    def test_recipients_accepted(self, good):
        assert imessage.check_recipient(f" {good} ") == good

    @pytest.mark.parametrize("bad", ["", "abc", "-e@x.com", "me@icloud.com\nx", "+1555\n5550123", "@x.com"])
    def test_recipients_refused_without_echoing(self, bad):
        with pytest.raises(ConfigError, match="IMESSAGE_TO") as caught:
            imessage.check_recipient(bad)
        assert bad.strip() == "" or bad not in str(caught.value)

    def test_repr_hides_the_recipient(self):
        assert PHONE not in repr(imessage.IMessageChannel(PHONE, timeout=1, runner=FakeRunner()))

    def test_from_env_needs_macos(self, monkeypatch):
        monkeypatch.setattr(imessage.sys, "platform", "linux")
        with pytest.raises(ConfigError, match="needs macOS"):
            imessage.from_env(ON)

    def test_from_env_on_macos(self, monkeypatch):
        monkeypatch.setattr(imessage.sys, "platform", "darwin")
        monkeypatch.setattr(imessage.os.path, "exists", lambda path: path == imessage.OSASCRIPT)
        monkeypatch.setattr(imessage, "load_env", lambda: None)
        monkeypatch.delenv(imessage.RECIPIENT_ENV, raising=False)
        with pytest.raises(ConfigError, match="IMESSAGE_TO"):
            imessage.from_env(ON)
        monkeypatch.setenv(imessage.RECIPIENT_ENV, PHONE)
        channel = imessage.from_env(NotifyConfig(timeout_seconds=4))
        assert channel.name == "imessage" and (channel._recipient, channel._timeout) == (PHONE, 4)


HOOK = "https://hooks.slack.com/services/T000/B000/s3cretXYZ"


class TestSlackChannel:
    def test_the_request_escapes_slack_markup(self):
        session = FakeSession()
        slack.SlackChannel(HOOK, timeout=5, session=session).send(
            _notice(title="AEGIS <!channel>", body="a & b <http://x|y>"))
        [(url, kwargs)] = session.calls
        assert url == HOOK and kwargs["timeout"] == 5 and kwargs["allow_redirects"] is False
        assert kwargs["json"] == {"text": "*AEGIS &lt;!channel&gt;*\na &amp; b &lt;http://x|y&gt;"}

    def test_failures_never_quote_the_url(self):
        leak = requests.ConnectionError(f"failed: {HOOK}")
        for session in (FakeSession(status=404), FakeSession(raises=leak)):
            with pytest.raises(NotifyError) as caught:
                slack.SlackChannel(HOOK, timeout=1, session=session).send(_notice())
            assert SECRET not in str(caught.value)

    @pytest.mark.parametrize("url", [
        "http://hooks.slack.com/services/T/B/x", "https://evil.example/services/T/B/x",
        "https://hooks.slack.com/other", "https://hooks.slack.com:8443/services/T/B/x",
        "https://hooks.slack.com/services/T/B/x?y=1",
    ])
    def test_bad_webhooks_are_refused_without_echoing(self, url):
        with pytest.raises(ConfigError, match="SLACK_WEBHOOK_URL") as caught:
            slack.check_webhook_url(url)
        assert url not in str(caught.value)

    def test_from_env(self, monkeypatch):
        monkeypatch.setattr(slack, "load_env", lambda: None)
        monkeypatch.setenv(slack.WEBHOOK_ENV, HOOK)
        channel = slack.from_env(NotifyConfig(timeout_seconds=2))
        assert channel.name == "slack" and SECRET not in repr(channel)


def test_the_factory_builds_the_configured_channel(monkeypatch):
    monkeypatch.setattr(slack, "load_env", lambda: None)
    monkeypatch.setattr(ntfy, "load_env", lambda: None)
    monkeypatch.setenv(slack.WEBHOOK_ENV, HOOK)
    monkeypatch.setenv(ntfy.TOPIC_URL_ENV, TOPIC)
    assert channels.channel_from_env(NotifyConfig(channel="slack")).name == "slack"
    assert channels.channel_from_env(NotifyConfig(channel="ntfy")).name == "ntfy"
    monkeypatch.setattr(imessage.sys, "platform", "win32")
    with pytest.raises(ConfigError, match="needs macOS"):
        channels.channel_from_env(NotifyConfig())


# --- config ---------------------------------------------------------------------


class TestConfig:
    def test_shipped_block_is_the_code_defaults_and_off(self):
        shipped = load_config(DEFAULT_CONFIG_PATH).notify
        assert shipped == NotifyConfig() and shipped.enabled is False

    def test_defaults(self):
        config = NotifyConfig()
        assert (config.channel, config.min_level, config.timeout_seconds) == ("imessage", "warning", 10.0)
        assert (config.lookback_minutes, config.max_attempts, config.max_per_run) == (60.0, 3, 20)
        assert "policy_needs_approval" in config.kinds

    @pytest.mark.parametrize("bad", [
        {"channel": "email"}, {"min_level": "loud"}, {"kinds": ["ok", " "]}, {"ignore_kinds": [""]}, {"max_attempts": 0},
        {"lookback_minutes": 0}, {"max_per_run": 0}, {"timeout_seconds": 0}, {"url": TOPIC},
    ])
    def test_refused(self, bad):
        with pytest.raises(ValueError):
            NotifyConfig(**bad)


# --- the CLI --------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cli.db"
    open_store(path).close()
    return path


@pytest.fixture
def config(monkeypatch):
    holder = {"config": AegisConfig(watchlist=["SPY"], notify=ON)}
    monkeypatch.setattr(cli, "get_config", lambda: holder["config"])
    return holder


def _seed(db, *events):
    conn = open_store(db)
    try:
        return [log_event(conn, e) for e in events]
    finally:
        conn.close()


def _run(db, *argv, channel=None):
    channel = channel if channel is not None else FakeChannel()
    extra = ["--db", str(db)] if db is not None else []
    return cli.main([*argv, *extra], channel_factory=lambda config: channel, clock=_clock)


class TestCli:
    def test_pending_lists_and_writes_nothing(self, db, config, capsys):
        [event] = _seed(db, _event(3, level=EventLevel.CRITICAL, kind="orphan_broker_order"))
        before = _rows(db)
        assert _run(db, "pending") == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "due now (1)"
        assert out[1] == f"  {event.id}   2026-10-14 14:57:00 UTC   p5   AEGIS CRITICAL Unknown order at the broker"
        assert _rows(db) == before

    def test_pending_says_when_sending_is_off(self, db, config, capsys):
        config["config"] = AegisConfig(watchlist=["SPY"])
        assert _run(db, "pending") == 0
        out = capsys.readouterr().out
        assert "due now (0)\n  (none)" in out and "sending is off" in out

    def test_send(self, db, config, capsys):
        [event] = _seed(db, _event(3))
        channel = FakeChannel()
        assert _run(db, "send", channel=channel) == 0
        assert capsys.readouterr().out == f"{event.id} order_refused: sent\n"
        assert [n.event_id for n in channel.sent] == [event.id]
        assert _run(db, "send", channel=channel) == 0
        assert capsys.readouterr().out == "nothing to send\n"

    def test_send_failure_exits_1(self, db, config, capsys):
        [event] = _seed(db, _event(3, id="bad"))
        assert _run(db, "send", channel=FakeChannel(failing={"bad"})) == 1
        captured = capsys.readouterr()
        assert "order_refused: failed" in captured.out and captured.err == "1 not sent\n"

    def test_send_refuses_while_disabled(self, db, config, capsys):
        config["config"] = AegisConfig(watchlist=["SPY"])
        _seed(db, _event(3))
        before = _rows(db)
        assert _run(db, "send") == 1
        assert "notify.enabled is false" in capsys.readouterr().err
        assert _rows(db) == before

    def test_send_never_creates_a_database(self, tmp_path, config, capsys):
        missing = tmp_path / "nope.db"
        assert _run(missing, "send") == 1
        assert "does not exist" in capsys.readouterr().err and not missing.exists()

    def test_test_needs_no_database(self, config, capsys):
        channel = FakeChannel()
        assert _run(None, "test", channel=channel) == 0
        [notice] = channel.sent
        assert notice.title == "AEGIS test" and notice.event_id == "test"
        assert capsys.readouterr().out == "test notice sent on imessage\n"

    def test_a_channel_error_is_one_clean_line(self, config, capsys):
        assert _run(None, "test", channel=FakeChannel(failing={"test"})) == 1
        assert capsys.readouterr().err == "notify test failed: notify failed: post to ntfy for test (HTTP 503)\n"

    def test_an_unexpected_error_prints_only_its_type(self, config, capsys):
        def boom(config):
            raise RuntimeError(f"could not reach {TOPIC}")

        assert cli.main(["test"], channel_factory=boom, clock=_clock) == 1
        err = capsys.readouterr().err
        assert err == "notify test failed (unexpected): RuntimeError\n" and SECRET not in err

    def test_bad_usage(self, capsys):
        assert cli.main(["shout"]) == 1
        assert "see --help" in capsys.readouterr().err


# --- never part of the order path -----------------------------------------------


def test_the_notifier_loads_nothing_from_the_order_path():
    """In a fresh interpreter, the notify package loads no policy engine, no
    broker package, no brain and no Alpaca client."""
    root = Path(__file__).resolve().parent.parent
    code = (
        "import sys, aegis.notify, aegis.notify.channels, aegis.notify.outbox\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'alpaca'"
        " or m.startswith(('aegis.policy', 'aegis.brain', 'aegis.exec')))\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == ""
