"""The friends' feed (``aegis.notify.feed``), its Discord channel
(``aegis.notify.discord``) and the ``notify feed-*`` commands.

Stores are tmp_path files; the channel and the HTTP session are stand-ins,
the clock is fixed and the account is a stand-in AccountState with made-up
figures. Nothing reaches Discord, a broker or Anthropic.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest
import requests

from aegis.cli import notify as cli
from aegis.config import AegisConfig, ConfigError, DEFAULT_CONFIG_PATH, FeedConfig, NotifyConfig, load_config
from aegis.data.models import AccountState, Position
from aegis.notify import discord
from aegis.notify.errors import NotifyError
from aegis.notify.feed import (
    FEED_RECAP,
    already_recapped,
    day_change,
    feed_outbox,
    post_for,
    recap_event,
    send_feed,
)
from aegis.notify.models import Notice
from aegis.notify.outbox import NOTIFY_FAILED, NOTIFY_SENT, outbox
from aegis.notify.summary import NEW_YORK
from aegis.store import Event, EventLevel, get_events_since, log_event, open_store
from aegis.store.models import (
    Broker,
    OptionType,
    Order,
    OrderSide,
    OrderStatus,
    Proposal,
    ProposalLeg,
)
from aegis.store.repo import insert_proposal, upsert_order

DAY = date(2026, 10, 28)
NOW = datetime.combine(DAY, time(16, 21), tzinfo=NEW_YORK).astimezone(timezone.utc)
ON = FeedConfig(enabled=True)
WEBHOOK = "https://discord.com/api/webhooks/123456/s3cret-token"
SECRET = "s3cret"
THESIS = "Data-center demand is accelerating and guidance looks conservative. " * 10


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "feed.db")
    yield connection
    connection.close()


def _clock():
    return NOW


def _event(kind, payload, minutes_ago=5, level=EventLevel.INFO, message="operator text"):
    return Event(occurred_at=NOW - timedelta(minutes=minutes_ago), level=level, kind=kind,
                 message=message, payload=payload)


def _proposal(conn, *, symbol="NVDA", instrument="equity", side="buy", quantity=3, limit_price=101.5,
              thesis=THESIS, legs=()) -> Proposal:
    return insert_proposal(conn, Proposal(
        cycle_id="c1", symbol=symbol, instrument=instrument, side=side, quantity=quantity,
        order_type="limit", limit_price=limit_price, thesis=thesis, confidence=0.62,
        invalidation="i", raw_model_output="{}", model_name="m", prompt_version="v",
    ), list(legs))


def _proposed(conn, **kwargs) -> Proposal:
    proposal = _proposal(conn, **kwargs)
    log_event(conn, _event("proposal", {"cycle_id": "c1", "proposal_id": proposal.id, "symbol": proposal.symbol},
                           message=f"cycle c1 proposed buy 3 equity NVDA ({proposal.id})"))
    return proposal


def _filled(conn, *, status="filled", kind="order_updated", side=OrderSide.BUY, coid="aegis-o1",
            minutes_ago=3, instrument=None) -> Order:
    proposal = _proposal(conn)
    order = upsert_order(conn, Order(
        proposal_id=proposal.id, client_order_id=coid, broker=Broker.PAPER, status=OrderStatus.SUBMITTED,
        symbol="NVDA", side=side, quantity=3, instrument=instrument,
    ))
    log_event(conn, _event(kind, {
        "client_order_id": coid, "broker_order_id": "b-1", "broker_status": status,
        "filled_quantity": 3.0, "avg_fill_price": 100.25,
    }, minutes_ago=minutes_ago, message=f"{coid}: {status} (3 of 3 filled)"))
    return order


class FakeChannel:
    def __init__(self, failing=()):
        self.name = discord.CHANNEL_NAME
        self.failing = set(failing)
        self.sent: list[Notice] = []

    def send(self, notice):
        if notice.event_id in self.failing:
            raise NotifyError("post to Discord", notice.event_id, detail="HTTP 429")
        self.sent.append(notice)


def _account(**overrides) -> AccountState:
    fields = dict(account_number="PA1234567", status="ACTIVE", equity=100_512.40, last_equity=100_000.00,
                  positions=[Position(symbol="SPY", qty=1, side="long", unrealized_pl=12.0)])
    return AccountState(**{**fields, **overrides})


# --- config -------------------------------------------------------------------


class TestConfig:
    def test_shipped_block_is_the_code_defaults_and_off(self):
        shipped = load_config(DEFAULT_CONFIG_PATH).feed
        assert shipped == FeedConfig()
        assert shipped.enabled is False and shipped.show_amounts is False

    def test_defaults(self):
        config = FeedConfig()
        assert config.posts == ("proposals", "fills", "recap")
        assert (config.thesis_chars, config.max_attempts, config.max_per_run) == (280, 3, 10)

    @pytest.mark.parametrize("bad", [
        {"enabled": "yes"}, {"enabled": 1}, {"show_amounts": "true"}, {"posts": ["trades"]},
        {"thesis_chars": -1}, {"thesis_chars": 5000}, {"max_attempts": 0}, {"max_per_run": 0},
        {"lookback_minutes": 0}, {"timeout_seconds": 0}, {"webhook_url": WEBHOOK},
    ])
    def test_refused(self, bad):
        with pytest.raises(ValueError):
            FeedConfig(**bad)

    def test_the_feed_has_no_say_in_the_operators_channel(self):
        with pytest.raises(ValueError):
            NotifyConfig(channel="discord")


# --- what friends read --------------------------------------------------------


class TestProposalPost:
    def test_ticker_direction_confidence_and_a_cut_thesis(self, conn):
        _proposed(conn)
        post = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts[0]
        assert post.notice.title == "AEGIS proposed: buy NVDA"
        lines = post.notice.body.splitlines()
        assert lines[0] == "Stock. Model confidence 62%."
        assert lines[1] == "The risk rules judge it next, so it may never trade."
        assert lines[2].startswith("Thesis: Data-center demand") and lines[2].endswith("...")
        assert len(lines[2]) == len("Thesis: ") + ON.thesis_chars

    def test_no_amounts_by_default(self, conn):
        proposal = _proposed(conn)
        body = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts[0].notice.body
        assert "3 shares" not in body and "$" not in body and "101" not in body
        assert proposal.id not in body and "operator text" not in body

    def test_amounts_when_opted_in(self, conn):
        _proposed(conn)
        config = FeedConfig(enabled=True, show_amounts=True)
        body = feed_outbox(conn, config, discord.CHANNEL_NAME, NOW).posts[0].notice.body
        assert body.startswith("Stock, 3 shares, limit $101.50. Model confidence 62%.")

    def test_thesis_off(self, conn):
        _proposed(conn)
        body = feed_outbox(conn, FeedConfig(enabled=True, thesis_chars=0), discord.CHANNEL_NAME, NOW).posts[0].notice.body
        assert "Thesis" not in body

    def test_options_name_the_legs_only_with_amounts(self, conn):
        legs = [
            ProposalLeg(proposal_id="p-opt", leg_index=n, symbol=f"NVDA261120C00{100 + 10 * n}000",
                        option_type=OptionType.CALL, side=OrderSide.BUY if n == 0 else OrderSide.SELL,
                        quantity=1, strike=100 + 10 * n, expiration=date(2026, 11, 20))
            for n in range(2)
        ]
        proposal = insert_proposal(conn, Proposal(
            id="p-opt", cycle_id="c1", symbol="NVDA", instrument="option", side="buy", quantity=1,
            order_type="limit", limit_price=3.2, thesis="t", confidence=0.55, invalidation="i",
            raw_model_output="{}", model_name="m", prompt_version="v",
        ), legs)
        event = log_event(conn, _event("proposal", {"proposal_id": proposal.id}))
        assert post_for(conn, event, ON).notice.body.startswith("Options, 2 legs. Model confidence 55%.")
        shown = post_for(conn, event, FeedConfig(show_amounts=True)).notice.body
        assert shown.startswith("Options, 2 legs: buy 1 2026-11-20 100 call, sell 1 2026-11-20 110 call, limit $3.20.")

    def test_a_missing_proposal_is_skipped(self, conn):
        event = log_event(conn, _event("proposal", {"proposal_id": "nope"}))
        assert post_for(conn, event, ON) is None


class TestFillPost:
    def test_a_full_fill(self, conn):
        _filled(conn)
        post = feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts[0]
        assert post.key == "fill:aegis-o1"
        assert post.notice.title == "AEGIS filled: bought NVDA"
        assert post.notice.body == "Stock order filled on the paper account."

    def test_a_sale_and_amounts(self, conn):
        _filled(conn, side=OrderSide.SELL)
        post = feed_outbox(conn, FeedConfig(enabled=True, show_amounts=True), discord.CHANNEL_NAME, NOW).posts[0]
        assert post.notice.title == "AEGIS filled: sold NVDA"
        assert post.notice.body == "Stock order filled on the paper account. 3 shares at $100.25 average."

    @pytest.mark.parametrize("status", ["new", "accepted", "partially_filled", "canceled"])
    def test_anything_short_of_filled_is_not_posted(self, conn, status):
        _filled(conn, status=status)
        assert feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts == ()

    def test_an_order_reported_filled_twice_is_posted_once(self, conn):
        _filled(conn, kind="order_submitted", minutes_ago=4)
        log_event(conn, _event("order_updated", {"client_order_id": "aegis-o1", "broker_status": "filled"}, minutes_ago=2))
        channel = FakeChannel()
        send_feed(conn, ON, channel, _clock)
        assert len(channel.sent) == 1
        assert send_feed(conn, ON, channel, _clock).sent == 0


class TestWhatNeverGoesOut:
    @pytest.mark.parametrize("kind,level", [
        ("policy_needs_approval", EventLevel.INFO), ("approval_answered", EventLevel.INFO),
        ("kill_switch", EventLevel.CRITICAL), ("order_refused", EventLevel.WARNING),
        ("orphan_broker_order", EventLevel.CRITICAL), ("daily_summary", EventLevel.INFO),
        ("brain_error", EventLevel.ERROR), ("cycle_end", EventLevel.INFO),
    ])
    def test_operator_events(self, conn, kind, level):
        log_event(conn, _event(kind, {"decision_id": "d-1", "symbol": "NVDA"}, level=level))
        assert feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts == ()

    def test_kinds_switched_off(self, conn):
        _proposed(conn)
        _filled(conn, coid="aegis-o2")
        assert feed_outbox(conn, FeedConfig(enabled=True, posts=()), discord.CHANNEL_NAME, NOW).posts == ()
        only_fills = feed_outbox(conn, FeedConfig(enabled=True, posts=("fills",)), discord.CHANNEL_NAME, NOW)
        assert [post.key for post in only_fills.posts] == ["fill:aegis-o2"]


# --- sending ------------------------------------------------------------------


class TestSend:
    def test_refuses_while_off(self, conn):
        with pytest.raises(NotifyError, match="feed.enabled is false"):
            send_feed(conn, FeedConfig(), FakeChannel(), _clock)

    def test_sends_once_and_records_on_its_own_channel(self, conn):
        _proposed(conn)
        channel = FakeChannel()
        report = send_feed(conn, ON, channel, _clock)
        assert (report.sent, report.failed) == (1, 0)
        sent = get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_SENT)
        assert [e.payload["channel"] for e in sent] == [discord.CHANNEL_NAME]
        assert send_feed(conn, ON, channel, _clock).sent == 0
        assert len(channel.sent) == 1

    def test_the_operators_outbox_is_untouched(self, conn):
        """Records on the feed's channel neither mark nor send anything for the operator."""
        _filled(conn)
        send_feed(conn, ON, FakeChannel(), _clock)
        operator = outbox(conn, NotifyConfig(enabled=True), "imessage", NOW)
        assert [n.kind for n in operator.notices] == ["order_updated"]

    def test_failures_are_retried_then_given_up(self, conn):
        proposal = _proposed(conn)
        event_id = get_events_since(conn, NOW - timedelta(hours=1), kind="proposal")[0].id
        channel = FakeChannel(failing={event_id})
        for attempt in range(1, ON.max_attempts + 1):
            report = send_feed(conn, ON, channel, _clock)
            assert report.failed == 1
        failed = get_events_since(conn, NOW - timedelta(hours=1), kind=NOTIFY_FAILED)
        assert [e.payload["attempt"] for e in failed] == [1, 2, 3]
        assert all(e.payload["key"] == f"proposal:{proposal.id}" for e in failed)
        assert feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).given_up == 1

    def test_old_events_are_not_replayed(self, conn):
        proposal = _proposal(conn)
        log_event(conn, _event("proposal", {"proposal_id": proposal.id}, minutes_ago=ON.lookback_minutes + 1))
        assert feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts == ()

    def test_max_per_run(self, conn):
        for n in range(3):
            _filled(conn, coid=f"aegis-o{n}", minutes_ago=5 - n)
        due = feed_outbox(conn, FeedConfig(enabled=True, max_per_run=2), discord.CHANNEL_NAME, NOW)
        assert [p.key for p in due.posts] == ["fill:aegis-o0", "fill:aegis-o1"] and due.waiting == 1


# --- the recap ----------------------------------------------------------------


class TestRecap:
    def _day(self, conn):
        for hour in (10, 12, 14, 15):
            at = datetime.combine(DAY, time(hour, 2), tzinfo=NEW_YORK)
            log_event(conn, Event(occurred_at=at, level=EventLevel.INFO, kind="cycle_end", message="x",
                                  payload={"cycle_id": f"c{hour}"}))
        _proposed(conn)
        _filled(conn)

    def test_counts_and_percent_only(self, conn):
        self._day(conn)
        event = log_event(conn, recap_event(conn, DAY, _account(), NOW))
        post = post_for(conn, event, ON)
        assert post.key == "recap:2026-10-28"
        assert post.notice.title == "AEGIS day recap, Wed Oct 28"
        assert post.notice.body == "4 brain cycles, 1 proposal, 1 fill.\nPaper account: up 0.51% today."
        for private in ("100,512", "512.40", "PA1234567", "1234", "SPY"):
            assert private not in post.notice.body

    def test_dollars_with_amounts(self, conn):
        event = log_event(conn, recap_event(conn, DAY, _account(), NOW))
        body = post_for(conn, event, FeedConfig(show_amounts=True)).notice.body
        assert body.endswith("Paper account: up 0.51% ($512.40) today.")
        assert "100,512" not in body

    def test_down_and_flat(self):
        assert day_change(_account(equity=99_000.0))[0] == pytest.approx(-1.0)
        assert day_change(_account(equity=None)) == (None, None)
        assert day_change(_account(last_equity=0.0)) == (None, None)

    def test_no_account(self, conn):
        event = log_event(conn, recap_event(conn, DAY, None, NOW))
        assert post_for(conn, event, ON).notice.body.endswith("Paper account change: not available today.")

    def test_once_a_day(self, conn):
        assert not already_recapped(conn, DAY)
        log_event(conn, recap_event(conn, DAY, None, NOW))
        assert already_recapped(conn, DAY)
        assert not already_recapped(conn, DAY + timedelta(days=1))

    def test_the_operator_does_not_get_it(self, conn):
        log_event(conn, recap_event(conn, DAY, _account(), NOW))
        assert outbox(conn, NotifyConfig(enabled=True), "imessage", NOW).notices == ()


# --- the Discord channel ------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self, status_code=204, error=None):
        self.status_code = status_code
        self.error = error
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return FakeResponse(self.status_code)


def _notice(title="AEGIS proposed: buy NVDA", body="Stock. Model confidence 62%."):
    return Notice(event_id="e1", occurred_at=NOW, level=EventLevel.INFO, kind="proposal", title=title, body=body)


class TestDiscordChannel:
    @pytest.mark.parametrize("url", [
        WEBHOOK, "https://discordapp.com/api/webhooks/1/t", "https://ptb.discord.com/api/webhooks/1/t",
    ])
    def test_accepted_urls(self, url):
        assert discord.check_webhook_url(f"  {url} ") == url

    @pytest.mark.parametrize("url", [
        "http://discord.com/api/webhooks/1/t", "https://discord.com.evil.example/api/webhooks/1/t",
        "https://evil.example/api/webhooks/1/t", "https://discord.com/channels/1/2",
        "https://discord.com:8443/api/webhooks/1/t", "https://user@discord.com/api/webhooks/1/t",
        "https://discord.com/api/webhooks/1/t?wait=true", "https://discord.com/api/webhooks/1/t#x",
        "not a url", "https://[::1",
    ])
    def test_refused_urls_never_echo_the_url(self, url):
        with pytest.raises(ConfigError) as caught:
            discord.check_webhook_url(url)
        assert "/1/t" not in str(caught.value) and "evil" not in str(caught.value)

    def test_the_request(self):
        session = FakeSession()
        discord.DiscordChannel(WEBHOOK, timeout=7, session=session).send(_notice())
        (call,) = session.calls
        assert call["url"] == WEBHOOK and call["timeout"] == 7 and call["allow_redirects"] is False
        payload = call["json"]
        assert payload["allowed_mentions"] == {"parse": []}
        assert payload["flags"] == discord.SUPPRESS_EMBEDS
        assert payload["content"].endswith("\n-# Paper account. Not investment advice.")
        assert payload["content"].startswith("**AEGIS proposed: buy NVDA**\n")

    def test_every_post_carries_the_disclaimer(self, conn):
        _proposed(conn)
        _filled(conn, coid="aegis-o9")
        log_event(conn, recap_event(conn, DAY, _account(), NOW))
        for post in feed_outbox(conn, ON, discord.CHANNEL_NAME, NOW).posts:
            assert discord.content_for(post.notice).endswith(discord.DISCLAIMER)

    def test_stored_text_cannot_ping_link_or_format(self):
        hostile = "@everyone [click](https://evil.example) **now** <@123> `x`\n-# fake small print"
        content = discord.content_for(_notice(body=hostile))
        assert "@everyone" not in content and "[click](" not in content and "**now**" not in content
        assert "<@123>" not in content and "\n-# fake" not in content
        assert content.count("\n-# ") == 1

    @pytest.mark.parametrize("status", [400, 404, 429, 500])
    def test_refused_post(self, status):
        channel = discord.DiscordChannel(WEBHOOK, timeout=1, session=FakeSession(status))
        with pytest.raises(NotifyError, match=f"HTTP {status}"):
            channel.send(_notice())

    def test_network_error_never_quotes_the_url(self):
        error = requests.ConnectionError(f"cannot reach {WEBHOOK}")
        channel = discord.DiscordChannel(WEBHOOK, timeout=1, session=FakeSession(error=error))
        with pytest.raises(NotifyError) as caught:
            channel.send(_notice())
        assert SECRET not in str(caught.value) and "ConnectionError" in str(caught.value)
        assert SECRET not in repr(channel)

    def test_from_env(self, monkeypatch):
        monkeypatch.setattr(discord, "load_env", lambda: None)
        monkeypatch.setenv(discord.WEBHOOK_ENV, WEBHOOK)
        assert isinstance(discord.from_env(ON), discord.DiscordChannel)
        monkeypatch.delenv(discord.WEBHOOK_ENV)
        with pytest.raises(ConfigError):
            discord.from_env(ON)


# --- the CLI ------------------------------------------------------------------


def _config(**feed) -> AegisConfig:
    return AegisConfig(watchlist=["SPY"], feed=FeedConfig(**feed))


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "cli.db"
    open_store(path).close()
    return path


def _run(argv, config, monkeypatch, channel=None, account=None):
    monkeypatch.setattr(cli, "get_config", lambda: config)
    reader = (lambda: account) if account is not None else (lambda: (_ for _ in ()).throw(RuntimeError("down")))
    return cli.main(argv, feed_channel_factory=lambda _: channel or FakeChannel(), clock=_clock,
                    account_reader=reader)


class TestCli:
    def test_send_refuses_while_off(self, store, monkeypatch, capsys):
        assert _run(["feed-send", "--db", str(store)], _config(), monkeypatch) == 1
        assert "feed.enabled is false" in capsys.readouterr().err

    def test_pending_prints_what_friends_would_read(self, store, monkeypatch, capsys):
        conn = open_store(store)
        _proposed(conn)
        conn.close()
        assert _run(["feed-pending", "--db", str(store)], _config(), monkeypatch) == 0
        out = capsys.readouterr().out
        assert "due now (1)" in out and "AEGIS proposed: buy NVDA" in out
        assert discord.DISCLAIMER in out and "posting is off" in out

    def test_send(self, store, monkeypatch, capsys):
        conn = open_store(store)
        _filled(conn)
        conn.close()
        channel = FakeChannel()
        assert _run(["feed-send", "--db", str(store)], _config(enabled=True), monkeypatch, channel) == 0
        assert [n.title for n in channel.sent] == ["AEGIS filled: bought NVDA"]
        assert "fill:aegis-o1: sent" in capsys.readouterr().out

    def test_a_failed_post_exits_1(self, store, monkeypatch, capsys):
        conn = open_store(store)
        _proposed(conn)
        event_id = get_events_since(conn, NOW - timedelta(hours=1), kind="proposal")[0].id
        conn.close()
        channel = FakeChannel(failing={event_id})
        assert _run(["feed-send", "--db", str(store)], _config(enabled=True), monkeypatch, channel) == 1
        assert "1 not posted" in capsys.readouterr().err

    def test_test_works_while_off(self, monkeypatch, capsys):
        channel = FakeChannel()
        assert _run(["feed-test"], _config(), monkeypatch, channel) == 0
        assert channel.sent[0].title == cli.FEED_TEST_TITLE

    def test_recap_once_a_day_and_dry_run(self, store, monkeypatch, capsys):
        argv = ["feed-recap", "--db", str(store), "--day", DAY.isoformat()]
        assert _run(argv + ["--dry-run"], _config(), monkeypatch, account=_account()) == 0
        assert "+0.51%" in capsys.readouterr().out
        conn = open_store(store)
        assert get_events_since(conn, NOW - timedelta(days=2), kind=FEED_RECAP) == []
        conn.close()
        assert _run(argv, _config(), monkeypatch, account=_account()) == 0
        assert _run(argv, _config(), monkeypatch, account=_account()) == 0
        assert "already written" in capsys.readouterr().out
        conn = open_store(store)
        assert len(get_events_since(conn, NOW - timedelta(days=2), kind=FEED_RECAP)) == 1
        conn.close()

    def test_recap_with_the_account_down(self, store, monkeypatch, capsys):
        assert _run(["feed-recap", "--db", str(store), "--day", DAY.isoformat()], _config(), monkeypatch) == 0
        assert "day change not available" in capsys.readouterr().out
