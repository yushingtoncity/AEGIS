"""The approval page's edges: the Review link in alerts, the config block,
and ``python -m aegis.cli.approve_web``. Offline: stand-in sessions, temp
stores, a fixed clock and a monkeypatched environment."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from test_notify import FakeSession, _event, _notice

from aegis.approve_web import auth
from aegis.approve_web.app import PIN_FAILED, UNLOCKED
from aegis.cli import approve_web as cli
from aegis.cli import notify as notify_cli
from aegis.config import AegisConfig, ApproveWebConfig, load_config
from aegis.notify import ntfy
from aegis.notify.models import Notice
from aegis.notify.outbox import notice_for, review_link
from aegis.store import Event, EventLevel, get_events_since, log_event, open_store

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
URL = "https://aegis.example-tailnet.ts.net"
DECISION = "0b9c1d2e-3f40-4a5b-8c6d-7e8f90a1b2c3"
WATCHLIST = ["SPY"]


def _needs(decision_id=DECISION):
    return _event(kind="policy_needs_approval", level=EventLevel.INFO,
                  payload={"decision_id": decision_id, "symbol": "SPY"})


# --- the Review link ---------------------------------------------------------------


class TestReviewLink:
    def test_an_approval_alert_links_to_its_decision_page(self):
        notice = notice_for(_needs(), URL)
        assert notice.link == f"{URL}/d/{DECISION}"
        assert f"Review: {URL}/d/{DECISION}" in notice.body
        assert "orders approve" in notice.body  # the CLI way stays

    def test_no_url_no_link(self):
        notice = notice_for(_needs())
        assert notice.link is None and "Review:" not in notice.body

    def test_only_approvals_get_a_link(self):
        assert review_link(_event(payload={"decision_id": DECISION}), URL) is None

    @pytest.mark.parametrize("decision_id", ["a,b", "a b", "../x", "", "x" * 65, 7])
    def test_an_odd_decision_id_gets_no_link(self, decision_id):
        assert review_link(_needs(decision_id), URL) is None

    def test_ntfy_turns_the_link_into_a_tap_and_a_button(self):
        session = FakeSession()
        link = f"{URL}/d/{DECISION}"
        ntfy.NtfyChannel("https://ntfy.example/t", timeout=1, session=session).send(
            _notice(link=link))
        headers = session.calls[0][1]["headers"]
        assert headers["Click"] == link
        assert headers["Actions"] == f"view, Review, {link}, clear=true"

    def test_a_notice_without_a_link_sends_no_action(self):
        session = FakeSession()
        ntfy.NtfyChannel("https://ntfy.example/t", timeout=1, session=session).send(_notice())
        assert "Actions" not in session.calls[0][1]["headers"]

    @pytest.mark.parametrize("link", ["http://x.ts.net/d/1", "https://x.ts.net/d/1,evil",
                                      "https://x.ts.net/d/1;x", "https://x.ts.net/d 1"])
    def test_a_notice_holds_only_a_clean_https_link(self, link):
        with pytest.raises(ValidationError):
            _notice(link=link)

    def test_the_notify_cli_adds_the_link_only_while_the_page_is_on(self, monkeypatch):
        monkeypatch.setattr(notify_cli, "load_env", lambda: None)
        monkeypatch.setenv(auth.URL_ENV, URL + "/")
        off = AegisConfig(watchlist=WATCHLIST)
        on = AegisConfig(watchlist=WATCHLIST, approve_web=ApproveWebConfig(enabled=True))
        assert notify_cli._review_url(off) is None
        assert notify_cli._review_url(on) == URL

    def test_a_bad_url_only_drops_the_link(self, monkeypatch, capsys):
        monkeypatch.setattr(notify_cli, "load_env", lambda: None)
        monkeypatch.setenv(auth.URL_ENV, "https://public.example.com")
        on = AegisConfig(watchlist=WATCHLIST, approve_web=ApproveWebConfig(enabled=True))
        assert notify_cli._review_url(on) is None
        err = capsys.readouterr().err
        assert "no Review link" in err and "public.example.com" not in err


# --- the config block --------------------------------------------------------------


class TestConfig:
    def test_shipped_off_with_the_documented_values(self):
        web = load_config().approve_web
        assert web.enabled is False
        assert web.port == 8765 and web.responder == "ayush"
        assert web.form_ttl_seconds == 300 and web.max_pin_failures == 5

    @pytest.mark.parametrize("fields", [
        {"enabled": "true"}, {"enabled": 1}, {"port": 80}, {"port": 70000}, {"responder": ""},
        {"responder": "a b"}, {"form_ttl_seconds": 0}, {"form_ttl_seconds": 3600},
        {"max_pin_failures": 0}, {"max_pin_failures": 50}, {"login": "x"},
    ])
    def test_bad_values_are_refused(self, fields):
        with pytest.raises(ValidationError):
            ApproveWebConfig(**fields)


# --- the CLI -------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "web.db"
    open_store(path).close()
    return path


@pytest.fixture
def config(monkeypatch):
    holder = {"config": AegisConfig(watchlist=WATCHLIST)}
    monkeypatch.setattr(cli, "get_config", lambda: holder["config"])
    monkeypatch.setattr(cli, "load_env", lambda: None)
    return holder


def _answers(*replies):
    queue = list(replies)
    return lambda prompt: queue.pop(0)


class TestCli:
    def test_hash_pin_prints_a_usable_line_and_never_the_pin(self, capsys):
        assert cli.main(["hash-pin"], ask=_answers("482916", "482916")) == 0
        out = capsys.readouterr().out
        line = out.splitlines()[0]
        assert line.startswith(f"{auth.PIN_HASH_ENV}=scrypt:") and "482916" not in out
        stored = auth.parse_pin_hash(line.split("=", 1)[1])
        assert auth.pin_ok("482916", stored)

    def test_hash_pin_refuses_two_different_pins(self, capsys):
        assert cli.main(["hash-pin"], ask=_answers("482916", "482917")) == 1
        assert "differ" in capsys.readouterr().err

    def test_hash_pin_refuses_a_short_pin(self, capsys):
        assert cli.main(["hash-pin"], ask=_answers("1234")) == 1

    def test_serve_refuses_while_switched_off(self, config, db, capsys):
        assert cli.main(["serve", "--db", str(db)]) == 1
        assert "approve_web.enabled is false" in capsys.readouterr().err

    def test_serve_refuses_an_incomplete_env_and_names_only_the_variable(
        self, config, db, monkeypatch, capsys
    ):
        config["config"] = AegisConfig(watchlist=WATCHLIST, approve_web=ApproveWebConfig(enabled=True))
        monkeypatch.setenv(auth.URL_ENV, URL)
        monkeypatch.setenv(auth.LOGIN_ENV, "operator@example.com")
        monkeypatch.setenv(auth.PIN_HASH_ENV, "not-a-hash")
        assert cli.main(["serve", "--db", str(db)]) == 1
        err = capsys.readouterr().err
        assert auth.PIN_HASH_ENV in err and "not-a-hash" not in err
        monkeypatch.delenv(auth.LOGIN_ENV)
        assert cli.main(["serve", "--db", str(db)]) == 1
        assert auth.LOGIN_ENV in capsys.readouterr().err

    def test_status_says_set_or_missing_never_the_value(self, config, db, monkeypatch, capsys):
        monkeypatch.setenv(auth.URL_ENV, URL)
        monkeypatch.delenv(auth.LOGIN_ENV, raising=False)
        monkeypatch.delenv(auth.PIN_HASH_ENV, raising=False)
        assert cli.main(["status", "--db", str(db)]) == 0
        out = capsys.readouterr().out
        assert f"{auth.URL_ENV}: set" in out and f"{auth.LOGIN_ENV}: missing" in out
        assert URL not in out and "0 of 5 (open)" in out

    def test_unlock_clears_the_count_on_the_record(self, config, db, capsys):
        conn = open_store(db)
        try:
            for _ in range(5):
                log_event(conn, Event(occurred_at=NOW, level=EventLevel.WARNING, kind=PIN_FAILED,
                                      message="wrong PIN"))
        finally:
            conn.close()
        assert cli.main(["status", "--db", str(db)]) == 0
        assert "5 of 5 (LOCKED)" in capsys.readouterr().out
        later = NOW + timedelta(minutes=1)
        assert cli.main(["unlock", "--by", "ayush", "--db", str(db)], clock=lambda: later) == 0
        assert "5 wrong PINs cleared" in capsys.readouterr().out
        conn = open_store(db)
        try:
            [event] = get_events_since(conn, NOW, kind=UNLOCKED)
        finally:
            conn.close()
        assert event.payload["by"] == "ayush"

    def test_unlock_needs_a_name(self, config, db, capsys):
        assert cli.main(["unlock", "--by", " ", "--db", str(db)]) == 1

    def test_a_missing_store_is_one_line(self, config, tmp_path, capsys):
        assert cli.main(["unlock", "--by", "ayush", "--db", str(tmp_path / "none.db")]) == 1
        assert "does not exist" in capsys.readouterr().err


def test_the_notice_link_field_defaults_to_none():
    assert Notice.model_fields["link"].default is None
