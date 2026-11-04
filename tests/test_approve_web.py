"""The phone approval page (aegis/approve_web): who may answer, what an
answer writes, and how every wrong request is turned away. Offline: the
store is a temp file, the clock fixed, and the one socket test talks to
127.0.0.1 only."""

import threading
import urllib.error
import urllib.request
from datetime import timedelta
from urllib.parse import urlencode

import pytest
from policy_factories import NOW, WATCHLIST, long_call
from test_policy_dispatch import LATER, _judged

from aegis.approve_web import auth
from aegis.approve_web.app import (
    LOCKED,
    PIN_FAILED,
    REFUSED,
    SECURITY_HEADERS,
    App,
    Request,
    Settings,
    failures_in_a_row,
    unlock,
)
from aegis.approve_web.server import LOOPBACK, make_server
from aegis.config import AegisConfig, ApproveWebConfig, BrokerConfig, ConfigError
from aegis.policy import dispatch
from aegis.policy.errors import PlacementError
from aegis.store import open_store
from aegis.store.models import ApprovalResponse
from aegis.store.repo import get_decision_approval, get_events_since

PIN = "482916"
PIN_HASH = auth.hash_pin(PIN)
URL = "https://aegis.example-tailnet.ts.net"
HOST = "aegis.example-tailnet.ts.net"
LOGIN = "operator@example.com"
CONFIG = AegisConfig(
    watchlist=list(WATCHLIST), broker=BrokerConfig(enabled=True),
    approve_web=ApproveWebConfig(enabled=True),
)
SETTINGS = Settings(page_url=URL, login=LOGIN, pin_hash=auth.parse_pin_hash(PIN_HASH))


# --- fixtures -------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "approve.db"
    open_store(path).close()
    return path


@pytest.fixture
def decision(db):
    conn = open_store(db)
    try:
        _, verdict = _judged(conn, long_call())
    finally:
        conn.close()
    assert verdict.verdict.value == "NEEDS_APPROVAL"
    return verdict


class Clock:
    def __init__(self):
        self.now = LATER

    def __call__(self):
        return self.now


class Ticks:
    """A monotonic clock for the form tokens."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def ticks():
    return Ticks()


@pytest.fixture
def app(db, ticks):
    return App(CONFIG, db, SETTINGS, clock=Clock(),
               tokens=auth.FormTokens(CONFIG.approve_web.form_ttl_seconds, monotonic=ticks))


def _headers(**extra):
    return {"Host": HOST, auth.IDENTITY_HEADER: LOGIN, **extra}


def _get(app, path, **headers):
    return app.handle(Request("GET", path, headers or _headers()))


def _token(app, decision_id):
    response = _get(app, f"/d/{decision_id}")
    assert response.status == 200
    marker = "name=token value='"
    start = response.html.index(marker) + len(marker)
    return response.html[start:response.html.index("'", start)]


def _post(app, decision_id, *, token, pin=PIN, answer="approve", headers=None, body=None):
    form = body if body is not None else urlencode({"token": token, "pin": pin, "answer": answer})
    sent = headers if headers is not None else _headers(
        **{"Content-Type": "application/x-www-form-urlencoded", "Origin": URL,
           "Sec-Fetch-Site": "same-origin"})
    return app.handle(Request("POST", f"/d/{decision_id}", sent, form.encode()))


def _kinds(db, kind):
    conn = open_store(db)
    try:
        return get_events_since(conn, NOW - timedelta(days=1), kind=kind)
    finally:
        conn.close()


def _approval(db, decision_id):
    conn = open_store(db)
    try:
        return get_decision_approval(conn, decision_id)
    finally:
        conn.close()


# --- the PIN --------------------------------------------------------------------


class TestPin:
    def test_a_hash_checks_its_own_pin_and_no_other(self):
        stored = auth.parse_pin_hash(PIN_HASH)
        assert auth.pin_ok(PIN, stored)
        assert not auth.pin_ok("482917", stored)
        assert not auth.pin_ok("", stored)
        assert not auth.pin_ok("x" * 65, stored)

    def test_the_hash_holds_no_dollar_sign_and_differs_each_time(self):
        assert "$" not in PIN_HASH and PIN not in PIN_HASH
        assert auth.hash_pin(PIN) != auth.hash_pin(PIN)

    @pytest.mark.parametrize("pin", ["12345", " 123456", "123456 ", "12\n3456", "x" * 65])
    def test_a_short_or_odd_pin_is_refused(self, pin):
        with pytest.raises(ValueError):
            auth.hash_pin(pin)

    @pytest.mark.parametrize("text", [
        "", "bcrypt:1:2:3:4:5", "scrypt:abc:8:1:AAAA:BBBB", "scrypt:1024:8:1:AAAAAAAAAAAAAAAAAAAAAA:"
        + "A" * 43, PIN_HASH + ":extra",
    ])
    def test_a_bad_or_weak_hash_is_a_config_error_that_does_not_quote_it(self, text):
        with pytest.raises(ConfigError) as caught:
            auth.parse_pin_hash(text)
        assert text not in str(caught.value) or not text


# --- the URL and the login ------------------------------------------------------


class TestUrlAndLogin:
    def test_a_serve_origin_is_kept_without_a_trailing_slash(self):
        assert auth.check_page_url("https://AEGIS.example-tailnet.ts.net/") == URL

    @pytest.mark.parametrize("url", [
        "http://aegis.example-tailnet.ts.net", "https://aegis.example.com", "https://ts.net",
        "https://aegis.example-tailnet.ts.net/approve", "https://u:p@aegis.example-tailnet.ts.net",
        "https://aegis.example-tailnet.ts.net?x=1", "https://aegis.ts.net.evil.com",
    ])
    def test_anything_but_a_tailnet_origin_is_refused(self, url):
        with pytest.raises(ConfigError):
            auth.check_page_url(url)

    def test_the_login_is_matched_without_case_and_never_blank(self):
        assert auth.identity_ok("Operator@Example.com", LOGIN)
        assert not auth.identity_ok(None, LOGIN)
        assert not auth.identity_ok("", LOGIN)
        assert not auth.identity_ok("someone@example.com", LOGIN)
        with pytest.raises(ConfigError):
            auth.check_login("two words")


# --- the form tokens ------------------------------------------------------------


class TestFormTokens:
    def test_a_token_works_once_for_its_own_decision(self, ticks):
        tokens = auth.FormTokens(300, monotonic=ticks)
        token = tokens.issue("d-1")
        assert not tokens.redeem(token, "d-2")  # wrong decision: and now it is spent
        assert not tokens.redeem(token, "d-1")
        again = tokens.issue("d-1")
        assert tokens.redeem(again, "d-1")
        assert not tokens.redeem(again, "d-1")

    def test_a_token_dies_after_its_time(self, ticks):
        tokens = auth.FormTokens(300, monotonic=ticks)
        token = tokens.issue("d-1")
        ticks.t += 300
        assert not tokens.redeem(token, "d-1")

    def test_no_token_and_an_unknown_token_are_refused(self, ticks):
        tokens = auth.FormTokens(300, monotonic=ticks)
        assert not tokens.redeem(None, "d-1") and not tokens.redeem("guess", "d-1")

    def test_the_store_of_tokens_is_bounded(self, ticks):
        tokens = auth.FormTokens(300, monotonic=ticks)
        first = tokens.issue("d-1")
        for _ in range(auth.MAX_LIVE_TOKENS):
            tokens.issue("d-1")
        assert not tokens.redeem(first, "d-1")


# --- the gate every request passes ------------------------------------------------


class TestGate:
    def test_no_identity_header_is_turned_away(self, app, decision):
        assert _get(app, f"/d/{decision.id}", Host=HOST).status == 403

    def test_another_login_is_turned_away(self, app, decision):
        response = _get(app, "/", Host=HOST, **{auth.IDENTITY_HEADER: "someone@example.com"})
        assert response.status == 403 and decision.id not in response.html

    def test_a_foreign_host_is_turned_away(self, app):
        assert _get(app, "/", Host="evil.example.com", **{auth.IDENTITY_HEADER: LOGIN}).status == 421

    def test_the_loopback_host_serve_forwards_to_is_accepted(self, app):
        host = f"127.0.0.1:{CONFIG.approve_web.port}"
        assert _get(app, "/", Host=host, **{auth.IDENTITY_HEADER: LOGIN}).status == 200

    @pytest.mark.parametrize("method", ["PUT", "DELETE", "HEAD", "OPTIONS"])
    def test_other_methods_are_refused(self, app, method):
        assert app.handle(Request(method, "/", _headers())).status == 405

    @pytest.mark.parametrize("path", ["/admin", "/d/", "/d/../etc", "/d/a/b", "/d/" + "a" * 65])
    def test_unknown_paths_are_not_found(self, app, path):
        assert _get(app, path).status == 404

    def test_every_response_carries_the_security_headers(self, app):
        for response in (_get(app, "/"), _get(app, "/nope"), _get(app, "/", Host=HOST)):
            for name, value in SECURITY_HEADERS.items():
                assert response.headers[name] == value
        assert "frame-ancestors 'none'" in SECURITY_HEADERS["Content-Security-Policy"]


# --- reading ------------------------------------------------------------------------


class TestReading:
    def test_the_list_shows_what_waits(self, app, decision):
        response = _get(app, "/")
        assert response.status == 200 and f"/d/{decision.id}" in response.html

    def test_a_decision_page_shows_the_order_and_a_form(self, app, decision):
        response = _get(app, f"/d/{decision.id}")
        assert response.status == 200
        assert "SPY" in response.html and "name=pin" in response.html
        assert "<script" not in response.html.lower()

    def test_model_text_is_escaped(self, db, ticks):
        conn = open_store(db)
        try:
            _, verdict = _judged(conn, long_call(thesis="<script>alert(1)</script> & more"))
        finally:
            conn.close()
        app = App(CONFIG, db, SETTINGS, clock=Clock(), tokens=auth.FormTokens(300, monotonic=ticks))
        html = _get(app, f"/d/{verdict.id}").html
        assert "<script>" not in html and "&lt;script&gt;" in html

    def test_an_unknown_decision_is_not_waiting(self, app, decision):
        response = _get(app, "/d/no-such-decision")
        assert response.status == 404 and "no such decision" in response.html


# --- answering ------------------------------------------------------------------------


class TestAnswering:
    def test_a_yes_is_recorded_as_a_phone_answer_by_the_responder(self, app, db, decision):
        response = _post(app, decision.id, token=_token(app, decision.id))
        assert response.status == 200 and "Approved" in response.html
        approval = _approval(db, decision.id)
        assert approval.response is ApprovalResponse.APPROVED
        assert approval.channel == "phone" and approval.responder == "ayush"
        assert approval.expires_at == LATER + timedelta(seconds=900)
        [event] = _kinds(db, dispatch.APPROVAL_ANSWERED)
        assert event.payload["channel"] == "phone"
        assert LOGIN not in event.message and LOGIN not in str(event.payload)

    def test_a_no_is_recorded_and_carries_no_expiry(self, app, db, decision):
        response = _post(app, decision.id, token=_token(app, decision.id), answer="reject")
        assert response.status == 200 and "Rejected" in response.html
        approval = _approval(db, decision.id)
        assert approval.response is ApprovalResponse.REJECTED and approval.expires_at is None

    def test_a_replayed_form_writes_nothing(self, app, db, decision):
        token = _token(app, decision.id)
        assert _post(app, decision.id, token=token).status == 200
        replay = _post(app, decision.id, token=token, answer="reject")
        assert replay.status == 403
        assert _approval(db, decision.id).response is ApprovalResponse.APPROVED
        assert len(_kinds(db, REFUSED)) == 1

    def test_a_second_answer_is_refused_by_the_store_rules(self, app, db, decision, ticks):
        _post(app, decision.id, token=_token(app, decision.id))
        fresh = app._tokens.issue(decision.id)  # a token the page would not hand out any more
        response = _post(app, decision.id, token=fresh, answer="reject")
        assert response.status == 409 and "already answered" in response.html
        assert _approval(db, decision.id).response is ApprovalResponse.APPROVED

    def test_a_page_answered_elsewhere_is_no_longer_offered(self, app, db, decision):
        conn = open_store(db)
        try:
            dispatch.answer(conn, decision.id, approved=False, by="ayush", config=CONFIG,
                            clock=lambda: LATER)
        finally:
            conn.close()
        assert _get(app, f"/d/{decision.id}").status == 404

    def test_an_expired_form_is_refused(self, app, db, decision, ticks):
        token = _token(app, decision.id)
        ticks.t += CONFIG.approve_web.form_ttl_seconds
        assert _post(app, decision.id, token=token).status == 403
        assert _approval(db, decision.id) is None

    def test_a_token_for_another_decision_is_refused(self, app, db, decision):
        token = app._tokens.issue("some-other-decision")
        assert _post(app, decision.id, token=token).status == 403
        assert _approval(db, decision.id) is None

    @pytest.mark.parametrize("headers", [
        {"Origin": "https://evil.example.com"},
        {"Sec-Fetch-Site": "cross-site"},
    ])
    def test_a_cross_site_post_is_refused(self, app, db, decision, headers):
        token = _token(app, decision.id)
        sent = _headers(**{"Content-Type": "application/x-www-form-urlencoded", **headers})
        assert _post(app, decision.id, token=token, headers=sent).status == 400
        assert _approval(db, decision.id) is None

    def test_a_post_without_identity_writes_nothing(self, app, db, decision):
        token = _token(app, decision.id)
        sent = {"Host": HOST, "Content-Type": "application/x-www-form-urlencoded"}
        assert _post(app, decision.id, token=token, headers=sent).status == 403
        assert _approval(db, decision.id) is None

    @pytest.mark.parametrize("body", [
        "token=a&pin=1&answer=maybe", "token=a&token=b&pin=1&answer=approve", "x" * 3000,
    ])
    def test_a_malformed_form_is_refused(self, app, db, decision, body):
        assert _post(app, decision.id, token="", body=body).status == 400
        assert _approval(db, decision.id) is None

    def test_a_form_must_be_form_encoded(self, app, db, decision):
        token = _token(app, decision.id)
        sent = _headers(**{"Content-Type": "application/json", "Origin": URL})
        assert _post(app, decision.id, token=token, headers=sent).status == 400

    def test_get_never_writes(self, app, db, decision):
        _get(app, f"/d/{decision.id}")
        _get(app, "/")
        assert _approval(db, decision.id) is None


# --- the PIN lockout --------------------------------------------------------------------


class TestLockout:
    def test_a_wrong_pin_writes_nothing_but_a_warning(self, app, db, decision):
        response = _post(app, decision.id, token=_token(app, decision.id), pin="000000")
        assert response.status == 403 and "4 tries left" in response.html
        assert _approval(db, decision.id) is None
        [event] = _kinds(db, PIN_FAILED)
        assert event.level.value == "warning" and "000000" not in str(event.payload)

    def test_too_many_wrong_pins_lock_the_page_until_unlocked(self, app, db, decision):
        for _ in range(CONFIG.approve_web.max_pin_failures):
            last = _post(app, decision.id, token=_token(app, decision.id), pin="000000")
        assert last.status == 423 and len(_kinds(db, LOCKED)) == 1
        assert _get(app, f"/d/{decision.id}").status == 423
        token = app._tokens.issue(decision.id)
        assert _post(app, decision.id, token=token).status == 423  # even the right PIN
        assert _approval(db, decision.id) is None
        conn = open_store(db)
        try:
            unlock(conn, "ayush", LATER)
            assert failures_in_a_row(conn) == 0
        finally:
            conn.close()
        assert _post(app, decision.id, token=_token(app, decision.id)).status == 200

    def test_a_phone_answer_clears_the_count(self, app, db, decision):
        _post(app, decision.id, token=_token(app, decision.id), pin="000000")
        _post(app, decision.id, token=_token(app, decision.id))
        conn = open_store(db)
        try:
            assert failures_in_a_row(conn) == 0
        finally:
            conn.close()


# --- the answer channel in the dispatcher -------------------------------------------------


def test_the_dispatcher_takes_only_known_channels(db, decision):
    conn = open_store(db)
    try:
        with pytest.raises(PlacementError, match="unknown answer channel"):
            dispatch.answer(conn, decision.id, approved=True, by="ayush", config=CONFIG,
                            channel="llm", clock=lambda: LATER)
        assert get_decision_approval(conn, decision.id) is None
    finally:
        conn.close()
    assert dispatch.APPROVAL_CHANNELS == {"cli", "phone"}


# --- the socket ---------------------------------------------------------------------------


def test_the_server_listens_on_the_loopback_address_only(app, decision):
    assert LOOPBACK == "127.0.0.1"
    server = make_server(app, 0, 5)
    host, port = server.server_address
    assert host == "127.0.0.1"
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/", headers={"Host": HOST, auth.IDENTITY_HEADER: LOGIN})
    try:
        with urllib.request.urlopen(request, timeout=5) as reply:
            body = reply.read().decode()
            assert reply.status == 200 and reply.headers["X-Frame-Options"] == "DENY"
    finally:
        thread.join(5)
        server.server_close()
    assert decision.id in body


def test_the_server_refuses_an_oversized_body_before_reading_it(app):
    server = make_server(app, 0, 5)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/d/x", data=b"x" * 5000, method="POST",
        headers={"Host": HOST, auth.IDENTITY_HEADER: LOGIN})
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        assert caught.value.code == 413
    finally:
        thread.join(5)
        server.server_close()
