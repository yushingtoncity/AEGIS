"""The approval page's logic: one request in, one response out, no socket.

``App.handle`` checks, in this order, for every request: the method and
path are ones the page has; the Host is the page's own name or the
loopback address Serve forwards to; Serve's identity header names the one
allowed login. A POST must also come from the page itself (``Origin`` /
``Sec-Fetch-Site``), carry a form body under ``MAX_BODY_BYTES``, redeem a
one time token issued for that decision, find the page not locked, and
carry the right PIN. Only then is ``dispatch.answer`` called, with
``channel="phone"`` and the configured responder; it writes the answer and
its audit event together, refuses a second answer, and places nothing.

This module never imports the brain, the broker or any order method; the
only dispatcher functions it calls are ``answer`` and ``awaiting_approval``
(``tests/test_approve_web_architecture.py`` pins both).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from aegis.approve_web import pages
from aegis.approve_web.auth import IDENTITY_HEADER, FormTokens, PinHash, identity_ok, pin_ok
from aegis.config import AegisConfig
from aegis.policy import dispatch
from aegis.policy.context import load_proposal
from aegis.policy.errors import PlacementError, PolicyError
from aegis.store.db import open_store
from aegis.store.errors import StoreError
from aegis.store.models import Event, EventLevel, PolicyDecision
from aegis.store.repo import get_decision, get_events_since, log_event

PIN_FAILED = "approve_web_pin_failed"
LOCKED = "approve_web_locked"
UNLOCKED = "approve_web_unlocked"
REFUSED = "approve_web_refused"
MAX_BODY_BYTES = 2048
"""A form holds a token, a PIN and one word. Fixed: it bounds what a
request can make the server read, nothing about trading."""
_DECISION_PATH = re.compile(r"^/d/([A-Za-z0-9-]{1,64})$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_FORM_TYPE = "application/x-www-form-urlencoded"
_NOTE = "answered on the phone page; Tailscale login checked"

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline';"
    " form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@dataclass(frozen=True)
class Settings:
    """What the page needs from .env, already checked (``auth``)."""

    page_url: str
    login: str
    pin_hash: PinHash


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes = b""

    def header(self, name: str) -> str | None:
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


@dataclass(frozen=True)
class Response:
    status: int
    html: str
    headers: dict[str, str] = field(default_factory=lambda: dict(SECURITY_HEADERS))


def _event(now: datetime, level: EventLevel, kind: str, message: str, **payload: object) -> Event:
    return Event(occurred_at=now, level=level, kind=kind, message=message, payload=payload)


def failures_in_a_row(conn: sqlite3.Connection) -> int:
    """Wrong PINs since the last unlock or the last answer given on the phone."""
    marks = [(e.occurred_at, 1) for e in get_events_since(conn, _EPOCH, kind=PIN_FAILED)]
    marks += [(e.occurred_at, 0) for e in get_events_since(conn, _EPOCH, kind=UNLOCKED)]
    marks += [
        (e.occurred_at, 0)
        for e in get_events_since(conn, _EPOCH, kind=dispatch.APPROVAL_ANSWERED)
        if (e.payload or {}).get("channel") == dispatch.PHONE_CHANNEL
    ]
    count = 0
    for _, failed in sorted(marks, key=lambda mark: mark[0]):
        count = count + 1 if failed else 0
    return count


def unlock(conn: sqlite3.Connection, by: str, now: datetime) -> Event:
    """Clear a lockout (``approve-web unlock`` on the mini), on the record."""
    return log_event(conn, _event(
        now, EventLevel.WARNING, UNLOCKED, f"phone approval page unlocked by {by}", by=by,
        failures=failures_in_a_row(conn),
    ))


def _view(conn: sqlite3.Connection, decision: PolicyDecision) -> pages.OrderView:
    under_review = load_proposal(conn, decision.proposal_id)
    order = under_review.proposal
    legs = under_review.legs
    symbol = legs[0].symbol if len(legs) == 1 else order.symbol
    leg_note = f" ({len(legs)} legs)" if len(legs) > 1 else ""
    limit = "market" if order.limit_price is None else f"limit {order.limit_price:g}"
    escalated = tuple(
        str(row["rule"]) for row in decision.rules_evaluated if row.get("outcome") == "ESCALATE"
    )
    return pages.OrderView(
        decision_id=decision.id,
        order=f"{order.side.value} {order.quantity:g} {symbol}{leg_note} {limit}",
        escalated=escalated, decided_at=decision.decided_at, confidence=order.confidence,
        thesis=order.thesis, invalidation=order.invalidation,
    )


class App:
    """The page. ``clock`` gives aware UTC instants; tests fix it."""

    def __init__(
        self, config: AegisConfig, db_path: Path, settings: Settings, *,
        clock: Callable[[], datetime], tokens: FormTokens | None = None,
    ) -> None:
        self._config = config
        self._web = config.approve_web
        self._db_path = db_path
        self._settings = settings
        self._clock = clock
        self._tokens = tokens or FormTokens(self._web.form_ttl_seconds)
        self._host = urlsplit(settings.page_url).netloc.lower()
        self._loopback = {f"127.0.0.1:{self._web.port}", f"localhost:{self._web.port}"}

    # --- the gate every request passes ------------------------------------------

    def handle(self, request: Request) -> Response:
        if request.method not in ("GET", "POST"):
            return Response(405, pages.message("Not allowed", "Only GET and POST.", back=False))
        match = _DECISION_PATH.match(request.path)
        if request.path != "/" and match is None:
            return Response(404, pages.message("Not found", "No such page.", back=False))
        host = (request.header("Host") or "").strip().lower()
        if host != self._host and host not in self._loopback:
            return Response(421, pages.message("Wrong address", "Open the link from the alert.",
                                               back=False))
        if not identity_ok(request.header(IDENTITY_HEADER), self._settings.login):
            return Response(403, pages.message(
                "Not you", "This page only answers your own phone over Tailscale.", back=False))
        try:
            if request.method == "GET":
                return self._get(match.group(1) if match else None)
            if match is None:
                return Response(405, pages.message("Not allowed", "Answer from a decision page."))
            return self._post(request, match.group(1))
        except (StoreError, PolicyError) as exc:
            return Response(503, pages.message("Store unavailable", f"{type(exc).__name__}: try again."))

    # --- reading ------------------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        return open_store(self._db_path)

    def _get(self, decision_id: str | None) -> Response:
        conn = self._open()
        try:
            waiting = dispatch.awaiting_approval(conn, self._config, self._clock())
            if decision_id is None:
                return Response(200, pages.waiting_list([_view(conn, d) for d in waiting]))
            decision = next((d for d in waiting if d.id == decision_id), None)
            if decision is None:
                return Response(404, pages.message(
                    "Not waiting", self._why_not_waiting(conn, decision_id)))
            if failures_in_a_row(conn) >= self._web.max_pin_failures:
                return self._locked()
            view = _view(conn, decision)
        finally:
            conn.close()
        token = self._tokens.issue(decision_id)
        form_minutes = max(1, round(self._web.form_ttl_seconds / 60))
        approval_minutes = max(1, round(self._config.broker.approval_ttl_seconds / 60))
        return Response(200, pages.answer_form(view, token, form_minutes, approval_minutes))

    @staticmethod
    def _why_not_waiting(conn: sqlite3.Connection, decision_id: str) -> str:
        if get_decision(conn, decision_id) is None:
            return "There is no such decision."
        return "This decision is not waiting on you: it was answered, superseded, or is not from today."

    def _locked(self) -> Response:
        return Response(423, pages.message(
            "Locked", "Too many wrong PINs. On the mini run: python -m aegis.cli.approve_web"
            " unlock --by NAME"))

    # --- answering ----------------------------------------------------------------

    def _same_origin(self, request: Request) -> bool:
        origin = request.header("Origin")
        if origin is not None and origin.strip().lower() != self._settings.page_url.lower():
            return False
        site = request.header("Sec-Fetch-Site")
        return site is None or site.strip().lower() in ("same-origin", "none")

    def _form(self, request: Request) -> dict[str, str] | None:
        kind = (request.header("Content-Type") or "").split(";")[0].strip().lower()
        if kind != _FORM_TYPE or len(request.body) > MAX_BODY_BYTES:
            return None
        try:
            fields = parse_qs(request.body.decode("ascii"), strict_parsing=False, max_num_fields=8)
        except (UnicodeDecodeError, ValueError):
            return None
        if any(len(values) != 1 for values in fields.values()):
            return None
        return {name: values[0] for name, values in fields.items()}

    def _post(self, request: Request, decision_id: str) -> Response:
        form = self._form(request) if self._same_origin(request) else None
        if form is None or form.get("answer") not in ("approve", "reject"):
            return self._refuse(decision_id, "a bad or cross site form", 400)
        if not self._tokens.redeem(form.get("token"), decision_id):
            return self._refuse(decision_id, "a used, expired or foreign form token", 403)
        conn = self._open()
        try:
            if failures_in_a_row(conn) >= self._web.max_pin_failures:
                return self._locked()
            if not pin_ok(form.get("pin", ""), self._settings.pin_hash):
                return self._wrong_pin(conn, decision_id)
            return self._answer(conn, decision_id, form["answer"] == "approve")
        finally:
            conn.close()

    def _refuse(self, decision_id: str, why: str, status: int) -> Response:
        conn = self._open()
        try:
            log_event(conn, _event(
                self._clock(), EventLevel.WARNING, REFUSED,
                f"phone approval page refused {why} for decision {decision_id}",
                decision_id=decision_id, why=why,
            ))
        finally:
            conn.close()
        return Response(status, pages.message("Refused", "That form cannot be used. Reload the page"
                                                         " from the alert and try again."))

    def _wrong_pin(self, conn: sqlite3.Connection, decision_id: str) -> Response:
        failures = failures_in_a_row(conn) + 1
        limit = self._web.max_pin_failures
        log_event(conn, _event(
            self._clock(), EventLevel.WARNING, PIN_FAILED,
            f"phone approval page: wrong PIN for decision {decision_id} ({failures} of {limit})",
            decision_id=decision_id, failures=failures, limit=limit,
        ))
        if failures >= limit:
            log_event(conn, _event(
                self._clock(), EventLevel.ERROR, LOCKED,
                f"phone approval page locked after {failures} wrong PINs in a row",
                failures=failures,
            ))
            return self._locked()
        return Response(403, pages.message(
            "Wrong PIN", f"{limit - failures} tries left before the page locks."))

    def _answer(self, conn: sqlite3.Connection, decision_id: str, approved: bool) -> Response:
        try:
            answer = dispatch.answer(
                conn, decision_id, approved=approved, by=self._web.responder, config=self._config,
                note=_NOTE, channel=dispatch.PHONE_CHANNEL, clock=self._clock,
            )
        except PlacementError as exc:
            return Response(409, pages.message("Not recorded", str(exc)))
        if not approved:
            return Response(200, pages.message("Rejected", "Recorded. Nothing will be placed."))
        return Response(200, pages.message(
            "Approved",
            f"Recorded, good until {pages.when(answer.expires_at)}. The next tick re-checks the"
            " order on fresh data and places it only if every rule still allows it."))
