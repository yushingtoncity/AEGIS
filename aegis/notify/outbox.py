"""Which events the operator hears about, and sending them once.

The notifier is a reader of the store's ``events`` table, which every part
of AEGIS already writes: a verdict that needs a human, an order reaching
the broker, a fill, an orphan order, the kill switch, a halt. It sends an
event when the event is at least ``notify.min_level`` or its kind is in
``notify.kinds`` (and not in ``notify.ignore_kinds``), it is younger than ``notify.lookback_minutes``, and it was
not sent before.

What it writes is its own events, through ``repo.log_event`` like any other:
``notify_sent`` (payload: the event id and the channel) after a push the
channel accepted, ``notify_failed`` after one it did not. Those records are
the memory of what went out, so the trading tables are never touched and no
schema changes. An event with ``max_attempts`` failures is given up on. The
notifier never sends an event of its own kinds (``notify_*``), so a broken
channel cannot feed itself.

The push and its record are two steps. If the process dies between them, or
the store refuses the record, the next run sends that notice again: a
duplicate push, never a lost one.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from aegis.config import NotifyConfig
from aegis.notify.errors import NotifyError
from aegis.notify.models import BODY_MAX, TITLE_MAX, Notice
from aegis.store.models import Event, EventLevel
from aegis.store.repo import get_events_since, log_event

NOTIFY_SENT = "notify_sent"
NOTIFY_FAILED = "notify_failed"
OWN_KIND_PREFIX = "notify_"
"""Every event kind the notifier writes starts with this, and none is ever sent."""

_LEVELS: tuple[EventLevel, ...] = (
    EventLevel.DEBUG, EventLevel.INFO, EventLevel.WARNING, EventLevel.ERROR, EventLevel.CRITICAL,
)

# ntfy's priority scale: 1 min, 3 default, 4 high, 5 urgent (overrides
# do-not-disturb on most phones). Urgent is kept for what needs action now.
_PRIORITY: dict[EventLevel, int] = {
    EventLevel.DEBUG: 2,
    EventLevel.INFO: 3,
    EventLevel.WARNING: 4,
    EventLevel.ERROR: 4,
    EventLevel.CRITICAL: 5,
}
_TAGS: dict[EventLevel, tuple[str, ...]] = {
    EventLevel.WARNING: ("warning",),
    EventLevel.ERROR: ("x",),
    EventLevel.CRITICAL: ("rotating_light",),
}

# The kinds a person reads more easily by a name than by the kind itself.
# The verdict and order kinds are the ones the policy engine and the
# dispatcher write (``engine.VERDICT_EVENT_KINDS``, ``dispatch.ORDER_*``);
# they are spelled here rather than imported, so the notifier needs nothing
# from the order path to run.
_TITLES: dict[str, str] = {
    "policy_needs_approval": "Approval needed",
    "approval_answered": "Approval answered",
    "order_submitted": "Order at the broker",
    "order_updated": "Order update",
    "order_unknown_outcome": "Order outcome unknown",
    "orphan_broker_order": "Unknown order at the broker",
    "risk_limit_tripped": "Daily loss limit tripped",
    "kill_switch": "Kill switch",
    "stand_down": "Stand-down",
    "daily_summary": "Day summary",
}
_MULTILINE_KINDS = frozenset({"daily_summary"})
"""Kinds whose message is laid out in lines (``aegis.notify.summary``): each
line is cleaned on its own and the breaks are kept, so a phone shows a list."""
_NEEDS_APPROVAL = "policy_needs_approval"
_APPROVE_HINT = "Approve: python -m aegis.cli.orders approve {decision_id} --by NAME"
_DECISION_ID_MAX = 64
"""A decision id is a uuid (36 characters); one longer than this is not one
the CLI would take, and the hint is left off rather than letting it crowd
out the message."""

Clock = Callable[[], datetime]


class Channel(Protocol):
    """Where notices go. ``send`` delivers one or raises ``NotifyError``."""

    name: str

    def send(self, notice: Notice) -> None: ...


# --- what goes out ------------------------------------------------------------


def level_rank(level: EventLevel) -> int:
    """Where ``level`` sits, debug 0 to critical 4."""
    return _LEVELS.index(level)


def wanted(event: Event, config: NotifyConfig) -> bool:
    """True for an event the operator should hear about, whether or not it
    was sent already: never one of the notifier's own or of an ignored kind,
    otherwise one at or above ``min_level`` or of a listed kind."""
    if event.kind.startswith(OWN_KIND_PREFIX) or event.kind in config.ignore_kinds:
        return False
    if event.kind in config.kinds:
        return True
    return level_rank(event.level) >= level_rank(EventLevel(config.min_level))


def _one_line(text: object) -> str:
    """Control and format characters out, whitespace runs to one space."""
    kept = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in str(text))
    return " ".join(kept.split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _title(event: Event) -> str:
    name = _TITLES.get(event.kind, event.kind)
    symbol = (event.payload or {}).get("symbol")
    label = f"{name} ({_one_line(symbol)})" if isinstance(symbol, str) and symbol.strip() else name
    prefix = "" if event.level in (EventLevel.DEBUG, EventLevel.INFO) else f"{event.level.value.upper()} "
    return _cut(f"AEGIS {prefix}{label}", TITLE_MAX)


def _body(event: Event) -> str:
    if event.kind in _MULTILINE_KINDS:
        lines = [_one_line(line) for line in str(event.message).splitlines()]
        message = "\n".join(line for line in lines if line) or event.kind
    else:
        message = _one_line(event.message) or event.kind
    hint = ""
    decision_id = (event.payload or {}).get("decision_id")
    if (event.kind == _NEEDS_APPROVAL and isinstance(decision_id, str)
            and 0 < len(decision_id.strip()) <= _DECISION_ID_MAX):
        hint = "\n\n" + _APPROVE_HINT.format(decision_id=_one_line(decision_id))
    return _cut(message, BODY_MAX - len(hint)) + hint


def notice_for(event: Event) -> Notice:
    """The push that tells about ``event``."""
    priority = _PRIORITY[event.level]
    tags = _TAGS.get(event.level, ())
    if event.kind == _NEEDS_APPROVAL:
        priority, tags = 4, ("bell",)  # an approval is only good for broker.approval_ttl_seconds
    return Notice(
        event_id=event.id, occurred_at=event.occurred_at, level=event.level, kind=event.kind,
        title=_title(event), body=_body(event), priority=priority, tags=tags,
    )


@dataclass(frozen=True)
class Outbox:
    """What a run would send now, oldest first, and what it holds back.

    ``waiting`` are wanted and unsent but past ``max_per_run``: the next run
    takes them. ``given_up`` failed ``max_attempts`` times and are never
    tried again. ``failures`` counts earlier failed attempts per event id.
    """

    notices: tuple[Notice, ...]
    waiting: int
    given_up: int
    failures: dict[str, int]


def _ids_in(records: Sequence[Event], channel: str) -> list[str]:
    found = []
    for record in records:
        payload: dict[str, Any] = record.payload or {}
        if payload.get("channel") == channel and isinstance(payload.get("event_id"), str):
            found.append(payload["event_id"])
    return found


def outbox(conn: sqlite3.Connection, config: NotifyConfig, channel: str, now: datetime) -> Outbox:
    """Read what is due on ``channel`` at ``now`` (naive taken as UTC). Writes nothing."""
    now = _aware(now)
    since = now - timedelta(minutes=config.lookback_minutes)
    sent = set(_ids_in(get_events_since(conn, since, kind=NOTIFY_SENT), channel))
    failures: dict[str, int] = {}
    for event_id in _ids_in(get_events_since(conn, since, kind=NOTIFY_FAILED), channel):
        failures[event_id] = failures.get(event_id, 0) + 1
    due: list[Event] = []
    given_up = 0
    for event in get_events_since(conn, since):
        if event.occurred_at > now or event.id in sent or not wanted(event, config):
            continue
        if failures.get(event.id, 0) >= config.max_attempts:
            given_up += 1
            continue
        due.append(event)
    taken = due[: config.max_per_run]
    return Outbox(
        notices=tuple(notice_for(event) for event in taken),
        waiting=len(due) - len(taken),
        given_up=given_up,
        failures=failures,
    )


def _aware(now: datetime) -> datetime:
    """A naive instant is taken as UTC, as the store's reads take it."""
    return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now


# --- sending ------------------------------------------------------------------


@dataclass(frozen=True)
class SendReport:
    """One run: a line per notice, and how many failed."""

    lines: tuple[str, ...]
    sent: int = 0
    failed: int = 0
    waiting: int = 0
    given_up: int = 0


def _record(now: datetime, level: EventLevel, kind: str, message: str, **payload: Any) -> Event:
    return Event(occurred_at=now, level=level, kind=kind, message=_one_line(message), payload=payload)


def send_due(
    conn: sqlite3.Connection, config: NotifyConfig, channel: Channel, clock: Clock
) -> SendReport:
    """Send every notice due now, oldest first, and record each attempt.

    A channel failure is recorded as ``notify_failed`` and the run goes on
    to the next notice; a store failure (``StoreError``) stops the run, as
    it does everywhere else. Refuses to run while ``notify.enabled`` is false.
    """
    if not config.enabled:
        raise NotifyError("send", detail="notify.enabled is false in config.yaml")
    due = outbox(conn, config, channel.name, clock())
    lines: list[str] = []
    sent = failed = 0
    for notice in due.notices:
        attempt = due.failures.get(notice.event_id, 0) + 1
        try:
            channel.send(notice)
        except NotifyError as exc:
            failed += 1
            log_event(conn, _record(
                clock(), EventLevel.WARNING, NOTIFY_FAILED,
                f"{notice.kind} {notice.event_id}: attempt {attempt} of {config.max_attempts}"
                f" on {channel.name} failed: {exc}",
                event_id=notice.event_id, channel=channel.name, attempt=attempt,
            ))
            lines.append(f"{notice.event_id} {notice.kind}: failed ({exc})")
            continue
        sent += 1
        log_event(conn, _record(
            clock(), EventLevel.INFO, NOTIFY_SENT,
            f"{notice.kind} {notice.event_id}: sent on {channel.name}",
            event_id=notice.event_id, channel=channel.name, attempt=attempt,
        ))
        lines.append(f"{notice.event_id} {notice.kind}: sent")
    return SendReport(
        lines=tuple(lines), sent=sent, failed=failed, waiting=due.waiting, given_up=due.given_up
    )
