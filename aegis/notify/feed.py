"""The friends' feed: what the paper account does, posted to a Discord channel.

A second audience next to the operator's notifications, with its own
config block (``feed:``) and its own records. It reads the same ``events``
table, but a post is never an event's message: each one is built here from
a short allowlist of fields looked up in the store, so whatever the
operator's notices carry (approval commands, decision ids, account
numbers, equity, positions, errors, the kill switch) has no path into it.

Three kinds of post, each switched on in ``feed.posts``:

- ``proposals``: a ``proposal`` event the brain wrote. The ticker, the
  direction, the instrument, the model's confidence and the first
  ``feed.thesis_chars`` characters of its thesis. Posted before the risk
  rules judge it, and it says so.
- ``fills``: an ``order_submitted`` / ``order_updated`` event whose broker
  status is ``filled``. Which ticker was bought or sold.
- ``recap``: the ``feed_recap`` event ``notify feed-recap`` writes after
  the close, at most one per New York day. Counts of cycles, proposals and
  fills, and the account's change on the day in percent.

Share counts, prices and dollar amounts are added only with
``feed.show_amounts``. Equity and positions are never posted.

Sending works like the operator's outbox (``aegis.notify.outbox``): each
attempt is recorded as ``notify_sent`` / ``notify_failed`` with
``channel: discord_feed``, which is how the feed remembers what went out.
One difference: a post is keyed by what it tells about (``proposal:<id>``,
``fill:<client order id>``, ``recap:<day>``), not by the event, so an order
reported filled twice is still posted once. The push and its record are two
steps: a crash between them posts that one again, never loses it.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from aegis.config import FeedConfig
from aegis.data.models import AccountState
from aegis.notify.errors import NotifyError
from aegis.notify.models import BODY_MAX, TITLE_MAX, Notice
from aegis.notify.outbox import NOTIFY_FAILED, NOTIFY_SENT, Channel
from aegis.notify.summary import day_bounds
from aegis.store.models import Event, EventLevel, Instrument, Order, OrderSide, Proposal
from aegis.store.repo import (
    get_events_since,
    get_order,
    get_proposal,
    get_proposal_legs,
    log_event,
)

FEED_RECAP = "feed_recap"
PROPOSAL_KIND = "proposal"
"""The kind ``aegis.brain.cycle`` logs once a proposal is stored."""
CYCLE_END_KIND = "cycle_end"
FILL_KINDS = frozenset({"order_submitted", "order_updated"})
"""The dispatcher's kinds (``dispatch.ORDER_SUBMITTED`` / ``ORDER_UPDATED``),
spelled here rather than imported so the feed needs nothing from the order path."""
FILLED = "filled"
"""The broker's word for an order done in full; a partial fill is not posted."""

Clock = Callable[[], datetime]


def _one_line(text: object) -> str:
    """Control and format characters out, whitespace runs to one space."""
    kept = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in str(text))
    return " ".join(kept.split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)].rstrip() + "..."


def _qty(value: float) -> str:
    return f"{value:g}"


def _money(value: float) -> str:
    return f"${abs(value):,.2f}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


# --- what a post says ---------------------------------------------------------


@dataclass(frozen=True)
class Post:
    """One post due, and the key that says what it tells about."""

    key: str
    notice: Notice


def _notice(event: Event, title: str, body: str) -> Notice:
    return Notice(
        event_id=event.id, occurred_at=event.occurred_at, level=EventLevel.INFO,
        kind=event.kind, title=_cut(title, TITLE_MAX), body=_cut(body, BODY_MAX),
    )


def _symbol(text: str) -> str:
    return _cut(_one_line(text), 12) or "?"


def _proposal_post(conn: sqlite3.Connection, event: Event, config: FeedConfig) -> Post | None:
    proposal_id = (event.payload or {}).get("proposal_id")
    if not isinstance(proposal_id, str):
        return None
    proposal = get_proposal(conn, proposal_id)
    if proposal is None:
        return None
    return Post(f"proposal:{proposal.id}", _notice(
        event, f"AEGIS proposed: {proposal.side.value} {_symbol(proposal.symbol)}",
        _proposal_body(conn, proposal, config),
    ))


def _proposal_body(conn: sqlite3.Connection, proposal: Proposal, config: FeedConfig) -> str:
    if proposal.instrument is Instrument.OPTION:
        legs = get_proposal_legs(conn, proposal.id)
        what = f"Options, {_plural(len(legs), 'leg')}" if legs else "Options"
        if config.show_amounts and legs:
            what += ": " + ", ".join(
                f"{leg.side.value} {_qty(leg.quantity)} {leg.expiration.isoformat()}"
                f" {_qty(leg.strike)} {leg.option_type.value}"
                for leg in legs
            )
    else:
        what = "Stock"
        if config.show_amounts:
            what += f", {_plural_qty(proposal.quantity, 'share')}"
    if config.show_amounts and proposal.limit_price is not None:
        what += f", limit {_money(proposal.limit_price)}"
    lines = [
        f"{what}. Model confidence {proposal.confidence:.0%}.",
        "The risk rules judge it next, so it may never trade.",
    ]
    thesis = _one_line(proposal.thesis)
    if config.thesis_chars and thesis:
        lines.append(f"Thesis: {_cut(thesis, config.thesis_chars)}")
    return "\n".join(lines)


def _plural_qty(value: float, word: str) -> str:
    return f"{_qty(value)} {word}" if value == 1 else f"{_qty(value)} {word}s"


def _fill_post(conn: sqlite3.Connection, event: Event, config: FeedConfig) -> Post | None:
    payload = event.payload or {}
    client_order_id = payload.get("client_order_id")
    if payload.get("broker_status") != FILLED or not isinstance(client_order_id, str):
        return None
    order = get_order(conn, client_order_id)
    if order is None:
        return None
    return Post(f"fill:{order.client_order_id}", _notice(
        event, _fill_title(order), _fill_body(order, payload, config)
    ))


def _fill_title(order: Order) -> str:
    symbol = _symbol(order.symbol)
    if order.instrument is Instrument.OPTION:
        return f"AEGIS filled: options order on {symbol}"
    verb = "bought" if order.side is OrderSide.BUY else "sold"
    return f"AEGIS filled: {verb} {symbol}"


def _fill_body(order: Order, payload: dict[str, Any], config: FeedConfig) -> str:
    what = "Options order" if order.instrument is Instrument.OPTION else "Stock order"
    line = f"{what} filled on the paper account."
    quantity, price = payload.get("filled_quantity"), payload.get("avg_fill_price")
    if config.show_amounts and isinstance(quantity, (int, float)) and quantity > 0:
        unit = "contract" if order.instrument is Instrument.OPTION else "share"
        line += f" {_plural_qty(float(quantity), unit)}"
        if isinstance(price, (int, float)) and price > 0:
            line += f" at {_money(float(price))} average"
        line += "."
    return line


def _recap_post(event: Event, config: FeedConfig) -> Post | None:
    payload = event.payload or {}
    day = payload.get("day")
    if not isinstance(day, str):
        return None
    try:
        label = date.fromisoformat(day)
    except ValueError:
        return None
    counts = [
        _plural(int(payload.get(name) or 0), word)
        for name, word in (("cycles", "brain cycle"), ("proposals", "proposal"), ("fills", "fill"))
    ]
    lines = [", ".join(counts) + "."]
    pct, usd = payload.get("change_pct"), payload.get("change_usd")
    if isinstance(pct, (int, float)):
        change = "flat" if round(pct, 2) == 0 else f"{'up' if pct > 0 else 'down'} {abs(pct):.2f}%"
        if config.show_amounts and isinstance(usd, (int, float)) and round(pct, 2) != 0:
            change += f" ({_money(usd)})"
        lines.append(f"Paper account: {change} today.")
    else:
        lines.append("Paper account change: not available today.")
    title = f"AEGIS day recap, {label.strftime('%a %b')} {label.day}"
    return Post(f"recap:{day}", _notice(event, title, "\n".join(lines)))


def post_for(conn: sqlite3.Connection, event: Event, config: FeedConfig) -> Post | None:
    """The post that tells friends about ``event``, or None when it is not
    one the feed posts (another kind, a kind switched off in ``feed.posts``,
    an order not filled, or a record it points to that cannot be found)."""
    if event.kind == PROPOSAL_KIND and "proposals" in config.posts:
        return _proposal_post(conn, event, config)
    if event.kind in FILL_KINDS and "fills" in config.posts:
        return _fill_post(conn, event, config)
    if event.kind == FEED_RECAP and "recap" in config.posts:
        return _recap_post(event, config)
    return None


# --- what is due --------------------------------------------------------------


@dataclass(frozen=True)
class FeedOutbox:
    """What a run would post now, oldest first, and what it holds back."""

    posts: tuple[Post, ...]
    waiting: int
    given_up: int
    failures: dict[str, int]


def _aware(now: datetime) -> datetime:
    return now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now


def _keys(conn: sqlite3.Connection, since: datetime, kind: str, channel: str) -> list[str]:
    found = []
    for record in get_events_since(conn, since, kind=kind):
        payload = record.payload or {}
        if payload.get("channel") == channel and isinstance(payload.get("key"), str):
            found.append(payload["key"])
    return found


def feed_outbox(
    conn: sqlite3.Connection, config: FeedConfig, channel: str, now: datetime
) -> FeedOutbox:
    """Read what is due on ``channel`` at ``now`` (naive taken as UTC). Writes nothing."""
    now = _aware(now)
    since = now - timedelta(minutes=config.lookback_minutes)
    sent = set(_keys(conn, since, NOTIFY_SENT, channel))
    failures: dict[str, int] = {}
    for key in _keys(conn, since, NOTIFY_FAILED, channel):
        failures[key] = failures.get(key, 0) + 1
    due: list[Post] = []
    seen: set[str] = set()
    given_up = 0
    for event in get_events_since(conn, since):
        if event.occurred_at > now:
            continue
        post = post_for(conn, event, config)
        if post is None or post.key in sent or post.key in seen:
            continue
        seen.add(post.key)
        if failures.get(post.key, 0) >= config.max_attempts:
            given_up += 1
            continue
        due.append(post)
    taken = due[: config.max_per_run]
    return FeedOutbox(
        posts=tuple(taken), waiting=len(due) - len(taken), given_up=given_up, failures=failures
    )


# --- sending ------------------------------------------------------------------


@dataclass(frozen=True)
class FeedReport:
    lines: tuple[str, ...]
    sent: int = 0
    failed: int = 0
    waiting: int = 0
    given_up: int = 0


def _record(now: datetime, level: EventLevel, kind: str, message: str, **payload: Any) -> Event:
    return Event(occurred_at=now, level=level, kind=kind, message=_one_line(message), payload=payload)


def send_feed(
    conn: sqlite3.Connection, config: FeedConfig, channel: Channel, clock: Clock
) -> FeedReport:
    """Post everything due now, oldest first, and record each attempt.

    A channel failure is recorded as ``notify_failed`` and the run goes on;
    a store failure stops it. Refuses to run while ``feed.enabled`` is false.
    """
    if not config.enabled:
        raise NotifyError("post to the feed", detail="feed.enabled is false in config.yaml")
    due = feed_outbox(conn, config, channel.name, clock())
    lines: list[str] = []
    sent = failed = 0
    for post in due.posts:
        notice = post.notice
        attempt = due.failures.get(post.key, 0) + 1
        common = dict(event_id=notice.event_id, key=post.key, channel=channel.name, attempt=attempt)
        try:
            channel.send(notice)
        except NotifyError as exc:
            failed += 1
            log_event(conn, _record(
                clock(), EventLevel.WARNING, NOTIFY_FAILED,
                f"{post.key}: attempt {attempt} of {config.max_attempts} on {channel.name}"
                f" failed: {exc}",
                **common,
            ))
            lines.append(f"{post.key}: failed ({exc})")
            continue
        sent += 1
        log_event(conn, _record(
            clock(), EventLevel.INFO, NOTIFY_SENT, f"{post.key}: sent on {channel.name}", **common
        ))
        lines.append(f"{post.key}: sent")
    return FeedReport(
        lines=tuple(lines), sent=sent, failed=failed, waiting=due.waiting, given_up=due.given_up
    )


# --- the day's recap ----------------------------------------------------------


def already_recapped(conn: sqlite3.Connection, day: date) -> bool:
    """True when ``day`` (a New York day) already has its recap."""
    start, _ = day_bounds(day)
    return any(
        (event.payload or {}).get("day") == day.isoformat()
        for event in get_events_since(conn, start - timedelta(days=1), kind=FEED_RECAP)
    )


def day_change(account: AccountState | None) -> tuple[float | None, float | None]:
    """The account's change since the last close, in percent and dollars;
    None and None when the figures are missing."""
    if account is None or account.equity is None or account.last_equity is None:
        return None, None
    if account.last_equity <= 0:
        return None, None
    usd = account.equity - account.last_equity
    return usd / account.last_equity * 100, usd


def recap_event(
    conn: sqlite3.Connection, day: date, account: AccountState | None, now: datetime
) -> Event:
    """The ``feed_recap`` event for ``day``: the counts read from the store
    and the account's change. Writes nothing; the caller logs it."""
    start, end = day_bounds(day)
    events = [e for e in get_events_since(conn, start) if e.occurred_at < end]
    cycles = sum(1 for e in events if e.kind == CYCLE_END_KIND)
    proposals = sum(1 for e in events if e.kind == PROPOSAL_KIND)
    fills = len({
        (e.payload or {}).get("client_order_id")
        for e in events
        if e.kind in FILL_KINDS and (e.payload or {}).get("broker_status") == FILLED
    })
    pct, usd = day_change(account)
    shown = "not available" if pct is None else f"{pct:+.2f}%"
    return Event(
        occurred_at=now, level=EventLevel.INFO, kind=FEED_RECAP,
        message=(
            f"feed recap for {day.isoformat()}: {cycles} cycles, {proposals} proposals,"
            f" {fills} fills, day change {shown}"
        ),
        payload={
            "day": day.isoformat(), "cycles": cycles, "proposals": proposals, "fills": fills,
            "change_pct": pct, "change_usd": usd,
        },
    )
