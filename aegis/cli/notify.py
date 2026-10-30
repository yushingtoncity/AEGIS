"""Tell the operator what happened: the operator's side of Phase 7.

Run:  python -m aegis.cli.notify pending [--db PATH]
      python -m aegis.cli.notify send [--db PATH]
      python -m aegis.cli.notify test
      python -m aegis.cli.notify summary [--db PATH] [--day YYYY-MM-DD] [--logs DIR] [--dry-run]
      python -m aegis.cli.notify feed-pending [--db PATH]
      python -m aegis.cli.notify feed-send [--db PATH]
      python -m aegis.cli.notify feed-test
      python -m aegis.cli.notify feed-recap [--db PATH] [--day YYYY-MM-DD] [--dry-run]

``pending`` lists the notices a ``send`` would push now (the events at or
above ``notify.min_level`` or of a kind in ``notify.kinds``, younger than
``notify.lookback_minutes``, not sent yet), and how many are held back for
the next run or given up after ``notify.max_attempts`` failures. It puts the
connection in SQLite's query-only mode and writes no row.

``send`` pushes them, oldest first, and records each attempt as a
``notify_sent`` or ``notify_failed`` event. It needs ``notify.enabled: true``
in config.yaml (shipped false) and the channel's address in .env
(``IMESSAGE_TO``, ``SLACK_WEBHOOK_URL`` or ``NTFY_TOPIC_URL``). It migrates
an existing store but never creates one. The Phase 9 daemon will run it
after each tick; until then, run it by hand or from cron.

``test`` pushes one fixed notice and touches no database, so the channel
can be checked before anything else is set up. It works with
``notify.enabled`` off.

Failures are one line on stderr and exit 1, never a traceback, and never
print the topic URL or the token. ``send`` exits 1 when any notice failed.
Ctrl-C exits 130.

``summary`` writes the end-of-day summary (``aegis.notify.summary``) as one
``daily_summary`` event, for the next ``send`` to deliver; it writes at
most one per New York day and says so when the day already has one. It
reads the paper account once (a failed read is reported in the text, not
an exit 1) and, with ``--logs``, the launchd logs in that folder.
``--dry-run`` prints the text and writes nothing. It works with
``notify.enabled`` off: only ``send`` pushes anything.

The ``feed-*`` commands are the friends' feed (``aegis.notify.feed``), a
one-way Discord channel with its own ``feed:`` block in config.yaml and
``DISCORD_WEBHOOK_URL`` in .env. ``feed-pending`` prints, in full, every
post a ``feed-send`` would make now, so what friends would read can be
checked before the feed is turned on; it writes nothing. ``feed-send``
posts them and needs ``feed.enabled: true``. ``feed-test`` posts one fixed
message and touches no database. ``feed-recap`` writes the day's
``feed_recap`` event (at most one per New York day) for the next
``feed-send``; like ``summary`` it reads the paper account once and keeps
only the day's change.

``main(argv, *, channel_factory=None, feed_channel_factory=None, clock=None, account_reader=None)``:
tests inject stand-in channels and a fixed clock.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from aegis.cli.policy import (
    GAP,
    INDENT,
    _clean,
    _db_path,
    _Failure,
    _open_unchanged,
    _Parser,
    _require_file,
    _require_schema,
    _ts,
    _UsageError,
)
from aegis.config import AegisConfig, ConfigError, FeedConfig, NotifyConfig, get_config
from aegis.data.models import AccountState, utcnow
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice
from aegis.notify import discord
from aegis.notify.channels import channel_from_env
from aegis.notify.feed import already_recapped, feed_outbox, recap_event, send_feed
from aegis.notify.outbox import Channel, outbox, send_due
from aegis.notify.summary import (
    already_summarized,
    build_summary,
    new_york_day,
    read_log_lines,
    summary_event,
)
from aegis.store.db import open_store
from aegis.store.errors import StoreError
from aegis.store.models import EventLevel
from aegis.store.repo import log_event

ChannelFactory = Callable[[NotifyConfig], Channel]
FeedChannelFactory = Callable[[FeedConfig], Channel]
AccountReader = Callable[[], AccountState]

NO_EVENTS = "there are no events in it"
TEST_EVENT_ID = "test"
TEST_TITLE = "AEGIS test"
TEST_BODY = "The notifier can reach you. Nothing happened in AEGIS; this is a test."
FEED_TEST_TITLE = "AEGIS feed test"
FEED_TEST_BODY = "The AEGIS feed can reach this channel. Nothing happened; this is a test."


def _notice_line(notice: Notice) -> str:
    return GAP.join(
        _clean(cell)
        for cell in (notice.event_id, _ts(notice.occurred_at), f"p{notice.priority}", notice.title)
    )


def _pending(db_path: Path, config: AegisConfig, channel_name: str, now: datetime) -> int:
    conn = _open_unchanged(db_path, NO_EVENTS)
    try:
        due = outbox(conn, config.notify, channel_name, now)
    finally:
        conn.close()
    print(f"due now ({len(due.notices)})")
    for line in [_notice_line(notice) for notice in due.notices] or ["(none)"]:
        print(f"{INDENT}{line}")
    if due.waiting:
        print(f"{due.waiting} more wait for the next run (notify.max_per_run is {config.notify.max_per_run})")
    if due.given_up:
        print(f"{due.given_up} given up after {config.notify.max_attempts} failed attempts")
    if not config.notify.enabled:
        print("note: notify.enabled is false in config.yaml: sending is off")
    return 0


def _send(
    db_path: Path, config: AegisConfig, factory: ChannelFactory, clock: Callable[[], datetime]
) -> int:
    if not config.notify.enabled:
        raise _Failure("notify.enabled is false in config.yaml: nothing is sent")
    channel = factory(config.notify)
    _require_file(db_path, NO_EVENTS)
    _require_schema(db_path)
    conn = open_store(db_path)
    try:
        report = send_due(conn, config.notify, channel, clock)
    finally:
        conn.close()
    for line in report.lines or ("nothing to send",):
        print(_clean(line))
    if report.waiting:
        print(f"{report.waiting} more wait for the next run")
    if report.failed:
        print(f"{report.failed} not sent", file=sys.stderr)
        return 1
    return 0


def _test(config: AegisConfig, factory: ChannelFactory, now: datetime) -> int:
    channel = factory(config.notify)
    channel.send(Notice(
        event_id=TEST_EVENT_ID, occurred_at=now, level=EventLevel.INFO, kind="notify_test",
        title=TEST_TITLE, body=TEST_BODY, priority=3, tags=("white_check_mark",),
    ))
    print(f"test notice sent on {channel.name}")
    return 0


def _feed_pending(db_path: Path, config: AegisConfig, now: datetime) -> int:
    conn = _open_unchanged(db_path, NO_EVENTS)
    try:
        due = feed_outbox(conn, config.feed, discord.CHANNEL_NAME, now)
    finally:
        conn.close()
    print(f"due now ({len(due.posts)})")
    if not due.posts:
        print(f"{INDENT}(none)")
    for post in due.posts:  # in full: this is what friends would read
        print(f"{INDENT}{_clean(post.key)}")
        print(discord.content_for(post.notice))
    if due.waiting:
        print(f"{due.waiting} more wait for the next run (feed.max_per_run is {config.feed.max_per_run})")
    if due.given_up:
        print(f"{due.given_up} given up after {config.feed.max_attempts} failed attempts")
    if not config.feed.enabled:
        print("note: feed.enabled is false in config.yaml: posting is off")
    return 0


def _feed_send(
    db_path: Path, config: AegisConfig, factory: FeedChannelFactory, clock: Callable[[], datetime]
) -> int:
    if not config.feed.enabled:
        raise _Failure("feed.enabled is false in config.yaml: nothing is posted")
    channel = factory(config.feed)
    _require_file(db_path, NO_EVENTS)
    _require_schema(db_path)
    conn = open_store(db_path)
    try:
        report = send_feed(conn, config.feed, channel, clock)
    finally:
        conn.close()
    for line in report.lines or ("nothing to post",):
        print(_clean(line))
    if report.waiting:
        print(f"{report.waiting} more wait for the next run")
    if report.failed:
        print(f"{report.failed} not posted", file=sys.stderr)
        return 1
    return 0


def _feed_test(config: AegisConfig, factory: FeedChannelFactory, now: datetime) -> int:
    channel = factory(config.feed)
    channel.send(Notice(
        event_id=TEST_EVENT_ID, occurred_at=now, level=EventLevel.INFO, kind="notify_test",
        title=FEED_TEST_TITLE, body=FEED_TEST_BODY,
    ))
    print(f"test post sent on {channel.name}")
    return 0


def _feed_recap(
    args: argparse.Namespace,
    db_path: Path,
    now: datetime,
    account_reader: AccountReader | None,
) -> int:
    day = args.day if args.day is not None else new_york_day(now)
    if args.dry_run:
        conn = _open_unchanged(db_path, NO_EVENTS)
    else:
        _require_file(db_path, NO_EVENTS)
        _require_schema(db_path)
        conn = open_store(db_path)
    try:
        if not args.dry_run and already_recapped(conn, day):
            print(f"the feed recap for {day.isoformat()} was already written")
            return 0
        account, _ = _read_account(account_reader)  # a failed read: the recap says "not available"
        event = recap_event(conn, day, account, now)
        if not args.dry_run:
            log_event(conn, event)
    finally:
        conn.close()
    print(_clean(event.message))
    if not args.dry_run:
        print("written as a feed_recap event; the next `notify feed-send` posts it")
    return 0


def _read_account(reader: AccountReader | None) -> tuple[AccountState | None, str | None]:
    """The account state, or None and why not: the summary goes out without it."""
    if reader is None:
        from aegis.data.account import get_account_state  # the broker client loads only here

        reader = get_account_state
    try:
        return reader(), None
    except Exception as exc:  # any failure, named by type only: the text goes to a phone
        return None, type(exc).__name__


def _summary(
    args: argparse.Namespace,
    db_path: Path,
    config: AegisConfig,
    now: datetime,
    account_reader: AccountReader | None,
) -> int:
    day = args.day if args.day is not None else new_york_day(now)
    if args.dry_run:
        conn = _open_unchanged(db_path, NO_EVENTS)
    else:
        _require_file(db_path, NO_EVENTS)
        _require_schema(db_path)
        conn = open_store(db_path)
    try:
        if not args.dry_run and already_summarized(conn, day):
            print(f"the summary for {day.isoformat()} was already written")
            return 0
        account, error = _read_account(account_reader)
        log_lines = read_log_lines(Path(args.logs).expanduser(), day) if args.logs else None
        summary = build_summary(
            conn, day, config, now=now, account=account, account_error=error, log_lines=log_lines
        )
        if not args.dry_run:
            log_event(conn, summary_event(summary, now))
    finally:
        conn.close()
    print(summary.text)
    if not args.dry_run:
        print("written as a daily_summary event; the next `notify send` delivers it")
    return 0


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a date (YYYY-MM-DD): {text!r}") from None


def _parser() -> _Parser:
    parser = _Parser(
        prog="python -m aegis.cli.notify",
        description="Push what happened in AEGIS to the operator, from the store's events.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("pending", "list the notices a send would push now; write nothing"),
        ("send", "push every notice due now and record each attempt"),
    ):
        commands.add_parser(name, parents=[common], help=help_text, description=help_text)
    summary_help = "write the end-of-day summary for the next send (once a day)"
    summary = commands.add_parser(
        "summary", parents=[common], help=summary_help, description=summary_help
    )
    summary.add_argument("--day", type=_day, help="the New York day (default: today)")
    summary.add_argument("--logs", help="the launchd log folder to read step exit codes from")
    summary.add_argument("--dry-run", action="store_true", help="print the text, write nothing")
    test_help = "push one test notice (no database needed)"
    commands.add_parser("test", help=test_help, description=test_help)
    for name, help_text in (
        ("feed-pending", "print every post a feed-send would make now; write nothing"),
        ("feed-send", "post everything due to the friends' feed and record each attempt"),
    ):
        commands.add_parser(name, parents=[common], help=help_text, description=help_text)
    recap_help = "write the day's recap for the next feed-send (once a day)"
    recap = commands.add_parser(
        "feed-recap", parents=[common], help=recap_help, description=recap_help
    )
    recap.add_argument("--day", type=_day, help="the New York day (default: today)")
    recap.add_argument("--dry-run", action="store_true", help="print the recap, write nothing")
    feed_test_help = "post one test message to the friends' feed (no database needed)"
    commands.add_parser("feed-test", help=feed_test_help, description=feed_test_help)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    channel_factory: ChannelFactory | None = None,
    feed_channel_factory: FeedChannelFactory | None = None,
    clock: Callable[[], datetime] | None = None,
    account_reader: AccountReader | None = None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        args = _parser().parse_args(argv)
    except _UsageError as exc:
        print(f"{_clean(exc)} (see --help)", file=sys.stderr)
        return 1

    now_of = clock if clock is not None else utcnow
    factory: ChannelFactory = channel_factory if channel_factory is not None else channel_from_env
    feed_factory: FeedChannelFactory = (
        feed_channel_factory if feed_channel_factory is not None else discord.from_env
    )
    try:
        config = get_config()
        if args.command == "test":
            return _test(config, factory, now_of())
        if args.command == "feed-test":
            return _feed_test(config, feed_factory, now_of())
        db_path = _db_path(args.db)
        if args.command == "feed-pending":
            return _feed_pending(db_path, config, now_of())
        if args.command == "feed-send":
            return _feed_send(db_path, config, feed_factory, now_of)
        if args.command == "feed-recap":
            return _feed_recap(args, db_path, now_of(), account_reader)
        if args.command == "summary":
            return _summary(args, db_path, config, now_of(), account_reader)
        if args.command == "pending":
            return _pending(db_path, config, config.notify.channel, now_of())
        return _send(db_path, config, factory, now_of)
    except (_Failure, ConfigError, StoreError, NotifyError) as exc:
        print(f"notify {args.command} failed: {_clean(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks, and never the exception's text:
        # an HTTP library's message can quote the secret topic URL
        print(f"notify {args.command} failed (unexpected): {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
