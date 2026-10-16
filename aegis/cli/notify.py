"""Tell the operator what happened: the operator's side of Phase 7.

Run:  python -m aegis.cli.notify pending [--db PATH]
      python -m aegis.cli.notify send [--db PATH]
      python -m aegis.cli.notify test

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

``main(argv, *, channel_factory=None, clock=None)``: tests inject a stand-in
channel and a fixed clock.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
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
from aegis.config import AegisConfig, ConfigError, NotifyConfig, get_config
from aegis.data.models import utcnow
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice
from aegis.notify.channels import channel_from_env
from aegis.notify.outbox import Channel, outbox, send_due
from aegis.store.db import open_store
from aegis.store.errors import StoreError
from aegis.store.models import EventLevel

ChannelFactory = Callable[[NotifyConfig], Channel]

NO_EVENTS = "there are no events in it"
TEST_EVENT_ID = "test"
TEST_TITLE = "AEGIS test"
TEST_BODY = "The notifier can reach you. Nothing happened in AEGIS; this is a test."


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
    test_help = "push one test notice (no database needed)"
    commands.add_parser("test", help=test_help, description=test_help)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    channel_factory: ChannelFactory | None = None,
    clock: Callable[[], datetime] | None = None,
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
    try:
        config = get_config()
        if args.command == "test":
            return _test(config, factory, now_of())
        db_path = _db_path(args.db)
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
