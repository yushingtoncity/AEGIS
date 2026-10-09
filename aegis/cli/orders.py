"""Place and follow paper orders by hand: the operator's side of Phase 6.

Run:  python -m aegis.cli.orders pending [--db PATH]
      python -m aegis.cli.orders status [CLIENT_ORDER_ID] [--all] [--db PATH]
      python -m aegis.cli.orders approve DECISION_ID --by NAME [--note TEXT] [--db PATH]
      python -m aegis.cli.orders reject DECISION_ID --by NAME [--note TEXT] [--db PATH]
      python -m aegis.cli.orders place DECISION_ID [--dry-run] [--db PATH]
      python -m aegis.cli.orders sync [--db PATH]
      python -m aegis.cli.orders cancel CLIENT_ORDER_ID [--db PATH]
      python -m aegis.cli.orders stand-down [--db PATH]
      python -m aegis.cli.orders tick [--db PATH]

Every command goes through ``aegis.policy.dispatch``, the one caller of the
broker; this module only reads the store and prints. docs/phase6/SPEC_PHASE6.md
is the design.

``pending`` lists today's NEEDS_APPROVAL verdicts nobody has answered, and
the decisions ready to place (an AUTO_EXECUTE verdict still young enough, or
an approved one whose approval has not expired). ``status`` lists the open
orders (``--all``: every order placed), or one order with its fills.

``approve`` / ``reject`` record a human's answer to one NEEDS_APPROVAL
verdict, once; an approval is good for ``broker.approval_ttl_seconds``. They
place nothing. ``place`` re-checks the proposal on fresh data and, if it
still stands, claims and sends the one order; ``--dry-run`` runs every check
and the re-check, and writes and sends nothing. Placing needs
``broker.enabled: true`` in config.yaml (shipped false); a dry run does not.

``sync`` brings every open order up to date with the broker and looks for
orders the store does not hold (one turns the kill switch on). ``cancel``
cancels one order. ``stand-down`` cancels every working order while the
kill switch is on, a halt is in force or the halt cannot be read, and does
nothing otherwise. ``tick`` is sync, stand-down, then place everything that
may go. These four work with placing off: turning it off never strands an
order that is already working.

What may touch the database: ``pending``, ``status`` and ``place --dry-run``
put the connection in SQLite's query-only mode and write no row (a missing
or empty database, or a pending migration, is a clean error); every other
command migrates an existing store but never creates one. ``--db`` is as in
the policy CLI, and ``:memory:`` is refused.

Failures are one line on stderr and exit 1, never a traceback: a refused
placement, a broker error, or (for sync, cancel, stand-down and tick) any
order that could not be resolved. Ctrl-C exits 130.

``main(argv, *, broker_factory=None, context_builder=None, clock=None)``:
tests inject a stand-in broker, a hand-made context and a fixed clock.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from aegis.cli.policy import (
    GAP,
    INDENT,
    _clean,
    _db_path,
    _Failure,
    _open_unchanged,
    _Parser,
    _price,
    _qty,
    _require_file,
    _require_schema,
    _ts,
    _UsageError,
)
from aegis.config import AegisConfig, ConfigError, get_config
from aegis.data.models import utcnow
from aegis.policy import dispatch
from aegis.policy.context import load_proposal
from aegis.policy.errors import PolicyError
from aegis.store.db import open_store
from aegis.store.errors import StoreError
from aegis.store.models import Order, PolicyDecision
from aegis.store.repo import get_execution_orders, get_order, get_proposal_trace

DRY_RUN_NOTE = "DRY RUN: nothing was written and nothing was sent"
NO_DECISION = "there is no decision in it"
NO_ORDER = "there is no order in it"


# --- opening the store ------------------------------------------------------------


def _writable(db_path: Path, why_not_memory: str) -> sqlite3.Connection:
    """The store for a command that writes: it must exist and hold a schema
    (never created here); a pending migration is applied."""
    _require_file(db_path, why_not_memory)
    _require_schema(db_path)
    return open_store(db_path)


# --- printing ---------------------------------------------------------------------


def _row(*cells: object) -> str:
    """One line of cells, each cleaned on its own (no stored text can break
    the line or restyle the terminal), joined by the report's gap."""
    return GAP.join(_clean(cell) for cell in cells if cell != "")


def _lines(title: str, lines: Sequence[str]) -> None:
    """``title (N)`` then the lines, already made with ``_row``."""
    print(f"{title} ({len(lines)})")
    for line in lines or ("(none)",):
        print(f"{INDENT}{line}")


def _decision_line(conn: sqlite3.Connection, decision: PolicyDecision) -> str:
    proposal = load_proposal(conn, decision.proposal_id)
    order = proposal.proposal
    symbol = proposal.legs[0].symbol if len(proposal.legs) == 1 else order.symbol
    legs = f" ({len(proposal.legs)} legs)" if len(proposal.legs) > 1 else ""
    limit = "market" if order.limit_price is None else f"limit {_price(order.limit_price)}"
    escalated = [
        row["rule"] for row in decision.rules_evaluated if row.get("outcome") == "ESCALATE"
    ]
    return _row(
        decision.id,
        decision.verdict.value,
        f"{order.side.value} {_qty(order.quantity)} {symbol}{legs} {limit}",
        f"decided {_ts(decision.decided_at)}",
        f"escalated: {', '.join(escalated)}" if escalated else "",
    )


def _order_line(order: Order) -> str:
    average = "" if order.avg_fill_price is None else f" at {_price(order.avg_fill_price)}"
    limit = "-" if order.limit_price is None else _price(order.limit_price)
    return _row(
        order.client_order_id,
        order.status.value,
        f"{order.side.value} {_qty(order.quantity)} {order.symbol} limit {limit}",
        f"filled {_qty(order.filled_quantity)}/{_qty(order.quantity)}{average}",
        f"broker {order.broker_status or '-'}",
        f"claimed {_ts(order.submitted_at, 'never')}",
        f"({order.status_reason})" if order.status_reason else "",
    )


def _report(report: dispatch.Report) -> int:
    for line in report.lines:
        print(_clean(line))
    if report.failures:
        print(f"{report.failures} not resolved", file=sys.stderr)
        return 1
    return 0


# --- the commands -----------------------------------------------------------------


def _pending(db_path: Path, config: AegisConfig, now: datetime) -> int:
    conn = _open_unchanged(db_path, NO_DECISION)
    try:
        waiting = [_decision_line(conn, d) for d in dispatch.awaiting_approval(conn, config, now)]
        ready = [_decision_line(conn, d) for d in dispatch.ready_decisions(conn, config, now)]
    finally:
        conn.close()
    _lines("awaiting approval", waiting)
    _lines("ready to place", ready)
    if not config.broker.enabled:
        print("note: broker.enabled is false in config.yaml: placing is off")
    return 0


def _status(db_path: Path, client_order_id: str | None, show_all: bool) -> int:
    conn = _open_unchanged(db_path, NO_ORDER)
    try:
        if client_order_id is None:
            orders = get_execution_orders(conn, open_only=not show_all)
            fills = []
        else:
            order = get_order(conn, client_order_id)
            if order is None or order.decision_id is None:
                raise _Failure(f"no order placed under the name {client_order_id}")
            orders = [order]
            trace = get_proposal_trace(conn, order.proposal_id)
            fills = [fill for fill in trace.fills if fill.order_id == order.id]
    finally:
        conn.close()
    if client_order_id is None:
        _lines("orders" if show_all else "open orders", [_order_line(o) for o in orders])
        return 0
    print(_order_line(orders[0]))
    _lines("fills", [_row(_ts(f.filled_at), f"{_qty(f.fill_quantity)} at {_price(f.fill_price)}")
                     for f in fills])
    return 0


def _answer(
    db_path: Path, config: AegisConfig, decision_id: str, approved: bool, by: str,
    note: str | None, clock: Callable[[], datetime],
) -> int:
    conn = _writable(db_path, NO_DECISION)
    try:
        answer = dispatch.answer(
            conn, decision_id, approved=approved, by=by, note=note, config=config, clock=clock
        )
    finally:
        conn.close()
    until = f", good until {_ts(answer.expires_at)}" if answer.expires_at is not None else ""
    print(_clean(f"{decision_id}: {answer.response.value} by {answer.responder}{until}"))
    if approved:
        print(f"next: python -m aegis.cli.orders place {_clean(decision_id)}")
    return 0


def _place(
    db_path: Path, config: AegisConfig, decision_id: str, dry_run: bool, **wiring: Any
) -> int:
    conn = _open_unchanged(db_path, NO_DECISION) if dry_run else _writable(db_path, NO_DECISION)
    try:
        placement = dispatch.place(conn, decision_id, config=config, dry_run=dry_run, **wiring)
    finally:
        conn.close()
    for line in placement.lines:
        print(_clean(line))
    if dry_run:
        print(DRY_RUN_NOTE)
    return 0


def _broker_command(db_path: Path, run: Callable[[sqlite3.Connection], dispatch.Report]) -> int:
    conn = _writable(db_path, NO_ORDER)
    try:
        report = run(conn)
    finally:
        conn.close()
    return _report(report)


# --- the command line -------------------------------------------------------------


def _parser() -> tuple[_Parser, dict[str, _Parser]]:
    parser = _Parser(
        prog="python -m aegis.cli.orders",
        description="Place and follow AEGIS's paper orders: answer approvals, place what the"
        " policy engine approved, follow it at the broker, and cancel.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    made: dict[str, _Parser] = {}

    def add(name: str, help_text: str) -> _Parser:
        made[name] = commands.add_parser(name, parents=[common], help=help_text,
                                         description=help_text)
        return made[name]

    add("pending", "list the verdicts awaiting approval and the decisions ready to place")
    status = add("status", "list the open orders, or one order with its fills")
    status.add_argument("client_order_id", nargs="?", metavar="CLIENT_ORDER_ID")
    status.add_argument("--all", action="store_true", help="every order placed, not only open ones")
    for name, verb in (("approve", "approve"), ("reject", "reject")):
        answer = add(name, f"{verb} one NEEDS_APPROVAL verdict (once; it places nothing)")
        answer.add_argument("decision_id", metavar="DECISION_ID")
        answer.add_argument("--by", required=True, metavar="NAME", help="who answers")
        answer.add_argument("--note", metavar="TEXT", help="why, in a few words")
    place = add("place", "re-check one decision and place its order")
    place.add_argument("decision_id", metavar="DECISION_ID")
    place.add_argument("--dry-run", action="store_true",
                       help="run every check and the re-check; write and send nothing")
    add("sync", "bring every open order up to date with the broker")
    cancel = add("cancel", "cancel one open order")
    cancel.add_argument("client_order_id", metavar="CLIENT_ORDER_ID")
    add("stand-down", "under the kill switch or a halt, cancel every working order")
    add("tick", "sync, stand down if the controls say stop, then place what may go")
    return parser, made


def main(
    argv: list[str] | None = None,
    *,
    broker_factory: dispatch.BrokerFactory | None = None,
    context_builder: dispatch.ContextBuilder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser, made = _parser()
    try:
        args = parser.parse_args(argv)
        for name in ("decision_id", "client_order_id", "by"):
            value = getattr(args, name, None)
            if value is not None and not value.strip():
                made[args.command].error(f"{name.upper()} must not be blank")
    except _UsageError as exc:
        print(f"{_clean(exc)} (see --help)", file=sys.stderr)
        return 1

    now_of = clock if clock is not None else utcnow
    wiring = {"broker_factory": broker_factory, "clock": now_of}
    try:
        db_path = _db_path(args.db)
        config = get_config()
        command = args.command
        if command == "pending":
            return _pending(db_path, config, now_of())
        if command == "status":
            return _status(db_path, args.client_order_id, args.all)
        if command in ("approve", "reject"):
            return _answer(db_path, config, args.decision_id, command == "approve", args.by,
                           args.note, now_of)
        if command == "place":
            return _place(db_path, config, args.decision_id, args.dry_run,
                          context_builder=context_builder, **wiring)
        if command == "sync":
            return _broker_command(db_path, lambda c: dispatch.sync(c, config=config, **wiring))
        if command == "cancel":
            return _broker_command(
                db_path,
                lambda c: dispatch.cancel(c, args.client_order_id, config=config, **wiring),
            )
        if command == "stand-down":
            return _broker_command(
                db_path, lambda c: dispatch.stand_down(c, config=config, **wiring)
            )
        return _broker_command(
            db_path,
            lambda c: dispatch.tick(c, config=config, context_builder=context_builder, **wiring),
        )
    except (_Failure, ConfigError, StoreError, PolicyError) as exc:
        print(f"orders {args.command} failed: {_clean(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(
            f"orders {args.command} failed (unexpected): {type(exc).__name__}: {_clean(exc)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
