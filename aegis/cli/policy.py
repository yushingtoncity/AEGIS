"""Run the policy engine by hand: judge a proposal, set the kill switch, read the limits.

Run:  python -m aegis.cli.policy evaluate (PROPOSAL_ID | --latest) [--dry-run] [--db PATH]
      python -m aegis.cli.policy kill {on,off,status} [--db PATH]
      python -m aegis.cli.policy limits [--db PATH]

``evaluate`` judges one stored proposal — by id, or with ``--latest`` the
newest proposal that has no decision yet — against a context built now, and
prints the proposal, the verdict (with the failing rule for a REJECT), every
rule with its outcome and detail in registry order, and the context the
verdict rested on. It records the decision through the engine; when the
daily loss limit tripped, the engine also persists the trading halt and the
report says so. ``--dry-run`` prints the same report and writes nothing: no
decision, no event, no halt. Exit 0 whenever a verdict was produced,
whatever the verdict. The context printed is the one that was judged: the
store's kill switch and halt are read once more after the context is built,
and the stricter of the two readings is what the report, the rule lines and
the recorded decision all show — a switch set while the quotes were being
fetched is ON in all three. The engine reads those flags itself as it
records, an instant later; should that reading have been stricter still, the
report is rebuilt from the store's flags as they stand afterwards, and it is
printed only when it reproduces the recorded rule lines exactly — when no
reading in hand does (the flags changed more than once in that instant, or
the halt it was judged under has been extended since), a ``note:`` line
says that the rule lines are what was judged and the context block is this
command's own earlier reading. A halt is announced as it stands in the
store after the write: ``trading HALTED until …`` when this evaluation's
halt is the one on record, ``trading already HALTED until …`` when a longer
one got there first.

``kill on`` / ``kill off`` set the operator's kill switch and log a
``kill_switch`` event with the previous state — every time, also when the
state does not change. ``kill status`` prints the switch, the trading halt
and anything the store could not read about either.

``limits`` prints each limit beside its current consumption and headroom:
the distance to the daily loss cap, the positions and trades used, the
buying power, the exposure per underlying, the halt and kill-switch state
and which account-level rules block trading right now. Every number comes
from ``aegis.policy.limits.limits_report`` — this module only formats it, so
the Phase 8 dashboard reading the same function shows the same figures.

What may touch the database: ``kill on|off`` opens the store the way the
loop does (creating and migrating it if need be); ``evaluate`` migrates a
database that exists but never creates one — a file that is empty, or that
no migration was ever applied to, is not a store, and is refused rather
than built into one; ``evaluate --dry-run``, ``kill status`` and ``limits``
neither create nor migrate — the connection is put in SQLite's query-only
mode, a missing or empty database or a pending migration is a clean error,
and the fix is ``python -m aegis.cli.db init``. Those three write no row.
They do open the file the way every reader of the store does
(``aegis.store.db.connect``), and that is not nothing at the file level: a
database that is not in WAL journal mode — a copy made in DELETE mode, a
restored backup — is switched to WAL, and a WAL left behind by a writer
that was killed is checkpointed into the main file when the connection
closes. Neither changes a row of any table.
``--db`` overrides ``store.db_path`` from config.yaml and is taken relative
to the current directory; ``--db :memory:`` is sqlite's in-memory sentinel,
never a file, and every command here refuses it — a proposal, a limit or a
kill switch in a database that vanishes on exit is of no use to anyone.

Failures are one line on stderr and exit 1, never a traceback; a bad
command line is a failure too (one line, exit 1, not argparse's exit 2);
Ctrl-C exits 130. Everything read from the store or written by a rule is
printed on one line with control characters removed, so a stored symbol can
neither restyle the terminal nor forge a line of the report; a character
stdout cannot encode prints as an escape.

``main(argv, *, context_builder=None)``: tests inject a builder that returns
a hand-made ``PolicyContext``; the default is
``aegis.policy.context.build_context``, which reads the live account, clock
and quotes.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import unicodedata
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

from aegis.config import AegisConfig, ConfigError, get_config
from aegis.data.models import utcnow
from aegis.policy.context import build_context, latest_undecided, load_proposal
from aegis.policy.engine import decide, evaluate, stricter_controls
from aegis.policy.errors import PolicyError
from aegis.policy.limits import limits_report
from aegis.policy.measures import money
from aegis.policy.models import (
    Evaluation,
    LimitsReport,
    PolicyContext,
    ProposalUnderReview,
    RuleOutcome,
)
from aegis.store.db import MEMORY_DB, connect, open_store, resolve_db_path, status
from aegis.store.errors import StoreError
from aegis.store.models import Controls, Event, EventLevel, PolicyDecision
from aegis.store.repo import get_controls, set_kill_switch

ContextBuilder = Callable[..., PolicyContext]
"""``build_context``'s shape: ``(conn, proposal=None, *, config=...) -> PolicyContext``."""

KILL_SWITCH_EVENT = "kill_switch"
"""The kind of the event ``kill on`` / ``kill off`` log with the change."""

INDENT = "  "
GAP = "   "
BREACHED = "BREACHED"
DRY_RUN_NOTE = "DRY RUN — nothing was written"
UNPROVEN_HALT = "UNKNOWN (cannot prove trading is not halted)"
STALE_CONTEXT_NOTE = (
    "note: the kill switch or the halt changed while this decision was being recorded:"
    " the rule lines are what was judged; the context block is this command's earlier reading"
)
INIT_HINT = "python -m aegis.cli.db init"

_USD = "USD"  # the unit of a ``LimitLine`` that holds dollars; the others hold counts
MARKET_HOURS_RULE = "market_hours"  # as ``LimitsReport.blocked_by`` names it


class _Failure(Exception):
    """A command that cannot do what was asked — no database, a pending
    migration, nothing to evaluate. ``main`` prints it as one line, exit 1."""


def _db_path(arg: str | None) -> Path:
    """An explicit --db resolves against the CWD; otherwise config.yaml decides."""
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)  # resolving it would name a file ":memory:" in the CWD
    return Path(arg).expanduser().resolve()


# --- formatting -----------------------------------------------------------------


def _clean(text: object) -> str:
    """Text for the terminal, on one line: control and format characters
    become spaces (an ESC in a stored symbol could restyle or hide the rest
    of the report), and whitespace runs — newlines included — collapse to
    one space, so no stored field can print a line of its own."""
    kept = "".join(
        " " if unicodedata.category(ch) in ("Cc", "Cf") else ch for ch in str(text)
    )
    return " ".join(kept.split())


def _utc(ts: datetime) -> datetime:
    """A naive instant is taken as UTC, so two instants always compare."""
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _ts(ts: datetime | None, missing: str = "unknown") -> str:
    """An instant in UTC to the second (a naive one is taken as UTC)."""
    if ts is None:
        return missing
    ts = _utc(ts)
    try:
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except OverflowError:  # the far end of the calendar, in another zone
        return _clean(ts.isoformat())


def _qty(value: float) -> str:
    """Every digit as stored, in plain notation: ``4``, ``0.5``, ``1000000``."""
    text = format(Decimal(repr(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _price(value: float) -> str:
    """A price: every digit, at least two decimals (``2.40``, ``561.275``)."""
    whole, _, fraction = _qty(value).partition(".")
    return f"{whole}.{fraction.ljust(2, '0')}"


def _strike(value: float) -> str:
    """A strike: every digit, at least one decimal (``640.0``, ``642.5``)."""
    whole, _, fraction = _qty(value).partition(".")
    return f"{whole}.{fraction.ljust(1, '0')}"


def _amount(value: float | None, unit: str) -> str:
    """One cell of the limits table: dollars as money, a count as an integer,
    ``-`` for a figure the line does not have."""
    if value is None or math.isnan(value):
        return "-"
    if unit == _USD:
        return money(value)
    if math.isinf(value):
        return "unlimited" if value > 0 else "-unlimited"
    return f"{value:,.0f}"


def _fields(rows: Sequence[tuple[str, str]], depth: int = 1) -> None:
    """``label   value`` lines, the values aligned past the longest label."""
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{INDENT * depth}{label.ljust(width)}{GAP}{value}")


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    """The rows as text, each column padded to its widest cell."""
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    return [
        GAP.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows
    ]


def _problems(title: str, lines: Sequence[str], depth: int = 0) -> None:
    """``title (N)`` then one cleaned line per problem (``(none)`` for none)."""
    pad = INDENT * depth
    print(f"{pad}{title} ({len(lines)})")
    if not lines:
        print(f"{pad}{INDENT}(none)")
    for line in lines:
        print(f"{pad}{INDENT}- {_clean(line)}")


def _switch(on: bool) -> str:
    return "on" if on else "off"


def _switch_text(on: bool) -> str:
    return "ON — every proposal is rejected" if on else "off"


def _halt_text(until: datetime | None, in_force: bool, now: datetime) -> str:
    """The trading halt in words: none / until <ts> (active) / expired at <ts> /
    UNKNOWN — in force with no instant still ahead of ``now``, which is a halt
    in force only because the stored value cannot be read. That is so with no
    instant on record, and also with one that has passed (a context may hold
    both an unreadable halt and the instant an earlier reading gave): an
    expired instant is never what keeps trading halted, and is not printed
    as "active"."""
    if not in_force:
        return "none" if until is None else f"expired at {_ts(until)}"
    if until is None:
        return UNPROVEN_HALT
    if _utc(until) > now:
        return f"until {_ts(until)} (active)"
    return f"{UNPROVEN_HALT}; the last halt that could be read expired at {_ts(until)}"


def _market_text(report: LimitsReport) -> str:
    if report.market_open is None:
        return "UNKNOWN (no market clock)"
    if not report.market_open:
        return f"CLOSED (next open {_ts(report.next_open)})"
    if MARKET_HOURS_RULE in report.blocked_by:
        # the clock says open and the rule still rejects: its close has passed
        return f"OPEN per a STALE clock (its close {_ts(report.next_close)} has passed)"
    return f"OPEN (next close {_ts(report.next_close)})"


def _state_rows(report: LimitsReport) -> list[tuple[str, str]]:
    return [
        ("market", _market_text(report)),
        ("kill switch", _switch_text(report.kill_switch)),
        ("halt", _halt_text(report.halt_until, report.halted, report.as_of)),
    ]


def _account_rows(report: LimitsReport) -> list[tuple[str, str]]:
    return [
        ("equity", money(report.equity)),
        ("start-of-day equity", money(report.start_of_day_equity)),
        ("today's P&L", money(report.daily_pnl)),
        ("buying power", money(report.buying_power)),
        ("options buying power", money(report.options_buying_power)),
    ]


# --- opening the database -------------------------------------------------------


def _require_file(db_path: Path, why_not_memory: str) -> None:
    """Refuse the in-memory sentinel, a database file that does not exist and
    one that is empty — before anything is opened: connecting to a zero-byte
    file would write a database header into it."""
    if str(db_path) == MEMORY_DB:
        raise _Failure(
            f"--db {MEMORY_DB} is an in-memory database, not a file: {why_not_memory}"
        )
    if not db_path.exists():
        raise _Failure(f"database {db_path} does not exist (run `{INIT_HINT}` to create it)")
    try:
        empty = db_path.is_file() and db_path.stat().st_size == 0
    except OSError as exc:
        raise StoreError("open", str(db_path), exc) from exc
    if empty:
        raise _Failure(
            f"database {db_path} is an empty file, not a store (run `{INIT_HINT}` to create it)"
        )


def _require_schema(db_path: Path) -> None:
    """Refuse a database no migration was ever applied to. ``evaluate`` may
    bring an existing store up to date; it must not build a schema into a
    file that holds none — there is no proposal in it to judge."""
    conn = connect(db_path)
    try:
        applied = status(conn, db_path).applied_migrations
    finally:
        conn.close()
    if not applied:
        raise _Failure(
            f"database {db_path} has no schema: no migration was ever applied"
            f" (run `{INIT_HINT}` to create it)"
        )


def _open_unchanged(db_path: Path, why_not_memory: str) -> sqlite3.Connection:
    """A connection for a command that must write no row: the database must
    exist, not be empty, and be fully migrated (``connect``, never
    ``open_store`` — a report applies no migration); the connection is then
    put in query-only mode, so SQLite refuses every statement that would
    write. (``connect`` itself keeps the store's journal mode — see the
    module docstring for what that means for a file that is not in WAL.)"""
    _require_file(db_path, why_not_memory)
    conn = connect(db_path)
    try:
        pending = status(conn, db_path).pending_migrations
        if pending:
            raise _Failure(
                f"database {db_path} has pending migrations ({', '.join(pending)}):"
                f" run `{INIT_HINT}` to apply them"
            )
        try:
            conn.execute("PRAGMA query_only=ON")
        except sqlite3.Error as exc:
            raise StoreError("open read-only", str(db_path), exc) from exc
    except BaseException:
        conn.close()
        raise
    return conn


# --- evaluate -------------------------------------------------------------------


def _print_proposal(proposal: ProposalUnderReview) -> None:
    order = proposal.proposal
    limit = f" @ {_price(order.limit_price)}" if order.limit_price is not None else ""
    terms = f"{order.side.value} {_qty(order.quantity)} {order.order_type.value}{limit}"
    print(f"proposal {_clean(order.id)}")
    _fields(
        [
            ("created", _ts(order.created_at)),
            ("symbol", f"{_clean(order.symbol)} ({order.instrument.value})"),
            ("order", terms),
            ("confidence", _qty(order.confidence)),  # as stored: an audit view never rounds
        ]
    )
    print(f"{INDENT}legs ({len(proposal.legs)})")
    if not proposal.legs:
        print(f"{INDENT * 2}(none)")
    for leg in proposal.legs:
        print(
            f"{INDENT * 2}[{leg.leg_index}] {leg.side.value} {_qty(leg.quantity)}"
            f" {leg.option_type.value} {_strike(leg.strike)} exp {leg.expiration.isoformat()}"
            f"  {_clean(leg.symbol)}"
        )


def _print_rules(rules: Sequence[dict[str, Any]]) -> None:
    """``rules (N)`` then one line per rule, as recorded: index, name, outcome, detail."""
    print(f"rules ({len(rules)})")
    names = [_clean(rule.get("rule", "(unnamed)")) for rule in rules]
    name_width = max((len(name) for name in names), default=0)
    outcome_width = max(len(outcome.value) for outcome in RuleOutcome)
    index_width = len(str(len(rules)))
    for index, (name, rule) in enumerate(zip(names, rules), start=1):
        outcome = _clean(rule.get("outcome", "?"))
        line = (
            f"{INDENT}[{str(index).rjust(index_width)}] {name.ljust(name_width)}{GAP}"
            f"{outcome.ljust(outcome_width)}{GAP}{_clean(rule.get('detail', ''))}"
        )
        print(line.rstrip())


def _halt_line(
    named: datetime | None, stored: Controls | None, now: datetime, dry_run: bool
) -> str | None:
    """What the report says of the halt this evaluation named (None: it named
    none). A dry run wrote nothing. A recording run says what the store holds
    AFTER the write (``stored``, read back): the engine's write is a
    compare-and-set, so a longer halt that reached the store first stays, and
    is not announced as this evaluation's. ``stored`` None — the store could
    not be read back — leaves the halt as the engine was asked to write it."""
    if named is None:
        return None
    if dry_run:
        return f"trading would be HALTED until {_ts(named)} (dry run: no halt was written)"
    if stored is None:
        return f"trading HALTED until {_ts(named)}"
    on_record = None if stored.halt_unknown else stored.halt_until
    if on_record is not None and _utc(on_record) == named:
        return f"trading HALTED until {_ts(named)}"
    if on_record is not None and _utc(on_record) > named:
        return (
            f"trading already HALTED until {_ts(on_record)}: a longer halt than this"
            f" evaluation's ({_ts(named)}) was on record first"
        )
    in_force = stored.halt_unknown or (on_record is not None and _utc(on_record) > now)
    return (
        f"trading HALTED until {_ts(named)} — but the store now reads:"
        f" {_halt_text(stored.halt_until, in_force, now)}"
    )


def _print_evaluation(
    proposal: ProposalUnderReview,
    decision: PolicyDecision,
    halt_line: str | None,
    report: LimitsReport,
    context: PolicyContext,
    dry_run: bool,
    note: str | None = None,
) -> None:
    _print_proposal(proposal)
    print()
    print(f"verdict: {decision.verdict.value}")
    if decision.failing_rule is not None:
        print(f"failing rule: {_clean(decision.failing_rule)}")
    print(DRY_RUN_NOTE if dry_run else f"decision {_clean(decision.id)} recorded")
    if halt_line is not None:
        print(halt_line)
    if note is not None:
        print(_clean(note))
    print()
    _print_rules(decision.rules_evaluated)
    print()
    positions = "unknown" if context.positions is None else str(len(context.positions))
    print("context")
    _fields(
        [
            ("as of", _ts(report.as_of)),
            *_state_rows(report),
            *_account_rows(report),
            ("positions", positions),
            ("open orders", str(len(context.open_orders))),
            ("orders today", str(context.orders_today)),
        ]
    )
    _problems("data problems", report.errors, depth=1)


def _read_back(conn: sqlite3.Connection) -> tuple[Controls | None, str | None]:
    """The store's stop flags after the engine has written, and a note for
    the report when they cannot be read: the decision IS recorded by then, so
    a failed read-back must not turn the command into a failure that hides it."""
    try:
        return get_controls(conn), None
    except StoreError as exc:
        return None, (
            "note: the store could not be read back after the decision was recorded"
            f" ({exc}): the context block is this command's earlier reading"
        )


def _as_judged(
    proposal: ProposalUnderReview,
    context: PolicyContext,
    decision: PolicyDecision,
    stored: Controls,
) -> tuple[PolicyContext, Evaluation] | None:
    """The context a recorded decision was judged under, and ``decide`` on it
    — or None when no reading in hand reproduces the decision's rule lines.

    ``evaluate`` reads the store's stop flags itself, after this command did,
    and judges under the stricter reading; a switch set (or a halt written)
    in between is in the decision and not in ``context``. ``stored`` is the
    store read back after the engine's writes. Two readings are tried, each
    merged into ``context`` the way the engine merges (``stricter_controls``):
    the flags as they stand, and the flags without their halt — the halt
    this very evaluation may just have written was not yet in force when it
    was judged. The one whose ``decide`` gives exactly the recorded rules is
    the context that was judged."""
    readings = [stored]
    if stored.halt_until is not None:
        readings.append(stored.model_copy(update={"halt_until": None}))
    for reading in readings:
        controls = stricter_controls(context.controls, reading)
        under = context
        if controls is not context.controls:
            under = context.model_copy(update={"controls": controls})
        evaluation = decide(proposal, under)
        if evaluation.rules_evaluated == decision.rules_evaluated:
            return under, evaluation
    return None


def _evaluate(
    db_path: Path,
    config: AegisConfig,
    builder: ContextBuilder,
    proposal_id: str | None,
    dry_run: bool,
) -> int:
    nothing = "there is no proposal in it to evaluate"
    if dry_run:
        conn = _open_unchanged(db_path, nothing)
    else:
        _require_file(db_path, nothing)  # never create a database to evaluate
        _require_schema(db_path)  # ... nor build one into a file that holds none
        conn = open_store(db_path)
    try:
        if proposal_id is None:
            proposal = latest_undecided(conn)
            if proposal is None:
                raise _Failure(f"no proposal without a decision in {db_path}")
        else:
            proposal = load_proposal(conn, proposal_id)
        context = builder(conn, proposal, config=config)
        # The context printed is the context judged. Building one takes
        # seconds, and a kill switch or a halt set meanwhile is honoured by
        # the engine — so the stop flags are read once more here (a read: the
        # query-only connection of a dry run allows it) and the stricter
        # reading goes into the report, the rule lines and the decision alike.
        controls = stricter_controls(context.controls, get_controls(conn))
        if controls is not context.controls:
            context = context.model_copy(update={"controls": controls})
        # Everything the report shows is worked out before anything is
        # written: a failure here leaves no decision behind. ``decide`` is the
        # engine's pure half — the same verdict ``evaluate`` records — and is
        # asked for the halt, which the stored decision does not carry, and
        # for the rule lines this context gives, to hold against the ones
        # the engine goes on to record.
        report = limits_report(context)
        judged = decide(proposal, context)
        decision = evaluate(proposal, context, conn, dry_run=dry_run)
        # The decision is recorded. What follows only reads: the store's stop
        # flags once more, when the engine's own reading of them was
        # stricter than this command's (the rule lines differ) or a halt was
        # named (the store says whether it is the one on record).
        stored, note = None, None
        differs = judged.rules_evaluated != decision.rules_evaluated
        if not dry_run and (differs or judged.halt_until is not None):
            stored, note = _read_back(conn)
        if differs and stored is not None:
            # The engine read the stop flags after this command did and
            # found a stricter reading still: the report is rebuilt from
            # the context that reproduces what was recorded. (Pure, but
            # for the read above: nothing more is written.)
            found = _as_judged(proposal, context, decision, stored)
            if found is None:
                note = STALE_CONTEXT_NOTE
            else:
                context, judged = found
                report = limits_report(context)
        halt_line = _halt_line(judged.halt_until, stored, context.now, dry_run)
    finally:
        conn.close()
    _print_evaluation(proposal, decision, halt_line, report, context, dry_run, note)
    return 0


# --- kill -----------------------------------------------------------------------


def _print_controls(controls: Controls, now: datetime, was: str | None = None) -> None:
    """The kill switch, the halt and what the store could not read of either.
    ``was`` is the state before a change (``kill on`` / ``kill off``)."""
    change = f" (was {was})" if was is not None else ""
    print(f"kill switch: {_switch_text(controls.kill_switch)}{change}")
    print(f"{INDENT}last set {_ts(controls.kill_switch_updated_at)}")
    until = controls.halt_until
    in_force = controls.halt_unknown or (until is not None and until > now)
    print(f"halt: {_halt_text(until, in_force, now)}")
    print(f"{INDENT}last set {_ts(controls.halt_until_updated_at)}")
    _problems("problems", controls.problems)


def _kill_set(db_path: Path, on: bool) -> int:
    if str(db_path) == MEMORY_DB:
        raise _Failure(
            f"--db {MEMORY_DB} is an in-memory database, not a file:"
            " a kill switch set there is gone when this command exits"
        )
    conn = open_store(db_path)
    try:
        was = _switch(get_controls(conn).kill_switch)
        now = utcnow()
        state = _switch(on)
        event = Event(
            occurred_at=now,
            level=EventLevel.WARNING if on else EventLevel.INFO,
            kind=KILL_SWITCH_EVENT,
            message=f"kill switch set {state.upper()} from the CLI (was {was})",
            payload={"state": state, "previous": was, "source": "cli"},
        )
        controls = set_kill_switch(conn, on, now=now, event=event)
    finally:
        conn.close()
    # the state printed is the one read back from the store, not the one asked for
    _print_controls(controls, now, was=was)
    if controls.kill_switch is not on:
        raise _Failure(
            f"the kill switch was set {state} but reads {_switch(controls.kill_switch).upper()}:"
            f" {'; '.join(controls.problems) or 'the controls table is not as migrated'}"
        )
    return 0


def _kill_status(db_path: Path) -> int:
    conn = _open_unchanged(db_path, "there are no controls in it to report")
    try:
        controls = get_controls(conn)
    finally:
        conn.close()
    _print_controls(controls, utcnow())
    return 0


# --- limits ---------------------------------------------------------------------


def _print_limits(report: LimitsReport) -> None:
    print(f"limits as of {_ts(report.as_of)}")
    _fields(_state_rows(report))
    if report.blocked_by:
        print(f"trading blocked by: {', '.join(_clean(name) for name in report.blocked_by)}")
    else:
        print("no account-level block")
    print()
    print("account")
    _fields(_account_rows(report))
    print()
    print(f"limits ({len(report.lines)})")
    rows = [("name", "limit", "used", "headroom", "unit", "")]
    rows += [
        (
            _clean(line.name),
            _amount(line.limit, line.unit),
            _amount(line.used, line.unit),
            _amount(line.headroom, line.unit),
            _clean(line.unit),
            BREACHED if line.breached else "",
        )
        for line in report.lines
    ]
    header, *body = _table(rows)
    print(f"{INDENT}{header}")
    for text, line in zip(body, report.lines):
        print(f"{INDENT}{text}")
        print(f"{INDENT * 3}{_clean(line.detail)}")
    print()
    print(f"exposure by underlying ({len(report.exposures)})")
    if not report.exposures:
        print(f"{INDENT}(none)")
    else:
        exposures = [("underlying", "exposure", "cap", "headroom", "")]
        exposures += [
            (
                _clean(line.underlying),
                _amount(line.exposure, _USD),
                _amount(line.cap, _USD),
                _amount(line.headroom, _USD),
                BREACHED if line.breached else "",
            )
            for line in report.exposures
        ]
        for text in _table(exposures):
            print(f"{INDENT}{text}")
    print()
    print("auto-execute tier")
    _fields(
        [
            ("enabled", "yes" if report.auto_execute_enabled else "no"),
            ("max notional", money(report.auto_execute_max_notional)),
        ]
    )
    print()
    _problems("data problems", report.errors)


def _limits(db_path: Path, config: AegisConfig, builder: ContextBuilder) -> int:
    conn = _open_unchanged(db_path, "there is nothing in it to measure against the limits")
    try:
        context = builder(conn, config=config)
    finally:
        conn.close()
    _print_limits(limits_report(context))
    return 0


# --- main -----------------------------------------------------------------------


class _UsageError(Exception):
    """A bad command line, raised in place of argparse's exit so ``main`` can
    report it as one line and exit 1, like every other failure."""


class _Parser(argparse.ArgumentParser):
    """``ArgumentParser`` whose errors raise ``_UsageError`` instead of
    printing the usage block and exiting 2 (the subcommand parsers are made
    from the same class, so theirs do too). ``--help`` still exits 0."""

    def error(self, message: str) -> NoReturn:
        raise _UsageError(f"{self.prog}: error: {message}")


def main(argv: list[str] | None = None, *, context_builder: ContextBuilder | None = None) -> int:
    # a report must not abort on the content it exists to show (a stored
    # symbol, the em dash in the dry-run note): a character the stdout
    # encoding lacks prints as a \uXXXX escape, as in the trace CLI
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = _Parser(
        prog="python -m aegis.cli.policy",
        description="Run the AEGIS policy engine: judge a proposal, set the kill switch,"
        " or print the limits.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    judge = commands.add_parser(
        "evaluate",
        parents=[common],
        help="judge one stored proposal and record the verdict",
        description="Judge one stored proposal against a context built now, print every"
        " rule's finding and record the decision. Exactly one of PROPOSAL_ID and --latest"
        " is required.",
    )
    judge.add_argument(
        "proposal_id",
        nargs="?",
        metavar="PROPOSAL_ID",
        help="the proposal's id (required unless --latest is given)",
    )
    judge.add_argument(
        "--latest",
        action="store_true",
        help="judge the newest proposal without a decision (instead of a PROPOSAL_ID)",
    )
    judge.add_argument(
        "--dry-run", action="store_true", help="print the verdict and write nothing"
    )
    kill = commands.add_parser(
        "kill",
        parents=[common],
        help="set or show the kill switch (on rejects every proposal)",
        description="Set or show the operator's kill switch. on and off log a kill_switch"
        " event with the previous state, every time; status writes nothing.",
    )
    kill.add_argument(
        "state",
        choices=["on", "off", "status"],
        help="on: reject every proposal; off: lift the switch; status: print the switch,"
        " the trading halt and anything the store could not read of either",
    )
    commands.add_parser(
        "limits",
        parents=[common],
        help="print each limit beside its consumption and headroom",
        description="Print each risk limit beside its current consumption and headroom,"
        " the halt and kill-switch state, and what blocks trading right now. Writes nothing.",
    )
    try:
        args = parser.parse_args(argv)
        if args.command == "evaluate":
            if args.proposal_id is not None and args.latest:
                judge.error("give a PROPOSAL_ID or --latest, not both")
            if args.proposal_id is None and not args.latest:
                judge.error("give a PROPOSAL_ID, or --latest for the newest undecided proposal")
            if args.proposal_id is not None and not args.proposal_id.strip():
                judge.error("PROPOSAL_ID must not be blank")
    except _UsageError as exc:
        print(f"{_clean(exc)} (see --help)", file=sys.stderr)
        return 1

    try:
        db_path = _db_path(args.db)
        if args.command == "kill":  # never waits on config.yaml when --db names the file
            if args.state == "status":
                return _kill_status(db_path)
            return _kill_set(db_path, args.state == "on")
        config = get_config()
        builder = context_builder if context_builder is not None else build_context
        if args.command == "evaluate":
            return _evaluate(db_path, config, builder, args.proposal_id, args.dry_run)
        return _limits(db_path, config, builder)
    except (_Failure, ConfigError, StoreError, PolicyError) as exc:
        print(f"policy {args.command} failed: {_clean(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(
            f"policy {args.command} failed (unexpected): {type(exc).__name__}: {_clean(exc)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
