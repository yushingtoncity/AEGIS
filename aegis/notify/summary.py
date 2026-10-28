"""The end-of-day summary: one text after the close saying how the day went.

``notify summary`` builds it and writes it as one ``daily_summary`` event;
``notify send`` then delivers it like any other notice, so it is retried
on failure and recorded by ``notify_sent``. At most one summary is written
per New York day, so the after-close ticks can all ask for it.

What it reads, all of it read-only:

- the store's events for the day: ``cycle_start`` / ``cycle_end`` (each
  brain cycle and its outcome and cost), ``proposal``, ``approval_answered``,
  ``order_submitted`` / ``order_updated``, ``forecast_end``, and every
  warning or worse;
- the day's policy verdicts, forecast runs, token usage and the controls;
- the account state the caller read (equity, today's change, positions),
  or the reason it could not be read;
- the launchd logs' lines for the day (``ops/launchd/aegis-run.sh`` writes
  ``<date> <time> <zone> [<mode>] exit <code>: <command>``), because a
  step's exit code is recorded nowhere else.

It is a report only: nothing here changes what may be traded.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from aegis.config import AegisConfig
from aegis.data.models import AccountState
from aegis.notify.models import BODY_MAX
from aegis.store.forecasts import get_run_forecasts, get_runs_for_session
from aegis.store.models import Event, EventLevel, Verdict
from aegis.store.repo import (
    get_controls,
    get_events_since,
    get_token_usage,
    get_verdicts_since,
)

SUMMARY_KIND = "daily_summary"
NEW_YORK = ZoneInfo("America/New_York")
"""The trading day is a New York day, as the launchd schedule is."""

LOG_FILES = ("cycle.log", "tick.log")
STALE_QUOTE_RULE = "quote_freshness"
MAX_PROBLEM_LINES = 6
"""Problems listed one per line before the rest are counted; a phone screen
holds about this many under the other lines."""
MAX_POSITIONS = 4
"""Positions named on the account line before the rest are counted."""

_LOG_LINE = re.compile(
    r"^(?P<day>\d{4}-\d{2}-\d{2}) (?P<hour>\d{2}):(?P<minute>\d{2}):\d{2} \S+ "
    r"\[(?P<mode>\w+)\] (?P<what>.*)$"
)
_EXIT = re.compile(r"^exit (?P<code>\d+): (?P<command>.+)$")
_OUTCOME_WORDS = {
    "no_trade": "no trade",
    "halted": "halted by the token budget",
    "failed": "FAILED",
}
_SKIP_PROBLEMS = {
    "skipped: busy": "a step was skipped because another was still running",
    "skipped: market clock unreadable": "the market clock could not be read",
}
# kinds the other lines already report, or that are not problems on their own
_NOT_PROBLEMS = frozenset({"policy_reject", "cycle_end", "forecast_end", SUMMARY_KIND})


@dataclass(frozen=True)
class DaySummary:
    """The text to send and how many problems it lists."""

    day: date
    text: str
    problems: int


def clock_time(at: datetime) -> str:
    """``10:00 am`` in New York time; naive taken as UTC."""
    aware = at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)
    local = aware.astimezone(NEW_YORK)
    return _hhmm(local.hour, local.minute)


def _hhmm(hour: int, minute: int) -> str:
    return f"{hour % 12 or 12}:{minute:02d} {'am' if hour < 12 else 'pm'}"


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """The New York day as ``[start, end)`` in UTC."""
    start = datetime.combine(day, time.min, tzinfo=NEW_YORK)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=NEW_YORK)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def new_york_day(now: datetime) -> date:
    aware = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return aware.astimezone(NEW_YORK).date()


def _money(value: float) -> str:
    return f"${abs(value):,.2f}"


def _change(value: float) -> str:
    if value > 0:
        return f"up {_money(value)}"
    if value < 0:
        return f"down {_money(value)}"
    return "flat"


def _qty(value: float) -> str:
    return f"{value:g}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


# --- the logs -----------------------------------------------------------------


@dataclass(frozen=True)
class LogLine:
    mode: str
    hour: int
    minute: int
    what: str


def read_log_lines(log_dir: Path, day: date) -> list[LogLine] | None:
    """The day's lines of the launchd logs, or None when there is no log
    folder (a machine without the schedule). An unreadable file is skipped."""
    if not log_dir.is_dir():
        return None
    found: list[LogLine] = []
    for name in LOG_FILES:
        try:
            text = (log_dir / name).read_text(errors="replace")
        except OSError:
            continue
        found.extend(parse_log_lines(text.splitlines(), day))
    return found


def parse_log_lines(lines: Sequence[str], day: date) -> list[LogLine]:
    """The runner's own lines for ``day`` (CLI output in between is left out)."""
    wanted = day.isoformat()
    found = []
    for line in lines:
        match = _LOG_LINE.match(line)
        if match is None or match["day"] != wanted:
            continue
        found.append(
            LogLine(match["mode"], int(match["hour"]), int(match["minute"]), match["what"].strip())
        )
    return found


def _log_problems(lines: Sequence[LogLine]) -> list[str]:
    failures: dict[str, list[LogLine]] = {}
    skips: Counter[str] = Counter()
    for line in lines:
        exit_match = _EXIT.match(line.what)
        if exit_match is not None and exit_match["code"] != "0":
            key = f"{exit_match['command'].strip()} exited {exit_match['code']}"
            failures.setdefault(key, []).append(line)
        else:
            skip = next((said for start, said in _SKIP_PROBLEMS.items() if line.what.startswith(start)), None)
            if skip is not None:
                skips[skip] += 1
    problems = []
    for key, hits in failures.items():
        first = _hhmm(hits[0].hour, hits[0].minute)
        problems.append(f"{key} at {first}" if len(hits) == 1 else f"{key} {len(hits)} times, first at {first}")
    for what, count in skips.items():
        problems.append(what if count == 1 else f"{what} ({count} times)")
    return problems


def _market_closed_all_day(lines: Sequence[LogLine] | None) -> bool:
    """True when every cycle the schedule fired found the market closed: a holiday."""
    if not lines:
        return False
    cycle_lines = [line.what for line in lines if line.mode == "cycle"]
    return bool(cycle_lines) and all(what == "skipped: market closed" for what in cycle_lines)


# --- the store ----------------------------------------------------------------


def _events(conn: sqlite3.Connection, start: datetime, end: datetime) -> list[Event]:
    return [
        event for event in get_events_since(conn, start)
        if _utc(event.occurred_at) < end
    ]


def _utc(at: datetime) -> datetime:
    return at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)


def already_summarized(conn: sqlite3.Connection, day: date) -> bool:
    start, _ = day_bounds(day)
    return any(
        (event.payload or {}).get("day") == day.isoformat()
        for event in get_events_since(conn, start, kind=SUMMARY_KIND)
    )


def _payload(event: Event, key: str) -> object:
    return (event.payload or {}).get(key)


def _cycles_line(events: Sequence[Event], expected: int) -> tuple[str, list[str]]:
    starts = [event for event in events if event.kind == "cycle_start"]
    ends = {_payload(e, "cycle_id"): e for e in events if e.kind == "cycle_end"}
    symbols = {_payload(e, "cycle_id"): _payload(e, "symbol") for e in events if e.kind == "proposal"}
    problems: list[str] = []
    if len(starts) < expected:
        problems.append(f"only {len(starts)} of {expected} brain cycles ran")
    parts: list[str] = []
    for start in starts:
        cycle_id = _payload(start, "cycle_id")
        end = ends.get(cycle_id)
        outcome = _payload(end, "outcome") if end is not None else None
        if outcome == "proposal":
            word = f"proposed {symbols.get(cycle_id) or 'a trade'}"
        elif isinstance(outcome, str):
            word = _OUTCOME_WORDS.get(outcome, outcome)
        else:
            word = "never finished"
            problems.append(f"the {clock_time(start.occurred_at)} cycle never finished")
        parts.append(f"{clock_time(start.occurred_at)} {word}")
    head = f"Cycles: {len(starts)} of {expected} ran"
    if not parts:
        return head, problems
    words = {part.split(" ", 2)[2] for part in parts}
    if len(words) == 1:
        return f"{head}, all {words.pop()}", problems
    return f"{head}. " + ", ".join(parts), problems


def _verdicts(conn: sqlite3.Connection, start: datetime, end: datetime) -> dict[Verdict, list]:
    return {
        verdict: [d for d in get_verdicts_since(conn, start, verdict) if _utc(d.decided_at) < end]
        for verdict in Verdict
    }


def _trades_line(
    conn: sqlite3.Connection, events: Sequence[Event], start: datetime, end: datetime
) -> tuple[str, list[str]]:
    verdicts = _verdicts(conn, start, end)
    proposals = sum(1 for e in events if e.kind == "proposal")
    approvals = sum(1 for e in events if e.kind == "approval_answered")
    submitted = [e for e in events if e.kind == "order_submitted"]
    updates = sum(1 for e in events if e.kind == "order_updated")
    problems: list[str] = []
    for decision in verdicts[Verdict.REJECT]:
        if decision.failing_rule == STALE_QUOTE_RULE:
            problems.append(f"stale quote reject at {clock_time(decision.decided_at)}")
    decided = sum(len(found) for found in verdicts.values())
    if not proposals and not decided and not submitted:
        return "Trades: none", problems
    counts = [
        f"{len(found)} {verdict.value.lower().replace('_', ' ')}"
        for verdict, found in verdicts.items() if found
    ]
    rules = Counter(d.failing_rule or "unknown" for d in verdicts[Verdict.REJECT])
    rejects = f" (rejects: {', '.join(sorted(rules))})" if rules else ""
    orders = ", ".join(
        str(_payload(e, "symbol") or "?") for e in submitted
    )
    line = (
        f"Trades: {_plural(proposals, 'proposal')}. Verdicts: {', '.join(counts) or 'none'}{rejects}. "
        f"{_plural(approvals, 'approval')}. Orders: {len(submitted)}"
        + (f" ({orders})" if orders else "")
        + (f", {_plural(updates, 'broker update')}" if updates else "")
    )
    return line, problems


def _account_line(account: AccountState | None, error: str | None) -> tuple[str, list[str]]:
    if account is None:
        reason = error or "not read"
        return f"Account: could not read ({reason})", [f"the account could not be read ({reason})"]
    parts = []
    if account.equity is not None:
        parts.append(f"equity {_money(account.equity)}")
        if account.last_equity is not None:
            parts.append(f"{_change(account.equity - account.last_equity)} today")
    line = "Account: " + (", ".join(parts) or "no equity figure")
    if not account.positions:
        return line + ". No positions", []
    shown = []
    for position in account.positions[:MAX_POSITIONS]:
        pnl = f" ({_change(position.unrealized_pl)})" if position.unrealized_pl is not None else ""
        shown.append(f"{position.symbol} {_qty(position.qty)}{pnl}")
    more = len(account.positions) - MAX_POSITIONS
    return line + ". " + ", ".join(shown) + (f", and {more} more" if more > 0 else ""), []


def _forecasts_line(
    conn: sqlite3.Connection, events: Sequence[Event], day: date, config: AegisConfig
) -> tuple[str, list[str]]:
    failed = [e for e in events if e.kind == "forecast_end" and _payload(e, "outcome") == "failed"]
    problems = [f"a forecast run failed at {clock_time(e.occurred_at)}" for e in failed]
    if not config.forecast.enabled:
        return "Forecasts: off", problems
    runs = get_runs_for_session(conn, day)
    if not runs:
        return "Forecasts: none ran", problems + ["the forecasts did not run"]
    count = sum(len(found) for found in get_run_forecasts(conn, [run.id for run in runs]).values())
    arms = ", ".join(run.arm.value for run in runs)
    return f"Forecasts: {_plural(len(runs), 'run')} ({arms}), {_plural(count, 'forecast')}", problems


def _cost_line(conn: sqlite3.Connection, events: Sequence[Event], day: date, config: AegisConfig) -> str:
    costs = [
        _payload(e, "estimated_cost_usd") for e in events if e.kind in ("cycle_end", "forecast_end")
    ]
    if any(not isinstance(cost, (int, float)) for cost in costs):
        spend = "est. cost n/a"
    else:
        spend = f"est. ${sum(costs):.2f}"
    # the budget guard counts a UTC day; the trading day sits inside one
    used = get_token_usage(conn, day).total
    budget = config.brain.daily_token_budget
    tokens = f"{used / 1000:,.0f}k" if used >= 1000 else str(used)
    return f"Claude: {spend}, {tokens} tokens ({used / budget * 100:.0f}% of the daily budget)"


def _event_problems(events: Sequence[Event]) -> list[str]:
    grouped: dict[str, list[Event]] = {}
    for event in events:
        if event.kind in _NOT_PROBLEMS or event.level in (EventLevel.DEBUG, EventLevel.INFO):
            continue
        grouped.setdefault(event.kind, []).append(event)
    problems = []
    for kind, found in grouped.items():
        first = clock_time(found[0].occurred_at)
        problems.append(f"{kind} at {first}" if len(found) == 1 else f"{kind} {len(found)} times, first at {first}")
    return problems


def _control_problems(conn: sqlite3.Connection, now: datetime) -> list[str]:
    controls = get_controls(conn)
    problems = []
    if controls.kill_switch:
        problems.append("the kill switch is ON")
    if controls.halt_unknown:
        problems.append("the daily loss halt state is unknown (treated as halted)")
    elif controls.halt_until is not None and _utc(controls.halt_until) > _utc(now):
        problems.append(f"trading is halted until {clock_time(controls.halt_until)}")
    return problems


# --- the summary ----------------------------------------------------------------


def build_summary(
    conn: sqlite3.Connection,
    day: date,
    config: AegisConfig,
    *,
    now: datetime,
    account: AccountState | None,
    account_error: str | None = None,
    log_lines: Sequence[LogLine] | None = None,
) -> DaySummary:
    """The day's summary text. ``log_lines`` None means no logs were read."""
    start, end = day_bounds(day)
    events = _events(conn, start, end)
    heading = f"{day.strftime('%a %b')} {day.day}"
    if _market_closed_all_day(log_lines) and not any(e.kind == "cycle_start" for e in events):
        text = f"{heading}\nMarket closed today, so nothing ran."
        return DaySummary(day, text, 0)

    cycles, cycle_problems = _cycles_line(events, config.notify.summary.expected_cycles)
    trades, trade_problems = _trades_line(conn, events, start, end)
    account_text, account_problems = _account_line(account, account_error)
    forecasts, forecast_problems = _forecasts_line(conn, events, day, config)
    cost = _cost_line(conn, events, day, config)
    problems = (
        _control_problems(conn, now) + cycle_problems + trade_problems + forecast_problems
        + _event_problems(events) + account_problems
        + (_log_problems(log_lines) if log_lines is not None else ["the launchd logs were not found"])
    )
    lines = [heading, cycles, trades, account_text, forecasts, cost]
    if not problems:
        lines.append("Problems: none")
    else:
        lines.append(f"Problems ({len(problems)}):")
        lines.extend(f"- {problem}" for problem in problems[:MAX_PROBLEM_LINES])
        if len(problems) > MAX_PROBLEM_LINES:
            lines.append(f"- and {len(problems) - MAX_PROBLEM_LINES} more (python -m aegis.cli.trace)")
    text = "\n".join(lines)
    if len(text) > BODY_MAX:
        text = text[: BODY_MAX - 3].rstrip() + "..."
    return DaySummary(day, text, len(problems))


def summary_event(summary: DaySummary, now: datetime) -> Event:
    """The ``daily_summary`` event ``notify send`` delivers."""
    return Event(
        occurred_at=now,
        level=EventLevel.INFO,
        kind=SUMMARY_KIND,
        message=summary.text,
        payload={"day": summary.day.isoformat(), "problems": summary.problems},
    )
