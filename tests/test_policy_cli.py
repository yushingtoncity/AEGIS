"""The policy CLI: ``evaluate``, ``kill`` and ``limits`` against tmp_path stores.

``main([...], context_builder=...)`` is called directly and its output read
through capsys. The builder is always injected — a hand-made context from
``policy_factories`` with the control flags read from the store under test —
so nothing here reaches the network; two autouse tripwires fail any test
that would open the configured (real) database or reach the live
``build_context`` by accident.

What is pinned: which commands write (and that ``--dry-run``, ``kill
status`` and ``limits`` leave the file byte-for-byte unchanged and never
create or migrate one — nor open an empty file at all), the exit codes,
every printed section, that the context ``evaluate`` prints is the context
that was judged (a kill switch set while the builder ran included), and that
no control character read from the store or typed on the command line
reaches the terminal.
"""

import ast
import inspect
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

import pytest
from policy_factories import (
    NEXT_OPEN,
    NOW,
    call_debit_spread,
    context_for,
    make_clock,
    make_context,
    make_leg,
    make_order,
    make_position,
    make_proposal,
    random_case,
    random_context,
)

from aegis.cli import policy as cli_policy
from aegis.cli import trace as cli_trace
from aegis.config import REPO_ROOT, ConfigError, get_config
from aegis.policy import engine
from aegis.policy.context import build_context
from aegis.policy.limits import limits_report
from aegis.policy.measures import structure_problems
from aegis.policy.models import ExposureLine
from aegis.policy.rules import RULE_NAMES
from aegis.store.db import applied_versions, connect, open_store, schema_version
from aegis.store.errors import StoreError
from aegis.store.models import Controls, EventLevel, Instrument, PolicyDecision, Verdict
from aegis.store.repo import (
    get_controls,
    get_proposal_trace,
    get_recent_events,
    insert_proposal,
    set_halt_until,
    set_kill_switch,
)

CLI_SOURCE = REPO_ROOT / "aegis" / "cli" / "policy.py"
MIGRATIONS = ("0001_initial", "0002_reasoning_cycles_and_legs", "0003_controls", "0004_execution")

RULE_LINE = re.compile(r"^  \[ ?(\d+)\] (\S+) {3,}(\S+) {3,}(.*)$")
"""One printed rule: ``  [ 4] daily_loss_limit   PASS   <detail>``."""

CONTEXT_BLOCK = [
    "context",
    "  as of                  2026-07-30 15:00:00 UTC",
    "  market                 OPEN (next close 2026-07-30 20:00:00 UTC)",
    "  kill switch            off",
    "  halt                   none",
    "  equity                 $100,000.00",
    "  start-of-day equity    $100,000.00",
    "  today's P&L            $0.00",
    "  buying power           $200,000.00",
    "  options buying power   $100,000.00",
    "  positions              0",
    "  open orders            0",
    "  orders today           0",
    "  data problems (0)",
    "    (none)",
]
"""The context block of ``evaluate`` for the default "everything is fine" context."""

LIMITS_REPORT = """\
limits as of 2026-07-30 15:00:00 UTC
  market        OPEN (next close 2026-07-30 20:00:00 UTC)
  kill switch   off
  halt          none
no account-level block

account
  equity                 $100,000.00
  start-of-day equity    $100,000.00
  today's P&L            -$500.00
  buying power           $200,000.00
  options buying power   $100,000.00

limits (5)
  name                   limit       used      headroom      unit
  daily_loss             $2,000.00   $500.00   $1,500.00     USD
      today's P&L -$500.00 is above the loss cap -$2,000.00, 2% of start-of-day equity \
$100,000.00 (risk_limits.daily_loss_limit_pct): $1,500.00 of headroom
  open_positions         5           1         4             positions
      1 of 5 underlyings held or pending (risk_limits.max_open_positions): AAPL
  daily_trades           10          3         7             trades
      3 of 10 orders sent today (risk_limits.max_daily_trades)
  buying_power           -           -         $200,000.00   USD
      $200,000.00 available to equity orders
  options_buying_power   -           -         $100,000.00   USD
      $100,000.00 available to option orders

exposure by underlying (1)
  underlying   exposure    cap         headroom
  AAPL         $2,000.00   $5,000.00   $3,000.00

auto-execute tier
  enabled        yes
  max notional   $1,000.00

data problems (1)
  - quote for AAPL: failed to fetch quote (ConnectionError: down)
"""
"""``limits`` for a context holding 10 AAPL, down $500 on the day, 3 orders
sent and one failed fetch — every figure hand-computed from the defaults (2%
of 100,000 = 2,000 loss cap; 5% of 100,000 = 5,000 position cap)."""


# --- helpers ------------------------------------------------------------------


class Builder:
    """Stands in for ``build_context``: records how it was called and returns
    a hand-made context — the "everything is fine" one for the proposal (or
    for no proposal), with the control flags read from the store under test,
    as the real builder reads them, and ``overrides`` on top."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        conn = args[0]
        proposal = args[1] if len(args) > 1 else None
        fields = {"controls": get_controls(conn), **self.overrides}
        if proposal is None:
            return make_context(**fields)
        return context_for(proposal, **fields)


def fixed(context):
    """A builder that returns ``context`` whatever it is asked."""

    def builder(conn, proposal=None, *, config=None):
        return context

    return builder


def seed(db_path, *proposals):
    """Create (and migrate) the database and store each proposal with its legs."""
    conn = open_store(db_path)
    try:
        for proposal in proposals:
            insert_proposal(conn, proposal.proposal, proposal.legs)
    finally:
        conn.close()


def every_row(db_path) -> dict[str, list[tuple]]:
    """Every row of every table in the database, by table."""
    conn = sqlite3.connect(db_path)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        return {
            name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
            for name in names
        }
    finally:
        conn.close()


def decisions_of(db_path, proposal_id):
    conn = connect(db_path)
    try:
        return get_proposal_trace(conn, proposal_id).decisions
    finally:
        conn.close()


def events_of(db_path, kind=None):
    """Every logged event, oldest first — optionally only one kind."""
    conn = connect(db_path)
    try:
        logged = list(reversed(get_recent_events(conn, limit=10_000)))
    finally:
        conn.close()
    return [event for event in logged if kind is None or event.kind == kind]


def controls_of(db_path) -> Controls:
    conn = connect(db_path)
    try:
        return get_controls(conn)
    finally:
        conn.close()


def loosen_controls(db_path, rows):
    """Rebuild ``controls`` by hand without its CHECKs and its no-delete
    trigger, holding exactly ``rows`` — the only way a store can come to hold
    a control the repository cannot read."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE controls")  # its trigger goes with it
        conn.execute("CREATE TABLE controls (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        conn.executemany("INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def drop_migration_0003(db_path):
    """Take 0003 out of a migrated store: the controls table is gone and 0003 is pending
    again (0004, which touches no part of 0003, stays applied)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE controls")
        conn.execute("DELETE FROM schema_version WHERE version = 3")
        conn.commit()
    finally:
        conn.close()


def versions_of(db_path) -> dict[int, str]:
    conn = sqlite3.connect(db_path)
    try:
        return applied_versions(conn)
    finally:
        conn.close()


def journal_mode_of(db_path) -> str:
    """The file's journal mode, read without changing it."""
    conn = sqlite3.connect(db_path)
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def listing(directory) -> list[str]:
    """Every file in ``directory``, by name — a ``-wal`` or ``-shm`` left behind included."""
    return sorted(path.name for path in directory.iterdir())


def elsewhere(db_path, write) -> None:
    """Another process writing to the store — the operator's ``kill on``, the
    loop — while this one is still building its context."""
    other = open_store(db_path)
    try:
        write(other)
    finally:
        other.close()


def damage_halt(db_path) -> None:
    """Leave the stored halt unreadable, by hand and around the repository
    (its CHECK bypassed) — the only way a store comes to hold one."""
    other = sqlite3.connect(db_path)
    try:
        other.execute("PRAGMA ignore_check_constraints = ON")
        other.execute("UPDATE controls SET value = 'tomorrow, probably' WHERE key = 'halt_until'")
        other.commit()
    finally:
        other.close()


def rule_lines(out: str) -> list[tuple[int, str, str, str]]:
    """The printed rules as ``(index, name, outcome, detail)``, in print order."""
    found = []
    for line in out.splitlines():
        match = RULE_LINE.match(line)
        if match:
            index, name, outcome, detail = match.groups()
            found.append((int(index), name, outcome, detail))
    return found


def cells(out: str, name: str) -> list[str]:
    """The cells of the table row that starts with ``name``."""
    (line,) = [line for line in out.splitlines() if line.startswith(f"  {name} ")]
    return re.split(r" {3,}", line.strip())


def control_characters(text: str) -> set[str]:
    """Every control or format character in ``text`` other than the newline."""
    return {ch for ch in text if ch != "\n" and unicodedata.category(ch) in ("Cc", "Cf")}


def over_the_auto_tier(**overrides):
    """5 AAPL at 200.20 = 1,001.00: one dollar over the default auto tier."""
    return make_proposal(quantity=5.0, limit_price=200.20, **overrides)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "policy.db"


@pytest.fixture
def store(db_path):
    """A migrated database holding the default proposal (``prop-0001``)."""
    seed(db_path, make_proposal())
    return db_path


@pytest.fixture(autouse=True)
def no_real_database_and_no_live_context(monkeypatch):
    """Tripwires: a test that forgets ``--db`` must not open the configured
    database (the user's real store), and one that forgets ``context_builder``
    must not reach the live builder (Alpaca)."""

    def configured_path(*args, **kwargs):
        raise AssertionError("a policy CLI test resolved the configured database path")

    def live_builder(*args, **kwargs):
        raise AssertionError("a policy CLI test reached the live build_context")

    monkeypatch.setattr(cli_policy, "resolve_db_path", configured_path)
    monkeypatch.setattr(cli_policy, "build_context", live_builder)


@pytest.fixture
def at_now(monkeypatch):
    """The CLI's own clock (``kill``'s timestamps and halt state) stands at ``NOW``."""
    monkeypatch.setattr(cli_policy, "utcnow", lambda: NOW)


@pytest.fixture
def opened(monkeypatch):
    """Record every connection the CLI opens, as ``(opener, connection)``."""
    seen = []
    real = {"open_store": cli_policy.open_store, "connect": cli_policy.connect}

    def recording(name):
        def opener(path):
            conn = real[name](path)
            seen.append((name, conn))
            return conn

        return opener

    for name in real:
        monkeypatch.setattr(cli_policy, name, recording(name))
    return seen


def assert_all_closed(opened):
    assert opened
    for _, conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


# --- evaluate -----------------------------------------------------------------


class TestEvaluate:
    def test_by_id_records_the_decision_and_prints_every_rule(self, store, capsys):
        builder = Builder()
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=builder) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()

        (decision,) = decisions_of(store, "prop-0001")
        assert decision.verdict is Verdict.AUTO_EXECUTE and decision.failing_rule is None
        assert decision.decided_at == NOW
        assert [rule["rule"] for rule in decision.rules_evaluated] == list(RULE_NAMES)
        assert lines[:11] == [
            "proposal prop-0001",
            "  created      2026-07-30 14:59:00 UTC",
            "  symbol       AAPL (equity)",
            "  order        buy 2 limit @ 200.00",
            "  confidence   0.8",
            "  legs (0)",
            "    (none)",
            "",
            "verdict: AUTO_EXECUTE",
            f"decision {decision.id} recorded",
            "",
        ]
        assert lines[11] == "rules (21)"
        # all twenty-one, in registry order, exactly as recorded
        assert rule_lines(captured.out) == [
            (index, rule["rule"], rule["outcome"], rule["detail"])
            for index, rule in enumerate(decision.rules_evaluated, start=1)
        ]
        assert [name for _, name, _, _ in rule_lines(captured.out)] == list(RULE_NAMES)
        assert lines[12].startswith("  [ 1] kill_switch ")
        assert lines[24].startswith("  [13] quote_freshness ")
        assert lines[25].startswith("  [14] limit_price_sanity ")
        assert lines[32].startswith("  [21] auto_tier ")
        assert lines[33] == "" and lines[34:] == CONTEXT_BLOCK
        assert "failing rule" not in captured.out and "HALTED" not in captured.out
        assert "DRY RUN" not in captured.out
        assert events_of(store) == []  # AUTO_EXECUTE logs no event: the decision is its record

    def test_latest_judges_the_newest_proposal_without_a_decision(self, db_path, capsys):
        older = make_proposal()
        newer = over_the_auto_tier(id="prop-0002", created_at=NOW)
        seed(db_path, older, newer)
        builder = Builder()
        argv = ["evaluate", "--latest", "--db", str(db_path)]

        assert cli_policy.main(argv, context_builder=builder) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "proposal prop-0002"
        assert "verdict: NEEDS_APPROVAL" in out.splitlines()
        assert len(rule_lines(out)) == 21
        (decision,) = decisions_of(db_path, "prop-0002")
        assert f"decision {decision.id} recorded" in out.splitlines()
        assert decisions_of(db_path, "prop-0001") == ()

        # the newest is decided now, so --latest moves on to the older one
        assert cli_policy.main(argv, context_builder=builder) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0] == "proposal prop-0001" and "verdict: AUTO_EXECUTE" in out
        assert len(decisions_of(db_path, "prop-0001")) == 1

        # and then there is nothing left to judge
        before = every_row(db_path)
        assert cli_policy.main(argv, context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy evaluate failed: no proposal without a decision in {db_path}\n"
        )
        assert every_row(db_path) == before and len(builder.calls) == 2

    def test_the_builder_gets_the_connection_the_proposal_and_the_config(self, store, capsys):
        builder = Builder()
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=builder) == 0
        ((args, kwargs),) = builder.calls
        conn, proposal = args
        assert isinstance(conn, sqlite3.Connection)
        assert proposal.id == "prop-0001" and proposal == make_proposal()
        assert list(kwargs) == ["config"] and kwargs["config"] is get_config()

    def test_the_default_builder_is_build_context(self, store, monkeypatch, capsys):
        assert inspect.signature(cli_policy.main).parameters["context_builder"].default is None
        seen = []

        def fake_build_context(conn, proposal=None, *, config=None, now=None):
            seen.append((proposal.id if proposal else None, config))
            return make_context() if proposal is None else context_for(proposal)

        # same shape as the real one, so the call the CLI makes fits build_context
        assert list(inspect.signature(fake_build_context).parameters) == list(
            inspect.signature(build_context).parameters
        )
        monkeypatch.setattr(cli_policy, "build_context", fake_build_context)
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)]) == 0
        assert cli_policy.main(["limits", "--db", str(store)]) == 0
        assert seen == [("prop-0001", get_config()), (None, get_config())]

    def test_an_option_prints_its_legs(self, db_path, capsys):
        seed(db_path, call_debit_spread())
        assert cli_policy.main(["evaluate", "--latest", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[:8] == [
            "proposal prop-0001",
            "  created      2026-07-30 14:59:00 UTC",
            "  symbol       SPY (option)",
            "  order        buy 1 limit @ 4.40",
            "  confidence   0.8",
            "  legs (2)",
            "    [0] buy 1 call 640.0 exp 2026-08-21  SPY260821C00640000",
            "    [1] sell 1 call 650.0 exp 2026-08-21  SPY260821C00650000",
        ]
        assert "verdict: NEEDS_APPROVAL" in lines  # an option never auto-executes
        outcomes = {name: outcome for _, name, outcome, _ in rule_lines("\n".join(lines))}
        assert outcomes["options_escalate"] == "ESCALATE" and outcomes["auto_tier"] == "ESCALATE"

    def test_quantities_and_prices_print_every_digit(self, db_path, capsys):
        seed(db_path, make_proposal(quantity=0.5, limit_price=561.275))
        seed(db_path, make_proposal(id="prop-0002", order_type="market", limit_price=None))
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        assert "  order        buy 0.5 limit @ 561.275" in capsys.readouterr().out.splitlines()
        assert cli_policy.main(["evaluate", "prop-0002", "--dry-run", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        assert "  order        buy 2 market" in capsys.readouterr().out.splitlines()

    @pytest.mark.parametrize(
        ("confidence", "shown"),
        [
            (0.555, "0.555"),  # not 0.56 ...
            (0.495, "0.495"),  # ... and not 0.50: this one is BELOW the 0.5 floor
            (0.8049999, "0.8049999"),
            (0.123456789012, "0.123456789012"),
            (0.1 + 0.2, "0.30000000000000004"),
            (0.0000001, "0.0000001"),
            (0.5, "0.5"),
            (1.0, "1"),
            (0.0, "0"),
        ],
    )
    def test_confidence_prints_every_digit_as_stored(self, db_path, capsys, confidence, shown):
        seed(db_path, make_proposal(confidence=confidence))
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        lines = capsys.readouterr().out.splitlines()
        assert [line for line in lines if line.startswith("  confidence ")] == [
            f"  confidence   {shown}"
        ]
        # What is printed reads back as exactly the stored number: nothing was rounded.
        assert float(shown) == confidence

    def test_a_confidence_under_the_floor_is_not_rounded_up_to_it(self, db_path, capsys):
        # 0.495 printed to two decimals would read "0.50" beside a FLAG for
        # being below 0.5 — the audit view would contradict the verdict.
        seed(db_path, make_proposal(confidence=0.495))
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        out = capsys.readouterr().out
        assert "  confidence   0.495" in out.splitlines()
        assert "verdict: FLAG_ONLY" in out.splitlines()

    def test_an_option_contract_declared_equity_is_rejected_and_recorded(self, db_path, capsys):
        # But for its shape, a perfect auto-tier buy: a limit at the mid of a
        # two-sided quote for that very symbol, on a watchlist underlying.
        contract = "SPY260821C00640000"
        seed(db_path, make_proposal(symbol=contract, limit_price=8.00))
        assert cli_policy.main(["evaluate", "--latest", "--db", str(db_path)],
                               context_builder=Builder()) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert f"  symbol       {contract} (equity)" in lines
        assert "verdict: REJECT" in lines and "failing rule: options_max_loss" in lines
        found = {name: (outcome, detail) for _, name, outcome, detail in rule_lines(out)}
        assert found["options_max_loss"] == (
            "REJECT",
            "the proposal is not a plain equity: the proposal is declared equity but its "
            "symbol SPY260821C00640000 is an option contract",
        )
        assert found["options_escalate"][0] == found["auto_tier"][0] == "ESCALATE"
        assert [name for name, (outcome, _) in found.items() if outcome != "PASS"] == [
            "options_max_loss", "options_escalate", "auto_tier",
        ]
        (decision,) = decisions_of(db_path, "prop-0001")
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "options_max_loss"

    def test_a_reject_names_the_failing_rule_and_still_exits_0(self, store, capsys):
        conn = open_store(store)
        set_kill_switch(conn, True, now=NOW)
        conn.close()
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "kill_switch"
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 3] == [
            "failing rule: kill_switch",
            f"decision {decision.id} recorded",
        ]
        assert rule_lines(captured.out)[0] == (
            1, "kill_switch", "REJECT", "the kill switch is ON: every proposal is rejected",
        )
        assert "  kill switch            ON — every proposal is rejected" in lines
        (event,) = events_of(store, "policy_reject")
        assert event.level is EventLevel.WARNING
        assert event.payload["decision_id"] == decision.id

    @pytest.mark.parametrize(
        "proposal, overrides, verdict, event_kind",
        [
            (make_proposal(), {}, "AUTO_EXECUTE", None),
            (over_the_auto_tier(), {}, "NEEDS_APPROVAL", "policy_needs_approval"),
            (make_proposal(confidence=0.2), {}, "FLAG_ONLY", "policy_flag_only"),
            (make_proposal(), {"clock": make_clock(False)}, "REJECT", "policy_reject"),
            (make_proposal(invalidation=" "), {}, "REJECT", "policy_reject"),
        ],
        ids=["auto", "needs-approval", "flag-only", "closed-market", "no-invalidation"],
    )
    def test_exit_0_whatever_the_verdict(
        self, db_path, capsys, proposal, overrides, verdict, event_kind
    ):
        seed(db_path, proposal)
        assert cli_policy.main(["evaluate", "--latest", "--db", str(db_path)],
                               context_builder=Builder(**overrides)) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert f"verdict: {verdict}" in captured.out.splitlines()
        assert ("failing rule: " in captured.out) is (verdict == "REJECT")
        (decision,) = decisions_of(db_path, proposal.id)
        assert decision.verdict.value == verdict
        assert [event.kind for event in events_of(db_path)] == ([event_kind] if event_kind else [])

    def test_a_tripped_loss_limit_prints_the_halt_and_persists_it(self, store, capsys):
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        lines = capsys.readouterr().out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 4] == [
            "failing rule: daily_loss_limit",
            f"decision {decision.id} recorded",
            "trading HALTED until 2026-07-31 13:30:00 UTC",
        ]
        assert "  today's P&L            -$2,500.00" in lines
        # the context block is the context the verdict rested on: no halt yet
        assert "  halt                   none" in lines
        assert controls_of(store).halt_until == NEXT_OPEN
        (trip,) = events_of(store, "risk_limit_tripped")
        assert trip.level is EventLevel.CRITICAL and trip.payload["proposal_id"] == "prop-0001"

        # the halt outlives the loss: a later evaluation with the P&L recovered is still rejected
        seed(store, make_proposal(id="prop-0002", created_at=NOW))
        assert cli_policy.main(["evaluate", "--latest", "--db", str(store)],
                               context_builder=Builder(daily_pnl=750.0)) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "proposal prop-0002" and "failing rule: halted" in lines
        assert "  halt                   until 2026-07-31 13:30:00 UTC (active)" in lines
        assert not any("HALTED" in line for line in lines)  # this evaluation set no halt
        assert len(events_of(store, "risk_limit_tripped")) == 1

    def test_the_context_block_counts_positions_and_orders(self, store, capsys):
        builder = Builder(
            positions=(make_position("MSFT"), make_position("NVDA")),
            open_orders=(make_order("QQQ"),),
            orders_today=3,
        )
        argv = ["evaluate", "prop-0001", "--dry-run", "--db", str(store)]
        assert cli_policy.main(argv, context_builder=builder) == 0
        lines = capsys.readouterr().out.splitlines()
        position = lines.index("  positions              2")
        assert lines[position + 1 : position + 3] == [
            "  open orders            1",
            "  orders today           3",
        ]
        # positions that could not be fetched are unknown, never "0"
        assert cli_policy.main(argv, context_builder=Builder(positions=None)) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "  positions              unknown" in lines
        assert "  open orders            0" in lines and "  orders today           0" in lines

    def test_evaluating_a_decided_proposal_by_id_adds_a_decision(self, store, capsys):
        argv = ["evaluate", "prop-0001", "--db", str(store)]
        assert cli_policy.main(argv, context_builder=Builder()) == 0
        assert cli_policy.main(argv, context_builder=Builder(clock=None)) == 0
        first, second = decisions_of(store, "prop-0001")
        assert (first.verdict, second.verdict) == (Verdict.AUTO_EXECUTE, Verdict.REJECT)
        assert second.failing_rule == "market_hours"
        out = capsys.readouterr().out
        assert f"decision {first.id} recorded" in out and f"decision {second.id} recorded" in out
        assert "  market                 UNKNOWN (no market clock)" in out.splitlines()

    def test_the_trace_cli_shows_the_recorded_decision(self, store, capsys):
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(orders_today=10)) == 0
        capsys.readouterr()
        assert cli_trace.main(["prop-0001", "--db", str(store)]) == 0
        out = capsys.readouterr().out
        assert "decisions (1)" in out and "failing rule: max_daily_trades" in out
        assert "rules evaluated (21)" in out

    def test_unknown_id_is_one_clean_line(self, store, capsys):
        before = every_row(store)
        builder = Builder()
        assert cli_policy.main(["evaluate", "nope", "--db", str(store)],
                               context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            "policy evaluate failed: policy failed: load proposal (no such proposal) for nope\n"
        )
        assert builder.calls == [] and every_row(store) == before

    def test_nothing_undecided_in_an_empty_store(self, db_path, capsys):
        seed(db_path)
        assert cli_policy.main(["evaluate", "--latest", "--db", str(db_path)],
                               context_builder=Builder()) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "no proposal without a decision" in captured.err
        assert captured.err.startswith("policy evaluate failed: ")
        assert captured.err.count("\n") == 1

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    @pytest.mark.parametrize("target", [["prop-0001"], ["--latest"]], ids=["by-id", "latest"])
    def test_a_missing_database_is_never_created(self, tmp_path, capsys, target, flags):
        missing = tmp_path / "nested" / "policy.db"
        builder = Builder()
        assert cli_policy.main(["evaluate", *target, *flags, "--db", str(missing)],
                               context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy evaluate failed: database {missing} does not exist"
            " (run `python -m aegis.cli.db init` to create it)\n"
        )
        assert not missing.exists() and not missing.parent.exists()
        assert builder.calls == []

    def test_the_connection_is_opened_with_open_store_and_closed(self, store, opened, capsys):
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        # A plain connection first, to see that the file holds a schema at
        # all (and closed again); then the store, as the loop opens it.
        assert [name for name, _ in opened] == ["connect", "open_store"]
        assert_all_closed(opened)

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_the_report_is_worked_out_before_anything_is_written(
        self, store, monkeypatch, capsys, flags
    ):
        """A failure while the report is being put together leaves no decision
        behind — and no halt, though this context would have tripped one."""

        def failing(context):
            raise RuntimeError("report failed")

        monkeypatch.setattr(cli_policy, "limits_report", failing)
        before = every_row(store)
        builder = Builder(daily_pnl=-2_500.0, clock=make_clock(False))
        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "policy evaluate failed (unexpected): RuntimeError: report failed\n"
        assert len(builder.calls) == 1  # it got as far as the context
        assert every_row(store) == before  # no decision, no event, no halt
        assert decisions_of(store, "prop-0001") == () and events_of(store) == []
        assert controls_of(store).halt_until is None

    def test_the_halt_is_worked_out_before_anything_is_written_too(
        self, store, monkeypatch, capsys
    ):
        def failing(proposal, context):
            raise RuntimeError("decide failed")

        monkeypatch.setattr(cli_policy, "decide", failing)
        before = every_row(store)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 1
        assert capsys.readouterr().err == (
            "policy evaluate failed (unexpected): RuntimeError: decide failed\n"
        )
        assert every_row(store) == before

    # The context printed is the context judged: the store's stop flags are
    # read once more after the builder returns, and the stricter reading is
    # what the report, the rule lines and the decision all show.

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_a_kill_switch_set_while_the_context_was_built_is_on_in_the_whole_report(
        self, store, capsys, flags
    ):
        built = []

        def stale(conn, proposal=None, *, config=None):
            context = context_for(proposal, controls=get_controls(conn))  # reads: off
            elsewhere(store, lambda other: set_kill_switch(other, True, now=NOW))
            built.append(context)
            return context  # ... and still says off

        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=stale) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert built[0].controls.kill_switch is False and controls_of(store).kill_switch is True
        lines = captured.out.splitlines()
        # the verdict,
        position = lines.index("verdict: REJECT")
        assert lines[position + 1] == "failing rule: kill_switch"
        # the rule line,
        assert rule_lines(captured.out)[0] == (
            1, "kill_switch", "REJECT", "the kill switch is ON: every proposal is rejected",
        )
        assert [n for _, n, outcome, _ in rule_lines(captured.out) if outcome != "PASS"] == [
            "kill_switch"
        ]
        # and the context block all say ON.
        assert "  kill switch            ON — every proposal is rejected" in lines
        assert not any(line.startswith("  kill switch") and line.endswith("off") for line in lines)
        if flags:
            assert lines[position + 2] == "DRY RUN — nothing was written"
            assert decisions_of(store, "prop-0001") == () and events_of(store) == []
        else:
            (decision,) = decisions_of(store, "prop-0001")
            assert lines[position + 2] == f"decision {decision.id} recorded"
            assert decision.verdict is Verdict.REJECT and decision.failing_rule == "kill_switch"
            assert decision.rules_evaluated[0]["outcome"] == "REJECT"
            assert rule_lines(captured.out) == [
                (index, rule["rule"], rule["outcome"], rule["detail"])
                for index, rule in enumerate(decision.rules_evaluated, start=1)
            ]

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_a_halt_written_while_the_context_was_built_is_in_the_whole_report(
        self, store, capsys, flags
    ):
        def stale(conn, proposal=None, *, config=None):
            context = context_for(proposal, controls=get_controls(conn))
            elsewhere(store, lambda other: set_halt_until(other, NEXT_OPEN, now=NOW))
            return context

        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=stale) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "verdict: REJECT" in lines and "failing rule: halted" in lines
        assert rule_lines(out)[1][1:3] == ("halted", "REJECT")
        assert "  halt                   until 2026-07-31 13:30:00 UTC (active)" in lines
        assert "  halt                   none" not in lines
        assert not any("HALTED" in line for line in lines)  # this evaluation set no halt
        assert len(decisions_of(store, "prop-0001")) == (0 if flags else 1)

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_a_longer_stored_halt_is_not_announced_as_a_new_one(self, store, capsys, flags):
        # The context trips the loss limit and names tomorrow's open; the
        # store, by the time the builder returns, holds a halt beyond it.
        far = NEXT_OPEN + timedelta(days=3)

        def stale(conn, proposal=None, *, config=None):
            context = context_for(proposal, controls=get_controls(conn), daily_pnl=-2_500.0)
            elsewhere(store, lambda other: set_halt_until(other, far, now=NOW))
            return context

        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=stale) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "failing rule: halted" in lines
        assert "  halt                   until 2026-08-03 13:30:00 UTC (active)" in lines
        assert not any("HALTED" in line for line in lines)  # nothing to announce
        assert controls_of(store).halt_until == far
        assert events_of(store, "risk_limit_tripped") == []

    def test_a_stop_the_context_holds_is_never_lifted_by_the_store(self, store, capsys):
        # The other way round: the context saw the switch ON, the store says off.
        builder = Builder(controls=Controls(kill_switch=True))
        assert controls_of(store).kill_switch is False
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=builder) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "failing rule: kill_switch" in lines
        assert "  kill switch            ON — every proposal is rejected" in lines

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_a_context_the_store_adds_nothing_to_is_judged_as_it_is(
        self, store, monkeypatch, capsys, flags
    ):
        context = context_for(make_proposal(), orders_today=3)
        seen = []
        real = cli_policy.evaluate

        def spy(proposal, given, conn, *, dry_run=False):
            seen.append(given)
            return real(proposal, given, conn, dry_run=dry_run)

        monkeypatch.setattr(cli_policy, "evaluate", spy)
        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=fixed(context)) == 0
        assert len(seen) == 1 and seen[0] is context  # the builder's own object, no copy
        assert "  orders today           3" in capsys.readouterr().out.splitlines()

    def test_the_engine_is_given_the_context_that_was_printed(self, store, monkeypatch, capsys):
        seen = []
        real = cli_policy.evaluate

        def spy(proposal, given, conn, *, dry_run=False):
            seen.append(given)
            return real(proposal, given, conn, dry_run=dry_run)

        def stale(conn, proposal=None, *, config=None):
            context = context_for(proposal, controls=get_controls(conn))
            elsewhere(store, lambda other: set_kill_switch(other, True, now=NOW))
            return context

        monkeypatch.setattr(cli_policy, "evaluate", spy)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=stale) == 0
        (given,) = seen
        assert given.controls.kill_switch is True  # already the stricter reading
        assert given.controls == controls_of(store)
        assert engine.stricter_controls(given.controls, controls_of(store)) is given.controls

    # The engine reads the stop flags itself as it records — an instant after
    # this command did. What reaches the store in that instant is in the
    # decision; the report is rebuilt so that its context block shows it too.

    @staticmethod
    def before_the_engines_read(monkeypatch, db_path, write) -> list:
        """``write`` reaches the store just before the ENGINE's read of the stop
        flags — after this command's own (``cli_policy.get_controls``)."""
        real = engine.get_controls
        landed = []

        def late(conn):
            if not landed:
                landed.append(True)
                elsewhere(db_path, write)
            return real(conn)

        monkeypatch.setattr(engine, "get_controls", late)
        return landed

    def test_a_kill_switch_set_between_this_commands_read_and_the_engines_is_on_throughout(
        self, store, monkeypatch, capsys
    ):
        landed = self.before_the_engines_read(
            monkeypatch, store, lambda other: set_kill_switch(other, True, now=NOW)
        )
        builder = Builder()
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=builder) == 0
        captured = capsys.readouterr()
        assert landed and captured.err == ""
        lines = captured.out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 4] == [
            "failing rule: kill_switch",
            f"decision {decision.id} recorded",
            "",  # no halt, and no note: the context that was judged was found
        ]
        assert rule_lines(captured.out)[0] == (
            1, "kill_switch", "REJECT", "the kill switch is ON: every proposal is rejected",
        )
        assert rule_lines(captured.out) == [
            (index, rule["rule"], rule["outcome"], rule["detail"])
            for index, rule in enumerate(decision.rules_evaluated, start=1)
        ]
        # ... and the context block is that of the context that was judged.
        assert "  kill switch            ON — every proposal is rejected" in lines
        assert "  kill switch            off" not in lines
        assert not any(line.startswith("note:") for line in lines)
        assert lines[lines.index("context") + 4] == "  halt                   none"

    def test_a_halt_written_in_that_instant_is_shown_and_not_announced_as_this_ones(
        self, store, monkeypatch, capsys
    ):
        # This command's context trips the loss limit and names tomorrow's open;
        # a longer halt reaches the store before the engine reads it.
        far = NEXT_OPEN + timedelta(days=3)
        self.before_the_engines_read(
            monkeypatch, store, lambda other: set_halt_until(other, far, now=NOW)
        )
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "failing rule: halted" in lines
        assert rule_lines(out)[3][3].endswith(
            ": a halt until 2026-08-03T13:30:00+00:00 is already in force"
        )
        assert "  halt                   until 2026-08-03 13:30:00 UTC (active)" in lines
        assert "  halt                   none" not in lines
        assert not any("HALTED" in line for line in lines)  # this evaluation set no halt
        assert not any(line.startswith("note:") for line in lines)
        assert controls_of(store).halt_until == far
        assert events_of(store, "risk_limit_tripped") == []

    def test_a_switch_set_in_that_instant_beside_a_trip_keeps_the_halt_this_one_wrote(
        self, store, monkeypatch, capsys
    ):
        # Read back after the engine's writes, the store holds the halt this
        # very evaluation wrote — which was not in force when it was judged.
        self.before_the_engines_read(
            monkeypatch, store, lambda other: set_kill_switch(other, True, now=NOW)
        )
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 5] == [
            "failing rule: kill_switch",
            f"decision {decision.id} recorded",
            "trading HALTED until 2026-07-31 13:30:00 UTC",
            "",
        ]
        assert [(name, outcome) for _, name, outcome, _ in rule_lines(out)][:4] == [
            ("kill_switch", "REJECT"), ("halted", "PASS"), ("market_hours", "PASS"),
            ("daily_loss_limit", "REJECT"),
        ]
        assert "  kill switch            ON — every proposal is rejected" in lines
        assert "  halt                   none" in lines  # as judged: the halt came after
        assert not any(line.startswith("note:") for line in lines)
        assert controls_of(store).halt_until == NEXT_OPEN

    def test_the_store_as_it_stands_is_the_first_reading_tried(self, store, monkeypatch, capsys):
        """Both readings ``_as_judged`` tries can reproduce the rule lines and
        still differ in the halt they print: with an unreadable halt in this
        command's context, ``halted`` rejects the same way whatever instant
        the controls hold. A switch and a readable halt then reach the store
        before the engine's read; the engine judged under that halt, so the
        context block shows it — the reading without it comes second."""
        far = NEXT_OPEN + timedelta(days=3)
        unreadable = Controls(halt_unknown=True, problems=("halt_until is unreadable",))

        def switch_and_halt(other):
            set_kill_switch(other, True, now=NOW)
            set_halt_until(other, far, now=NOW)

        self.before_the_engines_read(monkeypatch, store, switch_and_halt)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(controls=unreadable)) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        assert rule_lines(out) == [
            (index, rule["rule"], rule["outcome"], rule["detail"])
            for index, rule in enumerate(decision.rules_evaluated, start=1)
        ]
        assert rule_lines(out)[:2] == [
            (1, "kill_switch", "REJECT", "the kill switch is ON: every proposal is rejected"),
            (2, "halted", "REJECT",
             "cannot prove trading is not halted: halt_until is unreadable"),
        ]
        assert "  kill switch            ON — every proposal is rejected" in lines
        # ... the halt as the store holds it, not the reading with the halt left out
        assert "  halt                   until 2026-08-03 13:30:00 UTC (active)" in lines
        assert not any(line.startswith("  halt") and "UNKNOWN" in line for line in lines)
        assert not any(line.startswith("note:") for line in lines)

    def test_a_halt_the_store_refused_is_not_announced_as_written(
        self, store, monkeypatch, capsys
    ):
        """The last window: a longer halt lands after the engine read the flags
        and before it writes. The store's compare-and-set keeps the longer
        one, and the report says which halt is on record."""
        far = NEXT_OPEN + timedelta(days=3)
        real = engine.set_halt_until

        def late(conn, until, **kwargs):
            elsewhere(store, lambda other: set_halt_until(other, far, now=NOW))
            return real(conn, until, **kwargs)

        monkeypatch.setattr(engine, "set_halt_until", late)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        lines = capsys.readouterr().out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 5] == [
            "failing rule: daily_loss_limit",
            f"decision {decision.id} recorded",
            "trading already HALTED until 2026-08-03 13:30:00 UTC: a longer halt than this"
            " evaluation's (2026-07-31 13:30:00 UTC) was on record first",
            "",
        ]
        assert "trading HALTED until 2026-07-31 13:30:00 UTC" not in lines
        assert controls_of(store).halt_until == far
        assert events_of(store, "risk_limit_tripped") == []  # refused: row and event alike

    DAMAGED = "damaged"  # in place of an instant: the row left unreadable by hand

    @pytest.mark.parametrize(
        ("then", "reads"),
        [
            (None, "none"),
            (NOW + timedelta(hours=1), "until 2026-07-30 16:00:00 UTC (active)"),
            (NOW - timedelta(hours=1), "expired at 2026-07-30 14:00:00 UTC"),
            (NOW, "expired at 2026-07-30 15:00:00 UTC"),  # a halt until now has ended
            (DAMAGED, "UNKNOWN (cannot prove trading is not halted)"),
        ],
        ids=["cleared", "shortened", "shortened-to-the-past", "shortened-to-now", "damaged"],
    )
    def test_a_halt_changed_right_after_the_write_is_reported_as_the_store_reads(
        self, store, monkeypatch, capsys, then, reads
    ):
        """This evaluation's halt was written and logged — and is no longer what
        the store holds when it is read back: cleared or shortened by an
        operator, or damaged. The line says both, the second in the words
        ``kill status`` would use."""
        real = engine.record_decision

        def then_changed(conn, decision, *, event=None):
            recorded = real(conn, decision, event=event)
            if then == self.DAMAGED:
                damage_halt(store)
            else:
                elsewhere(store, lambda other: set_halt_until(other, then, now=NOW))
            return recorded

        monkeypatch.setattr(engine, "record_decision", then_changed)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        lines = capsys.readouterr().out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index(f"decision {decision.id} recorded")
        assert lines[position + 1 : position + 3] == [
            f"trading HALTED until 2026-07-31 13:30:00 UTC — but the store now reads: {reads}",
            "",  # no note: the rule lines are this command's own
        ]
        after = controls_of(store)
        if then == self.DAMAGED:
            assert after.halt_unknown is True and after.halt_until is None
        else:
            assert after.halt_until == then and after.halt_unknown is False
        assert len(events_of(store, "risk_limit_tripped")) == 1  # it was written, and logged

    def test_flags_that_changed_twice_are_said_to_have_changed(self, store, monkeypatch, capsys):
        """No reading in hand reproduces the decision: the switch was ON for
        the engine's read and is off again when the store is read back. The
        decision is reported as recorded, its rule lines as judged — and a
        note says the context block is not that context."""
        self.before_the_engines_read(
            monkeypatch, store, lambda other: set_kill_switch(other, True, now=NOW)
        )
        real = engine.record_decision

        def then_off(conn, decision, *, event=None):
            recorded = real(conn, decision, event=event)
            elsewhere(store, lambda other: set_kill_switch(other, False, now=NOW))
            return recorded

        monkeypatch.setattr(engine, "record_decision", then_off)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        assert decision.failing_rule == "kill_switch"
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 5] == [
            "failing rule: kill_switch",
            f"decision {decision.id} recorded",
            cli_policy.STALE_CONTEXT_NOTE,
            "",
        ]
        assert cli_policy.STALE_CONTEXT_NOTE.startswith("note: the kill switch or the halt changed")
        assert rule_lines(captured.out)[0][1:3] == ("kill_switch", "REJECT")  # as judged
        assert "  kill switch            off" in lines  # this command's own reading

    def test_a_store_that_cannot_be_read_back_does_not_hide_the_recorded_decision(
        self, store, monkeypatch, capsys
    ):
        real = cli_policy.get_controls
        reads = []

        def once(conn):
            reads.append(conn)
            if len(reads) > 1:
                raise StoreError("get controls", cause=sqlite3.OperationalError("disk gone"))
            return real(conn)

        monkeypatch.setattr(cli_policy, "get_controls", once)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        captured = capsys.readouterr()
        assert captured.err == "" and len(reads) == 2
        lines = captured.out.splitlines()
        (decision,) = decisions_of(store, "prop-0001")
        position = lines.index("verdict: REJECT")
        assert lines[position + 2 : position + 4] == [
            f"decision {decision.id} recorded",
            "trading HALTED until 2026-07-31 13:30:00 UTC",
        ]
        note = lines[position + 4]
        assert note.startswith(
            "note: the store could not be read back after the decision was recorded ("
        )
        assert "disk gone" in note and lines[position + 5] == ""

    def test_the_store_is_read_back_only_when_there_is_something_to_check(
        self, store, monkeypatch, capsys
    ):
        real = cli_policy.get_controls
        reads = []

        def counting(conn):
            reads.append(conn)
            return real(conn)

        monkeypatch.setattr(cli_policy, "get_controls", counting)
        argv = ["evaluate", "prop-0001", "--db", str(store)]
        assert cli_policy.main(argv, context_builder=fixed(make_context())) == 0
        assert len(reads) == 1  # nothing differed, no halt was named: no second read
        assert cli_policy.main([*argv, "--dry-run"],
                               context_builder=fixed(make_context(daily_pnl=-2_500.0))) == 0
        assert len(reads) == 2  # a dry run never reads back: it wrote nothing
        assert cli_policy.main(argv, context_builder=fixed(make_context(daily_pnl=-2_500.0))) == 0
        assert len(reads) == 4  # a halt was named: the store says what it holds
        capsys.readouterr()

    # An unreadable halt beside an instant — what the stricter reading holds
    # when the row was damaged after the context read it.

    @staticmethod
    def damaged_after_reading(db_path, **overrides):
        """A builder that reads the store's (readable) halt and then finds the
        row damaged by hand, as another process would leave it."""

        def builder(conn, proposal=None, *, config=None):
            context = context_for(proposal, controls=get_controls(conn), **overrides)
            damage_halt(db_path)
            return context

        return builder

    @pytest.mark.parametrize("flags", [[], ["--dry-run"]], ids=["recording", "dry-run"])
    def test_an_unreadable_halt_is_never_printed_as_active_on_an_expired_instant(
        self, store, capsys, flags
    ):
        expired = NOW - timedelta(hours=2)
        elsewhere(store, lambda other: set_halt_until(other, expired, now=NOW - timedelta(hours=3)))
        builder = self.damaged_after_reading(store)
        assert cli_policy.main(["evaluate", "prop-0001", *flags, "--db", str(store)],
                               context_builder=builder) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "failing rule: halted" in lines
        assert rule_lines(out)[1][3].startswith("cannot prove trading is not halted: halt_until")
        assert (
            "  halt                   UNKNOWN (cannot prove trading is not halted); the last"
            " halt that could be read expired at 2026-07-30 13:00:00 UTC"
        ) in lines
        assert not any("(active)" in line for line in lines)

    def test_a_trip_on_a_halt_damaged_after_it_was_read_repairs_the_store(self, store, capsys):
        # The longer reading is kept: the trip names 07-31, the context read 08-03.
        far = NEXT_OPEN + timedelta(days=3)
        elsewhere(store, lambda other: set_halt_until(other, far, now=NOW - timedelta(hours=3)))
        builder = self.damaged_after_reading(store, daily_pnl=-2_500.0)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=builder) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "failing rule: halted" in lines
        assert "trading HALTED until 2026-08-03 13:30:00 UTC" in lines
        assert "  halt                   until 2026-08-03 13:30:00 UTC (active)" in lines
        controls = controls_of(store)
        assert controls.halt_until == far and controls.halt_unknown is False
        assert controls.problems == ()
        (trip,) = events_of(store, "risk_limit_tripped")
        assert trip.payload["halt_until"] == far.isoformat()
        # The rule line beside "trading HALTED until …" does not contradict it:
        # the halt is this evaluation's to write, not one "already in force".
        _, name, outcome, detail = rule_lines(out)[3]
        assert (name, outcome) == ("daily_loss_limit", "REJECT")
        assert detail.endswith(
            ": trading is halted until 2026-08-03T13:30:00+00:00, the later instant another"
            " reading of the stored halt gave (one reading of it was unreadable)"
        )
        assert "already in force" not in out
        assert trip.message == detail

    def test_the_halt_in_words(self):
        text = cli_policy._halt_text
        soon, past = NOW + timedelta(seconds=1), NOW - timedelta(hours=2)
        assert text(None, False, NOW) == "none"
        assert text(past, False, NOW) == "expired at 2026-07-30 13:00:00 UTC"
        assert text(NOW, False, NOW) == "expired at 2026-07-30 15:00:00 UTC"
        assert text(soon, True, NOW) == "until 2026-07-30 15:00:01 UTC (active)"
        assert text(None, True, NOW) == "UNKNOWN (cannot prove trading is not halted)"
        # In force with an instant that is not ahead: only an unreadable halt does that.
        for instant in (past, NOW):
            assert text(instant, True, NOW).startswith(
                "UNKNOWN (cannot prove trading is not halted); the last halt that could be read"
                " expired at "
            )
        # a naive instant is taken as UTC, as everywhere
        assert text(soon.replace(tzinfo=None), True, NOW) == text(soon, True, NOW)

    def test_the_connection_is_closed_when_the_evaluation_fails(self, store, opened, capsys):
        def broken(conn, proposal=None, *, config=None):
            raise RuntimeError("boom")

        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=broken) == 1
        assert capsys.readouterr().err == (
            "policy evaluate failed (unexpected): RuntimeError: boom\n"
        )
        assert_all_closed(opened)
        assert decisions_of(store, "prop-0001") == ()

    def test_recording_migrates_a_database_that_exists(self, store, capsys):
        drop_migration_0003(store)
        assert versions_of(store) == {1: MIGRATIONS[0], 2: MIGRATIONS[1], 4: MIGRATIONS[3]}
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        assert "verdict: AUTO_EXECUTE" in capsys.readouterr().out
        assert versions_of(store) == dict(enumerate(MIGRATIONS, start=1))
        assert len(decisions_of(store, "prop-0001")) == 1

    def test_a_verdict_that_could_not_be_recorded_is_not_reported(
        self, store, monkeypatch, capsys
    ):
        def failing(conn, decision, *, event=None):
            raise StoreError("record decision", decision.id, sqlite3.OperationalError("disk full"))

        monkeypatch.setattr(engine, "record_decision", failing)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # no verdict, no "recorded"
        assert captured.err.startswith(
            "policy evaluate failed: store operation failed: record decision for "
        )
        assert captured.err.endswith("(OperationalError: disk full)\n")
        assert captured.err.count("\n") == 1 and "Traceback" not in captured.err

    def test_a_random_sweep_always_ends_in_a_printed_verdict(self, db_path, capsys):
        """Hostile contexts (unknown, NaN and infinite figures, broken structures)
        never break the report: every case prints the engine's verdict and twenty-one rules."""
        rng = random.Random(20260930)
        cases = [random_case(rng, index) for index in range(200)]
        seed(db_path, *(proposal for proposal, _ in cases))
        verdicts = set()
        for proposal, context in cases:
            # Each case meets a store with no stop of its own: a halt the case
            # before left behind is cleared, as an operator clears one, so the
            # verdict printed is the context's.
            elsewhere(db_path, lambda other: set_halt_until(other, None, now=NOW))
            expected = engine.decide(proposal, context)
            assert cli_policy.main(["evaluate", proposal.id, "--db", str(db_path)],
                                   context_builder=fixed(context)) == 0
            captured = capsys.readouterr()
            assert captured.err == ""
            lines = captured.out.splitlines()
            assert [line for line in lines if line.startswith("verdict: ")] == [
                f"verdict: {expected.verdict.value}"
            ]
            assert [(name, outcome) for _, name, outcome, _ in rule_lines(captured.out)] == [
                (result.name, result.outcome.value) for result in expected.results
            ]
            assert ("failing rule: " + str(expected.failing_rule) in lines) is (
                expected.failing_rule is not None
            )
            assert not control_characters(captured.out)
            verdicts.add(expected.verdict)
        assert verdicts == set(Verdict)
        # The sweep stays inside what RiskLimits accepts (finite numbers) and
        # still prints its extremes: the largest finite float, a count beyond one.
        limits = [context.limits for _, context in cases]
        assert all(math.isfinite(drawn.max_loss_per_trade) for drawn in limits)
        assert any(drawn.max_loss_per_trade == sys.float_info.max for drawn in limits)
        assert any(drawn.auto_execute.max_notional == sys.float_info.max for drawn in limits)
        assert any(drawn.max_daily_trades > sys.float_info.max for drawn in limits)
        assert any(drawn.max_contracts == 0 for drawn in limits)
        # ... and judges shapes that contradict themselves, of both instruments.
        assert any(p.is_equity and structure_problems(p) for p, _ in cases)
        assert any(p.is_option and structure_problems(p) for p, _ in cases)

    def test_a_sweep_on_a_store_that_keeps_its_halts_prints_what_was_judged(
        self, db_path, capsys
    ):
        """Nothing cleared between the cases: once a trip has halted the
        store, every later report shows that halt and the verdict under it —
        ``decide`` on the context with the stricter controls."""
        rng = random.Random(20260930)
        cases = [random_case(rng, index) for index in range(140)]  # the first trip is the 99th
        seed(db_path, *(proposal for proposal, _ in cases))
        stopped = 0
        for proposal, context in cases:
            stored = controls_of(db_path)
            under = context.model_copy(
                update={"controls": engine.stricter_controls(context.controls, stored)}
            )
            expected = engine.decide(proposal, under)
            assert cli_policy.main(["evaluate", proposal.id, "--db", str(db_path)],
                                   context_builder=fixed(context)) == 0
            captured = capsys.readouterr()
            assert captured.err == ""
            lines = captured.out.splitlines()
            assert [line for line in lines if line.startswith("verdict: ")] == [
                f"verdict: {expected.verdict.value}"
            ]
            assert [(name, outcome) for _, name, outcome, _ in rule_lines(captured.out)] == [
                (result.name, result.outcome.value) for result in expected.results
            ]
            report = limits_report(under)
            halt = cli_policy._halt_text(report.halt_until, report.halted, report.as_of)
            assert f"  halt                   {halt}" in lines
            # ... in the three states' own words, worked out here from the controls:
            # an instant is "active" only while it is still ahead.
            until, unknown = under.controls.halt_until, under.controls.halt_unknown
            if until is not None and until > context.now:
                assert halt.startswith("until ") and halt.endswith(" (active)")
            elif unknown:
                assert halt.startswith("UNKNOWN (cannot prove trading is not halted)")
            else:
                assert halt == "none" or halt.startswith("expired at ")
            if stored.halt_until is not None and stored.halt_until > context.now:
                assert "verdict: REJECT" in lines
                stopped += engine.decide(proposal, context).verdict is not Verdict.REJECT
        assert stopped >= 5  # proposals their own context would have let through


# --- evaluate --dry-run -------------------------------------------------------


class TestDryRun:
    def test_prints_the_verdict_and_writes_nothing(self, store, opened, capsys):
        before_rows, before_bytes = every_row(store), store.read_bytes()
        builder = Builder(clock=make_clock(False))
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)],
                               context_builder=builder) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        lines = captured.out.splitlines()
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 4] == [
            "failing rule: market_hours",
            "DRY RUN — nothing was written",
            "",
        ]
        assert "recorded" not in captured.out
        assert [name for _, name, _, _ in rule_lines(captured.out)] == list(RULE_NAMES)
        assert "  market                 CLOSED (next open 2026-07-31 13:30:00 UTC)" in lines

        # no decision, no event, no halt — not one row, not one byte
        assert every_row(store) == before_rows
        assert store.read_bytes() == before_bytes
        assert decisions_of(store, "prop-0001") == () and events_of(store) == []
        # connect, never open_store: a dry run applies no migration
        assert [name for name, _ in opened] == ["connect"]
        assert_all_closed(opened)

    def test_latest_stays_the_latest(self, db_path, capsys):
        seed(db_path, make_proposal(), make_proposal(id="prop-0002", created_at=NOW))
        argv = ["evaluate", "--latest", "--dry-run", "--db", str(db_path)]
        for _ in range(2):  # nothing was decided, so the same proposal comes up again
            assert cli_policy.main(argv, context_builder=Builder()) == 0
            out = capsys.readouterr().out
            assert out.splitlines()[0] == "proposal prop-0002"
            assert "DRY RUN — nothing was written" in out.splitlines()

    def test_a_trip_is_announced_but_no_halt_is_written(self, store, capsys):
        before = every_row(store)
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        lines = capsys.readouterr().out.splitlines()
        position = lines.index("failing rule: daily_loss_limit")
        assert lines[position + 1 : position + 3] == [
            "DRY RUN — nothing was written",
            "trading would be HALTED until 2026-07-31 13:30:00 UTC"
            " (dry run: no halt was written)",
        ]
        assert not any(line.startswith("trading HALTED") for line in lines)
        assert every_row(store) == before
        assert controls_of(store).halt_until is None
        assert events_of(store, "risk_limit_tripped") == []

    def test_the_engine_is_asked_for_a_dry_run(self, store, monkeypatch, capsys):
        seen = []
        real = cli_policy.evaluate

        def spy(proposal, context, conn, *, dry_run=False):
            seen.append(dry_run)
            return real(proposal, context, conn, dry_run=dry_run)

        monkeypatch.setattr(cli_policy, "evaluate", spy)
        argv = ["evaluate", "prop-0001", "--db", str(store)]
        assert cli_policy.main([*argv, "--dry-run"], context_builder=Builder()) == 0
        assert cli_policy.main(argv, context_builder=Builder()) == 0
        assert seen == [True, False]

    def test_the_connection_refuses_every_write(self, store, capsys):
        """Defence in depth: on a dry run SQLite itself is told to refuse
        writes, so not even the context builder can change the store."""
        before = every_row(store)

        def writing(conn, proposal=None, *, config=None):
            set_kill_switch(conn, True, now=NOW)
            return context_for(proposal)

        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)],
                               context_builder=writing) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith(
            "policy evaluate failed: store operation failed: set kill switch for kill_switch"
        )
        assert "readonly" in captured.err and captured.err.count("\n") == 1
        assert every_row(store) == before
        # without --dry-run the same connection is an ordinary, writable one
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=writing) == 0
        assert controls_of(store).kill_switch is True

    def test_pending_migrations_are_refused_not_applied(self, store, capsys):
        drop_migration_0003(store)
        before = every_row(store)
        builder = Builder()
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)],
                               context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy evaluate failed: database {store} has pending migrations (0003_controls):"
            " run `python -m aegis.cli.db init` to apply them\n"
        )
        assert builder.calls == []
        assert every_row(store) == before and versions_of(store) == {
            1: MIGRATIONS[0], 2: MIGRATIONS[1], 4: MIGRATIONS[3],
        }

    def test_an_unmigrated_file_is_refused_and_left_empty(self, db_path, capsys):
        connect(db_path).close()  # the file exists, but nothing has been applied
        assert cli_policy.main(["evaluate", "--latest", "--dry-run", "--db", str(db_path)],
                               context_builder=Builder()) == 1
        err = capsys.readouterr().err
        assert f"has pending migrations ({', '.join(MIGRATIONS)})" in err
        assert every_row(db_path) == {} and versions_of(db_path) == {}


# --- what the commands do to the FILE -----------------------------------------

READ_ONLY = [
    pytest.param(["evaluate", "--latest", "--dry-run"], id="evaluate --latest --dry-run"),
    pytest.param(["evaluate", "prop-0001", "--dry-run"], id="evaluate ID --dry-run"),
    pytest.param(["kill", "status"], id="kill status"),
    pytest.param(["limits"], id="limits"),
]
"""The commands that write no row."""
RECORDING = [
    pytest.param(["evaluate", "--latest"], id="evaluate --latest"),
    pytest.param(["evaluate", "prop-0001"], id="evaluate ID"),
]
"""``evaluate`` without ``--dry-run``: it writes, and may migrate — an existing store."""


class TestTheFile:
    """Which files a command will open at all, and what it leaves of them."""

    @pytest.mark.parametrize("argv", [*READ_ONLY, *RECORDING])
    def test_an_empty_file_is_refused_and_left_empty(self, tmp_path, opened, capsys, argv):
        empty = tmp_path / "empty.db"
        empty.write_bytes(b"")
        builder = Builder()
        assert cli_policy.main([*argv, "--db", str(empty)], context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy {argv[0]} failed: database {empty} is an empty file, not a store"
            " (run `python -m aegis.cli.db init` to create it)\n"
        )
        # Not one byte: connecting alone would have written a database header.
        assert empty.read_bytes() == b"" and empty.stat().st_size == 0
        assert listing(tmp_path) == ["empty.db"]  # no -wal, no -shm beside it
        assert opened == [] and builder.calls == []  # it was never even opened

    def test_connecting_is_what_would_have_filled_it(self, tmp_path):
        # Why the check comes before the connection, not after.
        empty = tmp_path / "empty.db"
        empty.write_bytes(b"")
        connect(empty).close()
        assert empty.stat().st_size > 0

    @pytest.mark.parametrize("argv", RECORDING)
    def test_recording_refuses_a_file_with_no_schema_and_builds_none(
        self, db_path, opened, capsys, argv
    ):
        connect(db_path).close()  # a database file, but no migration was ever applied
        before = db_path.read_bytes()
        assert before and every_row(db_path) == {} and versions_of(db_path) == {}
        builder = Builder()
        assert cli_policy.main([*argv, "--db", str(db_path)], context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy evaluate failed: database {db_path} has no schema: no migration was ever"
            " applied (run `python -m aegis.cli.db init` to create it)\n"
        )
        # Still no table, no schema_version, the very same bytes.
        assert every_row(db_path) == {} and versions_of(db_path) == {}
        assert db_path.read_bytes() == before
        assert listing(db_path.parent) == [db_path.name]
        assert [name for name, _ in opened] == ["connect"]  # looked at, never open_store
        assert_all_closed(opened)
        assert builder.calls == []

    def test_recording_still_migrates_a_store_that_has_a_schema(self, store, opened, capsys):
        # The schema check refuses only a file with NO migration: an older store is brought up.
        drop_migration_0003(store)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        assert [name for name, _ in opened] == ["connect", "open_store"]
        assert versions_of(store) == dict(enumerate(MIGRATIONS, start=1))

    def test_kill_on_is_the_one_command_that_makes_a_store_of_an_empty_file(
        self, tmp_path, capsys
    ):
        empty = tmp_path / "empty.db"
        empty.write_bytes(b"")
        assert cli_policy.main(["kill", "on", "--db", str(empty)]) == 0
        assert versions_of(empty) == dict(enumerate(MIGRATIONS, start=1))
        assert controls_of(empty).kill_switch is True

    @pytest.mark.parametrize("argv", READ_ONLY)
    def test_a_read_only_command_leaves_a_store_byte_for_byte_unchanged(
        self, store, capsys, argv
    ):
        # A store with history: a halt, its event, a decision and its event.
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder(daily_pnl=-2_500.0)) == 0
        seed(store, make_proposal(id="prop-0002", created_at=NOW))
        capsys.readouterr()
        assert journal_mode_of(store) == "wal"
        before = (store.read_bytes(), every_row(store), listing(store.parent))
        assert before[1]["policy_decisions"] and before[1]["events"]

        assert cli_policy.main([*argv, "--db", str(store)], context_builder=Builder()) == 0

        assert capsys.readouterr().err == ""
        assert (store.read_bytes(), every_row(store), listing(store.parent)) == before
        assert journal_mode_of(store) == "wal"

    @pytest.mark.parametrize("argv", READ_ONLY)
    def test_a_store_that_is_not_in_wal_mode_keeps_every_row(self, store, capsys, argv):
        """What "writes nothing" means at the file level, as the module
        docstring says it: no ROW of any table changes. ``connect`` — how every
        reader of the store opens it — does switch a file that is not in WAL
        journal mode (a copy made in DELETE mode, a restored backup) to WAL."""
        raw = sqlite3.connect(store)
        raw.execute("PRAGMA journal_mode=DELETE")
        raw.close()
        assert journal_mode_of(store) == "delete"
        rows = every_row(store)

        assert cli_policy.main([*argv, "--db", str(store)], context_builder=Builder()) == 0

        assert capsys.readouterr().err == ""
        assert every_row(store) == rows  # not a row, in any table
        assert decisions_of(store, "prop-0001") == () and events_of(store) == []
        assert journal_mode_of(store) == "wal"  # the documented file-level effect
        said = " ".join(cli_policy.__doc__.split())
        assert "Those three write no row." in said
        assert "is switched to WAL" in said and "Neither changes a row of any table." in said


# --- kill ---------------------------------------------------------------------


class TestKill:
    def test_on_and_off_persist_and_log_an_event_each_time(self, store, at_now, capsys):
        argv = ["--db", str(store)]
        expected = [  # (command, state after, level, previous)
            ("on", True, EventLevel.WARNING, "off"),
            ("on", True, EventLevel.WARNING, "on"),  # logged even when nothing changes
            ("off", False, EventLevel.INFO, "on"),
            ("off", False, EventLevel.INFO, "off"),
        ]
        for count, (command, state, level, previous) in enumerate(expected, start=1):
            assert cli_policy.main(["kill", command, *argv]) == 0
            captured = capsys.readouterr()
            assert captured.err == ""
            shown = "ON — every proposal is rejected" if state else "off"
            # spec D4: kill on says the switch does not cancel working orders
            hint = [cli_policy.KILL_HINT.format(working=0)] if state else []
            assert captured.out.splitlines() == [
                f"kill switch: {shown} (was {previous})",
                "  last set 2026-07-30 15:00:00 UTC",
                "halt: none",
                f"  last set {cli_policy._ts(controls_of(store).halt_until_updated_at)}",
                "problems (0)",
                "  (none)",
                *hint,
            ]
            controls = controls_of(store)
            assert controls.kill_switch is state and controls.kill_switch_updated_at == NOW
            logged = events_of(store)
            assert len(logged) == count and {event.kind for event in logged} == {"kill_switch"}
            event = logged[-1]
            assert event.level is level and event.occurred_at == NOW
            assert event.payload == {"state": command, "previous": previous, "source": "cli"}
            assert event.message == (
                f"kill switch set {command.upper()} from the CLI (was {previous})"
            )
        assert cli_policy.KILL_SWITCH_EVENT == "kill_switch"

    def test_the_switch_and_its_event_land_together(self, store, at_now, monkeypatch, capsys):
        seen = []
        real = cli_policy.set_kill_switch

        def spy(conn, on, *, now=None, event=None):
            seen.append((on, now, event))
            return real(conn, on, now=now, event=event)

        monkeypatch.setattr(cli_policy, "set_kill_switch", spy)
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 0
        ((on, now, event),) = seen
        assert on is True and now == NOW  # a real bool, never the string "on"
        assert event is not None and event.kind == "kill_switch"  # one transaction with the row

    def test_on_then_every_proposal_is_rejected(self, store, capsys):
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 0
        assert cli_policy.main(["evaluate", "--latest", "--db", str(store)],
                               context_builder=Builder()) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "verdict: REJECT" in lines and "failing rule: kill_switch" in lines
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=Builder()) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "trading blocked by: kill_switch" in lines
        assert "  kill switch   ON — every proposal is rejected" in lines

    def test_on_opens_the_store_as_the_loop_does(self, tmp_path, opened, capsys):
        """The one command that may create a database: the operator's stop
        must work on a fresh deployment too."""
        fresh = tmp_path / "fresh.db"
        assert cli_policy.main(["kill", "on", "--db", str(fresh)]) == 0
        assert [name for name, _ in opened] == ["open_store"]
        assert_all_closed(opened)
        assert versions_of(fresh) == dict(enumerate(MIGRATIONS, start=1))
        assert controls_of(fresh).kill_switch is True

    def test_set_needs_no_config(self, store, monkeypatch, capsys):
        """A broken config.yaml must not stand between the operator and the kill switch."""

        def broken():
            raise ConfigError("config file is not valid YAML")

        monkeypatch.setattr(cli_policy, "get_config", broken)
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 0
        assert cli_policy.main(["kill", "status", "--db", str(store)]) == 0
        assert controls_of(store).kill_switch is True
        capsys.readouterr()
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=Builder()) == 1
        assert capsys.readouterr().err == (
            "policy limits failed: config file is not valid YAML\n"
        )

    def test_status_prints_the_switch_and_writes_nothing(self, store, at_now, opened, capsys):
        conn = open_store(store)
        set_kill_switch(conn, True, now=NOW - timedelta(hours=1))
        conn.close()
        before_rows, before_bytes = every_row(store), store.read_bytes()
        assert cli_policy.main(["kill", "status", "--db", str(store)]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out.splitlines()[:3] == [
            "kill switch: ON — every proposal is rejected",
            "  last set 2026-07-30 14:00:00 UTC",
            "halt: none",
        ]
        assert captured.out.splitlines()[4:] == ["problems (0)", "  (none)"]
        assert every_row(store) == before_rows and store.read_bytes() == before_bytes
        assert events_of(store) == []  # reading the switch is not an event
        assert [name for name, _ in opened] == ["connect"]
        assert_all_closed(opened)

    @pytest.mark.parametrize(
        "until, shown",
        [
            (None, "halt: none"),
            (NOW + timedelta(seconds=1), "halt: until 2026-07-30 15:00:01 UTC (active)"),
            (NOW, "halt: expired at 2026-07-30 15:00:00 UTC"),  # until == now: over, as the rule
            (NOW - timedelta(days=1), "halt: expired at 2026-07-29 15:00:00 UTC"),
            (NEXT_OPEN, "halt: until 2026-07-31 13:30:00 UTC (active)"),
        ],
        ids=["none", "one-second-left", "ends-now", "long-over", "until-the-open"],
    )
    def test_status_prints_the_halt(self, store, at_now, capsys, until, shown):
        conn = open_store(store)
        set_halt_until(conn, until, now=NOW - timedelta(minutes=5))
        conn.close()
        assert cli_policy.main(["kill", "status", "--db", str(store)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[2:4] == [shown, "  last set 2026-07-30 14:55:00 UTC"]
        assert lines[0] == "kill switch: off"

    def test_status_shows_what_the_store_could_not_read(self, store, at_now, capsys):
        loosen_controls(store, [("halt_until", "tomorrow\x1b[2J", "not a time")])
        assert cli_policy.main(["kill", "status", "--db", str(store)]) == 0  # still exit 0
        captured = capsys.readouterr()
        assert captured.out.splitlines() == [
            "kill switch: ON — every proposal is rejected",  # fails closed
            "  last set unknown",
            "halt: UNKNOWN (cannot prove trading is not halted)",
            "  last set unknown",
            "problems (2)",
            "  - kill_switch is missing from the controls table: reading the kill switch as ON",
            "  - halt_until value 'tomorrow\\x1b[2J' is neither empty nor an ISO-8601 timestamp:"
            " cannot prove trading is not halted",
        ]
        assert not control_characters(captured.out)

    def test_off_restores_a_missing_row(self, store, at_now, capsys):
        loosen_controls(store, [("halt_until", "", NOW.isoformat())])
        assert cli_policy.main(["kill", "off", "--db", str(store)]) == 0
        assert capsys.readouterr().out.splitlines()[0] == "kill switch: off (was on)"
        (event,) = events_of(store)
        assert event.payload["previous"] == "on"  # an unreadable switch reads as ON
        assert controls_of(store) == Controls(
            kill_switch=False, kill_switch_updated_at=NOW, halt_until_updated_at=NOW
        )

    def test_a_switch_that_still_reads_on_is_a_failure(self, store, at_now, capsys):
        """The state printed is the one read back. A store that reads ON
        whatever was written (a control key this code does not know) must
        not let ``kill off`` report success."""
        rows = [
            ("kill_switch", "on", NOW.isoformat()),
            ("halt_until", "", NOW.isoformat()),
            ("panic", "yes", NOW.isoformat()),
        ]
        loosen_controls(store, rows)
        assert cli_policy.main(["kill", "off", "--db", str(store)]) == 1
        captured = capsys.readouterr()
        assert captured.out.splitlines()[0] == (
            "kill switch: ON — every proposal is rejected (was on)"
        )
        assert captured.err == (
            "policy kill failed: the kill switch was set off but reads ON: unrecognised control"
            " key 'panic': reading the kill switch as ON\n"
        )
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 0  # ON is what it reads

    def test_status_never_creates_a_database(self, tmp_path, capsys):
        missing = tmp_path / "nested" / "policy.db"
        assert cli_policy.main(["kill", "status", "--db", str(missing)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy kill failed: database {missing} does not exist"
            " (run `python -m aegis.cli.db init` to create it)\n"
        )
        assert not missing.exists() and not missing.parent.exists()

    def test_status_never_migrates(self, store, capsys):
        drop_migration_0003(store)
        before = every_row(store)
        assert cli_policy.main(["kill", "status", "--db", str(store)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy kill failed: database {store} has pending migrations (0003_controls):"
            " run `python -m aegis.cli.db init` to apply them\n"
        )
        assert every_row(store) == before
        # kill on, though, brings the store up to date and sets the switch
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 0
        assert versions_of(store) == dict(enumerate(MIGRATIONS, start=1))
        assert controls_of(store).kill_switch is True

    def test_a_failed_write_is_one_clean_line(self, store, monkeypatch, capsys):
        def failing(conn, on, *, now=None, event=None):
            raise StoreError("set kill switch", "kill_switch", sqlite3.OperationalError("locked"))

        monkeypatch.setattr(cli_policy, "set_kill_switch", failing)
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""  # never "kill switch: ON" for a switch that was not set
        assert captured.err == (
            "policy kill failed: store operation failed: set kill switch for kill_switch"
            " (OperationalError: locked)\n"
        )
        assert controls_of(store).kill_switch is False and events_of(store) == []


# --- limits -------------------------------------------------------------------


class TestLimits:
    def test_prints_each_limit_beside_its_consumption_and_headroom(self, store, opened, capsys):
        builder = Builder(
            positions=(make_position(),),
            daily_pnl=-500.0,
            orders_today=3,
            errors=("quote for AAPL: failed to fetch quote (ConnectionError: down)",),
        )
        before_rows, before_bytes = every_row(store), store.read_bytes()
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == LIMITS_REPORT

        # a report: nothing written, nothing migrated, the builder given no proposal
        assert every_row(store) == before_rows and store.read_bytes() == before_bytes
        assert [name for name, _ in opened] == ["connect"]
        assert_all_closed(opened)
        ((args, kwargs),) = builder.calls
        assert len(args) == 1 and isinstance(args[0], sqlite3.Connection)
        assert list(kwargs) == ["config"] and kwargs["config"] is get_config()

    def test_the_report_is_the_single_source(self, store, monkeypatch, capsys):
        """The CLI formats what ``limits_report`` returns and computes nothing
        itself: a doctored report is printed as it stands."""
        context = make_context()
        seen = []

        def doctored(given):
            seen.append(given)
            real = limits_report(given)
            # Numbers that CONTRADICT each other, so printing cannot be told
            # from recomputing by luck: a headroom that is not limit - used, a
            # line over its limit that is not marked, one under it that is.
            loss = real.lines[0].model_copy(
                update={"limit": 1_234.5, "used": 77.0, "headroom": 9.99, "breached": True}
            )
            positions = real.lines[1].model_copy(
                update={"limit": 5.0, "used": 9.0, "headroom": 1.0, "breached": False}
            )
            trades = real.lines[2].model_copy(
                update={"limit": 10.0, "used": 3.0, "headroom": 42.0, "breached": True}
            )
            return real.model_copy(
                update={
                    "equity": 123_456.78,
                    "daily_pnl": -1.0,
                    "blocked_by": ("some_rule",),
                    "lines": (loss, positions, trades, *real.lines[3:]),
                    "exposures": (
                        ExposureLine(
                            underlying="ZZZ", exposure=1.0, cap=2.0, headroom=123.45, breached=True
                        ),
                        ExposureLine(
                            underlying="YYY", exposure=9.0, cap=2.0, headroom=-7.0, breached=False
                        ),
                    ),
                    "auto_execute_enabled": False,
                    "auto_execute_max_notional": 42.0,
                }
            )

        monkeypatch.setattr(cli_policy, "limits_report", doctored)
        assert cli_policy.main(["limits", "--db", str(store)],
                               context_builder=fixed(context)) == 0
        out = capsys.readouterr().out
        assert seen == [context]
        assert "  equity                 $123,456.78" in out.splitlines()
        assert "  today's P&L            -$1.00" in out.splitlines()
        assert "trading blocked by: some_rule" in out.splitlines()
        assert cells(out, "daily_loss") == [
            "daily_loss", "$1,234.50", "$77.00", "$9.99", "USD", "BREACHED",
        ]
        assert cells(out, "open_positions") == ["open_positions", "5", "9", "1", "positions"]
        assert cells(out, "daily_trades") == ["daily_trades", "10", "3", "42", "trades", "BREACHED"]
        assert "exposure by underlying (2)" in out.splitlines()
        assert cells(out, "ZZZ") == ["ZZZ", "$1.00", "$2.00", "$123.45", "BREACHED"]
        assert cells(out, "YYY") == ["YYY", "$9.00", "$2.00", "-$7.00"]
        assert out.count("BREACHED") == 3
        assert "  enabled        no" in out.splitlines()
        assert "  max notional   $42.00" in out.splitlines()

    def test_evaluates_context_block_is_the_reports_too(self, store, monkeypatch, capsys):
        """The state and account lines under ``context`` come out of the same
        report — printed, not recomputed from the context."""
        def doctored(given):
            return limits_report(given).model_copy(
                update={
                    "equity": 1.23, "start_of_day_equity": 4.56, "daily_pnl": 7.89,
                    "buying_power": 0.12, "options_buying_power": 3.45,
                    "kill_switch": True, "halted": True, "halt_until": None,
                    "market_open": None, "errors": ("a doctored line",),
                }
            )

        monkeypatch.setattr(cli_policy, "limits_report", doctored)
        assert cli_policy.main(["evaluate", "prop-0001", "--dry-run", "--db", str(store)],
                               context_builder=Builder()) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[lines.index("context") + 2 : lines.index("context") + 10] == [
            "  market                 UNKNOWN (no market clock)",
            "  kill switch            ON — every proposal is rejected",
            "  halt                   UNKNOWN (cannot prove trading is not halted)",
            "  equity                 $1.23",
            "  start-of-day equity    $4.56",
            "  today's P&L            $7.89",
            "  buying power           $0.12",
            "  options buying power   $3.45",
        ]
        assert lines[-2:] == ["  data problems (1)", "    - a doctored line"]
        assert "verdict: AUTO_EXECUTE" in lines  # the verdict is the engine's, not the report's

    def test_a_winning_day_shows_more_headroom_than_the_cap(self, store, capsys):
        # Up 500 on the day: nothing of the 2,000 cap is used, and the
        # distance to it is 2,500 — not limit minus used.
        builder = Builder(daily_pnl=500.0)
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 0
        out = capsys.readouterr().out
        assert cells(out, "daily_loss") == [
            "daily_loss", "$2,000.00", "$0.00", "$2,500.00", "USD",
        ]
        assert "  today's P&L            $500.00" in out.splitlines()

    def test_breached_limits_are_marked(self, store, capsys):
        builder = Builder(
            daily_pnl=-2_000.0,  # at the cap: tripped
            orders_today=10,  # at the cap
            positions=tuple(
                make_position(symbol, market_value=value)
                for symbol, value in [
                    ("AAPL", 5_000.01),  # one cent over the 5,000 cap
                    ("MSFT", 5_000.0),  # at the cap: not over
                    ("NVDA", 100.0),
                    ("QQQ", 100.0),
                    ("SPY", 100.0),
                ]
            ),
        )
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 0
        out = capsys.readouterr().out
        assert cells(out, "daily_loss") == [
            "daily_loss", "$2,000.00", "$2,000.00", "$0.00", "USD", "BREACHED",
        ]
        assert cells(out, "open_positions") == [
            "open_positions", "5", "5", "0", "positions", "BREACHED",
        ]
        assert cells(out, "daily_trades") == ["daily_trades", "10", "10", "0", "trades", "BREACHED"]
        assert cells(out, "buying_power") == ["buying_power", "-", "-", "$200,000.00", "USD"]
        assert "exposure by underlying (5)" in out.splitlines()
        assert cells(out, "AAPL") == ["AAPL", "$5,000.01", "$5,000.00", "-$0.01", "BREACHED"]
        assert cells(out, "MSFT") == ["MSFT", "$5,000.00", "$5,000.00", "$0.00"]
        assert cells(out, "SPY") == ["SPY", "$100.00", "$5,000.00", "$4,900.00"]
        assert out.count("BREACHED") == 4
        assert "trading blocked by: daily_loss_limit, max_daily_trades" in out.splitlines()

    def test_one_under_each_cap_is_not_breached(self, store, capsys):
        builder = Builder(daily_pnl=-1_999.99, orders_today=9, positions=(make_position(),))
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 0
        out = capsys.readouterr().out
        assert cells(out, "daily_loss") == ["daily_loss", "$2,000.00", "$1,999.99", "$0.01", "USD"]
        assert cells(out, "daily_trades") == ["daily_trades", "10", "9", "1", "trades"]
        assert "BREACHED" not in out and "no account-level block" in out.splitlines()

    @pytest.mark.parametrize(
        "overrides, blocked",
        [
            ({}, None),
            ({"controls": Controls(kill_switch=True)}, "kill_switch"),
            ({"controls": Controls(halt_until=NEXT_OPEN)}, "halted"),
            ({"controls": Controls(halt_unknown=True)}, "halted"),
            ({"clock": make_clock(False)}, "market_hours"),
            ({"clock": None}, "market_hours"),
            ({"daily_pnl": -2_500.0}, "daily_loss_limit"),
            ({"orders_today": 10}, "max_daily_trades"),
            (
                {
                    "controls": Controls(kill_switch=True, halt_until=NEXT_OPEN),
                    "clock": make_clock(False),
                    "daily_pnl": None,
                    "orders_today": 11,
                },
                "kill_switch, halted, market_hours, daily_loss_limit, max_daily_trades",
            ),
        ],
        ids=[
            "nothing", "kill-switch", "halt", "unreadable-halt", "closed", "no-clock",
            "loss-cap", "trade-cap", "everything",
        ],
    )
    def test_blocked_by(self, store, capsys, overrides, blocked):
        assert cli_policy.main(["limits", "--db", str(store)],
                               context_builder=Builder(**overrides)) == 0
        lines = capsys.readouterr().out.splitlines()
        expected = "no account-level block" if blocked is None else f"trading blocked by: {blocked}"
        assert lines[4] == expected
        verdicts = [line for line in lines if line.startswith(("trading blocked", "no account"))]
        assert verdicts == [expected]

    @pytest.mark.parametrize(
        "overrides, market, halt",
        [
            ({}, "OPEN (next close 2026-07-30 20:00:00 UTC)", "none"),
            (
                {"clock": make_clock(False)},
                "CLOSED (next open 2026-07-31 13:30:00 UTC)",
                "none",
            ),
            ({"clock": None}, "UNKNOWN (no market clock)", "none"),
            (
                {"clock": make_clock(False, next_open=None, next_close=None)},
                "CLOSED (next open unknown)",
                "none",
            ),
            (  # the clock says open, but its close is not after now: the rule rejects
                {"clock": make_clock(next_close=NOW)},
                "OPEN per a STALE clock (its close 2026-07-30 15:00:00 UTC has passed)",
                "none",
            ),
            (
                {"clock": make_clock(next_close=NOW + timedelta(seconds=1))},
                "OPEN (next close 2026-07-30 15:00:01 UTC)",
                "none",
            ),
            (
                {"controls": Controls(halt_until=NOW + timedelta(seconds=1))},
                "OPEN (next close 2026-07-30 20:00:00 UTC)",
                "until 2026-07-30 15:00:01 UTC (active)",
            ),
            (
                {"controls": Controls(halt_until=NOW)},
                "OPEN (next close 2026-07-30 20:00:00 UTC)",
                "expired at 2026-07-30 15:00:00 UTC",
            ),
            (
                {"controls": Controls(halt_unknown=True)},
                "OPEN (next close 2026-07-30 20:00:00 UTC)",
                "UNKNOWN (cannot prove trading is not halted)",
            ),
        ],
        ids=[
            "open", "closed", "no-clock", "no-times", "stale-clock", "closing-in-a-second",
            "halted", "halt-over", "halt-unknown",
        ],
    )
    def test_market_and_halt_state(self, store, capsys, overrides, market, halt):
        assert cli_policy.main(["limits", "--db", str(store)],
                               context_builder=Builder(**overrides)) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[:4] == [
            "limits as of 2026-07-30 15:00:00 UTC",
            f"  market        {market}",
            "  kill switch   off",
            f"  halt          {halt}",
        ]

    def test_the_rule_name_the_market_line_reads_is_the_registered_one(self):
        assert cli_policy.MARKET_HOURS_RULE in RULE_NAMES
        report = limits_report(make_context(clock=make_clock(next_close=NOW)))
        assert report.market_open is True and report.blocked_by == (cli_policy.MARKET_HOURS_RULE,)

    def test_unknowns_are_never_printed_as_zero(self, store, capsys):
        builder = Builder(
            equity=None,
            start_of_day_equity=float("nan"),
            daily_pnl=None,
            buying_power=float("inf"),
            options_buying_power=None,
            cash=None,
            positions=None,
            errors=("account state: failed to fetch account (ConnectionError: down)",),
        )
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        for label in ("equity", "start-of-day equity", "today's P&L", "buying power",
                      "options buying power"):
            assert f"  {label.ljust(20)}   n/a" in lines
        assert cells(out, "daily_loss") == ["daily_loss", "-", "-", "-", "USD"]
        assert cells(out, "open_positions") == ["open_positions", "5", "-", "-", "positions"]
        assert cells(out, "buying_power") == ["buying_power", "-", "-", "-", "USD"]
        assert (
            "      cannot count open positions: positions are unknown"
            " (risk_limits.max_open_positions)"
        ) in lines
        assert "exposure by underlying (0)" in lines
        assert lines[lines.index("exposure by underlying (0)") + 1] == "  (none)"
        assert "trading blocked by: daily_loss_limit" in lines
        assert lines[-2:] == [
            "data problems (1)",
            "  - account state: failed to fetch account (ConnectionError: down)",
        ]
        assert "$0.00" not in out and "BREACHED" not in out

    def test_an_exposure_that_cannot_be_valued(self, store, capsys):
        position = make_position("MSFT", current_price=None, market_value=None)
        assert cli_policy.main(["limits", "--db", str(store)],
                               context_builder=Builder(positions=(position,))) == 0
        assert cells(capsys.readouterr().out, "MSFT") == ["MSFT", "-", "$5,000.00", "-"]

    def test_never_creates_or_migrates_a_database(self, tmp_path, store, capsys):
        missing = tmp_path / "nested" / "policy.db"
        builder = Builder()
        assert cli_policy.main(["limits", "--db", str(missing)], context_builder=builder) == 1
        assert capsys.readouterr().err == (
            f"policy limits failed: database {missing} does not exist"
            " (run `python -m aegis.cli.db init` to create it)\n"
        )
        assert not missing.exists() and not missing.parent.exists()

        drop_migration_0003(store)
        before = every_row(store)
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy limits failed: database {store} has pending migrations (0003_controls):"
            " run `python -m aegis.cli.db init` to apply them\n"
        )
        assert every_row(store) == before and builder.calls == []

    def test_the_connection_refuses_every_write(self, store, capsys):
        def writing(conn, proposal=None, *, config=None):
            conn.execute("DELETE FROM proposals")
            return make_context()

        before = every_row(store)
        assert cli_policy.main(["limits", "--db", str(store)], context_builder=writing) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("policy limits failed (unexpected): OperationalError: ")
        assert every_row(store) == before

    def test_a_random_sweep_of_contexts_always_prints(self, store, capsys):
        rng = random.Random(5)
        unlimited_counts = largest_tiers = 0
        for _ in range(150):
            context = random_context(rng, turbulence=rng.choice((0.1, 0.5, 1.0)))
            report = limits_report(context)
            # Every limit is finite (RiskLimits refuses the rest) — the sweep
            # still reaches a count no float can hold and the largest finite float.
            assert math.isfinite(report.auto_execute_max_notional)
            unlimited_counts += context.limits.max_daily_trades > sys.float_info.max
            largest_tiers += report.auto_execute_max_notional == sys.float_info.max
            assert cli_policy.main(["limits", "--db", str(store)],
                                   context_builder=fixed(context)) == 0
            captured = capsys.readouterr()
            assert captured.err == "" and not control_characters(captured.out)
            lines = captured.out.splitlines()
            assert f"limits ({len(report.lines)})" in lines
            assert f"exposure by underlying ({len(report.exposures)})" in lines
            assert f"data problems ({len(report.errors)})" in lines
            # BREACHED is printed exactly where the report says so
            breached = [line.name for line in report.lines if line.breached]
            breached += [line.underlying for line in report.exposures if line.breached]
            marked = [line.split()[0] for line in lines if line.endswith("BREACHED")]
            assert marked == breached
            expected = (
                f"trading blocked by: {', '.join(report.blocked_by)}"
                if report.blocked_by
                else "no account-level block"
            )
            assert lines[4] == expected
            if context.limits.max_daily_trades > sys.float_info.max:
                assert cells(captured.out, "daily_trades")[1] == "unlimited"
        assert unlimited_counts >= 5 and largest_tiers >= 5


# --- what reaches the terminal ------------------------------------------------

ESC = "\x1b"
EVIL_SYMBOL = f"ZZ{ESC}[2J{ESC}[31mTOP"
EVIL_ID = "prop-evil\nverdict: AUTO_EXECUTE\r\x07"
EVIL_ERROR = (
    f"quote for X: down{ESC}[0m\nverdict: AUTO_EXECUTE\n  [ 1] kill_switch   PASS   forged‮"
)


class TestOutputIsClean:
    def test_stored_text_cannot_restyle_the_terminal_or_forge_a_line(self, db_path, capsys):
        legs = [make_leg("buy", "call", 640.0, proposal_id=EVIL_ID, symbol=f"SPY{ESC}]0;pwned\x07")]
        proposal = make_proposal(
            id=EVIL_ID, symbol=EVIL_SYMBOL, instrument=Instrument.OPTION, legs=legs
        )
        seed(db_path, proposal)
        context = make_context(
            errors=(EVIL_ERROR,),
            controls=Controls(halt_unknown=True, problems=(f"halt_until is{ESC}[5m broken\n",)),
        )
        assert cli_policy.main(["evaluate", "--latest", "--db", str(db_path)],
                               context_builder=fixed(context)) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        out = captured.out
        assert not control_characters(out)
        lines = out.splitlines()
        # every stored field sits on its own, single line
        assert lines[0] == "proposal prop-evil verdict: AUTO_EXECUTE"
        assert lines[2] == "  symbol       ZZ [2J [31MTOP (option)"
        assert lines[6] == "    [0] buy 1 call 640.0 exp 2026-08-21  SPY ]0;PWNED"
        assert [line for line in lines if line.startswith("verdict:")] == ["verdict: REJECT"]
        printed = rule_lines(out)
        assert [name for _, name, _, _ in printed] == list(RULE_NAMES)  # twenty-one, none forged
        assert sum(line.startswith("  [") for line in lines) == 21
        (decision,) = decisions_of(db_path, EVIL_ID)
        # the details as recorded carry the raw symbol; the printed ones do not
        assert any(ESC in rule["detail"] for rule in decision.rules_evaluated)
        assert any("ZZ [2J [31MTOP" in detail for _, _, _, detail in printed)
        assert lines[-2:] == [
            "  data problems (1)",
            "    - quote for X: down [0m verdict: AUTO_EXECUTE [ 1] kill_switch PASS forged",
        ]

    def test_the_recorded_rules_are_printed_clean_whatever_they_hold(
        self, store, monkeypatch, capsys
    ):
        """The rule lines are the decision's own ``rules_evaluated`` — free-form
        JSON in the store — so every field of them goes through the cleaner."""
        rules = [
            {"rule": f"evil{ESC}[2J\nrule", "outcome": "PA\x07SS", "detail": "one\ntwo"},
            {},  # a row with no keys at all still prints one line
        ]

        def doctored(proposal, context, conn, *, dry_run=False):
            return PolicyDecision(
                id=f"dec{ESC}[0m\nision",
                proposal_id=proposal.id,
                decided_at=NOW,
                verdict=Verdict.REJECT,
                rules_evaluated=rules,
                failing_rule=f"evil{ESC}[2J\nrule",
            )

        monkeypatch.setattr(cli_policy, "evaluate", doctored)
        assert cli_policy.main(["evaluate", "prop-0001", "--db", str(store)],
                               context_builder=Builder()) == 0
        out = capsys.readouterr().out
        assert not control_characters(out)
        lines = out.splitlines()
        position = lines.index("verdict: REJECT")
        assert lines[position + 1 : position + 8] == [
            "failing rule: evil [2J rule",
            "decision dec [0m ision recorded",
            # These rules are no reading of the store's stop flags: the report says
            # that its context block is not the context they were judged under.
            cli_policy.STALE_CONTEXT_NOTE,
            "",
            "rules (2)",
            "  [1] evil [2J rule   PA SS      one two",
            "  [2] (unnamed)       ?",
        ]

    def test_limits_output_is_clean_too(self, store, capsys):
        context = make_context(
            positions=(make_position(f"MS{ESC}[2JFT\n  SPY   $0.00"),),
            errors=(EVIL_ERROR,),
        )
        assert cli_policy.main(["limits", "--db", str(store)],
                               context_builder=fixed(context)) == 0
        out = capsys.readouterr().out
        assert not control_characters(out)
        lines = out.splitlines()
        assert "exposure by underlying (1)" in lines
        row = lines[lines.index("exposure by underlying (1)") + 2]
        assert row.startswith("  MS [2JFT SPY $0.00   $2,000.00")
        assert len(lines) == len(LIMITS_REPORT.splitlines())  # no line was added by the text

    def test_an_error_message_is_one_clean_line(self, store, capsys):
        assert cli_policy.main(["evaluate", f"no{ESC}[2Jpe\nfake line", "--db", str(store)],
                               context_builder=Builder()) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and not control_characters(captured.err)
        assert captured.err == (
            "policy evaluate failed: policy failed: load proposal (no such proposal)"
            " for no [2Jpe fake line\n"
        )

    @pytest.mark.parametrize(
        ("argv", "shown"),
        [
            (
                ["evaluate", "prop-1", f"{ESC}[2J{ESC}[31mFORGED\nverdict: AUTO_EXECUTE"],
                "python -m aegis.cli.policy: error: unrecognized arguments:"
                " [2J [31mFORGED verdict: AUTO_EXECUTE (see --help)\n",
            ),
            (
                ["limits", f"--db{ESC}[2J\rverdict: AUTO_EXECUTE\x07"],
                "python -m aegis.cli.policy: error: unrecognized arguments:"
                " --db [2J verdict: AUTO_EXECUTE (see --help)\n",
            ),
            (
                ["kill", "status", "\u202eforged\u200b\n\nkill switch: off"],
                "python -m aegis.cli.policy: error: unrecognized arguments:"
                " forged kill switch: off (see --help)\n",
            ),
        ],
        ids=["escape-and-newline", "carriage-return-and-bell", "format-characters"],
    )
    def test_a_usage_error_is_one_clean_line(self, capsys, argv, shown):
        """What the user typed is echoed back by argparse: it goes through
        the cleaner like everything else."""
        assert cli_policy.main(argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.count("\n") == 1 and not control_characters(captured.err)
        assert captured.err == shown

    def test_a_character_stdout_cannot_encode_prints_as_an_escape(self, store):
        """The em dash of the dry-run note under an ASCII terminal: an escape, not a crash."""
        script = (
            "import sys\n"
            "from aegis.cli import policy\n"
            "sys.exit(policy.main(['kill', 'on', '--db', sys.argv[1]]))\n"
        )
        env = {**os.environ, "PYTHONIOENCODING": "ascii", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-c", script, str(store)],
            cwd=REPO_ROOT, capture_output=True, env=env, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == (
            b"kill switch: ON \\u2014 every proposal is rejected (was off)"
        )
        assert controls_of(store).kill_switch is True


# --- the command line ---------------------------------------------------------


class TestCommandLine:
    @pytest.mark.parametrize(
        "argv, message",
        [
            (
                ["evaluate", "prop-0001", "--latest"],
                "python -m aegis.cli.policy evaluate: error: give a PROPOSAL_ID or --latest,"
                " not both",
            ),
            (
                ["evaluate"],
                "python -m aegis.cli.policy evaluate: error: give a PROPOSAL_ID, or --latest"
                " for the newest undecided proposal",
            ),
            (
                ["evaluate", "--dry-run"],
                "python -m aegis.cli.policy evaluate: error: give a PROPOSAL_ID, or --latest",
            ),
            (
                ["evaluate", "  "],
                "python -m aegis.cli.policy evaluate: error: PROPOSAL_ID must not be blank",
            ),
            (
                ["evaluate", "a", "b"],
                "python -m aegis.cli.policy: error: unrecognized arguments: b",
            ),
            (
                ["kill", "maybe"],
                "python -m aegis.cli.policy kill: error: argument state: invalid choice: 'maybe'",
            ),
            (
                ["kill"],
                "python -m aegis.cli.policy kill: error: the following arguments are required:"
                " state",
            ),
            (["limits", "--nope"], "python -m aegis.cli.policy: error: unrecognized arguments"),
            (["approve"], "python -m aegis.cli.policy: error: argument command: invalid choice"),
            ([], "python -m aegis.cli.policy: error: the following arguments are required"),
        ],
        ids=[
            "both-targets", "no-target", "no-target-dry", "blank-id", "two-ids", "bad-state",
            "no-state", "unknown-flag", "unknown-command", "no-command",
        ],
    )
    def test_a_bad_command_line_is_one_line_and_exit_1(self, store, capsys, argv, message):
        before = every_row(store)
        builder = Builder()
        argv = [*argv, "--db", str(store)] if argv else argv  # --db belongs to a command
        assert cli_policy.main(argv, context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith(message)
        assert captured.err.count("\n") == 1 and captured.err.endswith("(see --help)\n")
        assert builder.calls == [] and every_row(store) == before

    @pytest.mark.parametrize(
        "argv, expected",
        [
            (
                ["--help"],
                [
                    "usage: python -m aegis.cli.policy [-h] {evaluate,kill,limits} ...",
                    "Run the AEGIS policy engine: judge a proposal, set the kill switch, or"
                    " print the limits.",
                    "evaluate judge one stored proposal and record the verdict",
                    "kill set or show the kill switch (on rejects every proposal)",
                    "limits print each limit beside its consumption and headroom",
                ],
            ),
            (
                ["evaluate", "--help"],
                [
                    "usage: python -m aegis.cli.policy evaluate [-h] [--db DB] [--latest]"
                    " [--dry-run] [PROPOSAL_ID]",
                    # the usage line shows both targets as optional: the text says what holds
                    "Judge one stored proposal against a context built now, print every rule's"
                    " finding and record the decision. Exactly one of PROPOSAL_ID and --latest"
                    " is required.",
                    "PROPOSAL_ID the proposal's id (required unless --latest is given)",
                    "--latest judge the newest proposal without a decision (instead of a"
                    " PROPOSAL_ID)",
                    "--dry-run print the verdict and write nothing",
                    "--db DB database file (default: store.db_path from config.yaml)",
                ],
            ),
            (
                ["kill", "--help"],
                [
                    "usage: python -m aegis.cli.policy kill [-h] [--db DB] {on,off,status}",
                    "Set or show the operator's kill switch. on and off log a kill_switch event"
                    " with the previous state, every time; status writes nothing.",
                    # each of the three states says what it does
                    "{on,off,status} on: reject every proposal; off: lift the switch; status:"
                    " print the switch, the trading halt and anything the store could not read"
                    " of either",
                    "--db DB database file (default: store.db_path from config.yaml)",
                ],
            ),
            (
                ["limits", "--help"],
                [
                    "usage: python -m aegis.cli.policy limits [-h] [--db DB]",
                    "Print each risk limit beside its current consumption and headroom, the"
                    " halt and kill-switch state, and what blocks trading right now. Writes"
                    " nothing.",
                    "--db DB database file (default: store.db_path from config.yaml)",
                ],
            ),
        ],
        ids=["top", "evaluate", "kill", "limits"],
    )
    def test_help_exits_0(self, capsys, monkeypatch, argv, expected):
        monkeypatch.setenv("COLUMNS", "200")  # wide enough that argparse wraps nothing
        with pytest.raises(SystemExit) as info:
            cli_policy.main(argv)
        assert info.value.code == 0
        captured = capsys.readouterr()
        assert captured.err == "" and "usage: python -m aegis.cli.policy" in captured.out
        said = " ".join(captured.out.split())  # one line, however the columns were padded
        for text in expected:
            assert text in said

    @pytest.mark.parametrize("command", ["evaluate", "kill", "limits"])
    def test_every_subcommand_describes_itself(self, capsys, monkeypatch, command):
        monkeypatch.setenv("COLUMNS", "200")
        with pytest.raises(SystemExit):
            cli_policy.main([command, "--help"])
        usage, description, *_ = capsys.readouterr().out.split("\n\n")
        assert usage.startswith(f"usage: python -m aegis.cli.policy {command} ")
        assert description.strip() and not description.startswith(("positional", "options"))

    @pytest.mark.parametrize(
        "argv",
        [
            ["evaluate", "--latest"],
            ["evaluate", "prop-0001", "--dry-run"],
            ["kill", "on"],
            ["kill", "off"],
            ["kill", "status"],
            ["limits"],
        ],
        ids=lambda argv: " ".join(argv),
    )
    def test_the_memory_sentinel_is_refused_and_never_becomes_a_file(
        self, tmp_path, monkeypatch, capsys, argv
    ):
        monkeypatch.chdir(tmp_path)
        builder = Builder()
        assert cli_policy.main([*argv, "--db", ":memory:"], context_builder=builder) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith(
            f"policy {argv[0]} failed: --db :memory: is an in-memory database, not a file: "
        )
        assert captured.err.count("\n") == 1
        assert list(tmp_path.iterdir()) == [] and builder.calls == []

    def test_a_relative_db_path_resolves_against_the_cwd(self, tmp_path, monkeypatch, capsys):
        seed(tmp_path / "rel.db", make_proposal())
        monkeypatch.chdir(tmp_path)
        assert cli_policy.main(["evaluate", "--latest", "--db", "rel.db"],
                               context_builder=Builder()) == 0
        assert len(decisions_of(tmp_path / "rel.db", "prop-0001")) == 1
        assert cli_policy.main(["kill", "status", "--db", "nope.db"]) == 1
        assert str(tmp_path / "nope.db") in capsys.readouterr().err

    def test_without_the_flag_the_configured_path_is_used(self, store, monkeypatch, capsys):
        monkeypatch.setattr(cli_policy, "resolve_db_path", lambda: store)
        assert cli_policy.main(["evaluate", "--latest"], context_builder=Builder()) == 0
        assert cli_policy.main(["kill", "on"]) == 0
        assert cli_policy.main(["kill", "status"]) == 0
        assert cli_policy.main(["limits"], context_builder=Builder()) == 0
        assert len(decisions_of(store, "prop-0001")) == 1
        assert controls_of(store).kill_switch is True

    def test_store_errors_are_one_clean_line(self, store, capsys):
        conn = sqlite3.connect(store)
        conn.execute(
            "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
            (99, "0099_from_the_future", NOW.isoformat()),
        )
        conn.commit()
        conn.close()
        for argv in (["evaluate", "--latest"], ["kill", "on"]):
            assert cli_policy.main([*argv, "--db", str(store)], context_builder=Builder()) == 1
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err.startswith(
                f"policy {argv[0]} failed: store operation failed: migrate"
            )
            assert "newer" in captured.err and captured.err.count("\n") == 1
            assert "Traceback" not in captured.err

    @pytest.mark.parametrize(
        "argv",
        [["evaluate", "--latest"], ["evaluate", "--latest", "--dry-run"], ["limits"]],
        ids=["evaluate", "dry-run", "limits"],
    )
    def test_unexpected_errors_never_dump_a_traceback(self, store, capsys, argv):
        def broken(conn, proposal=None, *, config=None):
            raise RuntimeError("boom\nsecond line")

        before = every_row(store)
        assert cli_policy.main([*argv, "--db", str(store)], context_builder=broken) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            f"policy {argv[0]} failed (unexpected): RuntimeError: boom second line\n"
        )
        assert every_row(store) == before

    @pytest.mark.parametrize(
        "argv",
        [["evaluate", "--latest"], ["limits"]],
        ids=["evaluate", "limits"],
    )
    def test_ctrl_c_exits_130(self, store, opened, capsys, argv):
        def interrupted(conn, proposal=None, *, config=None):
            raise KeyboardInterrupt

        before = every_row(store)
        assert cli_policy.main([*argv, "--db", str(store)], context_builder=interrupted) == 130
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err == ""
        assert_all_closed(opened)
        assert every_row(store) == before

    def test_ctrl_c_during_kill_exits_130(self, store, monkeypatch, capsys):
        def interrupted(path):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_policy, "open_store", interrupted)
        assert cli_policy.main(["kill", "on", "--db", str(store)]) == 130

    def test_a_config_error_is_one_clean_line(self, store, monkeypatch, capsys):
        def broken():
            raise ConfigError("config file config.yaml failed validation:\nwatchlist: missing")

        monkeypatch.setattr(cli_policy, "get_config", broken)
        builder = Builder()
        assert cli_policy.main(["evaluate", "--latest", "--db", str(store)],
                               context_builder=builder) == 1
        assert capsys.readouterr().err == (
            "policy evaluate failed: config file config.yaml failed validation:"
            " watchlist: missing\n"
        )
        assert builder.calls == []

    def test_runs_as_a_module(self, tmp_path):
        """``python -m aegis.cli.policy``: the entry point, in a fresh interpreter,
        on the one command that needs neither the config nor a context."""
        missing = tmp_path / "missing.db"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-m", "aegis.cli.policy", "kill", "status", "--db", str(missing)],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=120,
        )
        assert result.returncode == 1 and result.stdout == ""
        assert result.stderr == (
            f"policy kill failed: database {missing} does not exist"
            " (run `python -m aegis.cli.db init` to create it)\n"
        )
        assert not missing.exists()


# --- the module itself --------------------------------------------------------


def _cli_tree() -> ast.Module:
    return ast.parse(CLI_SOURCE.read_text(encoding="utf-8"), filename="policy.py")


def _imports(tree: ast.Module) -> dict[str, set[str]]:
    """``{module: names imported from it}`` (an ``import x`` maps ``x`` to an empty set)."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.setdefault(alias.name, set())
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            found.setdefault(node.module, set()).update(alias.name for alias in node.names)
    return found


class TestModule:
    def test_main_has_the_documented_signature(self):
        parameters = inspect.signature(cli_policy.main).parameters
        assert list(parameters) == ["argv", "context_builder"]
        assert parameters["argv"].default is None
        assert parameters["context_builder"].kind is inspect.Parameter.KEYWORD_ONLY
        assert cli_policy.__doc__ and "python -m aegis.cli.policy evaluate" in cli_policy.__doc__

    def test_no_risk_number_is_hardcoded(self):
        """Exit codes (0, 1, 130) and formatting widths only — every limit the
        CLI prints comes out of the report."""
        numbers = {
            node.value
            for node in ast.walk(_cli_tree())
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
        }
        assert numbers <= {0, 1, 2, 3, 130}
        assert all(isinstance(number, int) for number in numbers)

    def test_live_data_is_reached_only_through_the_context_builder(self):
        imports = _imports(_cli_tree())
        policy_modules = {name for name in imports if name.startswith("aegis.policy")}
        assert policy_modules == {
            "aegis.policy.context", "aegis.policy.engine", "aegis.policy.errors",
            "aegis.policy.limits", "aegis.policy.measures", "aegis.policy.models",
        }
        assert imports["aegis.policy.context"] == {
            "build_context", "latest_undecided", "load_proposal",
        }
        assert imports["aegis.policy.engine"] == {"decide", "evaluate", "stricter_controls"}
        assert imports["aegis.policy.limits"] == {"limits_report"}
        assert imports["aegis.policy.measures"] == {"money"}  # formatting, never a measurement
        # the data layer only for its clock helper; no fetcher, no client, no model
        assert {name for name in imports if name.startswith("aegis.data")} == {"aegis.data.models"}
        assert imports["aegis.data.models"] == {"utcnow"}
        forbidden = ("anthropic", "alpaca", "aegis.brain", "requests", "httpx", "httpx2")
        assert not [
            name
            for name in imports
            if any(name == module or name.startswith(module + ".") for module in forbidden)
        ]

    def test_nothing_private_is_borrowed_from_another_cli(self):
        imports = _imports(_cli_tree())
        assert not [name for name in imports if name.startswith("aegis.cli")]
        borrowed = {name for names in imports.values() for name in names if name.startswith("_")}
        assert borrowed == set()

    def test_the_only_sql_is_the_query_only_pragma(self):
        executed = [
            node.args[0].value
            for node in ast.walk(_cli_tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("execute", "executemany", "executescript")
        ]
        assert executed == ["PRAGMA query_only=ON"]

    def test_timestamps_print_in_utc_to_the_second(self):
        ts = cli_policy._ts
        assert ts(NOW) == "2026-07-30 15:00:00 UTC"
        assert ts(NOW.replace(tzinfo=None)) == "2026-07-30 15:00:00 UTC"  # naive: taken as UTC
        new_york = timezone(timedelta(hours=-4))
        # another zone is converted, not relabelled
        assert ts(NOW.astimezone(new_york)) == "2026-07-30 15:00:00 UTC"
        assert ts(NOW.replace(microsecond=999_999)) == "2026-07-30 15:00:00 UTC"
        assert ts(None) == "unknown" and ts(None, "never") == "never"
        # the far end of the calendar in a zone behind UTC cannot be converted: printed as it is
        assert ts(datetime.max.replace(tzinfo=new_york)) == "9999-12-31T23:59:59.999999-04:00"
        assert ts(datetime.max.replace(tzinfo=timezone.utc)) == "9999-12-31 23:59:59 UTC"

    def test_quantities_prices_and_table_cells(self):
        assert [cli_policy._qty(value) for value in (4.0, 0.5, 1e6, 100.0)] == [
            "4", "0.5", "1000000", "100",
        ]
        assert [cli_policy._price(value) for value in (2.4, 200.0, 561.275)] == [
            "2.40", "200.00", "561.275",
        ]
        assert [cli_policy._strike(value) for value in (640.0, 642.5)] == ["640.0", "642.5"]
        amount = cli_policy._amount
        assert amount(None, "USD") == "-" and amount(float("nan"), "trades") == "-"
        assert amount(1234.5, "USD") == "$1,234.50" and amount(-0.5, "USD") == "-$0.50"
        assert amount(float("inf"), "USD") == "unlimited"
        assert amount(12_000.0, "trades") == "12,000" and amount(-1.0, "positions") == "-1"
        assert amount(float("inf"), "trades") == "unlimited"
        assert amount(float("-inf"), "positions") == "-unlimited"

    def test_prints_go_through_the_cleaner(self):
        """Every stored string the CLI prints is wrapped in ``_clean`` — pinned
        on the helper itself: control and format characters become spaces and
        whitespace runs collapse."""
        clean = cli_policy._clean
        assert clean("a\x1b[2Jb") == "a [2Jb"
        assert clean("one\ntwo\r\n  three\t\x00") == "one two three"
        assert clean("rtl‮override​") == "rtl override"
        assert clean("line separator") == "line separator"
        assert clean("plain — text") == "plain — text"
        assert clean(12.5) == "12.5"
