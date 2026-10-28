"""The end-of-day summary (``aegis.notify.summary``) and ``notify summary``.

Stores are tmp_path files seeded with events, verdicts and forecast runs;
the account is a stand-in AccountState with made-up figures and no account
number; the launchd logs are written by the test. Nothing reaches a broker,
Anthropic or Messages.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from aegis.cli import notify as cli
from aegis.config import AegisConfig, ForecastConfig, NotifyConfig
from aegis.data.models import AccountState, Position
from aegis.notify.models import BODY_MAX
from aegis.notify.outbox import notice_for, wanted
from aegis.notify.summary import (
    NEW_YORK,
    SUMMARY_KIND,
    build_summary,
    clock_time,
    parse_log_lines,
    read_log_lines,
    summary_event,
)
from aegis.store import Event, EventLevel, get_events_since, log_event, open_store
from aegis.store.forecasts import insert_forecast_run
from aegis.store.models import ForecastArm, PolicyDecision, Proposal, Verdict
from aegis.store.repo import insert_proposal, record_decision, set_kill_switch
from forecast_factories import make_run, record

DAY = date(2026, 10, 28)  # a Wednesday
CYCLES = ((10, 0), (11, 30), (13, 30), (15, 0))


def at(hour: int, minute: int = 0, day: date = DAY) -> datetime:
    return datetime.combine(day, time(hour, minute), tzinfo=NEW_YORK).astimezone(timezone.utc)


NOW = at(16, 21)


def _config(**notify) -> AegisConfig:
    return AegisConfig(
        watchlist=["SPY"],
        notify=NotifyConfig(enabled=True, **notify),
        forecast=ForecastConfig(enabled=True, universe=["AAPL", "MSFT"]),
    )


def _account(**overrides) -> AccountState:
    fields = dict(
        account_number="", status="ACTIVE", equity=100_512.40, last_equity=100_478.00,
        positions=[Position(symbol="SPY", qty=1, side="long", unrealized_pl=12.0)],
    )
    return AccountState(**{**fields, **overrides})


def _cycle(conn, hour: int, minute: int, outcome: str = "no_trade", cost: float | None = 0.04,
           cycle_id: str | None = None, symbol: str | None = None) -> None:
    cycle_id = cycle_id or f"c{hour}{minute:02d}"
    log_event(conn, Event(occurred_at=at(hour, minute), level=EventLevel.INFO, kind="cycle_start",
                          message="started", payload={"cycle_id": cycle_id}))
    if symbol is not None:
        log_event(conn, Event(occurred_at=at(hour, minute + 1), level=EventLevel.INFO, kind="proposal",
                              message="proposed", payload={"cycle_id": cycle_id, "symbol": symbol}))
    if outcome:
        log_event(conn, Event(occurred_at=at(hour, minute + 2), level=EventLevel.INFO, kind="cycle_end",
                              message="ended", payload={"cycle_id": cycle_id, "outcome": outcome,
                                                        "estimated_cost_usd": cost}))


def _quiet_day(conn) -> None:
    for hour, minute in CYCLES:
        _cycle(conn, hour, minute)
    insert_forecast_run(conn, make_run("p", anchor=DAY), [record("p", "AAPL", 0.6), record("p", "MSFT", 0.4)])
    insert_forecast_run(conn, make_run("s", anchor=DAY, arm=ForecastArm.SHADOW),
                        [record("s", "AAPL", 0.5), record("s", "MSFT", 0.5)])
    log_event(conn, Event(occurred_at=at(15, 3), level=EventLevel.INFO, kind="forecast_end",
                          message="ok", payload={"outcome": "written", "estimated_cost_usd": 0.10}))


CLEAN_LOGS = [
    "2026-10-28 10:00:05 EDT [cycle] start",
    "2026-10-28 10:01:10 EDT [cycle] exit 0: aegis.cli.orders tick",
    "2026-10-28 16:05:00 EDT [tick] exit 0: aegis.cli.orders sync",
]


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "summary.db")
    yield connection
    connection.close()


def _build(conn, *, account=None, error=None, logs=CLEAN_LOGS, config=None, day=DAY):
    lines = parse_log_lines(logs, day) if logs is not None else None
    return build_summary(conn, day, config or _config(), now=NOW, account=account or (None if error else _account()),
                         account_error=error, log_lines=lines)


# --- the text -----------------------------------------------------------------


def test_a_quiet_day(conn):
    _quiet_day(conn)
    summary = _build(conn)
    assert summary.text.splitlines() == [
        "Wed Oct 28",
        "Cycles: 4 of 4 ran, all no trade",
        "Trades: none",
        "Account: equity $100,512.40, up $34.40 today. SPY 1 (up $12.00)",
        "Forecasts: 2 runs (primary, shadow), 4 forecasts",
        "Claude: est. $0.26, 0 tokens (0% of the daily budget)",
        "Problems: none",
    ]
    assert summary.problems == 0


def test_a_trading_day_names_the_proposals_verdicts_and_orders(conn):
    _quiet_day(conn)
    _cycle(conn, 12, 0, outcome="proposal", cycle_id="cx", symbol="NVDA")
    proposal = insert_proposal(conn, Proposal(
        cycle_id="cx", symbol="NVDA", instrument="equity", side="buy", quantity=1, order_type="limit",
        limit_price=100.0, thesis="t", confidence=0.6, invalidation="i", raw_model_output="{}",
        model_name="m", prompt_version="v",
    ))
    for verdict, rule in ((Verdict.AUTO_EXECUTE, None), (Verdict.REJECT, "position_cap")):
        record_decision(conn, PolicyDecision(proposal_id=proposal.id, decided_at=at(12, 3), verdict=verdict,
                                             rules_evaluated=[], failing_rule=rule))
    log_event(conn, Event(occurred_at=at(12, 4), level=EventLevel.INFO, kind="order_submitted",
                          message="sent", payload={"symbol": "NVDA"}))
    text = _build(conn, config=_config(summary={"expected_cycles": 5})).text
    assert "Cycles: 5 of 5 ran. 10:00 am no trade, 11:30 am no trade, 12:00 pm proposed NVDA" in text
    assert ("Trades: 1 proposal. Verdicts: 1 reject, 1 auto execute (rejects: position_cap). "
            "0 approvals. Orders: 1 (NVDA)") in text


def test_a_bad_day_lists_its_problems(conn, tmp_path):
    _cycle(conn, 10, 0)
    _cycle(conn, 11, 30, outcome="failed", cost=None)
    _cycle(conn, 13, 30, outcome="")
    log_event(conn, Event(occurred_at=at(11, 31), level=EventLevel.ERROR, kind="brain_error", message="boom"))
    log_event(conn, Event(occurred_at=at(15, 1), level=EventLevel.ERROR, kind="forecast_end",
                          message="failed", payload={"outcome": "failed"}))
    set_kill_switch(conn, True)
    logs = CLEAN_LOGS + [
        "2026-10-28 14:05:02 EDT [tick] exit 1: aegis.cli.orders tick",
        "2026-10-28 14:10:02 EDT [tick] exit 1: aegis.cli.orders tick",
        "2026-10-28 14:15:00 EDT [tick] skipped: busy (another AEGIS step holds /x/run.lock)",
        "2026-10-28 14:20:00 EDT [tick] skipped: market clock unreadable",
        "2026-10-27 14:05:02 EDT [tick] exit 1: aegis.cli.notify send",  # yesterday: not today's problem
    ]
    summary = _build(conn, logs=logs, config=_config(summary={"expected_cycles": 4}))
    lines = summary.text.splitlines()
    assert "Cycles: 3 of 4 ran. 10:00 am no trade, 11:30 am FAILED, 1:30 pm never finished" in lines
    assert "Forecasts: none ran" in lines
    assert "Claude: est. cost n/a" in summary.text
    assert lines[lines.index("Problems (9):") + 1:] == [
        "- the kill switch is ON",
        "- only 3 of 4 brain cycles ran",
        "- the 1:30 pm cycle never finished",
        "- a forecast run failed at 3:01 pm",
        "- the forecasts did not run",
        "- brain_error at 11:31 am",
        "- and 3 more (python -m aegis.cli.trace)",
    ]
    assert summary.problems == 9


def test_the_log_problems_read_as_sentences(conn):
    _quiet_day(conn)
    logs = [
        "2026-10-28 14:05:02 EDT [tick] exit 1: aegis.cli.orders tick",
        "2026-10-28 14:10:02 EDT [tick] exit 1: aegis.cli.orders tick",
        "2026-10-28 14:15:00 EDT [tick] skipped: busy (another AEGIS step holds /x/run.lock)",
        "2026-10-28 15:02:00 EDT [cycle] exit 2: aegis.cli.forecasts run",
        "aegis output that happens to say exit 1: in the middle",
    ]
    lines = _build(conn, logs=logs).text.splitlines()
    assert lines[-3:] == [
        "- aegis.cli.orders tick exited 1 2 times, first at 2:05 pm",
        "- aegis.cli.forecasts run exited 2 at 3:02 pm",
        "- a step was skipped because another was still running",
    ]


def test_a_stale_quote_reject_is_a_problem(conn):
    _quiet_day(conn)
    proposal = insert_proposal(conn, Proposal(
        cycle_id="cq", symbol="AMD", instrument="equity", side="buy", quantity=1, order_type="market",
        thesis="t", confidence=0.6, invalidation="i", raw_model_output="{}", model_name="m", prompt_version="v",
    ))
    record_decision(conn, PolicyDecision(proposal_id=proposal.id, decided_at=at(13, 33), verdict=Verdict.REJECT,
                                         rules_evaluated=[], failing_rule="quote_freshness"))
    text = _build(conn).text
    assert "(rejects: quote_freshness)" in text
    assert "- stale quote reject at 1:33 pm" in text


def test_an_account_that_cannot_be_read_still_sends_the_rest(conn):
    _quiet_day(conn)
    summary = _build(conn, error="DataError")
    assert "Account: could not read (DataError)" in summary.text
    assert "- the account could not be read (DataError)" in summary.text
    assert summary.text.startswith("Wed Oct 28\nCycles: 4 of 4 ran")


def test_many_positions_are_counted_not_listed(conn):
    _quiet_day(conn)
    positions = [Position(symbol=s, qty=2, side="long", unrealized_pl=-3.5) for s in ("A", "B", "C", "D", "E", "F")]
    text = _build(conn, account=_account(positions=positions, last_equity=None)).text
    assert "Account: equity $100,512.40. A 2 (down $3.50), B 2 (down $3.50), C 2 (down $3.50), D 2 (down $3.50), and 2 more" in text


def test_no_positions(conn):
    _quiet_day(conn)
    assert "Account: equity $100,512.40, flat today. No positions" in _build(
        conn, account=_account(positions=[], last_equity=100_512.40)).text


def test_forecasts_off_is_not_a_problem(conn):
    _quiet_day(conn)
    config = AegisConfig(watchlist=["SPY"], notify=NotifyConfig(enabled=True))
    text = _build(conn, config=config).text
    assert "Forecasts: off" in text and "Problems: none" in text


def test_missing_logs_are_a_problem(conn, tmp_path):
    _quiet_day(conn)
    assert read_log_lines(tmp_path / "nowhere", DAY) is None
    assert "- the launchd logs were not found" in _build(conn, logs=None).text


def test_a_holiday_says_only_that(conn):
    logs = [f"2026-10-28 {h:02d}:{m:02d}:05 EST [cycle] skipped: market closed" for h, m in CYCLES]
    summary = _build(conn, logs=logs)
    assert summary.text == "Wed Oct 28\nMarket closed today, so nothing ran."
    assert summary.problems == 0


def test_the_day_is_a_new_york_day(conn):
    _quiet_day(conn)
    log_event(conn, Event(occurred_at=at(23, 30, day=date(2026, 10, 27)), level=EventLevel.ERROR,
                          kind="brain_error", message="the evening before"))
    assert "Problems: none" in _build(conn).text


def test_twelve_hour_times():
    assert [clock_time(at(h, m)) for h, m in ((9, 5), (12, 0), (15, 30), (0, 10))] == [
        "9:05 am", "12:00 pm", "3:30 pm", "12:10 am",
    ]


def test_the_text_is_cut_to_what_a_notice_carries(conn):
    for minute in range(50):
        log_event(conn, Event(occurred_at=at(10, minute), level=EventLevel.WARNING, kind=f"k{'x' * 60}{minute}",
                              message="m"))
    assert len(_build(conn).text) <= BODY_MAX


def test_read_log_lines_reads_both_logs(tmp_path):
    (tmp_path / "cycle.log").write_text("2026-10-28 10:00:05 EDT [cycle] start\nnoise\n")
    (tmp_path / "tick.log").write_text("2026-10-28 10:05:05 EDT [tick] exit 1: aegis.cli.orders tick\n")
    lines = read_log_lines(tmp_path, DAY)
    assert [(line.mode, line.hour, line.minute) for line in lines] == [("cycle", 10, 0), ("tick", 10, 5)]


# --- the notice -------------------------------------------------------------------


def test_the_notice_keeps_the_lines_and_is_wanted_by_the_shipped_kinds(conn):
    _quiet_day(conn)
    event = summary_event(_build(conn), NOW)
    assert wanted(event, NotifyConfig(enabled=True))
    notice = notice_for(event)
    assert notice.title == "AEGIS Day summary"
    assert notice.body.splitlines()[:2] == ["Wed Oct 28", "Cycles: 4 of 4 ran, all no trade"]


def test_other_kinds_still_come_out_on_one_line():
    event = Event(occurred_at=NOW, level=EventLevel.WARNING, kind="order_refused", message="a\nb")
    assert notice_for(event).body == "a b"


def test_a_summary_line_cannot_smuggle_control_characters():
    event = Event(occurred_at=NOW, level=EventLevel.INFO, kind=SUMMARY_KIND, message="one\x1b[2J\n\ntwo\r")
    assert notice_for(event).body == "one [2J\ntwo"


# --- the CLI ----------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cli.db"
    conn = open_store(path)
    _quiet_day(conn)
    conn.close()
    return path


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(cli, "get_config", _config)


def _main(db, *argv, reader=_account):
    return cli.main(["summary", "--db", str(db), *argv], clock=lambda: NOW, account_reader=reader)


def _summaries(db):
    conn = open_store(db)
    try:
        return get_events_since(conn, at(0, 0), kind=SUMMARY_KIND)
    finally:
        conn.close()


def test_the_cli_writes_one_summary_a_day(db, configured, tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    assert _main(db, "--logs", str(tmp_path / "logs")) == 0
    out = capsys.readouterr().out
    assert out.startswith("Wed Oct 28\nCycles: 4 of 4 ran, all no trade")
    assert "written as a daily_summary event" in out
    [event] = _summaries(db)
    assert event.payload == {"day": "2026-10-28", "problems": 0}
    assert _main(db, "--logs", str(tmp_path / "logs")) == 0
    assert capsys.readouterr().out == "the summary for 2026-10-28 was already written\n"
    assert len(_summaries(db)) == 1


def test_dry_run_writes_nothing(db, configured, capsys):
    assert _main(db, "--dry-run") == 0
    assert "Problems (1):\n- the launchd logs were not found" in capsys.readouterr().out
    assert _summaries(db) == []


def test_an_account_read_that_raises_is_named_by_type_only(db, configured, capsys):
    def broken():
        raise RuntimeError("https://secret.example/key")

    assert _main(db, "--dry-run", reader=broken) == 0
    out = capsys.readouterr().out
    assert "Account: could not read (RuntimeError)" in out and "secret" not in out


def test_a_bad_day_is_a_usage_error(db, configured, capsys):
    assert _main(db, "--day", "yesterday") == 1
    assert "not a date" in capsys.readouterr().err


def test_the_cli_works_with_sending_off(db, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_config", lambda: AegisConfig(watchlist=["SPY"]))
    assert _main(db) == 0
    assert len(_summaries(db)) == 1
