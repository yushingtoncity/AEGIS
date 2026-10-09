"""``python -m aegis.cli.orders``: what each command prints, what it writes,
and how it fails. The broker is the dispatcher tests' stand-in, the context
a factory one and the clock fixed; nothing reaches the network."""

import sqlite3
from datetime import timedelta

import pytest
from policy_factories import WATCHLIST, long_call, make_proposal
from test_policy_dispatch import LATER, FakeBroker, _builder, _judged

from aegis.cli import orders as cli
from aegis.config import AegisConfig, BrokerConfig, ConfigError
from aegis.policy import dispatch
from aegis.store import get_order, open_store, set_kill_switch

ON = AegisConfig(watchlist=list(WATCHLIST), broker=BrokerConfig(enabled=True))
OFF = AegisConfig(watchlist=list(WATCHLIST))


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "orders.db"
    open_store(path).close()
    return path


@pytest.fixture
def config(monkeypatch):
    """The config the CLI reads; placing on unless a test says otherwise."""
    holder = {"config": ON}
    monkeypatch.setattr(cli, "get_config", lambda: holder["config"])
    return holder


def _run(db, *argv, broker=None, builder=None):
    broker = broker or FakeBroker()
    return cli.main(
        [*argv, "--db", str(db)],
        broker_factory=lambda conn, config: broker,
        context_builder=builder or _builder(),
        clock=lambda: LATER,
    )


def _with(db, act):
    conn = open_store(db)
    try:
        return act(conn)
    finally:
        conn.close()


def _rows(db):
    conn = sqlite3.connect(db)
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {t: conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall() for t in tables}
    finally:
        conn.close()


class TestReading:
    def test_pending_lists_what_waits_on_a_human_and_what_may_go(self, db, config, capsys):
        auto = _with(db, lambda c: _judged(c)[1])
        needs = _with(db, lambda c: _judged(c, long_call(id="prop-0002", cycle_id="c2"))[1])
        before = _rows(db)
        assert _run(db, "pending") == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "awaiting approval (1)"
        assert out[1].startswith(f"  {needs.id}   NEEDS_APPROVAL   buy 1 SPY")
        assert "escalated: options_escalate, auto_tier" in out[1]
        assert out[2] == "ready to place (1)"
        assert out[3].startswith(f"  {auto.id}   AUTO_EXECUTE   buy 2 AAPL limit 200.00")
        assert _rows(db) == before  # query-only

    def test_pending_says_when_placing_is_off(self, db, config, capsys):
        config["config"] = OFF
        assert _run(db, "pending") == 0
        out = capsys.readouterr().out
        assert "awaiting approval (0)\n  (none)\nready to place (0)\n  (none)\n" in out
        assert out.endswith("note: broker.enabled is false in config.yaml: placing is off\n")

    def test_status_lists_open_orders_and_one_order_with_its_fills(self, db, config, capsys):
        broker = FakeBroker()
        verdict = _with(db, lambda c: _judged(c)[1])
        assert _run(db, "place", verdict.id, broker=broker) == 0
        capsys.readouterr()
        assert _run(db, "status") == 0
        line = capsys.readouterr().out.splitlines()[1]
        assert line.startswith("  aegis-prop-0001   submitted   buy 2 AAPL limit 200.00   filled 0/2")
        broker.fill("aegis-prop-0001", "2", "199.9", "filled")
        assert _run(db, "sync", broker=broker) == 0
        capsys.readouterr()
        assert _run(db, "status") == 0
        assert capsys.readouterr().out == "open orders (0)\n  (none)\n"
        assert _run(db, "status", "--all") == 0
        assert "filled 2/2 at 199.90" in capsys.readouterr().out
        assert _run(db, "status", "aegis-prop-0001") == 0
        out = capsys.readouterr().out.splitlines()
        assert out[1] == "fills (1)" and out[2].endswith("2 at 199.90")

    def test_status_of_a_name_nobody_placed(self, db, config, capsys):
        assert _run(db, "status", "aegis-nowhere") == 1
        assert capsys.readouterr().err == (
            "orders status failed: no order placed under the name aegis-nowhere\n"
        )


class TestPlacing:
    def test_approve_then_place_then_once_only(self, db, config, capsys):
        verdict = _with(db, lambda c: _judged(c, long_call())[1])
        assert _run(db, "approve", verdict.id, "--by", "op", "--note", "fine") == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0].startswith(f"{verdict.id}: approved by op, good until ")
        assert out[1] == f"next: python -m aegis.cli.orders place {verdict.id}"
        broker = FakeBroker()
        assert _run(db, "place", verdict.id, broker=broker) == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0].startswith("claimed aegis-prop-0001: buy 1 SPY")
        assert out[-1].startswith("sent: accepted at the broker as ")
        assert _run(db, "place", verdict.id, broker=broker) == 1
        assert "already has order aegis-prop-0001" in capsys.readouterr().err

    def test_reject(self, db, config, capsys):
        verdict = _with(db, lambda c: _judged(c, long_call())[1])
        assert _run(db, "reject", verdict.id, "--by", "op") == 0
        assert capsys.readouterr().out == f"{verdict.id}: rejected by op\n"
        assert _run(db, "place", verdict.id) == 1
        assert "the answer was rejected" in capsys.readouterr().err

    def test_with_placing_off_place_refuses_and_a_dry_run_shows_it(self, db, config, capsys):
        config["config"] = OFF
        verdict = _with(db, lambda c: _judged(c)[1])
        before = _rows(db)
        broker = FakeBroker()
        assert _run(db, "place", verdict.id, broker=broker) == 1
        assert "broker.enabled is false" in capsys.readouterr().err
        assert _run(db, "place", verdict.id, "--dry-run", broker=broker) == 0
        out = capsys.readouterr().out.splitlines()
        assert out == [
            "would place aegis-prop-0001: buy 2 AAPL limit 200 day",
            "re-check: AUTO_EXECUTE (AUTO_EXECUTE (all 21 rules passed))",
            cli.DRY_RUN_NOTE,
        ]
        assert broker.calls == [] and _rows(db) == before

    def test_a_refused_send_is_one_line_and_exit_1(self, db, config, capsys):
        from test_policy_dispatch import _failure

        from aegis.execution.models import ExecutionOutcome

        verdict = _with(db, lambda c: _judged(c)[1])
        broker = FakeBroker()
        broker.send = [_failure(ExecutionOutcome.REJECTED, "insufficient buying power")]
        assert _run(db, "place", verdict.id, broker=broker) == 1
        err = capsys.readouterr().err
        assert err.startswith("orders place failed: policy failed: place order (rejected: ")
        assert err.count("\n") == 1


class TestTheBrokerCommands:
    def test_sync_cancel_stand_down_tick(self, db, config, capsys):
        broker = FakeBroker()
        _with(db, lambda c: _judged(c))
        assert _run(db, "tick", broker=broker) == 0
        assert "claimed aegis-prop-0001" in capsys.readouterr().out
        assert _run(db, "sync", broker=broker) == 0
        assert capsys.readouterr().out == "aegis-prop-0001: submitted, unchanged\n"
        assert _run(db, "stand-down", broker=broker) == 0
        assert capsys.readouterr().out == "controls clear: nothing to stand down\n"
        assert _run(db, "cancel", "aegis-prop-0001", broker=broker) == 0
        assert capsys.readouterr().out.startswith("aegis-prop-0001: cancel requested\n")
        assert _with(db, lambda c: get_order(c, "aegis-prop-0001")).status.value == "cancelled"

    def test_an_unresolved_order_is_exit_1(self, db, config, capsys):
        broker = FakeBroker()
        _with(db, lambda c: _judged(c))
        _run(db, "tick", broker=broker)
        broker.foreign = [broker.receipt("dashboard-1", "MSFT")]
        capsys.readouterr()
        assert _run(db, "sync", broker=broker) == 1
        captured = capsys.readouterr()
        assert "ORPHAN dashboard-1" in captured.out and captured.err == "1 not resolved\n"

    def test_stand_down_under_the_kill_switch_cancels(self, db, config, capsys):
        broker = FakeBroker()
        _with(db, lambda c: _judged(c))
        _run(db, "tick", broker=broker)
        _with(db, lambda c: set_kill_switch(c, True, now=LATER))
        capsys.readouterr()
        assert _run(db, "stand-down", broker=broker) == 0
        out = capsys.readouterr().out
        assert out.startswith("standing down: the kill switch is on\n")
        assert "aegis-prop-0001: submitted -> cancelled" in out


class TestFailing:
    @pytest.mark.parametrize(
        "argv",
        [
            ["approve", "dec-1"],  # no --by
            ["approve", "  ", "--by", "op"],
            ["approve", "dec-1", "--by", "  "],
            ["place"],
            ["cancel"],
            ["launch"],
        ],
    )
    def test_a_bad_command_line_is_one_line_and_exit_1(self, db, config, capsys, argv):
        assert _run(db, *argv) == 1
        err = capsys.readouterr().err
        assert err.endswith("(see --help)\n") and err.count("\n") == 1

    def test_help_exits_0(self, capsys):
        with pytest.raises(SystemExit) as info:
            cli.main(["--help"])
        assert info.value.code == 0
        assert "stand-down" in capsys.readouterr().out

    def test_a_database_that_does_not_exist_is_never_created(self, tmp_path, config, capsys):
        missing = tmp_path / "nowhere.db"
        for argv in (["pending"], ["sync"], ["approve", "d", "--by", "op"]):
            assert _run(missing, *argv) == 1
            assert "does not exist" in capsys.readouterr().err
        assert not missing.exists()

    def test_the_in_memory_database_is_refused(self, config, capsys):
        assert cli.main(["pending", "--db", ":memory:"]) == 1
        assert "in-memory database, not a file" in capsys.readouterr().err

    def test_missing_keys_are_one_line(self, db, config, capsys):
        def no_keys(conn, config):
            raise ConfigError("missing required environment variable ALPACA_API_KEY.")

        _with(db, lambda c: _judged(c))
        assert cli.main(["sync", "--db", str(db)], broker_factory=no_keys, clock=lambda: LATER) == 1
        assert capsys.readouterr().err == (
            "orders sync failed: missing required environment variable ALPACA_API_KEY.\n"
        )

    def test_an_unexpected_error_is_one_line_never_a_traceback(self, db, config, capsys):
        def broken(conn, config):
            raise RuntimeError("boom\nsecond line")

        assert cli.main(["sync", "--db", str(db)], broker_factory=broken, clock=lambda: LATER) == 1
        assert capsys.readouterr().err == (
            "orders sync failed (unexpected): RuntimeError: boom second line\n"
        )

    def test_ctrl_c_exits_130(self, db, config):
        def interrupted(conn, config):
            raise KeyboardInterrupt

        assert cli.main(["sync", "--db", str(db)], broker_factory=interrupted) == 130

    def test_the_default_broker_is_the_paper_one(self, db, config, monkeypatch):
        built = []
        monkeypatch.setattr(dispatch, "paper_broker", lambda conn, config: built.append(config) or FakeBroker())
        assert cli.main(["sync", "--db", str(db)], clock=lambda: LATER) == 0
        assert built == [ON]

    def test_the_default_clock_is_the_wall_clock(self, db, config, capsys):
        assert cli.main(["pending", "--db", str(db)]) == 0  # reads utcnow, writes nothing

    def test_a_proposal_named_with_control_characters_prints_on_one_line(self, db, config, capsys):
        proposal = make_proposal(thesis="SYNTHETIC TEST DATA.")
        _with(db, lambda c: _judged(c, proposal))
        broker = FakeBroker()
        _run(db, "tick", broker=broker)
        capsys.readouterr()
        assert _run(db, "status", "--all") == 0
        assert all("\x1b" not in line for line in capsys.readouterr().out.splitlines())


def test_the_approval_window_is_the_configs(db, config, capsys):
    verdict = _with(db, lambda c: _judged(c, long_call())[1])
    _run(db, "approve", verdict.id, "--by", "op")
    out = capsys.readouterr().out
    assert f"good until {cli._ts(LATER + timedelta(seconds=900))}" in out


def test_the_trace_shows_the_recheck_the_approval_and_the_order(db, config, capsys):
    """Invariant 5: the audit view of a proposal shows what Phase 6 recorded."""
    from aegis.cli import trace

    verdict = _with(db, lambda c: _judged(c, long_call())[1])
    _run(db, "approve", verdict.id, "--by", "op")
    _run(db, "place", verdict.id)
    capsys.readouterr()
    assert trace.main(["prop-0001", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "pre_submit re-check" in out
    assert f"answers decision {verdict.id}, good until " in out
    assert f"decision {verdict.id}   re-check " in out
    assert "filled 0 of 1   broker status accepted   synced 2026-07-30 15:00:10 UTC" in out
