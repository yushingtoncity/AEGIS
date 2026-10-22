"""``python -m aegis.cli.forecasts``: ``run`` with a fake model and fixture
inputs, ``score`` with fixture bars. No network, no keys."""

import hashlib
import json

import pytest

from aegis.brain.testing import FakeLLM
from aegis.cli import forecasts as cli
from aegis.config import BrainConfig, BrainStageConfig, get_config
from aegis.forecasts.inputs import MarketNotOpen
from aegis.store.db import open_store
from aegis.store.forecasts import get_forecast_runs, insert_forecast_run
from aegis.store.models import ForecastArm, ReasoningStage
from forecast_factories import (
    ANCHOR,
    BENCHMARK,
    NOW,
    STOCKS,
    bars_from,
    forecast_config,
    make_inputs,
    make_run,
    output_json,
    record,
)

SYMBOLS = (*STOCKS, BENCHMARK)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "aegis.db"


@pytest.fixture
def enabled(monkeypatch):
    holder = {"config": forecast_config()}
    monkeypatch.setattr(cli, "get_config", lambda: holder["config"])
    return holder


def _inputs(config, now):
    return make_inputs()


def _main(db, *args, llm=None, inputs_builder=_inputs, **kwargs):
    return cli.main(["run", "--db", str(db), *args], llm=llm, inputs_builder=inputs_builder, now=NOW, **kwargs)


def _llm(*texts):
    return FakeLLM({ReasoningStage.FORECAST: list(texts)})


def _runs(db):
    conn = open_store(db)
    try:
        return get_forecast_runs(conn, None, None)
    finally:
        conn.close()


class TestRun:
    def test_refused_while_disabled_and_nothing_is_opened(self, db, capsys):
        assert get_config().forecast.enabled is False  # the shipped config
        assert _main(db, llm=_llm()) == cli.EXIT_FAILED
        assert "forecast.enabled is false" in capsys.readouterr().err
        assert not db.exists()

    def test_both_arms_write_their_runs(self, db, enabled, capsys):
        llm = _llm(output_json(SYMBOLS), output_json(SYMBOLS, p1=0.45))
        assert _main(db, llm=llm) == cli.EXIT_OK
        runs = _runs(db)
        assert sorted(r.arm.value for r in runs) == ["primary", "shadow"]
        assert [c.model for c in llm.calls] == ["claude-opus-5-5", "claude-haiku-5-5"]
        out = capsys.readouterr().out
        assert "primary: WRITTEN run" in out and "shadow: WRITTEN run" in out and "ESTIMATE" in out

    def test_an_arm_that_already_ran_today_is_skipped(self, db, enabled, capsys):
        assert _main(db, "--arm", "primary", llm=_llm(output_json(SYMBOLS))) == cli.EXIT_OK
        llm = _llm(output_json(SYMBOLS))  # one answer only: a second primary call would fail
        assert _main(db, llm=llm) == cli.EXIT_OK
        assert [c.model for c in llm.calls] == ["claude-haiku-5-5"]
        assert "primary: already ran for the 2026-10-21 session; skipped" in capsys.readouterr().out
        assert _main(db, llm=_llm()) == cli.EXIT_OK  # both done: no inputs read, no calls
        assert len(_runs(db)) == 2

    def test_a_closed_market_exits_3_and_writes_nothing(self, db, enabled, capsys):
        def closed(config, now):
            raise MarketNotOpen("the market is closed")

        assert _main(db, llm=_llm(), inputs_builder=closed) == cli.EXIT_MARKET_NOT_OPEN
        assert "the market is closed" in capsys.readouterr().err
        assert _runs(db) == []

    def test_a_budget_halt_exits_2_and_stops_before_the_shadow_arm(self, db, enabled):
        tiny = BrainConfig(per_cycle_token_cap=10, daily_token_budget=10)
        enabled["config"] = enabled["config"].model_copy(update={"brain": tiny})
        llm = _llm(output_json(SYMBOLS), output_json(SYMBOLS))
        assert _main(db, llm=llm) == cli.EXIT_HALTED
        assert llm.calls == [] and _runs(db) == []

    def test_a_failed_arm_exits_1(self, db, enabled):
        bad = output_json(SYMBOLS, MSFT=None)
        llm = _llm(*[bad] * 8, output_json(SYMBOLS))
        assert _main(db, "--arm", "primary", llm=llm) == cli.EXIT_FAILED
        assert _runs(db) == []

    def test_an_unconfigured_shadow_arm_is_a_config_error(self, db, enabled, capsys):
        enabled["config"] = forecast_config(shadow=None)
        assert _main(db, "--arm", "shadow", llm=_llm()) == cli.EXIT_FAILED
        assert "forecast.shadow is not configured" in capsys.readouterr().err

    def test_a_bad_command_exits_1_not_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["fly"])
        assert exc.value.code == cli.EXIT_FAILED


D0, D1 = ANCHOR, ANCHOR.replace(day=22)


def _seeded(db):
    conn = open_store(db)
    try:
        insert_forecast_run(
            conn, make_run("p"), [record("p", "AAPL", 0.8), record("p", "MSFT", 0.3)]
        )
        insert_forecast_run(
            conn, make_run("s", arm=ForecastArm.SHADOW), [record("s", "AAPL", 0.5), record("s", "MSFT", 0.5)]
        )
    finally:
        conn.close()


def _fetcher(calls):
    closes = {"SPY": {D0: 100.0, D1: 101.0}, "AAPL": {D0: 50.0, D1: 51.0}, "MSFT": {D0: 50.0, D1: 50.0}}

    def fetch(symbol, lookback, adjustment):
        calls.append((symbol, lookback, adjustment))
        return bars_from(symbol, closes[symbol])

    return fetch


class TestScore:
    LATER = NOW.replace(month=11)

    def test_the_report_and_its_read_only_access(self, db, capsys):
        _seeded(db)
        digest = hashlib.sha256(db.read_bytes()).hexdigest()
        calls = []
        assert cli.main(["score", "--db", str(db)], fetcher=_fetcher(calls), now=self.LATER) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "PRIMARY TEST (pre-registered): primary / beats_benchmark / 1 session" in out
        assert "log loss 0.2899 vs coin 0.6931; gain +0.4032" in out
        assert "brier 0.0650 vs coin 0.2500" in out
        assert "PRIMARY vs SHADOW" in out
        assert {c[2] for c in calls} == {"all"}  # adjusted closes only
        assert sorted(c[0] for c in calls) == ["AAPL", "MSFT", "SPY"]
        assert hashlib.sha256(db.read_bytes()).hexdigest() == digest

    def test_json_output(self, db, capsys):
        _seeded(db)
        assert cli.main(["score", "--db", str(db), "--json", "--detail"], fetcher=_fetcher([]), now=self.LATER) == 0
        card = json.loads(capsys.readouterr().out)
        assert card["runs"] == 2 and card["primary"]["graded"] == 2 and len(card["forecasts"]) == 4
        assert card["price_errors"] == []

    def test_a_missing_database_is_not_created(self, db, capsys):
        assert cli.main(["score", "--db", str(db)], fetcher=_fetcher([]), now=self.LATER) == cli.EXIT_FAILED
        assert "does not exist" in capsys.readouterr().err and not db.exists()

    def test_an_empty_store_scores_nothing_and_fetches_nothing(self, db, capsys):
        open_store(db).close()
        calls = []
        assert cli.main(["score", "--db", str(db)], fetcher=_fetcher(calls), now=self.LATER) == cli.EXIT_OK
        assert calls == [] and "0 runs" in capsys.readouterr().out


class TestTheStage:
    def test_the_cli_uses_the_configured_arms_not_the_trading_stages(self, db, enabled):
        primary = BrainStageConfig(model="claude-sonnet-5-5", max_tokens=4000)
        enabled["config"] = forecast_config(primary=primary, shadow=None)
        llm = _llm(output_json(SYMBOLS))
        assert _main(db, llm=llm) == cli.EXIT_OK
        assert [(c.model, c.max_tokens) for c in llm.calls] == [("claude-sonnet-5-5", 4000)]
