"""Migration 0005 and the forecast repo functions, plus the forecast config
block: the shipped ``forecast.enabled`` pin and what its validators refuse."""

import sqlite3
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from aegis.config import BrainStageConfig, ForecastConfig, get_config
from aegis.store import db as store_db
from aegis.store.db import connect, list_migrations, migrate, open_store
from aegis.store.errors import StoreError
from aegis.store.forecasts import (
    get_forecast_runs,
    get_run_forecasts,
    get_runs_for_session,
    insert_forecast_run,
)
from aegis.store.models import ForecastArm, ForecastEvent, Reasoning, ReasoningStage
from aegis.store.repo import add_reasoning, get_cycle_reasoning
from forecast_factories import ANCHOR, make_run, record


@pytest.fixture
def conn(tmp_path):
    conn = open_store(tmp_path / "f.db")
    yield conn
    conn.close()


def _count(conn, table) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class TestMigration:
    def test_0005_on_a_version_4_store_keeps_reasoning_and_opens_the_forecast_stage(self, tmp_path):
        conn = connect(tmp_path / "v4.db")
        conn.execute(store_db._SCHEMA_VERSION_DDL)
        for version, name, file in list_migrations()[:4]:
            store_db._apply_migration(conn, version, name, file)
        for stage in (ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL):
            add_reasoning(conn, Reasoning(cycle_id="cyc-1", stage=stage, content=stage.value, tokens_in=7))
        with pytest.raises(StoreError):  # before 0005 the table refuses the new stage
            add_reasoning(conn, Reasoning(cycle_id="cyc-1", stage=ReasoningStage.FORECAST, content="x"))
        before = [(r.id, r.stage, r.content, r.tokens_in) for r in get_cycle_reasoning(conn, "cyc-1")]
        assert migrate(conn) == len(list_migrations())
        after = [(r.id, r.stage, r.content, r.tokens_in) for r in get_cycle_reasoning(conn, "cyc-1")]
        assert after == before and len(after) == 3
        add_reasoning(conn, Reasoning(cycle_id="run-1", stage=ReasoningStage.FORECAST, content="x"))
        indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
        assert {"ix_reasoning_proposal_id", "ix_reasoning_cycle_id", "ix_reasoning_created_at",
                "ix_forecast_runs_anchor_session", "ix_forecasts_run_id"} <= indexes
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.close()

    def test_the_table_still_refuses_an_unknown_stage(self, conn):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reasoning (id, cycle_id, stage, created_at, content) VALUES ('r', 'c', 'guess', 'now', 'x')"
            )


class TestRepo:
    def test_a_run_and_its_rows_round_trip(self, conn):
        run = make_run("run-1")
        rows = [record("run-1", "AAPL", 0.55), record("run-1", "SPY", 0.6, event=ForecastEvent.UP, horizon=5)]
        insert_forecast_run(conn, run, rows)
        assert get_forecast_runs(conn, None, None) == [run]
        assert get_run_forecasts(conn, ["run-1"]) == {"run-1": sorted(rows, key=lambda r: (r.symbol, r.event.value))}
        assert get_runs_for_session(conn, ANCHOR) == [run]
        assert get_runs_for_session(conn, ANCHOR + timedelta(days=1)) == []

    def test_an_empty_run_or_foreign_rows_are_refused(self, conn):
        with pytest.raises(StoreError, match="at least one forecast"):
            insert_forecast_run(conn, make_run("run-1"), [])
        with pytest.raises(StoreError):
            insert_forecast_run(conn, make_run("run-1"), [record("run-2", "AAPL", 0.5)])
        assert _count(conn, "forecast_runs") == 0

    def test_a_bad_row_leaves_nothing_behind(self, conn):
        rows = [record("run-1", "AAPL", 0.5), record("run-1", "AAPL", 0.6)]  # same key twice
        with pytest.raises(StoreError):
            insert_forecast_run(conn, make_run("run-1"), rows)
        assert _count(conn, "forecast_runs") == 0 and _count(conn, "forecasts") == 0

    def test_the_schema_holds_the_probability_bounds_and_the_arm(self, conn):
        insert_forecast_run(conn, make_run("run-1"), [record("run-1", "AAPL", 0.5)])
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO forecasts (id, run_id, symbol, event, horizon_sessions, probability, rationale)"
                " VALUES ('f', 'run-1', 'MSFT', 'beats_benchmark', 1, 0.99, 'r')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE forecast_runs SET arm = 'tertiary'")

    def test_since_and_until_select_by_anchor_session(self, conn):
        days = [date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)]
        for n, day in enumerate(days):
            insert_forecast_run(conn, make_run(f"run-{n}", anchor=day), [record(f"run-{n}", "AAPL", 0.5)])
        insert_forecast_run(
            conn, make_run("shadow", anchor=days[1], arm=ForecastArm.SHADOW), [record("shadow", "AAPL", 0.5)]
        )
        picked = get_forecast_runs(conn, days[1], days[2])
        assert sorted(r.id for r in picked) == ["run-1", "shadow"]


class TestConfig:
    def test_the_repo_ships_forecasts_off_on_a_thirty_name_universe(self):
        forecast = get_config().forecast
        assert forecast.enabled is False
        assert len(forecast.universe) == 30 and forecast.benchmark == "SPY"
        assert forecast.primary.track is None and forecast.shadow is not None and forecast.shadow.track is None

    def test_the_delisted_ea_slot_went_to_rblx(self):
        # EA was taken private on 2026-08-04 and has no bars; see the dated note in config.yaml
        universe = get_config().forecast.universe
        assert "EA" not in universe and "RBLX" in universe

    def test_enabled_must_be_a_real_boolean(self):
        with pytest.raises(ValidationError):
            ForecastConfig(enabled="yes")

    def test_the_benchmark_may_not_be_in_the_universe(self):
        with pytest.raises(ValidationError, match="must not also be in forecast.universe"):
            ForecastConfig(universe=("AAPL", "spy"))

    def test_symbols_are_cleaned_and_repeats_refused(self):
        assert ForecastConfig(universe=(" aapl ", "msft")).universe == ("AAPL", "MSFT")
        with pytest.raises(ValidationError, match="repeat"):
            ForecastConfig(universe=("AAPL", "aapl"))
        with pytest.raises(ValidationError, match="blank"):
            ForecastConfig(universe=("AAPL", " "))

    def test_an_arm_is_pinned_to_one_model(self):
        tracked = BrainStageConfig(model="claude-opus-5-5", max_tokens=8000, track="opus")
        with pytest.raises(ValidationError, match="pinned"):
            ForecastConfig(primary=tracked)

    def test_unknown_keys_are_refused(self):
        with pytest.raises(ValidationError):
            ForecastConfig(trade_on_it=True)
