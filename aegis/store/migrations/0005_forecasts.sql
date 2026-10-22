-- migration: 0005 forecasts: per-ticker probability forecasts, and a forecast reasoning stage
--
-- Applied exactly once by aegis.store.db.migrate, which runs every statement
-- in this file inside one BEGIN IMMEDIATE ... COMMIT together with the row it
-- inserts into schema_version (version 5, name "0005_forecasts"). That
-- atomicity is what makes the reasoning rebuild below safe, as in 0002: the
-- copy, the DROP and the RENAME land together with the version row or not at
-- all. Once this file has been applied anywhere it must never be edited:
-- later schema changes go in a new 0006_<name>.sql. No BEGIN/COMMIT in this
-- file: db.py owns the transaction.
--
-- Why: the forecast stage (aegis/brain/forecast.py) asks the model, once a
-- trading day, for a probability per ticker of one exactly defined event
-- (aegis/forecasts/events.py), so the paper phase produces dozens of
-- gradable forecasts a day. Forecasts are a measuring instrument only:
-- nothing that can place an order reads these tables
-- (tests/test_forecast_architecture.py pins it).
--
-- forecast_runs holds one row per run of one arm (primary or shadow): when
-- its inputs were taken and when it was written, the model, the prompt and
-- event versions, the trading date whose close starts every forecast's
-- clock, the universe asked and the exact inputs the model saw, so any row
-- can be audited later. forecasts holds one row per ticker per event per
-- horizon. Outcomes are not stored: the scorer recomputes them from
-- adjusted bars, so these tables stay a pure point-in-time record.
--
-- reasoning is rebuilt to allow stage 'forecast'. The budget guard sums
-- that table by UTC day and by cycle (a forecast run's id is its cycle id),
-- so a forecast run's tokens count against brain.daily_token_budget like
-- any other stage. SQLite cannot widen a CHECK in place. Nothing references
-- reasoning, so foreign_keys=ON allows the DROP; every column and row is
-- copied as it is.

CREATE TABLE IF NOT EXISTS reasoning_v3 (
    id          TEXT PRIMARY KEY NOT NULL,
    cycle_id    TEXT,
    proposal_id TEXT REFERENCES proposals (id),
    stage       TEXT NOT NULL CHECK (stage IN ('scan', 'thesis', 'proposal', 'forecast')),
    created_at  TEXT NOT NULL,
    content     TEXT NOT NULL,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    model_name  TEXT,
    latency_ms  REAL,
    CHECK (cycle_id IS NOT NULL OR proposal_id IS NOT NULL)
);
INSERT INTO reasoning_v3 (id, cycle_id, proposal_id, stage, created_at, content, tokens_in,
                          tokens_out, model_name, latency_ms)
    SELECT id, cycle_id, proposal_id, stage, created_at, content, tokens_in, tokens_out,
           model_name, latency_ms
    FROM reasoning;
DROP TABLE reasoning;
ALTER TABLE reasoning_v3 RENAME TO reasoning;
CREATE INDEX IF NOT EXISTS ix_reasoning_proposal_id ON reasoning (proposal_id);
CREATE INDEX IF NOT EXISTS ix_reasoning_cycle_id ON reasoning (cycle_id);
CREATE INDEX IF NOT EXISTS ix_reasoning_created_at ON reasoning (created_at);

CREATE TABLE IF NOT EXISTS forecast_runs (
    id              TEXT PRIMARY KEY NOT NULL,
    arm             TEXT NOT NULL CHECK (arm IN ('primary', 'shadow')),
    inputs_taken_at TEXT NOT NULL,
    completed_at    TEXT NOT NULL,
    anchor_session  TEXT NOT NULL,
    benchmark       TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    event_version   TEXT NOT NULL,
    universe        TEXT NOT NULL,
    inputs          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_forecast_runs_anchor_session ON forecast_runs (anchor_session);
CREATE INDEX IF NOT EXISTS ix_forecast_runs_completed_at ON forecast_runs (completed_at);

CREATE TABLE IF NOT EXISTS forecasts (
    id               TEXT PRIMARY KEY NOT NULL,
    run_id           TEXT NOT NULL REFERENCES forecast_runs (id),
    symbol           TEXT NOT NULL,
    event            TEXT NOT NULL CHECK (event IN ('beats_benchmark', 'up')),
    horizon_sessions INTEGER NOT NULL CHECK (horizon_sessions > 0),
    probability      REAL NOT NULL CHECK (probability BETWEEN 0.05 AND 0.95),
    rationale        TEXT NOT NULL,
    UNIQUE (run_id, symbol, event, horizon_sessions)
);
CREATE INDEX IF NOT EXISTS ix_forecasts_run_id ON forecasts (run_id);
