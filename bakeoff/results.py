"""The bakeoff's results file: its own SQLite database, never the audit DB.

Tables (all keyed by ``run_id``):

- ``runs``         one row per run: status, abort reason, spend, dataset hash
- ``contestants``  who ran in the run, with the resolved model and settings
- ``calls``        one row per (input, proposer): raw output, parse result,
                   validation errors, the dry run verdict of the candidate the
                   live cycle would have offered, latency, tokens, cost
- ``standins``     one row per candidate dry run (every candidate, every proposer)
- ``critiques``    one row per critic review of a baseline candidate, linked
                   to the stand in proposal id when the candidate resolved

``open_results`` refuses the production store path, so a misconfigured
``--results`` can never write into the audit log.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from aegis.store.db import resolve_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL CHECK (status IN
                      ('running', 'completed', 'aborted_spend_cap', 'aborted_errors', 'failed')),
    abort_reason    TEXT,
    dataset         TEXT NOT NULL,
    manifest_hash   TEXT NOT NULL,
    input_count     INTEGER NOT NULL,
    spend_cap_usd   REAL NOT NULL,
    spent_usd       REAL NOT NULL DEFAULT 0,
    seed            INTEGER NOT NULL,
    config_json     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contestants (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    name            TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('proposer', 'critic')),
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    settings_json   TEXT NOT NULL,
    position        INTEGER NOT NULL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS calls (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    input_id            TEXT NOT NULL,
    contestant          TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    attempts            INTEGER NOT NULL,
    raw_output          TEXT,
    response_model      TEXT,
    stop_reason         TEXT,
    parse_ok            INTEGER NOT NULL,
    validation_errors   TEXT,
    n_candidates        INTEGER,
    top_direction       TEXT,
    policy_verdict      TEXT,
    policy_reason       TEXT,
    standin_proposal_id TEXT,
    latency_ms          REAL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    thinking_tokens     INTEGER,
    cost_usd            REAL NOT NULL DEFAULT 0,
    error               TEXT,
    PRIMARY KEY (run_id, input_id, contestant)
);
CREATE TABLE IF NOT EXISTS standins (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    input_id            TEXT NOT NULL,
    contestant          TEXT NOT NULL,
    candidate_index     INTEGER NOT NULL,
    resolved            INTEGER NOT NULL,
    resolve_error       TEXT,
    standin_proposal_id TEXT,
    verdict             TEXT,
    failing_rule        TEXT,
    reason              TEXT,
    rule_outcomes_json  TEXT,
    PRIMARY KEY (run_id, input_id, contestant, candidate_index)
);
CREATE TABLE IF NOT EXISTS critiques (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    input_id            TEXT NOT NULL,
    candidate_index     INTEGER NOT NULL,
    standin_proposal_id TEXT,
    critic              TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('ok', 'invalid', 'error')),
    veto                INTEGER,
    reasons_json        TEXT,
    risk_flags_json     TEXT,
    confidence          REAL,
    fails_closed_veto   INTEGER NOT NULL,
    raw_output          TEXT,
    error               TEXT,
    attempts            INTEGER NOT NULL,
    latency_ms          REAL,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    thinking_tokens     INTEGER,
    cost_usd            REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, input_id, candidate_index)
);
"""


class ResultsError(Exception):
    """The results file cannot be used as asked."""


def open_results(path: str | Path, *, production_db: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating) the results DB; refuse the production store's file."""
    target = Path(path).expanduser().resolve()
    production = Path(production_db).expanduser().resolve() if production_db else resolve_db_path(None)
    if target == Path(production).resolve():
        raise ResultsError(f"{target} is the production audit DB; the bakeoff writes elsewhere")
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def as_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    """Insert one row; ``table`` is always one of this module's literals."""
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(row.values()))
