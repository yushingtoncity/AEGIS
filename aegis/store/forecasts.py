"""Typed repository functions for the forecast tables (migration 0005).

A forecast run is written whole: ``insert_forecast_run`` stores the run row
and every one of its forecasts in one transaction, so a run either has all
the probabilities the model gave or none. Nothing here updates or deletes a
row; forecasts are a point-in-time record and their outcomes are computed by
the scorer, never stored.

These functions live apart from ``aegis.store.repo`` so that the code which
can place orders never needs them: nothing under ``aegis/policy``,
``aegis/execution`` or ``aegis/cli/orders.py`` imports this module
(``tests/test_forecast_architecture.py`` pins it).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date, datetime

from aegis.store.db import transaction
from aegis.store.errors import StoreError
from aegis.store.models import ForecastArm, ForecastEvent, ForecastRecord, ForecastRun

_REPO_ERRORS = (sqlite3.Error, ValueError, TypeError, OverflowError)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _row_to_run(row: sqlite3.Row) -> ForecastRun:
    return ForecastRun(
        id=row["id"],
        arm=ForecastArm(row["arm"]),
        inputs_taken_at=datetime.fromisoformat(row["inputs_taken_at"]),
        completed_at=datetime.fromisoformat(row["completed_at"]),
        anchor_session=date.fromisoformat(row["anchor_session"]),
        benchmark=row["benchmark"],
        model_name=row["model_name"],
        requested_model=row["requested_model"],
        prompt_version=row["prompt_version"],
        event_version=row["event_version"],
        universe=tuple(json.loads(row["universe"])),
        inputs=json.loads(row["inputs"]),
    )


def _row_to_forecast(row: sqlite3.Row) -> ForecastRecord:
    return ForecastRecord(
        id=row["id"],
        run_id=row["run_id"],
        symbol=row["symbol"],
        event=ForecastEvent(row["event"]),
        horizon_sessions=int(row["horizon_sessions"]),
        probability=float(row["probability"]),
        rationale=row["rationale"],
    )


def insert_forecast_run(
    conn: sqlite3.Connection, run: ForecastRun, forecasts: Sequence[ForecastRecord]
) -> ForecastRun:
    """Write ``run`` and all of ``forecasts`` (each must name ``run.id``) in one transaction."""
    if not forecasts:
        raise StoreError("insert forecast run", run.id, ValueError("a run needs at least one forecast"))
    strays = sorted({f.run_id for f in forecasts if f.run_id != run.id})
    if strays:
        raise StoreError(
            "insert forecast run", run.id, ValueError(f"forecasts name other runs: {strays}")
        )
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO forecast_runs (id, arm, inputs_taken_at, completed_at, anchor_session,"
                " benchmark, model_name, requested_model, prompt_version, event_version, universe,"
                " inputs) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.arm.value,
                    _iso(run.inputs_taken_at),
                    _iso(run.completed_at),
                    run.anchor_session.isoformat(),
                    run.benchmark,
                    run.model_name,
                    run.requested_model,
                    run.prompt_version,
                    run.event_version,
                    json.dumps(list(run.universe)),
                    json.dumps(run.inputs, allow_nan=False),
                ),
            )
            conn.executemany(
                "INSERT INTO forecasts (id, run_id, symbol, event, horizon_sessions, probability,"
                " rationale) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        f.id,
                        f.run_id,
                        f.symbol,
                        f.event.value,
                        f.horizon_sessions,
                        f.probability,
                        f.rationale,
                    )
                    for f in forecasts
                ],
            )
    except _REPO_ERRORS as exc:
        raise StoreError("insert forecast run", run.id, exc) from exc
    return run


def get_forecast_runs(
    conn: sqlite3.Connection, since: date | None = None, until: date | None = None
) -> list[ForecastRun]:
    """Runs whose ``anchor_session`` is in ``[since, until)`` (either bound
    optional), oldest first."""
    clauses: list[str] = []
    params: list[str] = []
    if since is not None:
        clauses.append("anchor_session >= ?")
        params.append(since.isoformat())
    if until is not None:
        clauses.append("anchor_session < ?")
        params.append(until.isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        rows = conn.execute(
            "SELECT * FROM forecast_runs" + where + " ORDER BY anchor_session, completed_at, id",
            params,
        ).fetchall()
        return [_row_to_run(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get forecast runs", None, exc) from exc


def get_run_forecasts(conn: sqlite3.Connection, run_ids: Iterable[str]) -> dict[str, list[ForecastRecord]]:
    """Every forecast of each run in ``run_ids``, keyed by run id; a run
    with none (there should be none such) maps to an empty list."""
    result: dict[str, list[ForecastRecord]] = {run_id: [] for run_id in run_ids}
    try:
        for run_id in result:
            rows = conn.execute(
                "SELECT * FROM forecasts WHERE run_id = ? ORDER BY symbol, event, horizon_sessions",
                (run_id,),
            ).fetchall()
            result[run_id] = [_row_to_forecast(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("get run forecasts", None, exc) from exc
    return result


def get_runs_for_session(conn: sqlite3.Connection, anchor_session: date) -> list[ForecastRun]:
    """The runs already written for one trading date (so a second run the
    same day can be refused)."""
    return get_forecast_runs(conn, anchor_session, date.fromordinal(anchor_session.toordinal() + 1))
