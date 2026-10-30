"""Typed repository functions for the signal tables (migration 0006).

A fetch is written whole: ``record_fetch`` stores the attempt row and every
signal it brought in one transaction. Nothing here updates or deletes a row;
signals are a point-in-time record of what the brain could have seen.

The rate limiter's daily count and the reuse of a recent fetch both read
``signal_fetches``, so they hold across processes: every launchd run is a
fresh interpreter and an in-memory cache would forget everything.

These functions live apart from ``aegis.store.repo`` so the order path never
needs them: nothing under ``aegis/policy`` or ``aegis/execution`` imports
this module (``tests/test_signals_architecture.py`` pins it).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime

from aegis.store.db import transaction
from aegis.store.errors import StoreError
from aegis.store.models import SignalFetch, SignalFetchStatus, SignalRecord

_REPO_ERRORS = (sqlite3.Error, ValueError, TypeError, OverflowError)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dt(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def _row_to_fetch(row: sqlite3.Row) -> SignalFetch:
    return SignalFetch(
        id=row["id"],
        cycle_id=row["cycle_id"],
        source=row["source"],
        subject=row["subject"],
        endpoint=row["endpoint"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(row["finished_at"]),
        status=SignalFetchStatus(row["status"]),
        http_status=row["http_status"],
        error=row["error"],
        item_count=int(row["item_count"]),
    )


def _row_to_signal(row: sqlite3.Row) -> SignalRecord:
    return SignalRecord(
        id=row["id"],
        fetch_id=row["fetch_id"],
        source=row["source"],
        kind=row["kind"],
        subject=row["subject"],
        source_item_id=row["source_item_id"],
        endpoint=row["endpoint"],
        published_at=_dt(row["published_at"]),
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
        headline=row["headline"],
        summary=row["summary"],
        sentiment=row["sentiment"],
        sentiment_raw=row["sentiment_raw"],
        engagement=float(row["engagement"]),
        url=row["url"],
        confidence=float(row["confidence"]),
        raw_json=json.loads(row["raw_json"]),
    )


def record_fetch(
    conn: sqlite3.Connection, fetch: SignalFetch, signals: Sequence[SignalRecord] = ()
) -> SignalFetch:
    """Write ``fetch`` and all of ``signals`` (each must name ``fetch.id``) in one transaction."""
    strays = sorted({s.fetch_id for s in signals if s.fetch_id != fetch.id})
    if strays:
        raise StoreError("record signal fetch", fetch.id, ValueError(f"signals name other fetches: {strays}"))
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO signal_fetches (id, cycle_id, source, subject, endpoint, started_at,"
                " finished_at, status, http_status, error, item_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fetch.id,
                    fetch.cycle_id,
                    fetch.source,
                    fetch.subject,
                    fetch.endpoint,
                    _iso(fetch.started_at),
                    _iso(fetch.finished_at),
                    fetch.status.value,
                    fetch.http_status,
                    fetch.error,
                    fetch.item_count,
                ),
            )
            conn.executemany(
                "INSERT INTO signals (id, fetch_id, source, kind, subject, source_item_id, endpoint,"
                " published_at, fetched_at, headline, summary, sentiment, sentiment_raw, engagement,"
                " url, confidence, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.id,
                        s.fetch_id,
                        s.source,
                        s.kind,
                        s.subject,
                        s.source_item_id,
                        s.endpoint,
                        _iso(s.published_at),
                        _iso(s.fetched_at),
                        s.headline,
                        s.summary,
                        s.sentiment,
                        s.sentiment_raw,
                        s.engagement,
                        s.url,
                        s.confidence,
                        json.dumps(s.raw_json, allow_nan=False, default=str),
                    )
                    for s in signals
                ],
            )
    except _REPO_ERRORS as exc:
        raise StoreError("record signal fetch", fetch.id, exc) from exc
    return fetch


def latest_ok_fetch(
    conn: sqlite3.Connection, source: str, subject: str, since: datetime
) -> SignalFetch | None:
    """The newest successful fetch of ``source`` about ``subject`` that
    finished at or after ``since``, or None."""
    try:
        row = conn.execute(
            "SELECT * FROM signal_fetches WHERE source = ? AND subject = ? AND status = 'ok'"
            " AND finished_at >= ? ORDER BY finished_at DESC LIMIT 1",
            (source, subject, _iso(since)),
        ).fetchone()
    except sqlite3.Error as exc:
        raise StoreError("read signal fetches", f"{source}:{subject}", exc) from exc
    return _row_to_fetch(row) if row is not None else None


def requests_sent(conn: sqlite3.Connection, source: str, since: datetime) -> int:
    """Fetch attempts of ``source`` since ``since`` that reached (or tried to
    reach) the provider: every ``ok`` and ``error`` row, and a
    ``rate_limited`` row only when the provider itself answered (it has an
    ``http_status``). A refusal by our own limiter sent nothing."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM signal_fetches WHERE source = ? AND started_at >= ?"
            " AND (status IN ('ok', 'error') OR (status = 'rate_limited' AND http_status IS NOT NULL))",
            (source, _iso(since)),
        ).fetchone()
    except sqlite3.Error as exc:
        raise StoreError("count signal requests", source, exc) from exc
    return int(row[0])


def get_fetch_signals(conn: sqlite3.Connection, fetch_id: str) -> list[SignalRecord]:
    """Every signal one fetch stored, in insertion order."""
    try:
        rows = conn.execute(
            "SELECT * FROM signals WHERE fetch_id = ? ORDER BY rowid", (fetch_id,)
        ).fetchall()
        return [_row_to_signal(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("read signals", fetch_id, exc) from exc


def get_signals(conn: sqlite3.Connection, ids: Sequence[str]) -> list[SignalRecord]:
    """The signals with these ids, in the order given (unknown ids are skipped)."""
    if not ids:
        return []
    marks = ", ".join("?" for _ in ids)
    try:
        rows = conn.execute(f"SELECT * FROM signals WHERE id IN ({marks})", tuple(ids)).fetchall()
        by_id = {row["id"]: _row_to_signal(row) for row in rows}
    except _REPO_ERRORS as exc:
        raise StoreError("read signals", None, exc) from exc
    return [by_id[i] for i in ids if i in by_id]


def get_cycle_fetches(conn: sqlite3.Connection, cycle_id: str) -> list[SignalFetch]:
    """Every fetch attempt made during one brain cycle, oldest first."""
    try:
        rows = conn.execute(
            "SELECT * FROM signal_fetches WHERE cycle_id = ? ORDER BY started_at, rowid", (cycle_id,)
        ).fetchall()
        return [_row_to_fetch(row) for row in rows]
    except _REPO_ERRORS as exc:
        raise StoreError("read signal fetches", cycle_id, exc) from exc
