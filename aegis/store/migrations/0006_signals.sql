-- migration: 0006 signals: external news and social items shown to the scan stage, with provenance
--
-- Applied exactly once by aegis.store.db.migrate inside one BEGIN IMMEDIATE
-- ... COMMIT with its schema_version row (version 6, name "0006_signals").
-- Once applied anywhere it must never be edited: later changes go in a new
-- 0007_<name>.sql. No BEGIN/COMMIT here: db.py owns the transaction.
--
-- Why: aegis/signals fetches news and social posts as context for the scan
-- stage. Every fetch attempt is a signal_fetches row, good or bad, so a
-- degraded source is visible and the per-day request count the rate limiter
-- needs survives across processes (each launchd run is a new one). Every
-- item fetched is a signals row, before any filtering, with the provider's
-- own payload (raw_json), so what the brain could have seen is
-- reconstructible. Which items a cycle actually showed the model is a
-- signals_digest event in the existing events table.
--
-- Nothing on the order path reads these tables
-- (tests/test_signals_architecture.py pins it).

CREATE TABLE IF NOT EXISTS signal_fetches (
    id          TEXT PRIMARY KEY NOT NULL,
    cycle_id    TEXT,
    source      TEXT NOT NULL CHECK (source IN ('news', 'apewisdom')),
    subject     TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'cached', 'rate_limited', 'error', 'disabled')),
    http_status INTEGER,
    error       TEXT,
    item_count  INTEGER NOT NULL DEFAULT 0 CHECK (item_count >= 0)
);
CREATE INDEX IF NOT EXISTS ix_signal_fetches_source_started ON signal_fetches (source, started_at);
CREATE INDEX IF NOT EXISTS ix_signal_fetches_subject ON signal_fetches (source, subject, finished_at);

CREATE TABLE IF NOT EXISTS signals (
    id             TEXT PRIMARY KEY NOT NULL,
    fetch_id       TEXT NOT NULL REFERENCES signal_fetches (id),
    source         TEXT NOT NULL CHECK (source IN ('news', 'apewisdom')),
    kind           TEXT NOT NULL CHECK (kind IN ('news', 'mentions')),
    subject        TEXT NOT NULL,
    source_item_id TEXT NOT NULL,
    endpoint       TEXT NOT NULL,
    published_at   TEXT,
    fetched_at     TEXT NOT NULL,
    headline       TEXT NOT NULL,
    summary        TEXT,
    sentiment      REAL CHECK (sentiment IS NULL OR sentiment BETWEEN -1.0 AND 1.0),
    sentiment_raw  REAL,
    engagement     REAL NOT NULL CHECK (engagement >= 0),
    url            TEXT,
    confidence     REAL NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    raw_json       TEXT NOT NULL,
    UNIQUE (fetch_id, source_item_id)
);
CREATE INDEX IF NOT EXISTS ix_signals_fetch_id ON signals (fetch_id);
CREATE INDEX IF NOT EXISTS ix_signals_subject_published ON signals (subject, published_at);
