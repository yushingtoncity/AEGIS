-- migration: 0001 initial schema
--
-- Applied exactly once by aegis.store.db.migrate, which runs every statement
-- in this file inside one BEGIN IMMEDIATE ... COMMIT together with the row it
-- inserts into schema_version (version 1, name "0001_initial"). Once this
-- file has been applied anywhere it must never be edited: later schema
-- changes go in a new 0002_<name>.sql (and so on), applied in version order.
-- Every statement is idempotent (IF NOT EXISTS) so a re-run is harmless.
-- No BEGIN/COMMIT in this file: db.py owns the transaction.
--
-- Conventions: id TEXT PRIMARY KEY NOT NULL (uuid4; SQLite does not imply
-- NOT NULL for a non-INTEGER primary key, so a raw writer could otherwise
-- store NULL ids); timestamps TEXT (ISO-8601 with offset, always UTC); JSON
-- as TEXT; REAL for money and quantities; INTEGER for token counts. Enum
-- columns are CHECK-constrained to the string values of the matching
-- aegis.store.models enum. Every foreign key is indexed.

CREATE TABLE IF NOT EXISTS proposals (
    id               TEXT PRIMARY KEY NOT NULL,
    created_at       TEXT NOT NULL,
    cycle_id         TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    instrument       TEXT NOT NULL CHECK (instrument IN ('equity', 'option')),
    side             TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity         REAL NOT NULL CHECK (quantity > 0),
    order_type       TEXT NOT NULL CHECK (order_type IN ('market', 'limit')),
    limit_price      REAL,
    thesis           TEXT NOT NULL,
    confidence       REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    invalidation     TEXT NOT NULL,
    raw_model_output TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    prompt_version   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_proposals_created_at ON proposals (created_at);
CREATE INDEX IF NOT EXISTS ix_proposals_cycle_id ON proposals (cycle_id);

CREATE TABLE IF NOT EXISTS reasoning (
    id          TEXT PRIMARY KEY NOT NULL,
    proposal_id TEXT NOT NULL REFERENCES proposals (id),
    stage       TEXT NOT NULL CHECK (stage IN ('scan', 'thesis', 'proposal')),
    created_at  TEXT NOT NULL,
    content     TEXT NOT NULL,
    tokens_in   INTEGER,
    tokens_out  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_reasoning_proposal_id ON reasoning (proposal_id);

CREATE TABLE IF NOT EXISTS policy_decisions (
    id              TEXT PRIMARY KEY NOT NULL,
    proposal_id     TEXT NOT NULL REFERENCES proposals (id),
    decided_at      TEXT NOT NULL,
    verdict         TEXT NOT NULL
                    CHECK (verdict IN ('REJECT', 'FLAG_ONLY', 'NEEDS_APPROVAL', 'AUTO_EXECUTE')),
    rules_evaluated TEXT NOT NULL,
    failing_rule    TEXT,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS ix_policy_decisions_proposal_id ON policy_decisions (proposal_id);

CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY NOT NULL,
    proposal_id  TEXT NOT NULL REFERENCES proposals (id),
    requested_at TEXT NOT NULL,
    responded_at TEXT,
    response     TEXT CHECK (response IN ('approved', 'rejected', 'expired')),
    channel      TEXT NOT NULL,
    responder    TEXT
);
CREATE INDEX IF NOT EXISTS ix_approvals_proposal_id ON approvals (proposal_id);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY NOT NULL,
    proposal_id     TEXT NOT NULL REFERENCES proposals (id),
    client_order_id TEXT NOT NULL UNIQUE,
    broker          TEXT NOT NULL CHECK (broker IN ('paper', 'live')),
    broker_order_id TEXT,
    status          TEXT NOT NULL
                    CHECK (status IN ('proposed', 'gated', 'approved', 'submitted',
                                      'filled', 'partially_filled', 'cancelled', 'failed')),
    submitted_at    TEXT,
    updated_at      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity        REAL NOT NULL CHECK (quantity > 0),
    limit_price     REAL
);
CREATE INDEX IF NOT EXISTS ix_orders_proposal_id ON orders (proposal_id);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders (status);

CREATE TABLE IF NOT EXISTS fills (
    id            TEXT PRIMARY KEY NOT NULL,
    order_id      TEXT NOT NULL REFERENCES orders (id),
    filled_at     TEXT NOT NULL,
    fill_price    REAL NOT NULL,
    fill_quantity REAL NOT NULL CHECK (fill_quantity > 0),
    fees          REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_fills_order_id ON fills (order_id);

CREATE TABLE IF NOT EXISTS position_snapshots (
    id             TEXT PRIMARY KEY NOT NULL,
    taken_at       TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    quantity       REAL NOT NULL,
    avg_cost       REAL,
    market_value   REAL,
    unrealized_pnl REAL
);
CREATE INDEX IF NOT EXISTS ix_position_snapshots_taken_at ON position_snapshots (taken_at);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id             TEXT PRIMARY KEY NOT NULL,
    taken_at       TEXT NOT NULL,
    equity         REAL NOT NULL,
    cash           REAL NOT NULL,
    buying_power   REAL NOT NULL,
    daily_pnl      REAL NOT NULL,
    realized_pnl   REAL NOT NULL,
    unrealized_pnl REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pnl_snapshots_taken_at ON pnl_snapshots (taken_at);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY NOT NULL,
    occurred_at TEXT NOT NULL,
    level       TEXT NOT NULL CHECK (level IN ('debug', 'info', 'warning', 'error', 'critical')),
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    payload     TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_occurred_at ON events (occurred_at);
