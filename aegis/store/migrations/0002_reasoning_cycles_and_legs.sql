-- migration: 0002 reasoning rows per cycle, and proposal legs
--
-- Applied exactly once by aegis.store.db.migrate, which runs every statement
-- in this file inside one BEGIN IMMEDIATE ... COMMIT together with the row it
-- inserts into schema_version (version 2, name
-- "0002_reasoning_cycles_and_legs"). That atomicity is what makes the table
-- rebuild below safe: the copy, the DROP and the RENAME land together with
-- the version row or not at all, so a failure part-way leaves the 0001
-- reasoning table exactly as it was. Once this file has been applied
-- anywhere it must never be edited: later schema changes go in a new
-- 0003_<name>.sql (and so on), applied in version order.
-- No BEGIN/COMMIT in this file: db.py owns the transaction.
--
-- Why: 0001 declared reasoning.proposal_id NOT NULL, but the scan and thesis
-- stages run before any proposal exists and a NO_TRADE cycle never has one,
-- while the budget guard sums this table by UTC day and by cycle. SQLite
-- cannot relax a NOT NULL constraint in place, so the table is rebuilt:
-- copy into reasoning_v2 (cycle_id backfilled from the proposal, so every
-- existing row keeps its lineage), drop the old table (nothing references
-- reasoning, so foreign_keys=ON allows it), rename. A row must belong to a
-- cycle or a proposal — never neither. proposal_legs holds the typed legs
-- of an option proposal, one row per leg, ordered by leg_index.
--
-- The CREATE statements are idempotent (IF NOT EXISTS); the copy, DROP and
-- RENAME run exactly once because migrate never re-runs a recorded
-- migration, and they roll back with it if anything in this file fails.

CREATE TABLE IF NOT EXISTS reasoning_v2 (
    id          TEXT PRIMARY KEY NOT NULL,
    cycle_id    TEXT,
    proposal_id TEXT REFERENCES proposals (id),
    stage       TEXT NOT NULL CHECK (stage IN ('scan', 'thesis', 'proposal')),
    created_at  TEXT NOT NULL,
    content     TEXT NOT NULL,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    model_name  TEXT,
    latency_ms  REAL,
    CHECK (cycle_id IS NOT NULL OR proposal_id IS NOT NULL)
);
INSERT INTO reasoning_v2 (id, cycle_id, proposal_id, stage, created_at, content, tokens_in, tokens_out)
    SELECT r.id, p.cycle_id, r.proposal_id, r.stage, r.created_at, r.content, r.tokens_in, r.tokens_out
    FROM reasoning AS r LEFT JOIN proposals AS p ON p.id = r.proposal_id;
DROP TABLE reasoning;
ALTER TABLE reasoning_v2 RENAME TO reasoning;
CREATE INDEX IF NOT EXISTS ix_reasoning_proposal_id ON reasoning (proposal_id);
CREATE INDEX IF NOT EXISTS ix_reasoning_cycle_id ON reasoning (cycle_id);
CREATE INDEX IF NOT EXISTS ix_reasoning_created_at ON reasoning (created_at);

CREATE TABLE IF NOT EXISTS proposal_legs (
    id          TEXT PRIMARY KEY NOT NULL,
    proposal_id TEXT NOT NULL REFERENCES proposals (id),
    leg_index   INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),
    side        TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity    REAL NOT NULL CHECK (quantity > 0),
    strike      REAL NOT NULL CHECK (strike > 0),
    expiration  TEXT NOT NULL,
    UNIQUE (proposal_id, leg_index)
);
CREATE INDEX IF NOT EXISTS ix_proposal_legs_proposal_id ON proposal_legs (proposal_id);
