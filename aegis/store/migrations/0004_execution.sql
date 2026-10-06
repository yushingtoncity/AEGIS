-- migration: 0004 execution: orders tied to decisions, broker state, fills by broker id
--
-- Applied exactly once by aegis.store.db.migrate, which runs every statement
-- in this file inside one BEGIN IMMEDIATE ... COMMIT together with the row it
-- inserts into schema_version (version 4, name "0004_execution"), so the new
-- columns, indexes and triggers land together or not at all. Once this file
-- has been applied anywhere it must never be edited: later schema changes go
-- in a new 0005_<name>.sql. No BEGIN/COMMIT in this file: db.py owns the
-- transaction. (Each BEGIN ... END below is a trigger's body, not a
-- transaction; db.py's statement splitter keeps a trigger whole.)
--
-- Why: Phase 6 puts policy-approved orders on the paper account. Every
-- order must stay reconstructible (CLAUDE.md invariant 5): which decision
-- authorised it, which re-evaluation it passed just before it was sent,
-- which human approval (if any) it rests on, what the broker said, and every
-- fill. docs/phase6/SPEC_PHASE6.md has the full design.
--
-- ADD COLUMN only: the orders table is not rebuilt, because the foreign key
-- from fills would block the DROP. ADD COLUMN runs once (it has no IF NOT
-- EXISTS), which is safe because migrate never re-runs a recorded
-- migration. A column added with REFERENCES must default to NULL, so the
-- decision links are nullable; rows written before Phase 6 keep NULL there.
--
-- "Execution-era" rows are the orders with a decision_id: only
-- repo.claim_order writes one, and only repo.apply_broker_update moves it on.
-- Every trigger on orders below is limited to those rows, so an order
-- written before Phase 6 (through repo.upsert_order) behaves exactly as it
-- did. Likewise the approval triggers cover approvals tied to a decision.
--
-- The triggers are the store's own guard, under whatever the Python code
-- does: a status may only move forward, a terminal order is frozen, the
-- filled quantity only grows and never passes the order's quantity, and an
-- order's identity (what it buys, how much, at what limit, on whose
-- authority) never changes once claimed.

-- --- orders ----------------------------------------------------------------

ALTER TABLE orders ADD COLUMN decision_id TEXT REFERENCES policy_decisions (id);
ALTER TABLE orders ADD COLUMN regate_decision_id TEXT REFERENCES policy_decisions (id);
ALTER TABLE orders ADD COLUMN approval_id TEXT REFERENCES approvals (id);
ALTER TABLE orders ADD COLUMN instrument TEXT
    CHECK (instrument IS NULL OR instrument IN ('equity', 'option'));
ALTER TABLE orders ADD COLUMN order_type TEXT
    CHECK (order_type IS NULL OR order_type IN ('market', 'limit'));
ALTER TABLE orders ADD COLUMN time_in_force TEXT
    CHECK (time_in_force IS NULL OR time_in_force IN ('day'));
ALTER TABLE orders ADD COLUMN position_intent TEXT
    CHECK (position_intent IS NULL OR position_intent IN
           ('buy_to_open', 'buy_to_close', 'sell_to_open', 'sell_to_close'));
ALTER TABLE orders ADD COLUMN filled_quantity REAL NOT NULL DEFAULT 0
    CHECK (filled_quantity >= 0);
ALTER TABLE orders ADD COLUMN avg_fill_price REAL
    CHECK (avg_fill_price IS NULL OR avg_fill_price > 0);
ALTER TABLE orders ADD COLUMN broker_status TEXT;
ALTER TABLE orders ADD COLUMN status_reason TEXT;
ALTER TABLE orders ADD COLUMN last_synced_at TEXT;

CREATE INDEX IF NOT EXISTS ix_orders_decision_id ON orders (decision_id);
CREATE INDEX IF NOT EXISTS ix_orders_regate_decision_id ON orders (regate_decision_id);
CREATE INDEX IF NOT EXISTS ix_orders_approval_id ON orders (approval_id);
-- A broker order id names one claimed order, ever. (Rows written before
-- Phase 6 are left out, so this index can never fail on an existing store.)
CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_broker_order_id
    ON orders (broker_order_id) WHERE broker_order_id IS NOT NULL AND decision_id IS NOT NULL;
-- At most one placed order per proposal: re-evaluating a proposal whose
-- order already filled must not be able to place it again.
CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_one_per_proposal
    ON orders (proposal_id) WHERE decision_id IS NOT NULL;

-- A claimed order is born 'approved', with submitted_at set (it counts
-- toward the daily trade cap from that instant, so a crash cannot lose it),
-- nothing filled and no broker id yet.
CREATE TRIGGER IF NOT EXISTS orders_execution_born_approved
    BEFORE INSERT ON orders
    WHEN NEW.decision_id IS NOT NULL
     AND (NEW.status != 'approved' OR NEW.submitted_at IS NULL
          OR NEW.filled_quantity != 0 OR NEW.broker_order_id IS NOT NULL)
BEGIN
    SELECT RAISE(ABORT, 'a claimed order is born approved, with submitted_at set, nothing filled and no broker id');
END;

-- An order's decision link is set at birth and never changes, for any row.
CREATE TRIGGER IF NOT EXISTS orders_decision_link_fixed
    BEFORE UPDATE OF decision_id ON orders
    WHEN NEW.decision_id IS NOT OLD.decision_id
BEGIN
    SELECT RAISE(ABORT, 'an order''s decision_id never changes');
END;

-- A terminal order is frozen.
CREATE TRIGGER IF NOT EXISTS orders_execution_terminal_frozen
    BEFORE UPDATE ON orders
    WHEN OLD.decision_id IS NOT NULL AND OLD.status IN ('filled', 'cancelled', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'a filled, cancelled or failed order never changes');
END;

-- Status moves only forward.
CREATE TRIGGER IF NOT EXISTS orders_execution_status_forward
    BEFORE UPDATE OF status ON orders
    WHEN OLD.decision_id IS NOT NULL
     AND NEW.status IS NOT OLD.status
     AND NOT (
         (OLD.status = 'approved'
          AND NEW.status IN ('submitted', 'partially_filled', 'filled', 'cancelled', 'failed'))
      OR (OLD.status = 'submitted'
          AND NEW.status IN ('partially_filled', 'filled', 'cancelled', 'failed'))
      OR (OLD.status = 'partially_filled' AND NEW.status IN ('filled', 'cancelled'))
     )
BEGIN
    SELECT RAISE(ABORT, 'an order''s status only moves forward');
END;

-- The filled quantity only grows, and never past the order's quantity.
CREATE TRIGGER IF NOT EXISTS orders_execution_filled_quantity
    BEFORE UPDATE ON orders
    WHEN OLD.decision_id IS NOT NULL
     AND (NEW.filled_quantity < OLD.filled_quantity OR NEW.filled_quantity > NEW.quantity)
BEGIN
    SELECT RAISE(ABORT, 'an order''s filled quantity only grows, and never past its quantity');
END;

-- Failed means nothing was traded (a failed order does not count toward
-- the daily trade cap), so an order with anything filled never fails.
CREATE TRIGGER IF NOT EXISTS orders_execution_failed_unfilled
    BEFORE UPDATE ON orders
    WHEN OLD.decision_id IS NOT NULL AND NEW.status = 'failed' AND NEW.filled_quantity > 0
BEGIN
    SELECT RAISE(ABORT, 'an order with anything filled never fails');
END;

-- A claimed order is never deleted: it is the record of a trade (and of
-- the proposal's one order, and of a slot in the daily trade cap).
CREATE TRIGGER IF NOT EXISTS orders_execution_kept
    BEFORE DELETE ON orders
    WHEN OLD.decision_id IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'a claimed order is never deleted');
END;

-- What the order is, and on whose authority, never changes once claimed;
-- the broker's id is recorded at most once.
CREATE TRIGGER IF NOT EXISTS orders_execution_identity_fixed
    BEFORE UPDATE ON orders
    WHEN OLD.decision_id IS NOT NULL
     AND (NEW.proposal_id IS NOT OLD.proposal_id
          OR NEW.regate_decision_id IS NOT OLD.regate_decision_id
          OR NEW.approval_id IS NOT OLD.approval_id
          OR NEW.client_order_id IS NOT OLD.client_order_id
          OR NEW.broker IS NOT OLD.broker
          OR NEW.instrument IS NOT OLD.instrument
          OR NEW.symbol IS NOT OLD.symbol
          OR NEW.side IS NOT OLD.side
          OR NEW.quantity IS NOT OLD.quantity
          OR NEW.limit_price IS NOT OLD.limit_price
          OR NEW.order_type IS NOT OLD.order_type
          OR NEW.time_in_force IS NOT OLD.time_in_force
          OR NEW.position_intent IS NOT OLD.position_intent
          OR NEW.submitted_at IS NOT OLD.submitted_at
          OR (OLD.broker_order_id IS NOT NULL
              AND NEW.broker_order_id IS NOT OLD.broker_order_id))
BEGIN
    SELECT RAISE(ABORT, 'a claimed order''s identity never changes, and its broker id is set once');
END;

-- --- fills -----------------------------------------------------------------

-- The broker's own name for an execution; a replayed sync finds it taken.
ALTER TABLE fills ADD COLUMN broker_fill_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS ux_fills_broker_fill_id
    ON fills (broker_fill_id) WHERE broker_fill_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS fills_execution_price_positive
    BEFORE INSERT ON fills
    WHEN NEW.fill_price <= 0
     AND (SELECT decision_id FROM orders WHERE id = NEW.order_id) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'a fill of a claimed order has a positive price');
END;

-- A claimed order's fills add up to its filled quantity: they are only
-- ever added, never changed or removed.
CREATE TRIGGER IF NOT EXISTS fills_execution_fixed
    BEFORE UPDATE ON fills
    WHEN (SELECT decision_id FROM orders WHERE id = OLD.order_id) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'a fill of a claimed order never changes and is never deleted');
END;

CREATE TRIGGER IF NOT EXISTS fills_execution_kept
    BEFORE DELETE ON fills
    WHEN (SELECT decision_id FROM orders WHERE id = OLD.order_id) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'a fill of a claimed order never changes and is never deleted');
END;

-- --- approvals -------------------------------------------------------------

-- A human answer to one decision, valid until expires_at.
ALTER TABLE approvals ADD COLUMN decision_id TEXT REFERENCES policy_decisions (id);
ALTER TABLE approvals ADD COLUMN expires_at TEXT;
ALTER TABLE approvals ADD COLUMN note TEXT;
-- One answer per decision: a second "approve" cannot replace the first.
CREATE UNIQUE INDEX IF NOT EXISTS ux_approvals_decision_id
    ON approvals (decision_id) WHERE decision_id IS NOT NULL;

-- An approval answers a decision on its own proposal, so the proposal's
-- trace shows every approval an order of that proposal rests on.
CREATE TRIGGER IF NOT EXISTS approvals_decision_same_proposal
    BEFORE INSERT ON approvals
    WHEN NEW.decision_id IS NOT NULL
     AND (SELECT proposal_id FROM policy_decisions WHERE id = NEW.decision_id)
         IS NOT NEW.proposal_id
BEGIN
    SELECT RAISE(ABORT, 'an approval answers a decision on its own proposal');
END;

CREATE TRIGGER IF NOT EXISTS approvals_decision_link_fixed
    BEFORE UPDATE ON approvals
    WHEN NEW.decision_id IS NOT OLD.decision_id
      OR (OLD.decision_id IS NOT NULL
          AND (NEW.expires_at IS NOT OLD.expires_at OR NEW.proposal_id IS NOT OLD.proposal_id))
BEGIN
    SELECT RAISE(ABORT, 'an approval''s decision, proposal and expiry never change');
END;

-- Once a decision's approval is answered, the answer is final.
CREATE TRIGGER IF NOT EXISTS approvals_answer_once
    BEFORE UPDATE ON approvals
    WHEN OLD.decision_id IS NOT NULL AND OLD.response IS NOT NULL
     AND (NEW.response IS NOT OLD.response
          OR NEW.responded_at IS NOT OLD.responded_at
          OR NEW.responder IS NOT OLD.responder
          OR NEW.note IS NOT OLD.note)
BEGIN
    SELECT RAISE(ABORT, 'an approval is answered once');
END;

-- --- policy_decisions ------------------------------------------------------

-- 'evaluate': a verdict on a proposal. 'pre_submit': the re-evaluation
-- run just before an order is sent, on fresh data.
ALTER TABLE policy_decisions ADD COLUMN purpose TEXT NOT NULL DEFAULT 'evaluate'
    CHECK (purpose IN ('evaluate', 'pre_submit'));
