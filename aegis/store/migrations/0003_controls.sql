-- migration: 0003 operator controls (kill switch, daily-loss halt)
--
-- Applied exactly once by aegis.store.db.migrate, which runs every statement
-- in this file inside one BEGIN IMMEDIATE ... COMMIT together with the row it
-- inserts into schema_version (version 3, name "0003_controls"), so the
-- table, its two seed rows and the trigger land together or not at all.
-- Once this file has been applied anywhere it must never be edited: later
-- schema changes go in a new 0004_<name>.sql (and so on), applied in version
-- order. No BEGIN/COMMIT in this file: db.py owns the transaction. (The
-- BEGIN ... END below is the trigger's body, not a transaction; db.py's
-- statement splitter keeps a trigger whole.)
--
-- Why: the policy engine (Phase 5) must refuse every proposal while the
-- kill switch is on, and after the daily loss limit trips until the next
-- market open. Both flags live here rather than in memory so that they
-- survive a restart, and so that a P&L recovery later in the day does not
-- reopen trading. One row per control: key, value, and when it was last set.
--   kill_switch  'on' | 'off'
--   halt_until   '' (no halt) | the ISO-8601 instant trading may resume
--
-- Why each piece:
-- - CHECK on key: a typo'd key ('killswitch') fails loudly at the write
--   instead of landing as a row the engine never reads, which would leave
--   it blind to a control the operator believes is set. A new control
--   arrives in a later migration that widens this list.
-- - CHECKs on value: a state the engine cannot read ('maybe', 'tomorrow')
--   is unrepresentable for any normal writer. The GLOB pins the timestamp's
--   leading shape (YYYY-MM-DDTHH:MM:SS); repo.get_controls still parses the
--   whole value and fails closed on one that will not parse.
-- - The seed rows and the no-delete trigger: both rows exist from the moment
--   this migration lands and can never be deleted, so "the control is
--   missing" is not a state a reader has to interpret in normal operation.
--   A control is cleared by setting its value ('off', ''), never by DELETE.
--   INSERT OR IGNORE keeps a value already there, so re-running these
--   statements by hand can never switch a control back to its default.
--
-- repo.get_controls is the second line of defence: should either row be
-- missing or unreadable anyway (the table rebuilt by hand, the trigger
-- dropped), it reads the kill switch as ON and the halt as unknown.
--
-- updated_at is ISO-8601 UTC text like every other timestamp in the store;
-- the seeds use SQLite's clock (millisecond precision, explicit +00:00),
-- which datetime.fromisoformat reads back as an aware UTC value.

CREATE TABLE IF NOT EXISTS controls (
    key        TEXT PRIMARY KEY NOT NULL CHECK (key IN ('kill_switch', 'halt_until')),
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (key != 'kill_switch' OR value IN ('on', 'off')),
    CHECK (key != 'halt_until' OR value = ''
           OR value GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*')
);
INSERT OR IGNORE INTO controls (key, value, updated_at)
    VALUES ('kill_switch', 'off', strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));
INSERT OR IGNORE INTO controls (key, value, updated_at)
    VALUES ('halt_until', '', strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));
CREATE TRIGGER IF NOT EXISTS controls_keep_rows
    BEFORE DELETE ON controls
BEGIN
    SELECT RAISE(ABORT, 'controls rows are never deleted: set the value instead');
END;
