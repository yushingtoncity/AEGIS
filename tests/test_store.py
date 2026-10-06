"""Persistence layer: connection, migrations, schema, models, repository — tmp_path databases only.

The first half writes rows with parameterized SQL directly to pin the
contract the repository builds on: WAL semantics, atomic migrations, schema
constraints, and model round-trips. ``TestRepo`` then exercises the typed
repository functions end to end from the ``proposal_trace.json`` fixture, and
``TestControls`` migration 0003's ``controls`` table with the functions that
read and set the kill switch and the daily-loss halt.
"""

import ast
import io
import json
import re
import sqlite3
import subprocess
import sys
import time
import types
import typing
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import aegis.store
from aegis.cli import db as cli_db
from aegis.cli import trace as cli_trace
from aegis.config import REPO_ROOT, get_config
from aegis.data.models import OptionType
from aegis.store import db as store_db
from aegis.store import models as store_models
from aegis.store import repo
from aegis.store.db import (
    MIGRATIONS_DIR,
    TABLES,
    applied_versions,
    connect,
    list_migrations,
    migrate,
    open_store,
    resolve_db_path,
    schema_version,
    status,
    transaction,
)
from aegis.store.errors import StoreError
from aegis.store.models import (
    OPEN_ORDER_STATUSES,
    Approval,
    ApprovalResponse,
    Broker,
    Controls,
    Event,
    EventLevel,
    Fill,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PnlSnapshot,
    PolicyDecision,
    PositionSnapshot,
    Proposal,
    ProposalLeg,
    ProposalTrace,
    Reasoning,
    ReasoningStage,
    StoreStatus,
    TokenUsage,
    Verdict,
    new_id,
)
from aegis.store.repo import (
    add_reasoning,
    count_orders_submitted_between,
    get_controls,
    get_cycle_reasoning,
    get_cycle_token_usage,
    get_daily_pnl,
    get_latest_undecided_proposal,
    get_open_orders,
    get_order,
    get_proposal,
    get_proposal_legs,
    get_proposal_trace,
    get_proposals_since,
    get_recent_events,
    get_recent_proposals,
    get_token_usage,
    get_token_usage_by_model,
    insert_proposal,
    link_reasoning_to_proposal,
    log_event,
    record_approval,
    record_decision,
    record_fill,
    set_halt_until,
    set_kill_switch,
    snapshot_pnl,
    snapshot_positions,
    upsert_order,
)

NAIVE = datetime(2026, 7, 30, 14, 0, 0)
AWARE = NAIVE.replace(tzinfo=timezone.utc)
LATEST = 4
"""The newest migration: 0004_execution (Phase 6)."""


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "t.db"


@pytest.fixture
def conn(db_path):
    conn = open_store(db_path)
    yield conn
    conn.close()


@pytest.fixture
def trace_data(fixture):
    return fixture("proposal_trace.json")


def _sql_values(record):
    """A record's fields as SQLite values: enums → str, datetimes → ISO text."""
    values = []
    for value in record.model_dump().values():
        if isinstance(value, datetime):
            value = value.isoformat()
        values.append(getattr(value, "value", value))
    return tuple(values)


def _insert_event(conn, event_id="evt-0001", level="info"):
    conn.execute(
        "INSERT INTO events (id, occurred_at, level, kind, message, payload)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, AWARE.isoformat(), level, "test", "synthetic", None),
    )


def _insert_proposal(conn, proposal, **overrides):
    """Insert a proposal row; ``overrides`` replace column values (to hit CHECKs)."""
    row = dict(zip(proposal.model_dump(), _sql_values(proposal)))
    row.update(overrides)
    conn.execute(
        "INSERT INTO proposals (id, created_at, cycle_id, symbol, instrument, side, quantity,"
        " order_type, limit_price, thesis, confidence, invalidation, raw_model_output,"
        " model_name, prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(row.values()),
    )


_LEGACY_ORDER_COLUMNS = (
    "id", "proposal_id", "client_order_id", "broker", "broker_order_id", "status",
    "submitted_at", "updated_at", "symbol", "side", "quantity", "limit_price",
)
"""The orders columns of migration 0001: what an order written before Phase 6 has."""


def _insert_order(conn, order, **overrides):
    """Insert an order row as Phase 3 wrote it (the 0001 columns; 0004's keep their defaults);
    ``overrides`` replace column values (to hit CHECKs)."""
    row = dict(zip(order.model_dump(), _sql_values(order)))
    row.update(overrides)
    conn.execute(
        "INSERT INTO orders (id, proposal_id, client_order_id, broker, broker_order_id, status,"
        " submitted_at, updated_at, symbol, side, quantity, limit_price)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(row[column] for column in _LEGACY_ORDER_COLUMNS),
    )


def _schema_rows(conn):
    """Every schema_version row in full, so a re-applied migration cannot hide behind a count."""
    rows = conn.execute("SELECT rowid, version, name, applied_at FROM schema_version").fetchall()
    return [tuple(row) for row in rows]


# The test suite's own per-table counts: literal SQL, never the store's query
# table, so a wrong count query in db.py cannot vouch for itself.
_COUNT_SQL = {
    "proposals": "SELECT COUNT(*) FROM proposals",
    "proposal_legs": "SELECT COUNT(*) FROM proposal_legs",
    "reasoning": "SELECT COUNT(*) FROM reasoning",
    "policy_decisions": "SELECT COUNT(*) FROM policy_decisions",
    "approvals": "SELECT COUNT(*) FROM approvals",
    "orders": "SELECT COUNT(*) FROM orders",
    "fills": "SELECT COUNT(*) FROM fills",
    "position_snapshots": "SELECT COUNT(*) FROM position_snapshots",
    "pnl_snapshots": "SELECT COUNT(*) FROM pnl_snapshots",
    "events": "SELECT COUNT(*) FROM events",
    "controls": "SELECT COUNT(*) FROM controls",
}

# A freshly migrated store holds nothing but the two rows 0003 seeds in ``controls``.
_FRESH_COUNTS = {
    "proposals": 0, "proposal_legs": 0, "reasoning": 0, "policy_decisions": 0, "approvals": 0, "orders": 0,
    "fills": 0, "position_snapshots": 0, "pnl_snapshots": 0, "events": 0, "controls": 2,
}


def _count(conn, table):
    return conn.execute(_COUNT_SQL[table]).fetchone()[0]


def _bare_proposal(proposal_id):
    """The smallest valid proposal, for tests that need one without the fixture."""
    return Proposal(
        id=proposal_id, cycle_id="cycle-x", symbol="SPY", instrument=Instrument.EQUITY, side=OrderSide.BUY,
        quantity=1, order_type=OrderType.MARKET, thesis="synthetic", confidence=0.5, invalidation="synthetic",
        raw_model_output="{}", model_name="synthetic-model-v0", prompt_version="test-prompt-1",
    )


def _bare_order(order_id, client_order_id, **overrides):
    """A filled paper order on prop-0001 (the fixture proposal); ``overrides`` replace fields."""
    fields = {
        "id": order_id, "proposal_id": "prop-0001", "client_order_id": client_order_id, "broker": Broker.PAPER,
        "status": OrderStatus.FILLED, "symbol": "SPY", "side": OrderSide.BUY, "quantity": 1,
    }
    return Order(**{**fields, **overrides})


def _legs(proposal_id, count=2):
    """``count`` legs of an Oct-16 call vertical on ``proposal_id``: buy 640, sell 650 (660 …), by index."""
    return [
        ProposalLeg(
            id=f"leg-{proposal_id}-{n}", proposal_id=proposal_id, leg_index=n, symbol=f"SPY261016C00{640 + 10 * n}000",
            option_type=OptionType.CALL, side=OrderSide.BUY if n == 0 else OrderSide.SELL, quantity=1,
            strike=640 + 10 * n, expiration=date(2026, 10, 16),
        )
        for n in range(count)
    ]


def _optional(annotation):
    """Whether a model field's annotation admits None (``X | None``)."""
    origin = typing.get_origin(annotation)
    return origin in (types.UnionType, typing.Union) and type(None) in typing.get_args(annotation)


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


# Every row of every table that existed before 0003, verbatim (literal SQL again).
_ROWS_SQL = {
    "proposals": "SELECT * FROM proposals ORDER BY rowid",
    "proposal_legs": "SELECT * FROM proposal_legs ORDER BY rowid",
    "reasoning": "SELECT * FROM reasoning ORDER BY rowid",
    "policy_decisions": "SELECT * FROM policy_decisions ORDER BY rowid",
    "approvals": "SELECT * FROM approvals ORDER BY rowid",
    "orders": "SELECT * FROM orders ORDER BY rowid",
    "fills": "SELECT * FROM fills ORDER BY rowid",
    "position_snapshots": "SELECT * FROM position_snapshots ORDER BY rowid",
    "pnl_snapshots": "SELECT * FROM pnl_snapshots ORDER BY rowid",
    "events": "SELECT * FROM events ORDER BY rowid",
}


def _journal_rows(conn):
    return {table: [tuple(row) for row in conn.execute(sql).fetchall()] for table, sql in _ROWS_SQL.items()}


def _control_rows(conn):
    """The ``controls`` table verbatim, by key: ``[(key, value, updated_at), …]``."""
    rows = conn.execute("SELECT key, value, updated_at FROM controls ORDER BY key").fetchall()
    return [tuple(row) for row in rows]


def _triggers(conn, table=None):
    """Every trigger as ``(name, table)``, by name; only ``table``'s when given."""
    rows = conn.execute("SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger' ORDER BY name").fetchall()
    return [tuple(row) for row in rows if table is None or row[1] == table]


def _traced(conn, call):
    """Run ``call`` and return the transaction-shaping statements it executed, in order:
    BEGIN / COMMIT / ROLLBACK as issued, and each INSERT as ``INSERT INTO <table>``."""
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        call()
    finally:
        conn.set_trace_callback(None)
    shape = []
    for statement in statements:
        words = statement.split()
        if words[0] in ("BEGIN", "COMMIT", "ROLLBACK"):
            shape.append(" ".join(words))
        elif words[0] == "INSERT":
            shape.append(" ".join(words[:3]))
    return shape


# Run in a subprocess by TestConnect.test_two_processes_can_create_the_same_fresh_db:
# report readiness, spin at the starting gate (a sleep would stagger the two
# openers by more than the race window), then open the store and write.
_RACE_WORKER = """
import sys
from pathlib import Path
from aegis.store import Event, EventLevel, log_event, open_store
db, gate, ready = sys.argv[1:4]
Path(ready).touch()
while not Path(gate).exists():
    pass
conn = open_store(db)
for n in range(20):
    log_event(conn, Event(level=EventLevel.INFO, kind="race", message=str(n)))
conn.close()
print("ok")
"""


class TestConnect:
    def test_open_store_file_db(self, conn, db_path):
        assert db_path.exists()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL, the WAL pairing
        assert schema_version(conn) == LATEST
        assert _table_names(conn) == set(TABLES) | {"schema_version"}
        assert len(TABLES) == 11 and TABLES[-1] == "controls"
        assert conn.row_factory is sqlite3.Row
        assert conn.isolation_level is None

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "data" / "deeper" / "t.db"
        conn = open_store(nested)
        conn.close()
        assert nested.exists()

    def test_memory_db(self):
        conn = connect(":memory:")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "memory"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert migrate(conn) == LATEST
        assert _table_names(conn) == set(TABLES) | {"schema_version"}
        conn.close()

    def test_resolve_db_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # relative paths follow the repo root, never the CWD
        assert resolve_db_path("data/x.db") == REPO_ROOT / "data" / "x.db"
        assert resolve_db_path(tmp_path / "abs.db") == tmp_path / "abs.db"
        assert resolve_db_path(":memory:").name == ":memory:"
        assert resolve_db_path("~/x.db").is_absolute() and "~" not in str(resolve_db_path("~/x.db"))
        default = resolve_db_path()
        assert default.is_absolute()
        assert default == REPO_ROOT / get_config().store.db_path

    def test_unwritable_location_is_store_error(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(StoreError, match="connect") as info:
            connect(blocker / "t.db")
        assert isinstance(info.value.cause, OSError)
        assert str(blocker / "t.db") in str(info.value)

    def test_reader_sees_committed_state_while_writer_holds_lock(self, db_path):
        """WAL: the future dashboard reads while the loop is mid-transaction."""
        writer = open_store(db_path)
        reader = connect(db_path)
        try:
            with transaction(writer):
                _insert_event(writer)
                assert writer.in_transaction
                assert db_path.with_name("t.db-wal").exists()
                assert _count(reader, "events") == 0
                late = connect(db_path)  # even a brand-new connection gets in
                assert late.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
                assert _count(late, "events") == 0
                late.close()
                # a competing writer waits busy_timeout, then fails: raw here, so that
                # the repository can wrap it with the operation and id
                reader.execute("PRAGMA busy_timeout=50")
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    with transaction(reader):
                        pass
                assert not reader.in_transaction
            assert _count(reader, "events") == 1
        finally:
            writer.close()
            reader.close()

    def test_two_processes_can_create_the_same_fresh_db(self, tmp_path):
        """Switching a new file to WAL is the one statement busy_timeout does not cover."""
        db_path = tmp_path / "t.db"
        gate = tmp_path / "go"
        workers = []
        for n in range(2):
            ready = tmp_path / f"ready-{n}"
            proc = subprocess.Popen(
                [sys.executable, "-c", _RACE_WORKER, str(db_path), str(gate), str(ready)],
                cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            workers.append((proc, ready))
        deadline = time.monotonic() + 60
        while not all(ready.exists() for _, ready in workers) and time.monotonic() < deadline:
            time.sleep(0.005)
        gate.touch()  # both workers open the not-yet-existing database at once
        results = [proc.communicate(timeout=60) for proc, _ in workers]
        for (proc, _), (out, err) in zip(workers, results):
            assert proc.returncode == 0 and out.strip() == "ok", err
        conn = connect(db_path)
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert [row[:3] for row in _schema_rows(conn)] == [  # migrated once
            (1, 1, "0001_initial"), (2, 2, "0002_reasoning_cycles_and_legs"), (3, 3, "0003_controls"), (4, 4, "0004_execution"),
        ]
        assert _count(conn, "events") == 40
        conn.close()


class TestTransaction:
    def test_commit_on_success(self, conn):
        with transaction(conn):
            _insert_event(conn)
        assert not conn.in_transaction
        assert _count(conn, "events") == 1

    def test_rollback_on_exception(self, conn):
        with pytest.raises(RuntimeError, match="boom"):
            with transaction(conn):
                _insert_event(conn)
                raise RuntimeError("boom")
        assert not conn.in_transaction
        assert _count(conn, "events") == 0

    def test_sqlite_errors_pass_through_unwrapped(self, conn):
        """The repository wraps these with the operation and id; transaction() must not."""
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(conn):
                _insert_event(conn, level="loud")  # violates the CHECK constraint
        assert not conn.in_transaction
        assert _count(conn, "events") == 0

    def test_not_reentrant(self, conn):
        with pytest.raises(StoreError, match="not re-entrant"):
            with transaction(conn):
                _insert_event(conn)
                with transaction(conn):
                    pass
        assert not conn.in_transaction
        assert _count(conn, "events") == 0  # the outer block rolled back
        with transaction(conn):  # and the connection is usable afterwards
            _insert_event(conn)
        assert _count(conn, "events") == 1


class TestMigrations:
    def test_list_migrations(self):
        listed = [(version, name, path.name) for version, name, path in list_migrations()]
        assert listed == [
            (1, "0001_initial", "0001_initial.sql"),
            (2, "0002_reasoning_cycles_and_legs", "0002_reasoning_cycles_and_legs.sql"),
            (3, "0003_controls", "0003_controls.sql"),
            (4, "0004_execution", "0004_execution.sql"),
        ]
        assert all(path.parent == MIGRATIONS_DIR for _, _, path in list_migrations())

    def test_initial_file_leaves_transactions_to_db_py(self):
        text = (MIGRATIONS_DIR / "0001_initial.sql").read_text(encoding="utf-8")
        assert text.startswith("-- migration: 0001 initial schema")
        assert re.search(r"^\s*(BEGIN|COMMIT|END)\b", text, re.IGNORECASE | re.MULTILINE) is None
        assert text.count("CREATE TABLE IF NOT EXISTS") == 9
        assert "CREATE TABLE " not in text.replace("CREATE TABLE IF NOT EXISTS", "")
        assert "CREATE INDEX " not in text.replace("CREATE INDEX IF NOT EXISTS", "")

    def test_rebuild_file_leaves_transactions_to_db_py(self):
        """0002 rebuilds ``reasoning`` (copy, DROP, RENAME) and relies on db.py's transaction for safety."""
        text = (MIGRATIONS_DIR / "0002_reasoning_cycles_and_legs.sql").read_text(encoding="utf-8")
        assert text.startswith("-- migration: 0002 ")
        assert "atomic" in text  # the header says the one transaction is what makes the rebuild safe
        assert re.search(r"^\s*(BEGIN|COMMIT|END)\b", text, re.IGNORECASE | re.MULTILINE) is None
        assert text.count("CREATE TABLE IF NOT EXISTS") == 2
        assert "CREATE TABLE " not in text.replace("CREATE TABLE IF NOT EXISTS", "")
        assert "CREATE INDEX " not in text.replace("CREATE INDEX IF NOT EXISTS", "")
        body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
        statements = [" ".join(chunk.split()) for chunk in store_db._split_statements(body)]
        assert len(statements) == 9
        assert statements[2] == "DROP TABLE reasoning;"
        assert statements[3] == "ALTER TABLE reasoning_v2 RENAME TO reasoning;"
        # the multi-line copy is one statement, before the DROP, and backfills cycle_id from the proposal
        assert statements[1].startswith("INSERT INTO reasoning_v2 (id, cycle_id, proposal_id,")
        assert "SELECT r.id, p.cycle_id, r.proposal_id," in statements[1]
        assert statements[1].endswith("FROM reasoning AS r LEFT JOIN proposals AS p ON p.id = r.proposal_id;")

    def test_applying_twice_is_a_noop(self, conn, db_path):
        assert migrate(conn) == LATEST
        before = _schema_rows(conn)
        assert len(before) == LATEST
        assert migrate(conn) == LATEST
        assert _schema_rows(conn) == before  # the same rows, byte for byte: nothing re-applied
        assert applied_versions(conn) == {
            1: "0001_initial", 2: "0002_reasoning_cycles_and_legs", 3: "0003_controls", 4: "0004_execution",
        }
        for (applied_at,) in conn.execute("SELECT applied_at FROM schema_version").fetchall():
            assert datetime.fromisoformat(applied_at).tzinfo is not None
        reopened = open_store(db_path)
        assert schema_version(reopened) == LATEST
        assert _schema_rows(reopened) == before
        reopened.close()

    def test_refuses_newer_schema(self, conn, db_path):
        conn.execute(
            "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
            (99, "0099_from_the_future", AWARE.isoformat()),
        )
        with pytest.raises(StoreError, match="99"):
            migrate(conn)
        with pytest.raises(StoreError, match="newer"):
            open_store(db_path)

    def test_empty_db_reports_version_zero(self, db_path):
        conn = connect(db_path)
        assert applied_versions(conn) == {}
        assert schema_version(conn) == 0
        assert "schema_version" not in _table_names(conn)
        conn.close()

    @pytest.mark.parametrize(
        "names",
        [
            ["bad.sql"],
            ["0001_initial.sql", "0001_again.sql"],
            ["0000_zero.sql"],
            ["01_short.sql"],
            ["0002_Mixed-Case.sql"],
        ],
    )
    def test_malformed_migration_names(self, tmp_path, monkeypatch, names):
        for name in names:
            (tmp_path / name).write_text("CREATE TABLE IF NOT EXISTS x (id TEXT);\n", encoding="utf-8")
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", tmp_path)
        with pytest.raises(StoreError, match="list migrations"):
            list_migrations()

    def test_failed_migration_rolls_back_ddl_and_version_row(self, tmp_path, monkeypatch, db_path):
        """DDL + the schema_version row commit together or not at all."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        initial = MIGRATIONS_DIR / "0001_initial.sql"
        (migrations / initial.name).write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
        broken = migrations / "0002_broken.sql"
        broken.write_text(
            "-- migration: 0002 broken\n"
            "CREATE TABLE IF NOT EXISTS extra (id TEXT PRIMARY KEY);\n"
            "INSERT INTO no_such_table (x) VALUES (1);\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(db_path)
        with pytest.raises(StoreError, match="0002_broken") as info:
            migrate(conn)
        assert isinstance(info.value.cause, sqlite3.OperationalError)
        assert not conn.in_transaction
        assert schema_version(conn) == 1  # 0001 landed, 0002 did not
        assert "extra" not in _table_names(conn)

        broken.unlink()
        (migrations / "0002_extra.sql").write_text(
            "CREATE TABLE IF NOT EXISTS extra (id TEXT PRIMARY KEY);\n", encoding="utf-8"
        )
        assert migrate(conn) == 2
        assert applied_versions(conn) == {1: "0001_initial", 2: "0002_extra"}
        assert "extra" in _table_names(conn)
        conn.close()

    def test_semicolons_inside_literals_and_comments_do_not_split(self, tmp_path, monkeypatch):
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        (migrations / "0001_literals.sql").write_text(
            "-- header; with a semicolon\n"
            "CREATE TABLE IF NOT EXISTS notes (\n"
            "    id TEXT PRIMARY KEY,\n"
            "    tag TEXT NOT NULL CHECK (tag IN ('a;b', 'c'))  -- ';' in a literal\n"
            ");\n"
            "CREATE INDEX IF NOT EXISTS ix_notes_tag ON notes (tag)\n"  # no trailing ';'
            "-- trailing comment; nothing after it\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(":memory:")
        assert migrate(conn) == 1
        conn.execute("INSERT INTO notes (id, tag) VALUES (?, ?)", ("n1", "a;b"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO notes (id, tag) VALUES (?, ?)", ("n2", "zzz"))
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = ?", ("ix_notes_tag",)).fetchone()[0] == 1
        conn.close()

    def test_two_statements_on_one_line_are_two_statements(self, tmp_path, monkeypatch):
        """Chunks end where a statement ends, not where a line does — as executescript sees them."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        script = (
            "-- migration: 0001 one line\n"
            "CREATE TABLE IF NOT EXISTS a (id TEXT PRIMARY KEY); CREATE TABLE IF NOT EXISTS b (id TEXT PRIMARY KEY);\n"
            "INSERT INTO a (id) VALUES ('x;y'); INSERT INTO b (id) VALUES ('z')\n"
        )
        (migrations / "0001_one_line.sql").write_text(script, encoding="utf-8")
        chunks = store_db._split_statements(script)
        assert len(chunks) == 4 and "".join(chunks) == script
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(":memory:")
        assert migrate(conn) == 1
        assert conn.execute("SELECT id FROM a").fetchone()[0] == "x;y"
        assert conn.execute("SELECT id FROM b").fetchone()[0] == "z"
        conn.close()

    def test_apply_migration_rechecks_inside_the_write_lock(self, conn, monkeypatch):
        """Another process applied the migration between the outer check and BEGIN IMMEDIATE."""
        real = store_db.applied_versions
        # the outer check sees nothing applied; the one inside the write lock sees the truth
        monkeypatch.setattr(store_db, "applied_versions", lambda c: real(c) if c.in_transaction else {})
        before = _schema_rows(conn)
        migrate(conn)  # must not re-run the DDL or hit the UNIQUE version row
        assert not conn.in_transaction
        assert _schema_rows(conn) == before

    def test_dropped_table_is_reported_not_recreated(self, conn, db_path, capsys):
        """A recorded migration is never re-run: status and both CLIs say what is missing instead."""
        conn.execute("DROP TABLE events")
        assert migrate(conn) == LATEST
        assert "events" not in _table_names(conn)
        report = status(conn, db_path)
        assert report.pending_migrations == () and report.missing_tables == ("events",)
        assert report.row_counts["events"] == 0
        with pytest.raises(StoreError, match="log event for evt-gone") as info:
            log_event(conn, Event(id="evt-gone", level=EventLevel.INFO, kind="k", message="m"))
        assert "no such table: events" in str(info.value)

        assert cli_db.main(["status", "--db", str(db_path)]) == 1
        out = capsys.readouterr().out
        assert re.search(r"^  events\s+missing$", out, re.MULTILINE)
        assert re.search(r"^  orders\s+0$", out, re.MULTILINE)
        assert "missing tables: events" in out and "never re-run" in out
        assert cli_db.main(["init", "--db", str(db_path)]) == 1
        assert "missing tables: events" in capsys.readouterr().out
        assert "events" not in _table_names(conn)
        assert status(conn, db_path).missing_tables == ("events",)

    def test_upgrading_a_v1_database_keeps_reasoning_and_adds_legs(self, db_path, trace_data):
        """0002 rebuilds ``reasoning`` in place: every row survives with cycle_id backfilled from its proposal."""
        conn = connect(db_path)
        conn.execute(store_db._SCHEMA_VERSION_DDL)
        version, name, path = list_migrations()[0]
        store_db._apply_migration(conn, version, name, path)
        assert schema_version(conn) == 1 and "proposal_legs" not in _table_names(conn)
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", ("reasoning",)).fetchone()[0]
        assert "proposal_id TEXT NOT NULL" in ddl  # the 0001 shape
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        _insert_proposal(conn, _bare_proposal("prop-0002"))
        for n, row in enumerate(trace_data["reasoning"]):
            conn.execute(
                "INSERT INTO reasoning (id, proposal_id, stage, created_at, content, tokens_in, tokens_out)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row["id"], row["proposal_id"], row["stage"], row["created_at"], row["content"], row["tokens_in"],
                 None if n == 2 else row["tokens_out"]),
            )
        conn.execute(
            "INSERT INTO reasoning (id, proposal_id, stage, created_at, content) VALUES (?, ?, ?, ?, ?)",
            ("reas-0002", "prop-0002", "scan", AWARE.isoformat(), "other"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):  # the 0001 constraint 0002 lifts
            conn.execute(
                "INSERT INTO reasoning (id, proposal_id, stage, created_at, content) VALUES (?, NULL, ?, ?, ?)",
                ("x", "scan", AWARE.isoformat(), "c"),
            )

        assert migrate(conn) == LATEST  # 0002, and 0003 and 0004 after it
        assert not conn.in_transaction
        assert applied_versions(conn) == {
            1: "0001_initial", 2: "0002_reasoning_cycles_and_legs", 3: "0003_controls", 4: "0004_execution",
        }
        assert _table_names(conn) == set(TABLES) | {"schema_version"}  # reasoning_v2 is gone, proposal_legs is there
        rows = conn.execute(
            "SELECT id, cycle_id, proposal_id, stage, tokens_in, tokens_out, model_name, latency_ms"
            " FROM reasoning ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("reas-0001-proposal", "cycle-0001", "prop-0001", "proposal", 1700, None, None, None),
            ("reas-0001-scan", "cycle-0001", "prop-0001", "scan", 1200, 180, None, None),
            ("reas-0001-thesis", "cycle-0001", "prop-0001", "thesis", 1500, 220, None, None),
            ("reas-0002", "cycle-x", "prop-0002", "scan", None, None, None, None),
        ]
        trace = get_proposal_trace(conn, "prop-0001")
        assert [r.stage for r in trace.reasoning] == [
            ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL,
        ]
        assert {r.cycle_id for r in trace.reasoning} == {"cycle-0001"} and trace.legs == ()
        assert get_cycle_reasoning(conn, "cycle-x") == [Reasoning(
            id="reas-0002", cycle_id="cycle-x", proposal_id="prop-0002", stage=ReasoningStage.SCAN,
            created_at=AWARE, content="other",
        )]
        assert get_token_usage(conn, date(2026, 7, 30)) == TokenUsage(tokens_in=4400, tokens_out=400, calls=4)
        # the rebuilt table takes a cycle-only row, keeps the foreign key, and refuses a row under neither
        add_reasoning(conn, Reasoning(id="reas-cycle-only", cycle_id="cycle-new", stage=ReasoningStage.SCAN, content="c"))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO reasoning (id, cycle_id, proposal_id, stage, created_at, content) VALUES (?, ?, ?, ?, ?, ?)",
                ("x", "cycle-new", "no-such-proposal", "scan", AWARE.isoformat(), "c"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                "INSERT INTO reasoning (id, cycle_id, proposal_id, stage, created_at, content) VALUES (?, NULL, NULL, ?, ?, ?)",
                ("x", "scan", AWARE.isoformat(), "c"),
            )
        insert_proposal(conn, _bare_proposal("prop-0003"), _legs("prop-0003"))
        assert _count(conn, "proposal_legs") == 2
        assert status(conn, db_path).missing_tables == () and migrate(conn) == LATEST  # and nothing pending after
        conn.close()

    def test_controls_file_leaves_transactions_to_db_py(self):
        """0003's only BEGIN … END is its trigger body: one statement for the splitter, no transaction control."""
        text = (MIGRATIONS_DIR / "0003_controls.sql").read_text(encoding="utf-8")
        assert text.startswith("-- migration: 0003 ")
        header = text[: text.index("\nCREATE TABLE")]
        assert all(line.startswith("--") for line in header.splitlines())
        # the header says why each piece is there
        for piece in ("CHECK on key", "CHECKs on value", "seed rows", "no-delete trigger", "fails closed"):
            assert piece in header, piece
        assert "atomic" in header or "together or not at all" in header
        assert text.count("CREATE TABLE IF NOT EXISTS") == 1
        assert "CREATE TABLE " not in text.replace("CREATE TABLE IF NOT EXISTS", "")
        body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
        assert "CREATE TRIGGER " not in body.replace("CREATE TRIGGER IF NOT EXISTS", "")
        assert "INSERT " not in body.replace("INSERT OR IGNORE", "")
        statements = [" ".join(chunk.split()) for chunk in store_db._split_statements(body)]
        assert statements == [
            "CREATE TABLE IF NOT EXISTS controls ("
            " key TEXT PRIMARY KEY NOT NULL CHECK (key IN ('kill_switch', 'halt_until')),"
            " value TEXT NOT NULL,"
            " updated_at TEXT NOT NULL,"
            " CHECK (key != 'kill_switch' OR value IN ('on', 'off')),"
            " CHECK (key != 'halt_until' OR value = ''"
            " OR value GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*') );",
            "INSERT OR IGNORE INTO controls (key, value, updated_at)"
            " VALUES ('kill_switch', 'off', strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));",
            "INSERT OR IGNORE INTO controls (key, value, updated_at)"
            " VALUES ('halt_until', '', strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));",
            "CREATE TRIGGER IF NOT EXISTS controls_keep_rows BEFORE DELETE ON controls"
            " BEGIN SELECT RAISE(ABORT, 'controls rows are never deleted: set the value instead'); END;",
        ]
        # the file itself (header and all) splits the same way, and no chunk is BEGIN / COMMIT / END on its own
        assert len(store_db._split_statements(text)) == 4
        assert not any(re.match(r"(BEGIN|COMMIT|END|ROLLBACK)\b", statement, re.IGNORECASE) for statement in statements)
        assert re.findall(r"^\s*(BEGIN|COMMIT|END|ROLLBACK)\b", body, re.IGNORECASE | re.MULTILINE) == ["BEGIN", "END"]

    def test_a_trigger_body_is_one_statement(self, tmp_path, monkeypatch):
        """The ``;`` inside ``BEGIN … END`` never splits — however many statements the body holds, wherever
        the trigger sits in the file — and the trigger lands in the migration's transaction like any DDL."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        script = (
            "-- migration: 0001 triggers; a header with a semicolon\n"
            "CREATE TABLE IF NOT EXISTS a (id TEXT PRIMARY KEY, note TEXT);\n"
            "CREATE TABLE IF NOT EXISTS log (line TEXT);\n"
            "CREATE TRIGGER IF NOT EXISTS a_audit AFTER INSERT ON a\n"
            "BEGIN\n"
            "    INSERT INTO log (line) VALUES ('first; ' || NEW.id);  -- a ';' in a literal and in a comment;\n"
            "    INSERT INTO log (line) VALUES (CASE WHEN NEW.note IS NULL THEN 'bare' ELSE 'noted' END);\n"
            "END; INSERT INTO a (id) VALUES ('seed');\n"
            "CREATE TRIGGER IF NOT EXISTS a_keep BEFORE DELETE ON a BEGIN SELECT RAISE(ABORT, 'kept; always'); END\n"
        )
        (migrations / "0001_triggers.sql").write_text(script, encoding="utf-8")
        chunks = store_db._split_statements(script)
        assert len(chunks) == 5 and "".join(chunks) == script
        assert chunks[2].lstrip().startswith("CREATE TRIGGER IF NOT EXISTS a_audit") and chunks[2].endswith("END;")
        assert chunks[2].count(";") == 6  # the two body statements, three decoys, the END
        assert chunks[3].strip() == "INSERT INTO a (id) VALUES ('seed');"
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(":memory:")
        assert migrate(conn) == 1
        assert _triggers(conn) == [("a_audit", "a"), ("a_keep", "a")]
        assert [row[0] for row in conn.execute("SELECT line FROM log").fetchall()] == ["first; seed", "bare"]
        with pytest.raises(sqlite3.IntegrityError, match="kept; always"):
            conn.execute("DELETE FROM a")
        conn.close()

    def test_failed_migration_rolls_back_its_trigger_and_seed_rows(self, tmp_path, monkeypatch, db_path):
        """0003's shape — table, seeds, trigger — followed by a failure: none of it lands."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        for _, _, real in list_migrations():
            (migrations / real.name).write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        controls = migrations / "0003_controls.sql"
        controls.write_text(
            controls.read_text(encoding="utf-8") + "INSERT INTO no_such_table (x) VALUES (1);\n", encoding="utf-8"
        )
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        conn = connect(db_path)
        with pytest.raises(StoreError, match="apply migration for 0003_controls") as info:
            migrate(conn)
        assert isinstance(info.value.cause, sqlite3.OperationalError)
        assert not conn.in_transaction
        assert schema_version(conn) == 2 and "controls" not in _table_names(conn) and _triggers(conn) == []
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", MIGRATIONS_DIR)  # the real file applies cleanly on top
        assert migrate(conn) == LATEST
        assert [row[:2] for row in _control_rows(conn)] == [("halt_until", ""), ("kill_switch", "off")]
        assert _triggers(conn, "controls") == [("controls_keep_rows", "controls")]
        conn.close()

    def test_upgrading_a_v2_database_adds_controls_and_keeps_every_row(self, db_path, trace_data):
        """0003 (and 0004 on top) on a store with history: every existing value survives byte for
        byte, and the controls arrive seeded — kill switch off, no halt — behind their no-delete
        trigger. The history is written with the columns a version-2 store has."""
        conn = connect(db_path)
        conn.execute(store_db._SCHEMA_VERSION_DDL)
        for version, name, path in list_migrations()[:2]:
            store_db._apply_migration(conn, version, name, path)
        assert schema_version(conn) == 2 and "controls" not in _table_names(conn)
        assert status(conn, db_path).pending_migrations == ("0003_controls", "0004_execution")
        with pytest.raises(StoreError, match="get controls") as info:  # nothing to read before 0003
            get_controls(conn)
        assert "no such table: controls" in str(info.value)

        proposal, reasoning, decision, approval, order, fill = _seed_trace_legacy(conn, trace_data)
        cycle_only = add_reasoning(conn, Reasoning(
            id="reas-cycle-only", cycle_id="cycle-0009", stage=ReasoningStage.SCAN, content="c", created_at=AWARE,
        ))
        insert_proposal(conn, _bare_proposal("prop-0002"), _legs("prop-0002"))
        snapshot_positions(conn, [PositionSnapshot(symbol="SPY", quantity=2, taken_at=AWARE)])
        snapshot_pnl(conn, _pnl(AWARE, 1.0))
        log_event(conn, Event(id="evt-0001", occurred_at=AWARE, level=EventLevel.INFO, kind="k", message="m"))
        before = _journal_rows(conn)
        assert all(before[table] for table in _ROWS_SQL)  # every pre-0003 table holds at least one row
        schema_before = _schema_rows(conn)
        widths = _widths(conn)

        assert migrate(conn) == LATEST  # 0003, then 0004
        assert not conn.in_transaction
        assert applied_versions(conn) == {
            1: "0001_initial", 2: "0002_reasoning_cycles_and_legs", 3: "0003_controls", 4: "0004_execution",
        }
        assert _schema_rows(conn)[:2] == schema_before  # 0001 and 0002 were not re-applied
        assert _table_names(conn) == set(TABLES) | {"schema_version"}
        after = _journal_rows(conn)  # every old value as it was; 0004's columns come after them
        assert {table: [row[: widths[table]] for row in rows] for table, rows in after.items()} == before
        trace = get_proposal_trace(conn, proposal.id)
        assert trace == ProposalTrace(
            proposal=proposal, reasoning=tuple(reasoning), decisions=(decision,),
            approvals=(approval,), orders=(order,), fills=(fill,),
        )
        assert get_cycle_reasoning(conn, "cycle-0009") == [cycle_only]
        assert len(get_proposal_legs(conn, "prop-0002")) == 2

        rows = _control_rows(conn)
        assert [row[:2] for row in rows] == [("halt_until", ""), ("kill_switch", "off")]
        for _, _, updated_at in rows:  # SQLite's clock, as text datetime.fromisoformat reads as aware UTC
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", updated_at)
            seeded = datetime.fromisoformat(updated_at)
            assert seeded.utcoffset() == timedelta(0)
            assert abs(datetime.now(timezone.utc) - seeded) < timedelta(minutes=1)
        assert _triggers(conn, "controls") == [("controls_keep_rows", "controls")]
        controls = get_controls(conn)
        assert (controls.kill_switch, controls.halt_until, controls.halt_unknown, controls.problems) == (
            False, None, False, (),
        )
        report = status(conn, db_path)
        assert report.pending_migrations == () and report.missing_tables == ()
        assert report.row_counts["controls"] == 2 and report.row_counts["proposals"] == 2
        assert migrate(conn) == LATEST and _control_rows(conn) == rows  # and nothing pending after
        conn.close()


class TestSchemaConstraints:
    """The database enforces what the models enforce — a second line of defence."""

    def test_foreign_key_violation(self, conn):
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO reasoning (id, proposal_id, stage, created_at, content)"
                " VALUES (?, ?, ?, ?, ?)",
                ("r1", "no-such-proposal", "scan", AWARE.isoformat(), "x"),
            )

    def test_check_constraints(self, conn, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        for column, bad in (
            ("confidence", 1.5), ("confidence", -0.1), ("quantity", 0.0), ("quantity", -1.0), ("instrument", "future"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                _insert_proposal(conn, proposal, **{column: bad})
        assert _count(conn, "proposals") == 0
        _insert_proposal(conn, proposal)
        assert _count(conn, "proposals") == 1

    def test_quantity_checks_on_orders_and_fills(self, conn, trace_data):
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        order = Order.model_validate(trace_data["order"])
        for bad in (0.0, -1.0):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                _insert_order(conn, order, quantity=bad)
        _insert_order(conn, order)
        for bad in (0.0, -2.0):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute(
                    "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    ("fill-0", order.id, AWARE.isoformat(), 1.0, bad, 0.0),
                )
        assert _count(conn, "orders") == 1 and _count(conn, "fills") == 0

    def test_client_order_id_is_unique(self, conn, trace_data):
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        order = Order.model_validate(trace_data["order"])
        _insert_order(conn, order)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            _insert_order(conn, order.model_copy(update={"id": "ord-0002"}))
        assert _count(conn, "orders") == 1

    def test_every_enum_member_inserts_and_others_are_rejected(self, conn, trace_data):
        """Each CHECK lists exactly its enum's values: a too-narrow one would crash the loop."""
        enum_columns = {
            ("proposals", "instrument"): Instrument, ("proposals", "side"): OrderSide,
            ("proposals", "order_type"): OrderType, ("proposal_legs", "option_type"): OptionType,
            ("proposal_legs", "side"): OrderSide, ("reasoning", "stage"): ReasoningStage,
            ("policy_decisions", "verdict"): Verdict, ("approvals", "response"): ApprovalResponse,
            ("orders", "broker"): Broker, ("orders", "status"): OrderStatus, ("orders", "side"): OrderSide,
            ("events", "level"): EventLevel,
        }
        for (table, column), enum in enum_columns.items():
            ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()[0]
            values = ", ".join(repr(member.value) for member in enum)
            assert f"{column} IN ({values})" in " ".join(ddl.split()), (table, column)

        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        for field, enum in (("instrument", Instrument), ("side", OrderSide), ("order_type", OrderType)):
            for member in enum:
                insert_proposal(conn, proposal.model_copy(update={"id": f"prop-{member.value}", field: member}))
        insert_proposal(conn, proposal.model_copy(update={"id": "prop-legs"}), [
            ProposalLeg(
                proposal_id="prop-legs", leg_index=n, symbol="SPY", option_type=option_type, side=side,
                quantity=1, strike=1, expiration=AWARE.date(),
            )
            for n, (option_type, side) in enumerate((t, s) for t in OptionType for s in OrderSide)
        ])
        for stage in ReasoningStage:
            add_reasoning(conn, Reasoning(proposal_id=proposal.id, stage=stage, content="c"))
        for verdict in Verdict:
            record_decision(conn, PolicyDecision(proposal_id=proposal.id, verdict=verdict, rules_evaluated=[]))
        for response in ApprovalResponse:
            record_approval(conn, Approval(
                proposal_id=proposal.id, channel="cli", response=response, responded_at=AWARE,
            ))
        for status in OrderStatus:
            upsert_order(conn, Order(
                proposal_id=proposal.id, client_order_id="coid-" + status.value, broker=Broker.PAPER,
                status=status, symbol="SPY", side=OrderSide.BUY, quantity=1,
            ))
        for broker in Broker:
            upsert_order(conn, Order(
                proposal_id=proposal.id, client_order_id="coid-" + broker.value, broker=broker,
                status=OrderStatus.PROPOSED, symbol="SPY", side=OrderSide.SELL, quantity=1,
            ))
        for level in EventLevel:
            log_event(conn, Event(level=level, kind="k", message="m"))
        assert {table: _count(conn, table) for table in TABLES} == {
            "proposals": 8, "proposal_legs": 4, "reasoning": 3, "policy_decisions": 4, "approvals": 3,
            "orders": 10, "fills": 0, "position_snapshots": 0, "pnl_snapshots": 0, "events": 5, "controls": 2,
        }

        ts = AWARE.isoformat()
        rejected = (
            ("INSERT INTO reasoning (id, proposal_id, stage, created_at, content) VALUES (?, ?, ?, ?, ?)",
             ("x", proposal.id, "dream", ts, "c")),
            ("INSERT INTO proposal_legs (id, proposal_id, leg_index, symbol, option_type, side, quantity, strike,"
             " expiration) VALUES (?, ?, 9, 'SPY', ?, 'buy', 1, 1, ?)", ("x", proposal.id, "future", ts)),
            ("INSERT INTO proposal_legs (id, proposal_id, leg_index, symbol, option_type, side, quantity, strike,"
             " expiration) VALUES (?, ?, 9, 'SPY', 'call', ?, 1, 1, ?)", ("x", proposal.id, "hold", ts)),
            ("INSERT INTO policy_decisions (id, proposal_id, decided_at, verdict, rules_evaluated)"
             " VALUES (?, ?, ?, ?, ?)", ("x", proposal.id, ts, "MAYBE", "[]")),
            ("INSERT INTO approvals (id, proposal_id, requested_at, response, channel) VALUES (?, ?, ?, ?, ?)",
             ("x", proposal.id, ts, "maybe", "cli")),
            ("INSERT INTO events (id, occurred_at, level, kind, message) VALUES (?, ?, ?, ?, ?)",
             ("x", ts, "loud", "k", "m")),
        )
        for sql, params in rejected:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute(sql, params)
        for column, bad in (("side", "hold"), ("order_type", "stop")):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                _insert_proposal(conn, proposal, id="x", **{column: bad})
        order = Order.model_validate(trace_data["order"])
        for column, bad in (("broker", "crypto"), ("status", "lost"), ("side", "hold")):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                _insert_order(conn, order, **{column: bad})
        assert _count(conn, "proposals") == 8 and _count(conn, "orders") == 10
        assert _count(conn, "proposal_legs") == 4

    def test_columns_match_the_spec(self, conn):
        """Column names and declared types verbatim from the spec (and the models); ``id`` the PRIMARY KEY.

        The file is read with plain sqlite3, so the affinity matters: TEXT for
        timestamps and JSON, REAL for money and quantities, INTEGER for tokens.
        pydantic's lax mode would round-trip a TEXT price or a REAL token count
        unnoticed, but ``fill_price > 1000`` in the sqlite3 shell would then
        match the 12.15 fill.
        """
        expected = {
            "proposals": (Proposal, [
                ("id", "TEXT"), ("created_at", "TEXT"), ("cycle_id", "TEXT"), ("symbol", "TEXT"),
                ("instrument", "TEXT"), ("side", "TEXT"), ("quantity", "REAL"), ("order_type", "TEXT"),
                ("limit_price", "REAL"), ("thesis", "TEXT"), ("confidence", "REAL"), ("invalidation", "TEXT"),
                ("raw_model_output", "TEXT"), ("model_name", "TEXT"), ("prompt_version", "TEXT"),
            ]),
            "proposal_legs": (ProposalLeg, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("leg_index", "INTEGER"), ("symbol", "TEXT"),
                ("option_type", "TEXT"), ("side", "TEXT"), ("quantity", "REAL"), ("strike", "REAL"),
                ("expiration", "TEXT"),
            ]),
            "reasoning": (Reasoning, [
                ("id", "TEXT"), ("cycle_id", "TEXT"), ("proposal_id", "TEXT"), ("stage", "TEXT"),
                ("created_at", "TEXT"), ("content", "TEXT"), ("tokens_in", "INTEGER"), ("tokens_out", "INTEGER"),
                ("model_name", "TEXT"), ("latency_ms", "REAL"),
            ]),
            "policy_decisions": (PolicyDecision, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("decided_at", "TEXT"), ("verdict", "TEXT"),
                ("rules_evaluated", "TEXT"), ("failing_rule", "TEXT"), ("notes", "TEXT"),
                ("purpose", "TEXT"),  # 0004
            ]),
            "approvals": (Approval, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("requested_at", "TEXT"), ("responded_at", "TEXT"),
                ("response", "TEXT"), ("channel", "TEXT"), ("responder", "TEXT"),
                ("decision_id", "TEXT"), ("expires_at", "TEXT"), ("note", "TEXT"),  # 0004
            ]),
            "orders": (Order, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("client_order_id", "TEXT"), ("broker", "TEXT"),
                ("broker_order_id", "TEXT"), ("status", "TEXT"), ("submitted_at", "TEXT"), ("updated_at", "TEXT"),
                ("symbol", "TEXT"), ("side", "TEXT"), ("quantity", "REAL"), ("limit_price", "REAL"),
                # 0004
                ("decision_id", "TEXT"), ("regate_decision_id", "TEXT"), ("approval_id", "TEXT"),
                ("instrument", "TEXT"), ("order_type", "TEXT"), ("time_in_force", "TEXT"),
                ("position_intent", "TEXT"), ("filled_quantity", "REAL"), ("avg_fill_price", "REAL"),
                ("broker_status", "TEXT"), ("status_reason", "TEXT"), ("last_synced_at", "TEXT"),
            ]),
            "fills": (Fill, [
                ("id", "TEXT"), ("order_id", "TEXT"), ("filled_at", "TEXT"), ("fill_price", "REAL"),
                ("fill_quantity", "REAL"), ("fees", "REAL"),
                ("broker_fill_id", "TEXT"),  # 0004
            ]),
            "position_snapshots": (PositionSnapshot, [
                ("id", "TEXT"), ("taken_at", "TEXT"), ("symbol", "TEXT"), ("quantity", "REAL"),
                ("avg_cost", "REAL"), ("market_value", "REAL"), ("unrealized_pnl", "REAL"),
            ]),
            "pnl_snapshots": (PnlSnapshot, [
                ("id", "TEXT"), ("taken_at", "TEXT"), ("equity", "REAL"), ("cash", "REAL"),
                ("buying_power", "REAL"), ("daily_pnl", "REAL"), ("realized_pnl", "REAL"),
                ("unrealized_pnl", "REAL"),
            ]),
            "events": (Event, [
                ("id", "TEXT"), ("occurred_at", "TEXT"), ("level", "TEXT"), ("kind", "TEXT"),
                ("message", "TEXT"), ("payload", "TEXT"),
            ]),
        }
        # every record table; ``controls`` is keyed rows, not records (TestControls pins its columns)
        assert set(expected) == set(TABLES) - {"controls"}
        for table, (model, columns) in expected.items():
            rows = conn.execute(
                'SELECT name, type, "notnull", pk FROM pragma_table_info(?)', (table,)
            ).fetchall()
            assert [(row["name"], row["type"]) for row in rows] == columns, table
            assert list(model.model_fields) == [name for name, _ in columns], table
            assert [row["name"] for row in rows if row["pk"]] == ["id"], table
            # NOT NULL on exactly the columns whose field is not Optional — id included: a
            # non-INTEGER PRIMARY KEY does not imply NOT NULL in SQLite
            not_null = {row["name"] for row in rows if row["notnull"]}
            required = {name for name, field in model.model_fields.items() if not _optional(field.annotation)}
            assert not_null == required, table

    def test_id_is_not_null_on_every_table(self, conn, trace_data):
        """A non-INTEGER PRIMARY KEY does not imply NOT NULL in SQLite.

        Without the explicit declaration a raw writer (the dashboard, the
        sqlite3 shell) can store any number of NULL-id rows — NULLs are
        distinct in a unique index — and every repo read of that table then
        fails on the row that will not validate.
        """
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        _insert_order(conn, Order.model_validate(trace_data["order"]))
        ts = AWARE.isoformat()
        rows = {  # the id first, then otherwise valid values (parents exist for every FK)
            "proposals": ("INSERT INTO proposals VALUES (?, ?, 'c', 'SPY', 'equity', 'buy', 1, 'market', NULL,"
                          " 't', 0.5, 'i', '{}', 'm', 'v')", (ts,)),
            "proposal_legs": ("INSERT INTO proposal_legs VALUES (?, 'prop-0001', 0, 'SPY', 'call', 'buy', 1, 640,"
                              " '2026-10-16')", ()),
            "reasoning": ("INSERT INTO reasoning VALUES (?, 'cycle-0001', 'prop-0001', 'scan', ?, 'c', NULL, NULL,"
                          " NULL, NULL)", (ts,)),
            "policy_decisions": ("INSERT INTO policy_decisions VALUES (?, 'prop-0001', ?, 'REJECT', '[]', NULL, NULL,"
                                 " 'evaluate')", (ts,)),
            "approvals": ("INSERT INTO approvals VALUES (?, 'prop-0001', ?, NULL, NULL, 'cli', NULL, NULL, NULL,"
                          " NULL)", (ts,)),
            "orders": ("INSERT INTO orders VALUES (?, 'prop-0001', 'coid-x', 'paper', NULL, 'submitted', NULL, ?,"
                       " 'SPY', 'buy', 1, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL,"
                       " NULL)", (ts,)),
            "fills": ("INSERT INTO fills VALUES (?, 'ord-0001', ?, 1, 1, 0, NULL)", (ts,)),
            "position_snapshots": ("INSERT INTO position_snapshots VALUES (?, ?, 'SPY', 1, NULL, NULL, NULL)", (ts,)),
            "pnl_snapshots": ("INSERT INTO pnl_snapshots VALUES (?, ?, 1, 2, 3, 4, 5, 6)", (ts,)),
            "events": ("INSERT INTO events VALUES (?, ?, 'info', 'k', 'm', NULL)", (ts,)),
        }
        assert set(rows) == set(TABLES) - {"controls"}  # which has no id: its primary key is ``key``
        for table, (sql, params) in rows.items():
            declared = conn.execute(
                'SELECT "notnull" FROM pragma_table_info(?) WHERE name = ?', (table, "id")
            ).fetchone()[0]
            assert declared == 1, table
            with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
                conn.execute(sql, (None, *params))
            conn.execute(sql, ("id-" + table, *params))  # the same row with an id is fine
        assert {table: _count(conn, table) for table in TABLES} == {
            "proposals": 2, "proposal_legs": 1, "reasoning": 1, "policy_decisions": 1, "approvals": 1,
            "orders": 2, "fills": 1, "position_snapshots": 1, "pnl_snapshots": 1, "events": 1, "controls": 2,
        }
        trace = get_proposal_trace(conn, "prop-0001")  # and every read still maps
        assert len(trace.fills) == 1 and len(trace.legs) == 1 and len(trace.reasoning) == 1
        assert len(get_open_orders(conn)) == 1 and len(get_recent_events(conn)) == 1
        assert len(get_cycle_reasoning(conn, "cycle-0001")) == 1 and len(get_recent_proposals(conn)) == 2

    def test_every_foreign_key_and_lookup_column_is_indexed(self, conn):
        indexed = set()
        for table in TABLES:
            for index in conn.execute("SELECT name FROM pragma_index_list(?)", (table,)).fetchall():
                columns = conn.execute("SELECT name FROM pragma_index_info(?)", (index["name"],)).fetchall()
                indexed.add((table, columns[0]["name"]))
        expected = {
            ("proposal_legs", "proposal_id"),
            ("reasoning", "proposal_id"),
            ("reasoning", "cycle_id"),
            ("reasoning", "created_at"),
            ("policy_decisions", "proposal_id"),
            ("approvals", "proposal_id"),
            ("orders", "proposal_id"),
            ("orders", "client_order_id"),
            ("orders", "status"),
            ("fills", "order_id"),
            ("events", "occurred_at"),
            ("position_snapshots", "taken_at"),
            ("pnl_snapshots", "taken_at"),
            ("proposals", "created_at"),
            ("proposals", "cycle_id"),
        }
        assert expected <= indexed

    def test_proposal_legs_constraints(self, conn, trace_data):
        """FK to the proposal, CHECKs on type/side/quantity/strike, one row per (proposal, leg_index)."""
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        sql = (
            "INSERT INTO proposal_legs (id, proposal_id, leg_index, symbol, option_type, side, quantity, strike,"
            " expiration) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        good = ("leg-0", "prop-0001", 0, "SPY261016C00640000", "call", "buy", 1.0, 640.0, "2026-10-16")
        for position, bad, needle in (
            (1, "no-such-proposal", "FOREIGN KEY"), (4, "future", "CHECK"), (5, "hold", "CHECK"),
            (6, 0.0, "CHECK"), (6, -1.0, "CHECK"), (7, 0.0, "CHECK"), (7, -640.0, "CHECK"),
        ):
            params = list(good)
            params[position] = bad
            with pytest.raises(sqlite3.IntegrityError, match=needle):
                conn.execute(sql, params)
        assert _count(conn, "proposal_legs") == 0
        conn.execute(sql, good)
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            conn.execute(sql, ("leg-dup", "prop-0001", 0, "SPY261016P00630000", "put", "sell", 1.0, 630.0, "2026-10-16"))
        conn.execute(sql, ("leg-1", "prop-0001", 1, "SPY261016P00630000", "put", "sell", 1.0, 630.0, "2026-10-16"))
        assert _count(conn, "proposal_legs") == 2

    def test_reasoning_needs_a_cycle_or_a_proposal(self, conn, trace_data):
        """The 0002 CHECK: a row under neither a cycle nor a proposal is refused; either alone is fine."""
        _insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        sql = "INSERT INTO reasoning (id, cycle_id, proposal_id, stage, created_at, content) VALUES (?, ?, ?, ?, ?, ?)"
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(sql, ("x", None, None, "scan", AWARE.isoformat(), "c"))
        conn.execute(sql, ("cycle-only", "cycle-1", None, "scan", AWARE.isoformat(), "c"))
        conn.execute(sql, ("proposal-only", None, "prop-0001", "scan", AWARE.isoformat(), "c"))
        conn.execute(sql, ("both", "cycle-1", "prop-0001", "scan", AWARE.isoformat(), "c"))
        assert _count(conn, "reasoning") == 3
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(sql, ("x", "cycle-1", "no-such-proposal", "scan", AWARE.isoformat(), "c"))
        assert _count(conn, "reasoning") == 3


class TestModels:
    def test_naive_datetimes_are_taken_as_utc(self, trace_data):
        proposal = Proposal.model_validate({**trace_data["proposal"], "created_at": NAIVE})
        assert proposal.created_at == AWARE
        assert proposal.created_at.tzinfo is timezone.utc
        fill = Fill(order_id="o", filled_at="2026-07-30T14:00:00", fill_price=1.0, fill_quantity=1)
        assert fill.filled_at == AWARE
        event = Event(level=EventLevel.INFO, kind="k", message="m", occurred_at="2026-07-30T14:00:00Z")
        assert event.occurred_at == AWARE

    def test_other_offsets_are_converted_to_utc(self):
        eastern = datetime(2026, 7, 30, 10, 0, tzinfo=timezone(timedelta(hours=-4)))
        approval = Approval(proposal_id="p", channel="cli", requested_at=eastern, responded_at=eastern)
        assert approval.requested_at == AWARE
        assert approval.responded_at == AWARE
        assert approval.requested_at.utcoffset() == timedelta(0)

    def test_defaults(self):
        before = datetime.now(timezone.utc)
        snapshot = PnlSnapshot(
            equity=1.0, cash=1.0, buying_power=1.0, daily_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0
        )
        other = PnlSnapshot(
            equity=1.0, cash=1.0, buying_power=1.0, daily_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0
        )
        assert uuid.UUID(snapshot.id).version == 4
        assert snapshot.id != other.id
        assert snapshot.id == str(uuid.UUID(snapshot.id))  # canonical hyphenated form
        assert uuid.UUID(new_id()).version == 4
        assert before <= snapshot.taken_at <= datetime.now(timezone.utc)
        assert Approval(proposal_id="p", channel="cli").responded_at is None
        assert Order(
            proposal_id="p", client_order_id="c", broker=Broker.PAPER, status=OrderStatus.PROPOSED,
            symbol="spy", side=OrderSide.BUY, quantity=1,
        ).submitted_at is None
        assert Fill(order_id="o", fill_price=1.0, fill_quantity=1).fees == 0.0
        assert Event(level=EventLevel.INFO, kind="k", message="m").payload is None

    def test_symbols_are_uppercased(self, trace_data):
        proposal = Proposal.model_validate({**trace_data["proposal"], "symbol": " spy "})
        assert proposal.symbol == "SPY"
        order = Order.model_validate({**trace_data["order"], "symbol": "aapl"})
        assert order.symbol == "AAPL"
        with pytest.raises(ValidationError):
            Proposal.model_validate({**trace_data["proposal"], "symbol": "  "})

    def test_bounds(self, trace_data):
        for field, value in (("quantity", 0), ("quantity", -1), ("confidence", 1.01), ("confidence", -0.1)):
            with pytest.raises(ValidationError):
                Proposal.model_validate({**trace_data["proposal"], field: value})
        with pytest.raises(ValidationError):
            Fill(order_id="o", fill_price=1.0, fill_quantity=0)
        with pytest.raises(ValidationError):
            Order.model_validate({**trace_data["order"], "quantity": 0})
        assert PositionSnapshot(symbol="SPY", quantity=-2).quantity == -2  # shorts are fine

    def test_nan_and_inf_are_rejected(self, trace_data):
        """SQLite stores NaN as NULL, so a NaN price would silently read back as None."""
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValidationError, match="finite"):
                Order.model_validate({**trace_data["order"], "limit_price": bad})
            with pytest.raises(ValidationError, match="finite"):
                Proposal.model_validate({**trace_data["proposal"], "limit_price": bad})
            with pytest.raises(ValidationError, match="finite"):
                PositionSnapshot(symbol="SPY", quantity=1, avg_cost=bad)
            with pytest.raises(ValidationError, match="finite"):
                Fill(order_id="o", fill_price=bad, fill_quantity=1)
            with pytest.raises(ValidationError, match="finite"):
                PnlSnapshot(
                    equity=bad, cash=1.0, buying_power=1.0, daily_pnl=0.0, realized_pnl=0.0, unrealized_pnl=0.0,
                )

    def test_frozen(self, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        with pytest.raises(ValidationError):
            proposal.symbol = "QQQ"

    def test_enums_store_their_string_values(self):
        assert Verdict.NEEDS_APPROVAL.value == "NEEDS_APPROVAL"
        assert ReasoningStage("scan") is ReasoningStage.SCAN
        assert Instrument.OPTION == "option" and OrderType.LIMIT == "limit"
        assert ApprovalResponse.EXPIRED.value == "expired"
        assert [s.value for s in OrderStatus] == [
            "proposed", "gated", "approved", "submitted",
            "filled", "partially_filled", "cancelled", "failed",
        ]
        assert [level.value for level in EventLevel] == ["debug", "info", "warning", "error", "critical"]

    def test_open_order_statuses(self):
        assert OPEN_ORDER_STATUSES == {
            OrderStatus.PROPOSED, OrderStatus.GATED, OrderStatus.APPROVED,
            OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED,
        }
        assert set(OrderStatus) - OPEN_ORDER_STATUSES == {
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED,
        }
        assert isinstance(OPEN_ORDER_STATUSES, frozenset)

    def test_json_fields_round_trip(self, trace_data):
        decision = PolicyDecision.model_validate(trace_data["decision"])
        again = PolicyDecision.model_validate(decision.model_dump(mode="json"))
        assert again == decision
        assert again.rules_evaluated == trace_data["decision"]["rules_evaluated"]
        assert again.rules_evaluated[0]["passed"] is True
        event = Event(level=EventLevel.WARNING, kind="risk_limit", message="m", payload={"pct": 2.5, "n": [1, 2]})
        again = Event.model_validate(event.model_dump(mode="json"))
        assert again == event and again.payload == {"pct": 2.5, "n": [1, 2]}
        assert again.occurred_at.tzinfo is timezone.utc

    def test_trace_latest_properties(self, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        empty = ProposalTrace(proposal=proposal)
        assert empty.decision is None and empty.approval is None
        assert empty.legs == () and empty.reasoning == () and empty.orders == () and empty.fills == ()
        first = PolicyDecision.model_validate(trace_data["decision"])
        second = first.model_copy(update={"id": "dec-0002", "verdict": Verdict.REJECT})
        trace = ProposalTrace(proposal=proposal, decisions=(first, second))
        assert trace.decision is second
        approval = Approval.model_validate(trace_data["approval"])
        assert ProposalTrace(proposal=proposal, approvals=(approval,)).approval == approval

    def test_reasoning_needs_a_cycle_or_a_proposal(self):
        under_cycle = Reasoning(cycle_id="cycle-1", stage=ReasoningStage.SCAN, content="c")
        assert under_cycle.proposal_id is None
        assert under_cycle.model_name is None and under_cycle.latency_ms is None
        assert Reasoning(proposal_id="p", stage=ReasoningStage.SCAN, content="c").cycle_id is None  # the 0001 shape
        both = Reasoning(
            cycle_id="cycle-1", proposal_id="p", stage=ReasoningStage.SCAN, content="c",
            model_name="claude-haiku-4-5-20251001", latency_ms=812.5,
        )
        assert (both.model_name, both.latency_ms) == ("claude-haiku-4-5-20251001", 812.5)
        assert Reasoning.model_validate(both.model_dump(mode="json")) == both
        with pytest.raises(ValidationError, match="cycle_id or a proposal_id"):
            Reasoning(stage=ReasoningStage.SCAN, content="c")
        with pytest.raises(ValidationError, match="finite"):
            Reasoning(cycle_id="cycle-1", stage=ReasoningStage.SCAN, content="c", latency_ms=float("nan"))

    def test_proposal_leg(self):
        leg = ProposalLeg(
            proposal_id="p", leg_index=0, symbol=" spy261016c00640000 ", option_type="call", side="buy",
            quantity=1, strike=640, expiration="2026-10-16",
        )
        assert leg.symbol == "SPY261016C00640000"
        assert leg.option_type is OptionType.CALL and leg.side is OrderSide.BUY
        assert leg.expiration == date(2026, 10, 16) and uuid.UUID(leg.id).version == 4
        assert leg.model_dump(mode="json")["expiration"] == "2026-10-16"
        assert ProposalLeg.model_validate(leg.model_dump(mode="json")) == leg
        for field, bad in (
            ("leg_index", -1), ("quantity", 0), ("quantity", -1), ("strike", 0), ("strike", -640),
            ("symbol", "  "), ("option_type", "future"), ("side", "hold"), ("expiration", "soon"),
            ("strike", float("nan")),
        ):
            with pytest.raises(ValidationError):
                ProposalLeg.model_validate({**leg.model_dump(), field: bad})
        with pytest.raises(ValidationError):
            leg.strike = 650

    def test_store_status_model(self):
        report = StoreStatus(
            path="x.db", exists=True, schema_version=3, journal_mode="wal",
            applied_migrations=("0001_initial", "0002_reasoning_cycles_and_legs", "0003_controls", "0004_execution"),
            pending_migrations=(), row_counts={table: 0 for table in TABLES},
        )
        assert report.applied_migrations == ("0001_initial", "0002_reasoning_cycles_and_legs", "0003_controls", "0004_execution")
        assert set(report.row_counts) == set(TABLES) and len(report.row_counts) == 11


class TestFixture:
    def test_validates_into_every_model(self, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        reasoning = [Reasoning.model_validate(r) for r in trace_data["reasoning"]]
        decision = PolicyDecision.model_validate(trace_data["decision"])
        approval = Approval.model_validate(trace_data["approval"])
        order = Order.model_validate(trace_data["order"])
        fill = Fill.model_validate(trace_data["fill"])

        assert proposal.id == "prop-0001"
        assert proposal.instrument is Instrument.OPTION and proposal.side is OrderSide.BUY
        assert proposal.order_type is OrderType.LIMIT and proposal.limit_price == 12.2
        assert proposal.created_at.tzinfo is timezone.utc
        assert [r.stage for r in reasoning] == [
            ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL,
        ]
        assert [r.created_at for r in reasoning] == sorted(r.created_at for r in reasoning)
        assert all(r.proposal_id == proposal.id for r in reasoning)
        assert decision.proposal_id == proposal.id and decision.verdict is Verdict.NEEDS_APPROVAL
        assert len(decision.rules_evaluated) == 2 and decision.failing_rule is None
        assert approval.proposal_id == proposal.id
        assert approval.response is ApprovalResponse.APPROVED and approval.channel == "cli"
        assert approval.responded_at > approval.requested_at
        assert order.proposal_id == proposal.id and order.client_order_id == "aegis-prop-0001-1"
        assert order.broker is Broker.PAPER and order.status is OrderStatus.FILLED
        assert fill.order_id == order.id and fill.fill_quantity == order.quantity
        trace = ProposalTrace(
            proposal=proposal, reasoning=tuple(reasoning), decisions=(decision,),
            approvals=(approval,), orders=(order,), fills=(fill,),
        )
        assert trace.decision == decision and trace.approval == approval

    def test_fixture_is_obviously_synthetic(self, trace_data):
        for record in (trace_data["proposal"], *trace_data["reasoning"], trace_data["decision"]):
            text = " ".join(str(v) for v in record.values())
            assert "SYNTHETIC" in text or "synthetic" in text


class TestStatus:
    def test_after_open_store(self, conn, db_path):
        report = status(conn, db_path)
        assert report == StoreStatus(
            path=str(db_path), exists=True, schema_version=LATEST, journal_mode="wal",
            applied_migrations=("0001_initial", "0002_reasoning_cycles_and_legs", "0003_controls", "0004_execution"),
            pending_migrations=(), row_counts=_FRESH_COUNTS,  # the two seeded control rows, nothing else
        )
        assert list(report.row_counts) == list(TABLES)
        assert report.exists is True
        assert status(conn, db_path.with_name("other.db")).exists is False  # about the path, not the connection
        _insert_event(conn)
        assert status(conn, db_path).row_counts["events"] == 1

    def test_before_migrating(self, db_path):
        conn = connect(db_path)
        report = status(conn, db_path)
        assert report.exists is True and report.schema_version == 0
        assert report.applied_migrations == ()
        assert report.pending_migrations == ("0001_initial", "0002_reasoning_cycles_and_legs", "0003_controls", "0004_execution")
        assert report.row_counts == {table: 0 for table in TABLES}
        assert report.missing_tables == ()  # not created yet is not the same as dropped
        conn.close()

    def test_counts_every_table_independently(self, conn, db_path, trace_data):
        """Counts that differ per table, checked against literal SQL: a swapped query cannot hide."""
        _seed_trace(conn, trace_data)  # 1 proposal, 3 reasoning, 1 decision, 1 approval, 1 order, 1 fill
        record_fill(conn, Fill(order_id="ord-0001", fill_price=1.0, fill_quantity=1))
        snapshot_positions(conn, [PositionSnapshot(symbol="SPY", quantity=n) for n in range(4)])
        for n in range(5):
            snapshot_pnl(conn, _pnl(AWARE + timedelta(minutes=n), float(n)))
        for n in range(6):
            log_event(conn, Event(level=EventLevel.INFO, kind="k", message=str(n)))
        insert_proposal(conn, _bare_proposal("prop-0002"), _legs("prop-0002", 7))
        expected = {
            "proposals": 2, "proposal_legs": 7, "reasoning": 3, "policy_decisions": 1, "approvals": 1,
            "orders": 1, "fills": 2, "position_snapshots": 4, "pnl_snapshots": 5, "events": 6, "controls": 2,
        }
        assert status(conn, db_path).row_counts == expected
        assert {table: _count(conn, table) for table in TABLES} == expected

    def test_memory(self):
        conn = connect(":memory:")
        migrate(conn)
        report = status(conn, ":memory:")
        assert report.exists is True and report.journal_mode == "memory" and report.schema_version == LATEST
        conn.close()


def test_db_files_and_wal_sidecars_are_gitignored():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"*.db", "*.db-wal", "*.db-shm"} <= set(lines)


def _seed_trace_legacy(conn, trace_data):
    """``_seed_trace`` for a store from before 0004, where the repository's decision,
    approval, order and fill writers (which write 0004's columns) cannot run: the
    same records, written with the columns the older schema has. Returns them as
    a migrated store reads them back."""
    proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
    reasoning = [add_reasoning(conn, Reasoning.model_validate(r)) for r in trace_data["reasoning"]]
    decision = PolicyDecision.model_validate(trace_data["decision"])
    conn.execute(
        "INSERT INTO policy_decisions (id, proposal_id, decided_at, verdict, rules_evaluated, failing_rule, notes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (decision.id, decision.proposal_id, decision.decided_at.isoformat(), decision.verdict.value,
         json.dumps(decision.rules_evaluated), decision.failing_rule, decision.notes),
    )
    approval = Approval.model_validate(trace_data["approval"])
    conn.execute(
        "INSERT INTO approvals (id, proposal_id, requested_at, responded_at, response, channel, responder)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (approval.id, approval.proposal_id, approval.requested_at.isoformat(),
         approval.responded_at.isoformat() if approval.responded_at else None,
         approval.response.value if approval.response else None, approval.channel, approval.responder),
    )
    order = Order.model_validate(trace_data["order"])
    _insert_order(conn, order)
    fill = Fill.model_validate(trace_data["fill"])
    conn.execute(
        "INSERT INTO fills (id, order_id, filled_at, fill_price, fill_quantity, fees) VALUES (?, ?, ?, ?, ?, ?)",
        (fill.id, fill.order_id, fill.filled_at.isoformat(), fill.fill_price, fill.fill_quantity, fill.fees),
    )
    return proposal, reasoning, decision, approval, order, fill


def _widths(conn):
    """Each journal table's column count, to compare rows across a migration that adds columns."""
    return {
        table: len(conn.execute("SELECT name FROM pragma_table_info(?)", (table,)).fetchall()) for table in _ROWS_SQL
    }


def _seed_trace(conn, trace_data):
    """Write the fixture lineage through the repository; returns the records as stored."""
    proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
    reasoning = [add_reasoning(conn, Reasoning.model_validate(r)) for r in trace_data["reasoning"]]
    decision = record_decision(conn, PolicyDecision.model_validate(trace_data["decision"]))
    approval = record_approval(conn, Approval.model_validate(trace_data["approval"]))
    order = upsert_order(conn, Order.model_validate(trace_data["order"]))
    fill = record_fill(conn, Fill.model_validate(trace_data["fill"]))
    return proposal, reasoning, decision, approval, order, fill


def _pnl(taken_at, equity):
    """A snapshot with a distinct value per column, so swapped columns cannot round-trip as equal."""
    return PnlSnapshot(
        taken_at=taken_at, equity=equity, cash=100.0, buying_power=200.0,
        daily_pnl=10.0, realized_pnl=20.0, unrealized_pnl=30.0,
    )


class TestRepo:
    def test_full_chain_from_fixture(self, conn, trace_data):
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        assert get_proposal(conn, proposal.id) == proposal
        # stages inserted newest-first: the trace must order by created_at, not rowid
        reasoning = [Reasoning.model_validate(r) for r in trace_data["reasoning"]]
        for stage in reversed(reasoning):
            assert add_reasoning(conn, stage) == stage
        decision = record_decision(conn, PolicyDecision.model_validate(trace_data["decision"]))
        # the approval is recorded when requested, then again once answered
        requested = Approval.model_validate(trace_data["approval"]).model_copy(
            update={"responded_at": None, "response": None, "responder": None}
        )
        pending = record_approval(conn, requested)
        assert pending == requested and pending.response is None
        # the answer carries other request fields, which the upsert must ignore: the request is
        # immutable (an answer under another existing proposal must not re-parent it), and what
        # comes back is the row as stored, not the argument
        insert_proposal(conn, _bare_proposal("prop-0002"))
        answered = Approval.model_validate(trace_data["approval"]).model_copy(update={
            "proposal_id": "prop-0002", "channel": "slack",
            "requested_at": requested.requested_at + timedelta(hours=1),
        })
        approval = record_approval(conn, answered)
        assert approval == Approval.model_validate(trace_data["approval"])
        assert approval.proposal_id == "prop-0001" and approval.channel == "cli"
        assert approval.requested_at == requested.requested_at
        assert approval.response is ApprovalResponse.APPROVED
        assert approval.responded_at == answered.responded_at and approval.responder == "test-operator"
        assert _count(conn, "approvals") == 1
        row = conn.execute(
            "SELECT proposal_id, channel, requested_at, response, responder FROM approvals"
        ).fetchone()
        assert tuple(row) == ("prop-0001", "cli", requested.requested_at.isoformat(), "approved", "test-operator")
        assert get_proposal_trace(conn, "prop-0002").approvals == ()
        order = upsert_order(conn, Order.model_validate(trace_data["order"]))
        fill = record_fill(conn, Fill.model_validate(trace_data["fill"]))

        trace = get_proposal_trace(conn, proposal.id)
        assert trace.proposal == proposal
        assert [r.stage for r in trace.reasoning] == [
            ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL,
        ]
        assert list(trace.reasoning) == reasoning
        assert trace.decisions == (decision,)
        assert trace.decision.verdict is Verdict.NEEDS_APPROVAL
        assert trace.decision.rules_evaluated == trace_data["decision"]["rules_evaluated"]
        assert trace.decision.rules_evaluated[0]["passed"] is True
        assert trace.approvals == (approval,)
        assert trace.approval.response is ApprovalResponse.APPROVED
        assert trace.orders == (order,)
        assert trace.fills == (fill,)
        assert trace.fills[0].order_id == trace.orders[0].id
        assert trace.proposal.created_at == proposal.created_at
        assert trace.proposal.created_at.tzinfo is timezone.utc
        assert trace.approval.responded_at == answered.responded_at
        assert trace.orders[0].submitted_at.tzinfo is timezone.utc
        assert trace.fills[0].filled_at == datetime(2026, 7, 30, 14, 7, 19, tzinfo=timezone.utc)
        for record in (proposal, *reasoning, decision, approval, order, fill):
            assert type(record).model_validate(record.model_dump(mode="json")) == record

    def test_trace_of_a_bare_proposal(self, conn, trace_data):
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        trace = get_proposal_trace(conn, proposal.id)
        assert trace == ProposalTrace(proposal=proposal)
        assert trace.decision is None and trace.approval is None

    def test_raw_model_output_is_stored_verbatim(self, conn, trace_data):
        """The model's output is never parsed or re-serialised: prose, odd whitespace and all.

        The fixture's raw output happens to be canonical JSON, which a
        ``json.dumps(json.loads(...))`` round-trip would reproduce unnoticed.
        """
        raw = 'The model said:\n  {"a":1 ,\t"b": [ ] }  '
        proposal = insert_proposal(conn, Proposal.model_validate({**trace_data["proposal"], "raw_model_output": raw}))
        assert proposal.raw_model_output == raw
        assert conn.execute("SELECT raw_model_output FROM proposals WHERE id = ?", (proposal.id,)).fetchone()[0] == raw
        assert get_proposal(conn, proposal.id).raw_model_output == raw
        assert get_proposal_trace(conn, proposal.id).proposal.raw_model_output == raw

    def test_trace_only_contains_its_own_lineage(self, conn, trace_data):
        """A second proposal's rows must never leak into the first one's trace."""
        _seed_trace(conn, trace_data)
        other = insert_proposal(conn, Proposal.model_validate(
            {**trace_data["proposal"], "id": "prop-0002", "cycle_id": "cycle-0002"}
        ))
        add_reasoning(conn, Reasoning(
            id="reas-0002", proposal_id=other.id, stage=ReasoningStage.SCAN, content="other",
        ))
        record_decision(conn, PolicyDecision(
            id="dec-0002", proposal_id=other.id, verdict=Verdict.REJECT, rules_evaluated=[],
        ))
        record_approval(conn, Approval(id="appr-0002", proposal_id=other.id, channel="cli"))
        upsert_order(conn, Order(
            id="ord-0002", proposal_id=other.id, client_order_id="aegis-prop-0002-1", broker=Broker.PAPER,
            status=OrderStatus.FILLED, symbol="SPY", side=OrderSide.BUY, quantity=1,
        ))
        record_fill(conn, Fill(id="fill-0002", order_id="ord-0002", fill_price=1.0, fill_quantity=1))

        trace = get_proposal_trace(conn, "prop-0001")
        for records in (trace.reasoning, trace.decisions, trace.approvals, trace.orders):
            assert {record.proposal_id for record in records} == {"prop-0001"}
        assert {fill.order_id for fill in trace.fills} == {"ord-0001"}
        assert len(trace.reasoning) == 3 and len(trace.orders) == 1 and len(trace.fills) == 1
        theirs = get_proposal_trace(conn, "prop-0002")
        assert [r.id for r in theirs.reasoning] == ["reas-0002"]
        assert [d.id for d in theirs.decisions] == ["dec-0002"]
        assert [a.id for a in theirs.approvals] == ["appr-0002"]
        assert [o.id for o in theirs.orders] == ["ord-0002"]
        assert [f.id for f in theirs.fills] == ["fill-0002"]

    def test_trace_orders_every_section_oldest_first(self, conn, trace_data):
        """Written newest-first, so rowid order and timestamp order disagree in every section."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        late = AWARE + timedelta(minutes=10)
        record_decision(conn, PolicyDecision(
            id="dec-late", proposal_id=proposal.id, decided_at=late, verdict=Verdict.REJECT, rules_evaluated=[],
        ))
        record_decision(conn, PolicyDecision(
            id="dec-early", proposal_id=proposal.id, decided_at=AWARE, verdict=Verdict.NEEDS_APPROVAL,
            rules_evaluated=[],
        ))
        record_approval(conn, Approval(id="appr-late", proposal_id=proposal.id, requested_at=late, channel="cli"))
        record_approval(conn, Approval(id="appr-early", proposal_id=proposal.id, requested_at=AWARE, channel="cli"))
        base = Order.model_validate(trace_data["order"])
        upsert_order(conn, base.model_copy(update={"id": "ord-late", "client_order_id": "coid-late", "updated_at": late}))
        upsert_order(conn, base.model_copy(update={"id": "ord-early", "client_order_id": "coid-early", "updated_at": AWARE}))
        record_fill(conn, Fill(id="fill-late", order_id="ord-early", filled_at=late, fill_price=1.0, fill_quantity=1))
        record_fill(conn, Fill(id="fill-early", order_id="ord-early", filled_at=AWARE, fill_price=1.0, fill_quantity=1))

        trace = get_proposal_trace(conn, proposal.id)
        assert [d.id for d in trace.decisions] == ["dec-early", "dec-late"]
        assert trace.decision.id == "dec-late" and trace.decision.verdict is Verdict.REJECT
        assert [a.id for a in trace.approvals] == ["appr-early", "appr-late"]
        assert trace.approval.id == "appr-late"
        assert [o.id for o in trace.orders] == ["ord-early", "ord-late"]
        assert [f.id for f in trace.fills] == ["fill-early", "fill-late"]

    def test_equal_timestamps_keep_insertion_order(self, conn, trace_data):
        """Ties break on rowid, so what was recorded later comes later: 'latest is last' holds
        even when rows share a timestamp (a proposal re-judged within the same instant)."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        for stage in ReasoningStage:  # scan, thesis, proposal
            add_reasoning(conn, Reasoning(
                id="reas-" + stage.value, proposal_id=proposal.id, stage=stage, created_at=AWARE, content="c",
            ))
        for n, verdict in enumerate((Verdict.NEEDS_APPROVAL, Verdict.REJECT)):
            record_decision(conn, PolicyDecision(
                id=f"dec-{n}", proposal_id=proposal.id, decided_at=AWARE, verdict=verdict, rules_evaluated=[],
            ))
        for n in range(2):
            record_approval(conn, Approval(id=f"appr-{n}", proposal_id=proposal.id, requested_at=AWARE, channel="cli"))
            upsert_order(conn, _bare_order(f"ord-{n}", f"coid-{n}", status=OrderStatus.SUBMITTED, updated_at=AWARE))
            record_fill(conn, Fill(id=f"fill-{n}", order_id="ord-0", filled_at=AWARE, fill_price=1.0, fill_quantity=1))
        for n in range(3):
            log_event(conn, Event(id=f"evt-{n}", occurred_at=AWARE, level=EventLevel.INFO, kind="k", message=f"m{n}"))

        trace = get_proposal_trace(conn, proposal.id)
        assert [r.stage for r in trace.reasoning] == [
            ReasoningStage.SCAN, ReasoningStage.THESIS, ReasoningStage.PROPOSAL,
        ]
        assert [d.id for d in trace.decisions] == ["dec-0", "dec-1"]
        assert trace.decision.id == "dec-1" and trace.decision.verdict is Verdict.REJECT
        assert [a.id for a in trace.approvals] == ["appr-0", "appr-1"] and trace.approval.id == "appr-1"
        assert [o.id for o in trace.orders] == ["ord-0", "ord-1"]
        assert [f.id for f in trace.fills] == ["fill-0", "fill-1"]
        assert [o.id for o in get_open_orders(conn)] == ["ord-0", "ord-1"]
        assert [e.message for e in get_recent_events(conn)] == ["m2", "m1", "m0"]  # newest first

    def test_rejected_decision_keeps_its_failing_rule(self, conn, trace_data):
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        passed = record_decision(conn, PolicyDecision.model_validate(trace_data["decision"]))
        rejected = record_decision(conn, PolicyDecision(
            id="dec-reject", proposal_id=proposal.id, decided_at=passed.decided_at + timedelta(minutes=1),
            verdict=Verdict.REJECT,
            rules_evaluated=[{"rule": "max_position_pct", "limit": 5.0, "observed": 7.2, "passed": False}],
            failing_rule="max_position_pct", notes="over the cap",
        ))
        trace = get_proposal_trace(conn, proposal.id)
        assert trace.decisions == (passed, rejected)
        assert trace.decision == rejected
        assert trace.decision.failing_rule == "max_position_pct" and trace.decision.notes == "over the cap"
        assert trace.decision.rules_evaluated[0]["passed"] is False
        row = conn.execute(
            "SELECT verdict, failing_rule FROM policy_decisions WHERE id = ?", ("dec-reject",)
        ).fetchone()
        assert tuple(row) == ("REJECT", "max_position_pct")

    def test_nan_inside_json_columns_is_store_error(self, conn, trace_data):
        """JSON has no NaN token: refuse it rather than store text no other reader can parse."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        with pytest.raises(StoreError, match="log event for evt-nan") as info:
            log_event(conn, Event(
                id="evt-nan", level=EventLevel.INFO, kind="k", message="m", payload={"pct": float("nan")},
            ))
        assert isinstance(info.value.cause, ValueError)
        with pytest.raises(StoreError, match="record decision for dec-inf"):
            record_decision(conn, PolicyDecision(
                id="dec-inf", proposal_id=proposal.id, verdict=Verdict.REJECT,
                rules_evaluated=[{"rule": "r", "observed": float("inf"), "passed": False}],
            ))
        assert not conn.in_transaction
        assert _count(conn, "events") == 0 and _count(conn, "policy_decisions") == 0
        log_event(conn, Event(id="evt-ok", level=EventLevel.INFO, kind="k", message="m", payload={"pct": 2.5}))
        record_decision(conn, PolicyDecision.model_validate(trace_data["decision"]))
        assert conn.execute("SELECT json_valid(payload) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT json_valid(rules_evaluated) FROM policy_decisions").fetchone()[0] == 1

    def test_lock_timeout_names_the_operation_and_id(self, db_path):
        """The one sqlite3.Error a multi-process deployment will see must say what and which."""
        holder = open_store(db_path)
        writer = connect(db_path)
        writer.execute("PRAGMA busy_timeout=50")
        try:
            with transaction(holder):
                _insert_event(holder)
                with pytest.raises(StoreError, match="log event for blocked") as info:
                    log_event(writer, Event(id="blocked", level=EventLevel.INFO, kind="k", message="m"))
                assert isinstance(info.value.cause, sqlite3.OperationalError)
                assert "database is locked" in str(info.value)
                assert (info.value.what, info.value.key) == ("log event", "blocked")
                assert not writer.in_transaction
            # and the connection is usable again once the lock is released
            log_event(writer, Event(id="unblocked", level=EventLevel.INFO, kind="k", message="m"))
            assert _count(writer, "events") == 2
        finally:
            holder.close()
            writer.close()

    def test_upsert_order_is_idempotent_on_client_order_id(self, conn, trace_data):
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        insert_proposal(conn, _bare_proposal("prop-0002"))
        first = Order.model_validate({
            **trace_data["order"], "status": "approved", "broker_order_id": None, "submitted_at": None,
        })
        assert upsert_order(conn, first) == first
        later = AWARE + timedelta(minutes=1)
        second = first.model_copy(update={
            "id": "ord-0002", "status": OrderStatus.SUBMITTED,
            "broker_order_id": "paper-order-0001", "submitted_at": later, "updated_at": later,
            # the identity of the first insert is immutable: a status update arriving through a
            # live-broker code path must not flip a paper order to live, re-parent or re-symbol it
            "proposal_id": "prop-0002", "broker": Broker.LIVE, "symbol": "QQQ", "side": OrderSide.SELL,
        })
        stored = upsert_order(conn, second)
        assert _count(conn, "orders") == 1
        assert stored.id == "ord-0001"  # the original row's id, not ord-0002
        assert stored.proposal_id == "prop-0001" and stored.broker is Broker.PAPER
        assert stored.symbol == "SPY260821C00640000" and stored.side is OrderSide.BUY
        row = conn.execute("SELECT id, proposal_id, broker, symbol, side FROM orders").fetchone()
        assert tuple(row) == ("ord-0001", "prop-0001", "paper", "SPY260821C00640000", "buy")
        assert stored.status is OrderStatus.SUBMITTED
        assert stored.broker_order_id == "paper-order-0001"
        assert stored.submitted_at == later and stored.updated_at == later
        assert get_order(conn, first.client_order_id) == stored
        # a later update carrying no broker ids keeps the ones already recorded
        third = second.model_copy(update={
            "id": "ord-0003", "status": OrderStatus.FILLED, "broker_order_id": None,
            "submitted_at": None, "updated_at": later + timedelta(minutes=1), "quantity": 1.0,
            "limit_price": 12.5,
        })
        stored = upsert_order(conn, third)
        assert _count(conn, "orders") == 1
        assert stored.id == "ord-0001" and stored.status is OrderStatus.FILLED
        assert stored.broker_order_id == "paper-order-0001" and stored.submitted_at == later
        assert stored.quantity == 1.0 and stored.limit_price == 12.5
        assert stored.updated_at == third.updated_at
        assert get_order(conn, "aegis-prop-0001-1") == stored
        assert get_order(conn, "nope") is None

    def test_get_open_orders(self, conn, trace_data):
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        base = Order.model_validate(trace_data["order"])
        # inserted newest-first so rowid order and updated_at order disagree
        for offset, status in enumerate(OrderStatus):
            upsert_order(conn, base.model_copy(update={
                "id": "ord-" + status.value, "client_order_id": "coid-" + status.value,
                "status": status, "updated_at": AWARE - timedelta(minutes=offset),
            }))
        assert _count(conn, "orders") == len(OrderStatus)
        open_orders = get_open_orders(conn)
        assert {o.status for o in open_orders} == OPEN_ORDER_STATUSES
        assert OrderStatus.SUBMITTED in {o.status for o in open_orders}
        assert OrderStatus.PARTIALLY_FILLED in {o.status for o in open_orders}
        assert not {o.status for o in open_orders} & {
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED,
        }
        assert [o.updated_at for o in open_orders] == sorted(o.updated_at for o in open_orders)

    def test_get_daily_pnl_by_utc_day(self, conn, monkeypatch):
        day = date(2026, 7, 30)
        latest = snapshot_pnl(conn, _pnl(datetime(2026, 7, 30, 23, 59, 59, 999999, tzinfo=timezone.utc), 3.0))
        snapshot_pnl(conn, _pnl(datetime(2026, 7, 30, 9, 30, tzinfo=timezone.utc), 1.0))
        snapshot_pnl(conn, _pnl(datetime(2026, 7, 30, 16, 0, tzinfo=timezone.utc), 2.0))
        snapshot_pnl(conn, _pnl(datetime(2026, 7, 31, 0, 0, tzinfo=timezone.utc), 4.0))  # next day
        # 20:00 New York on the 30th is 00:00Z on the 31st: the UTC day counts
        eastern = timezone(timedelta(hours=-4))
        snapshot_pnl(conn, _pnl(datetime(2026, 7, 30, 20, 0, tzinfo=eastern), 5.0))
        assert get_daily_pnl(conn, day) == latest
        assert get_daily_pnl(conn, day).equity == 3.0
        row = conn.execute(
            "SELECT cash, buying_power, daily_pnl, realized_pnl, unrealized_pnl FROM pnl_snapshots WHERE id = ?",
            (latest.id,),
        ).fetchone()
        assert tuple(row) == (100.0, 200.0, 10.0, 20.0, 30.0)
        assert get_daily_pnl(conn, date(2026, 7, 31)).equity == 5.0  # same instant, latest row
        assert get_daily_pnl(conn, date(2026, 7, 29)) is None
        # a datetime as `day` (datetime is a date) means the UTC calendar day of that instant,
        # whatever its offset — never its local date
        assert get_daily_pnl(conn, datetime(2026, 7, 30, 20, 0, tzinfo=eastern)).equity == 5.0  # 00:00Z, the 31st
        assert get_daily_pnl(conn, datetime(2026, 7, 30, 12, 0, tzinfo=eastern)).equity == 3.0  # 16:00Z, the 30th
        assert get_daily_pnl(conn, datetime(2026, 7, 31, 0, 0)).equity == 5.0  # naive is UTC
        assert get_daily_pnl(conn, datetime(2026, 7, 30, 23, 59, tzinfo=timezone.utc)) == latest
        monkeypatch.setattr(repo, "utcnow", lambda: datetime(2026, 7, 30, 18, 0, tzinfo=timezone.utc))
        assert get_daily_pnl(conn) == latest
        monkeypatch.setattr(repo, "utcnow", lambda: datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))
        assert get_daily_pnl(conn) is None

    def test_snapshot_positions_is_one_transaction(self, conn):
        batch = [
            PositionSnapshot(
                id="pos-1", taken_at=AWARE, symbol="SPY", quantity=2, avg_cost=12.15,
                market_value=2450.0, unrealized_pnl=20.0,
            ),
            PositionSnapshot(id="pos-2", taken_at=AWARE, symbol="AAPL", quantity=-1),
        ]
        assert snapshot_positions(conn, batch) == batch
        assert snapshot_positions(conn, []) == []
        assert _count(conn, "position_snapshots") == 2
        rows = conn.execute(
            "SELECT id, quantity, avg_cost, market_value, unrealized_pnl FROM position_snapshots ORDER BY id"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("pos-1", 2.0, 12.15, 2450.0, 20.0), ("pos-2", -1.0, None, None, None),
        ]
        clashing = [
            PositionSnapshot(id="pos-3", taken_at=AWARE, symbol="QQQ", quantity=1),
            PositionSnapshot(id="pos-1", taken_at=AWARE, symbol="SPY", quantity=2),
        ]
        with pytest.raises(StoreError, match="snapshot positions for pos-1") as info:
            snapshot_positions(conn, clashing)
        assert isinstance(info.value.cause, sqlite3.IntegrityError)
        assert not conn.in_transaction
        assert _count(conn, "position_snapshots") == 2  # pos-3 rolled back with pos-1

    def test_foreign_key_violation_is_store_error(self, conn):
        orphan = Reasoning(
            id="reas-orphan", proposal_id="no-such-proposal", stage=ReasoningStage.SCAN, content="x"
        )
        with pytest.raises(StoreError, match="add reasoning for reas-orphan") as info:
            add_reasoning(conn, orphan)
        assert isinstance(info.value.cause, sqlite3.IntegrityError)
        assert "FOREIGN KEY" in str(info.value)
        assert (info.value.what, info.value.key) == ("add reasoning", "reas-orphan")
        assert not conn.in_transaction
        assert _count(conn, "reasoning") == 0
        with pytest.raises(StoreError, match="record fill for fill-orphan"):
            record_fill(conn, Fill(id="fill-orphan", order_id="no-such-order", fill_price=1.0, fill_quantity=1))
        with pytest.raises(StoreError, match="record decision for dec-orphan"):
            record_decision(conn, PolicyDecision(
                id="dec-orphan", proposal_id="no-such-proposal", verdict=Verdict.REJECT, rules_evaluated=[],
            ))
        with pytest.raises(StoreError, match="upsert order for coid-orphan"):
            upsert_order(conn, Order(
                proposal_id="no-such-proposal", client_order_id="coid-orphan", broker=Broker.PAPER,
                status=OrderStatus.PROPOSED, symbol="SPY", side=OrderSide.BUY, quantity=1,
            ))
        with pytest.raises(StoreError, match="record approval for appr-orphan") as info:
            record_approval(conn, Approval(id="appr-orphan", proposal_id="no-such-proposal", channel="cli"))
        assert isinstance(info.value.cause, sqlite3.IntegrityError)
        assert {table: _count(conn, table) for table in TABLES} == _FRESH_COUNTS  # nothing was written

    def test_duplicate_proposal_id_is_store_error(self, conn, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        assert insert_proposal(conn, proposal) == proposal
        with pytest.raises(StoreError, match="insert proposal for prop-0001") as info:
            insert_proposal(conn, proposal)
        assert isinstance(info.value.cause, sqlite3.IntegrityError)
        assert _count(conn, "proposals") == 1

    def test_duplicate_ids_are_store_errors(self, conn, trace_data):
        """Every id is its table's PRIMARY KEY: a retried write with the same id is refused, never a
        second row (a double-counted fill would corrupt the trace). Approvals upsert by design."""
        _seed_trace(conn, trace_data)
        log_event(conn, Event(id="evt-0001", level=EventLevel.INFO, kind="k", message="m"))
        repeats = (
            ("add reasoning for reas-0001-scan",
             lambda: add_reasoning(conn, Reasoning.model_validate(trace_data["reasoning"][0]))),
            ("record decision for dec-0001",
             lambda: record_decision(conn, PolicyDecision.model_validate(trace_data["decision"]))),
            ("record fill for fill-0001", lambda: record_fill(conn, Fill.model_validate(trace_data["fill"]))),
            ("upsert order for coid-other",  # a second client_order_id may not reuse an order id
             lambda: upsert_order(conn, Order.model_validate({**trace_data["order"], "client_order_id": "coid-other"}))),
            ("log event for evt-0001",
             lambda: log_event(conn, Event(id="evt-0001", level=EventLevel.INFO, kind="k", message="again"))),
        )
        for what, write in repeats:
            with pytest.raises(StoreError, match=what) as info:
                write()
            assert isinstance(info.value.cause, sqlite3.IntegrityError), what
            assert "UNIQUE constraint failed" in str(info.value), what
            assert not conn.in_transaction
        assert {table: _count(conn, table) for table in TABLES} == {
            "proposals": 1, "proposal_legs": 0, "reasoning": 3, "policy_decisions": 1, "approvals": 1,
            "orders": 1, "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 1, "controls": 2,
        }
        trace = get_proposal_trace(conn, "prop-0001")
        assert len(trace.reasoning) == 3 and len(trace.decisions) == 1 and len(trace.fills) == 1

    def test_int_too_large_for_sqlite_is_store_error(self, conn, trace_data):
        """Python ints are unbounded, a SQLite INTEGER is 64 bits, and the overflow is not a sqlite3.Error."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        with pytest.raises(StoreError, match="add reasoning for reas-big") as info:
            add_reasoning(conn, Reasoning(
                id="reas-big", proposal_id=proposal.id, stage=ReasoningStage.SCAN, content="c", tokens_in=2**63,
            ))
        assert isinstance(info.value.cause, OverflowError)
        assert not conn.in_transaction
        assert _count(conn, "reasoning") == 0
        with pytest.raises(StoreError, match="get recent events") as info:
            get_recent_events(conn, limit=2**63)
        assert isinstance(info.value.cause, OverflowError)
        largest = add_reasoning(conn, Reasoning(
            id="reas-max", proposal_id=proposal.id, stage=ReasoningStage.SCAN, content="c", tokens_in=2**63 - 1,
        ))
        assert get_proposal_trace(conn, proposal.id).reasoning == (largest,)

    def test_unknown_proposal_trace_is_store_error(self, conn):
        with pytest.raises(StoreError, match="proposal trace.* for nope") as info:
            get_proposal_trace(conn, "nope")
        assert info.value.key == "nope" and info.value.cause is None
        assert get_proposal(conn, "nope") is None

    @pytest.mark.parametrize(
        "write",
        [
            lambda c: insert_proposal(c, _bare_proposal("prop-0002")),
            lambda c: insert_proposal(c, _bare_proposal("prop-0003"), _legs("prop-0003")),
            lambda c: add_reasoning(c, Reasoning(proposal_id="prop-0001", stage=ReasoningStage.SCAN, content="c")),
            lambda c: link_reasoning_to_proposal(c, "cycle-0001", "prop-0001"),
            lambda c: record_decision(c, PolicyDecision(proposal_id="prop-0001", verdict=Verdict.REJECT, rules_evaluated=[])),
            lambda c: record_approval(c, Approval(proposal_id="prop-0001", channel="cli")),
            lambda c: upsert_order(c, _bare_order("ord-0009", "coid-0009")),
            lambda c: record_fill(c, Fill(order_id="ord-0001", fill_price=1.0, fill_quantity=1)),
            lambda c: snapshot_positions(c, [PositionSnapshot(symbol="SPY", quantity=1)]),
            lambda c: snapshot_pnl(c, _pnl(AWARE, 1.0)),
            lambda c: log_event(c, Event(level=EventLevel.INFO, kind="k", message="m")),
            lambda c: record_decision(
                c, PolicyDecision(proposal_id="prop-0001", verdict=Verdict.REJECT, rules_evaluated=[]),
                event=Event(level=EventLevel.WARNING, kind="policy_reject", message="m"),
            ),
            lambda c: set_kill_switch(c, True),
            lambda c: set_kill_switch(c, True, event=Event(level=EventLevel.WARNING, kind="kill_switch", message="m")),
            lambda c: set_halt_until(c, AWARE),
            lambda c: set_halt_until(c, None, event=Event(level=EventLevel.INFO, kind="k", message="m")),
        ],
        ids=[
            "insert_proposal", "insert_proposal_with_legs", "add_reasoning", "link_reasoning_to_proposal",
            "record_decision", "record_approval", "upsert_order", "record_fill", "snapshot_positions",
            "snapshot_pnl", "log_event", "record_decision_with_event", "set_kill_switch",
            "set_kill_switch_with_event", "set_halt_until", "set_halt_until_with_event",
        ],
    )
    def test_writes_never_join_a_callers_transaction(self, conn, trace_data, write):
        """Every write opens its own transaction — none autocommits into a caller's block."""
        _seed_trace(conn, trace_data)  # prop-0001 and ord-0001 exist for the FK-bearing writes
        counts = {table: _count(conn, table) for table in TABLES}
        controls = _control_rows(conn)
        with pytest.raises(StoreError, match="not re-entrant"):
            with transaction(conn):
                write(conn)
        assert not conn.in_transaction
        assert {table: _count(conn, table) for table in TABLES} == counts
        assert _control_rows(conn) == controls

    def test_reads_wrap_sqlite_errors(self, conn, trace_data):
        """A read that fails in SQLite surfaces as StoreError naming the operation and id."""
        _seed_trace(conn, trace_data)
        conn.execute("PRAGMA foreign_keys=OFF")  # so the parent tables can go
        for statement in (
            "DROP TABLE events", "DROP TABLE pnl_snapshots", "DROP TABLE orders", "DROP TABLE proposals",
            "DROP TABLE proposal_legs", "DROP TABLE reasoning", "DROP TABLE controls",
        ):
            conn.execute(statement)
        for what, call in (
            ("get controls", lambda: get_controls(conn)),
            ("get proposals since for 2026-07-30 14:00:00", lambda: get_proposals_since(conn, AWARE)),
            ("get latest undecided proposal", lambda: get_latest_undecided_proposal(conn)),
            ("count orders submitted between for 2026-07-30 14:00:00.* \\.\\. 2026-07-31 14:00:00",
             lambda: count_orders_submitted_between(conn, AWARE, AWARE + timedelta(days=1))),
            ("get recent events", lambda: get_recent_events(conn)),
            ("get daily pnl", lambda: get_daily_pnl(conn)),
            ("get open orders", lambda: get_open_orders(conn)),
            ("get order for aegis-prop-0001-1", lambda: get_order(conn, "aegis-prop-0001-1")),
            ("get proposal for prop-0001", lambda: get_proposal(conn, "prop-0001")),
            ("get recent proposals", lambda: get_recent_proposals(conn)),
            ("get proposal legs for prop-0001", lambda: get_proposal_legs(conn, "prop-0001")),
            ("get cycle reasoning for cycle-0001", lambda: get_cycle_reasoning(conn, "cycle-0001")),
            ("get token usage for 2026-07-30", lambda: get_token_usage(conn, date(2026, 7, 30))),
            ("get cycle token usage for cycle-0001", lambda: get_cycle_token_usage(conn, "cycle-0001")),
            ("get token usage by model for 2026-07-30", lambda: get_token_usage_by_model(conn, date(2026, 7, 30))),
            ("proposal trace for prop-0001", lambda: get_proposal_trace(conn, "prop-0001")),
        ):
            with pytest.raises(StoreError, match=what) as info:
                call()
            assert isinstance(info.value.cause, sqlite3.OperationalError)
            assert "no such table" in str(info.value)

    def test_reads_wrap_malformed_rows(self, conn, trace_data):
        """A row that will not convert back into a record is a StoreError, not a bare ValueError."""
        _seed_trace(conn, trace_data)
        conn.execute("UPDATE proposals SET created_at = ? WHERE id = ?", ("garbage", "prop-0001"))
        with pytest.raises(StoreError, match="get proposal for prop-0001") as info:
            get_proposal(conn, "prop-0001")
        assert isinstance(info.value.cause, ValueError)
        with pytest.raises(StoreError, match="proposal trace for prop-0001") as info:
            get_proposal_trace(conn, "prop-0001")
        assert isinstance(info.value.cause, ValueError)
        with pytest.raises(StoreError, match="get proposals since") as info:  # "garbage" sorts after any ISO bound
            get_proposals_since(conn, AWARE)
        assert isinstance(info.value.cause, ValueError)
        conn.execute("DELETE FROM policy_decisions")
        with pytest.raises(StoreError, match="get latest undecided proposal") as info:
            get_latest_undecided_proposal(conn)
        assert isinstance(info.value.cause, ValueError)

    def test_duplicate_pnl_snapshot_is_store_error(self, conn):
        snapshot = snapshot_pnl(conn, _pnl(AWARE, 1.0).model_copy(update={"id": "pnl-0001"}))
        with pytest.raises(StoreError, match="snapshot pnl for pnl-0001") as info:
            snapshot_pnl(conn, snapshot)
        assert isinstance(info.value.cause, sqlite3.IntegrityError)
        assert not conn.in_transaction
        assert _count(conn, "pnl_snapshots") == 1

    def test_events_round_trip_and_recent(self, conn):
        payload = {"pct": 2.5, "symbols": ["SPY", "QQQ"], "nested": {"ok": True, "n": None}}
        event = log_event(conn, Event(
            level=EventLevel.WARNING, kind="risk_limit", message="tripped", payload=payload, occurred_at=AWARE,
        ))
        [back] = get_recent_events(conn)
        assert back == event and back.payload == payload
        assert back.occurred_at == AWARE and back.occurred_at.tzinfo is timezone.utc
        assert conn.execute("SELECT payload FROM events").fetchone()[0].startswith("{")
        levels = (EventLevel.INFO, EventLevel.ERROR, EventLevel.INFO, EventLevel.DEBUG)
        for offset, level in enumerate(levels, start=1):
            log_event(conn, Event(
                level=level, kind="k", message="m" + str(offset), occurred_at=AWARE + timedelta(minutes=offset),
            ))
        recent = get_recent_events(conn)
        assert [e.message for e in recent] == ["m4", "m3", "m2", "m1", "tripped"]
        assert [e.occurred_at for e in recent] == sorted((e.occurred_at for e in recent), reverse=True)
        assert [e.message for e in get_recent_events(conn, level=EventLevel.INFO)] == ["m3", "m1"]
        assert get_recent_events(conn, limit=2) == recent[:2]
        assert get_recent_events(conn, limit=1, level=EventLevel.INFO) == [recent[1]]
        assert get_recent_events(conn, level=EventLevel.CRITICAL) == []
        heartbeat = log_event(conn, Event(level=EventLevel.DEBUG, kind="heartbeat", message="ok"))
        assert get_recent_events(conn, level=EventLevel.DEBUG)[0].payload is None
        assert heartbeat.payload is None

    def test_seed_helper_returns_records_as_stored(self, conn, trace_data):
        proposal, reasoning, decision, approval, order, fill = _seed_trace(conn, trace_data)
        trace = get_proposal_trace(conn, proposal.id)
        assert trace == ProposalTrace(
            proposal=proposal, reasoning=tuple(reasoning), decisions=(decision,),
            approvals=(approval,), orders=(order,), fills=(fill,),
        )
        assert {table: _count(conn, table) for table in TABLES} == {
            "proposals": 1, "proposal_legs": 0, "reasoning": 3, "policy_decisions": 1, "approvals": 1,
            "orders": 1, "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 0, "controls": 2,
        }

    def test_repo_functions_are_re_exported(self):
        for name in (
            "insert_proposal", "add_reasoning", "link_reasoning_to_proposal", "record_decision", "record_approval",
            "upsert_order", "record_fill", "snapshot_positions", "snapshot_pnl", "log_event", "get_proposal",
            "get_recent_proposals", "get_proposal_legs", "get_cycle_reasoning", "get_order", "get_open_orders",
            "get_daily_pnl", "get_token_usage", "get_cycle_token_usage", "get_token_usage_by_model",
            "get_recent_events", "get_proposal_trace", "get_controls", "set_kill_switch", "set_halt_until",
            "get_proposals_since", "get_latest_undecided_proposal", "count_orders_submitted_between",
        ):
            assert getattr(aegis.store, name) is getattr(repo, name)
            assert name in aegis.store.__all__
        for name in ("ProposalLeg", "TokenUsage", "Controls"):
            assert getattr(aegis.store, name) is getattr(store_models, name)
            assert name in aegis.store.__all__

    def test_reasoning_rows_live_under_a_cycle_before_any_proposal(self, conn):
        """Scan and thesis rows are written before a proposal exists (and never get one on NO_TRADE)."""
        late = AWARE + timedelta(minutes=1)
        thesis = add_reasoning(conn, Reasoning(
            id="reas-thesis", cycle_id="cycle-1", stage=ReasoningStage.THESIS, content="t", created_at=late,
            tokens_in=1500, tokens_out=220, model_name="claude-opus-5-5", latency_ms=812.5,
        ))
        scan = add_reasoning(conn, Reasoning(  # written second but earlier: the read orders by created_at
            id="reas-scan", cycle_id="cycle-1", stage=ReasoningStage.SCAN, content="s", created_at=AWARE,
            tokens_in=1200, tokens_out=180, model_name="claude-haiku-4-5-20251001", latency_ms=301.0,
        ))
        other = add_reasoning(conn, Reasoning(id="reas-other", cycle_id="cycle-2", stage=ReasoningStage.SCAN, content="o"))
        assert get_cycle_reasoning(conn, "cycle-1") == [scan, thesis]
        assert get_cycle_reasoning(conn, "cycle-2") == [other]
        assert get_cycle_reasoning(conn, "cycle-none") == []
        assert scan.proposal_id is None and other.model_name is None and other.latency_ms is None
        row = conn.execute(
            "SELECT cycle_id, proposal_id, model_name, latency_ms FROM reasoning WHERE id = ?", ("reas-scan",)
        ).fetchone()
        assert tuple(row) == ("cycle-1", None, "claude-haiku-4-5-20251001", 301.0)
        assert _count(conn, "proposals") == 0  # no proposal was ever needed
        for record in (scan, thesis, other):
            assert Reasoning.model_validate(record.model_dump(mode="json")) == record

    def test_link_reasoning_to_proposal_links_exactly_the_cycles_unlinked_rows(self, conn, trace_data):
        for n, stage in enumerate(ReasoningStage):
            add_reasoning(conn, Reasoning(
                id="reas-" + stage.value, cycle_id="cycle-0001", stage=stage, content="c",
                created_at=AWARE + timedelta(minutes=n),
            ))
        add_reasoning(conn, Reasoning(id="reas-other-cycle", cycle_id="cycle-0002", stage=ReasoningStage.SCAN, content="c"))
        insert_proposal(conn, _bare_proposal("prop-earlier").model_copy(update={"cycle_id": "cycle-0001"}))
        add_reasoning(conn, Reasoning(  # the same cycle, already linked elsewhere: must be left alone
            id="reas-taken", cycle_id="cycle-0001", proposal_id="prop-earlier", stage=ReasoningStage.PROPOSAL, content="c",
        ))
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))  # cycle-0001
        assert get_proposal_trace(conn, proposal.id).reasoning == ()

        assert link_reasoning_to_proposal(conn, "cycle-0001", proposal.id) == 3
        assert not conn.in_transaction
        trace = get_proposal_trace(conn, proposal.id)
        assert [r.id for r in trace.reasoning] == ["reas-scan", "reas-thesis", "reas-proposal"]
        assert {r.cycle_id for r in trace.reasoning} == {"cycle-0001"}
        assert {r.proposal_id for r in trace.reasoning} == {proposal.id}
        assert [r.id for r in get_proposal_trace(conn, "prop-earlier").reasoning] == ["reas-taken"]
        assert get_cycle_reasoning(conn, "cycle-0002")[0].proposal_id is None  # another cycle: untouched
        assert link_reasoning_to_proposal(conn, "cycle-0001", proposal.id) == 0  # nothing left to link
        assert link_reasoning_to_proposal(conn, "cycle-none", proposal.id) == 0
        with pytest.raises(
            StoreError, match="link reasoning to proposal for cycle cycle-0002 -> proposal no-such-proposal"
        ) as info:
            link_reasoning_to_proposal(conn, "cycle-0002", "no-such-proposal")
        assert isinstance(info.value.cause, sqlite3.IntegrityError) and "FOREIGN KEY" in str(info.value)
        assert not conn.in_transaction
        assert get_cycle_reasoning(conn, "cycle-0002")[0].proposal_id is None  # rolled back

    def test_insert_proposal_can_link_its_cycles_reasoning_in_the_same_transaction(self, conn, trace_data):
        """The proposal, its legs and the link of its cycle's unlinked reasoning rows land
        together or not at all: a failing link rolls the proposal and its legs back."""
        for stage in ReasoningStage:
            add_reasoning(conn, Reasoning(id="reas-" + stage.value, cycle_id="cycle-0001", stage=stage, content="c"))
        add_reasoning(conn, Reasoning(id="reas-other", cycle_id="cycle-0002", stage=ReasoningStage.SCAN, content="c"))
        proposal = Proposal.model_validate(trace_data["proposal"])  # cycle-0001
        conn.execute(
            "CREATE TRIGGER refuse_link BEFORE UPDATE OF proposal_id ON reasoning"
            " BEGIN SELECT RAISE(ABORT, 'link refused'); END"
        )
        with pytest.raises(StoreError, match=f"insert proposal for {proposal.id} .*link refused"):
            insert_proposal(conn, proposal, _legs(proposal.id), link_cycle_reasoning=True)
        assert not conn.in_transaction
        assert get_proposal(conn, proposal.id) is None and get_proposal_legs(conn, proposal.id) == []
        assert {r.proposal_id for r in get_cycle_reasoning(conn, "cycle-0001")} == {None}

        conn.execute("DROP TRIGGER refuse_link")
        assert insert_proposal(conn, proposal, _legs(proposal.id), link_cycle_reasoning=True) == proposal
        trace = get_proposal_trace(conn, proposal.id)
        assert [r.id for r in trace.reasoning] == ["reas-scan", "reas-thesis", "reas-proposal"]
        assert len(trace.legs) == 2
        assert get_cycle_reasoning(conn, "cycle-0002")[0].proposal_id is None  # another cycle: untouched
        # without the flag nothing is linked (the default, as before)
        other = _bare_proposal("prop-unlinked").model_copy(update={"cycle_id": "cycle-0002"})
        insert_proposal(conn, other)
        assert get_cycle_reasoning(conn, "cycle-0002")[0].proposal_id is None

    def test_legs_round_trip_through_insert_proposal_and_trace(self, conn, trace_data):
        proposal = Proposal.model_validate(trace_data["proposal"])
        legs = _legs(proposal.id)
        assert insert_proposal(conn, proposal, legs[::-1]) == proposal  # written index 1 first
        trace = get_proposal_trace(conn, proposal.id)
        assert trace.legs == tuple(legs) and get_proposal_legs(conn, proposal.id) == legs
        assert [leg.leg_index for leg in trace.legs] == [0, 1]
        assert trace.legs[0].expiration == date(2026, 10, 16) and trace.legs[1].side is OrderSide.SELL
        rows = conn.execute(
            "SELECT leg_index, symbol, option_type, side, quantity, strike, expiration FROM proposal_legs"
            " ORDER BY leg_index"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (0, "SPY261016C00640000", "call", "buy", 1.0, 640.0, "2026-10-16"),
            (1, "SPY261016C00650000", "call", "sell", 1.0, 650.0, "2026-10-16"),
        ]
        # an equity proposal has none, and another proposal's legs never leak into a trace
        equity = insert_proposal(conn, _bare_proposal("prop-0002"))
        assert get_proposal_legs(conn, equity.id) == [] and get_proposal_trace(conn, equity.id).legs == ()
        assert get_proposal_legs(conn, "nope") == []
        assert {leg.proposal_id for leg in get_proposal_trace(conn, proposal.id).legs} == {proposal.id}
        for record in legs:
            assert ProposalLeg.model_validate(record.model_dump(mode="json")) == record

    def test_insert_proposal_with_legs_is_all_or_none(self, conn):
        proposal = _bare_proposal("prop-0003")
        stray = _legs("prop-other", 1)
        with pytest.raises(StoreError, match=r"insert proposal \(leg 0 belongs to proposal prop-other\) for prop-0003") as info:
            insert_proposal(conn, proposal, stray)
        assert info.value.cause is None
        assert _count(conn, "proposals") == 0 and _count(conn, "proposal_legs") == 0  # refused before any write
        clashing = _legs(proposal.id) + [_legs(proposal.id, 1)[0].model_copy(update={"id": "leg-dup"})]  # index 0 twice
        with pytest.raises(StoreError, match="insert proposal for prop-0003") as info:
            insert_proposal(conn, proposal, clashing)
        assert isinstance(info.value.cause, sqlite3.IntegrityError) and "UNIQUE" in str(info.value)
        assert not conn.in_transaction
        assert _count(conn, "proposals") == 0 and _count(conn, "proposal_legs") == 0  # the proposal rolled back too
        insert_proposal(conn, proposal, _legs(proposal.id))
        assert _count(conn, "proposals") == 1 and _count(conn, "proposal_legs") == 2
        legless = _bare_proposal("prop-0004")
        assert insert_proposal(conn, legless, []) == legless and get_proposal_legs(conn, legless.id) == []

    def test_token_usage_by_utc_day_cycle_and_model(self, conn, monkeypatch):
        day = date(2026, 7, 30)
        eastern = timezone(timedelta(hours=-4))
        rows = (
            ("a", "cycle-1", datetime(2026, 7, 30, 9, 30, tzinfo=timezone.utc), 1000, 100, "haiku"),
            ("b", "cycle-1", datetime(2026, 7, 30, 23, 59, 59, tzinfo=timezone.utc), 2000, None, "opus"),  # NULL out → 0
            ("c", "cycle-2", datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc), None, None, None),  # no usage, no model
            ("d", "cycle-2", datetime(2026, 7, 30, 20, 0, tzinfo=eastern), 4000, 400, "haiku"),  # 00:00Z on the 31st
            ("e", "cycle-3", datetime(2026, 7, 29, 23, 59, tzinfo=timezone.utc), 8000, 800, "opus"),
        )
        for id_, cycle_id, created_at, tokens_in, tokens_out, model_name in rows:
            add_reasoning(conn, Reasoning(
                id=id_, cycle_id=cycle_id, stage=ReasoningStage.SCAN, content="c", created_at=created_at,
                tokens_in=tokens_in, tokens_out=tokens_out, model_name=model_name,
            ))
        usage = get_token_usage(conn, day)
        assert usage == TokenUsage(tokens_in=3000, tokens_out=100, calls=3) and usage.total == 3100
        assert get_token_usage(conn, date(2026, 7, 31)) == TokenUsage(tokens_in=4000, tokens_out=400, calls=1)
        assert get_token_usage(conn, date(2026, 7, 28)) == TokenUsage() and TokenUsage().total == 0
        assert get_token_usage(conn, datetime(2026, 7, 30, 20, 0, tzinfo=eastern)).calls == 1  # the 31st, in UTC
        assert get_token_usage(conn, datetime(2026, 7, 30, 12, 0)).calls == 3  # naive is UTC
        monkeypatch.setattr(repo, "utcnow", lambda: datetime(2026, 7, 30, 18, 0, tzinfo=timezone.utc))
        assert get_token_usage(conn) == usage
        assert get_cycle_token_usage(conn, "cycle-1") == TokenUsage(tokens_in=3000, tokens_out=100, calls=2)
        assert get_cycle_token_usage(conn, "cycle-2") == TokenUsage(tokens_in=4000, tokens_out=400, calls=2)
        assert get_cycle_token_usage(conn, "cycle-none") == TokenUsage()
        by_model = get_token_usage_by_model(conn, day)
        assert by_model == {
            "haiku": TokenUsage(tokens_in=1000, tokens_out=100, calls=1),
            "opus": TokenUsage(tokens_in=2000, tokens_out=0, calls=1),
            "unknown": TokenUsage(tokens_in=0, tokens_out=0, calls=1),
        }
        assert list(by_model) == ["haiku", "opus", "unknown"]
        assert get_token_usage_by_model(conn) == by_model
        assert get_token_usage_by_model(conn, date(2026, 7, 28)) == {}
        assert sum(u.total for u in by_model.values()) == usage.total

    def test_get_recent_proposals_newest_first(self, conn, trace_data):
        assert get_recent_proposals(conn) == []
        base = Proposal.model_validate(trace_data["proposal"])
        for n in (2, 0, 1):  # inserted out of order: the read must sort by created_at, not rowid
            insert_proposal(conn, base.model_copy(update={"id": f"prop-{n}", "created_at": AWARE + timedelta(minutes=n)}))
        tie = insert_proposal(conn, base.model_copy(update={"id": "prop-tie", "created_at": AWARE + timedelta(minutes=2)}))
        assert [p.id for p in get_recent_proposals(conn)] == ["prop-tie", "prop-2", "prop-1", "prop-0"]
        assert [p.id for p in get_recent_proposals(conn, limit=2)] == ["prop-tie", "prop-2"]
        assert get_recent_proposals(conn, limit=0) == []
        assert get_recent_proposals(conn)[0] == tie == get_proposal(conn, "prop-tie")

    def test_get_proposals_since_bounds_and_order(self, conn, trace_data):
        """What the duplicate rule looks through: created at or after the bound, newest first."""
        assert get_proposals_since(conn, AWARE) == []
        base = Proposal.model_validate(trace_data["proposal"])
        minute, micro = timedelta(minutes=1), timedelta(microseconds=1)
        created = {  # inserted out of order: the read must sort by created_at, not rowid
            "prop-2": AWARE + 2 * minute, "prop-0": AWARE, "prop-3": AWARE + 3 * minute, "prop-1": AWARE + minute,
            "prop-tie": AWARE + 2 * minute, "prop-frac": AWARE + minute + micro,
        }
        for proposal_id, created_at in created.items():
            insert_proposal(conn, base.model_copy(update={"id": proposal_id, "created_at": created_at}))

        def since(bound):
            return [proposal.id for proposal in get_proposals_since(conn, bound)]

        # newest first; the 14:02 tie breaks on rowid, the later insert first
        everything = ["prop-3", "prop-tie", "prop-2", "prop-frac", "prop-1", "prop-0"]
        assert since(AWARE - micro) == everything
        assert since(AWARE) == everything  # inclusive: prop-0 was created exactly at the bound
        assert since(AWARE + micro) == everything[:-1]  # one microsecond later it is out
        assert since(AWARE + minute) == everything[:-1]  # prop-1 at the bound
        assert since(AWARE + minute + micro) == everything[:-2]  # prop-frac at the bound, prop-1 just before it
        assert since(AWARE + minute + 2 * micro) == ["prop-3", "prop-tie", "prop-2"]
        assert since(AWARE + 3 * minute) == ["prop-3"]
        assert since(AWARE + 3 * minute + micro) == []
        assert since(NAIVE + 2 * minute) == ["prop-3", "prop-tie", "prop-2"]  # naive is UTC
        eastern = timezone(timedelta(hours=-4))
        assert since(datetime(2026, 7, 30, 10, 2, tzinfo=eastern)) == ["prop-3", "prop-tie", "prop-2"]  # 14:02Z
        assert get_proposals_since(conn, AWARE + 3 * minute) == [get_proposal(conn, "prop-3")]
        assert get_proposals_since(conn, AWARE) == get_recent_proposals(conn)  # one ordering, both reads
        for bad in ("2026-07-30T14:00:00+00:00", date(2026, 7, 30), None):  # a bound must be a datetime
            with pytest.raises(StoreError, match="get proposals since") as info:
                get_proposals_since(conn, bad)
            assert isinstance(info.value.cause, TypeError)

    def test_get_latest_undecided_proposal(self, conn, trace_data):
        """``evaluate --latest``: the newest proposal that has no policy decision at all."""
        assert get_latest_undecided_proposal(conn) is None  # an empty store
        base = Proposal.model_validate(trace_data["proposal"])

        def propose(proposal_id, minutes):
            return insert_proposal(conn, base.model_copy(
                update={"id": proposal_id, "created_at": AWARE + timedelta(minutes=minutes)}
            ))

        def decide(proposal_id, verdict):
            record_decision(conn, PolicyDecision(proposal_id=proposal_id, verdict=verdict, rules_evaluated=[]))

        # inserted newest-first, so rowid order and created_at order disagree
        new, mid, old = propose("prop-new", 2), propose("prop-mid", 1), propose("prop-old", 0)
        assert get_latest_undecided_proposal(conn) == new == get_proposal(conn, "prop-new")
        decide("prop-new", Verdict.REJECT)
        assert get_latest_undecided_proposal(conn) == mid  # a decided proposal is skipped, whatever the verdict
        decide("prop-mid", Verdict.NEEDS_APPROVAL)
        decide("prop-mid", Verdict.REJECT)  # judged twice is decided, and appears once
        assert get_latest_undecided_proposal(conn) == old
        decide("prop-old", Verdict.AUTO_EXECUTE)
        assert get_latest_undecided_proposal(conn) is None  # every proposal has a decision
        # a tie on created_at breaks on rowid: the later insert is the newer
        first, second = propose("prop-tie-a", 5), propose("prop-tie-b", 5)
        assert get_latest_undecided_proposal(conn) == second
        decide("prop-tie-b", Verdict.FLAG_ONLY)
        assert get_latest_undecided_proposal(conn) == first
        backdated = propose("prop-backdated", -60)  # written last, created first: not the latest
        assert get_latest_undecided_proposal(conn) == first
        decide("prop-tie-a", Verdict.REJECT)
        assert get_latest_undecided_proposal(conn) == backdated
        assert _count(conn, "proposals") == 6 and _count(conn, "policy_decisions") == 6

    def test_count_orders_submitted_between(self, conn, trace_data):
        """The daily trade cap's count: orders sent to the broker in ``[start, end)``, failed ones excepted."""
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        start = datetime(2026, 7, 30, 4, 0, tzinfo=timezone.utc)  # 00:00 in New York: the caller's trading day
        end = start + timedelta(days=1)
        micro = timedelta(microseconds=1)
        assert count_orders_submitted_between(conn, start, end) == 0
        orders = (
            ("before", start - micro, OrderStatus.FILLED),  # the previous day's last instant
            ("at-start", start, OrderStatus.FILLED),  # inclusive
            ("open", start + timedelta(hours=10), OrderStatus.SUBMITTED),
            ("partial", start + timedelta(hours=11), OrderStatus.PARTIALLY_FILLED),
            ("cancelled", end - micro, OrderStatus.CANCELLED),  # sent, then cancelled: it was still sent
            ("at-end", end, OrderStatus.FILLED),  # exclusive
            ("failed", start + timedelta(hours=12), OrderStatus.FAILED),  # the broker never accepted it
            ("never-sent", None, OrderStatus.APPROVED),  # NULL submitted_at
            ("never-sent-failed", None, OrderStatus.FAILED),
        )
        for name, submitted_at, state in orders:
            upsert_order(conn, _bare_order(
                "ord-" + name, "coid-" + name, status=state, submitted_at=submitted_at, updated_at=AWARE,
            ))
        assert _count(conn, "orders") == 9

        def count(lower, upper):
            return count_orders_submitted_between(conn, lower, upper)

        assert count(start, end) == 4  # at-start, open, partial, cancelled
        assert count(start - micro, end) == 5 and count(start + micro, end) == 3
        assert count(start, end + micro) == 5 and count(start, end - micro) == 3
        assert count(start - timedelta(days=1), start) == 1  # "before" belongs to the day before
        assert count(end, end + timedelta(days=1)) == 1  # and "at-end" to the day after
        assert count(start, start) == 0 and count(end, start) == 0  # an empty and an inverted window
        assert count(start - timedelta(days=365), end + timedelta(days=365)) == 6  # never the NULLs or the failed
        assert count(start.replace(tzinfo=None), end.replace(tzinfo=None)) == 4  # naive is UTC
        eastern = timezone(timedelta(hours=-4))
        assert count(datetime(2026, 7, 30, tzinfo=eastern), datetime(2026, 7, 31, tzinfo=eastern)) == 4
        # an order that fails afterwards stops counting; a status update never counts one twice
        upsert_order(conn, _bare_order("ord-x", "coid-open", status=OrderStatus.FAILED, updated_at=AWARE))
        assert count(start, end) == 3
        upsert_order(conn, _bare_order("ord-y", "coid-partial", status=OrderStatus.FILLED, updated_at=AWARE))
        assert count(start, end) == 3 and _count(conn, "orders") == 9
        for bad in (("2026-07-30", end), (start, date(2026, 7, 31)), (None, end)):  # bounds must be datetimes
            with pytest.raises(StoreError, match="count orders submitted between") as info:
                count(*bad)
            assert isinstance(info.value.cause, TypeError)


class TestControls:
    """Migration 0003 and the repository's control functions: the kill switch and the daily-loss halt."""

    def test_table_shape_and_seed_rows(self, conn):
        rows = conn.execute('SELECT name, type, "notnull", pk FROM pragma_table_info(?)', ("controls",)).fetchall()
        assert [tuple(row) for row in rows] == [
            ("key", "TEXT", 1, 1), ("value", "TEXT", 1, 0), ("updated_at", "TEXT", 1, 0),
        ]
        assert [row[:2] for row in _control_rows(conn)] == [("halt_until", ""), ("kill_switch", "off")]
        assert _triggers(conn, "controls") == [("controls_keep_rows", "controls")]
        assert list(Controls.model_fields) == [
            "kill_switch", "halt_until", "halt_unknown", "kill_switch_updated_at", "halt_until_updated_at", "problems",
        ]

    def test_check_constraints(self, conn):
        """A typo'd key and an unreadable value are refused at the write, however it is phrased."""
        ts = AWARE.isoformat()
        seeded = _control_rows(conn)
        for key in ("killswitch", "KILL_SWITCH", "kill_switch ", "halt", "halt_untill", ""):
            for value in ("on", ""):
                with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                    conn.execute("INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)", (key, value, ts))
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute("UPDATE controls SET key = ? WHERE key = ?", (key, "kill_switch"))
        bad_values = (
            ("kill_switch", "maybe"), ("kill_switch", ""), ("kill_switch", "ON"), ("kill_switch", "1"),
            ("kill_switch", "2026-07-30T14:00:00+00:00"),  # the other control's kind of value
            ("halt_until", "tomorrow"), ("halt_until", "off"), ("halt_until", " "), ("halt_until", "2026-07-30"),
            ("halt_until", "2026-07-30 14:00:00+00:00"), ("halt_until", " 2026-07-30T14:00:00+00:00"),
            ("halt_until", "26-07-30T14:00:00"),
        )
        for key, value in bad_values:
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute("UPDATE controls SET value = ? WHERE key = ?", (value, key))
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute("INSERT OR REPLACE INTO controls (key, value, updated_at) VALUES (?, ?, ?)", (key, value, ts))
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            conn.execute("UPDATE controls SET value = NULL WHERE key = ?", ("halt_until",))
        with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
            conn.execute("UPDATE controls SET updated_at = NULL WHERE key = ?", ("kill_switch",))
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):  # one row per control
            conn.execute("INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)", ("kill_switch", "on", ts))
        assert _control_rows(conn) == seeded  # nothing above changed a thing
        good_values = (
            ("kill_switch", "on"), ("kill_switch", "off"), ("halt_until", "2026-07-30T14:00:00+00:00"),
            ("halt_until", "2026-07-30T14:00:00.123456+00:00"), ("halt_until", "2026-07-30T14:00:00"), ("halt_until", ""),
        )
        for key, value in good_values:
            conn.execute("UPDATE controls SET value = ? WHERE key = ?", (value, key))
            assert (key, value) in [row[:2] for row in _control_rows(conn)]

    def test_rows_are_never_deleted(self, conn):
        refusal = "controls rows are never deleted: set the value instead"
        for statement in (
            "DELETE FROM controls WHERE key = 'kill_switch'", "DELETE FROM controls WHERE key = 'halt_until'",
            "DELETE FROM controls",
        ):
            with pytest.raises(sqlite3.IntegrityError, match=refusal):
                conn.execute(statement)
        with pytest.raises(sqlite3.IntegrityError, match=refusal):  # inside a transaction too, which then rolls back
            with transaction(conn):
                conn.execute("UPDATE controls SET value = 'on' WHERE key = 'kill_switch'")
                conn.execute("DELETE FROM controls WHERE key = 'halt_until'")
        assert not conn.in_transaction
        assert [row[:2] for row in _control_rows(conn)] == [("halt_until", ""), ("kill_switch", "off")]
        # REPLACE swaps a row for its successor without ever leaving the key absent
        conn.execute(
            "INSERT OR REPLACE INTO controls (key, value, updated_at) VALUES (?, ?, ?)",
            ("kill_switch", "on", AWARE.isoformat()),
        )
        assert _control_rows(conn)[1] == ("kill_switch", "on", AWARE.isoformat()) and _count(conn, "controls") == 2

    def test_rerunning_the_migration_by_hand_keeps_existing_values(self, conn):
        """``INSERT OR IGNORE``: the seeds never switch a control back to its default."""
        halt = AWARE + timedelta(hours=23, minutes=30)
        set_kill_switch(conn, True, now=AWARE)
        set_halt_until(conn, halt, now=AWARE + timedelta(minutes=1))
        before = _control_rows(conn)
        assert before == [
            ("halt_until", "2026-07-31T13:30:00+00:00", "2026-07-30T14:01:00+00:00"),
            ("kill_switch", "on", "2026-07-30T14:00:00+00:00"),
        ]
        script = (MIGRATIONS_DIR / "0003_controls.sql").read_text(encoding="utf-8")
        for _ in range(2):
            for statement in store_db._split_statements(script):
                conn.execute(statement)
        assert _control_rows(conn) == before  # value and updated_at, both rows
        assert _triggers(conn, "controls") == [("controls_keep_rows", "controls")]
        controls = get_controls(conn)
        assert controls.kill_switch is True and controls.halt_until == halt and controls.problems == ()

    def test_get_controls_defaults(self, conn):
        controls = get_controls(conn)
        assert isinstance(controls, Controls)
        assert controls.kill_switch is False and controls.halt_until is None and controls.halt_unknown is False
        assert controls.problems == ()
        for seeded in (controls.kill_switch_updated_at, controls.halt_until_updated_at):  # when 0003 was applied
            assert seeded.tzinfo is timezone.utc
            assert abs(datetime.now(timezone.utc) - seeded) < timedelta(minutes=1)
        assert get_controls(conn) == controls and not conn.in_transaction  # a read changes nothing
        with pytest.raises(ValidationError):
            controls.kill_switch = True  # frozen: the switch is set through the repository
        reader = connect(":memory:")  # an in-memory store reads the same
        migrate(reader)
        assert get_controls(reader).model_dump(exclude={"kill_switch_updated_at", "halt_until_updated_at"}) == {
            "kill_switch": False, "halt_until": None, "halt_unknown": False, "problems": (),
        }
        reader.close()

    def test_set_kill_switch_round_trip(self, conn, db_path, monkeypatch):
        seeded = get_controls(conn)
        on = set_kill_switch(conn, True, now=AWARE)
        assert on.kill_switch is True and on.kill_switch_updated_at == AWARE and on.problems == ()
        assert on == get_controls(conn) and not conn.in_transaction
        assert _control_rows(conn)[1] == ("kill_switch", "on", "2026-07-30T14:00:00+00:00")
        other = connect(db_path)  # committed: another connection — the dashboard, a restart — sees it
        assert get_controls(other) == on
        other.close()
        later = AWARE + timedelta(minutes=5)
        again = set_kill_switch(conn, True, now=later)  # no change of state is still a write: updated_at moves
        assert again.kill_switch is True and again.kill_switch_updated_at == later
        off = set_kill_switch(conn, False, now=NAIVE + timedelta(minutes=10))  # a naive ``now`` is UTC
        assert off.kill_switch is False and off.kill_switch_updated_at == AWARE + timedelta(minutes=10)
        assert _control_rows(conn)[1] == ("kill_switch", "off", "2026-07-30T14:10:00+00:00")
        eastern = timezone(timedelta(hours=-4))
        converted = set_kill_switch(conn, True, now=datetime(2026, 7, 30, 10, 15, tzinfo=eastern))
        assert converted.kill_switch_updated_at == AWARE + timedelta(minutes=15)
        assert _control_rows(conn)[1] == ("kill_switch", "on", "2026-07-30T14:15:00+00:00")
        monkeypatch.setattr(repo, "utcnow", lambda: datetime(2026, 8, 3, 9, 30, tzinfo=timezone.utc))
        assert set_kill_switch(conn, False).kill_switch_updated_at == datetime(2026, 8, 3, 9, 30, tzinfo=timezone.utc)
        # the halt row was never touched, and no event was logged unasked
        final = get_controls(conn)
        assert (final.halt_until, final.halt_unknown) == (None, False)
        assert final.halt_until_updated_at == seeded.halt_until_updated_at
        assert _count(conn, "controls") == 2 and _count(conn, "events") == 0

    def test_set_kill_switch_takes_a_bool_only(self, conn):
        """The truthiness of "off" (or of 0) is not a decision about the kill switch."""
        before = _control_rows(conn)
        for bad in ("off", "on", "", 1, 0, None):
            with pytest.raises(StoreError, match="set kill switch for kill_switch") as info:
                set_kill_switch(conn, bad, now=AWARE)
            assert isinstance(info.value.cause, TypeError) and "must be a bool" in str(info.value)
        with pytest.raises(StoreError, match="set kill switch for kill_switch") as info:
            set_kill_switch(conn, True, now="2026-07-30T14:00:00+00:00")
        assert isinstance(info.value.cause, TypeError) and "now must be a datetime" in str(info.value)
        assert _control_rows(conn) == before and not conn.in_transaction

    def test_set_halt_until_round_trip(self, conn, db_path, monkeypatch):
        seeded = get_controls(conn)
        until = datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)  # the next market open
        halted = set_halt_until(conn, until, now=AWARE)
        assert halted.halt_until == until and halted.halt_until.tzinfo is timezone.utc
        assert halted.halt_unknown is False and halted.halt_until_updated_at == AWARE and halted.problems == ()
        assert halted == get_controls(conn) and not conn.in_transaction
        assert _control_rows(conn)[0] == ("halt_until", "2026-07-31T13:30:00+00:00", "2026-07-30T14:00:00+00:00")
        other = connect(db_path)  # the halt survives a restart: it is in the file, not in a process
        assert get_controls(other).halt_until == until
        other.close()
        naive = set_halt_until(conn, datetime(2026, 7, 31, 14, 0), now=NAIVE + timedelta(minutes=1))  # naive is UTC
        assert naive.halt_until == datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
        assert naive.halt_until_updated_at == AWARE + timedelta(minutes=1)
        assert _control_rows(conn)[0] == ("halt_until", "2026-07-31T14:00:00+00:00", "2026-07-30T14:01:00+00:00")
        eastern = timezone(timedelta(hours=-4))
        converted = set_halt_until(conn, datetime(2026, 7, 31, 9, 30, tzinfo=eastern), now=AWARE)
        assert converted.halt_until == until and converted.halt_until.utcoffset() == timedelta(0)
        assert _control_rows(conn)[0][1] == "2026-07-31T13:30:00+00:00"  # stored as UTC text, not -04:00
        precise = set_halt_until(conn, until + timedelta(microseconds=123456), now=AWARE)
        assert precise.halt_until == until + timedelta(microseconds=123456)
        assert _control_rows(conn)[0][1] == "2026-07-31T13:30:00.123456+00:00"
        earlier = set_halt_until(conn, AWARE - timedelta(days=1), now=AWARE)  # a plain upsert: no "only extend" here
        assert earlier.halt_until == AWARE - timedelta(days=1)
        cleared = set_halt_until(conn, None, now=AWARE + timedelta(minutes=2))
        assert cleared.halt_until is None and cleared.halt_unknown is False and cleared.problems == ()
        assert cleared.halt_until_updated_at == AWARE + timedelta(minutes=2)
        assert _control_rows(conn)[0] == ("halt_until", "", "2026-07-30T14:02:00+00:00")
        monkeypatch.setattr(repo, "utcnow", lambda: datetime(2026, 8, 3, 9, 30, tzinfo=timezone.utc))
        assert set_halt_until(conn, until).halt_until_updated_at == datetime(2026, 8, 3, 9, 30, tzinfo=timezone.utc)
        # the kill switch row was never touched, and no event was logged unasked
        final = get_controls(conn)
        assert final.kill_switch is False and final.kill_switch_updated_at == seeded.kill_switch_updated_at
        assert _count(conn, "controls") == 2 and _count(conn, "events") == 0
        before = _control_rows(conn)
        for bad in ("2026-07-31T13:30:00+00:00", date(2026, 7, 31), 0, False):
            with pytest.raises(StoreError, match="set halt until for halt_until") as info:
                set_halt_until(conn, bad, now=AWARE)
            assert isinstance(info.value.cause, TypeError) and "until must be a datetime" in str(info.value)
        assert _control_rows(conn) == before

    def test_only_extend_writes_a_halt_only_when_it_is_later_than_the_stored_one(self, conn, db_path):
        """How the policy engine writes: a halt is extended, never shortened or re-announced — and that is
        decided here, against the stored row, not against what the caller remembers of it."""
        until = datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)  # the next market open
        eastern = timezone(timedelta(hours=-4))
        later_now = AWARE + timedelta(minutes=5)

        def trip(number):
            return Event(id=f"evt-trip-{number}", occurred_at=AWARE, level=EventLevel.CRITICAL,
                         kind="risk_limit_tripped", message="halted")

        seeded = get_controls(conn)
        # no halt on record: written, with its event
        first = set_halt_until(conn, until, now=AWARE, event=trip(1), only_extend=True)
        assert first.halt_until == until and first.halt_until_updated_at == AWARE and first.problems == ()
        assert first == get_controls(conn) and not conn.in_transaction
        written = ("halt_until", "2026-07-31T13:30:00+00:00", "2026-07-30T14:00:00+00:00")
        assert _control_rows(conn)[0] == written
        assert [event.id for event in get_recent_events(conn)] == ["evt-trip-1"]
        # a halt at least that long is already stored: not the row, not the event — and the controls come back as they stand
        not_later = (
            until,  # equal
            until - timedelta(microseconds=1), until - timedelta(days=1), AWARE - timedelta(days=365),  # earlier
            datetime(2026, 7, 31, 9, 30, tzinfo=eastern),  # the same instant, in another offset
            datetime(2026, 7, 31, 13, 30),  # ... and naive, which is UTC
            datetime(2026, 7, 31, 9, 29, 59, tzinfo=eastern),  # a second earlier, however it is written
        )
        for number, instant in enumerate(not_later, start=2):
            kept = set_halt_until(conn, instant, now=later_now, event=trip(number), only_extend=True)
            assert kept == first, instant
            assert _control_rows(conn)[0] == written, instant  # value AND updated_at: the row was not rewritten
            assert _count(conn, "events") == 1, instant
        other = connect(db_path)  # nothing uncommitted is left behind either
        assert get_controls(other) == first
        other.close()
        # one microsecond later is later: replaced, and announced
        longer = set_halt_until(conn, until + timedelta(microseconds=1), now=later_now, event=trip(9), only_extend=True)
        assert longer.halt_until == until + timedelta(microseconds=1) and longer.halt_until_updated_at == later_now
        assert _control_rows(conn)[0] == ("halt_until", "2026-07-31T13:30:00.000001+00:00", "2026-07-30T14:05:00+00:00")
        assert [event.id for event in get_recent_events(conn)] == ["evt-trip-9", "evt-trip-1"]
        # a later instant given in another offset extends too, and is stored as UTC
        far = set_halt_until(conn, datetime(2026, 8, 3, 9, 30, tzinfo=eastern), now=later_now, only_extend=True)
        assert far.halt_until == datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
        assert _control_rows(conn)[0][1] == "2026-08-03T13:30:00+00:00" and _count(conn, "events") == 2
        # the kill switch row was never touched
        final = get_controls(conn)
        assert final.kill_switch is False and final.kill_switch_updated_at == seeded.kill_switch_updated_at
        assert _count(conn, "controls") == 2

    def test_only_extend_replaces_an_expired_halt(self, conn):
        """An expired halt is simply an earlier one: "expired" needs a clock, and the comparison needs none."""
        expired = AWARE - timedelta(days=3)
        set_halt_until(conn, expired, now=AWARE - timedelta(days=4))
        replaced = set_halt_until(conn, AWARE + timedelta(hours=1), now=AWARE, only_extend=True)
        assert replaced.halt_until == AWARE + timedelta(hours=1) and replaced.halt_until_updated_at == AWARE
        # ... while a halt that is itself long past still does not shorten a later one
        assert set_halt_until(conn, expired, now=AWARE, only_extend=True).halt_until == AWARE + timedelta(hours=1)

    def test_only_extend_replaces_a_halt_nobody_can_read(self, conn):
        """Unreadable is not "longer": a halt without an end is replaced by one that has one."""
        event = Event(id="evt-trip", level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="halted")
        conn.execute("UPDATE controls SET value = ? WHERE key = ?", ("2026-13-45T99:99:99", "halt_until"))  # the CHECK lets it in
        unreadable = get_controls(conn)
        assert (unreadable.halt_until, unreadable.halt_unknown) == (None, True) and unreadable.problems
        fixed = set_halt_until(conn, AWARE, now=AWARE, event=event, only_extend=True)
        assert (fixed.halt_until, fixed.halt_unknown, fixed.problems) == (AWARE, False, ())
        assert _control_rows(conn)[0] == ("halt_until", "2026-07-30T14:00:00+00:00", "2026-07-30T14:00:00+00:00")
        assert [logged.id for logged in get_recent_events(conn)] == ["evt-trip"]
        # the row gone altogether (the trigger dropped by hand): restored, whatever the instant
        conn.execute("DROP TRIGGER controls_keep_rows")
        conn.execute("DELETE FROM controls WHERE key = ?", ("halt_until",))
        assert get_controls(conn).halt_unknown is True
        restored = set_halt_until(conn, AWARE - timedelta(days=1), now=AWARE, only_extend=True)
        assert (restored.halt_until, restored.halt_unknown, restored.problems) == (AWARE - timedelta(days=1), False, ())
        # '' is "no halt", and readable: any instant extends it
        set_halt_until(conn, None, now=AWARE)
        assert set_halt_until(conn, AWARE - timedelta(days=9), now=AWARE, only_extend=True).halt_until == AWARE - timedelta(days=9)

    def test_only_extend_needs_an_instant_and_a_real_bool(self, conn):
        halted = set_halt_until(conn, AWARE, now=AWARE)
        before = _control_rows(conn)
        with pytest.raises(StoreError, match="set halt until for halt_until") as info:
            set_halt_until(conn, None, now=AWARE, only_extend=True)  # "extend to no halt" would be a clear
        assert isinstance(info.value.cause, ValueError) and "only_extend needs an instant" in str(info.value)
        for bad in ("yes", "", 1, 0, None):
            with pytest.raises(StoreError, match="set halt until for halt_until") as info:
                set_halt_until(conn, AWARE + timedelta(days=1), now=AWARE, only_extend=bad)
            assert isinstance(info.value.cause, TypeError) and "only_extend must be a bool" in str(info.value)
        with pytest.raises(StoreError, match="set halt until for halt_until") as info:
            set_halt_until(conn, "2026-07-31T13:30:00+00:00", now=AWARE, only_extend=True)
        assert isinstance(info.value.cause, TypeError) and "until must be a datetime" in str(info.value)
        assert _control_rows(conn) == before and get_controls(conn) == halted and not conn.in_transaction
        with pytest.raises(TypeError):
            set_halt_until(conn, AWARE, AWARE, None, True)  # keyword-only: never a stray positional flag

    def test_the_plain_write_still_shortens_and_clears_for_the_operator(self, conn):
        """``only_extend`` is opt-in. Left out — or False — the write is the operator's: any instant, or none."""
        far = AWARE + timedelta(days=7)
        assert set_halt_until(conn, far, now=AWARE, only_extend=True).halt_until == far
        shorter = set_halt_until(conn, AWARE + timedelta(hours=1), now=AWARE)
        assert shorter.halt_until == AWARE + timedelta(hours=1)
        assert set_halt_until(conn, far, now=AWARE).halt_until == far
        assert set_halt_until(conn, AWARE - timedelta(days=1), now=AWARE, only_extend=False).halt_until == AWARE - timedelta(days=1)
        assert set_halt_until(conn, far, now=AWARE).halt_until == far
        cleared = set_halt_until(conn, None, now=AWARE + timedelta(minutes=1), only_extend=False)
        assert (cleared.halt_until, cleared.halt_unknown) == (None, False)
        assert _control_rows(conn)[0] == ("halt_until", "", "2026-07-30T14:01:00+00:00")

    def test_only_extend_compares_inside_the_write_transaction(self, conn, db_path):
        """The stored halt is read AFTER ``BEGIN IMMEDIATE`` took the write lock: no writer can slip in between
        the comparison and the write, and a caller whose own reading is stale cannot shorten what is stored."""
        until = AWARE + timedelta(hours=23, minutes=30)
        event = Event(id="evt-trip", level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="halted")
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            set_halt_until(conn, until, now=AWARE, event=event, only_extend=True)
        finally:
            conn.set_trace_callback(None)
        verbs = [" ".join(statement.split()[:2]) for statement in statements]
        assert verbs == [  # the first SELECT is the comparison, the second the read-back
            "BEGIN IMMEDIATE", "SELECT key,", "INSERT INTO", "INSERT INTO", "SELECT key,", "COMMIT",
        ]
        assert all("FROM controls" in statement for statement in statements if statement.startswith("SELECT"))
        # refused: the transaction holds the comparison and nothing else
        again = event.model_copy(update={"id": "evt-again"})
        assert _traced(conn, lambda: set_halt_until(conn, until, now=AWARE, event=again, only_extend=True)) == [
            "BEGIN IMMEDIATE", "COMMIT",
        ]
        assert _traced(conn, lambda: set_halt_until(conn, until + timedelta(seconds=1), now=AWARE, event=again, only_extend=True)) == [
            "BEGIN IMMEDIATE", "INSERT INTO controls", "INSERT INTO events", "COMMIT",
        ]
        assert _count(conn, "events") == 2
        # a second writer that read "no halt" before the first one wrote: its earlier halt does not land
        stale = connect(db_path)
        try:
            remembered = Controls()  # what it saw, long ago
            assert remembered.halt_until is None
            kept = set_halt_until(stale, AWARE + timedelta(hours=1), now=AWARE, only_extend=True,
                                  event=Event(id="evt-stale", level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="m"))
            assert kept.halt_until == until + timedelta(seconds=1) == get_controls(conn).halt_until
            assert _count(conn, "events") == 2
            # and while another writer holds the lock, the comparison waits for it like any write
            stale.execute("PRAGMA busy_timeout=50")
            with transaction(conn):
                _insert_event(conn)
                with pytest.raises(StoreError, match="set halt until for halt_until") as info:
                    set_halt_until(stale, until + timedelta(days=1), now=AWARE, only_extend=True)
                assert isinstance(info.value.cause, sqlite3.OperationalError) and "database is locked" in str(info.value)
            assert get_controls(stale).halt_until == until + timedelta(seconds=1)
        finally:
            stale.close()

    def test_only_extend_with_a_failing_event_rolls_the_halt_back(self, conn):
        log_event(conn, Event(id="evt-taken", occurred_at=AWARE, level=EventLevel.INFO, kind="k", message="m"))
        clash = Event(id="evt-taken", level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="again")
        before = _control_rows(conn)

        def write():
            with pytest.raises(StoreError, match="set halt until for halt_until") as info:
                set_halt_until(conn, AWARE, now=AWARE, event=clash, only_extend=True)
            assert "UNIQUE constraint failed: events.id" in str(info.value)

        assert _traced(conn, write) == ["BEGIN IMMEDIATE", "INSERT INTO controls", "INSERT INTO events", "ROLLBACK"]
        assert _control_rows(conn) == before and get_controls(conn).halt_until is None and _count(conn, "events") == 1
        # a write the comparison refuses never reaches the event: a clashing id cannot fail it
        set_halt_until(conn, AWARE, now=AWARE)
        assert set_halt_until(conn, AWARE, now=AWARE, event=clash, only_extend=True).halt_until == AWARE

    def test_event_lands_in_the_same_transaction_as_the_write(self, conn, trace_data):
        """A control change — or a verdict — and its audit event are one BEGIN … COMMIT."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        until = AWARE + timedelta(hours=23, minutes=30)
        kill = Event(
            id="evt-kill", occurred_at=AWARE, level=EventLevel.WARNING, kind="kill_switch", message="kill switch on",
            payload={"state": "on", "previous": "off", "source": "cli"},
        )
        halt = Event(
            id="evt-halt", occurred_at=AWARE, level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="halted",
            payload={"rule": "daily_loss_limit", "proposal_id": proposal.id, "halt_until": until.isoformat()},
        )
        verdict = Event(
            id="evt-verdict", occurred_at=AWARE, level=EventLevel.WARNING, kind="policy_reject", message="rejected",
            payload={"proposal_id": proposal.id, "decision_id": "dec-0001", "verdict": "REJECT"},
        )
        decision = PolicyDecision.model_validate(trace_data["decision"])
        results = {}
        assert _traced(conn, lambda: results.update(kill=set_kill_switch(conn, True, now=AWARE, event=kill))) == [
            "BEGIN IMMEDIATE", "INSERT INTO controls", "INSERT INTO events", "COMMIT",
        ]
        assert _traced(conn, lambda: results.update(halt=set_halt_until(conn, until, now=AWARE, event=halt))) == [
            "BEGIN IMMEDIATE", "INSERT INTO controls", "INSERT INTO events", "COMMIT",
        ]
        assert _traced(conn, lambda: results.update(decision=record_decision(conn, decision, event=verdict))) == [
            "BEGIN IMMEDIATE", "INSERT INTO policy_decisions", "INSERT INTO events", "COMMIT",
        ]
        assert results["kill"].kill_switch is True and results["kill"].halt_until is None
        assert results["halt"].kill_switch is True and results["halt"].halt_until == until
        assert results["decision"] == decision == get_proposal_trace(conn, proposal.id).decision
        assert get_recent_events(conn) == [verdict, halt, kill]  # newest first; each exactly as given
        assert get_recent_events(conn, level=EventLevel.CRITICAL)[0].payload["halt_until"] == "2026-07-31T13:30:00+00:00"
        assert get_controls(conn) == results["halt"] and not conn.in_transaction
        # without an event, the same writers log nothing — and log_event itself is that one INSERT
        assert _traced(conn, lambda: set_kill_switch(conn, False, now=AWARE)) == [
            "BEGIN IMMEDIATE", "INSERT INTO controls", "COMMIT",
        ]
        assert _traced(conn, lambda: log_event(conn, Event(level=EventLevel.INFO, kind="k", message="m"))) == [
            "BEGIN IMMEDIATE", "INSERT INTO events", "COMMIT",
        ]
        assert _count(conn, "events") == 4 and _count(conn, "policy_decisions") == 1

    def test_failing_event_rolls_the_write_back(self, conn, trace_data):
        """All or nothing: a control change or a decision whose event cannot be written does not happen."""
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        taken = log_event(conn, Event(id="evt-taken", occurred_at=AWARE, level=EventLevel.INFO, kind="k", message="m"))
        clash = Event(id="evt-taken", level=EventLevel.CRITICAL, kind="risk_limit_tripped", message="again")
        not_json = Event(id="evt-nan", level=EventLevel.CRITICAL, kind="k", message="m", payload={"pnl": float("nan")})
        decision = PolicyDecision.model_validate(trace_data["decision"])
        before = _control_rows(conn)
        for event, cause, needle in (
            (clash, sqlite3.IntegrityError, "UNIQUE constraint failed: events.id"),
            (not_json, ValueError, "Out of range float values are not JSON compliant"),
        ):
            writes = (
                ("set kill switch for kill_switch", lambda: set_kill_switch(conn, True, now=AWARE, event=event)),
                ("set halt until for halt_until", lambda: set_halt_until(conn, AWARE, now=AWARE, event=event)),
                ("record decision for dec-0001", lambda: record_decision(conn, decision, event=event)),
            )
            for what, write in writes:
                with pytest.raises(StoreError, match=what) as info:
                    write()
                assert isinstance(info.value.cause, cause) and needle in str(info.value), what
                assert not conn.in_transaction
                assert _control_rows(conn) == before, what  # the switch did not flip, the halt was not set
                assert _count(conn, "policy_decisions") == 0 and _count(conn, "events") == 1, what
        controls = get_controls(conn)
        assert (controls.kill_switch, controls.halt_until, controls.halt_unknown) == (False, None, False)
        assert get_recent_events(conn) == [taken] and get_proposal_trace(conn, proposal.id).decision is None
        # the other way round: a write that fails takes its event down with it
        orphan = PolicyDecision(id="dec-orphan", proposal_id="no-such-proposal", verdict=Verdict.REJECT, rules_evaluated=[])
        with pytest.raises(StoreError, match="record decision for dec-orphan") as info:
            record_decision(conn, orphan, event=Event(id="evt-orphan", level=EventLevel.WARNING, kind="k", message="m"))
        assert "FOREIGN KEY" in str(info.value) and _count(conn, "events") == 1
        # and the same writes go through once the event can be written
        fresh = clash.model_copy(update={"id": "evt-fresh"})
        assert set_kill_switch(conn, True, now=AWARE, event=fresh).kill_switch is True
        assert record_decision(conn, decision, event=fresh.model_copy(update={"id": "evt-fresh-2"})) == decision
        assert _count(conn, "events") == 3 and _count(conn, "policy_decisions") == 1

    def test_rolled_back_write_issues_rollback(self, conn):
        """The failed write above is a ROLLBACK of the control row, not a commit followed by an error."""
        log_event(conn, Event(id="evt-taken", level=EventLevel.INFO, kind="k", message="m"))
        clash = Event(id="evt-taken", level=EventLevel.INFO, kind="k", message="again")

        def write():
            with pytest.raises(StoreError, match="set kill switch"):
                set_kill_switch(conn, True, now=AWARE, event=clash)

        assert _traced(conn, write) == ["BEGIN IMMEDIATE", "INSERT INTO controls", "INSERT INTO events", "ROLLBACK"]
        assert get_controls(conn).kill_switch is False

    def test_missing_row_fails_closed(self, conn):
        """With the trigger dropped a row can be deleted: the read then assumes the worst, and says why."""
        conn.execute("DROP TRIGGER controls_keep_rows")
        conn.execute("DELETE FROM controls WHERE key = ?", ("kill_switch",))
        controls = get_controls(conn)  # never raises for a bad row
        assert controls.kill_switch is True and controls.kill_switch_updated_at is None
        assert (controls.halt_until, controls.halt_unknown) == (None, False)  # the halt row is intact
        assert controls.problems == (
            "kill_switch is missing from the controls table: reading the kill switch as ON",
        )
        # the writer restores the row (its INSERT half), and the read is clean again
        restored = set_kill_switch(conn, False, now=AWARE)
        assert restored.kill_switch is False and restored.kill_switch_updated_at == AWARE and restored.problems == ()

        conn.execute("DELETE FROM controls WHERE key = ?", ("halt_until",))
        controls = get_controls(conn)
        assert controls.kill_switch is False
        assert (controls.halt_until, controls.halt_unknown, controls.halt_until_updated_at) == (None, True, None)
        assert controls.problems == (
            "halt_until is missing from the controls table: cannot prove trading is not halted",
        )
        assert set_halt_until(conn, None, now=AWARE).halt_unknown is False

        conn.execute("DELETE FROM controls")  # both gone
        controls = get_controls(conn)
        assert controls.kill_switch is True and controls.halt_unknown is True and controls.halt_until is None
        assert len(controls.problems) == 2
        assert "kill_switch is missing" in controls.problems[0] and "halt_until is missing" in controls.problems[1]

    def test_unparseable_halt_that_passes_the_check_fails_closed(self, conn):
        """The GLOB pins the leading shape only: the read still parses the whole value."""
        for value in (
            "2026-13-45T99:99:99",  # the right shape, not a date
            "2026-07-31T13:30:00 and then some",
            "2026-07-31T13:30:00+99:99",
            "0001-01-01T00:00:00+05:00",  # parses, but has no UTC equivalent a datetime can hold
            "9999-12-31T23:59:59-05:00",
        ):
            conn.execute("UPDATE controls SET value = ? WHERE key = ?", (value, "halt_until"))  # the CHECK lets it in
            controls = get_controls(conn)
            assert (controls.halt_until, controls.halt_unknown) == (None, True), value
            assert controls.kill_switch is False
            assert controls.problems == (
                f"halt_until value {value!r} is neither empty nor an ISO-8601 timestamp:"
                " cannot prove trading is not halted",
            )
        for value, expected in (  # what does parse: naive is UTC, any other offset is converted
            ("2026-07-31T13:30:00", datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)),
            ("2026-07-31T09:30:00-04:00", datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)),
            ("2026-07-31T13:30:00Z", datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)),
            ("2026-07-31T13:30:00.250+00:00", datetime(2026, 7, 31, 13, 30, 0, 250000, tzinfo=timezone.utc)),
        ):
            conn.execute("UPDATE controls SET value = ? WHERE key = ?", (value, "halt_until"))
            controls = get_controls(conn)
            assert controls.halt_until == expected and controls.halt_until.tzinfo is timezone.utc, value
            assert controls.halt_unknown is False and controls.problems == ()

    def test_unreadable_updated_at_is_none_and_not_a_problem(self, conn):
        conn.execute("UPDATE controls SET updated_at = ?", ("last tuesday",))
        controls = get_controls(conn)
        assert controls.kill_switch_updated_at is None and controls.halt_until_updated_at is None
        assert (controls.kill_switch, controls.halt_until, controls.halt_unknown, controls.problems) == (
            False, None, False, (),
        )
        conn.execute("UPDATE controls SET updated_at = ? WHERE key = ?", ("2026-07-30T14:00:00", "kill_switch"))
        assert get_controls(conn).kill_switch_updated_at == AWARE  # naive is UTC

    @pytest.mark.parametrize(
        "rows, kill_switch, halt_unknown, needles",
        [
            ([("kill_switch", "maybe"), ("halt_until", "")], True, False, ["kill_switch value 'maybe' is neither"]),
            ([("kill_switch", "ON"), ("halt_until", "")], True, False, ["kill_switch value 'ON' is neither"]),
            ([("kill_switch", ""), ("halt_until", "")], True, False, ["kill_switch value '' is neither"]),
            ([("kill_switch", None), ("halt_until", "")], True, False, ["kill_switch value None is neither"]),
            ([("kill_switch", 0), ("halt_until", "")], True, False, ["kill_switch value 0 is neither"]),
            ([("kill_switch", b"off"), ("halt_until", "")], True, False, ["kill_switch value b'off' is neither"]),
            ([("kill_switch", "off"), ("halt_until", "tomorrow")], False, True, ["halt_until value 'tomorrow' is neither"]),
            ([("kill_switch", "off"), ("halt_until", None)], False, True, ["halt_until value None is neither"]),
            ([("kill_switch", "off"), ("halt_until", 1785504600)], False, True, ["halt_until value 1785504600 is"]),
            ([("kill_switch", "off"), ("halt_until", " ")], False, True, ["halt_until value ' ' is neither"]),
            ([("kill_switch", "off"), ("halt_until", "off")], False, True, ["halt_until value 'off' is neither"]),
            ([("kill_switch", "maybe"), ("halt_until", "soon")], True, True, ["kill_switch value", "halt_until value"]),
            # one row per control, or the read cannot tell which to believe
            ([("kill_switch", "off"), ("kill_switch", "off"), ("halt_until", "")], True, False,
             ["kill_switch has 2 rows in the controls table"]),
            ([("kill_switch", "off"), ("halt_until", ""), ("halt_until", "")], False, True,
             ["halt_until has 2 rows in the controls table"]),
            # a key this code does not know may be a stop it cannot read
            ([("kill_switch", "off"), ("halt_until", ""), ("killswitch", "on")], True, False,
             ["unrecognised control key 'killswitch'"]),
            ([("kill_switch", "off"), ("halt_until", ""), (None, "on")], True, False, ["unrecognised control key None"]),
            ([("KILL_SWITCH", "off"), ("halt_until", "")], True, False,
             ["kill_switch is missing", "unrecognised control key 'KILL_SWITCH'"]),
            ([], True, True, ["kill_switch is missing", "halt_until is missing"]),
        ],
    )
    def test_garbage_fails_closed(self, conn, rows, kill_switch, halt_unknown, needles):
        """The table rebuilt by hand without its constraints can hold anything; the read never raises
        and never reads garbage as "trading is allowed"."""
        conn.execute("DROP TABLE controls")  # takes the trigger with it
        conn.execute("CREATE TABLE controls (key, value, updated_at)")  # no CHECK, no NOT NULL, no primary key
        for key, value in rows:
            conn.execute("INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)", (key, value, AWARE.isoformat()))
        controls = get_controls(conn)
        assert controls.kill_switch is kill_switch and controls.halt_unknown is halt_unknown
        assert controls.halt_until is None
        assert len(controls.problems) == len(needles)
        for problem, needle in zip(controls.problems, needles):
            assert needle in problem
            # each line ends with what the read did about it
            if problem.startswith("halt_until"):
                assert problem.endswith(": cannot prove trading is not halted")
            else:
                assert problem.endswith(": reading the kill switch as ON")

    def test_problem_lines_quote_garbage_safely(self, conn):
        """A problem line is one printable line however hostile or long the stored value."""
        conn.execute("DROP TABLE controls")
        conn.execute("CREATE TABLE controls (key, value, updated_at)")
        hostile = "on\x1b[2J\r\nkill switch off" + "x" * 5000
        for key in ("kill_switch", "halt_until", hostile):
            conn.execute("INSERT INTO controls (key, value, updated_at) VALUES (?, ?, ?)", (key, hostile, hostile))
        controls = get_controls(conn)
        assert controls.kill_switch is True and controls.halt_unknown is True and len(controls.problems) == 3
        assert controls.kill_switch_updated_at is None and controls.halt_until_updated_at is None
        for problem in controls.problems:
            assert problem.isprintable() and len(problem) < 200
            assert "'on\\x1b[2J\\r\\nkill switch off" in problem and "..." in problem

    def test_controls_table_missing_is_store_error(self, conn, db_path):
        """Nothing to read is not a bad row: a store without the table is refused, by reads and writes alike."""
        conn.execute("DROP TABLE controls")
        with pytest.raises(StoreError, match="get controls") as info:
            get_controls(conn)
        assert isinstance(info.value.cause, sqlite3.OperationalError) and "no such table: controls" in str(info.value)
        assert (info.value.what, info.value.key) == ("get controls", None)
        for what, write in (
            ("set kill switch for kill_switch", lambda: set_kill_switch(conn, True, now=AWARE)),
            ("set halt until for halt_until", lambda: set_halt_until(conn, AWARE, now=AWARE)),
        ):
            with pytest.raises(StoreError, match=what) as info:
                write()
            assert isinstance(info.value.cause, sqlite3.OperationalError) and "no such table" in str(info.value)
            assert not conn.in_transaction
        report = status(conn, db_path)
        assert report.missing_tables == ("controls",) and report.row_counts["controls"] == 0
        assert migrate(conn) == LATEST and "controls" not in _table_names(conn)  # a recorded migration is never re-run

    def test_lock_timeout_names_the_control(self, db_path):
        holder = open_store(db_path)
        writer = connect(db_path)
        writer.execute("PRAGMA busy_timeout=50")
        try:
            with transaction(holder):
                _insert_event(holder)
                with pytest.raises(StoreError, match="set kill switch for kill_switch") as info:
                    set_kill_switch(writer, True, now=AWARE)
                assert isinstance(info.value.cause, sqlite3.OperationalError) and "database is locked" in str(info.value)
                assert (info.value.what, info.value.key) == ("set kill switch", "kill_switch")
                with pytest.raises(StoreError, match="set halt until for halt_until"):
                    set_halt_until(writer, AWARE, now=AWARE)
                assert get_controls(writer).kill_switch is False  # a reader is never blocked (WAL)
            assert set_kill_switch(writer, True, now=AWARE).kill_switch is True  # and the lock released, it lands
        finally:
            holder.close()
            writer.close()


_SQL_VERBS = ("SELECT", "INSERT", "UPDATE", "DELETE")

# The only SQL the store executes that is not a string literal at the call site: module
# constants built from literals at import (pinned below) and, in ``_apply_migration``, the
# statements read from a migration file.
_SQL_NAMES = {store_db: {"_SCHEMA_VERSION_DDL", "_COUNT_QUERIES", "statement"}, repo: {"_OPEN_ORDERS_SQL"}}


def _literal_sql(node, names):
    """Whether an executed SQL expression is string literals only (``+`` allowed) or a pinned name."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_sql(node.left, names) and _literal_sql(node.right, names)
    if isinstance(node, ast.Subscript):
        node = node.value
    return isinstance(node, ast.Name) and node.id in names


def _mentions_sql(node):
    """Whether any string literal under ``node`` contains a SQL verb."""
    return any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and any(verb in n.value for verb in _SQL_VERBS)
        for n in ast.walk(node)
    )


def test_store_sql_is_parameterized_only():
    """Every execute() gets literal SQL (or a pinned name), never a value spliced in — however the
    call is split across lines; no f-string, ``.format`` or ``%`` mentions SQL at all; and the one
    IN list is ``?`` marks only."""
    for module, names in _SQL_NAMES.items():
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"), module.__file__)
        for node in ast.walk(tree):
            where = module.__name__ + ":" + str(getattr(node, "lineno", "?"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("execute", "executemany", "executescript"):
                    assert node.args and _literal_sql(node.args[0], names), where + " executes SQL built from a value"
                if node.func.attr == "format":
                    assert not _mentions_sql(node.func.value), where + " formats SQL"
            elif isinstance(node, ast.JoinedStr):
                assert not _mentions_sql(node), where + " builds SQL with an f-string"
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                assert not _mentions_sql(node.left), where + " formats SQL with %"
    assert re.fullmatch(
        r"SELECT \* FROM orders WHERE status IN \((\?, )*\?\) ORDER BY updated_at, rowid",
        repo._OPEN_ORDERS_SQL,
    )
    assert repo._OPEN_ORDERS_SQL.count("?") == len(OPEN_ORDER_STATUSES) == len(repo._OPEN_STATUS_VALUES)
    assert set(repo._OPEN_STATUS_VALUES) == {status.value for status in OPEN_ORDER_STATUSES}
    assert store_db._SCHEMA_VERSION_DDL.startswith("CREATE TABLE IF NOT EXISTS schema_version (")
    assert store_db._COUNT_QUERIES == {table: "SELECT COUNT(*) FROM " + table for table in TABLES}


class TestCli:
    """``main([...])`` called directly against tmp_path databases; output via capsys."""

    def test_db_init_creates_and_is_idempotent(self, db_path, capsys):
        assert cli_db.main(["init", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert db_path.exists()
        assert str(db_path) in out and "(created)" in out
        assert "journal mode: wal" in out
        assert "schema version 4 (applied: 0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution)" in out
        assert "up to date" not in out

        assert cli_db.main(["init", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "(created)" not in out and "wal" in out
        assert (
            "schema version 4 (applied: 0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution) [up to date]"
        ) in out
        conn = connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == LATEST
        assert schema_version(conn) == LATEST
        conn.close()

    def test_db_status_missing_file_exits_1_without_creating_it(self, tmp_path, capsys):
        missing = tmp_path / "nested" / "t.db"
        assert cli_db.main(["status", "--db", str(missing)]) == 1
        captured = capsys.readouterr()
        assert str(missing) in captured.out and "exists: no" in captured.out
        assert "aegis.cli.db init" in captured.out
        assert not missing.exists() and not missing.parent.exists()

    def test_db_status_after_init_lists_eleven_tables(self, db_path, conn, trace_data, capsys):
        _seed_trace(conn, trace_data)
        insert_proposal(conn, _bare_proposal("prop-0002"), _legs("prop-0002"))
        assert cli_db.main(["status", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert str(db_path) in out and "exists: yes" in out
        assert "journal mode: wal" in out
        assert "schema version 4 (applied: 0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution)" in out
        assert "pending migrations: none" in out
        counts = dict(re.findall(r"^  (\w+)\s+(\d+)$", out, re.MULTILINE))
        assert list(counts) == list(TABLES) and len(counts) == 11  # every table, in TABLES order, controls last
        assert {t: int(n) for t, n in counts.items()} == {
            "proposals": 2, "proposal_legs": 2, "reasoning": 3, "policy_decisions": 1, "approvals": 1,
            "orders": 1, "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 0, "controls": 2,
        }

    def test_db_status_on_unmigrated_file(self, db_path, capsys):
        connect(db_path).close()  # the file exists, but nothing has been applied
        assert cli_db.main(["status", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "schema version 0 (applied: none)" in out
        assert "pending migrations: 0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution" in out
        assert re.search(r"^  events\s+0$", out, re.MULTILINE)
        assert re.search(r"^  proposal_legs\s+0$", out, re.MULTILINE)
        assert re.search(r"^  controls\s+0$", out, re.MULTILINE)  # not created yet: no seed rows either

    def test_db_relative_path_resolves_against_cwd(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        assert cli_db.main(["init", "--db", "rel.db"]) == 0
        assert (tmp_path / "rel.db").exists()
        assert str(tmp_path / "rel.db") in capsys.readouterr().out

    def test_db_without_flag_uses_config_path(self, tmp_path, monkeypatch, capsys):
        configured = tmp_path / "from-config" / "aegis.db"
        monkeypatch.setattr(cli_db, "resolve_db_path", lambda: configured)
        monkeypatch.setattr(cli_trace, "resolve_db_path", lambda: configured)
        assert cli_db.main(["status"]) == 1
        assert not configured.exists()
        assert cli_db.main(["init"]) == 0
        assert configured.exists()
        assert cli_db.main(["status"]) == 0
        assert cli_trace.main(["nope"]) == 1
        captured = capsys.readouterr()
        assert str(configured) in captured.out and "nope" in captured.err

    def test_db_errors_are_one_clean_line(self, db_path, monkeypatch, capsys):
        conn = open_store(db_path)
        conn.execute(
            "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
            (99, "0099_from_the_future", AWARE.isoformat()),
        )
        conn.close()
        assert cli_db.main(["init", "--db", str(db_path)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("db init failed: store operation failed: migrate")
        assert "newer" in captured.err and captured.err.count("\n") == 1
        assert "Traceback" not in captured.err

        def boom(path):
            raise RuntimeError("boom")

        monkeypatch.setattr(cli_db, "connect", boom)
        assert cli_db.main(["init", "--db", str(db_path)]) == 1
        err = capsys.readouterr().err
        assert err == "db init failed (unexpected): RuntimeError: boom\n"

        def interrupted(path):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_db, "connect", interrupted)
        assert cli_db.main(["status", "--db", str(db_path)]) == 130

    def test_trace_prints_the_full_lineage(self, db_path, conn, trace_data, capsys):
        proposal, reasoning, decision, approval, order, fill = _seed_trace(conn, trace_data)
        assert cli_trace.main([proposal.id, "--db", str(db_path)]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        out = captured.out
        lines = out.splitlines()

        assert lines[0] == "proposal prop-0001"
        # each header line pinned whole: the raw model output below repeats most of these values
        assert "  created      2026-07-30 14:05:00 UTC" in lines
        assert "  cycle        cycle-0001" in lines
        assert "  symbol       SPY260821C00640000 (option)" in lines
        assert "  order        buy 2 limit @ 12.20" in lines
        assert "  confidence   0.62" in lines
        assert "  model        synthetic-model-v0 (prompt test-prompt-1)" in lines
        assert "  legs (0)" in lines  # the fixture proposal carries no leg rows
        # thesis, invalidation and the raw model output are shown in full, indented
        assert "    " + proposal.thesis in lines
        assert "    " + proposal.invalidation in lines
        assert "    " + proposal.raw_model_output in lines

        # every stage, in order, with its content in full
        assert "reasoning (3)" in lines
        for stage in reasoning:
            assert f"[{stage.stage.value}]" in out and "    " + stage.content in lines
        assert out.index("[scan]") < out.index("[thesis]") < out.index("[proposal]")
        assert "tokens in 1200 / out 180" in out

        assert "decisions (1)" in lines
        assert decision.verdict.value in out  # NEEDS_APPROVAL
        assert "failing rule: none" in out
        assert "rules evaluated (2)" in out
        rule_lines = [line for line in lines if "max_position_pct" in line or "max_open_positions" in line]
        assert len(rule_lines) == 2  # one rule per line
        assert rule_lines[0].strip() == "max_position_pct: limit=5.0 observed=1.8 passed=True"
        assert rule_lines[1].strip() == "max_open_positions: limit=5 observed=2 passed=True"
        assert "      " + decision.notes in lines

        assert "approvals (1)" in lines
        assert "requested 2026-07-30 14:05:31 UTC via cli" in out
        assert "response: approved by test-operator at 2026-07-30 14:07:10 UTC" in out

        assert "orders (1)" in lines
        assert order.client_order_id in out and "paper (paper-order-0001)" in out
        assert "aegis-prop-0001-1   filled" in out
        assert "submitted 2026-07-30 14:07:12 UTC   updated 2026-07-30 14:07:20 UTC" in out
        assert "fills (1)" in out
        assert "2026-07-30 14:07:19 UTC   2 @ 12.15   fees 1.30" in out

    def test_trace_of_a_bare_proposal(self, db_path, conn, trace_data, capsys):
        proposal = insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        assert cli_trace.main([proposal.id, "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        for title in ("legs (0)", "reasoning (0)", "decisions (0)", "approvals (0)", "orders (0)"):
            assert title in out
        assert out.count("(none)") == 5

    def test_trace_prints_legs(self, db_path, conn, trace_data, capsys):
        proposal = Proposal.model_validate(trace_data["proposal"])
        insert_proposal(conn, proposal, _legs(proposal.id)[::-1])  # written index 1 first
        assert cli_trace.main([proposal.id, "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "  legs (2)" in lines
        assert "    [0] buy 1 call 640.0 exp 2026-10-16  SPY261016C00640000" in lines
        assert "    [1] sell 1 call 650.0 exp 2026-10-16  SPY261016C00650000" in lines
        assert out.index("legs (2)") < out.index("[0] buy") < out.index("[1] sell") < out.index("thesis:")
        assert out.count("(none)") == 4  # reasoning, decisions, approvals, orders — not legs
        assert [cli_trace._strike(v) for v in (640.0, 642.5, 7, 0.0001, 1234567.5)] == [
            "640.0", "642.5", "7.0", "0.0001", "1234567.5",
        ]
        assert cli_trace.main([proposal.id, "--db", str(db_path), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert [(leg["leg_index"], leg["symbol"], leg["side"], leg["expiration"]) for leg in data["legs"]] == [
            (0, "SPY261016C00640000", "buy", "2026-10-16"), (1, "SPY261016C00650000", "sell", "2026-10-16"),
        ]
        assert ProposalTrace.model_validate(data) == get_proposal_trace(conn, proposal.id)

    def test_trace_prints_a_rejection(self, db_path, conn, trace_data, capsys):
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        record_decision(conn, PolicyDecision(
            id="dec-reject", proposal_id="prop-0001", verdict=Verdict.REJECT,
            rules_evaluated=[{"rule": "max_position_pct", "limit": 5.0, "observed": 7.2, "passed": False}],
            failing_rule="max_position_pct", notes="over the cap",
        ))
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "decisions (1)" in out and "REJECT" in out
        assert "failing rule: max_position_pct" in out
        assert "max_position_pct: limit=5.0 observed=7.2 passed=False" in out
        assert "      over the cap" in out.splitlines()

    def test_trace_pending_approval_and_unfilled_market_order(self, db_path, conn, trace_data, capsys):
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        record_approval(conn, Approval(id="appr-p", proposal_id="prop-0001", channel="cli"))
        upsert_order(conn, Order(
            proposal_id="prop-0001", client_order_id="coid-m", broker=Broker.PAPER,
            status=OrderStatus.PROPOSED, symbol="SPY", side=OrderSide.SELL, quantity=10,
        ))
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "response: pending" in out
        assert "coid-m   proposed   paper (no broker order id)" in out
        assert "SPY   sell 10 market" in out
        assert "submitted -   updated" in out
        assert "fills (0)" in out and "(none)" in out

    def test_trace_lists_each_fill_under_its_own_order(self, db_path, conn, trace_data, capsys):
        insert_proposal(conn, Proposal.model_validate(trace_data["proposal"]))
        upsert_order(conn, _bare_order("ord-a", "coid-a", updated_at=AWARE))
        upsert_order(conn, _bare_order("ord-b", "coid-b", updated_at=AWARE + timedelta(minutes=1)))
        record_fill(conn, Fill(id="fill-a", order_id="ord-a", filled_at=AWARE, fill_price=1.0, fill_quantity=1))
        record_fill(conn, Fill(id="fill-b", order_id="ord-b", filled_at=AWARE, fill_price=2.0, fill_quantity=1))
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "orders (2)" in out
        assert out.count("fills (1)") == 2 and "fills (2)" not in out
        assert out.count("id fill-a") == 1 and out.count("id fill-b") == 1
        assert out.index("coid-a") < out.index("id fill-a") < out.index("coid-b") < out.index("id fill-b")

    def test_trace_prints_quantities_and_prices_exactly(self, db_path, conn, trace_data, capsys):
        """An audit view shows every stored digit: no rounding, no scientific notation."""
        insert_proposal(conn, Proposal.model_validate(
            {**trace_data["proposal"], "quantity": 0.5, "limit_price": 0.0001}
        ))
        upsert_order(conn, _bare_order(
            "ord-big", "coid-big", quantity=1000000, limit_price=1234567.891, updated_at=AWARE,
        ))
        upsert_order(conn, _bare_order(
            "ord-small", "coid-small", quantity=0.000123456789, updated_at=AWARE + timedelta(minutes=1),
        ))
        record_fill(conn, Fill(order_id="ord-big", fill_price=0.00011, fill_quantity=250000.5, fees=0.001))
        record_fill(conn, Fill(order_id="ord-small", fill_price=123.456789, fill_quantity=0.25, fees=0))
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "order        buy 0.5 limit @ 0.0001" in out
        assert "SPY   buy 1000000 limit @ 1234567.891" in out
        assert "SPY   buy 0.000123456789 market" in out
        assert "250000.5 @ 0.00011   fees 0.001" in out
        assert "0.25 @ 123.456789   fees 0.00" in out
        for rounded in ("1e+06", "1.23457e", "250000 @", "0.000123457", "@ 0.00 ", "@ 123.46"):
            assert rounded not in out, rounded
        # the helpers themselves: integral values drop the ".0", money keeps two decimals at least
        assert [cli_trace._qty(v) for v in (2.0, 10, 0.5, 1e6, 1e-5, 1234567.5)] == [
            "2", "10", "0.5", "1000000", "0.00001", "1234567.5",
        ]
        assert [cli_trace._money(v) for v in (12.2, 12.15, 1.3, 0.0, 0.0001, 123.456789, None)] == [
            "12.20", "12.15", "1.30", "0.00", "0.0001", "123.456789", "-",
        ]

    def test_memory_sentinel_never_becomes_a_file(self, tmp_path, monkeypatch, capsys):
        """``--db :memory:`` is sqlite's in-memory database, not a file called ':memory:' in the CWD."""
        monkeypatch.chdir(tmp_path)
        assert cli_db._db_path(":memory:") == Path(":memory:") == cli_trace._db_path(":memory:")
        assert cli_db.main(["init", "--db", ":memory:"]) == 0
        out = capsys.readouterr().out
        assert "database: :memory:" in out and "journal mode: memory" in out
        assert "schema version 4 (applied: 0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution)" in out
        assert cli_db.main(["status", "--db", ":memory:"]) == 1
        assert "exists: no" in capsys.readouterr().out
        assert cli_trace.main(["prop-0001", "--db", ":memory:"]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("trace failed: ")
        assert "in-memory" in captured.err and captured.err.count("\n") == 1
        assert list(tmp_path.iterdir()) == []

    def test_trace_unknown_id_is_one_clean_line(self, db_path, conn, capsys):
        assert cli_trace.main(["nope", "--db", str(db_path)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("trace failed: ") and "nope" in captured.err
        assert captured.err.count("\n") == 1 and "Traceback" not in captured.err

    def test_trace_missing_db_is_not_created(self, tmp_path, capsys):
        missing = tmp_path / "nested" / "t.db"
        assert cli_trace.main(["prop-0001", "--db", str(missing)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "does not exist" in captured.err and "aegis.cli.db init" in captured.err
        assert not missing.exists() and not missing.parent.exists()

    def test_trace_never_migrates(self, db_path, tmp_path, monkeypatch, capsys):
        """A read-only view applies no DDL: an unmigrated or behind database is refused, unchanged.

        Otherwise running trace from a newer checkout would upgrade the live
        database under the running loop, whose older code then refuses it.
        """
        connect(db_path).close()  # the file exists, but nothing has been applied
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("trace failed: ")
        assert "pending migrations (0001_initial, 0002_reasoning_cycles_and_legs, 0003_controls, 0004_execution)" in captured.err
        assert "aegis.cli.db init" in captured.err and captured.err.count("\n") == 1
        conn = connect(db_path)
        assert schema_version(conn) == 0 and _table_names(conn) == set()
        # a database one migration behind this code is refused the same way
        assert migrate(conn) == LATEST
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        for _, _, real in list_migrations():
            (migrations / real.name).write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        (migrations / "0005_extra.sql").write_text(
            "CREATE TABLE IF NOT EXISTS extra (id TEXT PRIMARY KEY NOT NULL);\n", encoding="utf-8"
        )
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 1
        assert "pending migrations (0005_extra)" in capsys.readouterr().err
        assert schema_version(conn) == LATEST and "extra" not in _table_names(conn)
        conn.close()

    def test_trace_prints_non_ascii_content_under_an_ascii_stdout(self, db_path, conn, trace_data, monkeypatch):
        """The audit view degrades to \\uXXXX escapes rather than aborting on the content it exists to show."""
        thesis = "SYNTHETIC — a café thesis with an em dash and a naïve accent"
        insert_proposal(conn, Proposal.model_validate({**trace_data["proposal"], "thesis": thesis}))
        ascii_stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")  # strict by default
        monkeypatch.setattr(sys, "stdout", ascii_stdout)
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 0
        ascii_stdout.flush()
        out = ascii_stdout.buffer.getvalue().decode("ascii")
        assert "    SYNTHETIC \\u2014 a caf\\xe9 thesis with an em dash and a na\\xefve accent" in out.splitlines()
        assert "orders (0)" in out  # the view ran to the end
        ascii_stdout = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        monkeypatch.setattr(sys, "stdout", ascii_stdout)
        assert cli_trace.main(["prop-0001", "--db", str(db_path), "--json"]) == 0
        ascii_stdout.flush()
        assert json.loads(ascii_stdout.buffer.getvalue().decode("ascii"))["proposal"]["thesis"] == thesis

    def test_trace_json(self, db_path, conn, trace_data, capsys):
        proposal, *_ = _seed_trace(conn, trace_data)
        assert cli_trace.main([proposal.id, "--db", str(db_path), "--json"]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        data = json.loads(captured.out)
        assert set(data) == {"proposal", "legs", "reasoning", "decisions", "approvals", "orders", "fills"}
        assert data["proposal"]["id"] == "prop-0001" and data["legs"] == []
        assert [r["cycle_id"] for r in data["reasoning"]] == [None] * 3  # the fixture rows predate cycles
        assert [r["stage"] for r in data["reasoning"]] == ["scan", "thesis", "proposal"]
        assert data["decisions"][0]["verdict"] == "NEEDS_APPROVAL"
        assert data["decisions"][0]["rules_evaluated"] == trace_data["decision"]["rules_evaluated"]
        assert data["orders"][0]["client_order_id"] == "aegis-prop-0001-1"
        assert data["fills"][0]["fill_price"] == 12.15
        trace = get_proposal_trace(conn, proposal.id)
        assert ProposalTrace.model_validate(data) == trace
        # the exact shape of model_dump(mode="json"): any datetime text would round-trip above
        assert data == trace.model_dump(mode="json")
        assert data["proposal"]["created_at"] == "2026-07-30T14:05:00Z"
        assert data["fills"][0]["filled_at"] == "2026-07-30T14:07:19Z"
        assert captured.out.startswith("{\n  ")  # indent=2

    def test_trace_errors_are_one_clean_line(self, db_path, conn, monkeypatch, capsys):
        def boom(conn, proposal_id):
            raise RuntimeError("boom")

        monkeypatch.setattr(cli_trace, "get_proposal_trace", boom)
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 1
        assert capsys.readouterr().err == "trace failed (unexpected): RuntimeError: boom\n"

        def interrupted(conn, proposal_id):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_trace, "get_proposal_trace", interrupted)
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 130

    def test_cli_prog_strings(self, capsys):
        for module, argv in ((cli_db, ["--help"]), (cli_trace, ["--help"])):
            with pytest.raises(SystemExit) as info:
                module.main(argv)
            assert info.value.code == 0
            assert capsys.readouterr().out.startswith("usage: python -m " + module.__name__)
