"""Persistence layer: connection, migrations, schema, models, repository — tmp_path databases only.

The first half writes rows with parameterized SQL directly to pin the
contract the repository builds on: WAL semantics, atomic migrations, schema
constraints, and model round-trips. ``TestRepo`` then exercises the typed
repository functions end to end from the ``proposal_trace.json`` fixture.
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
from aegis.store import db as store_db
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
    ProposalTrace,
    Reasoning,
    ReasoningStage,
    StoreStatus,
    Verdict,
    new_id,
)
from aegis.store.repo import (
    add_reasoning,
    get_daily_pnl,
    get_open_orders,
    get_order,
    get_proposal,
    get_proposal_trace,
    get_recent_events,
    insert_proposal,
    log_event,
    record_approval,
    record_decision,
    record_fill,
    snapshot_pnl,
    snapshot_positions,
    upsert_order,
)

NAIVE = datetime(2026, 7, 30, 14, 0, 0)
AWARE = NAIVE.replace(tzinfo=timezone.utc)


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


def _insert_order(conn, order, **overrides):
    """Insert an order row; ``overrides`` replace column values (to hit CHECKs)."""
    row = dict(zip(order.model_dump(), _sql_values(order)))
    row.update(overrides)
    conn.execute(
        "INSERT INTO orders (id, proposal_id, client_order_id, broker, broker_order_id, status,"
        " submitted_at, updated_at, symbol, side, quantity, limit_price)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(row.values()),
    )


def _schema_rows(conn):
    """Every schema_version row in full, so a re-applied migration cannot hide behind a count."""
    rows = conn.execute("SELECT rowid, version, name, applied_at FROM schema_version").fetchall()
    return [tuple(row) for row in rows]


# The test suite's own per-table counts: literal SQL, never the store's query
# table, so a wrong count query in db.py cannot vouch for itself.
_COUNT_SQL = {
    "proposals": "SELECT COUNT(*) FROM proposals",
    "reasoning": "SELECT COUNT(*) FROM reasoning",
    "policy_decisions": "SELECT COUNT(*) FROM policy_decisions",
    "approvals": "SELECT COUNT(*) FROM approvals",
    "orders": "SELECT COUNT(*) FROM orders",
    "fills": "SELECT COUNT(*) FROM fills",
    "position_snapshots": "SELECT COUNT(*) FROM position_snapshots",
    "pnl_snapshots": "SELECT COUNT(*) FROM pnl_snapshots",
    "events": "SELECT COUNT(*) FROM events",
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


def _optional(annotation):
    """Whether a model field's annotation admits None (``X | None``)."""
    origin = typing.get_origin(annotation)
    return origin in (types.UnionType, typing.Union) and type(None) in typing.get_args(annotation)


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


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
        assert schema_version(conn) == 1
        assert _table_names(conn) == set(TABLES) | {"schema_version"}
        assert len(TABLES) == 9
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
        assert migrate(conn) == 1
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
        assert [row[:3] for row in _schema_rows(conn)] == [(1, 1, "0001_initial")]  # migrated once
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
        assert listed == [(1, "0001_initial", "0001_initial.sql")]
        assert all(path.parent == MIGRATIONS_DIR for _, _, path in list_migrations())

    def test_initial_file_leaves_transactions_to_db_py(self):
        text = (MIGRATIONS_DIR / "0001_initial.sql").read_text(encoding="utf-8")
        assert text.startswith("-- migration: 0001 initial schema")
        assert re.search(r"^\s*(BEGIN|COMMIT|END)\b", text, re.IGNORECASE | re.MULTILINE) is None
        assert text.count("CREATE TABLE IF NOT EXISTS") == 9
        assert "CREATE TABLE " not in text.replace("CREATE TABLE IF NOT EXISTS", "")
        assert "CREATE INDEX " not in text.replace("CREATE INDEX IF NOT EXISTS", "")

    def test_applying_twice_is_a_noop(self, conn, db_path):
        assert migrate(conn) == 1
        before = _schema_rows(conn)
        assert len(before) == 1
        assert migrate(conn) == 1
        assert _schema_rows(conn) == before  # the same row, byte for byte: nothing re-applied
        assert applied_versions(conn) == {1: "0001_initial"}
        applied_at = conn.execute("SELECT applied_at FROM schema_version").fetchone()[0]
        assert datetime.fromisoformat(applied_at).tzinfo is not None
        reopened = open_store(db_path)
        assert schema_version(reopened) == 1
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
        assert migrate(conn) == 1
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
            ("proposals", "order_type"): OrderType, ("reasoning", "stage"): ReasoningStage,
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
            "proposals": 7, "reasoning": 3, "policy_decisions": 4, "approvals": 3, "orders": 10,
            "fills": 0, "position_snapshots": 0, "pnl_snapshots": 0, "events": 5,
        }

        ts = AWARE.isoformat()
        rejected = (
            ("INSERT INTO reasoning (id, proposal_id, stage, created_at, content) VALUES (?, ?, ?, ?, ?)",
             ("x", proposal.id, "dream", ts, "c")),
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
        assert _count(conn, "proposals") == 7 and _count(conn, "orders") == 10

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
            "reasoning": (Reasoning, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("stage", "TEXT"), ("created_at", "TEXT"),
                ("content", "TEXT"), ("tokens_in", "INTEGER"), ("tokens_out", "INTEGER"),
            ]),
            "policy_decisions": (PolicyDecision, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("decided_at", "TEXT"), ("verdict", "TEXT"),
                ("rules_evaluated", "TEXT"), ("failing_rule", "TEXT"), ("notes", "TEXT"),
            ]),
            "approvals": (Approval, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("requested_at", "TEXT"), ("responded_at", "TEXT"),
                ("response", "TEXT"), ("channel", "TEXT"), ("responder", "TEXT"),
            ]),
            "orders": (Order, [
                ("id", "TEXT"), ("proposal_id", "TEXT"), ("client_order_id", "TEXT"), ("broker", "TEXT"),
                ("broker_order_id", "TEXT"), ("status", "TEXT"), ("submitted_at", "TEXT"), ("updated_at", "TEXT"),
                ("symbol", "TEXT"), ("side", "TEXT"), ("quantity", "REAL"), ("limit_price", "REAL"),
            ]),
            "fills": (Fill, [
                ("id", "TEXT"), ("order_id", "TEXT"), ("filled_at", "TEXT"), ("fill_price", "REAL"),
                ("fill_quantity", "REAL"), ("fees", "REAL"),
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
        assert set(expected) == set(TABLES)
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
            "reasoning": ("INSERT INTO reasoning VALUES (?, 'prop-0001', 'scan', ?, 'c', NULL, NULL)", (ts,)),
            "policy_decisions": ("INSERT INTO policy_decisions VALUES (?, 'prop-0001', ?, 'REJECT', '[]', NULL, NULL)",
                                 (ts,)),
            "approvals": ("INSERT INTO approvals VALUES (?, 'prop-0001', ?, NULL, NULL, 'cli', NULL)", (ts,)),
            "orders": ("INSERT INTO orders VALUES (?, 'prop-0001', 'coid-x', 'paper', NULL, 'submitted', NULL, ?,"
                       " 'SPY', 'buy', 1, NULL)", (ts,)),
            "fills": ("INSERT INTO fills VALUES (?, 'ord-0001', ?, 1, 1, 0)", (ts,)),
            "position_snapshots": ("INSERT INTO position_snapshots VALUES (?, ?, 'SPY', 1, NULL, NULL, NULL)", (ts,)),
            "pnl_snapshots": ("INSERT INTO pnl_snapshots VALUES (?, ?, 1, 2, 3, 4, 5, 6)", (ts,)),
            "events": ("INSERT INTO events VALUES (?, ?, 'info', 'k', 'm', NULL)", (ts,)),
        }
        assert set(rows) == set(TABLES)
        for table, (sql, params) in rows.items():
            declared = conn.execute(
                'SELECT "notnull" FROM pragma_table_info(?) WHERE name = ?', (table, "id")
            ).fetchone()[0]
            assert declared == 1, table
            with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
                conn.execute(sql, (None, *params))
            conn.execute(sql, ("id-" + table, *params))  # the same row with an id is fine
        assert {table: _count(conn, table) for table in TABLES} == {
            "proposals": 2, "reasoning": 1, "policy_decisions": 1, "approvals": 1, "orders": 2,
            "fills": 1, "position_snapshots": 1, "pnl_snapshots": 1, "events": 1,
        }
        assert len(get_proposal_trace(conn, "prop-0001").fills) == 1  # and every read still maps
        assert len(get_open_orders(conn)) == 1 and len(get_recent_events(conn)) == 1

    def test_every_foreign_key_and_lookup_column_is_indexed(self, conn):
        indexed = set()
        for table in TABLES:
            for index in conn.execute("SELECT name FROM pragma_index_list(?)", (table,)).fetchall():
                columns = conn.execute("SELECT name FROM pragma_index_info(?)", (index["name"],)).fetchall()
                indexed.add((table, columns[0]["name"]))
        expected = {
            ("reasoning", "proposal_id"),
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
        assert empty.reasoning == () and empty.orders == () and empty.fills == ()
        first = PolicyDecision.model_validate(trace_data["decision"])
        second = first.model_copy(update={"id": "dec-0002", "verdict": Verdict.REJECT})
        trace = ProposalTrace(proposal=proposal, decisions=(first, second))
        assert trace.decision is second
        approval = Approval.model_validate(trace_data["approval"])
        assert ProposalTrace(proposal=proposal, approvals=(approval,)).approval == approval

    def test_store_status_model(self):
        report = StoreStatus(
            path="x.db", exists=True, schema_version=1, journal_mode="wal",
            applied_migrations=("0001_initial",), pending_migrations=(),
            row_counts={table: 0 for table in TABLES},
        )
        assert report.applied_migrations == ("0001_initial",)
        assert set(report.row_counts) == set(TABLES)


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
            path=str(db_path), exists=True, schema_version=1, journal_mode="wal",
            applied_migrations=("0001_initial",), pending_migrations=(),
            row_counts={table: 0 for table in TABLES},
        )
        assert report.exists is True
        assert status(conn, db_path.with_name("other.db")).exists is False  # about the path, not the connection
        _insert_event(conn)
        assert status(conn, db_path).row_counts["events"] == 1

    def test_before_migrating(self, db_path):
        conn = connect(db_path)
        report = status(conn, db_path)
        assert report.exists is True and report.schema_version == 0
        assert report.applied_migrations == ()
        assert report.pending_migrations == ("0001_initial",)
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
        expected = {
            "proposals": 1, "reasoning": 3, "policy_decisions": 1, "approvals": 1, "orders": 1,
            "fills": 2, "position_snapshots": 4, "pnl_snapshots": 5, "events": 6,
        }
        assert status(conn, db_path).row_counts == expected
        assert {table: _count(conn, table) for table in TABLES} == expected

    def test_memory(self):
        conn = connect(":memory:")
        migrate(conn)
        report = status(conn, ":memory:")
        assert report.exists is True and report.journal_mode == "memory" and report.schema_version == 1
        conn.close()


def test_db_files_and_wal_sidecars_are_gitignored():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"*.db", "*.db-wal", "*.db-shm"} <= set(lines)


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
        assert all(_count(conn, table) == 0 for table in TABLES)

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
            "proposals": 1, "reasoning": 3, "policy_decisions": 1, "approvals": 1, "orders": 1,
            "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 1,
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
            lambda c: add_reasoning(c, Reasoning(proposal_id="prop-0001", stage=ReasoningStage.SCAN, content="c")),
            lambda c: record_decision(c, PolicyDecision(proposal_id="prop-0001", verdict=Verdict.REJECT, rules_evaluated=[])),
            lambda c: record_approval(c, Approval(proposal_id="prop-0001", channel="cli")),
            lambda c: upsert_order(c, _bare_order("ord-0009", "coid-0009")),
            lambda c: record_fill(c, Fill(order_id="ord-0001", fill_price=1.0, fill_quantity=1)),
            lambda c: snapshot_positions(c, [PositionSnapshot(symbol="SPY", quantity=1)]),
            lambda c: snapshot_pnl(c, _pnl(AWARE, 1.0)),
            lambda c: log_event(c, Event(level=EventLevel.INFO, kind="k", message="m")),
        ],
        ids=[
            "insert_proposal", "add_reasoning", "record_decision", "record_approval", "upsert_order",
            "record_fill", "snapshot_positions", "snapshot_pnl", "log_event",
        ],
    )
    def test_writes_never_join_a_callers_transaction(self, conn, trace_data, write):
        """Every write opens its own transaction — none autocommits into a caller's block."""
        _seed_trace(conn, trace_data)  # prop-0001 and ord-0001 exist for the FK-bearing writes
        counts = {table: _count(conn, table) for table in TABLES}
        with pytest.raises(StoreError, match="not re-entrant"):
            with transaction(conn):
                write(conn)
        assert not conn.in_transaction
        assert {table: _count(conn, table) for table in TABLES} == counts

    def test_reads_wrap_sqlite_errors(self, conn, trace_data):
        """A read that fails in SQLite surfaces as StoreError naming the operation and id."""
        _seed_trace(conn, trace_data)
        conn.execute("PRAGMA foreign_keys=OFF")  # so the parent tables can go
        for statement in (
            "DROP TABLE events", "DROP TABLE pnl_snapshots", "DROP TABLE orders", "DROP TABLE proposals",
        ):
            conn.execute(statement)
        for what, call in (
            ("get recent events", lambda: get_recent_events(conn)),
            ("get daily pnl", lambda: get_daily_pnl(conn)),
            ("get open orders", lambda: get_open_orders(conn)),
            ("get order for aegis-prop-0001-1", lambda: get_order(conn, "aegis-prop-0001-1")),
            ("get proposal for prop-0001", lambda: get_proposal(conn, "prop-0001")),
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
            "proposals": 1, "reasoning": 3, "policy_decisions": 1, "approvals": 1, "orders": 1,
            "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 0,
        }

    def test_repo_functions_are_re_exported(self):
        for name in (
            "insert_proposal", "add_reasoning", "record_decision", "record_approval", "upsert_order",
            "record_fill", "snapshot_positions", "snapshot_pnl", "log_event", "get_proposal",
            "get_order", "get_open_orders", "get_daily_pnl", "get_recent_events", "get_proposal_trace",
        ):
            assert getattr(aegis.store, name) is getattr(repo, name)
            assert name in aegis.store.__all__


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
        assert "schema version 1 (applied: 0001_initial)" in out
        assert "up to date" not in out

        assert cli_db.main(["init", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "(created)" not in out and "wal" in out
        assert "schema version 1 (applied: 0001_initial) [up to date]" in out
        conn = connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        assert schema_version(conn) == 1
        conn.close()

    def test_db_status_missing_file_exits_1_without_creating_it(self, tmp_path, capsys):
        missing = tmp_path / "nested" / "t.db"
        assert cli_db.main(["status", "--db", str(missing)]) == 1
        captured = capsys.readouterr()
        assert str(missing) in captured.out and "exists: no" in captured.out
        assert "aegis.cli.db init" in captured.out
        assert not missing.exists() and not missing.parent.exists()

    def test_db_status_after_init_lists_nine_tables(self, db_path, conn, trace_data, capsys):
        _seed_trace(conn, trace_data)
        assert cli_db.main(["status", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert str(db_path) in out and "exists: yes" in out
        assert "journal mode: wal" in out
        assert "schema version 1 (applied: 0001_initial)" in out
        assert "pending migrations: none" in out
        counts = dict(re.findall(r"^  (\w+)\s+(\d+)$", out, re.MULTILINE))
        assert set(counts) == set(TABLES) and len(counts) == 9
        assert {t: int(n) for t, n in counts.items()} == {
            "proposals": 1, "reasoning": 3, "policy_decisions": 1, "approvals": 1, "orders": 1,
            "fills": 1, "position_snapshots": 0, "pnl_snapshots": 0, "events": 0,
        }

    def test_db_status_on_unmigrated_file(self, db_path, capsys):
        connect(db_path).close()  # the file exists, but nothing has been applied
        assert cli_db.main(["status", "--db", str(db_path)]) == 0
        out = capsys.readouterr().out
        assert "schema version 0 (applied: none)" in out
        assert "pending migrations: 0001_initial" in out
        assert re.search(r"^  events\s+0$", out, re.MULTILINE)

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
        for title in ("reasoning (0)", "decisions (0)", "approvals (0)", "orders (0)"):
            assert title in out
        assert out.count("(none)") == 4

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
        assert "schema version 1 (applied: 0001_initial)" in out
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
        assert captured.err.startswith("trace failed: ") and "pending migrations (0001_initial)" in captured.err
        assert "aegis.cli.db init" in captured.err and captured.err.count("\n") == 1
        conn = connect(db_path)
        assert schema_version(conn) == 0 and _table_names(conn) == set()
        # a database one migration behind this code is refused the same way
        assert migrate(conn) == 1
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        initial = MIGRATIONS_DIR / "0001_initial.sql"
        (migrations / initial.name).write_text(initial.read_text(encoding="utf-8"), encoding="utf-8")
        (migrations / "0002_extra.sql").write_text(
            "CREATE TABLE IF NOT EXISTS extra (id TEXT PRIMARY KEY NOT NULL);\n", encoding="utf-8"
        )
        monkeypatch.setattr(store_db, "MIGRATIONS_DIR", migrations)
        assert cli_trace.main(["prop-0001", "--db", str(db_path)]) == 1
        assert "pending migrations (0002_extra)" in capsys.readouterr().err
        assert schema_version(conn) == 1 and "extra" not in _table_names(conn)
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
        assert set(data) == {"proposal", "reasoning", "decisions", "approvals", "orders", "fills"}
        assert data["proposal"]["id"] == "prop-0001"
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
