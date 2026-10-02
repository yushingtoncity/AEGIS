"""SQLite connection, pragmas, versioned migrations, and the transaction helper.

Connections are opened with ``isolation_level=None`` — the sqlite3 module
never starts implicit transactions, every statement autocommits, and the
only transactions are the explicit ``BEGIN IMMEDIATE … COMMIT`` blocks that
``transaction`` issues. WAL journaling lets a dashboard read while the loop
writes; ``busy_timeout`` makes a second writer wait instead of failing. The
one statement SQLite will not apply the busy handler to — switching a
not-yet-WAL file to WAL, which needs an exclusive lock — is retried by hand
in ``_enable_wal`` for the same 5 s, so two processes may create the same
fresh database at once.

Migration atomicity (verified on CPython 3.14.7 / SQLite 3.53.4): under
``isolation_level=None`` the connection is in legacy transaction control,
and ``Connection.executescript`` issues an implicit COMMIT before running
its script — an explicit ``BEGIN`` followed by ``executescript`` therefore
ends the transaction, and the DDL and the ``schema_version`` row would
commit separately. ``migrate`` instead splits each ``.sql`` file into single
statements with ``sqlite3.complete_statement`` (which respects ``;`` inside
string literals, comments and a trigger's ``BEGIN … END`` body — 0003's
no-delete trigger stays one statement) and runs them through
``Connection.execute`` inside one ``BEGIN IMMEDIATE … COMMIT`` together with
the version row, so a migration lands whole or not at all. (The 3.12+
``autocommit=True`` mode would keep the transaction open across
``executescript``, but changes the transaction semantics of the whole
connection; splitting keeps the connection contract simple.) The statements
are idempotent regardless.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aegis.config import REPO_ROOT, get_config
from aegis.data.models import utcnow
from aegis.store.errors import StoreError
from aegis.store.models import StoreStatus

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
MEMORY_DB = ":memory:"
_WAL_RETRY_INTERVAL_S = 0.01

TABLES: tuple[str, ...] = (
    "proposals",
    "proposal_legs",
    "reasoning",
    "policy_decisions",
    "approvals",
    "orders",
    "fills",
    "position_snapshots",
    "pnl_snapshots",
    "events",
    "controls",
)

# One literal query per table: table names never come from user input and
# never get interpolated into SQL.
_COUNT_QUERIES: dict[str, str] = {
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

_MIGRATION_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")

_SCHEMA_VERSION_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_version ("
    " version INTEGER PRIMARY KEY,"
    " name TEXT NOT NULL,"
    " applied_at TEXT NOT NULL)"
)


def resolve_db_path(path: str | Path | None = None) -> Path:
    """The database file to use: an explicit argument wins, else config.

    Relative paths resolve against the repo root (config.yaml's ``db_path``
    is written relative to it); ``":memory:"`` passes through untouched.
    """
    raw = get_config().store.db_path if path is None else path
    if str(raw) == MEMORY_DB:
        return Path(MEMORY_DB)
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _enable_wal(conn: sqlite3.Connection) -> str:
    """``PRAGMA journal_mode=WAL``, retried while the file is locked, for ``busy_timeout`` at most.

    Switching a rollback-journal file to WAL needs an exclusive lock, and a
    connection that already holds a shared lock (it has read the header)
    gets SQLITE_BUSY straight back rather than a call to the busy handler —
    SQLite's deadlock-avoidance rule — so ``busy_timeout`` does not cover
    this one statement. Without the retry, two processes opening a
    brand-new database together (the loop and the dashboard on a fresh
    deployment) fail at once with "database is locked". Returns the
    resulting journal mode, lower-cased.
    """
    timeout_ms = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            return str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(_WAL_RETRY_INTERVAL_S)


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database (creating the file and its directory), set pragmas.

    Autocommit with explicit transactions (``isolation_level=None``), rows
    as ``sqlite3.Row``, foreign keys enforced, a 5 s busy timeout, WAL
    journaling (a file DB reports "wal"; ":memory:" reports "memory") and
    ``synchronous=NORMAL``, the usual pairing with WAL. Does not migrate —
    ``open_store`` does.
    """
    db_path = resolve_db_path(path)
    in_memory = str(db_path) == MEMORY_DB
    try:
        if not in_memory:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), isolation_level=None)
    except (sqlite3.Error, OSError) as exc:
        raise StoreError("connect", str(db_path), exc) from exc
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        journal_mode = _enable_wal(conn)
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error as exc:
        conn.close()
        raise StoreError("connect", str(db_path), exc) from exc
    except BaseException:  # a KeyboardInterrupt mid-pragma must not leak the handle
        conn.close()
        raise
    expected = "memory" if in_memory else "wal"
    if journal_mode != expected:
        conn.close()
        raise StoreError(
            f"connect (journal_mode is {journal_mode!r}, expected {expected!r})", str(db_path)
        )
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run the block inside ``BEGIN IMMEDIATE … COMMIT``; ROLLBACK and re-raise on error.

    ``BEGIN IMMEDIATE`` takes the write lock up front, so a competing writer
    waits (``busy_timeout``) here rather than failing at COMMIT, and WAL
    readers are never blocked. Not re-entrant: SQLite has no nested
    transactions, so nesting raises StoreError instead of silently joining
    the outer block — never nest it. Every ``sqlite3.Error`` — the busy
    timeout expiring at BEGIN, a constraint failing in the block, COMMIT
    itself — passes through unchanged after the rollback, so that the caller
    wraps it with the operation and id (the repository names the record,
    ``migrate`` the migration).
    """
    if conn.in_transaction:
        raise StoreError("begin transaction (already in one: transaction() is not re-entrant)")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def list_migrations() -> list[tuple[int, str, Path]]:
    """``(version, name, path)`` for every ``NNNN_name.sql`` in MIGRATIONS_DIR, by version.

    ``name`` is the file stem (e.g. ``0001_initial``) — what schema_version
    records and the CLIs print. A file that does not match the pattern, a
    version of 0, or two files sharing a version is a StoreError.
    """
    found: dict[int, tuple[str, Path]] = {}
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _MIGRATION_FILE_RE.match(path.name)
        if not match:
            raise StoreError("list migrations (expected NNNN_name.sql)", path.name)
        version = int(match.group(1))
        if version < 1:
            raise StoreError("list migrations (versions start at 0001)", path.name)
        if version in found:
            raise StoreError(f"list migrations (duplicate version {version:04d})", path.name)
        found[version] = (path.stem, path)
    return [(version, name, path) for version, (name, path) in sorted(found.items())]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def applied_versions(conn: sqlite3.Connection) -> dict[int, str]:
    """``{version: name}`` of every applied migration; ``{}`` before the first."""
    try:
        if not _table_exists(conn, "schema_version"):
            return {}
        rows = conn.execute("SELECT version, name FROM schema_version ORDER BY version").fetchall()
    except sqlite3.Error as exc:
        raise StoreError("read schema_version", cause=exc) from exc
    return {int(row[0]): str(row[1]) for row in rows}


def schema_version(conn: sqlite3.Connection) -> int:
    """The highest applied migration version; 0 for an empty database."""
    return max(applied_versions(conn), default=0)


def _split_statements(script: str) -> list[str]:
    """Split a migration file into single statements for ``Connection.execute``.

    A chunk ends at the first ``;`` that completes a statement, wherever it
    falls — two statements on one line are two chunks, like ``executescript``
    would see them. ``sqlite3.complete_statement`` understands string
    literals, comments and trigger bodies, so a ``;`` inside any of those
    never splits. Comment-only chunks (the file header, a trailing comment)
    are kept: ``execute`` treats them as empty statements. See the module
    docstring for why this replaces ``executescript``.
    """
    statements: list[str] = []
    start = 0
    for index, char in enumerate(script):
        if char == ";" and sqlite3.complete_statement(script[start : index + 1]):
            statements.append(script[start : index + 1])
            start = index + 1
    tail = script[start:]
    if tail.strip():
        statements.append(tail)
    return statements


def _apply_migration(conn: sqlite3.Connection, version: int, name: str, path: Path) -> None:
    """Apply one migration file and record it, atomically (see module docstring)."""
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StoreError("read migration", name, exc) from exc
    try:
        with transaction(conn):
            if version in applied_versions(conn):
                return  # another process applied it while we waited for the write lock
            for statement in _split_statements(script):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, utcnow().isoformat()),
            )
    except sqlite3.Error as exc:
        raise StoreError("apply migration", name, exc) from exc


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every unapplied migration in version order; return the resulting version.

    Idempotent: a second call finds nothing pending and changes nothing.
    Refuses (StoreError) a database whose version is higher than the newest
    migration file — that database was written by newer code.

    A recorded migration is never re-run, even though every statement in it
    is idempotent: a table dropped by hand stays missing after this call
    (``status`` lists it under ``missing_tables`` and ``db status`` says so)
    and is restored from its migration file, not by running this again.
    """
    migrations = list_migrations()
    newest = migrations[-1][0] if migrations else 0
    current = schema_version(conn)
    if current > newest:
        raise StoreError(
            f"migrate (database schema version {current} is newer than this code's {newest}"
            " — upgrade AEGIS before opening it)"
        )
    try:
        conn.execute(_SCHEMA_VERSION_DDL)
    except sqlite3.Error as exc:
        raise StoreError("create schema_version", cause=exc) from exc
    applied = applied_versions(conn)
    for version, name, path in migrations:
        if version not in applied:
            _apply_migration(conn, version, name, path)
    return schema_version(conn)


def open_store(path: str | Path | None = None) -> sqlite3.Connection:
    """``connect`` then ``migrate``: what the loop calls at startup.

    The CLIs do not: ``db init`` runs the two steps itself to report what
    changed, and ``db status`` / ``trace`` only ever ``connect`` — a report
    or an audit view must not apply a migration as a side effect.
    """
    conn = connect(path)
    try:
        migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def status(conn: sqlite3.Connection, path: str | Path) -> StoreStatus:
    """What the database at ``path`` (already open as ``conn``) looks like.

    A table that does not exist counts 0 rows. While migrations are pending
    that is expected; once none are, every table in ``TABLES`` must exist,
    and any that does not is reported in ``missing_tables`` — ``migrate``
    will not bring it back (see there).
    """
    db_path = Path(path)
    in_memory = str(db_path) == MEMORY_DB
    applied = applied_versions(conn)
    pending = tuple(name for version, name, _ in list_migrations() if version not in applied)
    try:
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        present = {table: _table_exists(conn, table) for table in TABLES}
        row_counts = {
            table: int(conn.execute(_COUNT_QUERIES[table]).fetchone()[0]) if present[table] else 0
            for table in TABLES
        }
    except sqlite3.Error as exc:
        raise StoreError("read status", str(db_path), exc) from exc
    return StoreStatus(
        path=str(db_path),
        exists=True if in_memory else db_path.exists(),
        schema_version=max(applied, default=0),
        journal_mode=journal_mode,
        applied_migrations=tuple(applied[version] for version in sorted(applied)),
        pending_migrations=pending,
        row_counts=row_counts,
        missing_tables=() if pending else tuple(table for table in TABLES if not present[table]),
    )
