"""Create, migrate, and inspect the Phase 3 SQLite store.

Run:  python -m aegis.cli.db init [--db PATH]
      python -m aegis.cli.db status [--db PATH]

``init`` opens the database (creating the file if needed), applies every
pending migration, and reports the result; running it again is a no-op.
``status`` never creates anything: it reports on a database that exists and
exits 1 for one that does not. Both exit 1 when a migrated database lacks a
table (dropped by hand — migrations are never re-run, so restore it from
the migration file). ``--db`` overrides ``store.db_path`` from config.yaml
and is taken relative to the current directory; ``--db :memory:`` is
sqlite's in-memory sentinel, never a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aegis.config import ConfigError
from aegis.store.db import (
    MEMORY_DB,
    TABLES,
    connect,
    migrate,
    resolve_db_path,
    schema_version,
    status,
)
from aegis.store.errors import StoreError
from aegis.store.models import StoreStatus


def _db_path(arg: str | None) -> Path:
    """An explicit --db resolves against the CWD; otherwise config.yaml decides."""
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)  # resolving it would name a file ":memory:" in the CWD
    return Path(arg).expanduser().resolve()


def _schema_line(report: StoreStatus) -> str:
    applied = ", ".join(report.applied_migrations) or "none"
    return f"schema version {report.schema_version} (applied: {applied})"


def _missing_tables(report: StoreStatus) -> int:
    """Exit status for the report: 1 with an explanation when a table is gone."""
    if not report.missing_tables:
        return 0
    print(
        f"missing tables: {', '.join(report.missing_tables)}"
        " (dropped by hand? a recorded migration is never re-run:"
        " restore them from aegis/store/migrations)"
    )
    return 1


def _init(db_path: Path) -> int:
    created = str(db_path) == MEMORY_DB or not db_path.exists()
    conn = connect(db_path)
    try:
        before = schema_version(conn)
        migrate(conn)
        report = status(conn, db_path)
    finally:
        conn.close()
    print(f"database: {report.path}" + (" (created)" if created else ""))
    print(f"journal mode: {report.journal_mode}")
    up_to_date = " [up to date]" if report.schema_version == before else ""
    print(_schema_line(report) + up_to_date)
    return _missing_tables(report)


def _status(db_path: Path) -> int:
    print(f"database: {db_path}")
    if str(db_path) == MEMORY_DB:
        print("exists: no (:memory: is an in-memory database, not a file)")
        return 1
    if not db_path.exists():
        print("exists: no (run `python -m aegis.cli.db init` to create it)")
        return 1
    conn = connect(db_path)
    try:
        report = status(conn, db_path)
    finally:
        conn.close()
    print("exists: yes")
    print(f"journal mode: {report.journal_mode}")
    print(_schema_line(report))
    print(f"pending migrations: {', '.join(report.pending_migrations) or 'none'}")
    print("row counts")
    for table in TABLES:
        count = "missing" if table in report.missing_tables else report.row_counts[table]
        print(f"  {table:<20}{count:>7}")
    return _missing_tables(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aegis.cli.db",
        description="Create, migrate, and inspect the AEGIS SQLite store.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "init", parents=[common], help="create the database if needed and apply pending migrations"
    )
    commands.add_parser(
        "status", parents=[common], help="report schema version, journal mode, and row counts"
    )
    args = parser.parse_args(argv)

    try:
        db_path = _db_path(args.db)
        return _init(db_path) if args.command == "init" else _status(db_path)
    except (ConfigError, StoreError) as exc:
        print(f"db {args.command} failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(
            f"db {args.command} failed (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
