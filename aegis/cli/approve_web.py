"""The phone approval page, the operator's side.

Run:  python -m aegis.cli.approve_web serve [--db PATH]
      python -m aegis.cli.approve_web hash-pin
      python -m aegis.cli.approve_web unlock --by NAME [--db PATH]
      python -m aegis.cli.approve_web status [--db PATH]

``serve`` runs the page on 127.0.0.1:``approve_web.port`` until stopped
(launchd keeps it up: ops/launchd/local.aegis.approve-web.plist). It
refuses while ``approve_web.enabled`` is false, and when ``APPROVE_WEB_URL``,
``APPROVE_WEB_LOGIN`` or ``APPROVE_WEB_PIN_HASH`` is missing or malformed in
.env; the error names the variable, never its value.

``hash-pin`` asks for a PIN twice (nothing echoed) and prints the line to
put in .env as ``APPROVE_WEB_PIN_HASH``. The PIN itself is never stored.

``unlock`` clears a lockout after ``approve_web.max_pin_failures`` wrong
PINs, on the record (an ``approve_web_unlocked`` event). ``status`` says
whether the page is on, whether .env holds what it needs (yes or no, never
a value), and how many wrong PINs in a row stand.

Failures are one line on stderr and exit 1. Ctrl-C exits 130.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from aegis.approve_web import auth
from aegis.approve_web.app import App, Settings, failures_in_a_row, unlock
from aegis.approve_web.server import LOOPBACK, make_server
from aegis.cli.policy import (
    _clean,
    _db_path,
    _Failure,
    _Parser,
    _require_file,
    _require_schema,
    _UsageError,
)
from aegis.config import AegisConfig, ConfigError, get_config, load_env, require_env
from aegis.data.models import utcnow
from aegis.store.db import connect, open_store
from aegis.store.errors import StoreError

NO_STORE = "the page reads and writes the store"


def settings_from_env() -> Settings:
    """The page's .env values, checked. ``ConfigError`` names what is wrong."""
    load_env()
    return Settings(
        page_url=auth.check_page_url(require_env(auth.URL_ENV)),
        login=auth.check_login(require_env(auth.LOGIN_ENV)),
        pin_hash=auth.parse_pin_hash(require_env(auth.PIN_HASH_ENV)),
    )


def _store(db_path: Path) -> Path:
    _require_file(db_path, NO_STORE)
    _require_schema(db_path)
    return db_path


def _serve(config: AegisConfig, db_path: Path, clock: Callable[[], datetime]) -> int:
    if not config.approve_web.enabled:
        raise _Failure("approve_web.enabled is false in config.yaml: the page stays off")
    settings = settings_from_env()
    app = App(config, _store(db_path), settings, clock=clock)
    server = make_server(app, config.approve_web.port, config.approve_web.request_timeout_seconds)
    print(f"approval page on http://{LOOPBACK}:{config.approve_web.port}"
          " (publish it with `tailscale serve`, never Funnel)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def _hash_pin(ask: Callable[[str], str]) -> int:
    pin = ask("PIN: ")
    try:
        auth.check_pin_shape(pin)
    except ValueError as exc:
        raise _Failure(str(exc)) from None
    if ask("PIN again: ") != pin:
        raise _Failure("the two PINs differ")
    print(f"{auth.PIN_HASH_ENV}={auth.hash_pin(pin)}")
    print("put that line in .env on the mini; the PIN itself is not stored anywhere")
    return 0


def _unlock(db_path: Path, by: str, clock: Callable[[], datetime]) -> int:
    conn = open_store(_store(db_path))
    try:
        event = unlock(conn, _clean(by), clock())
    finally:
        conn.close()
    print(f"unlocked ({(event.payload or {}).get('failures', 0)} wrong PINs cleared)")
    return 0


def _status(config: AegisConfig, db_path: Path) -> int:
    load_env()
    print(f"approve_web.enabled: {str(config.approve_web.enabled).lower()}")
    for name in (auth.URL_ENV, auth.LOGIN_ENV, auth.PIN_HASH_ENV):
        print(f"{name}: {'set' if os.getenv(name, '').strip() else 'missing'}")
    conn = connect(_store(db_path))
    try:
        failures = failures_in_a_row(conn)
    finally:
        conn.close()
    limit = config.approve_web.max_pin_failures
    state = "LOCKED" if failures >= limit else "open"
    print(f"wrong PINs in a row: {failures} of {limit} ({state})")
    return 0


def _parser() -> _Parser:
    parser = _Parser(prog="python -m aegis.cli.approve_web",
                     description="Run and look after the phone approval page.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", parents=[common], help="run the page on the loopback address")
    commands.add_parser("hash-pin", help="make the APPROVE_WEB_PIN_HASH line for .env")
    unlock_cmd = commands.add_parser("unlock", parents=[common], help="clear a PIN lockout")
    unlock_cmd.add_argument("--by", required=True, metavar="NAME", help="who unlocks")
    commands.add_parser("status", parents=[common], help="is the page on, set up, locked")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    clock: Callable[[], datetime] | None = None,
    ask: Callable[[str], str] = getpass.getpass,
) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "unlock" and not args.by.strip():
            raise _UsageError("--by must not be blank")
    except _UsageError as exc:
        print(f"{_clean(exc)} (see --help)", file=sys.stderr)
        return 1
    now_of = clock if clock is not None else utcnow
    try:
        if args.command == "hash-pin":
            return _hash_pin(ask)
        config = get_config()
        db_path = _db_path(args.db)
        if args.command == "serve":
            return _serve(config, db_path, now_of)
        if args.command == "unlock":
            return _unlock(db_path, args.by, now_of)
        return _status(config, db_path)
    except (_Failure, ConfigError, StoreError) as exc:
        print(f"approve-web {args.command} failed: {_clean(exc)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
