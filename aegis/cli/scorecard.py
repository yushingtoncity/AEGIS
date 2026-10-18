"""Grade past proposals by what the underlying did afterwards. Read-only.

Run:  python -m aegis.cli.scorecard [--since YYYY-MM-DD] [--until YYYY-MM-DD]
                                    [--group-by proposal|thesis|pair|symbol]
                                    [--detail] [--json] [--db PATH]

For every stored proposal it works out the call (bullish, bearish, or
neutral for a delta-flat option structure), takes the entry price from the
cycle's stored snapshot, and checks the underlying's close a few trading
sessions later (``scorecard.horizons_days`` in config.yaml). The summary
shows, per group and horizon, how many calls were graded, the hit rate, the
average signed return (positive means the calls were right on average), the
average confidence and the Brier score (lower is better; 0.25 is what always
saying 50% earns). Grouping by ``proposal`` (the default) compares the model
the proposal stage ran on, ``thesis`` the thesis stage's model, ``pair``
both. See ``aegis.scorecard.grade`` for exactly how a call is graded.

Later closes come from Alpaca daily bars (the free IEX feed when the account
has no paid data), so this needs the Alpaca keys in .env. The database is
opened with ``PRAGMA query_only``: the scorecard never writes, never applies
a migration, and never touches the broker. ``--db`` overrides
``store.db_path`` and is taken relative to the current directory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, time
from pathlib import Path

from aegis.config import ConfigError, get_config
from aegis.data.errors import DataError
from aegis.data.market import get_bars
from aegis.data.models import Bar, utcnow
from aegis.scorecard.gather import gather
from aegis.scorecard.grade import GROUP_KEYS, NEW_YORK, build_scorecard, first_session_day
from aegis.scorecard.models import HorizonResult, Scorecard, ScoredProposal
from aegis.store.db import MEMORY_DB, connect, resolve_db_path, status
from aegis.store.errors import StoreError

INDENT = "  "
BAR_FETCH_PADDING_DAYS = 7  # calendar days of slack before the first proposal, for its prior close

GROUP_LABELS = {
    "proposal": "proposal-stage model",
    "thesis": "thesis-stage model",
    "pair": "thesis -> proposal model pair",
    "symbol": "underlying",
}

BarFetcher = Callable[[str, int], list[Bar]]


def _db_path(arg: str | None) -> Path:
    """An explicit --db resolves against the CWD; otherwise config.yaml decides."""
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)
    return Path(arg).expanduser().resolve()


def _day(text: str) -> datetime:
    """A YYYY-MM-DD argument as midnight New York time."""
    try:
        return datetime.combine(date.fromisoformat(text), time(0), tzinfo=NEW_YORK)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """A connection that refuses every write (``PRAGMA query_only``)."""
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA query_only=ON")
    except sqlite3.Error as exc:
        conn.close()
        raise StoreError("open read-only", str(db_path), exc) from exc
    return conn


def _alpaca_daily_bars(symbol: str, lookback_days: int) -> list[Bar]:
    return get_bars(symbol, "1Day", lookback_days=lookback_days)


def fetch_bars(
    symbols: Iterable[str], lookback_days: int, fetcher: BarFetcher
) -> tuple[dict[str, list[Bar]], list[str]]:
    """Daily bars per symbol; a symbol whose fetch fails is reported, not fatal."""
    bars: dict[str, list[Bar]] = {}
    errors: list[str] = []
    for symbol in sorted(set(symbols)):
        try:
            bars[symbol] = fetcher(symbol, lookback_days)
        except DataError as exc:
            errors.append(f"{symbol}: {exc}")
    return bars, errors


def _pct(value: float | None) -> str:
    return f"{value * 100:+.2f}%" if value is not None else "-"


def _rate(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "-"


def _num(value: float | None, spec: str) -> str:
    return format(value, spec) if value is not None else "-"


def _horizon_cell(result: HorizonResult) -> str:
    if result.pending:
        return f"{result.days}d pending"
    if result.hit is None:
        moved = f" moved {_pct(result.underlying_return)}" if result.underlying_return is not None else ""
        return f"{result.days}d ungraded{moved}"
    return f"{result.days}d {_pct(result.directional_return)} {'hit' if result.hit else 'miss'}"


def _print_proposal(proposal: ScoredProposal) -> None:
    when = proposal.created_at.strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"entry {proposal.entry_price:.2f} ({proposal.entry_source})"
        if proposal.entry_price is not None
        else "no entry price"
    )
    print(
        f"{when}  {proposal.underlying}  {proposal.instrument.value} {proposal.side.value}"
        f"  {proposal.direction} ({proposal.direction_basis})"
        f"  conf {proposal.confidence:.2f}  {proposal.verdict or 'undecided'}  {entry}"
    )
    print(INDENT + "   ".join(_horizon_cell(result) for result in proposal.horizons))
    source = f" ({proposal.model_source})" if proposal.model_source else ""
    print(
        f"{INDENT}models {proposal.thesis_model or 'unknown'} -> {proposal.proposal_model}{source}"
        f"   id {proposal.proposal_id}"
    )
    if proposal.note:
        print(f"{INDENT}note: {proposal.note}")


def _print_scorecard(card: Scorecard, detail: bool, errors: Sequence[str]) -> None:
    since = card.since.date().isoformat() if card.since else "the start"
    until = card.until.date().isoformat() if card.until else "now"
    print(
        f"proposal scorecard: {len(card.proposals)} proposal(s) from {since} to {until},"
        f" grouped by {GROUP_LABELS[card.group_by]}"
    )
    print(
        "horizons: "
        + ", ".join(str(days) for days in card.horizons_days)
        + " trading session(s) after the proposal; options are graded on the underlying"
    )
    for error in errors:
        print(f"price fetch failed for {error}")
    if not card.proposals:
        print()
        print("no proposals in this range yet")
        return
    width = max(len("group"), *(len(summary.group) for summary in card.summaries))
    print()
    print(
        f"{'group':<{width}}  {'days':>4}  {'graded':>8}  {'hit rate':>8}"
        f"  {'avg signed':>10}  {'avg conf':>8}  {'brier':>6}"
    )
    for summary in card.summaries:
        graded = f"{summary.graded}/{summary.proposals}"
        print(
            f"{summary.group:<{width}}  {summary.horizon_days:>4}  {graded:>8}"
            f"  {_rate(summary.hit_rate):>8}  {_pct(summary.avg_directional_return):>10}"
            f"  {_num(summary.avg_confidence, '.2f'):>8}  {_num(summary.brier, '.3f'):>6}"
        )
    if detail:
        print()
        for proposal in card.proposals:
            _print_proposal(proposal)


def run(
    *,
    db_path: Path,
    since: datetime | None,
    until: datetime | None,
    group_by: str,
    now: datetime,
    fetcher: BarFetcher,
) -> tuple[Scorecard, list[str]]:
    """Gather, fetch later prices, grade. The CLI's whole job, minus printing."""
    config = get_config().scorecard
    conn = open_read_only(db_path)
    try:
        pending = status(conn, db_path).pending_migrations
        if pending:
            raise StoreError(
                f"database has pending migrations ({', '.join(pending)}):"
                " run `python -m aegis.cli.db init` to apply them",
                str(db_path),
            )
        inputs = gather(conn, since, until)
    finally:
        conn.close()
    errors: list[str] = []
    bars: dict[str, list[Bar]] = {}
    first = first_session_day(inputs)
    if first is not None:
        lookback = (now.astimezone(NEW_YORK).date() - first).days + BAR_FETCH_PADDING_DAYS
        bars, errors = fetch_bars((item.underlying for item in inputs), lookback, fetcher)
    card = build_scorecard(
        inputs,
        bars,
        config.horizons_days,
        now=now,
        neutral_delta=config.neutral_delta,
        group_by=group_by,
        since=since,
        until=until,
    )
    return card, errors


def main(argv: list[str] | None = None, *, fetcher: BarFetcher = _alpaca_daily_bars) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m aegis.cli.scorecard",
        description="Grade past proposals by what the underlying did afterwards (read-only).",
    )
    parser.add_argument("--since", type=_day, help="first day to include (YYYY-MM-DD, New York)")
    parser.add_argument("--until", type=_day, help="first day to leave out (YYYY-MM-DD, New York)")
    parser.add_argument(
        "--group-by", choices=tuple(GROUP_KEYS), default="proposal", help="what to compare"
    )
    parser.add_argument("--detail", action="store_true", help="also list every proposal")
    parser.add_argument("--json", action="store_true", help="print the scorecard as JSON")
    parser.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    args = parser.parse_args(argv)

    try:
        db_path = _db_path(args.db)
        if str(db_path) == MEMORY_DB or not db_path.exists():
            print(
                f"scorecard failed: database {db_path} does not exist"
                " (run `python -m aegis.cli.db init` to create it)",
                file=sys.stderr,
            )
            return 1
        card, errors = run(
            db_path=db_path,
            since=args.since,
            until=args.until,
            group_by=args.group_by,
            now=utcnow(),
            fetcher=fetcher,
        )
        if args.json:
            print(json.dumps({**card.model_dump(mode="json"), "price_errors": errors}, indent=2))
        else:
            _print_scorecard(card, args.detail, errors)
        return 0
    except (ConfigError, StoreError) as exc:
        print(f"scorecard failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(f"scorecard failed (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
