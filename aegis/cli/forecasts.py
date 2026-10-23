"""Make today's per-ticker forecasts, or grade the ones already made.

Run:  python -m aegis.cli.forecasts run [--db PATH] [--arm primary|shadow|all]
      python -m aegis.cli.forecasts score [--db PATH] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
                                          [--detail] [--json]

``run`` refuses while ``forecast.enabled`` is false. It reads the market
clock first and makes nothing unless the regular session is in progress (a
clock that cannot be read counts as closed), then reads the inputs once and
runs each configured arm on them: the primary model, then the shadow model.
An arm that already wrote a run for today's session is skipped, so a
repeated launch never doubles a day. Each run is journaled through the store
(``aegis.brain.forecast``) and its tokens count against
``brain.daily_token_budget``. Exit 0 when every arm asked wrote its run (or
had already), 2 when the budget guard halted one, 3 when the market was not
open, 1 on any failure: one line on stderr, never a traceback.

``score`` is read-only (``PRAGMA query_only``): it grades every run in
``[--since, --until)`` against split- and dividend-adjusted daily closes and
prints the pre-registered primary test, each arm's log loss and gain over a
coin flip with intervals, the no-skill baselines, the arm comparison and a
calibration table. Every cost printed is an ESTIMATE from the
``brain.prices_per_mtok`` placeholders, never a bill.

Nothing here reads or writes anything that can place an order.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

from aegis.brain.errors import BrainError
from aegis.brain.forecast import run_forecast_arm
from aegis.brain.llm import AnthropicLLM, LLMClient
from aegis.brain.models import ForecastInputs, ForecastRunResult
from aegis.cli.scorecard import BarFetcher, fetch_bars, open_read_only
from aegis.config import AegisConfig, ConfigError, get_config
from aegis.data.market import get_bars
from aegis.data.models import Bar, utcnow
from aegis.forecasts.inputs import ADJUSTED, MarketNotOpen, build_forecast_inputs
from aegis.forecasts.models import ForecastScorecard, GroupScore, Interval
from aegis.forecasts.score import build_forecast_scorecard
from aegis.scorecard.grade import NEW_YORK
from aegis.store.db import MEMORY_DB, open_store, resolve_db_path, status
from aegis.store.errors import StoreError
from aegis.store.forecasts import get_forecast_runs, get_run_forecasts, get_runs_for_session
from aegis.store.models import ForecastArm

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_HALTED = 2
EXIT_MARKET_NOT_OPEN = 3

CALENDAR_DAYS_PER_SESSION = 7 / 5  # weekdays only; holidays are covered by the padding below
BAR_FETCH_PADDING_DAYS = 30  # calendar days of slack before the baselines' first session

InputsBuilder = Callable[[AegisConfig, datetime], ForecastInputs]


def _db_path(arg: str | None) -> Path:
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)
    return Path(arg).expanduser().resolve()


def _day(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def _clean(text: str) -> str:
    """Model-written text on one line, control characters removed."""
    return "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in " ".join(text.split()))


def _alpaca_daily_bars(symbol: str, lookback_days: int, adjustment: str) -> list[Bar]:
    return get_bars(symbol, "1Day", lookback_days=lookback_days, adjustment=adjustment)


def _live_inputs(config: AegisConfig, now: datetime) -> ForecastInputs:
    return build_forecast_inputs(config, now=now)


# --- run ----------------------------------------------------------------------


def _arms(config: AegisConfig, choice: str) -> list[ForecastArm]:
    configured = [ForecastArm.PRIMARY] + ([ForecastArm.SHADOW] if config.forecast.shadow else [])
    if choice == "all":
        return configured
    arm = ForecastArm(choice)
    if arm not in configured:
        raise ConfigError(f"forecast.{choice} is not configured")
    return [arm]


def _print_run(result: ForecastRunResult) -> None:
    usage = result.usage
    tokens = f"{usage.tokens_in} in / {usage.tokens_out} out" if usage else "no tokens"
    cost = "n/a" if result.estimated_cost_usd is None else f"${result.estimated_cost_usd:.4f}"
    line = f"{result.arm}: {result.outcome.upper()} run {result.run_id}"
    if result.model_name:
        line += f" ({result.model_name})"
    print(line)
    print(f"  forecasts {result.forecasts}; tokens {tokens}; est. cost {cost} (ESTIMATE, not a bill)")
    if result.detail:
        print(f"  {_clean(result.detail)}")


def run(
    *,
    db_path: Path,
    arm_choice: str,
    llm: LLMClient | None,
    inputs_builder: InputsBuilder,
    now: datetime,
) -> int:
    config = get_config()
    if not config.forecast.enabled:
        print(
            "forecasts run refused: forecast.enabled is false in config.yaml (turn it on, on the mini)",
            file=sys.stderr,
        )
        return EXIT_FAILED
    arms = _arms(config, arm_choice)
    conn = open_store(db_path)
    try:
        session = now.astimezone(NEW_YORK).date()
        done = {run.arm for run in get_runs_for_session(conn, session)}
        todo = [arm for arm in arms if arm not in done]
        for arm in arms:
            if arm in done:
                print(f"{arm.value}: already ran for the {session} session; skipped")
        if not todo:
            return EXIT_OK
        try:
            inputs = inputs_builder(config, now)
        except MarketNotOpen as exc:
            print(f"forecasts run: nothing made: {exc}", file=sys.stderr)
            return EXIT_MARKET_NOT_OPEN
        client = llm if llm is not None else AnthropicLLM.from_config(config.brain)
        outcomes: list[str] = []
        for arm in todo:
            stage = config.forecast.primary if arm is ForecastArm.PRIMARY else config.forecast.shadow
            assert stage is not None  # _arms only lists configured arms
            result = run_forecast_arm(conn, client, inputs, arm, stage, config.brain)
            _print_run(result)
            outcomes.append(result.outcome)
            if result.outcome == "halted":
                break  # the budget is spent: the next arm would be refused too
    finally:
        conn.close()
    if "failed" in outcomes:
        return EXIT_FAILED
    if "halted" in outcomes:
        return EXIT_HALTED
    return EXIT_OK


# --- score --------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _range(interval: Interval | None, spec: str = "+.4f") -> str:
    return "n/a" if interval is None else f"[{interval.low:{spec}}, {interval.high:{spec}}]"


def _print_group(group: GroupScore) -> None:
    print(
        f"{group.arm} / {group.event} / {group.horizon} session(s)"
        f"  models {', '.join(group.models) or 'n/a'}"
    )
    print(
        f"  graded {group.graded} on {group.dates} dates; pending {group.pending};"
        f" ungradable {group.ungradable}; effective n {_num(group.effective_n, '.0f')}"
    )
    if not group.graded:
        return
    print(
        f"  log loss {_num(group.log_loss)} vs coin {group.coin_log_loss:.4f};"
        f" gain {_num(group.gain, '+.4f')} {_range(group.gain_interval)}"
    )
    print(
        f"  brier {_num(group.brier)} vs coin {group.coin_brier:.4f};"
        f" leaned right {group.lean_hits}/{group.leaning}"
        f" ({_pct(group.lean_hits / group.leaning if group.leaning else None)},"
        f" {_range(group.lean_hit_interval, '.1%')})"
    )
    for baseline in group.baselines:
        print(
            f"  vs {baseline.name}: its log loss {baseline.log_loss:.4f};"
            f" model gain {baseline.model_gain:+.4f} {_range(baseline.model_gain_interval)}"
        )
    if group.calibration:
        print("  calibration (forecast bin: n, mean forecast, happened, 95% CI)")
        for row in group.calibration:
            print(
                f"    {row.low:.2f}-{row.high:.2f}: {row.n:>4}  {row.mean_forecast:.3f}"
                f"  {row.frequency:.3f}  {_range(row.interval, '.3f')}"
            )


def _print_scorecard(card: ForecastScorecard, errors: Sequence[str]) -> None:
    window = f"{card.since or 'start'} to {card.until or 'now'}"
    print(f"Forecast scorecard, {window}: {card.runs} runs ({card.duplicate_runs} duplicate runs left out)")
    print("gain = ln 2 minus log loss per forecast: above 0 beats always saying 0.5.")
    test = card.primary
    print()
    print(
        f"PRIMARY TEST (pre-registered): {test.arm} / {test.event} / {test.horizon} session"
    )
    print(
        f"  graded {test.graded} on {test.dates} dates; gain {_num(test.gain, '+.4f')}"
        f" {_range(test.gain_interval)}; effective n {_num(test.effective_n, '.0f')}"
    )
    print(
        f"  p-value {_num(test.p_value)} against probabilities shuffled across tickers"
        f" within each date ({test.shuffles} shuffles)"
    )
    for group in card.groups:
        print()
        _print_group(group)
    if card.comparisons:
        print()
        print("PRIMARY vs SHADOW (same tickers and dates; above 0, primary did better)")
        for comparison in card.comparisons:
            print(
                f"  {comparison.event} / {comparison.horizon}: paired {comparison.paired};"
                f" {_num(comparison.primary_gain_over_shadow, '+.4f')} {_range(comparison.interval)}"
            )
    if card.ungradable_reasons:
        print()
        print("ungradable: " + "; ".join(f"{reason} ({n})" for reason, n in sorted(card.ungradable_reasons.items())))
    if errors:
        print()
        print("price errors (these tickers are pending or ungradable):")
        for error in errors:
            print(f"  {error}")


def score(
    *,
    db_path: Path,
    since: date | None,
    until: date | None,
    now: datetime,
    fetcher: BarFetcher,
    detail: bool,
) -> tuple[ForecastScorecard, list[str]]:
    config = get_config()
    conn = open_read_only(db_path)
    try:
        pending = status(conn, db_path).pending_migrations
        if pending:
            raise StoreError(
                f"database has pending migrations ({', '.join(pending)}):"
                " run `python -m aegis.cli.db init` to apply them",
                str(db_path),
            )
        runs = get_forecast_runs(conn, since, until)
        forecasts = get_run_forecasts(conn, [r.id for r in runs])
    finally:
        conn.close()
    benchmarks = {run.benchmark for run in runs} or {config.forecast.benchmark}
    if len(benchmarks) > 1:
        raise ConfigError(
            f"runs use different benchmarks ({', '.join(sorted(benchmarks))}): score them apart with --since/--until"
        )
    benchmark = benchmarks.pop()
    bars: dict = {}
    errors: list[str] = []
    if runs:
        first = min(run.anchor_session for run in runs)
        history = int(config.forecast.score.baseline_lookback_sessions * CALENDAR_DAYS_PER_SESSION)
        lookback = (now.astimezone(NEW_YORK).date() - first).days + history + BAR_FETCH_PADDING_DAYS
        symbols = sorted({symbol for run in runs for symbol in run.universe} | {benchmark})
        bars, errors = fetch_bars(symbols, lookback, fetcher, ADJUSTED)
    card = build_forecast_scorecard(
        runs,
        forecasts,
        bars,
        benchmark,
        now=now,
        config=config.forecast.score,
        since=since,
        until=until,
        detail=detail,
    )
    return card, errors


# --- main ---------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # one line and exit 1, never argparse's 2 (a budget halt here)
        print(f"forecasts: {message}", file=sys.stderr)
        raise SystemExit(EXIT_FAILED)


def main(
    argv: list[str] | None = None,
    *,
    llm: LLMClient | None = None,
    inputs_builder: InputsBuilder = _live_inputs,
    fetcher: BarFetcher = _alpaca_daily_bars,
    now: datetime | None = None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = _Parser(prog="python -m aegis.cli.forecasts", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)
    run_parser = commands.add_parser("run", help="make today's forecasts (refuses while forecast.enabled is false)")
    run_parser.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    run_parser.add_argument("--arm", choices=("primary", "shadow", "all"), default="all")
    score_parser = commands.add_parser("score", help="grade past forecasts (read-only)")
    score_parser.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    score_parser.add_argument("--since", type=_day, help="first anchor session to include (YYYY-MM-DD)")
    score_parser.add_argument("--until", type=_day, help="first anchor session to leave out (YYYY-MM-DD)")
    score_parser.add_argument("--detail", action="store_true", help="include every forecast (with --json)")
    score_parser.add_argument("--json", action="store_true", help="print the scorecard as JSON")
    args = parser.parse_args(argv)
    when = now or utcnow()
    try:
        db_path = _db_path(args.db)
        if args.command == "run":
            return run(db_path=db_path, arm_choice=args.arm, llm=llm, inputs_builder=inputs_builder, now=when)
        if str(db_path) == MEMORY_DB or not db_path.exists():
            print(
                f"forecasts score failed: database {db_path} does not exist"
                " (run `python -m aegis.cli.db init` to create it)",
                file=sys.stderr,
            )
            return EXIT_FAILED
        card, errors = score(
            db_path=db_path, since=args.since, until=args.until, now=when, fetcher=fetcher, detail=args.detail
        )
        if args.json:
            print(json.dumps({**card.model_dump(mode="json"), "price_errors": errors}, indent=2))
        else:
            _print_scorecard(card, errors)
        return EXIT_OK
    except (ConfigError, StoreError, BrainError, sqlite3.Error) as exc:
        print(f"forecasts {args.command} failed: {_clean(str(exc))}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(f"forecasts {args.command} failed (unexpected): {type(exc).__name__}: {_clean(str(exc))}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
