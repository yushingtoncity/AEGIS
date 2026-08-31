"""Market snapshot for one ticker: spot, ATM option chain, latest news.

Run:  python -m aegis.cli.snapshot AAPL [--expiration YYYY-MM-DD]

Data age is printed everywhere — free-plan options data runs ~15 minutes
behind, and when the market is closed everything shown is the latest
available (stale) data, labeled as such.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from aegis.config import ConfigError
from aegis.data.errors import DataError
from aegis.data.market import get_market_clock, get_option_chain, get_spot
from aegis.data.models import (
    ChainSnapshot,
    NewsItem,
    OptionSnapshot,
    OptionType,
    format_age,
    utcnow,
)
from aegis.data.news import get_news

STALE_AFTER_SECONDS = 15 * 60  # free-plan options data is ~15 minutes delayed
STRIKES_EACH_SIDE = 5


def _ts(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "unknown time"


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "-"


def _age_label(ts: datetime | None, market_open: bool) -> str:
    stale = (
        ts is None
        or not market_open
        or (utcnow() - ts).total_seconds() > STALE_AFTER_SECONDS
    )
    return f"as of {_ts(ts)}, age {format_age(ts)}" + (" [STALE]" if stale else "")


def _print_chain(chain: ChainSnapshot, market_open: bool) -> None:
    print()
    print(
        f"option chain {chain.underlying} exp {chain.expiration.isoformat()} "
        f"({len(chain.contracts)} contracts, {_age_label(chain.latest_quote_time, market_open)})"
    )
    strikes = chain.atm_strikes(STRIKES_EACH_SIDE)
    if not strikes:
        print("  no strikes available")
        return
    atm = chain.atm_strike()

    def _side(contract: OptionSnapshot | None) -> str:
        if contract is None:
            return f"{'-':>8} {'-':>7} {'-':>7} {'-':>8}"
        iv = f"{contract.implied_vol * 100:.1f}%" if contract.implied_vol is not None else "-"
        return (
            f"{_fmt(contract.mid):>8} {iv:>7} "
            f"{_fmt(contract.delta, '.3f'):>7} {_fmt(contract.theta, '.3f'):>8}"
        )

    header = f"{'mid':>8} {'IV':>7} {'delta':>7} {'theta':>8}"
    print(f"{'CALLS':^33} | {'strike':>8}  | {'PUTS':^33}")
    print(f"{header} | {'':>8}  | {header}")
    for strike in strikes:
        call = chain.contract_at(strike, OptionType.CALL)
        put = chain.contract_at(strike, OptionType.PUT)
        marker = "*" if atm is not None and abs(strike - atm) < 1e-6 else " "
        print(f"{_side(call)} | {strike:>8.2f}{marker} | {_side(put)}")
    if atm is not None:
        print(f"{'':>34}(* = ATM strike)")


def _print_news(items: list[NewsItem]) -> None:
    print()
    print("latest news")
    if not items:
        print("  (none)")
        return
    for item in items:
        ts = item.published_at.strftime("%Y-%m-%d %H:%M UTC") if item.published_at else "?"
        source = f"  ({item.source})" if item.source else ""
        print(f"  [{ts}] {item.headline}{source}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aegis.cli.snapshot",
        description="Print spot, the ATM option chain, and latest news for a ticker.",
    )
    parser.add_argument("symbol", help="ticker, e.g. SPY")
    parser.add_argument(
        "--expiration", help="chain expiration YYYY-MM-DD (default: nearest available)"
    )
    args = parser.parse_args(argv)
    symbol = args.symbol.strip().upper()
    expiration: date | None = None
    if args.expiration:
        try:
            expiration = date.fromisoformat(args.expiration)
        except ValueError:
            parser.error(f"--expiration must be YYYY-MM-DD, got {args.expiration!r}")

    try:
        failures = 0
        market_open = False
        clock_known = False
        try:
            clock = get_market_clock()
            market_open = clock.is_open
            clock_known = True
            status = (
                "OPEN"
                if market_open
                else f"CLOSED (next open {_ts(clock.next_open)})"
            )
        except (DataError, ConfigError) as exc:
            status = f"unknown ({exc})"

        print(f"{symbol} snapshot — {_ts(utcnow())}")
        print(f"market: {status}")
        if clock_known and not market_open:
            print("market is closed — data below is the latest available and is STALE")
        elif not clock_known:
            print("market status unknown — treating all data below as potentially STALE")
        print("-" * 78)

        try:
            quote = get_spot(symbol)
            print(
                f"spot {_fmt(quote.spot)}  "
                f"(bid {_fmt(quote.bid)} / ask {_fmt(quote.ask)}, "
                f"{_age_label(quote.spot_time, market_open)})"
            )
        except (DataError, ConfigError) as exc:
            print(f"spot unavailable: {exc}")
            failures += 1

        try:
            _print_chain(get_option_chain(symbol, expiration), market_open)
        except (DataError, ConfigError) as exc:
            print(f"\noption chain unavailable: {exc}")
            failures += 1

        try:
            _print_news(get_news(symbol, limit=5))
        except (DataError, ConfigError) as exc:
            print(f"\nnews unavailable: {exc}")
            failures += 1

        return 1 if failures else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(f"snapshot failed (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
