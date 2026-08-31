"""Smoke test: config, env, Alpaca paper auth, and data client init.

Run:  python -m aegis.cli.check
"""

from __future__ import annotations

import os
import sys

from aegis.config import ConfigError, get_config, load_env, require_env
from aegis.data.account import get_account_state
from aegis.data.clients import news_client, option_client, stock_client
from aegis.data.errors import DataError
from aegis.data.models import format_age


def _money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "n/a"


def main() -> int:
    print("AEGIS preflight check")
    print("=" * 44)
    try:
        config = get_config()
        ttls = config.cache.ttl_seconds
        print(f"[ok] config.yaml  watchlist: {', '.join(config.watchlist)}")
        print(
            f"     cache TTLs (s): quotes={ttls.quotes:g} chains={ttls.chains:g} news={ttls.news:g}"
            f" | bars: {config.bars.timeframe}, {config.bars.lookback_days}d lookback"
        )

        load_env()
        for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
            require_env(name)
        print("[ok] .env         Alpaca keys present")
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            print("[--] .env         ANTHROPIC_API_KEY not set (not needed until Phase 4)")

        account = get_account_state()
        print(
            f"[ok] alpaca paper account {account.masked_account_number}"
            f" ({account.status}, fetched {format_age(account.fetched_at)} ago)"
        )
        print(
            f"     equity {_money(account.equity)} | buying power {_money(account.buying_power)}"
            f" | cash {_money(account.cash)} | open positions: {len(account.positions)}"
        )

        for label, factory in (
            ("stock data client", stock_client),
            ("options data client", option_client),
            ("news client", news_client),
        ):
            factory()
            print(f"[ok] {label} initialized")

        print("=" * 44)
        print("All checks passed.")
        return 0
    except (ConfigError, DataError) as exc:
        print(f"\nCHECK FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(f"\nCHECK FAILED (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
