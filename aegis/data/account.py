"""Account reads: the paper account's state and open positions."""

from __future__ import annotations

from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from aegis.config import get_config
from aegis.data.cache import default_cache
from aegis.data.clients import trading_client
from aegis.data.errors import DataError
from aegis.data.models import AccountState, as_mapping, utcnow


def get_account_state() -> AccountState:
    """Account number/status/equity/buying power plus open positions, typed
    and cached (quotes TTL)."""
    ttl = get_config().cache.ttl_seconds.quotes

    def fetch() -> AccountState:
        try:
            account = trading_client().get_account()
            positions = trading_client().get_all_positions()
        except (APIError, RequestException) as exc:
            raise DataError("account state", None, exc) from exc
        return AccountState.from_alpaca(
            as_mapping(account),
            [dict(as_mapping(p)) for p in positions],
            fetched_at=utcnow(),
        )

    return default_cache.get_or_fetch(("account",), ttl, fetch)
