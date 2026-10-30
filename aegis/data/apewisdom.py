"""ApeWisdom: Reddit mention counts per ticker (free, community-run).

Documented at https://apewisdom.io/api/ (read 2026-10-30): ``GET
https://apewisdom.io/api/v1.0/filter/{filter}/page/{n}``, 100 results a
page, each with ``rank``, ``ticker``, ``name``, ``mentions``, ``upvotes``,
``rank_24h_ago`` and ``mentions_24h_ago``. The page names no key, no rate
limit and no terms, and offers no SLA, so the source is off by default and
fetched at most once per ``fetch_interval_minutes``.

It is an attention signal, not sentiment: a ticker's 24-hour mention count
and rank against the day before. Every signal is therefore unscored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from aegis.data.models import utcnow
from aegis.data.signals import Signal, SignalKind, SignalSource, SourceError, fetch_json

APEWISDOM_BASE = "https://apewisdom.io/api/v1.0/filter"
APEWISDOM_FILTER = "all-stocks"
"""Every stock subreddit ApeWisdom tracks, combined."""


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _headline(row: Mapping[str, Any], mentions: int) -> str:
    before = _int(row.get("mentions_24h_ago"))
    rank, rank_before = _int(row.get("rank")), _int(row.get("rank_24h_ago"))
    was = f" (a day earlier {before})" if before is not None else ""
    ranked = f", rank {rank}" if rank is not None else ""
    if rank is not None and rank_before is not None:
        ranked += f" (a day earlier {rank_before})"
    return f"{mentions} Reddit mentions in 24h{was}{ranked}"


def rows_to_signals(
    rows: Iterable[Mapping[str, Any]],
    tickers: Iterable[str],
    *,
    confidence: float,
    fetched_at: datetime,
    endpoint: str,
) -> list[Signal]:
    """The rows for ``tickers`` as unscored mention signals; a malformed row
    (no ticker, no mention count) is skipped."""
    wanted = {t.upper() for t in tickers}
    stamp = fetched_at.strftime("%Y%m%dT%H%M")
    signals: list[Signal] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        mentions = _int(row.get("mentions"))
        if ticker not in wanted or mentions is None or mentions < 0:
            continue
        signals.append(
            Signal(
                source=SignalSource.APEWISDOM,
                kind=SignalKind.MENTIONS,
                subject=ticker,
                source_item_id=f"{APEWISDOM_FILTER}:{ticker}:{stamp}",
                endpoint=endpoint,
                published_at=fetched_at,
                headline=_headline(row, mentions),
                engagement=float(mentions),
                confidence=confidence,
                raw=dict(row),
                fetched_at=fetched_at,
            )
        )
    return signals


def apewisdom_signals(
    tickers: Iterable[str],
    *,
    pages: int,
    confidence: float,
    timeout: float,
    session: Any = None,
) -> list[Signal]:
    """Up to ``pages`` pages of the all-stocks ranking, kept for ``tickers``."""
    wanted = [t.upper() for t in tickers]
    fetched_at = utcnow()
    rows: list[Mapping[str, Any]] = []
    for page in range(1, pages + 1):
        url = f"{APEWISDOM_BASE}/{APEWISDOM_FILTER}/page/{page}"
        body = fetch_json(url, timeout=timeout, session=session)
        if not isinstance(body, Mapping) or not isinstance(body.get("results"), list):
            raise SourceError("error", "apewisdom answered without a results list")
        rows.extend(body["results"])
        last_page = _int(body.get("pages"))
        if last_page is not None and page >= last_page:
            break
    endpoint = f"{APEWISDOM_BASE}/{APEWISDOM_FILTER}"
    return rows_to_signals(rows, wanted, confidence=confidence, fetched_at=fetched_at, endpoint=endpoint)
