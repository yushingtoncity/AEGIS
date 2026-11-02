"""One call per brain cycle: fetch or reuse, store, rank, digest.

For each enabled source and each of its subjects:

1. A successful fetch newer than ``fetch_interval_minutes`` is reused: its
   stored signals are read back and a ``cached`` row records the reuse. No
   request is made.
2. Otherwise the rate limiter is asked, the source is called, and the
   attempt is stored with every signal it brought (``ok``), or with the
   reason it failed (``rate_limited``, ``error``, ``disabled``).
3. A failed attempt falls back to the newest successful fetch within
   ``max_age_hours``, if any, so a flaky provider costs freshness, not the
   whole source; recency decay already discounts the older items. A WARNING
   ``signal_source_degraded`` event names the source, subject and reason.

Nothing a source does can fail the cycle: ``SourceError`` and every other
``Exception`` a source raises are recorded and skipped. A store failure
does propagate (``StoreError``); the cycle catches it, logs it and runs the
scan without signals.

Then every gathered signal is deduped, decayed and ranked (``rank``) and the
best fit into the digest (``digest``). The ranking clock is ``now`` when
given (tests, replays), else the wall clock after fetching.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from aegis.config import SignalsConfig
from aegis.data.models import utcnow
from aegis.data.signals import Signal, SourceError
from aegis.signals.digest import SignalsDigest, build_digest
from aegis.signals.rank import rank_signals
from aegis.signals.ratelimit import RateLimiter
from aegis.signals.source import Source, build_sources
from aegis.store.models import Event, EventLevel, SignalFetch, SignalFetchStatus, SignalRecord
from aegis.store.repo import log_event
from aegis.store.signals import get_fetch_signals, latest_ok_fetch, record_fetch

NEW_YORK = ZoneInfo("America/New_York")
"""The trading date ``alternate_days`` keys on is the exchange's, not UTC's."""
MAX_ERROR_CHARS = 300
"""An error stored on a fetch row is cut here; it is a reason, not a dump."""


class SourceStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    subject: str
    status: str
    detail: str | None = None
    items: int = 0


class CollectResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    digest: SignalsDigest
    statuses: tuple[SourceStatus, ...]
    considered: int


def signals_active(config: SignalsConfig, now: datetime) -> bool:
    """Whether this cycle gets signals: off when disabled; with
    ``alternate_days``, on only when the New York date's ordinal is even."""
    if not config.enabled:
        return False
    if config.assignment == "alternate_days":
        return now.astimezone(NEW_YORK).date().toordinal() % 2 == 0
    return True


def to_records(fetch_id: str, signals: Sequence[Signal]) -> list[SignalRecord]:
    """Signals as store rows of one fetch; a repeated provider id keeps its first copy."""
    seen: set[str] = set()
    records: list[SignalRecord] = []
    for s in signals:
        if s.source_item_id in seen:
            continue
        seen.add(s.source_item_id)
        records.append(
            SignalRecord(
                fetch_id=fetch_id,
                source=s.source.value,
                kind=s.kind.value,
                subject=s.subject,
                source_item_id=s.source_item_id,
                endpoint=s.endpoint,
                published_at=s.published_at,
                fetched_at=s.fetched_at,
                headline=s.headline,
                summary=s.summary,
                sentiment=s.sentiment,
                sentiment_raw=s.sentiment_raw,
                engagement=s.engagement,
                url=s.url,
                confidence=s.confidence,
                raw_json=s.raw,
            )
        )
    return records


def _degraded_event(cycle_id: str | None, status: SourceStatus, fallback: int) -> Event:
    used = f"; reusing {fallback} older items" if fallback else "; running without it"
    return Event(
        level=EventLevel.WARNING,
        kind="signal_source_degraded",
        message=f"signals: {status.source} for {status.subject} {status.status} ({status.detail}){used}",
        payload={
            "cycle_id": cycle_id,
            "source": status.source,
            "subject": status.subject,
            "status": status.status,
            "detail": status.detail,
            "fallback_items": fallback,
        },
    )


def _one_subject(
    conn: sqlite3.Connection,
    config: SignalsConfig,
    source: Source,
    limiter: RateLimiter,
    subject: str,
    cycle_id: str | None,
) -> tuple[list[SignalRecord], SourceStatus]:
    name = source.name.value
    started = utcnow()
    fresh = latest_ok_fetch(conn, name, subject, started - timedelta(minutes=config.fetch_interval_minutes))
    if fresh is not None:
        records = get_fetch_signals(conn, fresh.id)
        record_fetch(
            conn,
            SignalFetch(
                cycle_id=cycle_id, source=name, subject=subject, endpoint=source.endpoint,
                started_at=started, finished_at=utcnow(), status=SignalFetchStatus.CACHED,
                item_count=len(records),
            ),
        )
        return records, SourceStatus(source=name, subject=subject, status="cached", items=len(records))

    http_status: int | None = None
    try:
        limiter.acquire()
        signals = source.fetch(subject)
    except SourceError as exc:
        status, detail, http_status = exc.status, str(exc), exc.http_status
    except Exception as exc:  # a source bug or a provider schema change must not stop the cycle
        status, detail = "error", f"{type(exc).__name__}: {exc}"
    else:
        fetch = SignalFetch(
            cycle_id=cycle_id, source=name, subject=subject, endpoint=source.endpoint,
            started_at=started, finished_at=utcnow(), status=SignalFetchStatus.OK,
        )
        records = to_records(fetch.id, signals)
        record_fetch(conn, fetch.model_copy(update={"item_count": len(records)}), records)
        return records, SourceStatus(source=name, subject=subject, status="ok", items=len(records))

    detail = detail[:MAX_ERROR_CHARS]
    record_fetch(
        conn,
        SignalFetch(
            cycle_id=cycle_id, source=name, subject=subject, endpoint=source.endpoint,
            started_at=started, finished_at=utcnow(), status=SignalFetchStatus(status),
            http_status=http_status, error=detail,
        ),
    )
    older = latest_ok_fetch(conn, name, subject, started - timedelta(hours=config.max_age_hours))
    fallback = get_fetch_signals(conn, older.id) if older is not None else []
    result = SourceStatus(source=name, subject=subject, status=status, detail=detail, items=len(fallback))
    log_event(conn, _degraded_event(cycle_id, result, len(fallback)))
    return fallback, result


def collect_signals(
    conn: sqlite3.Connection,
    config: SignalsConfig,
    *,
    tickers: Sequence[str],
    cycle_id: str | None = None,
    sources: Sequence[Source] | None = None,
    now: datetime | None = None,
) -> CollectResult:
    """Gather, store and rank every enabled source's signals into the digest."""
    tickers = [t.upper() for t in (config.tickers or tickers)]
    sources = list(sources) if sources is not None else build_sources(config, tickers)
    records: list[SignalRecord] = []
    statuses: list[SourceStatus] = []
    for source in sources:
        limiter = RateLimiter(conn, source.name.value, getattr(config.sources, source.name.value))
        for subject in source.subjects(tickers, config.topics):
            got, status = _one_subject(conn, config, source, limiter, subject, cycle_id)
            records.extend(got)
            statuses.append(status)
    ranked = rank_signals(
        records,
        now if now is not None else utcnow(),
        half_life_hours=config.half_life_hours,
        max_age_hours=config.max_age_hours,
        min_weight=config.min_weight,
    )
    digest = build_digest(
        ranked,
        max_items=config.max_items,
        max_item_chars=config.max_item_chars,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    return CollectResult(digest=digest, statuses=tuple(statuses), considered=len(ranked))
