"""Dedupe, recency decay and ranking: pure functions over stored signals.

Ranking score, for a signal ``age`` hours old:

    recency = 0.5 ** (age / half_life_hours)
    weight  = confidence * recency
    score   = weight * (1 + log10(1 + engagement))

``weight`` is what fades: a signal whose weight is under ``min_weight``, or
that is older than ``max_age_hours``, is dropped. Engagement only orders the
survivors, compressed by the log so one heavily discussed ticker cannot
drown every headline. A signal published after ``now`` is never shown, so a replayed
cycle sees only what existed at its time (point-in-time). One with no
publish time is aged from when it was fetched.

Two signals are duplicates when they share a canonical URL or a normalized
title, across sources and subjects. The one with more engagement is kept
(ties: the earlier published, then the lower id, so the choice is stable);
the kept one lists every subject the group was about and the ids it absorbed.
A mention count is a snapshot, so only the newest per source and ticker is kept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from aegis.store.models import SignalRecord

_SECONDS_PER_HOUR = 3600.0
_TRACKING_PARAM = re.compile(r"^(utm_.*|ref|ref_src|s|t|fbclid|gclid|mc_cid|mc_eid)$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


class RankedSignal(BaseModel):
    """One signal that survived dedupe and decay, with its weights."""

    model_config = ConfigDict(frozen=True)

    record: SignalRecord
    subjects: tuple[str, ...]
    age_hours: float
    weight: float
    score: float
    duplicate_ids: tuple[str, ...] = ()


def canonical_url(url: str | None) -> str | None:
    """Lower-case scheme and host, no ``www.``, no fragment, no tracking
    parameters, no trailing slash; None for no URL."""
    if not url or not url.strip():
        return None
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAM.match(k)))
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))


def title_key(text: str) -> str:
    """Lower-cased words of a headline joined by spaces; punctuation and
    spacing never make two copies of one headline look different."""
    return " ".join(_WORD.findall(text.casefold()))


def _published(record: SignalRecord) -> datetime:
    return record.published_at or record.fetched_at


def age_hours(record: SignalRecord, now: datetime) -> float:
    return max(0.0, (now - _published(record)).total_seconds() / _SECONDS_PER_HOUR)


def recency(age: float, half_life_hours: float) -> float:
    """0.5 at one half-life, 0.25 at two."""
    return 0.5 ** (age / half_life_hours)


def score(weight: float, engagement: float) -> float:
    return weight * (1.0 + math.log10(1.0 + max(0.0, engagement)))


def _keys(record: SignalRecord) -> list[str]:
    if record.kind == "mentions":
        return [f"mentions:{record.source}:{record.subject}"]
    keys = []
    url = canonical_url(record.url)
    if url:
        keys.append(f"url:{url}")
    title = title_key(record.headline)
    if title:
        keys.append(f"title:{title}")
    return keys or [f"id:{record.id}"]


def _preference(record: SignalRecord) -> tuple[float, float, str]:
    # More engagement first; then the newer snapshot of a mention count or the
    # earlier copy of an article; then the id, so the order never depends on input order.
    when = _published(record).timestamp()
    return (-record.engagement, -when if record.kind == "mentions" else when, record.id)


def dedupe(records: Iterable[SignalRecord]) -> list[tuple[SignalRecord, tuple[str, ...], tuple[str, ...]]]:
    """Group duplicates and keep one per group: ``(kept, subjects, absorbed ids)``.

    Groups are joined transitively (A shares a URL with B, B a title with C:
    one group). The result is in the kept records' preference order.
    """
    items = sorted({r.id: r for r in records}.values(), key=_preference)
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for index, record in enumerate(items):
        for key in _keys(record):
            if key in owner:
                a, b = find(owner[key]), find(index)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)
    result = []
    for root in sorted(groups):
        members = [items[i] for i in groups[root]]
        kept = members[0]  # the root is the group's most preferred member
        subjects = tuple(dict.fromkeys(m.subject for m in members))
        result.append((kept, subjects, tuple(m.id for m in members[1:])))
    return result


def rank_signals(
    records: Sequence[SignalRecord],
    now: datetime,
    *,
    half_life_hours: float,
    max_age_hours: float,
    min_weight: float,
) -> list[RankedSignal]:
    """Drop what is from the future, too old or too faded; dedupe; rank by
    score, best first (ties: newer first, then id)."""
    visible = [r for r in records if _published(r) <= now]
    ranked: list[RankedSignal] = []
    for record, subjects, absorbed in dedupe(visible):
        age = age_hours(record, now)
        weight = record.confidence * recency(age, half_life_hours)
        if age > max_age_hours or weight < min_weight:
            continue
        ranked.append(
            RankedSignal(
                record=record,
                subjects=subjects,
                age_hours=age,
                weight=weight,
                score=score(weight, record.engagement),
                duplicate_ids=absorbed,
            )
        )
    ranked.sort(key=lambda s: (-s.score, s.age_hours, s.record.id))
    return ranked
