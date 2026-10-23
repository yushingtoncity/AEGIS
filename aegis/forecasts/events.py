"""What each forecast was a probability of, and whether it happened. Pure.

A forecast made during the ``anchor_session`` session is about the window
from that session's close to the close of the session ``horizon`` trading
sessions later, on split- and dividend-adjusted daily closes:

- ``beats_benchmark``: the stock's return over the window is strictly
  greater than the benchmark's over the same window;
- ``up``: the benchmark's own return over the window is strictly greater
  than zero.

A tie counts as "no". Trading sessions are the dates the benchmark has a
daily bar for, so a gap in a stock's bars never stretches its window; a
session counts as ended at 16:00 New York
(``aegis.scorecard.grade.session_close``). An outcome is:

- ``graded`` with ``happened`` True or False;
- ``pending`` while the target session has not closed;
- ``ungradable`` with a reason: the forecast was written after the anchor
  session's close (it could have seen the starting price), or a bar the
  window needs is missing (a halt, a delisting, a data gap) once the
  benchmark shows the session has passed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from aegis.data.models import Bar
from aegis.scorecard.grade import NEW_YORK, SESSION_CLOSE
from aegis.store.models import ForecastEvent

OutcomeStatus = Literal["graded", "pending", "ungradable"]


@dataclass(frozen=True)
class Outcome:
    status: OutcomeStatus
    happened: bool | None = None
    target_session: date | None = None
    reason: str | None = None


def close_of(day: date) -> datetime:
    """The instant a session on ``day`` counts as closed: 16:00 New York."""
    return datetime.combine(day, SESSION_CLOSE, tzinfo=NEW_YORK)


def closes_by_day(bars: Sequence[Bar]) -> dict[date, float]:
    """Each bar's close keyed by its New York trading date."""
    return {bar.timestamp.astimezone(NEW_YORK).date(): bar.close for bar in bars}


def outcome(
    *,
    event: ForecastEvent,
    symbol_closes: Mapping[date, float],
    benchmark_closes: Mapping[date, float],
    anchor_session: date,
    horizon: int,
    completed_at: datetime,
    now: datetime,
) -> Outcome:
    """Whether one forecast's event happened (see the module docstring)."""
    if completed_at >= close_of(anchor_session):
        return Outcome("ungradable", reason="written after the anchor session's close")
    sessions = sorted(benchmark_closes)
    later = [day for day in sessions if day > anchor_session]
    if anchor_session not in benchmark_closes or anchor_session not in symbol_closes:
        if now < close_of(anchor_session) or not later:
            return Outcome("pending")
        return Outcome("ungradable", reason="no bar for the anchor session")
    if len(later) < horizon:
        return Outcome("pending")
    target = later[horizon - 1]
    if now < close_of(target):
        return Outcome("pending", target_session=target)
    bench_return = benchmark_closes[target] / benchmark_closes[anchor_session] - 1.0
    if event is ForecastEvent.UP:
        return Outcome("graded", happened=bench_return > 0, target_session=target)
    if target not in symbol_closes:
        return Outcome("ungradable", target_session=target, reason="no bar for the target session")
    stock_return = symbol_closes[target] / symbol_closes[anchor_session] - 1.0
    return Outcome("graded", happened=stock_return > bench_return, target_session=target)

__all__ = ["Outcome", "OutcomeStatus", "close_of", "closes_by_day", "outcome"]
