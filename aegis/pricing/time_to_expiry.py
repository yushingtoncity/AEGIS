"""Time to expiry in years, under a calendar-day convention.

The convention: CALENDAR days (not trading days) over a ``day_count_basis``
of 365, with fractional days kept, so 0-3 DTE contracts are not rounded to
zero and a Friday-afternoon contract still has a few hours of time value.
The expiry instant is the exchange close on the expiration date —
``expiry_time`` in ``expiry_timezone`` (16:00 America/New_York by default,
which follows US daylight saving automatically via zoneinfo).

Everything is configurable through the keyword parameters. config.yaml's
``pricing`` block carries the operator's values (day_count_basis,
expiry_time, expiry_timezone); this package never reads config — callers
pass them in. ``expiry_time`` is "HH:MM" and ``expiry_timezone`` an IANA
name, both validated by config; malformed values raise the stdlib
ValueError / ZoneInfoNotFoundError. A non-positive ``day_count_basis``
raises ``PricingError`` (config validates it too, but direct callers can
pass anything).

``now`` defaults to the current UTC time; a naive ``now`` is treated as UTC.
Results are clamped at zero once the expiry instant has passed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from aegis.pricing.errors import PricingError

DEFAULT_EXPIRY_TIME = "16:00"
DEFAULT_EXPIRY_TIMEZONE = "America/New_York"
DEFAULT_DAY_COUNT_BASIS = 365

_SECONDS_PER_DAY = 86400


def expiry_instant(
    expiration: date,
    *,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> datetime:
    """Aware UTC datetime of the exchange close on the expiration day."""
    close = datetime.combine(
        expiration, time.fromisoformat(expiry_time), tzinfo=ZoneInfo(expiry_timezone)
    )
    return close.astimezone(timezone.utc)


def _seconds_to_expiry(
    expiration: date, now: datetime | None, expiry_time: str, expiry_timezone: str
) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    close = expiry_instant(
        expiration, expiry_time=expiry_time, expiry_timezone=expiry_timezone
    )
    return max(0.0, (close - now).total_seconds())


def time_to_expiry(
    expiration: date,
    now: datetime | None = None,
    *,
    day_count_basis: int = DEFAULT_DAY_COUNT_BASIS,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> float:
    """Years until the expiry instant: seconds / (86400 * day_count_basis), clamped at 0."""
    if day_count_basis <= 0:
        raise PricingError(
            "time to expiry",
            cause=ValueError(f"day_count_basis must be positive, got {day_count_basis!r}"),
        )
    seconds = _seconds_to_expiry(expiration, now, expiry_time, expiry_timezone)
    return seconds / (_SECONDS_PER_DAY * day_count_basis)


def calendar_days_to_expiry(
    expiration: date,
    now: datetime | None = None,
    *,
    expiry_time: str = DEFAULT_EXPIRY_TIME,
    expiry_timezone: str = DEFAULT_EXPIRY_TIMEZONE,
) -> float:
    """Fractional calendar days until the expiry instant, clamped at 0."""
    return _seconds_to_expiry(expiration, now, expiry_time, expiry_timezone) / _SECONDS_PER_DAY
