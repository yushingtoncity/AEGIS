"""Shared builders for the forecast tests: inputs, model output, daily bars,
trading sessions and stored runs. Fixtures only, never a live API."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone

from aegis.brain.models import ForecastInputs, ForecastTicker
from aegis.config import AegisConfig, ForecastConfig, get_config
from aegis.data.models import Bar
from aegis.scorecard.grade import NEW_YORK
from aegis.store.models import (
    FORECAST_EVENT_VERSION,
    ForecastArm,
    ForecastEvent,
    ForecastRecord,
    ForecastRun,
)

ANCHOR = date(2026, 10, 21)  # a Wednesday
NOW = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)  # 14:00 New York, mid-session
STOCKS = ("AAPL", "MSFT")
BENCHMARK = "SPY"


def forecast_config(**overrides) -> AegisConfig:
    """The repo's config with the forecast block turned on (or as overridden)."""
    fields = {"enabled": True, "universe": STOCKS, **overrides}
    return get_config().model_copy(update={"forecast": ForecastConfig(**fields)})


def make_inputs(stocks: Sequence[str] = STOCKS, benchmark: str = BENCHMARK, **ticker_fields) -> ForecastInputs:
    tickers = [ForecastTicker(symbol=s, role="stock", price=100.0, **ticker_fields) for s in stocks]
    tickers.append(ForecastTicker(symbol=benchmark, role="benchmark", price=500.0))
    return ForecastInputs(taken_at=NOW, anchor_session=ANCHOR, benchmark=benchmark, tickers=tuple(tickers))


def output_json(symbols: Sequence[str], p1: float = 0.55, p5: float = 0.5, **overrides) -> str:
    """A ForecastOutput as the model would write it; ``overrides`` maps a
    symbol to a replacement entry (or None to leave it out)."""
    entries = []
    for symbol in symbols:
        entry = {"symbol": symbol, "p_next_session": p1, "p_five_sessions": p5, "rationale": "no view"}
        if symbol in overrides:
            if overrides[symbol] is None:
                continue
            entry = {**entry, **overrides[symbol]}
        entries.append(entry)
    return json.dumps({"forecasts": entries})


def sessions(start: date, count: int) -> list[date]:
    """``count`` weekdays from ``start`` (inclusive), oldest first."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def bar(symbol: str, day: date, close: float, high: float | None = None, low: float | None = None) -> Bar:
    """A daily bar stamped the way Alpaca stamps them: midnight New York."""
    stamp = datetime.combine(day, time(0, 0), tzinfo=NEW_YORK).astimezone(timezone.utc)
    return Bar(
        symbol=symbol,
        timestamp=stamp,
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=1000,
    )


def bars_from(symbol: str, closes: Mapping[date, float]) -> list[Bar]:
    return [bar(symbol, day, close) for day, close in sorted(closes.items())]


def make_run(
    run_id: str = "run-1",
    *,
    arm: ForecastArm = ForecastArm.PRIMARY,
    anchor: date = ANCHOR,
    completed_at: datetime | None = None,
    universe: Sequence[str] = STOCKS,
    model_name: str = "model-a",
) -> ForecastRun:
    done = completed_at or datetime.combine(anchor, time(15, 0), tzinfo=NEW_YORK).astimezone(timezone.utc)
    return ForecastRun(
        id=run_id,
        arm=arm,
        inputs_taken_at=done - timedelta(minutes=1),
        completed_at=done,
        anchor_session=anchor,
        benchmark=BENCHMARK,
        model_name=model_name,
        requested_model=model_name,
        prompt_version="forecast-v1+system-v1",
        event_version=FORECAST_EVENT_VERSION,
        universe=tuple(universe),
        inputs={},
    )


def record(
    run_id: str,
    symbol: str,
    probability: float,
    *,
    event: ForecastEvent = ForecastEvent.BEATS_BENCHMARK,
    horizon: int = 1,
) -> ForecastRecord:
    return ForecastRecord(
        run_id=run_id,
        symbol=symbol,
        event=event,
        horizon_sessions=horizon,
        probability=probability,
        rationale="r",
    )
