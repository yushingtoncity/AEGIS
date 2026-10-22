"""The forecast stage: ``ForecastInputs`` in, one probability per ticker per event out.

A measuring instrument beside the trading cycle, never part of it. Once a
trading day each arm (the pinned primary model, and the cheaper shadow model
on the same inputs) is asked, for every universe ticker, the probability
that its adjusted close-to-close return beats the benchmark's over the next
session and over five sessions, from the close of the session in progress;
the benchmark itself is asked whether it closes up. The run is written to
``forecast_runs`` and ``forecasts`` through ``aegis.store.forecasts``. It
never writes a proposal, never sees the scan brief, the thesis or recent
proposals, and nothing that can place an order reads what it writes.

Each arm is its own run: its own id (used as the budget guard's cycle id,
so its tokens count against ``brain.daily_token_budget``), its own
``reasoning`` row (stage ``forecast``) and its own events:
``forecast_start``, then ``forecast_end`` with the outcome, and a
``brain_error`` when it failed. A run is written whole or not at all; spend
an error cut short is journaled exactly as the trading cycle does
(``journal_unrecorded_spend``).

The model's output is checked beyond its schema: every ticker asked appears
exactly once, and nothing else does. A failure is fed back for a retry like
any other stage.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime

from aegis.brain.cycle import brain_error_event, journal_unrecorded_spend, reasoning_row
from aegis.brain.errors import BrainError, BudgetExceeded
from aegis.brain.llm import BudgetGuard, GuardedLLM, LLMClient, estimate_cost
from aegis.brain.models import (
    ForecastInputs,
    ForecastOutput,
    ForecastRunResult,
    StageUsage,
)
from aegis.brain.prompts import build_forecast_prompt, neutralise_untrusted, prompt_version, system_prompt
from aegis.brain.schemas import OutputInvalid
from aegis.brain.stage import run_json_stage
from aegis.config import BrainConfig, BrainStageConfig
from aegis.data.models import utcnow
from aegis.store.forecasts import insert_forecast_run
from aegis.store.models import (
    FORECAST_EVENT_VERSION,
    FORECAST_HORIZONS,
    Event,
    EventLevel,
    ForecastArm,
    ForecastEvent,
    ForecastRecord,
    ForecastRun,
    ReasoningStage,
    new_id,
)
from aegis.store.repo import add_reasoning, log_event

Clock = Callable[[], datetime]


def check_coverage(output: ForecastOutput, inputs: ForecastInputs) -> None:
    """``OutputInvalid`` unless the output forecasts every ticker asked exactly once."""
    asked = inputs.symbols
    given = [forecast.symbol for forecast in output.forecasts]
    repeated = sorted({symbol for symbol in given if given.count(symbol) > 1})
    missing = [symbol for symbol in asked if symbol not in given]
    extra = sorted({symbol for symbol in given if symbol not in asked})
    problems: list[str] = []
    if missing:
        problems.append(f"missing tickers: {', '.join(missing)}")
    if extra:
        problems.append(f"tickers not asked: {', '.join(neutralise_untrusted(s) for s in extra)}")
    if repeated:
        problems.append(f"tickers given more than once: {', '.join(neutralise_untrusted(s) for s in repeated)}")
    if problems:
        raise OutputInvalid("; ".join(problems) + ". Give exactly one entry per ticker asked.")


def forecast_records(
    run_id: str, inputs: ForecastInputs, output: ForecastOutput
) -> list[ForecastRecord]:
    """The rows for a validated output: two per ticker (next session and five
    sessions), the stocks on ``beats_benchmark`` and the benchmark on ``up``,
    in the order the tickers were asked."""
    by_symbol = {forecast.symbol: forecast for forecast in output.forecasts}
    records: list[ForecastRecord] = []
    for ticker in inputs.tickers:
        forecast = by_symbol[ticker.symbol]
        event = ForecastEvent.UP if ticker.role == "benchmark" else ForecastEvent.BEATS_BENCHMARK
        probabilities = (forecast.p_next_session, forecast.p_five_sessions)
        for horizon, probability in zip(FORECAST_HORIZONS, probabilities, strict=True):
            records.append(
                ForecastRecord(
                    run_id=run_id,
                    symbol=ticker.symbol,
                    event=event,
                    horizon_sessions=horizon,
                    probability=probability,
                    rationale=forecast.rationale,
                )
            )
    return records


def run_forecast_stage(
    llm: LLMClient,
    inputs: ForecastInputs,
    stage_config: BrainStageConfig,
    *,
    max_retries: int,
    run_id: str | None = None,
) -> tuple[ForecastOutput, StageUsage, str]:
    """One validated ``ForecastOutput`` (with its usage and raw text) from
    ``stage_config``'s model, retrying invalid output ``max_retries`` times."""
    return run_json_stage(
        llm,
        stage=ReasoningStage.FORECAST,
        stage_config=stage_config,
        system=system_prompt(),
        user_prompt=build_forecast_prompt(inputs),
        output_model=ForecastOutput,
        max_retries=max_retries,
        cycle_id=run_id,
        post_validate=lambda output: check_coverage(output, inputs),
    )


def _event(kind: str, message: str, payload: dict[str, object], level: EventLevel = EventLevel.INFO) -> Event:
    return Event(level=level, kind=kind, message=message, payload=payload)


def _inputs_payload(inputs: ForecastInputs) -> dict[str, object]:
    """What the model saw, as stored on the run (headline text included: it
    is part of the record of what the forecast rested on)."""
    return inputs.model_dump(mode="json")


def run_forecast_arm(
    conn: sqlite3.Connection,
    llm: LLMClient,
    inputs: ForecastInputs,
    arm: ForecastArm,
    stage_config: BrainStageConfig,
    brain: BrainConfig,
    *,
    clock: Clock = utcnow,
) -> ForecastRunResult:
    """Run one arm end to end and journal it. A budget halt or a
    ``BrainError`` is an outcome (``halted`` / ``failed``); anything else
    (the store failing, an interrupt) is journaled and re-raised."""
    run_id = new_id()
    guard = BudgetGuard(conn, brain, run_id)
    guarded = GuardedLLM(llm, guard)
    base = {"run_id": run_id, "arm": arm.value, "model": stage_config.model}
    log_event(
        conn,
        _event(
            "forecast_start",
            f"forecast run {run_id} ({arm.value}, {stage_config.model}) started",
            {**base, "anchor_session": inputs.anchor_session.isoformat(), "tickers": len(inputs.tickers)},
        ),
    )
    try:
        output, usage, raw = run_forecast_stage(
            guarded, inputs, stage_config, max_retries=brain.max_retries, run_id=run_id
        )
    except BudgetExceeded as exc:
        pending = journal_unrecorded_spend(
            conn, guard, run_id, ReasoningStage.FORECAST, stage_config.model, (), exc
        )
        return _finish(conn, base, "halted", str(exc), pending, brain, level=EventLevel.WARNING)
    except BaseException as exc:
        pending = journal_unrecorded_spend(
            conn, guard, run_id, ReasoningStage.FORECAST, stage_config.model, (), exc
        )
        _best_effort_log(conn, brain_error_event(run_id, ReasoningStage.FORECAST, exc))
        if not isinstance(exc, BrainError):
            _best_effort_log(
                conn,
                _event("forecast_end", f"forecast run {run_id} failed", {**base, "outcome": "failed"}, EventLevel.ERROR),
            )
            raise
        return _finish(conn, base, "failed", str(exc), pending, brain, level=EventLevel.ERROR)

    add_reasoning(conn, reasoning_row(run_id, usage, raw))
    records = forecast_records(run_id, inputs, output)
    answered = usage.response_model or usage.model
    try:
        _write_run(conn, run_id, arm, inputs, usage, answered, records, clock)
    except BaseException as exc:
        _best_effort_log(conn, brain_error_event(run_id, ReasoningStage.FORECAST, exc))
        _best_effort_log(
            conn,
            _event("forecast_end", f"forecast run {run_id} failed", {**base, "outcome": "failed"}, EventLevel.ERROR),
        )
        raise
    return _finish(
        conn, base, "written", None, usage, brain, model_name=answered, forecasts=len(records)
    )


def _write_run(
    conn: sqlite3.Connection,
    run_id: str,
    arm: ForecastArm,
    inputs: ForecastInputs,
    usage: StageUsage,
    answered: str,
    records: list[ForecastRecord],
    clock: Clock,
) -> None:
    insert_forecast_run(
        conn,
        ForecastRun(
            id=run_id,
            arm=arm,
            inputs_taken_at=inputs.taken_at,
            completed_at=clock(),
            anchor_session=inputs.anchor_session,
            benchmark=inputs.benchmark,
            model_name=answered,
            requested_model=usage.model,
            prompt_version=prompt_version(ReasoningStage.FORECAST),
            event_version=FORECAST_EVENT_VERSION,
            universe=inputs.symbols,
            inputs=_inputs_payload(inputs),
        ),
        records,
    )


def _best_effort_log(conn: sqlite3.Connection, event: Event) -> None:
    try:
        log_event(conn, event)
    except Exception:  # never mask the error being handled
        pass


def _finish(
    conn: sqlite3.Connection,
    base: dict[str, object],
    outcome: str,
    detail: str | None,
    usage: StageUsage | None,
    brain: BrainConfig,
    *,
    level: EventLevel = EventLevel.INFO,
    model_name: str | None = None,
    forecasts: int = 0,
) -> ForecastRunResult:
    cost = estimate_cost([usage], brain.prices_per_mtok) if usage is not None else None
    payload = {
        **base,
        "outcome": outcome,
        "forecasts": forecasts,
        "tokens_in": usage.tokens_in if usage else 0,
        "tokens_out": usage.tokens_out if usage else 0,
        "estimated_cost_usd": cost,
        "detail": detail,
    }
    message = f"forecast run {base['run_id']} {outcome}" + (f": {detail}" if detail else "")
    _best_effort_log(conn, _event("forecast_end", message, payload, level))
    return ForecastRunResult(
        run_id=str(base["run_id"]),
        arm=str(base["arm"]),
        outcome=outcome,  # type: ignore[arg-type]
        model_name=model_name,
        forecasts=forecasts,
        usage=usage,
        estimated_cost_usd=cost,
        detail=detail,
    )
