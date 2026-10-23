"""What the forecast scorer reports. Frozen pydantic models, JSON-ready."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from aegis.forecasts.events import OutcomeStatus


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ScoredForecast(_Frozen):
    """One stored forecast with its outcome and what each baseline said."""

    run_id: str
    arm: str
    model_name: str
    anchor_session: date
    symbol: str
    event: str
    horizon: int
    probability: float
    status: OutcomeStatus
    happened: bool | None = None
    target_session: date | None = None
    reason: str | None = None
    baselines: dict[str, float] = {}
    """Baseline name -> the probability that no-skill rule would have given."""


class Interval(_Frozen):
    low: float
    high: float


class BaselineScore(_Frozen):
    """The model against one baseline on the same graded forecasts."""

    name: str
    log_loss: float
    model_gain: float
    """Mean of (baseline log loss - model log loss): above 0, the model did better."""
    model_gain_interval: Interval | None = None


class CalibrationBin(_Frozen):
    """Forecasts in [low, high): how often the event happened against how often it was forecast."""

    low: float
    high: float
    n: int
    mean_forecast: float
    frequency: float
    interval: Interval | None = None


class GroupScore(_Frozen):
    """One arm on one event and horizon."""

    arm: str
    models: tuple[str, ...]
    event: str
    horizon: int
    graded: int
    pending: int
    ungradable: int
    dates: int
    log_loss: float | None = None
    coin_log_loss: float
    """ln 2: always saying 0.5."""
    gain: float | None = None
    """Mean of (ln 2 - model log loss) per forecast: above 0, better than a coin flip."""
    gain_interval: Interval | None = None
    effective_n: float | None = None
    brier: float | None = None
    coin_brier: float = 0.25
    lean_hits: int = 0
    leaning: int = 0
    """Forecasts not exactly 0.5, and how many leaned the way the event went."""
    lean_hit_interval: Interval | None = None
    baselines: tuple[BaselineScore, ...] = ()
    calibration: tuple[CalibrationBin, ...] = ()


class PrimaryTest(_Frozen):
    """The pre-registered test: do the primary arm's next-session relative
    forecasts carry information? Gain over 0.5 against the same
    probabilities shuffled across tickers within each date."""

    arm: str
    event: str
    horizon: int
    graded: int
    dates: int
    gain: float | None = None
    gain_interval: Interval | None = None
    p_value: float | None = None
    shuffles: int
    effective_n: float | None = None


class ArmComparison(_Frozen):
    """Primary against shadow on the same tickers, dates, events and horizons."""

    event: str
    horizon: int
    paired: int
    primary_gain_over_shadow: float | None = None
    """Mean of (shadow log loss - primary log loss): above 0, the primary arm did better."""
    interval: Interval | None = None


class ForecastScorecard(_Frozen):
    generated_at: datetime
    since: date | None = None
    until: date | None = None
    runs: int
    duplicate_runs: int
    """Runs left out because an earlier run of the same arm covered the same session."""
    primary: PrimaryTest
    groups: tuple[GroupScore, ...]
    comparisons: tuple[ArmComparison, ...]
    ungradable_reasons: dict[str, int]
    forecasts: tuple[ScoredForecast, ...] = ()
