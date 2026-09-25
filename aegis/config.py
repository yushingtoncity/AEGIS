"""Load and validate config.yaml into typed pydantic settings.

Tunables live in config.yaml at the repo root. Secrets live in .env
(gitignored) and are loaded into the process environment by ``load_env``;
they are read via ``require_env`` and never stored on the config object.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

_TIMEFRAME_RE = re.compile(r"^(\d+)(Min|Hour|Day|Week|Month)$")

# Per-unit amount bounds, mirroring alpaca-py's TimeFrame.validate_timeframe —
# anything looser validates here but fails on every bar fetch.
_TIMEFRAME_AMOUNTS = {
    "Min": range(1, 60),
    "Hour": range(1, 24),
    "Day": (1,),
    "Week": (1,),
    "Month": (1, 2, 3, 6, 12),
}


class ConfigError(Exception):
    """Raised when config.yaml or the environment is missing or invalid."""


class CacheTTLs(BaseModel):
    """Per-category cache lifetimes, in seconds."""

    model_config = ConfigDict(extra="forbid")

    quotes: float = Field(default=5, gt=0)
    chains: float = Field(default=30, gt=0)
    news: float = Field(default=60, gt=0)


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ttl_seconds: CacheTTLs = CacheTTLs()


class BarsConfig(BaseModel):
    """Defaults for historical bar fetches."""

    model_config = ConfigDict(extra="forbid")

    timeframe: str = "1Day"
    lookback_days: int = Field(default=30, gt=0)

    @field_validator("timeframe")
    @classmethod
    def _valid_timeframe(cls, value: str) -> str:
        match = _TIMEFRAME_RE.match(value)
        if not match or int(match.group(1)) not in _TIMEFRAME_AMOUNTS[match.group(2)]:
            raise ValueError(
                f"invalid timeframe {value!r}: expected <amount><Min|Hour|Day|Week|Month> "
                "with amount in Min 1-59, Hour 1-23, Day/Week 1, Month 1/2/3/6/12 "
                "(e.g. 15Min or 1Day)"
            )
        return value


class PricingConfig(BaseModel):
    """Inputs to the Phase 2 pricing engine that a single quote cannot supply.

    ``risk_free_rate`` is a placeholder until a later phase sources it from a
    Treasury-yield feed; ``day_count_basis`` fixes the calendar-day
    time-to-expiry convention; ``expiry_time``/``expiry_timezone`` pin the
    instant a contract expires (the exchange close on expiration day).
    """

    model_config = ConfigDict(extra="forbid")

    risk_free_rate: float = Field(default=0.04, ge=-0.05, le=0.5)
    day_count_basis: int = Field(default=365, gt=0)
    contract_multiplier: int = Field(default=100, gt=0)
    expiry_time: str = "16:00"
    expiry_timezone: str = "America/New_York"

    @field_validator("expiry_time")
    @classmethod
    def _valid_clock_time(cls, value: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError(f"invalid expiry_time {value!r}: expected HH:MM (24h)")
        return value

    @field_validator("expiry_timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown expiry_timezone {value!r}") from exc
        return value


class StoreConfig(BaseModel):
    """Where the Phase 3 SQLite journal lives (relative to the repo root)."""

    model_config = ConfigDict(extra="forbid")

    db_path: str = "data/aegis.db"

    @field_validator("db_path")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("db_path must not be blank")
        return value.strip()


_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class BrainStageConfig(BaseModel):
    """Model, output cap and effort for one brain stage.

    ``effort`` is the current API's depth control (thinking is always on for
    the 5.x models and counts toward ``max_tokens``); ``None`` omits it,
    which Haiku 4.5 requires. There is no ``temperature``: the Messages API
    no longer accepts sampling parameters for these models.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = Field(gt=0)
    effort: str | None = None

    @field_validator("model")
    @classmethod
    def _non_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value.strip()

    @field_validator("effort")
    @classmethod
    def _valid_effort(cls, value: str | None) -> str | None:
        if value is not None and value not in _EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {_EFFORT_LEVELS} or null, got {value!r}")
        return value


class BrainStages(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan: BrainStageConfig = BrainStageConfig(
        model="claude-haiku-4-5-20251001", max_tokens=3000
    )
    thesis: BrainStageConfig = BrainStageConfig(
        model="claude-opus-5-5", max_tokens=4000, effort="medium"
    )
    proposal: BrainStageConfig = BrainStageConfig(
        model="claude-fable-5-1", max_tokens=3000, effort="low"
    )


class ModelPrice(BaseModel):
    """USD per million tokens — placeholders used only for cost estimates."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class SnapshotConfig(BaseModel):
    """How much market context the scan stage is shown per symbol."""

    model_config = ConfigDict(extra="forbid")

    strikes_each_side: int = Field(default=5, ge=0)
    headlines_per_symbol: int = Field(default=5, ge=0, le=50)
    stale_after_minutes: float = Field(default=15, gt=0)
    recent_proposals_limit: int = Field(default=10, ge=0)


class BrainConfig(BaseModel):
    """The Phase 4 agent brain: per-stage models, token budgets, price placeholders."""

    model_config = ConfigDict(extra="forbid")

    cadence_minutes: int = Field(default=30, gt=0)
    max_retries: int = Field(default=3, ge=0)
    per_cycle_token_cap: int = Field(default=60_000, gt=0)
    daily_token_budget: int = Field(default=500_000, gt=0)
    stages: BrainStages = BrainStages()
    prices_per_mtok: dict[str, ModelPrice] = {}
    snapshot: SnapshotConfig = SnapshotConfig()

    @model_validator(mode="after")
    def _daily_covers_a_cycle(self) -> "BrainConfig":
        if self.daily_token_budget < self.per_cycle_token_cap:
            raise ValueError(
                "daily_token_budget must be at least per_cycle_token_cap "
                f"({self.daily_token_budget} < {self.per_cycle_token_cap})"
            )
        return self

    def stage(self, name: str) -> BrainStageConfig:
        """The stage block by name ('scan', 'thesis', 'proposal')."""
        try:
            return getattr(self.stages, name)
        except AttributeError as exc:
            raise KeyError(f"unknown brain stage {name!r}") from exc


class RiskLimits(BaseModel):
    """Risk limits enforced by the Phase 5 policy engine.

    Placeholders for now — nothing reads them yet. They live in config from
    day one so limits are versioned configuration, not code.
    """

    model_config = ConfigDict(extra="forbid")

    max_position_pct: float = Field(default=5.0, gt=0, le=100)
    max_open_positions: int = Field(default=5, ge=0)
    daily_loss_limit_pct: float = Field(default=2.0, gt=0, le=100)
    no_trade_list: list[str] = []

    @field_validator("no_trade_list")
    @classmethod
    def _upper_symbols(cls, value: list[str]) -> list[str]:
        return [s.strip().upper() for s in value if s.strip()]


class AegisConfig(BaseModel):
    """The validated contents of config.yaml."""

    model_config = ConfigDict(extra="forbid")

    watchlist: list[str] = Field(min_length=1)
    cache: CacheConfig = CacheConfig()
    bars: BarsConfig = BarsConfig()
    pricing: PricingConfig = PricingConfig()
    store: StoreConfig = StoreConfig()
    brain: BrainConfig = BrainConfig()
    risk_limits: RiskLimits = RiskLimits()

    @field_validator("watchlist")
    @classmethod
    def _clean_watchlist(cls, value: list[str]) -> list[str]:
        cleaned = [s.strip().upper() for s in value if s.strip()]
        if not cleaned:
            raise ValueError("watchlist must contain at least one symbol")
        return cleaned


def load_config(path: str | Path | None = None) -> AegisConfig:
    """Read and validate a config file, raising ConfigError with a clean message."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {config_path}: {exc}") from exc
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {config_path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {config_path} must be a YAML mapping")
    try:
        return AegisConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"config file {config_path} failed validation:\n{exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AegisConfig:
    """The process-wide config singleton, loaded from the repo-root config.yaml."""
    return load_config()


def load_env() -> None:
    """Load .env from the repo root into the process environment (idempotent)."""
    load_dotenv(REPO_ROOT / ".env")


def require_env(name: str) -> str:
    """Return a required environment variable or raise ConfigError with guidance."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"missing required environment variable {name}. "
            "Copy .env.example to .env and fill in your keys."
        )
    return value
