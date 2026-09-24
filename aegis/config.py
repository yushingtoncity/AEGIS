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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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
