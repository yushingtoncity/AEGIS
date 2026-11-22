"""The bakeoff's own settings, read from ``bakeoff/bakeoff.yaml``.

Kept apart from ``config.yaml`` on purpose: the live cycle never reads this
file, and nothing here can change a production setting. The one thing the
bakeoff reads from ``config.yaml`` is the baseline contestant, so the
baseline is the thesis stage exactly as it is configured today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.brain.llm import model_price
from aegis.config import ConfigError, ModelPrice

BAKEOFF_DIR = Path(__file__).resolve().parent
DEFAULT_BAKEOFF_CONFIG = BAKEOFF_DIR / "bakeoff.yaml"

ThinkingLevel = Literal["low", "medium", "high"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetryConfig(_Strict):
    max_attempts: int = Field(default=5, ge=1)
    backoff_base_seconds: float = Field(default=2.0, gt=0)
    backoff_max_seconds: float = Field(default=60.0, gt=0)


class ContestantConfig(_Strict):
    """One model setup. ``anthropic`` contestants read their model from
    ``config.yaml`` (``from_brain_stage``); ``gemini`` ones name it here."""

    name: str = Field(min_length=1)
    provider: Literal["anthropic", "gemini"]
    from_brain_stage: Literal["thesis"] | None = None
    model: str | None = None
    thinking_level: ThinkingLevel | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _shape(self) -> ContestantConfig:
        if self.provider == "anthropic":
            if self.from_brain_stage is None:
                raise ValueError(
                    f"{self.name}: an anthropic contestant reads its model from config.yaml; "
                    "set from_brain_stage: thesis"
                )
            if self.model or self.thinking_level or self.max_output_tokens:
                raise ValueError(
                    f"{self.name}: model, thinking_level and max_output_tokens come from "
                    "config.yaml for an anthropic contestant; leave them out"
                )
        else:
            if not self.model or self.thinking_level is None or self.max_output_tokens is None:
                raise ValueError(
                    f"{self.name}: a gemini contestant needs model, thinking_level and "
                    "max_output_tokens"
                )
            if self.from_brain_stage is not None:
                raise ValueError(f"{self.name}: from_brain_stage is for anthropic contestants")
        return self


class PriceEntry(_Strict):
    """USD per million tokens; None until the operator fills it in."""

    input: float | None = Field(default=None, ge=0)
    output: float | None = Field(default=None, ge=0)


class BakeoffConfig(_Strict):
    spend_cap_usd: float = Field(default=5.0, gt=0)
    seed: int = 20261122
    blind_sample_size: int = Field(default=20, ge=1)
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_consecutive_errors: int = Field(default=3, ge=1)
    retry: RetryConfig = RetryConfig()
    contestants: tuple[ContestantConfig, ...]
    critic: ContestantConfig | None = None
    prices_per_mtok: dict[str, PriceEntry] = {}

    @field_validator("contestants")
    @classmethod
    def _contestants(cls, value: tuple[ContestantConfig, ...]) -> tuple[ContestantConfig, ...]:
        if not value:
            raise ValueError("at least one contestant is needed")
        names = [c.name for c in value]
        if len(set(names)) != len(names):
            raise ValueError(f"contestant names must be unique, got {names}")
        if value[0].provider != "anthropic" or value[0].from_brain_stage != "thesis":
            raise ValueError(
                "the first contestant is the baseline: provider anthropic, from_brain_stage thesis"
            )
        return value

    @property
    def baseline(self) -> ContestantConfig:
        return self.contestants[0]

    def price(self, model: str) -> ModelPrice | None:
        """The filled in price for ``model`` (a dated snapshot borrows its
        alias's entry, as in the live cost estimate); None when the entry is
        missing or either side is still null."""
        filled = {
            key: ModelPrice(input=entry.input, output=entry.output)
            for key, entry in self.prices_per_mtok.items()
            if entry.input is not None and entry.output is not None
        }
        return model_price(model, filled)


def load_bakeoff_config(path: str | Path | None = None) -> BakeoffConfig:
    """Read and validate ``bakeoff.yaml``; ``ConfigError`` with a one line reason."""
    config_path = Path(path) if path is not None else DEFAULT_BAKEOFF_CONFIG
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read bakeoff config {config_path}: {exc}") from exc
    try:
        return BakeoffConfig.model_validate(raw or {})
    except ValueError as exc:
        raise ConfigError(f"invalid bakeoff config {config_path}: {exc}") from exc
