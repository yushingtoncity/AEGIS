"""Typed records the scorecard reads and writes (in memory only, never stored)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aegis.store.models import Instrument, OrderSide, ProposalLeg

Direction = Literal["bullish", "bearish", "neutral", "unknown"]
EntrySource = Literal["snapshot", "prior_close"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProposalInputs(_Frozen):
    """Everything the store says about one proposal that grading needs.

    ``leg_deltas`` maps a leg's contract symbol to the per-share delta the
    cycle's snapshot recorded for it (missing when the snapshot had no
    Greeks for that contract). ``snapshot_spot`` is the underlying's spot in
    the same snapshot, the price the brain saw when it made the call.
    """

    proposal_id: str
    created_at: datetime
    cycle_id: str
    symbol: str
    underlying: str
    instrument: Instrument
    side: OrderSide
    quantity: float
    confidence: float
    proposal_model: str
    thesis_model: str | None = None
    model_source: str | None = None
    """``latest``, ``pinned`` or ``fallback`` from the cycle's ``model_selection``
    event for the proposal stage; None for cycles from before that event existed."""
    verdict: str | None = None
    legs: tuple[ProposalLeg, ...] = ()
    leg_deltas: dict[str, float] = {}
    snapshot_spot: float | None = None


class HorizonResult(_Frozen):
    """Where the underlying closed ``days`` sessions after the proposal."""

    days: int
    session: date | None = None
    close: float | None = None
    underlying_return: float | None = None
    """close / entry - 1, whatever the call was."""
    directional_return: float | None = None
    """underlying_return signed by the call: positive means the call was right."""
    hit: bool | None = None
    pending: bool = False


class ScoredProposal(_Frozen):
    proposal_id: str
    created_at: datetime
    symbol: str
    underlying: str
    instrument: Instrument
    side: OrderSide
    confidence: float
    proposal_model: str
    thesis_model: str | None = None
    model_source: str | None = None
    verdict: str | None = None
    direction: Direction
    direction_basis: str
    entry_price: float | None = None
    entry_source: EntrySource | None = None
    horizons: tuple[HorizonResult, ...] = ()
    note: str | None = None


class GroupSummary(_Frozen):
    """One group's record at one horizon. Rates and averages are None when
    nothing in the group was graded there yet."""

    group: str
    horizon_days: int
    proposals: int
    graded: int
    hits: int
    hit_rate: float | None = None
    avg_directional_return: float | None = None
    avg_confidence: float | None = None
    brier: float | None = None
    """Mean of (confidence - hit)^2 over graded calls: 0 is perfect, 0.25 is a coin flip at 50%."""


class Scorecard(_Frozen):
    generated_at: datetime
    since: datetime | None = None
    until: datetime | None = None
    group_by: str
    horizons_days: tuple[int, ...]
    proposals: tuple[ScoredProposal, ...]
    summaries: tuple[GroupSummary, ...]
