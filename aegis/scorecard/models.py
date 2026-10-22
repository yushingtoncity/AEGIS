"""Typed records the scorecard reads and writes (in memory only, never stored)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aegis.store.models import Instrument, OrderSide, ProposalLeg

Direction = Literal["bullish", "bearish", "neutral", "unknown"]
EntrySource = Literal["snapshot", "next_open"]
"""``snapshot``: the fresh spot the brain saw, moved onto the adjusted basis.
``next_open``: the open of the first session after a proposal made while the
market was shut (its snapshot spot is the last trade, older than the news)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LegQuote(_Frozen):
    """What the snapshot's chain said about one leg's contract."""

    bid: float | None = None
    ask: float | None = None
    implied_vol: float | None = None


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
    prompt_version: str | None = None
    """The proposal stage's prompt version: it says whether ``confidence``
    has its defined meaning (``aegis.brain.models.confidence_is_defined``)."""
    thesis_model: str | None = None
    model_source: str | None = None
    """``latest``, ``pinned`` or ``fallback`` from the cycle's ``model_selection``
    event for the proposal stage; None for cycles from before that event existed."""
    verdict: str | None = None
    legs: tuple[ProposalLeg, ...] = ()
    leg_deltas: dict[str, float] = {}
    snapshot_spot: float | None = None
    spot_time: datetime | None = None
    """When the snapshot spot traded; None when the snapshot had no timestamp."""
    spot_stale: bool = True
    """The snapshot's own staleness flag for the spot (market shut, no venue
    timestamp, or older than ``stale_after_seconds``). A stale spot is never
    the entry: news the brain read may postdate it."""
    leg_quotes: dict[str, LegQuote] = {}
    """Each leg's quote and implied vol from the same snapshot, by contract symbol."""
    risk_free_rate: float | None = None


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
    option_pnl: float | None = None
    """Options only: estimated dollar P&L of the whole structure if it had
    been bought at the snapshot's quotes and sold at this close. Each leg is
    marked by Black-Scholes at the stored implied vol (so theta and the
    underlying move count, an IV change does not), and the bid-ask spread is
    paid on the way in and again on the way out. An estimate, not a fill."""
    option_return: float | None = None
    """option_pnl over the net debit paid; None for a credit structure."""
    option_hit: bool | None = None
    """option_pnl > 0."""


class ScoredProposal(_Frozen):
    proposal_id: str
    created_at: datetime
    symbol: str
    underlying: str
    instrument: Instrument
    side: OrderSide
    confidence: float
    proposal_model: str
    prompt_version: str | None = None
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
    confidence_graded: int = 0
    """Graded calls whose confidence is scored: only at the
    ``CONFIDENCE_HORIZON_SESSIONS`` horizon, and only for proposals made
    under the defined meaning (``confidence_is_defined``). Zero elsewhere."""
    avg_confidence: float | None = None
    brier: float | None = None
    """Mean of (confidence - hit)^2 over the ``confidence_graded`` calls: 0 is perfect."""
    brier_reference: float | None = None
    """What always forecasting those calls' own hit rate would score,
    hit_rate * (1 - hit_rate). A Brier above it is worse than knowing nothing
    but the base rate; 0.25 is only the reference when the hit rate is 50%."""
    no_entry: int = 0
    """Proposals left ungraded because no clean entry price existed (a stale
    spot during the session, or no price at all)."""
    hit_rate_low: float | None = None
    hit_rate_high: float | None = None
    """95% Wilson interval for the hit rate."""
    avg_return_low: float | None = None
    avg_return_high: float | None = None
    """95% Student-t interval for the average signed return (needs 2 graded)."""
    long_hit_rate: float | None = None
    """Always-long baseline: the share of the same graded calls where the
    underlying simply rose. Beating it is the first test of skill over drift."""
    long_avg_return: float | None = None
    """Always-long baseline: the underlying's average return over the same calls."""
    shuffle_p: float | None = None
    """Direction-shuffled null: the chance that the same calls with their
    bullish/bearish labels shuffled earn an average signed return at least
    as high. 1.0 when every call points the same way (no direction to test)."""
    option_graded: int = 0
    option_hits: int = 0
    option_hit_rate: float | None = None
    """Share of option structures with a positive estimated P&L (see
    ``HorizonResult.option_pnl``); neutral structures count here too."""


class Scorecard(_Frozen):
    generated_at: datetime
    since: datetime | None = None
    until: datetime | None = None
    group_by: str
    horizons_days: tuple[int, ...]
    proposals: tuple[ScoredProposal, ...]
    summaries: tuple[GroupSummary, ...]
