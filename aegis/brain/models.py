"""Typed inputs and outputs of the three brain stages, and the cycle result.

Everything the stages exchange is one of these frozen models. The *input*
models (``MarketSnapshot`` and friends) are built from ``aegis.data`` and
``aegis.pricing`` by ``snapshot.py`` and rendered into prompts; the *output*
models (``ScanBrief``, ``ThesisOutput``, ``ProposalOutput``) are what the
model must emit as JSON and are validated with pydantic — a JSON schema
derived from them constrains decoding on the API side, and the pydantic
constraints that the schema cannot express (numeric bounds, cross-field
rules) are checked locally, with failures fed back to the model.

Two invariants the models enforce for the governance rules:

- Every ``TradeProposal`` and every ``ThesisCandidate`` carries a
  falsifiable ``invalidation`` condition (non-blank).
- A cycle's outcome is either exactly one ``TradeProposal`` or a
  ``no_trade`` with a reason. Declining is a first-class result.

Headline text is untrusted data (it is wrapped in delimiters in prompts);
nothing here interprets it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.data.models import OptionType
from aegis.pricing.models import Greeks, PositionSummary
from aegis.store.models import Instrument, OrderSide, OrderType, ReasoningStage

# --- shared -------------------------------------------------------------------


class Frozen(BaseModel):
    # extra="forbid": the API's constrained decoding already rejects unknown
    # keys, and local validation must agree — a misspelled field in model
    # output (or a canned fixture) is an error fed back, never silently
    # dropped from what gets stored as raw_model_output.
    model_config = ConfigDict(frozen=True, extra="forbid")


def _non_blank(value: str, what: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{what} must not be blank")
    return text


# --- market snapshot: input to the scan stage --------------------------------


class HeadlineItem(Frozen):
    """One news item. ``headline``/``summary`` are UNTRUSTED text."""

    headline: str
    summary: str | None = None
    source: str | None = None
    published_at: datetime | None = None
    age_seconds: float | None = None
    symbols: tuple[str, ...] = ()


class ContractSummary(Frozen):
    """One option contract as the scan stage sees it: quote plus IV/Greeks
    with their source (vendor or our model) — see ``EnrichedOption``."""

    symbol: str
    option_type: OptionType
    strike: float
    expiration: date
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    last: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    implied_vol: float | None = None
    iv_source: str | None = None
    greeks: Greeks | None = None
    greeks_source: str | None = None
    quote_age_seconds: float | None = None


class ChainSummary(Frozen):
    """The nearest-expiry chain, ATM ± N strikes, calls and puts."""

    underlying: str
    expiration: date
    days_to_expiry: float | None = None
    spot: float | None = None
    atm_strike: float | None = None
    contracts: tuple[ContractSummary, ...] = ()
    quote_age_seconds: float | None = None
    stale: bool = False


class SymbolSnapshot(Frozen):
    """Everything gathered for one watchlist symbol; fetch failures are
    recorded in ``errors`` rather than aborting the snapshot."""

    symbol: str
    spot: float | None = None
    bid: float | None = None
    ask: float | None = None
    spot_time: datetime | None = None
    spot_age_seconds: float | None = None
    stale: bool = False
    last_close: float | None = None
    change_5d_pct: float | None = None
    high_20d: float | None = None
    low_20d: float | None = None
    chain: ChainSummary | None = None
    headlines: tuple[HeadlineItem, ...] = ()
    errors: tuple[str, ...] = ()


class OpenPosition(Frozen):
    symbol: str
    qty: float
    side: str
    avg_entry_price: float | None = None
    market_value: float | None = None
    unrealized_pl: float | None = None
    current_price: float | None = None


class AccountSummary(Frozen):
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    positions: tuple[OpenPosition, ...] = ()
    fetched_at: datetime | None = None


class DataAge(Frozen):
    """Staleness facts the brief must carry regardless of what the model says."""

    market_open: bool | None = None
    """None when the market clock could not be fetched."""
    next_open: datetime | None = None
    next_close: datetime | None = None
    stale_after_seconds: float
    any_stale: bool = False
    warnings: tuple[str, ...] = ()
    """Human-readable staleness warnings, e.g. 'market is CLOSED — all quotes are the last available'."""


class MarketSnapshot(Frozen):
    """The scan stage's input: the whole watchlist plus account and data-age facts."""

    taken_at: datetime
    symbols: tuple[SymbolSnapshot, ...]
    account: AccountSummary | None = None
    data_age: DataAge
    risk_free_rate: float


# --- scan stage output --------------------------------------------------------


class SymbolBrief(Frozen):
    symbol: str
    summary: str
    iv_observation: str | None = None
    skew_observation: str | None = None
    catalysts: tuple[str, ...] = ()
    stale_data: bool = False


class ScanBrief(Frozen):
    """What the scan stage concluded. ``staleness_warnings`` always includes the
    deterministic ``DataAge.warnings`` — the scan stage merges them in, so a
    closed market or a delayed feed is never silently dropped."""

    symbols: tuple[SymbolBrief, ...]
    notable_observations: tuple[str, ...] = ()
    news_catalysts: tuple[str, ...] = ()
    staleness_warnings: tuple[str, ...] = ()


# --- thesis stage -------------------------------------------------------------


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class OptionStructure(str, Enum):
    """Structures the proposal stage knows how to resolve into legs.

    Strikes are listed low → high by the thesis stage: one for the single-leg
    structures, two for verticals, four for an iron condor.
    """

    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    CALL_DEBIT_SPREAD = "call_debit_spread"
    PUT_DEBIT_SPREAD = "put_debit_spread"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    PUT_CREDIT_SPREAD = "put_credit_spread"
    IRON_CONDOR = "iron_condor"


STRIKES_PER_STRUCTURE: dict[OptionStructure, int] = {
    OptionStructure.LONG_CALL: 1,
    OptionStructure.LONG_PUT: 1,
    OptionStructure.CALL_DEBIT_SPREAD: 2,
    OptionStructure.PUT_DEBIT_SPREAD: 2,
    OptionStructure.CALL_CREDIT_SPREAD: 2,
    OptionStructure.PUT_CREDIT_SPREAD: 2,
    OptionStructure.IRON_CONDOR: 4,
}


class ThesisCandidate(Frozen):
    """One trade idea. For options, ``structure``/``expiration``/``strikes``
    must name real contracts from the chain the scan stage was shown."""

    symbol: str
    direction: Direction
    instrument: Instrument
    structure: OptionStructure | None = None
    expiration: date | None = None
    strikes: tuple[float, ...] = ()
    rationale: str
    confidence: float = Field(ge=0, le=1)
    key_risk: str
    invalidation: str

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _non_blank(value, "symbol").upper()

    @field_validator("rationale", "key_risk", "invalidation")
    @classmethod
    def _text(cls, value: str, info) -> str:  # noqa: ANN001 - pydantic passes ValidationInfo
        return _non_blank(value, info.field_name)

    @model_validator(mode="after")
    def _option_fields(self) -> "ThesisCandidate":
        if self.instrument is Instrument.OPTION:
            if self.structure is None or self.expiration is None:
                raise ValueError("an option candidate needs structure and expiration")
            want = STRIKES_PER_STRUCTURE[self.structure]
            if len(self.strikes) != want:
                raise ValueError(
                    f"{self.structure.value} needs exactly {want} strike(s), got {len(self.strikes)}"
                )
            if list(self.strikes) != sorted(self.strikes) or len(set(self.strikes)) != want:
                raise ValueError("strikes must be distinct and listed low to high")
        elif self.structure is not None or self.strikes:
            raise ValueError("an equity candidate must not carry an option structure or strikes")
        return self


class ThesisOutput(Frozen):
    """Zero or more candidates, best first. Empty is a respectable answer."""

    candidates: tuple[ThesisCandidate, ...] = ()
    no_idea_reason: str | None = None

    @model_validator(mode="after")
    def _reason_when_empty(self) -> "ThesisOutput":
        if not self.candidates and not (self.no_idea_reason or "").strip():
            raise ValueError("no_idea_reason is required when there are no candidates")
        return self


# --- proposal stage -----------------------------------------------------------


class LegQuote(Frozen):
    """A resolved leg of the offered structure with its live quote."""

    contract: ContractSummary
    side: OrderSide
    quantity: int = Field(gt=0)


class InstrumentOffer(Frozen):
    """What the proposal stage is allowed to trade: the candidate resolved to
    concrete contracts (or the equity quote), priced by the pricing engine."""

    candidate: ThesisCandidate
    spot: float | None = None
    bid: float | None = None
    ask: float | None = None
    quote_age_seconds: float | None = None
    stale: bool = False
    legs: tuple[LegQuote, ...] = ()
    """Empty for equities."""
    pricing: PositionSummary | None = None
    """``analyze_position`` over the legs (dollar terms); None for equities or
    when a leg lacks a price."""
    net_mid: float | None = None
    """Per-share net premium at mid: positive = debit, negative = credit."""


class LegSpec(Frozen):
    """One leg as the model specifies it; persisted as ``aegis.store.models.ProposalLeg``."""

    symbol: str
    option_type: OptionType
    side: OrderSide
    quantity: int = Field(gt=0)
    strike: float = Field(gt=0)
    expiration: date

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _non_blank(value, "symbol").upper()


class TradeProposal(Frozen):
    """The brain's only trade-shaped output — a proposal, never an order.

    ``symbol`` is the ticker for equities, the OCC symbol for a single-leg
    option, and the underlying for a multi-leg structure (whose legs are
    listed). ``side`` is the net direction: buy = pay a debit / buy shares,
    sell = collect a credit / sell shares. ``quantity`` is shares or
    contracts per leg unit. ``limit_price`` is per share (net per spread
    for multi-leg) and required for limit orders.
    """

    symbol: str
    instrument: Instrument
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    limit_price: float | None = None
    thesis: str
    confidence: float = Field(ge=0, le=1)
    invalidation: str
    legs: tuple[LegSpec, ...] = ()

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _non_blank(value, "symbol").upper()

    @field_validator("thesis", "invalidation")
    @classmethod
    def _text(cls, value: str, info) -> str:  # noqa: ANN001
        return _non_blank(value, info.field_name)

    @model_validator(mode="after")
    def _consistent(self) -> "TradeProposal":
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("a limit order needs limit_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive")
        if self.instrument is Instrument.EQUITY and self.legs:
            raise ValueError("an equity proposal has no option legs")
        if self.instrument is Instrument.OPTION and not self.legs:
            raise ValueError("an option proposal must list its legs")
        return self


class ProposalOutput(Frozen):
    """Exactly one of: a proposal, or a no-trade with a reason."""

    outcome: Literal["trade", "no_trade"]
    proposal: TradeProposal | None = None
    no_trade_reason: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ProposalOutput":
        if self.outcome == "trade":
            if self.proposal is None:
                raise ValueError("outcome 'trade' requires a proposal")
        else:
            if not (self.no_trade_reason or "").strip():
                raise ValueError("outcome 'no_trade' requires no_trade_reason")
            if self.proposal is not None:
                raise ValueError("outcome 'no_trade' must not carry a proposal")
        return self


# --- usage and results --------------------------------------------------------


class StageUsage(Frozen):
    """Tokens, latency and attempts for one stage (summed over retries).

    ``model`` is the model the stage REQUESTED (config ``brain.stages``; the
    key for price estimates). ``response_model`` is the model the API says
    ANSWERED the last attempt; the reasoning row records it, falling back to
    ``model`` when no response arrived.
    """

    stage: ReasoningStage
    model: str
    response_model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


class ScanResult(Frozen):
    brief: ScanBrief
    usage: StageUsage
    raw_output: str
    prompt_version: str


class ThesisResult(Frozen):
    output: ThesisOutput
    usage: StageUsage
    raw_output: str
    prompt_version: str


class ProposalResult(Frozen):
    output: ProposalOutput
    offer: InstrumentOffer
    usage: StageUsage
    raw_output: str
    prompt_version: str


CycleOutcome = Literal["proposal", "no_trade", "halted", "failed"]


class CycleResult(Frozen):
    """What one brain cycle did, for the CLI and the scheduler."""

    cycle_id: str
    started_at: datetime
    finished_at: datetime
    outcome: CycleOutcome
    proposal_id: str | None = None
    proposal: TradeProposal | None = None
    no_trade_reason: str | None = None
    halt_reason: str | None = None
    brief: ScanBrief | None = None
    candidates: tuple[ThesisCandidate, ...] = ()
    usage: tuple[StageUsage, ...] = ()
    estimated_cost_usd: float | None = None
    """From config prices_per_mtok — an ESTIMATE; None if a model has no price."""

    @property
    def total_tokens_in(self) -> int:
        return sum(u.tokens_in for u in self.usage)

    @property
    def total_tokens_out(self) -> int:
        return sum(u.tokens_out for u in self.usage)
