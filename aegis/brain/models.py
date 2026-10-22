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

import re
from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.data.models import OptionType
from aegis.pricing.models import Greeks, PositionSummary
from aegis.store.models import (
    PROBABILITY_CEILING,
    PROBABILITY_FLOOR,
    Instrument,
    OrderSide,
    OrderType,
    ReasoningStage,
)

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
    """The chain of the nearest expiration in the DTE window (``risk_limits.min_dte``
    to ``brain.snapshot.max_dte``), ATM ± N strikes, calls and puts."""

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


CONFIDENCE_HORIZON_SESSIONS = 5
"""What a bullish or bearish candidate's (and proposal's) ``confidence`` is
a probability of, from thesis-v2 and proposal-v2 on: that the underlying
closes above (bullish) or below (bearish) its price at proposal time on the
close this many trading sessions later. A neutral structure's confidence
keeps its older, undefined meaning. Fixed, not tunable: changing it would
change the meaning of every stored confidence. The proposal scorecard grades
confidence at this horizon only, and ``risk_limits.min_confidence`` reads it
unchanged."""

CONFIDENCE_DEFINED_FROM = 2
"""The first proposal prompt version (``proposal-v2``) that asked for that
probability. Confidence from an earlier proposal named no event, so the
scorecard never scores it."""

_PROPOSAL_PROMPT = re.compile(r"proposal-v(\d+)\b")


def confidence_is_defined(prompt_version: str | None) -> bool:
    """Whether a stored proposal's confidence carries the defined meaning,
    from the ``prompt_version`` it was written under."""
    match = _PROPOSAL_PROMPT.match(prompt_version or "")
    return match is not None and int(match.group(1)) >= CONFIDENCE_DEFINED_FROM


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


# --- forecast stage -----------------------------------------------------------


ForecastRole = Literal["stock", "benchmark"]


class ForecastTicker(Frozen):
    """What the forecast stage is shown about one ticker.

    Every price-derived figure comes from split- and dividend-adjusted daily
    closes of COMPLETED sessions (``last_close_session`` is the latest), so
    a partial bar for today never enters them. ``price`` is the latest IEX
    trade or quote as of ``price_time`` and ``today_pct`` its move from that
    last close. Percent fields are in percent units (1.5 means 1.5%);
    ``rel_*`` are the stock's return minus the benchmark's over the same
    sessions. ``headlines`` are UNTRUSTED text, published before the run's
    ``taken_at``.
    """

    symbol: str
    role: ForecastRole
    price: float | None = None
    price_time: datetime | None = None
    price_age_seconds: float | None = None
    stale: bool = False
    last_close: float | None = None
    last_close_session: date | None = None
    today_pct: float | None = None
    rel_today_pct: float | None = None
    return_1d_pct: float | None = None
    return_5d_pct: float | None = None
    return_20d_pct: float | None = None
    rel_5d_pct: float | None = None
    rel_20d_pct: float | None = None
    range_20d_position: float | None = None
    """Where ``last_close`` sits in the 20-session low-high range: 0 at the low, 1 at the high."""
    volatility_20d_pct: float | None = None
    """Standard deviation of the last 20 daily close-to-close returns, in percent."""
    headlines: tuple[HeadlineItem, ...] = ()
    errors: tuple[str, ...] = ()


class ForecastInputs(Frozen):
    """Everything one forecast run is shown, taken once and shared by both arms.

    ``anchor_session`` is the trading date in progress when the inputs were
    taken: its close starts the clock of every forecast in the run.
    ``tickers`` lists the universe in config order, then the benchmark.
    """

    taken_at: datetime
    anchor_session: date
    benchmark: str
    tickers: tuple[ForecastTicker, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_benchmark_last(self) -> "ForecastInputs":
        roles = [ticker.role for ticker in self.tickers]
        if roles.count("benchmark") != 1 or self.tickers[-1].role != "benchmark":
            raise ValueError("forecast inputs need exactly one benchmark ticker, listed last")
        if self.tickers[-1].symbol != self.benchmark:
            raise ValueError("the benchmark ticker must be the configured benchmark")
        symbols = [ticker.symbol for ticker in self.tickers]
        if len(set(symbols)) != len(symbols):
            raise ValueError("forecast inputs must not repeat a ticker")
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every ticker asked, universe first, benchmark last."""
        return tuple(ticker.symbol for ticker in self.tickers)


class TickerForecast(Frozen):
    """The model's two probabilities for one ticker (see the forecast prompt
    for the events), each in [0.05, 0.95], and one short line of why."""

    symbol: str
    p_next_session: float = Field(ge=PROBABILITY_FLOOR, le=PROBABILITY_CEILING)
    p_five_sessions: float = Field(ge=PROBABILITY_FLOOR, le=PROBABILITY_CEILING)
    rationale: str

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return _non_blank(value, "symbol").upper()

    @field_validator("rationale")
    @classmethod
    def _text(cls, value: str) -> str:
        return _non_blank(value, "rationale")


class ForecastOutput(Frozen):
    """One ``TickerForecast`` per ticker asked; ``forecast.py`` checks that
    the set matches the inputs exactly."""

    forecasts: tuple[TickerForecast, ...] = Field(min_length=1)


class ForecastRunResult(Frozen):
    """What one arm's run did: its run id, outcome and spend."""

    run_id: str
    arm: str
    outcome: Literal["written", "halted", "failed"]
    model_name: str | None = None
    forecasts: int = 0
    usage: StageUsage | None = None
    estimated_cost_usd: float | None = None
    detail: str | None = None
