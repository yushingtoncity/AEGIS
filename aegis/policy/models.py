"""Typed inputs and outputs of the policy engine.

The engine is a pure function of two frozen values: the proposal under
review (the store's proposal row and its legs) and a ``PolicyContext``
(everything else a verdict may depend on — the limits, the account, the
clock, the quotes, the control flags, and the instant ``now``). Nothing in
a rule reads a wall clock, the network, the store or the config: if a
verdict depends on it, it is a field of one of these two models. That is
what makes "the same proposal and the same context always produce the same
verdict" true by construction, and what lets tests build a context by hand.

Unknown is never good news. A context field that could not be fetched is
``None`` (with the reason in ``errors``), a non-finite number counts as
unknown, and every rule that needs an unknown value REJECTs — the engine
fails closed.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegis.config import RiskLimits
from aegis.data.models import MarketClock, OptionSnapshot, Position, Quote, parse_occ_symbol
from aegis.pricing.models import PositionSummary
from aegis.store.models import (
    Controls,
    Instrument,
    Order,
    Proposal,
    ProposalLeg,
    Verdict,
)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def underlying_of(symbol: str) -> str:
    """The ticker a symbol trades on: the root of an OCC option symbol
    (``SPY260821C00640000`` → ``SPY``), else the symbol itself, upper-cased.

    The root is stripped, so the padded 21-character OCC form
    (``SPY   260821C00640000``) is SPY too — an unstripped ``'SPY   '`` would
    match nothing on the no-trade list, the watchlist or the books.
    """
    text = symbol.strip().upper()
    try:
        return parse_occ_symbol(text)[0].strip()
    except ValueError:
        return text


# --- the proposal -------------------------------------------------------------


class ProposalUnderReview(Frozen):
    """What the engine judges: a stored proposal and its option legs.

    The store keeps legs in ``proposal_legs`` (none for an equity), so this
    pairs the two rows the way ``repo.get_proposal`` and
    ``repo.get_proposal_legs`` return them. It does not judge the shape: an
    option proposal without legs, or legs on two underlyings, is a valid
    *input* that the rules reject. Only legs that belong to a different
    proposal are refused here — that is a caller's bug, not a proposal.
    """

    proposal: Proposal
    legs: tuple[ProposalLeg, ...] = ()

    @model_validator(mode="after")
    def _legs_belong(self) -> "ProposalUnderReview":
        for leg in self.legs:
            if leg.proposal_id != self.proposal.id:
                raise ValueError(
                    f"leg {leg.leg_index} belongs to proposal {leg.proposal_id}, "
                    f"not {self.proposal.id}"
                )
        return self

    @property
    def id(self) -> str:
        return self.proposal.id

    @property
    def is_option(self) -> bool:
        return self.proposal.instrument is Instrument.OPTION

    @property
    def is_equity(self) -> bool:
        return self.proposal.instrument is Instrument.EQUITY

    @property
    def underlying(self) -> str:
        """The ticker exposure is measured on: the first leg's root for an
        option with legs, else the root of the proposal's own symbol."""
        if self.is_option and self.legs:
            return underlying_of(self.legs[0].symbol)
        return underlying_of(self.proposal.symbol)

    @property
    def underlyings(self) -> tuple[str, ...]:
        """Every distinct ticker the proposal names — its symbol and each
        leg — sorted. More than one means the structure is inconsistent."""
        names = {underlying_of(self.proposal.symbol)}
        names.update(underlying_of(leg.symbol) for leg in self.legs)
        return tuple(sorted(names))

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every instrument symbol the proposal would trade: its own and each leg's, sorted."""
        return tuple(sorted({self.proposal.symbol, *(leg.symbol for leg in self.legs)}))


# --- the context --------------------------------------------------------------


class PolicyContext(Frozen):
    """Everything a verdict may depend on besides the proposal.

    Built by ``context.build_context`` from ``aegis.data`` and
    ``aegis.store``; tests construct it directly. ``now`` is the only clock
    a rule may consult, and ``market_date`` the only "today" (the calendar
    date at ``now`` in the exchange's timezone — what DTE and the trading
    day are counted from).

    ``next_open_date`` is the calendar date of ``clock.next_open`` in the same
    timezone (None without one): equal to ``market_date`` it means the market
    is closed BEFORE today's session, which is how ``daily_loss_limit`` tells
    a trip that must cover the coming session from one after the close.

    ``None`` means unknown, never zero: ``positions=None`` is "could not be
    fetched", ``positions=()`` is "holds nothing". ``errors`` says why each
    unknown is unknown. ``quote`` is the proposal symbol's own quote (an
    equity); ``leg_quotes`` holds one snapshot per option leg, matched by
    symbol; ``analysis`` is the pricing engine's expiry analysis of the
    whole structure at the proposal's price, in dollars (None for an equity
    or when it could not be computed). ``recent_proposals`` are the other
    proposals created inside the duplicate window and ``orders_today`` the
    orders sent to the broker on ``market_date``.
    """

    now: datetime
    market_date: date
    limits: RiskLimits
    watchlist: tuple[str, ...]
    contract_multiplier: int = Field(gt=0)
    controls: Controls
    clock: MarketClock | None = None
    next_open_date: date | None = None
    equity: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    options_buying_power: float | None = None
    start_of_day_equity: float | None = None
    daily_pnl: float | None = None
    positions: tuple[Position, ...] | None = None
    open_orders: tuple[Order, ...] = ()
    orders_today: int = Field(default=0, ge=0)
    recent_proposals: tuple[Proposal, ...] = ()
    quote: Quote | None = None
    leg_quotes: tuple[OptionSnapshot, ...] = ()
    analysis: PositionSummary | None = None
    errors: tuple[str, ...] = ()

    @field_validator("now")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("watchlist")
    @classmethod
    def _upper(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(symbol.strip().upper() for symbol in value)


# --- rule results and the evaluation ------------------------------------------


class RuleOutcome(str, Enum):
    PASS = "PASS"
    FLAG = "FLAG"  # record it, execute nothing, ask nobody
    ESCALATE = "ESCALATE"  # a human must approve
    REJECT = "REJECT"


class RuleResult(Frozen):
    """One rule's finding on one proposal: its name, outcome and a detail a
    human can act on (the numbers compared, or what was missing).

    ``halt_until`` is set by ``daily_loss_limit`` alone, and only when it
    trips: the instant trading may resume. A rule writes nothing — the
    engine persists the halt.
    """

    name: str
    outcome: RuleOutcome
    detail: str
    halt_until: datetime | None = None

    @field_validator("name", "detail")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Evaluation(Frozen):
    """The pure result of judging one proposal: every rule's result in
    registry order, and the verdict precedence resolved from them.

    ``failing_rule`` is the first rejecting rule when the verdict is REJECT,
    else None. ``halt_until`` is the halt the engine must persist — set only
    when ``daily_loss_limit`` tripped and no halt at least that long is
    already in force. A halt that cannot be read counts as none: the trip
    then names its own instant, or the later one another reading of the
    controls gave. ``decided_at`` is the context's ``now``.
    """

    proposal_id: str
    verdict: Verdict
    failing_rule: str | None = None
    results: tuple[RuleResult, ...]
    decided_at: datetime
    halt_until: datetime | None = None

    @property
    def non_pass(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.outcome is not RuleOutcome.PASS)

    @property
    def rules_evaluated(self) -> list[dict[str, str]]:
        """The store's shape: one ``{"rule", "outcome", "detail"}`` per rule, in order."""
        return [
            {"rule": r.name, "outcome": r.outcome.value, "detail": r.detail}
            for r in self.results
        ]


# --- the limits report (the CLI today, the dashboard in Phase 8) --------------


class LimitLine(Frozen):
    """One limit beside its current consumption. ``headroom`` is what is
    left before the limit binds (negative once breached); any of the three
    numbers is None when the context could not supply it."""

    name: str
    limit: float | None
    used: float | None
    headroom: float | None
    unit: str
    breached: bool
    detail: str


class ExposureLine(Frozen):
    """Exposure in one underlying against the per-position cap, in dollars."""

    underlying: str
    exposure: float | None
    cap: float | None
    headroom: float | None
    breached: bool


class LimitsReport(Frozen):
    """Where the account stands against its limits right now.

    ``blocked_by`` names the account-level rules that would reject any
    proposal at this instant (kill_switch, halted, market_hours,
    daily_loss_limit, max_daily_trades), in rule order; empty means a
    proposal would be judged on its own merits.
    """

    as_of: datetime
    market_open: bool | None
    next_open: datetime | None = None
    next_close: datetime | None = None
    kill_switch: bool
    halt_until: datetime | None = None
    halted: bool
    blocked_by: tuple[str, ...] = ()
    equity: float | None = None
    start_of_day_equity: float | None = None
    daily_pnl: float | None = None
    buying_power: float | None = None
    options_buying_power: float | None = None
    lines: tuple[LimitLine, ...] = ()
    exposures: tuple[ExposureLine, ...] = ()
    auto_execute_enabled: bool
    auto_execute_max_notional: float
    errors: tuple[str, ...] = ()
