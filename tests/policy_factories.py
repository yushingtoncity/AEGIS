"""Hand-built proposals and contexts for the policy tests — no network, no store, no clock.

Shared by every ``tests/test_policy_*.py`` module (import it as
``from policy_factories import ...``: pytest puts ``tests/`` on ``sys.path``).
Later stages may append helpers; the existing ones and their defaults are
pinned by ``tests/test_policy_rules.py::TestFactories`` and must not change.

The defaults are the "everything is fine" pair: ``make_proposal()`` is a
small equity LIMIT BUY of a watchlist symbol and ``make_context()`` an open
market, a healthy account and a fresh quote whose mid is that proposal's
limit price — together they make all twenty-one rules PASS (``passing()``), so
a test changes exactly the one thing it is about.

Everything is deterministic: ``NOW`` is a fixed Thursday during US market
hours, every id is a fixed string (the store models would otherwise mint a
uuid4), and every ``fetched_at`` is ``NOW`` (the data models would otherwise
read the wall clock) — so two calls with the same arguments build equal
values.

The option helpers trade one synthetic chain, ``CHAIN_MIDS`` (SPY near 640,
expiring ``EXPIRY``), and each structure's default limit price is the net
of those mids, so ``context_for(structure)`` is a context in which the
structure is priced exactly at its mid.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from aegis.config import RiskLimits
from aegis.data.models import MarketClock, OptionSnapshot, OptionType, Position, Quote
from aegis.policy.models import PolicyContext, ProposalUnderReview, RuleOutcome, RuleResult
from aegis.policy.rules import RULES
from aegis.pricing.errors import PricingError
from aegis.pricing.models import OptionLeg, PositionSummary, Side
from aegis.pricing.position import analyze_position
from aegis.store.models import (
    Broker,
    Controls,
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Proposal,
    ProposalLeg,
)

# --- the fixed instant --------------------------------------------------------

NOW = datetime(2026, 7, 30, 15, 0, tzinfo=timezone.utc)
"""Thursday 2026-07-30, 11:00 in New York: the market is open."""
MARKET_DATE = date(2026, 7, 30)
NEXT_CLOSE = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)  # 16:00 New York, today
NEXT_OPEN = datetime(2026, 7, 31, 13, 30, tzinfo=timezone.utc)  # 09:30 New York, tomorrow
PROPOSED_AT = NOW - timedelta(minutes=1)
"""When the default proposal was created: a minute before it is judged."""

# --- the default equity trade and account -------------------------------------

WATCHLIST = ("SPY", "QQQ", "AAPL", "NVDA", "MSFT")
PROPOSAL_ID = "prop-0001"
SYMBOL = "AAPL"
QUANTITY = 2.0
LIMIT_PRICE = 200.0
NOTIONAL = 400.0  # QUANTITY × LIMIT_PRICE: inside every default limit
EQUITY = 100_000.0
CASH = 100_000.0
BUYING_POWER = 200_000.0
OPTIONS_BUYING_POWER = 100_000.0
MULTIPLIER = 100

# --- the synthetic option chain -----------------------------------------------

UNDERLYING = "SPY"
EXPIRY = date(2026, 8, 21)  # 22 days after MARKET_DATE
CHAIN_MIDS: dict[tuple[OptionType, float], float] = {
    (OptionType.CALL, 640.0): 8.00,
    (OptionType.CALL, 650.0): 3.60,
    (OptionType.CALL, 660.0): 1.20,
    (OptionType.PUT, 630.0): 4.50,
    (OptionType.PUT, 620.0): 2.50,
    (OptionType.PUT, 610.0): 1.30,
}
"""Mid per (type, strike) of the contracts the option helpers trade."""

_UNSET: Any = object()


def occ(
    option_type: OptionType | str,
    strike: float,
    *,
    underlying: str = UNDERLYING,
    expiration: date = EXPIRY,
) -> str:
    """The OCC symbol of a contract: ``occ("call", 640)`` → ``SPY260821C00640000``."""
    letter = "C" if OptionType(option_type) is OptionType.CALL else "P"
    return f"{underlying.upper()}{expiration:%y%m%d}{letter}{round(strike * 1000):08d}"


def padded(symbol: str) -> str:
    """An OCC symbol in the padded 21-character form, its root left-justified
    to six characters: ``padded("SPY260821C00640000")`` →
    ``SPY   260821C00640000``. It parses as the same contract, and no rule may
    take it for a sound symbol. Anything too short to be an OCC symbol comes
    back as it is."""
    if len(symbol) < 16:
        return symbol
    root, tail = symbol[:-15], symbol[-15:]
    return f"{root:<6}{tail}"


# --- limits, clock, quotes, positions, orders ---------------------------------


def make_limits(**overrides: Any) -> RiskLimits:
    """``RiskLimits`` with its defaults (the config.yaml placeholders: 2% daily
    loss, 10 trades, 5 positions, watchlist only, 5% position cap, 60-minute
    duplicate window, no market orders, 5% price tolerance, quotes at most 120 s
    old for an equity and 1,200 s for an option, min DTE 7, $1,000
    max loss, 10 contracts, short sales escalate, confidence floor 0.5,
    auto-execute on up to $1,000). ``overrides`` replace fields;
    ``auto_execute`` may be a dict (``{"max_notional": 500}``)."""
    return RiskLimits(**overrides)


def exchange_date(instant: datetime) -> date:
    """The calendar date of ``instant`` in New York (a naive one is taken as
    UTC) — what ``build_context`` puts in ``market_date`` / ``next_open_date``."""
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(ZoneInfo("America/New_York")).date()


def make_clock(
    is_open: bool = True,
    *,
    next_open: datetime | None = NEXT_OPEN,
    next_close: datetime | None = NEXT_CLOSE,
) -> MarketClock:
    """A market clock fetched at ``NOW``: open, closing at 16:00 New York
    today and next opening at 09:30 tomorrow."""
    return MarketClock(
        is_open=is_open, next_open=next_open, next_close=next_close, fetched_at=NOW
    )


def make_quote(
    symbol: str = SYMBOL,
    mid: float = LIMIT_PRICE,
    *,
    spread: float = 0.04,
    bid: float | None = _UNSET,
    ask: float | None = _UNSET,
    at: datetime | None = NOW,
) -> Quote:
    """A two-sided equity quote centred on ``mid`` (default AAPL 199.98 /
    200.02), fetched at ``NOW``. ``bid`` / ``ask`` set a side outright —
    None for a one-sided quote, bid above ask for a crossed one. ``at`` is
    its venue timestamp (quote and last trade): ``NOW`` by default, so it is
    fresh in the default context; None for a quote with no timestamp."""
    return Quote(
        symbol=symbol,
        bid=mid - spread / 2 if bid is _UNSET else bid,
        ask=mid + spread / 2 if ask is _UNSET else ask,
        last=mid,
        quote_time=at,
        last_time=at,
        fetched_at=NOW,
    )


def make_snapshot(
    symbol: str,
    mid: float,
    *,
    spread: float = 0.10,
    bid: float | None = _UNSET,
    ask: float | None = _UNSET,
    underlying: str = UNDERLYING,
    at: datetime | None = NOW,
) -> OptionSnapshot:
    """One option contract's quote centred on ``mid`` (the bid floored at
    0.0), fetched at ``NOW``. ``bid`` / ``ask`` set a side outright; ``at``
    is the venue timestamp (``NOW`` by default; None for none)."""
    return OptionSnapshot(
        symbol=symbol,
        underlying=underlying,
        bid=max(0.0, mid - spread / 2) if bid is _UNSET else bid,
        ask=mid + spread / 2 if ask is _UNSET else ask,
        quote_time=at,
        fetched_at=NOW,
    )


def make_position(
    symbol: str = SYMBOL,
    qty: float = 10.0,
    side: str = "long",
    *,
    current_price: float | None = LIMIT_PRICE,
    market_value: float | None = _UNSET,
) -> Position:
    """An open position: by default 10 AAPL long at 200.00, market value
    2,000.00 (``abs(qty) × current_price`` unless ``market_value`` is given —
    pass None for an unknown one)."""
    if market_value is _UNSET:
        market_value = None if current_price is None else abs(qty) * current_price
    return Position(
        symbol=symbol, qty=qty, side=side, current_price=current_price, market_value=market_value
    )


def make_order(symbol: str = SYMBOL, **overrides: Any) -> Order:
    """An OPEN order (status submitted) on another proposal: by default buy
    1 AAPL at a 200.00 limit, id ``order-0001``. ``overrides`` replace fields."""
    fields: dict[str, Any] = {
        "id": "order-0001",
        "proposal_id": "prop-0000",
        "client_order_id": "client-0001",
        "broker": Broker.PAPER,
        "status": OrderStatus.SUBMITTED,
        "submitted_at": NOW - timedelta(minutes=30),
        "updated_at": NOW - timedelta(minutes=30),
        "symbol": symbol,
        "side": OrderSide.BUY,
        "quantity": 1.0,
        "limit_price": LIMIT_PRICE,
    }
    return Order(**{**fields, **overrides})


# --- proposals ----------------------------------------------------------------


def _proposal_fields() -> dict[str, Any]:
    return {
        "id": PROPOSAL_ID,
        "created_at": PROPOSED_AT,
        "cycle_id": "cycle-0001",
        "symbol": SYMBOL,
        "instrument": Instrument.EQUITY,
        "side": OrderSide.BUY,
        "quantity": QUANTITY,
        "order_type": OrderType.LIMIT,
        "limit_price": LIMIT_PRICE,
        "thesis": "SYNTHETIC TEST DATA. A small starter position.",
        "confidence": 0.8,
        "invalidation": "SYNTHETIC TEST DATA. A daily close below 190 invalidates the thesis.",
        "raw_model_output": "{}",
        "model_name": "synthetic-model-v0",
        "prompt_version": "test-prompt-1",
    }


def make_proposal(*, legs: Sequence[ProposalLeg] = (), **overrides: Any) -> ProposalUnderReview:
    """The proposal under review. Default: ``prop-0001``, an equity LIMIT BUY
    of 2 AAPL at 200.00 (notional 400.00), confidence 0.8, a real
    invalidation, created at ``PROPOSED_AT``. ``overrides`` replace
    ``Proposal`` fields; ``legs`` are attached as given (their
    ``proposal_id`` must be the proposal's id)."""
    proposal = Proposal(**{**_proposal_fields(), **overrides})
    return ProposalUnderReview(proposal=proposal, legs=tuple(legs))


def make_recent(**overrides: Any) -> Proposal:
    """ANOTHER stored proposal, for ``recent_proposals``: by default
    ``prop-0000``, the same trade as ``make_proposal()`` created ten minutes
    before it (so, inside the default window, a duplicate)."""
    fields = {
        **_proposal_fields(),
        "id": "prop-0000",
        "cycle_id": "cycle-0000",
        "created_at": PROPOSED_AT - timedelta(minutes=10),
    }
    return Proposal(**{**fields, **overrides})


def make_leg(
    side: OrderSide | str,
    option_type: OptionType | str,
    strike: float,
    quantity: float = 1.0,
    *,
    index: int = 0,
    proposal_id: str = PROPOSAL_ID,
    underlying: str = UNDERLYING,
    expiration: date = EXPIRY,
    symbol: str | None = None,
) -> ProposalLeg:
    """One option leg; its symbol is the OCC symbol of the contract unless
    ``symbol`` is given. Id ``<proposal_id>-leg-<index>``."""
    option_type = OptionType(option_type)
    return ProposalLeg(
        id=f"{proposal_id}-leg-{index}",
        proposal_id=proposal_id,
        leg_index=index,
        symbol=symbol or occ(option_type, strike, underlying=underlying, expiration=expiration),
        option_type=option_type,
        side=OrderSide(side),
        quantity=quantity,
        strike=strike,
        expiration=expiration,
    )


def make_option(
    legs: Sequence[tuple[Any, ...]],
    *,
    side: OrderSide | str,
    limit_price: float | None,
    quantity: float = 1.0,
    underlying: str = UNDERLYING,
    expiration: date = EXPIRY,
    **overrides: Any,
) -> ProposalUnderReview:
    """An option proposal from leg specs ``(side, option_type, strike)`` or
    ``(side, option_type, strike, ratio)``.

    Each leg trades ``ratio × quantity`` contracts (ratio 1 unless given), on
    ``underlying`` expiring ``expiration``. ``side`` is the net direction (buy
    = pay a debit, sell = collect a credit) and ``limit_price`` the positive
    net price per share. As the brain stores it, the proposal's symbol is the
    OCC symbol for a single leg and the underlying for several. ``overrides``
    replace ``Proposal`` fields.
    """
    proposal_id = overrides.get("id", PROPOSAL_ID)
    built = []
    for index, spec in enumerate(legs):
        leg_side, option_type, strike, *rest = spec
        ratio = rest[0] if rest else 1.0
        built.append(
            make_leg(
                leg_side,
                option_type,
                strike,
                ratio * quantity,
                index=index,
                proposal_id=proposal_id,
                underlying=underlying,
                expiration=expiration,
            )
        )
    fields: dict[str, Any] = {
        "symbol": built[0].symbol if len(built) == 1 else underlying,
        "instrument": Instrument.OPTION,
        "side": OrderSide(side),
        "quantity": quantity,
        "order_type": OrderType.LIMIT,
        "limit_price": limit_price,
    }
    return make_proposal(legs=built, **{**fields, **overrides})


def long_call(**overrides: Any) -> ProposalUnderReview:
    """Buy 1 SPY 640 call at an 8.00 debit: notional and max loss 800.00."""
    return make_option(
        [("buy", "call", 640.0)], **{"side": "buy", "limit_price": 8.00, **overrides}
    )


def call_debit_spread(**overrides: Any) -> ProposalUnderReview:
    """Buy the 640 call, sell the 650 call, at a 4.40 debit: max loss 440.00,
    max profit 560.00."""
    legs = [("buy", "call", 640.0), ("sell", "call", 650.0)]
    return make_option(legs, **{"side": "buy", "limit_price": 4.40, **overrides})


def put_credit_spread(**overrides: Any) -> ProposalUnderReview:
    """Sell the 630 put, buy the 620 put, for a 2.00 credit: max loss (and so
    notional) 800.00, max profit 200.00."""
    legs = [("sell", "put", 630.0), ("buy", "put", 620.0)]
    return make_option(legs, **{"side": "sell", "limit_price": 2.00, **overrides})


def naked_short_call(**overrides: Any) -> ProposalUnderReview:
    """Sell 1 SPY 650 call for a 3.60 credit with nothing against it:
    unlimited risk."""
    return make_option(
        [("sell", "call", 650.0)], **{"side": "sell", "limit_price": 3.60, **overrides}
    )


def iron_condor(**overrides: Any) -> ProposalUnderReview:
    """Sell the 650 call and the 630 put, buy the 660 call and the 620 put,
    for a 4.40 credit: max loss (and so notional) 560.00, max profit 440.00."""
    legs = [
        ("sell", "call", 650.0),
        ("buy", "call", 660.0),
        ("sell", "put", 630.0),
        ("buy", "put", 620.0),
    ]
    return make_option(legs, **{"side": "sell", "limit_price": 4.40, **overrides})


# --- contexts -----------------------------------------------------------------


def leg_quotes_for(
    proposal: ProposalUnderReview,
    mids: Sequence[float] | Mapping[str, float] | None = None,
    *,
    spread: float = 0.10,
) -> tuple[OptionSnapshot, ...]:
    """One ``OptionSnapshot`` per leg of ``proposal``, in leg order.

    ``mids`` is one mid per leg (a sequence, in leg order) or per leg symbol
    (a mapping); omitted, each leg is quoted at its ``CHAIN_MIDS`` mid. Every
    quote is ``mid ± spread / 2`` (the bid floored at 0.0).
    """
    snapshots = []
    for index, leg in enumerate(proposal.legs):
        if mids is None:
            mid = CHAIN_MIDS[(leg.option_type, leg.strike)]
        elif isinstance(mids, Mapping):
            mid = mids[leg.symbol]
        else:
            mid = mids[index]
        snapshots.append(make_snapshot(leg.symbol, mid, spread=spread))
    return tuple(snapshots)


def analysis_for(
    proposal: ProposalUnderReview,
    *,
    entry_price: float | None = None,
    contract_multiplier: int = MULTIPLIER,
) -> PositionSummary | None:
    """The pricing engine's expiry analysis of an option proposal at its own
    price — what ``context.analysis`` holds.

    ``entry_price`` is the signed net price per share (debit +, credit −) and
    defaults to the proposal's limit price signed by its side. The whole net
    premium is placed on one leg (the first bought leg for a debit, the first
    sold leg for a credit), which leaves the expiry payoff exact. None for an
    equity, a market order without ``entry_price``, or legs the pricing
    engine cannot analyse as one structure.
    """
    order = proposal.proposal
    if not proposal.is_option or not proposal.legs or len(proposal.underlyings) > 1:
        return None
    if entry_price is None:
        if order.limit_price is None:
            return None
        entry_price = order.limit_price if order.side is OrderSide.BUY else -order.limit_price
    if any(not leg.quantity.is_integer() for leg in proposal.legs):
        return None
    payer_side = OrderSide.BUY if entry_price >= 0 else OrderSide.SELL
    payer = next((leg for leg in proposal.legs if leg.side is payer_side), None)
    if payer is None:
        return None
    legs = [
        OptionLeg(
            option_type=leg.option_type,
            side=Side.LONG if leg.side is OrderSide.BUY else Side.SHORT,
            quantity=int(leg.quantity),
            strike=leg.strike,
            expiry=leg.expiration,
            premium=abs(entry_price) * order.quantity / leg.quantity if leg is payer else 0.0,
            symbol=leg.symbol,
        )
        for leg in proposal.legs
    ]
    try:
        return analyze_position(legs, contract_multiplier=contract_multiplier)
    except PricingError:
        return None


def make_context(**overrides: Any) -> PolicyContext:
    """The "everything is fine" context for ``make_proposal()``.

    ``now`` = ``NOW`` and ``market_date`` = ``MARKET_DATE``; default
    ``RiskLimits()``; the five-symbol ``WATCHLIST``; contract multiplier 100;
    kill switch off and no halt; the market open (``make_clock()``); equity
    100,000 with start-of-day equity 100,000 and P&L 0.0; cash 100,000,
    buying power 200,000, options buying power 100,000; no positions
    (``()`` — known to be none), no open orders, no orders today, no recent
    proposals; an AAPL quote whose mid is 200.00; no leg quotes, no
    analysis, no errors. ``overrides`` replace ``PolicyContext`` fields.
    """
    fields: dict[str, Any] = {
        "now": NOW,
        "market_date": MARKET_DATE,
        "limits": RiskLimits(),
        "watchlist": WATCHLIST,
        "contract_multiplier": MULTIPLIER,
        "controls": Controls(),
        "clock": make_clock(),
        "equity": EQUITY,
        "cash": CASH,
        "buying_power": BUYING_POWER,
        "options_buying_power": OPTIONS_BUYING_POWER,
        "start_of_day_equity": EQUITY,
        "daily_pnl": 0.0,
        "positions": (),
        "open_orders": (),
        "orders_today": 0,
        "recent_proposals": (),
        "quote": make_quote(),
        "leg_quotes": (),
        "analysis": None,
        "errors": (),
    }
    merged = {**fields, **overrides}
    if "next_open_date" not in overrides:
        # as build_context fills it: the exchange-local date of the clock's next open
        clock = merged["clock"]
        opens = None if clock is None else clock.next_open
        try:
            merged["next_open_date"] = None if opens is None else exchange_date(opens)
        except OverflowError:  # an instant at the edge of the calendar has no local date
            merged["next_open_date"] = None
    return PolicyContext(**merged)


def context_for(
    proposal: ProposalUnderReview,
    *,
    mids: Sequence[float] | Mapping[str, float] | None = None,
    **overrides: Any,
) -> PolicyContext:
    """``make_context()`` with the market data ``proposal`` needs.

    Equity: a quote for its symbol whose mid is its limit price (200.00 for
    a market order). Option: no equity quote, ``leg_quotes_for(proposal,
    mids)`` and ``analysis_for(proposal)``. ``overrides`` win over both.
    """
    order = proposal.proposal
    if proposal.is_equity:
        mid = order.limit_price if order.limit_price is not None else LIMIT_PRICE
        market: dict[str, Any] = {"quote": make_quote(order.symbol, mid)}
    else:
        market = {
            "quote": None,
            "leg_quotes": leg_quotes_for(proposal, mids),
            "analysis": analysis_for(proposal),
        }
    return make_context(**{**market, **overrides})


# --- running the rules --------------------------------------------------------


def results(
    proposal: ProposalUnderReview | None = None, context: PolicyContext | None = None
) -> dict[str, RuleResult]:
    """Every registered rule's result on the pair, keyed by rule name, in
    registry order. Defaults: ``make_proposal()`` and ``make_context()``."""
    proposal = make_proposal() if proposal is None else proposal
    context = make_context() if context is None else context
    return {rule.__name__: rule(proposal, context) for rule in RULES}


def outcomes(
    proposal: ProposalUnderReview | None = None, context: PolicyContext | None = None
) -> dict[str, RuleOutcome]:
    """Every rule's outcome on the pair, keyed by rule name."""
    return {name: result.outcome for name, result in results(proposal, context).items()}


def non_pass(
    proposal: ProposalUnderReview | None = None, context: PolicyContext | None = None
) -> dict[str, RuleOutcome]:
    """Only the rules that did not PASS, keyed by name — ``{}`` for the default pair."""
    return {
        name: outcome
        for name, outcome in outcomes(proposal, context).items()
        if outcome is not RuleOutcome.PASS
    }


def passing(
    proposal: ProposalUnderReview | None = None, context: PolicyContext | None = None
) -> bool:
    """Whether all twenty-one rules PASS on the pair — what the engine turns into
    AUTO_EXECUTE. True for the default pair; the sanity check a test runs
    before changing the one thing it is about."""
    return not non_pass(proposal, context)


# --- seeded random cases ------------------------------------------------------
#
# What the engine's randomized test and the limits report's no-drift sweep
# draw from. Every draw comes from the ``random.Random`` passed in, so a seed
# reproduces the whole sequence. ``turbulence`` is the probability that each
# dimension leaves its "everything is fine" default: 0.0 gives the default
# pair's world (a sound proposal in a healthy account), 1.0 changes
# everything at once. A proposal's shape, a low confidence, the side of an
# equity order, whether a sale is a market order and whether it has a long to
# close are drawn at fixed odds whatever the turbulence, so calm cases still
# reach every tier — and closing sales, the one kind of sell that may
# auto-execute, are common enough to meet every stop.

_NAN = float("nan")
_INF = float("inf")

RANDOM_TURBULENCE: tuple[float, ...] = (0.0, 0.03, 0.03, 0.1, 0.3, 0.7)
"""What ``random_case`` draws its turbulence from: mostly calm, sometimes a storm."""
RANDOM_FIGURES: tuple[float | None, ...] = (
    None, _NAN, _INF, -_INF, 0.0, -5_000.0, 1.0, 5_000.0, 100_000.0, 1e9,
)
"""An account figure that left its default: unknown, not finite, zero, negative, tiny, vast."""
RANDOM_PNLS: tuple[float | None, ...] = (
    None, _NAN, _INF, -_INF, 0.0, 750.0, -1_999.99, -2_000.0, -2_000.01, -50_000.0,
)
"""Today's P&L, on both sides of the default loss cap (2% of 100,000 = 2,000)."""
RANDOM_STRIKES: tuple[float, ...] = (610.0, 620.0, 630.0, 640.0, 650.0, 660.0)
DUST_QUANTITIES: tuple[float, ...] = (1e-9, 5e-10, 5e-324)
"""Sizes at or below the float-noise guard (1e-9): ``exceeds`` alone calls
them "not above" zero — a sale that small must still need a long to close."""
SPACED_SYMBOL = "AA PL"
"""A symbol with whitespace inside it: no ticker, whatever it resembles."""
PADDED_CALL = padded(occ("call", 640.0))
"""``SPY   260821C00640000`` — the 640 call in the padded 21-character OCC form."""
SALE_ODDS = 0.25
"""The odds an equity order is a sale, whatever the turbulence."""
CLOSING_ODDS = 0.6
"""The odds an equity sale finds a long position in its symbol to close."""
STOP_ODDS = 0.5
"""The odds a context that holds that long also carries one account-level stop."""
MARKET_SALE_ODDS = 0.2
"""The odds an equity sale is a MARKET order, whatever the turbulence — so
that closing sales at market, which only ``limit_price_sanity`` stops, are
common enough to be judged on an otherwise clean context."""
LARGEST_FINITE = sys.float_info.max
"""The largest number ``RiskLimits`` accepts: infinity and NaN are refused at load."""
VAST_COUNT = 10**400
"""A whole-number limit no float can hold — which ``RiskLimits`` accepts all the same."""


RANDOM_STOPS: tuple[dict[str, Any], ...] = (
    {"controls": Controls(kill_switch=True)},
    {"controls": Controls(halt_until=NEXT_OPEN)},
    {"controls": Controls(halt_unknown=True, problems=("halt_until is unreadable",))},
    {"clock": make_clock(False)},
    {"clock": None},
    {"daily_pnl": -2_500.0},
    {"orders_today": 10},
)
"""One account-level stop each, as ``make_context`` overrides: the kill
switch, a halt in force, a halt nobody can read, a closed market, no market
clock, a P&L under the default loss cap, the default trade cap reached."""


def random_limits(rng: random.Random, turbulence: float = 1.0) -> RiskLimits:
    """``RiskLimits`` with each field — independently, with probability
    ``turbulence`` — replaced by a value from its valid range, the extremes
    included (a cap of zero, a 100% position cap, an empty duplicate window,
    the largest finite float, a count no float can hold). Every value is one
    ``RiskLimits`` accepts: a limit is a finite number, never inf or NaN."""
    choices: dict[str, tuple[Any, ...]] = {
        "daily_loss_limit_pct": (0.5, 2.0, 10.0, 100.0),
        "halt_fallback_hours": (0.5, 24.0, 72.0, LARGEST_FINITE),
        "max_daily_trades": (0, 1, 10, 100, VAST_COUNT),
        "max_open_positions": (0, 1, 5, 50, VAST_COUNT),
        "no_trade_list": ([], ["AAPL"], ["SPY", "TSLA"]),
        "watchlist_only": (True, False),
        "max_position_pct": (0.1, 5.0, 25.0, 100.0),
        "duplicate_window_minutes": (0, 5, 60, 1440, VAST_COUNT),
        "allow_market_orders": (True, False),
        "limit_price_tolerance_pct": (0.0, 1.0, 5.0, 50.0, LARGEST_FINITE),
        "min_dte": (0, 1, 7, 45, VAST_COUNT),
        "max_loss_per_trade": (0.0, 250.0, 1_000.0, 1e6, LARGEST_FINITE),
        "max_contracts": (0, 1, 10, 100, VAST_COUNT),
        "reject_short_sales": (True, False),
        "min_confidence": (0.0, 0.5, 0.9, 1.0),
        "auto_execute": (
            {"enabled": False},
            {"max_notional": 0.0},
            {"max_notional": 500.0},
            {"max_notional": 1e5},
            {"max_notional": LARGEST_FINITE},
        ),
    }
    overrides = {
        name: rng.choice(values) for name, values in choices.items() if rng.random() < turbulence
    }
    return make_limits(**overrides)


def _random_equity(
    rng: random.Random, turbulence: float, fields: dict[str, Any]
) -> ProposalUnderReview:
    """An equity proposal: the default limit buy, with its symbol, size,
    price and order type each leaving the default at ``turbulence`` (the
    symbol sometimes for one with whitespace in it, the size sometimes for a
    billionth of a share or less). It is a sale at ``SALE_ODDS`` — or at
    ``turbulence``, when that is higher — and a sale is a market order at
    ``MARKET_SALE_ODDS``, however calm the case."""
    if rng.random() < turbulence:
        fields["symbol"] = rng.choice((*WATCHLIST, "TSLA", "GME", SPACED_SYMBOL))
    if rng.random() < max(turbulence, SALE_ODDS):
        fields["side"] = OrderSide.SELL
    if rng.random() < turbulence:
        fields["quantity"] = rng.choice(
            (0.5, 1.0, 5.0, 5.005, 25.0, 250.0, 1e6, *DUST_QUANTITIES)
        )
    if rng.random() < turbulence:
        fields["limit_price"] = rng.choice((140.0, 199.0, 200.2, 260.0, None, -1.0, 0.0))
    if rng.random() < turbulence:
        fields.update(order_type=OrderType.MARKET, limit_price=None)
    if fields.get("side") is OrderSide.SELL and rng.random() < MARKET_SALE_ODDS:
        fields.update(order_type=OrderType.MARKET, limit_price=None)
    return make_proposal(**fields)


def _random_dust_sale(
    rng: random.Random, turbulence: float, fields: dict[str, Any]
) -> ProposalUnderReview:
    """An equity SALE of a billionth of a share or less (``DUST_QUANTITIES``),
    otherwise ``_random_equity``'s: in a calm case nothing is held, so it is
    a short sale — however small — and never a "closing sale"."""
    order = _random_equity(rng, turbulence, fields).proposal
    dust = {"side": OrderSide.SELL, "quantity": rng.choice(DUST_QUANTITIES)}
    return ProposalUnderReview(proposal=order.model_copy(update=dust))


def _random_mislabelled(
    rng: random.Random, turbulence: float, fields: dict[str, Any]
) -> ProposalUnderReview:
    """A proposal DECLARED equity that is option-like: named by an option
    contract, carrying option legs, or both. Everything else is
    ``_random_equity``'s — so in a calm case it is, but for its shape, the
    limit buy that would auto-execute."""
    kind = rng.choice(("contract", "legs", "both"))
    legs = []
    if kind != "contract":
        legs = [
            make_leg(
                rng.choice(("buy", "sell")),
                rng.choice(("call", "put")),
                rng.choice(RANDOM_STRIKES),
                index=index,
                proposal_id=fields["id"],
            )
            for index in range(rng.randint(1, 2))
        ]
    order = _random_equity(rng, turbulence, fields).proposal
    if kind != "legs":
        # the contract's symbol — one time in four in the padded OCC form
        named = rng.choice((occ("call", 640.0),) * 3 + (PADDED_CALL,))
        order = order.model_copy(update={"symbol": named})
    return ProposalUnderReview(proposal=order, legs=tuple(legs))


def _random_structure(
    rng: random.Random, turbulence: float, fields: dict[str, Any]
) -> ProposalUnderReview:
    """One of the five named option structures (the naked short call
    included), with its size, expiry, price, order type and the symbol it is
    named by (its underlying, or one option contract — compact or padded)
    each leaving the default at ``turbulence``."""
    build = rng.choice(
        (long_call, call_debit_spread, put_credit_spread, naked_short_call, iron_condor)
    )
    if rng.random() < turbulence:
        fields["quantity"] = rng.choice((2.0, 11.0, 0.5))
    if rng.random() < turbulence:
        fields["expiration"] = rng.choice(
            (MARKET_DATE, MARKET_DATE + timedelta(days=1), MARKET_DATE - timedelta(days=1))
        )
    if rng.random() < turbulence:
        priced = build(**fields).proposal.limit_price
        fields["limit_price"] = round(priced * rng.choice((0.5, 1.3)), 2)
    if rng.random() < turbulence:
        fields.update(order_type=OrderType.MARKET, limit_price=None)
    if rng.random() < turbulence:
        fields["symbol"] = rng.choice((UNDERLYING, occ("call", 640.0), PADDED_CALL))
    return build(**fields)


def _random_legs(rng: random.Random, fields: dict[str, Any]) -> ProposalUnderReview:
    """An option proposal with zero to four arbitrary legs — as often as not
    a broken structure: no legs, two underlyings, mixed or past expiries,
    fractional or oversized quantities, a leg whose symbol is another
    contract's, a padded one or no contract at all, a symbol that is not the
    legs' (or is one in the padded OCC form)."""
    legs = [
        make_leg(
            rng.choice(("buy", "sell")),
            rng.choice(("call", "put")),
            rng.choice(RANDOM_STRIKES),
            rng.choice((1.0, 1.0, 2.0, 0.5, 11.0, 100.0)),
            index=index,
            proposal_id=fields["id"],
            underlying=rng.choice((UNDERLYING, UNDERLYING, UNDERLYING, "QQQ", "GME")),
            expiration=rng.choice(
                (
                    EXPIRY,
                    EXPIRY,
                    MARKET_DATE,
                    MARKET_DATE - timedelta(days=1),
                    EXPIRY + timedelta(days=28),
                )
            ),
            # Mostly the contract the leg describes; else another strike, the
            # other type, another expiration, the bare underlying, or a
            # contract in the padded OCC form.
            symbol=rng.choice(
                (
                    None,
                    None,
                    None,
                    None,
                    occ("call", 700.0),
                    occ("put", 640.0),
                    occ("call", 640.0, expiration=EXPIRY + timedelta(days=28)),
                    UNDERLYING,
                    PADDED_CALL,
                )
            ),
        )
        for index in range(rng.randint(0, 4))
    ]
    first = legs[0].symbol if legs else occ("call", 640.0)
    fields.update(
        symbol=rng.choice((UNDERLYING, first, "QQQ", occ("put", 500.0), padded(first))),
        instrument=Instrument.OPTION,
        side=rng.choice((OrderSide.BUY, OrderSide.SELL)),
        quantity=rng.choice((1.0, 2.0, 3.0)),
        order_type=OrderType.LIMIT,
        limit_price=rng.choice((0.5, 2.0, 4.4, 8.0, None)),
    )
    if rng.random() < 0.2:
        fields.update(order_type=OrderType.MARKET, limit_price=None)
    return make_proposal(legs=legs, **fields)


def random_proposal(
    rng: random.Random, turbulence: float = 1.0, *, proposal_id: str = PROPOSAL_ID
) -> ProposalUnderReview:
    """A proposal drawn from ``rng``: an equity order (just over half — two
    in a hundred of all proposals a sale of a billionth of a share or less),
    an "equity" that is really option-like (one in twenty), one of the named
    option structures (a quarter) or arbitrary legs (the rest).

    Two in ten carry a confidence below the default floor, whatever the
    turbulence; the invalidation, the creation time and each shape's own
    fields leave their defaults with probability ``turbulence``.
    """
    fields: dict[str, Any] = {"id": proposal_id}
    if rng.random() < 0.2:
        fields["confidence"] = rng.choice((0.0, 0.2, 0.49))
    elif rng.random() < turbulence:
        fields["confidence"] = rng.choice((0.5, 0.9, 1.0))
    if rng.random() < turbulence:
        fields["invalidation"] = rng.choice(("", "   "))
    if rng.random() < turbulence:
        fields["created_at"] = rng.choice((NOW - timedelta(hours=2), NOW + timedelta(minutes=5)))
    shape = rng.random()
    if shape < 0.53:
        return _random_equity(rng, turbulence, fields)
    if shape < 0.55:
        return _random_dust_sale(rng, turbulence, fields)
    if shape < 0.6:
        return _random_mislabelled(rng, turbulence, fields)
    if shape < 0.85:
        return _random_structure(rng, turbulence, fields)
    return _random_legs(rng, fields)


def _random_market(
    rng: random.Random, proposal: ProposalUnderReview, turbulence: float
) -> dict[str, Any]:
    """The market data for ``proposal`` — its quote, or its legs' quotes and
    the structure's analysis — sound by default (priced at its own limit),
    and at ``turbulence`` missing, crossed, one-sided, for another symbol or
    far from the limit."""
    order = proposal.proposal
    if proposal.is_equity:
        price = order.limit_price
        mid = price if price is not None and price > 0 else LIMIT_PRICE
        quote: Quote | None = make_quote(order.symbol, mid)
        if rng.random() < turbulence:
            quote = rng.choice(
                (
                    None,
                    make_quote("MSFT" if order.symbol != "MSFT" else "QQQ", mid),
                    make_quote(order.symbol, mid, bid=mid + 1.0, ask=mid - 1.0),
                    make_quote(order.symbol, mid, bid=None),
                    make_quote(order.symbol, mid, ask=_NAN),
                    make_quote(order.symbol, mid * 1.3),
                )
            )
        return {"quote": quote}
    mids = [CHAIN_MIDS.get((leg.option_type, leg.strike), 2.0) for leg in proposal.legs]
    snapshots = list(leg_quotes_for(proposal, mids))
    if snapshots and rng.random() < turbulence:
        index = rng.randrange(len(snapshots))
        symbol = proposal.legs[index].symbol
        damaged = rng.choice(
            (
                None,
                make_snapshot(symbol, 2.0, bid=3.0, ask=1.0),
                make_snapshot(symbol, 0.05, bid=0.0),
                make_snapshot(symbol, 2.0, bid=_NAN),
                make_snapshot(symbol, 2.0, ask=None),
            )
        )
        if damaged is None:
            del snapshots[index]
        else:
            snapshots[index] = damaged
    analysis = analysis_for(proposal)
    if rng.random() < turbulence:
        analysis = rng.choice(
            (None, analysis_for(naked_short_call()), analysis_for(iron_condor()))
        )
    quote = make_quote(UNDERLYING, 640.0) if rng.random() < turbulence else None
    return {"quote": quote, "leg_quotes": tuple(snapshots), "analysis": analysis}


def random_context(
    rng: random.Random,
    proposal: ProposalUnderReview | None = None,
    turbulence: float = 1.0,
) -> PolicyContext:
    """A context for ``proposal`` (default ``make_proposal()``) drawn from ``rng``.

    It starts as the context in which the proposal is priced at its own
    limit in a healthy account; then each dimension — every limit, the
    controls, the clock, each account figure (None, NaN, ±inf, zero and
    negatives included), the positions (two rows for one symbol and the
    padded OCC form among them), the open orders, the day's order count, the
    recent proposals, the watchlist, the contract multiplier, the market
    data — leaves that default with probability ``turbulence``.

    An equity SALE is treated apart, whatever the turbulence: at
    ``CLOSING_ODDS`` the account holds a long position in its symbol (so the
    sale closes it, unless a turbulent draw replaces the positions), and such
    a context carries, at ``STOP_ODDS``, exactly one account-level stop — the
    kill switch, a halt, an unreadable halt, a closed or missing clock, a
    tripped loss limit or a spent trade count — so closing sales meet every
    stop. A closing sale AT MARKET is given none: the market order is its one
    stop (while the limits do not allow them), so in a calm case nothing but
    ``limit_price_sanity`` stands between that sale and the auto tier.
    """
    proposal = make_proposal() if proposal is None else proposal
    order = proposal.proposal
    twin = order.model_copy(
        update={"id": "prop-0000", "created_at": order.created_at - timedelta(minutes=10)}
    )
    held = make_position(order.symbol if proposal.is_equity else SYMBOL)
    choices: dict[str, tuple[Any, ...]] = {
        "controls": (
            Controls(kill_switch=True),
            Controls(halt_until=NOW + timedelta(seconds=1)),
            Controls(halt_until=NOW),
            Controls(halt_until=NOW - timedelta(days=1)),
            Controls(halt_until=NEXT_OPEN + timedelta(days=1)),
            Controls(halt_unknown=True, problems=("halt_until is unreadable",)),
        ),
        "clock": (
            None,
            make_clock(False),
            make_clock(False, next_open=None, next_close=None),
            make_clock(next_close=NOW),
            make_clock(next_open=None),
            make_clock(next_open=NOW - timedelta(hours=1)),
        ),
        "equity": RANDOM_FIGURES,
        "cash": RANDOM_FIGURES,
        "buying_power": RANDOM_FIGURES,
        "options_buying_power": RANDOM_FIGURES,
        "start_of_day_equity": RANDOM_FIGURES,
        "daily_pnl": RANDOM_PNLS,
        "positions": (
            None,
            None,  # twice: unknown positions stay as common as before the rows below
            (held,),
            (held.model_copy(update={"qty": 1_000.0, "market_value": 200_000.0}),),
            (held.model_copy(update={"qty": -10.0, "side": "short"}),),
            (held.model_copy(update={"market_value": None, "current_price": None}),),
            (held.model_copy(update={"qty": _NAN}),),
            (make_position(occ("call", 640.0), 2.0, current_price=8.0, market_value=1_600.0),),
            tuple(make_position(symbol) for symbol in ("SPY", "QQQ", "NVDA", "MSFT", "TSLA")),
            # two rows for one symbol: a long beside a short, and two longs
            (held, held.model_copy(update={"qty": -10.0, "side": "short"})),
            (held.model_copy(update={"qty": -10.0, "side": "short"}), held),
            (held.model_copy(update={"qty": 4.0}), held.model_copy(update={"qty": 6.0})),
            # a contract the broker reported in the padded OCC form
            (make_position(PADDED_CALL, 2.0, current_price=8.0, market_value=1_600.0),),
        ),
        "open_orders": (
            (make_order(order.symbol),),
            (make_order("MSFT", id="order-0002", client_order_id="client-0002"),),
            (make_order("NVDA", limit_price=None),),
            (make_order(occ("call", 640.0), limit_price=8.0),),
            (make_order(PADDED_CALL, limit_price=8.0),),
        ),
        "orders_today": (0, 3, 9, 10, 11),
        "recent_proposals": (
            (twin,),
            (twin.model_copy(update={"created_at": order.created_at - timedelta(days=2)}),),
            (twin.model_copy(update={"created_at": order.created_at + timedelta(minutes=10)}),),
            (make_recent(symbol="MSFT"), order),
        ),
        "watchlist": ((), ("ZZZ",), WATCHLIST[:2]),
        "contract_multiplier": (1, 10),
        "errors": (("account: data fetch failed: unavailable",),),
    }
    fields = _random_market(rng, proposal, turbulence)
    fields["limits"] = random_limits(rng, turbulence)
    if proposal.is_equity and order.side is OrderSide.SELL and rng.random() < CLOSING_ODDS:
        fields["positions"] = (held,)
        if order.order_type is not OrderType.MARKET and rng.random() < STOP_ODDS:
            fields.update(rng.choice(RANDOM_STOPS))
    for name, values in choices.items():
        if rng.random() < turbulence:
            fields[name] = rng.choice(values)
    return make_context(**fields)


def random_case(
    rng: random.Random, index: int = 0
) -> tuple[ProposalUnderReview, PolicyContext]:
    """One (proposal, context) pair drawn from ``rng``, at a turbulence drawn
    from ``RANDOM_TURBULENCE``. The proposal's id is ``rand-<index>``, so a
    sweep's proposals can all be stored side by side."""
    turbulence = rng.choice(RANDOM_TURBULENCE)
    proposal = random_proposal(rng, turbulence, proposal_id=f"rand-{index:04d}")
    return proposal, random_context(rng, proposal, turbulence)
