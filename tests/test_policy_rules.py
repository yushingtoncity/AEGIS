"""The twenty policy rules and the measures behind them — contexts built by hand, no network.

Each rule is called directly (the engine is tested in
``test_policy_engine.py``): its PASS and non-PASS cases, both sides of every
boundary, every "unknown is never good news" path (None, NaN and ±inf for
each figure the rule reads), the float-noise cases, and inputs that are
hostile but valid. The measures are pinned first, since the rules — and the
limits report — are built on them. ``TestFactories`` pins the shared
factories every later policy test builds on.
"""

import ast
import random
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from policy_factories import (
    BUYING_POWER,
    CASH,
    CHAIN_MIDS,
    DUST_QUANTITIES,
    EQUITY,
    EXPIRY,
    LIMIT_PRICE,
    MARKET_DATE,
    MULTIPLIER,
    NEXT_CLOSE,
    NEXT_OPEN,
    NOTIONAL,
    NOW,
    OPTIONS_BUYING_POWER,
    PADDED_CALL,
    PROPOSAL_ID,
    PROPOSED_AT,
    QUANTITY,
    RANDOM_STOPS,
    SPACED_SYMBOL,
    SYMBOL,
    UNDERLYING,
    WATCHLIST,
    analysis_for,
    call_debit_spread,
    context_for,
    exchange_date,
    iron_condor,
    leg_quotes_for,
    long_call,
    make_clock,
    make_context,
    make_leg,
    make_limits,
    make_option,
    make_order,
    make_position,
    make_proposal,
    make_quote,
    make_recent,
    make_snapshot,
    naked_short_call,
    non_pass,
    occ,
    outcomes,
    padded,
    passing,
    put_credit_spread,
    random_case,
    results,
)
from pydantic import ValidationError

from aegis.brain.models import LegSpec, TradeProposal
from aegis.brain.proposal import trade_records
from aegis.config import REPO_ROOT, AutoExecuteConfig, RiskLimits
from aegis.data.models import OptionType
from aegis.policy import measures, rules
from aegis.policy.measures import (
    TOLERANCE,
    below,
    daily_loss_cap,
    entry_price,
    exceeds,
    exposure,
    held_underlyings,
    is_closing_sale,
    is_occ,
    is_plain_equity,
    known,
    leg_dte,
    long_quantity,
    mid_price,
    money,
    notional,
    quote_problems,
    signed_limit,
    structure_problems,
    worst_case_entry,
)
from aegis.policy.models import (
    PolicyContext,
    ProposalUnderReview,
    RuleOutcome,
    RuleResult,
    underlying_of,
)
from aegis.policy.rules import RULE_NAMES, RULES, account_rules
from aegis.store.models import Controls, Instrument, OrderSide, OrderStatus, OrderType

PASS, FLAG, ESCALATE, REJECT = (
    RuleOutcome.PASS,
    RuleOutcome.FLAG,
    RuleOutcome.ESCALATE,
    RuleOutcome.REJECT,
)

NAN = float("nan")
INF = float("inf")
LARGEST_FINITE = sys.float_info.max
"""The largest number ``RiskLimits`` accepts for a limit: infinity and NaN are refused at load."""
UNKNOWN = pytest.mark.parametrize(
    "unknown", [None, NAN, INF, -INF], ids=["none", "nan", "inf", "-inf"]
)
"""Every way a figure can be unknown: missing, not a number, not finite."""

EXPECTED_NAMES = (
    "kill_switch", "halted", "market_hours", "daily_loss_limit", "no_trade_list",
    "watchlist_only", "invalidation_present", "buying_power", "max_position_pct",
    "max_open_positions", "max_daily_trades", "duplicate", "limit_price_sanity",
    "options_min_dte", "options_max_loss", "options_max_contracts", "options_escalate",
    "short_sale", "min_confidence", "auto_tier",
)
ACCOUNT_RULE_NAMES = (
    "kill_switch", "halted", "market_hours", "daily_loss_limit", "max_daily_trades",
)
CONFIG_KEYS = {
    "daily_loss_limit": "(risk_limits.daily_loss_limit_pct)",
    "no_trade_list": "(risk_limits.no_trade_list)",
    "watchlist_only": "(risk_limits.watchlist_only)",
    "max_position_pct": "(risk_limits.max_position_pct)",
    "max_open_positions": "(risk_limits.max_open_positions)",
    "max_daily_trades": "(risk_limits.max_daily_trades)",
    "duplicate": "(risk_limits.duplicate_window_minutes)",
    "limit_price_sanity": "(risk_limits.limit_price_tolerance_pct)",
    "min_confidence": "(risk_limits.min_confidence)",
    "auto_tier": "(risk_limits.auto_execute.max_notional)",
}
OPTION_CONFIG_KEYS = {
    "options_min_dte": "(risk_limits.min_dte)",
    "options_max_loss": "(risk_limits.max_loss_per_trade)",
    "options_max_contracts": "(risk_limits.max_contracts)",
}

STRUCTURES = [long_call, call_debit_spread, put_credit_spread, naked_short_call, iron_condor]
DEFINED_RISK = [long_call, call_debit_spread, put_credit_spread, iron_condor]
ACCOUNT_FIGURES = ["equity", "buying_power", "start_of_day_equity", "daily_pnl"]

RATIO_LEGS = [("buy", "call", 640.0), ("sell", "call", 650.0, 2)]
"""Leg specs of a 1x2 call ratio spread: the sold leg trades twice the contracts."""
WASH_LEGS = [("buy", "call", 640.0), ("sell", "call", 640.0)]
"""Buy and sell the same contract: the legs net to nothing."""
TWO_UNDERLYINGS = [
    make_leg("buy", "call", 640.0),
    make_leg("sell", "call", 650.0, index=1, underlying="QQQ"),
]


def run(rule, proposal=None, context=None) -> RuleResult:
    """One rule on the pair; the defaults are the "everything is fine" pair."""
    proposal = make_proposal() if proposal is None else proposal
    context = make_context() if context is None else context
    result = rule(proposal, context)
    assert isinstance(result, RuleResult)
    assert result.name == rule.__name__
    return result


def sell(quantity=QUANTITY, **overrides) -> ProposalUnderReview:
    """An equity limit SELL of AAPL at 200.00."""
    return make_proposal(side=OrderSide.SELL, quantity=quantity, **overrides)


def holding(qty=10.0, **overrides) -> PolicyContext:
    """The default context holding ``qty`` AAPL long."""
    return make_context(positions=(make_position(qty=qty),), **overrides)


def option_proposal(legs, **overrides) -> ProposalUnderReview:
    """An option proposal with exactly these ``ProposalLeg``s (possibly none, or broken)."""
    fields = {"symbol": UNDERLYING, "instrument": Instrument.OPTION, "limit_price": 4.40}
    return make_proposal(legs=legs, **{**fields, **overrides})


CALL_640 = "SPY260821C00640000"
PADDED_640 = "SPY   260821C00640000"
"""The same contract in the padded 21-character OCC form (the root padded to six)."""
OTHER_DIGITS_640 = "SPY260821C\u0660\u0660\u0666\u0664\u0660\u0660\u0660\u0660"
"""The same contract again, its strike in Arabic-Indic digits: ``parse_occ_symbol``
reads it as the 640 call, and no string comparison takes it for ``CALL_640``."""

DUST = pytest.mark.parametrize("dust", [1e-9, 5e-10, 5e-324], ids=["1e-9", "5e-10", "5e-324"])
"""Sale sizes at or below the float-noise guard: ``exceeds`` calls them "not above" zero."""
SHORT_AAPL = (make_position(qty=-10, side="short"),)
"""The account is SHORT 10 AAPL: there is no long for a sale to close."""


def first_reject(proposal: ProposalUnderReview, context: PolicyContext) -> str | None:
    """The first rule, in registry order, that REJECTs the pair — what the
    engine reports as ``failing_rule`` — or None."""
    rejecting = [name for name, out in outcomes(proposal, context).items() if out is REJECT]
    return rejecting[0] if rejecting else None


def padded_call(**overrides) -> ProposalUnderReview:
    """Buy 1 SPY 640 call at 8.00, with the contract written in the padded OCC
    form everywhere: the proposal's own symbol and its leg's."""
    leg = make_leg("buy", "call", 640.0, symbol=PADDED_640)
    fields = {"symbol": PADDED_640, "quantity": 1.0, "limit_price": 8.00, **overrides}
    return option_proposal([leg], **fields)


def padded_context(proposal: ProposalUnderReview, **overrides) -> PolicyContext:
    """A context that gives a padded contract every chance: a quote for each
    leg under the leg's own (padded) symbol and the sound long call's analysis."""
    market = {
        "quote": None,
        "leg_quotes": tuple(make_snapshot(leg.symbol, 8.00) for leg in proposal.legs),
        "analysis": analysis_for(long_call()),
    }
    return make_context(**{**market, **overrides})


def _misnamed(option_type="call", strike=660.0, **symbol) -> list:
    """One long leg that SAYS it is the 660 call expiring ``EXPIRY`` and whose
    symbol is the contract ``symbol`` describes instead."""
    named = {"option_type": option_type, "strike": strike, **symbol}
    return [make_leg("buy", "call", 660.0, symbol=occ(named.pop("option_type"), **named))]


# Every way a proposal's shape can contradict itself that ``structure_problems``
# judges beyond the legs' own consistency — each alone, with the one finding.
BROKEN_SHAPES = {
    "another strike": (
        option_proposal(_misnamed(strike=700.0)),
        "leg SPY260821C00700000 names the 2026-08-21 700 call, but the leg says the "
        "2026-08-21 660 call",
    ),
    "the other option type": (
        option_proposal(_misnamed(option_type="put")),
        "leg SPY260821P00660000 names the 2026-08-21 660 put, but the leg says the "
        "2026-08-21 660 call",
    ),
    "another expiration": (
        option_proposal(_misnamed(expiration=date(2026, 9, 18))),
        "leg SPY260918C00660000 names the 2026-09-18 660 call, but the leg says the "
        "2026-08-21 660 call",
    ),
    "another root": (
        option_proposal(_misnamed(underlying="QQQ")),
        "the proposal is on SPY but its legs are on QQQ",
    ),
    "not an OCC symbol": (
        option_proposal([make_leg("buy", "call", 660.0, symbol="SPY")]),
        "leg SPY is not an OCC option symbol",
    ),
    "an OCC proposal symbol that is not its leg's": (
        option_proposal([make_leg("buy", "call", 660.0)], symbol=CALL_640),
        "the proposal is named by the option contract SPY260821C00640000, which is not its "
        "leg's SPY260821C00660000",
    ),
    "an OCC proposal symbol with two legs": (
        option_proposal(
            [make_leg("buy", "call", 640.0), make_leg("sell", "call", 650.0, index=1)],
            symbol=CALL_640,
        ),
        "the proposal is named by the option contract SPY260821C00640000 but has 2 legs: a "
        "multi-leg structure is named by its underlying",
    ),
    "an equity with legs": (
        make_proposal(symbol="SPY", legs=[make_leg("buy", "call", 640.0)]),
        "an equity proposal carries 1 option leg(s)",
    ),
    "an equity named by an option contract": (
        make_proposal(symbol=CALL_640, limit_price=8.00),
        "the proposal is declared equity but its symbol SPY260821C00640000 is an option "
        "contract",
    ),
    "a leg in the padded OCC form": (
        option_proposal([make_leg("buy", "call", 640.0, symbol=PADDED_640)]),
        "symbol 'SPY   260821C00640000' is not in the compact OCC form: it contains whitespace",
    ),
    "a leg whose strike is written in another script's digits": (
        option_proposal([make_leg("buy", "call", 640.0, symbol=OTHER_DIGITS_640)]),
        "symbol 'SPY260821C\\u0660\\u0660\\u0666\\u0664\\u0660\\u0660\\u0660\\u0660' is not in "
        "the compact OCC form: it contains non-ASCII characters",
    ),
}
BROKEN = pytest.mark.parametrize(
    ("proposal", "finding"), BROKEN_SHAPES.values(), ids=BROKEN_SHAPES.keys()
)


def flattering(proposal: ProposalUnderReview) -> PolicyContext:
    """A context that gives a broken shape every chance: a two-sided quote
    for its own symbol and for each leg at its limit price, and a sound
    structure's analysis claiming a small, defined loss."""
    order = proposal.proposal
    return make_context(
        quote=make_quote(order.symbol, order.limit_price),
        leg_quotes=tuple(make_snapshot(leg.symbol, order.limit_price) for leg in proposal.legs),
        analysis=analysis_for(call_debit_spread()),
    )


# --- the factories ------------------------------------------------------------


class TestFactories:
    def test_the_default_pair_passes_all_twenty_rules(self):
        found = results()
        assert tuple(found) == RULE_NAMES
        assert len(found) == 20
        assert {r.outcome for r in found.values()} == {PASS}
        assert passing() is True
        assert non_pass() == {}
        assert outcomes() == dict.fromkeys(RULE_NAMES, PASS)

    def test_passing_is_false_as_soon_as_one_rule_does_not_pass(self):
        context = make_context(controls=Controls(kill_switch=True))
        assert passing(context=context) is False
        assert non_pass(context=context) == {"kill_switch": REJECT}

    def test_now_is_a_weekday_inside_us_market_hours(self):
        local = NOW.astimezone(ZoneInfo("America/New_York"))
        assert NOW.tzinfo is timezone.utc
        assert local.weekday() < 5
        assert (local.hour, local.minute) == (11, 0)
        assert local.date() == MARKET_DATE
        assert NOW < NEXT_CLOSE < NEXT_OPEN
        assert PROPOSED_AT == NOW - timedelta(minutes=1)

    def test_everything_is_deterministic(self):
        assert make_proposal() == make_proposal()
        assert make_context() == make_context()
        for build in STRUCTURES:
            assert build() == build()
            assert context_for(build()) == context_for(build())
        assert make_order() == make_order()
        assert make_recent() == make_recent()
        assert results() == results()

    def test_default_proposal(self):
        proposal = make_proposal()
        order = proposal.proposal
        assert proposal.legs == ()
        assert (order.id, order.symbol) == (PROPOSAL_ID, SYMBOL)
        assert order.instrument is Instrument.EQUITY
        assert (order.side, order.order_type) == (OrderSide.BUY, OrderType.LIMIT)
        assert (order.quantity, order.limit_price) == (QUANTITY, LIMIT_PRICE) == (2.0, 200.0)
        assert order.quantity * order.limit_price == NOTIONAL == 400.0
        assert order.created_at == PROPOSED_AT
        assert order.confidence == 0.8
        assert order.invalidation.strip()
        assert SYMBOL in WATCHLIST

    def test_default_context(self):
        context = make_context()
        assert (context.now, context.market_date) == (NOW, MARKET_DATE)
        assert context.limits == RiskLimits()
        assert context.watchlist == WATCHLIST
        assert context.contract_multiplier == MULTIPLIER == 100
        assert context.controls == Controls()
        assert context.clock == make_clock()
        assert (context.clock.is_open, context.clock.next_open, context.clock.next_close) == (
            True, NEXT_OPEN, NEXT_CLOSE,
        )
        assert (context.equity, context.start_of_day_equity) == (EQUITY, EQUITY)
        assert context.daily_pnl == 0.0
        assert (context.cash, context.buying_power, context.options_buying_power) == (
            CASH, BUYING_POWER, OPTIONS_BUYING_POWER,
        )
        assert (EQUITY, CASH, BUYING_POWER, OPTIONS_BUYING_POWER) == (1e5, 1e5, 2e5, 1e5)
        assert context.positions == () and context.open_orders == ()
        assert context.orders_today == 0 and context.recent_proposals == ()
        assert context.quote.symbol == SYMBOL
        assert context.quote.mid == pytest.approx(LIMIT_PRICE)
        assert context.quote.bid < LIMIT_PRICE < context.quote.ask
        assert context.leg_quotes == () and context.analysis is None and context.errors == ()

    def test_overrides_replace_fields(self):
        assert make_proposal(symbol="msft", quantity=7).proposal.symbol == "MSFT"
        assert make_proposal(id="p-9").id == "p-9"
        assert make_context(orders_today=3).orders_today == 3
        assert make_limits(max_daily_trades=3).max_daily_trades == 3
        assert make_limits(auto_execute={"max_notional": 500}).auto_execute.max_notional == 500
        assert make_limits() == RiskLimits()
        recent = make_recent()
        assert (recent.id, recent.symbol, recent.side) == ("prop-0000", SYMBOL, OrderSide.BUY)
        assert recent.created_at == PROPOSED_AT - timedelta(minutes=10)
        order = make_order()
        assert order.status is OrderStatus.SUBMITTED and order.symbol == SYMBOL
        assert (order.quantity, order.limit_price) == (1.0, LIMIT_PRICE)
        position = make_position()
        assert (position.symbol, position.qty, position.side) == (SYMBOL, 10.0, "long")
        assert position.market_value == 2000.0
        assert make_position(qty=-4, side="short").market_value == 800.0
        assert make_position(market_value=None).market_value is None
        assert make_position(current_price=None).market_value is None

    def test_quotes_and_clock(self):
        quote = make_quote("MSFT", 410.0, spread=0.10)
        assert (quote.symbol, quote.fetched_at) == ("MSFT", NOW)
        assert (quote.bid, quote.ask) == pytest.approx((409.95, 410.05))
        assert make_quote(bid=None).bid is None and make_quote(ask=None).ask is None
        snapshot = make_snapshot("SPY260821C00640000", 0.02)
        assert (snapshot.bid, snapshot.ask) == pytest.approx((0.0, 0.07))
        assert make_snapshot("X", 1.0, bid=None).bid is None
        closed = make_clock(False, next_close=None)
        assert (closed.is_open, closed.next_open, closed.next_close) == (False, NEXT_OPEN, None)
        assert closed.fetched_at == NOW

    def test_occ_symbols(self):
        assert occ("call", 640) == "SPY260821C00640000"
        assert occ(OptionType.PUT, 97.5, underlying="aapl", expiration=date(2026, 9, 18)) == (
            "AAPL260918P00097500"
        )
        leg = make_leg("sell", "put", 630.0, 3, index=2)
        assert (leg.id, leg.leg_index, leg.proposal_id) == ("prop-0001-leg-2", 2, PROPOSAL_ID)
        assert (leg.symbol, leg.side, leg.quantity) == ("SPY260821P00630000", OrderSide.SELL, 3.0)
        assert (leg.strike, leg.expiration, leg.option_type) == (630.0, EXPIRY, OptionType.PUT)

    @pytest.mark.parametrize(
        ("build", "symbol", "side", "limit", "leg_count"),
        [
            (long_call, "SPY260821C00640000", OrderSide.BUY, 8.00, 1),
            (call_debit_spread, "SPY", OrderSide.BUY, 4.40, 2),
            (put_credit_spread, "SPY", OrderSide.SELL, 2.00, 2),
            (naked_short_call, "SPY260821C00650000", OrderSide.SELL, 3.60, 1),
            (iron_condor, "SPY", OrderSide.SELL, 4.40, 4),
        ],
    )
    def test_option_structures(self, build, symbol, side, limit, leg_count):
        proposal = build()
        order = proposal.proposal
        assert proposal.is_option and proposal.underlying == UNDERLYING
        assert (order.symbol, order.side, order.limit_price) == (symbol, side, limit)
        assert (order.quantity, order.order_type) == (1.0, OrderType.LIMIT)
        assert len(proposal.legs) == leg_count
        assert [leg.leg_index for leg in proposal.legs] == list(range(leg_count))
        assert {leg.expiration for leg in proposal.legs} == {EXPIRY}
        assert {leg.quantity for leg in proposal.legs} == {1.0}
        assert structure_problems(proposal) == ()
        # Each structure's default limit is the net of the chain mids it trades.
        context = context_for(proposal)
        assert context.quote is None
        assert mid_price(proposal, context) == pytest.approx(signed_limit(proposal))

    def test_structure_overrides(self):
        proposal = iron_condor(quantity=3, id="p-7", confidence=0.4)
        assert proposal.id == "p-7" and proposal.proposal.confidence == 0.4
        assert {leg.quantity for leg in proposal.legs} == {3.0}
        assert {leg.proposal_id for leg in proposal.legs} == {"p-7"}
        ratio = make_option(RATIO_LEGS, side="buy", limit_price=0.8, quantity=2)
        assert [leg.quantity for leg in ratio.legs] == [2.0, 4.0]
        later = long_call(expiration=date(2026, 9, 18), underlying="QQQ")
        assert later.legs[0].symbol == "QQQ260918C00640000"

    @pytest.mark.parametrize(
        ("build", "max_loss", "max_profit"),
        [
            (long_call, 800.0, None),
            (call_debit_spread, 440.0, 560.0),
            (put_credit_spread, 800.0, 200.0),
            (iron_condor, 560.0, 440.0),
        ],
    )
    def test_analysis_of_the_defined_risk_structures(self, build, max_loss, max_profit):
        analysis = analysis_for(build())
        assert analysis.unlimited_risk is False
        assert analysis.max_loss == pytest.approx(max_loss)
        assert analysis.max_profit == (None if max_profit is None else pytest.approx(max_profit))
        assert context_for(build()).analysis == analysis

    def test_analysis_of_a_naked_short_call_is_unlimited(self):
        analysis = analysis_for(naked_short_call())
        assert analysis.unlimited_risk is True and analysis.max_loss is None

    def test_analysis_scales_with_quantity_and_takes_an_entry_price(self):
        assert analysis_for(call_debit_spread(quantity=3)).max_loss == pytest.approx(1320.0)
        market = call_debit_spread(order_type=OrderType.MARKET, limit_price=None)
        assert analysis_for(market) is None
        assert analysis_for(market, entry_price=5.0).max_loss == pytest.approx(500.0)
        assert analysis_for(make_proposal()) is None
        assert analysis_for(option_proposal([])) is None
        assert analysis_for(option_proposal([make_leg("buy", "call", 640.0, 1.5)])) is None

    @pytest.mark.parametrize("build", DEFINED_RISK)
    def test_a_defined_risk_structure_only_escalates(self, build):
        proposal = build()
        assert non_pass(proposal, context_for(proposal)) == {
            "options_escalate": ESCALATE,
            "auto_tier": ESCALATE,
        }

    def test_leg_quotes_for(self):
        proposal = call_debit_spread()
        default = leg_quotes_for(proposal)
        assert [s.symbol for s in default] == [leg.symbol for leg in proposal.legs]
        assert [s.mid for s in default] == pytest.approx([8.00, 3.60])
        assert [s.mid for s in default] == pytest.approx(
            [CHAIN_MIDS[(leg.option_type, leg.strike)] for leg in proposal.legs]
        )
        assert [s.mid for s in leg_quotes_for(proposal, [9.0, 4.0])] == pytest.approx([9.0, 4.0])
        by_symbol = {proposal.legs[0].symbol: 7.0, proposal.legs[1].symbol: 3.0}
        assert [s.mid for s in leg_quotes_for(proposal, by_symbol)] == pytest.approx([7.0, 3.0])
        wide = leg_quotes_for(proposal, spread=1.0)
        assert (wide[0].bid, wide[0].ask) == pytest.approx((7.5, 8.5))
        assert all(s.fetched_at == NOW for s in default)
        repriced = context_for(proposal, mids=[9.0, 4.0])
        assert repriced.leg_quotes == leg_quotes_for(proposal, [9.0, 4.0])

    def test_context_for_an_equity_centres_the_quote_on_its_limit(self):
        proposal = make_proposal(symbol="MSFT", limit_price=410.0, quantity=1)
        context = context_for(proposal, orders_today=2)
        assert context.quote.symbol == "MSFT" and context.quote.mid == pytest.approx(410.0)
        assert context.orders_today == 2
        assert passing(proposal, context)
        assert context_for(make_proposal()) == make_context()

    def test_next_open_date_is_the_exchange_date_of_the_clocks_next_open(self):
        # As build_context fills it: derived from the clock unless overridden.
        assert exchange_date(NEXT_OPEN) == date(2026, 7, 31)
        assert make_context().next_open_date == date(2026, 7, 31) != MARKET_DATE
        # 01:00 UTC on the 31st is still the 30th in New York; a naive instant is UTC.
        late = datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc)
        assert exchange_date(late) == exchange_date(late.replace(tzinfo=None)) == MARKET_DATE
        same_day = make_clock(False, next_open=datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc))
        assert make_context(clock=same_day).next_open_date == MARKET_DATE
        assert make_context(clock=None).next_open_date is None
        assert make_context(clock=make_clock(next_open=None)).next_open_date is None
        assert make_context(next_open_date=None).next_open_date is None
        assert make_context(next_open_date=MARKET_DATE).next_open_date == MARKET_DATE
        # An instant at the edge of the calendar has no local date: unknown, not an error.
        edge = datetime.max.replace(tzinfo=timezone(timedelta(hours=-14)))
        assert make_context(clock=make_clock(next_open=edge)).next_open_date is None

    def test_padded_writes_the_21_character_occ_form(self):
        assert padded(CALL_640) == PADDED_640 == PADDED_CALL
        assert len(PADDED_640) == 21 and PADDED_640[:6] == "SPY   "
        assert padded("BRKB260821C00640000") == "BRKB  260821C00640000"
        assert padded(PADDED_640) == PADDED_640  # already padded
        assert padded("SPY") == "SPY"  # too short to be a contract: left as it is
        assert is_occ(PADDED_640)  # it parses — which is why the rules must judge it


class TestRandomGenerators:
    """The seeded generators the engine, limits and CLI sweeps draw from must
    reach the shapes rulings R4 and R5 are about."""

    SEED = 20261001
    CASES = 600

    def sweep(self):
        rng = random.Random(self.SEED)
        return [random_case(rng, index) for index in range(self.CASES)]

    def test_a_seed_reproduces_the_sweep(self):
        assert self.sweep() == self.sweep()

    def test_sales_of_a_billionth_of_a_share_or_less(self):
        assert DUST_QUANTITIES == (1e-9, 5e-10, 5e-324)
        assert all(not exceeds(quantity, 0.0) for quantity in DUST_QUANTITIES)
        dust = [
            (proposal, context)
            for proposal, context in self.sweep()
            if proposal.is_equity
            and proposal.proposal.side is OrderSide.SELL
            and proposal.proposal.quantity <= TOLERANCE
        ]
        assert len(dust) >= 10
        assert {proposal.proposal.quantity for proposal, _ in dust} == set(DUST_QUANTITIES)
        # Most of them with no long to close — and then never all twenty PASS.
        unbacked = [
            (proposal, context)
            for proposal, context in dust
            if context.positions is not None
            and not long_quantity(context, proposal.proposal.symbol)
        ]
        assert len(unbacked) >= 5
        for proposal, context in unbacked:
            assert not is_closing_sale(proposal, context)
            assert outcomes(proposal, context)["short_sale"] in (ESCALATE, REJECT)
            assert outcomes(proposal, context)["auto_tier"] is ESCALATE
            assert not passing(proposal, context)

    def test_padded_and_whitespace_symbols(self):
        cases = self.sweep()

        def named(proposal):
            return [proposal.proposal.symbol, *(leg.symbol for leg in proposal.legs)]

        def spaced(symbol):
            return any(character.isspace() for character in symbol)

        padded_shapes = [p for p, _ in cases if any(spaced(s) and is_occ(s) for s in named(p))]
        tickers = [p for p, _ in cases if any(spaced(s) and not is_occ(s) for s in named(p))]
        assert len(padded_shapes) >= 10 and len(tickers) >= 3
        assert any(p.is_equity for p in padded_shapes) and any(p.is_option for p in padded_shapes)
        assert {p.proposal.symbol for p in tickers} == {SPACED_SYMBOL}
        for proposal, context in cases:
            if any(spaced(symbol) for symbol in named(proposal)):
                found = outcomes(proposal, context)
                assert "contains whitespace" in " | ".join(structure_problems(proposal))
                assert found["options_max_loss"] is REJECT
                assert found["options_escalate"] is ESCALATE and found["auto_tier"] is ESCALATE
        # ... and on the books: a padded position, a padded open order.
        assert sum(
            1 for _, c in cases if any(spaced(row.symbol) for row in c.positions or ())
        ) >= 3
        assert sum(1 for _, c in cases if any(spaced(o.symbol) for o in c.open_orders)) >= 3

    def test_the_padded_contract_is_on_the_books_whatever_the_proposal_names(self):
        # The rows above may be copies of a spaced PROPOSAL symbol. These are the
        # generator's own: the padded 640 call as a position and as an open
        # order, beside a proposal that is not named by it — counted under SPY.
        cases = self.sweep()

        def carried(rows, proposal):
            return any(
                row.symbol == PADDED_CALL != proposal.proposal.symbol for row in rows or ()
            )

        padded_positions = [(p, c) for p, c in cases if carried(c.positions, p)]
        padded_orders = [(p, c) for p, c in cases if carried(c.open_orders, p)]
        assert len(padded_positions) >= 5 and len(padded_orders) >= 5
        for _, context in padded_positions:
            assert "SPY" in held_underlyings(context)
            assert exposure(context, "SPY") != 0.0  # 1,600.00 or more — or unknown
        known_books = [(p, c) for p, c in padded_orders if c.positions is not None]
        assert len(known_books) >= 5
        for proposal, context in padded_orders:
            if context.positions is None:  # nothing is known of what SPY holds
                assert held_underlyings(context) is None and exposure(context, "SPY") is None
                continue
            assert "SPY" in held_underlyings(context)
            assert exposure(context, "SPY") != 0.0
        # An open order on the padded contract makes a proposal that trades
        # that contract a repeat (rule 12(b) matches the instrument, not the
        # string) — and no other proposal.
        repeats = [(p, c) for p, c in padded_orders if CALL_640 in p.symbols]
        others = [(p, c) for p, c in padded_orders if CALL_640 not in p.symbols]
        assert len(repeats) >= 2 and len(others) >= 10
        for proposal, context in repeats:
            assert PADDED_CALL not in proposal.symbols
            result = outcomes(proposal, context)["duplicate"]
            assert result is REJECT, proposal.id
        unrepeated = [
            (p, c) for p, c in others
            if not c.recent_proposals and PADDED_CALL not in p.symbols
        ]
        assert len(unrepeated) >= 3
        for proposal, context in unrepeated:
            assert outcomes(proposal, context)["duplicate"] is PASS, proposal.id

    def test_closing_sales_at_market_with_nothing_else_against_them(self):
        # A market order is the one stop a calm closing sale carries: only
        # limit_price_sanity stands between it and the auto tier.
        cases = self.sweep()
        at_market = [
            (p, c)
            for p, c in cases
            if p.proposal.order_type is OrderType.MARKET and is_closing_sale(p, c)
        ]
        assert len(at_market) >= 15
        alone = [
            (p, c)
            for p, c in at_market
            if [name for name, out in outcomes(p, c).items() if out is REJECT]
            == ["limit_price_sanity"]
        ]
        assert len(alone) >= 3
        for proposal, context in alone:
            assert not context.limits.allow_market_orders
            assert "market orders are not allowed" in (
                results(proposal, context)["limit_price_sanity"].detail
            )
        # Where the limits allow market orders the same sale still needs a human
        # (the auto tier takes limit orders only) — it is never one that passes.
        assert not any(passing(p, c) for p, c in at_market)

    def test_two_rows_for_one_symbol_and_closing_sales_in_front_of_every_stop(self):
        cases = self.sweep()
        doubled = [
            context
            for _, context in cases
            if context.positions
            and len({row.symbol for row in context.positions}) < len(context.positions)
        ]
        assert len(doubled) >= 5
        sides = [{row.side for row in context.positions} for context in doubled]
        assert {"long", "short"} in sides and {"long"} in sides
        closing = [(p, c) for p, c in cases if is_closing_sale(p, c)]
        assert len(closing) >= 40
        assert sum(1 for p, c in closing if passing(p, c)) >= 5
        assert len(RANDOM_STOPS) == 7
        stopped = {
            "kill_switch": 0, "halted": 0, "market_hours": 0, "daily_loss_limit": 0,
            "max_daily_trades": 0,
        }
        for proposal, context in closing:
            for result in account_rules(context):
                stopped[result.name] += result.outcome is REJECT
        assert all(count >= 2 for count in stopped.values()), stopped


# --- the registry -------------------------------------------------------------


class TestRegistry:
    def test_twenty_rules_in_the_specified_order(self):
        assert RULE_NAMES == EXPECTED_NAMES
        assert len(RULES) == len(set(RULES)) == 20
        assert tuple(rule.__name__ for rule in RULES) == RULE_NAMES
        for rule in RULES:
            assert getattr(rules, rule.__name__) is rule

    @pytest.mark.parametrize("build", [make_proposal, *STRUCTURES])
    def test_every_result_carries_its_rules_name_and_a_detail(self, build):
        proposal = build()
        blind = make_context(positions=None, clock=None, quote=None)
        for context in (context_for(proposal), blind):
            for rule in RULES:
                result = rule(proposal, context)
                assert isinstance(result, RuleResult)
                assert result.name == rule.__name__
                assert result.detail.strip()
                assert "\n" not in result.detail

    def test_only_daily_loss_limit_ever_sets_a_halt(self):
        context = make_context(daily_pnl=-5000.0, controls=Controls(kill_switch=True), clock=None)
        found = results(context=context)
        assert [name for name, result in found.items() if result.halt_until is not None] == [
            "daily_loss_limit"
        ]

    def test_rules_are_deterministic(self):
        for build in [make_proposal, *STRUCTURES]:
            proposal = build()
            context = context_for(proposal)
            assert results(proposal, context) == results(proposal, context)
            assert results(build(), context_for(build())) == results(proposal, context)

    @pytest.mark.parametrize(("name", "key"), CONFIG_KEYS.items())
    def test_a_limit_rule_names_its_config_key_passing_and_failing(self, name, key):
        rule = getattr(rules, name)
        bad_proposal = make_proposal(
            symbol="TSLA", quantity=1000, limit_price=260.0, confidence=0.1
        )
        bad_context = make_context(
            daily_pnl=-9000.0,
            orders_today=99,
            limits=make_limits(no_trade_list=["TSLA"], max_open_positions=0),
            recent_proposals=(make_recent(symbol="TSLA", quantity=1),),
            quote=make_quote("TSLA", 200.0),
        )
        good, bad = run(rule), run(rule, bad_proposal, bad_context)
        assert good.outcome is PASS and bad.outcome is not PASS
        assert key in good.detail and key in bad.detail

    @pytest.mark.parametrize(("name", "key"), OPTION_CONFIG_KEYS.items())
    def test_an_option_rule_names_its_config_key_passing_and_failing(self, name, key):
        rule = getattr(rules, name)
        proposal = call_debit_spread()
        good = run(rule, proposal, context_for(proposal))
        tight = make_limits(min_dte=30, max_loss_per_trade=1.0, max_contracts=0)
        bad = run(rule, proposal, context_for(proposal, limits=tight))
        assert (good.outcome, bad.outcome) == (PASS, REJECT)
        assert key in good.detail and key in bad.detail


class TestAccountRules:
    CONTEXTS = {
        "fine": {},
        "kill switch": {"controls": Controls(kill_switch=True)},
        "halted": {"controls": Controls(halt_until=NOW + timedelta(hours=1))},
        "halt unknown": {
            "controls": Controls(halt_unknown=True, problems=("halt_until unreadable",))
        },
        "closed": {"clock": make_clock(False)},
        "no clock": {"clock": None},
        "loss tripped": {"daily_pnl": -2500.0},
        "pnl unknown": {"daily_pnl": None},
        "trade cap": {"orders_today": 10},
        "everything": {
            "controls": Controls(kill_switch=True, halt_until=NOW + timedelta(days=1)),
            "clock": make_clock(False),
            "daily_pnl": -2500.0,
            "orders_today": 11,
        },
    }

    def test_the_five_proposal_independent_rules_in_order(self):
        found = account_rules(make_context())
        assert tuple(result.name for result in found) == ACCOUNT_RULE_NAMES
        assert {result.outcome for result in found} == {PASS}

    @pytest.mark.parametrize("overrides", CONTEXTS.values(), ids=CONTEXTS.keys())
    @pytest.mark.parametrize("build", [make_proposal, naked_short_call])
    def test_equal_the_registered_rules_results_whatever_the_proposal(self, overrides, build):
        context = make_context(**overrides)
        registered = tuple(getattr(rules, name)(build(), context) for name in ACCOUNT_RULE_NAMES)
        assert account_rules(context) == registered

    @pytest.mark.parametrize("overrides", CONTEXTS.values(), ids=CONTEXTS.keys())
    def test_a_closing_sale_gets_the_same_five_results_as_any_proposal(self, overrides):
        # The one sell that may auto-execute is exempt from no account-level stop.
        closing = sell(2)
        context = holding(10, **overrides)
        assert is_closing_sale(closing, context)
        registered = tuple(getattr(rules, name)(closing, context) for name in ACCOUNT_RULE_NAMES)
        assert account_rules(context) == registered
        bought = tuple(
            getattr(rules, name)(make_proposal(), context) for name in ACCOUNT_RULE_NAMES
        )
        assert registered == bought
        # ... and they are the stops the same context holds with nothing in the account.
        bare = account_rules(make_context(**overrides))
        assert [r.outcome for r in registered] == [r.outcome for r in bare]

    @pytest.mark.parametrize(
        ("stop", "rule"),
        [
            ({"controls": Controls(kill_switch=True)}, "kill_switch"),
            ({"controls": Controls(halt_until=NEXT_OPEN)}, "halted"),
            ({"controls": Controls(halt_unknown=True)}, "halted"),
            ({"clock": make_clock(False)}, "market_hours"),
            ({"clock": None}, "market_hours"),
            ({"daily_pnl": -2500.0}, "daily_loss_limit"),
            ({"orders_today": 10}, "max_daily_trades"),
        ],
        ids=[
            "kill switch", "halted", "halt unknown", "closed", "no clock", "loss tripped",
            "trade cap",
        ],
    )
    def test_every_stop_stops_a_closing_sale(self, stop, rule):
        closing = sell(2)
        assert passing(closing, holding(10))  # but for the stop, all twenty PASS
        context = holding(10, **stop)
        found = results(closing, context)
        assert found[rule].outcome is REJECT
        assert first_reject(closing, context) == rule
        assert non_pass(closing, context) == {rule: REJECT}
        if rule == "daily_loss_limit":
            assert found[rule].halt_until == NEXT_OPEN

    @pytest.mark.parametrize(
        ("label", "blocked"),
        [
            ("fine", ()),
            ("kill switch", ("kill_switch",)),
            ("halted", ("halted",)),
            ("halt unknown", ("halted",)),
            ("closed", ("market_hours",)),
            ("no clock", ("market_hours",)),
            ("loss tripped", ("daily_loss_limit",)),
            ("pnl unknown", ("daily_loss_limit",)),
            ("trade cap", ("max_daily_trades",)),
            ("everything", ACCOUNT_RULE_NAMES),
        ],
    )
    def test_what_blocks_trading(self, label, blocked):
        found = account_rules(make_context(**self.CONTEXTS[label]))
        assert tuple(r.name for r in found if r.outcome is REJECT) == blocked
        assert {r.outcome for r in found} <= {PASS, REJECT}


# --- measures: numbers --------------------------------------------------------


class TestKnown:
    @pytest.mark.parametrize("value", [0, 1, -3, 2.5, -0.0, 1e308, 5e-324, 10**20])
    def test_real_finite_numbers_come_back_as_floats(self, value):
        result = known(value)
        assert isinstance(result, float) and result == float(value)

    @pytest.mark.parametrize(
        "value", [None, NAN, INF, -INF, True, False, "1.5", "", b"1", [1.0], 1 + 2j, 10**400]
    )
    def test_everything_else_is_unknown(self, value):
        assert known(value) is None


class TestExceeds:
    def test_plainly_over_and_under(self):
        assert exceeds(2.0, 1.0) and not exceeds(1.0, 2.0)
        assert below(1.0, 2.0) and not below(2.0, 1.0)

    def test_equal_is_neither_over_nor_under(self):
        for value in (0.0, 1.0, -5.0, 1001.0, 1e12):
            assert not exceeds(value, value) and not below(value, value)

    def test_the_spec_example_is_not_over(self):
        assert not exceeds(10 * 100.1, 1001.0)

    @pytest.mark.parametrize(
        ("value", "limit"),
        [(3 * 100.01, 300.03), (3 * 0.05, 0.15), (0.1 + 0.2, 0.3), (100.1 + 200.11, 300.21)],
    )
    def test_float_noise_never_flips_a_boundary(self, value, limit):
        assert value > limit  # plain comparison is fooled...
        assert not exceeds(value, limit)  # ...the measure is not
        assert not below(limit, value)

    def test_one_cent_is_not_noise(self):
        assert exceeds(1000.01, 1000.0) and below(999.99, 1000.0)
        assert exceeds(0.01, 0.0) and below(-0.01, 0.0)
        assert exceeds(1e6 + 0.01, 1e6) and below(1e6 - 0.01, 1e6)

    def test_the_tolerance_is_relative_above_one(self):
        # value - limit > TOLERANCE × max(1, |value|, |limit|): a cent stays
        # visible up to millions of dollars; at a billion the noise floor is $1.
        assert TOLERANCE == 1e-9
        assert not exceeds(1e9 + 0.5, 1e9) and exceeds(1e9 + 2.0, 1e9)
        assert not exceeds(1e12 + 1e-4, 1e12) and exceeds(1e12 + 1e4, 1e12)
        assert not exceeds(TOLERANCE / 2, 0.0) and exceeds(TOLERANCE * 2, 0.0)
        assert not exceeds(-1e9, -1e9 - 0.5) and exceeds(-1e9, -1e9 - 2.0)

    def test_infinity_exceeds_every_finite_limit(self):
        for limit in (0.0, 1000.0, 1e308, -1e308):
            assert exceeds(INF, limit)
            assert not exceeds(limit, INF)
            assert below(limit, INF)
            assert exceeds(limit, -INF)
        assert not exceeds(INF, INF) and not exceeds(-INF, -INF)
        assert not exceeds(-INF, 0.0)

    def test_nan_exceeds_nothing_and_nothing_exceeds_it(self):
        for other in (0.0, 1.0, INF, -INF, NAN):
            assert not exceeds(NAN, other) and not exceeds(other, NAN)
            assert not below(NAN, other) and not below(other, NAN)

    def test_ints_of_any_size_never_raise(self):
        assert exceeds(11, 10) and not exceeds(10, 10)
        assert not exceeds(5.0, 10**400) and exceeds(10**400, 5.0)
        assert exceeds(5.0, -(10**400))


class TestMoney:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1234.56, "$1,234.56"),
            (-1234.56, "-$1,234.56"),
            (0, "$0.00"),
            (0.005, "$0.01"),
            (-0.001, "$0.00"),
            (-0.0, "$0.00"),
            (1_000_000, "$1,000,000.00"),
            (0.1 + 0.2, "$0.30"),
            (INF, "unlimited"),
            (-INF, "-unlimited"),
            (None, "n/a"),
            (NAN, "n/a"),
            (10**400, "unlimited"),
        ],
    )
    def test_formatting(self, value, text):
        assert money(value) == text


class TestIsOcc:
    def test_option_symbols(self):
        assert is_occ("SPY260821C00640000") and is_occ(" spy260821p00640000 ")
        assert is_occ("BRKB260821C00640000")

    @pytest.mark.parametrize(
        "symbol", ["SPY", "", "AAPL", "SPY260821X00640000", "SPY261341C00640000"]
    )
    def test_everything_else(self, symbol):
        assert not is_occ(symbol)


# --- measures: the account ----------------------------------------------------


class TestLongQuantity:
    def test_nothing_held_is_zero(self):
        assert long_quantity(make_context(), SYMBOL) == 0.0

    def test_unknown_positions_are_unknown(self):
        assert long_quantity(make_context(positions=None), SYMBOL) is None

    def test_a_long_position_in_exactly_the_symbol(self):
        context = make_context(positions=(make_position(qty=10), make_position("MSFT", 7)))
        assert long_quantity(context, SYMBOL) == 10.0
        assert long_quantity(context, "aapl ") == 10.0
        assert long_quantity(context, "MSFT") == 7.0
        assert long_quantity(context, "SPY") == 0.0

    def test_an_option_position_is_not_a_position_in_its_underlying(self):
        context = make_context(positions=(make_position("SPY260821C00640000", 2),))
        assert long_quantity(context, "SPY") == 0.0
        assert long_quantity(context, "SPY260821C00640000") == 2.0

    def test_a_short_position_is_not_a_long_one(self):
        # Alpaca reports a short with a negative quantity and side "short".
        context = make_context(positions=(make_position(qty=-10, side="short"),))
        assert long_quantity(context, SYMBOL) == 0.0
        context = make_context(positions=(make_position(qty=10, side="short"),))
        assert long_quantity(context, SYMBOL) == 0.0

    @pytest.mark.parametrize("side", ["", "unknown", "LONGISH"])
    def test_a_side_that_is_not_long_proves_nothing(self, side):
        assert long_quantity(make_context(positions=(make_position(side=side),)), SYMBOL) == 0.0

    def test_side_is_read_case_insensitively(self):
        context = make_context(positions=(make_position(side=" Long "),))
        assert long_quantity(context, SYMBOL) == 10.0

    def test_a_long_with_a_negative_quantity_contradicts_itself(self):
        context = make_context(positions=(make_position(qty=-10, side="long"),))
        assert long_quantity(context, SYMBOL) == 0.0

    @pytest.mark.parametrize("qty", [NAN, INF, -INF])
    def test_an_unknown_quantity_is_unknown(self, qty):
        context = make_context(positions=(make_position(qty=qty, market_value=1.0),))
        assert long_quantity(context, SYMBOL) is None
        assert long_quantity(context, "MSFT") == 0.0  # another symbol's row is not consulted

    def test_two_long_rows_for_one_symbol_add_up(self):
        context = make_context(positions=(make_position(qty=4), make_position(qty=6)))
        assert long_quantity(context, SYMBOL) == 10.0
        three = (make_position(qty=4), make_position("MSFT", 7), make_position(qty=0.5))
        assert long_quantity(make_context(positions=three), SYMBOL) == 4.5

    def test_a_long_row_beside_a_short_row_is_not_a_long(self):
        # A symbol whose rows contradict each other cannot be shown to be held
        # long: not the long row alone, not the first row, not the net.
        long_row, short_row = make_position(qty=10), make_position(qty=-10, side="short")
        for rows in ((long_row, short_row), (short_row, long_row)):
            assert long_quantity(make_context(positions=rows), SYMBOL) == 0.0
        lopsided = (make_position(qty=100), make_position(qty=-1, side="short"))
        assert long_quantity(make_context(positions=lopsided), SYMBOL) == 0.0
        assert long_quantity(make_context(positions=lopsided[::-1]), SYMBOL) == 0.0

    def test_a_long_row_beside_a_row_of_unknown_quantity_is_unknown(self):
        unknown = make_position(qty=NAN, market_value=1.0)
        for rows in ((make_position(qty=10), unknown), (unknown, make_position(qty=10))):
            assert long_quantity(make_context(positions=rows), SYMBOL) is None


class TestIsClosingSale:
    def test_selling_what_is_held(self):
        assert is_closing_sale(sell(10), holding(10))
        assert is_closing_sale(sell(4), holding(10))

    def test_selling_more_than_is_held_is_not_closing(self):
        assert not is_closing_sale(sell(10.01), holding(10))
        assert not is_closing_sale(sell(11), holding(10))

    def test_float_noise_in_the_quantity_still_closes(self):
        assert is_closing_sale(sell(0.1 + 0.2), holding(0.3))

    def test_nothing_held_or_a_short_is_not_closing(self):
        assert not is_closing_sale(sell(1), make_context())
        short = make_context(positions=(make_position(qty=-10, side="short"),))
        assert not is_closing_sale(sell(1), short)

    def test_unknown_positions_prove_nothing(self):
        assert not is_closing_sale(sell(1), make_context(positions=None))
        unknown = make_context(positions=(make_position(qty=NAN, market_value=1.0),))
        assert not is_closing_sale(sell(1), unknown)

    def test_only_an_equity_sell_can_close(self):
        assert not is_closing_sale(make_proposal(), holding(10))  # a buy
        option_sell = naked_short_call()
        held = make_context(positions=(make_position(option_sell.proposal.symbol, 5),))
        assert not is_closing_sale(option_sell, held)

    def test_a_position_in_another_symbol_does_not_count(self):
        assert not is_closing_sale(sell(1), make_context(positions=(make_position("MSFT", 10),)))

    @DUST
    def test_a_sale_of_dust_closes_nothing_when_no_long_is_held(self, dust):
        # exceeds() alone calls a billionth of a share "not above" a holding of
        # zero: a closing sale needs a long to close, however small the sale.
        assert not exceeds(dust, 0.0)
        assert not is_closing_sale(sell(dust), make_context())
        assert not is_closing_sale(sell(dust), make_context(positions=SHORT_AAPL))
        elsewhere = make_context(positions=(make_position("MSFT", 10),))
        assert not is_closing_sale(sell(dust), elsewhere)
        assert not is_closing_sale(sell(dust), make_context(positions=None))
        empty_row = make_context(positions=(make_position(qty=0.0),))
        assert not is_closing_sale(sell(dust), empty_row)
        # ... while the same dust sold out of a real long does close.
        assert is_closing_sale(sell(dust), holding(10))
        assert is_closing_sale(sell(dust), holding(dust))

    def test_a_sale_against_a_long_row_and_a_short_row_is_not_closing(self):
        long_row, short_row = make_position(qty=10), make_position(qty=-10, side="short")
        for rows in ((long_row, short_row), (short_row, long_row)):
            assert not is_closing_sale(sell(2), make_context(positions=rows))
        two_longs = make_context(positions=(make_position(qty=4), make_position(qty=6)))
        assert is_closing_sale(sell(10), two_longs)  # 4 + 6
        assert not is_closing_sale(sell(10.01), two_longs)
        assert not is_closing_sale(sell(5), make_context(positions=(make_position(qty=4),)))


class TestExposure:
    def test_nothing_held_or_pending_is_zero(self):
        assert exposure(make_context(), SYMBOL) == 0.0

    def test_unknown_positions_are_unknown(self):
        assert exposure(make_context(positions=None), SYMBOL) is None

    def test_market_value_of_the_position(self):
        context = make_context(positions=(make_position(qty=10, market_value=2150.10),))
        assert exposure(context, SYMBOL) == pytest.approx(2150.10)
        assert exposure(context, " aapl ") == pytest.approx(2150.10)
        assert exposure(context, "MSFT") == 0.0

    def test_a_short_counts_by_magnitude(self):
        short = make_position(qty=-10, side="short", market_value=-2000.0)
        context = make_context(positions=(short,))
        assert exposure(context, SYMBOL) == pytest.approx(2000.0)

    def test_quantity_times_price_when_the_market_value_is_unknown(self):
        context = make_context(positions=(make_position(qty=-10, side="short", market_value=None),))
        assert exposure(context, SYMBOL) == pytest.approx(2000.0)
        for bad in (NAN, INF):
            context = make_context(positions=(make_position(qty=10, market_value=bad),))
            assert exposure(context, SYMBOL) == pytest.approx(2000.0)

    def test_an_option_position_valued_from_its_price_carries_the_multiplier(self):
        position = make_position("SPY260821C00640000", 2, current_price=8.0, market_value=None)
        assert exposure(make_context(positions=(position,)), "SPY") == pytest.approx(1600.0)
        assert exposure(make_context(positions=(position,), contract_multiplier=10), "SPY") == (
            pytest.approx(160.0)
        )

    def test_neither_a_value_nor_a_price_is_unknown(self):
        for position in (
            make_position(current_price=None, market_value=None),
            make_position(qty=NAN, market_value=None),
            make_position(current_price=NAN, market_value=NAN),
        ):
            context = make_context(positions=(position,))
            assert exposure(context, SYMBOL) is None
            assert exposure(context, "MSFT") == 0.0  # only this underlying's rows matter

    def test_stock_and_options_on_one_underlying_add_up(self):
        positions = (
            make_position("SPY", 3, current_price=640.0),
            make_position("SPY260821C00640000", 2, market_value=1600.0),
            make_position("SPY260821P00630000", -1, side="short", market_value=-450.0),
            make_position("QQQ", 1, market_value=555.0),
        )
        context = make_context(positions=positions)
        assert exposure(context, "SPY") == pytest.approx(1920.0 + 1600.0 + 450.0)
        assert exposure(context, "QQQ") == pytest.approx(555.0)

    def test_open_orders_count_at_their_limit_price(self):
        orders = (
            make_order(quantity=5, limit_price=200.0),
            make_order("MSFT", id="o2", client_order_id="c2", quantity=1, limit_price=410.0),
        )
        context = holding(10, open_orders=orders)
        assert exposure(context, SYMBOL) == pytest.approx(2000.0 + 1000.0)
        assert exposure(context, "MSFT") == pytest.approx(410.0)

    def test_an_open_option_order_carries_the_multiplier(self):
        order = make_order("SPY260821C00640000", quantity=2, limit_price=8.0)
        assert exposure(make_context(open_orders=(order,)), "SPY") == pytest.approx(1600.0)
        assert exposure(make_context(open_orders=(order,), contract_multiplier=10), "SPY") == (
            pytest.approx(160.0)
        )

    @pytest.mark.parametrize("price", [None, 0.0, -5.0])
    def test_an_open_order_without_a_positive_limit_price_cannot_be_valued(self, price):
        context = holding(10, open_orders=(make_order(limit_price=price),))
        assert exposure(context, SYMBOL) is None
        assert exposure(context, "MSFT") == 0.0

    def test_an_overflowing_total_is_unknown_not_infinite(self):
        positions = (make_position(market_value=1e308), make_position(market_value=1e308))
        assert exposure(make_context(positions=positions), SYMBOL) is None

    def test_a_padded_position_counts_under_its_underlying(self):
        # A contract reported in the 21-character OCC form is still SPY exposure.
        position = make_position(PADDED_640, 2.0, current_price=8.0, market_value=1600.0)
        assert position.symbol == PADDED_640
        context = make_context(positions=(position,))
        assert exposure(context, "SPY") == pytest.approx(1600.0)
        assert exposure(context, "SPY   ") == pytest.approx(1600.0)
        assert exposure(context, "QQQ") == 0.0
        # Valued from its price, it carries the multiplier like its compact twin.
        priced = make_position(PADDED_640, 2.0, current_price=8.0, market_value=None)
        assert exposure(make_context(positions=(priced,)), "SPY") == pytest.approx(1600.0)
        # ... and one that cannot be valued makes SPY's exposure unknown.
        unvalued = make_position(PADDED_640, 2.0, current_price=None, market_value=None)
        assert exposure(make_context(positions=(unvalued,)), "SPY") is None

    def test_a_padded_open_order_counts_under_its_underlying(self):
        order = make_order(PADDED_640, quantity=2, limit_price=8.0)
        assert order.symbol == PADDED_640
        context = make_context(open_orders=(order,))
        assert exposure(context, "SPY") == pytest.approx(1600.0)
        assert exposure(context, "AAPL") == 0.0
        compact = make_order(CALL_640, quantity=2, limit_price=8.0)
        assert exposure(make_context(open_orders=(compact,)), "SPY") == exposure(context, "SPY")
        unpriced = make_order(PADDED_640, limit_price=None)
        assert exposure(make_context(open_orders=(unpriced,)), "SPY") is None

    def test_the_underlying_is_read_by_underlying_of_alone(self):
        # Ruling R4: underlying_of strips the root, so no second helper does.
        assert underlying_of(PADDED_640) == underlying_of(CALL_640) == "SPY"
        assert underlying_of(" spy   260821c00640000 ") == "SPY"
        assert not hasattr(measures, "_held_underlying")


class TestHeldUnderlyings:
    def test_nothing_is_an_empty_set_and_unknown_is_none(self):
        assert held_underlyings(make_context()) == frozenset()
        assert held_underlyings(make_context(positions=None)) is None
        # Unknown positions stay unknown even when open orders are known.
        assert held_underlyings(make_context(positions=None, open_orders=(make_order(),))) is None

    def test_positions_and_open_orders_by_underlying(self):
        context = make_context(
            positions=(
                make_position("SPY", 3),
                make_position("SPY260821C00640000", 2),
                make_position("aapl", 1),
            ),
            open_orders=(
                make_order("QQQ260821P00500000"),
                make_order("SPY", id="o2", client_order_id="c2"),
            ),
        )
        assert held_underlyings(context) == frozenset({"SPY", "AAPL", "QQQ"})

    def test_padded_rows_count_under_their_underlying(self):
        padded_position = make_context(positions=(make_position(PADDED_640, 2),))
        assert held_underlyings(padded_position) == frozenset({"SPY"})
        padded_order = make_context(open_orders=(make_order("QQQ   260821P00500000"),))
        assert held_underlyings(padded_order) == frozenset({"QQQ"})
        # Padded and compact rows of one underlying are one position, not two.
        both = make_context(
            positions=(make_position(PADDED_640, 2), make_position(CALL_640, 1)),
            open_orders=(make_order("SPY"),),
        )
        assert held_underlyings(both) == frozenset({"SPY"})


class TestDailyLossCap:
    def test_percent_of_start_of_day_equity(self):
        assert daily_loss_cap(make_context()) == pytest.approx(2000.0)
        limits = make_limits(daily_loss_limit_pct=1.5)
        context = make_context(start_of_day_equity=50_000.0, limits=limits)
        assert daily_loss_cap(context) == pytest.approx(750.0)

    def test_it_is_measured_from_start_of_day_equity_not_current_equity(self):
        assert daily_loss_cap(make_context(equity=1.0)) == pytest.approx(2000.0)

    @pytest.mark.parametrize("start", [None, NAN, INF, -INF, 0.0, -100_000.0])
    def test_unknown_or_non_positive_equity_has_no_cap(self, start):
        assert daily_loss_cap(make_context(start_of_day_equity=start)) is None


# --- measures: prices ---------------------------------------------------------


class TestSignedLimit:
    def test_a_buy_pays_and_a_sell_receives(self):
        assert signed_limit(make_proposal()) == 200.0
        assert signed_limit(sell()) == -200.0
        assert signed_limit(call_debit_spread()) == 4.40
        assert signed_limit(put_credit_spread()) == -2.00

    def test_a_market_order_has_none(self):
        assert signed_limit(make_proposal(order_type=OrderType.MARKET, limit_price=None)) is None
        # Even when a stray price came along with it.
        assert signed_limit(make_proposal(order_type=OrderType.MARKET)) is None

    @pytest.mark.parametrize("price", [None, 0.0, -1.0])
    def test_a_missing_or_non_positive_limit_is_unusable(self, price):
        assert signed_limit(make_proposal(limit_price=price)) is None
        assert signed_limit(sell(limit_price=price)) is None


class TestMidPriceEquity:
    def test_the_quote_mid(self):
        assert mid_price(make_proposal(), make_context()) == pytest.approx(200.0)
        context = make_context(quote=make_quote(bid=199.0, ask=202.0))
        assert mid_price(make_proposal(), context) == pytest.approx(200.5)

    def test_the_side_does_not_sign_an_equity_mid(self):
        assert mid_price(sell(), make_context()) == pytest.approx(200.0)

    def test_no_quote(self):
        assert mid_price(make_proposal(), make_context(quote=None)) is None

    def test_a_quote_for_another_symbol_is_not_this_symbols_quote(self):
        assert mid_price(make_proposal(), make_context(quote=make_quote("MSFT", 200.0))) is None

    def test_the_symbol_match_ignores_case(self):
        context = make_context(quote=make_quote("aapl"))
        assert mid_price(make_proposal(), context) == pytest.approx(200.0)

    @pytest.mark.parametrize("bad", [None, NAN, INF, -INF, 0.0, -1.0])
    def test_a_side_that_is_unknown_or_not_positive(self, bad):
        assert mid_price(make_proposal(), make_context(quote=make_quote(bid=bad))) is None
        assert mid_price(make_proposal(), make_context(quote=make_quote(ask=bad))) is None

    def test_a_crossed_quote_has_no_mid(self):
        crossed = make_context(quote=make_quote(bid=201.0, ask=199.0))
        assert mid_price(make_proposal(), crossed) is None

    def test_a_locked_quote_is_not_crossed(self):
        context = make_context(quote=make_quote(bid=200.0, ask=200.0))
        assert mid_price(make_proposal(), context) == 200.0

    def test_the_last_trade_is_never_a_substitute(self):
        quote = make_quote(bid=None, ask=None)
        assert quote.last == 200.0
        assert mid_price(make_proposal(), make_context(quote=quote)) is None


class TestMidPriceOption:
    def test_a_single_long_leg(self):
        proposal = long_call()
        assert mid_price(proposal, context_for(proposal)) == pytest.approx(8.00)

    def test_a_debit_structure_is_positive(self):
        proposal = call_debit_spread()
        assert mid_price(proposal, context_for(proposal)) == pytest.approx(4.40)

    def test_a_credit_structure_is_negative(self):
        credits = ((put_credit_spread, -2.00), (naked_short_call, -3.60), (iron_condor, -4.40))
        for build, mid in credits:
            assert mid_price(build(), context_for(build())) == pytest.approx(mid)

    def test_the_sign_comes_from_the_legs_not_the_proposals_side(self):
        mislabelled = put_credit_spread(side="buy")
        assert mid_price(mislabelled, context_for(mislabelled)) == pytest.approx(-2.00)

    def test_it_is_per_proposal_unit_whatever_the_quantity(self):
        proposal = call_debit_spread(quantity=5)
        assert mid_price(proposal, context_for(proposal)) == pytest.approx(4.40)

    def test_ratio_legs_are_weighted(self):
        # 1 × 640C bought at 8.00, 2 × 650C sold at 3.60: 8.00 − 7.20 per unit.
        ratio = make_option(RATIO_LEGS, side="buy", limit_price=0.8, quantity=3)
        assert [leg.quantity for leg in ratio.legs] == [3.0, 6.0]
        assert mid_price(ratio, context_for(ratio)) == pytest.approx(0.80)

    def test_no_legs(self):
        assert mid_price(option_proposal([]), make_context()) is None

    def test_a_leg_without_a_snapshot(self):
        proposal = call_debit_spread()
        only_first = leg_quotes_for(proposal)[:1]
        assert mid_price(proposal, context_for(proposal, leg_quotes=only_first)) is None
        assert mid_price(proposal, context_for(proposal, leg_quotes=())) is None

    def test_the_equity_quote_is_not_used_for_an_option(self):
        proposal = long_call()
        context = make_context(quote=make_quote(proposal.proposal.symbol, 8.0))
        assert mid_price(proposal, context) is None

    def test_a_zero_bid_is_a_valid_option_quote(self):
        proposal = long_call()
        quotes = (make_snapshot(proposal.legs[0].symbol, 0.0, bid=0.0, ask=0.10),)
        assert mid_price(proposal, context_for(proposal, leg_quotes=quotes)) == pytest.approx(0.05)
        both_zero = (make_snapshot(proposal.legs[0].symbol, 0.0, bid=0.0, ask=0.0),)
        assert mid_price(proposal, context_for(proposal, leg_quotes=both_zero)) == 0.0

    @pytest.mark.parametrize("bad", [None, NAN, INF, -INF, -0.01])
    def test_a_leg_side_that_is_unknown_or_negative(self, bad):
        proposal = call_debit_spread()
        good = leg_quotes_for(proposal)
        for override in ({"bid": bad}, {"ask": bad}):
            broken = make_snapshot(proposal.legs[1].symbol, 3.60, **override)
            assert mid_price(proposal, context_for(proposal, leg_quotes=(good[0], broken))) is None

    def test_a_crossed_leg(self):
        proposal = call_debit_spread()
        crossed = make_snapshot(proposal.legs[1].symbol, 3.60, bid=3.70, ask=3.50)
        quotes = (leg_quotes_for(proposal)[0], crossed)
        assert mid_price(proposal, context_for(proposal, leg_quotes=quotes)) is None

    def test_the_first_snapshot_for_a_symbol_wins(self):
        proposal = long_call()
        symbol = proposal.legs[0].symbol
        quotes = (make_snapshot(symbol, 8.0), make_snapshot(symbol, 99.0))
        assert mid_price(proposal, context_for(proposal, leg_quotes=quotes)) == pytest.approx(8.0)


class TestQuoteProblems:
    def test_nothing_is_missing_for_the_defaults(self):
        assert quote_problems(make_proposal(), make_context()) == ()
        for build in STRUCTURES:
            assert quote_problems(build(), context_for(build())) == ()

    def test_equity(self):
        proposal = make_proposal()
        assert quote_problems(proposal, make_context(quote=None)) == ("no quote for AAPL",)
        other = quote_problems(proposal, make_context(quote=make_quote("MSFT")))
        assert other == ("the quote is for MSFT, not AAPL",)
        one_sided = quote_problems(proposal, make_context(quote=make_quote(bid=None)))
        assert one_sided == ("the AAPL quote has no usable two-sided bid/ask",)

    def test_option_names_each_leg(self):
        proposal = call_debit_spread()
        first, second = (leg.symbol for leg in proposal.legs)
        assert quote_problems(proposal, context_for(proposal, leg_quotes=())) == (
            f"no quote for leg {first}",
            f"no quote for leg {second}",
        )
        quotes = (leg_quotes_for(proposal)[0], make_snapshot(second, 3.6, ask=None))
        assert quote_problems(proposal, context_for(proposal, leg_quotes=quotes)) == (
            f"the quote for leg {second} has no usable two-sided bid/ask",
        )
        assert quote_problems(option_proposal([]), make_context()) == (
            "the option proposal has no legs",
        )

    def test_it_is_empty_exactly_when_there_is_a_mid(self):
        proposal = call_debit_spread()
        for context in (
            context_for(proposal),
            context_for(proposal, leg_quotes=()),
            context_for(proposal, leg_quotes=leg_quotes_for(proposal)[:1]),
        ):
            has_mid = mid_price(proposal, context) is not None
            assert (quote_problems(proposal, context) == ()) == has_mid


class TestWorstCaseEntry:
    def test_an_equity_buy_pays_the_ask(self):
        assert worst_case_entry(make_proposal(), make_context()) == pytest.approx(200.02)

    def test_an_equity_sell_receives_the_bid(self):
        assert worst_case_entry(sell(), make_context()) == pytest.approx(-199.98)

    def test_only_the_needed_side_is_needed(self):
        assert worst_case_entry(make_proposal(), make_context(quote=make_quote(bid=None))) == (
            pytest.approx(200.02)
        )
        no_ask = make_context(quote=make_quote(ask=None))
        assert worst_case_entry(sell(), no_ask) == pytest.approx(-199.98)

    @pytest.mark.parametrize("bad", [None, NAN, INF, 0.0, -1.0])
    def test_a_missing_needed_side(self, bad):
        assert worst_case_entry(make_proposal(), make_context(quote=make_quote(ask=bad))) is None
        assert worst_case_entry(sell(), make_context(quote=make_quote(bid=bad))) is None

    def test_no_quote_or_another_symbols(self):
        assert worst_case_entry(make_proposal(), make_context(quote=None)) is None
        assert worst_case_entry(make_proposal(), make_context(quote=make_quote("MSFT"))) is None

    def test_an_option_buys_at_the_ask_and_sells_at_the_bid(self):
        debit = call_debit_spread()  # 640C 7.95/8.05, 650C 3.55/3.65
        assert worst_case_entry(debit, context_for(debit)) == pytest.approx(8.05 - 3.55)
        credit = put_credit_spread()  # 630P 4.45/4.55, 620P 2.45/2.55
        assert worst_case_entry(credit, context_for(credit)) == pytest.approx(2.55 - 4.45)

    def test_ratio_legs_are_weighted(self):
        ratio = make_option(RATIO_LEGS, side="buy", limit_price=0.8)
        assert worst_case_entry(ratio, context_for(ratio)) == pytest.approx(8.05 - 2 * 3.55)

    def test_an_option_without_legs_or_quotes(self):
        assert worst_case_entry(option_proposal([]), make_context()) is None
        proposal = call_debit_spread()
        assert worst_case_entry(proposal, context_for(proposal, leg_quotes=())) is None
        bidless = make_snapshot(proposal.legs[1].symbol, 3.6, bid=None)
        no_bid = (leg_quotes_for(proposal)[0], bidless)
        assert worst_case_entry(proposal, context_for(proposal, leg_quotes=no_bid)) is None

    def test_a_zero_bid_on_a_sold_leg_is_a_valid_fill_at_nothing(self):
        proposal = naked_short_call()
        quotes = (make_snapshot(proposal.legs[0].symbol, 0.05, bid=0.0, ask=0.10),)
        assert worst_case_entry(proposal, context_for(proposal, leg_quotes=quotes)) == 0.0

    @pytest.mark.parametrize("bad", [None, NAN, INF, -INF, -1.0, -8.0])
    def test_a_bought_legs_ask_that_is_unknown_or_negative(self, bad):
        market = {"order_type": OrderType.MARKET, "limit_price": None}
        single = long_call(**market)
        quotes = (make_snapshot(single.legs[0].symbol, 8.0, bid=-9.0, ask=bad),)
        assert worst_case_entry(single, context_for(single, leg_quotes=quotes)) is None
        spread = call_debit_spread(**market)  # buys the 640 call, sells the 650 call
        good = leg_quotes_for(spread)
        broken = make_snapshot(spread.legs[0].symbol, 8.0, ask=bad)
        context = context_for(spread, leg_quotes=(broken, good[1]))
        assert worst_case_entry(spread, context) is None
        assert notional(spread, context) is None  # so no garbage quote sizes the order

    @pytest.mark.parametrize("bad", [None, NAN, INF, -INF, -1.0, -5.0])
    def test_a_sold_legs_bid_that_is_unknown_or_negative(self, bad):
        market = {"order_type": OrderType.MARKET, "limit_price": None}
        credit = put_credit_spread(**market)  # sells the 630 put, buys the 620 put
        good = leg_quotes_for(credit)
        broken = make_snapshot(credit.legs[0].symbol, 4.5, bid=bad)
        context = context_for(credit, leg_quotes=(broken, good[1]))
        assert worst_case_entry(credit, context) is None
        naked = naked_short_call(**market)
        quotes = (make_snapshot(naked.legs[0].symbol, 3.6, bid=bad),)
        assert worst_case_entry(naked, context_for(naked, leg_quotes=quotes)) is None

    def test_only_the_side_a_leg_trades_at_is_needed(self):
        # A bought leg fills at its ask, a sold leg at its bid: the other side
        # of each quote may be anything.
        market = {"order_type": OrderType.MARKET, "limit_price": None}
        spread = call_debit_spread(**market)
        quotes = (
            make_snapshot(spread.legs[0].symbol, 8.0, bid=-1.0, ask=8.05),
            make_snapshot(spread.legs[1].symbol, 3.6, bid=3.55, ask=NAN),
        )
        context = context_for(spread, leg_quotes=quotes)
        assert worst_case_entry(spread, context) == pytest.approx(8.05 - 3.55)
        # A 0.0 ask on a bought leg is a (strange but) valid fill at nothing.
        free = (make_snapshot(spread.legs[0].symbol, 0.0, bid=0.0, ask=0.0), quotes[1])
        assert worst_case_entry(spread, context_for(spread, leg_quotes=free)) == (
            pytest.approx(-3.55)
        )


class TestEntryPrice:
    def test_a_limit_order_trades_at_its_limit(self):
        assert entry_price(make_proposal(), make_context(quote=None)) == 200.0
        assert entry_price(sell(), make_context(quote=None)) == -200.0

    def test_a_market_order_trades_at_the_worst_case(self):
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)
        assert entry_price(market, make_context()) == pytest.approx(200.02)
        assert entry_price(market, make_context(quote=None)) is None

    def test_an_unusable_limit_is_not_replaced_by_the_quote(self):
        assert entry_price(make_proposal(limit_price=None), make_context()) is None


class TestNotional:
    def test_equity_is_quantity_times_price(self):
        assert notional(make_proposal(), make_context()) == pytest.approx(400.0)
        assert notional(sell(), make_context()) == pytest.approx(400.0)  # a magnitude, not signed

    def test_equity_does_not_use_the_contract_multiplier(self):
        context = make_context(contract_multiplier=10)
        assert notional(make_proposal(), context) == pytest.approx(400.0)

    def test_an_equity_market_order_is_sized_at_the_worst_case(self):
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)
        assert notional(market, make_context()) == pytest.approx(2 * 200.02)
        market_sell = sell(order_type=OrderType.MARKET, limit_price=None)
        assert notional(market_sell, make_context()) == pytest.approx(2 * 199.98)
        assert notional(market, make_context(quote=None)) is None

    def test_an_unusable_limit_price_has_no_notional(self):
        assert notional(make_proposal(limit_price=None), make_context()) is None

    def test_an_option_bought_uses_the_limit_and_the_multiplier(self):
        assert notional(long_call(), context_for(long_call())) == pytest.approx(800.0)
        proposal = call_debit_spread(quantity=3)
        assert notional(proposal, context_for(proposal)) == pytest.approx(4.40 * 100 * 3)
        tenths = context_for(proposal, contract_multiplier=10)
        assert notional(proposal, tenths) == pytest.approx(132.0)

    def test_an_option_bought_needs_no_analysis(self):
        proposal = long_call()
        assert notional(proposal, context_for(proposal, analysis=None)) == pytest.approx(800.0)

    def test_a_debit_that_is_really_a_credit_is_inconsistent(self):
        # A market "buy" of a structure that pays a credit at the worst case.
        proposal = put_credit_spread(side="buy", order_type=OrderType.MARKET, limit_price=None)
        context = context_for(proposal)
        assert worst_case_entry(proposal, context) < 0
        assert notional(proposal, context) is None

    def test_an_option_sold_is_sized_by_its_max_loss(self):
        for build, max_loss in ((put_credit_spread, 800.0), (iron_condor, 560.0)):
            assert notional(build(), context_for(build())) == pytest.approx(max_loss)
        proposal = put_credit_spread(quantity=4)
        assert notional(proposal, context_for(proposal)) == pytest.approx(3200.0)

    def test_unlimited_risk_is_an_infinite_notional(self):
        proposal = naked_short_call()
        assert notional(proposal, context_for(proposal)) == INF
        capped = context_for(proposal).analysis.model_copy(update={"unlimited_risk": False})
        assert capped.max_loss is None
        assert notional(proposal, context_for(proposal, analysis=capped)) == INF
        infinite = capped.model_copy(update={"max_loss": INF})
        assert notional(proposal, context_for(proposal, analysis=infinite)) == INF

    def test_the_unlimited_risk_flag_alone_is_an_infinite_notional(self):
        # The flag is read on its own, apart from max_loss: an analysis that
        # says "unlimited" beside a finite loss is still unlimited.
        proposal = put_credit_spread()
        sound = context_for(proposal).analysis
        assert (sound.unlimited_risk, sound.max_loss) == (False, pytest.approx(800.0))
        assert notional(proposal, context_for(proposal)) == pytest.approx(800.0)
        odd = sound.model_copy(update={"unlimited_risk": True})
        assert odd.max_loss == pytest.approx(800.0)
        context = context_for(proposal, analysis=odd)
        assert notional(proposal, context) == INF
        found = results(proposal, context)
        for name in ("buying_power", "max_position_pct", "options_max_loss"):
            assert found[name].outcome is REJECT, name
            assert "unlimited" in found[name].detail, name
        assert first_reject(proposal, context) == "buying_power"

    def test_an_option_sold_without_an_analysis_is_unknown(self):
        proposal = put_credit_spread()
        assert notional(proposal, context_for(proposal, analysis=None)) is None

    @pytest.mark.parametrize("bad", [NAN, -INF, -1.0])
    def test_an_unusable_max_loss_is_unknown(self, bad):
        proposal = put_credit_spread()
        analysis = context_for(proposal).analysis.model_copy(update={"max_loss": bad})
        assert notional(proposal, context_for(proposal, analysis=analysis)) is None

    def test_an_overflow_is_unknown_not_infinite(self):
        huge = make_proposal(quantity=1e200, limit_price=1e200)
        assert notional(huge, make_context()) is None
        assert notional(long_call(), context_for(long_call(), contract_multiplier=10**400)) is None


# --- measures: option structure -----------------------------------------------


class TestLegDte:
    def test_days_from_the_market_date(self):
        leg = make_leg("buy", "call", 640.0)
        assert leg_dte(leg, make_context()) == 22
        assert leg_dte(leg, make_context(market_date=EXPIRY)) == 0
        assert leg_dte(leg, make_context(market_date=EXPIRY + timedelta(days=3))) == -3

    def test_it_counts_from_market_date_not_from_now(self):
        # 01:00 UTC on the 31st is still the 30th in New York.
        context = make_context(now=datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc))
        leg = make_leg("buy", "call", 640.0, expiration=date(2026, 7, 31))
        assert leg_dte(leg, context) == 1


class TestIsPlainEquity:
    def test_shares_and_nothing_else(self):
        assert is_plain_equity(make_proposal())
        assert is_plain_equity(sell())
        assert is_plain_equity(make_proposal(order_type=OrderType.MARKET, limit_price=None))
        assert is_plain_equity(make_proposal(symbol="BRK.B"))

    @pytest.mark.parametrize("build", STRUCTURES)
    def test_never_an_option(self, build):
        assert not is_plain_equity(build())
        assert not is_plain_equity(option_proposal([]))

    def test_not_an_equity_that_carries_legs(self):
        assert not is_plain_equity(make_proposal(legs=[make_leg("buy", "call", 640.0)]))

    def test_not_an_equity_named_by_an_option_contract(self):
        assert not is_plain_equity(make_proposal(symbol=CALL_640))
        assert not is_plain_equity(make_proposal(symbol=" spy260821p00640000 "))

    def test_it_is_exactly_an_equity_without_structure_problems(self):
        for proposal, _ in BROKEN_SHAPES.values():
            assert not is_plain_equity(proposal)
        for proposal in (make_proposal(), sell(), make_proposal(symbol="TSLA")):
            assert is_plain_equity(proposal) and structure_problems(proposal) == ()

    @pytest.mark.parametrize("symbol", ["AA PL", "AAPL\tX", "BRK B", "A\nB", PADDED_640])
    def test_not_an_equity_whose_symbol_contains_whitespace(self, symbol):
        spaced = make_proposal(symbol=symbol)
        assert spaced.proposal.symbol == symbol  # the store only strips the ends
        assert not is_plain_equity(spaced)
        assert not is_plain_equity(sell(symbol=symbol))
        assert structure_problems(spaced)

    @pytest.mark.parametrize("symbol", ["AAPL\u0663", "\uff21\uff21\uff30\uff2c", OTHER_DIGITS_640])
    def test_not_an_equity_whose_symbol_is_not_ascii(self, symbol):
        # Arabic-Indic digits, full-width letters: no ticker — whatever it looks like.
        odd = make_proposal(symbol=symbol)
        assert odd.proposal.symbol == symbol and not symbol.isascii()
        assert not is_plain_equity(odd)
        assert not is_plain_equity(sell(symbol=symbol))
        assert "contains non-ASCII characters" in " | ".join(structure_problems(odd))


class TestStructureProblems:
    @pytest.mark.parametrize("build", STRUCTURES)
    def test_the_named_structures_are_sound(self, build):
        assert structure_problems(build()) == ()

    def test_a_plain_equity_is_sound(self):
        assert structure_problems(make_proposal()) == ()
        assert structure_problems(sell()) == ()
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)
        assert structure_problems(market) == ()

    def test_an_equity_that_carries_legs(self):
        legs = [make_leg("buy", "call", 640.0, 1.5), make_leg("sell", "call", 650.0, index=1)]
        with_legs = make_proposal(legs=legs)
        assert with_legs.is_equity
        assert structure_problems(with_legs) == ("an equity proposal carries 2 option leg(s)",)

    def test_an_equity_named_by_an_option_contract(self):
        mislabelled = make_proposal(symbol=CALL_640)
        assert mislabelled.is_equity and not mislabelled.legs
        assert structure_problems(mislabelled) == (
            "the proposal is declared equity but its symbol SPY260821C00640000 is an option "
            "contract",
        )
        # Both findings when it is both.
        both = make_proposal(symbol=CALL_640, legs=[make_leg("buy", "call", 640.0)])
        assert len(structure_problems(both)) == 2

    @BROKEN
    def test_each_new_finding_on_its_own(self, proposal, finding):
        assert structure_problems(proposal) == (finding,)

    def test_the_strike_is_compared_with_the_float_noise_guard(self):
        # 0.1 + 0.2 is 0.30000000000000004: the same strike as the symbol's 0.3.
        noisy = make_leg("buy", "call", 0.1 + 0.2, symbol=occ("call", 0.3))
        assert noisy.strike != 0.3
        assert structure_problems(option_proposal([noisy])) == ()
        # ... and a thousandth of a dollar is another contract.
        other = make_leg("buy", "call", 640.0, symbol=occ("call", 640.001))
        (problem,) = structure_problems(option_proposal([other]))
        assert "names the 2026-08-21 640.001 call" in problem

    def test_a_leg_symbol_is_read_whatever_its_case(self):
        leg = make_leg("buy", "call", 640.0, symbol=" spy260821c00640000 ")
        assert leg.symbol == CALL_640 and structure_problems(option_proposal([leg])) == ()

    def test_every_misnamed_leg_is_listed(self):
        legs = [
            make_leg("buy", "call", 640.0, symbol=occ("call", 700.0)),
            make_leg("sell", "call", 650.0, index=1),
            make_leg("sell", "put", 630.0, index=2, symbol="SPY"),
        ]
        problems = structure_problems(option_proposal(legs))
        assert len(problems) == 2
        assert problems[0].startswith("leg SPY260821C00700000 names the 2026-08-21 700 call")
        assert problems[1] == "leg SPY is not an OCC option symbol"

    def test_the_brains_proposals_are_sound(self):
        """As the brain stores them: a single leg named by its OCC symbol or by
        its underlying; a multi-leg structure named by its underlying."""
        by_contract = long_call()
        assert by_contract.proposal.symbol == by_contract.legs[0].symbol == CALL_640
        assert structure_problems(by_contract) == ()
        by_underlying = long_call(symbol=UNDERLYING)
        assert by_underlying.proposal.symbol == "SPY" and len(by_underlying.legs) == 1
        assert structure_problems(by_underlying) == ()
        for build in (call_debit_spread, put_credit_spread, iron_condor):
            multi = build()
            assert multi.proposal.symbol == "SPY" and len(multi.legs) > 1
            assert structure_problems(multi) == ()
        short = naked_short_call()
        assert short.proposal.symbol == short.legs[0].symbol
        assert structure_problems(short) == ()
        assert structure_problems(naked_short_call(symbol=UNDERLYING)) == ()

    @pytest.mark.parametrize(
        ("symbol", "legs"),
        [
            ("AAPL", []),
            (CALL_640, [("buy", "call", 640.0)]),
            ("SPY", [("buy", "call", 640.0)]),
            ("SPY", [("buy", "call", 640.0), ("sell", "call", 650.0)]),
            (
                "SPY",
                [
                    ("sell", "call", 650.0), ("buy", "call", 660.0),
                    ("sell", "put", 630.0), ("buy", "put", 620.0),
                ],
            ),
            ("SPY", [("buy", "put", 642.5)]),  # a fractional strike survives the symbol
        ],
        ids=[
            "equity", "single leg by contract", "single leg by underlying", "vertical",
            "iron condor", "fractional strike",
        ],
    )
    def test_the_rows_the_brain_stores_are_sound(self, symbol, legs):
        """Through the brain's own ``trade_records`` — the rows the engine is
        handed: every shape the brain may store passes the judge of shape."""
        trade = TradeProposal(
            symbol=symbol,
            instrument=Instrument.OPTION if legs else Instrument.EQUITY,
            side=OrderSide.BUY,
            quantity=2,
            order_type=OrderType.LIMIT,
            limit_price=4.40,
            thesis="SYNTHETIC TEST DATA.",
            confidence=0.7,
            invalidation="SYNTHETIC TEST DATA.",
            legs=tuple(
                LegSpec(
                    symbol=occ(option_type, strike),
                    option_type=option_type,
                    side=side,
                    quantity=2,
                    strike=strike,
                    expiration=EXPIRY,
                )
                for side, option_type, strike in legs
            ),
        )
        proposal, rows = trade_records(
            "cycle-0001", trade, raw_output="{}", model_name="synthetic-model-v0",
            prompt_version="test-prompt-1",
        )
        stored = ProposalUnderReview(proposal=proposal, legs=tuple(rows))
        assert len(stored.legs) == len(legs)
        assert structure_problems(stored) == ()
        assert is_plain_equity(stored) is (not legs)

    def test_no_legs(self):
        assert structure_problems(option_proposal([])) == ("the option proposal has no legs",)

    def test_legs_on_more_than_one_underlying(self):
        problems = structure_problems(option_proposal(TWO_UNDERLYINGS))
        assert problems == ("legs on more than one underlying (QQQ, SPY)",)

    def test_the_proposals_symbol_is_on_another_underlying(self):
        problems = structure_problems(long_call(symbol="QQQ"))
        assert problems == ("the proposal is on QQQ but its legs are on SPY",)
        # An OCC proposal symbol must be its only leg's own: the root is not enough.
        assert structure_problems(long_call(symbol="SPY260821C00999000")) == (
            "the proposal is named by the option contract SPY260821C00999000, which is not "
            "its leg's SPY260821C00640000",
        )
        # On another root it is both findings.
        assert structure_problems(long_call(symbol="QQQ260821C00640000")) == (
            "the proposal is on QQQ but its legs are on SPY",
            "the proposal is named by the option contract QQQ260821C00640000, which is not "
            "its leg's SPY260821C00640000",
        )

    def test_legs_with_different_expirations(self):
        legs = [
            make_leg("buy", "call", 640.0),
            make_leg("sell", "call", 650.0, index=1, expiration=date(2026, 9, 18)),
        ]
        problems = structure_problems(option_proposal(legs))
        assert problems == ("legs with different expirations (2026-08-21, 2026-09-18)",)

    def test_a_fractional_number_of_contracts(self):
        legs = [make_leg("buy", "call", 640.0, 1.5), make_leg("sell", "call", 650.0, 2, index=1)]
        problems = structure_problems(option_proposal(legs))
        assert problems == (
            "leg SPY260821C00640000 quantity 1.5 is not a whole number of contracts",
        )

    def test_ratio_legs_are_sound(self):
        ratio = make_option(RATIO_LEGS, side="buy", limit_price=0.8)
        assert structure_problems(ratio) == ()

    def test_a_symbol_with_whitespace_is_a_finding_for_either_instrument(self):
        assert structure_problems(make_proposal(symbol="AA PL")) == (
            "symbol 'AA PL' is not a ticker: it contains whitespace",
        )
        assert structure_problems(sell(symbol="BRK\tB")) == (
            "symbol 'BRK\\tB' is not a ticker: it contains whitespace",
        )
        # An "equity" named by a padded contract is both findings.
        assert structure_problems(make_proposal(symbol=PADDED_640)) == (
            "the proposal is declared equity but its symbol SPY   260821C00640000 is an "
            "option contract",
            "symbol 'SPY   260821C00640000' is not in the compact OCC form: it contains "
            "whitespace",
        )
        spaced_leg = make_leg("buy", "call", 640.0, symbol="SPY 640C")
        assert structure_problems(option_proposal([spaced_leg])) == (
            "the proposal is on SPY but its legs are on SPY 640C",
            "leg SPY 640C is not an OCC option symbol",
            "symbol 'SPY 640C' is not a ticker: it contains whitespace",
        )

    def test_the_padded_occ_form_is_a_finding_wherever_it_is_named(self):
        finding = (
            "symbol 'SPY   260821C00640000' is not in the compact OCC form: it contains "
            "whitespace"
        )
        # Named by the proposal and by its leg: one finding (a symbol is listed once),
        # and nothing else is wrong — underlying_of reads the root as SPY.
        everywhere = padded_call()
        assert everywhere.underlying == "SPY" and everywhere.underlyings == ("SPY",)
        assert structure_problems(everywhere) == (finding,)
        # The leg alone (the proposal named by its underlying).
        assert structure_problems(padded_call(symbol=UNDERLYING)) == (finding,)
        # The proposal alone: its symbol is then not its (compact) leg's either.
        own = long_call(symbol=PADDED_640)
        assert structure_problems(own) == (
            "the proposal is named by the option contract SPY   260821C00640000, which is "
            "not its leg's SPY260821C00640000",
            finding,
        )
        # One padded leg in a spread, and every padded leg listed.
        legs = [
            make_leg("buy", "call", 640.0, symbol=PADDED_640),
            make_leg("sell", "call", 650.0, index=1, symbol=padded(occ("call", 650.0))),
        ]
        problems = structure_problems(option_proposal(legs))
        assert problems == (
            finding,
            "symbol 'SPY   260821C00650000' is not in the compact OCC form: it contains "
            "whitespace",
        )

    def test_a_symbol_outside_ascii_is_a_finding_for_either_instrument(self):
        # Another spelling of the 640 call: it parses, its strike is the leg's,
        # and it is not the symbol the broker, the quotes or the books use.
        assert is_occ(OTHER_DIGITS_640) and underlying_of(OTHER_DIGITS_640) == "SPY"
        finding = (
            "symbol 'SPY260821C\\u0660\\u0660\\u0666\\u0664\\u0660\\u0660\\u0660\\u0660' is not "
            "in the compact OCC form: it contains non-ASCII characters"
        )
        leg = make_leg("buy", "call", 640.0, symbol=OTHER_DIGITS_640)
        everywhere = option_proposal(
            [leg], symbol=OTHER_DIGITS_640, quantity=1.0, limit_price=8.00
        )
        assert everywhere.proposal.symbol == leg.symbol == OTHER_DIGITS_640
        assert structure_problems(everywhere) == (finding,)
        # An "equity" named by it is both findings; a ticker gets the ticker's words.
        declared_equity = structure_problems(make_proposal(symbol=OTHER_DIGITS_640))
        assert len(declared_equity) == 2 and declared_equity[1] == finding
        assert structure_problems(make_proposal(symbol="AAPL\u0663")) == (
            "symbol 'AAPL\\u0663' is not a ticker: it contains non-ASCII characters",
        )
        # Padded as well: one finding that names both flaws.
        both = padded(OTHER_DIGITS_640)
        (problem,) = structure_problems(
            option_proposal([make_leg("buy", "call", 640.0, symbol=both)], symbol=both)
        )
        assert problem.endswith("it contains whitespace and non-ASCII characters")
        # And it stands between the shape and every tier, like the padded form.
        context = make_context(
            quote=None,
            leg_quotes=(make_snapshot(OTHER_DIGITS_640, 8.00),),
            analysis=analysis_for(long_call()),
        )
        assert non_pass(everywhere, context) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }

    def test_every_problem_is_listed(self):
        legs = [
            make_leg("buy", "call", 640.0, 0.5, underlying="QQQ"),
            make_leg(
                "sell", "call", 650.0, index=1, underlying="IWM", expiration=date(2026, 9, 18)
            ),
        ]
        problems = structure_problems(option_proposal(legs))
        assert len(problems) == 4
        assert problems[0].startswith("legs on more than one underlying")
        assert problems[1] == "the proposal is on SPY but its legs are on IWM, QQQ"
        assert problems[2].startswith("legs with different expirations")
        assert "not a whole number" in problems[3]


# --- rules 1–4: the account-level stops ---------------------------------------


class TestKillSwitch:
    def test_off_passes(self):
        assert run(rules.kill_switch).outcome is PASS

    def test_on_rejects(self):
        result = run(rules.kill_switch, context=make_context(controls=Controls(kill_switch=True)))
        assert result.outcome is REJECT
        assert "kill switch is ON" in result.detail

    def test_it_ignores_the_proposal(self):
        on = make_context(controls=Controls(kill_switch=True))
        assert run(rules.kill_switch, naked_short_call(), on) == run(rules.kill_switch, None, on)


class TestHalted:
    @staticmethod
    def halted_until(until, **controls):
        context = make_context(controls=Controls(halt_until=until, **controls))
        return run(rules.halted, context=context)

    def test_no_halt_passes(self):
        assert run(rules.halted).outcome is PASS

    def test_a_halt_in_the_future_rejects(self):
        result = self.halted_until(NEXT_OPEN)
        assert result.outcome is REJECT
        assert "2026-07-31T13:30:00+00:00" in result.detail

    def test_boundary_a_halt_that_ends_now_has_expired(self):
        assert self.halted_until(NOW).outcome is PASS
        assert self.halted_until(NOW + timedelta(seconds=1)).outcome is REJECT
        assert self.halted_until(NOW + timedelta(microseconds=1)).outcome is REJECT
        assert self.halted_until(NOW - timedelta(seconds=1)).outcome is PASS

    def test_an_expired_halt_says_when_it_expired(self):
        result = self.halted_until(NOW - timedelta(hours=2))
        assert result.outcome is PASS and "2026-07-30T13:00:00+00:00" in result.detail

    def test_the_halt_is_compared_as_an_instant_not_as_wall_time(self):
        eastern = ZoneInfo("America/New_York")
        assert self.halted_until(datetime(2026, 7, 30, 11, 0, tzinfo=eastern)).outcome is PASS
        assert self.halted_until(datetime(2026, 7, 30, 11, 0, 1, tzinfo=eastern)).outcome is REJECT

    def test_an_unreadable_halt_rejects(self):
        problems = ("halt_until row is missing",)
        result = self.halted_until(None, halt_unknown=True, problems=problems)
        assert result.outcome is REJECT
        assert "cannot prove trading is not halted" in result.detail
        assert "halt_until row is missing" in result.detail

    def test_an_unreadable_halt_rejects_even_without_a_stated_problem(self):
        assert self.halted_until(None, halt_unknown=True).outcome is REJECT
        assert self.halted_until(NOW - timedelta(days=1), halt_unknown=True).outcome is REJECT

    def test_a_positive_pnl_does_not_lift_a_halt(self):
        context = make_context(
            controls=Controls(halt_until=NEXT_OPEN), daily_pnl=5000.0, equity=105_000.0
        )
        assert run(rules.halted, context=context).outcome is REJECT
        assert run(rules.daily_loss_limit, context=context).outcome is PASS


class TestMarketHours:
    @staticmethod
    def at(clock, **overrides):
        return run(rules.market_hours, context=make_context(clock=clock, **overrides))

    def test_an_open_market_passes(self):
        result = run(rules.market_hours)
        assert result.outcome is PASS and "2026-07-30T20:00:00+00:00" in result.detail

    def test_a_closed_market_rejects(self):
        result = self.at(make_clock(False))
        assert result.outcome is REJECT
        assert "closed" in result.detail and "2026-07-31T13:30:00+00:00" in result.detail

    def test_three_in_the_morning(self):
        three_am = datetime(2026, 7, 30, 7, 0, tzinfo=timezone.utc)  # 03:00 New York
        clock = make_clock(False, next_open=datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc))
        assert self.at(clock, now=three_am).outcome is REJECT

    def test_no_clock_rejects(self):
        result = self.at(None)
        assert result.outcome is REJECT and "cannot confirm the market is open" in result.detail

    def test_a_closed_clock_without_a_next_open_still_rejects(self):
        result = self.at(make_clock(False, next_open=None))
        assert result.outcome is REJECT and "next open unknown" in result.detail

    def test_boundary_a_clock_that_says_open_past_its_own_close_is_stale(self):
        assert self.at(make_clock(), now=NEXT_CLOSE - timedelta(seconds=1)).outcome is PASS
        at_close = self.at(make_clock(), now=NEXT_CLOSE)
        assert at_close.outcome is REJECT and "stale" in at_close.detail
        assert self.at(make_clock(), now=NEXT_CLOSE + timedelta(seconds=1)).outcome is REJECT

    def test_an_open_clock_without_a_close_time_passes(self):
        result = self.at(make_clock(next_close=None))
        assert result.outcome is PASS and "close time unknown" in result.detail

    def test_a_naive_close_time_is_taken_as_utc(self):
        naive = NEXT_CLOSE.replace(tzinfo=None)
        assert self.at(make_clock(next_close=naive)).outcome is PASS
        assert self.at(make_clock(next_close=naive), now=NEXT_CLOSE).outcome is REJECT

    def test_how_long_ago_the_clock_was_fetched_is_not_this_rules_business(self):
        # Only is_open and next_close are consulted.
        late = NEXT_CLOSE - timedelta(minutes=1)
        assert self.at(make_clock(), now=late).outcome is PASS


class TestDailyLossLimit:
    @staticmethod
    def at(pnl=0.0, **overrides):
        return run(rules.daily_loss_limit, context=make_context(daily_pnl=pnl, **overrides))

    def test_flat_passes_with_the_headroom(self):
        result = self.at(0.0)
        assert result.outcome is PASS and result.halt_until is None
        assert "$2,000.00 of headroom" in result.detail
        assert "-$2,000.00" in result.detail and "2%" in result.detail

    def test_a_gain_passes(self):
        result = self.at(1500.0)
        assert result.outcome is PASS and "$3,500.00 of headroom" in result.detail

    def test_a_loss_inside_the_cap_passes(self):
        result = self.at(-1500.0)
        assert result.outcome is PASS and "$500.00 of headroom" in result.detail

    def test_boundary_at_the_cap_trips(self):
        assert self.at(-1999.99).outcome is PASS
        assert self.at(-2000.00).outcome is REJECT
        assert self.at(-2000.01).outcome is REJECT

    def test_float_noise_around_the_cap_counts_as_at_the_cap(self):
        # "At or below": a P&L within float noise of the cap has reached it.
        assert self.at(-2000.0 + 1e-10).outcome is REJECT
        assert self.at(-2000.0 - 1e-10).outcome is REJECT
        cap = 2.0 / 100 * 100_000.3
        assert self.at(-cap, start_of_day_equity=100_000.3).outcome is REJECT

    def test_a_trip_halts_until_the_next_open_and_says_so(self):
        result = self.at(-2500.0)
        assert result.outcome is REJECT
        assert result.halt_until == NEXT_OPEN
        assert result.halt_until.utcoffset() == timedelta(0)
        for text in ("-$2,500.00", "-$2,000.00", "2%", "$100,000.00", "2026-07-31T13:30:00+00:00"):
            assert text in result.detail
        assert "(risk_limits.daily_loss_limit_pct)" in result.detail

    def test_the_cap_is_a_percentage_of_start_of_day_equity(self):
        half = {"start_of_day_equity": 50_000.0}
        assert self.at(-999.99, **half).outcome is PASS
        assert self.at(-1000.00, **half).outcome is REJECT
        # Current equity is not the base: it has already fallen with the loss.
        assert self.at(-1999.0, equity=10.0).outcome is PASS

    def test_the_percentage_comes_from_the_limits(self):
        one = {"limits": make_limits(daily_loss_limit_pct=1.0)}
        assert self.at(-999.99, **one).outcome is PASS
        assert self.at(-1000.00, **one).outcome is REJECT
        everything = {"limits": make_limits(daily_loss_limit_pct=100.0)}
        assert self.at(-99_999.99, **everything).outcome is PASS
        assert self.at(-100_000.0, **everything).outcome is REJECT

    def test_a_closed_clocks_next_open_is_used_too(self):
        monday = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
        result = self.at(-2500.0, clock=make_clock(False, next_open=monday))
        assert result.halt_until == monday

    @pytest.mark.parametrize(
        "clock",
        [
            None,
            make_clock(next_open=None),
            make_clock(next_open=NOW),  # not in the future
            make_clock(next_open=NOW - timedelta(hours=1)),
        ],
        ids=["no clock", "no next open", "next open is now", "next open is past"],
    )
    def test_the_fallback_halt_when_the_clock_has_no_future_open(self, clock):
        result = self.at(-2500.0, clock=clock)
        assert result.outcome is REJECT
        assert result.halt_until == NOW + timedelta(hours=24)
        assert "2026-07-31T15:00:00+00:00" in result.detail

    def test_the_fallback_length_comes_from_the_limits(self):
        result = self.at(-2500.0, clock=None, limits=make_limits(halt_fallback_hours=6.5))
        assert result.halt_until == NOW + timedelta(hours=6, minutes=30)

    def test_a_next_open_one_second_ahead_is_used(self):
        soon = NOW + timedelta(seconds=1)
        assert self.at(-2500.0, clock=make_clock(next_open=soon)).halt_until == soon

    def test_a_naive_next_open_is_taken_as_utc(self):
        result = self.at(-2500.0, clock=make_clock(next_open=NEXT_OPEN.replace(tzinfo=None)))
        assert result.halt_until == NEXT_OPEN

    def test_a_next_open_in_another_zone_is_reported_in_utc(self):
        eastern = datetime(2026, 7, 31, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        result = self.at(-2500.0, clock=make_clock(next_open=eastern))
        assert result.halt_until == NEXT_OPEN
        assert result.halt_until.utcoffset() == timedelta(0)

    @pytest.mark.parametrize("hours", [LARGEST_FINITE, 1e300, 1e9])
    def test_an_unrepresentable_fallback_halts_to_the_end_of_the_calendar(self, hours):
        result = self.at(-2500.0, clock=None, limits=make_limits(halt_fallback_hours=hours))
        assert result.outcome is REJECT
        assert result.halt_until == datetime.max.replace(tzinfo=timezone.utc)

    @UNKNOWN
    def test_an_unknown_pnl_rejects_without_a_halt(self, unknown):
        result = self.at(unknown)
        assert result.outcome is REJECT and result.halt_until is None
        assert "cannot evaluate" in result.detail and "P&L is unknown" in result.detail

    @UNKNOWN
    def test_an_unknown_start_of_day_equity_rejects_without_a_halt(self, unknown):
        result = self.at(-5000.0, start_of_day_equity=unknown)
        assert result.outcome is REJECT and result.halt_until is None
        assert "cannot evaluate" in result.detail and "start-of-day equity" in result.detail

    @pytest.mark.parametrize("start", [0.0, -100_000.0])
    def test_a_non_positive_start_of_day_equity_rejects_without_a_halt(self, start):
        result = self.at(0.0, start_of_day_equity=start)
        assert result.outcome is REJECT and result.halt_until is None

    def test_both_unknown_names_both(self):
        result = self.at(None, start_of_day_equity=None)
        assert "P&L is unknown" in result.detail and "start-of-day equity" in result.detail

    def test_a_trip_under_a_longer_stored_halt_names_the_halt_in_force(self):
        # An earlier trip without a clock stored now + 24h; this one computes
        # the next open, which is sooner. The audit line must not name an
        # instant earlier than the halt the `halted` rule reports.
        in_force = NOW + timedelta(hours=24)
        assert in_force > NEXT_OPEN
        context = make_context(daily_pnl=-3000.0, controls=Controls(halt_until=in_force))
        result = run(rules.daily_loss_limit, context=context)
        assert result.outcome is REJECT
        assert "a halt until 2026-07-31T15:00:00+00:00 is already in force" in result.detail
        assert "2026-07-31T13:30:00+00:00" not in result.detail
        assert "trading is halted until" not in result.detail
        # The result still carries the instant this trip computed: whether to
        # write it is the engine's decision (it never shortens a halt).
        assert result.halt_until == NEXT_OPEN
        assert "2026-07-31T15:00:00+00:00" in run(rules.halted, context=context).detail

    @pytest.mark.parametrize(
        "stored",
        [None, NOW - timedelta(days=1), NOW + timedelta(minutes=5), NEXT_OPEN],
        ids=["none", "expired", "shorter", "the same instant"],
    )
    def test_a_trip_under_no_longer_halt_names_its_own_instant(self, stored):
        context = make_context(daily_pnl=-3000.0, controls=Controls(halt_until=stored))
        result = run(rules.daily_loss_limit, context=context)
        assert result.halt_until == NEXT_OPEN
        assert "trading is halted until 2026-07-31T13:30:00+00:00" in result.detail
        assert "already in force" not in result.detail

    def test_a_trip_on_an_unreadable_halt_beside_a_longer_instant_announces_that_instant(self):
        # What only the stricter of two readings holds: one could not read the
        # stored halt, the other gave an instant beyond this trip's. The engine
        # WRITES that instant then (a finite halt, no shorter than either
        # reading) — so the audit line, which is also the risk_limit_tripped
        # event's message, must not say it was "already in force".
        in_force = NOW + timedelta(hours=24)
        controls = Controls(
            halt_until=in_force, halt_unknown=True, problems=("halt_until is unreadable",)
        )
        context = make_context(daily_pnl=-3000.0, controls=controls)
        result = run(rules.daily_loss_limit, context=context)
        assert result.outcome is REJECT
        assert result.detail.endswith(
            " (risk_limits.daily_loss_limit_pct): trading is halted until"
            " 2026-07-31T15:00:00+00:00, the later instant another reading of the stored halt"
            " gave (one reading of it was unreadable)"
        )
        assert "already in force" not in result.detail
        assert "2026-07-31T13:30:00+00:00" not in result.detail  # not the trip's own instant
        # the result still carries the instant this trip computed; the engine takes the later
        assert result.halt_until == NEXT_OPEN

    @pytest.mark.parametrize(
        "stored",
        [None, NOW - timedelta(days=1), NOW + timedelta(minutes=5), NEXT_OPEN],
        ids=["none", "expired", "shorter", "the same instant"],
    )
    def test_a_trip_on_an_unreadable_halt_with_no_longer_instant_names_its_own(self, stored):
        controls = Controls(halt_until=stored, halt_unknown=True)
        context = make_context(daily_pnl=-3000.0, controls=controls)
        result = run(rules.daily_loss_limit, context=context)
        assert result.halt_until == NEXT_OPEN
        assert result.detail.endswith(": trading is halted until 2026-07-31T13:30:00+00:00")
        assert "another reading" not in result.detail

    def test_a_readable_longer_halt_is_still_said_to_be_in_force(self):
        # The wording for an unreadable halt is for that case alone.
        in_force = NOW + timedelta(hours=24)
        context = make_context(daily_pnl=-3000.0, controls=Controls(halt_until=in_force))
        result = run(rules.daily_loss_limit, context=context)
        assert result.detail.endswith(
            ": a halt until 2026-07-31T15:00:00+00:00 is already in force"
        )
        assert "another reading" not in result.detail

    def test_a_pass_within_a_fraction_of_a_cent_of_the_cap_shows_the_difference(self):
        result = self.at(-1999.996)
        assert result.outcome is PASS
        # money() prints both figures as -$2,000.00: the detail adds what separates them.
        assert "today's P&L -$2,000.00 is above the loss cap -$2,000.00 by $0.004," in (
            result.detail
        )
        assert result.detail.endswith(": $0.004 of headroom")
        assert "$0.00 of headroom" not in result.detail
        # A cent apart, the two figures already differ: nothing is added.
        plain = self.at(-1999.99)
        assert "-$1,999.99 is above the loss cap -$2,000.00, 2%" in plain.detail
        assert " by $" not in plain.detail and plain.detail.endswith(": $0.01 of headroom")


class TestDailyLossHaltCoversTheComingSession:
    """Ruling R1: a trip while the market is closed BEFORE today's session
    halts until that session's close — a halt to the opening bell would let a
    recovery by the open reopen trading the same day. Every other trip halts
    to the next open, as before."""

    EIGHT_AM = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)  # 08:00 New York
    TODAY_OPEN = datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc)  # 09:30 New York
    TODAY_CLOSE = NEXT_CLOSE  # 16:00 New York, the same trading date
    BEFORE_OPEN = make_clock(False, next_open=TODAY_OPEN, next_close=TODAY_CLOSE)

    @classmethod
    def tripped(cls, **overrides) -> RuleResult:
        fields = {"now": cls.EIGHT_AM, "clock": cls.BEFORE_OPEN, "daily_pnl": -2500.0}
        return run(rules.daily_loss_limit, context=make_context(**{**fields, **overrides}))

    def test_a_trip_at_eight_in_the_morning_halts_until_the_close(self):
        context = make_context(now=self.EIGHT_AM, clock=self.BEFORE_OPEN, daily_pnl=-2500.0)
        assert context.market_date == context.next_open_date == MARKET_DATE
        result = run(rules.daily_loss_limit, context=context)
        assert result.outcome is REJECT
        assert result.halt_until == self.TODAY_CLOSE
        assert result.halt_until != self.TODAY_OPEN
        assert result.halt_until.utcoffset() == timedelta(0)
        assert rules._halt_instant(context) == self.TODAY_CLOSE

    def test_the_detail_names_the_close_and_the_session_it_covers(self):
        result = self.tripped()
        assert (
            "trading is halted until 2026-07-30T20:00:00+00:00, the close of the session "
            "that opens at 2026-07-30T13:30:00+00:00"
        ) in result.detail
        for text in ("-$2,500.00", "-$2,000.00", "(risk_limits.daily_loss_limit_pct)"):
            assert text in result.detail

    def test_without_the_next_open_date_it_halts_to_the_next_open(self):
        # The rule cannot tell the open is today's: it falls back to the open.
        result = self.tripped(next_open_date=None)
        assert result.halt_until == self.TODAY_OPEN
        assert "trading is halted until 2026-07-30T13:30:00+00:00" in result.detail
        assert "the close of the session" not in result.detail

    @pytest.mark.parametrize(
        "next_close",
        [None, TODAY_OPEN, TODAY_OPEN - timedelta(seconds=1), EIGHT_AM - timedelta(hours=1)],
        ids=["unknown", "at the open", "before the open", "in the past"],
    )
    def test_a_close_that_is_unknown_or_not_after_the_open_halts_to_the_next_open(
        self, next_close
    ):
        clock = make_clock(False, next_open=self.TODAY_OPEN, next_close=next_close)
        result = self.tripped(clock=clock)
        assert result.halt_until == self.TODAY_OPEN
        assert "the close of the session" not in result.detail

    def test_a_close_one_second_after_the_open_is_used(self):
        close = self.TODAY_OPEN + timedelta(seconds=1)
        clock = make_clock(False, next_open=self.TODAY_OPEN, next_close=close)
        assert self.tripped(clock=clock).halt_until == close

    def test_an_after_hours_trip_halts_to_the_next_open(self):
        # 17:00 New York: the next open is tomorrow's, the next close tomorrow's too.
        five_pm = datetime(2026, 7, 30, 21, 0, tzinfo=timezone.utc)
        tomorrow_close = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        clock = make_clock(False, next_open=NEXT_OPEN, next_close=tomorrow_close)
        context = make_context(now=five_pm, clock=clock, daily_pnl=-2500.0)
        assert context.next_open_date == date(2026, 7, 31) != context.market_date
        result = run(rules.daily_loss_limit, context=context)
        assert result.halt_until == NEXT_OPEN
        assert "the close of the session" not in result.detail

    def test_the_trading_date_not_the_utc_date_decides(self):
        tomorrow_close = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        clock = make_clock(False, next_open=NEXT_OPEN, next_close=tomorrow_close)
        # 21:00 New York on the 30th is already the 31st in UTC — the same UTC
        # date as the next open, but the session belongs to tomorrow's trading date.
        evening = datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc)
        after_hours = make_context(
            now=evening, market_date=exchange_date(evening), clock=clock, daily_pnl=-2500.0
        )
        assert after_hours.market_date == date(2026, 7, 30)
        assert run(rules.daily_loss_limit, context=after_hours).halt_until == NEXT_OPEN
        # 01:00 New York on the 31st: now the open IS on the trading date.
        small_hours = datetime(2026, 7, 31, 5, 0, tzinfo=timezone.utc)
        pre_session = make_context(
            now=small_hours, market_date=exchange_date(small_hours), clock=clock,
            daily_pnl=-2500.0,
        )
        assert pre_session.market_date == pre_session.next_open_date == date(2026, 7, 31)
        assert run(rules.daily_loss_limit, context=pre_session).halt_until == tomorrow_close

    def test_a_weekend_trip_halts_to_mondays_open(self):
        saturday = datetime(2026, 8, 1, 15, 0, tzinfo=timezone.utc)
        monday_open = datetime(2026, 8, 3, 13, 30, tzinfo=timezone.utc)
        monday_close = datetime(2026, 8, 3, 20, 0, tzinfo=timezone.utc)
        clock = make_clock(False, next_open=monday_open, next_close=monday_close)
        context = make_context(
            now=saturday, market_date=exchange_date(saturday), clock=clock, daily_pnl=-2500.0
        )
        assert run(rules.daily_loss_limit, context=context).halt_until == monday_open

    def test_an_intraday_trip_halts_to_the_next_open(self):
        result = run(rules.daily_loss_limit, context=make_context(daily_pnl=-2500.0))
        assert result.halt_until == NEXT_OPEN
        assert "the close of the session" not in result.detail
        # An open clock is never "before the session", whatever its times and
        # the dates claim: the halt is to its next open, not to its close.
        soon = NOW + timedelta(hours=1)
        odd = make_context(daily_pnl=-2500.0, clock=make_clock(True, next_open=soon))
        assert odd.next_open_date == odd.market_date and soon < odd.clock.next_close
        assert run(rules.daily_loss_limit, context=odd).halt_until == soon
        closed = make_context(daily_pnl=-2500.0, clock=make_clock(False, next_open=soon))
        assert run(rules.daily_loss_limit, context=closed).halt_until == NEXT_CLOSE

    def test_a_closed_clock_whose_open_is_not_ahead_falls_back(self):
        # The clock is stale: its "next open" has passed. No session to cover.
        clock = make_clock(False, next_open=self.EIGHT_AM, next_close=self.TODAY_CLOSE)
        result = self.tripped(clock=clock, next_open_date=MARKET_DATE)
        assert result.halt_until == self.EIGHT_AM + timedelta(hours=24)

    def test_naive_and_zoned_clock_times_are_read_as_instants(self):
        naive = make_clock(
            False,
            next_open=self.TODAY_OPEN.replace(tzinfo=None),
            next_close=self.TODAY_CLOSE.replace(tzinfo=None),
        )
        assert self.tripped(clock=naive).halt_until == self.TODAY_CLOSE
        eastern = ZoneInfo("America/New_York")
        zoned = make_clock(
            False,
            next_open=datetime(2026, 7, 30, 9, 30, tzinfo=eastern),
            next_close=datetime(2026, 7, 30, 16, 0, tzinfo=eastern),
        )
        result = self.tripped(clock=zoned)
        assert result.halt_until == self.TODAY_CLOSE
        assert result.halt_until.utcoffset() == timedelta(0)

    def test_a_recovery_by_the_opening_bell_does_not_reopen_trading_that_day(self):
        halt = self.tripped().halt_until
        # 09:31 New York the same day: the market is open and the P&L is back.
        recovered = {
            "now": self.TODAY_OPEN + timedelta(minutes=1),
            "clock": make_clock(next_open=NEXT_OPEN, next_close=self.TODAY_CLOSE),
            "daily_pnl": -1000.0,
        }
        context = make_context(controls=Controls(halt_until=halt), **recovered)
        found = results(context=context)
        assert found["daily_loss_limit"].outcome is PASS
        assert found["market_hours"].outcome is PASS
        assert found["halted"].outcome is REJECT
        assert first_reject(make_proposal(), context) == "halted"
        assert non_pass(context=context) == {"halted": REJECT}
        # ... up to the close; a halt to the opening bell would have expired.
        last_minute = make_context(
            controls=Controls(halt_until=halt), **{**recovered, "now": halt - timedelta(seconds=1)}
        )
        assert run(rules.halted, context=last_minute).outcome is REJECT
        to_the_open = make_context(controls=Controls(halt_until=self.TODAY_OPEN), **recovered)
        assert passing(context=to_the_open)

    def test_a_longer_halt_already_in_force_is_still_named(self):
        in_force = self.TODAY_CLOSE + timedelta(days=1)
        result = self.tripped(controls=Controls(halt_until=in_force))
        assert result.halt_until == self.TODAY_CLOSE
        assert "a halt until 2026-07-31T20:00:00+00:00 is already in force" in result.detail

    def test_the_limits_report_sees_the_same_result(self):
        context = make_context(now=self.EIGHT_AM, clock=self.BEFORE_OPEN, daily_pnl=-2500.0)
        by_name = {result.name: result for result in account_rules(context)}
        assert by_name["daily_loss_limit"] == run(rules.daily_loss_limit, context=context)
        assert by_name["daily_loss_limit"].halt_until == self.TODAY_CLOSE


# --- rules 5–7: what may be traded --------------------------------------------


class TestNoTradeList:
    @staticmethod
    def listed(*symbols, proposal=None):
        context = make_context(limits=make_limits(no_trade_list=list(symbols)))
        return run(rules.no_trade_list, proposal, context)

    def test_an_empty_list_passes(self):
        assert run(rules.no_trade_list).outcome is PASS

    def test_a_listed_symbol_rejects(self):
        result = self.listed("AAPL")
        assert result.outcome is REJECT and "AAPL" in result.detail

    def test_another_symbol_listed_passes(self):
        assert self.listed("TSLA", "GME").outcome is PASS

    def test_the_list_is_case_insensitive(self):
        assert self.listed(" aapl ").outcome is REJECT

    def test_an_option_on_a_listed_underlying_rejects(self):
        assert self.listed("SPY", proposal=long_call()).outcome is REJECT
        assert self.listed("SPY", proposal=iron_condor()).outcome is REJECT
        assert self.listed("QQQ", proposal=iron_condor()).outcome is PASS

    def test_a_listed_contract_rejects(self):
        proposal = iron_condor()
        result = self.listed(proposal.legs[3].symbol, proposal=proposal)
        assert result.outcome is REJECT and proposal.legs[3].symbol in result.detail

    def test_a_leg_on_a_listed_underlying_rejects_whatever_the_proposal_symbol_says(self):
        legs = [make_leg("buy", "call", 640.0, underlying="GME")]
        assert self.listed("GME", proposal=option_proposal(legs)).outcome is REJECT

    def test_a_sale_of_a_listed_symbol_rejects_closing_or_not(self):
        listed = make_limits(no_trade_list=["AAPL"])
        assert run(rules.no_trade_list, sell(), make_context(limits=listed)).outcome is REJECT
        closing = holding(10, limits=listed)
        assert is_closing_sale(sell(2), closing)
        assert run(rules.no_trade_list, sell(2), closing).outcome is REJECT

    def test_a_padded_contract_on_a_listed_underlying_rejects(self):
        # Ruling R4. With the watchlist rule off, the no-trade list is the only
        # rule that names the underlying: the padded form must not slip past it.
        proposal = padded_call()
        limits = make_limits(no_trade_list=["SPY"], watchlist_only=False)
        context = padded_context(proposal, limits=limits)
        found = results(proposal, context)
        assert found["no_trade_list"].outcome is REJECT
        assert found["no_trade_list"].detail == (
            "SPY is on the no-trade list (risk_limits.no_trade_list)"
        )
        assert first_reject(proposal, context) == "no_trade_list"
        # ... and its shape is rejected on its own account, never auto-tier.
        assert found["options_max_loss"].outcome is REJECT
        assert "is not in the compact OCC form" in found["options_max_loss"].detail
        assert found["options_escalate"].outcome is ESCALATE
        assert found["auto_tier"].outcome is ESCALATE
        # The compact twin is rejected by the same rule with the same words.
        twin = long_call()
        compact = run(rules.no_trade_list, twin, context_for(twin, limits=limits))
        assert compact == found["no_trade_list"]

    def test_a_padded_leg_of_a_structure_named_by_its_underlying_rejects(self):
        proposal = padded_call(symbol=UNDERLYING)
        assert self.listed("SPY", proposal=proposal).outcome is REJECT
        assert self.listed("QQQ", proposal=proposal).outcome is PASS
        passed = self.listed("QQQ", proposal=proposal)
        assert passed.detail == "SPY is not on the no-trade list (risk_limits.no_trade_list)"

    def test_padded_and_compact_forms_of_a_listed_contract_match_each_other(self):
        # The list names the contract compactly, the proposal pads it — and the reverse.
        assert self.listed(CALL_640, proposal=padded_call()).outcome is REJECT
        assert self.listed(PADDED_640, proposal=long_call()).outcome is REJECT
        assert self.listed(PADDED_640, proposal=padded_call()).outcome is REJECT
        assert self.listed(occ("call", 650.0), proposal=padded_call()).outcome is PASS
        # A padded root on the list is its ticker.
        assert self.listed("SPY   ", proposal=long_call()).outcome is REJECT
        assert self.listed("spy", proposal=padded_call()).outcome is REJECT
        # A strike in another script's digits is the same contract, listed or proposed.
        leg = make_leg("buy", "call", 640.0, symbol=OTHER_DIGITS_640)
        odd = option_proposal([leg], symbol=OTHER_DIGITS_640)
        assert self.listed(CALL_640, proposal=odd).outcome is REJECT
        assert self.listed(OTHER_DIGITS_640, proposal=long_call()).outcome is REJECT
        assert self.listed(occ("call", 650.0), proposal=odd).outcome is PASS

    def test_an_equity_named_by_a_padded_contract_on_a_listed_underlying_rejects(self):
        mislabelled = make_proposal(symbol=PADDED_640, limit_price=8.00)
        result = self.listed("SPY", proposal=mislabelled)
        assert result.outcome is REJECT and "SPY is on the no-trade list" in result.detail


class TestWatchlistOnly:
    def test_a_watchlist_symbol_passes(self):
        assert run(rules.watchlist_only).outcome is PASS

    def test_a_symbol_outside_the_watchlist_rejects(self):
        result = run(rules.watchlist_only, make_proposal(symbol="TSLA"))
        assert result.outcome is REJECT and "TSLA" in result.detail

    def test_disabled_passes_and_says_so(self):
        context = make_context(limits=make_limits(watchlist_only=False))
        result = run(rules.watchlist_only, make_proposal(symbol="TSLA"), context)
        assert result.outcome is PASS and "disabled (risk_limits.watchlist_only)" in result.detail

    def test_an_option_is_judged_by_its_underlying(self):
        assert run(rules.watchlist_only, long_call()).outcome is PASS
        outside = long_call(underlying="IWM")
        assert run(rules.watchlist_only, outside).outcome is REJECT

    def test_every_underlying_must_be_in_the_watchlist(self):
        legs = [
            make_leg("buy", "call", 640.0),
            make_leg("sell", "call", 650.0, index=1, underlying="IWM"),
        ]
        result = run(rules.watchlist_only, option_proposal(legs))
        assert result.outcome is REJECT and "IWM" in result.detail and "SPY" not in result.detail

    def test_an_empty_watchlist_rejects_everything(self):
        assert run(rules.watchlist_only, context=make_context(watchlist=())).outcome is REJECT

    def test_the_watchlist_is_case_insensitive(self):
        assert run(rules.watchlist_only, context=make_context(watchlist=("aapl",))).outcome is PASS

    def test_a_short_sale_outside_the_watchlist_rejects(self):
        outside = make_proposal(symbol="TSLA", side=OrderSide.SELL)
        result = run(rules.watchlist_only, outside, context_for(outside))
        assert result.outcome is REJECT and "TSLA is not in the watchlist" in result.detail

    def test_a_closing_sale_outside_the_watchlist_rejects(self):
        # Selling a held long is still a trade in a symbol outside the watchlist.
        outside = make_proposal(symbol="TSLA", side=OrderSide.SELL)
        context = context_for(outside, positions=(make_position("TSLA", 10),))
        assert is_closing_sale(outside, context)
        result = run(rules.watchlist_only, outside, context)
        assert result.outcome is REJECT and "TSLA is not in the watchlist" in result.detail
        assert first_reject(outside, context) == "watchlist_only"
        assert non_pass(outside, context) == {"watchlist_only": REJECT, "auto_tier": ESCALATE}
        # ... and inside it, the same sale passes.
        assert run(rules.watchlist_only, sell(2), holding(10)).outcome is PASS

    def test_a_padded_contract_is_judged_by_its_ticker(self):
        proposal = padded_call()
        result = run(rules.watchlist_only, proposal, padded_context(proposal))
        assert result.outcome is PASS
        assert result.detail == "SPY is in the watchlist (risk_limits.watchlist_only)"
        iwm = padded(occ("call", 640.0, underlying="IWM"))
        outside = option_proposal([make_leg("buy", "call", 640.0, symbol=iwm)], symbol=iwm)
        assert run(rules.watchlist_only, outside).outcome is REJECT


class TestInvalidationPresent:
    def test_a_real_invalidation_passes(self):
        assert run(rules.invalidation_present).outcome is PASS

    @pytest.mark.parametrize("text", ["", " ", "\n\t  \r\n"])
    @pytest.mark.parametrize(
        "build",
        [make_proposal, sell, long_call, call_debit_spread, put_credit_spread, naked_short_call],
    )
    def test_an_empty_or_blank_invalidation_rejects(self, build, text):
        # Whatever is proposed — shares or options, bought or sold.
        proposal = build(invalidation=text)
        result = run(rules.invalidation_present, proposal, context_for(proposal))
        assert result.outcome is REJECT and "no invalidation" in result.detail

    @pytest.mark.parametrize("text", ["", "  "])
    def test_a_closing_sale_without_an_invalidation_rejects(self, text):
        closing = sell(2, invalidation=text)
        assert is_closing_sale(closing, holding(10))
        assert run(rules.invalidation_present, closing, holding(10)).outcome is REJECT
        assert non_pass(closing, holding(10)) == {"invalidation_present": REJECT}

    @pytest.mark.parametrize("build", [long_call, put_credit_spread])
    def test_an_option_without_an_invalidation_is_rejected_by_this_rule_first(self, build):
        proposal = build(invalidation="   ")
        assert first_reject(proposal, context_for(proposal)) == "invalidation_present"

    def test_any_text_at_all_passes(self):
        assert run(rules.invalidation_present, make_proposal(invalidation=" x ")).outcome is PASS


# --- rules 8–11: sizing and counts --------------------------------------------


class TestBuyingPower:
    @staticmethod
    def with_power(proposal=None, **overrides):
        proposal = make_proposal() if proposal is None else proposal
        return run(rules.buying_power, proposal, context_for(proposal, **overrides))

    def test_ample_buying_power_passes(self):
        result = run(rules.buying_power)
        assert result.outcome is PASS
        assert "$400.00" in result.detail and "$200,000.00" in result.detail

    def test_boundary_exactly_enough_passes_one_cent_short_rejects(self):
        assert self.with_power(buying_power=400.00).outcome is PASS
        result = self.with_power(buying_power=399.99)
        assert result.outcome is REJECT
        assert "$400.00" in result.detail and "$399.99" in result.detail

    def test_no_or_negative_buying_power_rejects(self):
        assert self.with_power(buying_power=0.0).outcome is REJECT
        assert self.with_power(buying_power=-5000.0).outcome is REJECT

    @pytest.mark.parametrize("power", [0.0, -0.0, -1e-10, -5000.0])
    def test_nothing_to_spend_admits_nothing_however_small_the_notional(self, power):
        # A notional of $2e-10 is "not above" zero by float noise alone: with
        # no buying power at all, a proposal that needs any is rejected.
        tiny = make_proposal(quantity=1e-12, limit_price=200.0)
        assert not exceeds(notional(tiny, context_for(tiny)), 0.0)
        result = self.with_power(tiny, buying_power=power)
        assert result.outcome is REJECT
        assert "cannot be met: buying power is" in result.detail
        assert not passing(tiny, context_for(tiny, buying_power=power))
        # The smallest positive buying power is enough for it.
        assert self.with_power(tiny, buying_power=1e-9).outcome is PASS

    def test_an_option_with_nothing_to_spend_rejects(self):
        result = self.with_power(long_call(), options_buying_power=0.0)
        assert result.outcome is REJECT
        assert "cannot be met: options buying power is $0.00" in result.detail
        fallback = self.with_power(long_call(), options_buying_power=None, cash=0.0)
        assert fallback.outcome is REJECT
        assert "the lesser of buying power and cash is $0.00" in fallback.detail
        riskless = put_credit_spread()
        analysis = context_for(riskless).analysis.model_copy(update={"max_loss": 0.0})
        nothing = context_for(riskless, analysis=analysis, options_buying_power=0.0)
        assert notional(riskless, nothing) == 0.0
        assert run(rules.buying_power, riskless, nothing).outcome is REJECT

    @DUST
    def test_a_sale_of_dust_with_no_long_is_not_exempt(self, dust):
        # Not a closing sale: unknown or absent buying power still rejects it.
        for power in (None, NAN, 0.0, -1.0):
            for positions in ((), SHORT_AAPL):
                context = make_context(positions=positions, buying_power=power)
                result = run(rules.buying_power, sell(dust), context)
                assert result.outcome is REJECT and "closing sale" not in result.detail

    def test_a_rejection_by_a_fraction_of_a_cent_shows_the_difference(self):
        result = self.with_power(buying_power=399.996)
        assert result.outcome is REJECT
        assert result.detail == "notional $400.00 is above buying power $400.00 by $0.004"
        # A cent apart the figures already differ: nothing is added.
        plain = self.with_power(buying_power=399.99)
        assert plain.detail == "notional $400.00 is above buying power $399.99"

    def test_float_noise_does_not_reject(self):
        spec = make_proposal(quantity=10, limit_price=100.1)
        assert self.with_power(spec, buying_power=1001.0).outcome is PASS
        noisy = make_proposal(quantity=3, limit_price=100.01)
        assert 3 * 100.01 > 300.03
        assert self.with_power(noisy, buying_power=300.03).outcome is PASS
        assert self.with_power(noisy, buying_power=300.02).outcome is REJECT

    @UNKNOWN
    def test_unknown_buying_power_rejects(self, unknown):
        result = self.with_power(buying_power=unknown)
        assert result.outcome is REJECT and "buying power is unknown" in result.detail

    def test_an_unknown_notional_rejects(self):
        result = self.with_power(make_proposal(limit_price=None))
        assert result.outcome is REJECT and "notional cannot be computed" in result.detail

    def test_an_equity_does_not_draw_on_options_buying_power_or_cash(self):
        assert self.with_power(options_buying_power=None, cash=None).outcome is PASS
        assert self.with_power(options_buying_power=0.0, cash=0.0).outcome is PASS

    def test_a_short_sale_consumes_buying_power(self):
        assert self.with_power(sell(), buying_power=400.00).outcome is PASS
        assert self.with_power(sell(), buying_power=399.99).outcome is REJECT

    def test_a_closing_sale_consumes_none(self):
        for power in (0.0, None, NAN, -1.0):
            result = run(rules.buying_power, sell(10), holding(10, buying_power=power))
            assert result.outcome is PASS and "closing sale" in result.detail

    def test_a_sale_that_is_not_shown_to_close_is_not_exempt(self):
        assert run(rules.buying_power, sell(11), holding(10, buying_power=0.0)).outcome is REJECT
        unknown = make_context(positions=None, buying_power=0.0)
        assert run(rules.buying_power, sell(1), unknown).outcome is REJECT

    def test_a_market_order_is_sized_at_the_ask(self):
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)  # ask 200.02
        assert self.with_power(market, buying_power=400.04).outcome is PASS
        assert self.with_power(market, buying_power=400.03).outcome is REJECT
        assert self.with_power(market, quote=None).outcome is REJECT

    def test_an_option_draws_on_options_buying_power(self):
        result = self.with_power(long_call())
        assert result.outcome is PASS and "options buying power $100,000.00" in result.detail
        assert self.with_power(long_call(), options_buying_power=800.00).outcome is PASS
        assert self.with_power(long_call(), options_buying_power=799.99).outcome is REJECT
        # Margin buying power does not help an option.
        margin = {"options_buying_power": 10.0, "buying_power": 1e9}
        assert self.with_power(long_call(), **margin).outcome is REJECT

    def test_an_option_falls_back_to_the_lesser_of_buying_power_and_cash(self):
        fallback = {"options_buying_power": None}
        result = self.with_power(long_call(), cash=800.00, **fallback)
        assert result.outcome is PASS and "lesser of buying power and cash" in result.detail
        assert self.with_power(long_call(), cash=799.99, **fallback).outcome is REJECT
        assert self.with_power(long_call(), buying_power=799.99, **fallback).outcome is REJECT

    @UNKNOWN
    def test_an_option_with_no_known_figure_rejects(self, unknown):
        for missing in ({"cash": unknown}, {"buying_power": unknown}):
            result = self.with_power(long_call(), options_buying_power=unknown, **missing)
            assert result.outcome is REJECT and "unknown" in result.detail

    @UNKNOWN
    def test_unknown_options_buying_power_alone_uses_the_fallback(self, unknown):
        assert self.with_power(long_call(), options_buying_power=unknown).outcome is PASS

    def test_a_credit_structure_is_sized_by_its_max_loss(self):
        assert self.with_power(put_credit_spread(), options_buying_power=800.00).outcome is PASS
        assert self.with_power(put_credit_spread(), options_buying_power=799.99).outcome is REJECT
        assert self.with_power(put_credit_spread(), analysis=None).outcome is REJECT

    def test_unlimited_risk_exceeds_any_buying_power(self):
        result = self.with_power(naked_short_call(), options_buying_power=1e15)
        assert result.outcome is REJECT and "unlimited" in result.detail


class TestMaxPositionPct:
    @staticmethod
    def sized(proposal=None, **overrides):
        proposal = make_proposal() if proposal is None else proposal
        return run(rules.max_position_pct, proposal, context_for(proposal, **overrides))

    def test_a_small_position_passes_and_shows_its_sums(self):
        result = run(rules.max_position_pct)
        assert result.outcome is PASS
        for text in ("$400.00", "AAPL", "$5,000.00", "5%", "$100,000.00"):
            assert text in result.detail

    def test_boundary_at_the_cap_passes_one_cent_over_rejects(self):
        assert self.sized(make_proposal(quantity=25)).outcome is PASS  # 5,000.00 of 5,000.00
        over = self.sized(make_proposal(quantity=25, limit_price=200.0004))  # 5,000.01
        assert over.outcome is REJECT
        assert self.sized(make_proposal(quantity=26)).outcome is REJECT

    def test_existing_exposure_counts(self):
        held = (make_position(qty=10, market_value=2000.00),)
        # 3,000 + 2,000
        assert self.sized(make_proposal(quantity=15), positions=held).outcome is PASS
        held_more = (make_position(qty=10, market_value=2000.01),)
        result = self.sized(make_proposal(quantity=15), positions=held_more)
        assert result.outcome is REJECT
        assert "$3,000.00" in result.detail and "$2,000.01" in result.detail
        assert "$5,000.01" in result.detail

    def test_exposure_is_per_underlying(self):
        elsewhere = (make_position("MSFT", 100, market_value=41_000.0),)
        assert self.sized(make_proposal(quantity=25), positions=elsewhere).outcome is PASS

    def test_options_on_the_underlying_count_against_a_stock_proposal(self):
        proposal = make_proposal(symbol="SPY", quantity=5, limit_price=640.0)  # 3,200
        calls = (make_position("SPY260821C00640000", 2, market_value=1800.0),)
        assert self.sized(proposal, positions=calls).outcome is PASS  # 5,000.00
        calls = (make_position("SPY260821C00640000", 2, market_value=1800.01),)
        assert self.sized(proposal, positions=calls).outcome is REJECT

    def test_open_orders_on_the_underlying_count(self):
        pending = (make_order(quantity=23, limit_price=200.0),)  # 4,600 pending
        assert self.sized(open_orders=pending).outcome is PASS  # + 400 = 5,000
        pending = (make_order(quantity=23, limit_price=200.01),)
        assert self.sized(open_orders=pending).outcome is REJECT

    def test_the_percentage_comes_from_the_limits_and_the_base_is_equity(self):
        one = make_limits(max_position_pct=1.0)
        assert self.sized(make_proposal(quantity=5), limits=one).outcome is PASS  # 1,000 of 1,000
        assert self.sized(make_proposal(quantity=6), limits=one).outcome is REJECT
        assert self.sized(make_proposal(quantity=5), equity=20_000.0).outcome is PASS  # cap 1,000
        assert self.sized(make_proposal(quantity=6), equity=20_000.0).outcome is REJECT

    def test_float_noise_does_not_reject(self):
        whole = make_limits(max_position_pct=100.0)
        noisy = make_proposal(quantity=3, limit_price=100.01)
        assert self.sized(noisy, limits=whole, equity=300.03).outcome is PASS
        assert self.sized(noisy, limits=whole, equity=300.02).outcome is REJECT
        # A sum that is over by an ulp: 100.1 + 200.11 > 300.21 in floats.
        one_share = make_proposal(quantity=1, limit_price=100.1)
        held = (make_position(qty=1, market_value=200.11),)
        assert 100.1 + 200.11 > 300.21
        assert self.sized(one_share, limits=whole, equity=300.21, positions=held).outcome is PASS
        spec = make_proposal(quantity=10, limit_price=100.1)
        assert self.sized(spec, limits=whole, equity=1001.0).outcome is PASS

    @UNKNOWN
    def test_unknown_equity_rejects(self, unknown):
        result = self.sized(equity=unknown)
        assert result.outcome is REJECT and "equity is unknown" in result.detail

    @pytest.mark.parametrize("equity", [0.0, -50_000.0])
    def test_non_positive_equity_rejects(self, equity):
        result = self.sized(equity=equity)
        assert result.outcome is REJECT and "equity is unknown or not positive" in result.detail
        # ...even for a structure that commits nothing: there is no cap to be inside.
        proposal = put_credit_spread()
        riskless = context_for(proposal).analysis.model_copy(update={"max_loss": 0.0})
        context = context_for(proposal, analysis=riskless, equity=equity)
        assert notional(proposal, context) == 0.0
        assert run(rules.max_position_pct, proposal, context).outcome is REJECT

    def test_an_unknown_notional_rejects(self):
        result = self.sized(make_proposal(limit_price=None))
        assert result.outcome is REJECT and "notional cannot be computed" in result.detail

    # Every kind of proposal the cap sizes: shares bought and sold short, an
    # option bought, an option structure sold. Unknown exposure is never zero
    # for any of them.
    SIZED = {
        "equity buy": make_proposal,
        "short sale": sell,
        "long call": long_call,
        "credit spread": put_credit_spread,
        "debit spread": call_debit_spread,
    }
    EVERY_KIND = pytest.mark.parametrize("build", SIZED.values(), ids=SIZED.keys())

    @EVERY_KIND
    def test_unknown_positions_reject(self, build):
        proposal = build()
        result = self.sized(proposal, positions=None)
        assert result.outcome is REJECT
        assert f"exposure in {proposal.underlying} is unknown" in result.detail
        assert self.sized(proposal).outcome is PASS  # known to hold nothing: it fits

    @EVERY_KIND
    def test_a_position_that_cannot_be_valued_rejects(self, build):
        proposal = build()
        underlying = proposal.underlying
        # A short row, so that an equity sale cannot be taken for a closing sale.
        rows = [
            make_position(underlying, -10, "short", current_price=None, market_value=None),
            make_position(underlying, NAN, market_value=NAN),
            make_position(underlying, -10, "short", current_price=NAN, market_value=INF),
        ]
        if proposal.is_option:  # an option position on the underlying, too
            rows.append(make_position(CALL_640, 2, current_price=None, market_value=None))
        for row in rows:
            result = self.sized(proposal, positions=(row,))
            assert result.outcome is REJECT
            assert f"exposure in {underlying} is unknown" in result.detail
        # ...but only when it is on this underlying.
        other = (make_position("MSFT", current_price=None, market_value=None),)
        assert self.sized(proposal, positions=other).outcome is PASS

    @EVERY_KIND
    def test_an_open_order_that_cannot_be_valued_rejects(self, build):
        proposal = build()
        underlying = proposal.underlying
        # On another contract of the underlying (or its shares): not a symbol
        # the proposal itself trades, yet exposure in the same underlying.
        pending = "SPY260821C00700000" if proposal.is_option else underlying
        for price in (None, 0.0, -1.0):
            unpriced = (make_order(pending, limit_price=price),)
            result = self.sized(proposal, open_orders=unpriced)
            assert result.outcome is REJECT
            assert f"exposure in {underlying} is unknown" in result.detail
        elsewhere = (make_order("MSFT", limit_price=None),)
        assert self.sized(proposal, open_orders=elsewhere).outcome is PASS

    def test_an_option_named_by_its_contract_counts_the_exposure_in_its_underlying(self):
        # A single-leg option is named by its OCC symbol; what it adds to is
        # the exposure in SPY, however that is held.
        proposal = long_call()  # 8.00 x 100 = 800.00; the cap is 5,000.00
        assert proposal.proposal.symbol == CALL_640 and proposal.underlying == "SPY"

        def stock(value):
            return (make_position("SPY", 10, current_price=value / 10, market_value=value),)

        at_cap = self.sized(proposal, positions=stock(4200.00))
        assert at_cap.outcome is PASS
        assert "existing SPY exposure $4,200.00" in at_cap.detail
        over = self.sized(proposal, positions=stock(4200.01))
        assert over.outcome is REJECT
        assert "notional $800.00 + existing SPY exposure $4,200.01 = $5,000.01" in over.detail
        # An option position on another SPY contract.
        def contract(value):
            return (make_position("SPY260821P00630000", 3, market_value=value),)

        assert self.sized(proposal, positions=contract(4200.00)).outcome is PASS
        assert self.sized(proposal, positions=contract(4200.01)).outcome is REJECT
        # An open order on another SPY contract: 6 x 7.00 x 100 = 4,200.00.
        other = "SPY260821C00650000"
        pending = (make_order(other, quantity=6, limit_price=7.00),)
        assert self.sized(proposal, open_orders=pending).outcome is PASS
        pending = (make_order(other, quantity=6, limit_price=7.01),)  # 4,206.00
        result = self.sized(proposal, open_orders=pending)
        assert result.outcome is REJECT and "existing SPY exposure $4,206.00" in result.detail
        # A credit structure, named by its underlying, is held to the same sum.
        spread = put_credit_spread()  # max loss 800.00
        assert self.sized(spread, positions=stock(4200.00)).outcome is PASS
        assert self.sized(spread, positions=stock(4200.01)).outcome is REJECT

    def test_a_rejection_by_a_fraction_of_a_cent_shows_the_difference(self):
        over = self.sized(make_proposal(quantity=25, limit_price=200.0001))  # 5,000.0025
        assert over.outcome is REJECT
        assert "= $5,000.00 is above the cap $5,000.00 by $0.0025," in over.detail
        plain = self.sized(make_proposal(quantity=25, limit_price=200.0004))  # 5,000.01
        assert "= $5,000.01 is above the cap $5,000.00, 5% of equity" in plain.detail
        assert " by $" not in plain.detail

    def test_a_closing_sale_passes_whatever_the_numbers(self):
        for overrides in ({}, {"equity": None}, {"equity": 1.0}):
            result = run(rules.max_position_pct, sell(10), holding(10, **overrides))
            assert result.outcome is PASS and "closing sale" in result.detail

    def test_a_short_sale_is_sized_like_any_position(self):
        assert self.sized(sell(25)).outcome is PASS
        assert self.sized(sell(26)).outcome is REJECT

    def test_an_option_bought_uses_the_multiplier_and_the_limit_price(self):
        assert self.sized(long_call(quantity=6)).outcome is PASS  # 8.00 × 100 × 6 = 4,800
        result = self.sized(long_call(quantity=7))  # 5,600
        assert result.outcome is REJECT and "$5,600.00" in result.detail and "SPY" in result.detail
        assert self.sized(long_call(quantity=7), contract_multiplier=10).outcome is PASS

    def test_a_credit_structure_uses_its_max_loss(self):
        assert self.sized(put_credit_spread(quantity=6)).outcome is PASS  # 800 × 6 = 4,800
        result = self.sized(put_credit_spread(quantity=7))
        assert result.outcome is REJECT and "$5,600.00" in result.detail
        assert self.sized(put_credit_spread(), analysis=None).outcome is REJECT

    def test_unlimited_risk_exceeds_any_cap(self):
        whole = make_limits(max_position_pct=100)
        result = self.sized(naked_short_call(), equity=1e15, limits=whole)
        assert result.outcome is REJECT and "unlimited" in result.detail


class TestMaxOpenPositions:
    @staticmethod
    def positions(*symbols):
        return tuple(make_position(symbol, 1) for symbol in symbols)

    def test_a_first_position_passes(self):
        result = run(rules.max_open_positions)
        assert result.outcome is PASS and "0 of 5" in result.detail

    def test_boundary_at_the_cap_a_new_underlying_rejects(self):
        four = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT"))
        assert run(rules.max_open_positions, context=four).outcome is PASS
        five = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT", "TSLA"))
        result = run(rules.max_open_positions, context=five)
        assert result.outcome is REJECT and "5 of 5" in result.detail and "AAPL" in result.detail
        six = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT", "TSLA", "GME"))
        assert run(rules.max_open_positions, context=six).outcome is REJECT

    def test_adding_to_a_held_underlying_passes_at_the_cap(self):
        five = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT", "AAPL"))
        result = run(rules.max_open_positions, context=five)
        assert result.outcome is PASS and "already held" in result.detail
        over = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT", "AAPL", "GME"))
        assert run(rules.max_open_positions, context=over).outcome is PASS

    def test_positions_are_counted_as_distinct_underlyings(self):
        # Five rows, three underlyings.
        rows = self.positions("SPY", "SPY260821C00640000", "SPY260821P00630000", "QQQ", "NVDA")
        result = run(rules.max_open_positions, context=make_context(positions=rows))
        assert result.outcome is PASS and "3 of 5" in result.detail

    def test_an_option_on_a_held_underlying_adds_no_position(self):
        five = make_context(positions=self.positions("SPY", "QQQ", "NVDA", "MSFT", "TSLA"))
        assert run(rules.max_open_positions, long_call(), five).outcome is PASS
        assert run(rules.max_open_positions, long_call(underlying="IWM"), five).outcome is REJECT

    def test_pending_orders_count(self):
        orders = tuple(
            make_order(symbol, id=f"o{n}", client_order_id=f"c{n}")
            for n, symbol in enumerate(("SPY", "QQQ"))
        )
        context = make_context(positions=self.positions("NVDA", "MSFT", "TSLA"), open_orders=orders)
        assert run(rules.max_open_positions, context=context).outcome is REJECT
        # A pending order on this very underlying means it is already counted.
        pending = (make_order(SYMBOL),)
        four = self.positions("SPY", "QQQ", "NVDA", "MSFT")
        context = make_context(positions=four, open_orders=pending)
        assert run(rules.max_open_positions, context=context).outcome is PASS

    @pytest.mark.parametrize(
        "build",
        [make_proposal, sell, long_call, put_credit_spread, naked_short_call],
    )
    def test_unknown_positions_reject(self, build):
        # Whatever is proposed: unknown positions are never "none held".
        proposal = build()
        result = run(rules.max_open_positions, proposal, context_for(proposal, positions=None))
        assert result.outcome is REJECT and "positions are unknown" in result.detail
        # ... even with open orders known, and even on an underlying pending there.
        pending = (make_order(proposal.underlying),)
        context = context_for(proposal, positions=None, open_orders=pending)
        assert run(rules.max_open_positions, proposal, context).outcome is REJECT

    def test_the_cap_comes_from_the_limits(self):
        one = make_limits(max_open_positions=1)
        assert run(rules.max_open_positions, context=make_context(limits=one)).outcome is PASS
        held = make_context(limits=one, positions=self.positions("SPY"))
        assert run(rules.max_open_positions, context=held).outcome is REJECT
        # A generous cap is honoured too: nothing here bounds what the config allows.
        many = self.positions(*(f"T{number:03d}" for number in range(120)))
        roomy = make_context(limits=make_limits(max_open_positions=150), positions=many)
        result = run(rules.max_open_positions, context=roomy)
        assert result.outcome is PASS and "120 of 150" in result.detail
        full = make_context(limits=make_limits(max_open_positions=120), positions=many)
        assert run(rules.max_open_positions, context=full).outcome is REJECT

    @pytest.mark.parametrize(
        "build",
        [sell, put_credit_spread, naked_short_call, iron_condor],
        ids=["short sale", "credit spread", "naked short call", "iron condor"],
    )
    def test_at_the_cap_a_sale_that_opens_a_new_underlying_rejects(self, build):
        # Only a CLOSING sale adds no position: a short sale, or a sold
        # structure, on an underlying not yet held is a new one.
        proposal = build()
        five = self.positions("MSFT", "NVDA", "QQQ", "TSLA", "AMZN")
        at_cap = context_for(proposal, positions=five)
        assert not is_closing_sale(proposal, at_cap)
        result = run(rules.max_open_positions, proposal, at_cap)
        assert result.outcome is REJECT
        assert f"{proposal.underlying} would be a new position: 5 of 5" in result.detail
        below = context_for(proposal, positions=five[:4])
        assert run(rules.max_open_positions, proposal, below).outcome is PASS

    def test_a_cap_of_zero_rejects_every_new_position(self):
        zero = make_context(limits=make_limits(max_open_positions=0))
        assert run(rules.max_open_positions, context=zero).outcome is REJECT

    def test_a_closing_sale_passes_at_the_cap(self):
        rows = (make_position(qty=10), *self.positions("SPY", "QQQ", "NVDA", "MSFT", "TSLA"))
        result = run(rules.max_open_positions, sell(10), make_context(positions=rows))
        assert result.outcome is PASS and "closing sale" in result.detail


class TestMaxDailyTrades:
    @staticmethod
    def after(orders_today, **overrides):
        context = make_context(orders_today=orders_today, **overrides)
        return run(rules.max_daily_trades, context=context)

    def test_none_today_passes(self):
        result = self.after(0)
        assert result.outcome is PASS and "0 of 10" in result.detail

    def test_boundary_at_the_cap_rejects(self):
        assert self.after(9).outcome is PASS
        result = self.after(10)
        assert result.outcome is REJECT and "10 orders sent today" in result.detail
        assert self.after(11).outcome is REJECT

    def test_the_detail_says_at_or_above_the_cap(self):
        # "at the cap" would be false of 12 orders against a cap of 10.
        for used in (10, 12):
            assert self.after(used).detail == (
                f"{used} orders sent today: at or above the cap of 10 "
                "(risk_limits.max_daily_trades)"
            )

    def test_the_cap_comes_from_the_limits(self):
        three = {"limits": make_limits(max_daily_trades=3)}
        assert self.after(2, **three).outcome is PASS
        assert self.after(3, **three).outcome is REJECT

    def test_a_cap_of_zero_rejects_the_first_trade(self):
        assert self.after(0, limits=make_limits(max_daily_trades=0)).outcome is REJECT


# --- rules 12–13: order hygiene -----------------------------------------------


class TestDuplicate:
    @staticmethod
    def against(*recent, proposal=None, **overrides):
        context = make_context(recent_proposals=tuple(recent), **overrides)
        return run(rules.duplicate, proposal, context)

    @staticmethod
    def earlier(by: timedelta, **overrides):
        return make_recent(created_at=PROPOSED_AT - by, **overrides)

    def test_nothing_recent_passes(self):
        assert run(rules.duplicate).outcome is PASS

    def test_the_same_trade_ten_minutes_earlier_rejects_and_names_it(self):
        result = self.against(make_recent())
        assert result.outcome is REJECT
        assert "prop-0000" in result.detail and "60 minutes" in result.detail

    def test_boundary_exactly_the_window_rejects_one_second_past_passes(self):
        assert self.against(self.earlier(timedelta(minutes=60))).outcome is REJECT
        assert self.against(self.earlier(timedelta(minutes=60, seconds=1))).outcome is PASS
        assert self.against(self.earlier(timedelta(minutes=59, seconds=59))).outcome is REJECT
        assert self.against(self.earlier(timedelta(minutes=60, microseconds=1))).outcome is PASS

    def test_the_window_comes_from_the_limits(self):
        five = {"limits": make_limits(duplicate_window_minutes=5)}
        assert self.against(self.earlier(timedelta(minutes=5)), **five).outcome is REJECT
        assert self.against(self.earlier(timedelta(minutes=5, seconds=1)), **five).outcome is PASS

    def test_the_window_is_measured_from_the_proposal_not_from_now(self):
        # Judged two hours after it was created, the pair is still ten minutes apart.
        late = self.against(make_recent(), now=NOW + timedelta(hours=2))
        assert late.outcome is REJECT

    def test_a_later_proposal_does_not_make_this_one_a_duplicate(self):
        assert self.against(self.earlier(timedelta(seconds=-1))).outcome is PASS
        assert self.against(self.earlier(timedelta(minutes=-30))).outcome is PASS

    def test_on_an_exact_tie_the_smaller_id_is_the_earlier(self):
        tie = timedelta(0)
        assert self.against(self.earlier(tie, id="prop-0000")).outcome is REJECT  # < prop-0001
        assert self.against(self.earlier(tie, id="prop-0002")).outcome is PASS

    @pytest.mark.parametrize("gap", [timedelta(0), timedelta(seconds=1), timedelta(minutes=60)])
    def test_two_proposals_never_reject_each_other(self, gap):
        first = make_proposal(id="prop-a", created_at=PROPOSED_AT - gap)
        second = make_proposal(id="prop-b", created_at=PROPOSED_AT)
        first_sees = make_context(recent_proposals=(second.proposal,))
        second_sees = make_context(recent_proposals=(first.proposal,))
        assert run(rules.duplicate, first, first_sees).outcome is PASS
        assert run(rules.duplicate, second, second_sees).outcome is REJECT

    def test_the_proposal_itself_in_the_recent_list_is_ignored(self):
        own = make_proposal().proposal
        assert self.against(own).outcome is PASS
        # Matched by id: a copy of it with an earlier timestamp is still itself.
        assert self.against(make_recent(id=PROPOSAL_ID)).outcome is PASS
        assert self.against(make_recent(id=PROPOSAL_ID), make_recent()).outcome is REJECT

    @pytest.mark.parametrize(
        "difference",
        [{"symbol": "MSFT"}, {"side": OrderSide.SELL}, {"instrument": Instrument.OPTION}],
        ids=["symbol", "side", "instrument"],
    )
    def test_a_different_symbol_side_or_instrument_is_not_a_duplicate(self, difference):
        assert self.against(make_recent(**difference)).outcome is PASS

    def test_size_and_price_do_not_matter(self):
        assert self.against(make_recent(quantity=500, limit_price=1.0)).outcome is REJECT

    def test_an_option_structure_is_matched_by_its_symbol_and_side(self):
        proposal = call_debit_spread()
        same = make_recent(symbol="SPY", instrument=Instrument.OPTION, side=OrderSide.BUY)
        assert self.against(same, proposal=proposal).outcome is REJECT
        stock = make_recent(symbol="SPY")
        assert self.against(stock, proposal=proposal).outcome is PASS

    def test_a_window_of_zero_only_catches_an_exact_tie(self):
        zero = {"limits": make_limits(duplicate_window_minutes=0)}
        assert self.against(self.earlier(timedelta(0)), **zero).outcome is REJECT
        assert self.against(self.earlier(timedelta(seconds=1)), **zero).outcome is PASS

    def test_an_open_order_on_the_symbol_rejects_and_names_it(self):
        result = run(rules.duplicate, context=make_context(open_orders=(make_order(),)))
        assert result.outcome is REJECT
        assert "client-0001" in result.detail and "AAPL" in result.detail

    def test_an_open_order_on_another_symbol_passes(self):
        context = make_context(open_orders=(make_order("MSFT"),))
        assert run(rules.duplicate, context=context).outcome is PASS

    def test_an_open_order_on_one_leg_rejects(self):
        proposal = iron_condor()
        order = make_order(proposal.legs[2].symbol)
        result = run(rules.duplicate, proposal, context_for(proposal, open_orders=(order,)))
        assert result.outcome is REJECT and proposal.legs[2].symbol in result.detail
        # The underlying's stock is not one of the structure's contracts.
        stock = make_order("QQQ")
        context = context_for(proposal, open_orders=(stock,))
        assert run(rules.duplicate, proposal, context).outcome is PASS

    def test_an_open_order_on_the_underlyings_shares_is_not_a_duplicate_of_an_option(self):
        # An option named by its contract trades that contract — an order
        # working SPY shares is another instrument, not a repeat of it.
        proposal = long_call()
        assert proposal.proposal.symbol == CALL_640 and proposal.symbols == (CALL_640,)
        shares = context_for(proposal, open_orders=(make_order("SPY"),))
        result = run(rules.duplicate, proposal, shares)
        assert result.outcome is PASS and "no open order on it" in result.detail
        # Nor is an order on another contract of the same underlying ...
        other = context_for(proposal, open_orders=(make_order("SPY260821C00650000"),))
        assert run(rules.duplicate, proposal, other).outcome is PASS
        # ... while one on the very contract is.
        same = context_for(proposal, open_orders=(make_order(CALL_640),))
        assert run(rules.duplicate, proposal, same).outcome is REJECT
        # A structure named by its underlying IS repeated by an order on that symbol.
        spread = call_debit_spread()
        assert "SPY" in spread.symbols
        pending = context_for(spread, open_orders=(make_order("SPY"),))
        assert run(rules.duplicate, spread, pending).outcome is REJECT

    @pytest.mark.parametrize(
        "spelling", [PADDED_640, OTHER_DIGITS_640], ids=["padded", "other digits"]
    )
    def test_an_open_order_on_the_contract_in_another_spelling_is_a_duplicate(self, spelling):
        # The books may carry a contract in the padded OCC form. Since ruling R4
        # it counts in its underlying's exposure — and it is the very contract
        # the proposal trades, so the proposal is a repeat: matched as the
        # instrument it names, not as a string.
        proposal = long_call()
        order = make_order(spelling, limit_price=8.0)
        assert order.symbol == spelling != CALL_640 and is_occ(spelling)
        context = context_for(proposal, open_orders=(order,))
        result = run(rules.duplicate, proposal, context)
        assert result.outcome is REJECT
        assert "client-0001" in result.detail and "is still open" in result.detail
        assert first_reject(proposal, context) == "duplicate"
        assert exposure(context, "SPY") == pytest.approx(800.0)
        assert held_underlyings(context) == frozenset({"SPY"})
        # Exactly what the compact order gives.
        compact = context_for(proposal, open_orders=(make_order(CALL_640, limit_price=8.0),))
        assert run(rules.duplicate, proposal, compact).outcome is REJECT
        assert exposure(compact, "SPY") == exposure(context, "SPY")
        # The reverse: the proposal in that spelling, the order compact.
        leg = make_leg("buy", "call", 640.0, symbol=spelling)
        odd = option_proposal([leg], symbol=spelling, quantity=1.0, limit_price=8.00)
        assert run(rules.duplicate, odd, compact).outcome is REJECT
        assert run(rules.duplicate, odd, context).outcome is REJECT
        # One leg of a structure named by its underlying.
        spread = call_debit_spread()
        working = context_for(spread, open_orders=(order,))
        assert run(rules.duplicate, spread, working).outcome is REJECT

    def test_another_contract_in_the_padded_form_is_not_a_duplicate(self):
        proposal = long_call()
        for symbol in (padded(occ("call", 650.0)), padded(occ("put", 640.0)), "SPY   "):
            other = context_for(proposal, open_orders=(make_order(symbol),))
            result = run(rules.duplicate, proposal, other)
            assert result.outcome is PASS and "no open order on it" in result.detail
        # ... nor do an equity's shares repeat a contract, however either is written.
        shares = make_context(open_orders=(make_order(PADDED_640),))
        assert run(rules.duplicate, make_proposal(symbol="SPY"), shares).outcome is PASS

    @pytest.mark.parametrize(
        "spelling", [PADDED_640, OTHER_DIGITS_640], ids=["padded", "other digits"]
    )
    def test_an_earlier_proposal_for_the_contract_in_another_spelling_is_a_duplicate(
        self, spelling
    ):
        proposal = long_call()
        twin = make_recent(symbol=spelling, instrument=Instrument.OPTION, limit_price=8.00)
        assert twin.symbol == spelling and twin.side is proposal.proposal.side
        result = self.against(twin, proposal=proposal)
        assert result.outcome is REJECT and "prop-0000" in result.detail
        # ... and the other way round.
        leg = make_leg("buy", "call", 640.0, symbol=spelling)
        odd = option_proposal([leg], symbol=spelling, quantity=1.0, limit_price=8.00)
        compact = make_recent(symbol=CALL_640, instrument=Instrument.OPTION, limit_price=8.00)
        assert self.against(compact, proposal=odd).outcome is REJECT
        # Another contract, another side or the shares: still not a repeat.
        for difference in (
            {"symbol": padded(occ("call", 650.0))},
            {"side": OrderSide.SELL},
            {"instrument": Instrument.EQUITY},
        ):
            other = twin.model_copy(update=difference)
            assert self.against(other, proposal=proposal).outcome is PASS

    def test_a_repeated_sale_is_a_duplicate_closing_or_not(self):
        earlier = make_recent(side=OrderSide.SELL)
        short = make_context(recent_proposals=(earlier,))
        assert run(rules.duplicate, sell(), short).outcome is REJECT
        closing = holding(10, recent_proposals=(earlier,))
        assert is_closing_sale(sell(2), closing)
        assert run(rules.duplicate, sell(2), closing).outcome is REJECT
        assert non_pass(sell(2), closing) == {"duplicate": REJECT}
        working = holding(10, open_orders=(make_order(side=OrderSide.SELL),))
        assert run(rules.duplicate, sell(2), working).outcome is REJECT
        # An earlier BUY of the symbol is another side: not a duplicate of the sale.
        bought = holding(10, recent_proposals=(make_recent(),))
        assert run(rules.duplicate, sell(2), bought).outcome is PASS

    def test_both_findings_are_named(self):
        result = self.against(make_recent(), open_orders=(make_order(),))
        assert "prop-0000" in result.detail and "client-0001" in result.detail

    def test_an_unrepresentable_window_is_simply_very_long(self):
        forever = {"limits": make_limits(duplicate_window_minutes=10**15)}
        assert self.against(self.earlier(timedelta(days=4000)), **forever).outcome is REJECT


def _prices_and_percentage(detail: str) -> tuple[float, float, float]:
    """The limit, the mid and the stated percentage a limit_price_sanity
    detail prints — read back from the text, as an auditor would."""
    compared, _, rest = detail.partition(": ")
    limit_text, _, mid_text = compared.removeprefix("limit ").partition(" vs ")
    mid_text = mid_text.removeprefix("net mid ").removeprefix("mid ").split(" ")[0]

    def dollars(text: str) -> float:
        return float(text.replace("$", "").replace(",", ""))

    return dollars(limit_text), dollars(mid_text), float(rest.partition("% away")[0])


class TestLimitPriceSanity:
    @staticmethod
    def priced(limit_price, **overrides):
        proposal = make_proposal(limit_price=limit_price)
        return run(rules.limit_price_sanity, proposal, make_context(**overrides))

    def test_a_limit_at_the_mid_passes(self):
        result = run(rules.limit_price_sanity)
        assert result.outcome is PASS
        assert "$200.00" in result.detail and "5%" in result.detail

    # A market order of every kind — shares bought, sold short and sold out of
    # a long; an option bought; structures sold: (build, the positions held).
    MARKET_ORDERS = {
        "equity buy": (make_proposal, ()),
        "short sale": (sell, ()),
        "closing sale": (sell, (make_position(qty=10),)),
        "long call": (long_call, ()),
        "debit spread": (call_debit_spread, ()),
        "credit spread": (put_credit_spread, ()),
        "naked short call": (naked_short_call, ()),
    }
    EVERY_MARKET_ORDER = pytest.mark.parametrize(
        ("build", "positions"), MARKET_ORDERS.values(), ids=MARKET_ORDERS.keys()
    )

    @EVERY_MARKET_ORDER
    def test_a_market_order_rejects(self, build, positions):
        market = build(order_type=OrderType.MARKET, limit_price=None)
        context = context_for(market, positions=positions)
        result = run(rules.limit_price_sanity, market, context)
        assert result.outcome is REJECT
        assert result.detail == "market orders are not allowed (risk_limits.allow_market_orders)"

    @EVERY_MARKET_ORDER
    def test_a_market_order_passes_when_allowed(self, build, positions):
        market = build(order_type=OrderType.MARKET, limit_price=None)
        allowed = make_limits(allow_market_orders=True)
        context = context_for(market, positions=positions, limits=allowed)
        result = run(rules.limit_price_sanity, market, context)
        assert result.outcome is PASS and "(risk_limits.allow_market_orders)" in result.detail

    def test_a_closing_market_sale_is_rejected_by_this_rule(self):
        closing = sell(2, order_type=OrderType.MARKET, limit_price=None)
        assert is_closing_sale(closing, holding(10))
        assert first_reject(closing, holding(10)) == "limit_price_sanity"
        assert non_pass(closing, holding(10)) == {
            "limit_price_sanity": REJECT, "auto_tier": ESCALATE,
        }

    def test_a_closing_sale_far_from_the_mid_rejects(self):
        closing = sell(2, limit_price=140.0)  # 30% under a 200.00 mid
        assert is_closing_sale(closing, holding(10))
        result = run(rules.limit_price_sanity, closing, holding(10))
        assert result.outcome is REJECT and "30% away" in result.detail
        assert non_pass(closing, holding(10)) == {"limit_price_sanity": REJECT}

    def test_boundary_exactly_the_tolerance_passes_a_cent_past_rejects(self):
        assert self.priced(210.00).outcome is PASS  # 5.0% above a 200.00 mid
        assert self.priced(210.01).outcome is REJECT
        assert self.priced(190.00).outcome is PASS  # 5.0% below
        assert self.priced(189.99).outcome is REJECT

    def test_thirty_percent_off_the_mid_rejects_and_says_how_far(self):
        result = self.priced(260.0)
        assert result.outcome is REJECT
        assert "$260.00" in result.detail and "$200.00" in result.detail and "30%" in result.detail
        assert self.priced(140.0).outcome is REJECT

    def test_a_sell_is_judged_the_same_way(self):
        assert run(rules.limit_price_sanity, sell(limit_price=190.0)).outcome is PASS
        assert run(rules.limit_price_sanity, sell(limit_price=140.0)).outcome is REJECT

    def test_the_tolerance_comes_from_the_limits(self):
        tight = {"limits": make_limits(limit_price_tolerance_pct=1.0)}
        assert self.priced(202.00, **tight).outcome is PASS
        assert self.priced(202.01, **tight).outcome is REJECT
        # A generous tolerance is honoured too: 800.00 is 300% from a 200.00 mid.
        loose = {"limits": make_limits(limit_price_tolerance_pct=500.0)}
        result = self.priced(800.00, **loose)
        assert result.outcome is PASS and "300% away, tolerance 500%" in result.detail
        assert self.priced(1200.00, **loose).outcome is PASS  # exactly 500%
        assert self.priced(1200.01, **loose).outcome is REJECT

    def test_the_detail_reproduces_its_percentage(self):
        # An option mid on the half cent: printed to the cent it would be
        # $1.21, and (1.26 - 1.21) / 1.21 is 4.13%, not the 4.56% stated.
        proposal = long_call(limit_price=1.26)
        result = run(rules.limit_price_sanity, proposal, context_for(proposal, mids=[1.205]))
        assert result.outcome is PASS
        assert result.detail.startswith(
            "limit $1.26 vs net mid $1.205 (debit +, credit -): 4.564315353% away, "
        )
        limit, mid, stated = _prices_and_percentage(result.detail)
        assert (limit, mid) == (1.26, 1.205)
        assert abs(limit - mid) / abs(mid) * 100 == pytest.approx(stated, rel=1e-9)
        assert abs(1.26 - 1.21) / 1.21 * 100 != pytest.approx(stated, rel=1e-3)

    @pytest.mark.parametrize(
        ("limit_price", "mid", "text"),
        [
            (200.0, 200.0, "limit $200.00 vs mid $200.00"),
            (200.004, 200.0, "limit $200.004 vs mid $200.00"),
            (200.0, 199.995, "limit $200.00 vs mid $199.995"),
            (0.1234, 0.125, "limit $0.1234 vs mid $0.125"),
            (1234.5, 1234.5678, "limit $1,234.50 vs mid $1,234.5678"),
        ],
    )
    def test_an_equity_price_is_printed_with_every_digit_it_has(self, limit_price, mid, text):
        proposal = make_proposal(limit_price=limit_price)
        quote = make_quote(mid=mid, spread=0.0)
        result = run(rules.limit_price_sanity, proposal, make_context(quote=quote))
        assert result.detail.startswith(text + ": ")
        limit, shown_mid, stated = _prices_and_percentage(result.detail)
        assert abs(limit - shown_mid) / shown_mid * 100 == pytest.approx(stated, rel=1e-6)

    def test_a_credit_structure_shows_signed_prices_that_reproduce_the_percentage(self):
        proposal = put_credit_spread(limit_price=2.085)  # a net mid of -2.00
        result = run(rules.limit_price_sanity, proposal, context_for(proposal))
        assert result.detail.startswith("limit -$2.085 vs net mid -$2.00 (debit +, credit -): ")
        limit, mid, stated = _prices_and_percentage(result.detail)
        assert (limit, mid) == (-2.085, -2.0)
        assert abs(limit - mid) / abs(mid) * 100 == pytest.approx(stated, rel=1e-9)

    def test_a_tolerance_of_zero_accepts_only_the_mid(self):
        exact = make_limits(limit_price_tolerance_pct=0.0)
        zero = {"limits": exact, "quote": make_quote(spread=0.0)}
        assert self.priced(200.00, **zero).outcome is PASS
        assert self.priced(200.01, **zero).outcome is REJECT

    def test_float_noise_does_not_reject(self):
        locked = make_quote(mid=100.1, spread=0.0)
        assert abs(105.105 - 100.1) / 100.1 * 100 > 5.0  # 5.00000000000001 in floats
        assert self.priced(105.105, quote=locked).outcome is PASS
        assert self.priced(105.12, quote=locked).outcome is REJECT

    @pytest.mark.parametrize("price", [None, 0.0, -200.0])
    def test_a_limit_order_without_a_usable_price_rejects(self, price):
        result = self.priced(price)
        assert result.outcome is REJECT and "no usable limit price" in result.detail

    def test_no_quote_rejects_and_names_the_symbol(self):
        result = self.priced(200.0, quote=None)
        assert result.outcome is REJECT and "no quote for AAPL" in result.detail

    def test_a_quote_for_another_symbol_rejects(self):
        result = self.priced(200.0, quote=make_quote("MSFT", 200.0))
        assert result.outcome is REJECT and "MSFT" in result.detail and "AAPL" in result.detail

    @pytest.mark.parametrize("bad", [None, NAN, INF, -INF, 0.0, -1.0])
    def test_an_unknown_bid_or_ask_rejects(self, bad):
        for quote in (make_quote(bid=bad), make_quote(ask=bad)):
            result = self.priced(200.0, quote=quote)
            assert result.outcome is REJECT and "no usable two-sided bid/ask" in result.detail

    def test_a_crossed_quote_rejects(self):
        assert self.priced(200.0, quote=make_quote(bid=201.0, ask=199.0)).outcome is REJECT

    def test_an_option_at_its_net_mid_passes(self):
        for build in STRUCTURES:
            proposal = build()
            assert run(rules.limit_price_sanity, proposal, context_for(proposal)).outcome is PASS

    def test_boundary_for_a_debit_structure(self):
        def priced(limit):
            proposal = call_debit_spread(limit_price=limit)  # net mid 4.40
            return run(rules.limit_price_sanity, proposal, context_for(proposal))

        assert priced(4.62).outcome is PASS  # 5% above, give or take an ulp
        assert priced(4.63).outcome is REJECT
        assert priced(4.18).outcome is PASS
        assert priced(4.17).outcome is REJECT

    def test_boundary_for_a_credit_structure(self):
        def priced(limit):
            proposal = put_credit_spread(limit_price=limit)  # net mid −2.00
            return run(rules.limit_price_sanity, proposal, context_for(proposal))

        assert priced(2.10).outcome is PASS
        assert priced(2.11).outcome is REJECT
        assert priced(1.90).outcome is PASS
        assert priced(1.89).outcome is REJECT

    def test_a_credit_mislabelled_as_a_debit_is_two_hundred_percent_away(self):
        proposal = put_credit_spread(side="buy")  # +2.00 against a net mid of −2.00
        result = run(rules.limit_price_sanity, proposal, context_for(proposal))
        assert result.outcome is REJECT and "200%" in result.detail
        mislabelled = call_debit_spread(side="sell")
        result = run(rules.limit_price_sanity, mislabelled, context_for(mislabelled))
        assert result.outcome is REJECT

    def test_a_missing_leg_quote_rejects_and_names_the_leg(self):
        proposal = call_debit_spread()
        quotes = leg_quotes_for(proposal)[:1]
        result = run(rules.limit_price_sanity, proposal, context_for(proposal, leg_quotes=quotes))
        assert result.outcome is REJECT
        assert f"no quote for leg {proposal.legs[1].symbol}" in result.detail
        assert proposal.legs[0].symbol not in result.detail

    @pytest.mark.parametrize("bad", [None, NAN, INF, -0.5])
    def test_an_unknown_leg_bid_or_ask_rejects(self, bad):
        proposal = call_debit_spread()
        good = leg_quotes_for(proposal)
        for override in ({"bid": bad}, {"ask": bad}):
            quotes = (good[0], make_snapshot(proposal.legs[1].symbol, 3.6, **override))
            context = context_for(proposal, leg_quotes=quotes)
            result = run(rules.limit_price_sanity, proposal, context)
            assert result.outcome is REJECT and proposal.legs[1].symbol in result.detail

    def test_a_zero_bid_leg_is_still_priced(self):
        proposal = long_call(limit_price=0.05)
        quotes = (make_snapshot(proposal.legs[0].symbol, 0.05, bid=0.0, ask=0.10),)
        context = context_for(proposal, leg_quotes=quotes)
        assert run(rules.limit_price_sanity, proposal, context).outcome is PASS

    def test_a_net_mid_of_zero_rejects(self):
        # Buy and sell the same contract: the legs net to nothing.
        proposal = make_option(WASH_LEGS, side="buy", limit_price=0.05)
        result = run(rules.limit_price_sanity, proposal, context_for(proposal))
        assert result.outcome is REJECT and "net mid of the legs is zero" in result.detail

    def test_an_option_without_legs_rejects(self):
        result = run(rules.limit_price_sanity, option_proposal([]))
        assert result.outcome is REJECT and "no legs" in result.detail


# --- rules 14–17: options -----------------------------------------------------


class TestOptionsMinDte:
    @staticmethod
    def expiring(days, **overrides):
        proposal = long_call(expiration=MARKET_DATE + timedelta(days=days))
        return run(rules.options_min_dte, proposal, make_context(**overrides))

    def test_an_equity_passes_as_not_an_option(self):
        result = run(rules.options_min_dte)
        assert result.outcome is PASS and result.detail == "not an option"

    def test_three_weeks_out_passes(self):
        result = run(rules.options_min_dte, long_call())
        assert result.outcome is PASS and "22 DTE" in result.detail

    def test_boundary_at_the_minimum_passes_one_day_less_rejects(self):
        assert self.expiring(1).outcome is PASS
        zero_dte = self.expiring(0)
        assert zero_dte.outcome is REJECT and "at 0 DTE" in zero_dte.detail
        assert self.expiring(-1).outcome is REJECT  # already expired

    @pytest.mark.parametrize("build", STRUCTURES)
    def test_the_boundary_holds_for_every_structure_bought_or_sold(self, build):
        # A 0DTE credit spread or condor is as close to expiry as a 0DTE long call.
        tomorrow = build(expiration=MARKET_DATE + timedelta(days=1))
        assert run(rules.options_min_dte, tomorrow, context_for(tomorrow)).outcome is PASS
        today = build(expiration=MARKET_DATE)
        result = run(rules.options_min_dte, today, context_for(today))
        assert result.outcome is REJECT
        for leg in today.legs:
            assert f"{leg.symbol} at 0 DTE" in result.detail
        assert "below the minimum of 1 (risk_limits.min_dte)" in result.detail
        expired = build(expiration=MARKET_DATE - timedelta(days=1))
        assert run(rules.options_min_dte, expired, context_for(expired)).outcome is REJECT

    @pytest.mark.parametrize("build", DEFINED_RISK)
    def test_a_zero_dte_defined_risk_structure_is_rejected_by_this_rule(self, build):
        today = build(expiration=MARKET_DATE)
        assert first_reject(today, context_for(today)) == "options_min_dte"

    def test_the_minimum_comes_from_the_limits(self):
        week = {"limits": make_limits(min_dte=7)}
        assert self.expiring(7, **week).outcome is PASS
        assert self.expiring(6, **week).outcome is REJECT

    def test_a_minimum_of_zero_allows_same_day_but_never_expired(self):
        zero = {"limits": make_limits(min_dte=0)}
        assert self.expiring(0, **zero).outcome is PASS
        assert self.expiring(-1, **zero).outcome is REJECT

    def test_dte_counts_from_the_market_date(self):
        proposal = long_call()
        on_the_day = make_context(market_date=EXPIRY)
        assert run(rules.options_min_dte, proposal, on_the_day).outcome is REJECT
        day_before = make_context(market_date=EXPIRY - timedelta(days=1))
        assert run(rules.options_min_dte, proposal, day_before).outcome is PASS

    def test_any_one_leg_too_close_rejects_and_only_it_is_listed(self):
        near = make_leg("sell", "call", 650.0, index=1, expiration=MARKET_DATE)
        proposal = option_proposal([make_leg("buy", "call", 640.0), near])
        result = run(rules.options_min_dte, proposal)
        assert result.outcome is REJECT
        assert near.symbol in result.detail and proposal.legs[0].symbol not in result.detail

    def test_an_option_without_legs_rejects(self):
        result = run(rules.options_min_dte, option_proposal([]))
        assert result.outcome is REJECT and "no legs" in result.detail

    def test_an_equity_that_carries_legs_is_left_to_options_max_loss(self):
        # Even an expired leg: this rule reads the declared instrument, and
        # options_max_loss owns the rejection of the shape.
        expired = make_leg("buy", "call", 640.0, expiration=MARKET_DATE - timedelta(days=1))
        result = run(rules.options_min_dte, make_proposal(legs=[expired]))
        assert result.outcome is PASS and result.detail == "not an option"
        named = run(rules.options_min_dte, make_proposal(symbol=CALL_640))
        assert named.outcome is PASS and named.detail == "not an option"


class TestOptionsMaxLoss:
    @staticmethod
    def with_loss(proposal, **overrides):
        return run(rules.options_max_loss, proposal, context_for(proposal, **overrides))

    @staticmethod
    def with_analysis(**update):
        proposal = put_credit_spread()
        analysis = context_for(proposal).analysis.model_copy(update=update)
        return run(rules.options_max_loss, proposal, context_for(proposal, analysis=analysis))

    def test_an_equity_passes_as_not_an_option(self):
        assert run(rules.options_max_loss).detail == "not an option"
        for plain in (sell(), make_proposal(symbol="TSLA", quantity=5000)):
            result = run(rules.options_max_loss, plain)
            assert result.outcome is PASS and result.detail == "not an option"

    @BROKEN
    def test_a_shape_that_contradicts_itself_rejects_with_the_finding(self, proposal, finding):
        # The shape is judged first, whatever the instrument says and however
        # small a loss the analysis in hand claims.
        result = run(rules.options_max_loss, proposal, flattering(proposal))
        assert result.outcome is REJECT
        lead = (
            "the structure cannot be analysed: "
            if proposal.is_option
            else "the proposal is not a plain equity: "
        )
        assert result.detail == lead + finding

    def test_a_mislabelled_equity_lists_every_finding(self):
        both = make_proposal(symbol=CALL_640, legs=[make_leg("buy", "call", 640.0)])
        result = run(rules.options_max_loss, both, flattering(both))
        assert result.outcome is REJECT
        assert result.detail == (
            "the proposal is not a plain equity: an equity proposal carries 1 option leg(s); "
            "the proposal is declared equity but its symbol SPY260821C00640000 is an option "
            "contract"
        )

    @pytest.mark.parametrize("build", DEFINED_RISK)
    def test_defined_risk_inside_the_cap_passes(self, build):
        result = self.with_loss(build())
        assert result.outcome is PASS and "$1,000.00" in result.detail

    def test_boundary_at_the_cap_passes_one_cent_over_rejects(self):
        at = {"limits": make_limits(max_loss_per_trade=800.00)}
        assert self.with_loss(long_call(), **at).outcome is PASS
        under = {"limits": make_limits(max_loss_per_trade=799.99)}
        result = self.with_loss(long_call(), **under)
        assert result.outcome is REJECT
        assert "$800.00" in result.detail and "$799.99" in result.detail
        assert self.with_analysis(max_loss=1000.00).outcome is PASS
        assert self.with_analysis(max_loss=1000.01).outcome is REJECT

    def test_the_loss_scales_with_the_quantity(self):
        assert self.with_loss(call_debit_spread(quantity=2)).outcome is PASS  # 880
        assert self.with_loss(call_debit_spread(quantity=3)).outcome is REJECT  # 1,320

    def test_the_cap_comes_from_the_limits(self):
        small = {"limits": make_limits(max_loss_per_trade=100.0)}
        assert self.with_loss(long_call(), **small).outcome is REJECT
        # A generous cap is honoured too: 3,000 calls at 8.00 can lose 2,400,000.00.
        huge = long_call(quantity=3000)
        generous = {"limits": make_limits(max_loss_per_trade=5_000_000.0)}
        result = self.with_loss(huge, **generous)
        assert result.outcome is PASS
        assert "max loss $2,400,000.00 is within the cap $5,000,000.00" in result.detail
        under = {"limits": make_limits(max_loss_per_trade=2_399_999.99)}
        assert self.with_loss(huge, **under).outcome is REJECT

    def test_a_rejection_by_a_fraction_of_a_cent_shows_the_difference(self):
        under = {"limits": make_limits(max_loss_per_trade=439.996)}
        result = self.with_loss(call_debit_spread(), **under)
        assert result.outcome is REJECT
        assert result.detail == (
            "max loss $440.00 is above the cap $440.00 by $0.004 "
            "(risk_limits.max_loss_per_trade)"
        )
        plain = self.with_loss(long_call(), limits=make_limits(max_loss_per_trade=799.99))
        assert " by $" not in plain.detail

    def test_float_noise_does_not_reject(self):
        proposal = call_debit_spread()
        loss = context_for(proposal).analysis.max_loss
        assert loss > 440.0  # 440.00000000000006 from the pricing engine
        at = make_limits(max_loss_per_trade=440.0)
        assert self.with_loss(proposal, limits=at).outcome is PASS

    @pytest.mark.parametrize("cap", [1000.0, 1e12, LARGEST_FINITE])
    def test_unlimited_risk_rejects_whatever_the_cap_says(self, cap):
        result = self.with_loss(naked_short_call(), limits=make_limits(max_loss_per_trade=cap))
        assert result.outcome is REJECT
        assert "unlimited risk" in result.detail and "unconditionally" in result.detail
        assert "(risk_limits.max_loss_per_trade)" in result.detail

    def test_the_unlimited_flag_wins_over_a_stated_max_loss(self):
        assert self.with_analysis(unlimited_risk=True, max_loss=1.0).outcome is REJECT

    def test_no_analysis_rejects(self):
        result = self.with_loss(call_debit_spread(), analysis=None)
        assert result.outcome is REJECT and "no position analysis" in result.detail

    @pytest.mark.parametrize("loss", [None, NAN, INF, -INF, -1.0])
    def test_a_max_loss_that_is_not_a_finite_amount_rejects(self, loss):
        result = self.with_analysis(max_loss=loss, unlimited_risk=False)
        assert result.outcome is REJECT and "no finite max loss" in result.detail

    def test_a_structure_that_cannot_lose_passes(self):
        assert self.with_analysis(max_loss=0.0).outcome is PASS

    def test_a_cap_of_zero_rejects_any_loss(self):
        zero = {"limits": make_limits(max_loss_per_trade=0.0)}
        assert self.with_loss(long_call(), **zero).outcome is REJECT

    @pytest.mark.parametrize(
        ("legs", "problem"),
        [
            ([], "no legs"),
            (TWO_UNDERLYINGS, "more than one underlying"),
            (
                [
                    make_leg("buy", "call", 640.0),
                    make_leg("sell", "call", 650.0, index=1, expiration=date(2026, 9, 18)),
                ],
                "different expirations",
            ),
            ([make_leg("buy", "call", 640.0, 1.5)], "not a whole number"),
            ([make_leg("buy", "call", 640.0, underlying="QQQ")], "the proposal is on SPY"),
        ],
        ids=["no legs", "two underlyings", "two expirations", "fractional", "symbol mismatch"],
    )
    def test_a_broken_structure_rejects_and_says_why(self, legs, problem):
        proposal = option_proposal(legs)
        # Even with an analysis that claims a tiny loss.
        analysis = context_for(call_debit_spread()).analysis
        result = run(rules.options_max_loss, proposal, make_context(analysis=analysis))
        assert result.outcome is REJECT
        assert "cannot be analysed" in result.detail and problem in result.detail


class TestOptionsMaxContracts:
    @staticmethod
    def sized(quantity, **overrides):
        proposal = long_call(quantity=quantity)
        return run(rules.options_max_contracts, proposal, make_context(**overrides))

    def test_an_equity_passes_as_not_an_option(self):
        assert run(rules.options_max_contracts).detail == "not an option"
        # ...however many shares it trades.
        assert run(rules.options_max_contracts, make_proposal(quantity=5000)).outcome is PASS

    def test_boundary_at_the_cap_passes_one_more_rejects(self):
        assert self.sized(1).outcome is PASS
        assert self.sized(10).outcome is PASS
        result = self.sized(11)
        assert result.outcome is REJECT and "x11" in result.detail and "10" in result.detail

    def test_the_cap_comes_from_the_limits(self):
        two = {"limits": make_limits(max_contracts=2)}
        assert self.sized(2, **two).outcome is PASS
        assert self.sized(3, **two).outcome is REJECT
        # A generous cap is honoured too: nothing here bounds what the config allows.
        many = {"limits": make_limits(max_contracts=500)}
        result = self.sized(300, **many)
        assert result.outcome is PASS and "largest leg x300, cap 500" in result.detail
        assert self.sized(500, **many).outcome is PASS
        assert self.sized(501, **many).outcome is REJECT

    def test_a_cap_of_zero_rejects_every_option(self):
        assert self.sized(1, limits=make_limits(max_contracts=0)).outcome is REJECT

    def test_it_is_each_legs_quantity_that_counts(self):
        # 6 units of a 1x2 ratio: the doubled leg is 12 contracts.
        ratio = make_option(RATIO_LEGS, side="buy", limit_price=0.8, quantity=6)
        result = run(rules.options_max_contracts, ratio)
        assert result.outcome is REJECT
        assert ratio.legs[1].symbol in result.detail and ratio.legs[0].symbol not in result.detail
        # Four legs of 10 are 40 contracts in all, and each is within the cap.
        assert run(rules.options_max_contracts, iron_condor(quantity=10)).outcome is PASS

    def test_a_fraction_over_the_cap_is_over(self):
        proposal = option_proposal([make_leg("buy", "call", 640.0, 10.5)])
        assert run(rules.options_max_contracts, proposal).outcome is REJECT

    def test_an_option_without_legs_rejects(self):
        result = run(rules.options_max_contracts, option_proposal([]))
        assert result.outcome is REJECT and "no legs" in result.detail

    def test_an_equity_that_carries_legs_is_left_to_options_max_loss(self):
        oversized = make_leg("buy", "call", 640.0, 500.0)
        result = run(rules.options_max_contracts, make_proposal(legs=[oversized]))
        assert result.outcome is PASS and result.detail == "not an option"
        named = run(rules.options_max_contracts, make_proposal(symbol=CALL_640))
        assert named.outcome is PASS and named.detail == "not an option"


class TestOptionsEscalate:
    def test_an_equity_passes(self):
        assert run(rules.options_escalate).outcome is PASS
        assert run(rules.options_escalate, sell()).outcome is PASS

    @pytest.mark.parametrize("build", STRUCTURES)
    def test_every_option_structure_escalates(self, build):
        proposal = build()
        assert run(rules.options_escalate, proposal, context_for(proposal)).outcome is ESCALATE

    def test_always_whatever_the_limits_or_the_size(self):
        generous = make_limits(
            auto_execute={"enabled": True, "max_notional": 1e9}, max_loss_per_trade=1e9, min_dte=0
        )
        tiny = long_call(limit_price=0.01)
        assert run(rules.options_escalate, tiny, make_context(limits=generous)).outcome is ESCALATE
        assert run(rules.options_escalate, option_proposal([])).outcome is ESCALATE

    @BROKEN
    def test_a_shape_that_contradicts_itself_escalates(self, proposal, finding):
        result = run(rules.options_escalate, proposal, flattering(proposal))
        assert result.outcome is ESCALATE
        if proposal.is_equity:  # option-like, whatever it is declared: it says why
            assert "not a plain equity" in result.detail and finding in result.detail

    def test_only_a_plain_equity_passes(self):
        for proposal in (make_proposal(), sell(), make_proposal(symbol="TSLA")):
            assert is_plain_equity(proposal)
            assert run(rules.options_escalate, proposal).outcome is PASS
        for proposal in [build() for build in STRUCTURES] + [
            pair[0] for pair in BROKEN_SHAPES.values()
        ]:
            assert not is_plain_equity(proposal)
            assert run(rules.options_escalate, proposal).outcome is ESCALATE


# --- rules 18–20: tiers -------------------------------------------------------


class TestShortSale:
    REJECTING = make_limits(reject_short_sales=True)

    def test_a_buy_passes(self):
        assert run(rules.short_sale).outcome is PASS

    def test_an_option_sell_is_not_examined(self):
        for build in (naked_short_call, put_credit_spread):
            context = make_context(positions=None, limits=self.REJECTING)
            result = run(rules.short_sale, build(), context)
            assert result.outcome is PASS and result.detail == "not an equity sell"

    def test_selling_a_held_long_passes(self):
        result = run(rules.short_sale, sell(10), holding(10))
        assert result.outcome is PASS and "closes" in result.detail
        assert run(rules.short_sale, sell(3), holding(10)).outcome is PASS

    def test_boundary_selling_more_than_is_held_escalates(self):
        assert run(rules.short_sale, sell(10), holding(10)).outcome is PASS
        result = run(rules.short_sale, sell(11), holding(10))
        assert result.outcome is ESCALATE and "only 10 AAPL held long" in result.detail

    def test_selling_with_nothing_held_escalates(self):
        result = run(rules.short_sale, sell())
        assert result.outcome is ESCALATE
        assert "short sale" in result.detail and "(risk_limits.reject_short_sales)" in result.detail

    def test_adding_to_a_short_escalates(self):
        short = make_context(positions=(make_position(qty=-10, side="short"),))
        assert run(rules.short_sale, sell(1), short).outcome is ESCALATE

    def test_a_long_in_another_symbol_does_not_help(self):
        other = make_context(positions=(make_position("MSFT", 10),))
        assert run(rules.short_sale, sell(1), other).outcome is ESCALATE

    def test_unknown_positions_escalate(self):
        result = run(rules.short_sale, sell(1), make_context(positions=None))
        assert result.outcome is ESCALATE and "positions are unknown" in result.detail
        unknown_qty = make_context(positions=(make_position(qty=NAN, market_value=1.0),))
        assert run(rules.short_sale, sell(1), unknown_qty).outcome is ESCALATE

    def test_with_the_flag_a_short_sale_rejects(self):
        for context in (
            make_context(limits=self.REJECTING),
            make_context(limits=self.REJECTING, positions=None),
            holding(10, limits=self.REJECTING),
        ):
            result = run(rules.short_sale, sell(11), context)
            assert result.outcome is REJECT and "(risk_limits.reject_short_sales)" in result.detail

    def test_with_the_flag_a_closing_sale_still_passes(self):
        assert run(rules.short_sale, sell(10), holding(10, limits=self.REJECTING)).outcome is PASS
        assert run(rules.short_sale, None, make_context(limits=self.REJECTING)).outcome is PASS

    @DUST
    @pytest.mark.parametrize("positions", [(), SHORT_AAPL], ids=["no position", "short"])
    def test_a_sale_of_dust_with_no_long_is_a_short_sale(self, dust, positions):
        # Finding 4.1: a billionth of a share sold with nothing held long (even
        # while short) used to "close an existing long position".
        proposal = sell(dust)
        context = context_for(proposal, positions=positions)
        result = run(rules.short_sale, proposal, context)
        assert result.outcome is ESCALATE
        assert "is a short sale: no long AAPL position is held" in result.detail
        assert "closes an existing long" not in result.detail
        # Never all twenty PASS: a human must look, and with the flag it is rejected.
        assert non_pass(proposal, context) == {"short_sale": ESCALATE, "auto_tier": ESCALATE}
        assert not passing(proposal, context)
        rejecting = context_for(proposal, positions=positions, limits=self.REJECTING)
        flagged = run(rules.short_sale, proposal, rejecting)
        assert flagged.outcome is REJECT and "short sales are rejected" in flagged.detail
        assert first_reject(proposal, rejecting) == "short_sale"

    @DUST
    def test_a_sale_of_dust_with_the_account_unreadable_is_rejected(self, dust):
        # ... and it no longer skips the sizing rules as "a closing sale".
        proposal = sell(dust)
        blind = context_for(proposal, equity=None, buying_power=None)
        assert non_pass(proposal, blind) == {
            "buying_power": REJECT, "max_position_pct": REJECT,
            "short_sale": ESCALATE, "auto_tier": ESCALATE,
        }

    @DUST
    def test_a_sale_of_dust_out_of_a_real_long_still_closes(self, dust):
        result = run(rules.short_sale, sell(dust), holding(10))
        assert result.outcome is PASS and "closes an existing long position" in result.detail

    def test_a_sale_against_a_long_row_and_a_short_row_escalates(self):
        long_row, short_row = make_position(qty=10), make_position(qty=-10, side="short")
        for rows in ((long_row, short_row), (short_row, long_row)):
            context = make_context(positions=rows)
            result = run(rules.short_sale, sell(2), context)
            assert result.outcome is ESCALATE and "no long AAPL position is held" in result.detail
            assert non_pass(sell(2), context) == {"short_sale": ESCALATE, "auto_tier": ESCALATE}
            rejecting = make_context(positions=rows, limits=self.REJECTING)
            assert run(rules.short_sale, sell(2), rejecting).outcome is REJECT

    def test_a_market_short_sale_is_a_short_sale(self):
        # The order type changes nothing: with market orders allowed, this
        # rule is what stands between a market short sale and the tape.
        market = sell(order_type=OrderType.MARKET, limit_price=None)
        allowed = make_limits(allow_market_orders=True)
        result = run(rules.short_sale, market, make_context(limits=allowed))
        assert result.outcome is ESCALATE and "short sale" in result.detail
        assert non_pass(market, make_context(limits=allowed)) == {
            "short_sale": ESCALATE, "auto_tier": ESCALATE,
        }
        both = make_limits(allow_market_orders=True, reject_short_sales=True)
        rejected = run(rules.short_sale, market, make_context(limits=both))
        assert rejected.outcome is REJECT
        assert first_reject(market, make_context(limits=both)) == "short_sale"
        # A market sale that closes a long still passes this rule.
        assert run(rules.short_sale, market, holding(10, limits=both)).outcome is PASS


class TestMinConfidence:
    @staticmethod
    def confident(confidence, floor=None):
        limits = make_limits() if floor is None else make_limits(min_confidence=floor)
        proposal = make_proposal(confidence=confidence)
        return run(rules.min_confidence, proposal, make_context(limits=limits))

    def test_above_the_floor_passes(self):
        result = self.confident(0.8)
        assert result.outcome is PASS and "0.8" in result.detail and "0.5" in result.detail

    def test_boundary_at_the_floor_passes_just_below_flags(self):
        assert self.confident(0.5).outcome is PASS
        result = self.confident(0.49)
        assert result.outcome is FLAG and "0.49" in result.detail
        assert self.confident(0.499999).outcome is FLAG

    def test_it_flags_and_never_rejects(self):
        assert self.confident(0.0).outcome is FLAG

    def test_the_floor_comes_from_the_limits(self):
        assert self.confident(0.7, floor=0.7).outcome is PASS
        assert self.confident(0.69, floor=0.7).outcome is FLAG
        assert self.confident(1.0, floor=1.0).outcome is PASS
        assert self.confident(0.99, floor=1.0).outcome is FLAG

    def test_a_floor_of_zero_flags_nothing(self):
        assert self.confident(0.0, floor=0.0).outcome is PASS

    def test_float_noise_does_not_flag(self):
        assert 0.3 < 0.1 + 0.2
        assert self.confident(0.3, floor=0.1 + 0.2).outcome is PASS
        assert self.confident(0.1 + 0.2, floor=0.3).outcome is PASS

    def test_a_market_order_below_the_floor_flags(self):
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None, confidence=0.2)
        allowed = make_context(limits=make_limits(allow_market_orders=True))
        result = run(rules.min_confidence, market, allowed)
        assert result.outcome is FLAG and "confidence 0.2, floor 0.5" in result.detail
        # With market orders allowed, the flag is the strongest finding on it.
        assert non_pass(market, allowed) == {"min_confidence": FLAG, "auto_tier": ESCALATE}
        assert first_reject(market, allowed) is None

    @pytest.mark.parametrize("build", [sell, *STRUCTURES])
    def test_any_proposal_below_the_floor_flags(self, build):
        proposal = build(confidence=0.2)
        assert run(rules.min_confidence, proposal, context_for(proposal)).outcome is FLAG

    def test_a_closing_sale_below_the_floor_flags(self):
        closing = sell(2, confidence=0.2)
        assert is_closing_sale(closing, holding(10))
        assert run(rules.min_confidence, closing, holding(10)).outcome is FLAG
        assert non_pass(closing, holding(10)) == {"min_confidence": FLAG}


class TestAutoTier:
    @staticmethod
    def tier(proposal=None, **overrides):
        proposal = make_proposal() if proposal is None else proposal
        return run(rules.auto_tier, proposal, context_for(proposal, **overrides))

    @staticmethod
    def up_to(max_notional):
        return make_limits(auto_execute={"max_notional": max_notional})

    def test_a_small_watchlist_limit_buy_qualifies(self):
        result = run(rules.auto_tier)
        assert result.outcome is PASS
        assert "$400.00" in result.detail and "$1,000.00" in result.detail

    def test_disabled_escalates(self):
        off = make_limits(auto_execute={"enabled": False})
        result = self.tier(limits=off)
        assert result.outcome is ESCALATE
        assert "disabled" in result.detail and "(risk_limits.auto_execute.enabled)" in result.detail

    @pytest.mark.parametrize("build", STRUCTURES)
    def test_an_option_never_qualifies(self, build):
        result = self.tier(build(limit_price=0.01))
        assert result.outcome is ESCALATE and "option" in result.detail

    @BROKEN
    def test_a_shape_that_contradicts_itself_never_qualifies(self, proposal, finding):
        result = run(rules.auto_tier, proposal, flattering(proposal))
        assert result.outcome is ESCALATE
        if proposal.is_equity:
            # Every other criterion is met — declared equity, a limit buy of a
            # watchlist underlying within the max notional: the shape alone escalates.
            assert result.detail == f"needs approval: not a plain equity: {finding}"
        else:
            assert "the instrument is option, not equity" in result.detail

    def test_three_rules_stand_between_a_mislabelled_option_and_auto_execute(self):
        for label in ("an equity with legs", "an equity named by an option contract"):
            proposal, _ = BROKEN_SHAPES[label]
            assert non_pass(proposal, flattering(proposal)) == {
                "options_max_loss": REJECT,
                "options_escalate": ESCALATE,
                "auto_tier": ESCALATE,
            }

    def test_a_short_sale_escalates(self):
        result = self.tier(sell())
        assert result.outcome is ESCALATE and "does not close" in result.detail
        assert self.tier(sell(), positions=None).outcome is ESCALATE
        assert self.tier(sell(11), positions=(make_position(qty=10),)).outcome is ESCALATE

    def test_a_closing_sale_qualifies(self):
        assert self.tier(sell(), positions=(make_position(qty=10),)).outcome is PASS
        assert self.tier(sell(2), positions=(make_position(qty=2),)).outcome is PASS

    def test_a_market_order_escalates_even_when_market_orders_are_allowed(self):
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)
        result = self.tier(market, limits=make_limits(allow_market_orders=True))
        assert result.outcome is ESCALATE and "market" in result.detail

    def test_a_symbol_outside_the_watchlist_escalates_even_with_watchlist_only_off(self):
        proposal = make_proposal(symbol="TSLA")
        result = self.tier(proposal, limits=make_limits(watchlist_only=False))
        assert result.outcome is ESCALATE and "TSLA is not in the watchlist" in result.detail
        assert self.tier(watchlist=()).outcome is ESCALATE

    def test_boundary_at_the_max_notional_passes_one_dollar_over_escalates(self):
        assert self.tier(make_proposal(quantity=5)).outcome is PASS  # 1,000.00
        result = self.tier(make_proposal(quantity=5, limit_price=200.20))  # 1,001.00
        assert result.outcome is ESCALATE
        assert "$1,001.00" in result.detail and "$1,000.00" in result.detail
        assert "(risk_limits.auto_execute.max_notional)" in result.detail
        cent_over = make_proposal(quantity=5, limit_price=200.002)  # 1,000.01
        assert self.tier(cent_over).outcome is ESCALATE
        assert self.tier(make_proposal(quantity=5, limit_price=199.80)).outcome is PASS

    def test_the_max_notional_comes_from_the_limits(self):
        small = make_limits(auto_execute={"max_notional": 400.0})
        assert self.tier(limits=small).outcome is PASS
        smaller = make_limits(auto_execute={"max_notional": 399.99})
        assert self.tier(limits=smaller).outcome is ESCALATE
        assert self.tier(limits=make_limits(auto_execute={"max_notional": 0.0})).outcome is ESCALATE

    def test_float_noise_does_not_escalate(self):
        noisy = make_proposal(quantity=3, limit_price=100.01)
        assert self.tier(noisy, limits=self.up_to(300.03)).outcome is PASS
        assert self.tier(noisy, limits=self.up_to(300.02)).outcome is ESCALATE
        spec = make_proposal(quantity=10, limit_price=100.1)
        assert self.tier(spec, limits=self.up_to(1001.0)).outcome is PASS

    def test_an_unknown_notional_escalates(self):
        result = self.tier(make_proposal(limit_price=None))
        assert result.outcome is ESCALATE and "notional cannot be computed" in result.detail

    def test_a_closing_sale_is_still_held_to_the_max_notional(self):
        big = sell(10)  # 2,000.00
        assert self.tier(big, positions=(make_position(qty=10),)).outcome is ESCALATE

    def test_every_unmet_criterion_is_listed(self):
        proposal = make_proposal(
            symbol="TSLA",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            limit_price=None,
            quantity=50,
        )
        off = make_limits(auto_execute={"enabled": False})
        result = self.tier(proposal, limits=off)
        assert result.outcome is ESCALATE
        for text in (
            "auto-execute is disabled",
            "does not close",
            "the order type is market",
            "TSLA is not in the watchlist",
            "is above $1,000.00",
        ):
            assert text in result.detail
        option = self.tier(naked_short_call(), limits=off)
        assert "the instrument is option" in option.detail and "unlimited" in option.detail

    def test_it_never_rejects_or_flags(self):
        for build in [make_proposal, sell, *STRUCTURES]:
            for overrides in ({}, {"positions": None, "watchlist": ()}):
                assert self.tier(build(), **overrides).outcome in (PASS, ESCALATE)

    # --- every criterion holds for a closing sale too (the one sell that qualifies) ---

    HELD = (make_position(qty=10),)

    def test_a_closing_sale_escalates_when_auto_execute_is_disabled(self):
        closing = sell(2)
        assert self.tier(closing, positions=self.HELD).outcome is PASS
        off = make_limits(auto_execute={"enabled": False})
        result = self.tier(closing, positions=self.HELD, limits=off)
        assert result.outcome is ESCALATE
        assert result.detail == (
            "needs approval: auto-execute is disabled (risk_limits.auto_execute.enabled)"
        )

    def test_a_closing_market_sale_escalates_even_when_market_orders_are_allowed(self):
        closing = sell(2, order_type=OrderType.MARKET, limit_price=None)
        allowed = make_limits(allow_market_orders=True)
        context = context_for(closing, positions=self.HELD, limits=allowed)
        assert is_closing_sale(closing, context)
        result = run(rules.auto_tier, closing, context)
        assert result.outcome is ESCALATE
        assert result.detail == "needs approval: the order type is market, not limit"
        assert non_pass(closing, context) == {"auto_tier": ESCALATE}

    def test_a_closing_sale_outside_the_watchlist_escalates_with_watchlist_only_off(self):
        closing = make_proposal(symbol="TSLA", side=OrderSide.SELL)
        context = context_for(
            closing,
            positions=(make_position("TSLA", 10),),
            limits=make_limits(watchlist_only=False),
        )
        assert is_closing_sale(closing, context)
        result = run(rules.auto_tier, closing, context)
        assert result.outcome is ESCALATE
        assert result.detail == "needs approval: TSLA is not in the watchlist"
        assert non_pass(closing, context) == {"auto_tier": ESCALATE}

    def test_a_closing_sale_without_a_usable_limit_price_escalates(self):
        for price in (None, 0.0, -200.0):
            closing = sell(2, limit_price=price)
            context = make_context(positions=self.HELD)
            assert is_closing_sale(closing, context)
            result = run(rules.auto_tier, closing, context)
            assert result.outcome is ESCALATE
            assert result.detail == "needs approval: the notional cannot be computed"

    def test_a_sale_of_an_option_contract_declared_equity_and_held_long_never_qualifies(self):
        # The D2 shape on the sell side: "shares" of SPY260821C00640000, sold
        # out of a long position in that very symbol.
        mislabelled = make_proposal(
            symbol=CALL_640, side=OrderSide.SELL, quantity=1.0, limit_price=8.00
        )
        context = make_context(
            positions=(make_position(CALL_640, 2, current_price=8.0, market_value=1600.0),),
            quote=make_quote(CALL_640, 8.00),
        )
        assert is_closing_sale(mislabelled, context)  # the books do show that long
        result = run(rules.auto_tier, mislabelled, context)
        assert result.outcome is ESCALATE
        assert result.detail == (
            "needs approval: not a plain equity: the proposal is declared equity but its "
            "symbol SPY260821C00640000 is an option contract"
        )
        assert non_pass(mislabelled, context) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        with_legs = make_proposal(
            symbol="SPY", side=OrderSide.SELL, quantity=1.0, limit_price=640.0,
            legs=[make_leg("sell", "call", 640.0)],
        )
        held = make_context(
            positions=(make_position("SPY", 10, current_price=640.0),),
            quote=make_quote("SPY", 640.0),
            limits=make_limits(max_position_pct=100.0),
        )
        assert is_closing_sale(with_legs, held)
        result = run(rules.auto_tier, with_legs, held)
        assert result.outcome is ESCALATE and "not a plain equity" in result.detail

    def test_an_equity_whose_symbol_contains_whitespace_never_qualifies(self):
        # Ruling R4: any symbol with whitespace is a malformed shape.
        spaced = make_proposal(symbol="AA PL")
        context = make_context(quote=make_quote("AA PL"), watchlist=(*WATCHLIST, "AA PL"))
        result = run(rules.auto_tier, spaced, context)
        assert result.outcome is ESCALATE
        assert result.detail == (
            "needs approval: not a plain equity: symbol 'AA PL' is not a ticker: it "
            "contains whitespace"
        )
        assert non_pass(spaced, context) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        rejected = run(rules.options_max_loss, spaced, context)
        assert rejected.detail == (
            "the proposal is not a plain equity: symbol 'AA PL' is not a ticker: it "
            "contains whitespace"
        )

    def test_a_padded_contract_never_qualifies(self):
        proposal = padded_call()
        context = padded_context(proposal)
        assert non_pass(proposal, context) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        declared_equity = make_proposal(symbol=PADDED_640, limit_price=8.00)
        flattered = make_context(quote=make_quote(PADDED_640, 8.00))
        result = run(rules.auto_tier, declared_equity, flattered)
        assert result.outcome is ESCALATE and "not a plain equity" in result.detail
        assert "is not in the compact OCC form" in result.detail
        assert non_pass(declared_equity, flattered) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }

    # --- the tier's size --------------------------------------------------------

    def test_a_generous_max_notional_is_honoured(self):
        # Nothing here bounds what the config allows: 100 shares at 200.00.
        large = make_proposal(quantity=100)
        generous = make_limits(auto_execute={"max_notional": 50_000.0}, max_position_pct=100.0)
        result = self.tier(large, limits=generous)
        assert result.outcome is PASS
        assert "notional $20,000.00 within $50,000.00" in result.detail
        assert passing(large, context_for(large, limits=generous))
        tight = make_limits(auto_execute={"max_notional": 19_999.99}, max_position_pct=100.0)
        assert self.tier(large, limits=tight).outcome is ESCALATE

    @pytest.mark.parametrize("quantity", [1e-12, 5e-324, 2.0])
    def test_a_tier_of_zero_admits_nothing(self, quantity):
        # A notional of $2e-10 is "not above" zero by float noise alone.
        proposal = make_proposal(quantity=quantity)
        zero = make_limits(auto_execute={"max_notional": 0.0})
        result = self.tier(proposal, limits=zero)
        assert result.outcome is ESCALATE
        assert (
            "the tier admits nothing: max_notional is $0.00 "
            "(risk_limits.auto_execute.max_notional)"
        ) in result.detail
        assert non_pass(proposal, context_for(proposal, limits=zero)) == {"auto_tier": ESCALATE}
        # The smallest positive tier admits the smallest notional.
        if quantity < 1:
            sliver = make_limits(auto_execute={"max_notional": 1e-6})
            assert self.tier(proposal, limits=sliver).outcome is PASS

    def test_an_escalation_by_a_fraction_of_a_cent_shows_the_difference(self):
        over = self.tier(make_proposal(quantity=5.0, limit_price=200.0008))  # 1,000.004
        assert over.outcome is ESCALATE
        assert over.detail == (
            "needs approval: notional $1,000.00 is above $1,000.00 by $0.004 "
            "(risk_limits.auto_execute.max_notional)"
        )
        plain = self.tier(make_proposal(quantity=5, limit_price=200.002))  # 1,000.01
        assert "notional $1,000.01 is above $1,000.00 (" in plain.detail

    # --- ruling R9: the criteria are the user's six, and no others -----------------

    @pytest.mark.parametrize("unknown", [None, NAN, INF, -INF, -1.0, 0.0])
    def test_an_equity_buy_auto_executes_with_cash_and_options_buying_power_unknown(
        self, unknown
    ):
        # No rule reads cash or options buying power for shares: all twenty
        # PASS — what the engine turns into AUTO_EXECUTE. A recorded decision
        # (R9), not an oversight.
        context = make_context(cash=unknown, options_buying_power=unknown)
        assert passing(context=context)
        assert run(rules.auto_tier, context=context) == run(rules.auto_tier)

    def test_an_equity_buy_auto_executes_with_an_unrelated_line_in_context_errors(self):
        # `errors` explains what could not be fetched; no rule reads it. What a
        # rule needs and lacks is None in its own field, and that rejects.
        errors = ("account state: the broker reported no usable cash",)
        context = make_context(errors=errors, cash=None)
        assert passing(context=context)
        assert run(rules.auto_tier, context=context) == run(rules.auto_tier)

    @pytest.mark.parametrize("unknown", [None, NAN, INF, -INF, -1.0, 0.0])
    def test_a_genuine_closing_sale_auto_executes_with_equity_and_buying_power_unknown(
        self, unknown
    ):
        # Selling shares that are held consumes no buying power and reduces
        # the position: rules 8 and 9 pass it before looking at either figure.
        closing = sell(2)
        context = holding(10, equity=unknown, buying_power=unknown)
        assert is_closing_sale(closing, context)
        assert passing(closing, context)
        assert run(rules.auto_tier, closing, context) == run(rules.auto_tier, closing, holding(10))
        # ... which is only true of a GENUINE closing sale: without the long
        # the same context rejects it.
        bare = make_context(equity=unknown, buying_power=unknown)
        assert first_reject(closing, bare) == "buying_power"

    def test_the_criteria_are_exactly_the_users_six_and_the_two_guards(self):
        # enabled; equity (and a plain one: D2); a buy or a closing sale; a limit
        # order; a watchlist symbol; a known notional within a positive tier (R5).
        everything_wrong = make_proposal(
            symbol=CALL_640.replace("SPY", "TSLA"),
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            limit_price=None,
        )
        off = make_limits(auto_execute={"enabled": False, "max_notional": 0.0})
        result = run(rules.auto_tier, everything_wrong, make_context(limits=off, quote=None))
        assert result.outcome is ESCALATE
        unmet = result.detail.removeprefix("needs approval: ").split("; ")
        assert [text.split(":")[0].split(" (")[0] for text in unmet] == [
            "auto-execute is disabled",
            "not a plain equity",
            "a sell that does not close an existing long equity position",
            "the order type is market, not limit",
            "TSLA is not in the watchlist",
            "the tier admits nothing",
            "the notional cannot be computed",
        ]


# --- unknown is never good news -----------------------------------------------


class TestUnknownIsNeverGoodNews:
    def test_a_context_that_knows_nothing_rejects_and_never_passes_everything(self):
        blind = make_context(
            clock=None, equity=None, cash=None, buying_power=None, options_buying_power=None,
            start_of_day_equity=None, daily_pnl=None, positions=None, quote=None,
            controls=Controls(
                kill_switch=True, halt_unknown=True, problems=("controls unreadable",)
            ),
            errors=("account: unavailable", "clock: unavailable", "quote: unavailable"),
        )
        found = outcomes(context=blind)
        rejected = {name for name, outcome in found.items() if outcome is REJECT}
        assert rejected == {
            "kill_switch", "halted", "market_hours", "daily_loss_limit", "buying_power",
            "max_position_pct", "max_open_positions", "limit_price_sanity",
        }
        assert not passing(context=blind)

    @pytest.mark.parametrize("field", [*ACCOUNT_FIGURES, "positions", "clock", "quote"])
    def test_each_missing_figure_alone_is_enough_to_reject(self, field):
        found = non_pass(context=make_context(**{field: None}))
        assert REJECT in found.values()

    @pytest.mark.parametrize("field", ACCOUNT_FIGURES)
    @pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_a_non_finite_figure_is_as_unknown_as_a_missing_one(self, field, bad):
        assert non_pass(context=make_context(**{field: bad})) == non_pass(
            context=make_context(**{field: None})
        )

    @pytest.mark.parametrize("field", ["cash", "options_buying_power"])
    def test_an_equity_does_not_need_what_it_does_not_read(self, field):
        assert passing(context=make_context(**{field: None}))

    @pytest.mark.parametrize("build", DEFINED_RISK)
    def test_an_option_with_no_market_data_rejects(self, build):
        proposal = build()
        found = non_pass(proposal, make_context(quote=None))
        assert found["limit_price_sanity"] is REJECT
        assert found["options_max_loss"] is REJECT

    @pytest.mark.parametrize("build", [long_call, call_debit_spread, put_credit_spread])
    def test_an_option_beside_a_position_that_cannot_be_valued_rejects(self, build):
        # Unknown exposure in the underlying is not zero exposure — for an
        # option exactly as for shares. But for it the structure would only
        # need approval; with it the first (and only) rejecting rule is the cap.
        proposal = build()
        assert non_pass(proposal, context_for(proposal)) == {
            "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        unvalued = (make_position("SPY", current_price=None, market_value=None),)
        context = context_for(proposal, positions=unvalued)
        assert non_pass(proposal, context) == {
            "max_position_pct": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        assert first_reject(proposal, context) == "max_position_pct"
        # Unknown positions altogether: the count of open positions is unknown too.
        blind = context_for(proposal, positions=None)
        assert non_pass(proposal, blind) == {
            "max_position_pct": REJECT, "max_open_positions": REJECT,
            "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        assert first_reject(proposal, blind) == "max_position_pct"

    def test_a_short_sale_beside_a_short_that_cannot_be_valued_rejects(self):
        unvalued = (make_position(qty=-10, side="short", current_price=None, market_value=None),)
        context = make_context(positions=unvalued)
        assert non_pass(sell(), context) == {
            "max_position_pct": REJECT, "short_sale": ESCALATE, "auto_tier": ESCALATE,
        }
        assert first_reject(sell(), context) == "max_position_pct"


# --- hostile but valid --------------------------------------------------------


class TestLimitsAreFiniteNumbers:
    """``RiskLimits`` refuses a limit that is not a finite number — an
    infinite cap would silently switch its rule off — so the hostile limits
    below stop at the largest values it accepts."""

    FLOATS = (
        "daily_loss_limit_pct", "halt_fallback_hours", "max_position_pct",
        "limit_price_tolerance_pct", "max_loss_per_trade", "min_confidence",
    )
    COUNTS = (
        "max_daily_trades", "max_open_positions", "duplicate_window_minutes", "min_dte",
        "max_contracts",
    )
    NON_FINITE = pytest.mark.parametrize("bad", [INF, -INF, NAN], ids=["inf", "-inf", "nan"])

    def test_these_are_all_the_numeric_limits(self):
        numeric = {
            name for name, field in RiskLimits.model_fields.items()
            if field.annotation in (int, float)
        }
        assert numeric == {*self.FLOATS, *self.COUNTS}
        assert AutoExecuteConfig.model_fields["max_notional"].annotation is float

    @NON_FINITE
    @pytest.mark.parametrize("name", [*FLOATS, *COUNTS])
    def test_a_non_finite_limit_is_refused(self, name, bad):
        for build in (RiskLimits, make_limits):
            with pytest.raises(ValidationError) as refused:
                build(**{name: bad})
            assert [error["type"] for error in refused.value.errors()] == ["finite_number"]
            assert refused.value.errors()[0]["loc"] == (name,)

    @NON_FINITE
    def test_a_non_finite_auto_execute_notional_is_refused(self, bad):
        with pytest.raises(ValidationError) as refused:
            make_limits(auto_execute={"max_notional": bad})
        assert [error["type"] for error in refused.value.errors()] == ["finite_number"]
        assert refused.value.errors()[0]["loc"] == ("auto_execute", "max_notional")
        with pytest.raises(ValidationError):
            AutoExecuteConfig(max_notional=bad)

    def test_the_largest_finite_values_are_accepted(self):
        vast = make_limits(
            halt_fallback_hours=LARGEST_FINITE,
            limit_price_tolerance_pct=LARGEST_FINITE,
            max_loss_per_trade=LARGEST_FINITE,
            auto_execute={"max_notional": LARGEST_FINITE},
            max_daily_trades=10**5000,
            max_contracts=10**400,
        )
        assert LARGEST_FINITE == sys.float_info.max < INF
        assert vast.halt_fallback_hours == vast.max_loss_per_trade == LARGEST_FINITE
        assert vast.limit_price_tolerance_pct == LARGEST_FINITE
        assert vast.auto_execute.max_notional == LARGEST_FINITE
        assert vast.max_daily_trades == 10**5000 and vast.max_contracts == 10**400

    def test_the_hostile_limits_are_ones_the_config_accepts(self):
        vast = _hostile_contexts()[4].limits
        assert vast.max_loss_per_trade == vast.halt_fallback_hours == LARGEST_FINITE
        assert vast.limit_price_tolerance_pct == LARGEST_FINITE
        assert vast.auto_execute.max_notional == LARGEST_FINITE
        assert RiskLimits.model_validate(vast.model_dump()) == vast
        zero = _hostile_contexts()[3].limits
        assert zero.max_loss_per_trade == zero.auto_execute.max_notional == 0.0
        assert RiskLimits.model_validate(zero.model_dump()) == zero


def _hostile_proposals() -> list[ProposalUnderReview]:
    mixed = [
        make_leg("buy", "call", 640.0, 0.5, underlying="QQQ"),
        make_leg("sell", "put", 650.0, 3.25, index=1, expiration=date(2020, 1, 17)),
        make_leg("sell", "call", 1e-9, 1e300, index=2, symbol="NOT-AN-OCC-SYMBOL"),
    ]
    return [
        make_proposal(),
        sell(),
        make_proposal(order_type=OrderType.MARKET, limit_price=None),
        sell(order_type=OrderType.MARKET, limit_price=None),
        make_proposal(limit_price=None),
        make_proposal(limit_price=-5.0),
        make_proposal(limit_price=1e-320, quantity=1e-320),
        make_proposal(limit_price=1e308, quantity=1e308, confidence=0.0),
        make_proposal(symbol="\x1b[31mEVIL\n", invalidation="\x00", confidence=1.0),
        make_proposal(symbol="SPY260821C00640000"),  # an equity that names a contract
        make_proposal(legs=[make_leg("buy", "call", 640.0, 1.5)]),  # an equity with legs
        option_proposal([]),
        option_proposal(mixed),
        option_proposal(mixed, order_type=OrderType.MARKET, limit_price=None, side=OrderSide.SELL),
        option_proposal([make_leg("buy", "call", 640.0, 0.25)], symbol="ZZZ"),
        *(proposal for proposal, _ in BROKEN_SHAPES.values()),
        *(build() for build in STRUCTURES),
        *(build(order_type=OrderType.MARKET, limit_price=None) for build in STRUCTURES),
        *(build(quantity=1e12) for build in STRUCTURES),
        make_option(WASH_LEGS, side="sell", limit_price=0.01),
    ]


def _hostile_contexts() -> list[PolicyContext]:
    zero_limits = RiskLimits(
        daily_loss_limit_pct=1e-300, halt_fallback_hours=1e-300, max_daily_trades=0,
        max_open_positions=0, no_trade_list=[], watchlist_only=True, max_position_pct=1e-300,
        duplicate_window_minutes=0, allow_market_orders=True, limit_price_tolerance_pct=0.0,
        min_dte=0, max_loss_per_trade=0.0, max_contracts=0, reject_short_sales=True,
        min_confidence=0.0, auto_execute={"enabled": True, "max_notional": 0.0},
    )
    vast_limits = RiskLimits(
        daily_loss_limit_pct=100, halt_fallback_hours=LARGEST_FINITE, max_daily_trades=10**5000,
        max_open_positions=10**400, no_trade_list=["", "  ", "aapl", "SPY260821C00640000"],
        watchlist_only=False, max_position_pct=100, duplicate_window_minutes=10**400,
        allow_market_orders=True, limit_price_tolerance_pct=LARGEST_FINITE, min_dte=10**400,
        max_loss_per_trade=LARGEST_FINITE, max_contracts=10**400, min_confidence=1.0,
        auto_execute={"enabled": False, "max_notional": LARGEST_FINITE},
    )
    weird_positions = (
        make_position("", 0.0, side=""),
        make_position(qty=NAN, market_value=NAN, current_price=NAN),
        make_position("SPY", INF, market_value=-INF, current_price=-INF),
        make_position(
            "SPY260821C00640000", -1e308, side="short", market_value=None, current_price=1e308
        ),
        make_position("not an occ symbol", 5, side="LONG", market_value=-3.0),
        make_position(qty=-10, side="long"),
    )
    weird_orders = (
        make_order(limit_price=None),
        make_order("SPY260821C00640000", id="o2", client_order_id="c2", limit_price=-1.0),
        make_order("SPY", id="o3", client_order_id="\x1b[2J", quantity=1e308, limit_price=1e308),
    )
    far = datetime.max.replace(tzinfo=timezone(timedelta(hours=-14)))
    near = datetime.min.replace(tzinfo=timezone(timedelta(hours=14)))
    condor = iron_condor()
    return [
        make_context(),
        context_for(condor),
        make_context(watchlist=()),
        make_context(limits=zero_limits),
        make_context(limits=vast_limits, orders_today=10**5000, contract_multiplier=10**400),
        make_context(quote=make_quote("MSFT")),
        make_context(quote=make_quote(bid=INF, ask=-INF)),
        make_context(quote=make_quote(bid=1e308, ask=1e308)),
        make_context(quote=make_quote(bid=5e-324, ask=5e-324)),
        make_context(positions=weird_positions, open_orders=weird_orders),
        make_context(positions=None, open_orders=weird_orders, clock=None, quote=None),
        make_context(
            equity=NAN, cash=INF, buying_power=-INF, options_buying_power=NAN,
            start_of_day_equity=INF, daily_pnl=NAN,
        ),
        make_context(equity=1e308, start_of_day_equity=1e308, daily_pnl=-1e308, buying_power=1e308),
        make_context(equity=-1.0, start_of_day_equity=5e-324, daily_pnl=-5e-324, buying_power=0.0),
        make_context(daily_pnl=-1e9, clock=make_clock(next_open=far, next_close=far)),
        make_context(daily_pnl=-1e9, clock=make_clock(False, next_open=near, next_close=near)),
        make_context(
            daily_pnl=-1e9, clock=make_clock(next_open=None, next_close=None), limits=vast_limits
        ),
        make_context(
            clock=make_clock(
                next_open=datetime(2026, 7, 31, 9, 30), next_close=datetime(2026, 7, 30, 16)
            ),
            controls=Controls(halt_until=datetime(2026, 7, 30, 15, 0), halt_unknown=True),
        ),
        make_context(
            now=datetime.max.replace(tzinfo=timezone.utc), daily_pnl=-1e9, clock=None,
            market_date=date.max,
        ),
        make_context(now=datetime.min.replace(tzinfo=timezone.utc), market_date=date.min),
        make_context(
            recent_proposals=(
                make_recent(),
                make_recent(id=PROPOSAL_ID),
                make_recent(id="", created_at=PROPOSED_AT),
                make_recent(id="zzz", created_at=datetime.max.replace(tzinfo=timezone.utc)),
                make_recent(id="aaa", created_at=datetime.min.replace(tzinfo=timezone.utc)),
            ),
        ),
        context_for(
            condor,
            leg_quotes=(
                make_snapshot(condor.legs[0].symbol, 1.0, bid=NAN, ask=INF),
                make_snapshot(condor.legs[1].symbol, 1.0, bid=2.0, ask=1.0),
                make_snapshot(condor.legs[2].symbol, 1e308, bid=1e308, ask=1e308),
                make_snapshot("SOMETHING-ELSE", 1.0),
            ),
            analysis=context_for(condor).analysis.model_copy(
                update={"max_loss": NAN, "unlimited_risk": False}
            ),
        ),
        context_for(
            condor,
            analysis=context_for(condor).analysis.model_copy(update={"max_loss": -INF}),
            contract_multiplier=1,
        ),
    ]


def _check_all_rules(
    proposal: ProposalUnderReview, context: PolicyContext
) -> tuple[RuleResult, ...]:
    found = tuple(rule(proposal, context) for rule in RULES)
    assert tuple(result.name for result in found) == RULE_NAMES
    for result in found:
        assert isinstance(result, RuleResult)
        assert isinstance(result.outcome, RuleOutcome)
        assert result.detail.strip()
        if result.name != "daily_loss_limit":
            assert result.halt_until is None
        elif result.halt_until is not None:
            assert result.outcome is REJECT
            assert result.halt_until.utcoffset() == timedelta(0)
    assert account_rules(context) == tuple(r for r in found if r.name in ACCOUNT_RULE_NAMES)
    return found


class TestHostileButValidInputs:
    def test_an_empty_watchlist(self):
        found = non_pass(context=make_context(watchlist=()))
        assert found == {"watchlist_only": REJECT, "auto_tier": ESCALATE}

    def test_zero_limits(self):
        context = _hostile_contexts()[3]
        found = _check_all_rules(make_proposal(), context)
        outcome = {result.name: result.outcome for result in found}
        assert outcome["max_daily_trades"] is REJECT
        assert outcome["max_open_positions"] is REJECT
        assert outcome["max_position_pct"] is REJECT
        assert outcome["auto_tier"] is ESCALATE
        assert outcome["min_confidence"] is PASS
        option = iron_condor()
        found = _check_all_rules(option, context_for(option, limits=context.limits))
        outcome = {result.name: result.outcome for result in found}
        assert outcome["options_max_contracts"] is REJECT and outcome["options_max_loss"] is REJECT

    def test_legs_with_fractional_quantities(self):
        proposal = option_proposal(
            [make_leg("buy", "call", 640.0, 0.5), make_leg("sell", "call", 650.0, 2.75, index=1)]
        )
        context = make_context(quote=None, leg_quotes=leg_quotes_for(proposal))
        outcome = {r.name: r.outcome for r in _check_all_rules(proposal, context)}
        assert outcome["options_max_loss"] is REJECT
        assert outcome["options_max_contracts"] is PASS
        assert outcome["options_escalate"] is ESCALATE

    def test_a_quote_for_another_symbol(self):
        context = make_context(quote=make_quote("MSFT"))
        outcome = {r.name: r.outcome for r in _check_all_rules(make_proposal(), context)}
        assert outcome["limit_price_sanity"] is REJECT
        market = make_proposal(order_type=OrderType.MARKET, limit_price=None)
        outcome = {r.name: r.outcome for r in _check_all_rules(market, context)}
        assert outcome["buying_power"] is REJECT and outcome["max_position_pct"] is REJECT

    def test_an_equity_proposal_that_carries_legs(self):
        proposal = make_proposal(legs=[make_leg("buy", "call", 640.0, 1.5, underlying="GME")])
        outcome = {r.name: r.outcome for r in _check_all_rules(proposal, make_context())}
        # The stray leg's underlying is still held to the watchlist.
        assert outcome["watchlist_only"] is REJECT
        # It is option-like whatever it is declared: rejected, and never auto-tier.
        assert outcome["options_max_loss"] is REJECT
        assert outcome["options_escalate"] is ESCALATE
        assert outcome["auto_tier"] is ESCALATE
        assert outcome["options_min_dte"] is PASS and outcome["options_max_contracts"] is PASS

    def test_limits_too_large_to_print_or_to_be_a_duration(self):
        context = _hostile_contexts()[4]
        found = {r.name: r for r in _check_all_rules(long_call(), context)}
        assert found["max_daily_trades"].outcome is REJECT  # 10**5000 of 10**5000
        assert "too large to print" in found["max_daily_trades"].detail
        assert found["options_max_contracts"].outcome is PASS
        assert found["options_min_dte"].outcome is REJECT

    def test_no_rule_raises_for_any_hostile_pair(self):
        proposals, contexts = _hostile_proposals(), _hostile_contexts()
        assert len(proposals) >= 30 and len(contexts) >= 20
        for proposal in proposals:
            for context in contexts:
                first = _check_all_rules(proposal, context)
                assert _check_all_rules(proposal, context) == first  # and deterministically

    def test_no_rule_raises_across_a_seeded_sweep_of_figures(self):
        rng = random.Random(20260730)
        figures = [None, NAN, INF, -INF, 0.0, -0.0, 1e-320, -1e-9, 0.01, 399.99, 400.0, 400.01]
        figures += [1e308, -1e308]
        proposals = _hostile_proposals()
        analyses = [None, analysis_for(iron_condor()), analysis_for(naked_short_call())]
        clocks = [None, make_clock(), make_clock(False), make_clock(next_close=NOW)]
        halt = Controls(halt_until=NOW + timedelta(seconds=1))
        controls = [Controls(), Controls(kill_switch=True), halt]
        for _ in range(400):
            proposal = rng.choice(proposals)
            position = make_position(
                qty=rng.choice(figures[1:]), market_value=rng.choice(figures)
            )
            quote = make_quote(bid=rng.choice(figures), ask=rng.choice(figures))
            context = make_context(
                equity=rng.choice(figures),
                cash=rng.choice(figures),
                buying_power=rng.choice(figures),
                options_buying_power=rng.choice(figures),
                start_of_day_equity=rng.choice(figures),
                daily_pnl=rng.choice(figures),
                orders_today=rng.choice([0, 9, 10, 11, 10**30]),
                positions=rng.choice([None, (), (position,)]),
                quote=rng.choice([None, quote, make_quote("QQQ")]),
                leg_quotes=tuple(
                    make_snapshot(leg.symbol, 1.0, bid=rng.choice(figures), ask=rng.choice(figures))
                    for leg in proposal.legs
                    if rng.random() < 0.8
                ),
                analysis=rng.choice(analyses),
                clock=rng.choice(clocks),
                controls=rng.choice(controls),
                watchlist=rng.choice([(), WATCHLIST, ("zzz",)]),
            )
            _check_all_rules(proposal, context)


# --- the pure core: no hardcoded limits, no outside reach ---------------------

POLICY_DIR = REPO_ROOT / "aegis" / "policy"
ALLOWED_IMPORTS = {
    "__future__", "math", "datetime", "collections.abc", "typing", "aegis.policy.models",
    "aegis.config", "aegis.data.models", "aegis.pricing.models", "aegis.store.models",
}
# 0/1/100 for signs and percent arithmetic; the float-noise tolerance in measures.py only.
ALLOWED_NUMBERS = {"rules.py": {0, 1, 100}, "measures.py": {0, 1, 100, 1e-9}}


def _tree(filename: str) -> ast.Module:
    return ast.parse((POLICY_DIR / filename).read_text(encoding="utf-8"), filename=filename)


def _imports(tree: ast.Module) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not used in aegis.policy"
            found.add(node.module)
    return found


def _numbers(tree: ast.Module) -> set[float]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
    }


def _literal_arithmetic(tree: ast.Module) -> list[int]:
    """Lines where two numeric literals are combined (``100 * 100``): a number
    the allowed literals 0/1/100 could spell without the scan above seeing it."""

    def numeric(node: ast.AST) -> bool:
        if isinstance(node, ast.UnaryOp):
            return numeric(node.operand)
        if isinstance(node, ast.BinOp):
            return numeric(node.left) and numeric(node.right)
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
        )

    return sorted(
        {node.lineno for node in ast.walk(tree) if isinstance(node, ast.BinOp) and numeric(node)}
    )


def _bounded_by_a_literal(tree: ast.Module) -> list[int]:
    """Lines where ``min`` / ``max`` takes a numeric literal beside something
    else (``min(limits.max_contracts, 100)``): a bound that is not config."""
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("min", "max")):
            continue
        literal = [
            isinstance(arg, (ast.Constant, ast.BinOp, ast.UnaryOp))
            and not any(isinstance(inner, (ast.Name, ast.Attribute)) for inner in ast.walk(arg))
            for arg in node.args
        ]
        if any(literal) and len(node.args) > 1:
            found.add(node.lineno)
    return sorted(found)


class TestPureCore:
    def test_the_arithmetic_checkers_see_what_they_should(self):
        hidden = ast.parse(
            "a = min(limits.max_notional, 100 * 100)\n"
            "b = max(cap, 100)\n"
            "c = 100 * 100 * 100\n"
            "d = -(100 * 100)\n"
        )
        assert _literal_arithmetic(hidden) == [1, 3, 4]
        assert _bounded_by_a_literal(hidden) == [1, 2]
        honest = ast.parse(
            "a = percent / 100 * equity\n"
            "b = min(power, cash)\n"
            "c = max(leg.quantity for leg in legs)\n"
            "d = abs(limit - mid) / mid * 100\n"
            "e = -cap\n"
        )
        assert _literal_arithmetic(honest) == [] and _bounded_by_a_literal(honest) == []

    def test_no_bound_is_spelled_with_the_allowed_literals(self):
        # rules.py compares against context.limits alone: no literal arithmetic,
        # and no min()/max() that caps a configured limit at a number.
        tree = _tree("rules.py")
        assert _literal_arithmetic(tree) == []
        assert _bounded_by_a_literal(tree) == []
        # measures.py: the one literal bound is exceeds()'s max(1.0, ...), the
        # float-noise scale — documented there as not a risk limit.
        measures_tree = _tree("measures.py")
        assert _literal_arithmetic(measures_tree) == []
        (line,) = _bounded_by_a_literal(measures_tree)
        source = (POLICY_DIR / "measures.py").read_text(encoding="utf-8").splitlines()
        assert "TOLERANCE * max(1.0, abs(value), abs(limit))" in source[line - 1]

    def test_the_checkers_see_what_they_should(self):
        snippet = ast.parse(
            "import os, sqlite3\nfrom aegis.store import repo\nfrom time import time\n"
            "CAP = 5000.0\nx = -1\ny = f'{x:.10g}'\nz = True\n"
        )
        assert _imports(snippet) == {"os", "sqlite3", "aegis.store", "time"}
        assert _numbers(snippet) == {5000.0, 1}

    @pytest.mark.parametrize("filename", ["measures.py", "rules.py"])
    def test_imports_stay_inside_the_allowlist(self, filename):
        allowed = ALLOWED_IMPORTS | ({"aegis.policy.measures"} if filename == "rules.py" else set())
        assert _imports(_tree(filename)) <= allowed

    @pytest.mark.parametrize("filename", ["measures.py", "rules.py"])
    def test_no_risk_number_is_hardcoded(self, filename):
        assert _numbers(_tree(filename)) <= ALLOWED_NUMBERS[filename]

    @pytest.mark.parametrize("filename", ["measures.py", "rules.py"])
    def test_no_clock_is_read(self, filename):
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(_tree(filename))
            if isinstance(node, ast.Call)
        }
        assert calls.isdisjoint({"utcnow", "now", "today", "time", "monotonic"})

    def test_the_tolerance_is_the_only_module_level_number_and_is_documented(self):
        assert measures.TOLERANCE == 1e-9
        source = (POLICY_DIR / "measures.py").read_text(encoding="utf-8")
        assert "NOT a risk limit" in source

    def test_each_rule_and_measure_is_documented(self):
        for rule in RULES:
            assert rule.__doc__ and rule.__doc__.strip()
        for name in (
            "known", "exceeds", "below", "money", "long_quantity", "is_closing_sale",
            "signed_limit", "mid_price", "quote_problems", "worst_case_entry", "entry_price",
            "notional", "exposure", "held_underlyings", "daily_loss_cap", "leg_dte",
            "structure_problems", "is_occ", "is_plain_equity",
        ):
            assert getattr(measures, name).__doc__.strip()
