"""The context builder — tmp_path stores, every fetcher monkeypatched, no network.

``build_context`` runs against a real (temporary) store with the four live
fetchers replaced through the seam the module exposes on purpose
(``aegis.policy.context.get_spot`` etc.), serving models parsed from the
canned fixtures the data-layer tests use. ``structure_analysis`` is pure and
is checked against hand-worked expiry arithmetic and against the pricing
engine fed the per-leg premiums it replaces. ``load_proposal`` /
``latest_undecided`` read proposals back from the store.
"""

import ast
import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from policy_factories import (
    CHAIN_MIDS,
    EXPIRY,
    MULTIPLIER,
    NEXT_CLOSE,
    NEXT_OPEN,
    NOW,
    PROPOSAL_ID,
    PROPOSED_AT,
    WATCHLIST,
    call_debit_spread,
    exchange_date,
    iron_condor,
    long_call,
    make_context,
    make_leg,
    make_option,
    make_order,
    make_proposal,
    make_quote,
    naked_short_call,
    non_pass,
    occ,
    outcomes,
    put_credit_spread,
    results,
)

import aegis.data.account
import aegis.data.market
import aegis.policy.context as context_module
from aegis.config import AegisConfig, ConfigError, PricingConfig, RiskLimits
from aegis.data.errors import DataError
from aegis.data.models import (
    AccountState,
    ChainSnapshot,
    MarketClock,
    OptionSnapshot,
    OptionType,
    Position,
    Quote,
)
from aegis.policy import measures
from aegis.policy.context import build_context, latest_undecided, load_proposal, structure_analysis
from aegis.policy.engine import decide, evaluate
from aegis.policy.errors import PolicyError
from aegis.policy.models import PolicyContext, ProposalUnderReview, RuleOutcome, underlying_of
from aegis.policy.rules import RULE_NAMES, account_rules
from aegis.pricing.errors import PricingError
from aegis.pricing.models import OptionLeg, PositionSummary, PremiumType, Side
from aegis.pricing.position import analyze_position
from aegis.store.db import TABLES, connect, open_store
from aegis.store.errors import StoreError
from aegis.store.models import (
    Instrument,
    OrderSide,
    OrderStatus,
    OrderType,
    PolicyDecision,
    Verdict,
)
from aegis.store.repo import (
    get_controls,
    get_open_orders,
    get_recent_events,
    insert_proposal,
    record_decision,
    set_halt_until,
    set_kill_switch,
    upsert_order,
)

PASS, ESCALATE, REJECT = RuleOutcome.PASS, RuleOutcome.ESCALATE, RuleOutcome.REJECT

UTC = timezone.utc
NAN = float("nan")
INF = float("inf")

FETCHER_NAMES = ("get_market_clock", "get_account_state", "get_spot", "get_option_chain")
# Captured at import, before any test replaces them: what the module really calls.
LIVE_FETCHERS = {name: getattr(context_module, name) for name in FETCHER_NAMES}

CONFIG = AegisConfig(watchlist=list(WATCHLIST))
"""Default limits, multiplier 100, the exchange in America/New_York."""

LATER_EXPIRY = date(2026, 9, 18)

# The canned SPY chain expiring EXPIRY, as (bid, ask). The 640 call is
# tests/fixtures/option_snapshot_full.json verbatim; the others are that
# payload with another symbol and quote.
CHAIN_QUOTES: dict[tuple[OptionType, float], tuple[float, float]] = {
    (OptionType.CALL, 640.0): (12.05, 12.35),  # mid 12.20
    (OptionType.CALL, 650.0): (7.00, 7.20),  # mid 7.10
    (OptionType.CALL, 660.0): (3.40, 3.60),  # mid 3.50
    (OptionType.PUT, 630.0): (6.90, 7.10),  # mid 7.00
    (OptionType.PUT, 620.0): (4.40, 4.60),  # mid 4.50
}
VERTICAL_MID = 5.10  # 12.20 - 7.10: the 640/650 call spread at the canned mids
CONDOR_MID = 6.10  # 7.10 - 3.50 + 7.00 - 4.50: the iron condor's credit


def config_with(*, pricing=None, **risk):
    """``CONFIG`` with some risk limits (and optionally the pricing block) replaced."""
    return AegisConfig(
        watchlist=list(WATCHLIST),
        risk_limits=RiskLimits(**risk),
        pricing=pricing or PricingConfig(),
    )


# --- the fake data layer ------------------------------------------------------


class Feeds:
    """Stands in for the four live fetchers: serves canned models (or raises a
    canned error) and records every call, in order."""

    def __init__(self, clock, account, quotes, chains):
        self.clock = clock
        self.account = account
        self.quotes = dict(quotes)
        self.chains = dict(chains)
        self.calls: list[tuple] = []

    @staticmethod
    def _serve(value):
        if isinstance(value, BaseException):
            raise value
        return value

    def get_market_clock(self):
        self.calls.append(("clock",))
        return self._serve(self.clock)

    def get_account_state(self):
        self.calls.append(("account",))
        return self._serve(self.account)

    def get_spot(self, symbol):
        self.calls.append(("spot", symbol))
        missing = DataError("spot quote", symbol, LookupError("no canned quote"))
        return self._serve(self.quotes.get(symbol, missing))

    def get_option_chain(self, underlying, expiration):
        self.calls.append(("chain", underlying, expiration))
        missing = DataError("option chain", underlying, LookupError("no canned chain"))
        return self._serve(self.chains.get((underlying, expiration), missing))

    def called(self, kind):
        return [call for call in self.calls if call[0] == kind]


def contract(payload, symbol, bid, ask, underlying="SPY"):
    """The option fixture's payload re-quoted for another contract."""
    quote = {**payload["latest_quote"], "symbol": symbol, "bid_price": bid, "ask_price": ask}
    return OptionSnapshot.from_alpaca(
        symbol, underlying, {**payload, "symbol": symbol, "latest_quote": quote}, fetched_at=NOW
    )


def chain_of(payload, underlying="SPY", expiration=EXPIRY, quotes=None):
    quotes = CHAIN_QUOTES if quotes is None else quotes
    contracts = [
        contract(
            payload,
            occ(option_type, strike, underlying=underlying, expiration=expiration),
            bid,
            ask,
            underlying,
        )
        for (option_type, strike), (bid, ask) in quotes.items()
    ]
    return ChainSnapshot(
        underlying=underlying, expiration=expiration, contracts=contracts, fetched_at=NOW
    )


@pytest.fixture(autouse=True)
def no_live_fetch(monkeypatch):
    """No test may reach the data layer: until ``feeds`` installs its fakes,
    each fetcher is a tripwire, and a test that tripped one fails."""
    tripped = []

    def tripwire(name):
        def fetch(*args):
            tripped.append((name, *args))
            raise DataError(f"{name} (not faked in this test)")

        return fetch

    for name in FETCHER_NAMES:
        monkeypatch.setattr(context_module, name, tripwire(name))
    yield
    assert tripped == []


@pytest.fixture
def account(fixture):
    data = fixture("account.json")
    return AccountState.from_alpaca(data["account"], data["positions"], fetched_at=NOW)


@pytest.fixture
def open_clock(fixture):
    return MarketClock.from_alpaca(fixture("clock.json"), fetched_at=NOW)


@pytest.fixture
def spy_quote(fixture):
    data = fixture("stock_quote.json")
    return Quote.from_alpaca("SPY", quote=data["quote"], trade=data["trade"], fetched_at=NOW)


@pytest.fixture
def option_payload(fixture):
    return fixture("option_snapshot_full.json")


@pytest.fixture
def feeds(monkeypatch, account, open_clock, spy_quote, option_payload):
    """The four fetchers, faked from the fixtures: an open market, the paper
    account, a SPY quote and the SPY chain expiring ``EXPIRY``."""
    fake = Feeds(
        clock=open_clock,
        account=account,
        quotes={"SPY": spy_quote},
        chains={("SPY", EXPIRY): chain_of(option_payload)},
    )
    for name in FETCHER_NAMES:
        monkeypatch.setattr(context_module, name, getattr(fake, name))
    return fake


@pytest.fixture
def conn(tmp_path):
    connection = open_store(tmp_path / "aegis.db")
    yield connection
    connection.close()


# --- proposals and store rows -------------------------------------------------


def stored(conn, proposal: ProposalUnderReview) -> ProposalUnderReview:
    insert_proposal(conn, proposal.proposal, proposal.legs)
    return proposal


def spy_buy(**overrides) -> ProposalUnderReview:
    """An equity LIMIT BUY of 1 SPY at the fixture quote's mid (636.42)."""
    return make_proposal(**{"symbol": "SPY", "quantity": 1.0, "limit_price": 636.42, **overrides})


def spy_vertical(**overrides) -> ProposalUnderReview:
    """Buy the 640 call, sell the 650 call, at the canned chain's net mid."""
    return call_debit_spread(**{"limit_price": VERTICAL_MID, **overrides})


def spy_condor(**overrides) -> ProposalUnderReview:
    return iron_condor(**{"limit_price": CONDOR_MID, **overrides})


def decide_on(conn, proposal_id, verdict=Verdict.REJECT, decided_at=NOW):
    return record_decision(
        conn,
        PolicyDecision(
            proposal_id=proposal_id, decided_at=decided_at, verdict=verdict, rules_evaluated=[]
        ),
    )


def add_order(conn, number, submitted_at, *, status=OrderStatus.FILLED, symbol="MSFT"):
    """One order of the (already stored) default proposal, sent at ``submitted_at``."""
    return upsert_order(
        conn,
        make_order(
            symbol,
            id=f"order-{number:04d}",
            client_order_id=f"client-{number:04d}",
            proposal_id=PROPOSAL_ID,
            status=status,
            submitted_at=submitted_at,
            updated_at=submitted_at or NOW,
        ),
    )


def dump(conn) -> dict[str, list[tuple]]:
    """Every row of every table — what "nothing was written" is compared on."""
    queries = {table: f"SELECT * FROM {table} ORDER BY rowid" for table in TABLES}
    return {table: [tuple(row) for row in conn.execute(sql)] for table, sql in queries.items()}


def build(conn, proposal=None, *, config=CONFIG, now=NOW) -> PolicyContext:
    return build_context(conn, proposal, config=config, now=now)


def split_premium_analysis(proposal, mids=None, multiplier=MULTIPLIER) -> PositionSummary:
    """The pricing engine on the legs at their OWN premiums — what
    ``structure_analysis`` must reproduce from the net price alone."""
    mids = CHAIN_MIDS if mids is None else mids
    return analyze_position(
        [
            OptionLeg(
                option_type=leg.option_type,
                side=Side.LONG if leg.side is OrderSide.BUY else Side.SHORT,
                quantity=int(leg.quantity),
                strike=leg.strike,
                expiry=leg.expiration,
                premium=mids[(leg.option_type, leg.strike)],
            )
            for leg in proposal.legs
        ],
        contract_multiplier=multiplier,
    )


def net_mid(proposal, mids=None) -> float:
    """The signed net price per proposal unit at the per-leg mids (debit +, credit -)."""
    mids = CHAIN_MIDS if mids is None else mids
    return sum(
        (1 if leg.side is OrderSide.BUY else -1)
        * (leg.quantity / proposal.proposal.quantity)
        * mids[(leg.option_type, leg.strike)]
        for leg in proposal.legs
    )


def ratio_spread(**overrides) -> ProposalUnderReview:
    """Buy 1 of the 640 call, sell 2 of the 650 call: a 0.80 debit, one call naked."""
    legs = [("buy", "call", 640.0), ("sell", "call", 650.0, 2.0)]
    return make_option(legs, **{"side": "buy", "limit_price": 0.80, **overrides})


def backspread(**overrides) -> ProposalUnderReview:
    """Sell 1 of the 640 call, buy 2 of the 650 call, for a 0.80 credit."""
    legs = [("sell", "call", 640.0), ("buy", "call", 650.0, 2.0)]
    return make_option(legs, **{"side": "sell", "limit_price": 0.80, **overrides})


def butterfly(**overrides) -> ProposalUnderReview:
    """Buy the 640 and 660 calls, sell 2 of the 650: a 2.00 debit."""
    legs = [("buy", "call", 640.0), ("sell", "call", 650.0, 2.0), ("buy", "call", 660.0)]
    return make_option(legs, **{"side": "buy", "limit_price": 2.00, **overrides})


def short_strangle(**overrides) -> ProposalUnderReview:
    """Sell the 650 call and the 630 put for an 8.10 credit: unlimited risk."""
    legs = [("sell", "call", 650.0), ("sell", "put", 630.0)]
    return make_option(legs, **{"side": "sell", "limit_price": 8.10, **overrides})


def long_put(**overrides) -> ProposalUnderReview:
    return make_option(
        [("buy", "put", 630.0)], **{"side": "buy", "limit_price": 4.50, **overrides}
    )


# --- load_proposal / latest_undecided -----------------------------------------


class TestLoadProposal:
    def test_an_equity_proposal_has_no_legs(self, conn):
        proposal = stored(conn, make_proposal())
        loaded = load_proposal(conn, PROPOSAL_ID)
        assert isinstance(loaded, ProposalUnderReview)
        assert loaded == proposal
        assert loaded.legs == ()
        assert loaded.is_equity

    def test_an_option_proposal_comes_with_its_legs_in_leg_order(self, conn):
        condor = iron_condor()
        # Written out of order: the legs come back by leg_index, not by insertion.
        insert_proposal(conn, condor.proposal, list(reversed(condor.legs)))
        loaded = load_proposal(conn, PROPOSAL_ID)
        assert loaded == condor
        assert [leg.leg_index for leg in loaded.legs] == [0, 1, 2, 3]
        assert loaded.is_option

    def test_only_its_own_legs(self, conn):
        stored(conn, call_debit_spread())
        stored(conn, put_credit_spread(id="prop-0002"))
        assert load_proposal(conn, PROPOSAL_ID).legs == call_debit_spread().legs
        assert load_proposal(conn, "prop-0002").legs == put_credit_spread(id="prop-0002").legs

    def test_an_unknown_id_is_a_policy_error(self, conn):
        stored(conn, make_proposal())
        with pytest.raises(PolicyError) as info:
            load_proposal(conn, "no-such-proposal")
        assert info.value.what == "load proposal (no such proposal)"
        assert info.value.key == "no-such-proposal"
        assert str(info.value) == (
            "policy failed: load proposal (no such proposal) for no-such-proposal"
        )

    def test_a_store_failure_propagates(self, tmp_path):
        raw = connect(tmp_path / "unmigrated.db")  # no tables at all
        try:
            with pytest.raises(StoreError, match="get proposal"):
                load_proposal(raw, PROPOSAL_ID)
        finally:
            raw.close()

    def test_reads_only(self, conn):
        stored(conn, iron_condor())
        before = dump(conn)
        load_proposal(conn, PROPOSAL_ID)
        assert dump(conn) == before


class TestLatestUndecided:
    def test_an_empty_store_has_none(self, conn):
        assert latest_undecided(conn) is None

    def test_the_newest_proposal_without_a_decision_with_its_legs(self, conn):
        stored(conn, make_proposal(id="prop-old", created_at=NOW - timedelta(hours=2)))
        newest = stored(conn, iron_condor(id="prop-new", created_at=NOW - timedelta(minutes=5)))
        found = latest_undecided(conn)
        assert found == newest
        assert len(found.legs) == 4

    def test_a_decided_proposal_is_skipped(self, conn):
        older = stored(conn, make_proposal(id="prop-old", created_at=NOW - timedelta(hours=2)))
        stored(conn, iron_condor(id="prop-new", created_at=NOW - timedelta(minutes=5)))
        decide_on(conn, "prop-new")
        assert latest_undecided(conn) == older

    def test_any_verdict_counts_as_decided(self, conn):
        stored(conn, make_proposal())
        decide_on(conn, PROPOSAL_ID, Verdict.AUTO_EXECUTE)
        assert latest_undecided(conn) is None

    def test_a_store_failure_propagates(self, tmp_path):
        raw = connect(tmp_path / "unmigrated.db")
        try:
            with pytest.raises(StoreError, match="get latest undecided proposal"):
                latest_undecided(raw)
        finally:
            raw.close()


# --- structure_analysis -------------------------------------------------------


def analysed(proposal, entry_price, multiplier=MULTIPLIER):
    return structure_analysis(proposal, entry_price=entry_price, contract_multiplier=multiplier)


class TestStructureAnalysis:
    def test_a_single_long_call(self):
        summary = analysed(long_call(), 8.00)
        assert isinstance(summary, PositionSummary)
        assert summary.net_premium == pytest.approx(800.0)
        assert summary.premium_type is PremiumType.DEBIT
        assert summary.max_loss == pytest.approx(800.0)
        assert summary.max_profit is None and summary.unlimited_profit is True
        assert summary.unlimited_risk is False
        assert summary.breakevens == pytest.approx((648.0,))

    def test_a_debit_vertical(self):
        summary = analysed(call_debit_spread(), 4.40)
        assert summary.net_premium == pytest.approx(440.0)
        assert summary.premium_type is PremiumType.DEBIT
        assert summary.max_loss == pytest.approx(440.0)  # the debit paid
        assert summary.max_profit == pytest.approx(560.0)  # the width less the debit
        assert summary.breakevens == pytest.approx((644.40,))
        assert (summary.unlimited_risk, summary.unlimited_profit) == (False, False)

    def test_a_credit_vertical(self):
        summary = analysed(put_credit_spread(), -2.00)
        assert summary.net_premium == pytest.approx(-200.0)
        assert summary.premium_type is PremiumType.CREDIT
        assert summary.max_loss == pytest.approx(800.0)  # the width less the credit
        assert summary.max_profit == pytest.approx(200.0)  # the credit kept
        assert summary.breakevens == pytest.approx((628.0,))
        assert (summary.unlimited_risk, summary.unlimited_profit) == (False, False)

    def test_an_iron_condor(self):
        summary = analysed(iron_condor(), -4.40)
        assert summary.net_premium == pytest.approx(-440.0)
        assert summary.premium_type is PremiumType.CREDIT
        assert summary.max_loss == pytest.approx(560.0)
        assert summary.max_profit == pytest.approx(440.0)
        assert summary.breakevens == pytest.approx((625.60, 654.40))
        assert (summary.unlimited_risk, summary.unlimited_profit) == (False, False)

    def test_a_naked_short_call_has_unlimited_risk(self):
        summary = analysed(naked_short_call(), -3.60)
        assert summary.unlimited_risk is True
        assert summary.max_loss is None
        assert summary.max_profit == pytest.approx(360.0)
        assert summary.net_premium == pytest.approx(-360.0)
        assert summary.breakevens == pytest.approx((653.60,))

    def test_a_short_strangle_has_unlimited_risk(self):
        summary = analysed(short_strangle(), -8.10)
        assert summary.unlimited_risk is True and summary.max_loss is None
        assert summary.max_profit == pytest.approx(810.0)

    def test_ratio_legs_one_by_two(self):
        # Long 1 x 640C, short 2 x 650C for a 0.80 debit: one short call is uncovered.
        summary = analysed(ratio_spread(), 0.80)
        assert [leg.quantity for leg in summary.legs] == [1, 2]
        assert summary.net_premium == pytest.approx(80.0)
        assert summary.unlimited_risk is True and summary.max_loss is None
        assert summary.max_profit == pytest.approx(920.0)  # at 650: 10.00 - 0.80
        assert summary.breakevens == pytest.approx((640.80, 659.20))

    def test_ratio_legs_backspread(self):
        # Short 1 x 640C, long 2 x 650C for a 0.80 credit: the worst is 650 at expiry.
        summary = analysed(backspread(), -0.80)
        assert summary.net_premium == pytest.approx(-80.0)
        assert summary.unlimited_profit is True and summary.max_profit is None
        assert summary.unlimited_risk is False
        assert summary.max_loss == pytest.approx(920.0)  # 10.00 of width - 0.80
        assert summary.breakevens == pytest.approx((640.80, 659.20))

    def test_the_premium_is_spread_over_the_carrying_legs_contracts(self):
        # The carrying leg holds 2 contracts per unit: 0.50 per unit is 0.25 a contract.
        summary = analysed(backspread(side="buy", limit_price=0.50), 0.50)
        short_640, long_650 = summary.legs
        assert (short_640.premium, long_650.premium) == (0.0, 0.25)
        assert summary.net_premium == pytest.approx(50.0)

    def test_the_whole_premium_sits_on_the_first_bought_leg_of_a_debit(self):
        summary = analysed(butterfly(), 2.00)
        assert [leg.premium for leg in summary.legs] == [2.00, 0.0, 0.0]
        assert [leg.side for leg in summary.legs] == [Side.LONG, Side.SHORT, Side.LONG]

    def test_the_whole_premium_sits_on_the_first_sold_leg_of_a_credit(self):
        summary = analysed(iron_condor(), -4.40)
        assert [leg.premium for leg in summary.legs] == [4.40, 0.0, 0.0, 0.0]
        assert [leg.side for leg in summary.legs] == [
            Side.SHORT, Side.LONG, Side.SHORT, Side.LONG,
        ]

    def test_the_legs_are_the_proposals_contracts(self):
        proposal = put_credit_spread()
        summary = analysed(proposal, -2.00)
        for leg, source in zip(summary.legs, proposal.legs, strict=True):
            assert leg.symbol == source.symbol
            assert leg.option_type is source.option_type
            assert leg.strike == source.strike
            assert leg.expiry == source.expiration
            assert type(leg.quantity) is int and leg.quantity == source.quantity
        assert summary.contract_multiplier == MULTIPLIER

    @pytest.mark.parametrize(
        "make",
        [
            long_call,
            long_put,
            call_debit_spread,
            put_credit_spread,
            naked_short_call,
            short_strangle,
            iron_condor,
            ratio_spread,
            backspread,
            butterfly,
        ],
    )
    @pytest.mark.parametrize("quantity", [1.0, 3.0])
    def test_reproduces_the_split_premium_analysis(self, make, quantity):
        """Placing the net premium on one leg gives the same expiry picture
        as pricing every leg at its own premium."""
        proposal = make(quantity=quantity)
        expected = split_premium_analysis(proposal)
        summary = analysed(proposal, net_mid(proposal))
        assert summary is not None
        assert summary.net_premium == pytest.approx(expected.net_premium)
        assert summary.premium_type is expected.premium_type
        assert summary.max_loss == pytest.approx(expected.max_loss)
        assert summary.max_profit == pytest.approx(expected.max_profit)
        assert summary.breakevens == pytest.approx(expected.breakevens)
        assert summary.unlimited_risk is expected.unlimited_risk
        assert summary.unlimited_profit is expected.unlimited_profit

    def test_dollar_figures_scale_with_the_proposal_quantity(self):
        one = analysed(call_debit_spread(), 4.40)
        three = analysed(call_debit_spread(quantity=3.0), 4.40)
        assert [leg.quantity for leg in three.legs] == [3, 3]
        assert three.legs[0].premium == pytest.approx(4.40)  # still per share
        assert three.net_premium == pytest.approx(3 * one.net_premium)
        assert three.max_loss == pytest.approx(3 * one.max_loss)
        assert three.max_profit == pytest.approx(3 * one.max_profit)
        assert three.breakevens == pytest.approx(one.breakevens)

    def test_the_contract_multiplier_is_the_callers(self):
        summary = analysed(call_debit_spread(), 4.40, multiplier=10)
        assert summary.contract_multiplier == 10
        assert summary.max_loss == pytest.approx(44.0)
        assert summary.max_profit == pytest.approx(56.0)

    def test_the_price_is_the_proposals_not_the_leg_mids(self):
        # The same spread bought for more is a bigger loss, whatever it quotes at.
        assert analysed(call_debit_spread(), 6.00).max_loss == pytest.approx(600.0)
        assert analysed(call_debit_spread(), 6.00).max_profit == pytest.approx(400.0)

    def test_a_credit_structure_entered_as_a_debit_is_analysed_at_that_debit(self):
        # Short 630P / long 620P "bought" for 2.00: loses the width AND the debit.
        summary = analysed(put_credit_spread(side="buy"), 2.00)
        assert summary.net_premium == pytest.approx(200.0)
        assert summary.max_loss == pytest.approx(1200.0)
        assert summary.max_profit == pytest.approx(0.0)

    @pytest.mark.parametrize("make", [call_debit_spread, iron_condor, naked_short_call])
    def test_the_net_premium_is_the_entry_price_in_dollars(self, make):
        proposal = make(quantity=7.0)
        for entry in (0.01, 1.23, 4.40, 17.77):
            signed = entry if proposal.proposal.side is OrderSide.BUY else -entry
            summary = analysed(proposal, signed)
            assert summary.net_premium == pytest.approx(signed * 7.0 * MULTIPLIER, rel=1e-12)

    def test_a_zero_price_is_carried_by_a_bought_leg(self):
        summary = analysed(call_debit_spread(), 0.0)
        assert summary.net_premium == 0.0
        assert summary.max_loss == 0.0
        assert summary.max_profit == pytest.approx(1000.0)

    def test_pure_and_repeatable(self):
        assert analysed(iron_condor(), -4.40) == analysed(iron_condor(), -4.40)

    # --- None: there is no analysis to trust ---

    def test_none_for_an_equity(self):
        assert analysed(make_proposal(), 200.0) is None

    def test_none_for_an_equity_even_with_legs_attached(self):
        # The instrument decides: legs on an equity proposal are not a structure.
        dressed = make_proposal(symbol="SPY", limit_price=4.40, legs=call_debit_spread().legs)
        assert dressed.is_equity and len(dressed.legs) == 2
        assert analysed(dressed, 4.40) is None

    @pytest.mark.parametrize("unknown", [None, NAN, INF, -INF], ids=["none", "nan", "inf", "-inf"])
    def test_none_when_the_entry_price_is_unknown(self, unknown):
        assert analysed(call_debit_spread(), unknown) is None

    def test_none_without_legs(self):
        bare = make_proposal(symbol="SPY", instrument=Instrument.OPTION, limit_price=4.40)
        assert measures.structure_problems(bare)
        assert analysed(bare, 4.40) is None

    def test_none_for_legs_on_two_underlyings(self):
        legs = [
            make_leg("buy", "call", 640.0, index=0),
            make_leg("sell", "call", 650.0, index=1, underlying="QQQ"),
        ]
        mixed = make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        assert measures.structure_problems(mixed)
        assert analysed(mixed, 4.40) is None

    def test_none_when_the_proposal_is_on_another_underlying(self):
        other = call_debit_spread(symbol="QQQ")
        assert measures.structure_problems(other)
        assert analysed(other, 4.40) is None

    def test_none_for_mixed_expirations(self):
        legs = [
            make_leg("buy", "call", 640.0, index=0, expiration=LATER_EXPIRY),
            make_leg("sell", "call", 640.0, index=1),
        ]
        calendar = make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        assert measures.structure_problems(calendar)
        assert analysed(calendar, 4.40) is None

    def test_none_for_a_fractional_contract_quantity(self):
        legs = [make_leg("buy", "call", 640.0, 1.5, index=0)]
        fractional = make_proposal(
            legs=legs, symbol=legs[0].symbol, instrument=Instrument.OPTION, quantity=1.5
        )
        assert measures.structure_problems(fractional)
        assert analysed(fractional, 8.00) is None

    def test_none_when_a_debit_has_no_bought_leg_to_carry_it(self):
        assert analysed(naked_short_call(side="buy"), 3.60) is None
        assert analysed(naked_short_call(), 0.0) is None  # zero counts as a debit

    def test_none_when_a_credit_has_no_sold_leg_to_carry_it(self):
        assert analysed(long_call(side="sell"), -8.00) is None

    def test_none_when_a_leg_symbol_names_another_strike(self):
        # The order would trade the 700 call; the analysis would be of the 650.
        legs = [
            make_leg("sell", "call", 650.0, index=0),
            make_leg("buy", "call", 660.0, index=1, symbol=occ("call", 700.0)),
        ]
        spread = make_proposal(
            legs=legs, symbol="SPY", instrument=Instrument.OPTION, side="sell", limit_price=2.40
        )
        assert measures.structure_problems(spread) == (
            "leg SPY260821C00700000 names the 2026-08-21 700 call, but the leg says the "
            "2026-08-21 660 call",
        )
        assert analysed(spread, -2.40) is None

    def test_none_when_a_leg_symbol_names_the_other_option_type(self):
        legs = [make_leg("buy", "call", 640.0, symbol=occ("put", 640.0))]
        proposal = make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        assert measures.structure_problems(proposal) == (
            "leg SPY260821P00640000 names the 2026-08-21 640 put, but the leg says the "
            "2026-08-21 640 call",
        )
        assert analysed(proposal, 8.00) is None

    def test_none_when_a_leg_symbol_names_another_expiration(self):
        legs = [make_leg("buy", "call", 640.0, symbol=occ("call", 640.0, expiration=LATER_EXPIRY))]
        proposal = make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        (problem,) = measures.structure_problems(proposal)
        assert problem.startswith(f"leg {legs[0].symbol} names the {LATER_EXPIRY.isoformat()} ")
        assert problem.endswith("but the leg says the 2026-08-21 640 call")
        assert analysed(proposal, 8.00) is None

    def test_none_when_a_leg_symbol_is_not_an_option_contract(self):
        legs = [make_leg("buy", "call", 640.0, symbol="SPY")]
        proposal = make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        assert measures.structure_problems(proposal) == (
            "leg SPY is not an OCC option symbol",
        )
        assert analysed(proposal, 8.00) is None

    def test_the_shape_is_judged_by_structure_problems_alone(self, monkeypatch):
        """``measures.structure_problems`` is the single judge: its finding is
        enough to refuse an analysis, and nothing here second-guesses it."""
        assert not hasattr(context_module, "_contract_problems")
        legs = [
            make_leg("sell", "call", 650.0, index=0),
            make_leg("buy", "call", 660.0, index=1, symbol=occ("call", 700.0)),
        ]
        misnamed = make_proposal(
            legs=legs, symbol="SPY", instrument=Instrument.OPTION, side="sell", limit_price=2.40
        )
        assert analysed(misnamed, -2.40) is None
        monkeypatch.setattr(measures, "structure_problems", lambda proposal: ())
        assert analysed(misnamed, -2.40) is not None  # no second check behind it
        monkeypatch.setattr(measures, "structure_problems", lambda proposal: ("judged unsound",))
        assert analysed(call_debit_spread(), 4.40) is None

    def test_none_when_the_pricing_engine_refuses(self, monkeypatch):
        def refuse(legs, *, contract_multiplier):
            raise PricingError("expiry payoff across mixed expiries")

        monkeypatch.setattr(context_module, "analyze_position", refuse)
        assert analysed(call_debit_spread(), 4.40) is None

    def test_none_for_a_multiplier_the_pricing_engine_rejects(self):
        assert analysed(call_debit_spread(), 4.40, multiplier=0) is None

    def test_none_not_an_overflow_for_sizes_no_float_can_hold(self):
        huge = call_debit_spread(quantity=1e308)
        assert measures.structure_problems(huge) == ()
        assert analysed(huge, 4.40) is None
        assert analysed(call_debit_spread(), 1e308, multiplier=10**6) is None
        assert analysed(call_debit_spread(), 4.40, multiplier=10**400) is None

    # --- the net-premium sanity check ---

    def test_a_summary_with_another_net_premium_is_refused(self, monkeypatch):
        real = analyze_position

        def skewed(legs, *, contract_multiplier):
            summary = real(legs, contract_multiplier=contract_multiplier)
            return summary.model_copy(update={"net_premium": summary.net_premium + 1.0})

        monkeypatch.setattr(context_module, "analyze_position", skewed)
        assert analysed(call_debit_spread(), 4.40) is None  # 441.00 is not 440.00

    @pytest.mark.parametrize("premium", [NAN, INF, -440.0], ids=["nan", "inf", "wrong-sign"])
    def test_a_summary_with_an_unusable_net_premium_is_refused(self, monkeypatch, premium):
        real = analyze_position

        def broken(legs, *, contract_multiplier):
            summary = real(legs, contract_multiplier=contract_multiplier)
            return summary.model_copy(update={"net_premium": premium})

        monkeypatch.setattr(context_module, "analyze_position", broken)
        assert analysed(call_debit_spread(), 4.40) is None

    def test_float_noise_in_the_net_premium_is_not_a_mismatch(self, monkeypatch):
        real = analyze_position

        def noisy(legs, *, contract_multiplier):
            summary = real(legs, contract_multiplier=contract_multiplier)
            return summary.model_copy(update={"net_premium": summary.net_premium * (1 + 1e-13)})

        monkeypatch.setattr(context_module, "analyze_position", noisy)
        assert analysed(call_debit_spread(), 4.40) is not None


# --- build_context: the account, the clock and the instant --------------------


class TestFieldMapping:
    def test_the_account_figures(self, conn, feeds, account):
        context = build(conn)
        assert context.equity == 100_000.00
        assert context.cash == 25_000.50
        assert context.buying_power == 200_000.00
        assert context.options_buying_power == 25_000.50
        assert context.positions == tuple(account.positions)
        assert context.errors == ()

    def test_start_of_day_equity_is_the_brokers_last_equity(self, conn, feeds, account):
        context = build(conn)
        assert context.start_of_day_equity == account.last_equity == 99_251.10

    def test_daily_pnl_is_equity_less_last_equity(self, conn, feeds):
        context = build(conn)
        assert context.daily_pnl == 100_000.00 - 99_251.10
        assert context.daily_pnl == pytest.approx(748.90)

    def test_a_losing_day_is_a_negative_pnl(self, conn, feeds, account):
        feeds.account = account.model_copy(update={"equity": 97_000.0})
        context = build(conn)
        assert context.daily_pnl == pytest.approx(-2_251.10)
        # 2% of 99,251.10 is 1,985.02: the builder's numbers trip the loss rule.
        assert results(make_proposal(), context)["daily_loss_limit"].outcome is REJECT

    def test_the_positions_are_the_accounts(self, conn, feeds):
        (position,) = build(conn).positions
        assert isinstance(position, Position)
        assert (position.symbol, position.qty, position.side) == ("AAPL", 10.0, "long")
        assert position.market_value == 2150.10

    def test_holding_nothing_is_an_empty_tuple_not_unknown(self, conn, feeds, account):
        feeds.account = account.model_copy(update={"positions": []})
        assert build(conn).positions == ()

    def test_every_position_is_carried_in_the_accounts_order(self, conn, feeds, account):
        # Longs, a short and an option contract, in no particular order: a
        # position the builder dropped would be exposure the engine never saw.
        held = [
            Position(symbol="MSFT", qty=1.0, side="long", market_value=410.0),
            Position(symbol="AAPL", qty=-24.0, side="short", market_value=-4_800.0),
            Position(symbol=occ("call", 640.0), qty=2.0, side="long", market_value=1_600.0),
            Position(symbol="NVDA", qty=3.0, side="long", market_value=390.0),
            Position(symbol="QQQ", qty=-1.0, side="short", market_value=-560.0),
        ]
        feeds.account = account.model_copy(update={"positions": held})
        context = build(conn)
        assert context.positions == tuple(held)
        assert len(context.positions) == 5
        assert all(built is given for built, given in zip(context.positions, held))
        assert [p.symbol for p in context.positions] == [
            "MSFT", "AAPL", "SPY260821C00640000", "NVDA", "QQQ",
        ]
        assert [p.side for p in context.positions] == ["long", "short", "long", "long", "short"]
        assert measures.held_underlyings(context) == {"MSFT", "AAPL", "SPY", "NVDA", "QQQ"}
        assert measures.exposure(context, "AAPL") == 4_800.0  # the short counts, as a size
        assert measures.exposure(context, "SPY") == 1_600.0
        assert measures.exposure(context, "QQQ") == 560.0

    @pytest.mark.parametrize(
        "others_first",
        [
            [("MSFT", 1.0, 410.0)],
            [("MSFT", 1.0, 410.0), ("NVDA", 3.0, 390.0)],
            [],
        ],
        ids=["second", "last-of-three", "only"],
    )
    @pytest.mark.parametrize("side", ["long", "short"])
    def test_exposure_held_anywhere_in_the_account_counts_against_the_cap(
        self, conn, feeds, account, others_first, side
    ):
        # 24 AAPL worth 4,800 (long or short); 2 more at 200 make 5,200 —
        # over 5% of 100,000. Wherever the AAPL row sits in the account.
        sign = 1.0 if side == "long" else -1.0
        held = [
            Position(symbol=symbol, qty=qty, side="long", market_value=value)
            for symbol, qty, value in others_first
        ]
        held.append(
            Position(symbol="AAPL", qty=sign * 24.0, side=side, market_value=sign * 4_800.0)
        )
        feeds.account = account.model_copy(update={"positions": held})
        feeds.quotes["AAPL"] = make_quote()
        proposal = stored(conn, make_proposal())  # buy 2 AAPL at the 200.00 mid
        context = build(conn, proposal)
        assert measures.exposure(context, "AAPL") == 4_800.0
        found = results(proposal, context)["max_position_pct"]
        assert found.outcome is REJECT
        assert "$4,800.00" in found.detail and "$5,200.00" in found.detail
        evaluation = decide(proposal, context)
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == "max_position_pct"
        # Take the AAPL row away and the same buy auto-executes: that row decided it.
        feeds.account = account.model_copy(update={"positions": held[:-1]})
        assert decide(proposal, build(conn, proposal)).verdict is Verdict.AUTO_EXECUTE

    def test_the_clock_is_the_fetched_clock(self, conn, feeds, open_clock):
        context = build(conn)
        assert context.clock == open_clock
        assert context.clock.is_open is True
        assert context.clock.next_close == NEXT_CLOSE
        assert context.clock.next_open == NEXT_OPEN

    def test_the_closed_clock_fixture_rejects_a_trade_at_3am(self, conn, feeds, fixture):
        three_am = datetime(2026, 7, 31, 7, 0, tzinfo=UTC)  # 03:00 in New York
        feeds.clock = MarketClock.from_alpaca(fixture("clock_closed.json"), fetched_at=three_am)
        context = build(conn, now=three_am)
        assert context.clock.is_open is False
        assert [r.name for r in account_rules(context) if r.outcome is REJECT] == ["market_hours"]

    def test_limits_watchlist_and_multiplier_come_from_the_config(self, conn, feeds):
        config = AegisConfig(
            watchlist=["spy", "tsla"],
            risk_limits=RiskLimits(max_daily_trades=3, min_dte=7),
            pricing=PricingConfig(contract_multiplier=10),
        )
        context = build(conn, config=config)
        assert context.limits is config.risk_limits
        assert context.watchlist == ("SPY", "TSLA")
        assert context.contract_multiplier == 10

    def test_the_default_config_is_get_config(self, conn, feeds, monkeypatch):
        custom = config_with(max_open_positions=1)
        monkeypatch.setattr(context_module, "get_config", lambda: custom)
        assert build_context(conn, now=NOW).limits is custom.risk_limits

    def test_a_given_config_is_used_as_is(self, conn, feeds, monkeypatch):
        def never():
            raise AssertionError("get_config must not be called when a config is passed")

        monkeypatch.setattr(context_module, "get_config", never)
        assert build(conn).limits is CONFIG.risk_limits

    def test_the_result_is_a_frozen_policy_context(self, conn, feeds):
        context = build(conn)
        assert isinstance(context, PolicyContext)
        with pytest.raises(Exception):
            context.equity = 0.0

    def test_the_same_inputs_build_the_same_context(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        assert build(conn, proposal) == build(conn, proposal)


class TestAccountGaps:
    """A figure the broker did not report is unknown — None, with a line saying so."""

    def test_no_last_equity_means_no_start_of_day_and_no_pnl(self, conn, feeds, account):
        feeds.account = account.model_copy(update={"last_equity": None})
        context = build(conn)
        assert context.start_of_day_equity is None
        assert context.daily_pnl is None
        assert context.equity == 100_000.00
        assert context.errors == ("account state: the broker reported no usable last_equity",)
        assert results(make_proposal(), context)["daily_loss_limit"].outcome is REJECT

    def test_no_equity_means_no_pnl(self, conn, feeds, account):
        feeds.account = account.model_copy(update={"equity": None})
        context = build(conn)
        assert context.equity is None and context.daily_pnl is None
        assert context.start_of_day_equity == 99_251.10
        assert context.errors == ("account state: the broker reported no usable equity",)

    @pytest.mark.parametrize("junk", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
    def test_a_non_finite_figure_is_unknown(self, conn, feeds, account, junk):
        feeds.account = account.model_copy(update={"equity": junk, "buying_power": junk})
        context = build(conn)
        assert context.equity is None and context.buying_power is None
        assert context.daily_pnl is None
        assert context.errors == (
            "account state: the broker reported no usable equity, buying_power",
        )

    def test_no_options_buying_power(self, conn, feeds, fixture):
        data = fixture("account.json")
        del data["account"]["options_buying_power"]
        feeds.account = AccountState.from_alpaca(data["account"], data["positions"])
        context = build(conn)
        assert context.options_buying_power is None
        assert context.buying_power == 200_000.00 and context.cash == 25_000.50
        assert context.errors == (
            "account state: the broker reported no usable options_buying_power",
        )


class TestNow:
    def test_now_is_injectable(self, conn, feeds):
        moment = datetime(2026, 7, 30, 17, 45, 12, tzinfo=UTC)
        assert build(conn, now=moment).now == moment

    def test_a_naive_now_is_taken_as_utc(self, conn, feeds):
        context = build(conn, now=datetime(2026, 7, 30, 15, 0))
        assert context.now == NOW
        assert context.now.tzinfo is not None

    def test_another_offset_is_converted_to_utc(self, conn, feeds):
        new_york = timezone(timedelta(hours=-4))
        context = build(conn, now=datetime(2026, 7, 30, 11, 0, tzinfo=new_york))
        assert context.now == NOW
        assert context.now.utcoffset() == timedelta(0)
        assert context.market_date == date(2026, 7, 30)

    def test_the_default_is_the_wall_clock(self, conn, feeds, monkeypatch):
        moment = datetime(2026, 7, 30, 18, 30, tzinfo=UTC)
        monkeypatch.setattr(context_module, "utcnow", lambda: moment)
        context = build_context(conn, config=CONFIG)
        assert context.now == moment
        assert context.market_date == date(2026, 7, 30)

    def test_a_given_now_never_reads_the_clock(self, conn, feeds, monkeypatch):
        def never():
            raise AssertionError("the wall clock must not be read when now is passed")

        monkeypatch.setattr(context_module, "utcnow", never)
        assert build(conn, stored(conn, spy_buy())).now == NOW

    @pytest.mark.parametrize("bad", ["2026-07-30T15:00:00+00:00", date(2026, 7, 30), 1785423600])
    def test_a_now_that_is_not_a_datetime_is_the_callers_mistake(self, conn, feeds, bad):
        with pytest.raises(PolicyError, match="now must be a datetime"):
            build(conn, now=bad)
        assert feeds.calls == []


class TestMarketDate:
    """``market_date`` is the calendar date at ``now`` in the exchange's
    timezone — not the UTC date."""

    @pytest.mark.parametrize(
        ("now", "expected"),
        [
            # Summer: New York is UTC-4, so its midnight is 04:00 UTC.
            (datetime(2026, 7, 30, 15, 0, tzinfo=UTC), date(2026, 7, 30)),
            (datetime(2026, 7, 31, 0, 0, tzinfo=UTC), date(2026, 7, 30)),  # 20:00 on the 30th
            (datetime(2026, 7, 31, 3, 59, 59, tzinfo=UTC), date(2026, 7, 30)),
            (datetime(2026, 7, 31, 4, 0, 0, tzinfo=UTC), date(2026, 7, 31)),
            (datetime(2026, 7, 30, 3, 59, 59, tzinfo=UTC), date(2026, 7, 29)),
            (datetime(2026, 7, 30, 4, 0, 0, tzinfo=UTC), date(2026, 7, 30)),
            # Winter: New York is UTC-5, so its midnight is 05:00 UTC.
            (datetime(2026, 1, 15, 4, 59, 59, tzinfo=UTC), date(2026, 1, 14)),
            (datetime(2026, 1, 15, 5, 0, 0, tzinfo=UTC), date(2026, 1, 15)),
        ],
    )
    def test_across_the_utc_and_new_york_day_boundary(self, conn, feeds, now, expected):
        assert build(conn, now=now).market_date == expected

    def test_the_timezone_is_the_configs(self, conn, feeds):
        tokyo = config_with(pricing=PricingConfig(expiry_timezone="Asia/Tokyo"))
        # 15:00 UTC is midnight in Tokyo: already the 31st there.
        assert build(conn, config=tokyo).market_date == date(2026, 7, 31)
        assert build(conn).market_date == date(2026, 7, 30)

    def test_dte_is_counted_from_the_market_date(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        assert CONFIG.risk_limits.min_dte == 7
        # 03:00 UTC on the 15th is still the evening of the 14th in New York:
        # 7 DTE, at the floor. By 15:00 UTC the same day it is 6 DTE.
        evening_before = build(conn, proposal, now=datetime(2026, 8, 15, 3, 0, tzinfo=UTC))
        assert evening_before.market_date == date(2026, 8, 14)
        assert results(proposal, evening_before)["options_min_dte"].outcome is PASS
        next_day = build(conn, proposal, now=datetime(2026, 8, 15, 15, 0, tzinfo=UTC))
        assert next_day.market_date == date(2026, 8, 15)
        assert results(proposal, next_day)["options_min_dte"].outcome is REJECT  # 6 DTE
        expiration_day = build(conn, proposal, now=datetime(2026, 8, 21, 15, 0, tzinfo=UTC))
        assert expiration_day.market_date == EXPIRY
        assert results(proposal, expiration_day)["options_min_dte"].outcome is REJECT  # 0DTE


class TestNextOpenDate:
    """``next_open_date``: the calendar date of the clock's next open in the
    exchange's timezone — the same zone ``market_date`` is counted in. Equal
    to ``market_date`` it says the market is closed BEFORE today's session,
    which is how a daily-loss trip then halts through that session."""

    THREE_AM = datetime(2026, 7, 31, 7, 0, tzinfo=UTC)  # 03:00 in New York on the 31st
    OPEN_31 = datetime(2026, 7, 31, 13, 30, tzinfo=UTC)  # 09:30 in New York
    CLOSE_31 = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)  # 16:00 in New York

    @pytest.fixture
    def closed_clock(self, fixture):
        """tests/fixtures/clock_closed.json: 03:00 on the 31st, the next open
        09:30 and the next close 16:00 that same day."""
        return MarketClock.from_alpaca(fixture("clock_closed.json"), fetched_at=self.THREE_AM)

    def test_during_the_session_the_next_open_is_tomorrow(self, conn, feeds):
        context = build(conn)
        assert context.clock.is_open and context.clock.next_open == NEXT_OPEN
        assert context.next_open_date == date(2026, 7, 31)
        assert context.market_date == date(2026, 7, 30) != context.next_open_date

    def test_before_the_open_it_is_the_market_date_itself(self, conn, feeds, closed_clock):
        feeds.clock = closed_clock
        context = build(conn, now=self.THREE_AM)
        assert context.clock.is_open is False and context.clock.next_open == self.OPEN_31
        assert context.next_open_date == context.market_date == date(2026, 7, 31)

    def test_after_the_close_it_is_a_later_date(self, conn, feeds, open_clock):
        evening = datetime(2026, 7, 30, 21, 0, tzinfo=UTC)  # 17:00 in New York
        feeds.clock = open_clock.model_copy(update={"is_open": False})
        context = build(conn, now=evening)
        assert context.market_date == date(2026, 7, 30)
        assert context.next_open_date == date(2026, 7, 31)

    @pytest.mark.parametrize(
        ("next_open", "expected"),
        [
            # The date is the EXCHANGE's, not UTC's: 02:00 UTC on the 31st is
            # still 22:00 on the 30th in New York ...
            (datetime(2026, 7, 31, 2, 0, tzinfo=UTC), date(2026, 7, 30)),
            (datetime(2026, 7, 31, 3, 59, 59, tzinfo=UTC), date(2026, 7, 30)),
            (datetime(2026, 7, 31, 4, 0, tzinfo=UTC), date(2026, 7, 31)),
            # ... whatever offset the clock reports it in,
            (datetime(2026, 7, 31, 9, 30, tzinfo=timezone(timedelta(hours=-4))), date(2026, 7, 31)),
            (datetime(2026, 8, 1, 1, 0, tzinfo=timezone(timedelta(hours=9))), date(2026, 7, 31)),
            # and a naive instant is taken as UTC.
            (datetime(2026, 7, 31, 2, 0), date(2026, 7, 30)),
            # Winter: New York is UTC-5.
            (datetime(2026, 1, 16, 4, 59, 59, tzinfo=UTC), date(2026, 1, 15)),
            (datetime(2026, 1, 16, 5, 0, tzinfo=UTC), date(2026, 1, 16)),
        ],
    )
    def test_it_is_the_exchange_local_date_of_the_next_open(
        self, conn, feeds, next_open, expected
    ):
        feeds.clock = MarketClock.model_construct(
            is_open=False, next_open=next_open, next_close=None, fetched_at=NOW
        )
        assert build(conn).next_open_date == expected

    def test_the_timezone_is_the_configs_as_for_the_market_date(self, conn, feeds):
        tokyo = config_with(pricing=PricingConfig(expiry_timezone="Asia/Tokyo"))
        # The fixture's next open, 13:30 UTC on the 31st, is 22:30 that day in Tokyo;
        # an open at 15:00 UTC would already be the 1st there.
        assert build(conn, config=tokyo).next_open_date == date(2026, 7, 31)
        feeds.clock = feeds.clock.model_copy(
            update={"next_open": datetime(2026, 7, 31, 15, 0, tzinfo=UTC)}
        )
        assert build(conn, config=tokyo).next_open_date == date(2026, 8, 1)
        assert build(conn).next_open_date == date(2026, 7, 31)

    def test_none_without_a_clock_or_without_a_next_open(self, conn, feeds, open_clock):
        feeds.clock = DataError("market clock", None, ConnectionError("down"))
        context = build(conn)
        assert context.clock is None and context.next_open_date is None
        feeds.clock = open_clock.model_copy(update={"next_open": None})
        context = build(conn)
        assert context.clock is not None and context.next_open_date is None
        assert context.errors == ()  # a clock without a next open is not a failed fetch

    @pytest.mark.parametrize(
        ("edge", "expected"),
        [
            # New York is behind UTC: the first instant of the calendar has no date there,
            (datetime.min.replace(tzinfo=UTC), None),
            # the last one does;
            (datetime.max.replace(tzinfo=UTC), date(9999, 12, 31)),
            # and an instant with no UTC equivalent at all has none anywhere.
            (datetime.min.replace(tzinfo=timezone(timedelta(hours=12))), None),
            (datetime.max.replace(tzinfo=timezone(timedelta(hours=-12))), None),
        ],
        ids=["first-instant", "last-instant", "before-the-calendar", "past-the-calendar"],
    )
    def test_an_instant_at_the_edge_of_the_calendar_never_breaks_the_build(
        self, conn, feeds, edge, expected
    ):
        feeds.clock = MarketClock.model_construct(
            is_open=False, next_open=edge, next_close=None, fetched_at=NOW
        )
        context = build(conn)  # never raises
        assert context.clock.next_open == edge
        assert context.next_open_date == expected

    def test_it_agrees_with_the_factories_exchange_date(self, conn, feeds):
        # What hand-built contexts derive is what the builder fills in.
        context = build(conn)
        assert context.next_open_date == exchange_date(context.clock.next_open)
        assert make_context(clock=context.clock).next_open_date == context.next_open_date

    def test_a_trip_before_the_open_halts_to_that_sessions_close(
        self, conn, feeds, account, closed_clock
    ):
        """The builder's own numbers, end to end: at 03:00 the account is
        down more than 2%; the halt the rule names is the 16:00 close."""
        feeds.clock = closed_clock
        feeds.account = account.model_copy(update={"equity": 97_000.0})
        context = build(conn, now=self.THREE_AM)
        tripped = results(make_proposal(), context)["daily_loss_limit"]
        assert tripped.outcome is REJECT
        assert tripped.halt_until == self.CLOSE_31 != self.OPEN_31
        assert tripped.detail.endswith(
            "trading is halted until 2026-07-31T20:00:00+00:00, the close of the session "
            "that opens at 2026-07-31T13:30:00+00:00"
        )
        # Without the field the builder now fills, the same trip would halt
        # only to the opening bell.
        blind = context.model_copy(update={"next_open_date": None})
        assert results(make_proposal(), blind)["daily_loss_limit"].halt_until == self.OPEN_31

    def test_the_halt_survives_a_recovery_by_the_opening_bell(
        self, conn, feeds, account, closed_clock, open_clock
    ):
        """The live path, with only the fetchers faked: ``build_context`` →
        ``evaluate`` before the open with the account down, then again at
        09:31 with the P&L recovered — still REJECT, by ``halted``."""
        feeds.quotes["AAPL"] = make_quote()
        first = stored(conn, make_proposal(created_at=self.THREE_AM - timedelta(minutes=1)))
        feeds.clock = closed_clock
        feeds.account = account.model_copy(update={"equity": 97_000.0, "positions": []})
        decision = evaluate(first, build(conn, first, now=self.THREE_AM), conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "market_hours"
        assert get_controls(conn).halt_until == self.CLOSE_31
        (trip,) = [e for e in get_recent_events(conn) if e.kind == "risk_limit_tripped"]
        assert trip.payload["halt_until"] == "2026-07-31T20:00:00+00:00"

        at_0931 = self.OPEN_31 + timedelta(minutes=1)
        second = stored(
            conn, make_proposal(id="prop-0002", created_at=at_0931 - timedelta(seconds=30))
        )
        feeds.clock = open_clock.model_copy(
            update={
                "next_open": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),  # Monday
                "next_close": self.CLOSE_31,
            }
        )
        feeds.account = account.model_copy(update={"positions": []})  # up 748.90 on the day
        feeds.quotes["AAPL"] = make_quote(at=at_0931)  # a quote as fresh as the morning
        recovered = build(conn, second, now=at_0931)
        assert recovered.daily_pnl > 0 and recovered.clock.is_open
        assert recovered.controls.halt_until == self.CLOSE_31
        decision = evaluate(second, recovered, conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        outcome = {entry["rule"]: entry["outcome"] for entry in decision.rules_evaluated}
        assert [name for name, value in outcome.items() if value != "PASS"] == ["halted"]
        # But for the halt, this is the order that auto-executes at 09:31.
        unhalted = recovered.model_copy(update={"controls": recovered.controls.model_copy(
            update={"halt_until": None}
        )})
        assert decide(second, unhalted).verdict is Verdict.AUTO_EXECUTE
        assert len([e for e in get_recent_events(conn) if e.kind == "risk_limit_tripped"]) == 1


# --- build_context: the store -------------------------------------------------

DAY_START = datetime(2026, 7, 30, 4, 0, tzinfo=UTC)  # 00:00 in New York on the 30th
DAY_END = datetime(2026, 7, 31, 4, 0, tzinfo=UTC)  # 00:00 in New York on the 31st


class TestOrdersToday:
    @pytest.fixture
    def orders(self, conn):
        """Orders sent either side of both ends of the New York day of the 30th."""
        stored(conn, make_proposal(created_at=NOW - timedelta(days=30)))
        add_order(conn, 1, DAY_START - timedelta(seconds=1))  # 23:59:59 on the 29th
        add_order(conn, 2, DAY_START)  # the first instant of the 30th
        add_order(conn, 3, NOW)
        add_order(conn, 4, datetime(2026, 7, 31, 1, 0, tzinfo=UTC))  # 21:00 on the 30th
        add_order(conn, 5, DAY_END - timedelta(seconds=1))  # 23:59:59 on the 30th
        add_order(conn, 6, DAY_END)  # the first instant of the 31st
        add_order(conn, 7, NOW, status=OrderStatus.FAILED)  # the broker never took it
        add_order(conn, 8, None, status=OrderStatus.PROPOSED)  # never sent

    def test_counts_the_exchange_local_day(self, conn, feeds, orders):
        # Orders 2, 3, 4 and 5. The UTC day of the 30th would hold 1, 2 and 3.
        assert build(conn).orders_today == 4

    def test_the_same_day_late_in_the_new_york_evening(self, conn, feeds, orders):
        # 03:30 UTC on the 31st is 23:30 on the 30th in New York.
        late = datetime(2026, 7, 31, 3, 30, tzinfo=UTC)
        context = build(conn, now=late)
        assert context.market_date == date(2026, 7, 30)
        assert context.orders_today == 4

    def test_the_count_starts_again_at_local_midnight(self, conn, feeds, orders):
        assert build(conn, now=DAY_END).orders_today == 1  # order 6 alone
        assert build(conn, now=DAY_START - timedelta(seconds=1)).orders_today == 1  # order 1

    def test_no_orders_is_zero(self, conn, feeds):
        assert build(conn).orders_today == 0

    @pytest.mark.parametrize(
        ("now", "start", "end"),
        [
            (NOW, DAY_START, DAY_END),
            # Clocks go back on 2026-11-01: a 25-hour day, 04:00 UTC to 05:00 UTC.
            (
                datetime(2026, 11, 1, 12, 0, tzinfo=UTC),
                datetime(2026, 11, 1, 4, 0, tzinfo=UTC),
                datetime(2026, 11, 2, 5, 0, tzinfo=UTC),
            ),
            # Clocks go forward on 2026-03-08: a 23-hour day, 05:00 UTC to 04:00 UTC.
            (
                datetime(2026, 3, 8, 12, 0, tzinfo=UTC),
                datetime(2026, 3, 8, 5, 0, tzinfo=UTC),
                datetime(2026, 3, 9, 4, 0, tzinfo=UTC),
            ),
        ],
        ids=["summer", "fall-back", "spring-forward"],
    )
    def test_the_bounds_are_local_midnight_to_local_midnight(
        self, conn, feeds, monkeypatch, now, start, end
    ):
        asked = []

        def spy(connection, lower, upper):
            asked.append((lower, upper))
            return 7

        monkeypatch.setattr(context_module, "count_orders_submitted_between", spy)
        assert build(conn, now=now).orders_today == 7
        assert asked == [(start, end)]
        assert all(bound.utcoffset() == timedelta(0) for bound in asked[0])

    def test_an_order_in_the_extra_hour_of_a_25_hour_day_counts(self, conn, feeds):
        stored(conn, make_proposal(created_at=NOW))
        add_order(conn, 1, datetime(2026, 11, 2, 4, 30, tzinfo=UTC))  # 23:30 EST on the 1st
        assert build(conn, now=datetime(2026, 11, 1, 12, 0, tzinfo=UTC)).orders_today == 1

    def test_the_trade_cap_is_enforced_on_the_builders_count(self, conn, feeds, orders):
        context = build(conn, config=config_with(max_daily_trades=4))
        assert results(make_proposal(), context)["max_daily_trades"].outcome is REJECT
        context = build(conn, config=config_with(max_daily_trades=5))
        assert results(make_proposal(), context)["max_daily_trades"].outcome is PASS


class TestOpenOrders:
    def test_only_orders_still_open(self, conn, feeds):
        stored(conn, make_proposal(created_at=NOW - timedelta(days=30)))
        add_order(conn, 1, NOW, status=OrderStatus.SUBMITTED, symbol="MSFT")
        add_order(conn, 2, NOW, status=OrderStatus.PARTIALLY_FILLED, symbol="NVDA")
        add_order(conn, 3, NOW, status=OrderStatus.FILLED, symbol="QQQ")
        add_order(conn, 4, NOW, status=OrderStatus.CANCELLED, symbol="QQQ")
        context = build(conn)
        assert isinstance(context.open_orders, tuple)
        assert context.open_orders == tuple(get_open_orders(conn))
        assert sorted(order.symbol for order in context.open_orders) == ["MSFT", "NVDA"]

    def test_none_is_an_empty_tuple(self, conn, feeds):
        assert build(conn).open_orders == ()

    def test_an_open_order_on_the_symbol_makes_the_proposal_a_duplicate(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        add_order(conn, 1, NOW, status=OrderStatus.SUBMITTED, symbol="SPY")
        assert results(proposal, build(conn, proposal))["duplicate"].outcome is REJECT

    def test_open_orders_on_other_symbols_are_carried_too(self, conn, feeds):
        # Every open order is in the context, whatever the proposal trades: a
        # pending order in another underlying is still a position being opened.
        proposal = stored(conn, make_proposal())  # AAPL
        add_order(conn, 1, NOW, status=OrderStatus.SUBMITTED, symbol="MSFT")
        add_order(conn, 2, NOW, status=OrderStatus.SUBMITTED, symbol="SPY")
        add_order(conn, 3, NOW, status=OrderStatus.SUBMITTED, symbol="AAPL")
        add_order(conn, 4, NOW, status=OrderStatus.PARTIALLY_FILLED, symbol=occ("call", 640.0))
        with_proposal = build(conn, proposal)
        assert sorted(order.symbol for order in with_proposal.open_orders) == [
            "AAPL", "MSFT", "SPY", "SPY260821C00640000",
        ]
        assert with_proposal.open_orders == tuple(get_open_orders(conn))
        assert with_proposal.open_orders == build(conn).open_orders  # proposal or not
        option = stored(conn, spy_vertical(id="prop-0002"))
        assert build(conn, option).open_orders == with_proposal.open_orders

    def test_pending_orders_in_other_underlyings_fill_the_position_cap(
        self, conn, feeds, account
    ):
        # Nothing held; five orders pending in five other underlyings — the
        # default max_open_positions. A sixth underlying is one too many.
        feeds.account = account.model_copy(update={"positions": []})
        feeds.quotes["AAPL"] = make_quote()
        proposal = stored(conn, make_proposal())  # buy 2 AAPL at the 200.00 mid
        for number, symbol in enumerate(["MSFT", "NVDA", "QQQ", "SPY"], start=1):
            add_order(conn, number, NOW, status=OrderStatus.SUBMITTED, symbol=symbol)
        four = build(conn, proposal)
        assert len(four.open_orders) == 4
        assert results(proposal, four)["max_open_positions"].outcome is PASS
        assert decide(proposal, four).verdict is Verdict.AUTO_EXECUTE

        add_order(conn, 5, NOW, status=OrderStatus.SUBMITTED, symbol="TSLA")
        five = build(conn, proposal)
        assert sorted(order.symbol for order in five.open_orders) == [
            "MSFT", "NVDA", "QQQ", "SPY", "TSLA",
        ]
        assert measures.held_underlyings(five) == {"MSFT", "NVDA", "QQQ", "SPY", "TSLA"}
        assert results(proposal, five)["max_open_positions"].outcome is REJECT
        evaluation = decide(proposal, five)
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == "max_open_positions"


class TestRecentProposals:
    """The other proposals inside the duplicate window (60 minutes by
    default), counted back from the proposal's own creation."""

    @pytest.fixture
    def proposal(self, conn):
        """``prop-0001`` (created at PROPOSED_AT) among four others."""
        others = {
            "ten-before": PROPOSED_AT - timedelta(minutes=10),
            "at-the-edge": PROPOSED_AT - timedelta(minutes=60),
            "just-outside": PROPOSED_AT - timedelta(minutes=60, seconds=1),
            "after": PROPOSED_AT + timedelta(seconds=30),
        }
        for other_id, created_at in others.items():
            stored(conn, make_proposal(id=other_id, created_at=created_at))
        return stored(conn, make_proposal())

    @staticmethod
    def ids(context):
        return [other.id for other in context.recent_proposals]

    def test_the_proposal_itself_is_excluded(self, conn, feeds, proposal):
        context = build(conn, proposal)
        assert PROPOSAL_ID not in self.ids(context)
        assert isinstance(context.recent_proposals, tuple)

    def test_the_window_is_inclusive_and_newest_first(self, conn, feeds, proposal):
        assert self.ids(build(conn, proposal)) == ["after", "ten-before", "at-the-edge"]

    def test_the_window_is_the_configs(self, conn, feeds, proposal):
        context = build(conn, proposal, config=config_with(duplicate_window_minutes=10))
        assert self.ids(context) == ["after", "ten-before"]
        context = build(conn, proposal, config=config_with(duplicate_window_minutes=9))
        assert self.ids(context) == ["after"]
        context = build(conn, proposal, config=config_with(duplicate_window_minutes=0))
        assert self.ids(context) == ["after"]

    def test_a_proposal_judged_late_still_sees_its_own_window(self, conn, feeds, proposal):
        # Three hours on, a window counted back from NOW would hold nothing.
        context = build(conn, proposal, now=NOW + timedelta(hours=3))
        assert self.ids(context) == ["after", "ten-before", "at-the-edge"]
        assert results(proposal, context)["duplicate"].outcome is REJECT

    def test_a_proposal_stamped_after_now_is_windowed_from_now(self, conn, feeds, proposal):
        # now is 5 minutes BEFORE the proposal's timestamp (clock skew): the
        # window reaches back from the earlier of the two.
        context = build(conn, proposal, now=PROPOSED_AT - timedelta(minutes=5))
        assert self.ids(context) == ["after", "ten-before", "at-the-edge", "just-outside"]

    def test_without_a_proposal_the_window_ends_at_now(self, conn, feeds, proposal):
        # NOW is a minute after PROPOSED_AT, so "at-the-edge" is 61 minutes old.
        assert self.ids(build(conn)) == ["after", PROPOSAL_ID, "ten-before"]

    @pytest.mark.parametrize("minutes", [10**11, 10**13, 10**30])
    def test_a_window_longer_than_the_calendar_takes_everything(
        self, conn, feeds, proposal, minutes
    ):
        context = build(conn, proposal, config=config_with(duplicate_window_minutes=minutes))
        assert self.ids(context) == ["after", "ten-before", "at-the-edge", "just-outside"]

    def test_an_earlier_twin_makes_the_proposal_a_duplicate(self, conn, feeds, proposal):
        found = results(proposal, build(conn, proposal))["duplicate"]
        assert found.outcome is REJECT
        assert "ten-before" in found.detail

    def test_a_proposal_alone_is_not_its_own_duplicate(self, conn, feeds):
        proposal = stored(conn, make_proposal())
        context = build(conn, proposal)
        assert context.recent_proposals == ()
        assert results(proposal, context)["duplicate"].outcome is PASS

    def test_recent_proposals_on_other_symbols_and_instruments_are_carried(
        self, conn, feeds, proposal
    ):
        # The context holds every other proposal in the window, not only the
        # ones on this proposal's symbol: what counts as a duplicate is for
        # the rule to say.
        ten_before = PROPOSED_AT - timedelta(minutes=10)
        stored(conn, make_proposal(id="other-symbol", symbol="MSFT", created_at=ten_before))
        stored(conn, make_proposal(id="other-side", side=OrderSide.SELL, created_at=ten_before))
        stored(conn, spy_vertical(id="an-option", created_at=ten_before))
        context = build(conn, proposal)
        assert set(self.ids(context)) == {
            "after", "ten-before", "at-the-edge", "other-symbol", "other-side", "an-option",
        }
        by_id = {other.id: other for other in context.recent_proposals}
        assert by_id["other-symbol"].symbol == "MSFT"
        assert by_id["an-option"].instrument is Instrument.OPTION
        assert len(build(conn).recent_proposals) == 6  # and without a proposal alike


class TestControls:
    def test_the_controls_are_the_stores(self, conn, feeds):
        context = build(conn)
        assert context.controls == get_controls(conn)
        assert context.controls.kill_switch is False and context.controls.halt_until is None

    def test_the_kill_switch(self, conn, feeds):
        set_kill_switch(conn, True, now=NOW)
        context = build(conn)
        assert context.controls.kill_switch is True
        assert [r.name for r in account_rules(context) if r.outcome is REJECT] == ["kill_switch"]

    def test_a_halt_in_force(self, conn, feeds):
        set_halt_until(conn, NEXT_OPEN, now=NOW)
        context = build(conn)
        assert context.controls.halt_until == NEXT_OPEN
        assert [r.name for r in account_rules(context) if r.outcome is REJECT] == ["halted"]
        # ... and once now reaches it, the same stored halt no longer blocks.
        later = build(conn, now=NEXT_OPEN)
        assert "halted" not in [r.name for r in account_rules(later) if r.outcome is REJECT]

    # The fetches take seconds. The stop flags are read AFTER them, so a
    # switch set while the quotes were on their way is in the context.

    @pytest.mark.parametrize(
        ("make", "fetches"),
        [
            (None, [("clock",), ("account",)]),
            (spy_buy, [("clock",), ("account",), ("spot", "SPY")]),
            (spy_vertical, [("clock",), ("account",), ("chain", "SPY", EXPIRY)]),
        ],
        ids=["no-proposal", "equity", "option"],
    )
    def test_the_flags_are_read_once_after_every_fetch(
        self, conn, feeds, monkeypatch, make, fetches
    ):
        real = context_module.get_controls

        def reading(connection):
            feeds.calls.append(("controls",))
            return real(connection)

        monkeypatch.setattr(context_module, "get_controls", reading)
        proposal = None if make is None else stored(conn, make())
        build(conn, proposal)
        assert feeds.calls == [*fetches, ("controls",)]

    @pytest.mark.parametrize("slow", ["get_market_clock", "get_account_state", "get_spot"])
    def test_a_kill_switch_set_during_a_fetch_is_in_the_context(
        self, conn, tmp_path, feeds, monkeypatch, slow
    ):
        fetch = getattr(feeds, slow)

        def slow_fetch(*args):
            # The operator, from another process, while this fetch is in flight.
            other = open_store(tmp_path / "aegis.db")
            try:
                set_kill_switch(other, True, now=NOW)
                set_halt_until(other, NEXT_OPEN, now=NOW)
            finally:
                other.close()
            return fetch(*args)

        monkeypatch.setattr(context_module, slow, slow_fetch)
        proposal = stored(conn, spy_buy())
        assert get_controls(conn).kill_switch is False
        context = build(conn, proposal)
        assert context.controls.kill_switch is True
        assert context.controls.halt_until == NEXT_OPEN
        assert context.controls == get_controls(conn)
        evaluation = decide(proposal, context)
        assert evaluation.verdict is Verdict.REJECT and evaluation.failing_rule == "kill_switch"

    def test_a_kill_switch_set_after_the_build_still_rejects_at_evaluate(
        self, conn, tmp_path, feeds
    ):
        """Between ``build_context`` and ``evaluate``: the context says off,
        the store says ON — and the store is what the engine records under."""
        proposal = stored(conn, spy_buy())
        context = build(conn, proposal)
        assert context.controls.kill_switch is False
        assert decide(proposal, context).verdict is Verdict.AUTO_EXECUTE
        other = open_store(tmp_path / "aegis.db")
        try:
            set_kill_switch(other, True, now=NOW)
        finally:
            other.close()
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "kill_switch"
        assert decision.rules_evaluated[0]["detail"] == (
            "the kill switch is ON: every proposal is rejected"
        )


class TestStore:
    def test_a_store_failure_propagates(self, tmp_path, feeds):
        raw = connect(tmp_path / "unmigrated.db")  # no tables at all
        try:
            with pytest.raises(StoreError, match="get open orders"):  # the first store read
                build(raw)
            assert feeds.calls == []  # ... which comes before any fetch
        finally:
            raw.close()

    def test_a_store_without_its_controls_is_refused(self, conn, feeds):
        # Everything else is there: the build still fails — there is nothing to gate with.
        stored(conn, spy_buy())
        conn.execute("DROP TABLE controls")
        with pytest.raises(StoreError, match="get controls"):
            build(conn, load_proposal(conn, PROPOSAL_ID))

    @pytest.mark.parametrize(
        "name",
        [
            "get_controls",
            "get_open_orders",
            "count_orders_submitted_between",
            "get_proposals_since",
        ],
    )
    def test_every_store_read_propagates_its_error(self, conn, feeds, monkeypatch, name):
        def fail(*args):
            raise StoreError(name, cause=RuntimeError("disk gone"))

        monkeypatch.setattr(context_module, name, fail)
        with pytest.raises(StoreError, match="disk gone"):
            build(conn, make_proposal())

    def test_building_writes_nothing(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        stored(conn, spy_buy(id="prop-0002"))
        add_order(conn, 1, NOW, status=OrderStatus.SUBMITTED)
        before = dump(conn)
        build(conn, proposal)
        build(conn, load_proposal(conn, "prop-0002"))
        build(conn)
        assert dump(conn) == before
        assert before["controls"] and before["proposals"]  # the dump is of real rows

    def test_a_failed_build_writes_nothing_either(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        feeds.clock = feeds.account = DataError("everything", None, OSError("down"))
        feeds.chains = {}
        before = dump(conn)
        build(conn, proposal)
        assert dump(conn) == before


# --- build_context: the proposal's market data --------------------------------


class TestWithoutAProposal:
    def test_the_account_level_picture_only(self, conn, feeds):
        context = build(conn)
        assert context.quote is None
        assert context.leg_quotes == ()
        assert context.analysis is None
        assert context.errors == ()
        assert context.equity == 100_000.00 and context.clock is not None

    def test_fetches_the_clock_and_the_account_and_nothing_else(self, conn, feeds):
        build(conn)
        assert feeds.calls == [("clock",), ("account",)]

    def test_the_account_rules_can_run_on_it(self, conn, feeds):
        assert [result.outcome for result in account_rules(build(conn))] == [PASS] * 5


class TestEquityProposal:
    def test_the_quote_is_the_proposal_symbols(self, conn, feeds, spy_quote):
        context = build(conn, stored(conn, spy_buy()))
        assert context.quote == spy_quote
        assert (context.quote.bid, context.quote.ask) == (636.40, 636.44)
        assert context.leg_quotes == () and context.analysis is None
        assert context.errors == ()

    def test_one_spot_fetch_and_no_chain(self, conn, feeds):
        build(conn, stored(conn, spy_buy()))
        assert feeds.calls == [("clock",), ("account",), ("spot", "SPY")]

    def test_an_equity_with_legs_attached_is_still_an_equity(self, conn, feeds, spy_quote):
        dressed = stored(conn, spy_buy(legs=call_debit_spread().legs))
        context = build(conn, dressed)
        assert feeds.calls == [("clock",), ("account",), ("spot", "SPY")]  # no chain
        assert context.quote == spy_quote
        assert context.leg_quotes == () and context.analysis is None
        assert context.errors == ()

    def test_every_rule_passes_on_the_fixtures(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        context = build(conn, proposal)
        assert non_pass(proposal, context) == {}
        assert list(results(proposal, context)) == list(RULE_NAMES)

    def test_the_engine_auto_executes_it(self, conn, feeds):
        engine = pytest.importorskip("aegis.policy.engine")
        proposal = stored(conn, spy_buy())
        evaluation = engine.decide(proposal, build(conn, proposal))
        assert evaluation.verdict is Verdict.AUTO_EXECUTE
        assert evaluation.failing_rule is None

    def test_a_limit_far_from_the_fetched_mid_is_rejected(self, conn, feeds):
        proposal = stored(conn, spy_buy(limit_price=445.49))  # 30% under 636.42
        found = results(proposal, build(conn, proposal))["limit_price_sanity"]
        assert found.outcome is REJECT

    def test_a_sale_fetches_its_quote_like_a_buy(self, conn, feeds, account):
        # The fixture account holds 10 AAPL at 215.01.
        feeds.quotes["AAPL"] = make_quote("AAPL", 215.01)
        sale = stored(
            conn, make_proposal(side=OrderSide.SELL, quantity=2.0, limit_price=215.01)
        )
        context = build(conn, sale)
        assert feeds.called("spot") == [("spot", "AAPL")]
        assert feeds.calls == [("clock",), ("account",), ("spot", "AAPL")]
        assert context.quote is feeds.quotes["AAPL"]
        assert context.errors == ()
        assert results(sale, context)["limit_price_sanity"].outcome is PASS
        # Two of the ten shares held: a closing sale, and it auto-executes.
        assert measures.is_closing_sale(sale, context) is True
        evaluation = decide(sale, context)
        assert evaluation.verdict is Verdict.AUTO_EXECUTE and evaluation.non_pass == ()
        # With nothing held the same sale is a short sale: a human decides.
        feeds.account = account.model_copy(update={"positions": []})
        short = build(conn, sale)
        assert short.quote is feeds.quotes["AAPL"]
        evaluation = decide(sale, short)
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert {r.name: r.outcome for r in evaluation.non_pass} == {
            "short_sale": ESCALATE, "auto_tier": ESCALATE,
        }

    def test_exposure_counts_the_fetched_position(self, conn, feeds, fixture):
        data = fixture("stock_quote.json")
        feeds.quotes["AAPL"] = Quote.from_alpaca(
            "AAPL", quote={**data["quote"], "bid_price": 215.00, "ask_price": 215.02}
        )
        # 14 x 215.01 = 3,010.14 on top of the 2,150.10 held: over 5% of 100,000.
        proposal = stored(conn, make_proposal(quantity=14.0, limit_price=215.01))
        context = build(conn, proposal)
        assert measures.exposure(context, "AAPL") == 2150.10
        assert results(proposal, context)["max_position_pct"].outcome is REJECT


class TestOptionProposal:
    def test_the_leg_quotes_are_the_chains_snapshots_in_leg_order(self, conn, feeds):
        proposal = stored(conn, spy_condor())
        context = build(conn, proposal)
        assert [snapshot.symbol for snapshot in context.leg_quotes] == [
            leg.symbol for leg in proposal.legs
        ]
        assert [(s.bid, s.ask) for s in context.leg_quotes] == [
            (7.00, 7.20), (3.40, 3.60), (6.90, 7.10), (4.40, 4.60),
        ]
        assert all(isinstance(s, OptionSnapshot) for s in context.leg_quotes)
        assert context.quote is None
        assert context.errors == ()

    def test_the_fixture_contract_is_served_verbatim(self, conn, feeds, option_payload):
        context = build(conn, stored(conn, spy_vertical()))
        long_leg = context.leg_quotes[0]
        assert long_leg == OptionSnapshot.from_alpaca(
            "SPY260821C00640000", "SPY", option_payload, fetched_at=NOW
        )
        assert (long_leg.bid, long_leg.ask, long_leg.delta) == (12.05, 12.35, 0.5321)

    @pytest.mark.parametrize(
        "spell",
        [str.lower, lambda symbol: f"  {symbol} ", lambda symbol: f"{symbol.lower()}\n"],
        ids=["lower-case", "padded", "lower-case-and-trailing-newline"],
    )
    def test_a_chain_whose_symbols_are_not_normalised_still_quotes_the_legs(
        self, conn, feeds, option_payload, spell
    ):
        # The snapshot model does not normalise its symbol; the builder must
        # still find each leg's contract in the chain.
        chain = chain_of(option_payload)
        feeds.chains[("SPY", EXPIRY)] = chain.model_copy(
            update={
                "contracts": [
                    snapshot.model_copy(update={"symbol": spell(snapshot.symbol)})
                    for snapshot in chain.contracts
                ]
            }
        )
        served = feeds.chains[("SPY", EXPIRY)].contracts
        assert all(snapshot.symbol != snapshot.symbol.strip().upper() for snapshot in served)
        proposal = stored(conn, spy_condor())
        context = build(conn, proposal)
        assert context.errors == ()
        assert [(s.bid, s.ask) for s in context.leg_quotes] == [
            (7.00, 7.20), (3.40, 3.60), (6.90, 7.10), (4.40, 4.60),
        ]
        assert [s.symbol.strip().upper() for s in context.leg_quotes] == [
            leg.symbol for leg in proposal.legs
        ]
        # ... and the rules read those quotes: priced at its mid, it only needs approval.
        assert measures.mid_price(proposal, context) == pytest.approx(-CONDOR_MID)
        assert non_pass(proposal, context) == {"options_escalate": ESCALATE, "auto_tier": ESCALATE}

    def test_one_chain_fetch_for_a_two_leg_spread(self, conn, feeds):
        build(conn, stored(conn, spy_vertical()))
        assert feeds.calls == [("clock",), ("account",), ("chain", "SPY", EXPIRY)]

    def test_one_chain_fetch_for_four_legs_on_one_expiration(self, conn, feeds):
        build(conn, stored(conn, spy_condor()))
        assert feeds.called("chain") == [("chain", "SPY", EXPIRY)]
        assert feeds.called("spot") == []

    def test_the_chain_is_asked_for_by_underlying_and_date(self, conn, feeds):
        build(conn, stored(conn, long_call(limit_price=12.20)))
        ((_, underlying, expiration),) = feeds.called("chain")
        assert underlying == "SPY"  # the root of the OCC symbol, not the symbol
        assert type(expiration) is date and expiration == EXPIRY

    def test_one_fetch_per_distinct_expiration(self, conn, feeds, option_payload):
        feeds.chains[("SPY", LATER_EXPIRY)] = chain_of(option_payload, expiration=LATER_EXPIRY)
        legs = [
            make_leg("buy", "call", 640.0, index=0, expiration=LATER_EXPIRY),
            make_leg("sell", "call", 640.0, index=1),
            make_leg("sell", "call", 650.0, index=2, expiration=LATER_EXPIRY),
            make_leg("buy", "call", 650.0, index=3),
        ]
        calendar = stored(
            conn, make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        )
        context = build(conn, calendar)
        assert feeds.called("chain") == [
            ("chain", "SPY", LATER_EXPIRY),
            ("chain", "SPY", EXPIRY),
        ]
        assert [s.symbol for s in context.leg_quotes] == [leg.symbol for leg in legs]

    def test_one_fetch_per_distinct_underlying(self, conn, feeds, option_payload):
        feeds.chains[("QQQ", EXPIRY)] = chain_of(option_payload, underlying="QQQ")
        legs = [
            make_leg("buy", "call", 640.0, index=0),
            make_leg("sell", "call", 650.0, index=1, underlying="QQQ"),
            make_leg("buy", "call", 660.0, index=2),
        ]
        mixed = stored(conn, make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION))
        context = build(conn, mixed)
        assert feeds.called("chain") == [("chain", "SPY", EXPIRY), ("chain", "QQQ", EXPIRY)]
        assert [s.underlying for s in context.leg_quotes] == ["SPY", "QQQ", "SPY"]

    def test_a_contract_listed_twice_is_quoted_once(self, conn, feeds):
        legs = [
            make_leg("buy", "call", 640.0, index=0),
            make_leg("buy", "call", 640.0, index=1),
        ]
        doubled = stored(
            conn, make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        )
        context = build(conn, doubled)
        assert [s.symbol for s in context.leg_quotes] == [legs[0].symbol]
        assert feeds.called("chain") == [("chain", "SPY", EXPIRY)]

    def test_an_option_without_legs_fetches_no_chain(self, conn, feeds):
        bare = stored(conn, make_proposal(symbol="SPY", instrument=Instrument.OPTION))
        context = build(conn, bare)
        assert feeds.calls == [("clock",), ("account",)]
        assert context.leg_quotes == () and context.analysis is None
        assert context.errors == ("position analysis: the option proposal has no legs",)
        assert results(bare, context)["options_max_loss"].outcome is REJECT

    # --- the analysis ---

    def test_the_analysis_is_the_structure_at_the_proposals_limit(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        context = build(conn, proposal)
        assert context.analysis == structure_analysis(
            proposal, entry_price=VERTICAL_MID, contract_multiplier=MULTIPLIER
        )
        assert context.analysis.max_loss == pytest.approx(510.0)
        assert context.analysis.max_profit == pytest.approx(490.0)
        assert context.analysis.unlimited_risk is False

    def test_the_analysis_uses_the_limit_not_the_quotes(self, conn, feeds):
        proposal = stored(conn, spy_vertical(limit_price=5.30))
        assert build(conn, proposal).analysis.max_loss == pytest.approx(530.0)

    def test_a_credit_structure_is_analysed_at_its_credit(self, conn, feeds):
        proposal = stored(conn, spy_condor())
        context = build(conn, proposal)
        assert context.analysis.net_premium == pytest.approx(-610.0)
        assert context.analysis.max_loss == pytest.approx(390.0)
        assert measures.notional(proposal, context) == pytest.approx(390.0)

    def test_a_market_order_is_analysed_at_its_worst_case_fill(self, conn, feeds):
        proposal = stored(conn, spy_vertical(order_type=OrderType.MARKET, limit_price=None))
        context = build(conn, proposal)
        # Buy the 640 call at its 12.35 ask, sell the 650 call at its 7.00 bid.
        assert measures.entry_price(proposal, context) == pytest.approx(5.35)
        assert context.analysis.max_loss == pytest.approx(535.0)

    def test_the_multiplier_is_the_configs(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        tens = config_with(pricing=PricingConfig(contract_multiplier=10))
        context = build(conn, proposal, config=tens)
        assert context.analysis.contract_multiplier == 10
        assert context.analysis.max_loss == pytest.approx(51.0)

    def test_the_analysis_is_what_the_split_premiums_give(self, conn, feeds):
        proposal = stored(conn, spy_condor())
        mids = {key: (bid + ask) / 2 for key, (bid, ask) in CHAIN_QUOTES.items()}
        expected = split_premium_analysis(proposal, mids)
        analysis = build(conn, proposal).analysis
        assert analysis.max_loss == pytest.approx(expected.max_loss)
        assert analysis.max_profit == pytest.approx(expected.max_profit)
        assert analysis.breakevens == pytest.approx(expected.breakevens)

    @pytest.mark.parametrize("make", [spy_vertical, spy_condor])
    def test_a_sound_structure_only_needs_approval(self, conn, feeds, make):
        proposal = stored(conn, make())
        context = build(conn, proposal)
        assert non_pass(proposal, context) == {"options_escalate": ESCALATE, "auto_tier": ESCALATE}

    def test_the_engine_sends_an_option_to_a_human(self, conn, feeds):
        engine = pytest.importorskip("aegis.policy.engine")
        proposal = stored(conn, spy_vertical())
        evaluation = engine.decide(proposal, build(conn, proposal))
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL

    def test_a_naked_short_call_is_built_as_unlimited_risk(self, conn, feeds):
        proposal = stored(conn, naked_short_call(limit_price=7.10))
        context = build(conn, proposal)
        assert context.analysis.unlimited_risk is True
        assert context.analysis.max_loss is None
        assert measures.notional(proposal, context) == math.inf
        found = results(proposal, context)
        assert found["options_max_loss"].outcome is REJECT
        assert "unlimited risk" in found["options_max_loss"].detail

    # --- no analysis: the reason is on record ---

    def test_a_broken_structure_has_no_analysis_and_says_why(self, conn, feeds, option_payload):
        feeds.chains[("SPY", LATER_EXPIRY)] = chain_of(option_payload, expiration=LATER_EXPIRY)
        legs = [
            make_leg("buy", "call", 640.0, index=0, expiration=LATER_EXPIRY),
            make_leg("sell", "call", 640.0, index=1),
        ]
        calendar = stored(
            conn, make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION)
        )
        context = build(conn, calendar)
        assert context.analysis is None
        assert context.errors == (
            "position analysis: legs with different expirations (2026-08-21, 2026-09-18)",
        )
        assert results(calendar, context)["options_max_loss"].outcome is REJECT

    def test_a_leg_whose_symbol_is_another_contract_has_no_analysis(self, conn, feeds):
        # Declared: short the 650 call, long the 660. The long leg's SYMBOL is
        # the 640 call: the analysis would not be of what the order trades.
        legs = [
            make_leg("sell", "call", 650.0, index=0),
            make_leg("buy", "call", 660.0, index=1, symbol=occ("call", 640.0)),
        ]
        proposal = stored(
            conn,
            make_proposal(
                legs=legs,
                symbol="SPY",
                instrument=Instrument.OPTION,
                side=OrderSide.SELL,
                limit_price=3.60,
            ),
        )
        context = build(conn, proposal)
        assert context.analysis is None
        assert context.errors == (
            "position analysis: leg SPY260821C00640000 names the 2026-08-21 640 call, "
            "but the leg says the 2026-08-21 660 call",
        )
        found = results(proposal, context)
        assert found["options_max_loss"].outcome is REJECT
        assert measures.notional(proposal, context) is None  # a credit has no notional either
        assert found["buying_power"].outcome is REJECT

    def test_the_build_says_what_the_pricing_engine_refused(self, conn, feeds, monkeypatch):
        def refuse(legs, *, contract_multiplier):
            raise PricingError("expiry payoff", "SPY", OverflowError("P&L exceeds the float range"))

        monkeypatch.setattr(context_module, "analyze_position", refuse)
        context = build(conn, stored(conn, spy_vertical()))
        assert context.analysis is None
        assert context.errors == (
            "position analysis: cannot compute expiry payoff for SPY "
            "(OverflowError: P&L exceeds the float range)",
        )

    def test_the_build_names_an_arithmetic_failure(self, conn, feeds):
        context = build(conn, stored(conn, spy_vertical(quantity=1e308)))
        assert context.analysis is None
        (line,) = context.errors
        assert line.startswith("position analysis: ")

    def test_a_mislabelled_credit_has_no_analysis(self, conn, feeds):
        # Selling a call "for a debit": nothing bought to carry it.
        proposal = stored(conn, naked_short_call(side="buy", limit_price=7.10))
        context = build(conn, proposal)
        assert context.analysis is None
        assert context.errors == (
            "position analysis: the structure has no buy leg to carry a net debit",
        )
        assert results(proposal, context)["options_max_loss"].outcome is REJECT


# --- build_context: failing fetchers ------------------------------------------


def down(what, symbol=None):
    return DataError(what, symbol, ConnectionError("down"))


def rejecting(proposal, context):
    return {name for name, outcome in outcomes(proposal, context).items() if outcome is REJECT}


class TestFailingFetchers:
    """A data-layer failure never aborts the build: the field stays None,
    ``errors`` says why, and the rules that needed it reject."""

    def test_the_market_clock(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        feeds.clock = down("market clock")
        context = build(conn, proposal)
        assert context.clock is None
        assert context.errors == (
            "market clock: failed to fetch market clock (ConnectionError: down)",
        )
        assert context.equity == 100_000.00 and context.quote is not None  # the rest is intact
        assert rejecting(proposal, context) == {"market_hours"}

    def test_the_account(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        feeds.account = down("account state")
        context = build(conn, proposal)
        for name in (
            "equity",
            "cash",
            "buying_power",
            "options_buying_power",
            "start_of_day_equity",
            "daily_pnl",
            "positions",
        ):
            assert getattr(context, name) is None, name
        assert context.errors == (
            "account state: failed to fetch account state (ConnectionError: down)",
        )
        assert context.clock is not None and context.quote is not None
        assert rejecting(proposal, context) == {
            "daily_loss_limit", "buying_power", "max_position_pct", "max_open_positions",
        }

    def test_the_spot_quote(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        feeds.quotes["SPY"] = down("spot quote", "SPY")
        context = build(conn, proposal)
        assert context.quote is None
        assert context.errors == (
            "quote for SPY: failed to fetch spot quote for SPY (ConnectionError: down)",
        )
        assert rejecting(proposal, context) == {"quote_freshness", "limit_price_sanity"}

    def test_the_option_chain(self, conn, feeds):
        proposal = stored(conn, spy_vertical())
        feeds.chains[("SPY", EXPIRY)] = down("option chain", "SPY")
        context = build(conn, proposal)
        assert context.leg_quotes == ()
        assert context.errors == (
            "option chain for SPY 2026-08-21: "
            "failed to fetch option chain for SPY (ConnectionError: down)",
        )
        assert feeds.called("chain") == [("chain", "SPY", EXPIRY)]  # asked once, not per leg
        # A limit order still has its price, so the structure is still analysed.
        assert context.analysis.max_loss == pytest.approx(510.0)
        assert rejecting(proposal, context) == {"quote_freshness", "limit_price_sanity"}

    def test_a_market_order_without_quotes_has_no_price_and_no_analysis(self, conn, feeds):
        proposal = stored(conn, spy_vertical(order_type=OrderType.MARKET, limit_price=None))
        feeds.chains[("SPY", EXPIRY)] = down("option chain", "SPY")
        context = build(conn, proposal)
        assert context.analysis is None
        assert context.errors == (
            "option chain for SPY 2026-08-21: "
            "failed to fetch option chain for SPY (ConnectionError: down)",
            "position analysis: the proposal's entry price is unknown",
        )
        assert {"options_max_loss", "buying_power", "max_position_pct"} <= rejecting(
            proposal, context
        )

    def test_one_failed_chain_leaves_the_others_quotes(self, conn, feeds, option_payload):
        feeds.chains[("QQQ", EXPIRY)] = down("option chain", "QQQ")
        legs = [
            make_leg("buy", "call", 640.0, index=0),
            make_leg("sell", "call", 650.0, index=1, underlying="QQQ"),
        ]
        mixed = stored(conn, make_proposal(legs=legs, symbol="SPY", instrument=Instrument.OPTION))
        context = build(conn, mixed)
        assert [s.symbol for s in context.leg_quotes] == [legs[0].symbol]
        assert context.errors[0] == (
            "option chain for QQQ 2026-08-21: "
            "failed to fetch option chain for QQQ (ConnectionError: down)"
        )

    def test_a_leg_missing_from_its_chain(self, conn, feeds, option_payload):
        quotes = dict(CHAIN_QUOTES)
        del quotes[(OptionType.PUT, 620.0)]
        feeds.chains[("SPY", EXPIRY)] = chain_of(option_payload, quotes=quotes)
        proposal = stored(conn, spy_condor())
        context = build(conn, proposal)
        missing = occ("put", 620.0)
        assert [s.symbol for s in context.leg_quotes] == [leg.symbol for leg in proposal.legs[:3]]
        assert context.errors == (
            f"leg quote for {missing}: not in the SPY 2026-08-21 option chain",
        )
        found = results(proposal, context)["limit_price_sanity"]
        assert found.outcome is REJECT and missing in found.detail

    def test_an_empty_chain(self, conn, feeds):
        feeds.chains[("SPY", EXPIRY)] = ChainSnapshot(underlying="SPY", expiration=EXPIRY)
        proposal = stored(conn, spy_vertical())
        context = build(conn, proposal)
        assert context.leg_quotes == ()
        assert len(context.errors) == 2
        assert all(line.startswith("leg quote for SPY260821C006") for line in context.errors)

    def test_a_config_error_is_handled_like_a_data_error(self, conn, feeds):
        # What the data layer raises when the Alpaca keys are not in the environment.
        missing_keys = ConfigError(
            "missing required environment variable ALPACA_API_KEY. "
            "Copy .env.example to .env and fill in your keys."
        )
        proposal = stored(conn, spy_buy())
        feeds.clock = feeds.account = feeds.quotes["SPY"] = missing_keys
        context = build(conn, proposal)
        assert (context.clock, context.equity, context.positions, context.quote) == (None,) * 4
        assert context.errors == (
            f"market clock: {missing_keys}",
            f"account state: {missing_keys}",
            f"quote for SPY: {missing_keys}",
        )

    def test_an_error_line_is_one_line(self, conn, feeds):
        feeds.clock = ConfigError(
            "config file x failed validation:\n  watchlist\n    Field required"
        )
        context = build(conn)
        assert context.errors == (
            "market clock: config file x failed validation: watchlist Field required",
        )

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("could not convert string to float: 'n/a'"),
            KeyError("equity"),
            TypeError("x"),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_any_other_failure_of_a_feed_is_recorded_with_its_type(self, conn, feeds, error):
        feeds.account = error
        context = build(conn, stored(conn, spy_buy()))
        assert context.equity is None and context.positions is None
        assert context.errors == (f"account state: {type(error).__name__}: {error}",)

    def test_a_failure_without_a_message_is_named_by_its_type(self, conn, feeds):
        feeds.clock = RuntimeError()
        feeds.account = ConfigError("")
        assert build(conn).errors == ("market clock: RuntimeError", "account state: ConfigError")

    def test_an_interrupt_is_not_swallowed(self, conn, feeds):
        feeds.clock = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            build(conn)

    def test_everything_failing_at_once_still_builds(self, conn, feeds):
        proposal = stored(conn, spy_buy())
        feeds.clock = down("market clock")
        feeds.account = down("account state")
        feeds.quotes = {}
        context = build(conn, proposal)
        assert isinstance(context, PolicyContext)
        assert (context.clock, context.equity, context.positions, context.quote) == (None,) * 4
        assert [line.split(":")[0] for line in context.errors] == [
            "market clock", "account state", "quote for SPY",
        ]
        # What came from the store and the config is all still there.
        assert context.controls == get_controls(conn)
        assert context.limits is CONFIG.risk_limits and context.market_date == date(2026, 7, 30)
        assert rejecting(proposal, context) == {
            "market_hours", "daily_loss_limit", "buying_power", "max_position_pct",
            "max_open_positions", "quote_freshness", "limit_price_sanity",
        }

    def test_everything_failing_for_an_option_still_builds(self, conn, feeds):
        proposal = stored(conn, spy_condor())
        feeds.clock = down("market clock")
        feeds.account = down("account state")
        feeds.chains = {}
        context = build(conn, proposal)
        assert context.leg_quotes == ()
        assert len(context.errors) == 3
        assert {
            "market_hours", "daily_loss_limit", "buying_power", "quote_freshness",
            "limit_price_sanity",
        } <= rejecting(proposal, context)

    @pytest.mark.parametrize("failing", ["clock", "account", "quote", "chain"])
    def test_then_the_engine_rejects(self, conn, feeds, failing):
        engine = pytest.importorskip("aegis.policy.engine")
        proposal = stored(conn, spy_vertical() if failing == "chain" else spy_buy())
        if failing == "clock":
            feeds.clock = down("market clock")
        elif failing == "account":
            feeds.account = down("account state")
        elif failing == "quote":
            feeds.quotes = {}
        else:
            feeds.chains = {}
        evaluation = engine.decide(proposal, build(conn, proposal))
        assert evaluation.verdict is Verdict.REJECT
        expected = {
            "clock": "market_hours",
            "account": "daily_loss_limit",
            # a missing quote has no age either: the staleness rule names it first
            "quote": "quote_freshness",
            "chain": "quote_freshness",
        }
        assert evaluation.failing_rule == expected[failing]


# --- build_context: hostile feeds, seeded --------------------------------------

FAILURES = (
    lambda: DataError("feed", "SPY", ConnectionError("down")),
    lambda: ConfigError("missing required environment variable ALPACA_API_KEY"),
    lambda: ValueError("could not convert string to float: 'n/a'"),
)
JUNK = (None, NAN, INF, -INF, 0.0, -1.0, 1e300)


def random_snapshot(rng, symbol, underlying):
    bid = rng.choice([None, 0.0, 0.05, 1.20, 7.00, NAN])
    ask = rng.choice([None, 0.0, 0.10, 1.10, 7.20, INF])  # sometimes crossed, sometimes junk
    return OptionSnapshot(symbol=symbol, underlying=underlying, bid=bid, ask=ask, fetched_at=NOW)


def random_proposal(rng, number) -> ProposalUnderReview:
    proposal_id = f"fuzz-{number:04d}"
    fields = {
        "id": proposal_id,
        "created_at": NOW + timedelta(minutes=rng.randint(-600, 5)),
        "side": rng.choice(list(OrderSide)),
        "quantity": rng.choice([1.0, 2.0, 3.0, 0.5, 250.0, 1e12]),
        "order_type": rng.choice(list(OrderType)),
        "limit_price": rng.choice([None, 0.0, -1.0, 0.05, 2.40, 200.0, 636.42, 1e9]),
        "confidence": rng.random(),
        "invalidation": rng.choice(["", "  ", "a close below 190"]),
    }
    if rng.random() < 0.4:
        # Sometimes an "equity" that is option-like: named by a contract, carrying legs.
        symbol = rng.choice(["SPY", "AAPL", "TSLA", occ("call", 640.0)])
        legs = []
        if rng.random() < 0.25:
            legs = [make_leg("buy", "call", 640.0, proposal_id=proposal_id)]
        return make_proposal(legs=legs, symbol=symbol, instrument=Instrument.EQUITY, **fields)
    # Half the option proposals are one sound structure; the rest are hostile:
    # other underlyings and expirations, fractional or absurd sizes, legs whose
    # symbol is another contract's or no contract at all.
    sound = rng.random() < 0.5
    legs = []
    for index in range(rng.randint(1, 4) if sound else rng.randint(0, 4)):
        legs.append(
            make_leg(
                rng.choice(list(OrderSide)),
                rng.choice(list(OptionType)),
                rng.choice([610.0, 620.0, 630.0, 640.0, 650.0, 660.0]),
                rng.choice([1.0, 2.0] if sound else [1.0, 1.5, 50.0, 1e308]),
                index=index,
                proposal_id=proposal_id,
                underlying="SPY" if sound else rng.choice(["SPY", "QQQ"]),
                expiration=EXPIRY if sound else rng.choice([LATER_EXPIRY, date(2026, 7, 30)]),
                symbol=None if sound else rng.choice([None, occ("put", 500.0), "SPY", "NOT-ONE"]),
            )
        )
    symbol = "SPY" if sound else rng.choice(["SPY", "QQQ", legs[0].symbol if legs else "SPY"])
    return make_proposal(legs=legs, symbol=symbol, instrument=Instrument.OPTION, **fields)


def random_feeds(rng, proposal, account, open_clock, closed_clock) -> Feeds:
    def maybe(value):
        return rng.choice(FAILURES)() if rng.random() < 0.25 else value

    figures = {
        name: rng.choice(JUNK) if rng.random() < 0.2 else getattr(account, name)
        for name in ("equity", "last_equity", "cash", "buying_power", "options_buying_power")
    }
    positions = [
        Position(
            symbol=rng.choice(["SPY", "AAPL", occ("call", 640.0)]),
            qty=rng.choice([10.0, -5.0, 0.0, NAN]),
            side=rng.choice(["long", "short", ""]),
            market_value=rng.choice([None, 2150.10, -900.0, INF]),
            current_price=rng.choice([None, 215.01, NAN]),
        )
        for _ in range(rng.randint(0, 3))
    ]
    quotes = {
        symbol: maybe(
            Quote(
                symbol=rng.choice([symbol, symbol, "MSFT"]),  # sometimes for another symbol
                bid=rng.choice([None, 199.98, 636.40, 700.0, NAN]),
                ask=rng.choice([None, 200.02, 636.44, 0.0, INF]),
                fetched_at=NOW,
            )
        )
        for symbol in ("SPY", "AAPL", "TSLA")
    }
    chains = {}
    for leg in proposal.legs:
        underlying = underlying_of(leg.symbol)
        listed = [
            random_snapshot(rng, other.symbol, underlying)
            for other in proposal.legs
            if rng.random() < 0.8  # sometimes a leg is not in its chain
        ]
        chains[(underlying, leg.expiration)] = maybe(
            ChainSnapshot(underlying=underlying, expiration=leg.expiration, contracts=listed)
        )
    return Feeds(
        clock=maybe(rng.choice([open_clock, closed_clock])),
        account=maybe(account.model_copy(update={**figures, "positions": positions})),
        quotes=quotes,
        chains=chains,
    )


class TestNeverRaises:
    """Whatever the feeds do and whatever the proposal says, the build ends in
    a context and the twenty-one rules end in twenty-one results."""

    def test_seeded_hostile_feeds_and_proposals(
        self, conn, monkeypatch, account, open_clock, fixture
    ):
        rng = random.Random(20260730)
        closed_clock = MarketClock.from_alpaca(fixture("clock_closed.json"), fetched_at=NOW)
        seen_outcomes, with_errors, with_analysis, failed_fetches = set(), 0, 0, 0
        misshapen = {True: 0, False: 0}  # broken shapes, by "declared equity?"
        for number in range(400):
            proposal = stored(conn, random_proposal(rng, number))
            fake = random_feeds(rng, proposal, account, open_clock, closed_clock)
            for name in FETCHER_NAMES:
                monkeypatch.setattr(context_module, name, getattr(fake, name))
            now = NOW + timedelta(minutes=rng.randint(-300, 900))
            if rng.random() < 0.2:
                now = now.replace(tzinfo=None)  # naive: taken as UTC

            context = build(conn, proposal, now=now)

            assert isinstance(context, PolicyContext)
            assert proposal.id not in [other.id for other in context.recent_proposals]
            assert all(line and "\n" not in line for line in context.errors)
            failed = isinstance(fake.clock, Exception) or isinstance(fake.account, Exception)
            if failed:
                failed_fetches += 1
                assert context.errors  # a failed fetch is always on record
            if isinstance(fake.clock, Exception):
                assert context.clock is None
            if context.clock is None:
                assert context.next_open_date is None
            else:
                assert context.next_open_date == exchange_date(context.clock.next_open)
            if isinstance(fake.account, Exception):
                assert context.equity is None and context.positions is None
            for figure in ("equity", "cash", "buying_power", "start_of_day_equity", "daily_pnl"):
                value = getattr(context, figure)
                assert value is None or math.isfinite(value)  # unknown is None, never NaN
            if proposal.is_equity:
                assert context.analysis is None and context.leg_quotes == ()
            elif context.analysis is None:
                assert any(line.startswith("position analysis: ") for line in context.errors)
            else:
                with_analysis += 1
                assert measures.structure_problems(proposal) == ()
            found = results(proposal, context)  # no rule raises on a built context
            assert list(found) == list(RULE_NAMES)
            # A shape that contradicts itself is rejected whatever was fetched,
            # and nothing option-like is ever left to the auto tier.
            if measures.structure_problems(proposal):
                misshapen[proposal.is_equity] += 1
                assert found["options_max_loss"].outcome is REJECT
            if not measures.is_plain_equity(proposal):
                assert found["options_escalate"].outcome is ESCALATE
                assert found["auto_tier"].outcome is ESCALATE
            seen_outcomes.update(result.outcome for result in found.values())
            with_errors += bool(context.errors)
        # The generator reached the cases it is there for.
        assert seen_outcomes == set(RuleOutcome)
        assert failed_fetches > 50 and with_errors > 100 and with_analysis > 30
        assert misshapen[True] > 30 and misshapen[False] > 50


# --- the module itself --------------------------------------------------------

SOURCE = Path(context_module.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def imported_modules(tree) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


class TestModule:
    def test_the_fetchers_are_the_data_layers_bound_at_module_level(self):
        assert LIVE_FETCHERS["get_market_clock"] is aegis.data.market.get_market_clock
        assert LIVE_FETCHERS["get_spot"] is aegis.data.market.get_spot
        assert LIVE_FETCHERS["get_option_chain"] is aegis.data.market.get_option_chain
        assert LIVE_FETCHERS["get_account_state"] is aegis.data.account.get_account_state

    def test_the_fetchers_are_imported_at_the_top_level(self):
        top_level = {
            (node.module, alias.name)
            for node in TREE.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert {
            ("aegis.data.market", "get_market_clock"),
            ("aegis.data.market", "get_option_chain"),
            ("aegis.data.market", "get_spot"),
            ("aegis.data.account", "get_account_state"),
        } <= top_level
        # ... and nowhere else: no import hides inside a function.
        nested = [
            node
            for node in ast.walk(TREE)
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in TREE.body
        ]
        assert nested == []

    def test_imports_stay_inside_the_deterministic_perimeter(self):
        forbidden = (
            "anthropic", "alpaca", "aegis.brain", "aegis.execution", "aegis.data.clients",
            "aegis.data.news", "requests", "httpx", "httpx2", "urllib", "socket", "random",
            "secrets", "subprocess",
        )
        for module in imported_modules(TREE):
            for name in forbidden:
                assert module != name and not module.startswith(name + "."), module

    def test_no_risk_number_is_written_in_the_module(self):
        literals = {
            node.value
            for node in ast.walk(TREE)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        }
        assert literals <= {0, 1}

    def test_no_print(self):
        calls = [
            node
            for node in ast.walk(TREE)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert calls == []

    def test_the_public_names(self):
        defined = {
            node.name
            for node in TREE.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        }
        assert defined == {
            "load_proposal", "latest_undecided", "structure_analysis", "build_context",
        }
        assert all(callable(getattr(context_module, name)) for name in defined)
