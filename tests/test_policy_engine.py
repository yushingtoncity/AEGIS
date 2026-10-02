"""The policy engine: every rule run, one verdict resolved, the decision recorded.

Contexts are built by hand (``policy_factories``) and stores are tmp_path
databases — no network, no wall clock. The evil cases, the tiers and the
precedence go through ``decide`` (the pure half); the halt persistence and
the audit trail go through ``evaluate`` on a real store; the fail-closed
guards are exercised with rules that raise, lie about their name or go
missing. A seeded randomized sweep then checks, for hundreds of arbitrary
proposals and contexts, that the engine always returns exactly one verdict
and that AUTO_EXECUTE never appears beside a rule that did not pass — and
holds every verdict to invariants read off the inputs alone (the kill
switch on means REJECT, whatever any rule reported).

``evaluate`` is also pinned against a store that changed after the context
was built: the stop flags are read again and the stricter reading judged
(``TestStopFlagsReadAgain``), and a halt is only ever extended, by a
compare-and-set in the store (``TestHaltOnlyExtends``).
"""

import ast
import functools
import inspect
import itertools
import math
import random
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import pytest
from policy_factories import (
    LIMIT_PRICE,
    MARKET_DATE,
    NEXT_CLOSE,
    NEXT_OPEN,
    NOW,
    PROPOSAL_ID,
    PROPOSED_AT,
    SYMBOL,
    call_debit_spread,
    context_for,
    iron_condor,
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
    naked_short_call,
    occ,
    put_credit_spread,
    random_case,
)

from aegis.config import REPO_ROOT, QuoteAgeLimits
from aegis.policy import engine
from aegis.policy.engine import (
    ENGINE_INTEGRITY,
    RISK_LIMIT_TRIPPED,
    VERDICT_EVENT_KINDS,
    decide,
    evaluate,
    resolve_verdict,
    run_rules,
)
from aegis.policy.measures import is_occ, is_plain_equity, structure_problems
from aegis.policy.models import (
    Evaluation,
    PolicyContext,
    ProposalUnderReview,
    RuleOutcome,
    RuleResult,
)
from aegis.policy.rules import RULE_NAMES, RULES
from aegis.store import repo
from aegis.store.db import TABLES, open_store
from aegis.store.errors import StoreError
from aegis.store.models import (
    Controls,
    Event,
    EventLevel,
    Instrument,
    OrderSide,
    OrderType,
    PolicyDecision,
    Verdict,
)
from aegis.store.repo import (
    get_controls,
    get_proposal_trace,
    get_recent_events,
    insert_proposal,
)

PASS, FLAG, ESCALATE, REJECT = (
    RuleOutcome.PASS,
    RuleOutcome.FLAG,
    RuleOutcome.ESCALATE,
    RuleOutcome.REJECT,
)
OUTCOMES = (PASS, FLAG, ESCALATE, REJECT)

STRUCTURES = [long_call, call_debit_spread, put_credit_spread, naked_short_call, iron_condor]
DEFINED_RISK = [long_call, call_debit_spread, put_credit_spread, iron_condor]

LARGEST_FINITE = sys.float_info.max
"""The largest number ``RiskLimits`` accepts for a limit: infinity and NaN are refused at load."""

LOSS = -2_500.0
"""A day's P&L under the default cap (2% of 100,000 = 2,000): the daily loss limit trips."""
TRIP_DETAIL = (
    "today's P&L -$2,500.00 is at or below the loss cap -$2,000.00, 2% of start-of-day "
    "equity $100,000.00 (risk_limits.daily_loss_limit_pct): trading is halted until "
    "2026-07-31T13:30:00+00:00"
)

THREE_AM = datetime(2026, 7, 30, 7, 0, tzinfo=timezone.utc)
"""03:00 in New York on the trading date: hours before the open."""
OPENS_TODAY = datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc)


# --- helpers ------------------------------------------------------------------


def judge(proposal=None, context=None) -> Evaluation:
    """``decide`` on the pair; the defaults are the "everything is fine" pair."""
    proposal = make_proposal() if proposal is None else proposal
    context = make_context() if context is None else context
    evaluation = decide(proposal, context)
    assert isinstance(evaluation, Evaluation)
    assert tuple(result.name for result in evaluation.results) == RULE_NAMES
    return evaluation


def found(evaluation: Evaluation) -> dict[str, RuleOutcome]:
    """The rules that did not PASS, by name."""
    return {result.name: result.outcome for result in evaluation.non_pass}


def result_of(evaluation: Evaluation, name: str) -> RuleResult:
    (result,) = [r for r in evaluation.results if r.name == name]
    return result


def expected_verdict(outcomes) -> Verdict:
    """The precedence, restated independently of the engine."""
    outcomes = list(outcomes)
    if REJECT in outcomes:
        return Verdict.REJECT
    if FLAG in outcomes:
        return Verdict.FLAG_ONLY
    if ESCALATE in outcomes:
        return Verdict.NEEDS_APPROVAL
    return Verdict.AUTO_EXECUTE


def assert_rejected_by(evaluation: Evaluation, rule: str) -> RuleResult:
    """The verdict is REJECT, ``rule`` is the failing rule, that rule's own
    outcome is REJECT, and no rule before it in the registry rejected."""
    assert evaluation.verdict is Verdict.REJECT
    assert evaluation.failing_rule == rule
    failing = result_of(evaluation, rule)
    assert failing.outcome is REJECT
    earlier = evaluation.results[: RULE_NAMES.index(rule)]
    assert all(result.outcome is not REJECT for result in earlier)
    return failing


def verdict_results(expected=RULE_NAMES, **outcomes: RuleOutcome) -> tuple[RuleResult, ...]:
    """Synthetic results for ``expected``: PASS unless ``outcomes`` says otherwise."""
    assert set(outcomes) <= set(expected)
    return tuple(
        RuleResult(name=name, outcome=outcomes.get(name, PASS), detail=f"synthetic {name}")
        for name in expected
    )


def fake_rule(name: str, outcome: RuleOutcome = PASS, calls: list | None = None):
    """A rule function named ``name`` returning ``outcome`` (and recording its call)."""

    def rule(proposal, context):
        if calls is not None:
            calls.append(name)
        return RuleResult(name=name, outcome=outcome, detail=f"fake {name}")

    rule.__name__ = name
    return rule


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "policy.db"


@pytest.fixture
def conn(db_path):
    conn = open_store(db_path)
    yield conn
    conn.close()


def stored(conn, proposal: ProposalUnderReview | None = None) -> ProposalUnderReview:
    """``proposal`` (default ``make_proposal()``), inserted into the store with its legs."""
    proposal = make_proposal() if proposal is None else proposal
    insert_proposal(conn, proposal.proposal, proposal.legs)
    return proposal


def events(conn, kind: str | None = None) -> list[Event]:
    """Every logged event, oldest first — optionally only one kind."""
    logged = list(reversed(get_recent_events(conn, limit=10_000)))
    return [event for event in logged if kind is None or event.kind == kind]


def decisions(conn) -> list[tuple]:
    rows = conn.execute("SELECT * FROM policy_decisions ORDER BY rowid").fetchall()
    return [tuple(row) for row in rows]


def every_row(conn) -> dict[str, list[tuple]]:
    """Every row of every table in the database, by table."""
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    ]
    assert set(TABLES) < set(names)  # the eleven tables and schema_version
    return {
        name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name} ORDER BY rowid")]
        for name in names
    }


@pytest.fixture
def calls(monkeypatch):
    """Spy on the engine's two store writes: each call is recorded as
    ``(name, record, event)`` and passed on to the real function. The halt
    write must always be the compare-and-set one (``only_extend=True``): the
    engine never asks the store for a plain upsert of a halt."""
    seen = []
    real_record, real_halt = engine.record_decision, engine.set_halt_until

    def record_decision(conn, decision, *, event=None):
        seen.append(("record_decision", decision, event))
        return real_record(conn, decision, event=event)

    def set_halt_until(conn, until, *, now=None, event=None, only_extend=False):
        assert only_extend is True
        seen.append(("set_halt_until", until, event))
        return real_halt(conn, until, now=now, event=event, only_extend=only_extend)

    monkeypatch.setattr(engine, "record_decision", record_decision)
    monkeypatch.setattr(engine, "set_halt_until", set_halt_until)
    return seen


def names_of(calls) -> list[str]:
    return [name for name, _, _ in calls]


def _replacing(name: str, replacement) -> tuple:
    """The registry with the rule called ``name`` swapped for ``replacement``."""
    assert name in RULE_NAMES
    return tuple(replacement if rule.__name__ == name else rule for rule in RULES)


class Untouchable:
    """Stands in for a connection that must not be used: any use fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the connection was touched: .{name}")


def low_confidence(**overrides) -> ProposalUnderReview:
    return make_proposal(confidence=0.2, **overrides)


def over_the_auto_tier(**overrides) -> ProposalUnderReview:
    """5 AAPL at 200.20 = 1,001.00: one dollar over the default auto_execute.max_notional."""
    return make_proposal(quantity=5.0, limit_price=200.20, **overrides)


def market_order(**overrides) -> ProposalUnderReview:
    return make_proposal(order_type=OrderType.MARKET, limit_price=None, **overrides)


def closing_sale(**overrides) -> ProposalUnderReview:
    """Sell 2 AAPL at the 200.00 mid — a closing sale wherever ``HELD`` is held."""
    return make_proposal(side=OrderSide.SELL, **overrides)


HELD = (make_position(qty=10.0),)
"""10 AAPL long: what makes ``closing_sale()`` a closing sale — the one kind
of sell that can reach AUTO_EXECUTE."""

ONE_SHOT = {
    "generator": lambda items: (item for item in items),
    "iter": iter,
    "map": lambda items: map(lambda item: item, items),
    "filter": lambda items: filter(lambda item: True, items),
}
"""Ways to hand ``resolve_verdict`` its results as an iterable that can be read once."""

THE_TWENTY_ONE = (
    "kill_switch", "halted", "market_hours", "daily_loss_limit", "no_trade_list",
    "watchlist_only", "invalidation_present", "buying_power", "max_position_pct",
    "max_open_positions", "max_daily_trades", "duplicate", "quote_freshness",
    "limit_price_sanity", "options_min_dte", "options_max_loss", "options_max_contracts",
    "options_escalate", "short_sale", "min_confidence", "auto_tier",
)
"""The user's twenty-one rules, in order, written out — not read from the registry."""

EARLY = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
"""08:00 in New York on the trading date: the market opens in ninety minutes."""
AT_0931 = datetime(2026, 7, 30, 13, 31, tzinfo=timezone.utc)
"""09:31 in New York the same day: a minute into the session."""


# --- run_rules ----------------------------------------------------------------


class TestRunRules:
    def test_runs_all_twenty_one_registered_rules_in_order(self):
        proposal, context = make_proposal(), make_context()
        results = run_rules(proposal, context)
        assert isinstance(results, tuple)
        assert len(results) == 21
        assert tuple(result.name for result in results) == RULE_NAMES
        assert results == tuple(rule(proposal, context) for rule in RULES)

    def test_the_default_registry_is_the_rules_modules(self):
        assert run_rules(make_proposal(), make_context()) == run_rules(
            make_proposal(), make_context(), RULES
        )

    @pytest.mark.parametrize("position", range(21))
    def test_it_never_short_circuits(self, position):
        # Whichever rule rejects, every rule before and after it still runs.
        calls = []
        rules = [
            fake_rule(name, REJECT if index == position else PASS, calls)
            for index, name in enumerate(RULE_NAMES)
        ]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == list(RULE_NAMES)
        assert [r.outcome for r in results].count(REJECT) == 1
        assert results[position].outcome is REJECT

    def test_every_rule_runs_even_when_every_rule_rejects(self):
        calls = []
        rules = [fake_rule(name, REJECT, calls) for name in RULE_NAMES]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == list(RULE_NAMES)
        assert {r.outcome for r in results} == {REJECT}

    def test_the_real_rules_all_run_on_a_proposal_everything_is_wrong_with(self):
        proposal = market_order(invalidation="", confidence=0.0, symbol="GME", quantity=1e6)
        context = make_context(
            controls=Controls(kill_switch=True, halt_until=NOW + timedelta(days=1)),
            clock=make_clock(False),
            daily_pnl=LOSS,
            orders_today=10,
            recent_proposals=(make_recent(symbol="GME"),),
            limits=make_limits(no_trade_list=["GME"]),
        )
        results = run_rules(proposal, context)
        assert tuple(r.name for r in results) == RULE_NAMES
        rejected = [r.name for r in results if r.outcome is REJECT]
        assert rejected == [
            "kill_switch", "halted", "market_hours", "daily_loss_limit", "no_trade_list",
            "watchlist_only", "invalidation_present", "buying_power", "max_position_pct",
            "max_daily_trades", "duplicate", "quote_freshness", "limit_price_sanity",
        ]

    def test_a_rule_that_raises_rejects_and_the_others_still_run(self):
        calls = []

        def broken(proposal, context):
            calls.append("broken")
            raise RuntimeError("boom")

        rules = [fake_rule("first", PASS, calls), broken, fake_rule("last", PASS, calls)]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == ["first", "broken", "last"]
        assert results[1] == RuleResult(
            name="broken", outcome=REJECT, detail="rule failed: RuntimeError: boom"
        )
        assert [r.outcome for r in results] == [PASS, REJECT, PASS]
        assert results[1].halt_until is None

    @pytest.mark.parametrize(
        ("error", "detail"),
        [
            (ZeroDivisionError("division by zero"), "ZeroDivisionError: division by zero"),
            (KeyError("equity"), "KeyError: 'equity'"),
            (ValueError("line one\n  line two\ttab\r\n"), "ValueError: line one line two tab"),
            (AssertionError(), "AssertionError: no message"),
            (RecursionError("maximum recursion depth exceeded"), "RecursionError: maximum"),
            (StoreError("get controls"), "StoreError: store operation failed: get controls"),
        ],
        ids=["arithmetic", "lookup", "multi-line", "no-message", "recursion", "store"],
    )
    def test_the_failure_is_one_line_naming_the_type_and_the_message(self, error, detail):
        def broken(proposal, context):
            raise error

        (result,) = run_rules(make_proposal(), make_context(), [broken])
        assert result.name == "broken" and result.outcome is REJECT
        assert result.detail.startswith(f"rule failed: {detail}")
        assert "\n" not in result.detail and "\t" not in result.detail

    def test_an_exception_that_cannot_be_printed_still_rejects(self):
        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("no words")

        def broken(proposal, context):
            raise Unprintable()

        (result,) = run_rules(make_proposal(), make_context(), [broken])
        assert result == RuleResult(
            name="broken", outcome=REJECT, detail="rule failed: Unprintable: no message"
        )

    @pytest.mark.parametrize(
        ("returned", "type_name"),
        [
            (None, "NoneType"),
            (True, "bool"),
            ("PASS", "str"),
            (PASS, "RuleOutcome"),
            ({"name": "wrong_type", "outcome": "PASS", "detail": "ok"}, "dict"),
            (("wrong_type", PASS, "ok"), "tuple"),
            (Verdict.AUTO_EXECUTE, "Verdict"),
        ],
        ids=["none", "bool", "str", "outcome", "dict", "tuple", "verdict"],
    )
    def test_a_rule_that_returns_the_wrong_type_rejects(self, returned, type_name):
        calls = []

        def wrong_type(proposal, context):
            return returned

        rules = [fake_rule("first", PASS, calls), wrong_type, fake_rule("last", PASS, calls)]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == ["first", "last"]
        assert results[1] == RuleResult(
            name="wrong_type",
            outcome=REJECT,
            detail=f"rule failed: TypeError: returned {type_name}, not a RuleResult",
        )

    def test_a_rule_that_answers_under_another_name_rejects_under_its_own(self):
        # A PASS filed under another rule's name must not count as that rule passing.
        def impostor(proposal, context):
            return RuleResult(name="kill_switch", outcome=PASS, detail="all clear")

        calls = []
        rules = [fake_rule("first", PASS, calls), impostor, fake_rule("last", PASS, calls)]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == ["first", "last"]
        assert [r.name for r in results] == ["first", "impostor", "last"]
        assert results[1] == RuleResult(
            name="impostor",
            outcome=REJECT,
            detail=(
                "rule failed: ValueError: returned a result named 'kill_switch', not 'impostor'"
            ),
        )

    def test_a_result_with_a_malformed_outcome_rejects(self):
        def malformed(proposal, context):
            return RuleResult.model_construct(name="malformed", outcome="PASS", detail="ok")

        (result,) = run_rules(make_proposal(), make_context(), [malformed])
        assert result.outcome is REJECT
        assert result.detail == (
            "rule failed: ValueError: returned the outcome 'PASS', not a RuleOutcome"
        )

    # A result built around the model's validation (``model_construct``) can
    # carry anything in any field: the engine checks all of them.

    @pytest.mark.parametrize("detail", [None, "", "   \n", 7, b"all clear"], ids=repr)
    def test_a_result_without_a_real_detail_rejects(self, detail):
        def hollow(proposal, context):
            return RuleResult.model_construct(
                name="hollow", outcome=PASS, detail=detail, halt_until=None
            )

        calls = []
        rules = [fake_rule("first", PASS, calls), hollow, fake_rule("last", PASS, calls)]
        results = run_rules(make_proposal(), make_context(), rules)
        assert calls == ["first", "last"]  # the others still run
        assert [r.outcome for r in results] == [PASS, REJECT, PASS]
        assert results[1].name == "hollow"
        assert results[1].detail.startswith("rule failed: ValueError: returned the detail ")
        assert results[1].detail.endswith(", not a description")
        assert isinstance(results[1], RuleResult) and results[1].detail.strip()

    def test_the_detail_failure_quotes_what_was_returned(self):
        def hollow(proposal, context):
            return RuleResult.model_construct(
                name="hollow", outcome=PASS, detail=None, halt_until=None
            )

        (result,) = run_rules(make_proposal(), make_context(), [hollow])
        assert result == RuleResult(
            name="hollow",
            outcome=REJECT,
            detail="rule failed: ValueError: returned the detail None, not a description",
        )

    @pytest.mark.parametrize(
        "halt", ["tomorrow", 1785504600, date(2026, 7, 31), True], ids=repr
    )
    def test_a_result_whose_halt_is_not_an_instant_rejects(self, halt):
        def garbled(proposal, context):
            return RuleResult.model_construct(
                name="garbled", outcome=REJECT, detail="tripped", halt_until=halt
            )

        (result,) = run_rules(make_proposal(), make_context(), [garbled])
        assert result == RuleResult(
            name="garbled",
            outcome=REJECT,
            detail=f"rule failed: ValueError: returned the halt {halt!r}, not a datetime",
        )
        assert result.halt_until is None  # the garbage is not carried on

    def test_a_well_formed_constructed_result_is_accepted(self):
        # The check is on the fields, not on how the result was built.
        def sound(proposal, context):
            return RuleResult.model_construct(
                name="sound", outcome=REJECT, detail="tripped", halt_until=NEXT_OPEN
            )

        (result,) = run_rules(make_proposal(), make_context(), [sound])
        assert (result.outcome, result.detail, result.halt_until) == (REJECT, "tripped", NEXT_OPEN)

    def test_a_callable_without_a_name_cannot_pass(self):
        nameless = functools.partial(RULES[0])  # a partial has no __name__
        results = run_rules(make_proposal(), make_context(), [fake_rule("first"), nameless])
        assert results[1].name == "unnamed_rule_2"
        assert results[1].outcome is REJECT
        assert "returned a result named 'kill_switch'" in results[1].detail

    def test_a_failed_registered_rule_keeps_its_place_and_name(self):
        def daily_loss_limit(proposal, context):
            raise LookupError("no account")

        rules = _replacing("daily_loss_limit", daily_loss_limit)
        results = run_rules(make_proposal(), make_context(), rules)
        assert tuple(r.name for r in results) == RULE_NAMES
        assert {r.name: r.outcome for r in results if r.outcome is not PASS} == {
            "daily_loss_limit": REJECT
        }
        assert resolve_verdict(results) == (Verdict.REJECT, "daily_loss_limit")

    def test_an_interrupt_is_not_swallowed(self):
        def interrupted(proposal, context):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_rules(make_proposal(), make_context(), [interrupted])

    def test_any_sequence_of_rules_and_none_at_all(self):
        assert run_rules(make_proposal(), make_context(), []) == ()
        assert run_rules(make_proposal(), make_context(), list(RULES)) == run_rules(
            make_proposal(), make_context(), RULES
        )


# --- resolve_verdict ----------------------------------------------------------


class TestResolveVerdict:
    def test_all_twenty_one_pass_is_auto_execute(self):
        assert resolve_verdict(verdict_results()) == (Verdict.AUTO_EXECUTE, None)

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_any_single_reject_is_reject_naming_that_rule(self, name):
        assert resolve_verdict(verdict_results(**{name: REJECT})) == (Verdict.REJECT, name)

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_any_single_flag_is_flag_only(self, name):
        assert resolve_verdict(verdict_results(**{name: FLAG})) == (Verdict.FLAG_ONLY, None)

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_any_single_escalate_is_needs_approval(self, name):
        assert resolve_verdict(verdict_results(**{name: ESCALATE})) == (
            Verdict.NEEDS_APPROVAL, None,
        )

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_auto_execute_needs_every_rule_to_pass(self, name):
        for outcome in (FLAG, ESCALATE, REJECT):
            verdict, _ = resolve_verdict(verdict_results(**{name: outcome}))
            assert verdict is not Verdict.AUTO_EXECUTE

    def test_a_flag_and_an_escalate_is_flag_only(self):
        for flagged, escalated in (("min_confidence", "auto_tier"), ("auto_tier", "kill_switch")):
            results = verdict_results(**{flagged: FLAG, escalated: ESCALATE})
            assert resolve_verdict(results) == (Verdict.FLAG_ONLY, None)

    def test_a_reject_beats_a_flag_and_an_escalate_wherever_it_is(self):
        for name in (RULE_NAMES[0], "duplicate", RULE_NAMES[-1]):
            others = [n for n in RULE_NAMES if n != name]
            results = verdict_results(**{name: REJECT, others[0]: FLAG, others[-1]: ESCALATE})
            assert resolve_verdict(results) == (Verdict.REJECT, name)

    @pytest.mark.parametrize(
        ("earlier", "later"), [(3, 12), (0, 19), (7, 8), (11, 12)], ids=str
    )
    def test_the_failing_rule_is_the_first_reject_in_registry_order(self, earlier, later):
        first, second = RULE_NAMES[earlier], RULE_NAMES[later]
        results = verdict_results(**{second: REJECT, first: REJECT})
        assert resolve_verdict(results) == (Verdict.REJECT, first)

    def test_every_combination_of_outcomes_on_a_small_registry(self):
        expected = ("a", "b", "c", "d")
        for combination in itertools.product(OUTCOMES, repeat=len(expected)):
            results = verdict_results(expected, **dict(zip(expected, combination)))
            verdict, failing_rule = resolve_verdict(results, expected)
            assert verdict is expected_verdict(combination)
            rejecting = [name for name, outcome in zip(expected, combination) if outcome is REJECT]
            assert failing_rule == (rejecting[0] if rejecting else None)
            assert (verdict is Verdict.AUTO_EXECUTE) == (set(combination) == {PASS})

    def test_results_and_expected_may_be_any_sequence(self):
        assert resolve_verdict(list(verdict_results()), list(RULE_NAMES)) == (
            Verdict.AUTO_EXECUTE, None,
        )

    # A one-shot iterable is read ONCE: the integrity guard must not use it
    # up and leave the precedence an empty second pass — where "no REJECT, no
    # FLAG, every outcome PASS" would all be vacuously true.

    @pytest.mark.parametrize("one_shot", ONE_SHOT.values(), ids=list(ONE_SHOT))
    def test_a_one_shot_iterable_of_rejecting_results_is_rejected(self, one_shot):
        rejecting = verdict_results(kill_switch=REJECT, market_hours=REJECT, auto_tier=ESCALATE)
        assert resolve_verdict(one_shot(rejecting)) == (Verdict.REJECT, "kill_switch")
        assert resolve_verdict(one_shot(rejecting), one_shot(RULE_NAMES)) == (
            Verdict.REJECT, "kill_switch",
        )
        later = verdict_results(duplicate=REJECT, min_confidence=FLAG)
        assert resolve_verdict(one_shot(later)) == (Verdict.REJECT, "duplicate")
        # ... as are the real rules' results, with the kill switch on at 3am
        stopped = make_context(controls=Controls(kill_switch=True), clock=make_clock(False))
        results = run_rules(make_proposal(), stopped)
        assert [r.name for r in results if r.outcome is REJECT] == ["kill_switch", "market_hours"]
        assert resolve_verdict(one_shot(results)) == (Verdict.REJECT, "kill_switch")

    @pytest.mark.parametrize("one_shot", ONE_SHOT.values(), ids=list(ONE_SHOT))
    def test_a_one_shot_iterable_resolves_like_the_tuple_it_yields(self, one_shot):
        for outcomes in (
            {},
            {"min_confidence": FLAG},
            {"auto_tier": ESCALATE},
            {"min_confidence": FLAG, "auto_tier": ESCALATE},
            {"auto_tier": REJECT},
        ):
            results = verdict_results(**outcomes)
            assert resolve_verdict(one_shot(results)) == resolve_verdict(results)
            assert resolve_verdict(one_shot(results), one_shot(RULE_NAMES)) == (
                resolve_verdict(results)
            )
        # The names are read once too: a one-shot ``expected`` still guards.
        assert resolve_verdict(verdict_results(), one_shot(RULE_NAMES[:-1])) == (
            Verdict.REJECT, ENGINE_INTEGRITY,
        )
        assert resolve_verdict(one_shot(verdict_results()[:-1]), one_shot(RULE_NAMES)) == (
            Verdict.REJECT, ENGINE_INTEGRITY,
        )

    @pytest.mark.parametrize("one_shot", ONE_SHOT.values(), ids=list(ONE_SHOT))
    def test_an_empty_iterable_is_an_integrity_reject(self, one_shot):
        assert resolve_verdict(one_shot(())) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict(one_shot(()), one_shot(())) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict([], []) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict(one_shot(()), one_shot(RULE_NAMES)) == (
            Verdict.REJECT, ENGINE_INTEGRITY,
        )

    # The integrity guard: all-PASS results would be AUTO_EXECUTE, unless the
    # results are not exactly the registered rules.

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_integrity_a_missing_result(self, name):
        results = tuple(r for r in verdict_results() if r.name != name)
        assert len(results) == 20
        assert resolve_verdict(results) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_an_extra_result(self):
        extra = RuleResult(name="one_more_rule", outcome=PASS, detail="extra")
        for position in (0, 7, 20):
            results = list(verdict_results())
            results.insert(position, extra)
            assert resolve_verdict(results) == (Verdict.REJECT, ENGINE_INTEGRITY)

    @pytest.mark.parametrize(("left", "right"), [(0, 1), (0, 19), (9, 10), (18, 19)], ids=str)
    def test_integrity_reordered_results(self, left, right):
        results = list(verdict_results())
        results[left], results[right] = results[right], results[left]
        assert sorted(r.name for r in results) == sorted(RULE_NAMES)
        assert resolve_verdict(results) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_a_duplicated_result(self):
        results = verdict_results()
        appended = (*results, results[-1])
        in_place = (results[0], *results)
        replacing = (*results[:5], results[4], *results[6:])  # twenty-one results, one twice
        assert len(replacing) == 21
        for broken in (appended, in_place, replacing):
            assert resolve_verdict(broken) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_a_renamed_result(self):
        results = list(verdict_results())
        results[3] = RuleResult(name="daily_loss_limit ", outcome=PASS, detail="renamed")
        assert resolve_verdict(results) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_no_results_at_all(self):
        assert resolve_verdict(()) == (Verdict.REJECT, ENGINE_INTEGRITY)
        # ... not even against an empty registry: nothing that ran no rules may execute.
        assert resolve_verdict((), ()) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_something_that_is_not_a_rule_result(self):
        results = list(verdict_results())
        for impostor in (None, "kill_switch", {"name": "kill_switch", "outcome": PASS}):
            assert resolve_verdict([impostor, *results[1:]]) == (
                Verdict.REJECT, ENGINE_INTEGRITY,
            )
        forged = RuleResult.model_construct(name="kill_switch", outcome="PASS", detail="ok")
        assert resolve_verdict([forged, *results[1:]]) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_integrity_is_checked_before_any_precedence(self):
        # A rule's own REJECT is present, but the verdict cannot be trusted to name it.
        results = verdict_results(duplicate=REJECT, min_confidence=FLAG)
        assert resolve_verdict(results) == (Verdict.REJECT, "duplicate")
        assert resolve_verdict(results[1:]) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict((*results, results[0])) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_the_expected_names_are_what_is_checked(self):
        small = verdict_results(("a", "b"))
        assert resolve_verdict(small, ("a", "b")) == (Verdict.AUTO_EXECUTE, None)
        assert resolve_verdict(small) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict(small, ("b", "a")) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict(small, ("a", "b", "c")) == (Verdict.REJECT, ENGINE_INTEGRITY)
        assert resolve_verdict(verdict_results(), ("a", "b")) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_engine_integrity_is_not_a_rule(self):
        assert ENGINE_INTEGRITY == "engine_integrity"
        assert ENGINE_INTEGRITY not in RULE_NAMES


# --- decide -------------------------------------------------------------------


class TestDecide:
    def test_the_default_pair_is_auto_execute_with_all_twenty_one_passing(self):
        evaluation = judge()
        assert evaluation.verdict is Verdict.AUTO_EXECUTE
        assert evaluation.failing_rule is None
        assert len(evaluation.results) == 21
        assert {result.outcome for result in evaluation.results} == {PASS}
        assert evaluation.non_pass == ()
        assert evaluation.proposal_id == PROPOSAL_ID
        assert evaluation.halt_until is None

    def test_it_is_run_rules_and_resolve_verdict(self):
        proposal = low_confidence(quantity=5.0, limit_price=200.20)
        context = context_for(proposal)
        evaluation = decide(proposal, context)
        assert evaluation.results == run_rules(proposal, context)
        assert (evaluation.verdict, evaluation.failing_rule) == resolve_verdict(evaluation.results)

    def test_decided_at_is_the_contexts_now_never_the_wall_clock(self):
        long_ago = datetime(2001, 9, 10, 14, 0, tzinfo=timezone.utc)
        context = make_context(now=long_ago, market_date=long_ago.date())
        assert judge(context=context).decided_at == long_ago
        assert judge().decided_at == NOW

    @pytest.mark.parametrize(
        "build",
        [make_proposal, low_confidence, market_order, *STRUCTURES],
        ids=lambda build: build.__name__,
    )
    def test_deterministic_the_same_inputs_give_an_equal_evaluation(self, build):
        for overrides in ({}, {"daily_pnl": LOSS}, {"controls": Controls(kill_switch=True)}):
            first = decide(build(), context_for(build(), **overrides))
            second = decide(build(), context_for(build(), **overrides))
            assert first == second
            assert first.model_dump_json() == second.model_dump_json()

    def test_it_does_not_change_its_inputs(self):
        proposal, context = iron_condor(), context_for(iron_condor(), daily_pnl=LOSS)
        before = (proposal.model_dump_json(), context.model_dump_json())
        decide(proposal, context)
        assert (proposal.model_dump_json(), context.model_dump_json()) == before

    def test_rules_evaluated_is_the_stores_shape_for_every_rule(self):
        evaluation = judge(low_confidence())
        assert evaluation.rules_evaluated == [
            {"rule": r.name, "outcome": r.outcome.value, "detail": r.detail}
            for r in evaluation.results
        ]
        assert [entry["rule"] for entry in evaluation.rules_evaluated] == list(RULE_NAMES)

    # The halt to persist.

    def test_no_trip_no_halt(self):
        assert judge(context=make_context(daily_pnl=-1_999.99)).halt_until is None

    def test_a_trip_names_the_halt_the_rule_named(self):
        evaluation = judge(context=make_context(daily_pnl=LOSS))
        assert evaluation.halt_until == NEXT_OPEN
        assert evaluation.halt_until == result_of(evaluation, "daily_loss_limit").halt_until
        assert evaluation.halt_until.utcoffset() == timedelta(0)

    def test_an_unknown_pnl_rejects_without_a_halt(self):
        evaluation = judge(context=make_context(daily_pnl=None))
        assert_rejected_by(evaluation, "daily_loss_limit")
        assert evaluation.halt_until is None

    def test_a_halt_already_that_long_is_not_written_again(self):
        for longer in (timedelta(0), timedelta(seconds=1), timedelta(days=3)):
            in_force = NEXT_OPEN + longer
            context = make_context(daily_pnl=LOSS, controls=Controls(halt_until=in_force))
            evaluation = judge(context=context)
            assert result_of(evaluation, "daily_loss_limit").halt_until == NEXT_OPEN
            assert evaluation.halt_until is None

    def test_a_shorter_or_expired_halt_is_extended(self):
        for in_force in (
            NEXT_OPEN - timedelta(seconds=1), NOW + timedelta(hours=1), NOW - timedelta(days=1),
        ):
            context = make_context(daily_pnl=LOSS, controls=Controls(halt_until=in_force))
            assert judge(context=context).halt_until == NEXT_OPEN

    def test_an_unreadable_halt_is_replaced_by_the_trips(self):
        controls = Controls(halt_unknown=True, problems=("halt_until is unreadable",))
        evaluation = judge(context=make_context(daily_pnl=LOSS, controls=controls))
        assert_rejected_by(evaluation, "halted")
        assert evaluation.halt_until == NEXT_OPEN

    @pytest.mark.parametrize(
        ("instant", "named"),
        [
            (NOW - timedelta(days=1), NEXT_OPEN),
            (NEXT_OPEN - timedelta(seconds=1), NEXT_OPEN),
            (NEXT_OPEN, NEXT_OPEN),
            (NEXT_OPEN + timedelta(days=3), NEXT_OPEN + timedelta(days=3)),
        ],
        ids=["expired", "shorter", "the-same", "longer"],
    )
    def test_an_unreadable_halt_beside_an_instant_is_replaced_by_the_later_of_the_two(
        self, instant, named
    ):
        # What stricter_controls can yield and no single reading of the store
        # holds: one reading unreadable, another an instant. The halt is in
        # force whatever the instant says; a trip names a finite halt that
        # cuts neither its own instant nor that reading short.
        controls = Controls(
            halt_unknown=True, halt_until=instant, problems=("halt_until is unreadable",)
        )
        evaluation = judge(context=make_context(daily_pnl=LOSS, controls=controls))
        halted = assert_rejected_by(evaluation, "halted")
        assert halted.detail == "cannot prove trading is not halted: halt_until is unreadable"
        assert result_of(evaluation, "daily_loss_limit").halt_until == NEXT_OPEN
        assert evaluation.halt_until == named
        # Without a trip there is no halt to write, readable or not.
        calm = judge(context=make_context(controls=controls))
        assert_rejected_by(calm, "halted")
        assert calm.halt_until is None

    def test_the_halt_is_named_whichever_rule_fails_first(self):
        context = make_context(daily_pnl=LOSS, controls=Controls(kill_switch=True), clock=None)
        evaluation = judge(context=context)
        assert_rejected_by(evaluation, "kill_switch")
        assert evaluation.halt_until == NOW + timedelta(hours=24)  # no clock: the fallback

    def test_a_naive_halt_from_a_rule_is_compared_as_utc(self, monkeypatch):
        naive = NEXT_OPEN.replace(tzinfo=None)

        def daily_loss_limit(proposal, context):
            return RuleResult(
                name="daily_loss_limit", outcome=REJECT, detail="tripped", halt_until=naive
            )

        monkeypatch.setattr(engine, "RULES", _replacing("daily_loss_limit", daily_loss_limit))
        assert judge().halt_until == NEXT_OPEN
        halted = make_context(controls=Controls(halt_until=NEXT_OPEN))
        assert judge(context=halted).halt_until is None

    def test_only_the_daily_loss_limit_can_name_a_halt(self, monkeypatch):
        # Any other rule's halt_until is not the engine's to persist.
        def max_daily_trades(proposal, context):
            return RuleResult(
                name="max_daily_trades", outcome=REJECT, detail="at the cap", halt_until=NEXT_OPEN
            )

        monkeypatch.setattr(engine, "RULES", _replacing("max_daily_trades", max_daily_trades))
        evaluation = judge()
        assert_rejected_by(evaluation, "max_daily_trades")
        assert result_of(evaluation, "max_daily_trades").halt_until == NEXT_OPEN
        assert evaluation.halt_until is None

    @pytest.mark.parametrize("outcome", [PASS, FLAG, ESCALATE], ids=lambda o: o.value)
    def test_a_halt_is_taken_from_a_reject_only(self, monkeypatch, outcome):
        # A daily_loss_limit result that did not trip has no halt to give,
        # whatever its halt_until field holds.
        def daily_loss_limit(proposal, context):
            return RuleResult(
                name="daily_loss_limit", outcome=outcome, detail="not tripped",
                halt_until=NEXT_OPEN,
            )

        monkeypatch.setattr(engine, "RULES", _replacing("daily_loss_limit", daily_loss_limit))
        evaluation = judge()
        carried = result_of(evaluation, "daily_loss_limit")
        assert carried.outcome is outcome and carried.halt_until == NEXT_OPEN
        assert evaluation.verdict is expected_verdict([outcome])
        assert evaluation.halt_until is None
        # ... while the very same result as a REJECT is the halt

        def tripped(proposal, context):
            return daily_loss_limit(proposal, context).model_copy(update={"outcome": REJECT})

        tripped.__name__ = "daily_loss_limit"
        monkeypatch.setattr(engine, "RULES", _replacing("daily_loss_limit", tripped))
        assert judge().halt_until == NEXT_OPEN

    @pytest.mark.parametrize("outcome", [PASS, FLAG, ESCALATE], ids=lambda o: o.value)
    def test_a_halt_on_a_result_that_did_not_reject_is_never_written(
        self, conn, calls, monkeypatch, outcome
    ):
        def daily_loss_limit(proposal, context):
            return RuleResult(
                name="daily_loss_limit", outcome=outcome, detail="not tripped",
                halt_until=NEXT_OPEN,
            )

        monkeypatch.setattr(engine, "RULES", _replacing("daily_loss_limit", daily_loss_limit))
        decision = evaluate(stored(conn), make_context(), conn)
        assert decision.verdict is expected_verdict([outcome])
        assert names_of(calls) == ["record_decision"]
        assert get_controls(conn).halt_until is None
        assert events(conn, RISK_LIMIT_TRIPPED) == []


# --- the evil cases -----------------------------------------------------------


class TestEvilCases:
    """Each asserts the verdict, the failing rule and that rule's own outcome."""

    def test_ten_times_the_size_cap(self):
        # The cap is 5% of 100,000 = 5,000; 250 AAPL at 200 is 50,000.
        proposal = make_proposal(quantity=250.0)
        evaluation = judge(proposal)
        failing = assert_rejected_by(evaluation, "max_position_pct")
        assert "$50,000.00" in failing.detail and "$5,000.00" in failing.detail
        assert found(evaluation) == {"max_position_pct": REJECT, "auto_tier": ESCALATE}

    def test_a_trade_at_three_in_the_morning(self):
        proposal = make_proposal(created_at=THREE_AM - timedelta(minutes=1))
        context = make_context(
            now=THREE_AM, clock=make_clock(False, next_open=OPENS_TODAY, next_close=NEXT_CLOSE)
        )
        evaluation = judge(proposal, context)
        failing = assert_rejected_by(evaluation, "market_hours")
        assert "the market is closed" in failing.detail
        assert found(evaluation) == {"market_hours": REJECT}
        assert evaluation.decided_at == THREE_AM

    def test_the_kill_switch_is_on(self):
        evaluation = judge(context=make_context(controls=Controls(kill_switch=True)))
        assert_rejected_by(evaluation, "kill_switch")
        assert found(evaluation) == {"kill_switch": REJECT}

    @pytest.mark.parametrize(
        "cap",
        [0.0, 1_000.0, 1e12, LARGEST_FINITE],
        ids=["zero", "small", "vast", "largest-finite"],
    )
    def test_a_naked_short_call_whatever_the_loss_cap_says(self, cap):
        proposal = naked_short_call()
        context = context_for(proposal, limits=make_limits(max_loss_per_trade=cap))
        assert context.limits.max_loss_per_trade == cap
        evaluation = judge(proposal, context)
        assert evaluation.verdict is Verdict.REJECT
        # Fact one: options_max_loss rejects it unconditionally, whatever the cap.
        max_loss = result_of(evaluation, "options_max_loss")
        assert max_loss.outcome is REJECT
        assert "unlimited" in max_loss.detail
        assert "unlimited risk" in max_loss.detail and "unconditionally" in max_loss.detail
        # Fact two: unlimited risk is also an unlimited notional, which no
        # buying power covers — and buying_power (rule 8) comes before
        # options_max_loss (rule 15) in the registry, so it is the rule the
        # decision names.
        assert evaluation.failing_rule == "buying_power"
        assert RULE_NAMES.index("buying_power") < RULE_NAMES.index("options_max_loss")
        failing = assert_rejected_by(evaluation, "buying_power")
        assert "notional unlimited" in failing.detail
        assert found(evaluation) == {
            "buying_power": REJECT, "max_position_pct": REJECT, "options_max_loss": REJECT,
            "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }

    def test_a_naked_short_call_is_rejected_by_options_max_loss_on_its_own(self):
        # Take the two sizing rejections away and the verdict does not move.
        proposal = naked_short_call()
        context = context_for(proposal, limits=make_limits(max_loss_per_trade=LARGEST_FINITE))
        results = tuple(
            result.model_copy(update={"outcome": PASS})
            if result.name in ("buying_power", "max_position_pct")
            else result
            for result in judge(proposal, context).results
        )
        assert resolve_verdict(results) == (Verdict.REJECT, "options_max_loss")

    def test_no_account_is_large_enough_for_a_naked_short_call(self):
        proposal = naked_short_call()
        rich = {"equity": 1e15, "cash": 1e15, "buying_power": 1e15, "options_buying_power": 1e15}
        evaluation = judge(proposal, context_for(proposal, **rich))
        assert_rejected_by(evaluation, "buying_power")
        assert result_of(evaluation, "options_max_loss").outcome is REJECT

    def test_an_option_contract_declared_equity_is_rejected_never_auto_executed(self):
        # Otherwise a perfect auto-tier buy: declared equity, a limit at the
        # mid of a two-sided quote for that very symbol, a watchlist
        # underlying, a small notional — counted WITHOUT the contract multiplier.
        contract = "SPY260821C00640000"
        proposal = make_proposal(symbol=contract, limit_price=8.00)
        context = make_context(quote=make_quote(contract, 8.00))
        assert proposal.is_equity and proposal.legs == ()
        assert context.quote.symbol == contract and context.quote.bid < 8.00 < context.quote.ask
        evaluation = judge(proposal, context)
        failing = assert_rejected_by(evaluation, "options_max_loss")
        assert failing.detail == (
            "the proposal is not a plain equity: the proposal is declared equity but its "
            "symbol SPY260821C00640000 is an option contract"
        )
        assert evaluation.verdict is not Verdict.AUTO_EXECUTE
        # Three independent rules stand in the way; everything else passes.
        assert found(evaluation) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }
        assert "$16.00" in result_of(evaluation, "buying_power").detail  # 2 x 8.00, no x100
        # The same order for plain shares would have auto-executed: the shape alone stops it.
        shares = make_proposal(symbol="SPY", limit_price=8.00)
        plain = judge(shares, make_context(quote=make_quote("SPY", 8.00)))
        assert plain.verdict is Verdict.AUTO_EXECUTE

    def test_a_mislabelled_option_is_stopped_by_each_of_three_rules_on_its_own(self):
        contract = "SPY260821C00640000"
        proposal = make_proposal(symbol=contract, limit_price=8.00)
        results = judge(proposal, make_context(quote=make_quote(contract, 8.00))).results
        guards = ("options_max_loss", "options_escalate", "auto_tier")
        for standing in guards:
            # Take the other two away: the one left still keeps it from auto-executing.
            weakened = tuple(
                result.model_copy(update={"outcome": PASS})
                if result.name in guards and result.name != standing
                else result
                for result in results
            )
            verdict, _ = resolve_verdict(weakened)
            assert verdict is not Verdict.AUTO_EXECUTE
            assert verdict is (
                Verdict.REJECT if standing == "options_max_loss" else Verdict.NEEDS_APPROVAL
            )

    def test_an_equity_that_carries_option_legs_is_rejected(self):
        proposal = make_proposal(symbol="SPY", legs=[make_leg("buy", "call", 640.0)])
        context = make_context(quote=make_quote("SPY", LIMIT_PRICE))
        evaluation = judge(proposal, context)
        failing = assert_rejected_by(evaluation, "options_max_loss")
        assert "an equity proposal carries 1 option leg(s)" in failing.detail
        assert found(evaluation) == {
            "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }

    def test_a_leg_whose_symbol_is_another_contract_is_rejected(self):
        # Declared: short the 650 call, long the 660 — a defined-risk credit
        # spread. The long leg's SYMBOL is the 700 call: the order would trade
        # a wider spread than the one the analysis in hand describes.
        declared = make_option(
            [("sell", "call", 650.0), ("buy", "call", 660.0)], side="sell", limit_price=2.40
        )
        misnamed = ProposalUnderReview(
            proposal=declared.proposal,
            legs=(
                declared.legs[0],
                make_leg("buy", "call", 660.0, index=1, symbol=occ("call", 700.0)),
            ),
        )
        assert misnamed.legs[1].strike == 660.0 and misnamed.legs[1].symbol.endswith("00700000")
        sound = judge(declared, context_for(declared))
        assert sound.verdict is Verdict.NEEDS_APPROVAL
        evaluation = judge(misnamed, context_for(declared))  # the same flattering context
        failing = assert_rejected_by(evaluation, "quote_freshness")  # no quote for the 700
        assert "no quote for leg SPY260821C00700000" in failing.detail
        price = result_of(evaluation, "limit_price_sanity")
        assert price.outcome is REJECT and "no quote for leg SPY260821C00700000" in price.detail
        max_loss = result_of(evaluation, "options_max_loss")
        assert max_loss.outcome is REJECT
        assert max_loss.detail == (
            "the structure cannot be analysed: leg SPY260821C00700000 names the 2026-08-21 "
            "700 call, but the leg says the 2026-08-21 660 call"
        )
        # ... and with a quote for the contract it names, options_max_loss is the failing rule.
        quoted = context_for(misnamed, mids=[3.60, 1.20], analysis=context_for(declared).analysis)
        assert_rejected_by(judge(misnamed, quoted), "options_max_loss")

    def test_a_zero_dte_contract(self):
        proposal = long_call(expiration=MARKET_DATE)
        evaluation = judge(proposal, context_for(proposal))
        failing = assert_rejected_by(evaluation, "options_min_dte")
        assert "0 DTE" in failing.detail
        assert found(evaluation) == {
            "options_min_dte": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
        }

    def test_a_duplicate_inside_the_window(self):
        evaluation = judge(context=make_context(recent_proposals=(make_recent(),)))
        failing = assert_rejected_by(evaluation, "duplicate")
        assert "prop-0000" in failing.detail
        assert found(evaluation) == {"duplicate": REJECT}

    def test_a_symbol_on_the_no_trade_list(self):
        context = make_context(limits=make_limits(no_trade_list=[SYMBOL]))
        evaluation = judge(context=context)
        assert_rejected_by(evaluation, "no_trade_list")
        assert found(evaluation) == {"no_trade_list": REJECT}

    def test_a_market_order(self):
        evaluation = judge(market_order())
        failing = assert_rejected_by(evaluation, "limit_price_sanity")
        assert "market orders are not allowed" in failing.detail
        assert found(evaluation) == {"limit_price_sanity": REJECT, "auto_tier": ESCALATE}

    @pytest.mark.parametrize("limit", [140.0, 260.0], ids=["30% under", "30% over"])
    def test_a_limit_price_thirty_percent_off_the_mid(self, limit):
        evaluation = judge(make_proposal(limit_price=limit))  # the mid is 200.00
        failing = assert_rejected_by(evaluation, "limit_price_sanity")
        assert "30% away" in failing.detail
        assert found(evaluation) == {"limit_price_sanity": REJECT}

    def test_confidence_below_the_floor_is_flag_only(self):
        evaluation = judge(low_confidence())
        assert evaluation.verdict is Verdict.FLAG_ONLY
        assert evaluation.failing_rule is None
        assert result_of(evaluation, "min_confidence").outcome is FLAG
        assert found(evaluation) == {"min_confidence": FLAG}

    @pytest.mark.parametrize("text", ["", "   \n"], ids=["empty", "blank"])
    def test_a_missing_invalidation(self, text):
        evaluation = judge(make_proposal(invalidation=text))
        assert_rejected_by(evaluation, "invalidation_present")
        assert found(evaluation) == {"invalidation_present": REJECT}

    def test_after_the_loss_limit_tripped_a_recovered_pnl_does_not_reopen_trading(self):
        context = make_context(controls=Controls(halt_until=NEXT_OPEN), daily_pnl=1_500.0)
        evaluation = judge(context=context)
        assert_rejected_by(evaluation, "halted")
        assert result_of(evaluation, "daily_loss_limit").outcome is PASS
        assert found(evaluation) == {"halted": REJECT}

    def test_a_context_that_knows_nothing_is_rejected(self):
        blind = make_context(
            clock=None, equity=None, cash=None, buying_power=None, options_buying_power=None,
            start_of_day_equity=None, daily_pnl=None, positions=None, quote=None,
            errors=("account: unavailable", "clock: unavailable", "quote: unavailable"),
        )
        evaluation = judge(context=blind)
        assert_rejected_by(evaluation, "market_hours")
        assert evaluation.halt_until is None
        for proposal in (build() for build in STRUCTURES):
            assert judge(proposal, blind).verdict is Verdict.REJECT

    def test_everything_wrong_at_once_still_shows_every_rule(self):
        proposal = market_order(invalidation="", confidence=0.0, quantity=1e6)
        context = make_context(
            controls=Controls(kill_switch=True, halt_until=NOW + timedelta(days=1)),
            clock=make_clock(False),
            daily_pnl=LOSS,
            orders_today=10,
            recent_proposals=(make_recent(),),
            limits=make_limits(no_trade_list=[SYMBOL]),
        )
        evaluation = judge(proposal, context)
        assert_rejected_by(evaluation, "kill_switch")
        assert len(evaluation.results) == 21
        assert len(evaluation.non_pass) >= 12


# --- the tiers ----------------------------------------------------------------


class TestTiers:
    def test_a_watchlist_limit_buy_within_the_max_notional_auto_executes(self):
        evaluation = judge()  # 2 AAPL at 200.00 = 400.00, under the 1,000.00 tier
        assert evaluation.verdict is Verdict.AUTO_EXECUTE
        assert evaluation.failing_rule is None
        assert [result.outcome for result in evaluation.results] == [PASS] * 21

    def test_exactly_at_the_max_notional_still_auto_executes(self):
        evaluation = judge(make_proposal(quantity=5.0))  # 5 x 200.00 = 1,000.00
        assert evaluation.verdict is Verdict.AUTO_EXECUTE
        assert found(evaluation) == {}

    def test_one_dollar_over_needs_approval(self):
        proposal = over_the_auto_tier()
        evaluation = judge(proposal, context_for(proposal))
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert evaluation.failing_rule is None
        assert found(evaluation) == {"auto_tier": ESCALATE}
        assert "$1,001.00" in result_of(evaluation, "auto_tier").detail

    def test_the_max_notional_comes_from_the_limits(self):
        proposal = over_the_auto_tier()
        raised = make_limits(auto_execute={"max_notional": 1_001.0})
        assert judge(proposal, context_for(proposal, limits=raised)).verdict is Verdict.AUTO_EXECUTE
        lowered = make_limits(auto_execute={"max_notional": 399.0})
        assert judge(context=make_context(limits=lowered)).verdict is Verdict.NEEDS_APPROVAL

    def test_auto_execute_disabled_needs_approval(self):
        context = make_context(limits=make_limits(auto_execute={"enabled": False}))
        evaluation = judge(context=context)
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert found(evaluation) == {"auto_tier": ESCALATE}

    @pytest.mark.parametrize("build", DEFINED_RISK, ids=lambda b: b.__name__)
    def test_a_defined_risk_option_structure_needs_approval(self, build):
        proposal = build()
        evaluation = judge(proposal, context_for(proposal))
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert evaluation.failing_rule is None
        assert found(evaluation) == {"options_escalate": ESCALATE, "auto_tier": ESCALATE}

    @pytest.mark.parametrize("build", STRUCTURES, ids=lambda b: b.__name__)
    def test_no_option_ever_auto_executes_however_generous_the_limits(self, build):
        generous = make_limits(
            max_loss_per_trade=1e12, max_contracts=10**6, max_position_pct=100.0, min_dte=0,
            limit_price_tolerance_pct=1e6, watchlist_only=False, allow_market_orders=True,
            max_open_positions=10**6, max_daily_trades=10**6, min_confidence=0.0,
            auto_execute={"enabled": True, "max_notional": 1e12},
        )
        for overrides in ({}, {"quantity": 2.0}, {"confidence": 1.0}):
            proposal = build(**overrides)
            context = context_for(
                proposal, limits=generous, equity=1e12, cash=1e12, buying_power=1e12,
                options_buying_power=1e12, start_of_day_equity=1e12,
            )
            evaluation = judge(proposal, context)
            assert evaluation.verdict in (Verdict.NEEDS_APPROVAL, Verdict.REJECT)
            assert result_of(evaluation, "options_escalate").outcome is ESCALATE
            assert result_of(evaluation, "auto_tier").outcome is ESCALATE

    def test_a_closing_sale_auto_executes_and_a_short_sale_does_not(self):
        sale = make_proposal(side=OrderSide.SELL)
        closing = judge(sale, make_context(positions=(make_position(qty=10.0),)))
        assert closing.verdict is Verdict.AUTO_EXECUTE
        short = judge(sale)  # nothing held
        assert short.verdict is Verdict.NEEDS_APPROVAL
        assert found(short) == {"short_sale": ESCALATE, "auto_tier": ESCALATE}
        strict = make_context(limits=make_limits(reject_short_sales=True))
        assert_rejected_by(judge(sale, strict), "short_sale")

    def test_a_symbol_outside_the_watchlist_never_auto_executes(self):
        proposal = make_proposal(symbol="TSLA")
        assert_rejected_by(judge(proposal, context_for(proposal)), "watchlist_only")
        open_list = make_limits(watchlist_only=False)
        evaluation = judge(proposal, context_for(proposal, limits=open_list))
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert found(evaluation) == {"auto_tier": ESCALATE}


# --- closing sales: the one sell that may auto-execute -------------------------

ACCOUNT_STOPS = [
    pytest.param({"controls": Controls(kill_switch=True)}, "kill_switch", id="kill-switch"),
    pytest.param({"controls": Controls(halt_until=NEXT_OPEN)}, "halted", id="halt-in-force"),
    pytest.param(
        {"controls": Controls(halt_unknown=True, problems=("halt_until is unreadable",))},
        "halted",
        id="halt-unreadable",
    ),
    pytest.param({"clock": make_clock(False)}, "market_hours", id="market-closed"),
    pytest.param({"clock": None}, "market_hours", id="no-clock"),
    pytest.param({"clock": make_clock(next_close=NOW)}, "market_hours", id="stale-clock"),
    pytest.param({"daily_pnl": LOSS}, "daily_loss_limit", id="loss-limit-tripped"),
    pytest.param({"daily_pnl": None}, "daily_loss_limit", id="pnl-unknown"),
    pytest.param({"orders_today": 10}, "max_daily_trades", id="trade-cap"),
]
"""Each account-level stop on its own, with the rule that must reject for it."""

PROPOSAL_STOPS = [
    pytest.param({}, {"limits": make_limits(no_trade_list=[SYMBOL])}, "no_trade_list",
                 id="no-trade-list"),
    pytest.param({"invalidation": "  "}, {}, "invalidation_present", id="blank-invalidation"),
    pytest.param({"invalidation": ""}, {}, "invalidation_present", id="no-invalidation"),
    pytest.param({}, {"recent_proposals": (make_recent(side=OrderSide.SELL),)}, "duplicate",
                 id="duplicate-in-the-window"),
    pytest.param({}, {"open_orders": (make_order(SYMBOL, side=OrderSide.SELL),)}, "duplicate",
                 id="open-order-on-the-symbol"),
    pytest.param({"limit_price": 140.0}, {}, "limit_price_sanity", id="30%-under-the-mid"),
    pytest.param({"limit_price": 260.0}, {}, "limit_price_sanity", id="30%-over-the-mid"),
    pytest.param({}, {"quote": make_quote(at=NOW - timedelta(seconds=121))}, "quote_freshness",
                 id="stale-quote"),
    pytest.param({}, {"quote": make_quote(at=None)}, "quote_freshness", id="undated-quote"),
]
"""What is wrong with the sale itself (proposal overrides, context overrides)
and the rule that must reject it."""


class TestClosingSales:
    """A closing sale — an equity SELL of a long that is held — is exempt from
    the sizing rules and reaches the auto tier. It is exempt from nothing
    else: every stop and every sanity rule holds for it as for a buy."""

    def test_the_sale_these_cases_start_from_auto_executes(self):
        evaluation = judge(closing_sale(), make_context(positions=HELD))
        assert evaluation.verdict is Verdict.AUTO_EXECUTE
        assert found(evaluation) == {}

    @pytest.mark.parametrize(("stop", "rule"), ACCOUNT_STOPS)
    def test_every_account_level_stop_rejects_a_closing_sale(self, stop, rule):
        evaluation = judge(closing_sale(), make_context(positions=HELD, **stop))
        assert_rejected_by(evaluation, rule)
        assert found(evaluation) == {rule: REJECT}  # the stop, and nothing but the stop
        assert evaluation.verdict is not Verdict.AUTO_EXECUTE
        # The trip still names its halt for the engine to persist.
        expected_halt = NEXT_OPEN if stop.get("daily_pnl") == LOSS else None
        assert evaluation.halt_until == expected_halt
        assert result_of(evaluation, "daily_loss_limit").halt_until == expected_halt

    @pytest.mark.parametrize(("stop", "rule"), ACCOUNT_STOPS)
    def test_the_stops_hold_for_a_sale_of_the_whole_position_too(self, stop, rule):
        whole = closing_sale(quantity=10.0, limit_price=100.0)  # 1,000.00: inside the tier
        context = context_for(whole, positions=HELD, **stop)
        assert judge(whole, context_for(whole, positions=HELD)).verdict is Verdict.AUTO_EXECUTE
        assert_rejected_by(judge(whole, context), rule)

    def test_a_tripped_loss_limit_on_a_closing_sale_persists_the_halt(self, conn, calls):
        sale = stored(conn, closing_sale())
        decision = evaluate(sale, make_context(positions=HELD, daily_pnl=LOSS), conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "daily_loss_limit"
        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert get_controls(conn).halt_until == NEXT_OPEN
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1

    @pytest.mark.parametrize(("overrides", "context_overrides", "rule"), PROPOSAL_STOPS)
    def test_every_sanity_rule_rejects_a_closing_sale(self, overrides, context_overrides, rule):
        sale = closing_sale(**overrides)
        evaluation = judge(sale, make_context(positions=HELD, **context_overrides))
        assert_rejected_by(evaluation, rule)
        assert found(evaluation) == {rule: REJECT}
        # But for that one thing it is the sale that auto-executes.
        assert judge(closing_sale(), make_context(positions=HELD)).verdict is Verdict.AUTO_EXECUTE

    def test_a_closing_sale_with_no_quote_is_rejected_by_both_quote_rules(self):
        # No quote at all: nothing to date and nothing to price. The age is
        # named first; the price check rejects it too.
        evaluation = judge(closing_sale(), make_context(positions=HELD, quote=None))
        failing = assert_rejected_by(evaluation, "quote_freshness")
        assert "no quote for AAPL" in failing.detail
        assert found(evaluation) == {"quote_freshness": REJECT, "limit_price_sanity": REJECT}

    def test_a_closing_sale_at_market_is_rejected_unless_market_orders_are_allowed(self):
        at_market = closing_sale(order_type=OrderType.MARKET, limit_price=None)
        evaluation = judge(at_market, make_context(positions=HELD))
        failing = assert_rejected_by(evaluation, "limit_price_sanity")
        assert "market orders are not allowed" in failing.detail
        assert found(evaluation) == {"limit_price_sanity": REJECT, "auto_tier": ESCALATE}
        allowed = make_context(positions=HELD, limits=make_limits(allow_market_orders=True))
        evaluation = judge(at_market, allowed)
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL  # never the auto tier
        assert found(evaluation) == {"auto_tier": ESCALATE}

    def test_a_closing_sale_below_the_confidence_floor_is_flag_only(self):
        evaluation = judge(closing_sale(confidence=0.2), make_context(positions=HELD))
        assert evaluation.verdict is Verdict.FLAG_ONLY
        assert evaluation.failing_rule is None
        assert found(evaluation) == {"min_confidence": FLAG}

    def test_a_closing_sale_outside_the_watchlist_is_rejected(self):
        sale = closing_sale(symbol="TSLA")
        held = (make_position("TSLA", qty=10.0),)
        evaluation = judge(sale, context_for(sale, positions=held))
        assert_rejected_by(evaluation, "watchlist_only")
        assert found(evaluation) == {"watchlist_only": REJECT, "auto_tier": ESCALATE}

    @pytest.mark.parametrize(
        ("limits", "context_overrides"),
        [
            ({"auto_execute": {"enabled": False}}, {}),
            ({"auto_execute": {"max_notional": 399.0}}, {}),
            ({"auto_execute": {"max_notional": 0.0}}, {}),
            ({"watchlist_only": False}, {"watchlist": ("SPY",)}),
        ],
        ids=["tier-disabled", "over-the-tier", "tier-admits-nothing", "off-the-watchlist"],
    )
    def test_a_closing_sale_needs_every_auto_tier_criterion(self, limits, context_overrides):
        context = make_context(positions=HELD, limits=make_limits(**limits), **context_overrides)
        evaluation = judge(closing_sale(), context)
        assert evaluation.verdict is Verdict.NEEDS_APPROVAL
        assert found(evaluation) == {"auto_tier": ESCALATE}

    @pytest.mark.parametrize(
        "positions",
        [
            (),
            None,
            (make_position(qty=1.0),),  # one share held, two sold: one would be short
            (make_position(qty=-10.0, side="short"),),
            (make_position("MSFT", qty=10.0),),  # a long, in another symbol
        ],
        ids=["nothing-held", "positions-unknown", "more-than-held", "short", "another-symbol"],
    )
    def test_a_sale_with_no_long_to_close_never_auto_executes(self, positions):
        evaluation = judge(closing_sale(), make_context(positions=positions))
        assert evaluation.verdict is not Verdict.AUTO_EXECUTE
        assert result_of(evaluation, "short_sale").outcome is ESCALATE
        assert result_of(evaluation, "auto_tier").outcome is ESCALATE
        strict = make_context(positions=positions, limits=make_limits(reject_short_sales=True))
        assert result_of(judge(closing_sale(), strict), "short_sale").outcome is REJECT


# --- precedence ---------------------------------------------------------------


class TestPrecedence:
    def test_a_flag_and_an_escalate_is_flag_only(self):
        proposal = over_the_auto_tier(confidence=0.2)
        evaluation = judge(proposal, context_for(proposal))
        assert found(evaluation) == {"min_confidence": FLAG, "auto_tier": ESCALATE}
        assert evaluation.verdict is Verdict.FLAG_ONLY
        assert evaluation.failing_rule is None

    @pytest.mark.parametrize("build", DEFINED_RISK, ids=lambda b: b.__name__)
    def test_a_low_confidence_option_is_flag_only_not_needs_approval(self, build):
        proposal = build(confidence=0.1)
        evaluation = judge(proposal, context_for(proposal))
        assert found(evaluation) == {
            "options_escalate": ESCALATE, "min_confidence": FLAG, "auto_tier": ESCALATE,
        }
        assert evaluation.verdict is Verdict.FLAG_ONLY

    def test_a_reject_beats_a_flag_and_an_escalate(self):
        proposal = over_the_auto_tier(confidence=0.2, invalidation="")
        evaluation = judge(proposal, context_for(proposal))
        assert found(evaluation) == {
            "invalidation_present": REJECT, "min_confidence": FLAG, "auto_tier": ESCALATE,
        }
        assert_rejected_by(evaluation, "invalidation_present")

    def test_a_reject_after_the_flag_and_the_escalate_in_registry_order_still_wins(self):
        # short_sale (18) escalates, min_confidence (19) flags; the reject is duplicate (12).
        proposal = make_proposal(side=OrderSide.SELL, confidence=0.2)
        twin = make_recent(side=OrderSide.SELL)
        evaluation = judge(proposal, make_context(recent_proposals=(twin,)))
        assert found(evaluation) == {
            "duplicate": REJECT, "short_sale": ESCALATE, "min_confidence": FLAG,
            "auto_tier": ESCALATE,
        }
        assert_rejected_by(evaluation, "duplicate")

    def test_the_failing_rule_is_the_first_reject_in_registry_order(self):
        # A later-rule reject (a market order: limit_price_sanity, rule 13) together
        # with an earlier-rule reject (the kill switch, rule 1).
        later_only = judge(market_order())
        assert_rejected_by(later_only, "limit_price_sanity")
        both = judge(market_order(), make_context(controls=Controls(kill_switch=True)))
        assert found(both) == {
            "kill_switch": REJECT, "limit_price_sanity": REJECT, "auto_tier": ESCALATE,
        }
        assert_rejected_by(both, "kill_switch")

    @pytest.mark.parametrize(
        ("overrides", "context_overrides", "first"),
        [
            ({"invalidation": ""}, {"recent_proposals": (make_recent(),)}, "invalidation_present"),
            ({}, {"recent_proposals": (make_recent(),), "orders_today": 10}, "max_daily_trades"),
            ({"quantity": 250.0}, {"clock": None}, "market_hours"),
            ({"limit_price": 260.0}, {"daily_pnl": LOSS}, "daily_loss_limit"),
            ({"quantity": 250.0, "limit_price": 260.0}, {}, "max_position_pct"),
        ],
        ids=["7+12", "11+12", "3+9", "4+13", "9+13"],
    )
    def test_two_rejects_name_the_earlier_rule(self, overrides, context_overrides, first):
        evaluation = judge(make_proposal(**overrides), make_context(**context_overrides))
        rejected = [name for name, outcome in found(evaluation).items() if outcome is REJECT]
        assert len(rejected) == 2 and rejected[0] == first
        assert_rejected_by(evaluation, first)

    def test_the_verdict_always_matches_the_outcomes(self):
        pairs = [
            (make_proposal(), make_context()),
            (low_confidence(), make_context()),
            (over_the_auto_tier(), context_for(over_the_auto_tier())),
            (market_order(), make_context()),
            *((build(), context_for(build())) for build in STRUCTURES),
        ]
        for proposal, context in pairs:
            evaluation = judge(proposal, context)
            outcomes = [result.outcome for result in evaluation.results]
            assert evaluation.verdict is expected_verdict(outcomes)


# --- halt persistence, end to end on a store ----------------------------------


class TestHaltPersistence:
    def test_a_trip_rejects_persists_the_halt_and_logs_one_event(self, conn):
        proposal = stored(conn)
        assert get_controls(conn).halt_until is None
        decision = evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "daily_loss_limit"
        controls = get_controls(conn)
        assert controls.halt_until == NEXT_OPEN  # the clock's next open
        assert controls.halt_until_updated_at == NOW  # the context's now, not the wall clock
        assert controls.kill_switch is False and controls.problems == ()
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert RISK_LIMIT_TRIPPED == "risk_limit_tripped"
        assert tripped.level is EventLevel.CRITICAL
        assert tripped.occurred_at == NOW
        assert tripped.message == TRIP_DETAIL
        assert tripped.payload == {
            "rule": "daily_loss_limit",
            "proposal_id": PROPOSAL_ID,
            "daily_pnl": -2_500.0,
            "start_of_day_equity": 100_000.0,
            "daily_loss_limit_pct": 2.0,
            "cap": 2_000.0,
            "halt_until": "2026-07-31T13:30:00+00:00",
        }

    def test_after_a_restart_a_recovered_pnl_is_still_rejected_by_the_halt(self, conn, db_path):
        proposal = stored(conn)
        evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        conn.close()

        reopened = open_store(db_path)  # a new process: nothing in memory survives
        try:
            controls = get_controls(reopened)
            assert controls.halt_until == NEXT_OPEN
            later = NOW + timedelta(hours=2)
            assert later < controls.halt_until
            fresh = make_context(
                now=later, controls=controls, daily_pnl=1_500.0, quote=make_quote(at=later)
            )
            decision = evaluate(proposal, fresh, reopened)
            assert decision.verdict is Verdict.REJECT
            assert decision.failing_rule == "halted"
            outcome = {entry["rule"]: entry["outcome"] for entry in decision.rules_evaluated}
            assert outcome["halted"] == "REJECT"
            assert outcome["daily_loss_limit"] == "PASS"  # the P&L recovered; the halt did not lift
            assert [name for name, value in outcome.items() if value != "PASS"] == ["halted"]
            # ... and that rejection wrote no new halt and no new trip event.
            assert get_controls(reopened).halt_until == NEXT_OPEN
            assert get_controls(reopened).halt_until_updated_at == NOW
            assert len(events(reopened, RISK_LIMIT_TRIPPED)) == 1
        finally:
            reopened.close()

    @pytest.mark.parametrize(
        ("offset", "halted"),
        [
            (timedelta(seconds=-1), True),
            (timedelta(0), False),
            (timedelta(seconds=1), False),
            (timedelta(hours=3), False),
        ],
        ids=["a second before", "at the instant", "a second after", "hours after"],
    )
    def test_at_and_after_the_halt_instant_trading_resumes(self, conn, offset, halted):
        proposal = stored(conn)
        evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        now = NEXT_OPEN + offset
        next_day = make_context(
            now=now,
            market_date=date(2026, 7, 31),
            quote=make_quote(at=now),
            controls=get_controls(conn),
            clock=make_clock(
                next_open=NEXT_OPEN + timedelta(days=3),
                next_close=NEXT_OPEN + timedelta(hours=6, minutes=30),
            ),
            daily_pnl=0.0,
        )
        evaluation = decide(proposal, next_day)
        assert (result_of(evaluation, "halted").outcome is REJECT) == halted
        if halted:
            assert_rejected_by(evaluation, "halted")
        else:
            assert evaluation.verdict is Verdict.AUTO_EXECUTE
            assert "expired" in result_of(evaluation, "halted").detail

    def test_a_second_trip_while_halted_writes_no_second_halt_or_event(self, conn, calls):
        proposal = stored(conn)
        evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        assert names_of(calls) == ["set_halt_until", "record_decision"]

        later = NOW + timedelta(hours=1)
        still_losing = make_context(now=later, controls=get_controls(conn), daily_pnl=-3_000.0)
        assert decide(proposal, still_losing).halt_until is None
        decision = evaluate(proposal, still_losing, conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        outcome = {entry["rule"]: entry["outcome"] for entry in decision.rules_evaluated}
        assert outcome["daily_loss_limit"] == "REJECT"  # it tripped again ...
        assert names_of(calls) == ["set_halt_until", "record_decision", "record_decision"]
        controls = get_controls(conn)
        assert controls.halt_until == NEXT_OPEN
        assert controls.halt_until_updated_at == NOW  # ... but the row was not rewritten
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1
        assert len(events(conn, "policy_reject")) == 2

    def test_a_trip_while_the_stored_halt_is_unreadable_writes_the_finite_halt(
        self, conn, calls
    ):
        proposal = stored(conn)
        # A stored halt nobody can read (the CHECK is bypassed, as a damaged
        # row would be): the store fails closed — halted, until nobody knows when.
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE controls SET value = 'tomorrow, probably' WHERE key = 'halt_until'")
        conn.commit()
        conn.execute("PRAGMA ignore_check_constraints = OFF")
        unreadable = get_controls(conn)
        assert unreadable.halt_unknown is True and unreadable.halt_until is None
        assert unreadable.problems

        context = make_context(daily_pnl=LOSS, controls=unreadable)
        assert decide(proposal, context).halt_until == NEXT_OPEN
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "halted"  # the unreadable halt already rejects ...
        outcome = {entry["rule"]: entry["outcome"] for entry in decision.rules_evaluated}
        assert outcome["halted"] == "REJECT" and outcome["daily_loss_limit"] == "REJECT"
        # ... and the trip replaces it with a halt that has an end.
        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert calls[0][1] == NEXT_OPEN
        controls = get_controls(conn)
        assert controls.halt_until == NEXT_OPEN
        assert controls.halt_unknown is False and controls.problems == ()
        assert controls.halt_until_updated_at == NOW
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert tripped.payload["halt_until"] == "2026-07-31T13:30:00+00:00"
        # Once readable, the halt is in force until its instant, and then it lifts.
        still = make_context(now=NOW + timedelta(hours=1), controls=controls)
        assert_rejected_by(decide(proposal, still), "halted")
        after = make_context(
            now=NEXT_OPEN,
            market_date=date(2026, 7, 31),
            controls=controls,
            clock=make_clock(
                next_open=NEXT_OPEN + timedelta(days=3),
                next_close=NEXT_OPEN + timedelta(hours=6, minutes=30),
            ),
        )
        assert result_of(decide(proposal, after), "halted").outcome is PASS

    @pytest.mark.parametrize(
        ("read", "written"),
        [
            (NOW - timedelta(hours=2), NEXT_OPEN),
            (NEXT_OPEN + timedelta(days=3), NEXT_OPEN + timedelta(days=3)),
        ],
        ids=["an-expired-halt", "a-longer-halt"],
    )
    def test_a_halt_damaged_after_a_context_read_it_is_repaired_by_the_trip(
        self, conn, calls, read, written
    ):
        """The state only ``stricter_controls`` produces: the context read a
        halt, the row became unreadable before ``evaluate`` looked again. The
        trip writes a finite halt — the later of its own instant and the one
        the context read — so the store never stays unreadable behind it."""
        proposal = stored(conn)
        repo.set_halt_until(conn, read, now=NOW - timedelta(hours=3))
        context = make_context(daily_pnl=LOSS, controls=get_controls(conn))
        assert context.controls.halt_until == read and not context.controls.halt_unknown
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE controls SET value = 'tomorrow, probably' WHERE key = 'halt_until'")
        conn.commit()
        conn.execute("PRAGMA ignore_check_constraints = OFF")
        assert get_controls(conn).halt_unknown is True

        decision = evaluate(proposal, context, conn)

        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        assert decision.rules_evaluated[1]["detail"].startswith(
            "cannot prove trading is not halted: halt_until value"
        )
        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert calls[0][1] == written
        controls = get_controls(conn)
        assert controls.halt_until == written  # finite, and no shorter than either reading
        assert controls.halt_unknown is False and controls.problems == ()
        assert controls.halt_until_updated_at == NOW
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert tripped.payload["halt_until"] == written.isoformat()
        # The audit line — and the event's message, which is that line — names
        # the halt this evaluation wrote, never as one "already in force".
        detail = decision.rules_evaluated[3]["detail"]
        assert decision.rules_evaluated[3]["rule"] == "daily_loss_limit"
        assert f": trading is halted until {written.isoformat()}" in detail
        assert "already in force" not in detail
        assert detail.endswith("(one reading of it was unreadable)") is (written != NEXT_OPEN)
        assert tripped.message == detail

    def test_a_store_repaired_since_an_unreadable_reading_is_asked_and_not_rewritten(
        self, conn, calls
    ):
        # The other way round: the CONTEXT's reading was the unreadable one, and
        # the store holds a readable halt by the time evaluate looks again.
        proposal = stored(conn)
        longer = NEXT_OPEN + timedelta(days=3)
        repo.set_halt_until(conn, longer, now=NOW - timedelta(hours=3))
        unreadable = Controls(halt_unknown=True, problems=("halt_until value 'x' is unreadable",))
        decision = evaluate(proposal, make_context(daily_pnl=LOSS, controls=unreadable), conn)
        assert decision.failing_rule == "halted"
        # The engine asks for the later instant; the store, which holds it, writes nothing.
        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert calls[0][1] == longer
        controls = get_controls(conn)
        assert controls.halt_until == longer
        assert controls.halt_until_updated_at == NOW - timedelta(hours=3)
        assert events(conn, RISK_LIMIT_TRIPPED) == []
        # The audit line is true this way round too: it says until when trading
        # is halted and why two readings are behind it — not who wrote the halt.
        assert decision.rules_evaluated[3]["detail"].endswith(
            f": trading is halted until {longer.isoformat()}, the later instant another"
            " reading of the stored halt gave (one reading of it was unreadable)"
        )
        # A readable halt SHORTER than the trip's is extended to the trip's, once.
        repo.set_halt_until(conn, NOW + timedelta(hours=1), now=NOW - timedelta(hours=3))
        evaluate(proposal, make_context(daily_pnl=LOSS, controls=unreadable), conn)
        assert get_controls(conn).halt_until == NEXT_OPEN
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1

    def test_the_fallback_when_the_clock_has_no_next_open(self, conn):
        proposal = stored(conn)
        context = make_context(
            daily_pnl=LOSS,
            clock=make_clock(next_open=None),
            limits=make_limits(halt_fallback_hours=6.0),
        )
        decision = evaluate(proposal, context, conn)
        assert decision.failing_rule == "daily_loss_limit"
        assert get_controls(conn).halt_until == NOW + timedelta(hours=6)
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert tripped.payload["halt_until"] == "2026-07-30T21:00:00+00:00"

    def test_the_fallback_when_there_is_no_clock_at_all(self, conn):
        proposal = stored(conn)
        decision = evaluate(proposal, make_context(daily_pnl=LOSS, clock=None), conn)
        assert decision.failing_rule == "market_hours"  # the first rejecting rule ...
        assert get_controls(conn).halt_until == NOW + timedelta(hours=24)  # ... and still halted
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1

    def test_the_halt_is_persisted_whichever_rule_the_decision_names(self, conn):
        proposal = stored(conn)
        context = make_context(daily_pnl=LOSS, controls=Controls(kill_switch=True))
        decision = evaluate(proposal, context, conn)
        assert decision.failing_rule == "kill_switch"
        assert get_controls(conn).halt_until == NEXT_OPEN
        assert [event.kind for event in events(conn)] == [RISK_LIMIT_TRIPPED, "policy_reject"]

    def test_a_later_trip_extends_a_halt_and_never_shortens_one(self, conn):
        proposal = stored(conn)
        repo.set_halt_until(conn, NOW + timedelta(hours=1), now=NOW - timedelta(hours=1))
        shorter = make_context(daily_pnl=LOSS, controls=get_controls(conn))
        evaluate(proposal, shorter, conn)
        assert get_controls(conn).halt_until == NEXT_OPEN  # extended
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1

        far = NEXT_OPEN + timedelta(days=7)
        repo.set_halt_until(conn, far, now=NOW)
        longer = make_context(daily_pnl=LOSS, controls=get_controls(conn))
        evaluate(proposal, longer, conn)
        assert get_controls(conn).halt_until == far  # not shortened
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1

    def test_an_unknown_pnl_rejects_but_halts_nothing(self, conn, calls):
        proposal = stored(conn)
        decision = evaluate(proposal, make_context(daily_pnl=None), conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "daily_loss_limit"
        assert names_of(calls) == ["record_decision"]
        assert get_controls(conn).halt_until is None
        assert events(conn, RISK_LIMIT_TRIPPED) == []

    def test_the_halt_lands_before_the_decision_and_survives_its_failure(self, conn, calls):
        # The proposal was never stored: recording the decision fails on its
        # foreign key — after the halt and its event were committed.
        proposal = make_proposal()
        with pytest.raises(StoreError, match="record decision"):
            evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert get_controls(conn).halt_until == NEXT_OPEN
        assert [event.kind for event in events(conn)] == [RISK_LIMIT_TRIPPED]
        assert decisions(conn) == []

    def test_a_failing_halt_write_propagates_and_no_decision_is_recorded(
        self, conn, monkeypatch
    ):
        proposal = stored(conn)
        recorded = []

        def failing_halt(conn, until, *, now=None, event=None, only_extend=False):
            raise StoreError("set halt until", "halt_until")

        monkeypatch.setattr(engine, "set_halt_until", failing_halt)
        monkeypatch.setattr(engine, "record_decision", lambda *a, **k: recorded.append(a))
        with pytest.raises(StoreError, match="set halt until"):
            evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        assert recorded == []
        assert decisions(conn) == [] and events(conn) == []

    def test_the_halt_and_its_event_are_one_write(self, conn, calls):
        proposal = stored(conn)
        evaluate(proposal, make_context(daily_pnl=LOSS), conn)
        name, until, event = calls[0]
        assert name == "set_halt_until" and until == NEXT_OPEN
        assert isinstance(event, Event) and event.kind == RISK_LIMIT_TRIPPED
        assert event == events(conn, RISK_LIMIT_TRIPPED)[0]

    # A trip BEFORE the open: the halt covers the session that is about to
    # begin, so a P&L that recovers by the opening bell does not reopen
    # trading that day.

    def test_a_trip_before_the_open_keeps_trading_halted_through_that_session(
        self, conn, db_path
    ):
        closed = make_clock(False, next_open=OPENS_TODAY, next_close=NEXT_CLOSE)
        first = stored(conn, make_proposal(created_at=EARLY - timedelta(minutes=1)))
        pre_session = make_context(
            now=EARLY, clock=closed, daily_pnl=LOSS, controls=get_controls(conn)
        )
        assert pre_session.market_date == pre_session.next_open_date == MARKET_DATE
        decision = evaluate(first, pre_session, conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "market_hours"  # closed: the first rule to reject
        outcome = {entry["rule"]: entry for entry in decision.rules_evaluated}
        assert outcome["daily_loss_limit"]["outcome"] == "REJECT"
        assert outcome["daily_loss_limit"]["detail"].endswith(
            "trading is halted until 2026-07-30T20:00:00+00:00, the close of the session "
            "that opens at 2026-07-30T13:30:00+00:00"
        )
        # The halt lasts to the CLOSE of today's session — not to its open.
        assert get_controls(conn).halt_until == NEXT_CLOSE != OPENS_TODAY
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert tripped.payload["halt_until"] == "2026-07-30T20:00:00+00:00"
        assert tripped.occurred_at == EARLY
        conn.close()

        # 09:31 the same day, in a new process: the market is open and the
        # P&L has recovered — the very case the halt exists for.
        reopened = open_store(db_path)
        try:
            second = stored(
                reopened,
                make_proposal(id="prop-0002", created_at=AT_0931 - timedelta(seconds=30)),
            )
            fresh = make_context(now=AT_0931, daily_pnl=1_500.0, controls=get_controls(reopened))
            assert fresh.clock.is_open and fresh.market_date == MARKET_DATE
            decision = evaluate(second, fresh, reopened)
            assert decision.verdict is Verdict.REJECT
            assert decision.failing_rule == "halted"
            outcome = {entry["rule"]: entry["outcome"] for entry in decision.rules_evaluated}
            assert outcome["daily_loss_limit"] == "PASS"  # recovered — and still halted
            assert [name for name, value in outcome.items() if value != "PASS"] == ["halted"]
            # Without the halt this is the order that auto-executes.
            unhalted = fresh.model_copy(update={"controls": Controls()})
            assert decide(second, unhalted).verdict is Verdict.AUTO_EXECUTE
            # Nothing new was written, and the halt holds to the last minute of the session.
            assert get_controls(reopened).halt_until == NEXT_CLOSE
            assert len(events(reopened, RISK_LIMIT_TRIPPED)) == 1
            last_minute = fresh.model_copy(update={"now": NEXT_CLOSE - timedelta(minutes=1)})
            assert_rejected_by(decide(second, last_minute), "halted")
            # The next morning the halt has expired and trading resumes.
            next_morning = make_context(
                now=NEXT_OPEN + timedelta(minutes=1),
                market_date=date(2026, 7, 31),
                quote=make_quote(at=NEXT_OPEN + timedelta(minutes=1)),
                controls=get_controls(reopened),
                clock=make_clock(
                    next_open=NEXT_OPEN + timedelta(days=3),
                    next_close=NEXT_OPEN + timedelta(hours=6, minutes=30),
                ),
            )
            assert decide(second, next_morning).verdict is Verdict.AUTO_EXECUTE
        finally:
            reopened.close()

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, NEXT_CLOSE),
            ({"next_open_date": None}, OPENS_TODAY),
            ({"clock": make_clock(False, next_open=OPENS_TODAY, next_close=None)}, OPENS_TODAY),
            (
                {"clock": make_clock(False, next_open=OPENS_TODAY, next_close=OPENS_TODAY)},
                OPENS_TODAY,
            ),
        ],
        ids=["covers-the-session", "no-next-open-date", "no-close", "close-not-after-open"],
    )
    def test_the_pre_session_halt_the_engine_persists(self, conn, overrides, expected):
        fields = {
            "now": EARLY,
            "clock": make_clock(False, next_open=OPENS_TODAY, next_close=NEXT_CLOSE),
            "daily_pnl": LOSS,
            **overrides,
        }
        proposal = stored(conn, make_proposal(created_at=EARLY - timedelta(minutes=1)))
        evaluate(proposal, make_context(**fields), conn)
        assert get_controls(conn).halt_until == expected
        (tripped,) = events(conn, RISK_LIMIT_TRIPPED)
        assert tripped.payload["halt_until"] == expected.isoformat()

    def test_an_after_hours_and_an_intraday_trip_still_halt_to_the_next_open(self, conn):
        after_hours = make_context(
            now=NEXT_CLOSE + timedelta(hours=1),  # 17:00 New York: tomorrow's open is next
            clock=make_clock(
                False, next_open=NEXT_OPEN, next_close=NEXT_OPEN + timedelta(hours=6, minutes=30)
            ),
            daily_pnl=LOSS,
        )
        assert after_hours.next_open_date == date(2026, 7, 31) != after_hours.market_date
        assert decide(make_proposal(), after_hours).halt_until == NEXT_OPEN
        intraday = make_context(daily_pnl=LOSS)
        assert intraday.clock.is_open
        assert decide(make_proposal(), intraday).halt_until == NEXT_OPEN
        evaluate(stored(conn), intraday, conn)
        assert get_controls(conn).halt_until == NEXT_OPEN


# --- the stop flags are read again before anything is recorded -----------------


def operator(db_path, write) -> None:
    """Another process changing the store — the operator's CLI, the loop —
    while this one still holds a context it built earlier."""
    other = open_store(db_path)
    try:
        write(other)
    finally:
        other.close()


@pytest.fixture
def judged(monkeypatch):
    """Record the context ``evaluate`` actually hands to ``decide``."""
    seen = []
    real = engine.decide

    def decide_spy(proposal, context):
        seen.append(context)
        return real(proposal, context)

    monkeypatch.setattr(engine, "decide", decide_spy)
    return seen


class TestStopFlagsReadAgain:
    """A context is a snapshot. ``evaluate`` reads the store's kill switch
    and halt once more and judges under the stricter of the two readings."""

    def test_a_kill_switch_set_after_the_context_was_built_rejects(
        self, conn, db_path, calls, judged
    ):
        proposal = stored(conn)
        context = make_context(controls=get_controls(conn))  # built: the switch is off
        assert decide(proposal, context).verdict is Verdict.AUTO_EXECUTE
        operator(db_path, lambda other: repo.set_kill_switch(other, True, now=NOW))

        decision = evaluate(proposal, context, conn)

        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "kill_switch"
        assert decision.rules_evaluated[0] == {
            "rule": "kill_switch",
            "outcome": "REJECT",
            "detail": "the kill switch is ON: every proposal is rejected",
        }
        assert decision.notes == "REJECT by kill_switch (REJECT: kill_switch)"
        assert names_of(calls) == ["record_decision"]  # exactly once, as ever
        # What was recorded is decide() on the context with the store's controls ...
        (under,) = judged
        assert under is not context and under.controls == get_controls(conn)
        assert under.model_dump(exclude={"controls"}) == context.model_dump(exclude={"controls"})
        assert decision.rules_evaluated == decide(proposal, under).rules_evaluated
        assert get_proposal_trace(conn, proposal.id).decision == decision
        (event,) = events(conn)
        assert event.kind == "policy_reject" and event.payload["failing_rule"] == "kill_switch"
        # ... and the caller's context is untouched: decide stays a pure function of it.
        assert context.controls.kill_switch is False
        assert decide(proposal, context).verdict is Verdict.AUTO_EXECUTE

    def test_a_halt_written_after_the_context_was_built_rejects(
        self, conn, db_path, calls, judged
    ):
        proposal = stored(conn)
        context = make_context(controls=get_controls(conn), daily_pnl=1_500.0)
        operator(db_path, lambda other: repo.set_halt_until(other, NEXT_OPEN, now=NOW))

        decision = evaluate(proposal, context, conn)

        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        outcome = {entry["rule"]: entry for entry in decision.rules_evaluated}
        assert outcome["halted"]["detail"] == (
            "trading is halted until 2026-07-31T13:30:00+00:00 (now 2026-07-30T15:00:00+00:00)"
        )
        assert [name for name, entry in outcome.items() if entry["outcome"] != "PASS"] == [
            "halted"
        ]
        assert names_of(calls) == ["record_decision"]
        assert judged[0].controls.halt_until == NEXT_OPEN
        assert get_controls(conn).halt_until == NEXT_OPEN  # read, not rewritten
        assert events(conn, RISK_LIMIT_TRIPPED) == []

    def test_a_halt_that_became_unreadable_after_the_context_was_built_rejects(
        self, conn, calls
    ):
        proposal = stored(conn)
        context = make_context(controls=get_controls(conn))
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE controls SET value = 'tomorrow, probably' WHERE key = 'halt_until'")
        conn.execute("PRAGMA ignore_check_constraints = OFF")
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        halted = decision.rules_evaluated[1]
        assert halted["detail"].startswith("cannot prove trading is not halted: halt_until value")
        assert names_of(calls) == ["record_decision"]

    def test_a_halt_from_another_trip_is_honoured_and_not_re_announced(
        self, conn, db_path, calls
    ):
        # The loop judged a losing proposal while this context was being built.
        first = stored(conn)
        second = stored(conn, make_proposal(id="prop-0002", symbol="MSFT"))
        stale = context_for(second, controls=get_controls(conn), daily_pnl=LOSS)
        assert decide(second, stale).halt_until == NEXT_OPEN  # it would announce the halt ...
        operator(db_path, lambda other: evaluate(first, make_context(daily_pnl=LOSS), other))
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1
        calls.clear()  # the other evaluator's writes are not this one's

        decision = evaluate(second, stale, conn)

        assert decision.failing_rule == "halted"  # ... but the store already holds it
        assert names_of(calls) == ["record_decision"]
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1
        assert get_controls(conn).halt_until == NEXT_OPEN

    @pytest.mark.parametrize(
        ("held", "rule"),
        [
            (Controls(kill_switch=True), "kill_switch"),
            (Controls(halt_until=NEXT_OPEN), "halted"),
            (Controls(halt_unknown=True, problems=("halt_until is unreadable",)), "halted"),
        ],
        ids=["kill-switch", "halt", "unreadable-halt"],
    )
    def test_nothing_the_context_holds_is_lifted_by_the_store(
        self, conn, calls, judged, held, rule
    ):
        # The store says "no stop at all"; the context saw one. The context stands.
        proposal = stored(conn)
        assert get_controls(conn).model_dump(include={"kill_switch", "halt_until"}) == {
            "kill_switch": False, "halt_until": None,
        }
        context = make_context(controls=held)
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == rule
        assert names_of(calls) == ["record_decision"]
        assert judged == [context] and judged[0] is context  # judged exactly as given
        assert decision.rules_evaluated == decide(proposal, context).rules_evaluated

    def test_a_later_halt_in_the_context_is_not_shortened_by_an_earlier_stored_one(
        self, conn, judged
    ):
        proposal = stored(conn)
        repo.set_halt_until(conn, NOW + timedelta(hours=1), now=NOW)
        two_hours_on = NOW + timedelta(hours=2)
        context = make_context(controls=Controls(halt_until=NEXT_OPEN), now=two_hours_on)
        decision = evaluate(proposal, context, conn)
        assert decision.failing_rule == "halted"  # the stored halt expired; the context's did not
        assert judged[0] is context

    @pytest.mark.parametrize(
        "build",
        [
            lambda conn: make_context(),
            lambda conn: make_context(controls=get_controls(conn)),
            lambda conn: make_context(controls=Controls(kill_switch=True)),
            lambda conn: make_context(controls=Controls(halt_until=NEXT_OPEN), daily_pnl=LOSS),
            lambda conn: make_context(daily_pnl=LOSS),
        ],
        ids=["hand-made", "read-from-the-store", "kill-on", "halted-and-tripping", "tripping"],
    )
    def test_when_the_store_adds_nothing_the_context_is_used_as_it_is(
        self, conn, calls, judged, build
    ):
        proposal = stored(conn)
        context = build(conn)
        expected = decide(proposal, context)
        judged.clear()
        decision = evaluate(proposal, context, conn)
        assert len(judged) == 1 and judged[0] is context  # the same object, no copy
        assert decision.verdict is expected.verdict
        assert decision.rules_evaluated == expected.rules_evaluated
        assert names_of(calls).count("record_decision") == 1

    def test_the_store_is_read_exactly_once_and_before_anything_is_written(
        self, conn, monkeypatch
    ):
        order = []
        real_read, real_record, real_halt = (
            engine.get_controls, engine.record_decision, engine.set_halt_until,
        )

        def get_controls_spy(connection):
            order.append("get_controls")
            assert connection is conn
            return real_read(connection)

        def record(connection, decision, *, event=None):
            order.append("record_decision")
            return real_record(connection, decision, event=event)

        def halt(connection, until, **kwargs):
            order.append("set_halt_until")
            return real_halt(connection, until, **kwargs)

        monkeypatch.setattr(engine, "get_controls", get_controls_spy)
        monkeypatch.setattr(engine, "record_decision", record)
        monkeypatch.setattr(engine, "set_halt_until", halt)
        evaluate(stored(conn), make_context(daily_pnl=LOSS), conn)
        assert order == ["get_controls", "set_halt_until", "record_decision"]

    def test_the_engine_reads_through_the_stores_own_function(self):
        assert engine.get_controls is repo.get_controls

    def test_a_dry_run_reads_nothing_and_judges_the_context_as_given(
        self, conn, db_path, judged, monkeypatch
    ):
        proposal = stored(conn)
        context = make_context(controls=get_controls(conn))
        operator(db_path, lambda other: repo.set_kill_switch(other, True, now=NOW))

        def never(connection):
            raise AssertionError("a dry run read the store")

        monkeypatch.setattr(engine, "get_controls", never)
        dry = evaluate(proposal, context, Untouchable(), dry_run=True)
        assert dry.verdict is Verdict.AUTO_EXECUTE  # the context's own verdict
        assert judged == [context] and judged[0] is context
        assert decisions(conn) == []

    def test_a_store_that_cannot_be_read_records_nothing(self, conn, monkeypatch):
        proposal = stored(conn)
        recorded = []

        def unreadable(connection):
            raise StoreError("get controls", cause=RuntimeError("disk gone"))

        monkeypatch.setattr(engine, "get_controls", unreadable)
        monkeypatch.setattr(engine, "record_decision", lambda *a, **k: recorded.append(a))
        with pytest.raises(StoreError, match="get controls"):
            evaluate(proposal, make_context(), conn)
        assert recorded == [] and decisions(conn) == [] and events(conn) == []


LATER = NEXT_OPEN + timedelta(days=3)
EARLIER = NOW + timedelta(hours=1)
EXPIRED = NOW - timedelta(days=1)
STOP_READINGS = [
    Controls(),
    Controls(kill_switch=True),
    Controls(halt_until=EXPIRED),
    Controls(halt_until=EARLIER),
    Controls(halt_until=LATER),
    Controls(halt_unknown=True, problems=("halt_until is unreadable",)),
    Controls(kill_switch=True, halt_until=LATER),
    Controls(kill_switch=True, halt_unknown=True, problems=("kill_switch is missing",)),
]
"""Readings of the stop flags, from "no stop at all" to "every stop there is"."""
STOP_IDS = [
    "none", "kill", "halt-expired", "halt-soon", "halt-late", "halt-unknown", "kill+halt",
    "kill+unknown",
]


def _stops(controls: Controls, now: datetime = NOW) -> tuple[bool, bool]:
    """(the kill switch rejects, the halt rejects) under ``controls`` — worked
    out here, not asked of the rules."""
    halt = controls.halt_until
    return controls.kill_switch, controls.halt_unknown or (halt is not None and halt > now)


class TestStricterControls:
    """``stricter_controls(seen, stored)``: every stop either reading holds."""

    def test_it_is_public_documented_and_pure(self):
        assert engine.stricter_controls.__doc__ and engine.stricter_controls.__doc__.strip()
        assert list(inspect.signature(engine.stricter_controls).parameters) == ["seen", "stored"]
        assert not hasattr(engine, "_stricter")
        seen, stored_ = Controls(halt_until=EARLIER), Controls(kill_switch=True)
        before = (seen.model_dump_json(), stored_.model_dump_json())
        first = engine.stricter_controls(seen, stored_)
        assert first == engine.stricter_controls(seen, stored_)
        assert (seen.model_dump_json(), stored_.model_dump_json()) == before

    @pytest.mark.parametrize("seen", STOP_READINGS, ids=STOP_IDS)
    @pytest.mark.parametrize("stored_", STOP_READINGS, ids=STOP_IDS)
    def test_a_stop_in_either_reading_is_a_stop_in_the_result(self, seen, stored_):
        merged = engine.stricter_controls(seen, stored_)
        assert isinstance(merged, Controls)
        assert merged.kill_switch is (seen.kill_switch or stored_.kill_switch)
        assert merged.halt_unknown is (seen.halt_unknown or stored_.halt_unknown)
        halts = [halt for halt in (seen.halt_until, stored_.halt_until) if halt is not None]
        assert merged.halt_until == (max(halts) if halts else None)
        # Whatever instant it is asked at, nothing either reading stops is let through.
        for now in (EXPIRED - timedelta(days=1), NOW, EARLIER, LATER - timedelta(seconds=1), LATER):
            for index in (0, 1):
                either = _stops(seen, now)[index] or _stops(stored_, now)[index]
                assert _stops(merged, now)[index] is either
        # ... and the rules agree, on the default pair.
        rejected = {
            name
            for controls in (seen, stored_)
            for name, outcome in found(judge(context=make_context(controls=controls))).items()
        }
        under = found(judge(context=make_context(controls=merged)))
        assert set(under) == rejected

    @pytest.mark.parametrize("seen", STOP_READINGS, ids=STOP_IDS)
    def test_the_seen_reading_comes_back_itself_when_the_store_adds_nothing(self, seen):
        assert engine.stricter_controls(seen, seen) is seen
        assert engine.stricter_controls(seen, Controls()) is seen
        assert engine.stricter_controls(seen, seen.model_copy()) is seen
        # a store that holds LESS than the context saw adds nothing either
        if seen.halt_until is not None:
            shorter = Controls(halt_until=seen.halt_until - timedelta(seconds=1))
            assert engine.stricter_controls(seen, shorter) is seen
        # ... nor do the store's timestamps or notes, on their own
        touched = Controls(kill_switch_updated_at=NOW, halt_until_updated_at=NOW)
        assert engine.stricter_controls(seen, touched) is seen

    def test_what_the_store_adds_is_taken_with_its_explanation(self):
        seen = Controls(
            halt_until=EARLIER,
            kill_switch_updated_at=NOW - timedelta(days=1),
            problems=("seen: a note", "both: halt_until is unreadable"),
        )
        stored_ = Controls(
            kill_switch=True,
            halt_until=LATER,
            kill_switch_updated_at=NOW,
            halt_until_updated_at=NOW - timedelta(minutes=5),
            problems=("both: halt_until is unreadable", "stored: kill_switch is missing"),
        )
        assert engine.stricter_controls(seen, stored_) == Controls(
            kill_switch=True,
            halt_until=LATER,
            kill_switch_updated_at=NOW,  # when the store's rows were last set
            halt_until_updated_at=NOW - timedelta(minutes=5),
            # what either reading could not read, the context's first, each line once
            problems=(
                "seen: a note",
                "both: halt_until is unreadable",
                "stored: kill_switch is missing",
            ),
        )
        # The explanation the halted rule prints is still there after the merge.
        unreadable = Controls(halt_unknown=True, problems=("halt_until value 'x' is unreadable",))
        merged = engine.stricter_controls(unreadable, Controls(kill_switch=True))
        assert merged.problems == ("halt_until value 'x' is unreadable",)
        halted = result_of(judge(context=make_context(controls=merged)), "halted")
        assert halted.detail == (
            "cannot prove trading is not halted: halt_until value 'x' is unreadable"
        )

    def test_one_second_is_enough_to_tell_two_halts_apart(self):
        seen = Controls(halt_until=NEXT_OPEN)
        assert engine.stricter_controls(seen, Controls(halt_until=NEXT_OPEN)) is seen
        a_second_on = NEXT_OPEN + timedelta(seconds=1)
        later = engine.stricter_controls(seen, Controls(halt_until=a_second_on))
        assert later is not seen and later.halt_until == a_second_on
        # the same instant written in another offset is the same halt
        new_york = timezone(timedelta(hours=-4))
        same = Controls(halt_until=NEXT_OPEN.astimezone(new_york))
        assert engine.stricter_controls(seen, same) is seen


# --- a halt is only ever extended, decided against the store ------------------

FALLBACK = NOW + timedelta(hours=24)
"""The halt a trip names with no clock: ``halt_fallback_hours`` (24) from now —
ninety minutes LATER than tomorrow's 13:30 open."""


class TestHaltOnlyExtends:
    def test_two_contexts_built_before_the_first_write_never_shorten_the_halt(
        self, conn, calls
    ):
        first = stored(conn)
        second = stored(conn, make_proposal(id="prop-0002", symbol="MSFT"))
        before = get_controls(conn)
        # Both built while the store held no halt: one without a clock (the
        # fallback), one with it (the next open, which is EARLIER).
        no_clock = make_context(daily_pnl=LOSS, clock=None, controls=before)
        with_clock = context_for(second, daily_pnl=LOSS, controls=before)
        assert decide(first, no_clock).halt_until == FALLBACK
        assert decide(second, with_clock).halt_until == NEXT_OPEN < FALLBACK

        evaluate(first, no_clock, conn)
        assert get_controls(conn).halt_until == FALLBACK
        decision = evaluate(second, with_clock, conn)

        assert get_controls(conn).halt_until == FALLBACK  # still the later instant
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1  # and announced once
        assert get_controls(conn).halt_until_updated_at == NOW
        # The second verdict was judged under the halt in force, and says so.
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "halted"
        outcome = {entry["rule"]: entry for entry in decision.rules_evaluated}
        assert outcome["daily_loss_limit"]["outcome"] == "REJECT"
        assert outcome["daily_loss_limit"]["detail"].endswith(
            ": a halt until 2026-07-31T15:00:00+00:00 is already in force"
        )
        assert names_of(calls) == ["set_halt_until", "record_decision", "record_decision"]

    def test_the_store_refuses_to_shorten_even_when_the_engine_asks(
        self, conn, calls, monkeypatch
    ):
        """The race the re-read cannot close: the longer halt lands AFTER
        ``evaluate`` read the flags and before it writes. The write itself is
        a compare-and-set, so the store keeps the later instant."""
        first = stored(conn)
        second = stored(conn, make_proposal(id="prop-0002", symbol="MSFT"))
        before = get_controls(conn)
        evaluate(first, make_context(daily_pnl=LOSS, clock=None, controls=before), conn)
        assert get_controls(conn).halt_until == FALLBACK
        calls.clear()

        monkeypatch.setattr(engine, "get_controls", lambda connection: before)  # a stale read
        decision = evaluate(second, context_for(second, daily_pnl=LOSS, controls=before), conn)

        assert names_of(calls) == ["set_halt_until", "record_decision"]
        assert calls[0][1] == NEXT_OPEN < FALLBACK  # the engine did ask for the earlier halt
        controls = repo.get_controls(conn)
        assert controls.halt_until == FALLBACK  # ... and the store did not take it
        assert controls.halt_until_updated_at == NOW
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1  # nor its event
        assert decision.verdict is Verdict.REJECT  # the decision is recorded all the same
        assert len(decisions(conn)) == 2

    def test_a_longer_halt_from_a_stale_context_does_extend(self, conn, calls):
        first = stored(conn)
        second = stored(conn, make_proposal(id="prop-0002", symbol="MSFT"))
        before = get_controls(conn)
        with_clock = make_context(daily_pnl=LOSS, controls=before)
        no_clock = context_for(second, daily_pnl=LOSS, clock=None, controls=before)
        evaluate(first, with_clock, conn)
        assert get_controls(conn).halt_until == NEXT_OPEN
        evaluate(second, no_clock, conn)
        assert get_controls(conn).halt_until == FALLBACK  # a real extension is written
        kinds = [event.kind for event in events(conn)]
        assert kinds.count(RISK_LIMIT_TRIPPED) == 2  # ... and announced: the halt changed
        assert names_of(calls).count("set_halt_until") == 2

    def test_one_context_reused_for_two_proposals_announces_the_halt_once(self, conn, calls):
        first = stored(conn)
        second = stored(conn, make_proposal(id="prop-0002", symbol="MSFT"))
        context = make_context(daily_pnl=LOSS, controls=get_controls(conn))
        evaluate(first, context, conn)
        evaluate(second, context, conn)
        evaluate(first, context, conn)
        assert names_of(calls).count("set_halt_until") == 1
        assert names_of(calls).count("record_decision") == 3
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == 1
        assert get_controls(conn).halt_until == NEXT_OPEN

    def test_the_engine_never_clears_or_shortens_an_operators_halt(self, conn):
        proposal = stored(conn)
        far = NEXT_OPEN + timedelta(days=30)
        repo.set_halt_until(conn, far, now=NOW - timedelta(hours=3))
        for context in (
            make_context(daily_pnl=LOSS),  # built before the operator's halt
            make_context(daily_pnl=LOSS, clock=None),
            make_context(daily_pnl=1_500.0),
        ):
            assert evaluate(proposal, context, conn).failing_rule == "halted"
            controls = get_controls(conn)
            assert controls.halt_until == far
            assert controls.halt_until_updated_at == NOW - timedelta(hours=3)
        assert events(conn, RISK_LIMIT_TRIPPED) == []


# --- the audit trail ----------------------------------------------------------


def _reject_case():
    return make_proposal(invalidation=""), make_context()


def _flag_case():
    return low_confidence(), make_context()


def _approval_case():
    proposal = iron_condor()
    return proposal, context_for(proposal)


def _auto_case():
    return make_proposal(), make_context()


VERDICT_CASES = [
    pytest.param(_reject_case, Verdict.REJECT, id="REJECT"),
    pytest.param(_flag_case, Verdict.FLAG_ONLY, id="FLAG_ONLY"),
    pytest.param(_approval_case, Verdict.NEEDS_APPROVAL, id="NEEDS_APPROVAL"),
    pytest.param(_auto_case, Verdict.AUTO_EXECUTE, id="AUTO_EXECUTE"),
]


class TestPersistence:
    @pytest.mark.parametrize(("case", "verdict"), VERDICT_CASES)
    def test_record_decision_is_called_exactly_once_per_evaluation(
        self, conn, calls, case, verdict
    ):
        proposal, context = case()
        stored(conn, proposal)
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is verdict
        assert names_of(calls) == ["record_decision"]
        _, recorded, _ = calls[0]
        assert recorded == decision
        assert len(decisions(conn)) == 1

    def test_once_per_evaluation_however_many_times_a_proposal_is_judged(self, conn, calls):
        proposal = stored(conn)
        first = evaluate(proposal, make_context(), conn)
        second = evaluate(proposal, make_context(now=NOW + timedelta(minutes=5)), conn)
        assert names_of(calls) == ["record_decision", "record_decision"]
        assert first.id != second.id
        trace = get_proposal_trace(conn, proposal.id)
        assert trace.decisions == (first, second)
        assert trace.decision == second

    @pytest.mark.parametrize(("case", "verdict"), VERDICT_CASES)
    def test_rules_evaluated_lists_all_twenty_one_rules_in_order(self, conn, case, verdict):
        proposal, context = case()
        stored(conn, proposal)
        decision = evaluate(proposal, context, conn)
        listed = decision.rules_evaluated
        assert len(listed) == 21
        assert [entry["rule"] for entry in listed] == list(RULE_NAMES)
        for entry in listed:
            assert set(entry) == {"rule", "outcome", "detail"}
            assert entry["outcome"] in {"PASS", "FLAG", "ESCALATE", "REJECT"}
            assert isinstance(entry["detail"], str) and entry["detail"].strip()
        evaluation = decide(proposal, context)
        assert listed == [
            {"rule": r.name, "outcome": r.outcome.value, "detail": r.detail}
            for r in evaluation.results
        ]

    def test_the_decision_carries_the_verdict_the_failing_rule_and_the_contexts_now(self, conn):
        proposal = stored(conn, make_proposal(invalidation=""))
        decision = evaluate(proposal, make_context(), conn)
        assert isinstance(decision, PolicyDecision)
        assert decision.proposal_id == PROPOSAL_ID
        assert decision.decided_at == NOW
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "invalidation_present"
        assert decision.notes == "REJECT by invalidation_present (REJECT: invalidation_present)"

    @pytest.mark.parametrize(
        ("build", "context_overrides", "notes"),
        [
            (make_proposal, {}, "AUTO_EXECUTE (all 21 rules passed)"),
            (low_confidence, {}, "FLAG_ONLY (FLAG: min_confidence)"),
            (
                lambda: over_the_auto_tier(confidence=0.2),
                {},
                "FLAG_ONLY (FLAG: min_confidence; ESCALATE: auto_tier)",
            ),
            (iron_condor, {}, "NEEDS_APPROVAL (ESCALATE: options_escalate, auto_tier)"),
            (
                market_order,
                {"controls": Controls(kill_switch=True)},
                "REJECT by kill_switch (REJECT: kill_switch, limit_price_sanity; "
                "ESCALATE: auto_tier)",
            ),
        ],
        ids=["auto", "flag", "flag+escalate", "approval", "reject"],
    )
    def test_the_notes_are_a_one_line_summary(self, build, context_overrides, notes):
        proposal = build()
        decision = evaluate(
            proposal, context_for(proposal, **context_overrides), Untouchable(), dry_run=True
        )
        assert decision.notes == notes
        assert "\n" not in decision.notes

    @pytest.mark.parametrize(("case", "verdict"), VERDICT_CASES)
    def test_the_decision_round_trips_through_the_proposal_trace(self, conn, case, verdict):
        proposal, context = case()
        stored(conn, proposal)
        decision = evaluate(proposal, context, conn)
        trace = get_proposal_trace(conn, proposal.id)
        assert trace.decisions == (decision,)
        assert trace.decision == decision
        assert trace.decision.verdict is verdict
        assert trace.decision.rules_evaluated == decision.rules_evaluated
        assert trace.legs == proposal.legs

    @pytest.mark.parametrize(
        ("case", "verdict", "kind", "level"),
        [
            (_reject_case, Verdict.REJECT, "policy_reject", EventLevel.WARNING),
            (_flag_case, Verdict.FLAG_ONLY, "policy_flag_only", EventLevel.INFO),
            (_approval_case, Verdict.NEEDS_APPROVAL, "policy_needs_approval", EventLevel.INFO),
        ],
        ids=["REJECT", "FLAG_ONLY", "NEEDS_APPROVAL"],
    )
    def test_a_verdict_event_is_logged_with_the_decision(
        self, conn, calls, case, verdict, kind, level
    ):
        proposal, context = case()
        stored(conn, proposal)
        decision = evaluate(proposal, context, conn)
        assert decision.verdict is verdict
        (event,) = events(conn)
        assert event.kind == kind == VERDICT_EVENT_KINDS[verdict]
        assert event.level is level
        assert event.occurred_at == NOW
        evaluation = decide(proposal, context)
        assert event.payload == {
            "proposal_id": proposal.id,
            "decision_id": decision.id,
            "verdict": verdict.value,
            "failing_rule": decision.failing_rule,
            "symbol": proposal.proposal.symbol,
            "non_pass": [
                {"rule": r.name, "outcome": r.outcome.value, "detail": r.detail}
                for r in evaluation.non_pass
            ],
        }
        assert event.payload["non_pass"]
        assert event.message == (
            f"proposal {proposal.id} ({proposal.proposal.side.value} "
            f"{proposal.proposal.symbol}): {decision.notes}"
        )
        # It went to the store with the decision — one call, one transaction.
        (_, _, passed) = calls[0]
        assert passed == event

    def test_auto_execute_logs_no_event(self, conn, calls):
        proposal = stored(conn)
        decision = evaluate(proposal, make_context(), conn)
        assert decision.verdict is Verdict.AUTO_EXECUTE
        assert events(conn) == []
        assert calls == [("record_decision", decision, None)]
        assert Verdict.AUTO_EXECUTE not in VERDICT_EVENT_KINDS
        assert set(VERDICT_EVENT_KINDS) == set(Verdict) - {Verdict.AUTO_EXECUTE}

    def test_the_failing_rule_in_the_event_is_null_unless_rejected(self, conn):
        proposal = stored(conn, low_confidence())
        evaluate(proposal, make_context(), conn)
        (event,) = events(conn)
        assert event.payload["failing_rule"] is None
        assert event.payload["non_pass"] == [
            {
                "rule": "min_confidence",
                "outcome": "FLAG",
                "detail": "below the floor: confidence 0.2, floor 0.5 (risk_limits.min_confidence)",
            }
        ]

    def test_dry_run_returns_the_same_decision_unsaved(self, conn):
        proposal = stored(conn, make_proposal(invalidation=""))
        dry = evaluate(proposal, make_context(), conn, dry_run=True)
        real = evaluate(proposal, make_context(), conn)
        assert isinstance(dry, PolicyDecision)
        assert dry.model_dump(exclude={"id"}) == real.model_dump(exclude={"id"})
        assert get_proposal_trace(conn, proposal.id).decisions == (real,)

    @pytest.mark.parametrize(("case", "verdict"), VERDICT_CASES)
    def test_dry_run_never_touches_the_connection(self, calls, case, verdict):
        proposal, context = case()
        decision = evaluate(proposal, context, Untouchable(), dry_run=True)
        assert decision.verdict is verdict
        assert calls == []

    def test_dry_run_leaves_every_table_unchanged(self, conn, db_path, calls):
        # A store with history, and an evaluation that would write all it can:
        # a halt, its event, a decision and the decision's event.
        proposal = stored(conn, make_proposal(invalidation=""))
        evaluate(proposal, make_context(), conn)
        tripping = make_context(daily_pnl=LOSS, now=NOW + timedelta(minutes=1))
        calls.clear()
        before = every_row(conn)
        changes = conn.total_changes
        files = [db_path, db_path.with_name(db_path.name + "-wal")]
        contents = [path.read_bytes() for path in files]

        decision = evaluate(proposal, tripping, conn, dry_run=True)

        assert decision.verdict is Verdict.REJECT
        assert decide(proposal, tripping).halt_until == NEXT_OPEN  # it would have halted
        assert calls == []
        assert conn.total_changes == changes
        assert every_row(conn) == before
        assert [path.read_bytes() for path in files] == contents
        assert get_controls(conn).halt_until is None
        assert len(before["policy_decisions"]) == 1 and len(before["events"]) == 1

        # The same evaluation, for real, changes exactly what the dry run left alone.
        evaluate(proposal, tripping, conn)
        after = every_row(conn)
        changed = {name for name in after if after[name] != before[name]}
        assert changed == {"controls", "events", "policy_decisions"}
        assert len(after["policy_decisions"]) == 2 and len(after["events"]) == 3

    def test_a_store_failure_propagates_as_a_store_error(self, conn, db_path):
        unknown = make_proposal(id="never-stored")
        with pytest.raises(StoreError, match="record decision") as raised:
            evaluate(unknown, make_context(), conn)
        assert "FOREIGN KEY" in str(raised.value)
        assert decisions(conn) == [] and events(conn) == []

        # A connection that cannot be read fails at the first thing evaluate
        # asks of it — the store's stop flags — and nothing is recorded.
        proposal = stored(conn)
        closed = open_store(db_path)
        closed.close()
        with pytest.raises(StoreError, match="get controls"):
            evaluate(proposal, make_context(), closed)
        assert decisions(conn) == []

    def test_a_decision_that_cannot_be_recorded_leaves_no_event_behind(self, conn):
        with pytest.raises(StoreError):
            evaluate(make_proposal(id="never-stored", invalidation=""), make_context(), conn)
        assert events(conn) == []

    def test_evaluate_returns_what_record_decision_returns(self, conn, monkeypatch):
        proposal = stored(conn)
        marker = PolicyDecision(
            proposal_id=proposal.id, verdict=Verdict.REJECT, rules_evaluated=[], notes="as stored"
        )
        monkeypatch.setattr(engine, "record_decision", lambda conn, decision, *, event: marker)
        assert evaluate(proposal, make_context(), conn) is marker

    def test_the_engine_writes_through_the_stores_own_functions(self):
        assert engine.record_decision is repo.record_decision
        assert engine.set_halt_until is repo.set_halt_until


# --- the engine fails closed --------------------------------------------------


class TestFailClosedEngine:
    """``decide`` on the default pair — AUTO_EXECUTE with the real registry —
    with one rule of the registry broken."""

    def test_the_default_pair_would_auto_execute(self):
        assert judge().verdict is Verdict.AUTO_EXECUTE

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_a_rule_that_raises_rejects_and_the_others_still_run(self, monkeypatch, name):
        def broken(proposal, context):
            raise RuntimeError(f"{name} blew up")

        broken.__name__ = name
        monkeypatch.setattr(engine, "RULES", _replacing(name, broken))
        evaluation = decide(make_proposal(), make_context())
        assert tuple(result.name for result in evaluation.results) == RULE_NAMES
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == name
        assert found(evaluation) == {name: REJECT}
        assert result_of(evaluation, name).detail == (
            f"rule failed: RuntimeError: {name} blew up"
        )

    @pytest.mark.parametrize("returned", [None, "PASS", PASS, True], ids=repr)
    def test_a_rule_that_returns_the_wrong_type_rejects(self, monkeypatch, returned):
        def auto_tier(proposal, context):
            return returned

        monkeypatch.setattr(engine, "RULES", _replacing("auto_tier", auto_tier))
        evaluation = decide(make_proposal(), make_context())
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == "auto_tier"
        assert found(evaluation) == {"auto_tier": REJECT}
        assert "not a RuleResult" in result_of(evaluation, "auto_tier").detail

    def test_a_rule_that_answers_under_the_wrong_name_rejects(self, monkeypatch):
        def kill_switch(proposal, context):
            return RuleResult(name="halted", outcome=PASS, detail="no halt")

        monkeypatch.setattr(engine, "RULES", _replacing("kill_switch", kill_switch))
        evaluation = decide(make_proposal(), make_context())
        assert tuple(result.name for result in evaluation.results) == RULE_NAMES
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == "kill_switch"
        assert found(evaluation) == {"kill_switch": REJECT}

    @pytest.mark.parametrize("detail", [None, "", "  "], ids=repr)
    def test_an_auto_tier_pass_without_a_detail_is_not_a_pass(self, monkeypatch, detail):
        # Twenty PASS outcomes — one of them a result nobody validated. It
        # must not be the AUTO_EXECUTE it looks like, with a null in the audit trail.
        def auto_tier(proposal, context):
            return RuleResult.model_construct(
                name="auto_tier", outcome=PASS, detail=detail, halt_until=None
            )

        monkeypatch.setattr(engine, "RULES", _replacing("auto_tier", auto_tier))
        evaluation = decide(make_proposal(), make_context())
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == "auto_tier"
        assert found(evaluation) == {"auto_tier": REJECT}
        assert all(isinstance(r.detail, str) and r.detail.strip() for r in evaluation.results)
        assert all(entry["detail"] for entry in evaluation.rules_evaluated)

    @pytest.mark.parametrize("halt", ["tomorrow", 0, date(2026, 7, 31)], ids=repr)
    def test_a_trip_with_a_garbled_halt_rejects_and_nothing_raises(
        self, conn, calls, monkeypatch, halt
    ):
        def daily_loss_limit(proposal, context):
            return RuleResult.model_construct(
                name="daily_loss_limit", outcome=REJECT, detail="tripped", halt_until=halt
            )

        monkeypatch.setattr(engine, "RULES", _replacing("daily_loss_limit", daily_loss_limit))
        evaluation = decide(make_proposal(), make_context())  # an evaluation WITH a verdict
        assert_rejected_by(evaluation, "daily_loss_limit")
        assert found(evaluation) == {"daily_loss_limit": REJECT}
        assert result_of(evaluation, "daily_loss_limit").detail == (
            f"rule failed: ValueError: returned the halt {halt!r}, not a datetime"
        )
        assert evaluation.halt_until is None
        # ... recorded like any rejection, and no halt is written from it.
        decision = evaluate(stored(conn), make_context(), conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "daily_loss_limit"
        assert names_of(calls) == ["record_decision"]
        assert get_controls(conn).halt_until is None and events(conn, RISK_LIMIT_TRIPPED) == []

    @pytest.mark.parametrize("name", RULE_NAMES)
    def test_a_rule_missing_from_the_registry_is_an_integrity_reject(self, monkeypatch, name):
        monkeypatch.setattr(
            engine, "RULES", tuple(rule for rule in RULES if rule.__name__ != name)
        )
        evaluation = decide(make_proposal(), make_context())
        assert len(evaluation.results) == 20
        assert {result.outcome for result in evaluation.results} == {PASS}
        assert evaluation.verdict is Verdict.REJECT
        assert evaluation.failing_rule == ENGINE_INTEGRITY

    def test_an_extra_reordered_or_repeated_rule_is_an_integrity_reject(self, monkeypatch):
        extra = (*RULES, fake_rule("one_more_rule"))
        reordered = (RULES[1], RULES[0], *RULES[2:])
        repeated = (*RULES, RULES[0])
        for registry in (extra, reordered, repeated, ()):
            monkeypatch.setattr(engine, "RULES", registry)
            evaluation = decide(make_proposal(), make_context())
            assert evaluation.verdict is Verdict.REJECT
            assert evaluation.failing_rule == ENGINE_INTEGRITY
            assert all(result.outcome is PASS for result in evaluation.results)

    def test_decide_checks_the_names_it_imported_whatever_rules_is_patched_to(self, monkeypatch):
        """What the integrity guard proves, and no more: the rules that ran
        are the rules ``RULE_NAMES`` lists. ``RULE_NAMES`` is derived from the
        registry, so the guard is not what pins "exactly these twenty-one" — a
        registry shrunk together with its names is judged on twenty rules.
        That pin is the literal list: the next test here (``THE_TWENTY_ONE``), and
        ``TestRegistry.test_twenty_one_rules_in_the_specified_order`` in
        tests/test_policy_rules.py."""
        monkeypatch.setattr(engine, "RULES", RULES[:-1])
        assert decide(make_proposal(), make_context()).failing_rule == ENGINE_INTEGRITY
        assert engine.RULE_NAMES == RULE_NAMES and len(engine.RULE_NAMES) == 21
        # Shrink the names too and the guard has nothing left to object to:
        # one dollar over the auto tier, with auto_tier gone, auto-executes.
        proposal = over_the_auto_tier()
        context = context_for(proposal)
        monkeypatch.setattr(engine, "RULES", RULES)
        assert decide(proposal, context).verdict is Verdict.NEEDS_APPROVAL
        monkeypatch.setattr(engine, "RULES", RULES[:-1])
        monkeypatch.setattr(engine, "RULE_NAMES", RULE_NAMES[:-1])
        blessed = decide(proposal, context)
        assert (blessed.verdict, len(blessed.results)) == (Verdict.AUTO_EXECUTE, 20)

    def test_the_engine_checks_against_the_users_twenty_one_rules(self):
        """The engine's own anchor: the names it compares the results with
        are the user's twenty-one, written out here — not whatever the registry
        happens to hold."""
        assert len(THE_TWENTY_ONE) == len(set(THE_TWENTY_ONE)) == 21
        assert engine.RULE_NAMES == THE_TWENTY_ONE
        assert tuple(rule.__name__ for rule in engine.RULES) == THE_TWENTY_ONE
        assert inspect.signature(resolve_verdict).parameters["expected"].default == THE_TWENTY_ONE
        assert inspect.signature(run_rules).parameters["rules"].default is engine.RULES
        judged = decide(make_proposal(), make_context())
        assert tuple(r.name for r in judged.results) == THE_TWENTY_ONE
        # one name fewer, one more, or two swapped: no longer the twenty-one
        results = verdict_results(THE_TWENTY_ONE)
        assert resolve_verdict(results) == (Verdict.AUTO_EXECUTE, None)
        assert resolve_verdict(results[:-1]) == (Verdict.REJECT, ENGINE_INTEGRITY)

    def test_an_integrity_reject_is_recorded_and_explained(self, conn, monkeypatch):
        proposal = stored(conn)
        monkeypatch.setattr(engine, "RULES", RULES[:-1])
        decision = evaluate(proposal, make_context(), conn)
        assert decision.verdict is Verdict.REJECT
        assert decision.failing_rule == "engine_integrity"
        assert decision.notes == (
            "REJECT by engine_integrity (20 rule results do not match the 21 registered rules)"
        )
        assert len(decision.rules_evaluated) == 20
        (event,) = events(conn)
        assert event.kind == "policy_reject" and event.level is EventLevel.WARNING
        assert event.payload["failing_rule"] == "engine_integrity"
        assert get_proposal_trace(conn, proposal.id).decision == decision

    def test_a_failed_rule_is_recorded_with_its_failure(self, conn, monkeypatch):
        def buying_power(proposal, context):
            raise ZeroDivisionError("division by zero")

        proposal = stored(conn)
        monkeypatch.setattr(engine, "RULES", _replacing("buying_power", buying_power))
        decision = evaluate(proposal, make_context(), conn)
        assert decision.verdict is Verdict.REJECT and decision.failing_rule == "buying_power"
        assert len(decision.rules_evaluated) == 21
        assert decision.rules_evaluated[RULE_NAMES.index("buying_power")] == {
            "rule": "buying_power",
            "outcome": "REJECT",
            "detail": "rule failed: ZeroDivisionError: division by zero",
        }


# --- the randomized sweep -----------------------------------------------------

SEED = 20260930
CASES = 600
MISLABELLED_ONLY = {
    "options_max_loss": REJECT, "options_escalate": ESCALATE, "auto_tier": ESCALATE,
}
"""What does not pass when an "equity" that is really option-like is otherwise
a perfect auto-tier order: the three rules that judge its shape, and no other."""


def _sweep() -> list[tuple[ProposalUnderReview, PolicyContext]]:
    rng = random.Random(SEED)
    return [random_case(rng, index) for index in range(CASES)]


# The invariants below are worked out from the INPUTS alone — the proposal's
# and the context's own fields — never from a rule's outcome or a measure in
# aegis.policy. A rule that stopped holding (for closing sales, say) would
# still report an outcome consistent with the verdict; it could not make
# these agree.

OCC_SHAPE = re.compile(r"^[A-Z]{1,6} *\d{6}[CP]\d{8}$")
"""An OCC option symbol, compact or padded — spelled out here, not asked of the measures."""

KILL_SWITCH_ON = "the kill switch is on"
HALTED = "a halt is in force, or cannot be read"
NO_OPEN_MARKET = "the clock is missing, closed or stale"
PNL_UNKNOWN = "today's P&L or the start-of-day equity is unknown"
LOSS_CAP_REACHED = "today's P&L is at or below the loss cap"
TRADE_CAP_REACHED = "the orders sent today are at the cap"
MARKET_ORDER = "a market order, and market orders are not allowed"
BLANK_INVALIDATION = "the invalidation is blank"
NO_TRADE_LISTED = "the symbol is on the no-trade list"
MISSING_QUOTE = "a quote the price check needs is not there"
UNDATED_QUOTE = "a quote the price check needs has no timestamp"
STALE_QUOTE = "a quote the price check needs is older than its limit"
QUOTE_STOPS = (MISSING_QUOTE, UNDATED_QUOTE, STALE_QUOTE)
ACCOUNT_LEVEL = (
    KILL_SWITCH_ON, HALTED, NO_OPEN_MARKET, PNL_UNKNOWN, LOSS_CAP_REACHED, TRADE_CAP_REACHED,
)
"""The stops that hold whatever the proposal is — a closing sale included."""


def _same(left: str, right: str) -> bool:
    return left.strip().upper() == right.strip().upper()


def _quote_stops(proposal: ProposalUnderReview, context: PolicyContext) -> list[str]:
    """Why the quotes the limit is judged against cannot be trusted, read off
    the inputs: the declared equity's own quote, or each leg's (an option
    with no legs has none to show), dated by ``quote_time`` and measured to
    ``context.now`` against the instrument's ``max_quote_age_seconds``."""
    order, ages = proposal.proposal, context.limits.max_quote_age_seconds
    if order.instrument is Instrument.EQUITY:
        own = context.quote
        quotes = [own if own is not None and _same(own.symbol, order.symbol) else None]
        limit = ages.equity
    else:
        quotes = [
            next((q for q in context.leg_quotes if _same(q.symbol, leg.symbol)), None)
            for leg in proposal.legs
        ] or [None]
        limit = ages.option
    reasons = set()
    for quote in quotes:
        if quote is None:
            reasons.add(MISSING_QUOTE)
        elif quote.quote_time is None:
            reasons.add(UNDATED_QUOTE)
        else:
            age = (context.now - quote.quote_time).total_seconds()
            if age - limit > 1e-9 * max(1.0, abs(age), abs(limit)):
                reasons.add(STALE_QUOTE)
    return [reason for reason in QUOTE_STOPS if reason in reasons]


def _must_reject(proposal: ProposalUnderReview, context: PolicyContext) -> list[str]:
    """Every reason, read off the inputs, why this pair may only be REJECTed."""
    order, limits, controls, clock = (
        proposal.proposal, context.limits, context.controls, context.clock,
    )
    reasons = []
    if controls.kill_switch:
        reasons.append(KILL_SWITCH_ON)
    halt = controls.halt_until
    if controls.halt_unknown or (halt is not None and halt > context.now):
        reasons.append(HALTED)
    if clock is None or not clock.is_open:
        reasons.append(NO_OPEN_MARKET)
    elif clock.next_close is not None and context.now >= clock.next_close:
        reasons.append(NO_OPEN_MARKET)
    pnl, start = context.daily_pnl, context.start_of_day_equity
    if pnl is None or start is None or not math.isfinite(pnl) or not math.isfinite(start):
        reasons.append(PNL_UNKNOWN)
    elif start <= 0:
        reasons.append(PNL_UNKNOWN)  # no cap can be a percentage of nothing
    elif pnl <= -(limits.daily_loss_limit_pct / 100 * start):
        reasons.append(LOSS_CAP_REACHED)
    if context.orders_today >= limits.max_daily_trades:
        reasons.append(TRADE_CAP_REACHED)
    if order.order_type is OrderType.MARKET and not limits.allow_market_orders:
        reasons.append(MARKET_ORDER)
    if not order.invalidation.strip():
        reasons.append(BLANK_INVALIDATION)
    if order.symbol in limits.no_trade_list:
        reasons.append(NO_TRADE_LISTED)
    reasons.extend(_quote_stops(proposal, context))
    return reasons


def _closes_a_long(proposal: ProposalUnderReview, context: PolicyContext) -> bool:
    """An equity SELL of no more than the account holds long in that very
    symbol — every row the account reports for it being a plain long."""
    order = proposal.proposal
    if order.instrument is not Instrument.EQUITY or order.side is not OrderSide.SELL:
        return False
    if context.positions is None:
        return False
    rows = [position for position in context.positions if position.symbol == order.symbol]
    if not rows:
        return False
    if any(row.side != "long" or not math.isfinite(row.qty) or row.qty <= 0 for row in rows):
        return False
    return order.quantity <= sum(row.qty for row in rows)


NOT_EQUITY = "the instrument is not equity"
HAS_LEGS = "the proposal carries option legs"
OPTION_SYMBOL = "the symbol is an option contract's"
SPACED = "the symbol has whitespace in it"
TIER_OFF = "the auto tier is disabled"
TIER_EMPTY = "the auto tier admits nothing"
NOT_A_LIMIT = "not a limit order"
OFF_WATCHLIST = "the symbol is not on the watchlist"
LOW_CONFIDENCE = "the confidence is below the floor"
NO_PRICE = "a limit order without a usable price"
OVER_THE_TIER = "the notional is over the auto tier"
NOT_CLOSING = "a sale with no long to close"
NOT_PLAIN_EQUITY = (NOT_EQUITY, HAS_LEGS, OPTION_SYMBOL, SPACED)


def _never_auto_execute(proposal: ProposalUnderReview, context: PolicyContext) -> list[str]:
    """Every reason, read off the inputs, why this pair may not AUTO_EXECUTE."""
    order, limits = proposal.proposal, context.limits
    reasons = []
    if order.instrument is not Instrument.EQUITY:
        reasons.append(NOT_EQUITY)
    if proposal.legs:
        reasons.append(HAS_LEGS)
    if OCC_SHAPE.match(order.symbol.strip().upper()):
        reasons.append(OPTION_SYMBOL)
    if any(character.isspace() for character in order.symbol):
        reasons.append(SPACED)
    if not limits.auto_execute.enabled:
        reasons.append(TIER_OFF)
    if limits.auto_execute.max_notional <= 0:
        reasons.append(TIER_EMPTY)
    if order.order_type is not OrderType.LIMIT:
        reasons.append(NOT_A_LIMIT)
    elif order.limit_price is None or not order.limit_price > 0:
        reasons.append(NO_PRICE)
    elif order.quantity * order.limit_price > limits.auto_execute.max_notional * 1.000001 + 0.01:
        reasons.append(OVER_THE_TIER)  # clearly over: a cent and a millionth past it
    if order.symbol not in context.watchlist:
        reasons.append(OFF_WATCHLIST)
    if order.confidence < limits.min_confidence:
        reasons.append(LOW_CONFIDENCE)
    if order.side is OrderSide.SELL and not _closes_a_long(proposal, context):
        reasons.append(NOT_CLOSING)
    return reasons


class TestRandomized:
    def test_the_generator_is_seeded_and_varied(self):
        cases = _sweep()
        assert cases == _sweep()  # the same seed, the same cases
        assert len(cases) == CASES >= 500
        assert len({proposal.id for proposal, _ in cases}) == CASES
        shapes = Counter(
            (p.proposal.instrument.value, p.proposal.side.value, p.proposal.order_type.value)
            for p, _ in cases
        )
        assert len(shapes) == 8  # equity/option x buy/sell x market/limit
        assert sum(1 for p, _ in cases if p.is_option and not p.legs) >= 5
        assert sum(1 for p, _ in cases if len(p.underlyings) > 1) >= 20
        assert sum(1 for _, c in cases if c.clock is None) >= 10
        assert sum(1 for _, c in cases if c.clock is not None and not c.clock.is_open) >= 10
        assert sum(1 for _, c in cases if c.controls.kill_switch) >= 10
        assert sum(1 for _, c in cases if c.positions is None) >= 10
        assert sum(1 for _, c in cases if c.positions) >= 30
        assert sum(1 for _, c in cases if c.open_orders) >= 30
        assert sum(1 for p, c in cases if p.is_equity and c.quote is None) >= 5
        assert sum(1 for _, c in cases if c.limits != make_limits()) >= 200
        # Every limit is one RiskLimits accepts — finite — and the sweep still
        # reaches the extremes it accepts: zero, the largest finite float, and
        # whole numbers no float can hold.
        limits = [c.limits for _, c in cases]
        for name in ("halt_fallback_hours", "limit_price_tolerance_pct", "max_loss_per_trade"):
            values = [getattr(drawn, name) for drawn in limits]
            assert all(math.isfinite(value) for value in values), name
            assert sum(1 for value in values if value == LARGEST_FINITE) >= 5, name
        tiers = [drawn.auto_execute.max_notional for drawn in limits]
        assert all(math.isfinite(value) for value in tiers)
        assert sum(1 for value in tiers if value == LARGEST_FINITE) >= 5
        assert sum(1 for value in tiers if value == 0.0) >= 5
        for name in ("max_daily_trades", "max_open_positions", "max_contracts", "min_dte"):
            values = [getattr(drawn, name) for drawn in limits]
            assert sum(1 for value in values if value == 0) >= 5, name
            assert sum(1 for value in values if value > LARGEST_FINITE) >= 5, name
        # The shapes that contradict themselves, each of them.
        problems = [" | ".join(structure_problems(p)) for p, _ in cases]
        assert sum(1 for p, _ in cases if p.is_equity and p.legs) >= 10
        assert sum(1 for p, _ in cases if p.is_equity and is_occ(p.proposal.symbol)) >= 10
        for finding in (
            " names the ",  # a leg whose symbol is another contract
            " is not an OCC option symbol",
            ", which is not its leg's ",  # an OCC proposal symbol that is not its only leg's
            ": a multi-leg structure is named by its underlying",
            "an equity proposal carries ",
            "is declared equity but its symbol ",
        ):
            assert sum(1 for text in problems if finding in text) >= 10, finding
        sound = [p for (p, _), text in zip(cases, problems) if p.is_option and not text]
        assert len(sound) >= 100
        for figure in ("equity", "buying_power", "start_of_day_equity", "daily_pnl"):
            values = [getattr(c, figure) for _, c in cases]
            assert any(value is None for value in values)
            assert any(value is not None and value != value for value in values)  # NaN
            assert any(value in (float("inf"), float("-inf")) for value in values)
            assert any(value is not None and value < 0 for value in values)

    def test_every_case_gets_exactly_one_verdict_consistent_with_its_rules(self):
        verdicts: Counter = Counter()
        failing: Counter = Counter()
        shapes: Counter = Counter()  # broken shapes seen, by "declared equity?"
        stopped_by_shape_alone = 0
        outcomes_seen: dict[str, set] = {name: set() for name in RULE_NAMES}
        must_reject: Counter = Counter()  # how often each independent stop applied
        never_auto: Counter = Counter()  # ... and each bar to the auto tier
        closing_sales: Counter = Counter()  # closing sales, by verdict
        closing_stops: Counter = Counter()  # the stops closing sales met
        closing_at_market = 0  # closing sales whose ONLY stop was the market order ...
        stopped_by_the_order_type_alone = 0  # ... and that no other rule rejected either
        dust_sales: Counter = Counter()  # sales of a billionth of a share or less, by verdict
        for proposal, context in _sweep():
            evaluation = decide(proposal, context)  # nothing raises

            # exactly one verdict, out of the four
            assert isinstance(evaluation.verdict, Verdict)
            assert evaluation.verdict in (
                Verdict.REJECT, Verdict.FLAG_ONLY, Verdict.NEEDS_APPROVAL, Verdict.AUTO_EXECUTE,
            )
            # all twenty-one rules, in order
            assert len(evaluation.results) == 21
            assert tuple(result.name for result in evaluation.results) == RULE_NAMES
            outcomes = [result.outcome for result in evaluation.results]
            assert all(isinstance(outcome, RuleOutcome) for outcome in outcomes)
            # the verdict is the precedence of the outcomes
            assert evaluation.verdict is expected_verdict(outcomes)
            # the failing rule is the first REJECT, or None
            rejecting = [r.name for r in evaluation.results if r.outcome is REJECT]
            assert evaluation.failing_rule == (rejecting[0] if rejecting else None)
            assert (evaluation.failing_rule is not None) == (evaluation.verdict is Verdict.REJECT)
            # AUTO_EXECUTE never appears beside a rule that did not pass
            if evaluation.verdict is Verdict.AUTO_EXECUTE:
                assert set(outcomes) == {PASS}
                assert evaluation.non_pass == ()
                assert proposal.is_equity
                assert is_plain_equity(proposal) and structure_problems(proposal) == ()
            if set(outcomes) == {PASS}:
                assert evaluation.verdict is Verdict.AUTO_EXECUTE
            if proposal.is_option:
                assert evaluation.verdict is not Verdict.AUTO_EXECUTE
            # a shape that contradicts itself is rejected, whatever it is declared ...
            if structure_problems(proposal):
                assert result_of(evaluation, "options_max_loss").outcome is REJECT
                assert evaluation.verdict is Verdict.REJECT
                shapes[proposal.is_equity] += 1
            # ... and anything option-like needs a human at the very least
            if not is_plain_equity(proposal):
                assert result_of(evaluation, "options_escalate").outcome is ESCALATE
                assert result_of(evaluation, "auto_tier").outcome is ESCALATE
                assert evaluation.verdict is not Verdict.AUTO_EXECUTE
            if found(evaluation) == MISLABELLED_ONLY and proposal.is_equity:
                stopped_by_shape_alone += 1
            # deterministic, and anchored to the context's clock
            assert decide(proposal, context) == evaluation
            assert evaluation.decided_at == context.now
            assert evaluation.proposal_id == proposal.id
            # a halt only ever comes from a tripped daily loss limit
            if evaluation.halt_until is not None:
                assert result_of(evaluation, "daily_loss_limit").halt_until is not None
                assert evaluation.verdict is Verdict.REJECT
            for result in evaluation.results:
                assert result.detail.strip() and "\n" not in result.detail
                outcomes_seen[result.name].add(result.outcome)

            # --- the independent invariants: the inputs alone say what the
            # verdict may be, whatever each rule reported ---
            stops = _must_reject(proposal, context)
            if stops:
                assert evaluation.verdict is Verdict.REJECT, (proposal.id, stops)
            if KILL_SWITCH_ON in stops:
                assert evaluation.failing_rule == "kill_switch"  # rule 1: nothing precedes it
            barred = _never_auto_execute(proposal, context)
            if barred:
                assert evaluation.verdict is not Verdict.AUTO_EXECUTE, (proposal.id, barred)
            if LOW_CONFIDENCE in barred:  # a FLAG: never past FLAG_ONLY
                assert evaluation.verdict in (Verdict.REJECT, Verdict.FLAG_ONLY), proposal.id
            if evaluation.verdict is Verdict.AUTO_EXECUTE:
                assert stops == [] and barred == []
                order = proposal.proposal
                assert order.side is OrderSide.BUY or _closes_a_long(proposal, context)
            must_reject.update(stops)
            never_auto.update(barred)
            if _closes_a_long(proposal, context):
                closing_sales[evaluation.verdict] += 1
                closing_stops.update(stops)
                if stops == [MARKET_ORDER]:
                    closing_at_market += 1
                    stopped_by_the_order_type_alone += rejecting == ["limit_price_sanity"]
            if proposal.proposal.side is OrderSide.SELL and proposal.is_equity:
                if proposal.proposal.quantity <= 1e-9:
                    dust_sales[evaluation.verdict] += 1

            verdicts[evaluation.verdict] += 1
            failing[evaluation.failing_rule] += 1

        # The generator really produced each of the four verdicts — and plenty of each.
        assert set(verdicts) == set(Verdict)
        assert sum(verdicts.values()) == CASES
        assert all(count >= 30 for count in verdicts.values()), verdicts
        # ... and every rule both passed and failed to pass somewhere in the sweep.
        for name, seen in outcomes_seen.items():
            assert PASS in seen and len(seen) >= 2, name
        assert len([name for name in failing if name is not None]) >= 12
        # Broken shapes of both instruments were judged — and some mislabelled
        # equities were, but for their shape, perfect auto-tier orders.
        assert shapes[True] >= 20 and shapes[False] >= 50, shapes
        assert stopped_by_shape_alone >= 5
        # Every independent invariant was put to the test, many times over ...
        for reason in (
            *ACCOUNT_LEVEL, MARKET_ORDER, BLANK_INVALIDATION, NO_TRADE_LISTED, MISSING_QUOTE,
        ):
            assert must_reject[reason] >= 10, (reason, must_reject)
        # The base sweep stamps every quote it builds at NOW: only the aged
        # sweep below puts a stale or undated quote in front of the engine.
        assert must_reject[STALE_QUOTE] == must_reject[UNDATED_QUOTE] == 0
        for reason in (
            *NOT_PLAIN_EQUITY, TIER_OFF, TIER_EMPTY, NOT_A_LIMIT, OFF_WATCHLIST, LOW_CONFIDENCE,
            NO_PRICE, OVER_THE_TIER, NOT_CLOSING,
        ):
            assert never_auto[reason] >= 10, (reason, never_auto)
        # ... and closing sales — the one sell that can auto-execute — met
        # every account-level stop, were rejected each time, and auto-executed
        # only where nothing stood in the way.
        assert sum(closing_sales.values()) >= 40, closing_sales
        assert closing_sales[Verdict.AUTO_EXECUTE] >= 5 and closing_sales[Verdict.REJECT] >= 20
        for reason in ACCOUNT_LEVEL:
            assert closing_stops[reason] >= 3, (reason, closing_stops)
        # "A market order is rejected while market orders are not allowed" was
        # decisive for a SELL too: closing sales at market with no other stop
        # in the way, several of which no other rule rejected — so an
        # exemption for sells, or for closing sales, cannot hide behind
        # another rule's REJECT.
        assert closing_stops[MARKET_ORDER] >= 10, closing_stops
        assert closing_at_market >= 5 and stopped_by_the_order_type_alone >= 3, (
            closing_at_market, stopped_by_the_order_type_alone,
        )
        # A sale of a billionth of a share or less is judged like any sale.
        assert sum(dust_sales.values()) >= 15, dust_sales
        assert dust_sales[Verdict.AUTO_EXECUTE] <= closing_sales[Verdict.AUTO_EXECUTE]

    def test_every_case_can_be_recorded_and_read_back(self, conn, calls):
        cases = _sweep()
        halts = 0
        expected_events = 0
        for proposal, context in cases:
            stored(conn, proposal)
            # Each case meets a store with no stop of its own: the halt the
            # case before may have left is cleared the way an operator clears
            # one, so the store adds nothing and what is recorded is the
            # context's own verdict. (The next test leaves the halts in.)
            repo.set_halt_until(conn, None, now=NOW)
            evaluation = decide(proposal, context)
            dry = evaluate(proposal, context, Untouchable(), dry_run=True)
            assert dry.verdict is evaluation.verdict
            decision = evaluate(proposal, context, conn)
            assert decision.model_dump(exclude={"id"}) == dry.model_dump(exclude={"id"})
            assert decision.verdict is evaluation.verdict
            assert decision.failing_rule == evaluation.failing_rule
            assert decision.rules_evaluated == evaluation.rules_evaluated
            assert len(decision.rules_evaluated) == 21
            halts += evaluation.halt_until is not None
            expected_events += evaluation.verdict is not Verdict.AUTO_EXECUTE
        # record_decision exactly once per evaluation; the halt only when one was named
        assert names_of(calls).count("record_decision") == CASES
        assert names_of(calls).count("set_halt_until") == halts > 0
        assert len(decisions(conn)) == CASES
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == halts
        assert len(events(conn)) == halts + expected_events
        for proposal, context in cases[::25]:
            trace = get_proposal_trace(conn, proposal.id)
            assert len(trace.decisions) == 1
            assert trace.decision.verdict is decide(proposal, context).verdict

    def test_every_case_recorded_on_a_store_that_keeps_its_halts(self, conn, calls):
        """The same sweep with nothing cleared in between, as in the loop:
        each trip's halt stays in the store for the cases after it. Every
        decision is ``decide`` under the stricter of the context's and the
        store's stop flags; the stored halt only ever moves later, and each
        move is announced exactly once."""
        extended = 0
        asked = 0  # trips that named a halt: each one asks the store, which decides
        rejected_by_the_stored_halt = 0
        for proposal, context in _sweep():
            stored(conn, proposal)
            before = get_controls(conn)
            under = context.model_copy(
                update={"controls": engine.stricter_controls(context.controls, before)}
            )
            expected = decide(proposal, under)

            decision = evaluate(proposal, context, conn)

            assert decision.verdict is expected.verdict
            assert decision.failing_rule == expected.failing_rule
            assert decision.rules_evaluated == expected.rules_evaluated
            # Independent of the rules: a stored halt in force means REJECT.
            if before.halt_until is not None and before.halt_until > context.now:
                assert decision.verdict is Verdict.REJECT
                rejected_by_the_stored_halt += decide(proposal, context).verdict is not (
                    Verdict.REJECT
                )
            after = get_controls(conn)
            assert after.kill_switch is False and after.halt_unknown is False
            if before.halt_until is not None:
                assert after.halt_until is not None and after.halt_until >= before.halt_until
            if after.halt_until != before.halt_until:
                extended += 1
                assert after.halt_until == expected.halt_until
                assert after.halt_until_updated_at == context.now
            else:
                assert after.halt_until_updated_at == before.halt_until_updated_at
            if expected.halt_until is not None:
                asked += 1
                if after.halt_until == before.halt_until:
                    # Asked and not written: only a context whose own reading
                    # of the halt was unreadable asks for one the store —
                    # readable — already holds.
                    assert under.controls.halt_unknown
                    assert before.halt_until == expected.halt_until
        assert extended >= 1
        assert names_of(calls).count("set_halt_until") == asked >= extended
        assert len(events(conn, RISK_LIMIT_TRIPPED)) == extended
        assert names_of(calls).count("record_decision") == CASES == len(decisions(conn))
        # Proposals their own context would have let through were stopped by the halt.
        assert rejected_by_the_stored_halt >= 50


# --- the randomized sweep, with the quotes aged -------------------------------
#
# Every case of the sweep above, judged again with the timestamps of the
# quotes its price check reads moved — under limits drawn per case — once
# for each way a quote can be dated. Drawn from its own seed, so the base
# sweep's cases stay exactly as they were.

AGE_SEED = 20261002
AGE_LIMITS: tuple[dict[str, float] | None, ...] = (
    None,  # the defaults: 120 s for an equity, 1,200 s for an option
    None,
    {"equity": 0.0, "option": 0.0},
    {"equity": 30.0, "option": 1e9},
    {"equity": 1e9, "option": 30.0},
    {"equity": 900.0, "option": 900.0},
)
"""``max_quote_age_seconds`` per case: the defaults twice as often as any other."""
OVER_BY: tuple[float, ...] = (0.001, 1.0, 60.0, 86_400.0)
"""How far past its limit a stale quote is dated — at least a millionth of
the limit, so it is past the float-noise guard (a part in a billion) too."""
AGINGS = ("stale", "one stale", "undated", "at the limit", "ahead")
"""Every relevant quote past its limit; one of them past it, the rest inside;
one of them with no timestamp; every one exactly at its limit; every one
dated after the as-of time (fetched after the context's clock was read)."""
FRESH_AGINGS = ("at the limit", "ahead")


def _restamped(quote, context: PolicyContext, age: float | None):
    stamp = None if age is None else context.now - timedelta(seconds=age)
    return quote.model_copy(update={"quote_time": stamp})


def _aged(
    rng: random.Random, proposal: ProposalUnderReview, context: PolicyContext, aging: str
) -> PolicyContext:
    """``context`` with the quotes the price check reads re-dated by ``aging``."""
    ages = context.limits.max_quote_age_seconds
    if proposal.is_equity:
        limit = ages.equity
        positions = [0] if context.quote is not None else []
    else:
        limit = ages.option
        positions = list(range(len(context.leg_quotes)))
    if not positions:
        return context
    chosen = rng.choice(positions)
    by_position = {}
    for position in positions:
        if aging == "stale" or (aging == "one stale" and position == chosen):
            by_position[position] = limit + max(rng.choice(OVER_BY), limit * 1e-6)
        elif aging == "undated" and position == chosen:
            by_position[position] = None
        elif aging == "at the limit":
            by_position[position] = limit
        elif aging == "ahead":
            by_position[position] = -rng.choice((0.5, 2.0, 30.0))
        else:
            by_position[position] = rng.uniform(0.0, limit)
    if proposal.is_equity:
        return context.model_copy(
            update={"quote": _restamped(context.quote, context, by_position[0])}
        )
    quotes = tuple(
        _restamped(quote, context, by_position[position])
        for position, quote in enumerate(context.leg_quotes)
    )
    return context.model_copy(update={"leg_quotes": quotes})


def _aged_sweep():
    """``(proposal, context with its drawn limits, aging, aged context)``
    for every case of the base sweep and every aging."""
    rng = random.Random(AGE_SEED)
    cases = []
    for proposal, context in _sweep():
        drawn = rng.choice(AGE_LIMITS)
        if drawn is not None:
            limits = context.limits.model_copy(
                update={"max_quote_age_seconds": QuoteAgeLimits(**drawn)}
            )
            context = context.model_copy(update={"limits": limits})
        for aging in AGINGS:
            cases.append((proposal, context, aging, _aged(rng, proposal, context, aging)))
    return cases


class TestRandomizedQuoteAges:
    def test_the_aged_sweep_is_seeded_and_reaches_every_aging(self):
        cases = _aged_sweep()
        assert cases == _aged_sweep()
        assert len(cases) == CASES * len(AGINGS)
        stops = Counter(
            (aging, reason)
            for proposal, _, aging, aged in cases
            for reason in _quote_stops(proposal, aged)
        )
        assert stops[("stale", STALE_QUOTE)] >= 400
        assert stops[("one stale", STALE_QUOTE)] >= 400
        assert stops[("undated", UNDATED_QUOTE)] >= 400
        for aging in FRESH_AGINGS:
            assert stops[(aging, STALE_QUOTE)] == stops[(aging, UNDATED_QUOTE)] == 0
        # one stale leg among fresh ones, many times over
        mixed = [
            aged
            for proposal, _, aging, aged in cases
            if aging == "one stale" and proposal.is_option and len(aged.leg_quotes) >= 2
        ]
        assert len(mixed) >= 50
        # both instruments under every limit drawn, the defaults included
        drawn = Counter(
            (proposal.is_equity, aged.limits.max_quote_age_seconds)
            for proposal, _, _, aged in cases
        )
        assert len(drawn) == 2 * (len(AGE_LIMITS) - 1)

    def test_auto_execute_never_appears_beside_a_stale_or_undated_quote(self):
        verdicts: Counter = Counter()
        turned = Counter()  # base AUTO_EXECUTE cases, by aging, and what became of them
        for proposal, context, aging, aged in _aged_sweep():
            evaluation = decide(proposal, aged)
            reasons = _quote_stops(proposal, aged)
            freshness = result_of(evaluation, "quote_freshness")
            if reasons:
                assert evaluation.verdict is Verdict.REJECT, (proposal.id, aging, reasons)
                assert freshness.outcome is REJECT, (proposal.id, aging, freshness.detail)
            else:
                assert freshness.outcome is PASS, (proposal.id, aging, freshness.detail)
            if evaluation.verdict is Verdict.AUTO_EXECUTE:
                assert reasons == [] and aging in FRESH_AGINGS
            verdicts[aging, evaluation.verdict] += 1
            # A quote's date moves no other rule: rules 1-12 read no quote
            # time, so where the base case auto-executed, a stale or undated
            # quote is the FIRST rejection — and a fresh one changes nothing.
            base = decide(proposal, context)
            if aging in FRESH_AGINGS:
                assert evaluation.verdict is base.verdict, (proposal.id, aging)
                assert evaluation.failing_rule == base.failing_rule
            if base.verdict is Verdict.AUTO_EXECUTE:
                if aging in FRESH_AGINGS:
                    turned[aging, evaluation.verdict] += 1
                else:
                    assert evaluation.failing_rule == "quote_freshness", (proposal.id, aging)
                    assert {r.name for r in evaluation.non_pass} == {"quote_freshness"}
                    turned[aging, evaluation.verdict] += 1
        for aging in ("stale", "one stale", "undated"):
            assert verdicts[aging, Verdict.AUTO_EXECUTE] == 0
            assert turned[aging, Verdict.REJECT] >= 20, turned
        for aging in FRESH_AGINGS:
            assert verdicts[aging, Verdict.AUTO_EXECUTE] >= 20, verdicts
            assert turned[aging, Verdict.AUTO_EXECUTE] >= 20, turned


# --- the engine module itself -------------------------------------------------

ENGINE_SOURCE = REPO_ROOT / "aegis" / "policy" / "engine.py"
ALLOWED_IMPORTS = {
    "__future__", "sqlite3", "collections.abc", "datetime", "typing",
    "aegis.policy.measures", "aegis.policy.rules", "aegis.policy.models",
    "aegis.store.models", "aegis.store.repo",
}


def _engine_tree() -> ast.Module:
    return ast.parse(ENGINE_SOURCE.read_text(encoding="utf-8"), filename="engine.py")


class TestEngineModule:
    def test_imports_stay_inside_the_allowlist(self):
        imported = set()
        for node in ast.walk(_engine_tree()):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                imported.add(node.module)
        assert imported <= ALLOWED_IMPORTS
        assert {"aegis.policy.rules", "aegis.store.repo"} <= imported

    def test_the_store_writers_are_imported_by_name_at_module_level(self):
        (statement,) = [
            node
            for node in _engine_tree().body
            if isinstance(node, ast.ImportFrom) and node.module == "aegis.store.repo"
        ]
        # the two writers, and the one read evaluate makes: the stop flags, once more
        assert sorted(alias.name for alias in statement.names) == [
            "get_controls", "record_decision", "set_halt_until",
        ]

    def test_no_clock_is_read(self):
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(_engine_tree())
            if isinstance(node, ast.Call)
        }
        assert called.isdisjoint({"utcnow", "now", "today", "time", "monotonic"})

    def test_no_risk_number_is_hardcoded(self):
        numbers = {
            node.value
            for node in ast.walk(_engine_tree())
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
        }
        assert numbers <= {0, 1, 100}

    def test_the_public_functions_are_documented(self):
        for function in (run_rules, resolve_verdict, decide, evaluate):
            assert function.__doc__ and function.__doc__.strip()

    def test_the_event_kinds(self):
        assert RISK_LIMIT_TRIPPED == "risk_limit_tripped"
        assert VERDICT_EVENT_KINDS == {
            Verdict.REJECT: "policy_reject",
            Verdict.FLAG_ONLY: "policy_flag_only",
            Verdict.NEEDS_APPROVAL: "policy_needs_approval",
        }

    def test_the_default_limit_price_is_the_default_quotes_mid(self):
        # What the evil cases above lean on: the default context prices AAPL at 200.00.
        assert make_context().quote.mid == pytest.approx(LIMIT_PRICE)
        assert make_proposal().proposal.created_at == PROPOSED_AT
