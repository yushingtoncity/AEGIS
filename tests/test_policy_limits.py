"""The limits report — contexts built by hand, no network, no store.

``limits_report`` is what ``python -m aegis.cli.policy limits`` prints and
what the Phase 8 dashboard will read. Each line's numbers are checked
against values worked out by hand, ``blocked_by`` against each account-level
stop, and every unknown against "None, never zero". ``TestNoDrift`` then
sweeps hundreds of seeded random contexts and holds the report to the rules:
a limit the report shows as breached is one its rule rejects on, and a
headroom the report shows is the room the rule really leaves.
"""

import ast
import math
import random
import sys
from datetime import date, datetime, timedelta, timezone

import pytest
from policy_factories import (
    BUYING_POWER,
    EQUITY,
    MARKET_DATE,
    NEXT_CLOSE,
    NEXT_OPEN,
    NOTIONAL,
    NOW,
    OPTIONS_BUYING_POWER,
    RANDOM_TURBULENCE,
    WATCHLIST,
    long_call,
    make_clock,
    make_context,
    make_limits,
    make_order,
    make_position,
    make_proposal,
    occ,
    random_context,
)

from aegis.config import REPO_ROOT
from aegis.policy.limits import limits_report
from aegis.policy.measures import exceeds, held_underlyings, known, money, notional
from aegis.policy.models import (
    ExposureLine,
    LimitLine,
    LimitsReport,
    PolicyContext,
    RuleOutcome,
)
from aegis.policy.rules import (
    account_rules,
    buying_power,
    max_open_positions,
    max_position_pct,
)
from aegis.store.models import Controls

REJECT = RuleOutcome.REJECT
NAN = float("nan")
INF = float("inf")
LARGEST_FINITE = sys.float_info.max
"""The largest number ``RiskLimits`` accepts for a limit: infinity and NaN are refused at load."""
UNKNOWN = pytest.mark.parametrize(
    "unknown", [None, NAN, INF, -INF], ids=["none", "nan", "inf", "-inf"]
)

LINE_NAMES = (
    "daily_loss", "open_positions", "daily_trades", "buying_power", "options_buying_power",
)
ACCOUNT_RULE_NAMES = (
    "kill_switch", "halted", "market_hours", "daily_loss_limit", "max_daily_trades",
)

CALL = occ("call", 640.0)
"""An SPY option contract, for positions and orders on SPY that are not SPY stock."""


def line(report: LimitsReport, name: str) -> LimitLine:
    (found,) = [candidate for candidate in report.lines if candidate.name == name]
    return found


def report_line(name: str, **overrides) -> LimitLine:
    """The named line of the report on ``make_context(**overrides)``."""
    return line(limits_report(make_context(**overrides)), name)


def numbers(limit_line: LimitLine) -> tuple[float | None, float | None, float | None, bool]:
    return (limit_line.limit, limit_line.used, limit_line.headroom, limit_line.breached)


def exposure_of(report: LimitsReport, underlying: str) -> ExposureLine:
    (found,) = [candidate for candidate in report.exposures if candidate.underlying == underlying]
    return found


# --- the report as a whole ----------------------------------------------------


class TestReport:
    def test_the_default_context_is_a_healthy_account(self):
        report = limits_report(make_context())
        assert isinstance(report, LimitsReport)
        assert report.as_of == NOW
        assert report.market_open is True
        assert (report.next_open, report.next_close) == (NEXT_OPEN, NEXT_CLOSE)
        assert report.kill_switch is False
        assert report.halt_until is None and report.halted is False
        assert report.blocked_by == ()
        assert (report.equity, report.start_of_day_equity, report.daily_pnl) == (
            EQUITY, EQUITY, 0.0,
        )
        assert (report.buying_power, report.options_buying_power) == (
            BUYING_POWER, OPTIONS_BUYING_POWER,
        )
        assert report.exposures == ()
        assert report.auto_execute_enabled is True
        assert report.auto_execute_max_notional == 1_000.0
        assert report.errors == ()

    def test_the_five_lines_in_order_with_their_units(self):
        report = limits_report(make_context())
        assert tuple(found.name for found in report.lines) == LINE_NAMES
        assert [found.unit for found in report.lines] == [
            "USD", "positions", "trades", "USD", "USD",
        ]
        for found in report.lines:
            assert isinstance(found, LimitLine)
            assert found.detail.strip() and "\n" not in found.detail
            assert found.breached is False

    def test_the_default_lines_by_hand(self):
        report = limits_report(make_context())
        # 2% of 100,000 start-of-day equity; nothing lost; all of it left
        assert numbers(line(report, "daily_loss")) == (2_000.0, 0.0, 2_000.0, False)
        assert numbers(line(report, "open_positions")) == (5.0, 0.0, 5.0, False)
        assert numbers(line(report, "daily_trades")) == (10.0, 0.0, 10.0, False)
        assert numbers(line(report, "buying_power")) == (None, None, 200_000.0, False)
        assert numbers(line(report, "options_buying_power")) == (None, None, 100_000.0, False)

    def test_as_of_is_the_contexts_now_never_the_wall_clock(self):
        # An instant with seconds and microseconds: as_of is the context's
        # now to the last digit, not the minute it falls in.
        long_ago = datetime(2001, 9, 10, 14, 3, 42, 123456, tzinfo=timezone.utc)
        report = limits_report(make_context(now=long_ago, market_date=long_ago.date()))
        assert report.as_of == long_ago
        assert (report.as_of.second, report.as_of.microsecond) == (42, 123456)
        assert report.as_of.isoformat() == "2001-09-10T14:03:42.123456+00:00"

    @pytest.mark.parametrize(
        "now",
        [
            NOW + timedelta(seconds=42, microseconds=123_456),
            NOW + timedelta(microseconds=1),
            NOW - timedelta(microseconds=1),
            NOW.replace(second=59, microsecond=999_999),
        ],
        ids=["seconds-and-microseconds", "one-microsecond-on", "one-microsecond-back", "59.999999"],
    )
    def test_as_of_keeps_every_digit_of_the_instant(self, now):
        report = limits_report(make_context(now=now))
        assert report.as_of == now and report.as_of.isoformat() == now.isoformat()
        assert report.as_of.utcoffset() == timedelta(0)

    def test_pure_the_same_context_gives_an_equal_report_and_is_left_alone(self):
        def build() -> PolicyContext:
            return make_context(
                daily_pnl=-750.0,
                positions=(make_position(), make_position(CALL, 2.0, current_price=8.0)),
                open_orders=(make_order("QQQ"),),
                orders_today=4,
                controls=Controls(halt_until=NEXT_OPEN),
            )

        context = build()
        before = context.model_dump_json()
        first = limits_report(context)
        assert limits_report(context) == first
        assert limits_report(build()) == first
        assert context.model_dump_json() == before

    def test_the_auto_execute_tier_comes_from_the_limits(self):
        limits = make_limits(auto_execute={"enabled": False, "max_notional": 250.0})
        report = limits_report(make_context(limits=limits))
        assert report.auto_execute_enabled is False
        assert report.auto_execute_max_notional == 250.0

    def test_the_contexts_errors_are_passed_on(self):
        errors = ("account: data fetch failed", "clock: data fetch failed")
        assert limits_report(make_context(errors=errors)).errors == errors

    def test_the_report_needs_no_proposal_and_no_market_data(self):
        bare = make_context(quote=None, leg_quotes=(), analysis=None, recent_proposals=())
        assert limits_report(bare) == limits_report(make_context())


# --- the market and the controls ----------------------------------------------


class TestMarketAndControls:
    def test_an_open_market(self):
        report = limits_report(make_context())
        assert (report.market_open, report.next_open, report.next_close) == (
            True, NEXT_OPEN, NEXT_CLOSE,
        )

    def test_a_closed_market(self):
        opens = datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc)
        clock = make_clock(False, next_open=opens, next_close=NEXT_CLOSE)
        report = limits_report(make_context(clock=clock))
        assert (report.market_open, report.next_open, report.next_close) == (
            False, opens, NEXT_CLOSE,
        )
        assert report.blocked_by == ("market_hours",)

    def test_no_clock_is_unknown_not_closed(self):
        report = limits_report(make_context(clock=None))
        assert report.market_open is None
        assert report.next_open is None and report.next_close is None
        assert report.blocked_by == ("market_hours",)

    def test_a_clock_without_times(self):
        report = limits_report(make_context(clock=make_clock(next_open=None, next_close=None)))
        assert report.market_open is True
        assert report.next_open is None and report.next_close is None
        assert report.blocked_by == ()

    def test_a_stale_clock_still_says_what_it_says_but_blocks(self):
        report = limits_report(make_context(clock=make_clock(next_close=NOW)))
        assert report.market_open is True  # what the clock said ...
        assert report.blocked_by == ("market_hours",)  # ... and what the rule makes of it

    def test_the_kill_switch(self):
        assert limits_report(make_context()).kill_switch is False
        report = limits_report(make_context(controls=Controls(kill_switch=True)))
        assert report.kill_switch is True
        assert report.blocked_by == ("kill_switch",)

    @pytest.mark.parametrize(
        ("until", "halted"),
        [
            (None, False),
            (NOW + timedelta(seconds=1), True),
            (NEXT_OPEN, True),
            (NOW, False),
            (NOW - timedelta(days=1), False),
        ],
        ids=["no halt", "a second ahead", "until the next open", "ending now", "expired"],
    )
    def test_halted_is_exactly_the_halted_rules_condition(self, until, halted):
        report = limits_report(make_context(controls=Controls(halt_until=until)))
        assert report.halt_until == until  # shown as stored, in force or not
        assert report.halted is halted
        assert report.blocked_by == (("halted",) if halted else ())

    def test_an_unreadable_halt_counts_as_halted(self):
        controls = Controls(halt_unknown=True, problems=("halt_until is unreadable",))
        report = limits_report(make_context(controls=controls))
        assert report.halt_until is None
        assert report.halted is True
        assert report.blocked_by == ("halted",)


# --- blocked_by ---------------------------------------------------------------


class TestBlockedBy:
    def test_nothing_blocks_a_healthy_account(self):
        assert limits_report(make_context()).blocked_by == ()

    @pytest.mark.parametrize(
        ("overrides", "blocked"),
        [
            ({"controls": Controls(kill_switch=True)}, "kill_switch"),
            ({"controls": Controls(halt_until=NEXT_OPEN)}, "halted"),
            ({"controls": Controls(halt_unknown=True)}, "halted"),
            ({"clock": make_clock(False)}, "market_hours"),
            ({"clock": None}, "market_hours"),
            ({"daily_pnl": -2_000.0}, "daily_loss_limit"),
            ({"daily_pnl": None}, "daily_loss_limit"),
            ({"start_of_day_equity": None}, "daily_loss_limit"),
            ({"orders_today": 10}, "max_daily_trades"),
        ],
        ids=[
            "kill switch", "halt", "unreadable halt", "closed market", "no clock", "loss cap",
            "unknown P&L", "unknown start-of-day equity", "trade cap",
        ],
    )
    def test_each_account_level_stop_alone(self, overrides, blocked):
        assert limits_report(make_context(**overrides)).blocked_by == (blocked,)

    def test_several_stops_are_named_in_rule_order(self):
        context = make_context(
            orders_today=11,
            daily_pnl=-5_000.0,
            clock=make_clock(False),
            controls=Controls(kill_switch=True, halt_until=NEXT_OPEN),
        )
        assert limits_report(context).blocked_by == ACCOUNT_RULE_NAMES
        partial = make_context(orders_today=10, clock=None)
        assert limits_report(partial).blocked_by == ("market_hours", "max_daily_trades")

    def test_only_the_account_level_rules_can_block(self):
        # A full book, no buying power and an exposure over its cap reject
        # proposals, not the account: they show on their lines instead.
        context = make_context(
            buying_power=0.0,
            limits=make_limits(max_open_positions=1),
            positions=(make_position(qty=1_000.0),),
        )
        report = limits_report(context)
        assert report.blocked_by == ()
        assert line(report, "open_positions").breached is True
        assert exposure_of(report, "AAPL").breached is True

    def test_it_is_the_rejecting_account_rules_whatever_the_context(self):
        for overrides in (
            {},
            {"controls": Controls(kill_switch=True), "daily_pnl": NAN},
            {"clock": make_clock(next_close=NOW), "orders_today": 10**30},
            {"start_of_day_equity": 0.0, "controls": Controls(halt_until=NOW)},
        ):
            context = make_context(**overrides)
            rejecting = tuple(r.name for r in account_rules(context) if r.outcome is REJECT)
            assert limits_report(context).blocked_by == rejecting


# --- the daily loss line ------------------------------------------------------


class TestDailyLossLine:
    @pytest.mark.parametrize(
        ("pnl", "used", "headroom", "breached"),
        [
            (0.0, 0.0, 2_000.0, False),
            (750.0, 0.0, 2_750.0, False),  # a gain uses none of the cap and adds room
            (-500.0, 500.0, 1_500.0, False),
            (-1_999.99, 1_999.99, 0.01, False),
            (-2_000.0, 2_000.0, 0.0, True),  # at the cap is tripped
            (-2_000.01, 2_000.01, -0.01, True),
            (-2_500.0, 2_500.0, -500.0, True),
        ],
        ids=["flat", "gain", "loss", "a cent inside", "at the cap", "a cent over", "over"],
    )
    def test_the_numbers_by_hand(self, pnl, used, headroom, breached):
        found = report_line("daily_loss", daily_pnl=pnl)
        assert found.limit == 2_000.0  # 2% of 100,000
        assert found.used == pytest.approx(used)
        assert found.headroom == pytest.approx(headroom)
        assert found.breached is breached
        assert found.unit == "USD"

    def test_no_loss_is_zero_used_not_minus_zero(self):
        assert str(report_line("daily_loss", daily_pnl=0.0).used) == "0.0"

    def test_the_cap_is_a_percentage_of_start_of_day_equity_from_the_limits(self):
        # Today's equity is irrelevant: the cap is measured from where the day started.
        found = report_line(
            "daily_loss",
            limits=make_limits(daily_loss_limit_pct=1.5),
            start_of_day_equity=80_000.0,
            equity=50_000.0,
            daily_pnl=-300.0,
        )
        assert numbers(found) == (pytest.approx(1_200.0), 300.0, pytest.approx(900.0), False)

    def test_the_detail_is_the_rules_own(self):
        for pnl in (0.0, -500.0, -2_500.0, None):
            context = make_context(daily_pnl=pnl)
            (rule,) = [r for r in account_rules(context) if r.name == "daily_loss_limit"]
            assert line(limits_report(context), "daily_loss").detail == rule.detail
        assert "(risk_limits.daily_loss_limit_pct)" in report_line("daily_loss").detail

    @UNKNOWN
    def test_an_unknown_pnl(self, unknown):
        found = report_line("daily_loss", daily_pnl=unknown)
        assert numbers(found) == (2_000.0, None, None, False)  # the cap is still known
        assert "today's P&L is unknown" in found.detail

    @pytest.mark.parametrize("start", [None, NAN, INF, -INF, 0.0, -100.0], ids=str)
    def test_an_unknown_or_non_positive_start_of_day_equity(self, start):
        found = report_line("daily_loss", start_of_day_equity=start, daily_pnl=-500.0)
        assert numbers(found) == (None, 500.0, None, False)  # the loss so far is still known
        assert "start-of-day equity is unknown or not positive" in found.detail

    def test_both_unknown(self):
        found = report_line("daily_loss", start_of_day_equity=None, daily_pnl=None)
        assert numbers(found) == (None, None, None, False)
        assert found.detail.startswith("cannot evaluate the daily loss limit")

    def test_unknown_is_not_breached_but_it_blocks(self):
        report = limits_report(make_context(daily_pnl=None))
        assert line(report, "daily_loss").breached is False
        assert report.blocked_by == ("daily_loss_limit",)

    def test_a_trip_under_a_longer_halt_names_the_halt_the_report_shows(self):
        # The line's detail and the report's own halt_until must be the same
        # instant: a longer halt already on record is the one in force.
        in_force = NOW + timedelta(hours=24)  # later than tomorrow's 13:30 open
        report = limits_report(
            make_context(daily_pnl=-3_000.0, controls=Controls(halt_until=in_force))
        )
        assert report.halt_until == in_force and report.halted is True
        found = line(report, "daily_loss")
        assert found.breached is True
        assert found.detail.endswith(": a halt until 2026-07-31T15:00:00+00:00 is already in force")
        assert "2026-07-31T13:30:00" not in found.detail  # not the earlier next open
        assert report.blocked_by == ("halted", "daily_loss_limit")

    def test_a_trip_before_the_open_reports_the_session_close_as_the_halt(self):
        early = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)  # 08:00 in New York
        opens = datetime(2026, 7, 30, 13, 30, tzinfo=timezone.utc)
        context = make_context(
            now=early,
            clock=make_clock(False, next_open=opens, next_close=NEXT_CLOSE),
            daily_pnl=-2_500.0,
        )
        report = limits_report(context)
        assert report.blocked_by == ("market_hours", "daily_loss_limit")
        assert (report.market_open, report.next_open, report.next_close) == (
            False, opens, NEXT_CLOSE,
        )
        assert line(report, "daily_loss").detail.endswith(
            "trading is halted until 2026-07-30T20:00:00+00:00, the close of the session "
            "that opens at 2026-07-30T13:30:00+00:00"
        )
        assert report.halt_until is None and report.halted is False  # a report writes no halt


# --- the open positions line --------------------------------------------------


class TestOpenPositionsLine:
    def test_nothing_held(self):
        found = report_line("open_positions")
        assert numbers(found) == (5.0, 0.0, 5.0, False)
        assert found.detail == (
            "0 of 5 underlyings held or pending (risk_limits.max_open_positions)"
        )

    def test_distinct_underlyings_held_or_pending_by_hand(self):
        # AAPL stock, two SPY contracts and SPY stock (one underlying), a pending QQQ order.
        found = report_line(
            "open_positions",
            positions=(
                make_position("AAPL"),
                make_position(CALL, 1.0, current_price=8.0),
                make_position(occ("put", 630.0), 1.0, current_price=4.5),
                make_position("SPY"),
            ),
            open_orders=(make_order("QQQ"),),
        )
        assert numbers(found) == (5.0, 3.0, 2.0, False)
        assert found.detail == (
            "3 of 5 underlyings held or pending (risk_limits.max_open_positions): AAPL, QQQ, SPY"
        )
        assert found.unit == "positions"

    @pytest.mark.parametrize(
        ("held", "used", "headroom", "breached"),
        [(4, 4.0, 1.0, False), (5, 5.0, 0.0, True), (6, 6.0, -1.0, True)],
        ids=["one below", "at the cap", "over"],
    )
    def test_boundary_at_the_cap_is_breached(self, held, used, headroom, breached):
        symbols = ("SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA")[:held]
        positions = tuple(make_position(symbol) for symbol in symbols)
        assert numbers(report_line("open_positions", positions=positions)) == (
            5.0, used, headroom, breached,
        )

    def test_the_cap_comes_from_the_limits(self):
        found = report_line(
            "open_positions",
            limits=make_limits(max_open_positions=2),
            positions=(make_position("SPY"),),
        )
        assert numbers(found) == (2.0, 1.0, 1.0, False)
        zero = report_line("open_positions", limits=make_limits(max_open_positions=0))
        assert numbers(zero) == (0.0, 0.0, 0.0, True)

    def test_unknown_positions(self):
        found = report_line("open_positions", positions=None, open_orders=(make_order(),))
        assert numbers(found) == (5.0, None, None, False)
        assert "positions are unknown" in found.detail

    def test_a_cap_too_large_for_a_float_is_unlimited_not_an_error(self):
        found = report_line(
            "open_positions",
            limits=make_limits(max_open_positions=10**5000),
            positions=(make_position(),),
        )
        assert numbers(found) == (INF, 1.0, INF, False)
        assert "a number too large to print" in found.detail


# --- the daily trades line ----------------------------------------------------


class TestDailyTradesLine:
    @pytest.mark.parametrize(
        ("orders", "headroom", "breached"),
        [(0, 10.0, False), (3, 7.0, False), (9, 1.0, False), (10, 0.0, True), (12, -2.0, True)],
        ids=["none", "three", "one left", "at the cap", "over"],
    )
    def test_the_numbers_by_hand(self, orders, headroom, breached):
        found = report_line("daily_trades", orders_today=orders)
        assert numbers(found) == (10.0, float(orders), headroom, breached)
        assert found.unit == "trades"

    def test_the_cap_comes_from_the_limits(self):
        limits = make_limits(max_daily_trades=3)
        assert numbers(report_line("daily_trades", limits=limits, orders_today=2)) == (
            3.0, 2.0, 1.0, False,
        )
        assert numbers(report_line("daily_trades", limits=limits, orders_today=3)) == (
            3.0, 3.0, 0.0, True,
        )
        zero = report_line("daily_trades", limits=make_limits(max_daily_trades=0))
        assert numbers(zero) == (0.0, 0.0, 0.0, True)

    def test_the_detail_is_the_rules_own(self):
        for orders in (3, 10):
            context = make_context(orders_today=orders)
            (rule,) = [r for r in account_rules(context) if r.name == "max_daily_trades"]
            assert line(limits_report(context), "daily_trades").detail == rule.detail
        assert report_line("daily_trades", orders_today=3).detail == (
            "3 of 10 orders sent today (risk_limits.max_daily_trades)"
        )

    def test_counts_too_large_for_a_float_are_infinite_not_an_error(self):
        vast = 10**5000
        found = report_line(
            "daily_trades", limits=make_limits(max_daily_trades=vast), orders_today=vast - 1
        )
        assert numbers(found) == (INF, INF, 1.0, False)
        at_cap = report_line(
            "daily_trades", limits=make_limits(max_daily_trades=vast), orders_today=vast
        )
        assert numbers(at_cap) == (INF, INF, 0.0, True)


# --- the buying power lines ---------------------------------------------------


class TestBuyingPowerLines:
    def test_the_figures_by_hand(self):
        report = limits_report(make_context(buying_power=12_345.67, options_buying_power=890.12))
        power, options = line(report, "buying_power"), line(report, "options_buying_power")
        assert numbers(power) == (None, None, 12_345.67, False)
        assert numbers(options) == (None, None, 890.12, False)
        assert power.detail == "$12,345.67 available to equity orders"
        assert options.detail == "$890.12 available to option orders"
        assert (power.unit, options.unit) == ("USD", "USD")

    def test_zero_and_negative_buying_power_are_figures_not_unknowns(self):
        assert report_line("buying_power", buying_power=0.0).headroom == 0.0
        assert report_line("buying_power", buying_power=-250.0).headroom == -250.0
        assert report_line("options_buying_power", options_buying_power=0.0).headroom == 0.0

    @UNKNOWN
    def test_unknown_buying_power(self, unknown):
        found = report_line("buying_power", buying_power=unknown)
        assert numbers(found) == (None, None, None, False)
        assert "buying power is unknown" in found.detail

    @UNKNOWN
    def test_unknown_options_buying_power_names_the_fallback_the_rule_uses(self, unknown):
        # The buying_power rule holds an option order to the lesser of buying power and cash.
        found = report_line(
            "options_buying_power", options_buying_power=unknown, buying_power=9_000.0, cash=4_000.0
        )
        assert numbers(found) == (None, None, None, False)
        assert found.detail == (
            "options buying power is unknown: option orders are held to the lesser of "
            "buying power and cash, $4,000.00"
        )
        swapped = report_line(
            "options_buying_power", options_buying_power=unknown, buying_power=3_000.0, cash=4_000.0
        )
        assert swapped.detail.endswith("$3,000.00")

    @pytest.mark.parametrize("missing", ["buying_power", "cash"])
    def test_unknown_options_buying_power_with_no_fallback(self, missing):
        found = report_line("options_buying_power", options_buying_power=None, **{missing: None})
        assert numbers(found) == (None, None, None, False)
        assert "every option order is rejected" in found.detail

    def test_the_equity_line_does_not_depend_on_cash_or_options_buying_power(self):
        found = report_line("buying_power", cash=None, options_buying_power=None)
        assert numbers(found) == (None, None, BUYING_POWER, False)


# --- the account figures ------------------------------------------------------


class TestAccountFigures:
    FIGURES = ("equity", "start_of_day_equity", "daily_pnl", "buying_power", "options_buying_power")

    def test_known_figures_are_passed_on(self):
        report = limits_report(
            make_context(
                equity=98_765.43, start_of_day_equity=100_000.0, daily_pnl=-1_234.57,
                buying_power=150_000.0, options_buying_power=40_000.0,
            )
        )
        assert [getattr(report, figure) for figure in self.FIGURES] == [
            98_765.43, 100_000.0, -1_234.57, 150_000.0, 40_000.0,
        ]

    @pytest.mark.parametrize("figure", FIGURES)
    @UNKNOWN
    def test_an_unknown_figure_is_none_never_zero(self, figure, unknown):
        report = limits_report(make_context(**{figure: unknown}))
        assert getattr(report, figure) is None


# --- exposures ----------------------------------------------------------------


class TestExposures:
    def test_nothing_held_no_lines(self):
        assert limits_report(make_context()).exposures == ()

    def test_one_line_per_underlying_sorted_by_hand(self):
        context = make_context(
            positions=(
                make_position("QQQ", 4.0, current_price=500.0),  # 2,000.00
                make_position("AAPL", 10.0),  # 10 x 200.00 = 2,000.00
                make_position(CALL, 2.0, current_price=8.0, market_value=1_600.0),
            ),
            open_orders=(
                make_order("AAPL", quantity=3.0, limit_price=190.0),  # 570.00 pending
                make_order(  # 1 contract x 3.60 x 100 = 360.00 pending, on SPY
                    occ("call", 650.0), id="order-0002", client_order_id="client-0002",
                    limit_price=3.60,
                ),
                make_order(
                    "NVDA", id="order-0003", client_order_id="client-0003", limit_price=120.0
                ),
            ),
        )
        exposures = limits_report(context).exposures
        assert [found.underlying for found in exposures] == ["AAPL", "NVDA", "QQQ", "SPY"]
        cap = 5_000.0  # 5% of 100,000 equity
        assert exposures == (
            ExposureLine(
                underlying="AAPL", exposure=2_570.0, cap=cap, headroom=2_430.0, breached=False
            ),
            ExposureLine(
                underlying="NVDA", exposure=120.0, cap=cap, headroom=4_880.0, breached=False
            ),
            ExposureLine(
                underlying="QQQ", exposure=2_000.0, cap=cap, headroom=3_000.0, breached=False
            ),
            ExposureLine(
                underlying="SPY", exposure=1_960.0, cap=cap, headroom=3_040.0, breached=False
            ),
        )

    @pytest.mark.parametrize(
        ("value", "headroom", "breached"),
        [
            (4_999.99, 0.01, False),
            (5_000.0, 0.0, False),  # at the cap is not over it
            (5_000.01, -0.01, True),
            (6_000.0, -1_000.0, True),
        ],
        ids=["a cent inside", "at the cap", "a cent over", "over"],
    )
    def test_boundary_over_the_cap_is_breached(self, value, headroom, breached):
        context = make_context(positions=(make_position(market_value=value),))
        (found,) = limits_report(context).exposures
        assert found.exposure == value and found.cap == 5_000.0
        assert found.headroom == pytest.approx(headroom)
        assert found.breached is breached

    def test_float_noise_at_the_cap_is_not_a_breach(self):
        # 3 x 333.35 is 1000.0500000000001 in floats: above a cap of 1,000.05 only by noise.
        assert 3 * 333.35 > 1_000.05
        context = make_context(
            equity=1_000.05,
            limits=make_limits(max_position_pct=100.0),
            positions=(make_position(qty=3.0, current_price=333.35, market_value=3 * 333.35),),
        )
        (found,) = limits_report(context).exposures
        assert found.cap == 1_000.05
        assert found.breached is False
        # ... as the rule sees it: a proposal that adds nothing to speak of still fits.
        probe = make_proposal(quantity=1e-9, limit_price=1e-9)
        assert max_position_pct(probe, context).outcome is RuleOutcome.PASS

    def test_the_cap_is_a_percentage_of_equity_from_the_limits(self):
        context = make_context(
            equity=40_000.0,
            start_of_day_equity=100_000.0,
            limits=make_limits(max_position_pct=25.0),
            positions=(make_position(),),
        )
        (found,) = limits_report(context).exposures
        assert (found.exposure, found.cap, found.headroom) == (2_000.0, 10_000.0, 8_000.0)

    def test_a_short_counts_by_magnitude(self):
        short = make_position(qty=-10.0, side="short", market_value=-2_000.0)
        (found,) = limits_report(make_context(positions=(short,))).exposures
        assert (found.exposure, found.headroom) == (2_000.0, 3_000.0)

    def test_an_option_position_valued_from_its_price_carries_the_multiplier(self):
        position = make_position(CALL, 2.0, current_price=8.0, market_value=None)
        (found,) = limits_report(make_context(positions=(position,))).exposures
        assert (found.underlying, found.exposure) == ("SPY", 1_600.0)  # 2 x 8.00 x 100

    def test_an_exposure_that_cannot_be_valued_is_unknown_not_zero(self):
        position = make_position(market_value=None, current_price=None)
        context = make_context(positions=(position, make_position("SPY")))
        exposures = limits_report(context).exposures
        assert exposures[0] == ExposureLine(
            underlying="AAPL", exposure=None, cap=5_000.0, headroom=None, breached=False
        )
        assert exposures[1].exposure == 2_000.0
        pending = make_context(open_orders=(make_order(limit_price=None),))
        assert limits_report(pending).exposures == (
            ExposureLine(
                underlying="AAPL", exposure=None, cap=5_000.0, headroom=None, breached=False
            ),
        )

    @pytest.mark.parametrize("equity", [None, NAN, INF, 0.0, -5.0], ids=str)
    def test_unknown_or_non_positive_equity_has_no_cap(self, equity):
        context = make_context(equity=equity, positions=(make_position(),))
        (found,) = limits_report(context).exposures
        assert found == ExposureLine(
            underlying="AAPL", exposure=2_000.0, cap=None, headroom=None, breached=False
        )

    def test_unknown_positions_give_no_exposure_lines(self):
        context = make_context(positions=None, open_orders=(make_order(),))
        report = limits_report(context)
        assert report.exposures == ()
        assert line(report, "open_positions").used is None


# --- no drift from the rules --------------------------------------------------

SEED = 20260930
CONTEXTS = 500
NEWCOMER = "ZZNEW"
"""A symbol no random context holds: a proposal on it would open a new position."""


def _contexts() -> list[PolicyContext]:
    rng = random.Random(SEED)
    return [
        random_context(rng, turbulence=rng.choice(RANDOM_TURBULENCE)) for _ in range(CONTEXTS)
    ]


def _rejects(result) -> bool:
    return result.outcome is REJECT


def _as_shown(count: int) -> float:
    """A count as a ``LimitLine`` carries it: a float — infinite when the
    limit is a whole number no float can hold (``RiskLimits`` accepts those)."""
    try:
        return float(count)
    except OverflowError:
        return INF


class TestNoDrift:
    def test_the_sweep_is_seeded_and_varied(self):
        contexts = _contexts()
        assert contexts == _contexts()
        assert len(contexts) == CONTEXTS
        assert sum(1 for c in contexts if c == make_context()) >= 50  # calm ones too
        assert len({c.model_dump_json() for c in contexts}) >= 300
        # Every limit is one RiskLimits accepts — and the sweep reaches the
        # extremes it accepts: zero, the largest finite float, a count beyond one.
        limits = [c.limits for c in contexts]
        assert all(
            math.isfinite(value)
            for drawn in limits
            for value in (
                drawn.halt_fallback_hours, drawn.max_loss_per_trade,
                drawn.limit_price_tolerance_pct, drawn.auto_execute.max_notional,
            )
        )
        assert any(drawn.max_daily_trades == 0 for drawn in limits)
        assert any(drawn.max_open_positions == 0 for drawn in limits)
        assert any(drawn.halt_fallback_hours == LARGEST_FINITE for drawn in limits)
        assert any(drawn.auto_execute.max_notional == LARGEST_FINITE for drawn in limits)
        assert any(_as_shown(drawn.max_daily_trades) == INF for drawn in limits)
        assert any(_as_shown(drawn.max_open_positions) == INF for drawn in limits)

    def test_blocked_by_and_the_stop_flags_agree_with_the_account_rules(self):
        seen = set()
        for context in _contexts():
            report = limits_report(context)
            account = {result.name: result for result in account_rules(context)}
            assert tuple(account) == ACCOUNT_RULE_NAMES
            assert report.blocked_by == tuple(
                name for name, result in account.items() if _rejects(result)
            )
            assert report.kill_switch is _rejects(account["kill_switch"])
            assert report.halted is _rejects(account["halted"])
            seen.update(report.blocked_by)
            seen.add(report.halted)
        assert seen == {*ACCOUNT_RULE_NAMES, True, False}

    def test_the_daily_loss_line_agrees_with_the_daily_loss_rule(self):
        flags = set()
        for context in _contexts():
            found = line(limits_report(context), "daily_loss")
            (rule,) = [r for r in account_rules(context) if r.name == "daily_loss_limit"]
            tripped = rule.halt_until is not None  # the rule names a halt only when it trips
            assert found.breached is tripped
            unknown = found.headroom is None
            assert _rejects(rule) == (found.breached or unknown)
            if not unknown:
                assert found.headroom == pytest.approx(context.daily_pnl + found.limit)
                assert found.breached == (not exceeds(found.headroom, 0.0))
                assert found.used == max(0.0, -context.daily_pnl)
            assert found.detail == rule.detail
            flags.add((found.breached, unknown))
        assert flags == {(True, False), (False, False), (False, True)}

    def test_the_daily_trades_line_agrees_with_the_trade_cap_rule(self):
        flags = set()
        for context in _contexts():
            found = line(limits_report(context), "daily_trades")
            (rule,) = [r for r in account_rules(context) if r.name == "max_daily_trades"]
            assert found.breached is _rejects(rule)
            assert found.breached == (found.headroom <= 0)
            assert found.headroom == found.limit - found.used
            assert (found.limit, found.used) == (
                _as_shown(context.limits.max_daily_trades), context.orders_today,
            )
            flags.add(found.breached)
        assert flags == {True, False}

    def test_the_open_positions_line_agrees_with_the_position_count_rule(self):
        flags = set()
        newcomer = make_proposal(symbol=NEWCOMER)
        for context in _contexts():
            found = line(limits_report(context), "open_positions")
            rule = max_open_positions(newcomer, context)
            held = held_underlyings(context)
            if held is None:
                assert (found.used, found.headroom, found.breached) == (None, None, False)
                assert _rejects(rule)  # unknown positions: the rule rejects
            else:
                assert NEWCOMER not in held
                assert found.used == len(held)
                assert found.headroom == found.limit - found.used
                # breached means exactly: a new underlying would be rejected
                assert found.breached is _rejects(rule)
            flags.add((found.breached, held is None))
        assert flags == {(True, False), (False, False), (False, True)}

    def test_the_buying_power_line_is_the_room_the_rule_leaves(self):
        flags = set()
        proposal = make_proposal()  # an equity buy of 400.00
        for context in _contexts():
            found = line(limits_report(context), "buying_power")
            rule = buying_power(proposal, context)
            assert found.headroom == known(context.buying_power)
            if found.headroom is None:
                assert _rejects(rule)
            else:
                assert _rejects(rule) == exceeds(NOTIONAL, found.headroom)
            flags.add((found.headroom is None, _rejects(rule)))
        assert flags == {(True, True), (False, True), (False, False)}

    def test_the_options_buying_power_line_is_the_room_the_rule_leaves(self):
        flags = set()
        proposal = long_call()
        for context in _contexts():
            found = line(limits_report(context), "options_buying_power")
            cost = notional(proposal, context)  # 8.00 x the context's contract multiplier
            rule = buying_power(proposal, context)
            assert found.headroom == known(context.options_buying_power)
            if found.headroom is not None:
                assert _rejects(rule) == exceeds(cost, found.headroom)
                continue
            power, cash = known(context.buying_power), known(context.cash)
            if power is None or cash is None:
                assert "every option order is rejected" in found.detail
                assert _rejects(rule)
                flags.add("no fallback")
            else:
                fallback = min(power, cash)
                assert found.detail.endswith(f"buying power and cash, {money(fallback)}")
                assert _rejects(rule) == exceeds(cost, fallback)
                flags.add(("fallback", _rejects(rule)))
        assert flags == {"no fallback", ("fallback", True), ("fallback", False)}

    def test_each_exposure_line_agrees_with_the_position_cap_rule(self):
        flags = set()
        for context in _contexts():
            report = limits_report(context)
            held = held_underlyings(context)
            if held is None:
                assert report.exposures == ()
                continue
            assert [found.underlying for found in report.exposures] == sorted(held)
            for found in report.exposures:
                # A one-dollar buy of the underlying: the rule adds it to the same exposure
                # and compares the sum with the same cap.
                probe = make_proposal(symbol=found.underlying, quantity=1.0, limit_price=1.0)
                rule = max_position_pct(probe, context)
                if found.exposure is None or found.cap is None:
                    assert (found.headroom, found.breached) == (None, False)
                    assert _rejects(rule)  # unknown: the rule rejects
                    flags.add("unknown")
                    continue
                assert found.headroom == pytest.approx(found.cap - found.exposure)
                assert found.breached == exceeds(found.exposure, found.cap)
                assert _rejects(rule) == exceeds(1.0 + found.exposure, found.cap)
                if found.breached:
                    assert _rejects(rule)
                flags.add(found.breached)
        assert flags == {"unknown", True, False}

    def test_the_account_figures_are_the_known_ones(self):
        for context in _contexts():
            report = limits_report(context)
            for figure in TestAccountFigures.FIGURES:
                assert getattr(report, figure) == known(getattr(context, figure))
            assert report.as_of == context.now
            assert report.errors == context.errors
            assert report.halt_until == context.controls.halt_until
            assert report.market_open == (None if context.clock is None else context.clock.is_open)
            assert limits_report(context) == report  # and deterministically


# --- hostile but valid --------------------------------------------------------


class TestHostileButValidContexts:
    def test_the_report_never_raises(self):
        vast = make_limits(
            daily_loss_limit_pct=100, halt_fallback_hours=LARGEST_FINITE,
            max_daily_trades=10**5000, max_open_positions=10**400, max_position_pct=100,
            max_loss_per_trade=LARGEST_FINITE,
            auto_execute={"enabled": False, "max_notional": LARGEST_FINITE},
        )
        tiny = make_limits(
            daily_loss_limit_pct=1e-300, max_daily_trades=0, max_open_positions=0,
            max_position_pct=1e-300, auto_execute={"max_notional": 0.0},
        )
        weird_positions = (
            make_position("", 0.0, side=""),
            make_position(qty=NAN, market_value=NAN, current_price=NAN),
            make_position("SPY", INF, market_value=-INF, current_price=-INF),
            make_position(CALL, -1e308, side="short", market_value=None, current_price=1e308),
            make_position("\x1b[31mEVIL\n", 5.0, market_value=-3.0),
        )
        weird_orders = (
            make_order(limit_price=None),
            make_order(CALL, id="o2", client_order_id="c2", limit_price=-1.0),
            make_order("SPY", id="o3", client_order_id="c3", quantity=1e308, limit_price=1e308),
        )
        far = datetime.max.replace(tzinfo=timezone.utc)
        contexts = [
            make_context(watchlist=()),
            make_context(limits=vast, orders_today=10**5000, contract_multiplier=10**400),
            make_context(limits=tiny, positions=weird_positions, open_orders=weird_orders),
            make_context(positions=None, open_orders=weird_orders, clock=None, quote=None),
            make_context(
                equity=NAN, cash=INF, buying_power=-INF, options_buying_power=NAN,
                start_of_day_equity=INF, daily_pnl=NAN,
            ),
            make_context(equity=1e308, start_of_day_equity=1e308, daily_pnl=1e308),
            make_context(equity=5e-324, start_of_day_equity=5e-324, daily_pnl=-5e-324),
            make_context(now=far, market_date=date.max, daily_pnl=-1e9, clock=None),
            make_context(
                now=datetime.min.replace(tzinfo=timezone.utc), market_date=date.min,
                controls=Controls(halt_until=far, kill_switch=True),
            ),
            make_context(
                clock=make_clock(
                    next_open=datetime(2026, 7, 31, 9, 30), next_close=datetime(2026, 7, 30, 16)
                ),
                controls=Controls(halt_unknown=True),
            ),
        ]
        for context in contexts:
            report = limits_report(context)
            assert tuple(found.name for found in report.lines) == LINE_NAMES
            assert report.blocked_by == tuple(
                r.name for r in account_rules(context) if r.outcome is REJECT
            )
            for found in report.lines:
                assert found.detail.strip()
                for number in (found.limit, found.used, found.headroom):
                    assert number is None or number == number  # never NaN
            for found in report.exposures:
                for number in (found.exposure, found.cap, found.headroom):
                    assert number is None or abs(number) < INF  # None when unknown, never inf/NaN
            assert limits_report(context) == report

    def test_an_overflowing_headroom_is_unknown_not_infinite(self):
        found = report_line(
            "daily_loss",
            start_of_day_equity=1e308,
            daily_pnl=1e308,
            limits=make_limits(daily_loss_limit_pct=100),
        )
        assert (found.limit, found.used, found.headroom) == (1e308, 0.0, None)
        assert found.breached is False

    def test_an_empty_watchlist_changes_nothing_here(self):
        assert limits_report(make_context(watchlist=())) == limits_report(
            make_context(watchlist=WATCHLIST)
        )

    def test_the_market_date_is_not_the_reports_business(self):
        assert limits_report(make_context(market_date=MARKET_DATE)) == limits_report(
            make_context(market_date=date(2030, 1, 1))
        )


# --- the limits module itself -------------------------------------------------

LIMITS_SOURCE = REPO_ROOT / "aegis" / "policy" / "limits.py"


def _limits_tree() -> ast.Module:
    return ast.parse(LIMITS_SOURCE.read_text(encoding="utf-8"), filename="limits.py")


class TestLimitsModule:
    def test_it_imports_only_the_pure_core(self):
        imported = set()
        for node in ast.walk(_limits_tree()):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                imported.add(node.module)
        assert imported == {
            "__future__", "math", "aegis.policy.measures", "aegis.policy.models",
            "aegis.policy.rules",
        }

    def test_no_clock_is_read(self):
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(_limits_tree())
            if isinstance(node, ast.Call)
        }
        assert called.isdisjoint({"utcnow", "now", "today", "time", "monotonic"})

    def test_no_risk_number_is_hardcoded(self):
        found = {
            node.value
            for node in ast.walk(_limits_tree())
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
        }
        assert found <= {0, 1, 100}

    def test_the_report_uses_the_rules_and_the_measures_not_its_own_arithmetic(self):
        names = {
            alias.name
            for node in ast.walk(_limits_tree())
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert {
            "account_rules", "daily_loss_cap", "exposure", "held_underlyings", "exceeds", "known",
        } <= names

    def test_limits_report_is_documented(self):
        assert limits_report.__doc__ and limits_report.__doc__.strip()
