"""Where the account stands against its limits right now: ``limits_report``.

One pure function of a ``PolicyContext`` — what ``python -m aegis.cli.policy
limits`` prints and what the Phase 8 dashboard will read. No number here is
computed differently from the rule that enforces it: the report calls
``rules.account_rules`` for what blocks trading and for the daily-loss and
daily-trades findings, and the measures in ``measures`` for the caps, the
exposure and the counts. A limit shown as breached is one its rule rejects
on, by construction.

Unknown is never good news here either: a figure the context could not
supply (or one that is not a finite number) is reported as None — never as
zero — with the line's detail saying what is missing, and the rule that
needs it shows up in ``blocked_by`` when it rejects.
"""

from __future__ import annotations

import math

from aegis.policy.measures import (
    daily_loss_cap,
    exceeds,
    exposure,
    held_underlyings,
    known,
    money,
)
from aegis.policy.models import (
    ExposureLine,
    LimitLine,
    LimitsReport,
    PolicyContext,
    RuleOutcome,
    RuleResult,
)
from aegis.policy.rules import account_rules, daily_loss_limit, halted, max_daily_trades

_USD = "USD"


def _count(value: int) -> float:
    """A count as the float a ``LimitLine`` carries; one beyond the float
    range (the limits accept any integer) is infinite, not an error."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _shown(value: int) -> str:
    try:
        return str(value)
    except ValueError:  # an int beyond the interpreter's digit limit for printing
        return "a number too large to print"


def _daily_loss_line(context: PolicyContext, finding: RuleResult) -> LimitLine:
    """Distance to the daily loss cap. ``finding`` is the ``daily_loss_limit``
    rule's own result: it rejects when a figure is unknown or the cap is
    reached, so with both figures known a rejection is the trip itself."""
    pnl = known(context.daily_pnl)
    cap = daily_loss_cap(context)
    both = pnl is not None and cap is not None
    return LimitLine(
        name="daily_loss",
        limit=cap,
        used=None if pnl is None else max(0.0, -pnl),
        headroom=known(pnl + cap) if both else None,
        unit=_USD,
        breached=both and finding.outcome is RuleOutcome.REJECT,
        detail=finding.detail,
    )


def _open_positions_line(context: PolicyContext) -> LimitLine:
    """Distinct underlyings held or pending against ``max_open_positions`` —
    at the cap, ``max_open_positions`` rejects any new underlying."""
    key = "risk_limits.max_open_positions"
    cap = context.limits.max_open_positions
    held = held_underlyings(context)
    if held is None:
        return LimitLine(
            name="open_positions",
            limit=_count(cap),
            used=None,
            headroom=None,
            unit="positions",
            breached=False,
            detail=f"cannot count open positions: positions are unknown ({key})",
        )
    used = len(held)
    names = f": {', '.join(sorted(held))}" if held else ""
    return LimitLine(
        name="open_positions",
        limit=_count(cap),
        used=_count(used),
        headroom=_count(cap - used),
        unit="positions",
        breached=used >= cap,
        detail=f"{used} of {_shown(cap)} underlyings held or pending ({key}){names}",
    )


def _daily_trades_line(context: PolicyContext, finding: RuleResult) -> LimitLine:
    """Orders sent today against ``max_daily_trades``. ``finding`` is the
    ``max_daily_trades`` rule's own result: it rejects exactly at the cap."""
    used, cap = context.orders_today, context.limits.max_daily_trades
    return LimitLine(
        name="daily_trades",
        limit=_count(cap),
        used=_count(used),
        headroom=_count(cap - used),
        unit="trades",
        breached=finding.outcome is RuleOutcome.REJECT,
        detail=finding.detail,
    )


def _buying_power_line(context: PolicyContext) -> LimitLine:
    """What an equity order may spend: the ``buying_power`` rule's figure."""
    figure = known(context.buying_power)
    if figure is None:
        detail = "buying power is unknown: every equity order that needs it is rejected"
    else:
        detail = f"{money(figure)} available to equity orders"
    return LimitLine(
        name="buying_power",
        limit=None,
        used=None,
        headroom=figure,
        unit=_USD,
        breached=False,
        detail=detail,
    )


def _options_buying_power_line(context: PolicyContext) -> LimitLine:
    """What an option order may spend. When the broker did not report it the
    ``buying_power`` rule holds option orders to the lesser of buying power
    and cash, and the detail says so."""
    figure = known(context.options_buying_power)
    if figure is not None:
        detail = f"{money(figure)} available to option orders"
    else:
        power, cash = known(context.buying_power), known(context.cash)
        if power is None or cash is None:
            detail = (
                "options buying power is unknown, and so is buying power or cash: "
                "every option order is rejected"
            )
        else:
            detail = (
                "options buying power is unknown: option orders are held to the lesser of "
                f"buying power and cash, {money(min(power, cash))}"
            )
    return LimitLine(
        name="options_buying_power",
        limit=None,
        used=None,
        headroom=figure,
        unit=_USD,
        breached=False,
        detail=detail,
    )


def _position_cap(context: PolicyContext) -> float | None:
    """The per-underlying cap in dollars, as ``max_position_pct`` computes
    it: that percentage of equity. None when equity is unknown or not positive."""
    equity = known(context.equity)
    if equity is None or equity <= 0:
        return None
    return known(context.limits.max_position_pct / 100 * equity)


def _exposure_lines(context: PolicyContext) -> tuple[ExposureLine, ...]:
    """One line per underlying held or pending, sorted; none when the
    positions are unknown (the ``open_positions`` line says so)."""
    held = held_underlyings(context)
    if held is None:
        return ()
    cap = _position_cap(context)
    lines = []
    for underlying in sorted(held):
        committed = exposure(context, underlying)
        both = committed is not None and cap is not None
        lines.append(
            ExposureLine(
                underlying=underlying,
                exposure=committed,
                cap=cap,
                headroom=known(cap - committed) if both else None,
                breached=both and exceeds(committed, cap),
            )
        )
    return tuple(lines)


def limits_report(context: PolicyContext) -> LimitsReport:
    """Each limit beside its current consumption and headroom, at ``context.now``.

    ``blocked_by`` names the account-level rules that reject every proposal
    at this instant, in rule order, and ``halted`` is exactly the ``halted``
    rule's finding (an unreadable halt counts as halted). ``lines`` are, in
    order: ``daily_loss`` (the cap, the day's loss so far, the distance to
    the cap), ``open_positions``, ``daily_trades``, ``buying_power`` and
    ``options_buying_power`` (no limit of their own — the headroom is the
    figure). ``exposures`` holds each held or pending underlying against
    the per-position cap. Pure: it reads the context and nothing else.
    """
    limits = context.limits
    clock = context.clock
    account = account_rules(context)
    findings = {finding.name: finding for finding in account}
    return LimitsReport(
        as_of=context.now,
        market_open=None if clock is None else clock.is_open,
        next_open=None if clock is None else clock.next_open,
        next_close=None if clock is None else clock.next_close,
        kill_switch=context.controls.kill_switch,
        halt_until=context.controls.halt_until,
        halted=findings[halted.__name__].outcome is RuleOutcome.REJECT,
        blocked_by=tuple(
            finding.name for finding in account if finding.outcome is RuleOutcome.REJECT
        ),
        equity=known(context.equity),
        start_of_day_equity=known(context.start_of_day_equity),
        daily_pnl=known(context.daily_pnl),
        buying_power=known(context.buying_power),
        options_buying_power=known(context.options_buying_power),
        lines=(
            _daily_loss_line(context, findings[daily_loss_limit.__name__]),
            _open_positions_line(context),
            _daily_trades_line(context, findings[max_daily_trades.__name__]),
            _buying_power_line(context),
            _options_buying_power_line(context),
        ),
        exposures=_exposure_lines(context),
        auto_execute_enabled=limits.auto_execute.enabled,
        auto_execute_max_notional=limits.auto_execute.max_notional,
        errors=context.errors,
    )
