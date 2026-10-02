"""The twenty rules: each a pure function of (proposal, context) → one ``RuleResult``.

``RULES`` is the registry, in the order the engine runs them and the audit
trail lists them; each function is named exactly as its rule. A rule reads
nothing but its two arguments — no clock, no store, no config — and every
number it compares comes from ``context.limits`` (``risk_limits`` in
config.yaml) or ``context.contract_multiplier``; none is written here.

Outcomes: REJECT stops the proposal; FLAG records it and executes nothing;
ESCALATE sends it to a human; PASS lets it through this rule. Unknown is
never good news: a rule that needs a figure the context does not have
REJECTs and says what is missing. The two tier rules (``short_sale``,
``auto_tier``) ESCALATE instead, because all they decide is whether a human
must look. A rule never raises for any proposal and context the models
accept; comparisons go through ``measures.exceeds`` / ``below`` so float
noise never flips a boundary.

The detail of every result states the numbers compared, the config key the
limit comes from, or exactly what is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone

from aegis.policy.measures import (
    TOLERANCE,
    below,
    daily_loss_cap,
    exceeds,
    exposure,
    held_underlyings,
    is_closing_sale,
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
)
from aegis.policy.models import PolicyContext, ProposalUnderReview, RuleOutcome, RuleResult
from aegis.store.models import OrderSide, OrderType

Rule = Callable[[ProposalUnderReview, PolicyContext], RuleResult]


# --- helpers ------------------------------------------------------------------


def _passed(name: str, detail: str) -> RuleResult:
    return RuleResult(name=name, outcome=RuleOutcome.PASS, detail=detail)


def _rejected(name: str, detail: str, *, halt_until: datetime | None = None) -> RuleResult:
    return RuleResult(
        name=name, outcome=RuleOutcome.REJECT, detail=detail, halt_until=halt_until
    )


def _escalated(name: str, detail: str) -> RuleResult:
    return RuleResult(name=name, outcome=RuleOutcome.ESCALATE, detail=detail)


def _flagged(name: str, detail: str) -> RuleResult:
    return RuleResult(name=name, outcome=RuleOutcome.FLAG, detail=detail)


def _aware(value: datetime) -> datetime:
    """A naive timestamp is taken as UTC (the data layer's convention), so it
    can be compared with ``context.now``."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _when(value: datetime) -> str:
    return _aware(value).isoformat(timespec="seconds")


def _number(value: float) -> str:
    """A quantity, percentage or count without float noise (``0.3``, not
    ``0.30000000000000004``)."""
    return f"{value:.10g}"


def _count(value: int) -> str:
    try:
        return str(value)
    except ValueError:  # an int beyond the interpreter's digit limit for printing
        return "a number too large to print"


def _sliver(difference: float) -> str:
    """A difference of less than a cent between two dollar figures
    (``$0.004``): six significant digits — enough to show it, short of the
    float noise the subtraction itself leaves behind (``0.002500000001``)."""
    return f"${abs(difference):.6g}"


def _by(value: float, limit: float) -> str:
    """`` by $0.004`` for two dollar figures that differ and print alike.

    ``money`` shows cents, ``exceeds`` resolves far finer: a detail whose
    outcome hinges on a fraction of a cent would read "$1,000.00 is above
    $1,000.00". Called where the outcome says the two differ; empty whenever
    their texts already tell them apart."""
    if money(value) != money(limit):
        return ""
    return f" by {_sliver(value - limit)}"


def _price(value: float) -> str:
    """A price as a detail quotes it: ``money`` when cents show every digit,
    else every digit (``$1.205``) — a mid on the half cent, shown rounded,
    would not reproduce the percentage computed from it."""
    cents = float(f"{value:.2f}")
    if not (exceeds(value, cents) or below(value, cents)):
        return money(value)
    return f"{'-' if value < 0 else ''}${abs(value):,.10g}"


def _compact(symbol: str) -> str:
    """A symbol as the no-trade list and the duplicate rule match it:
    upper-cased, whitespace removed, every decimal digit written in ASCII —
    so each spelling of one instrument is one name. The padded OCC form
    (``SPY   260821C00640000``) and a strike in another script's digits both
    parse as the compact contract, and must match it and its listed names."""
    folded = (str(int(ch)) if ch.isdecimal() else ch for ch in "".join(symbol.split()))
    return "".join(folded).upper()


def _names(values: Iterable[str]) -> str:
    return ", ".join(sorted(values)) or "none"


def _trade(proposal: ProposalUnderReview) -> str:
    """``buy 4 AAPL`` — the proposal in three words for a detail line."""
    order = proposal.proposal
    return f"{order.side.value} {_number(order.quantity)} {order.symbol}"


# --- 1–4 and 11: the account-level rules --------------------------------------
#
# These five look only at the context, so the limits report can ask "would
# anything be rejected right now?" without a proposal: ``account_rules``
# returns exactly what the registered rules return.


def _kill_switch(context: PolicyContext) -> RuleResult:
    name = "kill_switch"
    if context.controls.kill_switch:
        return _rejected(name, "the kill switch is ON: every proposal is rejected")
    return _passed(name, "the kill switch is off")


def _halted(context: PolicyContext) -> RuleResult:
    name = "halted"
    controls = context.controls
    if controls.halt_unknown:
        reasons = "; ".join(controls.problems) or "the stored halt_until is unreadable"
        return _rejected(name, f"cannot prove trading is not halted: {reasons}")
    until = controls.halt_until
    if until is None:
        return _passed(name, "no trading halt on record")
    if _aware(until) > context.now:
        return _rejected(
            name, f"trading is halted until {_when(until)} (now {_when(context.now)})"
        )
    return _passed(name, f"the trading halt expired at {_when(until)}")


def _market_hours(context: PolicyContext) -> RuleResult:
    name = "market_hours"
    clock = context.clock
    if clock is None:
        return _rejected(name, "cannot confirm the market is open: no market clock")
    if not clock.is_open:
        opens = clock.next_open
        reopens = f"next open {_when(opens)}" if opens is not None else "next open unknown"
        return _rejected(name, f"the market is closed ({reopens})")
    close = clock.next_close
    if close is not None and context.now >= _aware(close):
        return _rejected(
            name,
            "cannot confirm the market is open: the clock says open but its close "
            f"{_when(close)} is not after now {_when(context.now)} (a stale clock)",
        )
    closes = f"closes {_when(close)}" if close is not None else "close time unknown"
    return _passed(name, f"the market is open ({closes})")


def _coming_session_close(context: PolicyContext) -> datetime | None:
    """The close of TODAY's session when the market is closed before it:
    the clock is closed, its next open is still ahead and falls on the
    trading date itself (``next_open_date == market_date``), and it names a
    close after that open. None otherwise — during the session, after the
    close (the next open is on a later date), or when the context cannot
    tell (no clock, no ``next_open_date``, no usable close)."""
    clock = context.clock
    if clock is None or clock.is_open or clock.next_open is None or clock.next_close is None:
        return None
    if context.next_open_date is None or context.next_open_date != context.market_date:
        return None
    opens, closes = _aware(clock.next_open), _aware(clock.next_close)
    if not context.now < opens < closes:
        return None
    return closes


def _halt_instant(context: PolicyContext) -> datetime:
    """When trading may resume after the daily loss limit trips.

    The clock's next open when it has one after now: an intraday or an
    after-hours trip stops trading until the next session. A trip BEFORE
    today's session lasts until that session's close instead
    (``_coming_session_close``) — a halt to the opening bell would let a
    recovery by the open reopen trading the same day. With no usable next
    open, ``halt_fallback_hours`` from now. An instant beyond the calendar
    is the end of the calendar."""
    forever = datetime.max.replace(tzinfo=timezone.utc)
    clock = context.clock
    until = _coming_session_close(context)
    if until is None and clock is not None and clock.next_open is not None:
        next_open = _aware(clock.next_open)
        if next_open > context.now:
            until = next_open
    if until is not None:
        try:
            return until.astimezone(timezone.utc)
        except OverflowError:
            return forever
    try:
        return context.now + timedelta(hours=context.limits.halt_fallback_hours)
    except (OverflowError, ValueError):
        return forever


def _daily_loss_limit(context: PolicyContext) -> RuleResult:
    name = "daily_loss_limit"
    key = "risk_limits.daily_loss_limit_pct"
    pnl = known(context.daily_pnl)
    cap = daily_loss_cap(context)
    if pnl is None or cap is None:
        missing = []
        if pnl is None:
            missing.append("today's P&L is unknown")
        if cap is None:
            missing.append("start-of-day equity is unknown or not positive")
        return _rejected(name, f"cannot evaluate the daily loss limit: {'; '.join(missing)}")
    percent = _number(context.limits.daily_loss_limit_pct)
    basis = f"{percent}% of start-of-day equity {money(context.start_of_day_equity)} ({key})"
    if not exceeds(pnl, -cap):
        until = _halt_instant(context)
        # The detail names the halt actually in force. A longer one already
        # on record is not shortened by this trip, and the engine writes
        # nothing — unless the stored halt is unreadable: then the controls
        # hold that longer instant only because another reading gave it, and
        # the engine writes a finite halt no shorter than it. That halt is
        # this evaluation's to announce, not one "already in force".
        in_force = context.controls.halt_until
        if in_force is not None and _aware(in_force) > until:
            if context.controls.halt_unknown:
                halt = (
                    f"trading is halted until {_when(in_force)}, the later instant another "
                    "reading of the stored halt gave (one reading of it was unreadable)"
                )
            else:
                halt = f"a halt until {_when(in_force)} is already in force"
        elif _coming_session_close(context) is not None:
            # Tripped before today's open: the halt covers the coming session.
            halt = (
                f"trading is halted until {_when(until)}, the close of the session that "
                f"opens at {_when(context.clock.next_open)}"
            )
        else:
            halt = f"trading is halted until {_when(until)}"
        return _rejected(
            name,
            f"today's P&L {money(pnl)} is at or below the loss cap {money(-cap)}, {basis}: "
            f"{halt}",
            halt_until=until,
        )
    apart = _by(pnl, -cap)
    headroom = _sliver(pnl + cap) if apart else money(pnl + cap)
    return _passed(
        name,
        f"today's P&L {money(pnl)} is above the loss cap {money(-cap)}{apart}, {basis}: "
        f"{headroom} of headroom",
    )


def _max_daily_trades(context: PolicyContext) -> RuleResult:
    name = "max_daily_trades"
    key = "risk_limits.max_daily_trades"
    used, cap = context.orders_today, context.limits.max_daily_trades
    if used >= cap:
        return _rejected(
            name,
            f"{_count(used)} orders sent today: at or above the cap of {_count(cap)} ({key})",
        )
    return _passed(name, f"{_count(used)} of {_count(cap)} orders sent today ({key})")


def account_rules(context: PolicyContext) -> tuple[RuleResult, ...]:
    """The five results that do not depend on the proposal — kill_switch,
    halted, market_hours, daily_loss_limit, max_daily_trades, in that order.
    A REJECT among them blocks every proposal at this instant; the limits
    report shows them as ``blocked_by``."""
    return (
        _kill_switch(context),
        _halted(context),
        _market_hours(context),
        _daily_loss_limit(context),
        _max_daily_trades(context),
    )


def kill_switch(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT while the operator's kill switch is on."""
    return _kill_switch(context)


def halted(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT while a trading halt is in force (``halt_until`` after now), or
    when the stored halt cannot be read."""
    return _halted(context)


def market_hours(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT unless the market clock confirms the market is open now."""
    return _market_hours(context)


def daily_loss_limit(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT once today's P&L is at or below minus ``daily_loss_limit_pct``
    of start-of-day equity, and name the instant the halt lasts until — the
    next market open, or the close of today's session when the trip comes
    before its open, so a recovery later that day never reopens trading. The
    engine persists it. An unknown P&L or equity rejects without a halt."""
    return _daily_loss_limit(context)


# --- 5–7: what may be traded --------------------------------------------------


def no_trade_list(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a proposal that names a symbol or underlying on the no-trade list."""
    name = "no_trade_list"
    key = "risk_limits.no_trade_list"
    listed = {_compact(symbol) for symbol in context.limits.no_trade_list}
    hits = {_compact(named) for named in (*proposal.underlyings, *proposal.symbols)} & listed
    if hits:
        return _rejected(name, f"{_names(hits)} is on the no-trade list ({key})")
    clear = {_compact(underlying) for underlying in proposal.underlyings}
    return _passed(name, f"{_names(clear)} is not on the no-trade list ({key})")


def watchlist_only(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """When enabled, REJECT a proposal with an underlying outside the watchlist."""
    name = "watchlist_only"
    key = "risk_limits.watchlist_only"
    if not context.limits.watchlist_only:
        return _passed(name, f"disabled ({key})")
    outside = [symbol for symbol in proposal.underlyings if symbol not in context.watchlist]
    if outside:
        return _rejected(name, f"{_names(outside)} is not in the watchlist ({key})")
    return _passed(name, f"{_names(proposal.underlyings)} is in the watchlist ({key})")


def invalidation_present(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a proposal that does not say what would prove it wrong."""
    name = "invalidation_present"
    if not proposal.proposal.invalidation.strip():
        return _rejected(name, "the proposal has no invalidation condition")
    return _passed(name, "the proposal states an invalidation condition")


# --- 8–10: sizing -------------------------------------------------------------


def _available(
    proposal: ProposalUnderReview, context: PolicyContext
) -> tuple[float | None, str]:
    """The buying power a proposal draws on, and what to call it. Options
    cannot be bought on margin: they draw on options buying power, or — when
    the broker did not report it — the lesser of buying power and cash."""
    if proposal.is_equity:
        return known(context.buying_power), "buying power"
    options = known(context.options_buying_power)
    if options is not None:
        return options, "options buying power"
    power, cash = known(context.buying_power), known(context.cash)
    if power is None or cash is None:
        return None, "options buying power"
    return min(power, cash), "the lesser of buying power and cash"


def buying_power(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a proposal whose notional is above the buying power available
    to it — or that needs any at all when none (zero or less) is available."""
    name = "buying_power"
    if is_closing_sale(proposal, context):
        return _passed(name, "a closing sale consumes no buying power")
    cost = notional(proposal, context)
    available, label = _available(proposal, context)
    if cost is None:
        return _rejected(name, "cannot evaluate buying power: the notional cannot be computed")
    if available is None:
        return _rejected(name, f"cannot evaluate buying power: {label} is unknown")
    if available <= 0:
        # Nothing to spend is nothing to spend: float noise must not let a
        # notional of a fraction of a cent pass as "within" zero.
        return _rejected(
            name, f"notional {money(cost)} cannot be met: {label} is {money(available)}"
        )
    if exceeds(cost, available):
        return _rejected(
            name,
            f"notional {money(cost)} is above {label} {money(available)}{_by(cost, available)}",
        )
    return _passed(name, f"notional {money(cost)} is within {label} {money(available)}")


def max_position_pct(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT when this proposal's notional plus the exposure already held or
    pending in its underlying is above ``max_position_pct`` of equity."""
    name = "max_position_pct"
    key = "risk_limits.max_position_pct"
    if is_closing_sale(proposal, context):
        return _passed(name, "a closing sale reduces the position")
    underlying = proposal.underlying
    equity = known(context.equity)
    if equity is None or equity <= 0:
        return _rejected(
            name, f"cannot evaluate the position cap: equity is unknown or not positive ({key})"
        )
    cost = notional(proposal, context)
    if cost is None:
        return _rejected(
            name, f"cannot evaluate the position cap: the notional cannot be computed ({key})"
        )
    held = exposure(context, underlying)
    if held is None:
        return _rejected(
            name,
            f"cannot evaluate the position cap: existing exposure in {underlying} is unknown "
            f"({key})",
        )
    cap = context.limits.max_position_pct / 100 * equity
    total = cost + held
    percent = _number(context.limits.max_position_pct)
    sums = (
        f"notional {money(cost)} + existing {underlying} exposure {money(held)} = {money(total)}"
    )
    basis = f"{percent}% of equity {money(equity)} ({key})"
    if exceeds(total, cap):
        return _rejected(name, f"{sums} is above the cap {money(cap)}{_by(total, cap)}, {basis}")
    return _passed(name, f"{sums} is within the cap {money(cap)}, {basis}")


def max_open_positions(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a proposal that would open a NEW underlying while the count of
    distinct underlyings held or pending is at the cap."""
    name = "max_open_positions"
    key = "risk_limits.max_open_positions"
    held = held_underlyings(context)
    if held is None:
        return _rejected(name, f"cannot count open positions: positions are unknown ({key})")
    cap = context.limits.max_open_positions
    usage = f"{len(held)} of {_count(cap)} underlyings held or pending ({key})"
    if is_closing_sale(proposal, context):
        return _passed(name, f"a closing sale opens no position: {usage}")
    underlying = proposal.underlying
    if underlying in held:
        return _passed(name, f"{underlying} is already held or pending: {usage}")
    if len(held) >= cap:
        return _rejected(
            name, f"{underlying} would be a new position: {usage}, at the cap"
        )
    return _passed(name, f"{underlying} would be a new position: {usage}")


def max_daily_trades(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT once the orders sent to the broker today reach ``max_daily_trades``."""
    return _max_daily_trades(context)


# --- 12–13: order hygiene -----------------------------------------------------


def _window(context: PolicyContext) -> timedelta:
    try:
        return timedelta(minutes=context.limits.duplicate_window_minutes)
    except OverflowError:
        return timedelta.max


def duplicate(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a repeat: an EARLIER proposal for the same symbol, instrument
    and side inside ``duplicate_window_minutes``, or an open order already
    working one of the proposal's symbols. Only the earlier of two proposals
    makes the other a duplicate, so two never reject each other.

    A symbol is matched as the instrument it names, not as a string: the
    padded OCC form of a contract is that contract (``_compact``), so an
    order the books carry in another spelling still makes the proposal a
    repeat — as it already counts in the exposure of its underlying."""
    name = "duplicate"
    key = "risk_limits.duplicate_window_minutes"
    order = proposal.proposal
    window = _window(context)
    minutes = _count(context.limits.duplicate_window_minutes)
    findings = []
    for other in context.recent_proposals:
        if other.id == order.id:
            continue
        if (_compact(other.symbol), other.instrument, other.side) != (
            _compact(order.symbol),
            order.instrument,
            order.side,
        ):
            continue
        age = order.created_at - other.created_at
        earlier = age > timedelta(0) or (age == timedelta(0) and other.id < order.id)
        if earlier and age <= window:
            findings.append(
                f"proposal {other.id} ({other.side.value} {other.symbol}) was created at "
                f"{_when(other.created_at)}, within {minutes} minutes before this one ({key})"
            )
    traded = {_compact(symbol) for symbol in proposal.symbols}
    for open_order in context.open_orders:
        if _compact(open_order.symbol) in traded:
            findings.append(
                f"order {open_order.client_order_id} on {open_order.symbol} is still open "
                f"({open_order.status.value})"
            )
    if findings:
        return _rejected(name, "duplicate: " + "; ".join(findings))
    return _passed(
        name,
        f"no earlier {order.side.value} proposal for {order.symbol} within {minutes} minutes "
        f"({key}) and no open order on it",
    )


def limit_price_sanity(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a market order unless ``allow_market_orders``; REJECT a limit
    order with no usable price, no current mid to judge it by, or a price
    further than ``limit_price_tolerance_pct`` from that mid."""
    name = "limit_price_sanity"
    key = "risk_limits.limit_price_tolerance_pct"
    order = proposal.proposal
    if order.order_type is not OrderType.LIMIT:
        if context.limits.allow_market_orders:
            return _passed(
                name, "a market order, which is allowed (risk_limits.allow_market_orders)"
            )
        return _rejected(
            name, "market orders are not allowed (risk_limits.allow_market_orders)"
        )
    limit = signed_limit(proposal)
    if limit is None:
        return _rejected(name, "the limit order has no usable limit price")
    mid = mid_price(proposal, context)
    if mid is None:
        missing = "; ".join(quote_problems(proposal, context)) or "no usable quote"
        return _rejected(name, f"no current mid to judge the limit price by: {missing}")
    tolerance = context.limits.limit_price_tolerance_pct
    if proposal.is_equity:
        deviation = abs(abs(limit) - mid) / mid * 100
        compared = f"limit {_price(abs(limit))} vs mid {_price(mid)}"
    else:
        if abs(mid) <= TOLERANCE:
            return _rejected(
                name, "the net mid of the legs is zero: no price to judge the limit by"
            )
        deviation = abs(limit - mid) / abs(mid) * 100
        # Signed, so a credit proposed as a debit is ~200% away, not 0%.
        compared = f"limit {_price(limit)} vs net mid {_price(mid)} (debit +, credit -)"
    away = f"{_number(deviation)}% away, tolerance {_number(tolerance)}% ({key})"
    if exceeds(deviation, tolerance):
        return _rejected(name, f"{compared}: {away}")
    return _passed(name, f"{compared}: {away}")


# --- 14–17: options -----------------------------------------------------------


def options_min_dte(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT an option proposal with any leg closer to expiry than ``min_dte``."""
    name = "options_min_dte"
    key = "risk_limits.min_dte"
    if not proposal.is_option:
        return _passed(name, "not an option")
    if not proposal.legs:
        return _rejected(name, f"the option proposal has no legs to check ({key})")
    floor = context.limits.min_dte
    days = [(leg.symbol, leg_dte(leg, context)) for leg in proposal.legs]
    short = [f"{symbol} at {dte} DTE" for symbol, dte in days if dte < floor]
    if short:
        return _rejected(name, f"{'; '.join(short)}: below the minimum of {_count(floor)} ({key})")
    nearest = min(dte for _, dte in days)
    return _passed(name, f"nearest leg at {nearest} DTE, minimum {_count(floor)} ({key})")


def options_max_loss(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT a proposal whose shape is not what its instrument says — an
    option structure that cannot be analysed, an "equity" that carries legs
    or names an option contract — and an option structure whose max loss is
    above ``max_loss_per_trade``, cannot be established, or is unlimited:
    the last unconditionally, whatever the limit says."""
    name = "options_max_loss"
    key = "risk_limits.max_loss_per_trade"
    # The shape is judged first, for every proposal: a mislabelled option
    # must not pass here as "not an option".
    problems = structure_problems(proposal)
    if problems:
        what = (
            "the structure cannot be analysed"
            if proposal.is_option
            else "the proposal is not a plain equity"
        )
        return _rejected(name, f"{what}: {'; '.join(problems)}")
    if not proposal.is_option:
        return _passed(name, "not an option")
    analysis = context.analysis
    if analysis is None:
        return _rejected(name, "cannot evaluate max loss: no position analysis for the structure")
    if analysis.unlimited_risk:
        return _rejected(
            name,
            f"the structure has unlimited risk: rejected unconditionally, whatever the cap ({key})",
        )
    loss = known(analysis.max_loss)
    if loss is None or loss < 0:
        return _rejected(name, "cannot evaluate max loss: the analysis has no finite max loss")
    cap = context.limits.max_loss_per_trade
    if exceeds(loss, cap):
        return _rejected(
            name, f"max loss {money(loss)} is above the cap {money(cap)}{_by(loss, cap)} ({key})"
        )
    return _passed(name, f"max loss {money(loss)} is within the cap {money(cap)} ({key})")


def options_max_contracts(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """REJECT an option proposal with any leg larger than ``max_contracts``."""
    name = "options_max_contracts"
    key = "risk_limits.max_contracts"
    if not proposal.is_option:
        return _passed(name, "not an option")
    if not proposal.legs:
        return _rejected(name, f"the option proposal has no legs to check ({key})")
    cap = context.limits.max_contracts
    over = [
        f"{leg.symbol} x{_number(leg.quantity)}"
        for leg in proposal.legs
        if exceeds(leg.quantity, cap)
    ]
    if over:
        return _rejected(name, f"{'; '.join(over)}: above the cap of {_count(cap)} per leg ({key})")
    largest = max(leg.quantity for leg in proposal.legs)
    return _passed(
        name, f"largest leg x{_number(largest)}, cap {_count(cap)} per leg ({key})"
    )


def options_escalate(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """ESCALATE every proposal that is not a plain equity: options — and
    anything option-like, whatever its instrument says — never auto-execute."""
    name = "options_escalate"
    if proposal.is_option:
        return _escalated(name, "an option proposal always needs human approval")
    if not is_plain_equity(proposal):
        return _escalated(
            name,
            "not a plain equity, so it always needs human approval: "
            + "; ".join(structure_problems(proposal)),
        )
    return _passed(name, "not an option")


# --- 18–20: tiers -------------------------------------------------------------


def _why_not_closing(proposal: ProposalUnderReview, context: PolicyContext) -> str:
    """Why an equity sell cannot be shown to close a long position."""
    order = proposal.proposal
    held = long_quantity(context, order.symbol)
    if held is None:
        return f"positions are unknown, so it cannot be shown to close a long {order.symbol}"
    if held <= 0:
        return f"no long {order.symbol} position is held"
    return f"only {_number(held)} {order.symbol} held long"


def short_sale(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """An equity sell that does not close an existing long is a short sale:
    ESCALATE, or REJECT when ``reject_short_sales``."""
    name = "short_sale"
    key = "risk_limits.reject_short_sales"
    order = proposal.proposal
    if not proposal.is_equity or order.side is not OrderSide.SELL:
        return _passed(name, "not an equity sell")
    if is_closing_sale(proposal, context):
        return _passed(name, f"{_trade(proposal)} closes an existing long position")
    detail = f"{_trade(proposal)} is a short sale: {_why_not_closing(proposal, context)}"
    if context.limits.reject_short_sales:
        return _rejected(name, f"{detail}; short sales are rejected ({key})")
    return _escalated(name, f"{detail}; needs approval ({key})")


def min_confidence(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """FLAG a proposal whose confidence is below ``min_confidence``."""
    name = "min_confidence"
    key = "risk_limits.min_confidence"
    confidence, floor = proposal.proposal.confidence, context.limits.min_confidence
    compared = f"confidence {_number(confidence)}, floor {_number(floor)} ({key})"
    if below(confidence, floor):
        return _flagged(name, f"below the floor: {compared}")
    return _passed(name, compared)


def auto_tier(proposal: ProposalUnderReview, context: PolicyContext) -> RuleResult:
    """PASS only a proposal that may execute without a human: auto-execute
    enabled, a plain equity (declared equity, no legs, not an option
    symbol), a buy or a closing sale, a limit order, a watchlist symbol, and
    a known notional within a positive ``auto_execute.max_notional``.
    Anything else is ESCALATEd, with every unmet criterion listed."""
    name = "auto_tier"
    key = "risk_limits.auto_execute"
    order = proposal.proposal
    auto = context.limits.auto_execute
    unmet = []
    if not auto.enabled:
        unmet.append(f"auto-execute is disabled ({key}.enabled)")
    if not proposal.is_equity:
        unmet.append(f"the instrument is {order.instrument.value}, not equity")
    elif not is_plain_equity(proposal):
        unmet.append("not a plain equity: " + "; ".join(structure_problems(proposal)))
    if order.side is not OrderSide.BUY and not is_closing_sale(proposal, context):
        unmet.append("a sell that does not close an existing long equity position")
    if order.order_type is not OrderType.LIMIT:
        unmet.append(f"the order type is {order.order_type.value}, not limit")
    if proposal.underlying not in context.watchlist:
        unmet.append(f"{proposal.underlying} is not in the watchlist")
    if auto.max_notional <= 0:
        # A tier of zero admits nothing — not even a notional below float noise.
        unmet.append(
            f"the tier admits nothing: max_notional is {money(auto.max_notional)} "
            f"({key}.max_notional)"
        )
    cost = notional(proposal, context)
    if cost is None:
        unmet.append("the notional cannot be computed")
    elif auto.max_notional > 0 and exceeds(cost, auto.max_notional):
        unmet.append(
            f"notional {money(cost)} is above {money(auto.max_notional)}"
            f"{_by(cost, auto.max_notional)} ({key}.max_notional)"
        )
    if unmet:
        return _escalated(name, "needs approval: " + "; ".join(unmet))
    return _passed(
        name,
        f"{_trade(proposal)} qualifies: equity limit order on a watchlist symbol, notional "
        f"{money(cost)} within {money(auto.max_notional)} ({key}.max_notional)",
    )


# --- the registry -------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    kill_switch,
    halted,
    market_hours,
    daily_loss_limit,
    no_trade_list,
    watchlist_only,
    invalidation_present,
    buying_power,
    max_position_pct,
    max_open_positions,
    max_daily_trades,
    duplicate,
    limit_price_sanity,
    options_min_dte,
    options_max_loss,
    options_max_contracts,
    options_escalate,
    short_sale,
    min_confidence,
    auto_tier,
)
"""Every rule, in the order the engine runs them and the audit trail lists them."""

RULE_NAMES: tuple[str, ...] = tuple(rule.__name__ for rule in RULES)
"""The registered rule names, in registry order — each rule function is
named exactly as the ``name`` of the result it returns."""
