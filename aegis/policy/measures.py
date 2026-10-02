"""Pure measurements shared by the rules and the limits report.

Every function here is a pure function of the proposal and the
``PolicyContext`` — no clock, no I/O, no config read — so a number shown by
``limits_report`` is computed by exactly the code the rule that enforces it
uses, and the two can never drift apart.

Unknown is never good news. ``known`` is the one gate a figure passes
through before it is compared: ``None``, NaN and ±inf are all "unknown", and
a measure that needs an unknown figure returns ``None`` so its caller can
reject. The only infinity produced on purpose is ``notional`` for a
structure with unlimited risk, which then exceeds every finite cap.

Sign conventions: a price paid is positive (a debit), a price received is
negative (a credit); quantities, notionals and exposures are magnitudes.
"""

from __future__ import annotations

import math
from datetime import timezone

from aegis.data.models import OptionSnapshot, Quote, parse_occ_symbol
from aegis.policy.models import PolicyContext, ProposalUnderReview, underlying_of
from aegis.store.models import OrderSide, OrderType, ProposalLeg

TOLERANCE = 1e-9
"""Relative float-noise guard for comparisons. NOT a risk limit: it only
keeps ``3 * 0.05`` (0.15000000000000002) from counting as over a cap of
0.15. Every limit a rule enforces comes from ``context.limits``."""


# --- numbers ------------------------------------------------------------------


def known(value: object) -> float | None:
    """``value`` as a float when it is a real, finite number — else None.

    None, NaN, ±inf, a bool and anything that is not a number are all
    "unknown" (so is an int too large to be a float).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _as_float(value: float) -> float:
    """``float(value)``, with an int beyond the float range taken as ±inf."""
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def exceeds(value: float, limit: float) -> bool:
    """Whether ``value`` is above ``limit`` by more than float noise.

    ``value - limit > TOLERANCE * max(1, |value|, |limit|)``, so a figure
    AT its cap never counts as over it. ``math.inf`` exceeds every finite
    limit (and nothing exceeds an infinite one). NaN is not a number to
    compare: it exceeds nothing and nothing exceeds it — callers pass
    figures through ``known`` first and reject the unknown ones.
    """
    value, limit = _as_float(value), _as_float(limit)
    if math.isnan(value) or math.isnan(limit):
        return False
    if math.isinf(value) or math.isinf(limit):
        return value > limit
    return value - limit > TOLERANCE * max(1.0, abs(value), abs(limit))


def below(value: float, limit: float) -> bool:
    """Whether ``value`` is under ``limit`` by more than float noise."""
    return exceeds(limit, value)


def money(value: float | None) -> str:
    """Dollars for a rule detail: ``$1,234.56``, ``-$1,234.56``, ``unlimited``
    for infinity and ``n/a`` for an unknown (None or NaN) figure."""
    if value is None:
        return "n/a"
    value = _as_float(value)
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "unlimited" if value > 0 else "-unlimited"
    text = f"{abs(value):,.2f}"
    # A loss smaller than half a cent prints as $0.00, not -$0.00.
    negative = value < 0 and text.strip("0.,") != ""
    return f"-${text}" if negative else f"${text}"


def is_occ(symbol: str) -> bool:
    """Whether ``symbol`` is an OCC option symbol (``SPY260821C00640000``)."""
    try:
        parse_occ_symbol(symbol)
    except ValueError:
        return False
    return True


def _same_symbol(left: str, right: str) -> bool:
    return left.strip().upper() == right.strip().upper()


def _has_whitespace(symbol: str) -> bool:
    return any(character.isspace() for character in symbol)


# --- what the account holds ---------------------------------------------------


def long_quantity(context: PolicyContext, symbol: str) -> float | None:
    """Shares (or contracts) held LONG in exactly ``symbol``.

    0.0 when nothing is held there or the position is short; None when the
    positions are unknown or the matching position's quantity is. A position
    counts as long only when its side says "long" and its quantity is not
    negative — a row that contradicts itself cannot be shown to be a long.
    """
    if context.positions is None:
        return None
    total = 0.0
    for position in context.positions:
        if not _same_symbol(position.symbol, symbol):
            continue
        quantity = known(position.qty)
        if quantity is None:
            return None
        if position.side.strip().lower() != "long" or quantity < 0:
            return 0.0
        total += quantity
    return known(total)


def is_closing_sale(proposal: ProposalUnderReview, context: PolicyContext) -> bool:
    """True only for an EQUITY SELL of no more than the long position held in
    the proposal's symbol. Unknown positions can prove nothing: False. Nor
    can a sale close a long that is not there: with nothing held long it is
    False however small the sale — ``exceeds`` alone would call a billionth
    of a share "not above" a holding of zero."""
    order = proposal.proposal
    if not proposal.is_equity or order.side is not OrderSide.SELL:
        return False
    held = long_quantity(context, order.symbol)
    if held is None or held <= 0:
        return False
    return not exceeds(order.quantity, held)


def exposure(context: PolicyContext, underlying: str) -> float | None:
    """Dollars already committed to ``underlying``, held or pending.

    The sum of ``abs(market_value)`` over positions on that underlying
    (``abs(qty) × current_price`` when the market value is unknown, times
    the contract multiplier for an option position, whose price is quoted
    per share) plus ``quantity × limit_price`` over open orders on it (times
    the contract multiplier for an OCC option symbol). A row belongs to the
    underlying ``underlying_of`` reads from its symbol — the padded OCC form
    included.

    None when it cannot be valued: positions unknown, a position with
    neither a market value nor a quantity and price, or an open order
    without a positive limit price. 0.0 when nothing is held or pending.
    """
    if context.positions is None:
        return None
    target = underlying.strip().upper()
    multiplier = known(context.contract_multiplier)
    total = 0.0
    for position in context.positions:
        if underlying_of(position.symbol) != target:
            continue
        value = known(position.market_value)
        if value is None:
            quantity, price = known(position.qty), known(position.current_price)
            if quantity is None or price is None:
                return None
            value = quantity * price
            if is_occ(position.symbol):
                if multiplier is None:
                    return None
                value *= multiplier
        total += abs(value)
    for order in context.open_orders:
        if underlying_of(order.symbol) != target:
            continue
        price = known(order.limit_price)
        if price is None or price <= 0:
            return None
        value = order.quantity * price
        if is_occ(order.symbol):
            if multiplier is None:
                return None
            value *= multiplier
        total += value
    return known(total)


def held_underlyings(context: PolicyContext) -> frozenset[str] | None:
    """The underlying of every position and every open order — what
    ``max_open_positions`` counts (a contract reported in the padded OCC form
    counts under its ticker, as ``underlying_of`` reads it). None when the
    positions are unknown."""
    if context.positions is None:
        return None
    names = {underlying_of(position.symbol) for position in context.positions}
    names.update(underlying_of(order.symbol) for order in context.open_orders)
    return frozenset(names)


def daily_loss_cap(context: PolicyContext) -> float | None:
    """The day's loss cap as a positive dollar amount:
    ``daily_loss_limit_pct`` percent of start-of-day equity. None when that
    equity is unknown or not positive."""
    start = known(context.start_of_day_equity)
    percent = known(context.limits.daily_loss_limit_pct)
    if start is None or percent is None or start <= 0:
        return None
    return known(percent / 100 * start)


# --- prices -------------------------------------------------------------------


def signed_limit(proposal: ProposalUnderReview) -> float | None:
    """The proposal's limit price, signed: positive for a buy (a debit
    paid), negative for a sell (a credit or proceeds received). None for a
    market order, or when the limit price is missing, not finite or not
    positive."""
    order = proposal.proposal
    if order.order_type is not OrderType.LIMIT:
        return None
    price = known(order.limit_price)
    if price is None or price <= 0:
        return None
    return price if order.side is OrderSide.BUY else -price


def _own_quote(proposal: ProposalUnderReview, context: PolicyContext) -> Quote | None:
    """The context's equity quote, only when it is for this proposal's symbol."""
    quote = context.quote
    if quote is None or not _same_symbol(quote.symbol, proposal.proposal.symbol):
        return None
    return quote


def _leg_snapshot(leg: ProposalLeg, context: PolicyContext) -> OptionSnapshot | None:
    """The first snapshot in ``leg_quotes`` whose symbol is the leg's."""
    for snapshot in context.leg_quotes:
        if _same_symbol(snapshot.symbol, leg.symbol):
            return snapshot
    return None


def _leg_mid(leg: ProposalLeg, context: PolicyContext) -> float | None:
    """The leg's quote mid; None without a snapshot, or when its bid or ask
    is unknown, negative or crossed. A 0.0 bid is a valid option quote."""
    snapshot = _leg_snapshot(leg, context)
    if snapshot is None:
        return None
    bid, ask = known(snapshot.bid), known(snapshot.ask)
    if bid is None or ask is None or bid < 0 or ask < 0 or below(ask, bid):
        return None
    return known(snapshot.mid)


def _leg_sign(leg: ProposalLeg) -> int:
    return 1 if leg.side is OrderSide.BUY else -1


def _leg_ratio(leg: ProposalLeg, proposal: ProposalUnderReview) -> float:
    """Contracts of this leg per unit of the proposal (2.0 for the doubled
    leg of a 1x2 ratio spread)."""
    return leg.quantity / proposal.proposal.quantity


def mid_price(proposal: ProposalUnderReview, context: PolicyContext) -> float | None:
    """The current mid the proposal's limit price is judged against.

    Equity: the quote mid — only when the context's quote is for this
    proposal's symbol and its bid and ask are known, positive and not
    crossed. Option: the SIGNED net mid per proposal unit, the sum over legs
    of ``sign × (leg.quantity / proposal.quantity) × leg mid`` with sign +1
    for a bought leg and -1 for a sold one — positive for a debit structure,
    negative for a credit. None when there are no legs or any leg lacks a
    usable quote.
    """
    if proposal.is_equity:
        quote = _own_quote(proposal, context)
        if quote is None:
            return None
        bid, ask = known(quote.bid), known(quote.ask)
        if bid is None or ask is None or bid <= 0 or ask <= 0 or below(ask, bid):
            return None
        return known(quote.mid)
    if not proposal.legs:
        return None
    total = 0.0
    for leg in proposal.legs:
        mid = _leg_mid(leg, context)
        if mid is None:
            return None
        total += _leg_sign(leg) * _leg_ratio(leg, proposal) * mid
    return known(total)


def quote_problems(proposal: ProposalUnderReview, context: PolicyContext) -> tuple[str, ...]:
    """Why ``mid_price`` is None, named per symbol or leg (empty when it is
    not): what a rule detail tells the operator is missing."""
    if proposal.is_equity:
        symbol = proposal.proposal.symbol
        if context.quote is None:
            return (f"no quote for {symbol}",)
        if _own_quote(proposal, context) is None:
            return (f"the quote is for {context.quote.symbol}, not {symbol}",)
        if mid_price(proposal, context) is None:
            return (f"the {symbol} quote has no usable two-sided bid/ask",)
        return ()
    if not proposal.legs:
        return ("the option proposal has no legs",)
    problems = []
    for leg in proposal.legs:
        if _leg_snapshot(leg, context) is None:
            problems.append(f"no quote for leg {leg.symbol}")
        elif _leg_mid(leg, context) is None:
            problems.append(f"the quote for leg {leg.symbol} has no usable two-sided bid/ask")
    if not problems and mid_price(proposal, context) is None:
        problems.append("the net mid of the legs is not a finite number")
    return tuple(problems)


def priced_quotes(
    proposal: ProposalUnderReview, context: PolicyContext
) -> tuple[tuple[str, Quote | OptionSnapshot | None], ...]:
    """The quotes ``mid_price`` judges the limit price against, each beside
    the symbol it must be for.

    Equity: the proposal's own quote — None when the context holds none for
    its symbol. Option: one snapshot per distinct leg symbol, in leg order —
    None for a leg that has none. Empty for an option with no legs, which
    has nothing to price."""
    if proposal.is_equity:
        return ((proposal.proposal.symbol, _own_quote(proposal, context)),)
    quotes: dict[str, tuple[str, OptionSnapshot | None]] = {}
    for leg in proposal.legs:
        key = leg.symbol.strip().upper()
        if key not in quotes:
            quotes[key] = (leg.symbol, _leg_snapshot(leg, context))
    return tuple(quotes.values())


def quote_age(
    quote: Quote | OptionSnapshot, context: PolicyContext
) -> float | None:
    """Seconds from the quote's venue timestamp (``quote_time``, the time of
    the bid and ask a mid is made of) to the context's as-of time
    ``context.now``: negative for a quote stamped after it, None for a quote
    with no timestamp. Never measured from ``fetched_at`` — that says when
    we read the quote, not how old the market it shows is."""
    stamped = quote.quote_time
    if stamped is None:
        return None
    if stamped.utcoffset() is None:  # naive, or a tzinfo that names no offset
        stamped = stamped.replace(tzinfo=timezone.utc)
    return known((context.now - stamped).total_seconds())


def worst_case_entry(proposal: ProposalUnderReview, context: PolicyContext) -> float | None:
    """The signed per-unit price a MARKET order could be filled at.

    Equity buy: +ask; equity sell: -bid. Option: what the bought legs cost
    at their asks minus what the sold legs fetch at their bids, each scaled
    by its ratio to the proposal quantity. None when a needed quote is
    missing (an equity ask or bid must be positive; an option's may be 0.0).
    """
    order = proposal.proposal
    if proposal.is_equity:
        quote = _own_quote(proposal, context)
        if quote is None:
            return None
        price = known(quote.ask if order.side is OrderSide.BUY else quote.bid)
        if price is None or price <= 0:
            return None
        return price if order.side is OrderSide.BUY else -price
    if not proposal.legs:
        return None
    total = 0.0
    for leg in proposal.legs:
        snapshot = _leg_snapshot(leg, context)
        if snapshot is None:
            return None
        price = known(snapshot.ask if leg.side is OrderSide.BUY else snapshot.bid)
        if price is None or price < 0:
            return None
        total += _leg_sign(leg) * _leg_ratio(leg, proposal) * price
    return known(total)


def entry_price(proposal: ProposalUnderReview, context: PolicyContext) -> float | None:
    """The signed per-unit price the proposal would trade at: its limit for
    a limit order, the worst-case fill for a market order."""
    if proposal.proposal.order_type is OrderType.LIMIT:
        return signed_limit(proposal)
    return worst_case_entry(proposal, context)


def notional(proposal: ProposalUnderReview, context: PolicyContext) -> float | None:
    """Dollars the proposal commits. None means it cannot be computed, and
    callers reject.

    Equity: ``quantity × |entry price|``. Option bought (side buy, a debit):
    ``entry price × contract multiplier × quantity`` — None when the entry
    price is not positive, since a "debit" that is really a credit is
    inconsistent. Option sold (side sell, a credit): the structure's max
    loss from ``context.analysis`` — ``math.inf`` when the risk is
    unlimited, None when there is no analysis or its figure is unusable.
    """
    order = proposal.proposal
    if proposal.is_option and order.side is OrderSide.SELL:
        analysis = context.analysis
        if analysis is None:
            return None
        if analysis.unlimited_risk or analysis.max_loss is None:
            return math.inf
        if analysis.max_loss == math.inf:
            return math.inf
        loss = known(analysis.max_loss)
        return None if loss is None or loss < 0 else loss
    price = entry_price(proposal, context)
    if price is None:
        return None
    if proposal.is_equity:
        return known(order.quantity * abs(price))
    multiplier = known(context.contract_multiplier)
    if price <= 0 or multiplier is None:
        return None
    return known(price * multiplier * order.quantity)


# --- option structure ---------------------------------------------------------


def leg_dte(leg: ProposalLeg, context: PolicyContext) -> int:
    """Calendar days from the trading date to the leg's expiration: 0 means
    it expires today, negative that it already has."""
    return (leg.expiration - context.market_date).days


def _leg_contract_problem(leg: ProposalLeg) -> str | None:
    """Why a leg's ``symbol`` is not the contract the leg describes — None
    when it is. The analysis is computed from the leg's ``option_type``,
    ``strike`` and ``expiration``; the order would trade its ``symbol``. If
    the two disagree, the max loss analysed is not the max loss of the trade."""
    try:
        _, expiration, option_type, strike = parse_occ_symbol(leg.symbol)
    except ValueError:
        return f"leg {leg.symbol} is not an OCC option symbol"
    same_strike = not (exceeds(strike, leg.strike) or below(strike, leg.strike))
    if (expiration, option_type) == (leg.expiration, leg.option_type) and same_strike:
        return None
    return (
        f"leg {leg.symbol} names the {expiration.isoformat()} {strike:.10g} "
        f"{option_type.value}, but the leg says the {leg.expiration.isoformat()} "
        f"{leg.strike:.10g} {leg.option_type.value}"
    )


def _symbol_problems(proposal: ProposalUnderReview) -> list[str]:
    """One finding per symbol the proposal names — its own and each leg's —
    that contains whitespace or a character outside ASCII. No tradable
    symbol has either: the padded OCC form ``SPY   260821C00640000`` parses
    (and ``underlying_of`` reads its root as SPY, so the rules keyed on the
    underlying still see it), and so does a contract whose strike is written
    in another script's digits — but neither is the compact symbol the
    broker, the quotes and the books use. A malformed structure, never one
    that may pass as sound. (The symbol is quoted in ASCII, so the detail
    shows which characters are not what they look like.)"""
    problems = []
    named = [proposal.proposal.symbol, *(leg.symbol for leg in proposal.legs)]
    for symbol in dict.fromkeys(named):
        flaws = []
        if _has_whitespace(symbol):
            flaws.append("whitespace")
        if not symbol.isascii():
            flaws.append("non-ASCII characters")
        if not flaws:
            continue
        form = "is not in the compact OCC form: it" if is_occ(symbol) else "is not a ticker: it"
        problems.append(f"symbol {symbol!a} {form} contains {' and '.join(flaws)}")
    return problems


def structure_problems(proposal: ProposalUnderReview) -> tuple[str, ...]:
    """Why a proposal's shape is not what its instrument says — the single
    judge of shape, for both instruments. Empty when the shape is sound.

    Either instrument: a symbol — the proposal's own or a leg's — that
    contains whitespace (the padded OCC form included) or a character
    outside ASCII (a strike in another script's digits included): no
    tradable symbol does, so the shape is malformed whatever else is right
    about it.

    An OPTION proposal, one line per finding: no legs; legs on more than one
    underlying; a proposal symbol whose underlying is not the legs'; legs
    with different expirations; a leg quantity that is not a whole number of
    contracts; a leg whose ``symbol`` is not an OCC option symbol, or is the
    symbol of another contract than the leg's own ``expiration``,
    ``option_type`` and ``strike`` describe (its root is held to the
    proposal's underlying by the two findings above); a proposal whose OWN
    symbol is an OCC option symbol that is not the symbol of its only leg (a
    multi-leg structure is named by its underlying).

    An EQUITY proposal: it carries option legs; its symbol is an OCC option
    symbol (its notional would be counted without the contract multiplier,
    as if the contract were shares).
    """
    order = proposal.proposal
    if not proposal.is_option:
        problems = []
        if proposal.legs:
            problems.append(f"an equity proposal carries {len(proposal.legs)} option leg(s)")
        if is_occ(order.symbol):
            problems.append(
                f"the proposal is declared equity but its symbol {order.symbol} is an "
                "option contract"
            )
        return tuple(problems + _symbol_problems(proposal))
    if not proposal.legs:
        return ("the option proposal has no legs", *_symbol_problems(proposal))
    problems = []
    underlyings = sorted({underlying_of(leg.symbol) for leg in proposal.legs})
    if len(underlyings) > 1:
        problems.append(f"legs on more than one underlying ({', '.join(underlyings)})")
    own = underlying_of(order.symbol)
    if own not in underlyings:
        problems.append(
            f"the proposal is on {own} but its legs are on {', '.join(underlyings)}"
        )
    expirations = sorted({leg.expiration for leg in proposal.legs})
    if len(expirations) > 1:
        dates = ", ".join(expiration.isoformat() for expiration in expirations)
        problems.append(f"legs with different expirations ({dates})")
    for leg in proposal.legs:
        if not leg.quantity.is_integer():
            problems.append(
                f"leg {leg.symbol} quantity {leg.quantity:.10g} is not a whole number of contracts"
            )
    for leg in proposal.legs:
        mismatch = _leg_contract_problem(leg)
        if mismatch is not None:
            problems.append(mismatch)
    if is_occ(order.symbol):
        if len(proposal.legs) > 1:
            problems.append(
                f"the proposal is named by the option contract {order.symbol} but has "
                f"{len(proposal.legs)} legs: a multi-leg structure is named by its underlying"
            )
        elif not _same_symbol(order.symbol, proposal.legs[0].symbol):
            problems.append(
                f"the proposal is named by the option contract {order.symbol}, which is not "
                f"its leg's {proposal.legs[0].symbol}"
            )
    return tuple(problems + _symbol_problems(proposal))


def is_plain_equity(proposal: ProposalUnderReview) -> bool:
    """Whether the proposal is shares and nothing else: declared equity, no
    option legs, and a symbol that is a plain ticker — not an OCC option
    symbol, no whitespace and nothing outside ASCII in it. Anything else is
    option-like or unreadable, whatever its instrument says."""
    symbol = proposal.proposal.symbol
    return (
        proposal.is_equity
        and not proposal.legs
        and not is_occ(symbol)
        and not _has_whitespace(symbol)
        and symbol.isascii()
    )
