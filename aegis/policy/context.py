"""Build the engine's two inputs: the proposal under review and its ``PolicyContext``.

This is the ONE module in ``aegis.policy`` that talks to the outside world.
It alone imports the live fetchers (``aegis.data.market`` / ``account``) and
it alone reads the clock; the rules, the measures, the engine and the limits
report see only the frozen values built here, which is what keeps a verdict
a pure function of (proposal, context). The fetchers are called through this
module's globals (``get_spot`` etc.), so a test replaces them with
``monkeypatch.setattr(aegis.policy.context, "get_spot", fake)`` and never
reaches Alpaca.

Where each field comes from:

- the STORE — the control flags, the open orders, the count of orders sent
  on the trading day (the exchange-local calendar day of ``now``) and the
  other proposals inside the duplicate window. A ``StoreError`` propagates:
  without the store no verdict could be recorded anyway. The control flags
  are the LAST thing read, after every data-layer call: those take seconds,
  and a kill switch set in the meantime belongs in the context.
- the DATA LAYER — the market clock (and from it ``next_open_date``, the
  exchange-local date its next open falls on), the account (equity, cash,
  buying power, positions; start-of-day equity is the broker's
  ``last_equity`` and today's P&L is ``equity - last_equity``), the proposal
  symbol's quote for an equity, and one option chain per distinct
  (underlying, expiration) of an option's legs, from which each leg's
  snapshot is picked by symbol.
- the PRICING ENGINE — ``structure_analysis``: the expiry analysis of an
  option structure at the proposal's own price.

Failure policy: unknown is never good news, and it is never fatal either. A
data-layer call that fails does not abort the build — its fields stay
``None`` and ``errors`` gains one line, ``"<what>: <error>"`` — so every
proposal still reaches the engine, where each rule that needs the missing
value REJECTs and says so. The same goes for a figure the broker did not
report and for an analysis that could not be computed: ``errors`` says why
each unknown is unknown.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from aegis.config import AegisConfig, ConfigError, get_config
from aegis.data.account import get_account_state
from aegis.data.errors import DataError
from aegis.data.market import get_market_clock, get_option_chain, get_spot
from aegis.data.models import ChainSnapshot, OptionSnapshot, utcnow
from aegis.policy import measures
from aegis.policy.errors import PolicyError
from aegis.policy.models import PolicyContext, ProposalUnderReview, underlying_of
from aegis.pricing.errors import PricingError
from aegis.pricing.models import OptionLeg, PositionSummary, Side
from aegis.pricing.position import analyze_position
from aegis.store.models import OrderSide, Proposal
from aegis.store.repo import (
    count_orders_submitted_between,
    get_controls,
    get_latest_undecided_proposal,
    get_open_orders,
    get_proposal,
    get_proposal_legs,
    get_proposals_since,
)

_FetchedT = TypeVar("_FetchedT")

# The account figures the context carries, by their ``AccountState`` names.
_ACCOUNT_FIGURES = ("equity", "last_equity", "cash", "buying_power", "options_buying_power")


# --- the proposal -------------------------------------------------------------


def _under_review(conn: sqlite3.Connection, proposal: Proposal) -> ProposalUnderReview:
    """A stored proposal paired with its legs (by leg_index; none for an equity)."""
    return ProposalUnderReview(
        proposal=proposal, legs=tuple(get_proposal_legs(conn, proposal.id))
    )


def load_proposal(conn: sqlite3.Connection, proposal_id: str) -> ProposalUnderReview:
    """The stored proposal ``proposal_id`` and its option legs, as the engine judges them.

    An id that does not exist is the caller's mistake: ``PolicyError``. A
    ``StoreError`` propagates.
    """
    proposal = get_proposal(conn, proposal_id)
    if proposal is None:
        raise PolicyError("load proposal (no such proposal)", proposal_id)
    return _under_review(conn, proposal)


def latest_undecided(conn: sqlite3.Connection) -> ProposalUnderReview | None:
    """The newest proposal with no policy decision yet, with its legs — None
    when every proposal has been judged (or there is none)."""
    proposal = get_latest_undecided_proposal(conn)
    return None if proposal is None else _under_review(conn, proposal)


# --- the structure's analysis -------------------------------------------------


def _analyse(
    proposal: ProposalUnderReview, entry_price: float | None, contract_multiplier: int
) -> tuple[PositionSummary | None, str]:
    """``structure_analysis`` plus, when there is none, the reason in words."""
    if not proposal.is_option:
        return None, "not an option"
    problems = measures.structure_problems(proposal)
    if problems:
        return None, "; ".join(problems)
    price = measures.known(entry_price)
    if price is None:
        return None, "the proposal's entry price is unknown"
    order = proposal.proposal
    # A net debit is carried by a bought leg, a net credit by a sold one.
    payer_side = OrderSide.BUY if price >= 0 else OrderSide.SELL
    payer = next((leg for leg in proposal.legs if leg.side is payer_side), None)
    if payer is None:
        paid = "debit" if payer_side is OrderSide.BUY else "credit"
        return None, f"the structure has no {payer_side.value} leg to carry a net {paid}"
    try:
        legs = [
            OptionLeg(
                option_type=leg.option_type,
                side=Side.LONG if leg.side is OrderSide.BUY else Side.SHORT,
                quantity=int(leg.quantity),
                strike=leg.strike,
                expiry=leg.expiration,
                premium=abs(price) * order.quantity / leg.quantity if leg is payer else 0.0,
                symbol=leg.symbol,
            )
            for leg in proposal.legs
        ]
        summary = analyze_position(legs, contract_multiplier=contract_multiplier)
        expected = price * order.quantity * contract_multiplier
    except (PricingError, ValueError, ArithmeticError) as exc:
        # ValueError: a leg the pricing model refuses; ArithmeticError: a
        # size no float can hold. Either way there is no analysis.
        return None, _one_line(exc, named=not isinstance(exc, PricingError))
    premium = measures.known(summary.net_premium)
    if (
        premium is None
        or measures.exceeds(premium, expected)
        or measures.below(premium, expected)
    ):
        return None, (
            f"the analysed net premium {measures.money(summary.net_premium)} is not the "
            f"proposal's {measures.money(expected)}"
        )
    return summary, ""


def structure_analysis(
    proposal: ProposalUnderReview, *, entry_price: float | None, contract_multiplier: int
) -> PositionSummary | None:
    """The pricing engine's expiry analysis of an option structure at the
    proposal's own price — what ``PolicyContext.analysis`` holds. Pure.

    ``entry_price`` is the signed net price per proposal unit, per share: a
    debit positive, a credit negative (``measures.entry_price``). The legs
    go to ``analyze_position`` as they are (BUY → long, SELL → short), but
    the premium is the PROPOSAL's price, not the leg mids: the whole net
    premium, ``entry_price × quantity``, is placed on one leg — the first
    bought leg for a debit, the first sold leg for a credit — and every
    other leg gets 0.0. The expiry payoff depends only on the net premium,
    so max loss, max profit and the breakevens are exact; the per-leg
    premiums in the returned summary are an allocation, not quotes.

    None — and a rule that needs the analysis then rejects — for an equity;
    when ``measures.structure_problems`` — the single judge of shape — finds
    anything (the legs are not one structure, or a leg's symbol is not the
    OCC symbol of the contract the leg describes, so the analysis would not
    be of what the order trades); when ``entry_price`` is unknown; when no
    leg of the needed side exists to carry the premium; when the pricing
    engine refuses the legs (``PricingError``); or when the summary's net
    premium is not ``entry_price × quantity × contract_multiplier``.
    """
    summary, _ = _analyse(proposal, entry_price, contract_multiplier)
    return summary


# --- the context --------------------------------------------------------------


def _one_line(error: BaseException, *, named: bool = False) -> str:
    """An error's message as one line (a validation error's spans several),
    after its type when ``named``; the type alone when it has no message."""
    lines = [line.strip() for line in str(error).splitlines()]
    message = " ".join(line for line in lines if line)
    name = type(error).__name__
    if not message:
        return name
    return f"{name}: {message}" if named else message


def _fetch(
    errors: list[str], what: str, fetcher: Callable[..., _FetchedT], *args: Any
) -> _FetchedT | None:
    """One guarded data-layer call: its result, or None with a line in ``errors``.

    ``DataError`` and ``ConfigError`` are how the data layer reports a
    failed fetch. Anything else a feed can throw (a payload that will not
    parse, say) leaves the value just as unknown, so it is recorded the same
    way, with its type — the build never aborts on a read.
    """
    try:
        return fetcher(*args)
    except (DataError, ConfigError) as exc:
        errors.append(f"{what}: {_one_line(exc)}")
    except Exception as exc:
        errors.append(f"{what}: {_one_line(exc, named=True)}")
    return None


def _aware_utc(value: datetime) -> datetime:
    """Naive is taken as UTC, any other offset converted — the store's convention."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _local_date(instant: datetime | None, zone: ZoneInfo) -> date | None:
    """The calendar date of ``instant`` in ``zone`` (a naive one is taken as
    UTC) — how ``next_open_date`` is read off the clock's next open, in the
    zone ``market_date`` is counted in. None without an instant, and for one
    at the edge of the calendar, which has no local date."""
    if instant is None:
        return None
    try:
        return _aware_utc(instant).astimezone(zone).date()
    except OverflowError:
        return None


def _trading_day(market_date: date, zone: ZoneInfo) -> tuple[datetime, datetime]:
    """The exchange-local calendar day ``market_date`` as UTC instants:
    ``[00:00 local, next 00:00 local)``. Each bound is its own local
    midnight, so a 23- or 25-hour daylight-saving day is still one day."""
    start = datetime.combine(market_date, time.min, tzinfo=zone)
    end = datetime.combine(market_date + timedelta(days=1), time.min, tzinfo=zone)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _window_start(anchor: datetime, minutes: int) -> datetime:
    """``anchor`` minus the duplicate window; the beginning of the calendar
    when the window reaches back further than a datetime can."""
    try:
        return anchor - timedelta(minutes=minutes)
    except OverflowError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _leg_quotes(proposal: ProposalUnderReview, errors: list[str]) -> tuple[OptionSnapshot, ...]:
    """Each leg's snapshot, in leg order, from its (underlying, expiration)
    chain — one fetch per distinct chain. A chain that cannot be fetched, or
    a leg its chain does not list, is a line in ``errors`` and no quote."""
    chains: dict[tuple[str, date], ChainSnapshot | None] = {}
    quotes: list[OptionSnapshot] = []
    quoted: set[str] = set()
    for leg in proposal.legs:
        key = underlying, expiration = underlying_of(leg.symbol), leg.expiration
        described = f"{underlying} {expiration.isoformat()}"
        if key not in chains:
            chains[key] = _fetch(
                errors, f"option chain for {described}", get_option_chain, underlying, expiration
            )
        chain = chains[key]
        if chain is None or leg.symbol in quoted:
            continue
        quoted.add(leg.symbol)
        snapshot = next(
            (c for c in chain.contracts if c.symbol.strip().upper() == leg.symbol), None
        )
        if snapshot is None:
            errors.append(f"leg quote for {leg.symbol}: not in the {described} option chain")
        else:
            quotes.append(snapshot)
    return tuple(quotes)


def build_context(
    conn: sqlite3.Connection,
    proposal: ProposalUnderReview | None = None,
    *,
    config: AegisConfig | None = None,
    now: datetime | None = None,
) -> PolicyContext:
    """Everything a verdict on ``proposal`` may depend on, read once and frozen.

    ``config`` defaults to ``get_config()`` and ``now`` to the wall clock (a
    naive ``now`` is taken as UTC). ``market_date`` is the calendar date at
    ``now`` in ``pricing.expiry_timezone`` — the trading day orders are
    counted on and DTE is counted from — and ``next_open_date`` the calendar
    date of the clock's next open in that same timezone (None without a
    clock or without a next open). Without a proposal the context holds
    the account-level picture only (what the limits report shows): no
    quote, no leg quotes, no analysis.

    Reads only: nothing is written to the store. A ``StoreError`` (or a
    ``ConfigError`` from loading the default config) propagates; a failed
    data-layer call does not — see the module docstring.
    """
    if config is None:
        config = get_config()
    if now is None:
        now = utcnow()
    elif not isinstance(now, datetime):
        raise PolicyError(f"build context (now must be a datetime, not {type(now).__name__})")
    now = _aware_utc(now)
    limits = config.risk_limits
    exchange_zone = ZoneInfo(config.pricing.expiry_timezone)
    market_date = now.astimezone(exchange_zone).date()

    # --- the store: a failure here propagates (the control flags are read
    # from it too — further down, once the data layer has answered) ---
    open_orders = tuple(get_open_orders(conn))
    day_start, day_end = _trading_day(market_date, exchange_zone)
    orders_today = count_orders_submitted_between(conn, day_start, day_end)
    # The duplicate rule looks back from the proposal's own creation, which
    # is earlier than now for a proposal judged late.
    anchor = now if proposal is None else min(now, proposal.proposal.created_at)
    recent = get_proposals_since(conn, _window_start(anchor, limits.duplicate_window_minutes))
    if proposal is not None:
        recent = [other for other in recent if other.id != proposal.id]

    # --- the data layer: a failure leaves None and a line in ``errors`` ---
    errors: list[str] = []
    fields: dict[str, Any] = {
        "now": now,
        "market_date": market_date,
        "limits": limits,
        "watchlist": tuple(config.watchlist),
        "contract_multiplier": config.pricing.contract_multiplier,
        "open_orders": open_orders,
        "orders_today": orders_today,
        "recent_proposals": tuple(recent),
    }
    clock = _fetch(errors, "market clock", get_market_clock)
    fields["clock"] = clock
    # The trading date the clock's next open falls on: equal to market_date
    # it tells ``daily_loss_limit`` the market is closed BEFORE today's
    # session, so a trip then halts through that session, not to its open.
    fields["next_open_date"] = _local_date(
        None if clock is None else clock.next_open, exchange_zone
    )
    state = _fetch(errors, "account state", get_account_state)
    if state is not None:
        figures = {name: measures.known(getattr(state, name)) for name in _ACCOUNT_FIGURES}
        missing = [name for name, value in figures.items() if value is None]
        if missing:
            errors.append(f"account state: the broker reported no usable {', '.join(missing)}")
        equity, last_equity = figures["equity"], figures["last_equity"]
        fields.update(
            equity=equity,
            cash=figures["cash"],
            buying_power=figures["buying_power"],
            options_buying_power=figures["options_buying_power"],
            start_of_day_equity=last_equity,
            daily_pnl=(
                None
                if equity is None or last_equity is None
                else measures.known(equity - last_equity)
            ),
            positions=tuple(state.positions),
        )
    if proposal is not None and proposal.is_equity:
        symbol = proposal.proposal.symbol
        fields["quote"] = _fetch(errors, f"quote for {symbol}", get_spot, symbol)
    elif proposal is not None:
        fields["leg_quotes"] = _leg_quotes(proposal, errors)
    # The stop flags are read LAST, after every network call above: the
    # fetches take seconds, and a kill switch set meanwhile must be in the
    # context. (The engine reads them once more before it records a verdict.)
    fields["controls"] = get_controls(conn)
    if proposal is not None and not proposal.is_equity:
        # The analysis is priced from the context so far: a market order's
        # entry price is its worst-case fill at the leg quotes just fetched.
        draft = PolicyContext(**fields)
        analysis, why_not = _analyse(
            proposal, measures.entry_price(proposal, draft), config.pricing.contract_multiplier
        )
        if analysis is None:
            errors.append(f"position analysis: {why_not}")
        fields["analysis"] = analysis
    return PolicyContext(**fields, errors=tuple(errors))
