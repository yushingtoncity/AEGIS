"""Stage 3 — proposal: a candidate resolved to concrete contracts in, a
``ProposalOutput`` (one ``TradeProposal`` or a no-trade with a reason) out.

``resolve_offer`` is the pure half: it turns a ``ThesisCandidate`` into an
``InstrumentOffer`` from the snapshot alone — for an option, the structure's
legs looked up in the symbol's chain (each leg quantity 1, the strikes
read low to high as the thesis stage listed them), priced by
``analyze_position`` at the contracts' mids; for an equity, the share
quote. A contract that is not in the chain, or one without a mid, is a
``BrainError`` keyed by the symbol — the cycle moves on to the next
candidate — and so is an equity without a two-sided quote: its limit price
could not be bounded (the IEX feed outside market hours keeps the last
trade but drops the bid/ask). A crossed quote (bid above ask), on the
equity or on any leg, is skipped the same way: its bounds would be an
empty interval. Leg semantics per structure (K1 < K2 < K3 < K4)::

    long_call            buy call K1
    long_put             buy put K1
    call_debit_spread    buy call K1, sell call K2
    put_debit_spread     buy put K2, sell put K1
    call_credit_spread   sell call K1, buy call K2
    put_credit_spread    sell put K2, buy put K1
    iron_condor          buy put K1, sell put K2, sell call K3, buy call K4

``run_proposal`` shows the model the offer, the pricing block, the account
and the data age (``build_proposal_prompt``) and lets it choose only what
the prompt says it may: quantity, order type and limit price — or decline.
``validate_proposal`` is the loop's ``post_validate`` and holds a trade to
the offer: the instrument and symbol are the offer's (a single-leg option
may be named by its OCC symbol or by the underlying — the prompt asks for
the OCC symbol, the store keys legs by contract either way), the legs match
the offered legs exactly and in order with one quantity throughout, an
option is a limit order whose side follows the premium type (buy a debit,
sell a credit), an equity's side follows the candidate's direction (bullish
buys, bearish sells), and any limit price — option or equity alike — sits
between the (net) bid and the (net) ask ``offer_net_quote`` states in the
prompt, widened by ``LIMIT_PRICE_SLACK`` of the spread on each side. The
rejection message states the bounds. An offer that cannot state them at all
(built by hand with a leg that has no bid/ask, or a crossed one;
``resolve_offer`` never builds one) has any limit price rejected outright,
with the message telling the model to decline (or, for an equity, to use a
market order): a limit the prompt could not bound would be held to nothing.
A trade must also not contradict itself — no ``no_trade_reason``, no limit
price on an equity market order — and must be storable: ``trade_records``
builds the store rows the cycle will write, and whatever they refuse (a
non-finite price, a lone surrogate in a text field) is fed back rather
than crashing the cycle after the stage succeeded. Model-authored values
quoted in a rejection are passed through ``neutralise_untrusted``.
``quantity >= 1`` is already the model's own rule.
"""

from __future__ import annotations

import math

from pydantic import ValidationError

from aegis.brain.errors import BrainError
from aegis.brain.llm import LLMClient
from aegis.brain.models import (
    ContractSummary,
    Direction,
    InstrumentOffer,
    LegQuote,
    MarketSnapshot,
    OptionStructure,
    ProposalOutput,
    ProposalResult,
    SymbolSnapshot,
    ThesisCandidate,
    TradeProposal,
)
from aegis.brain.prompts import (
    build_proposal_prompt,
    neutralise_untrusted,
    offer_net_quote,
    prompt_version,
    system_prompt,
)
from aegis.brain.schemas import OutputInvalid
from aegis.brain.stage import run_json_stage
from aegis.config import BrainConfig
from aegis.data.models import OptionType
from aegis.pricing.errors import PricingError
from aegis.pricing.models import OptionLeg, PremiumType, Side
from aegis.pricing.position import analyze_position
from aegis.store.models import (
    Instrument,
    OrderSide,
    OrderType,
    Proposal,
    ProposalLeg,
    ReasoningStage,
)

LIMIT_PRICE_SLACK = 0.10
"""Fraction of the net bid/ask spread the limit-price bounds are widened by, each side."""

STRIKE_TOLERANCE = 1e-6
PRICE_TOLERANCE = 1e-6
"""A limit price this close to a bound is on it (float noise in the bound arithmetic)."""

STRUCTURE_LEG_PLAN: dict[OptionStructure, tuple[tuple[int, OptionType, OrderSide], ...]] = {
    OptionStructure.LONG_CALL: ((0, OptionType.CALL, OrderSide.BUY),),
    OptionStructure.LONG_PUT: ((0, OptionType.PUT, OrderSide.BUY),),
    OptionStructure.CALL_DEBIT_SPREAD: (
        (0, OptionType.CALL, OrderSide.BUY),
        (1, OptionType.CALL, OrderSide.SELL),
    ),
    OptionStructure.PUT_DEBIT_SPREAD: (
        (1, OptionType.PUT, OrderSide.BUY),
        (0, OptionType.PUT, OrderSide.SELL),
    ),
    OptionStructure.CALL_CREDIT_SPREAD: (
        (0, OptionType.CALL, OrderSide.SELL),
        (1, OptionType.CALL, OrderSide.BUY),
    ),
    OptionStructure.PUT_CREDIT_SPREAD: (
        (1, OptionType.PUT, OrderSide.SELL),
        (0, OptionType.PUT, OrderSide.BUY),
    ),
    OptionStructure.IRON_CONDOR: (
        (0, OptionType.PUT, OrderSide.BUY),
        (1, OptionType.PUT, OrderSide.SELL),
        (2, OptionType.CALL, OrderSide.SELL),
        (3, OptionType.CALL, OrderSide.BUY),
    ),
}
"""(strike index low→high, option type, side) per leg, in the order the legs
are offered and stored — the mapping the module docstring spells out. The
thesis stage checks a candidate's strikes against it too, so each strike is
held to the option type its leg needs."""


def _plain(value: float) -> str:
    """A strike without trailing zeros: 640, 642.5."""
    return f"{value:g}"


def _price(value: float) -> str:
    """A price to the sub-cent, trailing zeros stripped: 4.19, 561.275 — exact
    enough that the stated bounds never contradict the check."""
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _quote(value: float | None) -> str:
    return "n/a" if value is None else _price(value)


# --- resolve --------------------------------------------------------------------


def _crossed(key: str, instrument: str, bid: float, ask: float) -> BrainError:
    """A crossed quote (bid above ask) is treated like a missing one: the
    limit-price bounds it would give are an empty interval, so no price the
    model could choose would pass — the candidate is skipped instead."""
    return BrainError(
        "resolve offer (crossed quote)",
        key,
        ValueError(
            f"{instrument} has a crossed quote (bid {_price(bid)} > ask {_price(ask)}) — "
            "a limit price could not be bounded"
        ),
    )


def _find_contract(
    symbol: SymbolSnapshot, option_type: OptionType, strike: float
) -> ContractSummary | None:
    chain = symbol.chain
    if chain is None:
        return None
    for contract in chain.contracts:
        if contract.option_type is not option_type:
            continue
        if abs(contract.strike - strike) <= STRIKE_TOLERANCE:
            return contract
    return None


def resolve_offer(
    candidate: ThesisCandidate, snapshot: MarketSnapshot, *, contract_multiplier: int
) -> InstrumentOffer:
    """The candidate as concrete, priced legs (or the equity quote) from the snapshot.

    ``BrainError("resolve offer", symbol)`` when the symbol is not in the
    snapshot, an equity has no spot, an option candidate names an
    expiration other than the chain's, a leg's contract is not in the chain
    or has no mid, or the pricing engine rejects the legs;
    ``BrainError("resolve offer (no bid/ask quote)", symbol)`` for an equity
    whose bid or ask is missing (like an option leg without a mid);
    ``BrainError("resolve offer (crossed quote)", symbol)`` when the equity's
    or any leg's bid is above its ask.
    """
    symbol = next((s for s in snapshot.symbols if s.symbol == candidate.symbol), None)
    if symbol is None:
        raise BrainError(
            "resolve offer", candidate.symbol, ValueError("symbol is not in the snapshot")
        )
    if candidate.instrument is Instrument.EQUITY:
        if symbol.spot is None:
            raise BrainError("resolve offer", candidate.symbol, ValueError("no spot price"))
        if symbol.bid is None or symbol.ask is None:
            raise BrainError(
                "resolve offer (no bid/ask quote)",
                candidate.symbol,
                ValueError(
                    f"no bid/ask quote (bid {_quote(symbol.bid)}, ask {_quote(symbol.ask)}) — "
                    "a limit price could not be bounded"
                ),
            )
        if symbol.bid > symbol.ask:
            raise _crossed(candidate.symbol, candidate.symbol, symbol.bid, symbol.ask)
        return InstrumentOffer(
            candidate=candidate,
            spot=symbol.spot,
            bid=symbol.bid,
            ask=symbol.ask,
            quote_age_seconds=symbol.spot_age_seconds,
            stale=symbol.stale,
        )

    chain = symbol.chain
    if chain is None or not chain.contracts:
        raise BrainError(
            "resolve offer", candidate.symbol, ValueError("no option chain in the snapshot")
        )
    if candidate.expiration != chain.expiration:
        raise BrainError(
            "resolve offer",
            candidate.symbol,
            ValueError(
                f"expiration {candidate.expiration} is not in the snapshot "
                f"(chain expiration {chain.expiration.isoformat()})"
            ),
        )
    assert candidate.structure is not None  # the model validator guarantees it for options
    legs: list[LegQuote] = []
    for strike_index, option_type, side in STRUCTURE_LEG_PLAN[candidate.structure]:
        strike = candidate.strikes[strike_index]
        contract = _find_contract(symbol, option_type, strike)
        if contract is None:
            raise BrainError(
                "resolve offer",
                candidate.symbol,
                ValueError(
                    f"no {option_type.value} at strike {_plain(strike)} "
                    f"exp {chain.expiration.isoformat()} in the chain"
                ),
            )
        if contract.mid is None:
            raise BrainError(
                "resolve offer",
                candidate.symbol,
                ValueError(f"{contract.symbol} has no mid price"),
            )
        if contract.bid is not None and contract.ask is not None and contract.bid > contract.ask:
            raise _crossed(candidate.symbol, contract.symbol, contract.bid, contract.ask)
        legs.append(LegQuote(contract=contract, side=side, quantity=1))

    try:
        pricing = analyze_position(
            [
                OptionLeg(
                    option_type=leg.contract.option_type,
                    side=Side.LONG if leg.side is OrderSide.BUY else Side.SHORT,
                    quantity=leg.quantity,
                    strike=leg.contract.strike,
                    expiry=leg.contract.expiration,
                    premium=leg.contract.mid or 0.0,
                    greeks=leg.contract.greeks,
                    symbol=leg.contract.symbol,
                )
                for leg in legs
            ],
            contract_multiplier=contract_multiplier,
        )
    except PricingError as exc:
        raise BrainError("resolve offer", candidate.symbol, exc) from exc
    net_mid = sum(
        (1 if leg.side is OrderSide.BUY else -1) * leg.quantity * (leg.contract.mid or 0.0)
        for leg in legs
    )
    return InstrumentOffer(
        candidate=candidate,
        spot=chain.spot if chain.spot is not None else symbol.spot,
        quote_age_seconds=chain.quote_age_seconds,
        stale=chain.stale,
        legs=tuple(legs),
        pricing=pricing,
        net_mid=net_mid,
    )


# --- validate -------------------------------------------------------------------


def accepted_symbols(offer: InstrumentOffer) -> tuple[str, ...]:
    """The ``proposal.symbol`` values a trade on this offer may carry: the
    ticker for an equity or a multi-leg structure; for a single leg, its
    OCC symbol (what the prompt asks for) or the underlying."""
    underlying = offer.candidate.symbol.upper()
    if len(offer.legs) == 1:
        return (offer.legs[0].contract.symbol.upper(), underlying)
    return (underlying,)


def limit_price_bounds(offer: InstrumentOffer) -> tuple[float, float] | None:
    """``[net bid, net ask]`` widened by ``LIMIT_PRICE_SLACK`` of the spread
    on each side, as positive per-share magnitudes; None when a leg (or the
    equity) lacks the quote to state them, or when the quote is crossed (bid
    above ask — an empty interval; ``resolve_offer`` never builds such an
    offer, but one built by hand is refused rather than held to nothing)."""
    net = offer_net_quote(offer)
    if net.bid is None or net.ask is None or net.bid > net.ask + PRICE_TOLERANCE:
        return None
    slack = LIMIT_PRICE_SLACK * max(0.0, net.ask - net.bid)
    return net.bid - slack, net.ask + slack


# The side an equity trade takes for the candidate's view; a neutral view has none.
_EQUITY_SIDE: dict[Direction, OrderSide] = {
    Direction.BULLISH: OrderSide.BUY,
    Direction.BEARISH: OrderSide.SELL,
}


def trade_records(
    cycle_id: str,
    trade: TradeProposal,
    *,
    raw_output: str,
    model_name: str,
    prompt_version: str,
) -> tuple[Proposal, list[ProposalLeg]]:
    """The store rows for a validated trade: the proposal (the verbatim model
    output as ``raw_output``) and one leg per ``LegSpec``, in order.

    Raises what the store records raise — a ``ValidationError`` (a non-finite
    price, say) or an ``OverflowError`` (a quantity no float can hold) — which
    is why ``validate_proposal`` builds them too, before the cycle has to.
    """
    proposal = Proposal(
        cycle_id=cycle_id,
        symbol=trade.symbol,
        instrument=trade.instrument,
        side=trade.side,
        quantity=float(trade.quantity),
        order_type=trade.order_type,
        limit_price=trade.limit_price,
        thesis=trade.thesis,
        confidence=trade.confidence,
        invalidation=trade.invalidation,
        raw_model_output=raw_output,
        model_name=model_name,
        prompt_version=prompt_version,
    )
    legs = [
        ProposalLeg(
            proposal_id=proposal.id,
            leg_index=index,
            symbol=leg.symbol,
            option_type=leg.option_type,
            side=leg.side,
            quantity=float(leg.quantity),
            strike=leg.strike,
            expiration=leg.expiration,
        )
        for index, leg in enumerate(trade.legs)
    ]
    return proposal, legs


def _storage_problems(trade: TradeProposal) -> list[str]:
    """What would stop ``trade`` from becoming store rows, as problems to feed
    back — local validation must agree with what the store accepts, or a
    trade that passed would crash the cycle instead of being corrected.
    The rows are built exactly as the cycle builds them, and every text
    field must encode as UTF-8 (SQLite cannot bind a lone surrogate)."""
    try:
        trade_records("(validation)", trade, raw_output="", model_name="", prompt_version="")
    except ValidationError as exc:
        return [
            f"proposal: the {exc.title} row would refuse it — "
            f"{'.'.join(str(part) for part in error['loc']) or 'root'}: "
            f"{neutralise_untrusted(error['msg'])}"
            for error in exc.errors()
        ]
    except (OverflowError, ValueError) as exc:
        detail = neutralise_untrusted(str(exc))
        return [f"proposal: cannot be stored ({type(exc).__name__}: {detail})"]
    texts = [
        ("symbol", trade.symbol),
        ("thesis", trade.thesis),
        ("invalidation", trade.invalidation),
        *((f"legs.{index}.symbol", leg.symbol) for index, leg in enumerate(trade.legs)),
    ]
    problems = []
    for field, text in texts:
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            problems.append(
                f"proposal.{field}: holds a lone UTF-16 surrogate, which is not a character"
            )
    return problems


def validate_proposal(output: ProposalOutput, offer: InstrumentOffer) -> None:
    """Hold a ``trade`` outcome to the offer (see the module docstring);
    ``OutputInvalid`` lists every problem as ``proposal.<field>: <expected>``.
    A ``no_trade`` outcome has nothing to check beyond its own validator.

    Beyond the offer, a trade must be self-consistent and storable: its
    ``no_trade_reason`` is null, an equity market order carries no limit
    price, an equity's side follows the candidate's direction (bullish buys,
    bearish sells), a limit price is a finite number, and — once everything
    else passes — the store rows ``trade_records`` builds from it accept it.
    Model-authored values quoted back (a symbol) are neutralised: the
    message is fed to the model as a user turn.
    """
    proposal = output.proposal
    if output.outcome != "trade" or proposal is None:
        return
    candidate = offer.candidate
    problems: list[str] = []

    if output.no_trade_reason is not None:
        problems.append(
            "no_trade_reason: must be null when outcome is 'trade' — a trade carries no reason "
            "not to trade; return outcome 'no_trade' (and proposal null) to decline"
        )
    if proposal.instrument is not candidate.instrument:
        problems.append(
            f"proposal.instrument: expected {candidate.instrument.value} (the offered "
            f"candidate), got {proposal.instrument.value}"
        )
    accepted = accepted_symbols(offer)
    if proposal.symbol.upper() not in accepted:
        problems.append(
            f"proposal.symbol: expected {' or '.join(accepted)}, "
            f"got {neutralise_untrusted(proposal.symbol)}"
        )

    if len(proposal.legs) != len(offer.legs):
        problems.append(
            f"proposal.legs: expected {len(offer.legs)} leg(s) exactly as offered, "
            f"got {len(proposal.legs)}"
        )
    else:
        for index, (leg, quote) in enumerate(zip(proposal.legs, offer.legs)):
            loc = f"proposal.legs.{index}"
            contract = quote.contract
            if leg.symbol.strip().upper() != contract.symbol.upper():
                problems.append(
                    f"{loc}.symbol: expected {contract.symbol}, "
                    f"got {neutralise_untrusted(leg.symbol)}"
                )
            if leg.option_type is not contract.option_type:
                problems.append(
                    f"{loc}.option_type: expected {contract.option_type.value}, "
                    f"got {leg.option_type.value}"
                )
            if leg.side is not quote.side:
                problems.append(f"{loc}.side: expected {quote.side.value}, got {leg.side.value}")
            if abs(leg.strike - contract.strike) > STRIKE_TOLERANCE:
                problems.append(
                    f"{loc}.strike: expected {_plain(contract.strike)}, got {_plain(leg.strike)}"
                )
            if leg.expiration != contract.expiration:
                problems.append(
                    f"{loc}.expiration: expected {contract.expiration.isoformat()}, "
                    f"got {leg.expiration.isoformat()}"
                )
            if leg.quantity != proposal.quantity:
                problems.append(
                    f"{loc}.quantity: expected {proposal.quantity} (the proposal's quantity, "
                    f"the same on every leg), got {leg.quantity}"
                )

    if proposal.instrument is Instrument.OPTION:
        if proposal.order_type is not OrderType.LIMIT:
            problems.append(
                "proposal.order_type: options must be limit orders, "
                f"got {proposal.order_type.value}"
            )
        net = offer_net_quote(offer)
        if offer.legs and net.premium_type is not None:
            side = OrderSide.BUY if net.premium_type is PremiumType.DEBIT else OrderSide.SELL
            if proposal.side is not side:
                problems.append(
                    f"proposal.side: the offer is a {net.premium_type.value} — side must be "
                    f"{side.value}, got {proposal.side.value}"
                )
    elif candidate.instrument is Instrument.EQUITY:
        want = _EQUITY_SIDE.get(candidate.direction)
        if want is None:
            problems.append(
                "proposal.side: the candidate's view is neutral, and an equity trade has no "
                "neutral side — return no_trade"
            )
        elif proposal.side is not want:
            problems.append(
                f"proposal.side: the candidate is {candidate.direction.value} — an equity's "
                f"side follows the direction: {want.value}, got {proposal.side.value}"
            )

    price = proposal.limit_price
    market_equity = (
        proposal.order_type is OrderType.MARKET and proposal.instrument is Instrument.EQUITY
    )
    if price is not None and market_equity:
        problems.append(
            f"proposal.limit_price: must be null for a market order, got {_price(price)} — "
            "set it to null, or make order_type limit"
        )
    elif price is not None and not math.isfinite(price):
        problems.append(f"proposal.limit_price: must be a finite number, got {price}")
    elif price is not None:
        bounds = limit_price_bounds(offer)
        if bounds is None:
            # No quote to hold the price to: an unbounded limit is not a rule.
            alternative = "return no_trade" if offer.legs else "return no_trade, or a market order"
            problems.append(
                f"proposal.limit_price: {_price(price)} cannot be checked — the offer shows no "
                f"bid/ask to state the bounds (the quote is unavailable); {alternative}"
            )
        else:
            low, high = bounds
            if price < low - PRICE_TOLERANCE or price > high + PRICE_TOLERANCE:
                net = offer_net_quote(offer)
                assert net.bid is not None and net.ask is not None  # bounds exist only with both
                quote = "the net bid" if offer.legs else "the bid"
                problems.append(
                    f"proposal.limit_price: {_price(price)} is outside "
                    f"[{_price(low)}, {_price(high)}] — {quote} {_price(net.bid)} / ask "
                    f"{_price(net.ask)} per share widened by {LIMIT_PRICE_SLACK:.0%} of the "
                    "spread on each side"
                )

    if not problems:
        problems = _storage_problems(proposal)
    if problems:
        raise OutputInvalid("output did not match the offer:\n" + "\n".join(problems))


# --- run ------------------------------------------------------------------------


def run_proposal(
    llm: LLMClient,
    offer: InstrumentOffer,
    snapshot: MarketSnapshot,
    config: BrainConfig,
    *,
    cycle_id: str | None = None,
) -> ProposalResult:
    """Run the proposal stage on ``offer`` with ``config.stages.proposal`` and
    ``config.max_retries``; the account and data age come from ``snapshot``."""
    output, usage, raw = run_json_stage(
        llm,
        stage=ReasoningStage.PROPOSAL,
        stage_config=config.stages.proposal,
        system=system_prompt(),
        user_prompt=build_proposal_prompt(offer, snapshot.account, snapshot.data_age),
        output_model=ProposalOutput,
        max_retries=config.max_retries,
        cycle_id=cycle_id,
        post_validate=lambda parsed: validate_proposal(parsed, offer),
    )
    return ProposalResult(
        output=output,
        offer=offer,
        usage=usage,
        raw_output=raw,
        prompt_version=prompt_version(ReasoningStage.PROPOSAL),
    )
