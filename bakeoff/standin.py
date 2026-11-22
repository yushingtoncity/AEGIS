"""A stand in proposal for a thesis candidate, judged by the real policy rules offline.

The thesis stage emits candidates, not sized orders, so the bakeoff sizes
each one mechanically and with no model call:

- the candidate is resolved against its snapshot with the brain's own
  ``resolve_offer`` (a candidate that names a contract the chain lacks, or
  an equity without a two sided quote, has no stand in, as in the live cycle);
- quantity 1, a limit order at the mid: the equity's ``(bid + ask) / 2``, or
  the structure's net mid from ``offer_net_quote``; side from the direction
  (equity) or the premium type (options: buy a debit, sell a credit).

``replay_context`` builds the ``PolicyContext`` from the stored snapshot, with
``now`` set to the snapshot's ``taken_at`` so quote freshness, market hours
and days to expiry are judged as of the original cycle, not today. Quote
times are rebuilt the same way: an equity quote at the snapshot's
``spot_time``, an option leg at ``taken_at`` minus its stored quote age.

The snapshot does not hold everything a live context reads. What is missing
is filled with a stated assumption (``REPLAY_ASSUMPTIONS``) and the report
prints those assumptions next to the rules they feed. The verdict is
therefore APPROXIMATE, and policy pass rate is a secondary metric.

``evaluate_candidate`` calls ``aegis.policy.engine.decide``, which is pure:
it writes nothing and places nothing. This module imports no order path.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from aegis.brain.errors import BrainError
from aegis.brain.models import Direction, InstrumentOffer, MarketSnapshot, ThesisCandidate
from aegis.brain.prompts import offer_net_quote
from aegis.brain.proposal import resolve_offer
from aegis.config import AegisConfig
from aegis.data.models import MarketClock, OptionSnapshot, Position, Quote
from aegis.policy import measures
from aegis.policy.context import structure_analysis
from aegis.policy.engine import decide
from aegis.policy.models import Evaluation, PolicyContext, ProposalUnderReview
from aegis.pricing.models import PremiumType
from aegis.store.models import Controls, Instrument, OrderSide, OrderType, Proposal, ProposalLeg

STANDIN_QUANTITY = 1
"""Fixed on purpose: the bakeoff compares ideas, not sizing. It almost never
trips a position cap or the daily loss halt, which the report says."""

STANDIN_PROMPT_VERSION = "bakeoff-standin-v1"
_NAMESPACE = uuid.UUID("6f5b2c1e-3d4a-4b8e-9c7f-0a1b2c3d4e5f")
"""Fixed namespace so a stand in's id is the same on every rerun of the same
(run, input, contestant, candidate)."""

REPLAY_ASSUMPTIONS: dict[str, str] = {
    "kill_switch": "controls assumed clear (kill switch off): the store's controls at the time are not replayed",
    "halted": "controls assumed clear (no daily loss halt in force)",
    "daily_loss_limit": "start of day equity assumed equal to the snapshot equity, so daily P&L is 0: the snapshot has no last_equity",
    "buying_power": "options buying power assumed equal to the snapshot's buying power: the snapshot has none of its own",
    "max_open_positions": "no open orders assumed: orders are not in the snapshot",
    "max_daily_trades": "no orders assumed sent that day",
    "duplicate": "no other proposals assumed inside the duplicate window: each stand in is judged alone",
    "quote_freshness": "the equity quote time is the snapshot's spot_time, which is the last trade time when there was a trade",
}
"""Rule name → the assumption that feeds it in a replay. These rules are not
really exercised by the bakeoff, whatever they return."""


class StandinResult(BaseModel):
    """One candidate's dry run: the stand in (when it resolved) and its verdict."""

    model_config = ConfigDict(frozen=True)

    candidate_index: int
    resolved: bool
    resolve_error: str | None = None
    proposal_id: str | None = None
    verdict: str | None = None
    failing_rule: str | None = None
    reason: str | None = None
    rule_outcomes: tuple[tuple[str, str], ...] = ()
    """(rule, outcome) for every registered rule, in registry order."""


def standin_id(run_id: str, input_id: str, contestant: str, candidate_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{run_id}|{input_id}|{contestant}|{candidate_index}"))


def _mid(bid: float | None, ask: float | None) -> float | None:
    return None if bid is None or ask is None else (bid + ask) / 2


def standin_proposal(
    offer: InstrumentOffer,
    *,
    proposal_id: str,
    cycle_id: str,
    created_at: datetime,
    model_name: str,
    raw_output: str,
) -> ProposalUnderReview:
    """The offer as a quantity 1 limit order at the mid, with its legs."""
    candidate = offer.candidate
    if candidate.instrument is Instrument.EQUITY:
        side = OrderSide.BUY if candidate.direction is Direction.BULLISH else OrderSide.SELL
        limit = _mid(offer.bid, offer.ask)
    else:
        net = offer_net_quote(offer)
        side = OrderSide.BUY if net.premium_type is PremiumType.DEBIT else OrderSide.SELL
        limit = net.mid
    proposal = Proposal(
        id=proposal_id,
        created_at=created_at,
        cycle_id=cycle_id,
        symbol=offer.legs[0].contract.symbol if len(offer.legs) == 1 else candidate.symbol,
        instrument=candidate.instrument,
        side=side,
        quantity=float(STANDIN_QUANTITY),
        order_type=OrderType.LIMIT,
        limit_price=limit,
        thesis=candidate.rationale,
        confidence=candidate.confidence,
        invalidation=candidate.invalidation,
        raw_model_output=raw_output,
        model_name=model_name,
        prompt_version=STANDIN_PROMPT_VERSION,
    )
    legs = tuple(
        ProposalLeg(
            proposal_id=proposal_id,
            leg_index=index,
            symbol=leg.contract.symbol,
            option_type=leg.contract.option_type,
            side=leg.side,
            quantity=float(STANDIN_QUANTITY * leg.quantity),
            strike=leg.contract.strike,
            expiration=leg.contract.expiration,
        )
        for index, leg in enumerate(offer.legs)
    )
    return ProposalUnderReview(proposal=proposal, legs=legs)


def _option_quote(offer_leg_symbol: str, snapshot: MarketSnapshot) -> OptionSnapshot | None:
    for symbol in snapshot.symbols:
        chain = symbol.chain
        if chain is None:
            continue
        for contract in chain.contracts:
            if contract.symbol.upper() != offer_leg_symbol.upper():
                continue
            age = contract.quote_age_seconds
            greeks = contract.greeks
            return OptionSnapshot(
                fetched_at=snapshot.taken_at,
                symbol=contract.symbol,
                underlying=chain.underlying,
                expiration=contract.expiration,
                strike=contract.strike,
                option_type=contract.option_type,
                bid=contract.bid,
                ask=contract.ask,
                last=contract.last,
                volume=contract.volume,
                open_interest=contract.open_interest,
                implied_vol=contract.implied_vol,
                delta=greeks.delta if greeks is not None else None,
                gamma=greeks.gamma if greeks is not None else None,
                theta=greeks.theta if greeks is not None else None,
                vega=greeks.vega if greeks is not None else None,
                rho=greeks.rho if greeks is not None else None,
                quote_time=None if age is None else snapshot.taken_at - timedelta(seconds=age),
            )
    return None


def replay_context(
    snapshot: MarketSnapshot, proposal: ProposalUnderReview, config: AegisConfig
) -> PolicyContext:
    """The policy context as of ``snapshot.taken_at``, from the snapshot and
    the ``REPLAY_ASSUMPTIONS``; ``errors`` names anything the snapshot lacks."""
    now = snapshot.taken_at
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    zone = ZoneInfo(config.pricing.expiry_timezone)
    age = snapshot.data_age
    errors: list[str] = []
    clock = None
    if age.market_open is not None:
        clock = MarketClock(
            fetched_at=now, is_open=age.market_open, next_open=age.next_open, next_close=age.next_close
        )
    else:
        errors.append("market clock: unknown in the snapshot")
    account = snapshot.account
    fields: dict[str, object] = {
        "now": now,
        "market_date": now.astimezone(zone).date(),
        "next_open_date": age.next_open.astimezone(zone).date() if age.next_open else None,
        "limits": config.risk_limits,
        "watchlist": tuple(config.watchlist),
        "contract_multiplier": config.pricing.contract_multiplier,
        "controls": Controls(),
        "clock": clock,
        "open_orders": (),
        "orders_today": 0,
        "recent_proposals": (),
    }
    if account is not None:
        fields.update(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            options_buying_power=account.buying_power,
            start_of_day_equity=account.equity,
            daily_pnl=0.0 if account.equity is not None else None,
            positions=tuple(
                Position(
                    symbol=p.symbol,
                    qty=p.qty,
                    side=p.side,
                    avg_entry_price=p.avg_entry_price,
                    market_value=p.market_value,
                    unrealized_pl=p.unrealized_pl,
                    current_price=p.current_price,
                )
                for p in account.positions
            ),
        )
    else:
        errors.append("account state: not in the snapshot")
    if proposal.is_equity:
        symbol = next((s for s in snapshot.symbols if s.symbol == proposal.proposal.symbol), None)
        if symbol is not None:
            fields["quote"] = Quote(
                fetched_at=now, symbol=symbol.symbol, bid=symbol.bid, ask=symbol.ask, quote_time=symbol.spot_time
            )
    else:
        quotes = [_option_quote(leg.symbol, snapshot) for leg in proposal.legs]
        fields["leg_quotes"] = tuple(q for q in quotes if q is not None)
        draft = PolicyContext(**fields)
        fields["analysis"] = structure_analysis(
            proposal,
            entry_price=measures.entry_price(proposal, draft),
            contract_multiplier=config.pricing.contract_multiplier,
        )
    return PolicyContext(**fields, errors=tuple(errors))


def _reason(evaluation: Evaluation) -> str:
    non_pass = evaluation.non_pass
    if not non_pass:
        return "every rule passed"
    first = next((r for r in non_pass if r.name == evaluation.failing_rule), non_pass[0])
    return f"{first.name}: {first.detail}"


def evaluate_candidates(
    candidates: Sequence[ThesisCandidate],
    snapshot: MarketSnapshot,
    config: AegisConfig,
    *,
    run_id: str,
    input_id: str,
    cycle_id: str,
    contestant: str,
    model_name: str,
    raw_output: str,
) -> list[StandinResult]:
    """A stand in and a dry run verdict for every candidate, in order."""
    results: list[StandinResult] = []
    for index, candidate in enumerate(candidates):
        try:
            offer = resolve_offer(
                candidate, snapshot, contract_multiplier=config.pricing.contract_multiplier
            )
        except BrainError as exc:
            reason = exc.cause if exc.cause is not None else exc
            results.append(StandinResult(candidate_index=index, resolved=False, resolve_error=str(reason)))
            continue
        pid = standin_id(run_id, input_id, contestant, index)
        under_review = standin_proposal(
            offer,
            proposal_id=pid,
            cycle_id=cycle_id,
            created_at=snapshot.taken_at,
            model_name=model_name,
            raw_output=raw_output,
        )
        evaluation = decide(under_review, replay_context(snapshot, under_review, config))
        results.append(
            StandinResult(
                candidate_index=index,
                resolved=True,
                proposal_id=pid,
                verdict=evaluation.verdict.value,
                failing_rule=evaluation.failing_rule,
                reason=_reason(evaluation),
                rule_outcomes=tuple((r.name, r.outcome.value) for r in evaluation.results),
            )
        )
    return results


def first_resolved(results: Sequence[StandinResult]) -> StandinResult | None:
    """The candidate the live cycle would have offered: the first that resolves."""
    return next((r for r in results if r.resolved), None)
