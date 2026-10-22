"""Read what the store knows about past proposals, for grading. Reads only.

Joins each proposal with its legs, its latest policy verdict, the model the
thesis stage ran on (``reasoning.model_name``), the proposal stage's
``model_selection`` choice, and the cycle's ``market_snapshot`` (the spot,
whether it was stale, and the contract deltas, quotes and implied vols the
brain saw). A cycle whose snapshot event is missing or
no longer validates is graded without it; nothing here raises on old rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import NamedTuple

from aegis.brain.cycle import MARKET_SNAPSHOT_EVENT, snapshot_from_event
from aegis.brain.models import MarketSnapshot
from aegis.data.models import parse_occ_symbol
from aegis.scorecard.models import LegQuote, ProposalInputs
from aegis.store.models import Proposal, ProposalLeg, ReasoningStage
from aegis.store.repo import (
    get_cycle_events,
    get_cycle_reasoning,
    get_latest_verdict,
    get_proposal_legs,
    get_proposals_between,
)

MODEL_SELECTION_EVENT = "model_selection"  # written by aegis.brain.cycle.select_models


def underlying_of(proposal: Proposal, legs: tuple[ProposalLeg, ...]) -> str:
    """The stock or ETF whose price decides the call."""
    for symbol in (proposal.symbol, *(leg.symbol for leg in legs)):
        try:
            return parse_occ_symbol(symbol)[0]
        except ValueError:
            continue
    return proposal.symbol


def _snapshot(conn: sqlite3.Connection, cycle_id: str) -> MarketSnapshot | None:
    for event in reversed(get_cycle_events(conn, cycle_id, MARKET_SNAPSHOT_EVENT)):
        try:
            return snapshot_from_event(event)
        except ValueError:
            continue
    return None


def _proposal_model_source(conn: sqlite3.Connection, cycle_id: str) -> str | None:
    for event in reversed(get_cycle_events(conn, cycle_id, MODEL_SELECTION_EVENT)):
        stages = (event.payload or {}).get("stages")
        if not isinstance(stages, list):
            continue
        for stage in stages:
            if isinstance(stage, dict) and stage.get("stage") == ReasoningStage.PROPOSAL.value:
                source = stage.get("source")
                return str(source) if source is not None else None
    return None


def _thesis_model(conn: sqlite3.Connection, cycle_id: str) -> str | None:
    models = [
        row.model_name
        for row in get_cycle_reasoning(conn, cycle_id)
        if row.stage is ReasoningStage.THESIS and row.model_name
    ]
    return models[-1] if models else None


class _SnapshotFacts(NamedTuple):
    spot: float | None = None
    spot_time: datetime | None = None
    spot_stale: bool = True
    deltas: dict[str, float] = {}
    quotes: dict[str, LegQuote] = {}
    risk_free_rate: float | None = None


def _from_snapshot(
    snapshot: MarketSnapshot | None, underlying: str, legs: tuple[ProposalLeg, ...]
) -> _SnapshotFacts:
    """The underlying's spot (and how fresh it was) and each leg's stored delta
    and quote, as far as the snapshot has them."""
    if snapshot is None:
        return _SnapshotFacts()
    entry = next((s for s in snapshot.symbols if s.symbol == underlying), None)
    if entry is None:
        return _SnapshotFacts(risk_free_rate=snapshot.risk_free_rate)
    deltas: dict[str, float] = {}
    quotes: dict[str, LegQuote] = {}
    if entry.chain is not None:
        wanted = {leg.symbol for leg in legs}
        for contract in entry.chain.contracts:
            if contract.symbol not in wanted:
                continue
            if contract.greeks is not None:
                deltas[contract.symbol] = contract.greeks.delta
            quotes[contract.symbol] = LegQuote(
                bid=contract.bid, ask=contract.ask, implied_vol=contract.implied_vol
            )
    return _SnapshotFacts(
        spot=entry.spot,
        spot_time=entry.spot_time,
        spot_stale=entry.stale,
        deltas=deltas,
        quotes=quotes,
        risk_free_rate=snapshot.risk_free_rate,
    )


def gather(
    conn: sqlite3.Connection, since: datetime | None = None, until: datetime | None = None
) -> list[ProposalInputs]:
    """Every proposal in ``[since, until)``, oldest first, with what grading needs."""
    snapshots: dict[str, MarketSnapshot | None] = {}
    inputs = []
    for proposal in get_proposals_between(conn, since, until):
        legs = tuple(get_proposal_legs(conn, proposal.id))
        underlying = underlying_of(proposal, legs)
        if proposal.cycle_id not in snapshots:
            snapshots[proposal.cycle_id] = _snapshot(conn, proposal.cycle_id)
        facts = _from_snapshot(snapshots[proposal.cycle_id], underlying, legs)
        verdict = get_latest_verdict(conn, proposal.id)
        inputs.append(
            ProposalInputs(
                proposal_id=proposal.id,
                created_at=proposal.created_at,
                cycle_id=proposal.cycle_id,
                symbol=proposal.symbol,
                underlying=underlying,
                instrument=proposal.instrument,
                side=proposal.side,
                quantity=proposal.quantity,
                confidence=proposal.confidence,
                proposal_model=proposal.model_name,
                thesis_model=_thesis_model(conn, proposal.cycle_id),
                model_source=_proposal_model_source(conn, proposal.cycle_id),
                verdict=verdict.verdict.value if verdict is not None else None,
                legs=legs,
                leg_deltas=facts.deltas,
                snapshot_spot=facts.spot,
                spot_time=facts.spot_time,
                spot_stale=facts.spot_stale,
                leg_quotes=facts.quotes,
                risk_free_rate=facts.risk_free_rate,
            )
        )
    return inputs
