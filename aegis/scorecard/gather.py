"""Read what the store knows about past proposals, for grading. Reads only.

Joins each proposal with its legs, its latest policy verdict, the model the
thesis stage ran on (``reasoning.model_name``), the proposal stage's
``model_selection`` choice, and the cycle's ``market_snapshot`` (the spot and
the contract deltas the brain saw). A cycle whose snapshot event is missing or
no longer validates is graded without it; nothing here raises on old rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from aegis.brain.cycle import MARKET_SNAPSHOT_EVENT, snapshot_from_event
from aegis.brain.models import MarketSnapshot
from aegis.data.models import parse_occ_symbol
from aegis.scorecard.models import ProposalInputs
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


def _from_snapshot(
    snapshot: MarketSnapshot | None, underlying: str, legs: tuple[ProposalLeg, ...]
) -> tuple[float | None, dict[str, float]]:
    """The underlying's spot and each leg's stored delta, as far as the snapshot has them."""
    if snapshot is None:
        return None, {}
    entry = next((s for s in snapshot.symbols if s.symbol == underlying), None)
    if entry is None:
        return None, {}
    deltas: dict[str, float] = {}
    if entry.chain is not None:
        wanted = {leg.symbol for leg in legs}
        for contract in entry.chain.contracts:
            if contract.symbol in wanted and contract.greeks is not None:
                deltas[contract.symbol] = contract.greeks.delta
    return entry.spot, deltas


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
        spot, deltas = _from_snapshot(snapshots[proposal.cycle_id], underlying, legs)
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
                leg_deltas=deltas,
                snapshot_spot=spot,
            )
        )
    return inputs
