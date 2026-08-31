"""The broker execution boundary. Interface only — Phase 6 implements it.

AEGIS's core safety invariant lives at this seam:

    The LLM proposes; deterministic code disposes.

The agent brain (Phase 4) emits structured TradeProposal objects. The
deterministic policy engine (Phase 5) validates them against the
risk_limits block of config.yaml and converts survivors into approved
orders. ONLY the policy engine may hand an order to an Executor — no code
path may ever route model output to a broker API directly. Treat any
import of aegis.execution outside aegis.policy as a review-blocking defect.

Executor is deliberately broker-agnostic so implementations are swappable:
Phase 6 ships a PaperExecutor over Alpaca paper trading; a live-broker
executor, if one ever exists, is a separate implementation behind this same
interface, selected by configuration — never by code changes elsewhere.

The ``Any`` placeholders below become typed models when Phase 5 defines
ApprovedOrder and Phase 6 defines OrderReceipt/ExecutionError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Executor(ABC):
    """Abstract order-execution interface for one brokerage account."""

    @abstractmethod
    def submit_order(self, approved_order: Any) -> Any:
        """Submit a single policy-approved order to the broker.

        Args:
            approved_order: An ApprovedOrder produced by the Phase 5 policy
                engine. Implementations must verify the approval marker and
                reject anything else — defense in depth, in case a future
                caller violates the policy-only rule.

        Returns:
            A typed broker receipt (order id, status, fill details).

        Raises:
            ExecutionError (Phase 6): on broker rejection or transport
                failure. Implementations never retry silently; the decision
                to retry belongs to the caller, on the record.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a previously submitted order by its broker order id.

        Cancelling an already-filled or unknown order raises
        ExecutionError; cancelling an already-cancelled order is a no-op.
        """

    @abstractmethod
    def get_open_orders(self) -> list[Any]:
        """Return all currently open orders for the account as typed models.

        Used by the policy engine to enforce max_open_positions and to
        reconcile state after restarts.
        """

    @abstractmethod
    def close_position(self, symbol: str) -> Any:
        """Flatten the entire position in ``symbol`` and return the receipt.

        The policy engine's emergency lever (e.g. daily_loss_limit_pct
        breach); must work even when the account is otherwise restricted
        from opening new positions.
        """
