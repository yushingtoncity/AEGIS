"""The broker execution boundary.

AEGIS's core safety invariant lives at this seam:

    The LLM proposes; deterministic code disposes.

The agent brain (Phase 4) emits structured proposals. The deterministic
policy engine (Phase 5) judges them against the risk_limits block of
config.yaml, and only an order it authorised becomes an ApprovedOrder.
ONLY the policy engine may hand an order to an Executor; no code path may
ever route model output to a broker API directly. Any import of this
package outside aegis.policy is a review-blocking defect, and
tests/test_policy_architecture.py fails on one.

Executor is deliberately broker-agnostic so implementations are swappable.
Phase 6 ships PaperExecutor (aegis/execution/paper.py) over the Alpaca
paper account. A live-broker implementation, if one ever exists, is a
separate class behind this same interface, chosen by configuration, never
by code changes elsewhere.

The models are imported for the annotations only, and relatively, both on
purpose. Relatively: an absolute import would put the package's own name in
this file as an identifier, and tests/test_policy_architecture.py holds this
file to naming the Executor class and nothing else. For the annotations only
(under TYPE_CHECKING, never run): tests/test_brain_architecture.py proves its
harness catches this very file loaded by path, outside any package, so the
file must still run on its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ApprovedOrder, OrderReceipt


class Executor(ABC):
    """Order execution for one brokerage account.

    Every method raises ``ExecutionError`` on failure, with an ``outcome``
    that says what the failed call left behind at the broker. Nothing
    retries on its own: whether to try again is the caller's decision, on
    the record.
    """

    @abstractmethod
    def submit_order(self, order: ApprovedOrder) -> OrderReceipt:
        """Send one policy-approved order to the broker, exactly once.

        Before anything leaves the process, an implementation verifies the
        order against the store: the row ``claim_order`` wrote for it must
        exist, match it field for field, and be unsent, and the controls
        must still be clear. Anything else fails with outcome ``not_sent``
        and no request at all, in case a future caller breaks the
        policy-only rule.
        """

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> None:
        """Ask the broker to cancel one order by its broker order id.

        Cancelling an order the broker already cancelled (or let expire) is
        a no-op. Cancelling a filled or unknown order raises
        ``ExecutionError``. A cancel request that is accepted may still lose
        the race with a fill: read the order back to learn which won.
        """

    @abstractmethod
    def get_open_orders(self) -> list[OrderReceipt]:
        """Every order the broker holds open for the account, AEGIS's or not.

        Used to reconcile the store with the broker, and to find any order
        the store does not know.
        """

    @abstractmethod
    def get_order(self, client_order_id: str) -> OrderReceipt | None:
        """The broker's receipt for the order named ``client_order_id``, or
        None when the broker has no such order. A read: it changes nothing.

        How a caller learns what became of an order whose submission ended
        with an unknown outcome, and how it follows an order to its end.
        """

    @abstractmethod
    def close_position(self, symbol: str) -> OrderReceipt:
        """Flatten the entire position in ``symbol`` and return the receipt
        of the order that does it.

        The emergency lever. Phase 6 implements it and calls it from
        nowhere (spec decision D5): a daily-loss trip halts new trading but
        does not flatten.
        """
