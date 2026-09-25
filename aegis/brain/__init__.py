"""Phase 4 — the agent brain: scan → thesis → proposal.

The brain has ZERO execution authority. It reads market data through
``aegis.data``, prices structures with ``aegis.pricing``, calls Claude
through ``aegis.brain.llm`` (the single place that touches the Anthropic
API), and its only outputs are rows in the store: one ``reasoning`` row per
stage and, when it has a compelling idea, one ``proposals`` row. Nothing
consumes proposals yet — the Phase 5 policy engine will gate every one of
them deterministically before any order can exist. Consequently no module
under ``aegis.brain`` may import ``aegis.execution``; ``tests/test_brain_
architecture.py`` fails the build if one does.

Declining to trade is a valid and often correct output: a cycle ends in a
``proposals`` row or a ``no_trade`` event, never in a forced trade. News
text is untrusted data and is delimited as such in every prompt.
"""

from aegis.brain.errors import BrainError, BudgetExceeded

__all__ = ["BrainError", "BudgetExceeded"]
