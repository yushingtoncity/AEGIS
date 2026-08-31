"""Phase 4 — the agent brain (stub).

Will assemble market context from aegis.data, prompt Claude, and parse its
output into structured TradeProposal objects. The brain only ever
PROPOSES: its entire output surface is typed proposals handed to
aegis.policy for deterministic vetting. It holds no broker credentials and
must never import aegis.execution.
"""
