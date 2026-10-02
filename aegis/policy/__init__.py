"""Phase 5 — the policy engine: the deterministic gate between a proposal and any order.

The LLM proposes; this package disposes. Everything here is pure,
deterministic Python — no model calls, ever: the same proposal and the same
``PolicyContext`` always produce the same verdict. ``engine.evaluate`` runs
all twenty rules in ``rules`` on every proposal (never short-circuiting, so
the audit trail shows each one), resolves exactly one verdict by precedence
— REJECT, then FLAG_ONLY, then NEEDS_APPROVAL, else AUTO_EXECUTE — and
records it through the store. AUTO_EXECUTE is unreachable unless every rule
passed, the auto tier included. Unknown data is never good news: a rule that
cannot see what it needs rejects.

Every limit comes from ``risk_limits`` in config.yaml; none is hardcoded.
The kill switch and the daily-loss halt live in the store's ``controls``
table, so they survive a restart.

This is also the only package that may ever reference an ``Executor``
(``aegis.execution``) — ``tests/test_policy_architecture.py`` fails the
build if any other module does.
"""

from aegis.policy.errors import PolicyError

__all__ = ["PolicyError"]
