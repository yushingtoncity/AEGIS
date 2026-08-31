"""AEGIS — an agentic trading platform.

Two principles are load-bearing everywhere in this codebase:

1. The LLM proposes, deterministic code disposes. The agent brain (Phase 4)
   emits structured trade proposals; a deterministic policy engine (Phase 5)
   gates them. Model output must never reach a broker API directly.
2. Execution is an abstraction. ``aegis.execution.base.Executor`` defines the
   broker interface; paper and live brokers are swappable implementations.

All external reads go through ``aegis.data`` and return typed pydantic
models, never raw API JSON. Tunables live in config.yaml; secrets in .env.
"""

__version__ = "0.1.0"
