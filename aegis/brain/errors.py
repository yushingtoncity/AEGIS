"""Error types for the agent brain."""

from __future__ import annotations


class BrainError(Exception):
    """A brain stage could not produce its output.

    Raised for an LLM call that failed after retries (or was refused), model
    output that never validated within ``max_retries``, a budget halt, or
    input that cannot be turned into a prompt. Mirrors ``DataError`` /
    ``StoreError``: carries what was being done and for which key (a stage,
    a cycle id, a symbol) so CLIs print one clean line.
    """

    def __init__(
        self,
        what: str,
        key: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.what = what
        self.key = key
        self.cause = cause
        target = f"{what} for {key}" if key else what
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        super().__init__(f"brain failed: {target}{detail}")


class BudgetExceeded(BrainError):
    """The token budget guard refused a call (a ``budget_halt`` event was logged)."""
