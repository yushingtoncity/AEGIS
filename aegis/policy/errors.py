"""Error types for the policy engine."""

from __future__ import annotations


class PolicyError(Exception):
    """The policy engine was asked something it cannot answer.

    Raised only for a caller's mistake — a proposal id that does not exist,
    legs that belong to another proposal. It is never how a proposal is
    turned down: every evaluation of a real proposal ends in a verdict, and
    missing or broken data is a REJECT with the reason in the rule's detail,
    not an exception. Mirrors ``DataError`` / ``StoreError`` / ``BrainError``:
    carries what was being done and for which key so CLIs print one clean
    line.
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
        super().__init__(f"policy failed: {target}{detail}")
