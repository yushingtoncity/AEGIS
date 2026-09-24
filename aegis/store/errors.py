"""Error types for the persistence layer."""

from __future__ import annotations


class StoreError(Exception):
    """A store operation failed.

    Mirrors ``aegis.data.DataError``: carries what was being done and for
    which key (a record id, a database path, a migration name) so CLIs print
    one clean line instead of a traceback. Every ``sqlite3.Error`` that
    escapes the store is wrapped in one of these — callers never see a raw
    ``IntegrityError``.
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
        super().__init__(f"store operation failed: {target}{detail}")
