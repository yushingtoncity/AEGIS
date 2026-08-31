"""Error types for the data layer."""

from __future__ import annotations


class DataError(Exception):
    """An external data fetch failed.

    Carries what was being fetched and for which symbol so callers (CLIs
    especially) can print one clean line instead of a traceback.
    """

    def __init__(
        self,
        what: str,
        symbol: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.what = what
        self.symbol = symbol
        self.cause = cause
        target = f"{what} for {symbol}" if symbol else what
        detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
        super().__init__(f"failed to fetch {target}{detail}")
