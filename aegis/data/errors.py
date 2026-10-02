"""Error types for the data layer."""

from __future__ import annotations

from datetime import date


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


class NoEligibleExpiration(DataError):
    """No listed expiration passed the caller's test, so no chain was fetched.

    Not a failed fetch: the listing was read and none of its dates qualified.
    ``listed`` holds every active expiration (nearest first) for the
    caller's warning. The message keeps ``DataError``'s shape so a caller
    that only catches ``DataError`` still prints one clean line.
    """

    def __init__(self, symbol: str, listed: tuple[date, ...] = ()) -> None:
        self.listed = listed
        super().__init__("option chain (no eligible expiration)", symbol)
