"""Error types for the pricing engine."""

from __future__ import annotations


class PricingError(Exception):
    """A pricing computation was asked something it cannot answer.

    Raised only for malformed input (a position mixing expiries, a leg with
    a negative quantity, a snapshot with no strike). Numerical edge cases —
    zero time to expiry, zero volatility, prices outside the no-arbitrage
    band — are handled by returning intrinsic values or ``None``, never by
    raising. Mirrors ``aegis.data.DataError``: carries what was being
    computed and for which contract so CLIs print one clean line.
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
        super().__init__(f"cannot compute {target}{detail}")
