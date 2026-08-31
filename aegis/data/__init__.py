"""The data layer — the only gateway to external market/account/news reads.

Everything outside this package consumes typed pydantic models from
``aegis.data.models``; raw API JSON never crosses this boundary. Every model
carries a ``fetched_at`` UTC timestamp so staleness is always visible (free
Alpaca options data is ~15 minutes delayed). A TTL cache sits in front of
all fetches; failures are wrapped in ``DataError`` with context.
"""

from aegis.data.errors import DataError

__all__ = ["DataError"]
