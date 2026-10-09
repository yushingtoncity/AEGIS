"""The data layer — the only gateway to external market/account/news reads.

Everything outside this package consumes typed pydantic models from
``aegis.data.models``; raw API JSON never crosses this boundary. Every model
carries a ``fetched_at`` UTC timestamp so staleness is always visible (free
Alpaca options data is ~15 minutes delayed). A TTL cache sits in front of
all fetches; failures are wrapped in ``DataError`` with context.
"""

import warnings

from aegis.data.errors import DataError

# alpaca-py still imports ``websockets.legacy``, which websockets 14+
# deprecates at import time. That warning is the dependency's to act on, not
# ours, and every module that reaches Alpaca lives under this package — or,
# for ``aegis.execution.paper``, imports it before Alpaca — so it is silenced
# here, once, before any of them is imported, and a ``python -W error`` run
# of anything that touches the data layer stays clean. Nothing else is
# filtered.
warnings.filterwarnings(
    "ignore", message="websockets.legacy is deprecated", category=DeprecationWarning
)

__all__ = ["DataError"]
