"""Phase 2 — pricing engine.

Pure math over ``aegis.data`` model types: Black-Scholes prices, analytic
Greeks, implied volatility, time-to-expiry, multi-leg position risk, and an
``enrich`` step that puts vendor and computed values side by side. Zero
network calls, zero broker access; every function is deterministic. See
``black_scholes`` for the modelling assumptions (European exercise, constant
vol and rate) and why results are approximations for American-style US
equity options.
"""

from aegis.pricing.errors import PricingError
from aegis.pricing.models import (
    EnrichedOption,
    Greeks,
    OptionLeg,
    PositionSummary,
    PremiumType,
    Side,
)

__all__ = [
    "EnrichedOption",
    "Greeks",
    "OptionLeg",
    "PositionSummary",
    "PremiumType",
    "PricingError",
    "Side",
]
