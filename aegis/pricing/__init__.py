"""Phase 2 — pricing engine (stub).

Will compute our own Greeks and implied volatility (Black-Scholes /
binomial) as the fallback when Alpaca omits them — common for near-dated,
deep-OTM, or unquoted contracts — plus fair-value estimates the brain can
cite in proposals. Pure functions over aegis.data models; no I/O and no
broker access.
"""
