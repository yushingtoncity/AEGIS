"""Signals: external news and social context for the scan stage.

The brain's scan stage can be shown a short, ranked digest of recent news
headlines and Reddit mention counts about the tickers it watches.
This package decides what goes in that digest; the fetching itself lives in
``aegis/data`` (``news``, ``apewisdom``) like every external read.

- ``source``: the ``Source`` interface and one adapter per provider.
- ``ratelimit``: a per-source limiter whose daily count is read from the store.
- ``rank``: dedupe across sources, recency decay, ranking. Pure functions.
- ``digest``: strips links and markup, caps length, fits the token cap.
- ``collect``: one call per cycle. Fetches (or reuses a recent fetch),
  stores every item with its provenance, and returns the digest. A source
  that fails is recorded and skipped; it never fails the cycle.

Governance: signals are context, never authority. Their text is untrusted
and is rendered only inside the scan prompt's UNTRUSTED_NEWS block. Nothing
on the order path imports this package or reads its tables
(``tests/test_signals_architecture.py``), so the policy engine, the risk
limits, the no-trade list and the kill switch see exactly what they saw
before it existed.
"""
