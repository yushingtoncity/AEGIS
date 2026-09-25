<!-- version: thesis-v1 -->
# Thesis stage

Decide whether the market brief below supports any trade idea and, if it does, describe the best few as ThesisCandidate entries. Everything you know about the market is in the brief and the contract list: you are not shown quotes here, and you must not invent them. The snapshot behind the brief was taken at $taken_at UTC; $market.

## Market brief (from the scan stage)

$brief

## Open positions

$positions

## Risk limits (read-only)

$risk_limits

The policy engine enforces these; propose within them. A candidate that would breach a limit is wasted work, and a symbol on the no-trade list must not appear at all.

## Recent proposals

$recent_proposals

Do not repeat these unless something changed — a new catalyst, a materially different price, a different structure. If all you would do is re-propose an earlier idea, say so in no_idea_reason instead.

## Available contracts

$contracts

Option candidates must use exactly one of these expirations and only strikes from the matching list — no other expiration, no strike that is not listed, no interpolation between strikes. Equity candidates need no expiration, structure or strikes.

## Structures

Strikes are listed low to high; K1 is the lowest strike.

$structures

## Task

Produce the ThesisOutput JSON object:

- candidates: zero to $max_candidates entries, best first, each with
  - symbol: a watchlist symbol from the brief
  - direction: "bullish", "bearish" or "neutral" — it must agree with the structure's view listed above; an equity candidate is bullish (buy shares) or bearish (sell shares), never neutral
  - instrument: "equity" or "option"
  - structure: one of the structure names above for an option; null for equity
  - expiration: the chain expiration (YYYY-MM-DD) for an option; null for equity
  - strikes: exactly the number of strikes the structure needs, listed low to high, every one from the available list; an empty list for equity
  - rationale: why this trade, in the numbers from the brief — levels, implied vol, days to expiry, the catalyst
  - confidence: 0 to 1, calibrated — 0.5 means a coin flip; stale data and thin liquidity lower it
  - key_risk: the single thing most likely to make this lose money
  - invalidation: a falsifiable condition — a price level, a date or an event — that would prove the idea wrong
- no_idea_reason: when candidates is empty, why (a closed or stale market, no edge at these implied vols, everything already proposed, positions full); null otherwise

An empty candidates list with a no_idea_reason is a respectable answer — the system expects it most of the time. Do not manufacture a candidate to fill the list.
