<!-- version: proposal-v1 -->
# Proposal stage

The thesis stage's best candidate has been resolved to concrete contracts and priced. Decide whether to propose it and, if so, size it and price it. The legs are fixed: you may choose the quantity, the order type and the limit price, or return no_trade with a reason.

## Candidate

$candidate

## Offer

$offer

## Pricing (one unit of the structure, held to expiry)

$pricing

## Account

$account

## Data age

$data_age

## Rules

- The legs are fixed. Copy them into the proposal exactly as listed — the same symbol, type, side, strike and expiration, in the same order — and give every leg the same quantity as the proposal's quantity. Changing, adding or dropping a leg is a rejection.
- Options must be limit orders (order_type "limit"). Equities may be "market" or "limit".
- limit_price is the net per-share price of one unit of the structure, as a positive number, stated between the net bid and the net ask shown above — say in the thesis which price you chose and why. The mid is a reasonable default; nearer the far side fills faster and costs more. For a DEBIT the proposal's side is "buy" (you pay it); for a CREDIT the side is "sell" (you receive it). For an equity the limit price is the share price and the side follows the direction.
- If the offer shows the bid or ask as n/a, there is no quote to check a limit price against and a limit order is rejected: return no_trade, or for an equity a market order if the idea survives without a quote.
- quantity is whole contracts per leg, or whole shares, at least 1. Size against the account: the policy engine caps any single position at a small percentage of equity, so a quantity that commits more than a few percent of equity to the max loss will be rejected.
- thesis restates the idea in one or two paragraphs with the concrete numbers from this prompt. invalidation restates a falsifiable condition — a price level, a date or an event — that, if it occurs, means the position should be closed.
- Return no_trade when the pricing does not justify the idea (a spread that eats the edge, a max loss out of proportion to the payoff, a breakeven too far away), when the quotes are marked [STALE] and the market is closed, or when the candidate no longer looks right on these numbers. Say why.

## Task

Produce the ProposalOutput JSON object:

- outcome: "trade" or "no_trade"
- proposal: for "trade", the object {symbol, instrument, side, quantity, order_type, limit_price, thesis, confidence, invalidation, legs}; null for "no_trade"
  - symbol: exactly the "proposal symbol" shown in the offer — the OCC symbol for a single-leg option, the underlying ticker for a multi-leg structure, the ticker for an equity
  - legs: one entry per offered leg, in order, each {symbol (the OCC symbol), option_type ("call"/"put"), side ("buy"/"sell"), quantity, strike, expiration (YYYY-MM-DD)}; an empty list for an equity
  - confidence: 0 to 1
- no_trade_reason: for "no_trade", the reason; null for "trade"
