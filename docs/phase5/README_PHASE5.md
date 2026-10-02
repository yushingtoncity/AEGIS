## Policy engine (Phase 5)

`aegis/policy` is the deterministic gate between a proposal and any order —
the "deterministic code disposes" half of the governing principle. It is
pure Python with **no model calls, ever**: the same proposal and the same
context always produce the same verdict. Every proposal gets exactly one of
four verdicts, and `AUTO_EXECUTE` is unreachable unless every one of the
twenty rules passed.

It is also the only package that may ever reference an `Executor`.
`tests/test_policy_architecture.py` statically scans every module outside
`aegis/policy` and `aegis/execution` and fails on any import of
`aegis.execution` (in any form), any identifier containing `executor`, the
interface's order methods (`submit_order`, `cancel_order`, `close_position`),
or any string naming the package; a fresh-interpreter run then imports every
one of those modules and fails if any of them asked for the execution package
or holds a module, class or instance from it. The static scan is a tripwire,
not a sandbox — a deliberately obfuscated, dormant reference is beyond it —
which is why the runtime check exists. The same test pins the engine's
purity: nothing under `aegis/policy` imports `anthropic`, the brain, the
broker or a network library, and only `context.py` may read the clock or
fetch data.

### Inputs: the proposal and the context

`engine.evaluate(proposal, context, conn)` judges a stored proposal (with its
option legs) against a frozen `PolicyContext`. If a verdict depends on
something, it is a field of one of those two values — no rule reads a wall
clock, the network, the store or the config. `context.build_context` fills
the context from `aegis.data` and `aegis.store`:

- account state — equity, cash, buying power, options buying power — and open
  positions; start-of-day equity (Alpaca's `last_equity`) and today's P&L
- the market clock (`get_market_clock`: `is_open`, `next_open`, `next_close`)
- current quotes: the instrument's own, and one per leg for an option
- the pricing engine's expiry analysis of the structure at the proposal's
  price — max loss, max profit, breakevens, `unlimited_risk`
- from the store: open orders, orders sent today, the other proposals inside
  the duplicate window, and the control flags
- the limits (`risk_limits`), the watchlist and the contract multiplier

Tests build contexts by hand, so the whole suite runs with no network.

**Unknown is never good news.** A value that could not be fetched is `None`
with the reason recorded; NaN and infinity count as unknown; and a rule that
needs an unknown value rejects. A rule that raises is itself a REJECT, and the
engine still runs the rest.

### The twenty rules

Every rule is a pure function of `(proposal, context)` returning a name, an
outcome (`PASS`, `FLAG`, `ESCALATE` or `REJECT`) and a human-readable detail
with the numbers it compared. Every number comes from `risk_limits` in
`config.yaml`; nothing in `aegis/policy` hardcodes one. All twenty always
run — the engine never short-circuits — so the audit trail shows each.

| # | Rule | Trips when | Outcome |
| - | ---- | ---------- | ------- |
| 1 | `kill_switch` | the kill switch is on | REJECT |
| 2 | `halted` | `halt_until` is in the future (or unreadable) | REJECT |
| 3 | `market_hours` | the market is closed, or the clock is missing or stale | REJECT |
| 4 | `daily_loss_limit` | today's P&L ≤ −`daily_loss_limit_pct` of start-of-day equity — and this trip sets the halt | REJECT |
| 5 | `no_trade_list` | the symbol or its underlying is on `no_trade_list` | REJECT |
| 6 | `watchlist_only` | enabled and the underlying is not in the watchlist | REJECT |
| 7 | `invalidation_present` | the invalidation is empty | REJECT |
| 8 | `buying_power` | notional > available buying power (options: options buying power) | REJECT |
| 9 | `max_position_pct` | notional + existing exposure in the underlying > `max_position_pct` of equity | REJECT |
| 10 | `max_open_positions` | a new underlying while distinct underlyings held or pending ≥ `max_open_positions` | REJECT |
| 11 | `max_daily_trades` | orders sent today ≥ `max_daily_trades` | REJECT |
| 12 | `duplicate` | an earlier proposal with the same symbol, instrument and side within `duplicate_window_minutes`, or an open order on the instrument | REJECT |
| 13 | `limit_price_sanity` | a market order while `allow_market_orders` is false; a limit more than `limit_price_tolerance_pct` from the current mid; no mid to check against | REJECT |
| 14 | `options_min_dte` | any leg with fewer than `min_dte` days to expiration | REJECT |
| 15 | `options_max_loss` | max loss > `max_loss_per_trade`; **unlimited risk, unconditionally**; a structure that cannot be analysed or whose shape is inconsistent (see below) | REJECT |
| 16 | `options_max_contracts` | any leg quantity > `max_contracts` | REJECT |
| 17 | `options_escalate` | the instrument is an option — always (and anything option-like under an equity label) | ESCALATE |
| 18 | `short_sale` | an equity sell that does not close an existing long | ESCALATE (REJECT if `reject_short_sales`) |
| 19 | `min_confidence` | confidence < `min_confidence` | FLAG |
| 20 | `auto_tier` | anything but: `auto_execute.enabled`, a plain equity, a buy or a sell closing a long, a limit order, a watchlist symbol, notional ≤ `auto_execute.max_notional` | ESCALATE |

Definitions the rules share (`aegis/policy/measures.py`):

- **Notional** — equity: quantity × price (the limit, or the worst-case quote
  for a market order). Option bought for a debit: limit × the contract
  multiplier (100) × quantity. Option sold for a credit: the structure's max
  loss. Unlimited risk is an infinite notional.
- **Exposure** is measured per underlying: the market value of every position
  on the ticker (shares and contracts) plus the notional of open orders on it.
- **A position** is a distinct underlying held or pending, so a four-leg
  condor is one position, not four.
- **A closing sale** is an equity sell of no more than the long quantity
  held — and there must be one. It consumes no buying power, adds no exposure
  and no position, and can qualify for the auto tier. Every other rule still
  applies to it: the kill switch, a halt, a closed market, the loss limit and
  the trade cap stop a closing sale like anything else.
- **DTE** counts calendar days from the trading date (the date in the
  exchange's timezone) to expiration: 0 expires today.
- **Shape** — the engine does not trust a proposal to be what its label
  says. Every option leg's symbol must be the OCC symbol of the contract the
  leg describes (root, expiration, type, strike); all legs share one
  underlying and one expiration; a proposal named by a contract symbol has
  exactly that one leg; an equity proposal carries no legs and is not named
  by an option symbol; and no symbol contains whitespace (the padded OCC
  form included). Anything else is rejected by `options_max_loss`, and can
  never reach the auto tier.
- Comparisons carry a tiny float-noise tolerance (one part in a billion), so
  `10 × 100.1` is not "above" `1001.00`.

### Verdict precedence

After all twenty have run, the verdict is resolved in this order:

1. any `REJECT` → **`REJECT`**, with `failing_rule` set to the first rejecting rule
2. else any `FLAG` → **`FLAG_ONLY`** — recorded and flagged; nothing executes and nobody is asked
3. else any `ESCALATE` → **`NEEDS_APPROVAL`** — a human must approve
4. else → **`AUTO_EXECUTE`** — all twenty passed, the auto tier included

Options never auto-execute in this phase (`options_escalate`), and a result
set that is not exactly the twenty registered rules in order is itself a
REJECT (`engine_integrity`). `failing_rule` is the *first* rejecting rule in
the table's order, not the most specific one: a naked short call, whose
notional is unlimited, is recorded as failing `buying_power` (rule 8), with
`max_position_pct` and `options_max_loss` rejecting it further down the same
list. The decision is written through the store's
`record_decision` — exactly once per evaluation, with every rule's outcome
and detail in `rules_evaluated` — together with an event for every verdict
other than `AUTO_EXECUTE` (`policy_reject`, `policy_flag_only`,
`policy_needs_approval`), in the same transaction.

### Controls: the kill switch and the halt

Migration `0003_controls.sql` adds a `controls` table (`key`, `value`,
`updated_at`) with two rows that always exist:

| Key | Value | Meaning |
| --- | ----- | ------- |
| `kill_switch` | `on` / `off` | on: every proposal is rejected |
| `halt_until` | an ISO timestamp, or empty | trading is halted until that instant |

When `daily_loss_limit` trips, the engine sets `halt_until` to the next
market open and logs a `risk_limit_tripped` event in the same transaction.
Because the halt lives in the store, it survives a restart, and a P&L
recovery later in the day does not reopen trading — the `halted` rule keeps
rejecting until the next open. Two refinements keep that promise at the
edges: a trip *before* the day's session opens halts through that session's
close (otherwise the halt would expire at the opening bell and a recovery by
then would reopen trading the same day), and if the clock has no next open at
that moment the halt lasts `halt_fallback_hours`. A halt is only ever
extended: the write is a compare-and-set against the stored value, so a
later trip — or an evaluation working from an older snapshot — neither
shortens nor re-logs it.

The stop flags are read as late as possible. Building a context takes several
network calls, so the engine reads the kill switch and the halt from the
store once more just before it records a verdict and judges under the
stricter of the two readings: a switch set while a context was being built
is honoured, and nothing the context already holds is lifted.

The table defends itself: the key and value are CHECK-constrained, a trigger
refuses to delete either row, and the read fails closed — a missing
kill-switch row reads as ON and an unreadable halt reads as halted.

### Limits (`config.yaml` → `risk_limits`)

Every value is a labelled placeholder sized for the $100k paper account; live
sizing gets revisited before Phase 11.

| Key | Placeholder | Used by |
| --- | ----------- | ------- |
| `daily_loss_limit_pct` | 2.0 | `daily_loss_limit` |
| `halt_fallback_hours` | 24 | the halt, when the clock has no next open |
| `max_daily_trades` | 10 | `max_daily_trades` |
| `max_open_positions` | 5 | `max_open_positions` |
| `no_trade_list` | `[]` | `no_trade_list` |
| `watchlist_only` | true | `watchlist_only` |
| `max_position_pct` | 5.0 | `max_position_pct` |
| `duplicate_window_minutes` | 60 | `duplicate` |
| `allow_market_orders` | false | `limit_price_sanity` |
| `limit_price_tolerance_pct` | 5.0 | `limit_price_sanity` |
| `min_dte` | 1 | `options_min_dte` |
| `max_loss_per_trade` | 1000.0 | `options_max_loss` |
| `max_contracts` | 10 | `options_max_contracts` |
| `reject_short_sales` | false | `short_sale` |
| `min_confidence` | 0.5 | `min_confidence` |
| `auto_execute.enabled` | true | `auto_tier` |
| `auto_execute.max_notional` | 1000.0 | `auto_tier` |

`min_dte: 1` rejects same-day (0DTE) contracts and nothing else. The brain
currently reads the nearest expiration, usually 0–1 DTE on these tickers, so
a stricter value should be paired with a later chain on the brain side.

### Policy CLIs

```bash
python -m aegis.cli.policy evaluate PROPOSAL_ID   # judge one proposal and record the decision
python -m aegis.cli.policy evaluate --latest      # ...the most recent proposal without a decision
python -m aegis.cli.policy evaluate --latest --dry-run   # print the verdict and every rule; write nothing
python -m aegis.cli.policy kill on|off|status     # the kill switch (every toggle is logged as an event)
python -m aegis.cli.policy limits                 # each limit beside its consumption and headroom
```

`evaluate` prints the proposal, the verdict (and the failing rule for a
REJECT), all twenty rules with outcome and detail, then the context it judged
against (market state, account figures, data problems). It exits 0 whenever a
verdict was produced, whatever the verdict; evaluating a proposal that already
has a decision adds another one. `limits` prints the distance to the daily loss cap, positions
used, trades today, exposure per underlying and the halt state; it is a thin
formatter over `aegis.policy.limits.limits_report`, the same function the
Phase 8 dashboard will read. Both use live Alpaca data; neither `limits`,
`kill status` nor `--dry-run` ever creates or migrates a database. All take
`--db PATH`.
