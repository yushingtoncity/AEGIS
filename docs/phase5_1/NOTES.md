# Phase 5.1: the expiry floor and quote freshness

Two follow-ups from the Phase 5 review, on branch
`phase-5-1-dte-and-quote-age`. The prompt that asked for them is the
explicit approval CLAUDE.md invariant 3 requires for the `risk_limits` change
and the new rule. Nothing else in `risk_limits` or in the existing twenty
rules changed meaning.

## What changed

### 1. The options expiry floor

**Why.** `risk_limits.min_dte` was 1 and the brain's snapshot read the
nearest expiration, usually 0 to 1 DTE on these tickers. So the brain lived in
the gamma-heavy end of the chain by construction, and raising the floor alone
would have made every option idea die at `options_min_dte`.

**What.**

- `risk_limits.min_dte` goes from 1 to 7 (config.yaml and the code default,
  which a test keeps equal).
- Each symbol's chain now comes from the **nearest listed expiration whose DTE
  is at least `risk_limits.min_dte` and at most `brain.snapshot.max_dte`**
  (new, default 45). The floor is read from `risk_limits.min_dte` directly in
  `aegis/brain/snapshot.py` (`_dte_window`), so there is one number for both
  the brain and the policy engine.
- DTE is the pricing engine's calendar-day convention,
  `aegis.pricing.time_to_expiry.calendar_days_to_expiry` (fractional calendar
  days to the 16:00 New York close on expiration day), rounded down to whole
  days (`whole_days_to_expiry`). Rounding down is what keeps the brain from
  ever being looser than the policy engine: through the session the count
  matches the policy's DTE (calendar days from the trading date), after the
  close it is one less, and it is never more. A test walks every two hours of
  2026, both daylight-saving changes included, to show it.
- An expiration whose close has passed is never eligible, whatever the floor.
  The seeded property test that holds the brain's pick against the real
  `options_min_dte` rule found this one: with `min_dte: 0`, a contract that
  expired yesterday read as 0 DTE (the pricing helper clamps at zero) and was
  "eligible", while the policy saw -1 DTE. Fixed in `in_dte_window`.
- No expiration in the window: the symbol's errors get
  `no eligible expiration (7-45 DTE, risk_limits.min_dte, brain.snapshot.max_dte): none of the N listed expiration(s) is in the window; option chain omitted`,
  which the data-age warnings carry into the brief. The symbol has no chain,
  the cycle goes on, and the old `option chain missing` line is not added for
  it (it would read as a failed fetch). A nearer expiration is never used in
  its place.
- The snapshot also checks the chain it gets back: a chain outside the window
  is refused with `... the chain fetched expires YYYY-MM-DD, at N DTE; option
  chain omitted`.
- The thesis stage only ever sees the one chain per symbol, and
  `validate_candidates` already refuses an option candidate on any other
  expiration, so every candidate comes from the eligible expiration.

**Where the selection lives, and why there.** The brain decides which dates
qualify; the data layer fetches. `aegis.data.market.get_option_chain` gained
a keyword-only `eligible: Callable[[date], bool]`: with no explicit
expiration it reads the listing, fetches the nearest date `eligible` accepts,
and raises `NoEligibleExpiration` (a `DataError` subclass carrying the
listing) when it accepts none. Every other caller is unchanged.

The obvious alternative, having the brain call `list_expirations` itself,
would have broken `tests/test_brain_architecture.py`. Its fresh-interpreter
harness replaces exactly six fetchers on the snapshot module and refuses the
broker's trading client; a seventh fetcher would reach the refused client
(and on the mini, with keys, the network). CLAUDE.md forbids editing that
test, so the selection goes through the one chain fetcher it already fakes.
That harness passes unchanged.

### 2. `quote_freshness`

**Why.** No rule judged quote age, and free-plan option quotes run about 15
minutes behind, so `limit_price_sanity` could judge a limit against a stale
mid.

**What.** A new rule, `quote_freshness`, runs immediately before
`limit_price_sanity`. It is **rule 13 of 21**; `limit_price_sanity` moves to
14 and everything after it moves down one.

- It REJECTs when any quote the price check depends on is older than
  `risk_limits.max_quote_age_seconds` for its instrument: the equity's own
  quote (`equity: 120`), or every option leg's (`option: 1200`, above the
  roughly 900 s free-plan delay). Exactly at the limit passes; the comparison
  goes through `measures.exceeds`, so float noise never flips it.
- Age runs from the quote's venue timestamp to the context's as-of time
  (`context.now`), never from `fetched_at`. For an equity the timestamp is
  `quote_time`, the time of the bid and ask the mid is made of, not the last
  trade.
- Fail closed: a quote with no timestamp, a quote that is not there (or is for
  another symbol), and an option with no legs all REJECT.
- It checks the quotes whatever the order type. A market order has no limit to
  judge, but its notional is the worst-case fill at those same quotes.
- The detail names each offending quote, its age and the limit, with the
  config key, in the house style:
  `the quote for leg SPY260821C00650000 is 1201s old, above the limit of 1200s (risk_limits.max_quote_age_seconds.option)`.
  A pass names the oldest quote:
  `the quote for AAPL is 0s old, within the limit of 120s (risk_limits.max_quote_age_seconds.equity)`.
- Measures: `priced_quotes` (the quotes `mid_price` reads, by symbol) and
  `quote_age`, both in `aegis/policy/measures.py`, pure like the rest.

**A behaviour change worth knowing.** A missing quote used to be rejected
first by `limit_price_sanity` ("no current mid"). It is still rejected there,
but `quote_freshness` now comes first, so `failing_rule` for a proposal with
no quote is `quote_freshness`. The tests that pinned the old failing rule were
updated to say so.

**Open question, not decided here.** A quote dated after the as-of time counts
as fresh, with no bound on how far ahead. That is deliberate for small
amounts: `build_context` reads the clock before it fetches quotes, so a
real-time quote is routinely a little ahead. Bounding it would need a new
tunable, which this change does not add.

## How the counts changed

Everything that said twenty now says twenty-one: the engine's and the
package's docstrings, config.yaml's comment, the README, and every test
that pinned the count or the list (`THE_TWENTY` is now `THE_TWENTY_ONE`).
The limits report does not list rules, so it needed no change. The CLI
numbers rules from the recorded list, so `evaluate` now prints
`rules (21)` with `[13] quote_freshness` and `[14] limit_price_sanity`.

Existing tests that changed, and why:

- **Counts and lists** (engine, rules, CLI, context): 20 to 21, 19 to 20.
- **`min_dte` boundary tests** were written against the old default of 1.
  They now test the new default of 7 (7 passes, 6 rejects, 0DTE and expired
  still reject), and the test that the floor comes from the limits uses 14
  and 1.
- **Missing quotes**: `failing_rule` is now `quote_freshness` (see above), and
  where a test listed the set of rejecting rules, `quote_freshness` joins
  `limit_price_sanity`. The closing-sale "no quote" case moved into its own
  test, because two rules reject it; stale and undated cases took its place in
  that parametrized list.
- **Halt tests that move `now` hours ahead** (`TestHaltPersistence`, and one
  in `TestNextOpenDate`) now give the context a quote stamped at that `now`.
  Their subject is the halt; with the default quote stamped at the fixtures'
  `NOW`, the new rule correctly called it stale. The factories gained a
  keyword-only `at=` on `make_quote` and `make_snapshot` for this; their
  defaults are unchanged.

`tests/test_brain_architecture.py` and `tests/test_policy_architecture.py`
are untouched and pass.

## New tests

- `tests/test_brain_expiry_window.py` (new): whole-day DTE against the
  pricing convention; never above the policy's DTE across 2026; the 0, 1, 3,
  8, 15 listing picks 8; nothing between 7 and 45 gives the warning and no
  chain; both ends of the window included; expired contracts never eligible;
  the floor follows `risk_limits.min_dte` in a test config; the cap follows
  `brain.snapshot.max_dte`; 2,000 seeded picks all pass the real
  `options_min_dte`; the data layer's `eligible` seam (nearest accepted
  fetched, none accepted raises and fetches nothing, explicit expiration not
  second-guessed, cache keyed by the date chosen); the snapshot end to end over
  the real `get_option_chain` with only the Alpaca client and listings faked;
  a chain outside the window refused; the thesis prompt and
  `validate_candidates` only admitting the eligible expiration; a full FakeLLM
  cycle running on with one symbol that has no eligible expiration.
- `tests/test_policy_rules.py`: `TestPricedQuotes`, `TestQuoteAge`,
  `TestQuoteFreshness` (fresh passes; one second over rejects; exactly at the
  limit passes; float noise at the limit; missing timestamp rejects; missing
  quote and wrong-symbol quote reject; `fetched_at` and the last trade are
  ignored; one stale leg among fresh legs rejects and names only that leg;
  every offending leg named; each instrument held to its own limit; limits
  from the config; a zero limit; a market order held to the same limit; the
  staleness named as `failing_rule` ahead of a bad price).
- `tests/test_policy_engine.py`: the sweep's input-derived "must reject"
  reasons gained missing, undated and stale quotes; and
  `TestRandomizedQuoteAges` re-dates every case of the seeded 600-case sweep
  five ways (all stale, one stale, one undated, all exactly at the limit, all
  dated after the as-of time) under limits drawn per case, from its own seed
  so the base sweep's cases are unchanged. AUTO_EXECUTE never appears beside a
  stale or undated quote; every case the base sweep auto-executes becomes a
  REJECT with `failing_rule` `quote_freshness` and nothing else non-passing;
  and the at-the-limit and dated-ahead versions keep the base verdict exactly.
- `tests/test_config.py`: the new fields' defaults and validation (negative,
  infinite, NaN, unknown keys refused).

## Evidence

Python 3.14.8 in a `uv` venv, as CLAUDE.md asks. (The `uv` in the cloud
container only offered 3.14.0rc2, whose typing internals break pydantic at
import; `uv` was upgraded to get the 3.14.8 release.)

| | Passed | Skipped |
| - | - | - |
| Baseline, `main` at `f8f361b` | 4429 | 1 |
| This branch, `python -m pytest -q` | 4517 | 1 |
| This branch, `python -W error -m pytest -q` | 4517 | 1 |

The one skip is a case-insensitive-file-system check in
`tests/test_brain_architecture.py`; the cloud container's file system is
case-sensitive. It was skipped in the baseline too.

### Mutations

MUTATION_RESULTS

## Run on the mini after merge

No live API call was made from the cloud session; there are no keys there.

1. `git checkout main && git pull && source .venv/bin/activate`
2. `python -m pytest -q`: expect 4518 passed on a case-insensitive file
   system (the macOS default), where the one test skipped in the cloud runs;
   4517 passed and 1 skipped on a case-sensitive one
3. `python -m aegis.cli.policy limits`
4. During market hours: `python -m aegis.cli.brain once`, and confirm the
   brief shows chains at 7 or more DTE
5. If that cycle produced a proposal: `python -m aegis.cli.policy evaluate
   --latest`, and confirm `quote_freshness` appears as rule 13 of 21, right
   before `limit_price_sanity`
