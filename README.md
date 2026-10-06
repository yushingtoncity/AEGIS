# AEGIS

An agentic trading platform. Phase 0–5.1 status: project skeleton, the data
layer (Alpaca paper account, stocks, options, news), a pure-math pricing
engine, a SQLite audit log with operator CLIs, the agent brain that proposes
trades into that log, and the deterministic policy engine that gives every
proposal a verdict. Phase 5.1 keeps the brain's option chains at or past the
policy's DTE floor and adds a quote-age rule to the engine. Nothing trades yet — by design: there is no Executor
implementation until Phase 6, so a verdict is all that happens to a proposal.

## Architecture principles

1. **The LLM proposes; deterministic code disposes.** Claude emits
   structured trade proposals (`aegis/brain`) and a deterministic policy
   engine (`aegis/policy`) gates each one against the `risk_limits` in
   `config.yaml`. Nothing in this codebase may ever let model output reach a
   broker API directly.
2. **Execution is an abstraction.** `aegis/execution/base.py` defines an
   abstract `Executor` interface so a paper broker and a live broker are
   swappable implementations behind the same seam. Only the policy engine may
   reference an Executor — `tests/test_policy_architecture.py` fails if any
   other module does.

Supporting rules: all external reads go through `aegis/data` and return typed
pydantic models (every one carrying a `fetched_at` UTC timestamp), never raw
API JSON; all tunables live in `config.yaml`; all secrets live in `.env`,
which is gitignored.

## Setup

Requires Python 3.12+ (tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (POSIX: source .venv/bin/activate)
pip install -r requirements.txt
```

### Alpaca paper keys (free)

1. Sign up at <https://app.alpaca.markets/signup> (no funding needed).
2. Switch the dashboard to the **Paper** account (top-left account switcher).
3. Generate an API key pair on the overview page.
4. Create your env file and fill in both values:

```bash
copy .env.example .env         # POSIX: cp .env.example .env
```

`ANTHROPIC_API_KEY` can stay empty until Phase 4. `.env` is gitignored —
never commit keys, never put them in `config.yaml`.

## Running

Smoke test — config, env, Alpaca paper auth, data clients:

```bash
python -m aegis.cli.check
```

Market snapshot — spot with data age, the nearest-expiry ATM ± 5 strikes
chain (calls and puts side by side), and the last 5 headlines:

```bash
python -m aegis.cli.snapshot SPY
```

Data-age note: free-plan Alpaca options data is roughly 15 minutes delayed
(the `indicative` feed), so the CLIs print venue timestamps and age on
everything and label stale/closed-market data explicitly.

Tests (no live API calls — fixtures only):

```bash
python -m pytest -q
```

## Pricing engine (Phase 2)

`aegis/pricing` is pure math: Black-Scholes prices, analytic Greeks, implied
volatility, time to expiry, multi-leg position risk, and an `enrich` step that
puts vendor and model values side by side. It makes no network calls, reads
no config, and imports only model types from `aegis.data`; every function is
deterministic. Its job is to be the cross-check and the fallback when Alpaca
omits IV or Greeks (common for near-dated, deep-OTM, or unquoted contracts) —
never the source of truth.

**Modelling assumptions — read before trusting a number.**

- **European exercise.** US equity options are American style, so every
  price, Greek and IV here is an approximation. The error is largest for
  deep in-the-money puts (early exercise has value) and for dividend payers
  (a discrete dividend is not a continuous yield).
- **Constant volatility and rate**, log-normal spot, no dividends unless a
  continuous yield `q` is passed.
- **Risk-free rate** comes from `pricing.risk_free_rate` in `config.yaml`. It
  is a labelled placeholder that should track short-term Treasury yields; a
  later phase may source it from a feed.
- **Time to expiry** is calendar days (fractional, so 0–3 DTE contracts are
  not rounded to zero) over `pricing.day_count_basis` (365), measured to the
  exchange close on expiration day (`pricing.expiry_time` in
  `pricing.expiry_timezone`, default 16:00 America/New_York, DST-aware).
- **Greek units** are the ones traders quote: theta per calendar day, vega
  per 1 vol point (0.01 σ), rho per 1 percentage point of rate; delta and
  gamma unscaled. Per-share values; position-level values apply
  `pricing.contract_multiplier` (100).
- **Implied vol** is solved from the quote **mid** with `brentq` on
  [1e-4, 5.0] and returns `None` (never raises) when no vol explains the
  price — at or below intrinsic, above the bracket's ceiling, a zero quote,
  or an expired contract. Near expiry vega → 0, so IV is unstable and should
  be treated with suspicion.
- **Edge cases never produce NaN.** `T <= 0` returns intrinsic value;
  `sigma <= 0` returns discounted intrinsic (the deterministic forward);
  Greeks take the matching limits. Only malformed input (non-positive or
  non-finite spot/strike, an empty position, mixed expiries) raises
  `PricingError`, which carries context like `DataError`.
- **Position analysis** is evaluated at expiry on the exact grid
  {0, every strike, one probe beyond the highest strike}, so ratio spreads
  and backspreads fall out without per-strategy formulas; `unlimited_risk`
  and `unlimited_profit` come from the slope past the last strike. All legs
  must share one expiry (calendar/diagonal spreads are out of scope). Early
  assignment on short legs is ignored.

Sanity anchor (Hull): S=42, K=40, r=0.10, σ=0.20, T=0.5 → call 4.76, put
0.81. `tests/test_pricing.py` pins this, put-call parity on a seeded grid,
an IV round trip, and finite-difference checks of every Greek.

## Persistence and audit log (Phase 3)

`aegis/store` is a SQLite journal through the standard library `sqlite3` —
no ORM, so the file stays inspectable with any SQLite client. Every proposal
the brain makes, each reasoning stage behind it, the policy verdict, the
human approval, the orders and fills that followed, plus periodic position
and P&L snapshots and operational events, are all written here so any action
AEGIS takes is reconstructible after the fact.

- **Path** comes from `store.db_path` in `config.yaml` (default
  `data/aegis.db`, relative to the repo root). `*.db` and the WAL sidecars
  `*.db-wal` / `*.db-shm` are gitignored.
- **WAL journaling** is enabled on every connection, with foreign keys on and
  a 5 s busy timeout, so the future dashboard can read while the loop writes.
- **Schema** lives in versioned SQL files under `aegis/store/migrations/`
  (`0001_initial.sql`, `0002_reasoning_cycles_and_legs.sql`,
  `0003_controls.sql`, `0004_execution.sql`, …). `open_store` applies pending migrations at startup inside one transaction per file and records each in
  `schema_version`; every statement is `IF NOT EXISTS`, so applying twice is
  a no-op, and a database written by newer code is refused rather than
  half-read. An applied file is never edited — schema changes are new files.
- **Repository layer** (`aegis/store/repo.py`): typed functions, pydantic
  models in and out, parameterized queries only (a test greps for anything
  else). Writes: `insert_proposal`, `add_reasoning`, `record_decision`,
  `record_approval` (request, then answer, one row), `upsert_order`,
  `record_fill`, `snapshot_positions`, `snapshot_pnl`, `log_event`, and the
  control flags `set_kill_switch` / `set_halt_until`, and for Phase 6
  `claim_order` / `apply_broker_update` (see "Execution records" below).
  `record_decision` and
  the two control writers take an optional `event`, written in the same
  transaction as the row it describes. Reads: `get_proposal`, `get_order`,
  `get_open_orders`, `get_daily_pnl`, `get_recent_events`, `get_controls`,
  `get_proposals_since`, `get_latest_undecided_proposal`,
  `count_orders_submitted_between`, `get_decision`, `get_decision_approval`,
  `get_proposal_order`, `get_order_by_broker_id`, and `get_proposal_trace(proposal_id)`,
  which returns the whole lineage in one call. `client_order_id` is the idempotency key:
  `upsert_order` called twice with the same one yields exactly one row,
  keeping the original `id`. Failures surface as `StoreError` with the
  operation and record id, never a raw `sqlite3` exception.
- **Timestamps** are timezone-aware UTC in the models and ISO-8601 text with
  offset in the file; JSON columns hold strict JSON; `raw_model_output` is
  stored verbatim and never parsed.

### Schema overview

| Table | Purpose | Key columns |
| ----- | ------- | ----------- |
| `proposals` | one structured trade proposal from the brain | `id`, `created_at`, `cycle_id`, `symbol`, `instrument`, `side`, `quantity`, `order_type`, `limit_price`, `thesis`, `confidence` (0–1), `invalidation`, `raw_model_output`, `model_name`, `prompt_version` |
| `reasoning` | each agent stage of a brain cycle, linked to the proposal it led to (if any) | `cycle_id`, `proposal_id` → proposals (null until a proposal exists; a NO_TRADE cycle keeps its rows under the `cycle_id`), `stage` (scan / thesis / proposal), `created_at`, `content` (the model's raw output), `tokens_in`, `tokens_out`, `model_name` (the model that answered), `latency_ms` |
| `proposal_legs` | the legs of a multi-leg option proposal | `proposal_id` → proposals, `leg_index`, `symbol` (OCC), `option_type`, `side`, `quantity`, `strike`, `expiration` |
| `policy_decisions` | the deterministic gate's verdict | `proposal_id`, `decided_at`, `verdict` (REJECT / FLAG_ONLY / NEEDS_APPROVAL / AUTO_EXECUTE), `rules_evaluated` (JSON), `failing_rule`, `notes`, `purpose` (`evaluate`, or `pre_submit` for the re-check just before an order is sent) |
| `approvals` | human sign-off requests and responses | `proposal_id`, `requested_at`, `responded_at`, `response` (approved / rejected / expired), `channel`, `responder`, `decision_id` → policy_decisions (one approval per decision), `expires_at`, `note` |
| `orders` | order state; one row per `client_order_id` | `proposal_id`, `client_order_id` UNIQUE, `broker` (paper / live), `broker_order_id`, `status` (proposed … filled / cancelled / failed), `submitted_at`, `updated_at`, `symbol`, `side`, `quantity`, `limit_price`; since 0004: `decision_id`, `regate_decision_id`, `approval_id` (the authority behind the order), `instrument`, `order_type`, `time_in_force`, `position_intent`, `filled_quantity`, `avg_fill_price`, `broker_status`, `status_reason`, `last_synced_at` |
| `fills` | executions | `order_id` → orders, `filled_at`, `fill_price`, `fill_quantity`, `fees`, `broker_fill_id` (UNIQUE when set) |
| `position_snapshots` | positions as seen each cycle | `taken_at`, `symbol`, `quantity`, `avg_cost`, `market_value`, `unrealized_pnl` |
| `pnl_snapshots` | account P&L each cycle | `taken_at`, `equity`, `cash`, `buying_power`, `daily_pnl`, `realized_pnl`, `unrealized_pnl` |
| `events` | risk-limit trips, kill-switch toggles, heartbeats, errors | `occurred_at`, `level`, `kind`, `message`, `payload` (JSON) |
| `controls` | the operator's stop flags; exactly two rows, never deleted | `key` (`kill_switch` / `halt_until`), `value` (`on` / `off`; an ISO timestamp or empty), `updated_at` |
| `schema_version` | applied migrations | `version`, `name`, `applied_at` |

Every `id` is a uuid4 string; enum columns are CHECK-constrained to the
values above; every foreign key is indexed. (`controls` is keyed by its
`key`.)

### Store CLIs

```bash
python -m aegis.cli.db init            # create the file if needed, apply pending migrations
python -m aegis.cli.db status          # schema version, journal mode, row counts (never creates)
python -m aegis.cli.trace PROPOSAL_ID  # the full lineage, printed in full (--json for scripts)
```

All three take `--db PATH` to override the configured file. `trace` is
read-only: it refuses a database with pending migrations rather than
applying them as a side effect.

### Execution records (Phase 6, store side)

Migration `0004_execution.sql` gets the store ready for real paper orders.
It only adds columns, indexes and triggers, so every row written before it
reads and behaves as it did. An order written by Phase 6 is tied to the
decision that authorised it (`decision_id`), to the `pre_submit`
re-evaluation it passed just before it was sent (`regate_decision_id`), and
to the human approval it rests on, if any (`approval_id`). Those orders are
written and moved only by two functions:

- `claim_order` is the last check and the first record, in one
  `BEGIN IMMEDIATE` transaction. It re-reads the controls (the kill switch,
  an active halt, or a halt it cannot read all refuse), checks the links
  (an `evaluate` verdict of AUTO_EXECUTE or NEEDS_APPROVAL on this
  proposal, a `pre_submit` decision on the same proposal, and for
  NEEDS_APPROVAL an approved, unexpired approval of that exact decision),
  refuses a second order for the same proposal, and counts today's orders
  against `max_daily_trades`. Then it inserts the row as `approved` with
  `submitted_at` set, so the order counts toward the cap from that instant
  even if the process dies before it is sent. A refusal raises
  `ClaimRefused` with the reason and writes nothing.
- `apply_broker_update` records what the broker says. The broker reports
  cumulative figures, so each growth of the filled quantity becomes one
  fill, priced so the fills add up to the broker's average, and named
  `<broker_order_id>:<cumulative>`, so replaying the same update records
  nothing twice. Figures that do not add up (a shrinking total, more than
  the order's quantity, no positive price for the new fill) are refused.

Triggers hold the same rules under any writer: a claimed order is born
`approved`, its status only moves forward, a filled, cancelled or failed
order is frozen, the filled quantity only grows and never passes the
quantity, what the order is and on whose authority never changes, and the
broker's id is set once. An approval tied to a decision is answered once.
`upsert_order` keeps working for rows written before Phase 6 and refuses
the new ones. The design is in `docs/phase6/SPEC_PHASE6.md`; the executor
and the CLI that use these come in the next two PRs.

## Agent brain (Phase 4)

`aegis/brain` is the part of AEGIS that thinks. It has **zero execution
authority**: it reads market data through `aegis.data`, prices structures
with `aegis.pricing`, calls Claude through one module (`aegis/brain/llm.py`),
and its only outputs are rows in the store — one `reasoning` row per stage
and, when it has a compelling idea, one `proposals` row (plus
`proposal_legs` for multi-leg structures). The policy engine (Phase 5,
below) gates every one of them; the brain never sees or calls it. No module under
`aegis/brain` may import `aegis.execution`. `tests/test_brain_architecture.py`
enforces that three ways: a static scan of every brain module (and the brain
CLI) for any import of `aegis.execution`, any dynamic-import or
process-spawning machinery, any string naming an execution path, and any
handle on the broker's order API; a fresh-interpreter run of a full cycle
and of the CLI with an audit hook that fails if anything under
`aegis/execution` is imported, opened or executed; and a test-suite guard
that makes `aegis.execution` unimportable in every brain test.

### The pipeline

Each cycle (`aegis.brain.cycle.run_cycle`) gets a `cycle_id`, logs
`cycle_start`, records the market snapshot it saw as an event, then runs
three stages, each with its own pydantic input and output models and its
own model from `brain.stages` in `config.yaml`:

| Stage | Default model | Input | Output |
| ----- | ------------- | ----- | ------ |
| **scan** | `claude-haiku-4-5-20251001` | the watchlist snapshot: spot, the chain of the nearest expiration 7–45 DTE (ATM ± 5; see "Which expiration") with IV and Greeks from the vendor or our pricing engine, headlines, account and positions, data-age flags | `ScanBrief`: per-symbol summary, IV/skew observations, catalysts, staleness warnings |
| **thesis** | `claude-opus-5-5` | the brief, positions, a read-only summary of `risk_limits`, recent proposals from the store (so it does not repeat itself), the contracts actually available | `ThesisOutput`: zero or more `ThesisCandidate`s (symbol, direction, instrument/structure, rationale, confidence, key risk, invalidation). Empty with a reason is a respectable answer |
| **proposal** | `claude-fable-5-1` | the best candidate resolved to concrete contracts with live quotes, the pricing engine's max loss / max profit / breakevens / net Greeks for the structure, account state | `ProposalOutput`: a `TradeProposal` (symbol, instrument, side, quantity, order type, limit price, thesis, confidence, invalidation, legs) **or** `no_trade` with a reason |

Model output is constrained JSON (the API's structured outputs, with a
schema derived from the pydantic model) and is then validated locally.
Output that fails validation — including a proposal whose legs do not match
the offered contracts — is fed back to the model with the error and retried
up to `brain.max_retries`; it is never hand-patched. After the last failure
the stage raises `BrainError` and a `brain_error` event is logged.

A cycle ends in exactly one of: a `proposals` row (reasoning rows linked to
it), a `no_trade` event, a `budget_halt`, or a logged failure — always
followed by `cycle_end`. When the market is closed the cycle still runs, and
the brief carries the deterministic staleness warnings whether or not the
model repeats them.

### Which expiration

Each symbol's chain comes from the **nearest listed expiration whose DTE is
at least `risk_limits.min_dte` and at most `brain.snapshot.max_dte`** (7 and
45 as shipped). The floor is read from `risk_limits.min_dte` itself, not
copied into the brain's block, so the brain and the policy engine's
`options_min_dte` share one number and cannot drift apart.

- **DTE** here is the pricing engine's calendar-day convention
  (`calendar_days_to_expiry`: fractional calendar days to the exchange close
  on expiration day) in whole days, rounded down. Through the session that
  matches the policy's DTE (calendar days from the trading date); after the
  close it is one less; it is never more. So a chain the brain picks is never
  one `options_min_dte` rejects. An expiration whose close has passed is
  never eligible, whatever the floor.
- The data layer does the fetching: `get_option_chain(symbol,
  eligible=...)` reads the expiration listing and fetches only the nearest
  date the brain's test accepts. The snapshot then checks the chain it got
  back, refuses one outside the window, and leaves out any contract in it
  that is listed under another expiration.
- **No eligible expiration** (nothing listed between the floor and the cap)
  is a data warning, not an error: the symbol gets a `no eligible expiration
  (7-45 DTE, risk_limits.min_dte, brain.snapshot.max_dte): …; option chain
  omitted` line, no chain, and the cycle goes on. A nearer expiration is never
  used in its place.
- The thesis stage is shown only that chain and may only name its
  expiration and strikes (`validate_candidates`), so every option candidate
  comes from the eligible expiration. A symbol with no chain is offered for
  equity candidates only.

The operator snapshot CLI (`python -m aegis.cli.snapshot`) still shows the
nearest expiration, or the one `--expiration` names.

### Prompts and the injection guard

Prompts live in `aegis/brain/prompts/` — a shared system preamble plus one
template per stage, each with a version string; the composite
`prompt_version` is recorded on every proposal row. The preamble states the
governance model plainly: you propose, you never execute; a deterministic
policy engine gates every proposal; every proposal must carry a falsifiable
invalidation condition; declining to trade is always a valid and often the
correct output.

All news text is **untrusted data**. It is wrapped in
`<<<UNTRUSTED_NEWS_BEGIN>>>` / `<<<UNTRUSTED_NEWS_END>>>` delimiters,
anything inside that looks like a delimiter is neutralised, and the model
is told to treat the contents strictly as data, never as instructions, no
matter what they claim. `tests/test_brain_prompts.py` asserts a fixture
headline is wrapped and the guard text is present.

### Budget guard

Same philosophy as the daily loss limit. Before every model call the guard
sums today's tokens (UTC day) and this cycle's tokens from the `reasoning`
table (plus what the cycle has spent but not yet written), adds the call's
estimated input and `max_tokens`, and refuses the call — logging a
`budget_halt` event and ending the cycle as `halted` — if
`brain.daily_token_budget` or `brain.per_cycle_token_cap` would be crossed.
Both are labelled placeholders in `config.yaml`; so are the
`brain.prices_per_mtok` entries, which are used only to print cost
*estimates*. Every call records tokens in and out, latency and the model
that answered on its reasoning row, and rate limits / transient API errors
are retried with exponential backoff before a `BrainError` is raised. No
billed token escapes the budget: when a stage fails after the API has
answered — output that never validated, a response truncated at
`max_tokens`, a refusal, a crash mid-cycle — its tokens are still written
to the reasoning table (with whatever the model returned) before the
`brain_error` and `cycle_end` events are logged.

Note on the API surface: the current Messages API and `anthropic` 1.x SDK
have no `temperature` parameter for these models; the per-stage knob is
`effort` (thinking is always on for Opus 5.5 and Fable 5.1 and counts toward
`max_tokens`). That is why the per-stage `max_tokens` are 3,000 / 4,000 /
3,000 rather than the initial 1,500 / 2,000 / 800: live runs truncated at the
smaller caps (a terse five-symbol brief needs ~1,700 tokens; the proposal
stage spent ~670 thinking before its ~500-token JSON began). A truncated
answer is never retried — it is a configuration problem, and the error names
the key to raise. `ANTHROPIC_API_KEY` is read from `.env` like the Alpaca keys.

### Brain CLIs

```bash
python -m aegis.cli.brain once                 # one real cycle: live Alpaca data, real Anthropic calls
python -m aegis.cli.brain stage scan           # a single stage for debugging (thesis, proposal; --cycle ID reuses a prior cycle)
python -m aegis.cli.brain usage                # today's tokens and estimated spend from the store
```

`once` prints the brief, the candidates, the proposal or `NO_TRADE`, tokens
per stage, and an estimated cost clearly labelled as an estimate; follow it
with `python -m aegis.cli.trace <proposal_id>` to see the stored lineage.

## Policy engine (Phase 5)

`aegis/policy` is the deterministic gate between a proposal and any order —
the "deterministic code disposes" half of the governing principle. It is
pure Python with **no model calls, ever**: the same proposal and the same
context always produce the same verdict. Every proposal gets exactly one of
four verdicts — `REJECT`, `FLAG_ONLY`, `NEEDS_APPROVAL` or `AUTO_EXECUTE` —
and `AUTO_EXECUTE` is unreachable unless every one of the twenty-one rules
passed. A verdict is all this phase produces: there is no Executor
implementation yet, so even `AUTO_EXECUTE` places no order until Phase 6.

It is also the only package that may ever reference an `Executor`.
`tests/test_policy_architecture.py` statically scans every module outside
`aegis/policy` and `aegis/execution` and fails on any import of
`aegis.execution` (in any form), any identifier containing `executor`, the
interface's order methods (`submit_order`, `cancel_order`, `close_position`),
or any string naming the package; a fresh-interpreter run then imports every
one of those modules and fails if any of them asked for the execution
package or holds a module, class or instance from it. The static scan is a
tripwire, not a sandbox — a deliberately obfuscated, dormant reference is
beyond it, and the test's docstring says exactly what it does not see. The
same test pins the engine's purity: nothing under `aegis/policy` imports
`anthropic`, the brain, the broker or a network library, and only
`context.py` may read the clock or fetch data.

The spec the phase was built against, and the decisions and review rulings
taken along the way, are kept in `docs/phase5/`; the Phase 5.1 follow-ups
(the expiry floor and `quote_freshness`) are in `docs/phase5_1/NOTES.md`.

### Inputs: the proposal and the context

`engine.evaluate(proposal, context, conn)` judges a stored proposal (with its
option legs) against a frozen `PolicyContext` and records the verdict;
`engine.decide(proposal, context)` is its pure half. If a verdict depends on
something, it is a field of one of those two values — no rule reads a wall
clock, the network, the store or the config. `context.build_context` fills
the context from `aegis.data` and `aegis.store`:

- account state — equity, cash, buying power, options buying power — and open
  positions; start-of-day equity (Alpaca's `last_equity`) and today's P&L
- the market clock (`aegis.data.market.get_market_clock`: `is_open`,
  `next_open`, `next_close`)
- current quotes: the instrument's own, and one per leg for an option
- the pricing engine's expiry analysis of the structure at the proposal's
  price — max loss, max profit, breakevens, `unlimited_risk`
- from the store: open orders, orders sent today, the other proposals inside
  the duplicate window, and the control flags
- the limits (`risk_limits`), the watchlist and the contract multiplier

Tests build contexts by hand, so the whole suite runs with no network.

**Unknown is never good news.** A value that could not be fetched is `None`
with the reason recorded; NaN and infinity count as unknown; and a rule that
needs an unknown value rejects. A failed fetch never aborts the build — the
proposal still reaches the engine and is rejected there, on the record. A
rule that raises is itself a REJECT, and the engine still runs the rest.

### The twenty-one rules

Every rule is a pure function of `(proposal, context)` returning a name, an
outcome (`PASS`, `FLAG`, `ESCALATE` or `REJECT`) and a human-readable detail
with the numbers it compared. Every number comes from `risk_limits` in
`config.yaml`; nothing in `aegis/policy` hardcodes one. All twenty-one always
run, in this order — the engine never short-circuits — so the audit trail
shows what each one found.

| # | Rule | Trips when | Outcome |
| - | ---- | ---------- | ------- |
| 1 | `kill_switch` | the kill switch is on | REJECT |
| 2 | `halted` | `halt_until` is in the future, or cannot be read | REJECT |
| 3 | `market_hours` | the market is closed, or the clock is missing or stale | REJECT |
| 4 | `daily_loss_limit` | today's P&L ≤ −`daily_loss_limit_pct` of start-of-day equity — and this trip sets the halt | REJECT |
| 5 | `no_trade_list` | the symbol or its underlying is on `no_trade_list` | REJECT |
| 6 | `watchlist_only` | enabled and the underlying is not in the watchlist | REJECT |
| 7 | `invalidation_present` | the invalidation is empty | REJECT |
| 8 | `buying_power` | notional > the buying power available to it (options: options buying power), or none is available | REJECT |
| 9 | `max_position_pct` | notional + existing exposure in the underlying > `max_position_pct` of equity | REJECT |
| 10 | `max_open_positions` | a new underlying while distinct underlyings held or pending ≥ `max_open_positions` | REJECT |
| 11 | `max_daily_trades` | orders sent today ≥ `max_daily_trades` | REJECT |
| 12 | `duplicate` | an earlier proposal with the same symbol, instrument and side within `duplicate_window_minutes`, or an open order on one of the proposal's symbols | REJECT |
| 13 | `quote_freshness` | a quote the price check depends on (the equity's own, or any option leg's) is older than `max_quote_age_seconds` for its instrument, has no venue timestamp, or is not there | REJECT |
| 14 | `limit_price_sanity` | a market order while `allow_market_orders` is false; a limit more than `limit_price_tolerance_pct` from the current mid; no usable limit or no mid to check it against | REJECT |
| 15 | `options_min_dte` | any leg with fewer than `min_dte` days to expiration | REJECT |
| 16 | `options_max_loss` | max loss > `max_loss_per_trade`; **unlimited risk, unconditionally**; a structure that cannot be analysed or whose shape is not what its instrument says (see Shape) | REJECT |
| 17 | `options_max_contracts` | any leg quantity > `max_contracts` | REJECT |
| 18 | `options_escalate` | the instrument is an option — always (and anything option-like under an equity label) | ESCALATE |
| 19 | `short_sale` | an equity sell that does not close an existing long | ESCALATE (REJECT if `reject_short_sales`) |
| 20 | `min_confidence` | confidence < `min_confidence` | FLAG |
| 21 | `auto_tier` | anything but: `auto_execute.enabled`, a plain equity, a buy or a sell closing a long, a limit order, a watchlist symbol, a known notional ≤ `auto_execute.max_notional` | ESCALATE |

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
- **Quote age** runs from a quote's venue timestamp (`quote_time`: the time
  of the bid and ask a mid is made of, not the last trade) to the context's
  as-of time (`now`), never from `fetched_at`. `quote_freshness` sits
  immediately before `limit_price_sanity`, so a limit judged against a stale
  mid is rejected for the staleness, which `failing_rule` names. It checks
  the quotes whatever the order type: a market order's notional is the
  worst-case fill at those same quotes.
- **Shape** — the engine does not trust a proposal to be what its label
  says. Every option leg's symbol must be the OCC symbol of the contract the
  leg describes (root, expiration, type, strike); all legs share one
  underlying and one expiration; a proposal named by a contract symbol has
  exactly that one leg; an equity proposal carries no legs and is not named
  by an option symbol; and no symbol contains whitespace (the padded OCC
  form included) or a non-ASCII character. Anything else is rejected by
  `options_max_loss`, and can never reach the auto tier. Symbols are matched
  as the instrument they name, so another spelling of a listed or already
  working contract is still that contract to `no_trade_list` and `duplicate`.
- Comparisons carry a tiny float-noise tolerance (one part in a billion), so
  `10 × 100.1` is not "above" `1001.00`. It is not a limit, and it never
  turns nothing into something: zero buying power, a zero auto tier and a
  sale against no holding admit nothing, however small the order.

### Verdict precedence

After all twenty-one have run, the verdict is resolved in this order:

1. any `REJECT` → **`REJECT`**, with `failing_rule` set to the first rejecting rule
2. else any `FLAG` → **`FLAG_ONLY`** — recorded and flagged; nothing executes and nobody is asked
3. else any `ESCALATE` → **`NEEDS_APPROVAL`** — a human must approve
4. else → **`AUTO_EXECUTE`** — all twenty-one passed, the auto tier included

Options never auto-execute in this phase (`options_escalate`), and a result
set that is not exactly the twenty-one registered rules in order is itself a
REJECT (`engine_integrity`). `failing_rule` is the *first* rejecting rule in
the table's order, not the most specific one: a naked short call, whose
notional is unlimited, is recorded as failing `buying_power` (rule 8), with
`max_position_pct` and `options_max_loss` rejecting it further down the same
list. The decision is written through the store's `record_decision` —
exactly once per evaluation, with every rule's outcome and detail in
`rules_evaluated` — together with an event for every verdict other than
`AUTO_EXECUTE` (`policy_reject`, `policy_flag_only`,
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
close (a halt to the opening bell would expire just as trading began, and a
recovery by then would reopen it the same day), and if the clock has no next
open at that moment the halt lasts `halt_fallback_hours`. The engine only
ever extends a halt: its write is a compare-and-set against the stored
value, so a later trip — or an evaluation working from an older snapshot —
neither shortens nor re-logs it. Shortening or clearing a halt is an
operator's decision, and this phase ships no command for it: the repository
layer's `set_halt_until(conn, None)` is the way, or waiting it out.

The stop flags are read as late as possible. Building a context takes several
network calls, so the engine reads the kill switch and the halt from the
store once more just before it records a verdict and judges under the
stricter of the two readings: a switch set while a context was being built
is honoured, and nothing the context already holds is lifted.

The table defends itself: the key and value are CHECK-constrained, a trigger
refuses to delete either row, and the read fails closed — a missing or
unreadable kill-switch row reads as ON and an unreadable halt reads as
halted, each with the reason listed by `kill status`.

### Limits (`config.yaml` → `risk_limits`)

Every value is a labelled placeholder sized for the $100k paper account; live
sizing gets revisited before Phase 11. Unknown keys, non-finite numbers and
out-of-range values are refused when the config loads.

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
| `max_quote_age_seconds.equity` | 120 | `quote_freshness` |
| `max_quote_age_seconds.option` | 1200 | `quote_freshness` |
| `min_dte` | 7 | `options_min_dte`, and the floor of the brain's expiration window |
| `max_loss_per_trade` | 1000.0 | `options_max_loss` |
| `max_contracts` | 10 | `options_max_contracts` |
| `reject_short_sales` | false | `short_sale` |
| `min_confidence` | 0.5 | `min_confidence` |
| `auto_execute.enabled` | true | `auto_tier` |
| `auto_execute.max_notional` | 1000.0 | `auto_tier` |

`min_dte: 7` keeps option ideas out of the gamma-heavy last week before
expiry. The brain reads its chain from the nearest expiration at or above
this same value (see "Which expiration" under the agent brain), so its option
ideas are never built on contracts this rule rejects. The option quote limit,
1,200 s, sits above the roughly 900 s delay of free-plan option quotes; the
equity limit, 120 s, is for real-time IEX quotes.

`brain.snapshot.max_dte` (45) is not a risk limit — it lives in the brain's
block and only caps how far out the brain looks for a chain.

### Policy CLIs

```bash
python -m aegis.cli.policy evaluate PROPOSAL_ID   # judge one proposal and record the decision
python -m aegis.cli.policy evaluate --latest      # ...the newest proposal without a decision
python -m aegis.cli.policy evaluate --latest --dry-run   # print the verdict and every rule; write nothing
python -m aegis.cli.policy kill on|off|status     # the kill switch (every on/off is logged as a kill_switch event)
python -m aegis.cli.policy limits                 # each limit beside its consumption and headroom
```

`evaluate` prints the proposal, the verdict (and the failing rule for a
REJECT), all twenty-one rules with outcome and detail, then the context it judged
against (market state, kill switch and halt, account figures, data
problems). When the loss limit tripped it says so — `trading HALTED until …`,
or `trading already HALTED until …` when a longer halt was on record first.
It exits 0 whenever a verdict was produced, whatever the verdict; with
nothing to judge (`--latest` and no undecided proposal, an unknown id) it
prints one line and exits 1. Evaluating a proposal that already has a
decision adds another one — the trace shows them all.

`kill status` prints the switch, the halt and anything the store could not
read about either. `limits` prints the market, switch and halt state, which
account-level rules block trading right now, the distance to the daily loss
cap, positions and trades used, buying power, exposure per underlying
against the position cap, and the auto-execute tier; it is a thin formatter
over `aegis.policy.limits.limits_report`, the same function the Phase 8
dashboard will read, and every figure in it is computed by the code the
rule that enforces it uses.

`evaluate` and `limits` read live Alpaca data. `evaluate` brings an existing
store up to date but never creates one; `evaluate --dry-run`, `kill status`
and `limits` write no row and neither create nor migrate a database — a
pending migration is a clean error whose fix is `python -m aegis.cli.db
init`. All take `--db PATH` (`:memory:` is refused); a failure is one line on
stderr and exit 1, never a traceback.

### What the gate does not do

- It places no order. Phase 6 adds the paper Executor; until then every
  verdict, `AUTO_EXECUTE` included, is a row in `policy_decisions`.
- It does not reject a quote dated after the context's as-of time. The
  context's clock is read before its quotes are fetched, so a real-time
  quote is routinely a little ahead of it; such a quote counts as fresh, and
  there is no bound on how far ahead it may be.
- `auto_tier` is exactly the criteria listed in rule 21. It does not ask
  whether figures no rule needed for this proposal (cash, options buying
  power on an equity order) could be read.

## Repo layout

```
config.yaml          all tunables (watchlist, cache TTLs, bars, pricing, store, risk_limits)
aegis/
  config.py          loads and validates config.yaml (pydantic)
  data/              the only gateway to external reads — typed models out
  pricing/           Black-Scholes, Greeks, IV, time to expiry, position risk, enrich
  store/             SQLite audit log: models, db + migrations/, typed repo
  brain/             scan -> thesis -> proposal: llm client, prompts/, snapshot, stages, cycle
  policy/            the deterministic gate: models, context, measures, rules, engine, limits
  execution/base.py  abstract Executor interface (no implementation yet)
  cli/               operator tools: check, snapshot, db, trace, brain, policy
  notify/ dashboard/ stubs for later phases
tests/               config, cache, model-parsing, pricing, position, store, brain and policy tests (canned JSON, FakeLLM, hand-built contexts, tmp_path DBs)
docs/phase5/         working documents of the policy-engine phase: spec, decisions, review rulings
docs/phase5_1/       the Phase 5.1 follow-ups: what changed, why, and the evidence
docs/phase6/         the Phase 6 spec: decisions, flow, and the 6a/6b/6c breakdown
```

## Roadmap

| Phase | Deliverable |
| ----- | ----------- |
| 0–1   | Skeleton + data layer |
| 2     | Pricing: our own Greeks/IV fallback when Alpaca omits them |
| 3     | Persistence: SQLite journal of proposals, orders, fills |
| 4     | Brain: Claude proposes structured trades |
| 5     | Policy engine: deterministic gate enforcing risk_limits |
| 5.1   | Expiry floor for the brain's chains, and the quote-age rule (this) |
| 6     | Execution: paper broker behind the Executor interface |
| 7     | Notifications |
| 8     | Dashboard |
| 9     | Daemon: scheduled autonomous loop |
| 10    | Paper campaign: supervised live-paper trading run |
