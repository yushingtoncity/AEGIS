# AEGIS

An agentic trading platform. Phase 0–4 status: project skeleton, the data
layer (Alpaca paper account, stocks, options, news), a pure-math pricing
engine, a SQLite audit log with operator CLIs, and the agent brain that
proposes trades into that log. Nothing trades yet — by design.

## Architecture principles

1. **The LLM proposes; deterministic code disposes.** In later phases, Claude
   emits structured trade proposals and a deterministic policy engine
   (`aegis/policy`, Phase 5) gates them against the `risk_limits` in
   `config.yaml`. Nothing in this codebase may ever let model output reach a
   broker API directly.
2. **Execution is an abstraction.** `aegis/execution/base.py` defines an
   abstract `Executor` interface so a paper broker and a live broker are
   swappable implementations behind the same seam. Only the policy engine may
   call an Executor.

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
  (`0001_initial.sql`, `0002_reasoning_cycles_and_legs.sql`, …). `open_store` applies pending
  migrations at startup inside one transaction per file and records each in
  `schema_version`; every statement is `IF NOT EXISTS`, so applying twice is
  a no-op, and a database written by newer code is refused rather than
  half-read. An applied file is never edited — schema changes are new files.
- **Repository layer** (`aegis/store/repo.py`): typed functions, pydantic
  models in and out, parameterized queries only (a test greps for anything
  else). Writes: `insert_proposal`, `add_reasoning`, `record_decision`,
  `record_approval` (request, then answer, one row), `upsert_order`,
  `record_fill`, `snapshot_positions`, `snapshot_pnl`, `log_event`. Reads:
  `get_proposal`, `get_order`, `get_open_orders`, `get_daily_pnl`,
  `get_recent_events`, and `get_proposal_trace(proposal_id)`, which returns
  the whole lineage in one call. `client_order_id` is the idempotency key:
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
| `policy_decisions` | the deterministic gate's verdict | `proposal_id`, `decided_at`, `verdict` (REJECT / FLAG_ONLY / NEEDS_APPROVAL / AUTO_EXECUTE), `rules_evaluated` (JSON), `failing_rule`, `notes` |
| `approvals` | human sign-off requests and responses | `proposal_id`, `requested_at`, `responded_at`, `response` (approved / rejected / expired), `channel`, `responder` |
| `orders` | order state; one row per `client_order_id` | `proposal_id`, `client_order_id` UNIQUE, `broker` (paper / live), `broker_order_id`, `status` (proposed … filled / cancelled / failed), `submitted_at`, `updated_at`, `symbol`, `side`, `quantity`, `limit_price` |
| `fills` | executions | `order_id` → orders, `filled_at`, `fill_price`, `fill_quantity`, `fees` |
| `position_snapshots` | positions as seen each cycle | `taken_at`, `symbol`, `quantity`, `avg_cost`, `market_value`, `unrealized_pnl` |
| `pnl_snapshots` | account P&L each cycle | `taken_at`, `equity`, `cash`, `buying_power`, `daily_pnl`, `realized_pnl`, `unrealized_pnl` |
| `events` | risk-limit trips, kill-switch toggles, heartbeats, errors | `occurred_at`, `level`, `kind`, `message`, `payload` (JSON) |
| `schema_version` | applied migrations | `version`, `name`, `applied_at` |

Every `id` is a uuid4 string; enum columns are CHECK-constrained to the
values above; every foreign key is indexed.

### Store CLIs

```bash
python -m aegis.cli.db init            # create the file if needed, apply pending migrations
python -m aegis.cli.db status          # schema version, journal mode, row counts (never creates)
python -m aegis.cli.trace PROPOSAL_ID  # the full lineage, printed in full (--json for scripts)
```

All three take `--db PATH` to override the configured file. `trace` is
read-only: it refuses a database with pending migrations rather than
applying them as a side effect.

## Agent brain (Phase 4)

`aegis/brain` is the part of AEGIS that thinks. It has **zero execution
authority**: it reads market data through `aegis.data`, prices structures
with `aegis.pricing`, calls Claude through one module (`aegis/brain/llm.py`),
and its only outputs are rows in the store — one `reasoning` row per stage
and, when it has a compelling idea, one `proposals` row (plus
`proposal_legs` for multi-leg structures). Nothing consumes proposals yet;
the Phase 5 policy engine will gate every one of them. No module under
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
| **scan** | `claude-haiku-4-5-20251001` | the watchlist snapshot: spot, nearest-expiry chain (ATM ± 5) with IV and Greeks from the vendor or our pricing engine, headlines, account and positions, data-age flags | `ScanBrief`: per-symbol summary, IV/skew observations, catalysts, staleness warnings |
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

## Repo layout

```
config.yaml          all tunables (watchlist, cache TTLs, bars, pricing, store, risk_limits)
aegis/
  config.py          loads and validates config.yaml (pydantic)
  data/              the only gateway to external reads — typed models out
  pricing/           Black-Scholes, Greeks, IV, time to expiry, position risk, enrich
  store/             SQLite audit log: models, db + migrations/, typed repo
  brain/             scan -> thesis -> proposal: llm client, prompts/, snapshot, stages, cycle
  execution/base.py  abstract Executor interface (no implementation yet)
  cli/               operator tools: check, snapshot, db, trace, brain
  policy/ notify/ dashboard/   stubs for later phases
tests/               config, cache, model-parsing, pricing, position, store and brain tests (canned JSON, FakeLLM, tmp_path DBs)
```

## Roadmap

| Phase | Deliverable |
| ----- | ----------- |
| 0–1   | Skeleton + data layer |
| 2     | Pricing: our own Greeks/IV fallback when Alpaca omits them |
| 3     | Persistence: SQLite journal of proposals, orders, fills |
| 4     | Brain: Claude proposes structured trades (this) |
| 5     | Policy engine: deterministic gate enforcing risk_limits |
| 6     | Execution: paper broker behind the Executor interface |
| 7     | Notifications |
| 8     | Dashboard |
| 9     | Daemon: scheduled autonomous loop |
| 10    | Paper campaign: supervised live-paper trading run |
