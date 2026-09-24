# AEGIS

An agentic trading platform. Phase 0–2 status: project skeleton, the data
layer (Alpaca paper account, stocks, options, news) with two operator CLIs,
and a pure-math pricing engine. Nothing trades yet — by design.

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

## Repo layout

```
config.yaml          all tunables (watchlist, cache TTLs, bars, pricing, store, risk_limits)
aegis/
  config.py          loads and validates config.yaml (pydantic)
  data/              the only gateway to external reads — typed models out
  pricing/           Black-Scholes, Greeks, IV, time to expiry, position risk, enrich
  execution/base.py  abstract Executor interface (no implementation yet)
  cli/               operator tools: check, snapshot
  brain/ policy/ notify/ store/ dashboard/   stubs for later phases
tests/               config, cache, model-parsing, pricing and position tests (canned JSON, seeded RNG)
```

## Roadmap

| Phase | Deliverable |
| ----- | ----------- |
| 0–1   | Skeleton + data layer |
| 2     | Pricing: our own Greeks/IV fallback when Alpaca omits them (this) |
| 3     | Persistence: SQLite journal of proposals, orders, fills |
| 4     | Brain: Claude proposes structured trades |
| 5     | Policy engine: deterministic gate enforcing risk_limits |
| 6     | Execution: paper broker behind the Executor interface |
| 7     | Notifications |
| 8     | Dashboard |
| 9     | Daemon: scheduled autonomous loop |
| 10    | Paper campaign: supervised live-paper trading run |
