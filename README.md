# AEGIS

An agentic trading platform. Phase 0–1 status: project skeleton and the data
layer (Alpaca paper account, stocks, options, news) with two operator CLIs.
Nothing trades yet — by design.

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

## Repo layout

```
config.yaml          all tunables (watchlist, cache TTLs, bars, risk_limits)
aegis/
  config.py          loads and validates config.yaml (pydantic)
  data/              the only gateway to external reads — typed models out
  execution/base.py  abstract Executor interface (no implementation yet)
  cli/               operator tools: check, snapshot
  pricing/ brain/ policy/ notify/ store/ dashboard/   stubs for later phases
tests/               config, cache, and model-parsing tests with canned JSON
```

## Roadmap

| Phase | Deliverable |
| ----- | ----------- |
| 0–1   | Skeleton + data layer (this) |
| 2     | Pricing: our own Greeks/IV fallback when Alpaca omits them |
| 3     | Persistence: SQLite journal of proposals, orders, fills |
| 4     | Brain: Claude proposes structured trades |
| 5     | Policy engine: deterministic gate enforcing risk_limits |
| 6     | Execution: paper broker behind the Executor interface |
| 7     | Notifications |
| 8     | Dashboard |
| 9     | Daemon: scheduled autonomous loop |
| 10    | Paper campaign: supervised live-paper trading run |
