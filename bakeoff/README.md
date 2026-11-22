# Offline model bakeoff (thesis stage)

Compares thesis stage quality and cost across models on frozen historical
inputs. It is research tooling, not part of the trading system:

- It never imports the order path, the policy dispatcher, the brain cycle,
  the CLIs or the notifiers, and nothing under `aegis/` imports it
  (`tests/test_bakeoff_architecture.py`).
- It reads the audit DB **read only** and writes its results to its own
  SQLite file. `open_results` refuses the production DB path.
- The policy verdicts it records come from `aegis.policy.engine.decide`, which
  is pure: nothing is written and nothing is placed.
- The shadow critic's verdicts are logged only. They change nothing.

## Contestants

| Name | Role | Model | Setting |
| --- | --- | --- | --- |
| `baseline` | proposer | the thesis stage exactly as `config.yaml` configures it (model, track, effort, max_tokens read at run time) | effort from config |
| `neon-medium` | proposer | `gemini-4-neon` | thinking level medium |
| `neon-high` | proposer | `gemini-4-neon` | thinking level high |
| `neon-critic` | critic, log only | `gemini-4-neon` | thinking level medium |

All proposers get the identical system prompt, user prompt and JSON schema,
one call each, with no corrective retry. Settings live in `bakeoff/bakeoff.yaml`.

## What "reconstructed" means

The audit log does not store the rendered thesis prompt. `extract` rebuilds
each input from what it does store: the cycle's `market_snapshot` event, the
cycle's scan output, the proposals table (for the recent proposals list) and
the risk limits in the `config.yaml` it reads. Every record is labelled
`reconstructed` and hashed; every contestant, baseline included, sees the same
frozen bytes, and the report never compares against the historical live
outputs. `tests/test_bakeoff_dataset.py` proves the rebuild is byte identical
to what a cycle written by the real brain sent.

## Files (all private, gitignored)

- `bakeoff/data/thesis-inputs-vN.jsonl` and `.manifest.json`: the frozen
  dataset. It contains account equity and positions from the snapshots.
- `bakeoff/out/results.db`: every call, stand in dry run and critique.
- `bakeoff/out/report-<run>.md`, `blind-<run>.csv`, `blind-<run>-key.json`.

## Runbook (on the mini)

Extraction needs the production DB, so it runs on the mini. Keep the bakeoff
away from the live runtime: a separate worktree and a separate venv, because
the launchd jobs run from the live working copy and installing `google-genai`
changes shared packages (on 2026-11-22 it would have downgraded `websockets`,
which alpaca-py uses). From the live repo root:

1. A worktree and its own venv:

   ```
   git fetch origin model-bakeoff
   git worktree add ../aegis-bakeoff origin/model-bakeoff
   cd ../aegis-bakeoff
   uv venv -p 3.14 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt -r bakeoff/requirements-bakeoff.txt
   ```

2. Put `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` in this worktree's `.env`
   (never commit it).
3. Fill in the Claude price in `bakeoff/bakeoff.yaml` under `prices_per_mtok`,
   and verify the Gemini 4 Neon price. The preflight refuses to run while a
   price is blank.
4. Freeze the dataset. `--db` and `--config` point at the LIVE files, so the
   risk limits and recent proposals limit are the ones the live cycle read;
   the DB is opened read only:

   ```
   python -m bakeoff --config ../aegis/config.yaml extract --db ../aegis/data/aegis.db --since 2026-10-23 --limit 50
   ```

5. Preflight, no paid calls. It confirms `gemini-4-neon` is in your key's
   model list and stops if it is not (there is no substitute model):

   ```
   python -m bakeoff --config ../aegis/config.yaml preflight --dataset bakeoff/data/thesis-inputs-v1.jsonl
   ```

6. Run. This spends money, at most `spend_cap_usd` (5 USD by default;
   `--spend-cap` overrides it for one run):

   ```
   python -m bakeoff --config ../aegis/config.yaml run --dataset bakeoff/data/thesis-inputs-v1.jsonl
   ```

7. Grade `bakeoff/out/blind-<run>.csv` by hand without opening the key, then
   unblind with `blind-<run>-key.json`.

Adjust `../aegis` if the live checkout lives elsewhere.

`python -m bakeoff report --run <run>` regenerates the report from the
results file.

## Metrics

Primary: schema validity rate, the blind grade, cost. Secondary: policy pass
rate on quantity 1 stand ins, judged as of the original cycle's timestamp
with the assumptions the report lists beside each rule. The report also
shows which policy rules fired and which were never exercised, the critic's
veto rate, how often a veto matches a policy REJECT, and every candidate the
critic vetoed but the policy engine passed.
