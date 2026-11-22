# Proposal: capture what each stage was actually sent

Status: draft, for a future pull request. Nothing here is implemented on the
`model-bakeoff` branch, and that branch does not touch the live cycle.

## Problem

The audit log stores each stage's raw output (`reasoning.content`), its token
counts, model and latency, the cycle's `market_snapshot` event and the
proposals table. It does not store the rendered prompt a stage was sent, the
risk limits block inside that prompt, or which template produced it.

So the offline bakeoff has to rebuild every thesis input from the snapshot,
the scan output, the proposals table and a `config.yaml` it reads today. The
rebuild is byte identical for cycles written by the current code
(`tests/test_bakeoff_dataset.py`), but it can drift silently in three cases:

1. `config.yaml` `risk_limits` changed after the cycle ran. The rebuild then
   shows today's limits, not the ones the model saw.
2. A template (`aegis/brain/prompts/*.md`) or `build_thesis_prompt` changed
   after the cycle ran. The rebuild then renders the new template.
3. The recent proposals list is rebuilt by a timestamp cutoff; a proposal
   written in the same second as the scan row could land on either side.

None of these can be detected after the fact, which is why every bakeoff
record is labelled `reconstructed`.

## Proposed change

One new migration (never an edit to an applied one) adding three nullable
columns to `reasoning`:

| Column | Holds |
| --- | --- |
| `prompt_version` | the stage's recorded template version, for example `thesis-v2+system-v1`, or `scan-v2+system-v1+signals-v1` |
| `prompt_sha256` | sha256 of system prompt, user prompt and canonical schema JSON, joined by a NUL byte (the same hash the bakeoff uses, `bakeoff.dataset.input_sha256`) |
| `risk_limits_sha256` | sha256 of the canonical JSON of the `risk_limits` block the prompt rendered |

And one new table, `stage_prompts`, keyed by `prompt_sha256`, holding the full
system prompt, user prompt and schema once per distinct hash. Storing by hash
keeps the database small: the system prompt and schema repeat every cycle.

Written by the cycle in the same transaction as the reasoning row, through
`aegis/store` (invariant 5). Old rows keep NULLs; nothing is backfilled.

## What it buys

- A replay can load the exact bytes instead of rebuilding them, and label the
  record `captured` instead of `reconstructed`.
- A rebuilt input can be checked against the stored hash, so drift is caught
  instead of assumed away.
- The scorecard and any later study can group results by template and by
  risk limits version without guessing from dates.

## Costs and risks

- Storage: one user prompt per cycle per stage, roughly 10 to 30 KB each. At
  four cycles a day that is a few MB a year.
- The user prompt carries account equity and positions from the snapshot, the
  same data the `market_snapshot` event already holds. No new kind of data is
  stored, and the audit DB is already private.
- It touches the live cycle and the store, so it needs its own review, tests
  that the columns are written for every stage, and a migration test. It
  also needs one owner decision: whether a failed capture stops the cycle
  (fail closed, closest to invariant 5) or is logged as a WARNING and the
  cycle goes on.

## Out of scope

Prompt or template changes, any change to what a stage is sent, and any
policy rule. This only records what was sent.
