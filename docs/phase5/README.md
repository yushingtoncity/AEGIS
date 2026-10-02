# Phase 5 working documents

The documents the policy engine was built and reviewed against, kept as they
were written. They are a record of how the phase was decided, not a
description of the code: where one of them differs from the code, the code
and the main `README.md` ("Policy engine (Phase 5)") are what is true.

Read them in this order — each later document supersedes the earlier ones
where they differ:

| File | What it is |
| ---- | ---------- |
| `SPEC_PHASE5.md` | The binding spec the build was done against: ground rules, the contract of each module, the twenty rules, the tests asked for. |
| `DECISIONS_PHASE5.md` | Decisions D1–D4, taken after the first integrated build: finite limits only, `structure_problems` as the single judge of a proposal's shape, a runtime backstop for the Executor rule that survives Phase 6. |
| `RULINGS_R1.md` | Rulings R1–R12 on the first adversarial review (48 findings and four open architecture items): the pre-session halt, the stop flags re-read before a verdict is recorded, the extend-only halt, padded and whitespace symbols, zero caps, the runtime "who holds an Executor" check and its stated limit. |
| `README_PHASE5.md` | The README section as drafted during the build. The final text is in the main `README.md`. |

Notes for the reader:

- Paths such as `/Users/test1/aegis` are the machine the work was done on.
- `RULINGS_R1.md` cites findings by id (`1.2`, `3.14`, `NEW-1-R1-R1`). The
  findings files themselves were working data and are not kept here; the
  rulings restate what each one was about, and the Phase 5 commit message
  summarises them.
- "The suite is currently RED" in `RULINGS_R1.md` describes the tree at the
  moment the rulings were written, part-way through a fix. The phase was
  committed with every test passing.
