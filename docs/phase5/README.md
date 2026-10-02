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
| `FINDINGS_R1.json` | The 48 findings of the first adversarial review, as the reviewers reported them: id, file, severity, title, detail, repro and suggested fix. `RULINGS_R1.md` rules on each by id. |
| `UNRESOLVED_DECISIONS.json` | The four architecture items (`NEW-1-R1-R1`, `D3-R2-R1`, `NEW-1-R1-R2`, `NEW-1-R1-R3`) still open after the decisions loop, with their repros; ruling R10 closes them. |

Notes for the reader:

- Paths such as `/Users/test1/aegis` are the machine the work was done on.
- `RULINGS_R1.md` cites findings by id (`1.2`, `3.14`, `NEW-1-R1-R1`); the
  two JSON files hold them. They are the reviewers' raw reports: a repro or
  line number in them describes the tree as it was then, and a "suggested
  fix" is a proposal the rulings accepted, changed or declined.
- "The suite is currently RED" in `RULINGS_R1.md` describes the tree at the
  moment the rulings were written, part-way through a fix. The phase was
  committed with every test passing.
