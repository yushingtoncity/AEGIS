# AEGIS Status

Code facts verified at main `1cadcc3` (merge of PR #15) on 2026-10-20.
If main has moved past `1cadcc3` with changes outside docs/, re-check before relying on this file.

## How to use this file

- Canonical status for AEGIS. Code and config at a commit outrank this file; this file outranks PR descriptions, assistant memory, chat summaries and older docs.
- Read it at the start of any AEGIS session, then check whether main has moved.
- Changed only through reviewed pull requests, ideally in the same PR as the change it describes. Claude Code writes it; ChatGPT and the Claude Project read and review it; the owner approves and merges.
- Labels: **[Code]** read from the repo at the commit above. **[Test]** recorded test evidence. **[Reported]** stated by the owner, not re-checked.
- Public repository: no credentials, machine names, addresses, usernames, local paths, account details, balances, positions, P&L or unpublished research.
- Agent rules live in [CLAUDE.md](../CLAUDE.md). Architecture and phase history live in [README.md](../README.md). This file links to them instead of repeating them.

## What AEGIS is

AEGIS is an agentic trading platform on an Alpaca paper account. The LLM proposes and deterministic code disposes: the brain writes proposals to the store, and the policy engine gives each one a verdict. Only the policy engine's dispatcher calls an Executor, and the only Executor is the paper one. See README "Architecture principles" and CLAUDE.md "Non-negotiable invariants". [Code]

## Component status

| Component | State | Evidence |
| --- | --- | --- |
| Data layer, `aegis/data` | active | [Code] |
| Pricing, `aegis/pricing` | active | [Code] |
| Store / audit log, `aegis/store` | active | [Code] |
| Brain (scan, thesis, proposal), `aegis/brain` | active; stages can track the newest model per config (`aegis/brain/model_choice.py`, `brain.stages.*.track`) | [Code] |
| Policy engine (21 rules), `aegis/policy` | active (`aegis/policy/rules.py`, `RULES`) | [Code] |
| Execution | shipped, off by default (`broker.enabled: false`). `PaperExecutor` in `aegis/execution/paper.py` is the only Executor; its one caller is `aegis/policy/dispatch.py`; manual orders via `aegis/cli/orders.py` | [Code] |
| Notifications, `aegis/notify` (iMessage, Slack, ntfy) | shipped, off by default (`notify.enabled: false`) | [Code] |
| Proposal scorecard, `aegis/scorecard` and `aegis/cli/scorecard.py` | shipped, read-only | [Code] |
| Dashboard, `aegis/dashboard` | stub; Phase 8 not approved | [Code] |
| Scheduler daemon (Phase 9) | not built | [Code] |

## Safety defaults [Code]

- `config.yaml` `broker.enabled: false`; `broker.kind` accepts only `paper` (`aegis/config.py`, `BrokerConfig`).
- `config.yaml` `notify.enabled: false` (`aegis/config.py`, `NotifyConfig`).
- Tests pin both shipped defaults: `tests/test_config.py::test_shipped_broker_block_matches_the_code_defaults` and `tests/test_notify.py::TestConfig::test_shipped_block_is_the_code_defaults_and_off`.
- `.env` is gitignored; `.env.example` lists key names with empty values.
- Turning broker or notifications on is the owner's action only (CLAUDE.md invariant 8).

## Approved decisions

- The LLM proposes and deterministic code disposes; nothing under `aegis/brain` imports `aegis.execution` (CLAUDE.md invariants 1 and 2; `tests/test_brain_architecture.py`). [Code]
- Paper trading only; live trading needs separate explicit authorization (CLAUDE.md invariant 8). [Code]
- Risk limits live in config and are never loosened without owner approval (CLAUDE.md invariant 3). [Code]
- Branches and pull requests only; the owner reviews and merges (CLAUDE.md "How to work here"). [Code]
- The proposal stage runs on the newest Opus rather than Fable until the scorecard can compare them (commit `d1e8540`, PR #11). [Code]
- GitHub main is the source of truth, Windows is the development machine, and the Mac is the paper runtime (owner decision, 2026-10-20). [Reported]
- This file is the single canonical status, written only through reviewed PRs (owner decision, 2026-10-20). [Reported]

## Environments

| Role | Purpose | Last known commit | Label |
| --- | --- | --- | --- |
| GitHub main | source of truth | `1cadcc3` | [Code] |
| Windows dev | development | main at `1cadcc3` when this file was written | [Code] |
| Mac runtime | paper operations | `fff521a`, one PR behind main | [Reported] |

- PR #15 changed only `CLAUDE.md` and two test files (`git diff --stat fff521a 1cadcc3`), so the Mac's runtime code is identical to main. [Code]
- Backups and credential-file permissions were set up on 2026-10-20; details are kept private. [Reported]

## Tests

| Environment | Command | Commit | Passed | Skipped | Failed | Label |
| --- | --- | --- | --- | --- | --- | --- |
| Windows dev | `python -m pytest -q` | `d7149a1` (PR #15 head) | 5,419 | 1 | 0 | [Test] (source: owner report) |
| Mac | `python -m pytest -q` in an isolated temporary checkout | `d7149a1` | 5,420 | 0 | 0 | [Test] (source: owner report) |

- Main `1cadcc3` has the same tree as `d7149a1`, so these results apply to main unchanged. [Code]
- The Windows skip is intentional: `tests/test_brain_architecture.py::TestRuntime::test_the_harness_catches_a_load_by_any_name[symlink]` skips when the account can't create symlinks (Windows without Developer Mode or elevation). [Code]
- Before PR #15, Windows had 3 portability failures; PR #15 fixed them in `tests/test_brain_architecture.py` and `tests/test_pricing.py`. Resolved. [Code] [Test]
- Test policy: documentation-only changes don't need a new run. Rerun when code, tests, config or dependencies change, and record the counts with the commit.

## Open issues

1. README.md is partly outdated [Code]:
   - the intro still says "Phase 0–5.1" and that nothing trades and no Executor exists until Phase 6;
   - the Policy engine section repeats "no Executor implementation yet";
   - the Agent brain stage table lists the code-default models, not the shipped `config.yaml` stages, which track the newest Haiku and Opus since PR #11;
   - the Roadmap marks 5.1 as current and omits the scorecard;
   - the Setup note on `ANTHROPIC_API_KEY` ("can stay empty until Phase 4") is dated;
   - the repo-layout test list omits the notify, orders CLI and scorecard tests.

   The Execution, Notifications and Proposal scorecard sections are current. Fix in a separate docs PR.
2. Mac runtime checks pending: update the Mac to main, first scorecard run, notification test, brain model-selection check. [Reported]
3. Merged feature branches remain on the remote, every one an ancestor of main (`git ls-remote --heads origin`, `git merge-base --is-ancestor`): `claude/auto-latest-models-z7rxu6`, `claude/offline-aegis-work-5sytso`, `claude/option-close-tier-detail-8pa3e5`, `claude/project-thread-tycmn6`, `claude/proposal-scorecard-pzq07u`, `claude/vigilant-johnson-h8jcnq`, `claude/windows-test-portability-agent-rules`, `phase-5-policy-engine`. Deleting them needs owner approval. [Code]
4. `notify.enabled` is validated as a plain `bool` while `broker.enabled` uses `StrictBool` (`aegis/config.py`). Aligning them is an owner decision. [Code]

## Next priorities (set by owner)

1. Refresh the outdated README sections (docs PR).
2. Mac runtime checks (above).
3. Delete merged remote branches (owner approval).
4. Choose the next track: the Phase 8 read-only dashboard or the research roadmap built on scorecard evidence.

## Recent changes

Merged pull requests, newest first (`git log --merges main`). Git history is the full record.

- #15 test: Windows portability for three offline tests; agent safety rules in CLAUDE.md
- #14 feat: read-only proposal scorecard
- #13 fix: describe an option closing sale accurately in auto_tier
- #11 feat: brain stages follow the newest Haiku and Opus
- #10 Phase 7: notifications over the event log (iMessage, Slack, ntfy)
- #12 Let the policy engine close an option the account holds long
- #9 Phase 6c follow-up: act on the independent review
- #7 Phase 6c: place paper orders (dispatcher, orders CLI, broker config)
- #8 fix: refuse an approval that could never be placed (onto #7)
- #6 Phase 6b: execution package (PaperExecutor, typed Executor, D8 rule)
- #5 Phase 6a: store side of execution (migration 0004, claim_order, apply_broker_update)
- #4 Phase 5.1: expiry floor for the brain's chains, and the quote_freshness rule
- #3 Chore/ecc project setup
- #2 Phase 5: policy engine
