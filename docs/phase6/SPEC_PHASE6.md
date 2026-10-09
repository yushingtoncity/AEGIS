# Phase 6: execution on the paper account

The spec Phase 6 is built against. It comes from a planning pass (five
readers over the code, three independent designs, one critic that checked
each design against the code and the installed alpaca-py) and from the
user's answers to the decisions below. Where this file and the code differ
once a PR lands, the code and the main README are what is true.

## What Phase 6 does

Puts policy-approved orders on the Alpaca **paper** account and records
everything that happens to them, through the store, so every order stays
reconstructible (CLAUDE.md invariant 5).

In scope:

- Equity LIMIT DAY orders, placed after an AUTO_EXECUTE verdict or after a
  human approves a NEEDS_APPROVAL one.
- Single-leg option LIMIT DAY orders, after a human approves (options never
  reach the auto tier).

Refused when an `ApprovedOrder` is minted, so they fail closed:
multi-leg (MLEG) orders, market/stop/bracket orders, GTC on options,
replace, streaming updates, automatic flattening, and the daemon (Phase 9).

## Decisions (approved by the user)

| # | Decision | Answer |
| - | -------- | ------ |
| D1 | New timing gates (CLAUDE.md invariant 3) | Approved: `max_decision_age_seconds: 300`, `approval_ttl_seconds: 900`, `not_found_grace_seconds: 120`, in a new `broker:` config block, not under `risk_limits`. |
| D2 | Does AUTO_EXECUTE place real paper orders? | Only through `orders place` / `orders tick`, and only with `broker.enabled: true` (shipped `false`). `policy evaluate` never places. |
| D3 | Re-gate for approved orders | A full re-evaluation at submit time, recorded as a `pre_submit` decision; an approved order passes only if every rule that escalates was already escalating when the human approved. |
| D4 | Kill switch / halt vs working orders | `orders stand-down` (also run by `orders tick`) cancels working orders. `policy kill on` stays a store-only write and prints the hint. |
| D5 | Flatten on a daily-loss trip | No. `close_position` is implemented and tested; nothing calls it. |
| D6 | Options scope | Single-leg only, approval only. Multi-leg refused until the debit/credit sign is checked live. |
| D7 | Approval before Phase 7 | CLI `orders pending` / `approve` / `reject`, write-once, tied to one decision. |
| D8 | Tighten architecture rule (f) | Yes: Alpaca's other order calls (`cancel_order_by_id`, `cancel_orders`, `replace_order_by_id`, `close_all_positions`, `exercise_options_position`) are banned outside `aegis/data` and `aegis/execution`. |
| D9 | Daily trade cap | An order counts from the moment it is claimed (`submitted_at` set at the claim), so a crash cannot lose it; an order not found at the broker after the grace period becomes `cancelled` and still counts. |
| D10 | An order at the broker the store does not know | CRITICAL `orphan_broker_order` event, and the kill switch goes on. |
| D11 | Retry | Never automatic. One order per proposal; a failed order needs a new proposal. |
| D12 | Indicative option feed | Options stay approval-only. Check the feed live on the mini; consider the real-time options feed before any option auto tier. |

## The flow (`orders place DECISION_ID`)

1. **Eligibility.** The decision is AUTO_EXECUTE and younger than
   `max_decision_age_seconds`, or NEEDS_APPROVAL with an approval row tied to
   that decision, `response = approved`, not past `expires_at`. The proposal
   has no order yet. `broker.enabled` is true.
2. **Re-gate.** A fresh `build_context` and `engine.evaluate`, recorded as a
   `policy_decisions` row with `purpose = 'pre_submit'`.
3. **Claim** (`repo.claim_order`, one transaction): re-read the controls
   (fail closed on the kill switch, an active halt or an unreadable one),
   recount today's orders against `max_daily_trades`, insert the order row
   as `approved` with `submitted_at` set, plus an `order_claimed` event.
4. **Submit.** The executor verifies the row and the decision/approval links
   read-only, then sends exactly one request.
5. **Outcome.** A receipt moves the row to `submitted`. A definite rejection
   is `failed`. Anything unclear (timeout, 5xx, duplicate id, a body that
   will not parse) stays `approved` with `status_reason = 'unknown_outcome'`,
   a CRITICAL event, and one lookup. Nothing retries on its own.

`orders sync` polls the broker: fills are recorded as deltas of the
cumulative filled quantity, so replaying a sync records nothing twice.

## The PRs

### 6a: the store

Migration `0004_execution.sql`: ADD COLUMN, indexes and triggers only. The
`orders` table is not rebuilt (the foreign key from `fills` would block it).

- `orders`: `decision_id`, `regate_decision_id`, `approval_id` (foreign keys,
  NULL on rows written before Phase 6), `instrument`, `order_type`,
  `time_in_force`, `position_intent`, `filled_quantity` (default 0),
  `avg_fill_price`, `broker_status`, `status_reason`, `last_synced_at`.
  Partial UNIQUE indexes: `broker_order_id` where set; `proposal_id` where
  `decision_id` is set (one placed order per proposal).
- `fills`: `broker_fill_id`, partial UNIQUE.
- `approvals`: `decision_id`, `expires_at`, `note`.
- `policy_decisions`: `purpose` (`evaluate` | `pre_submit`, default
  `evaluate`).
- Triggers, on execution-era rows (`decision_id` set) only, so every row
  written before Phase 6 behaves exactly as it did:
  - born `approved` with `submitted_at` set;
  - status moves only forward: `approved` → `submitted` | `partially_filled`
    | `filled` | `cancelled` | `failed`; `submitted` → `partially_filled` |
    `filled` | `cancelled` | `failed`; `partially_filled` → `filled` |
    `cancelled`; `filled`, `cancelled` and `failed` never change;
  - `filled_quantity` never decreases and never exceeds `quantity`;
  - the order's identity (proposal, decision links, client id, broker,
    instrument, symbol, side, quantity, limit price, type, time in force,
    intent) never changes, and `broker_order_id` is set at most once;
  - an order with anything filled never becomes `failed` (a failed order
    does not count toward the daily cap);
  - a claimed order is never deleted, and its fills are never changed or
    deleted.
  - An approval's answer is written once; its decision link, proposal and
    expiry never change; it answers a decision on its own proposal.
  - A fill's price is positive.

Repository: `claim_order`, `apply_broker_update`, `get_order_by_broker_id`,
`get_proposal_order`, `get_decision`, `get_decision_approval`. The existing
`upsert_order` and `record_fill` refuse execution-era rows (they change
only through `claim_order` and `apply_broker_update`); their behaviour on
legacy rows is unchanged. `claim_order` also refuses a pre_submit re-check
of REJECT or FLAG_ONLY, and an AUTO_EXECUTE order whose re-check now
needs approval. `record_decision` writes `purpose`; `record_approval` writes the
new approval columns.

### 6b: the execution package

`aegis/execution/models.py` (`ApprovedOrder`, `OrderReceipt`,
`ExecutionError` with an `outcome` of `rejected` / `not_sent` / `unknown`),
typed signatures in `base.py`, and `aegis/execution/paper.py`:
`PaperExecutor`, the only place a trading client is built for orders. It
asserts the paper host (and its HTTP session refuses any other), sets
alpaca-py's retries to zero, gives every request a timeout, and reads raw
JSON. `submit_order` verifies the ApprovedOrder against the claimed store
row and the controls, read-only, before it sends anything. All of it is
tested against a fake client, and the hardening through alpaca-py's own
request path with a recording transport.

As built:

- `base.py` imports its models relatively (an absolute import would put the
  identifier `execution` in `base.py`) and only under `TYPE_CHECKING`: the
  brain's architecture harness loads `base.py` by file path, outside any
  package, to prove it catches such a load, so the file must run on its own.
- The interface gained one read, `get_order(client_order_id)`, beside
  `get_open_orders`: it is how a caller follows an order, and resolves an
  unknown outcome. Like `get_open_orders` it is a read that the store also
  defines, so the architecture test lists it as an interface read, not as
  an order method.
- D8 is rule (g) of `tests/test_policy_architecture.py`, enforced on every
  module outside `aegis/execution`, `aegis/policy` and `aegis/data`
  included (stricter than "outside data and execution": no data module
  needs those calls).
- `ApprovedOrder.quantity` is a whole number (fractional shares are out of
  scope); `limit_price` is a Decimal with at most 4 places.

Notes for 6c, from the 6b review (what the executor deliberately leaves to
the dispatcher):

- Mint, then claim. Build the `ApprovedOrder` from the proposal, then the
  store `Order` from it (`quantity=float(...)`, `limit_price=float(...)`),
  and claim that. The reverse needs `int()` and a Decimal from the float.
- Claim and send in one call. The executor sends a claimed row it has never
  seen before, whatever its age; it does not check how long ago it was
  claimed, nor the approval's expiry again. `place` must send right after
  its own claim, and only then.
- Record every attempt. An interrupted send (Ctrl-C mid-request) leaves the
  row `approved` with nothing on record, which the executor would send
  again. `place` writes `status_reason` for any outcome but a receipt, and
  `sync` resolves an `approved` row by `get_order` before anything else.
- The static rules read names. A module that calls the data layer's
  trading client with alpaca-py's generic `post` / `delete` on `/orders`
  names nothing rules (f) and (g) look for; review catches that, the rules
  do not.

### 6c: dispatch, CLI, config, docs

`aegis/policy/dispatch.py` (the only caller of an Executor; not one of the
pure modules), `python -m aegis.cli.orders` (`pending`, `status`, `approve`,
`reject`, `place [--dry-run]`, `sync`, `cancel`, `stand-down`, `tick`), the
`broker:` config block, README, and the mini checklist.

## Live checks for the mini (before 6c merges)

1. `client_order_id` length and charset limits; what Alpaca returns for a
   duplicate id.
2. With retries at zero, a 429/504 sends exactly one request; the HTTP
   timeout fires.
3. Equity limit order lifecycle, marketable and not, in and after hours.
4. Cancelling a filled order (expect 422) and an already-cancelled one.
5. A single-leg option DAY limit order; GTC on an option is refused.
6. MLEG debit/credit sign (before MLEG comes into scope).
7. `close_position` on an OCC symbol.
8. Kill the process mid-`place`; `sync` adopts the order or marks it
   `cancelled`.
9. Kill switch on with a working order; `tick` cancels it.
10. Whether indicative-feed option quotes carry fresh timestamps while their
    prices lag (the Phase 5.1 run showed 0.85 s old leg quotes).
