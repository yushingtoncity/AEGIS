# Phone approval page: design

Approved by the owner on 2026-11-04 (option A of the phone approval plan). This file is the design as built.

## The problem

A NEEDS_APPROVAL verdict sends an ntfy alert with the line `orders approve ID --by NAME`. Answering meant SSH or Remote Control into the mini, activating the venv and typing the command: four or five steps on a phone.

## What already held, and still holds

1. `dispatch.answer()` writes one approvals row per decision. A second answer is refused, and migration 0004 enforces one approval per decision with a unique index and a trigger that refuses changing an answer. A replay can never flip or repeat an answer.
2. A yes is good for `broker.approval_ttl_seconds` (900) from the moment it is given; `place` refuses it after that.
3. An answer places nothing. The 5 minute tick calls `dispatch.place()`, which re-runs the policy engine on fresh data and refuses if the re-check rejects or escalates on a rule the approval did not cover.
4. Every answer writes an `approval_answered` event in the same transaction.

The page is only a new way to deliver the owner's yes or no to `dispatch.answer()`. Nothing about what may be traded changed: no risk limit, auto execute threshold, TTL or broker setting.

## The threat

ntfy.sh is a public relay: anyone who learns the topic reads every alert, decision ids included. The mini sits on a home LAN. So knowing an alert, a link or a decision id must never be enough to answer.

## How it works

1. A stdlib HTTP server (`aegis/approve_web/server.py`) listens on `127.0.0.1:approve_web.port` only. The address is a literal, never config, pinned by `tests/test_approve_web_architecture.py`.
2. `tailscale serve` on the mini publishes it as `https://<name>.<tailnet>.ts.net` to the owner's tailnet only, with a Tailscale certificate. No router change and no open port. Funnel (public exposure) is never used.
3. When `approve_web.enabled` is on and `APPROVE_WEB_URL` checks out, a `policy_needs_approval` push carries a `Review` button and a tap target (ntfy `Actions` and `Click` headers) pointing at `/d/<decision_id>`, plus a `Review:` line in the body. The CLI hint stays. The link holds no secret.
4. The page shows the order, why it escalated, the model's thesis (escaped), and a form with a PIN box and Approve and Reject.
5. A good answer calls `dispatch.answer(..., channel="phone", by=approve_web.responder)`.

## Every request passes, in order

1. Method GET or POST and a known path (`/` or `/d/<id>`, id `[A-Za-z0-9-]{1,64}`).
2. Host is the page's own name or the loopback address Serve forwards to.
3. `Tailscale-User-Login` equals `APPROVE_WEB_LOGIN` (case folded, constant time). Serve sets this header for tailnet traffic and sets none for Funnel traffic, and since the server only listens on loopback, nothing but Serve can hand it the header.

A POST must also:

4. Come from the page itself: an `Origin` header, when present, equals the page's origin, and `Sec-Fetch-Site`, when present, is `same-origin` or `none`.
5. Be `application/x-www-form-urlencoded`, at most 2,048 bytes, each field once.
6. Redeem a one time form token. A token is issued on each GET of a decision page, bound to that decision, held only in memory, dies after `approve_web.form_ttl_seconds` (300), and is gone after its first use, right or wrong.
7. Find the page not locked: fewer than `approve_web.max_pin_failures` (5) wrong PINs in a row. A good phone answer or `approve-web unlock` resets the count.
8. Carry the right PIN, checked in constant time against the scrypt hash in `APPROVE_WEB_PIN_HASH` (N 2^15, r 8, p 1, 16 byte salt). The PIN is never stored.

## The audit trail

- The approvals row: `channel = "phone"`, `responder = approve_web.responder`, note "answered on the phone page; Tailscale login checked". The login itself is checked but never stored or pushed, because `approval_answered` events go out over ntfy.
- `approval_answered` event payload gains `channel`.
- New events, all through `aegis/store`: `approve_web_pin_failed` (warning, so it reaches the phone), `approve_web_locked` (error), `approve_web_unlocked` (warning), `approve_web_refused` (warning: a used, expired or foreign token, or a bad or cross site form).
- The server's own log is the stdlib request line and status, never headers or bodies.

## LLM proposes, deterministic code disposes

- The page never calls a model. Nothing in `aegis/approve_web` or its CLI imports `aegis.brain`, `aegis.forecasts`, `anthropic`, `alpaca`, `aegis.data.clients` or `aegis.execution`, and from the dispatcher it uses only `answer`, `awaiting_approval` and two constants. Nothing under `aegis/brain` or `aegis/forecasts` mentions the page. All pinned in `tests/test_approve_web_architecture.py`; `tests/test_policy_architecture.py` scans the package like every other one outside the policy engine.
- `dispatch.answer()` accepts only the channels in `dispatch.APPROVAL_CHANNELS` (`cli`, `phone`), fixed in code.
- The model has no path to the tailnet, the PIN or the page.

## What it does not protect against

- It assumes `tailscale serve` sets the identity header itself rather than passing on one a client sent, as Tailscale's Serve docs describe. The PIN is the second wall if that ever fails.

- Someone with the owner's unlocked phone, on the tailnet, who knows or guesses the PIN within 5 tries. The policy engine still re-checks the order, and removing the phone from the tailnet cuts it off at once.
- A process already running on the mini can reach the loopback port and forge the identity header. It would still need the PIN, and a compromised mini is already game over for every other control.
- A device on the tailnet other than the owner's that belongs to the same Tailscale login. Use a Tailscale access rule to limit the page's port to the phone (SETUP.md).
