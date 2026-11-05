# Phone approval page: setup on the mini

Every step here changes the mini's runtime, `.env` or a network service, so it runs only with the owner's approval. The sign ins, the Serve approval and the PIN are always the owner's own. Run commands one at a time; do not paste `#` comments into zsh.

## 1. Tailscale

The Mac app needs the Mac's admin password and a System Settings approval. The command line build from Homebrew needs neither, so that is what runs on the mini, in user mode networking:

1. `brew install tailscale`.
2. Run `tailscaled` in user mode under a user LaunchAgent (`~/Library/LaunchAgents/local.aegis.tailscaled.plist`, RunAtLoad and KeepAlive on, ThrottleInterval 60, log `~/Library/Logs/AEGIS/tailscaled.log`), so it comes back after a reboot and login.
3. `tailscale up --hostname aegis` prints a login link; the owner opens it and signs in.
4. Install the Tailscale app on the phone and sign in to the same account.
5. `tailscale status` lists the mini and the phone.
6. Optional but recommended: in the Tailscale admin console, add an access rule so only the phone can reach the mini's port 443.

In user mode the mini's own traffic does not go through Tailscale, so the mini cannot open its own tailnet URL (curl from the mini gives no connection). That is expected; test from the phone.

## 2. Publish the page to the tailnet only

1. `tailscale serve --bg 8765` (8765 is `approve_web.port`; it forwards `https://<mini>.<tailnet>.ts.net` to `127.0.0.1:8765`). The first time, Tailscale says Serve is not enabled on the tailnet and prints a link; the owner opens it and approves (it also turns on HTTPS certificates), and the waiting command finishes.
2. `tailscale serve status` shows the URL. Never run `tailscale funnel`: Funnel would make it public, and the page refuses Funnel traffic anyway because it carries no identity header.

## 3. The .env values (never in config.yaml)

1. `cd ~/AEGIS` and `source .venv/bin/activate`.
2. The owner runs `python -m aegis.cli.approve_web hash-pin | grep '^APPROVE_WEB_PIN_HASH=' >> .env` and types a PIN of 6 to 64 characters twice (nothing is echoed; if the two differ, nothing is added). `.env` is hidden in Finder because its name starts with a dot; `open -e .env` opens it in TextEdit.
3. Add `APPROVE_WEB_URL=` the URL from `tailscale serve status`, with no path.
4. Add `APPROVE_WEB_LOGIN=` your Tailscale login, as `tailscale status` or the admin console shows it.
5. `python -m aegis.cli.approve_web status` should say all three are set.

## 4. Turn it on

1. In the mini's local `config.yaml`, set `approve_web.enabled: true` (an uncommitted edit, like `broker.enabled`).
2. `ops/launchd/aegisctl.sh approve-web-install` loads the page as its own launchd job (it is not part of `aegisctl.sh install`). Its log is `~/Library/Logs/AEGIS/approve-web.log`.
3. From the phone, open the URL. You should see "Waiting on you".
4. The next approval alert carries a Review button.

## Off switch

- Fastest: `ops/launchd/aegisctl.sh approve-web-remove`, or `tailscale serve reset`.
- Or set `approve_web.enabled: false`; the job then stops at its next restart and alerts lose the Review link.
- A lost phone: remove it from the tailnet in the admin console.

## Live checklist (the owner, on the mini)

- [ ] `approve-web status` shows enabled, all three values set, `0 of 5 (open)`.
- [ ] From the phone over cellular (wifi off), the URL opens the list.
- [ ] From a browser not on the tailnet, the URL does not load.
- [ ] `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/` on the mini prints `403` (no identity header).
- [ ] A wrong PIN shows "4 tries left" and an ntfy warning arrives.
- [ ] On the next real approval alert, Review opens the order, the PIN plus Approve records it, `orders pending` lists it as ready to place, and `trace` shows the approval with channel `phone`.
