# Phone approval page: setup on the mini

The owner does every step here. Nothing in this file is done by an agent, because it touches the mini's runtime, `.env` and a network service. Run commands one at a time; do not paste the `#` comments into zsh.

## 1. Tailscale

1. Install Tailscale on the mini and on the phone, and sign in to both with the same account.
2. On the mini, check it is up: `tailscale status`.
3. Optional but recommended: in the Tailscale admin console, add an access rule so only the phone can reach the mini's port 443.

### Without the Mac app (user mode, what the mini runs)

The Mac app needs an admin password and a VPN approval in System Settings. The Homebrew CLI in user mode needs neither, and is enough for `tailscale serve`:

1. `brew install tailscale`.
2. Save `~/Library/LaunchAgents/local.aegis.tailscaled.plist` running `/opt/homebrew/opt/tailscale/bin/tailscaled --tun=userspace-networking --statedir=<state dir> --socket=<state dir>/tailscaled.sock`, with `RunAtLoad` and `KeepAlive` true, `ThrottleInterval` 60 and its log in `~/Library/Logs/AEGIS/tailscaled.log`. Load it with `launchctl bootstrap gui/$(id -u) <plist>`.
3. Every `tailscale` command then takes `--socket=<state dir>/tailscaled.sock` before the subcommand. `tailscale --socket=... up --hostname=aegis` prints a login link; sign in from the phone.
4. The first `tailscale serve` asks you to enable Serve (and HTTPS) for the tailnet with a second link, and waits until you do.
5. In user mode the mini's own traffic does not go through the tailnet, so `curl https://aegis.<tailnet>.ts.net/` on the mini fails to connect. Test from the phone.
6. Off: `launchctl bootout gui/$(id -u)/local.aegis.tailscaled`.

## 2. Publish the page to the tailnet only

1. `tailscale serve --bg 8765` (8765 is `approve_web.port`; it forwards `https://<mini>.<tailnet>.ts.net` to `127.0.0.1:8765`).
2. `tailscale serve status` shows the URL. Never run `tailscale funnel`: Funnel would make it public, and the page refuses Funnel traffic anyway because it carries no identity header.

## 3. The .env values (never in config.yaml)

1. `cd ~/AEGIS` and `source .venv/bin/activate`.
2. `python -m aegis.cli.approve_web hash-pin` and type a PIN of 6 to 64 characters twice. Copy the printed `APPROVE_WEB_PIN_HASH=...` line into `.env`, or append it directly: `python -m aegis.cli.approve_web hash-pin | grep '^APPROVE_WEB_PIN_HASH=' >> .env`.
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
