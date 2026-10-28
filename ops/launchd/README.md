# Running AEGIS on a schedule (launchd, on the mini)

Until the Phase 9 daemon exists, two LaunchAgents run the existing CLI
commands on a timer. They run in your logged-in user session, which is what
lets Messages send the texts. Nothing here loosens a limit or skips the gate.
The policy engine still judges every proposal, `orders tick` is still the only
thing that places, and `broker.enabled` and `notify.enabled` decide, as
always, whether anything is sent at all.

| Job | When | What |
|---|---|---|
| `local.aegis.cycle` | weekdays 10:00, 11:30, 13:30, 15:00 local (New York) time | market check, `brain once`, `policy evaluate <the id it printed>`, `orders tick`, `notify send`. The 15:00 cycle then runs `forecasts run` (the day's per-ticker forecasts) |
| `local.aegis.tick` | every 5 minutes, but only acts 09:25 to 16:30 ET on weekdays | market open: `orders tick`, `notify send`. Close to 16:30: `orders sync`, `notify send` (places nothing). From 16:20 it also runs `notify summary` before `notify send`: the end-of-day text |

Both jobs call `aegis-run.sh` with this checkout's `.venv/bin/python`. The
following rules apply:

- **Market check:** the market clock is asked first. If the market is closed,
  or the clock can't be read, the step does nothing (fail closed). That covers
  holidays and early closes.
- **Tick timing:** the tick runs straight after the verdict, because an
  AUTO_EXECUTE verdict is only acted on for `broker.max_decision_age_seconds`
  (5 minutes).
- **No overlap:** one lock (`run.lock` in the log folder) is shared by both
  jobs, so no two steps ever run at once. Two `notify send` runs could
  otherwise text the same notice twice. A step that finds the lock taken logs
  `skipped: busy`. A lock whose owner has died is cleared.
- **Forecasts:** only a cycle between 15:00 and 16:00 ET runs `forecasts run`,
  after the trading steps, so it never delays them. It makes one run per arm
  per day (a repeat skips), refuses while `forecast.enabled` is false (logged
  as `exit 1`), and nothing that trades reads what it writes.
- **Day summary:** the first tick after 16:20 writes one text about the day (cycles and
  their outcomes, verdicts, approvals and orders, equity and positions, the
  forecasts, the estimated Claude cost, and any problems, including a step
  that logged a non-zero exit in `cycle.log` or `tick.log`), and the
  `notify send` after it delivers it. A later tick finds it already
  written. On a holiday, when every cycle found the market closed, the text
  says only that. `python -m aegis.cli.notify summary --dry-run --logs
  ~/Library/Logs/AEGIS` prints today's text without writing it.
- **Logs:** they go to `~/Library/Logs/AEGIS/cycle.log` and `tick.log`. A log
  rolls over at 10 MB, keeping one old copy. The store stays the record, and
  `python -m aegis.cli.trace` shows it.

## Before installing

- **Time zone:** the mini must be on New York time, because the cycle times are
  local. `aegisctl.sh install` refuses otherwise.
- **Sleep:** the mini must not sleep during market hours. Check `pmset -g`. If
  `sleep` is above 0, turn on "Prevent automatic sleeping when the display is
  off" in System Settings.
- **Login:** LaunchAgents only run while you are logged in. After a power cut,
  nothing runs until someone logs in.
- **Cost:** each cycle makes three Claude calls, about $0.06 to $0.11 by the
  config's price estimates. `brain.daily_token_budget` caps a day whatever the
  schedule. `python -m aegis.cli.brain usage` shows the real figure. The daily
  forecast adds two calls (the primary and shadow arms); `forecasts run`
  prints each arm's tokens and estimated cost in `cycle.log`.

## Install, pause, remove

From the checkout on the mini:

```
ops/launchd/aegisctl.sh install
ops/launchd/aegisctl.sh status
ops/launchd/aegisctl.sh pause
ops/launchd/aegisctl.sh resume
ops/launchd/aegisctl.sh remove
```

What each command does:

- **`install`:** fills this checkout's path into the two plists, copies them to
  `~/Library/LaunchAgents` and loads them with `launchctl bootstrap`.
- **`status`:** shows which jobs are loaded and the last log lines.
- **`pause`:** unloads both jobs with `launchctl bootout`. The plists stay.
- **`resume`:** loads both jobs again.
- **`remove`:** unloads both jobs and deletes their plists. The logs are kept.

The first time a job sends a text, macOS may ask again whether it may control
Messages. Run `launchctl kickstart gui/$(id -u)/local.aegis.tick` during market
hours with someone at the screen to click Allow.

To stop trading but keep the schedule, use `python -m aegis.cli.policy kill
on` then `python -m aegis.cli.orders stand-down`, or set `broker.enabled:
false`. Cycles keep recording proposals, but nothing is placed.
