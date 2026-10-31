#!/bin/bash
# Run one scheduled AEGIS step on the mini. launchd calls this; see
# ops/launchd/README.md.
#
#   aegis-run.sh cycle   market open: brain once, judge its proposal, tick, notify,
#                        post to the friends' feed; the 15:00 cycle then makes the
#                        day's forecasts
#   aegis-run.sh tick    market open: tick, notify, feed; close to 16:30 ET: sync,
#                        notify, feed; from 16:20 ET it also writes the day's summary
#                        and feed recap first
#
# The feed steps pass --if-enabled: while feed.enabled is false they exit 0
# and post nothing, so a machine without the feed logs no failure.
#
# Only existing CLI commands run here: the policy engine still judges every
# proposal and `orders tick` is still the only thing that places. The market
# clock is asked first; closed, or a clock that cannot be read, means the
# step does nothing (fail closed). One lock is shared by both jobs, so no two
# steps ever run at once (two `notify send` runs could text the same notice
# twice). Environment overrides, for tests: AEGIS_PYTHON, AEGIS_LOG_DIR,
# AEGIS_NY_CLOCK ("<ISO weekday> <HHMM>", New York time).

set -u

AEGIS_HOME="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${AEGIS_PYTHON:-$AEGIS_HOME/.venv/bin/python}"
LOG_DIR="${AEGIS_LOG_DIR:-$HOME/Library/Logs/AEGIS}"
LOCK="$LOG_DIR/run.lock"
MAX_LOG_BYTES=10485760        # roll a log over at 10 MB, keeping one old copy
STALE_LOCK_MINUTES=10         # a lock with no live owner older than this is cleared
TICK_WINDOW_START=925         # the tick asks the market clock from 09:25 ET on weekdays...
WRAP_UP_END=1630              # ...to 16:30 ET: after the close it syncs and notifies, placing nothing
FORECAST_FROM=1500            # a cycle from 15:00 ET until the close also runs `forecasts run` (once a day:
FORECAST_UNTIL=1600           # it skips an arm that already ran today, and refuses while forecast.enabled is false)
SUMMARY_FROM=1620             # an after-close tick from 16:20 ET writes the end-of-day summary (once a day: the later ticks find it written)

MODE="${1:-}"
case "$MODE" in
  cycle|tick) ;;
  *) echo "usage: aegis-run.sh {cycle|tick}" >&2; exit 64 ;;
esac

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$MODE.log"
cd "$AEGIS_HOME" || exit 1

if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
  mv -f "$LOG" "$LOG.1"
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') [$MODE] $*" >> "$LOG"; }

# Run one CLI step, its output into the log, and log its exit code.
step() {
  "$PY" -m "$@" >> "$LOG" 2>&1
  local code=$?
  log "exit $code: $*"
  return $code
}

take_lock() {
  if mkdir "$LOCK" 2>/dev/null; then
    echo $$ > "$LOCK/pid"
    return 0
  fi
  local owner
  owner="$(cat "$LOCK/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    return 1
  fi
  if [ -z "$owner" ] && [ -z "$(find "$LOCK" -maxdepth 0 -mmin +"$STALE_LOCK_MINUTES" 2>/dev/null)" ]; then
    return 1  # just taken; its owner has not written its pid yet
  fi
  rm -rf "$LOCK"
  log "cleared a stale lock (owner ${owner:-unknown} is gone)"
  mkdir "$LOCK" 2>/dev/null || return 1
  echo $$ > "$LOCK/pid"
}

# "open", "closed", or "unknown" when the clock cannot be read.
market_state() {
  local state
  state="$("$PY" -c '
import sys
from aegis.data.market import get_market_clock
try:
    clock = get_market_clock()
except Exception as exc:
    print(f"market clock unreadable: {type(exc).__name__}", file=sys.stderr)
    sys.exit(2)
print("open" if clock.is_open else "closed")
' 2>> "$LOG")"
  case "$state" in
    open|closed) echo "$state" ;;
    *) echo unknown ;;
  esac
}

# New York weekday (1-7) and time as HHMM, for the tick's window.
NY_NOW="${AEGIS_NY_CLOCK:-$(TZ=America/New_York date '+%u %H%M')}"
NY_DAY="${NY_NOW%% *}"
NY_HHMM="$((10#${NY_NOW##* }))"

in_tick_window() {
  [ "$NY_DAY" -le 5 ] && [ "$NY_HHMM" -ge "$TICK_WINDOW_START" ] && [ "$NY_HHMM" -lt "$WRAP_UP_END" ]
}

run_cycle() {
  local state output proposal
  state="$(market_state)"
  if [ "$state" != open ]; then
    log "skipped: market $state"
    return 0
  fi
  log "start"
  output="$("$PY" -m aegis.cli.brain once 2>&1)"
  log "exit $?: aegis.cli.brain once"
  printf '%s\n' "$output" >> "$LOG"
  proposal="$(printf '%s\n' "$output" | sed -n 's/^proposal \([0-9a-fA-F-]\{36\}\)$/\1/p' | head -n 1)"
  if [ -n "$proposal" ]; then
    step aegis.cli.policy evaluate "$proposal"
  fi
  step aegis.cli.orders tick
  step aegis.cli.notify send
  step aegis.cli.notify feed-send --if-enabled
  # last, so the trading steps never wait on it; it reads nothing they write
  if [ "$NY_HHMM" -ge "$FORECAST_FROM" ] && [ "$NY_HHMM" -lt "$FORECAST_UNTIL" ]; then
    step aegis.cli.forecasts run
  fi
  log "done"
}

run_tick() {
  local state
  state="$(market_state)"
  if [ "$state" = open ]; then
    step aegis.cli.orders tick
    step aegis.cli.notify send
    step aegis.cli.notify feed-send --if-enabled
  elif [ "$state" = closed ] && [ "$NY_HHMM" -ge 1600 ]; then
    step aegis.cli.orders sync
    if [ "$NY_HHMM" -ge "$SUMMARY_FROM" ]; then
      step aegis.cli.notify summary --logs "$LOG_DIR"
      step aegis.cli.notify feed-recap --if-enabled
    fi
    step aegis.cli.notify send
    step aegis.cli.notify feed-send --if-enabled
  elif [ "$state" = unknown ]; then
    log "skipped: market clock unreadable"
  fi
}

if [ "$MODE" = tick ] && ! in_tick_window; then
  exit 0  # nights and weekends: no lock, no clock, no log line
fi
if ! take_lock; then
  log "skipped: busy (another AEGIS step holds $LOCK)"
  exit 0
fi
trap 'rm -rf "$LOCK"' EXIT

if [ "$MODE" = cycle ]; then run_cycle; else run_tick; fi
