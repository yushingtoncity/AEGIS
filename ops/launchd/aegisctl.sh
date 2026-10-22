#!/bin/bash
# Install, pause, resume or remove the AEGIS launchd jobs on the mini.
# See ops/launchd/README.md.
#
#   aegisctl.sh install   fill in this checkout's path, load both jobs
#   aegisctl.sh pause     unload both jobs (the plists stay)
#   aegisctl.sh resume    load both jobs again
#   aegisctl.sh remove    unload both jobs and delete their plists
#   aegisctl.sh status    what launchd has loaded, and the last log lines

set -u

AEGIS_HOME="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$AEGIS_HOME/ops/launchd"
AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/AEGIS"
DOMAIN="gui/$(id -u)"
JOBS="local.aegis.cycle local.aegis.tick"

die() { echo "aegisctl: $*" >&2; exit 1; }

load() {
  for job in $JOBS; do
    launchctl bootstrap "$DOMAIN" "$AGENTS/$job.plist" || die "could not load $job"
    echo "loaded $job"
  done
}

unload() {
  for job in $JOBS; do
    launchctl bootout "$DOMAIN/$job" 2>/dev/null && echo "unloaded $job"
  done
  return 0
}

install_jobs() {
  [ -x "$AEGIS_HOME/.venv/bin/python" ] || die "no $AEGIS_HOME/.venv/bin/python"
  case "$(readlink /etc/localtime)" in
    */America/New_York) ;;
    *) die "the mini's time zone is not America/New_York; the cycle times are local time" ;;
  esac
  case "$AEGIS_HOME$LOG_DIR" in
    *"|"*) die "a path holds '|'" ;;
  esac
  mkdir -p "$AGENTS" "$LOG_DIR"
  unload
  for job in $JOBS; do
    sed -e "s|__AEGIS_HOME__|$AEGIS_HOME|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
      "$HERE/$job.plist" > "$AGENTS/$job.plist"
    plutil -lint "$AGENTS/$job.plist" > /dev/null || die "$job.plist does not parse"
  done
  load
  echo "logs: $LOG_DIR"
}

case "${1:-}" in
  install) install_jobs ;;
  pause) unload ;;
  resume) load ;;
  remove)
    unload
    for job in $JOBS; do rm -f "$AGENTS/$job.plist"; done
    echo "removed; the logs in $LOG_DIR are kept"
    ;;
  status)
    launchctl list | grep -E 'local\.aegis\.' || echo "no AEGIS job is loaded"
    for log in cycle tick; do
      [ -f "$LOG_DIR/$log.log" ] && { echo "--- $log.log"; tail -n 5 "$LOG_DIR/$log.log"; }
    done
    ;;
  *) echo "usage: aegisctl.sh {install|pause|resume|remove|status}" >&2; exit 64 ;;
esac
