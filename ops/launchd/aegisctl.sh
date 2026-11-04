#!/bin/bash
# Install, pause, resume or remove the AEGIS launchd jobs on the mini.
# See ops/launchd/README.md.
#
#   aegisctl.sh install   fill in this checkout's path, load both jobs
#   aegisctl.sh pause     unload both jobs (the plists stay)
#   aegisctl.sh resume    load both jobs again
#   aegisctl.sh remove    unload both jobs and delete their plists
#   aegisctl.sh status    what launchd has loaded, and the last log lines
#   aegisctl.sh approve-web-install   load the phone approval page job on its own
#   aegisctl.sh approve-web-remove    unload it and delete its plist

set -u

AEGIS_HOME="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$AEGIS_HOME/ops/launchd"
AGENTS="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/AEGIS"
DOMAIN="gui/$(id -u)"
JOBS="local.aegis.cycle local.aegis.tick"
WEB_JOB="local.aegis.approve-web"   # opt in, never part of JOBS: see docs/approve_web/SETUP.md

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

install_web() {
  [ -x "$AEGIS_HOME/.venv/bin/python" ] || die "no $AEGIS_HOME/.venv/bin/python"
  case "$AEGIS_HOME$LOG_DIR" in
    *"|"*) die "a path holds '|'" ;;
  esac
  mkdir -p "$AGENTS" "$LOG_DIR"
  launchctl bootout "$DOMAIN/$WEB_JOB" 2>/dev/null
  sed -e "s|__AEGIS_HOME__|$AEGIS_HOME|g" -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$HERE/$WEB_JOB.plist" > "$AGENTS/$WEB_JOB.plist"
  plutil -lint "$AGENTS/$WEB_JOB.plist" > /dev/null || die "$WEB_JOB.plist does not parse"
  launchctl bootstrap "$DOMAIN" "$AGENTS/$WEB_JOB.plist" || die "could not load $WEB_JOB"
  echo "loaded $WEB_JOB; log: $LOG_DIR/approve-web.log"
}

case "${1:-}" in
  install) install_jobs ;;
  approve-web-install) install_web ;;
  approve-web-remove)
    launchctl bootout "$DOMAIN/$WEB_JOB" 2>/dev/null && echo "unloaded $WEB_JOB"
    rm -f "$AGENTS/$WEB_JOB.plist"
    ;;
  pause) unload ;;
  resume) load ;;
  remove)
    unload
    for job in $JOBS; do rm -f "$AGENTS/$job.plist"; done
    echo "removed; the logs in $LOG_DIR are kept"
    ;;
  status)
    launchctl list | grep -E 'local\.aegis\.' || echo "no AEGIS job is loaded"
    for log in cycle tick approve-web; do
      [ -f "$LOG_DIR/$log.log" ] || continue
      echo "--- $log.log"
      tail -n 5 "$LOG_DIR/$log.log"
    done
    ;;
  *) echo "usage: aegisctl.sh {install|pause|resume|remove|status|approve-web-install|approve-web-remove}" >&2; exit 64 ;;
esac
