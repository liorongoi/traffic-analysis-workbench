#!/bin/bash
# Runs the LanePulse server of an install made without root (install-server.sh as an ordinary user).
#   run-server.sh               in the foreground (what systemd runs)
#   run-server.sh --background  starts it in the background unless it is running already
#   run-server.sh --ensure      the same, unless it was stopped on purpose (lanepulse stop); cron runs this
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
CONF="$BASE/lanepulse.conf"
set -a
# shellcheck disable=SC1090
. "$CONF"
set +a
LOG="$BASE/logs/server.log"
PIDFILE="$BASE/logs/server.pid"
mkdir -p "$BASE/logs"
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-}" in
  --background|--ensure)
    if [ "$1" = --ensure ] && [ -e "$BASE/stopped" ]; then exit 0; fi
    rm -f "$BASE/stopped"
    running && exit 0
    # A new session, so it keeps running after the terminal or SSH connection that started it closes.
    setsid "$0" </dev/null >/dev/null 2>&1 &
    exit 0;;
esac
# The log is kept under 10 MB, with the 3 before it.
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 10485760 ]; then
  for i in 2 1; do [ -f "$LOG.$i" ] && mv -f "$LOG.$i" "$LOG.$((i + 1))"; done
  mv -f "$LOG" "$LOG.1"
fi
echo $$ > "$PIDFILE"
export TWB_MODELS="$MODELS"
exec "$BASE/current/python/bin/python3" -c "import sys; from trafficwb.web.app import main; sys.exit(main(sys.argv[1:]))" \
  serve --lan --data "$DATA" --port "$PORT" --backup-dir "$BACKUPS" >> "$LOG" 2>&1
