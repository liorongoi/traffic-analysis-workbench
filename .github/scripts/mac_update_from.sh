#!/bin/bash
# Updating from an earlier LanePulse Mac build, as a person does it: the earlier app is opened (and set up, if it is
# the online kind) and left running with data in ~/LanePulse, then the new app is dragged over it and opened. Checks
# that the new version is the one answering, that SAM 3.1 is offered and can be chosen, and that the data is kept.
#   mac_update_from.sh <earlier .dmg> <new .dmg> <new version>
set -euxo pipefail
OLD_DMG="$1"; NEW_DMG="$2"; VER="$3"
: "${LANEPULSE_HOME:?}"
APP="$RUNNER_TEMP/Applications/LanePulse.app"
URL=http://127.0.0.1:8765
PYH="$pythonLocation/bin/python3"
health() { curl -fsS -m 5 "$URL/api/health"; }
copy_app() {
  rm -rf "$APP"; mkdir -p "$(dirname "$APP")"
  hdiutil attach -nobrowse -mountpoint "$RUNNER_TEMP/mnt" "$1"
  cp -R "$RUNNER_TEMP/mnt/LanePulse.app" "$APP"
  hdiutil detach "$RUNNER_TEMP/mnt"
}

echo "== The earlier build"
copy_app "$OLD_DMG"
"$APP/Contents/MacOS/LanePulse"
health; echo
mkdir -p "$LANEPULSE_HOME/data/keep-me"

echo "== The new build dragged over it, and opened, while the earlier one runs"
copy_app "$NEW_DMG"
"$APP/Contents/MacOS/LanePulse"
v="$(health | "$PYH" -c 'import json,sys; print(json.load(sys.stdin).get("version", ""))')"
test "$v" = "$VER" || { echo "the earlier version is still the one answering ($v)"; exit 1; }
test -d "$LANEPULSE_HOME/data/keep-me" || { echo "the update lost the data folder"; exit 1; }
curl -fsS "$URL/api/sam3" | "$PYH" -c 'import json,sys; s = json.load(sys.stdin); print({k: s[k] for k in ("installed", "reason", "memory_gb")}); assert s["installed"], s'
"$PYH" .github/scripts/mac_sam_screen.py "$URL" "$RUNNER_TEMP/mac-new-mission.png"

echo "== Opened again with the internet blocked: it starts without downloading"
"$LANEPULSE_HOME/Stop LanePulse.command"
for _ in $(seq 30); do health >/dev/null 2>&1 || break; sleep 1; done
HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 "$APP/Contents/MacOS/LanePulse"
health; echo
"$LANEPULSE_HOME/Stop LanePulse.command"
echo "Update OK."
