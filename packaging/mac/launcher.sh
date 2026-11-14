#!/bin/bash
# LanePulse.app: starts the LanePulse server if it is not running, then opens it in the browser.
# Everything LanePulse keeps (data, backups, models, logs) is in ~/LanePulse; the app itself changes nothing
# inside its own folder. The first time it is opened, and after an update, it sets up what it runs on (Python, the
# packages and the models) in ~/LanePulse/runtime, from the files the app carries (no internet), in a Terminal window
# that shows the progress.
APP="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$APP/Contents/Resources"
VER="$(cat "$RES/version.txt")"
BUILD="$(cat "$RES/build.txt" 2>/dev/null || echo "$VER")"
HOME_DIR="${LANEPULSE_HOME:-$HOME/LanePulse}"
DATA="$HOME_DIR/data"; MODELS="$HOME_DIR/models"; BACKUPS="$HOME_DIR/backups"; LOGS="$HOME_DIR/logs"
RT="$HOME_DIR/runtime"; PY="$RT/python/bin/python3"
PORT="${LANEPULSE_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
LAUNCH="$HOME_DIR/.launch"
mkdir -p "$DATA" "$MODELS" "$BACKUPS" "$LOGS" "$LAUNCH"

# The launch screen: a page in the browser that shows the version and what the app is doing, then moves on to
# LanePulse when it answers. It reads status.js, which lp_status rewrites: lp_status <state> [text] [detail] [step].
# States: check, setup, start, ready, failed.
js() { printf '%s' "$1" | tr '\r\n\t' '   ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | cut -c1-400; }
lp_status() {
  printf 'lpStatus({"state":"%s","text":"%s","detail":"%s","step":"%s","version":"%s","url":"%s"});\n' \
    "$1" "$(js "${2:-}")" "$(js "${3:-}")" "${4:-}" "$VER" "$URL" > "$LAUNCH/status.js.tmp" && mv -f "$LAUNCH/status.js.tmp" "$LAUNCH/status.js"
}
PAGE=""
if [ -z "${LANEPULSE_NO_BROWSER:-}" ] && [ -f "$RES/launch/index.html" ]; then
  rm -rf "$LAUNCH/fonts"; [ ! -d "$RES/launch/fonts" ] || cp -R "$RES/launch/fonts" "$LAUNCH/fonts"
  sed -e "s|__LP_VERSION__|$VER|g" -e "s|__LP_BUILD__|$BUILD|g" -e "s|__LP_URL__|$URL|g" "$RES/launch/index.html" > "$LAUNCH/index.html"
  PAGE="$LAUNCH/index.html"
fi
# (While a setup runs, its window keeps the status up to date.)
{ [ -f "$HOME_DIR/.setup.pid" ] && kill -0 "$(cat "$HOME_DIR/.setup.pid")" 2>/dev/null; } || lp_status check
# Opened again by the setup window when it is done: the launch screen is already showing.
[ -z "$PAGE" ] || [ -n "${LANEPULSE_PAGE_OPEN:-}" ] || open "$PAGE" 2>/dev/null || PAGE=""

fail() {
  lp_status failed "$1" "" "${2:-start}"
  osascript -e "display alert \"LanePulse\" message \"$1\" as critical" >/dev/null 2>&1 || echo "$1" >&2
  exit 1
}
up() { curl -fsS -m 2 "$URL/api/health" >/dev/null 2>&1; }

echo "$APP" > "$HOME_DIR/.app-path"

# After an update, stop the older version this app started, so the new one runs.
if up && [ -f "$HOME_DIR/.server.pid" ] && [ "$(cat "$HOME_DIR/.server-version" 2>/dev/null)" != "$BUILD" ]; then
  kill "$(cat "$HOME_DIR/.server.pid")" 2>/dev/null || true
  for _ in $(seq 30); do up || break; sleep 1; done
fi

# First time, or a new build: set up the runtime. Only one setup at a time.
if [ "$(cat "$RT/.build" 2>/dev/null)" != "$BUILD" ]; then
  LOCK="$HOME_DIR/.setup.pid"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    # The setup window keeps the launch screen up to date; the screen just opened follows it.
    [ -n "$PAGE" ] && exit 0
    fail "LanePulse is still being set up. The Terminal window shows the progress; LanePulse opens when it is done." setup
  fi
  if [ -n "${LANEPULSE_NO_BROWSER:-}" ]; then
    # Without a screen (tests): set up here, then continue.
    echo $$ > "$LOCK"
    "$RES/setup/setup.sh" "$HOME_DIR" "$BUILD" 2>&1 | tee "$LOGS/setup.log"
    rc="${PIPESTATUS[0]}"; rm -f "$LOCK"
    [ "$rc" = 0 ] || fail "LanePulse could not be set up. Details are in $LOGS/setup.log" setup
  else
    cat > "$HOME_DIR/Set up LanePulse.command" <<'CMD'
#!/bin/bash
# Sets up LanePulse (written by the app; it runs this the first time and after an update), then opens it.
H="$(cd "$(dirname "$0")" && pwd)"
APP="$(cat "$H/.app-path")"
echo $$ > "$H/.setup.pid"
printf '\033]0;Setting up LanePulse\007'
echo "Setting up LanePulse. This downloads about 2 GB and takes a few minutes; keep this window open."
VER="$(cat "$APP/Contents/Resources/version.txt")"
js() { printf '%s' "$1" | tr '\r\n\t' '   ' | perl -pe 's/\e\[[0-9;?]*[A-Za-z]//g' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' | cut -c1-400; }
st() { printf 'lpStatus({"state":"%s","text":"%s","detail":"%s","step":"setup","version":"%s"});\n' "$1" "$(js "$2")" "$(js "$3")" "$VER" \
  > "$H/.launch/status.js.tmp" && mv -f "$H/.launch/status.js.tmp" "$H/.launch/status.js"; }
# Shows the setup's current step and its latest line on the launch screen while it runs.
( while [ -f "$H/.setup.pid" ]; do
    LOG="$H/logs/setup.log"
    STEP="$(grep '^== ' "$LOG" 2>/dev/null | tail -1 | cut -c4-)"
    LAST="$(tail -c 600 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -1)"
    st setup "${STEP:-Checking the disk space}" "$LAST"
    sleep 1
  done ) &
if "$APP/Contents/Resources/setup/setup.sh" "$H" "$(cat "$APP/Contents/Resources/build.txt")" 2>&1 | tee "$H/logs/setup.log"; [ "${PIPESTATUS[0]}" = 0 ]; then
  rm -f "$H/.setup.pid"; wait
  echo "Opening LanePulse. You can close this window."
  LANEPULSE_PAGE_OPEN=1 "$APP/Contents/MacOS/LanePulse"
  sleep 2
else
  rm -f "$H/.setup.pid"; wait
  st failed "Setup did not finish. Open LanePulse again: it continues from where it stopped." \
    "$(grep 'Setup did not finish' "$H/logs/setup.log" | tail -1)"
  echo
  echo "Nothing was changed outside the LanePulse folder. Open LanePulse again (see the reason above):"
  echo "it continues from where it stopped. The details are in $H/logs/setup.log"
  read -r -p "Press Return to close."
fi
CMD
    chmod +x "$HOME_DIR/Set up LanePulse.command"
    lp_status setup "Preparing the setup" "" setup
    open -a Terminal "$HOME_DIR/Set up LanePulse.command" \
      || fail "LanePulse could not open Terminal to set itself up. Double-click \"Set up LanePulse.command\" in $HOME_DIR."
    exit 0
  fi
fi

# A LanePulse server of another build still answering (one started some other way, or whose process number was
# lost) would show the older screens: stop it, whatever started it. Only a LanePulse process is stopped.
running="$(curl -fsS -m 2 "$URL/api/health" 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null || true)"
if up && [ "$running" != "$VER" ]; then
  for pid in $(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null); do
    ps -o command= -p "$pid" 2>/dev/null | grep -q trafficwb && kill "$pid" 2>/dev/null
  done
  for _ in $(seq 30); do up || break; sleep 1; done
fi

lp_status start
if ! up; then
  if [ -f "$LOGS/server.log" ] && [ "$(stat -f%z "$LOGS/server.log" 2>/dev/null || echo 0)" -gt 10000000 ]; then
    mv -f "$LOGS/server.log" "$LOGS/server.old.log"
  fi
  echo "--- $(date) LanePulse $VER starting" >> "$LOGS/server.log"
  TWB_MODELS="$MODELS" nohup "$PY" -c 'import sys; from trafficwb.web.app import main; sys.exit(main(sys.argv[1:]))' \
    serve --data "$DATA" --port "$PORT" --backup-dir "$BACKUPS" >>"$LOGS/server.log" 2>&1 < /dev/null &
  echo $! > "$HOME_DIR/.server.pid"
  disown 2>/dev/null || true
  echo "$BUILD" > "$HOME_DIR/.server-version"
  for _ in $(seq 90); do up && break; sleep 1; done
  up || fail "LanePulse did not start. Details are in $LOGS/server.log"
fi

# Helpers in ~/LanePulse (rewritten each time, so they follow the app if it moves).
cat > "$HOME_DIR/Stop LanePulse.command" <<'CMD'
#!/bin/bash
# Stops the LanePulse server. Processing stops at once and continues from where it stopped next time.
H="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$H/.server.pid" ] && kill "$(cat "$H/.server.pid")" 2>/dev/null; then echo "LanePulse stopped."; else echo "LanePulse was not running."; fi
rm -f "$H/.server.pid"
sleep 2
CMD
cat > "$HOME_DIR/Import Model.command" <<'CMD'
#!/bin/bash
# Adds a trained model (.zip from LanePulse training) to LanePulse on this Mac. It does not change the model in
# use: evaluate it on the learning page and promote it there.
H="$(cd "$(dirname "$0")" && pwd)"
read -r -p "Drag the model .zip into this window, then press Return: " Z
Z="${Z%\'}"; Z="${Z#\'}"; Z="${Z%% }"; Z="${Z//\\ / }"
"$H/runtime/python/bin/python3" -c 'import sys; from trafficwb.model_import import main; sys.exit(main(sys.argv[1:]))' \
  --models "$H/models" import "$Z"
read -r -p "Press Return to close."
CMD
cat > "$HOME_DIR/Help another computer.command" <<'CMD'
#!/bin/bash
# Lets this Mac help another LanePulse server analyse long videos. Asks for the server's address and the pairing
# code (both on that server's Workers screen) unless they are given: --server http://... --code XXXX-XXXX-XXXX
H="$(cd "$(dirname "$0")" && pwd)"
printf '\033]0;LanePulse - helping another computer\007'
TWB_MODELS="$H/models" "$H/runtime/python/bin/python3" -m trafficwb.workers "$@"
read -r -p "Press Return to close."
CMD
chmod +x "$HOME_DIR/Stop LanePulse.command" "$HOME_DIR/Import Model.command" "$HOME_DIR/Help another computer.command"

lp_status ready
# The launch screen moves on to LanePulse by itself; without it, open LanePulse directly.
[ -n "${LANEPULSE_NO_BROWSER:-}" ] || [ -n "$PAGE" ] || open "$URL"
