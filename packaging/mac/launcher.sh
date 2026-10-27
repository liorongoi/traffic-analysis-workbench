#!/bin/bash
# LanePulse.app: starts the LanePulse server if it is not running, then opens it in the browser.
# Everything LanePulse keeps (data, backups, models, logs) is in ~/LanePulse; the app itself changes nothing
# inside its own folder. Nothing is downloaded.
APP="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$APP/Contents/Resources"
PY="$RES/python/bin/python3"
VER="$(cat "$RES/version.txt")"
BUILD="$(cat "$RES/build.txt" 2>/dev/null || echo "$VER")"
HOME_DIR="${LANEPULSE_HOME:-$HOME/LanePulse}"
DATA="$HOME_DIR/data"; MODELS="$HOME_DIR/models"; BACKUPS="$HOME_DIR/backups"; LOGS="$HOME_DIR/logs"
PORT="${LANEPULSE_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$DATA" "$MODELS" "$BACKUPS" "$LOGS"

fail() {
  osascript -e "display alert \"LanePulse\" message \"$1\" as critical" >/dev/null 2>&1 || echo "$1" >&2
  exit 1
}
up() { curl -fsS -m 2 "$URL/api/health" >/dev/null 2>&1; }

# Model packs shipped with this version, next to any trained ones you imported. Never overwritten.
for card in "$RES"/models/*/*/model.json; do
  [ -e "$card" ] || continue
  d="$(dirname "$card")"; rel="${d#"$RES/models/"}"
  if [ ! -e "$MODELS/$rel" ]; then mkdir -p "$MODELS/$(dirname "$rel")" && cp -R "$d" "$MODELS/$rel"; fi
done
echo "$APP" > "$HOME_DIR/.app-path"

# After an update, stop the older version this app started, so the new one runs.
if up && [ -f "$HOME_DIR/.server.pid" ] && [ "$(cat "$HOME_DIR/.server-version" 2>/dev/null)" != "$BUILD" ]; then
  kill "$(cat "$HOME_DIR/.server.pid")" 2>/dev/null || true
  for _ in $(seq 30); do up || break; sleep 1; done
fi

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
APP="$(cat "$H/.app-path")"
read -r -p "Drag the model .zip into this window, then press Return: " Z
Z="${Z%\'}"; Z="${Z#\'}"; Z="${Z%% }"; Z="${Z//\\ / }"
"$APP/Contents/Resources/python/bin/python3" -c 'import sys; from trafficwb.model_import import main; sys.exit(main(sys.argv[1:]))' \
  --models "$H/models" import "$Z"
read -r -p "Press Return to close."
CMD
chmod +x "$HOME_DIR/Stop LanePulse.command" "$HOME_DIR/Import Model.command"

[ -n "${LANEPULSE_NO_BROWSER:-}" ] || open "$URL"
