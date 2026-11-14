# How an install made without root starts by itself: sourced by install-server.sh, lanepulse and uninstall-server.sh.
# Prefers systemd's per-user services (when this user may run them while logged out: "linger"), else cron, else
# nothing (started by hand). Sets SERVICE to systemd-user, cron or none.
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CRON_MARK="# LanePulse (installed by install-server.sh)"

_linger_on() { [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" = yes ]; }
_systemd_user_ok() {
  command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 || return 1
  _linger_on && return 0
  # Most systems let a user turn this on for themselves; some need the administrator.
  loginctl enable-linger "$(id -un)" >/dev/null 2>&1 && _linger_on
}
_cron_remove() {
  command -v crontab >/dev/null 2>&1 || return 0
  local cur; cur="$(crontab -l 2>/dev/null || true)"
  if printf '%s\n' "$cur" | grep -qF "$CRON_MARK"; then
    printf '%s\n' "$cur" | grep -vF "$CRON_MARK" | crontab -
  fi
}

user_service_stop() {   # stops it, whichever way it runs
  local base="$1"
  touch "$base/stopped"
  if [ -f "$UNIT_DIR/lanepulse.service" ]; then systemctl --user stop lanepulse.service 2>/dev/null || true; fi
  if [ -f "$base/logs/server.pid" ]; then
    local pid; pid="$(cat "$base/logs/server.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$base/logs/server.pid"
  fi
}

user_service_start() {
  local base="$1"
  rm -f "$base/stopped"
  case "${SERVICE:-none}" in
    systemd-user) systemctl --user restart lanepulse.service;;
    *) "$base/run-server.sh" --background;;
  esac
}

user_service_install() {
  local base="$1"
  user_service_stop "$base"
  rm -f "$base/stopped"
  if _systemd_user_ok; then
    _cron_remove
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/lanepulse.service" <<UNIT
[Unit]
Description=LanePulse traffic analysis server
After=network-online.target

[Service]
ExecStart=$base/run-server.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable lanepulse.service >/dev/null 2>&1
    systemctl --user restart lanepulse.service
    SERVICE=systemd-user
  else
    if [ -f "$UNIT_DIR/lanepulse.service" ]; then
      systemctl --user disable lanepulse.service >/dev/null 2>&1 || true
      rm -f "$UNIT_DIR/lanepulse.service"
    fi
    SERVICE=none
    if command -v crontab >/dev/null 2>&1; then
      _cron_remove || true
      if { crontab -l 2>/dev/null || true
           echo "@reboot $base/run-server.sh --ensure $CRON_MARK"
           echo "* * * * * $base/run-server.sh --ensure $CRON_MARK"; } | crontab - 2>/dev/null; then
        SERVICE=cron
      fi
    fi
    "$base/run-server.sh" --background
  fi
}

user_service_remove() {
  local base="$1"
  user_service_stop "$base"
  if [ -f "$UNIT_DIR/lanepulse.service" ]; then
    systemctl --user disable lanepulse.service >/dev/null 2>&1 || true
    rm -f "$UNIT_DIR/lanepulse.service"
    systemctl --user daemon-reload 2>/dev/null || true
  fi
  _cron_remove
}
