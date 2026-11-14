#!/bin/bash
# Removes the LanePulse server program. Data, backups and models are kept.
#   lanepulse uninstall   (or: ~/lanepulse/uninstall-server.sh, or for the system service: sudo /opt/lanepulse/uninstall-server.sh)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$(id -u)" != 0 ] && [ -f "$HERE/lanepulse.conf" ]; then
  # The install made without root, in this folder.
  CONF="$HERE/lanepulse.conf"
  # shellcheck disable=SC1090
  . "$CONF"
  BASE="$HERE"
  # shellcheck source=user-service.sh
  . "$BASE/user-service.sh"
  user_service_remove "$BASE"
  [ "$(sed -n 's/^CONF=//p' "$HOME/.local/bin/lanepulse" 2>/dev/null | tr -d '"')" = "$CONF" ] && rm -f "$HOME/.local/bin/lanepulse"
  rm -rf "$BASE"/lanepulse-* "$BASE/current" "$BASE/previous" "$BASE/logs" "$BASE/stopped"
  rm -f "$BASE/run-server.sh" "$BASE/user-service.sh" "$BASE/system.sh" "$BASE/lanepulse.conf" "$BASE/uninstall-server.sh"
  echo "LanePulse removed. Kept: data in $DATA, backups in $BACKUPS, models in $MODELS. Installing again picks them up."
  exit 0
fi
[ "$(id -u)" = 0 ] || { echo "Run with sudo."; exit 1; }
CONF=/etc/lanepulse/lanepulse.conf
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"
systemctl disable --now lanepulse.service 2>/dev/null || true
rm -f /etc/systemd/system/lanepulse.service /etc/logrotate.d/lanepulse /usr/local/bin/lanepulse
[ "$(readlink /usr/bin/lanepulse 2>/dev/null)" = /usr/local/bin/lanepulse ] && rm -f /usr/bin/lanepulse
systemctl daemon-reload
if [ -f /opt/lanepulse/system.sh ]; then
  # shellcheck source=system.sh
  . /opt/lanepulse/system.sh
  [ -n "${PORT:-}" ] && fw_close "${ALLOW:-any}" "$PORT"
  selinux_unlabel /opt/lanepulse
fi
rm -rf /opt/lanepulse /etc/lanepulse
echo "LanePulse removed. Kept: data in ${DATA:-/srv/lanepulse/data}, backups in ${BACKUPS:-/srv/lanepulse/backups},"
echo "models in ${MODELS:-/srv/lanepulse/models}. Installing again picks them up."
