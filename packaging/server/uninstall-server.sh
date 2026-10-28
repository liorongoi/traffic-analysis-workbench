#!/bin/bash
# Removes the LanePulse server program. Data and backups are kept.   sudo /opt/lanepulse/uninstall-server.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Run with sudo."; exit 1; }
CONF=/etc/lanepulse/lanepulse.conf
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
