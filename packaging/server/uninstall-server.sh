#!/bin/bash
# Removes the LanePulse server program. Data and backups are kept.   sudo /opt/lanepulse/uninstall-server.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Run with sudo."; exit 1; }
CONF=/etc/lanepulse/lanepulse.conf
[ -f "$CONF" ] && . "$CONF"
systemctl disable --now lanepulse.service 2>/dev/null || true
rm -f /etc/systemd/system/lanepulse.service /etc/logrotate.d/lanepulse /usr/local/bin/lanepulse
systemctl daemon-reload
if command -v ufw >/dev/null 2>&1 && [ -n "${PORT:-}" ]; then
  if [ "${ALLOW:-any}" = any ]; then ufw delete allow "$PORT/tcp" >/dev/null 2>&1 || true
  else ufw delete allow from "$ALLOW" to any port "$PORT" proto tcp >/dev/null 2>&1 || true; fi
fi
rm -rf /opt/lanepulse /etc/lanepulse
echo "LanePulse removed. Kept: data in ${DATA:-/srv/lanepulse/data}, backups in ${BACKUPS:-/srv/lanepulse/backups},"
echo "models in ${MODELS:-/srv/lanepulse/models}. Installing again picks them up."
