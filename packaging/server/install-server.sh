#!/bin/bash
# Installs or updates the LanePulse server on Ubuntu (22.04, 24.04) or Red Hat Enterprise Linux 8 or 9 (and Rocky or
# AlmaLinux), x86_64. Run from the unpacked package:
#   sudo ./install-server.sh [--data DIR] [--backups DIR] [--models DIR] [--port N] [--allow-from CIDR|any]
# Nothing is downloaded. The web screens are then open to other computers at http://<this server>:<port>,
# with no password: keep the server on a trusted network (see INSTALL.txt).
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
DATA=/srv/lanepulse/data
BACKUPS=/srv/lanepulse/backups
MODELS=/srv/lanepulse/models
PORT=8765
ALLOW=""
SKIP_CHECK=""
BASE=/opt/lanepulse
CONF=/etc/lanepulse/lanepulse.conf
# Keep the settings of an earlier install unless given again below.
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --backups) BACKUPS="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --allow-from) ALLOW="$2"; shift 2;;
    --skip-check) SKIP_CHECK=1; shift;;
    -h|--help) sed -n 2,6p "$0"; exit 0;;
    *) echo "Unknown option $1"; exit 2;;
  esac
done
say() { printf '\n== %s\n' "$*"; }
# shellcheck source=system.sh
. "$PKG/system.sh"
[ "$(id -u)" = 0 ] || { echo "Run with sudo: sudo $0 $*"; exit 1; }
[ "$(uname -m)" = x86_64 ] || { echo "This package is for x86_64 servers."; exit 1; }
VER="$(sed -n 's/.*"version": "\([^"]*\)".*/\1/p' "$PKG/server.json")"
MIN_DRIVER="$(sed -n 's/.*"min_driver": "\([^"]*\)".*/\1/p' "$PKG/server.json")"
echo "LanePulse server $VER"
if [ -r /etc/os-release ]; then
  OS_PRETTY="$(. /etc/os-release; echo "${PRETTY_NAME:-}")"; OS_ID="$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")"
  echo "System: $OS_PRETTY"
  case " $OS_ID " in
    *" ubuntu "*|*" rhel "*|*" fedora "*|*" centos "*) ;;
    *) echo "Note: tested on Ubuntu and Red Hat Enterprise Linux 9; continuing.";;
  esac
fi
selinux_on && echo "SELinux: $(getenforce 2>/dev/null || echo on); LanePulse's files are labelled for it."

say "1. Checking the package"
(cd "$PKG" && sha256sum --quiet -c SHA256SUMS) || { echo "Some files are damaged or missing; copy the package again."; exit 1; }
echo "All files verified."

say "2. NVIDIA GPU"
GPU="cpu"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
  if [ -n "$MIN_DRIVER" ] && [ "$(printf '%s\n%s\n' "$MIN_DRIVER" "$DRV" | sort -V | head -1)" != "$MIN_DRIVER" ]; then
    echo "The NVIDIA driver $DRV is older than $MIN_DRIVER, which this package's CUDA libraries need."
    echo "LanePulse will run on the CPU until the driver is updated (see INSTALL.txt, NVIDIA driver)."
  else
    GPU="cuda"; echo "Driver $DRV: the GPU will be used."
  fi
else
  echo "No working NVIDIA driver found (nvidia-smi). LanePulse will run on the CPU."
  echo "Install the driver (INSTALL.txt, NVIDIA driver), reboot, then run this installer again."
fi

say "3. Service account and folders"
id lanepulse >/dev/null 2>&1 || useradd --system --home-dir "$BASE" --shell /usr/sbin/nologin lanepulse
# video/render: access to the GPU device files on some systems.
for g in video render; do getent group "$g" >/dev/null && usermod -aG "$g" lanepulse || true; done
mkdir -p "$BASE" "$MODELS" "$BASE/logs" "$DATA" "$BACKUPS" /etc/lanepulse
chown lanepulse:lanepulse "$DATA" "$BACKUPS" "$BASE/logs"
chown -R lanepulse:lanepulse "$MODELS"

say "4. Program files"
DEST="$BASE/lanepulse-$VER"
if [ -e "$DEST" ]; then DEST="$BASE/lanepulse-$VER-repair-$(date +%Y%m%d-%H%M%S)"; echo "This version is installed already; repairing into $DEST."; fi
mkdir -p "$DEST"
cp -a "$PKG/runtime/python" "$DEST/python"
cp -a "$PKG/server.json" "$PKG/INSTALL.txt" "$PKG/TRAINING.txt" "$PKG/offline_smoke.py" "$DEST/"
chmod -R a+rX "$DEST"
for card in "$PKG"/models/*/*/model.json; do
  d="$(dirname "$card")"; rel="${d#"$PKG/models/"}"
  if [ ! -e "$MODELS/$rel" ]; then mkdir -p "$MODELS/$(dirname "$rel")"; cp -a "$d" "$MODELS/$rel"; fi
done
chown -R lanepulse:lanepulse "$MODELS"
PY="$DEST/python/bin/python3"
echo "Installed into $DEST"

if [ -z "$SKIP_CHECK" ]; then
  say "5. Install check (a small mission end to end, offline)"
  W="$(mktemp -d)"; chown lanepulse "$W"
  if ! runuser -u lanepulse -- env TWB_MODELS="$MODELS" HOME="$W" "$PY" "$DEST/offline_smoke.py" --work "$W/check" --models "$MODELS"; then
    rm -rf "$W" "$DEST"; echo "The install check failed; nothing was switched."; exit 1
  fi
  rm -rf "$W"
fi

say "6. Backup before switching"
if [ -L "$BASE/current" ] && [ -f "$DATA/control.db" ]; then
  runuser -u lanepulse -- "$BASE/current/python/bin/python3" -c 'import sys; from trafficwb.admin import main; sys.exit(main(sys.argv[1:]))' \
    backup --data "$DATA" --to "$BACKUPS" || { echo "Backup before the update failed; nothing was switched."; exit 1; }
else
  echo "Nothing to back up yet."
fi
[ -L "$BASE/current" ] && ln -sfn "$(readlink "$BASE/current")" "$BASE/previous"
ln -sfn "$DEST" "$BASE/current"
cat > "$CONF" <<CFG
# LanePulse server settings, read by the service and by the next install. Change, then: sudo lanepulse restart
DATA="$DATA"
BACKUPS="$BACKUPS"
MODELS="$MODELS"
PORT="$PORT"
ALLOW="$ALLOW"
# auto, cuda or cpu
TWB_ACCELERATOR="${TWB_ACCELERATOR:-auto}"
CFG
install -m 0755 "$PKG/lanepulse" /usr/local/bin/lanepulse
# Red Hat's sudo does not look in /usr/local/bin, so "sudo lanepulse" needs this link.
[ -e /usr/bin/lanepulse ] || ln -s /usr/local/bin/lanepulse /usr/bin/lanepulse
install -m 0755 "$PKG/uninstall-server.sh" "$BASE/uninstall-server.sh"
install -m 0644 "$PKG/system.sh" "$BASE/system.sh"
selinux_label "$BASE"

say "7. Service"
cat > /etc/systemd/system/lanepulse.service <<UNIT
[Unit]
Description=LanePulse traffic analysis server
After=network-online.target remote-fs.target
Wants=network-online.target

[Service]
User=lanepulse
Group=lanepulse
EnvironmentFile=$CONF
Environment=TWB_MODELS=$MODELS
Environment=HOME=$BASE
ExecStart=$BASE/current/python/bin/python3 -c "import sys; from trafficwb.web.app import main; sys.exit(main(sys.argv[1:]))" serve --lan --data \${DATA} --port \${PORT} --backup-dir \${BACKUPS}
Restart=on-failure
RestartSec=10
StandardOutput=append:$BASE/logs/server.log
StandardError=append:$BASE/logs/server.log

[Install]
WantedBy=multi-user.target
UNIT
cat > /etc/logrotate.d/lanepulse <<ROT
$BASE/logs/*.log {
  size 10M
  rotate 3
  copytruncate
  missingok
  notifempty
}
ROT
systemctl daemon-reload
systemctl enable lanepulse.service >/dev/null 2>&1
systemctl restart lanepulse.service
for _ in $(seq 90); do curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null || { tail -30 "$BASE/logs/server.log"; echo "The server did not start."; exit 1; }
echo "Running; starts by itself when the server boots."

say "8. Firewall"
if [ -z "$ALLOW" ]; then
  # The server's own local network, e.g. 192.168.1.0/24.
  ALLOW="$(ip -o -f inet addr show scope global | awk '{print $4}' | head -1)"
  ALLOW="$("$PY" -c 'import ipaddress,sys; print(ipaddress.ip_network(sys.argv[1], strict=False))' "$ALLOW" 2>/dev/null || echo any)"
  sed -i "s|^ALLOW=.*|ALLOW=\"$ALLOW\"|" "$CONF"
fi
if fw_active; then
  fw_open "$ALLOW" "$PORT"
  echo "Firewall ($(fw_name)): port $PORT open to $ALLOW."
else
  echo "The firewall ($(fw_name)) is off, so any computer that can reach this server can open LanePulse."
  echo "To allow only $ALLOW:  sudo lanepulse firewall   (keeps SSH open)"
fi

IP="$(hostname -I | awk '{print $1}')"
say "Done"
echo "LanePulse $VER is running ($GPU). Open it from another computer:  http://$IP:$PORT"
echo "Manage it with:  lanepulse status | logs | restart | stop | start | import-model <zip> | backup"
