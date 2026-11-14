#!/bin/bash
# Installs or updates the LanePulse server on Ubuntu (22.04, 24.04) or Red Hat Enterprise Linux 8 or 9 (and Rocky or
# AlmaLinux), x86_64. Run from the unpacked package, as an ordinary user (no root, no sudo):
#   ./install-server.sh [--data DIR] [--backups DIR] [--models DIR] [--port N] [--allow-from CIDR|any]
# It installs into ~/lanepulse (program, data, backups, models) and starts by itself, also after a reboot.
# Run with sudo instead to install it as a system service (/opt/lanepulse, data in /srv/lanepulse, its own
# account, the firewall opened). Nothing is downloaded. The web screens are then open to other computers at
# http://<this server>:<port>, with the password the installer prints (change it with lanepulse password).
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
if [ "$(id -u)" = 0 ]; then
  MODE=system
  BASE=/opt/lanepulse
  CONF=/etc/lanepulse/lanepulse.conf
  DATA=/srv/lanepulse/data
  BACKUPS=/srv/lanepulse/backups
  MODELS=/srv/lanepulse/models
  SUDO="sudo "
else
  MODE=user
  BASE="${LANEPULSE_HOME:-$HOME/lanepulse}"
  CONF="$BASE/lanepulse.conf"
  DATA="$BASE/data"
  BACKUPS="$BASE/backups"
  MODELS="$BASE/models"
  SUDO=""
fi
PORT=8765
ALLOW=""
SKIP_CHECK=""
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
    -h|--help) sed -n 2,8p "$0"; exit 0;;
    *) echo "Unknown option $1"; exit 2;;
  esac
done
say() { printf '\n== %s\n' "$*"; }
# shellcheck source=system.sh
. "$PKG/system.sh"
# As the service account (system install) or as this user (own install).
as_svc() { if [ "$MODE" = system ]; then runuser -u lanepulse -- "$@"; else "$@"; fi; }
[ "$(uname -m)" = x86_64 ] || { echo "This package is for x86_64 servers."; exit 1; }
VER="$(sed -n 's/.*"version": "\([^"]*\)".*/\1/p' "$PKG/server.json")"
MIN_DRIVER="$(sed -n 's/.*"min_driver": "\([^"]*\)".*/\1/p' "$PKG/server.json")"
echo "LanePulse server $VER ($([ "$MODE" = system ] && echo "system service" || echo "for user $(id -un), in $BASE, no root needed"))"
if [ -r /etc/os-release ]; then
  OS_PRETTY="$(. /etc/os-release; echo "${PRETTY_NAME:-}")"; OS_ID="$(. /etc/os-release; echo "${ID:-} ${ID_LIKE:-}")"
  echo "System: $OS_PRETTY"
  case " $OS_ID " in
    *" ubuntu "*|*" rhel "*|*" fedora "*|*" centos "*) ;;
    *) echo "Note: tested on Ubuntu and Red Hat Enterprise Linux 9; continuing.";;
  esac
fi
[ "$MODE" = system ] && selinux_on && echo "SELinux: $(getenforce 2>/dev/null || echo on); LanePulse's files are labelled for it."

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

if [ "$GPU" = cuda ] && [ "$MODE" = user ] && ! [ -r /dev/nvidiactl ] && ! [ -w /dev/nvidiactl ]; then
  echo "This account cannot open the GPU (/dev/nvidiactl). LanePulse will run on the CPU; ask the administrator to"
  echo "add $(id -un) to the group that owns /dev/nvidia* (often 'video'), then run this installer again."
fi

say "3. Folders"
mkdir -p "$BASE" "$MODELS" "$BASE/logs" "$DATA" "$BACKUPS" "$(dirname "$CONF")"
if [ "$MODE" = system ]; then
  id lanepulse >/dev/null 2>&1 || useradd --system --home-dir "$BASE" --shell /usr/sbin/nologin lanepulse
  # video/render: access to the GPU device files on some systems.
  for g in video render; do getent group "$g" >/dev/null && usermod -aG "$g" lanepulse || true; done
  chown lanepulse:lanepulse "$DATA" "$BACKUPS" "$BASE/logs"
  chown -R lanepulse:lanepulse "$MODELS"
fi
# Room for the program (Python, PyTorch, CUDA libraries) next to the previous version, and the models.
NEED_KB=$(( $(du -sk "$PKG" | cut -f1) + 1048576 ))
FREE_KB="$(df -Pk "$BASE" | awk 'NR==2 {print $4}')"
if [ "$FREE_KB" -lt "$NEED_KB" ]; then
  echo "Not enough free space for $BASE: $((FREE_KB / 1048576)) GB free, about $((NEED_KB / 1048576)) GB needed."
  exit 1
fi
echo "Program: $BASE   Data: $DATA   Backups: $BACKUPS   Models: $MODELS"

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
[ "$MODE" = system ] && chown -R lanepulse:lanepulse "$MODELS"
PY="$DEST/python/bin/python3"
echo "Installed into $DEST"

if [ -z "$SKIP_CHECK" ]; then
  say "5. Install check (a small mission end to end, offline)"
  W="$(mktemp -d)"; [ "$MODE" = system ] && chown lanepulse "$W"
  if ! as_svc env TWB_MODELS="$MODELS" HOME="$W" "$PY" "$DEST/offline_smoke.py" --work "$W/check" --models "$MODELS"; then
    rm -rf "$W" "$DEST"; echo "The install check failed; nothing was switched."; exit 1
  fi
  rm -rf "$W"
fi

say "6. Backup before switching"
if [ -L "$BASE/current" ] && [ -f "$DATA/control.db" ]; then
  as_svc "$BASE/current/python/bin/python3" -c 'import sys; from trafficwb.admin import main; sys.exit(main(sys.argv[1:]))' \
    backup --data "$DATA" --to "$BACKUPS" || { echo "Backup before the update failed; nothing was switched."; exit 1; }
else
  echo "Nothing to back up yet."
fi
[ -L "$BASE/current" ] && ln -sfn "$(readlink "$BASE/current")" "$BASE/previous"
ln -sfn "$DEST" "$BASE/current"
cat > "$CONF" <<CFG
# LanePulse server settings, read by the service and by the next install. Change, then: ${SUDO}lanepulse restart
MODE="$MODE"
BASE="$BASE"
DATA="$DATA"
BACKUPS="$BACKUPS"
MODELS="$MODELS"
PORT="$PORT"
ALLOW="$ALLOW"
# auto, cuda or cpu
TWB_ACCELERATOR="${TWB_ACCELERATOR:-auto}"
# How it starts by itself (own install): systemd-user, cron or none
SERVICE="${SERVICE:-}"
CFG
install -m 0755 "$PKG/uninstall-server.sh" "$BASE/uninstall-server.sh"
install -m 0644 "$PKG/system.sh" "$BASE/system.sh"
install -m 0755 "$PKG/run-server.sh" "$BASE/run-server.sh"
install -m 0644 "$PKG/user-service.sh" "$BASE/user-service.sh"
if [ "$MODE" = system ]; then
  install -m 0755 "$PKG/lanepulse" /usr/local/bin/lanepulse
  # Red Hat's sudo does not look in /usr/local/bin, so "sudo lanepulse" needs this link.
  [ -e /usr/bin/lanepulse ] || ln -s /usr/local/bin/lanepulse /usr/bin/lanepulse
  selinux_label "$BASE"
else
  # The lanepulse command, in ~/.local/bin (on the PATH of new logins; added to it when missing).
  mkdir -p "$HOME/.local/bin"
  sed "s|^CONF=.*|CONF=\"$CONF\"|" "$PKG/lanepulse" > "$HOME/.local/bin/lanepulse"
  chmod 0755 "$HOME/.local/bin/lanepulse"
  for rc in "$HOME/.bashrc" "$HOME/.profile"; do
    if ! grep -qs 'LanePulse: the lanepulse command' "$rc"; then
      printf '\n# LanePulse: the lanepulse command\ncase ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH";; esac\n' >> "$rc"
    fi
  done
fi

say "7. Start, now and whenever the server boots"
if [ "$MODE" = user ]; then
  # shellcheck source=user-service.sh
  . "$PKG/user-service.sh"
  user_service_install "$BASE" "$CONF"
  sed -i "s|^SERVICE=.*|SERVICE=\"$SERVICE\"|" "$CONF"
  for _ in $(seq 90); do curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null || { tail -30 "$BASE/logs/server.log"; echo "The server did not start."; exit 1; }
  case "$SERVICE" in
    systemd-user) echo "Running; starts by itself when the server boots (systemd, for user $(id -un)).";;
    cron) echo "Running; starts by itself when the server boots, and again within a minute if it stops (cron).";;
    *) echo "Running, but it will not start by itself after a reboot on this server: run $BASE/run-server.sh --background then.";;
  esac
else
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
fi

say "8. Firewall"
if [ -z "$ALLOW" ]; then
  # The server's own local network, e.g. 192.168.1.0/24.
  ALLOW="$( (ip -o -f inet addr show scope global 2>/dev/null || true) | awk '{print $4}' | head -1)"
  ALLOW="$("$PY" -c 'import ipaddress,sys; print(ipaddress.ip_network(sys.argv[1], strict=False))' "$ALLOW" 2>/dev/null || echo any)"
  sed -i "s|^ALLOW=.*|ALLOW=\"$ALLOW\"|" "$CONF"
fi
if [ "$MODE" = user ]; then
  echo "Other computers reach it on port $PORT. If they cannot (a firewall on this server), the server's administrator"
  echo "allows it once with:  sudo firewall-cmd --permanent --add-port=$PORT/tcp && sudo firewall-cmd --reload   (Red Hat)"
  echo "                  or:  sudo ufw allow from $ALLOW to any port $PORT proto tcp                         (Ubuntu)"
elif fw_active; then
  fw_open "$ALLOW" "$PORT"
  echo "Firewall ($(fw_name)): port $PORT open to $ALLOW."
else
  echo "The firewall ($(fw_name)) is off, so any computer that can reach this server can open LanePulse."
  echo "To allow only $ALLOW:  sudo lanepulse firewall   (keeps SSH open)"
fi

IP="$( (hostname -I 2>/dev/null || true) | awk '{print $1}')"; IP="${IP:-$(hostname)}"
say "Done"
KEY="$(as_svc "$BASE/current/python/bin/python3" -c 'import sys; from trafficwb.web.app import main; sys.exit(main(sys.argv[1:]))' password --data "$DATA" --show)"
echo "LanePulse $VER is running ($GPU). Open it from another computer:  http://$IP:$PORT"
echo "Password (the browser asks for it once):  $KEY"
echo "  Choose your own in LanePulse (the key icon at the top), or here: ${SUDO}lanepulse password"
echo "  Forgotten later? ${SUDO}lanepulse password sets a new one."
echo "Manage it with:  lanepulse status | logs | restart | stop | start | import-model <zip> | backup"
