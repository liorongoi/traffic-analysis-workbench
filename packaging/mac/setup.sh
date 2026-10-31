#!/bin/bash
# Sets up what LanePulse.app runs on, in <LanePulse folder>/runtime: its own Python, the packages, the models and the
# training starting weights. Downloads only the exact files this build lists (python.txt, packages.lock and
# fetch_models.py), each checked against its SHA-256, and changes nothing outside the LanePulse folder.
#   setup.sh <LanePulse folder> <build>
# Run by the app the first time it is opened and after an update. Safe to run again after a failure: it continues.
set -euo pipefail
SETUP="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="$1"; BUILD="$2"
RT="$HOME_DIR/runtime"; NEW="$HOME_DIR/runtime.new"; MODELS="$HOME_DIR/models"
mkdir -p "$HOME_DIR" "$MODELS" "$HOME_DIR/logs"
step() { echo; echo "== $*"; }
die() { echo; echo "Setup did not finish: $*" >&2; exit 1; }

free_gb="$(df -Pk "$HOME_DIR" | awk 'NR==2 {print int($4 / 1048576)}')"
[ "$free_gb" -ge 6 ] || die "LanePulse needs about 6 GB of free disk space; this Mac has ${free_gb} GB free."
curl -fsS -m 20 -o /dev/null https://pypi.org/simple/ 2>/dev/null \
  || die "the internet could not be reached (pypi.org). Connect to the internet and open LanePulse again."

LOCK_ID="$(cat "$SETUP/python.txt" "$SETUP/packages.lock" | shasum -a 256 | cut -c1-64)"
rm -f "$RT/.build"
if [ -x "$RT/python/bin/python3" ] && [ "$(cat "$RT/.packages" 2>/dev/null)" = "$LOCK_ID" ]; then
  step "Updating LanePulse (the packages it uses are unchanged)"
else
  step "Downloading Python"
  rm -rf "$NEW"; mkdir -p "$NEW"
  PYURL="$(sed -n 1p "$SETUP/python.txt")"; PYSHA="$(sed -n 2p "$SETUP/python.txt")"
  curl -fL --retry 4 --progress-bar -o "$NEW/python.tar.gz" "$PYURL" || die "Python could not be downloaded."
  [ "$(shasum -a 256 "$NEW/python.tar.gz" | cut -c1-64)" = "$PYSHA" ] || die "the downloaded Python is not the expected file."
  tar -xzf "$NEW/python.tar.gz" -C "$NEW" && rm -f "$NEW/python.tar.gz"
  step "Downloading the packages (PyTorch, ONNX Runtime, OpenVINO, OpenCV and others)"
  "$NEW/python/bin/python3" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --require-hashes \
    --progress-bar on -r "$SETUP/packages.lock" || die "the packages could not be downloaded."
  echo "$LOCK_ID" > "$NEW/.packages"
  rm -rf "$RT.old"
  [ ! -e "$RT" ] || mv "$RT" "$RT.old"
  mv "$NEW" "$RT"
  rm -rf "$RT.old"
fi
PY="$RT/python/bin/python3"
"$PY" -m pip install --quiet --disable-pip-version-check --no-cache-dir --no-deps --force-reinstall "$SETUP"/trafficwb-*.whl \
  || die "LanePulse could not be installed."

step "Downloading the models (vehicles, plates, SAM 3.1: about 1 GB) and the training starting weights"
mkdir -p "$RT/python/share/lanepulse"
# Python's own downloads check certificates against the list pip carries.
SSL_CERT_FILE="$("$PY" -c 'import pip._vendor.certifi as c; print(c.where())')" \
  "$PY" "$SETUP/scripts/fetch_models.py" --dest "$MODELS" --training-weights "$RT/python/share/lanepulse" \
  || die "the models could not be downloaded."

step "Checking LanePulse runs on this Mac"
"$PY" "$SETUP/scripts/pc_check.py" --training || die "a part of LanePulse does not load on this Mac (see above)."
"$PY" -c "import trafficwb.web.app" || die "LanePulse does not load on this Mac (see above)."

echo "$BUILD" > "$RT/.build"
step "LanePulse is set up. From now on it runs without the internet."
