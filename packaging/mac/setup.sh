#!/bin/bash
# Sets up what LanePulse.app runs on, in <LanePulse folder>/runtime: its own Python, the packages, the models and the
# training starting weights. The normal build carries all of them and needs no internet; a build without them
# downloads only the exact files it lists (python.txt, packages.lock and fetch_models.py), each checked against its
# SHA-256. Changes nothing outside the LanePulse folder.
#   setup.sh <LanePulse folder> <build>
# Run by the app the first time it is opened and after an update. Safe to run again after a failure: it continues.
set -euo pipefail
SETUP="$(cd "$(dirname "$0")" && pwd)"
HOME_DIR="$1"; BUILD="$2"
RT="$HOME_DIR/runtime"; NEW="$HOME_DIR/runtime.new"; MODELS="$HOME_DIR/models"
mkdir -p "$HOME_DIR" "$MODELS" "$HOME_DIR/logs"
step() { echo; echo "== $*"; }
die() { echo; echo "Setup did not finish: $*" >&2; exit 1; }

# Offline when the app carries the files (the normal build); otherwise they are downloaded.
OFFLINE=""
[ -f "$SETUP/python.tar.gz" ] && [ -d "$SETUP/wheels" ] && [ -d "$SETUP/models" ] && OFFLINE=1
need_gb=6
[ -z "$OFFLINE" ] || need_gb=$(( 6 + $(du -sk "$SETUP/models" | cut -f1) / 1048576 ))
free_gb="$(df -Pk "$HOME_DIR" | awk 'NR==2 {print int($4 / 1048576)}')"
[ "$free_gb" -ge "$need_gb" ] || die "LanePulse needs about $need_gb GB of free disk space; this Mac has ${free_gb} GB free."
if [ -z "$OFFLINE" ]; then
  curl -fsS -m 20 -o /dev/null https://pypi.org/simple/ 2>/dev/null \
    || die "the internet could not be reached (pypi.org). Connect to the internet and open LanePulse again."
fi

LOCK_ID="$(cat "$SETUP/python.txt" "$SETUP/packages.lock" | shasum -a 256 | cut -c1-64)"
rm -f "$RT/.build"
if [ -x "$RT/python/bin/python3" ] && [ "$(cat "$RT/.packages" 2>/dev/null)" = "$LOCK_ID" ]; then
  step "Updating LanePulse (the packages it uses are unchanged)"
else
  rm -rf "$NEW"; mkdir -p "$NEW"
  PYURL="$(sed -n 1p "$SETUP/python.txt")"; PYSHA="$(sed -n 2p "$SETUP/python.txt")"
  if [ -n "$OFFLINE" ]; then
    step "Setting up Python (carried in the app)"
    cp "$SETUP/python.tar.gz" "$NEW/python.tar.gz"
  else
    step "Downloading Python"
    curl -fL --retry 4 --progress-bar -o "$NEW/python.tar.gz" "$PYURL" || die "Python could not be downloaded."
  fi
  [ "$(shasum -a 256 "$NEW/python.tar.gz" | cut -c1-64)" = "$PYSHA" ] || die "the Python file is not the expected one."
  tar -xzf "$NEW/python.tar.gz" -C "$NEW" && rm -f "$NEW/python.tar.gz"
  if [ -n "$OFFLINE" ]; then
    step "Installing the packages (PyTorch, ONNX Runtime, OpenVINO, OpenCV and others, carried in the app)"
    # The wheels were downloaded at build time against the SHA-256 in packages.lock.
    "$NEW/python/bin/python3" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --no-index \
      --progress-bar on "$SETUP"/wheels/*.whl || die "the packages could not be installed."
  else
    step "Downloading the packages (PyTorch, ONNX Runtime, OpenVINO, OpenCV and others)"
    "$NEW/python/bin/python3" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --require-hashes \
      --progress-bar on -r "$SETUP/packages.lock" || die "the packages could not be downloaded."
  fi
  echo "$LOCK_ID" > "$NEW/.packages"
  rm -rf "$RT.old"
  [ ! -e "$RT" ] || mv "$RT" "$RT.old"
  mv "$NEW" "$RT"
  rm -rf "$RT.old"
fi
PY="$RT/python/bin/python3"
"$PY" -m pip install --quiet --disable-pip-version-check --no-cache-dir --no-deps --force-reinstall "$SETUP"/trafficwb-*.whl \
  || die "LanePulse could not be installed."

mkdir -p "$RT/python/share/lanepulse"
if [ -n "$OFFLINE" ]; then
  step "Installing the models (vehicles in every size, plates, SAM 3.1) and the training starting weights"
  # Model packs already here (trained or imported ones, or the same version from before) are kept as they are.
  rsync -a --ignore-existing "$SETUP/models/" "$MODELS/" || die "the models could not be copied."
  if [ -d "$SETUP/training_weights" ]; then
    cp -f "$SETUP/training_weights/"* "$RT/python/share/lanepulse/" || die "the training starting weights could not be copied."
  fi
else
  step "Downloading the models (vehicles, plates, SAM 3.1: about 1 GB) and the training starting weights"
  # Python's own downloads check certificates against the list pip carries.
  SSL_CERT_FILE="$("$PY" -c 'import pip._vendor.certifi as c; print(c.where())')" \
    "$PY" "$SETUP/scripts/fetch_models.py" --dest "$MODELS" --training-weights "$RT/python/share/lanepulse" \
    || die "the models could not be downloaded."
fi

step "Checking LanePulse runs on this Mac"
"$PY" "$SETUP/scripts/pc_check.py" --training || die "a part of LanePulse does not load on this Mac (see above)."
"$PY" -c "import trafficwb.web.app" || die "LanePulse does not load on this Mac (see above)."

echo "$BUILD" > "$RT/.build"
step "LanePulse is set up. From now on it runs without the internet."
