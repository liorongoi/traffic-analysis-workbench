#!/bin/bash
# Sets up LanePulse training on an Apple-silicon Mac (M1 to M5). No administrator rights needed.
# Run it from the training kit folder:   bash setup-training-mac.sh
# It needs the internet once (Python packages and the starting weights). Everything goes into
# ~/LanePulse-training; delete that folder to remove it.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
DIR="${LANEPULSE_TRAINING:-$HOME/LanePulse-training}"
mkdir -p "$DIR"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This is for an Apple-silicon Mac. On a computer with an NVIDIA GPU use setup-training-windows.ps1."
  exit 1
fi

# Python 3.11 or later: the one installed, or one fetched by uv into your home folder.
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Getting Python 3.12 (into your home folder, no administrator rights)..."
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install 3.12
  PY="$(uv python find 3.12)"
fi
echo "Python: $PY"

"$PY" -m venv "$DIR/venv"
"$DIR/venv/bin/python" -m pip install --quiet --upgrade pip
WHL="$(ls "$KIT"/trafficwb-*.whl | sort | tail -1)"
echo "Installing LanePulse training and PyTorch (a few minutes)..."
"$DIR/venv/bin/python" -m pip install --quiet --upgrade "${WHL}[train]"

cp "$KIT/TRAINING.txt" "$DIR/TRAINING.txt" 2>/dev/null || true
cat > "$DIR/Train LanePulse.command" <<'CMD'
#!/bin/bash
# Double-click to train the LanePulse vehicle detector on an exported dataset.
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "LanePulse training"
echo
read -r -p "Drag the exported dataset folder into this window, then press Return: " DS
DS="${DS%\'}"; DS="${DS#\'}"; DS="${DS%% }"; DS="${DS//\\ / }"
read -r -p "Rounds of training (Return for 40): " EP
"$DIR/venv/bin/trafficwb-train" run --dataset "$DS" --out "$DIR/models" --epochs "${EP:-40}"
ZIP="$(ls -t "$DIR"/models/*.zip 2>/dev/null | head -1)"
if [ -n "$ZIP" ]; then
  echo
  echo "Done. Copy this file to the LanePulse computer and drag it onto import-model.cmd there:"
  echo "  $ZIP"
  open -R "$ZIP"
fi
read -r -p "Press Return to close."
CMD
chmod +x "$DIR/Train LanePulse.command"

echo
"$DIR/venv/bin/trafficwb-train" device
echo
echo "Ready. To train, double-click \"Train LanePulse.command\" in $DIR"
open "$DIR" 2>/dev/null || true
