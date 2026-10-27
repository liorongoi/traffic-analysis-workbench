#!/bin/bash
# Builds the LanePulse server package for Ubuntu and Red Hat (x86_64) with NVIDIA GPU support, run in CI on Linux.
#   packaging/server/build_server.sh <python-build-standalone x86_64 linux .tar.gz> <out folder>
# Before: the web screens built (scripts/build_ui.py) and the model packs fetched (scripts/fetch_models.py).
# The package carries its own Python and every library, including NVIDIA's CUDA and cuDNN libraries for ONNX
# Runtime, so the server needs only the NVIDIA driver and downloads nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTGZ="$1"; OUT="$2"
VER="$(python3 -c 'import re,sys; print(re.search(r"__version__ = \"(.+?)\"", open(sys.argv[1]).read()).group(1))' "$ROOT/src/trafficwb/__init__.py")"
NAME="lanepulse-server-$VER-linux-x86_64"
PKG="$OUT/$NAME"
rm -rf "$PKG"; mkdir -p "$PKG/runtime"
tar -xzf "$PYTGZ" -C "$PKG/runtime"
PY="$PKG/runtime/python/bin/python3"
# PyTorch for training on the learning page, built for CUDA 12.8 (the same CUDA 12 family as ONNX Runtime's
# libraries), then the app. Training uses the server's GPU; nothing is fetched on the server.
"$PY" -m pip install --quiet --no-cache-dir --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install --quiet --no-cache-dir --disable-pip-version-check "$ROOT[probe,cuda,train]"
"$PY" -c 'import torch, onnxruntime; assert torch.version.cuda, "PyTorch without CUDA"; print("torch", torch.__version__, "CUDA", torch.version.cuda, "onnxruntime", onnxruntime.__version__)'
test -f "$ROOT/training_weights/yolox_s.pth" || { echo "The training starting weights are missing: run scripts/fetch_models.py first." >&2; exit 1; }
mkdir -p "$PKG/runtime/python/share/lanepulse"
cp "$ROOT/training_weights/yolox_s.pth" "$PKG/runtime/python/share/lanepulse/"
test -f "$("$PY" -c 'import trafficwb.web, pathlib; print(pathlib.Path(trafficwb.web.__file__).parent / "static" / "index.html")')" \
  || { echo "The web screens are missing: run scripts/build_ui.py first." >&2; exit 1; }
mkdir -p "$PKG/models"
for card in "$ROOT"/models/*/*/model.json; do
  d="$(dirname "$card")"; rel="${d#"$ROOT/models/"}"
  mkdir -p "$PKG/models/$(dirname "$rel")"; cp -R "$d" "$PKG/models/$rel"
done
# The CUDA libraries shipped decide the oldest NVIDIA driver that can run them (Linux minimums from NVIDIA).
CUDA_MAJOR="$("$PY" -c 'import importlib.metadata as m
for d in m.distributions():
    n = (d.metadata["Name"] or "").lower()
    if n.startswith("nvidia-cuda-runtime"):
        print(d.version.split(".")[0]); break' || true)"
case "$CUDA_MAJOR" in
  11) MIN_DRIVER="450.80.02";; 12) MIN_DRIVER="525.60.13";; 13) MIN_DRIVER="580.65.06";; *) MIN_DRIVER="";;
esac
ORT="$("$PY" -c 'import onnxruntime as o; print(o.__version__)')"
printf '{"version": "%s", "python": "3.12", "onnxruntime": "%s", "cuda_major": "%s", "min_driver": "%s"}\n' \
  "$VER" "$ORT" "$CUDA_MAJOR" "$MIN_DRIVER" > "$PKG/server.json"
cp "$ROOT/packaging/server/install-server.sh" "$ROOT/packaging/server/uninstall-server.sh" "$ROOT/packaging/server/system.sh" \
   "$ROOT/packaging/server/lanepulse" "$ROOT/scripts/offline_smoke.py" "$PKG/"
cp "$ROOT/packaging/server/INSTALL-SERVER.txt" "$PKG/INSTALL.txt"
cp "$ROOT/packaging/training/TRAINING.txt" "$PKG/TRAINING.txt"
chmod +x "$PKG/install-server.sh" "$PKG/uninstall-server.sh" "$PKG/lanepulse"
(cd "$PKG" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "Built $OUT/$NAME.tar.gz ($(du -h "$OUT/$NAME.tar.gz" | cut -f1)); CUDA $CUDA_MAJOR, driver $MIN_DRIVER or later"
