#!/usr/bin/env bash
# Offline install of LanePulse on Linux (used to test the bundle; the target is Windows).
#   ./install.sh <target dir> [data dir]
set -euo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
T="${1:?target folder}"
D="${2:-$T/data}"
PY="${PYTHON:-python3}"
cd "$B" && sha256sum --quiet -c SHA256SUMS && cd - >/dev/null
echo "Bundle checksums verified."
V="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$B/bundle.json")"
ORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ort"])' "$B/bundle.json")"
mkdir -p "$T"
[ -e "$T/app-$V" ] && { echo "$T/app-$V exists"; exit 1; }
"$PY" -m venv "$T/app-$V"
mkdir -p "$T/wheelhouse"
for w in "$B"/wheelhouse/*.whl; do [ -e "$T/wheelhouse/$(basename "$w")" ] || cp "$w" "$T/wheelhouse/"; done
"$T/app-$V/bin/python" -m pip install --no-index --find-links "$T/wheelhouse" "trafficwb[probe,$ORT]==$V"
mkdir -p "$T/models"
for d in "$B"/models/*/*/; do
  [ -d "$d" ] || continue
  rel="${d#"$B"/models/}"
  [ -e "$T/models/$rel" ] || { mkdir -p "$T/models/$rel" && cp -r "$d." "$T/models/$rel"; }
done
"$T/app-$V/bin/python" "$B/offline_smoke.py" --work "$(mktemp -d)/check" --models "$T/models" ${SMOKE_IMAGES:+--images "$SMOKE_IMAGES"}
cat > "$T/start.sh" <<EOS
#!/usr/bin/env bash
export TWB_MODELS="$T/models"
exec "$T/app-$V/bin/trafficwb-web" serve --data "$D" "\$@"
EOS
chmod +x "$T/start.sh"
echo "$T/app-$V" > "$T/current.txt"
echo "Installed. Start with $T/start.sh"