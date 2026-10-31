#!/bin/bash
# Builds LanePulse.app and LanePulse-<version>.dmg for Apple-silicon Macs (run on a Mac, in CI).
#   packaging/mac/build_app.sh <python-build-standalone .tar.gz> <its download URL> <out folder>
# Before: the web screens built (scripts/build_ui.py).
#
# An online installer: the app is small and, the first time it is opened (and after an update), downloads what it
# needs into ~/LanePulse/runtime: its own Python, the packages, the models and the training starting weights. Every
# download is pinned here, at build time, to an exact file and its SHA-256, so the Mac gets exactly what was tested.
# After that LanePulse runs offline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTGZ="$1"; PYURL="$2"; OUT="$3"
VER="$(python3 -c 'import re,sys; print(re.search(r"__version__ = \"(.+?)\"", open(sys.argv[1]).read()).group(1))' "$ROOT/src/trafficwb/__init__.py")"
APP="$OUT/LanePulse.app"
SETUP="$APP/Contents/Resources/setup"
WORK="$OUT/work"
rm -rf "$APP" "$WORK"; mkdir -p "$APP/Contents/MacOS" "$SETUP" "$WORK"

# The LanePulse package itself, with the web screens.
python3 -m pip wheel --quiet --no-deps --disable-pip-version-check "$ROOT" -w "$SETUP"
WHEEL="$(ls "$SETUP"/trafficwb-*.whl)"
unzip -l "$WHEEL" | grep -q "trafficwb/web/static/index.html" || { echo "The web screens are missing: run scripts/build_ui.py first." >&2; exit 1; }

# The packages, resolved with the same Python the Mac will use, for macOS 12 or newer, wheels only; then written as a
# list of exact files and their SHA-256 for pip --require-hashes.
tar -xzf "$PYTGZ" -C "$WORK"
"$WORK/python/bin/python3" -m pip install --quiet --disable-pip-version-check --dry-run --ignore-installed \
  --only-binary=:all: --platform macosx_12_0_arm64 --python-version 3.12 --implementation cp --target "$WORK/t" \
  --report "$WORK/report.json" "$WHEEL[probe,cpu,train]"
python3 - "$WORK/report.json" "$SETUP/packages.lock" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
lines = []
for item in report["install"]:
    info = item["download_info"]
    if info["url"].startswith("file:"):
        continue  # trafficwb itself, shipped in the app
    sha = info["archive_info"].get("hashes", {}).get("sha256") or info["archive_info"]["hash"].split("=", 1)[1]
    if not info["url"].endswith(".whl"):
        sys.exit(f"not a wheel: {info['url']}")
    lines.append(f"{item['metadata']['name']} @ {info['url']} --hash=sha256:{sha}")
open(sys.argv[2], "w").write("\n".join(sorted(lines, key=str.lower)) + "\n")
print(f"{len(lines)} packages pinned")
PY
grep -qi "^torch @" "$SETUP/packages.lock" && grep -qi "^onnxruntime @" "$SETUP/packages.lock" && grep -qi "^openvino @" "$SETUP/packages.lock"

# Python itself: the same python-build-standalone file, by URL and SHA-256.
printf '%s\n%s\n' "$PYURL" "$(shasum -a 256 "$PYTGZ" | cut -c1-64)" > "$SETUP/python.txt"

# The model and weight downloader, each file checked against its SHA-256 (fetch_models.py finds the SAM licence
# text two folders up, as in the source tree).
mkdir -p "$SETUP/scripts" "$SETUP/packaging/licenses"
cp "$ROOT/scripts/fetch_models.py" "$ROOT/scripts/pc_check.py" "$SETUP/scripts/"
cp "$ROOT/packaging/licenses/SAM-LICENSE.txt" "$SETUP/packaging/licenses/"
cp "$ROOT/packaging/mac/setup.sh" "$SETUP/setup.sh"
chmod +x "$SETUP/setup.sh"

echo "$VER" > "$APP/Contents/Resources/version.txt"
# Tells this build from an older one of the same version, so opening it sets up again and replaces an older server.
echo "$VER-$(date -u +%Y%m%d%H%M%S)-${GITHUB_SHA:-local}" | cut -c1-48 > "$APP/Contents/Resources/build.txt"
cp "$ROOT/packaging/mac/launcher.sh" "$APP/Contents/MacOS/LanePulse"
chmod +x "$APP/Contents/MacOS/LanePulse"
cp "$ROOT/packaging/mac/lanepulse.icns" "$APP/Contents/Resources/lanepulse.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>LanePulse</string>
  <key>CFBundleDisplayName</key><string>LanePulse</string>
  <key>CFBundleIdentifier</key><string>org.lanepulse.app</string>
  <key>CFBundleExecutable</key><string>LanePulse</string>
  <key>CFBundleIconFile</key><string>lanepulse</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VER</string>
  <key>CFBundleVersion</key><string>$VER</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
</dict></plist>
PLIST
rm -rf "$WORK"
# Not from an Apple developer account: an ad-hoc signature, so macOS asks once (see INSTALL-MAC.txt).
codesign --force --deep --sign - "$APP"
mkdir -p "$OUT/dmg"
rm -rf "$OUT/dmg/"*
cp -R "$APP" "$OUT/dmg/"
ln -s /Applications "$OUT/dmg/Applications"
cp "$ROOT/packaging/mac/INSTALL-MAC.txt" "$OUT/dmg/Read me first.txt"
hdiutil create -quiet -volname "LanePulse $VER" -srcfolder "$OUT/dmg" -ov -format UDZO "$OUT/LanePulse-$VER.dmg"
rm -rf "$OUT/dmg"
echo "Built $OUT/LanePulse-$VER.dmg ($(du -h "$OUT/LanePulse-$VER.dmg" | cut -f1))"
