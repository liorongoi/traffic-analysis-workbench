#!/bin/bash
# Builds LanePulse.app and LanePulse-<version>.dmg for Apple-silicon Macs (run on a Mac, in CI).
#   packaging/mac/build_app.sh <python-build-standalone .tar.gz> <out folder>
# Before: the web screens built (scripts/build_ui.py) and the model packs fetched (scripts/fetch_models.py).
# The app carries its own Python and every package, so the Mac needs nothing else and downloads nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTGZ="$1"; OUT="$2"
VER="$(python3 -c 'import re,sys; print(re.search(r"__version__ = \"(.+?)\"", open(sys.argv[1]).read()).group(1))' "$ROOT/src/trafficwb/__init__.py")"
APP="$OUT/LanePulse.app"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
tar -xzf "$PYTGZ" -C "$APP/Contents/Resources"
PY="$APP/Contents/Resources/python/bin/python3"
"$PY" -m pip install --quiet --no-cache-dir --disable-pip-version-check "$ROOT[probe,cpu]"
test -f "$("$PY" -c 'import trafficwb.web, pathlib; print(pathlib.Path(trafficwb.web.__file__).parent / "static" / "index.html")')" \
  || { echo "The web screens are missing: run scripts/build_ui.py first." >&2; exit 1; }
for card in "$ROOT"/models/*/*/model.json; do
  d="$(dirname "$card")"; rel="${d#"$ROOT/models/"}"
  mkdir -p "$APP/Contents/Resources/models/$(dirname "$rel")"
  cp -R "$d" "$APP/Contents/Resources/models/$rel"
done
echo "$VER" > "$APP/Contents/Resources/version.txt"
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
