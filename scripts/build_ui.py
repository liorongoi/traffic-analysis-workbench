"""Build the web screens into src/trafficwb/web/static (staging machine or CI only).

Needs Node.js 20+. The workstation never runs this: it gets the built files in the package.
"""
import shutil
import subprocess
import sys
from pathlib import Path

UI = Path(__file__).resolve().parent.parent / "ui"


def main() -> int:
    npm = shutil.which("npm")
    if not npm:
        print("npm not found: install Node.js 20 or later on the staging machine", file=sys.stderr)
        return 1
    for args in (["ci"], ["run", "build"]):
        r = subprocess.run([npm, *args], cwd=UI)
        if r.returncode:
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
