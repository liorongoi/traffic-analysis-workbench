"""Disconnected install-to-export check (ENV-04). Run with the installed application's Python.

Creates synthetic images named with UTC times, makes a mission, scans, detects with the
installed vehicle detector, exports CSV and Excel, takes and verifies a backup, and checks
that the local web app opens directly without a session API. It uses no network. Prints one JSON line and exits 0 on success.

    python offline_smoke.py --work <empty folder> [--models <models folder>]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def no_network() -> str:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        return "network reachable"
    except OSError:
        return "no network"


def web_check(data: Path) -> tuple[str, dict]:
    """Start the real web server on a free local port and check the page, health and removed session route."""
    import threading
    import urllib.request
    from urllib.error import HTTPError

    import uvicorn

    from trafficwb.web.app import create_app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(create_app(data, start_monitoring=False), host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # loopback only, never a proxy
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        page = opener.open(f"http://127.0.0.1:{port}/", timeout=10).read().decode("utf-8")
        health = json.loads(opener.open(f"http://127.0.0.1:{port}/api/health", timeout=10).read())
        try:
            opener.open(f"http://127.0.0.1:{port}/api/session", timeout=10)
        except HTTPError as exc:
            if exc.code != 404:
                raise
        else:
            raise AssertionError("The removed session endpoint is still available.")
    finally:
        server.should_exit = True
        th.join(10)
    return page, health


SAM3_MIN_FREE = 8.5e9     # one SAM 3.1 image peaks at about 7 GB of memory


def sam_check() -> dict:
    """SAM 3.1, when its pack is installed: load it and, with enough free memory, outline a road on a synthetic street.

    A failure is reported, not fatal: LanePulse works without SAM 3.1, and the screens say why it is unavailable.
    With TWB_REQUIRE_SAM3=1 (the build machine's tests) any failure fails the check.
    """
    import cv2
    import numpy as np
    import psutil

    from trafficwb import sam3
    st = sam3.available()
    if not st["pack"]:
        return {"installed": False}
    t0 = time.time()
    out: dict = {"installed": True}
    try:
        eng = sam3.engine()
        out["pack"] = f"{eng.pack.name} {eng.pack.version}"
        eng.load()
        out["load_s"] = eng.load_seconds
        free = psutil.virtual_memory().available
        if free < SAM3_MIN_FREE:
            out["inference"] = f"skipped: {free / 1e9:.1f} GB of memory free, {SAM3_MIN_FREE / 1e9:.1f} GB needed for a full check"
        else:
            img = np.full((720, 1280, 3), (200, 170, 140), np.uint8)               # sky
            cv2.fillPoly(img, [np.array([[560, 300], [720, 300], [1280, 720], [0, 720]], np.int32)], (75, 75, 78))
            for y in range(330, 720, 60):                                           # lane dashes
                cv2.line(img, (640, y), (640, y + 30), (235, 235, 235), 6)
            out["road_areas"] = len(sam3.find_areas(eng, img, ["road"], threshold=0.3))
            out["inference"] = "ok"
        eng.release()
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported in words
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        if os.environ.get("TWB_REQUIRE_SAM3") == "1":
            raise
        print(f"warning: SAM 3.1 did not pass its check on this computer ({out['error']}); "
              "everything else works, and SAM 3.1 reports the error when it is used.", file=sys.stderr)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def step(name: str) -> None:
    """Says on stderr which part of the check runs, so a check that stops says where."""
    print(f"check: {name}", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    import faulthandler
    faulthandler.enable()   # a crash inside a compiled library still prints where it happened
    from trafficwb import safe_console
    safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--models")
    ap.add_argument("--images", help="use these images (names with UTC times) instead of synthetic ones")
    a = ap.parse_args(argv)
    if a.models:
        os.environ["TWB_MODELS"] = a.models
    import cv2
    import numpy as np

    from trafficwb import __version__, admin
    from trafficwb.cli import main as mission_cli
    from trafficwb.store import DataRoot

    work = Path(a.work)
    if work.exists() and any(work.iterdir()):
        print(f"{work} is not empty", file=sys.stderr)
        return 2
    src, out, data, bk = work / "src", work / "out", work / "data", work / "backup"
    for d in (src, out):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    old = time.time() - 3600
    import shutil
    given = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if a.images else []
    for p in given:
        shutil.copy2(p, src / p.name)
        os.utime(src / p.name, (old, old))
    for k in range(0 if given else 4):
        img = np.full((720, 1280, 3), 110, np.uint8)
        cv2.rectangle(img, (0, 400), (1280, 720), (70, 70, 70), -1)
        img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
        t = datetime(2026, 10, 21, 8, tzinfo=timezone.utc) + timedelta(minutes=5 * k)
        p = src / f"CAM1_{t.strftime('%Y%m%dT%H%M%SZ')}.jpg"
        cv2.imwrite(str(p), img)
        os.utime(p, (old, old))
    t0 = time.time()
    step("mission, scan and detect")
    assert mission_cli(["create", "--data", str(data), "--name", "Offline check – בדיקה", "--country", "IL",
                        "--zone", "Asia/Jerusalem", "--source", str(src), "--output", str(out)]) == 0
    mid = DataRoot(data).missions()[0]["id"]
    assert mission_cli(["scan", "--data", str(data), "--mission", mid, "--stability", "0"]) == 0
    assert mission_cli(["detect", "--data", str(data), "--mission", mid]) == 0
    step("export")
    assert mission_cli(["export", "--data", str(data), "--mission", mid, "--formats", "csv,xlsx"]) == 0
    exported = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert any(n.endswith(".csv") for n in exported) and any(n.endswith(".xlsx") for n in exported), exported
    step("backup")
    b = admin.backup(data, bk, "smoke")
    assert admin.verify(b) == []
    step("web server")
    page, health = web_check(data)
    assert '<div id="root">' in page and health.get("status") == "ok"
    from trafficwb.models import find
    from trafficwb.runtime import make_session
    step("vehicle detector")
    _, info = make_session(find("yolox-s").weights)
    step("SAM 3.1")
    sam = sam_check()
    from trafficwb.store import DataRoot as _DR
    m = _DR(data).open_mission(mid)
    n_img = m.con.execute("SELECT COUNT(*) FROM source_version").fetchone()[0]
    n_det = m.con.execute("SELECT COUNT(*) FROM detection").fetchone()[0]
    print(json.dumps({"ok": True, "version": __version__, "network": no_network(), "images": n_img, "detections": n_det,
                      "exported": exported, "backup": b.name, "accelerator": info.record(), "sam3": sam,
                      "seconds": round(time.time() - t0, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
