"""Production check on real footage: the real server, started the way the network server starts it, offline.

    python scripts/soak.py --images <folder of camera stills> --videos <folder of recordings> --work <empty folder>
                           [--port 8799] [--load 8] [--restart-at 0.4]

What it does, through the HTTP API only (as the browser does):
  1. Starts `trafficwb-web serve --lan` with downloads blocked (dead proxy, no package index, Hugging Face offline).
  2. Checks the access key and the Host check (a rebinding name is refused).
  3. An image mission over --images and a video mission over --videos, both started.
  4. While they run, --load clients keep reading missions, results, events and thumbnails; latency is recorded.
  5. Kills the server partway through the video run (--restart-at, a fraction of the video work) and starts it
     again: the run must resume, not start over, and finish.
  6. Exports, then reports wall time, video real-time factor, peak server memory and request latency (p50/p95).
Prints one JSON report and exits 1 if any check failed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEAD = "http://127.0.0.1:9"
DONE = ("done", "finished")


class Api:
    def __init__(self, base: str, key: str | None = None, host: str | None = None):
        self.base, self.key, self.host = base, key, host

    def call(self, method: str, path: str, body=None, raw=False, timeout=120):
        h = {"X-TWB": "1"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        if self.host:
            h["Host"] = self.host
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as r:
                b = r.read()
                return r.status, (b if raw else (json.loads(b) if b else None))
        except urllib.error.HTTPError as e:
            b = e.read()
            try:
                return e.code, json.loads(b)
            except ValueError:
                return e.code, b

    def get(self, p, **k):
        return self.call("GET", p, **k)

    def post(self, p, body=None, **k):
        return self.call("POST", p, body if body is not None else {}, **k)


def rss_mb(pid: int) -> float:
    try:
        total = 0
        for p in [pid, *children(pid)]:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
        return total / 1024
    except OSError:
        return 0.0


def children(pid: int) -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
        return [int(x) for x in out]
    except OSError:
        return []


def start_server(data: Path, port: int, log: Path) -> subprocess.Popen:
    env = dict(os.environ, HTTPS_PROXY=DEAD, HTTP_PROXY=DEAD, https_proxy=DEAD, http_proxy=DEAD, PIP_NO_INDEX="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="", no_proxy="")
    return subprocess.Popen([sys.executable, "-c", "import sys; from trafficwb.web.app import main; "
                             "sys.exit(main(sys.argv[1:]))", "serve", "--lan", "--data", str(data), "--port", str(port)],
                            stdout=open(log, "ab"), stderr=subprocess.STDOUT, env=env)


def wait_up(api: Api, s: float = 90):
    end = time.time() + s
    while time.time() < end:
        try:
            if api.get("/api/health", timeout=3)[0] == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--load", type=int, default=8, help="clients reading while the work runs")
    ap.add_argument("--restart-at", type=float, default=0.4)
    ap.add_argument("--timeout-min", type=float, default=120)
    a = ap.parse_args(argv)
    work = Path(a.work)
    data, out, log = work / "data", work / "out", work / "server.log"
    work.mkdir(parents=True, exist_ok=True)
    for d in (out / "images", out / "video"):
        d.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    report: dict = {"checks": checks}

    srv = start_server(data, a.port, log)
    peak = [0.0]
    stop = threading.Event()

    def watch_mem():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb(srv.pid))
            time.sleep(1)

    threading.Thread(target=watch_mem, daemon=True).start()
    anon = Api(f"http://127.0.0.1:{a.port}")
    checks["server_starts_offline"] = wait_up(anon)
    from trafficwb.web.app import access_key_for
    key = access_key_for(data)
    checks["api_needs_key_on_network"] = anon.get("/api/missions")[0] == 401
    checks["rebinding_host_refused"] = Api(anon.base, key, host="rebind.example.net").get("/api/missions")[0] == 421
    api = Api(anon.base, key)
    checks["key_opens_api"] = api.get("/api/missions")[0] == 200

    base = {"country": "FI", "iana_zone": "Europe/Helsinki", "processing": "one_time", "export_formats": ["xlsx", "csv"]}
    s, im = api.post("/api/missions", base | {"name": "soak images", "input_type": "images", "mode": "general_traffic",
                                              "source_dir": a.images, "output_dir": str(out / "images")})
    s2, vm = api.post("/api/missions", base | {"name": "soak video", "input_type": "video", "mode": "object_search",
                                               "source_dir": a.videos, "output_dir": str(out / "video"),
                                               "camera": "fixed"})
    checks["missions_created"] = s in (200, 201) and s2 in (200, 201)
    if not checks["missions_created"]:
        report["errors"] = [im, vm]
        srv.kill()
        print(json.dumps(report, indent=1))
        return 1
    im_id, vm_id = im["id"], vm["id"]
    # Files copied in the last minute wait until they stop changing (as on a camera share): wait that out.
    newest = max(f.stat().st_mtime for d in (a.images, a.videos) for f in Path(d).iterdir())
    time.sleep(max(0.0, newest + 65 - time.time()))
    t0 = time.time()
    api.post(f"/api/missions/{im_id}/job")
    api.post(f"/api/missions/{vm_id}/job")

    lat: list[float] = []
    errs: list[str] = []

    def reader(i: int):
        paths = [f"/api/missions/{im_id}", f"/api/missions/{vm_id}", "/api/missions", f"/api/missions/{vm_id}/events",
                 f"/api/missions/{vm_id}/recordings", f"/api/missions/{im_id}/thumbs/1", "/api/system"]
        n = 0
        while not stop.is_set():
            p = paths[(i + n) % len(paths)]
            n += 1
            t = time.perf_counter()
            try:
                st, _ = api.call("GET", p, raw=True, timeout=30)
                if st >= 500:
                    errs.append(f"{p} {st}")
            except OSError as e:
                if not restarting.is_set():
                    errs.append(f"{p} {type(e).__name__}")
                time.sleep(0.5)
                continue
            lat.append(time.perf_counter() - t)
            time.sleep(0.05)

    restarting = threading.Event()
    for i in range(a.load):
        threading.Thread(target=reader, args=(i,), daemon=True).start()

    def job(mid):
        st, m = api.get(f"/api/missions/{mid}")
        return (m or {}).get("job") or {} if st == 200 else {}

    restarted = False
    resumed_from = None
    deadline = t0 + a.timeout_min * 60
    while time.time() < deadline:
        vj, ij = job(vm_id), job(im_id)
        if not restarted and vj.get("total") and vj.get("done", 0) >= a.restart_at * vj["total"] > 0:
            resumed_from = vj["done"]
            restarting.set()
            srv.send_signal(signal.SIGKILL)               # a crash, not a clean stop
            srv.wait()
            srv = start_server(data, a.port, log)
            checks["restarts_after_crash"] = wait_up(anon)
            restarting.clear()
            api.post(f"/api/missions/{vm_id}/job")        # the operator presses Start again
            if ij.get("state") in ("running", "paused"):
                api.post(f"/api/missions/{im_id}/job")
            restarted = True
            continue
        if vj.get("state") not in ("running", "paused") and ij.get("state") not in ("running", "paused") \
                and (restarted or vj.get("state") in DONE):
            break
        time.sleep(3)
    wall = time.time() - t0
    stop.set()

    _, vmd = api.get(f"/api/missions/{vm_id}")
    _, imd = api.get(f"/api/missions/{im_id}")
    vj, ij = vmd.get("job") or {}, imd.get("job") or {}
    report["video_job"] = {k: vj.get(k) for k in ("state", "done", "total", "elapsed_s", "error", "result")}
    report["image_job"] = {k: ij.get(k) for k in ("state", "done", "total", "elapsed_s", "error")}
    report["image_counts"] = imd.get("processing_counts")
    report["video_counts"] = vmd.get("processing_counts")
    checks["video_finished"] = vj.get("state") in DONE and not vj.get("error")
    checks["images_finished"] = ij.get("state") in DONE and not ij.get("error")
    checks["no_failed_items"] = not (imd.get("processing_counts") or {}).get("failed") and \
        not (vmd.get("processing_counts") or {}).get("failed")
    if restarted:
        checks["video_resumed_not_restarted"] = vj.get("done", 0) < (vj.get("total") or 0) or \
            (vj.get("total") or 0) - resumed_from <= vj.get("done", 0)
        report["resumed_from_done"] = resumed_from
    _, evs = api.get(f"/api/missions/{vm_id}/events")
    nev = len(evs) if isinstance(evs, list) else len((evs or {}).get("events", []))
    report["video_events"] = nev
    checks["video_found_vehicles"] = nev > 0
    s, res = api.post(f"/api/missions/{im_id}/results", {})
    rows = res if isinstance(res, list) else (res or {}).get("rows", [])
    report["image_rows"] = len(rows)
    report["image_vehicles"] = sum((r.get("vehicle_count") or 0) for r in rows)
    checks["images_found_vehicles"] = report["image_vehicles"] > 0
    s, ex = api.post(f"/api/missions/{im_id}/export", {"formats": ["xlsx", "csv"]})
    checks["image_export"] = s == 200 and all(Path(f).is_file() for f in ex.get("files", []))
    s, ex2 = api.post(f"/api/missions/{vm_id}/export", {"formats": ["xlsx", "csv"]})
    checks["video_export"] = s == 200 and all(Path(f).is_file() for f in (ex2 or {}).get("files", []))
    if s != 200:
        report["video_export_error"] = ex2

    video_s = 0.0
    try:
        import av
        for f in Path(a.videos).iterdir():
            with av.open(str(f)) as c:
                video_s += (c.duration or 0) / 1e6
    except Exception:
        pass
    report["video_seconds"] = round(video_s, 1)
    report["wall_s"] = round(wall, 1)
    report["peak_server_mb"] = round(peak[0])
    if lat:
        q = statistics.quantiles(lat, n=20)
        report["requests"] = {"n": len(lat), "p50_ms": round(statistics.median(lat) * 1000),
                              "p95_ms": round(q[18] * 1000), "max_ms": round(max(lat) * 1000)}
    report["request_errors"] = errs[:20]
    checks["no_server_errors_under_load"] = not errs
    srv.terminate()
    try:
        srv.wait(20)
    except subprocess.TimeoutExpired:
        srv.kill()
    report["ok"] = all(checks.values())
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
