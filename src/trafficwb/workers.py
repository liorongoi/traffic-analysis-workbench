"""Workers on other computers (phase 5): pairing codes, the operator's switch, and the worker program.

On the server, the operator turns "other computers may help" on and makes a pairing code on the Workers screen.
On any other computer with LanePulse installed, open its "help another computer" launcher (Windows: Start menu
"LanePulse - help another computer"; Mac: ~/LanePulse/Help another computer.command; Linux server:
``lanepulse worker``) and give it the server's address and the code, or run it with them:

    trafficwb-worker --server http://<server>:8765 --code <pairing code>

The worker asks for a segment, reads only that time range of the recording from the server (HTTP range requests),
runs the mission's exact detector (the same model, checked by its checksum and fetched once if missing), posts back
the tracks it found, and keeps nothing of the recording. It renews its lease while it works; if it stops, the
segment goes back to the queue after two minutes. The server keeps only a hash of each pairing code; revoking a code
stops its workers at their next request.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import socket
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .store import DataRoot, utcnow

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no 0/O, 1/I: read aloud and typed easily


def _hash(code: str) -> str:
    return hashlib.sha256(code.strip().upper().replace("-", "").replace(" ", "").encode()).hexdigest()


class Pairing:
    """Pairing codes and the switch, kept in the data root's control database."""

    def __init__(self, data_root):
        self.con = DataRoot(data_root).con

    def enabled(self) -> bool:
        r = self.con.execute("SELECT value FROM app_setting WHERE key='workers_enabled'").fetchone()
        return bool(r and json.loads(r[0]))

    def set_enabled(self, on: bool, author: str) -> None:
        self.con.execute("INSERT INTO app_setting(key,value,author,updated_at) VALUES ('workers_enabled',?,?,?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value, author=excluded.author, "
                         "updated_at=excluded.updated_at", (json.dumps(bool(on)), author, utcnow()))

    def new_code(self, label: str, author: str) -> dict:
        raw = "".join(secrets.choice(ALPHABET) for _ in range(12))
        code = "-".join(raw[i:i + 4] for i in range(0, 12, 4))
        cid = self.con.execute("INSERT INTO worker_code(label,code_hash,author,created_at) VALUES (?,?,?,?)",
                               (label.strip() or "worker", _hash(code), author, utcnow())).lastrowid
        return {"id": cid, "label": label.strip() or "worker", "code": code}

    def codes(self) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT id, label, author, created_at, revoked_at FROM worker_code ORDER BY id DESC")]

    def revoke(self, code_id: int) -> None:
        self.con.execute("UPDATE worker_code SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (utcnow(), code_id))

    def check(self, code: str | None) -> int | None:
        if not code:
            return None
        r = self.con.execute("SELECT id FROM worker_code WHERE code_hash=? AND revoked_at IS NULL",
                             (_hash(code),)).fetchone()
        return r[0] if r else None


def remote_spec(jobs, m) -> dict:
    """What another computer needs to run this mission's detector exactly as this one does."""
    cfg = m.config()
    choice = getattr(cfg, "detector", "standard")
    spec = {"detector": choice, "words": getattr(cfg, "sam3_words", None), "standard": None}
    if choice != "sam3":
        det, card = jobs.standard_for(m)
        pack = getattr(det, "pack", None)
        if pack is not None:
            spec["standard"] = {"name": pack.name, "version": pack.version, "sha256": pack.sha256}
            jobs.hub_packs[pack.sha256] = Path(pack.path)
    return spec


# ---------------------------------------------------------------- the worker program

class WorkerStopped(RuntimeError):
    pass


class Worker:
    """One worker on another computer. ``detector_factory(spec)`` may stand in for the real detectors (tests)."""

    def __init__(self, server: str, code: str, name: str | None = None, cache: Path | None = None,
                 detector_factory: Callable[[dict], object] | None = None, renew_s: float = 20.0,
                 log: Callable[[str], None] = print):
        self.server = server.rstrip("/")
        self.code = code
        self.name = name or f"{socket.gethostname()}-{os.getpid()}"
        self.cache = Path(cache or Path.home() / ".lanepulse-worker")
        self.detector_factory = detector_factory
        self.renew_s = renew_s
        self.log = log
        self._detectors: dict[str, object] = {}
        self.segments = 0

    # ---------- talking to the server ----------
    def _call(self, path: str, body: dict | None = None, timeout: float = 60):
        data = json.dumps(dict(body or {}, code=self.code, name=self.name)).encode()
        req = urllib.request.Request(self.server + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "X-TWB": "1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return None if r.status == 204 else json.loads(r.read() or b"null")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
                detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            except (ValueError, AttributeError):
                pass
            if exc.code in (401, 403):
                raise WorkerStopped(str(detail)) from None
            raise RuntimeError(f"{exc.code}: {detail}") from None

    def can(self) -> list[str]:
        out = ["standard"]
        if self.detector_factory is None:
            try:
                from .sam3 import available
                if available().get("ok"):
                    out.append("sam3")
            except Exception:  # noqa: BLE001
                pass
        else:
            out.append("sam3")
        return out

    # ---------- the detector ----------
    def detector(self, spec: dict):
        key = json.dumps({k: spec.get(k) for k in ("detector", "words", "standard", "algo", "low_score")}, sort_keys=True)
        if key not in self._detectors:
            self._detectors[key] = (self.detector_factory(spec) if self.detector_factory else self._build(spec))
        return self._detectors[key]

    def _pack(self, want: dict):
        from . import models
        for card in models.installed("vehicle_detection"):
            if card.get("sha256") == want["sha256"]:
                return models.find(card["name"], card["version"])
        local = self.cache / "models" / want["sha256"]
        if not (local / "model.json").is_file():
            self.log(f"fetching model {want['name']} {want['version']} from the server (once)")
            local.parent.mkdir(parents=True, exist_ok=True)
            url = f"{self.server}/api/worker/model/{want['sha256']}?code={urllib.request.quote(self.code)}"
            with tempfile.TemporaryDirectory(dir=local.parent) as tmp:
                arc = Path(tmp) / "pack.tar.gz"
                urllib.request.urlretrieve(url, arc)
                with tarfile.open(arc) as t:
                    t.extractall(Path(tmp) / "pack", filter="data")
                os.replace(Path(tmp) / "pack", local)
        pack = models.ModelPack.load(local)              # checks the weights against the checksum
        if pack.sha256 != want["sha256"]:
            raise RuntimeError("The model fetched from the server does not match its checksum.")
        return pack

    def _build(self, spec: dict):
        from .detect import YoloxDetector
        from .video_process import _low_threshold
        from .web.jobs import JobManager
        std = None
        if spec["detector"] != "sam3":
            if not spec.get("standard"):
                raise RuntimeError("The server did not say which model to use.")
            pack = self._pack(spec["standard"])
            std = JobManager._with_classes((YoloxDetector(pack), pack.card), spec.get("words"))
        det = std[0] if std else None
        if spec["detector"] in ("sam3", "both"):
            from .sam3 import Sam3Detector, engine, vehicle_prompts
            sam = Sam3Detector(engine(), prompts=vehicle_prompts(spec.get("words")))
            if spec["detector"] == "sam3":
                det = sam
            else:
                from .combined import CombinedDetector
                det = CombinedDetector(std[0], std[1], sam, sam.engine.pack.card)
        if spec.get("algo") == "byte":
            det = _low_threshold(det, spec.get("low_score", 0.1))
        return det

    # ---------- the work ----------
    def once(self) -> bool:
        """Ask for one segment and do it. False when there was nothing to do."""
        from . import video_segments as vs
        from .tracker import TrackerParams
        from .video_process import _kind, _plain
        info = {"cpus": os.cpu_count(), "system": platform.platform(terse=True), "python": platform.python_version()}
        task = self._call("/api/worker/next", {"can": self.can(), "info": info})
        if not task:
            return False
        lease, spec = task["lease"], task["spec"]
        sg = task["segment"]
        inf = float("inf")
        seg = vs.Segment(sg["idx"], sg["own_a"], inf if sg["own_b"] is None else sg["own_b"], sg["run_a"],
                         inf if sg["run_b"] is None else sg["run_b"])
        stop = threading.Event()
        lost = threading.Event()

        def renew():
            while not stop.wait(self.renew_s):
                try:
                    if not self._call(f"/api/worker/lease/{lease}/renew").get("ok"):
                        lost.set()
                        return
                except Exception:  # noqa: BLE001 (a network blip: try again next time)
                    pass
        threading.Thread(target=renew, daemon=True).start()

        def checkpoint():
            if lost.is_set():
                raise WorkerStopped("the server gave this segment to another worker")
        try:
            det = self.detector(spec)
            video = f"{self.server}/api/worker/lease/{lease}/video?code={urllib.request.quote(self.code)}"
            self.log(f"segment {seg.idx + 1} of recording {task['version_id']} ({seg.own_a:.0f} s on)")
            res = vs.analyse(seg, video, spec["width"], spec["height"], vs.make_detect(_kind(det), det), spec["grid_s"],
                             TrackerParams(**spec["tracker"]), spec["moving"], spec["sam_every"], checkpoint)
            stop.set()
            out = self._call(f"/api/worker/lease/{lease}/result", {"result": _plain(res)}, timeout=300)
            if out and out.get("ok"):
                self.segments += 1
            return True
        except WorkerStopped:
            stop.set()
            raise
        except Exception as exc:  # noqa: BLE001
            stop.set()
            self.log(f"segment failed: {type(exc).__name__}: {exc}")
            try:
                self._call(f"/api/worker/lease/{lease}/give-back", {"error": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001
                pass
            return True

    def run(self, stop: threading.Event | None = None, idle_s: float = 5.0) -> None:
        stop = stop or threading.Event()
        self.log(f"LanePulse worker {self.name} helping {self.server}")
        while not stop.is_set():
            try:
                if not self.once():
                    stop.wait(idle_s)
            except WorkerStopped as exc:
                self.log(f"stopped by the server: {exc}")
                return
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                self.log(f"cannot reach the server ({exc}); trying again")
                stop.wait(max(idle_s, 10))


def main(argv=None) -> int:
    import argparse
    from . import safe_console
    safe_console()
    ap = argparse.ArgumentParser(prog="trafficwb-worker",
                                 description="Help a LanePulse server analyse video, from this computer.")
    ap.add_argument("--server", help="the server's address, e.g. http://192.168.1.20:8765 (asked when left out)")
    ap.add_argument("--code", help="the pairing code from the server's Workers screen (asked when left out)")
    ap.add_argument("--name", help="how this computer is listed on the server (default: its name)")
    args = ap.parse_args(argv)
    try:
        # Opened by double-clicking the "help another computer" launcher: ask for what the command line left out.
        server = (args.server or input("The LanePulse server's address (from its Workers screen): ")).strip()
        code = (args.code or input("The pairing code: ")).strip()
        if not server or not code:
            print("Both the server's address and a pairing code are needed.")
            return 2
        if "://" not in server:
            server = "http://" + server
        Worker(server, code, args.name).run()
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
