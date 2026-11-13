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


MAX_ASKING = 20       # computers waiting for an answer at once (more are told to try later)


class Requests:
    """Computers that found this server on the network and ask to help. Allowing one gives its own key the standing of
    a pairing code (listed with the codes, stopped the same way); nothing is typed on either computer."""

    def __init__(self, data_root):
        self.con = DataRoot(data_root).con

    def server_id(self) -> str:
        r = self.con.execute("SELECT value FROM app_setting WHERE key='server_id'").fetchone()
        if r:
            return json.loads(r[0])
        sid = secrets.token_hex(8)
        self.con.execute("INSERT OR IGNORE INTO app_setting(key,value,author,updated_at) VALUES ('server_id',?,?,?)",
                         (json.dumps(sid), "system", utcnow()))
        return json.loads(self.con.execute("SELECT value FROM app_setting WHERE key='server_id'").fetchone()[0])

    def ask(self, key: str, name: str, info: dict, address: str | None) -> str:
        """asking | allowed | refused | busy"""
        h = _hash(key)
        now = utcnow()
        code = self.con.execute("SELECT revoked_at FROM worker_code WHERE code_hash=?", (h,)).fetchone()
        r = self.con.execute("SELECT id, state FROM worker_request WHERE key_hash=?", (h,)).fetchone()
        if r is None:
            if code is not None and code["revoked_at"] is None:
                return "allowed"
            waiting = self.con.execute("SELECT COUNT(*) FROM worker_request WHERE state='asking'").fetchone()[0]
            if waiting >= MAX_ASKING:
                return "busy"
            self.con.execute("INSERT INTO worker_request(name,key_hash,info,address,created_at,seen_at) "
                             "VALUES (?,?,?,?,?,?)", (name[:80], h, json.dumps(info)[:2000], address, now, now))
            return "asking"
        state = r["state"]
        if state == "allowed" and (code is None or code["revoked_at"] is not None):
            state = "asking"                     # its code was stopped: it asks again
        self.con.execute("UPDATE worker_request SET name=?, info=?, address=?, seen_at=?, state=? WHERE id=?",
                         (name[:80], json.dumps(info)[:2000], address, now, state, r["id"]))
        return state

    def asking(self) -> list[dict]:
        out = []
        for r in self.con.execute("SELECT id, name, info, address, created_at, seen_at FROM worker_request "
                                  "WHERE state='asking' ORDER BY id"):
            d = dict(r)
            d["info"] = json.loads(d["info"] or "{}")
            out.append(d)
        return out

    def decide(self, request_id: int, allow: bool, author: str) -> None:
        r = self.con.execute("SELECT name, key_hash FROM worker_request WHERE id=?", (request_id,)).fetchone()
        if r is None:
            raise KeyError(request_id)
        if allow:
            if self.con.execute("SELECT 1 FROM worker_code WHERE code_hash=?", (r["key_hash"],)).fetchone():
                self.con.execute("UPDATE worker_code SET revoked_at=NULL, label=?, author=? WHERE code_hash=?",
                                 (r["name"], author, r["key_hash"]))
            else:
                self.con.execute("INSERT INTO worker_code(label,code_hash,author,created_at) VALUES (?,?,?,?)",
                                 (r["name"], r["key_hash"], author, utcnow()))
        self.con.execute("UPDATE worker_request SET state=?, decided_by=?, decided_at=? WHERE id=?",
                         ("allowed" if allow else "refused", author, utcnow(), request_id))


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
        from .detect import make_detector
        from .video_process import _low_threshold
        from .web.jobs import JobManager
        std = None
        if spec["detector"] != "sam3":
            if not spec.get("standard"):
                raise RuntimeError("The server did not say which model to use.")
            pack = self._pack(spec["standard"])
            std = JobManager._with_classes((make_detector(pack), pack.card), spec.get("words"))
        det = std[0] if std else None
        from .detectors import uses_sam
        if uses_sam(spec["detector"]):
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

    def run(self, stop: threading.Event | None = None, idle_s: float = 5.0,
            refind: Callable[[], str | None] | None = None) -> None:
        """Help until stopped. ``refind`` looks for the server again when it cannot be reached (its address changed)."""
        stop = stop or threading.Event()
        self.log(f"LanePulse worker {self.name} helping {self.server}")
        misses = 0
        while not stop.is_set():
            try:
                if not self.once():
                    stop.wait(idle_s)
                misses = 0
            except WorkerStopped as exc:
                self.log(f"stopped by the server: {exc}")
                return
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                misses += 1
                self.log(f"cannot reach the server ({exc}); trying again")
                if refind and misses >= 3:
                    url = refind()
                    if url and url != self.server:
                        self.log(f"found the server again at {url}")
                        self.server = url
                stop.wait(max(idle_s, 10))


# ---------------------------------------------------------------- finding the server and asking to help

def identity(cache: Path | None = None) -> dict:
    """This computer's own key (made once, kept in the worker's folder) and its saved choices."""
    cache = Path(cache or Path.home() / ".lanepulse-worker")
    f = cache / "identity.json"
    try:
        d = json.loads(f.read_text())
        if len(d.get("key", "")) >= 32:
            return d
    except (OSError, ValueError):
        pass
    d = {"key": secrets.token_hex(24)}
    save_identity(d, cache)
    return d


def save_identity(d: dict, cache: Path | None = None) -> None:
    cache = Path(cache or Path.home() / ".lanepulse-worker")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / "identity.json.tmp"
    tmp.write_text(json.dumps(d))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, cache / "identity.json")


def _hello(server: str, key: str, name: str, timeout: float = 15) -> dict:
    info = {"cpus": os.cpu_count(), "system": platform.platform(terse=True)}
    req = urllib.request.Request(server.rstrip("/") + "/api/worker/hello", method="POST",
                                 data=json.dumps({"key": key, "name": name, "info": info}).encode(),
                                 headers={"Content-Type": "application/json", "X-TWB": "1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


ASK_TEXT = {
    "asking": "Waiting for the server to allow this computer: on its Workers screen, choose Allow for {name}.",
    "off": "Allowed. The server is not taking help right now; this computer starts when it is turned on there.",
    "busy": "Too many computers are asking that server at once; trying again soon.",
}


def ask_until_allowed(server: str, key: str, name: str, log=print, stop: threading.Event | None = None,
                      wait_s: float = 5.0) -> bool:
    """Ask the server to let this computer help, and wait for the operator's answer. False when refused or stopped."""
    stop = stop or threading.Event()
    said = None
    while not stop.is_set():
        try:
            state = _hello(server, key, name)["state"]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            state, text = "unreachable", f"cannot reach {server} ({exc}); trying again"
        else:
            if state == "allowed":
                return True
            if state == "refused":
                log("The server's operator chose not to take help from this computer.")
                return False
            text = ASK_TEXT.get(state, state).format(name=name)
        if text != said:
            log(text)
            said = text
        stop.wait(wait_s if state in ("asking", "unreachable") else 30)
    return False


def pick_server(found: list[dict], interactive: bool, chosen_id: str | None = None) -> dict | None:
    if not found:
        return None
    for f in found:
        if chosen_id and f["id"] == chosen_id:
            return f
    if len(found) == 1 or not interactive:
        return found[0] if len(found) == 1 else None
    print("LanePulse servers on this network:")
    for k, f in enumerate(found, 1):
        print(f"  {k}. {f['name']} ({f['url']})")
    while True:
        a = input(f"Which one should this computer help? (1-{len(found)}): ").strip()
        if a.isdigit() and 1 <= int(a) <= len(found):
            return found[int(a) - 1]


def start_at_signin(on: bool = True) -> str:
    """Start helping (finding the server on its own) whenever this computer signs in. Returns what was done."""
    import subprocess
    import sys
    models = os.environ.get("TWB_MODELS", "")
    if sys.platform == "win32":
        folder = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / \
            "Microsoft/Windows/Start Menu/Programs/Startup"
        f = folder / "LanePulse - help another computer.cmd"
        if not on:
            f.unlink(missing_ok=True)
            return "This computer no longer starts helping when it signs in."
        folder.mkdir(parents=True, exist_ok=True)
        pyw = Path(sys.executable).with_name("pythonw.exe")
        exe = pyw if pyw.is_file() else Path(sys.executable)
        f.write_text("@echo off\r\n"
                     + (f'set TWB_MODELS={models}\r\n' if models else "")
                     + f'start "" /min "{exe}" -m trafficwb.workers --quiet\r\n', encoding="ascii")
        return f"This computer starts helping when it signs in ({f})."
    if sys.platform == "darwin":
        f = Path.home() / "Library/LaunchAgents/com.lanepulse.worker.plist"
        if not on:
            subprocess.run(["launchctl", "unload", str(f)], capture_output=True)
            f.unlink(missing_ok=True)
            return "This Mac no longer starts helping when it signs in."
        f.parent.mkdir(parents=True, exist_ok=True)
        log = Path.home() / ".lanepulse-worker" / "worker.log"
        env = f"<key>EnvironmentVariables</key><dict><key>TWB_MODELS</key><string>{models}</string></dict>" if models else ""
        f.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.lanepulse.worker</string>
<key>ProgramArguments</key><array><string>{sys.executable}</string><string>-m</string><string>trafficwb.workers</string><string>--quiet</string></array>
{env}
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
<key>StandardOutPath</key><string>{log}</string><key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
""")
        subprocess.run(["launchctl", "load", "-w", str(f)], capture_output=True)
        return f"This Mac starts helping when it signs in ({f})."
    f = Path.home() / ".config/systemd/user/lanepulse-worker.service"
    if not on:
        subprocess.run(["systemctl", "--user", "disable", "--now", "lanepulse-worker"], capture_output=True)
        f.unlink(missing_ok=True)
        return "This computer no longer starts helping by itself."
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("[Unit]\nDescription=LanePulse: help another LanePulse server\nAfter=network-online.target\n\n"
                 f"[Service]\n{'Environment=TWB_MODELS=' + models + chr(10) if models else ''}"
                 f"ExecStart={sys.executable} -m trafficwb.workers --quiet\nRestart=always\nRestartSec=30\n\n"
                 "[Install]\nWantedBy=default.target\n")
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "lanepulse-worker"], capture_output=True)
    return ("This computer starts helping by itself." if r.returncode == 0 else
            f"Wrote {f}; turn it on with: systemctl --user enable --now lanepulse-worker")


def main(argv=None) -> int:
    import argparse
    import sys
    from . import discovery, safe_console
    safe_console()
    ap = argparse.ArgumentParser(prog="trafficwb-worker",
                                 description="Help a LanePulse server analyse video, from this computer. With no "
                                             "options it finds the server on this network and asks to help; the "
                                             "server's operator allows it on the Workers screen.")
    ap.add_argument("--server", help="the server's address, e.g. http://192.168.1.20:8765 (found on the network "
                                     "when left out)")
    ap.add_argument("--code", help="a pairing code from the server's Workers screen (instead of asking the operator)")
    ap.add_argument("--name", help="how this computer is listed on the server (default: its name)")
    ap.add_argument("--start-at-signin", choices=["on", "off"],
                    help="help by itself whenever this computer signs in (on), or stop doing that (off)")
    ap.add_argument("--quiet", action="store_true", help="never ask anything (used when started at sign-in)")
    args = ap.parse_args(argv)
    if args.start_at_signin:
        print(start_at_signin(args.start_at_signin == "on"))
        return 0
    if sys.stdout is None:                   # started at sign-in without a window: keep a log instead
        log_dir = Path.home() / ".lanepulse-worker"
        log_dir.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(log_dir / "worker.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    interactive = not args.quiet and sys.stdin is not None and sys.stdin.isatty()
    name = args.name or socket.gethostname()
    me = identity()

    def log(text):
        print(time.strftime("%H:%M:%S"), text, flush=True)

    def refind() -> str | None:
        f = pick_server(discovery.find(), False, me.get("server_id"))
        return f["url"] if f else None
    try:
        server = (args.server or "").strip()
        found_on_network = not server
        while not server:
            log("Looking for a LanePulse server on this network...")
            f = pick_server(discovery.find(), interactive, me.get("server_id"))
            if f:
                server = f["url"]
                log(f"Found {f['name']} at {server}")
                if f.get("id"):
                    me["server_id"] = f["id"]
                    save_identity(me)
            elif interactive:
                print("No LanePulse server answered on this network. It must be started with network access, and on a")
                print("separate network or a VPN it cannot be found: type its address (shown on its Workers screen).")
                server = input("The server's address (Return to look again): ").strip()
            else:
                time.sleep(30)
        if "://" not in server:
            server = "http://" + server
        code = (args.code or "").strip()
        if not code:
            if not ask_until_allowed(server, me["key"], name, log):
                return 1
            code = me["key"]
            if interactive and "signin_asked" not in me:
                a = input("Help this server by itself whenever this computer is on? [Y/n]: ").strip().lower()
                me["signin_asked"] = True
                save_identity(me)
                if a in ("", "y", "yes"):
                    log(start_at_signin(True))
                    log("It is now helping in the background as well; you can close this window.")
                    return 0
        Worker(server, code, name, log=log).run(refind=refind if found_on_network else None)
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
