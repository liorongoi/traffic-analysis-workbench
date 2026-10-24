"""Source ingestion: reconciliation scans, file stability, versions.

A reconciliation scan is authoritative. Filesystem change notifications (when
available) only make the next scan happen sooner, so a lost notification can
delay a file but never lose it, and repeated notifications cannot create
duplicates.

Rules:
- Originals are only read, never modified, moved or deleted.
- A file is ingested only when stable: its last-modified time is at least
  `stability_seconds` old, its size and mtime have not changed since the last
  observation (if any), and, for JPEG, the end-of-image marker is present. A
  JPEG that stays truncated across a full stability interval is recorded as
  corrupt.
- Each distinct content at a path is a new source version. Earlier versions,
  and anything attached to them, are kept.
- Identical bytes under different names are separate files, never merged.
- Application output and data folders are never scanned.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .inventory import IMAGE_EXT, VIDEO_EXT, read_image_header
from .store import Mission, utcnow
from .timeparse import MalformedName, parse_capture_time

DEFAULT_STABILITY_SECONDS = 30


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ScanResult:
    scan_id: int
    status: str
    new_versions: list[int] = field(default_factory=list)  # eligible for processing
    malformed: list[str] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)
    waiting_for_stability: int = 0
    newly_missing: list[str] = field(default_factory=list)
    unchanged: int = 0
    ignored_by_extension: dict[str, int] = field(default_factory=dict)


class Ingestor:
    def __init__(self, mission: Mission, stability_seconds: int = DEFAULT_STABILITY_SECONDS):
        self.m = mission
        self.con = mission.con
        self.stability = timedelta(seconds=stability_seconds)
        self.extensions = IMAGE_EXT if mission.input_type == "images" else VIDEO_EXT

    def _excluded_dirs(self) -> list[Path]:
        cfg = self.m.config()
        return [Path(cfg.output_dir).resolve(), self.m.root.path.resolve()]

    def _walk(self, root: Path, recursive: bool):
        excluded = self._excluded_dirs()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            d = Path(dirpath)
            dirnames[:] = sorted(
                n for n in dirnames if recursive and not any((d / n).resolve() == e for e in excluded)
            )
            for n in sorted(filenames):
                yield d / n

    def scan(self, now: datetime | None = None) -> ScanResult:
        now = now or datetime.now(timezone.utc)
        now_s = _iso(now)
        scan_id = self.con.execute(
            "INSERT INTO scan(started_at,status) VALUES (?, 'running')", (now_s,)
        ).lastrowid
        res = ScanResult(scan_id, "completed")
        ignored: dict[str, int] = {}
        for root in self.con.execute("SELECT * FROM source_root").fetchall():
            rpath = Path(root["path"])
            if not rpath.is_dir() or not os.access(rpath, os.R_OK | os.X_OK):
                res.status = "source_unavailable"
                self.m.notify("error", "source_unavailable",
                              f"Source folder {rpath} is not reachable. Nothing was marked missing; "
                              "the next scan retries.", path=str(rpath))
                continue
            seen: set[str] = set()
            for path in self._walk(rpath, bool(root["recursive"])):
                ext = path.suffix.lower()
                if ext not in self.extensions:
                    ignored[ext or "(none)"] = ignored.get(ext or "(none)", 0) + 1
                    continue
                rel = path.relative_to(rpath).as_posix()
                seen.add(rel)
                try:
                    self._observe(root["id"], rpath, rel, path, now, res)
                except OSError as exc:
                    self.m.notify("warning", "read_failed", f"Could not read {rel}: {exc}", path=rel)
            self._mark_missing(root["id"], seen, now_s, res)
            self.con.execute(
                "DELETE FROM pending_file WHERE root_id=? AND rel_path NOT IN (SELECT value FROM json_each(?))",
                (root["id"], json.dumps(sorted(seen))),
            )
        res.ignored_by_extension = ignored
        res.waiting_for_stability = self.con.execute("SELECT COUNT(*) FROM pending_file").fetchone()[0]
        if res.malformed and self.m.input_type == "video":
            # Video times are relative to the stream start, so a recording is still analysed; it gets no absolute UTC.
            self.m.notify("warning", "malformed_names",
                          f"{len(res.malformed)} recording(s) have names without a start time. They are analysed, "
                          "but their events have no absolute UTC.", examples=res.malformed[:10])
        elif res.malformed:
            self.m.notify("warning", "malformed_names",
                          f"{len(res.malformed)} file(s) have names without a valid capture time and were not queued.",
                          examples=res.malformed[:10])
        if res.corrupt:
            self.m.notify("warning", "corrupt_files", f"{len(res.corrupt)} file(s) are corrupt or incomplete.",
                          examples=res.corrupt[:10])
        summary = {k: (len(v) if isinstance(v, list) else v) for k, v in res.__dict__.items() if k not in ("scan_id", "status")}
        self.con.execute("UPDATE scan SET finished_at=?, status=?, summary=? WHERE id=?",
                         (utcnow(), res.status, json.dumps(summary), scan_id))
        return res

    def _observe(self, root_id: int, rpath: Path, rel: str, path: Path, now: datetime, res: ScanResult) -> None:
        st = path.stat()
        sf = self.con.execute("SELECT * FROM source_file WHERE root_id=? AND rel_path=?", (root_id, rel)).fetchone()
        if sf and sf["last_size"] == st.st_size and sf["last_mtime_ns"] == st.st_mtime_ns:
            if sf["missing_since"]:
                self.con.execute("UPDATE source_file SET missing_since=NULL WHERE id=?", (sf["id"],))
            res.unchanged += 1
            return
        pend = self.con.execute("SELECT * FROM pending_file WHERE root_id=? AND rel_path=?", (root_id, rel)).fetchone()
        same_as_pending = pend is not None and pend["size"] == st.st_size and pend["mtime_ns"] == st.st_mtime_ns
        mtime = datetime.fromtimestamp(st.st_mtime_ns / 1e9, timezone.utc)
        old_enough = now - mtime >= self.stability
        if pend is not None and not same_as_pending:
            self._set_pending(root_id, rel, st, now)
            return
        if not old_enough:
            if pend is None:
                self._set_pending(root_id, rel, st, now)
            return

        integrity, detail = "ok", None
        if path.suffix.lower() in IMAGE_EXT:
            info = read_image_header(path)
            if info.format is None:
                integrity, detail = "unsupported", info.error
            elif info.error:
                if "partial" in info.error and (pend is None or now - _parse_iso(pend["unchanged_since"]) < self.stability):
                    # Possibly still being copied: wait one full stability interval before calling it corrupt.
                    if pend is None:
                        self._set_pending(root_id, rel, st, now)
                    return
                integrity, detail = "corrupt", info.error
        else:
            integrity, detail = "unverified", "video integrity is checked by the decoder"

        self._ingest(root_id, rel, path, st, now, integrity, detail, sf, res)
        self.con.execute("DELETE FROM pending_file WHERE root_id=? AND rel_path=?", (root_id, rel))

    def _set_pending(self, root_id: int, rel: str, st: os.stat_result, now: datetime) -> None:
        self.con.execute(
            "INSERT INTO pending_file(root_id,rel_path,size,mtime_ns,unchanged_since) VALUES (?,?,?,?,?) "
            "ON CONFLICT(root_id,rel_path) DO UPDATE SET size=excluded.size, mtime_ns=excluded.mtime_ns, "
            "unchanged_since=excluded.unchanged_since",
            (root_id, rel, st.st_size, st.st_mtime_ns, _iso(now)),
        )

    def _ingest(self, root_id, rel, path, st, now, integrity, detail, sf, res) -> None:
        sha = _sha256(path)
        if path.stat().st_mtime_ns != st.st_mtime_ns or path.stat().st_size != st.st_size:
            self._set_pending(root_id, rel, path.stat(), now)  # changed while hashing
            return
        now_s = _iso(now)
        self.con.execute("BEGIN")
        try:
            if sf is None:
                sf_id = self.con.execute(
                    "INSERT INTO source_file(root_id,rel_path,first_seen) VALUES (?,?,?)", (root_id, rel, now_s)
                ).lastrowid
                prev = None
            else:
                sf_id = sf["id"]
                prev = self.con.execute(
                    "SELECT * FROM source_version WHERE source_file_id=? AND superseded_by IS NULL", (sf_id,)
                ).fetchone()
            self.con.execute("UPDATE source_file SET last_size=?, last_mtime_ns=?, missing_since=NULL WHERE id=?",
                             (st.st_size, st.st_mtime_ns, sf_id))
            if prev is not None and prev["sha256"] == sha:
                self.con.execute("COMMIT")  # touched, content unchanged: no new version
                res.unchanged += 1
                return
            try:
                ct = parse_capture_time(path.name)
                cap = (_iso(ct.utc), ct.precision.value, ct.pattern, json.dumps(ct.fields), "ok", None)
            except MalformedName as exc:
                cap = (None, None, None, None, "malformed", exc.reason)
            rename = None
            if prev is None:
                r = self.con.execute(
                    "SELECT v.id FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                    "WHERE v.sha256=? AND f.missing_since IS NOT NULL AND v.superseded_by IS NULL LIMIT 1", (sha,)
                ).fetchone()
                rename = r["id"] if r else None
            vid = self.con.execute(
                "INSERT INTO source_version(source_file_id,size,mtime_ns,sha256,capture_utc,capture_precision,"
                "name_pattern,name_fields,parse_status,parse_detail,integrity,integrity_detail,first_seen,"
                "possible_rename_of) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sf_id, st.st_size, st.st_mtime_ns, sha, *cap, integrity, detail, now_s, rename),
            ).lastrowid
            if prev is not None:
                self.con.execute("UPDATE source_version SET superseded_by=? WHERE id=?", (vid, prev["id"]))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        if cap[4] == "malformed":
            res.malformed.append(rel)
        elif integrity in ("corrupt", "unsupported"):
            res.corrupt.append(rel)
        else:
            res.new_versions.append(vid)

    def _mark_missing(self, root_id: int, seen: set[str], now_s: str, res: ScanResult) -> None:
        rows = self.con.execute(
            "SELECT id, rel_path FROM source_file WHERE root_id=? AND missing_since IS NULL", (root_id,)
        ).fetchall()
        gone = [r for r in rows if r["rel_path"] not in seen]
        for r in gone:
            # Only a flag: versions, results and corrections stay intact.
            self.con.execute("UPDATE source_file SET missing_since=? WHERE id=?", (now_s, r["id"]))
            res.newly_missing.append(r["rel_path"])


def cycle(mission: Mission, now: datetime | None = None, stability_seconds: int = DEFAULT_STABILITY_SECONDS) -> tuple[ScanResult, int | None]:
    """One scan plus, when there are new eligible files, a processing run.

    One-time missions get exactly one run (the initial snapshot). Continuous
    missions get a new batch run for each scan that finds new files.
    """
    cfg = mission.config()
    has_runs = mission.con.execute("SELECT COUNT(*) FROM run").fetchone()[0] > 0
    if cfg.processing == "one_time" and has_runs:
        return ScanResult(scan_id=-1, status="skipped_one_time_complete"), None
    res = Ingestor(mission, stability_seconds).scan(now)
    if cfg.processing == "one_time" and res.waiting_for_stability:
        return res, None  # wait until the snapshot is complete
    # Queue every eligible current version not yet in any run, not just this scan's finds,
    # so files ingested by an interrupted or deferred cycle are never left behind.
    parse_ok = "v.parse_status IN ('ok','malformed')" if mission.input_type == "video" else "v.parse_status='ok'"
    todo = [r[0] for r in mission.con.execute(
        f"SELECT v.id FROM source_version v WHERE v.superseded_by IS NULL AND {parse_ok} "
        "AND v.integrity IN ('ok','unverified') AND NOT EXISTS "
        "(SELECT 1 FROM run_item i WHERE i.source_version_id=v.id) ORDER BY v.id")]
    run_id = mission.create_run("initial" if not has_runs else "continuous_batch", todo)
    if run_id is not None:
        mission.notify("info", "new_data", f"{len(todo)} new file(s) queued in run {run_id}.", run_id=run_id)
    return res, run_id


class Monitor:
    """Background monitoring for a continuous mission, independent of any browser.

    The enabled flag is part of the mission configuration, so `resume_enabled`
    restarts monitoring after an application or workstation restart.
    """

    def __init__(self, mission: Mission, interval_seconds: float = 60, stability_seconds: int = DEFAULT_STABILITY_SECONDS,
                 on_cycle: Callable[[ScanResult, int | None], None] | None = None):
        self.mission = mission
        self.interval = interval_seconds
        self.stability = stability_seconds
        self.on_cycle = on_cycle
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    def nudge(self) -> None:
        """Called on a filesystem notification: scan soon. Repeated calls collapse into one scan."""
        self._wake.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name=f"monitor-{self.mission.id}", daemon=True)
        self._thread.start()
        self._start_watcher()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._observer is not None:
            self._observer.stop()
        if self._thread:
            self._thread.join(timeout=10)

    def _start_watcher(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler  # type: ignore
            from watchdog.observers import Observer  # type: ignore
        except ImportError:
            return  # polling only
        mon = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):  # noqa: D401
                mon.nudge()

        obs = Observer()
        for root in self.mission.con.execute("SELECT path, recursive FROM source_root"):
            if Path(root["path"]).is_dir():
                obs.schedule(Handler(), root["path"], recursive=bool(root["recursive"]))
        obs.daemon = True
        obs.start()
        self._observer = obs

    def _loop(self) -> None:
        # The mission's SQLite connection belongs to the creating thread; use our own.
        m = Mission(type(self.mission.root)(self.mission.root.path), self.mission.id)
        while not self._stop.is_set():
            if m.config().monitoring_enabled:
                try:
                    res, run_id = cycle(m, stability_seconds=self.stability)
                    if self.on_cycle:
                        self.on_cycle(res, run_id)
                except Exception as exc:  # keep monitoring alive; surface the failure
                    m.notify("error", "scan_failed", f"Scan failed: {exc}")
            self._wake.wait(self.interval)
            self._wake.clear()


def resume_enabled(root, **kw) -> list[Monitor]:
    """Start monitors for every continuous mission whose monitoring flag is on."""
    monitors = []
    for row in root.missions():
        m = root.open_mission(row["id"])
        cfg = m.config()
        if cfg.processing == "continuous" and cfg.monitoring_enabled:
            mon = Monitor(m, **kw)
            mon.start()
            monitors.append(mon)
    return monitors
