"""Background work for the web app: scan-and-process jobs and continuous monitoring (PRG-01, MIS-07).

Work runs in server threads, so closing the browser never stops it. Each job opens its
own database connections. Pause, resume and cancel act between images.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..ingest import Monitor, cycle
from ..process import Processor, RunControl
from ..store import DataRoot


@dataclass
class Job:
    mission_id: str
    kind: str                         # scan_and_process | process | search | evaluate
    started: float = field(default_factory=time.time)
    finished: float | None = None
    control: RunControl = field(default_factory=RunControl)
    done: int = 0
    total: int = 0
    rate_start: float | None = None   # time of the first finished image, for the ETA
    rate_done: int = 0
    state: str = "running"            # running | paused | finished | cancelled | failed
    error: str | None = None
    report: list = field(default_factory=list)
    thread: threading.Thread | None = None
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        now = self.finished or time.time()
        eta = None
        if self.rate_start and self.rate_done >= 3 and self.state in ("running", "paused") and self.total:
            per = (now - self.rate_start) / self.rate_done
            eta = round(per * max(0, self.total - self.done))
        state = "paused" if self.state == "running" and self.control.paused else self.state
        return {"kind": self.kind, "state": state, "stage": self.control.stage, "done": self.done, "total": self.total,
                "elapsed_s": round(now - self.started), "eta_s": eta, "error": self.error, "result": self.result,
                "eta_basis": "measured on this run" if eta is not None else "not enough images yet"}


class JobManager:
    def __init__(self, data_root: Path, detector_factory: Callable[[], tuple[object, dict]],
                 monitor_interval: float = 60, stability_seconds: int = 30,
                 plate_factory: Callable[[], tuple] | None = None,
                 model_loader: Callable[[str, str], object] | None = None):
        self.data_root = Path(data_root)
        self.model_loader = model_loader
        self.detector_factory = detector_factory
        self.plate_factory = plate_factory
        self._detector = None
        self._plates = None
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.monitors: dict[str, Monitor] = {}
        self.monitor_interval = monitor_interval
        self.stability = stability_seconds

    def detector(self):
        with self._lock:
            if self._detector is None:
                self._detector = self.detector_factory()
            return self._detector

    def reset_detector(self) -> None:
        """After a promotion or rollback, the next job loads the new production model."""
        with self._lock:
            self._detector = None

    def load_model(self, name: str, version: str):
        if self.model_loader:
            return self.model_loader(name, version)
        from .. import models
        from ..detect import YoloxDetector
        return YoloxDetector(models.find(name, version))

    def plate_models(self):
        with self._lock:
            if self._plates is None:
                from ..plates import default_plate_models
                self._plates = (self.plate_factory or default_plate_models)()
            return self._plates

    def current(self, mission_id: str) -> Job | None:
        return self.jobs.get(mission_id)

    def busy(self, mission_id: str) -> bool:
        j = self.jobs.get(mission_id)
        return j is not None and j.state in ("running", "paused")

    def start(self, mission_id: str, kind: str = "scan_and_process", **params) -> Job:
        with self._lock:
            if self.busy(mission_id):
                return self.jobs[mission_id]
            job = Job(mission_id, kind, params=params)
            job.thread = threading.Thread(target=self._run, args=(job,), name=f"job-{mission_id}", daemon=True)
            self.jobs[mission_id] = job
        job.thread.start()
        return job

    def _run(self, job: Job) -> None:
        try:
            root = DataRoot(self.data_root)
            m = root.open_mission(job.mission_id)
            if job.kind == "search":
                self._search(job, m)
                return
            if job.kind == "evaluate":
                self._evaluate(job, m)
                return
            if job.kind == "scan_and_process":
                job.control.stage = "scanning source folder"
                cycle(m, stability_seconds=self.stability)
            job.total = m.con.execute(
                "SELECT COUNT(*) FROM run_item WHERE processing_state IN ('queued','running')").fetchone()[0]
            job.control.stage = "loading model"
            det, card = self.detector()

            def progress(n, total):
                job.done += 1
                job.rate_done += 1
                if job.rate_start is None:
                    job.rate_start = time.time()
                    job.rate_done = 0

            if m.input_type == "video":
                from ..video_process import VideoProcessor
                proc = VideoProcessor(m, det, card, control=job.control, plate_models=self.plate_models)
            else:
                proc = Processor(m, det, card, control=job.control, plate_models=self.plate_models)
            job.report = [r.__dict__ for r in proc.process_all(progress)]
            if m.input_type == "images" and m.config().mode == "plate_recognition" and not job.control.cancelled:
                job.control.stage = "reading plates"
                job.report.append({"plates": proc._plates()})   # plates left from an earlier cancelled run
            job.state = "cancelled" if job.control.cancelled else "finished"
        except Exception as exc:  # surface, never crash the server
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            try:
                DataRoot(self.data_root).open_mission(job.mission_id).notify(
                    "error", "job_failed", f"Processing stopped: {exc}", trace=traceback.format_exc()[-2000:])
            except Exception:
                pass
        finally:
            job.finished = time.time()

    def _search(self, job: Job, m) -> None:
        from ..search import SearchService

        def progress(n, total):
            job.total = total
            job.done = n
            job.rate_done += 1
            if job.rate_start is None:
                job.rate_start = time.time()
                job.rate_done = 0

        run_id = SearchService(m).run(job.params["request_id"], job.params["author"], control=job.control,
                                      progress=progress)
        job.result = {"search_run": run_id}
        job.state = "cancelled" if job.control.cancelled else "finished"

    def _evaluate(self, job: Job, m) -> None:
        from ..learning import LearningService
        p = job.params
        job.control.stage = "loading model"
        det = self.load_model(p["name"], p["version"]) if p.get("name") else self.detector()[0]
        job.control.stage = "evaluating"

        def progress(n, total):
            job.total = total
            job.done = n
            job.rate_done += 1
            if job.rate_start is None:
                job.rate_start = time.time()
                job.rate_done = 0

        try:
            ev = LearningService(m).evaluate(p["dataset_id"], det, p["author"], split=p.get("split", "test"),
                                             iou=p.get("iou", 0.5), control=job.control, progress=progress)
            job.result = {"evaluation": ev["id"]}
            job.state = "finished"
        except Exception as exc:
            from ..process import RunCancelled
            if not isinstance(exc, RunCancelled):
                raise
            job.state = "cancelled"

    # ---------- monitoring ----------
    def set_monitoring(self, mission_id: str, enabled: bool) -> None:
        if enabled and mission_id not in self.monitors:
            m = DataRoot(self.data_root).open_mission(mission_id)

            def on_cycle(res, run_id, mid=mission_id):
                if run_id is not None and not self.busy(mid):
                    self.start(mid, kind="process")

            mon = Monitor(m, interval_seconds=self.monitor_interval, stability_seconds=self.stability, on_cycle=on_cycle)
            mon.start()
            self.monitors[mission_id] = mon
        elif not enabled and mission_id in self.monitors:
            self.monitors.pop(mission_id).stop()

    def resume_monitoring(self) -> list[str]:
        """After a restart, monitoring resumes for every continuous mission that had it on."""
        root = DataRoot(self.data_root)
        started = []
        for row in root.missions():
            cfg = root.open_mission(row["id"]).config()
            if cfg.processing == "continuous" and cfg.monitoring_enabled:
                self.set_monitoring(row["id"], True)
                started.append(row["id"])
        return started

    def shutdown(self) -> None:
        for mid in list(self.monitors):
            self.monitors.pop(mid).stop()
        for j in self.jobs.values():
            if j.state in ("running", "paused"):
                j.control.cancel()
