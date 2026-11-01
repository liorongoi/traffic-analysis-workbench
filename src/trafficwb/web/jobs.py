"""Background work for the web app: scan-and-process jobs and continuous monitoring (PRG-01, MIS-07).

Work runs in server threads, so closing the browser never stops it. Each job opens its
own database connections. Pause, resume and cancel act between images (between training steps for training).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..ingest import Monitor, cycle
from ..process import Processor, RunControl
from ..store import DataRoot

_resource_lock = threading.Lock()
_training_lock = threading.Lock()      # one training at a time on this computer: it uses the whole GPU or CPU
_resource_process = None


def _resource_usage() -> dict | None:
    """Small, non-blocking resource sample for the operator's progress screen."""
    global _resource_process
    try:
        import psutil
        with _resource_lock:
            if _resource_process is None:
                _resource_process = psutil.Process()
            proc = _resource_process
            cpu = proc.cpu_percent(interval=None)
            rss = proc.memory_info().rss
            memory = psutil.virtual_memory()
        return {"process_cpu_percent": round(cpu, 1), "process_rss_bytes": int(rss),
                "system_memory_available_bytes": int(memory.available), "system_memory_total_bytes": int(memory.total)}
    except Exception:
        # Progress reporting must never turn a missing process metric into a failed job request.
        return None


def scan_summary(m, res, queued: int) -> dict:
    """What a scan found, and in plain words why files were not queued, for the Processing card."""
    q = lambda sql: m.con.execute(sql).fetchone()[0]
    cur = "FROM source_version v JOIN source_file f ON f.id=v.source_file_id WHERE v.superseded_by IS NULL"
    no_time = 0 if m.input_type == "video" else q(f"SELECT COUNT(*) {cur} AND v.parse_status='malformed'")
    examples = [r[0] for r in m.con.execute(
        f"SELECT f.rel_path {cur} AND v.parse_status='malformed' ORDER BY f.rel_path LIMIT 5")] if no_time else []
    damaged = q(f"SELECT COUNT(*) {cur} AND v.integrity IN ('corrupt','unsupported')")
    ignored = sum((res.ignored_by_extension or {}).values())
    notes = []
    if res.status == "skipped_one_time_complete":
        notes.append("This is a one-time mission and its run has already happened, so the folder was not scanned "
                      "again. Create a new mission for images added later.")
    if res.status == "source_unavailable":
        notes.append("The source folder could not be reached.")
    if no_time:
        notes.append(f"{no_time} file(s) were skipped because their names hold no capture time this app can read "
                     f"(for example {', '.join(examples[:3])}).")
    if res.waiting_for_stability:
        notes.append(f"{res.waiting_for_stability} file(s) changed less than a minute ago and are waiting until they "
                     "stop changing. Scan again in a minute.")
    if damaged:
        notes.append(f"{damaged} file(s) are damaged or not a supported image.")
    if ignored:
        notes.append(f"{ignored} file(s) are not {'videos' if m.input_type == 'video' else 'images'} and were ignored.")
    if not queued and not notes:
        notes.append("No new files: everything in the source folder has already been processed.")
    return {"queued": queued, "no_capture_time": no_time, "no_capture_time_examples": examples,
            "waiting_for_stability": res.waiting_for_stability, "damaged": damaged, "ignored": ignored,
            "status": res.status, "notes": notes}


@dataclass
class Job:
    mission_id: str
    kind: str                         # scan_and_process | process | search | evaluate | train
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
        if self.kind == "train" and self.rate_start and self.rate_done >= 3 and self.state in ("running", "paused"):
            per = (now - self.rate_start) / self.rate_done          # seconds per training step, measured here
            left = max(0, self.total - self.done)
            # validation at the end of every epoch, and the export at the end, add a little
            eta = round(per * left * 1.1 + 60) if left else None
        elif self.rate_start and self.rate_done >= 3 and self.state in ("running", "paused") and self.total:
            per = (now - self.rate_start) / self.rate_done
            eta = round(per * max(0, self.total - self.done))
        state = "paused" if self.state == "running" and self.control.paused else self.state
        return {"kind": self.kind, "state": state, "stage": self.control.stage, "done": self.done, "total": self.total,
                "elapsed_s": round(now - self.started), "eta_s": eta, "error": self.error, "result": self.result,
                "eta_basis": "measured on this run" if eta is not None else
                ("not enough training steps yet" if self.kind == "train" else "not enough images yet"),
                "resource_usage": _resource_usage()}


@dataclass
class AreaTask:
    """SAM 3.1 looking for named areas (road, parking lot) on one image of a camera view."""
    id: str
    mission_id: str
    view_id: int
    version_id: int
    prompts: list[str]
    kind: str
    threshold: float
    started: float = field(default_factory=time.time)
    finished: float | None = None
    state: str = "running"            # running | finished | failed
    stage: str = "starting"
    error: str | None = None
    proposals: list = field(default_factory=list)
    size: list = field(default_factory=list)

    def snapshot(self) -> dict:
        return {"id": self.id, "view_id": self.view_id, "version_id": self.version_id, "prompts": self.prompts,
                "kind": self.kind, "state": self.state, "stage": self.stage, "error": self.error,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1),
                "proposals": self.proposals, "size": self.size}


class JobManager:
    def __init__(self, data_root: Path, detector_factory: Callable[[], tuple[object, dict]],
                 monitor_interval: float = 60, stability_seconds: int = 30,
                 plate_factory: Callable[[], tuple] | None = None,
                 model_loader: Callable[[str, str], object] | None = None,
                 sam_factory: Callable[[], object] | None = None):
        self.data_root = Path(data_root)
        self.sam_factory = sam_factory          # the SAM 3.1 engine; tests pass a small stand-in
        self._sam_detector = None
        self.area_tasks: dict[str, AreaTask] = {}
        self.model_loader = model_loader
        self.detector_factory = detector_factory
        self.plate_factory = plate_factory
        self._detector = None
        self._plates = None
        self._lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        # Training runs beside a mission's processing (it can take a night on a CPU), so it has its own slot.
        self.training: dict[str, Job] = {}
        self.monitors: dict[str, Monitor] = {}
        self.monitor_interval = monitor_interval
        self.stability = stability_seconds

    def detector(self):
        with self._lock:
            if self._detector is None:
                self._detector = self.detector_factory()
            return self._detector

    def sam_engine(self):
        if self.sam_factory:
            return self.sam_factory()
        from ..sam3 import engine
        return engine()

    def detector_for(self, m):
        """The detector a mission chose: SAM 3.1 for still-image missions that asked for it, else the standard one."""
        choice = getattr(m.config(), "detector", "standard") if m.input_type == "images" else "standard"
        if choice in ("sam3", "both"):
            from ..sam3 import Sam3Detector, vehicle_prompts
            eng = self.sam_engine()
            prompts = vehicle_prompts(getattr(m.config(), "sam3_words", None))
            with self._lock:
                cur = self._sam_detector
                if cur is None or cur[0].engine is not eng or cur[0].prompts != prompts:
                    self._sam_detector = (Sam3Detector(eng, prompts=prompts), eng.pack.card)
                sam = self._sam_detector
            if choice == "sam3":
                return sam
            from ..combined import CombinedDetector
            std, std_card = self.detector()
            both = CombinedDetector(std, std_card, sam[0], sam[1])
            return both, both.card()
        return self.detector()

    # ---------- scan areas found by SAM 3.1 ----------
    def find_areas(self, mission_id: str, view_id: int, version_id: int, prompts: list[str], kind: str,
                   threshold: float) -> "AreaTask":
        import uuid
        task = AreaTask(uuid.uuid4().hex[:12], mission_id, view_id, version_id, prompts, kind, threshold)
        with self._lock:
            for k in [k for k, t in self.area_tasks.items() if t.finished and time.time() - t.finished > 3600]:
                del self.area_tasks[k]
            self.area_tasks[task.id] = task
        threading.Thread(target=self._areas, args=(task,), name=f"areas-{task.id}", daemon=True).start()
        return task

    def _areas(self, t: "AreaTask") -> None:
        from dataclasses import asdict as _asdict
        from ..sam3 import find_areas
        from ..views import ViewService
        try:
            t.stage = "loading SAM 3.1"
            eng = self.sam_engine()
            eng.load()
            t.stage = "reading the image"
            img = ViewService(DataRoot(self.data_root).open_mission(t.mission_id)).image(t.version_id)
            t.stage = "looking for " + ", ".join(f"\u201c{p}\u201d" for p in t.prompts)
            t.proposals = [_asdict(a) for a in find_areas(eng, img, t.prompts, t.kind, t.threshold)]
            t.size = [int(img.shape[1]), int(img.shape[0])]
            t.state = "finished"
            t.stage = "done"
        except Exception as exc:
            t.state = "failed"
            t.error = str(exc) or type(exc).__name__
        finally:
            t.finished = time.time()

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
            slot = self.training if kind == "train" else self.jobs
            old = slot.get(mission_id)
            if old is not None and old.state in ("running", "paused"):
                return old
            job = Job(mission_id, kind, params=params)
            job.thread = threading.Thread(target=self._run, args=(job,), name=f"{kind}-{mission_id}", daemon=True)
            slot[mission_id] = job
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
            if job.kind == "train":
                self._train(job, m)
                return
            if job.kind == "compare_backgrounds":
                self._compare(job, m)
                return
            if job.kind == "scan_and_process":
                job.control.stage = "scanning source folder"
                res, _ = cycle(m, stability_seconds=self.stability)
            job.total = m.con.execute(
                "SELECT COUNT(*) FROM run_item WHERE processing_state IN ('queued','running')").fetchone()[0]
            if job.kind == "scan_and_process":
                job.result["scan"] = scan_summary(m, res, job.total)
            job.control.stage = "loading model"
            det, card = self.detector_for(m)
            if det.__class__.__name__ == "Sam3Detector":
                job.control.stage = "loading SAM 3.1 (about a minute the first time)"
                det.engine.load()

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
                proc = Processor(m, det, card, control=job.control, plate_models=self.plate_models,
                                 sam_engine=self._sam_if_available)
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
                    "error", "job_failed", f"{'Training' if job.kind == 'train' else 'Processing'} stopped: {exc}",
                    trace=traceback.format_exc()[-2000:])
            except Exception:
                pass
        finally:
            job.finished = time.time()

    def _sam_if_available(self):
        """SAM 3.1 when it is installed and this computer can hold it; otherwise an error (callers carry on)."""
        if not self.sam_factory:
            from ..sam3 import available
            st = available()
            if not st.get("installed"):
                raise RuntimeError(st.get("reason") or "SAM 3.1 is not installed")
            if st.get("memory_note"):
                raise RuntimeError(st["memory_note"])
        return self.sam_engine()

    def _compare(self, job: Job, m) -> None:
        from ..process import compare_with_backgrounds

        def progress(n, total):
            job.total = total
            job.done = n
            job.rate_done += 1
            if job.rate_start is None:
                job.rate_start = time.time()
                job.rate_done = 0

        job.control.stage = "comparing with the empty backgrounds"
        job.result = compare_with_backgrounds(m, self._sam_if_available, job.control, progress)
        job.state = "cancelled" if job.control.cancelled else "finished"

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

    def training_job(self, mission_id: str) -> Job | None:
        return self.training.get(mission_id)

    def training_busy(self) -> Job | None:
        for j in self.training.values():
            if j.kind == "train" and j.state in ("running", "paused"):
                return j
        return None

    def _train(self, job: Job, m) -> None:
        """Export the dataset, fine-tune on this computer, add the new model to the installed ones and evaluate it.

        Nothing is promoted: the operator compares the evaluation with the model in use and promotes it if better.
        """
        import shutil
        import tempfile
        from .. import models
        from ..learning import LearningService
        from ..model_import import import_pack
        p = job.params
        if not _training_lock.acquire(blocking=False):
            raise RuntimeError("Another training is running on this computer. Wait for it to finish.")
        work = None
        try:
            job.control.stage = "loading PyTorch"
            from .. import train as tr
            ls = LearningService(m)
            work = Path(tempfile.mkdtemp(prefix="lanepulse-train-", dir=p.get("work_dir")))
            job.control.stage = "copying the dataset images"
            ds_dir = ls.export_dataset(p["dataset_id"], out_dir=work)
            log_lines: list[str] = []

            def log(*a):
                line = " ".join(str(x) for x in a)
                log_lines.append(line)
                del log_lines[:-40]
                job.result["log"] = list(log_lines)
                if line.startswith("Device: "):
                    job.result["device"] = line[len("Device: "):]

            def progress(step, steps):
                job.total = steps
                job.done = step
                job.rate_done += 1
                if job.rate_start is None:                    # the first step includes warm-up; not timed
                    job.rate_start = time.time()
                    job.rate_done = 0

            def on_epoch(ep, epochs, row):
                job.result["epoch"] = ep
                job.result["epochs"] = epochs
                job.result.setdefault("history", []).append(row)
                v = row.get("val_loss")
                job.control.stage = f"training: epoch {ep} of {epochs} done" + (f", validation loss {v:.3f}" if v is not None else "")

            job.control.stage = "training: first epoch"
            # Image loading in separate processes on a Linux GPU server (started fresh, never forked from this
            # multi-threaded server); elsewhere, and on a CPU, it is read in this thread.
            workers = min(4, os.cpu_count() or 1) if sys.platform.startswith("linux") and p.get("device") != "cpu" else 0
            z = tr.train(ds_dir, work / "out", init=p.get("init") or "coco", epochs=int(p["epochs"]), device=p.get("device") or "auto",
                         workers=workers, log=log, progress=progress, control=job.control, on_epoch=on_epoch, download=False,
                         mp_context="spawn")
            job.control.stage = "adding the new model"
            r = import_pack(z, p.get("models_root") or models.models_root())
            job.result["model"] = {"name": r["name"], "version": r["version"]}
            ds = ls.dataset(p["dataset_id"])
            test_images = ((ds.get("stats") or {}).get("splits") or {}).get("test", {}).get("images", 0)
            if test_images and p.get("evaluate", True):
                job.control.stage = "evaluating the new model on the test images"
                job.done, job.total, job.rate_start, job.rate_done = 0, 0, None, 0

                def eprogress(n, total):
                    job.total = total
                    job.done = n

                try:
                    det = self.load_model(r["name"], r["version"])
                    ev = ls.evaluate(p["dataset_id"], det, p["author"], split="test", control=job.control, progress=eprogress)
                    job.result["evaluation"] = ev["id"]
                except Exception as exc:              # the model is installed; evaluating it can be redone by hand
                    from ..process import RunCancelled
                    if isinstance(exc, RunCancelled):
                        job.state = "cancelled"
                        return
                    job.result["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            job.state = "finished"
        except Exception as exc:
            try:
                from ..train import TrainCancelled
            except Exception:                         # PyTorch missing: the message below says so
                TrainCancelled = ()
            if TrainCancelled and isinstance(exc, TrainCancelled):
                job.state = "cancelled"
                return
            if isinstance(exc, ModuleNotFoundError) and exc.name == "torch":
                raise RuntimeError("PyTorch is not part of this installation, so it cannot train. Install the current "
                                   "LanePulse version.") from None
            raise
        finally:
            _training_lock.release()
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)

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
        for j in [*self.jobs.values(), *self.training.values()]:
            if j.state in ("running", "paused"):
                j.control.cancel()
