"""Run processing for still-image missions (general traffic mode).

For each queued run item: read the image, find its camera view, detect
vehicles, decide region membership, and store immutable detection rows.
Items are committed one at a time, so an interrupted run resumes where it
stopped and never duplicates detections. A failing image is recorded with its
error and the run continues.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .detect import Detector
from .night import measure_condition
from .night_service import NightService
from .regions import RULES, RegionSet
from .store import Mission, utcnow
from .views import ViewService, read_image

MAX_ATTEMPTS = 3


class RunCancelled(Exception):
    pass


class RunControl:
    """Pause, resume and cancel from another thread. Checked between images, never mid-write."""

    def __init__(self):
        import threading
        self._go = threading.Event()
        self._go.set()
        self.cancelled = False
        self.stage = "idle"

    def pause(self) -> None:
        self._go.clear()

    def resume(self) -> None:
        self._go.set()

    def cancel(self) -> None:
        self.cancelled = True
        self._go.set()

    @property
    def paused(self) -> bool:
        return not self._go.is_set()

    def checkpoint(self) -> None:
        self._go.wait()
        if self.cancelled:
            raise RunCancelled()


@dataclass
class RunReport:
    run_id: int
    done: int = 0
    failed: int = 0
    skipped: int = 0
    detections: int = 0
    errors: list[str] = field(default_factory=list)
    night: dict = field(default_factory=dict)


def register_model(con, describe: dict, card: dict) -> int:
    m = describe["model"]
    row = con.execute("SELECT id FROM model_version WHERE name=? AND version=? AND sha256=?",
                      (m["name"], m["version"], m["sha256"])).fetchone()
    if row:
        return row[0]
    return con.execute(
        "INSERT INTO model_version(name,version,sha256,task,license,card,first_used) VALUES (?,?,?,?,?,?,?)",
        (m["name"], m["version"], m["sha256"], card.get("task", "vehicle_detection"), card.get("license", "unknown"),
         json.dumps(card, ensure_ascii=False), utcnow())).lastrowid


class Processor:
    def __init__(self, mission: Mission, detector: Detector, card: dict | None = None,
                 region_rule: str = "bottom_center", min_overlap: float = 0.5, control: RunControl | None = None):
        if region_rule not in RULES:
            raise ValueError(f"region rule must be one of {RULES}")
        self.m = mission
        self.con = mission.con
        self.detector = detector
        self.card = card if card is not None else getattr(getattr(detector, "pack", None), "card", {})
        self.rule = region_rule
        self.min_overlap = min_overlap
        self.views = ViewService(mission)
        self.night = NightService(mission)
        self.control = control or RunControl()

    def pending_runs(self) -> list[int]:
        # Includes runs interrupted after their last image but before night analysis and completion.
        return [r[0] for r in self.con.execute(
            "SELECT id FROM run WHERE state IN ('queued','running') OR id IN "
            "(SELECT run_id FROM run_item WHERE processing_state IN ('queued','running')) ORDER BY id")]

    def process_all(self, progress: Callable[[int, int], None] | None = None) -> list[RunReport]:
        return [self.process_run(r, progress) for r in self.pending_runs()]

    def process_run(self, run_id: int, progress: Callable[[int, int], None] | None = None) -> RunReport:
        rep = RunReport(run_id)
        desc = self.detector.describe()
        model_id = register_model(self.con, desc, self.card)
        run = self.con.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
        models = json.loads(run["model_versions"] or "{}")
        models["vehicle_detection"] = desc["model"]
        pre = json.loads(run["preprocessing"] or "{}")
        pre["vehicle_detection"] = desc["preprocessing"]
        pre["night"] = self.night.describe()
        pre["region_rule"] = {"rule": self.rule, "min_overlap": self.min_overlap if self.rule == "overlap" else None}
        self.con.execute("UPDATE run SET state='running', started_at=COALESCE(started_at,?), model_versions=?, preprocessing=? "
                         "WHERE id=?", (utcnow(), json.dumps(models), json.dumps(pre), run_id))

        # Group new images into views first so regions can be applied.
        self.control.stage = "grouping views"
        self.views.assign_or_propose()

        items = self.con.execute(
            "SELECT i.*, f.rel_path FROM run_item i JOIN source_version v ON v.id=i.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id WHERE i.run_id=? AND i.processing_state IN ('queued','running') "
            "ORDER BY i.id", (run_id,)).fetchall()
        total = len(items)
        region_cache: dict[int, RegionSet] = {}
        self.control.stage = "detecting vehicles"
        for n, it in enumerate(items, 1):
            try:
                self.control.checkpoint()
            except RunCancelled:
                self.con.execute("UPDATE run_item SET processing_state='cancelled' WHERE run_id=? AND "
                                 "processing_state IN ('queued','running')", (run_id,))
                self.con.execute("UPDATE run SET state='cancelled', finished_at=? WHERE id=?", (utcnow(), run_id))
                self.m.notify("warning", "run_cancelled", f"Run {run_id} was cancelled after {rep.done} image(s).",
                              run_id=run_id)
                self.control.stage = "cancelled"
                return rep
            if it["attempts"] >= MAX_ATTEMPTS:
                self.con.execute("UPDATE run_item SET processing_state='failed', error_code='too_many_attempts' WHERE id=?",
                                 (it["id"],))
                rep.failed += 1
                continue
            self.con.execute("UPDATE run_item SET processing_state='running', stage='detect', attempts=attempts+1 WHERE id=?",
                             (it["id"],))
            try:
                img = read_image(self.views._path(it["source_version_id"]))
                dets = self.detector.detect(img)
            except Exception as exc:  # one bad image must not stop the run
                self.con.execute("UPDATE run_item SET processing_state='failed', error_code=?, error_detail=? WHERE id=?",
                                 (type(exc).__name__, str(exc)[:500], it["id"]))
                self.m.notify("warning", "processing_failed", f"Could not process {it['rel_path']}: {exc}",
                              run_id=run_id, source_version_id=it["source_version_id"])
                rep.failed += 1
                rep.errors.append(f"{it['rel_path']}: {exc}")
                continue
            view_id = self.views.current_view(it["source_version_id"])
            regions = None
            if view_id is not None:
                self.views.check_drift(it["source_version_id"])
                if self.views.regions_applicable(view_id):
                    regions = region_cache.get(view_id) or RegionSet.load(self.con, view_id)
                    region_cache[view_id] = regions
            now = utcnow()
            self.con.execute("BEGIN")
            # Detections and the 'done' state commit together: a crash leaves neither, and a done item is never redone.
            for d in dets:
                inside = None
                if regions is not None:
                    inside = int(regions.contains_box(d.box, self.rule, self.min_overlap))
                elif view_id is None:
                    inside = int(RegionSet([]).contains_box(d.box, self.rule, self.min_overlap))  # unassigned: whole frame
                self.con.execute(
                    "INSERT INTO detection(run_item_id,source_version_id,view_id,x1,y1,x2,y2,cls,score,class_scores,"
                    "in_region,region_rule,model_version_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (it["id"], it["source_version_id"], view_id, *(float(v) for v in d.box), d.cls, float(d.score), json.dumps(d.class_scores),
                     inside, self.rule if inside is not None else None, model_id, now))
            self.night.record_item(it["id"], it["source_version_id"], img, measure_condition(img))
            self.con.execute("UPDATE run_item SET processing_state='done', stage=NULL, error_code=NULL, error_detail=NULL "
                             "WHERE id=?", (it["id"],))
            self.con.execute("COMMIT")
            rep.done += 1
            rep.detections += len(dets)
            if progress:
                progress(n, total)
        # Night: backgrounds need many images, so lights are grouped once the images are in.
        self.control.stage = "night analysis"
        rep.night = self.night.refresh()
        self.control.stage = "finished"
        left = self.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=? AND processing_state IN ('queued','running')",
                                (run_id,)).fetchone()[0]
        if not left:
            failed = self.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=? AND processing_state='failed'",
                                      (run_id,)).fetchone()[0]
            self.con.execute("UPDATE run SET state=?, finished_at=? WHERE id=?",
                             ("completed" if not failed else "completed_with_errors", utcnow(), run_id))
            self.m.notify("info" if not failed else "warning", "run_finished",
                          f"Run {run_id} finished: {rep.done} processed, {failed} failed.", run_id=run_id)
        return rep
