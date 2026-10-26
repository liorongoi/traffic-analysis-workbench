"""Processing of video recordings: decode on a time grid, detect, track, and record events (VID-02, VID-04).

Rules:
- Work is committed in chunks of analysed frames together with a checkpoint
  (position on the time grid and the full tracker state). An interrupted run
  resumes from the last checkpoint and produces the same tracks and events as
  an uninterrupted one. Memory stays bounded: only active tracks are held.
- Tracks and their observations are machine output and never change. Each
  confirmed track becomes one event. Tentative tracks that never reach
  confirmation are kept as discarded, with the reason.
- Tracks end at scene cuts and at the end of the recording (Q14 default). No
  identity is carried across recordings.
- Event times are presentation times in seconds from the stream start.
  Absolute UTC is added only when the recording's start is known from a
  confirmed filename or operator correction, with its precision.
- Plate recognition missions read the plate of each vehicle event on a few of
  its frames after processing (see plates.combine_frame_reads).
- Not in this version: camera views and regions for video, night light
  tracking (blob, pair, body).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .process import RunCancelled, RunControl, register_model
from .store import Mission, utcnow
from .tracker import Closed, Tracker, TrackerParams
from .video import frames, is_scene_cut, probe, scene_signature
from .views import ViewService

MAX_ATTEMPTS = 3


@dataclass
class VideoSettings:
    interval_s: float = 0.2          # analysis every 0.2 s of presentation time (5 per second)
    chunk_frames: int = 50           # analysed frames per committed chunk
    tracker: TrackerParams = field(default_factory=TrackerParams)


@dataclass
class VideoReport:
    run_id: int
    done: int = 0
    failed: int = 0
    frames: int = 0
    events: int = 0
    errors: list[str] = field(default_factory=list)
    plates: dict | None = None


class VideoProcessor:
    def __init__(self, mission: Mission, detector, card: dict | None = None, settings: VideoSettings | None = None,
                 control: RunControl | None = None, plate_models: Callable[[], tuple] | None = None):
        self.m = mission
        self.plate_models = plate_models   # loaded only when a plate recognition mission needs it
        self.con = mission.con
        self.detector = detector
        self.card = card if card is not None else getattr(getattr(detector, "pack", None), "card", {})
        self.s = settings or VideoSettings()
        self.control = control or RunControl()
        self.views = ViewService(mission)

    def pending_runs(self) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT id FROM run WHERE state IN ('queued','running') OR id IN "
            "(SELECT run_id FROM run_item WHERE processing_state IN ('queued','running')) ORDER BY id")]

    def process_all(self, progress: Callable[[int, int], None] | None = None) -> list[VideoReport]:
        reps = [self.process_run(r, progress) for r in self.pending_runs()]
        if self.m.config().mode == "plate_recognition" and not self.control.cancelled:
            self.control.stage = "reading plates"
            try:
                plates = self._plates()     # also plates left from an earlier, cancelled run
            except RunCancelled:
                self.m.notify("warning", "plates_cancelled", "Plate reading was cancelled; plates not yet read are "
                              "read on the next run.")
                return reps
            if reps:
                reps[-1].plates = plates
            self.control.stage = "finished"
        return reps

    def _plates(self) -> dict:
        from .models import ModelPackError
        from .plates import PlateService, default_plate_models
        try:
            finder, reader = (self.plate_models or default_plate_models)()
        except ModelPackError as exc:
            self.m.notify("error", "plate_models_missing", f"Plates were not read: {exc}",
                          action="Copy the plate model packs from the staging machine (scripts/fetch_models.py).")
            return {"error": str(exc)}
        return PlateService(self.m, finder, reader).refresh(self.control)

    # ---------- probe ----------
    def probe_version(self, version_id: int) -> dict:
        row = self.con.execute("SELECT * FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone()
        if row:
            return dict(row)
        v = self.con.execute("SELECT capture_utc, capture_precision FROM source_version WHERE id=?", (version_id,)).fetchone()
        p = probe(self.views._path(version_id))
        self.con.execute(
            "INSERT INTO video_probe(source_version_id,container,codec,width,height,duration_s,nominal_fps,frame_count,"
            "variable_frame_rate,creation_time,start_utc,start_precision,start_source,error,probed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version_id, p.container, p.codec, p.width, p.height, p.duration_s, p.nominal_fps, p.frame_count,
             None if p.variable_frame_rate is None else int(p.variable_frame_rate), p.creation_time,
             v["capture_utc"], v["capture_precision"] if v["capture_utc"] else None,
             "filename" if v["capture_utc"] else "none", p.error, utcnow()))
        return dict(self.con.execute("SELECT * FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone())

    # ---------- run ----------
    def process_run(self, run_id: int, progress: Callable[[int, int], None] | None = None) -> VideoReport:
        rep = VideoReport(run_id)
        desc = self.detector.describe()
        model_id = register_model(self.con, desc, self.card)
        run = self.con.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
        models = json.loads(run["model_versions"] or "{}")
        models["vehicle_detection"] = desc["model"]
        pre = json.loads(run["preprocessing"] or "{}")
        pre["vehicle_detection"] = desc["preprocessing"]
        pre["video"] = {"interval_s": self.s.interval_s, "chunk_frames": self.s.chunk_frames,
                        "tracker": asdict(self.s.tracker), "time_base": "presentation timestamps",
                        "regions": "not applied to video in this version"}
        self.con.execute("UPDATE run SET state='running', started_at=COALESCE(started_at,?), model_versions=?, "
                         "preprocessing=? WHERE id=?", (utcnow(), json.dumps(models), json.dumps(pre), run_id))
        self._model_id = model_id
        items = self.con.execute(
            "SELECT i.*, f.rel_path FROM run_item i JOIN source_version v ON v.id=i.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id WHERE i.run_id=? AND i.processing_state IN ('queued','running') "
            "ORDER BY i.id", (run_id,)).fetchall()
        total = len(items)
        self.control.stage = "analysing video"
        for n, it in enumerate(items, 1):
            if it["attempts"] >= MAX_ATTEMPTS:
                self.con.execute("UPDATE run_item SET processing_state='failed', error_code='too_many_attempts' WHERE id=?",
                                 (it["id"],))
                rep.failed += 1
                continue
            self.con.execute("UPDATE run_item SET processing_state='running', stage='video', attempts=attempts+1 WHERE id=?",
                             (it["id"],))
            try:
                ok = self._process_item(it, rep, model_id)
            except RunCancelled:
                self.con.execute("UPDATE run_item SET processing_state='queued', stage=NULL, attempts=attempts-1 "
                                 "WHERE id=?", (it["id"],))
                self.con.execute("UPDATE run_item SET processing_state='cancelled' WHERE run_id=? AND "
                                 "processing_state='queued' AND id!=?", (run_id, it["id"]))
                self.con.execute("UPDATE run SET state='cancelled', finished_at=? WHERE id=?", (utcnow(), run_id))
                self.m.notify("warning", "run_cancelled",
                              f"Run {run_id} was cancelled. The recording in progress keeps its checkpoint.", run_id=run_id)
                self.control.stage = "cancelled"
                return rep
            except Exception as exc:
                if self.con.in_transaction:
                    self.con.execute("ROLLBACK")
                self.con.execute("UPDATE run_item SET processing_state='failed', error_code=?, error_detail=? WHERE id=?",
                                 (type(exc).__name__, str(exc)[:500], it["id"]))
                self.m.notify("warning", "processing_failed", f"Could not process {it['rel_path']}: {exc}",
                              run_id=run_id, source_version_id=it["source_version_id"])
                rep.failed += 1
                rep.errors.append(f"{it['rel_path']}: {exc}")
                continue
            if ok:
                rep.done += 1
            else:
                rep.failed += 1
            if progress:
                progress(n, total)
        self.control.stage = "finished"
        left = self.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=? AND processing_state IN ('queued','running')",
                                (run_id,)).fetchone()[0]
        if not left:
            failed = self.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=? AND processing_state='failed'",
                                      (run_id,)).fetchone()[0]
            self.con.execute("UPDATE run SET state=?, finished_at=? WHERE id=?",
                             ("completed" if not failed else "completed_with_errors", utcnow(), run_id))
            self.m.notify("info" if not failed else "warning", "run_finished",
                          f"Run {run_id} finished: {rep.done} recording(s) analysed, {rep.events} events, {failed} failed.",
                          run_id=run_id)
        return rep

    def _process_item(self, it, rep: VideoReport, model_id: int) -> bool:
        vid = it["source_version_id"]
        pr = self.probe_version(vid)
        if pr["error"] or not pr["width"]:
            self.con.execute("UPDATE run_item SET processing_state='failed', error_code='undecodable', error_detail=? "
                             "WHERE id=?", (pr["error"] or "no video stream", it["id"]))
            self.m.notify("warning", "video_undecodable", f"Cannot decode {it['rel_path']}: {pr['error']}",
                          source_version_id=vid)
            return False
        path = self.views._path(vid)
        cp = self.con.execute("SELECT * FROM video_checkpoint WHERE run_item_id=?", (it["id"],)).fetchone()
        if cp:
            tracker = Tracker.from_state(json.loads(cp["tracker_state"]))
            last_t, nxt, count = cp["last_t"], cp["next_sample_t"], cp["frames_analysed"]
            import numpy as np
            prev = np.frombuffer(cp["prev_signature"], np.uint8).reshape(36, 64) if cp["prev_signature"] else None
        else:
            tracker = Tracker(pr["width"], pr["height"], self.s.tracker)
            last_t, nxt, count, prev = None, 0.0, 0, None
        ids = {r["track_key"]: r["id"] for r in self.con.execute(
            "SELECT id, track_key FROM track WHERE run_item_id=?", (it["id"],))}
        in_chunk = 0
        self.con.execute("BEGIN")
        try:
            for fr, nxt_after in frames(path, self.s.interval_s, after_t=last_t, next_sample_t=nxt):
                sig = scene_signature(fr.image)
                if is_scene_cut(prev, sig):
                    self.con.execute("INSERT INTO video_marker(run_item_id,t,kind) VALUES (?,?,?)",
                                     (it["id"], fr.t, "scene_cut"))
                    self._finish(it, tracker.close_all("scene_cut"), ids, pr, rep)
                prev = sig
                dets = [(list(map(float, d.box)), d.cls, float(d.score)) for d in self.detector.detect(fr.image)]
                obs, closed, new = tracker.update(fr.t, dets)
                for tr in new:
                    ids[tr.key] = self.con.execute(
                        "INSERT INTO track(run_item_id,track_key,state) VALUES (?,?,?)",
                        (it["id"], tr.key, tr.state)).lastrowid
                for tr, box, cls, score in obs:
                    self.con.execute(
                        "INSERT INTO track_observation(track_id,t,x1,y1,x2,y2,cls,score) VALUES (?,?,?,?,?,?,?,?)",
                        (ids[tr.key], round(fr.t, 6), *[round(float(b), 2) for b in box], cls, round(score, 4)))
                self._finish(it, closed, ids, pr, rep)
                last_t, nxt, count = fr.t, nxt_after, count + 1
                rep.frames += 1
                in_chunk += 1
                if in_chunk >= self.s.chunk_frames:
                    self._checkpoint(it["id"], last_t, nxt, count, tracker, prev)
                    self.con.execute("COMMIT")
                    in_chunk = 0
                    self.control.checkpoint()          # pause or cancel only between committed chunks
                    self.con.execute("BEGIN")
            end_t = pr["duration_s"] if pr["duration_s"] else last_t
            self._finish(it, tracker.close_all("end_of_recording", end_t), ids, pr, rep)
            self._checkpoint(it["id"], last_t if last_t is not None else 0.0, nxt, count, tracker, prev)
            self.con.execute("UPDATE run_item SET processing_state='done', stage=NULL, error_code=NULL, error_detail=NULL "
                             "WHERE id=?", (it["id"],))
            self.con.execute("COMMIT")
        except RunCancelled:
            raise
        except Exception:
            if self.con.in_transaction:
                self.con.execute("ROLLBACK")
            raise
        return True

    def _checkpoint(self, item_id, last_t, nxt, count, tracker: Tracker, prev) -> None:
        self.con.execute(
            "INSERT INTO video_checkpoint(run_item_id,last_t,next_sample_t,frames_analysed,tracker_state,prev_signature,"
            "updated_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(run_item_id) DO UPDATE SET last_t=excluded.last_t, "
            "next_sample_t=excluded.next_sample_t, frames_analysed=excluded.frames_analysed, "
            "tracker_state=excluded.tracker_state, prev_signature=excluded.prev_signature, updated_at=excluded.updated_at",
            (item_id, last_t, nxt, count, json.dumps(tracker.state()), prev.tobytes() if prev is not None else None,
             utcnow()))

    def _finish(self, it, closed: list[Closed], ids: dict, pr: dict, rep: VideoReport) -> None:
        for c in closed:
            tr = c.track
            tid = ids[tr.key]
            if not c.confirmed:
                self.con.execute("UPDATE track SET state='discarded', closed_reason='too_short' WHERE id=?", (tid,))
                continue
            self.con.execute("UPDATE track SET state='closed', closed_reason=? WHERE id=?", (c.reason, tid))
            n = self.con.execute("SELECT COUNT(*) FROM video_event WHERE run_item_id=?", (it["id"],)).fetchone()[0] + 1
            total = sum(tr.votes.values()) or 1.0
            cls = max(sorted(tr.votes), key=lambda k: tr.votes[k])
            share = round(tr.votes[cls] / total, 3)
            w, h = pr["width"], pr["height"]
            entry = [round((tr.first_box[0] + tr.first_box[2]) / 2 / w, 4), round(tr.first_box[3] / h, 4)]
            exit_ = [round((tr.box[0] + tr.box[2]) / 2 / w, 4), round(tr.box[3] / h, 4)]
            flags = []
            e = self.s.tracker.edge_px
            first_edge = (tr.first_box[0] <= e or tr.first_box[1] <= e or tr.first_box[2] >= w - e or tr.first_box[3] >= h - e)
            if not first_edge and tr.first_t > self.s.tracker.gap_s:
                flags.append("appeared_inside_frame")
            if c.reason == "lost":
                flags.append("ended_inside_frame")
            if tr.related:
                flags.append("possible_duplicate")
            if share < 0.7:
                flags.append("class_uncertain")
            if c.reason == "scene_cut":
                flags.append("cut_by_scene_change")
            if c.reason == "end_of_recording":
                flags.append("cut_by_end_of_recording")
            related = []
            for key in tr.related:
                r = self.con.execute("SELECT e.event_key FROM video_event e JOIN track t ON t.id=e.track_id "
                                     "WHERE t.run_item_id=? AND t.track_key=?", (it["id"], key)).fetchone()
                if r:
                    related.append(r[0])
            self.con.execute(
                "INSERT INTO video_event(run_item_id,source_version_id,event_key,track_id,first_t,last_t,cls,cls_share,"
                "class_votes,entry,exit,end_reason,flags,related,thumb_t,thumb_box,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (it["id"], it["source_version_id"], f"E-{n:04d}", tid, round(tr.first_t, 6), round(tr.t, 6), cls, share,
                 json.dumps(tr.votes, sort_keys=True), json.dumps(entry), json.dumps(exit_), c.reason, json.dumps(flags),
                 json.dumps(related), round(tr.best_t, 6), json.dumps(tr.best_box), utcnow()))
            rep.events += 1
