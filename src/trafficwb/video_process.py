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
- Camera views use the first presentation frame as a signature; named and
  inclusion/exclusion regions filter reviewed event boxes during search.
- Not in this version: night-light tracking (blob, pair, body).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .process import RunCancelled, RunControl, register_model
from .store import Mission, utcnow
from .tracker import Closed, Tracker, TrackerParams
from .video import camera_motion, frames, is_scene_cut, motion_frame, probe, scene_signature
from .views import ViewService

MAX_ATTEMPTS = 3


@dataclass
class VideoSettings:
    interval_s: float = 0.2          # analysis every 0.2 s of presentation time (5 per second)
    chunk_frames: int = 50           # analysed frames per committed chunk
    tracker: TrackerParams = field(default_factory=TrackerParams)
    sam3_interval_s: float | None = None   # SAM 3.1 every this many seconds (None: the mission's choice, else 1 s)
    camera: str | None = None              # fixed | moving (None: the mission's choice)
    workers: int | None = None             # segments analysed side by side (None: the mission's choice, else 1)
    segment_s: float = 300.0               # segment length when workers > 1
    overlap_s: float = 3.0                 # analysed on both sides of each cut, to join tracks across it
    low_score: float = 0.10                # the standard detector reports boxes down to this score in video, for
                                           # the tracker's second pass (only boxes from high_score start tracks)


def _low_threshold(detector, score: float):
    """The same detector reporting weaker boxes too (shares the loaded model), for ByteTrack's second pass."""
    if hasattr(detector, "standard") and hasattr(detector, "sam"):
        if hasattr(detector.standard, "with_threshold"):
            import copy
            c = copy.copy(detector)
            c.standard = detector.standard.with_threshold(min(score, detector.standard.settings.score_threshold))
            return c
        return detector
    if hasattr(detector, "with_threshold"):
        return detector.with_threshold(min(score, detector.settings.score_threshold))
    return detector


def _kind(detector) -> str:
    """standard | sam3 | both, from the detector's shape (test doubles are standard)."""
    if hasattr(detector, "standard") and hasattr(detector, "sam"):
        return "both"
    if hasattr(detector, "prompts") and hasattr(detector, "engine"):
        return "sam3"
    return "standard"


@dataclass
class VideoReport:
    run_id: int
    done: int = 0
    failed: int = 0
    frames: int = 0
    events: int = 0
    errors: list[str] = field(default_factory=list)
    plates: dict | None = None
    video_s: float = 0.0                   # seconds of recording analysed in this call
    wall_s: float = 0.0                    # seconds it took

    @property
    def realtime_factor(self) -> float | None:
        """How many seconds of video were analysed per second of work (above 1: faster than real time)."""
        return round(self.video_s / self.wall_s, 2) if self.wall_s > 0 and self.video_s > 0 else None


def _with_found(d, who: str):
    from dataclasses import replace
    try:
        return replace(d, found_by=who)
    except TypeError:
        return d


def _plain(res) -> dict:
    return {"idx": res.idx, "cuts": res.cuts, "frames": res.frames, "seconds": res.seconds,
            "pieces": [{"key": p.key, "family": p.family, "obs": p.obs, "reason": p.reason, "related": p.related}
                       for p in res.pieces]}


def _result(d: dict):
    from .video_segments import Piece, SegmentResult
    return SegmentResult(d["idx"], [Piece(p["key"], p["family"], [tuple(o) for o in p["obs"]], p["reason"], p["related"])
                                    for p in d["pieces"]], d["cuts"], d["frames"], d["seconds"])


def _pack(a) -> bytes | None:
    if a is None:
        return None
    import numpy as np
    return np.asarray(a.shape, np.int32).tobytes() + np.ascontiguousarray(a, np.uint8).tobytes()


def _unpack(b: bytes | None):
    if not b:
        return None
    import numpy as np
    h, w = np.frombuffer(b[:8], np.int32)
    return np.frombuffer(b[8:], np.uint8).reshape(int(h), int(w)).copy()


def probe_version(con, views, version_id: int) -> dict:
    """The recording's probe (size, length, frame rate, start time), probed once and kept in video_probe."""
    row = con.execute("SELECT * FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone()
    if row:
        return dict(row)
    v = con.execute("SELECT capture_utc, capture_precision FROM source_version WHERE id=?", (version_id,)).fetchone()
    p = probe(views._path(version_id))
    con.execute(
        "INSERT INTO video_probe(source_version_id,container,codec,width,height,duration_s,nominal_fps,frame_count,"
        "variable_frame_rate,creation_time,start_utc,start_precision,start_source,error,probed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (version_id, p.container, p.codec, p.width, p.height, p.duration_s, p.nominal_fps, p.frame_count,
         None if p.variable_frame_rate is None else int(p.variable_frame_rate), p.creation_time,
         v["capture_utc"], v["capture_precision"] if v["capture_utc"] else None,
         "filename" if v["capture_utc"] else "none", p.error, utcnow()))
    return dict(con.execute("SELECT * FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone())


class VideoProcessor:
    def __init__(self, mission: Mission, detector, card: dict | None = None, settings: VideoSettings | None = None,
                 control: RunControl | None = None, plate_models: Callable[[], tuple] | None = None,
                 only: set[int] | None = None):
        self.m = mission
        self.only = set(only) if only else None    # analyse only these recordings (the catalog's ticked ones)
        self.plate_models = plate_models   # loaded only when a plate recognition mission needs it
        self.con = mission.con
        self.detector = detector
        self.card = card if card is not None else getattr(getattr(detector, "pack", None), "card", {})
        self.s = settings or VideoSettings()
        self.control = control or RunControl()
        self.views = ViewService(mission)
        # SAM 3.1 takes seconds per frame, so it looks at every Nth analysed frame and the tracker carries what it
        # found between them. With both detectors the standard one sees every frame; with SAM 3.1 alone the grid
        # itself is spaced out to the SAM 3.1 interval.
        self.kind = _kind(detector)
        sam_s = self.s.sam3_interval_s or getattr(mission.config(), "sam3_interval_s", None) or 1.0
        self.sam_every = max(1, round(sam_s / self.s.interval_s)) if self.kind == "both" else 1
        self.grid_s = max(self.s.interval_s, sam_s) if self.kind == "sam3" else self.s.interval_s
        self.sam_s = sam_s
        self.camera = self.s.camera or getattr(mission.config(), "camera", None) or "fixed"
        self.workers = max(1, int(self.s.workers or getattr(mission.config(), "video_workers", None) or 1))
        if self.s.tracker.algo == "byte":
            self.detector = _low_threshold(detector, self.s.low_score)
        if self.kind != "standard" and not self.s.tracker.jump_per_s:
            # SAM 3.1 sees a target about once a second: a person walks several body widths in that time.
            from dataclasses import replace
            self.s = replace(self.s, tracker=replace(self.s.tracker, jump_per_s=3.0))


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
        return probe_version(self.con, self.views, version_id)

    def pending_runs(self) -> list[int]:
        runs = [r[0] for r in self.con.execute(
            "SELECT id FROM run WHERE state IN ('queued','running') OR id IN "
            "(SELECT run_id FROM run_item WHERE processing_state IN ('queued','running')) ORDER BY id")]
        if self.only is None:
            return runs
        q = ",".join("?" * len(self.only)) or "NULL"
        sel = {r[0] for r in self.con.execute(
            f"SELECT DISTINCT run_id FROM run_item WHERE processing_state IN ('queued','running') AND "
            f"source_version_id IN ({q})", tuple(self.only))}
        return [r for r in runs if r in sel]

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
        pre["video"] = {"interval_s": self.grid_s, "chunk_frames": self.s.chunk_frames, "detector": self.kind,
                        "camera": self.camera, "camera_motion": "optical flow + RANSAC similarity" if self.camera == "moving"
                        else "none (fixed camera)",
                        **({"sam3_every_frames": self.sam_every, "sam3_interval_s": self.sam_s}
                           if self.kind != "standard" else {}),
                        "tracker": asdict(self.s.tracker), "time_base": "presentation timestamps",
                        "view_signature": "first presentation frame", "regions": "applied to event search boxes"}
        self.con.execute("UPDATE run SET state='running', started_at=COALESCE(started_at,?), model_versions=?, "
                         "preprocessing=? WHERE id=?", (utcnow(), json.dumps(models), json.dumps(pre), run_id))
        self._model_id = model_id
        self.control.stage = "grouping camera views"
        self.views.assign_or_propose(min_size=1)
        items = self.con.execute(
            "SELECT i.*, f.rel_path FROM run_item i JOIN source_version v ON v.id=i.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id WHERE i.run_id=? AND i.processing_state IN ('queued','running') "
            "ORDER BY i.id", (run_id,)).fetchall()
        if self.only is not None:
            items = [it for it in items if it["source_version_id"] in self.only]
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
        if not cp and self.workers > 1 and (pr["duration_s"] or 0) > self.s.segment_s * 1.25:
            return self._process_segmented(it, pr, rep, path)
        if cp:
            tracker = Tracker.from_state(json.loads(cp["tracker_state"]))
            last_t, nxt, count = cp["last_t"], cp["next_sample_t"], cp["frames_analysed"]
            import numpy as np
            prev = np.frombuffer(cp["prev_signature"], np.uint8).reshape(36, 64) if cp["prev_signature"] else None
            pm = _unpack(cp["prev_motion"]) if "prev_motion" in cp.keys() else None
        else:
            tracker = Tracker(pr["width"], pr["height"], self.s.tracker)
            last_t, nxt, count, prev, pm = None, 0.0, 0, None, None
        moving = self.camera == "moving"
        ids = {r["track_key"]: r["id"] for r in self.con.execute(
            "SELECT id, track_key FROM track WHERE run_item_id=?", (it["id"],))}
        in_chunk = 0
        self.con.execute("BEGIN")
        try:
            for fr, nxt_after in frames(path, self.grid_s, after_t=last_t, next_sample_t=nxt):
                sig = scene_signature(fr.image)
                motion = None
                cut = is_scene_cut(prev, sig)
                if moving:
                    # A moving camera: measure how the picture moved and take it out before tracking. A pan or
                    # zoom is never a cut; a textured frame that matches the previous one nowhere is.
                    mf = motion_frame(fr.image)
                    motion, why = camera_motion(pm, mf, pr["width"], [t.box for t in tracker.tracks])
                    cut = why == "no_match" or (why == "no_texture" and pm is not None and cut)
                    pm = mf
                if cut:
                    self.con.execute("INSERT INTO video_marker(run_item_id,t,kind) VALUES (?,?,?)",
                                     (it["id"], fr.t, "scene_cut"))
                    self._finish(it, tracker.close_all("scene_cut"), ids, pr, rep)
                prev = sig
                dets, found = [], {}
                for d in self._detect(fr.image, count):
                    box = list(map(float, d.box))
                    found[id(box)] = getattr(d, "found_by", None) or ("sam3" if self.kind == "sam3" else "standard")
                    dets.append((box, d.cls, float(d.score)))
                obs, closed, new = tracker.update(fr.t, dets, motion=motion)
                for tr in new:
                    ids[tr.key] = self.con.execute(
                        "INSERT INTO track(run_item_id,track_key,state) VALUES (?,?,?)",
                        (it["id"], tr.key, tr.state)).lastrowid
                for tr, box, cls, score in obs:
                    self.con.execute(
                        "INSERT INTO track_observation(track_id,t,x1,y1,x2,y2,cls,score,found_by) VALUES (?,?,?,?,?,?,?,?,?)",
                        (ids[tr.key], round(fr.t, 6), *[round(float(b), 2) for b in box], cls, round(score, 4),
                         found.get(id(box))))
                self._finish(it, closed, ids, pr, rep)
                last_t, nxt, count = fr.t, nxt_after, count + 1
                rep.frames += 1
                in_chunk += 1
                if in_chunk >= self.s.chunk_frames:
                    self._checkpoint(it["id"], last_t, nxt, count, tracker, prev, pm)
                    self.con.execute("COMMIT")
                    in_chunk = 0
                    self.control.checkpoint()          # pause or cancel only between committed chunks
                    self.con.execute("BEGIN")
            end_t = pr["duration_s"] if pr["duration_s"] else last_t
            self._finish(it, tracker.close_all("end_of_recording", end_t), ids, pr, rep)
            self._checkpoint(it["id"], last_t if last_t is not None else 0.0, nxt, count, tracker, prev, pm)
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

    # ---------- several segments side by side ----------
    def _process_segmented(self, it, pr: dict, rep: "VideoReport", path) -> bool:
        """Cut the recording into segments, analyse them in a pool of workers sharing the detector, join the
        tracks at the cuts and record them as one run would. Finished segments are kept, so a cancelled run
        resumes with the segments still to do."""
        import time as _time
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from . import video_segments as vs
        t0 = _time.monotonic()
        # The overlap on each side holds at least one SAM 3.1 frame, so a target only SAM 3.1 finds is seen by both
        # segments on two frames or more around the cut and can be joined.
        ov = max(self.s.overlap_s, self.sam_s if self.kind != "standard" else 0.0)
        segs = vs.plan(pr["duration_s"], self.s.segment_s, ov, self.grid_s)
        done = {r["idx"]: _result(json.loads(r["result"])) for r in self.con.execute(
            "SELECT idx, result FROM video_segment WHERE run_item_id=? AND state='done'", (it["id"],))}
        todo = [s for s in segs if s.idx not in done]
        moving = self.camera == "moving"
        sam_every = self.sam_every if self.kind == "both" else 1

        def detect(img, sam: bool):
            if self.kind == "both" and not sam:
                from dataclasses import replace
                return [replace(d, found_by="standard") for d in self.detector.standard.detect(img)]
            return [d if getattr(d, "found_by", None) else _with_found(d, "sam3" if self.kind == "sam3" else "standard")
                    for d in self.detector.detect(img)]

        self.control.stage = f"analysing video in {len(segs)} segments, {min(self.workers, len(segs))} at a time"
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="segment") as pool:
            running = {pool.submit(vs.analyse, s, path, pr["width"], pr["height"], detect, self.grid_s,
                                   self.s.tracker, moving, sam_every, self.control.checkpoint): s for s in todo}
            try:
                while running:
                    fin, _ = wait(running, return_when=FIRST_COMPLETED)
                    for f in fin:
                        s = running.pop(f)
                        res = f.result()                       # a cancel raised inside a worker comes out here
                        done[s.idx] = res
                        self.con.execute(
                            "INSERT INTO video_segment(run_item_id,idx,own_start,own_end,state,result,frames,seconds,"
                            "updated_at) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(run_item_id,idx) DO UPDATE SET "
                            "state=excluded.state, result=excluded.result, frames=excluded.frames, "
                            "seconds=excluded.seconds, updated_at=excluded.updated_at",
                            (it["id"], s.idx, s.own_a, None if s.own_b == float("inf") else s.own_b, "done",
                             json.dumps(_plain(res)), res.frames, res.seconds, utcnow()))
                        self.control.checkpoint()      # pause or cancel between segments too
            except BaseException:
                self.control.cancel() if not self.control.cancelled else None
                for f in running:
                    f.cancel()
                raise
        # All segments are done: join them and write tracks, observations and events in one transaction.
        joined = sorted(vs.join(segs, done), key=lambda j: (j.obs[0][0], j.obs[0][1]))
        keys: dict[tuple, str] = {}
        for n, j in enumerate(joined, 1):
            for pid in j.ids:
                keys[tuple(pid)] = f"T{n}"
        end_t = pr["duration_s"]
        self.con.execute("BEGIN")
        try:
            ids: dict[str, int] = {}
            closed = []
            from .tracker import Closed
            for n, j in enumerate(joined, 1):
                tr, ok = vs.to_track(j, f"T{n}", self.s.tracker.min_hits)
                tr.related = sorted({keys[tuple(r)] for r in j.related if tuple(r) in keys} - {tr.key})
                ids[tr.key] = self.con.execute("INSERT INTO track(run_item_id,track_key,state) VALUES (?,?,?)",
                                               (it["id"], tr.key, tr.state)).lastrowid
                self.con.executemany(
                    "INSERT INTO track_observation(track_id,t,x1,y1,x2,y2,cls,score,found_by) VALUES (?,?,?,?,?,?,?,?,?)",
                    [(ids[tr.key], t, *box, cls, score, fb) for t, box, cls, score, fb in j.obs])
                reason = j.reason
                if reason == "segment_end":
                    e = self.s.tracker.edge_px
                    edge = (tr.box[0] <= e or tr.box[1] <= e or tr.box[2] >= pr["width"] - e or tr.box[3] >= pr["height"] - e)
                    reason = "left_frame" if end_t - tr.t > 0.5 and edge else "end_of_recording"
                closed.append(Closed(tr, reason, ok))
            for s in segs:
                for t in done[s.idx].cuts:
                    self.con.execute("INSERT INTO video_marker(run_item_id,t,kind) VALUES (?,?,?)", (it["id"], t, "scene_cut"))
            self._finish(it, sorted(closed, key=lambda c: (c.track.t, c.track.key)), ids, pr, rep)
            frames_n = sum(done[s.idx].frames for s in segs)
            last_t = max((j.obs[-1][0] for j in joined), default=0.0)
            self.con.execute(
                "INSERT INTO video_checkpoint(run_item_id,last_t,next_sample_t,frames_analysed,tracker_state,updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(run_item_id) DO UPDATE SET last_t=excluded.last_t, "
                "frames_analysed=excluded.frames_analysed, updated_at=excluded.updated_at",
                (it["id"], last_t, end_t, frames_n, json.dumps({"segments": len(segs)}), utcnow()))
            self.con.execute("UPDATE run_item SET processing_state='done', stage=NULL, error_code=NULL, error_detail=NULL "
                             "WHERE id=?", (it["id"],))
            self.con.execute("COMMIT")
        except Exception:
            if self.con.in_transaction:
                self.con.execute("ROLLBACK")
            raise
        rep.frames += sum(done[s.idx].frames for s in segs if s in todo)
        rep.video_s += pr["duration_s"]
        rep.wall_s += _time.monotonic() - t0
        return True

    def _detect(self, img, count: int):
        """Detections on one analysed frame; ``count`` frames of this recording were analysed before it, so a
        resumed run calls SAM 3.1 on the same frames as an uninterrupted one."""
        if self.kind == "both" and count % self.sam_every:
            from dataclasses import replace
            return [replace(d, found_by="standard") for d in self.detector.standard.detect(img)]
        return self.detector.detect(img)

    def _checkpoint(self, item_id, last_t, nxt, count, tracker: Tracker, prev, pm=None) -> None:
        self.con.execute(
            "INSERT INTO video_checkpoint(run_item_id,last_t,next_sample_t,frames_analysed,tracker_state,prev_signature,"
            "prev_motion,updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_item_id) DO UPDATE SET "
            "last_t=excluded.last_t, next_sample_t=excluded.next_sample_t, frames_analysed=excluded.frames_analysed, "
            "tracker_state=excluded.tracker_state, prev_signature=excluded.prev_signature, "
            "prev_motion=excluded.prev_motion, updated_at=excluded.updated_at",
            (item_id, last_t, nxt, count, json.dumps(tracker.state()), prev.tobytes() if prev is not None else None,
             _pack(pm), utcnow()))

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
