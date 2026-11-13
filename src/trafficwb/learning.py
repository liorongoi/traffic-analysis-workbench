"""Learning, part 1: eligible evidence, dataset versions, evaluation and model promotion (LRN-01..04).

Rules (design package, deliverable 07):
- Only explicit reviewed labels are used. An image an operator approved
  individually is a **full** review: every vehicle inside the included regions
  is marked, so unmarked areas there are background. An image with corrections
  but no approval is a **partial** review: only the corrected objects are
  labels, everything else is ignored, never treated as background. Bulk
  approved and unreviewed images are never used.
- A dataset version is a frozen list of images, labels and splits. Splits are
  group-wise by camera view and date block, so near-duplicate frames of one
  view and day never sit in two splits. Once an image group is in a test set,
  later dataset versions keep it in test.
- Evaluation runs a detector on a split and compares it with the labels:
  precision, recall, F1, AP at IoU 0.5 and per-image count error, overall, by
  light condition and by view. Precision and count error use full reviews
  only; partial reviews add recall for their corrected objects.
- No training adapter is implemented yet; each algorithm declares its planned
  adapter (LRN-02) and nothing is shown as trainable.
- The production model per task changes only when an operator promotes a
  version; rollback moves the pointer back. Nothing is promoted automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from .detect import COCO_ALIASES, VEHICLE_CLASSES
from .localtime import to_local
from .process import RunControl
from .regions import RegionSet
from .review import ReviewService
from .store import CONTROL_SCHEMA, Mission, connect, utcnow
from .views import ViewService, read_image


class LearningError(ValueError):
    pass


# LRN-02: every algorithm declares its learning adapter. "planned" means designed, not implemented.
ADAPTERS = [
    {"algorithm": "Vehicle detector", "adapter": "fine-tune on this computer from the learning page (NVIDIA or Apple GPU, or the CPU)",
     "status": "available"},
    {"algorithm": "Vehicle detector thresholds", "adapter": "thresholds per view", "status": "planned"},
    {"algorithm": "Colour reader", "adapter": "small classifier trained on corrected colours", "status": "planned"},
    {"algorithm": "Night light grouping", "adapter": "pair classifier on reviewed groups", "status": "planned"},
    {"algorithm": "Persistent-light background", "adapter": "background refit", "status": "planned"},
    {"algorithm": "Plate finder", "adapter": "fine-tune per country (GPU)", "status": "planned"},
    {"algorithm": "Plate reader", "adapter": "fine-tune per country and layout (GPU)", "status": "planned"},
    {"algorithm": "Video tracker", "adapter": "association parameters", "status": "planned"},
    {"algorithm": "View grouping", "adapter": "merge thresholds", "status": "planned"},
    {"algorithm": "Search interpreter", "adapter": "none: vocabulary revisions only", "status": "not_applicable"},
]

SPLITS = ("train", "val", "test")

# Classes a model can learn: the five vehicle classes and "person" (COCO class 0). A mission's own class named
# person, pedestrian, people... is learnt as person; other own classes are found by SAM 3.1 only, so their boxes
# are neither labels nor background.
TRAINABLE = (*VEHICLE_CLASSES, "person")

# Frames taken from a reviewed recording (video learning).
FRAMES_PER_EVENT = 5          # an event's best frame plus up to 4 more spread over its track
BACKGROUND_EVERY_S = 5.0      # an approved recording: also a frame every 5 s, so empty road is learnt too
FRAME_MIN_GAP_S = 0.4         # frames closer than this are one
FRAMES_PER_RECORDING = 300


def trainable(cls: str | None) -> str | None:
    """The class a model learns a label as, or None when it cannot learn it."""
    if cls in VEHICLE_CLASSES:
        return cls
    return "person" if COCO_ALIASES.get(cls or "", cls) == "person" else None


def _refs(refs) -> list[tuple[str, int]]:
    """Dataset references as (mission_id, dataset_id), in order, each once."""
    out: list[tuple[str, int]] = []
    for r in refs or []:
        if isinstance(r, dict):
            key = (str(r["mission_id"]), int(r["dataset_id"]))
        elif hasattr(r, "mission_id"):
            key = (str(r.mission_id), int(r.dataset_id))
        else:
            key = (str(r[0]), int(r[1]))
        if key not in out:
            out.append(key)
    return out


def data_key(sources: list[dict], split: str, iou: float) -> str:
    """Two evaluations with the same key looked at exactly the same images with the same match rule."""
    return json.dumps([sorted([s["mission_id"], int(s["dataset_id"])] for s in sources), split, round(float(iou), 4)])


def improved(new: dict | None, base: dict | None) -> bool | None:
    """Did the new model do better than the one it started from on the same images? F1 first, then AP at IoU 0.5."""
    if not new or not base:
        return None
    a, b = new.get("f1"), base.get("f1")
    if a is None or b is None:
        a, b = new.get("recall"), base.get("recall")
    if a is None or b is None:
        return None
    if abs(a - b) > 1e-4:
        return a > b
    ap_a, ap_b = new.get("ap50"), base.get("ap50")
    return ap_a is not None and ap_b is not None and ap_a > ap_b + 1e-4


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _between(obs: list[tuple[float, list[float]]], t: float, max_gap: float = 2.0) -> list[float] | None:
    """A track's box at time t, interpolated between its observations around t (None when they are far apart)."""
    before = [(tt, b) for tt, b in obs if tt < t]
    after = [(tt, b) for tt, b in obs if tt > t]
    if not before or not after:
        return None
    (ta, a), (tb, b) = before[-1], after[0]
    if tb - ta > max_gap:
        return None
    w = (t - ta) / (tb - ta)
    return [round(a[i] + w * (b[i] - a[i]), 2) for i in range(4)]


@dataclass
class ItemLabels:
    source_version_id: int
    review: str                       # full | partial | bulk | none
    positives: list[dict]             # {"box": [...], "cls": str, "uid": str, "source": machine|human|candidate}
    ignore_boxes: list[list[float]]   # unresolved night candidates: never background
    last_revision_id: int | None


class LearningService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.rs = ReviewService(mission)
        self.views = ViewService(mission)
        self.zone = mission.config().iana_zone
        self.video = mission.input_type == "video"

    # ---------- eligibility ----------
    def review_kind(self, vid: int) -> str:
        if self.video:
            return self._recording_kind(vid)
        st = self.rs.review_status(vid)
        if st == "approved":
            return "full"
        if st == "bulk_approved_unreviewed":
            return "bulk"
        touched = self.con.execute(
            "SELECT (SELECT COUNT(*) FROM annotation_revision WHERE source_version_id=?) + "
            "(SELECT COUNT(*) FROM candidate_revision WHERE source_version_id=?)", (vid, vid)).fetchone()[0]
        return "partial" if touched else "none"

    def labels(self, vid: int) -> ItemLabels:
        kind = self.review_kind(vid)
        pos: list[dict] = []
        ignore: list[list[float]] = []
        if kind in ("full", "partial"):
            for o in self.rs.objects(vid, include_removed=False):
                if kind == "partial" and o.correction in (None, "none"):
                    continue            # untouched machine output is not a label in a partial review
                if trainable(o.cls) is None:
                    # a class of the mission's own that no model here can learn (SAM 3.1 finds it): the area must
                    # not be taught as background either
                    ignore.append([round(float(v), 2) for v in o.box])
                    continue
                pos.append({"box": [round(float(v), 2) for v in o.box], "cls": o.cls, "uid": o.uid,
                            "source": "human" if o.origin == "human" or o.correction not in (None, "none") else "machine"})
            for c in self.rs.candidates(vid):
                if c.status == "confirmed":
                    pos.append({"box": [round(float(v), 2) for v in c.box], "cls": c.cls or "car", "uid": c.uid,
                                "source": "candidate"})
                elif c.status == "unresolved" and kind == "full":
                    ignore.append([round(float(v), 2) for v in c.box])
        last = self.con.execute(
            "SELECT MAX(id) FROM annotation_revision WHERE source_version_id=?", (vid,)).fetchone()[0]
        return ItemLabels(vid, kind, pos, ignore, last)

    # ---------- video: frames of reviewed recordings ----------
    def _latest_item(self, vid: int) -> int | None:
        return self.con.execute("SELECT MAX(id) FROM run_item WHERE source_version_id=? AND processing_state='done'",
                                (vid,)).fetchone()[0]

    def _events(self, vid: int) -> list:
        from .video_review import VideoService
        item = self._latest_item(vid)
        return [e for e in VideoService(self.m).events(vid) if e.run_item_id == item]

    @staticmethod
    def _reviewed(e) -> bool:
        return e.status != "pending" or e.cls != e.machine_cls or bool(getattr(e, "fixes", None))

    def recording_approved(self, vid: int) -> bool:
        r = self.con.execute("SELECT action FROM recording_review WHERE source_version_id=? ORDER BY id DESC LIMIT 1",
                             (vid,)).fetchone()
        return bool(r) and r[0] == "approve"

    def _recording_kind(self, vid: int) -> str:
        evs = self._events(vid)
        if self.recording_approved(vid) and not any(e.status == "pending" for e in evs):
            return "full"
        return "partial" if any(self._reviewed(e) for e in evs) else "none"

    def approve_recording(self, vid: int, author: str, confirm_rest: bool = False) -> dict:
        """The operator says every target of the recording is an event and every event is decided: its frames may
        then teach background too (as approving a still image does). With ``confirm_rest``, the events not decided
        yet are confirmed first, in one step."""
        if not self.video:
            raise LearningError("Only a recording can be approved here.")
        if self._latest_item(vid) is None:
            raise LearningError("Only an analysed recording can be approved.")
        if confirm_rest:
            from .video_review import VideoService
            svc = VideoService(self.m)
            for e in self._events(vid):
                if e.status == "pending":
                    svc.revise(e.id, "confirm", author, "confirmed when the recording was approved")
        pending = sum(1 for e in self._events(vid) if e.status == "pending")
        if pending:
            raise LearningError(f"{pending} event(s) of this recording are not decided yet. Confirm or reject every "
                                "event before approving the recording.")
        self.con.execute("INSERT INTO recording_review(source_version_id,action,author,created_at) VALUES (?,?,?,?)",
                         (vid, "approve", author, utcnow()))
        return {"source_version_id": vid, "review": self.review_kind(vid)}

    def reopen_recording(self, vid: int, author: str) -> dict:
        self.con.execute("INSERT INTO recording_review(source_version_id,action,author,created_at) VALUES (?,?,?,?)",
                         (vid, "reopen", author, utcnow()))
        return {"source_version_id": vid, "review": self.review_kind(vid)}

    def video_frames(self, vid: int) -> tuple[str, list[dict]]:
        """The training frames of one recording and their labels: (review kind, [{t, positives, ignore_boxes}]).

        - A reviewed event (confirmed, duplicate, merged, or given another class) gives its best frame and up to
          four more spread over its track; its box on each frame is a label of its class.
        - An approved recording (full review) also gives a frame every 5 s. There, a rejected detection and
          everything nobody marked is background; a box the tracker saw outside any event, and an event's position
          on a frame it was not matched on, are ignored (neither label nor background).
        - Otherwise (partial review) only the reviewed events' boxes are labels; the rest of the frame is ignored.
        """
        kind = self.review_kind(vid)
        if kind not in ("full", "partial"):
            return kind, []
        evs = self._events(vid)
        item = self._latest_item(vid)
        track_of = {r[0]: r[1] for r in self.con.execute("SELECT id, track_id FROM video_event WHERE run_item_id=?", (item,))}
        obs: dict[int, list[tuple[float, list[float]]]] = {}
        for r in self.con.execute(
                "SELECT o.track_id, o.t, o.x1, o.y1, o.x2, o.y2 FROM track_observation o JOIN track k ON k.id=o.track_id "
                "WHERE k.run_item_id=? ORDER BY o.t", (item,)):
            obs.setdefault(r[0], []).append((round(r[1], 6), [round(float(v), 2) for v in r[2:6]]))
        seen = sorted({t for lst in obs.values() for t, _ in lst})        # frames the analysis looked at
        for e in evs:                       # a box the operator drew replaces the machine's on that frame
            if e.fixes:
                tid = track_of.get(e.id)
                lst = [o for o in obs.get(tid, []) if not any(abs(o[0] - f["t"]) < 1e-3 for f in e.fixes)]
                lst += [(round(f["t"], 6), [round(float(v), 2) for v in f["box"]]) for f in e.fixes]
                obs[tid] = sorted(lst, key=lambda o: o[0])
        teach = [e for e in evs if e.status != "rejected" and self._reviewed(e)]
        times: list[float] = []
        for e in teach:
            ts = [t for t, _ in obs.get(track_of.get(e.id), []) if e.first_t - 1e-6 <= t <= e.last_t + 1e-6]
            if not ts:
                continue
            # the best frame, then each time the frame farthest from those already taken
            pick = [next((t for t in ts if abs(t - e.thumb_t) < 1e-3), ts[len(ts) // 2])]
            while len(pick) < min(FRAMES_PER_EVENT, len(ts)):
                pick.append(max((t for t in ts if t not in pick), key=lambda t: min(abs(t - q) for q in pick)))
            times += pick
        if kind == "full":
            dur = self.con.execute("SELECT duration_s FROM video_probe WHERE source_version_id=?", (vid,)).fetchone()
            k = 0.0
            while dur and dur[0] and k < dur[0]:
                # a frame the analysis looked at (so every target on it is known) nearest to every 5 s
                near = min(seen, key=lambda t: abs(t - k)) if seen else None
                if near is not None and abs(near - k) <= BACKGROUND_EVERY_S / 2:
                    times.append(near)
                k += BACKGROUND_EVERY_S
        # frames the operator drew on come first (the surest labels), then the others kept apart from every one taken
        must = sorted({round(f["t"], 6) for e in teach for f in e.fixes})
        chosen = list(must)
        for t in sorted(set(times) - set(must)):
            if all(abs(t - c) >= FRAME_MIN_GAP_S for c in chosen):
                chosen.append(t)
        chosen.sort()
        if len(chosen) > FRAMES_PER_RECORDING:
            step = len(chosen) / FRAMES_PER_RECORDING
            chosen = [chosen[int(i * step)] for i in range(FRAMES_PER_RECORDING)]
        by_track = {track_of.get(e.id): e for e in evs}
        out = []
        for t in chosen:
            pos, ign = [], []
            for tid, lst in obs.items():
                box = next((b for tt, b in lst if abs(tt - t) < 1e-3), None)
                e = by_track.get(tid)
                if e is None:                                  # a track that never became an event
                    if box is not None and kind == "full":
                        ign.append(box)
                    continue
                alive = e.first_t - 1e-6 <= t <= e.last_t + 1e-6
                if e.status == "rejected":
                    continue                                   # background where the recording is approved
                if not self._reviewed(e):
                    if box is not None and kind == "full":
                        ign.append(box)
                    continue
                if box is not None and alive:
                    cls = trainable(e.cls)
                    if cls is None:
                        ign.append(box)
                    else:
                        pos.append({"box": box, "cls": e.cls, "uid": f"{e.key}@{t:.3f}", "source": "human"})
                elif alive and kind == "full":
                    gap = _between(lst, t, float("inf") if e.origin == "human" else 2.0)
                    if gap is not None:
                        ign.append(gap)
                elif box is not None and kind == "full":
                    ign.append(box)                            # trimmed off the event by the operator: unsure
            out.append({"t": t, "positives": pos, "ignore_boxes": ign})
        return kind, out

    def _images(self) -> list:
        return self.con.execute(
            "SELECT v.id, v.sha256, v.capture_utc, f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND "
            "i.processing_state='done') ORDER BY v.capture_utc, v.id").fetchall()

    def eligibility(self) -> dict:
        counts = {"full": 0, "partial": 0, "bulk": 0, "none": 0}
        for r in self._images():
            counts[self.review_kind(r["id"])] += 1
        if self.video:
            return {"recordings": counts, "images": counts, "eligible": counts["full"] + counts["partial"],
                    "rules": {"full": "approved recording: frames of every reviewed event, rejected detections and empty "
                                      "road taught as background",
                              "partial": "some events reviewed: frames of those events, their boxes only, the rest ignored",
                              "none": "no event reviewed: never used"}}
        return {"images": counts, "eligible": counts["full"] + counts["partial"],
                "rules": {"full": "approved individually: positives and background inside included regions",
                          "partial": "corrected but not approved: corrected objects only, the rest ignored",
                          "bulk": "never used", "none": "never used"}}

    # ---------- datasets ----------
    def _group(self, r, block_days: int) -> tuple[str, str]:
        vid = r["id"]
        view = self.views.current_view(vid)
        view_key = f"view-{view}" if view is not None else "unassigned"
        if r["capture_utc"]:
            utc = datetime.fromisoformat(r["capture_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
            d = to_local(utc, self.zone).local.date()
            start = date.fromordinal(d.toordinal() - (d.toordinal() % block_days))
            block = start.isoformat()
        else:
            block = "no-date"
        return view_key, block

    @staticmethod
    def _bucket(seed: int, key: str) -> float:
        h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64

    def _samples(self, r) -> list[tuple[ItemLabels, float | None, str]]:
        """The samples one source gives: a still image is one; a recording gives one per training frame."""
        if not self.video:
            return [(self.labels(r["id"]), None, r["sha256"])]
        kind, frames = self.video_frames(r["id"])
        if not frames:
            return [(ItemLabels(r["id"], kind, [], [], None), None, r["sha256"])]
        return [(ItemLabels(r["id"], kind, f["positives"], f["ignore_boxes"], None), f["t"],
                 hashlib.sha256(f"{r['sha256']}|{f['t']:.6f}".encode()).hexdigest()) for f in frames]

    def image_of(self, row: dict):
        """The pixels of a dataset member: the still image, or the recording's frame at its time."""
        path = self.views._path(row["source_version_id"])
        if row.get("frame_t") is None:
            return read_image(path)
        from .video import frame_at
        f = frame_at(path, float(row["frame_t"]))
        if f is None or abs(f.t - float(row["frame_t"])) > 0.05:
            raise LearningError(f"The frame at {row['frame_t']:.2f} s of {row.get('rel_path') or path.name} cannot be "
                                "read any more.")
        return f.image

    def create_dataset(self, name: str, author: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                       block_days: int = 1, seed: int = 0) -> dict:
        if not name.strip():
            raise LearningError("Give the dataset version a name.")
        if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-6:
            raise LearningError("Split ratios must be three non-negative numbers that add up to 1.")
        if block_days < 1 or block_days > 366:
            raise LearningError("A date block is 1 to 366 days.")
        prev = self.con.execute("SELECT id FROM training_dataset ORDER BY id DESC LIMIT 1").fetchone()
        frozen = {}
        if prev:
            # Groups already in a test set stay in test (the test set is frozen across versions).
            for r in self.con.execute("SELECT DISTINCT view_key, block FROM dataset_member WHERE split='test'"):
                frozen[(r["view_key"], r["block"])] = "test"
            for r in self.con.execute("SELECT DISTINCT d.source_version_id FROM dataset_member d WHERE d.split='test'"):
                frozen[("vid", str(r[0]))] = "test"
        members, seen_sha, dupes, skipped = [], {}, 0, {"bulk": 0, "none": 0, "no_labels": 0}
        for r in self._images():
            for lab, frame_t, sha in self._samples(r):
                if lab.review in ("bulk", "none"):
                    skipped[lab.review] += 1
                    continue
                if lab.review == "partial" and not lab.positives:
                    skipped["no_labels"] += 1
                    continue
                if sha in seen_sha:
                    dupes += 1            # identical bytes are one sample, kept with the first copy
                    continue
                seen_sha[sha] = r["id"]
                g = self._group(r, block_days)
                members.append((r, lab, g, frame_t, sha))
        groups = sorted({m[2] for m in members})
        split_of = {}
        for g in groups:
            if g in frozen:
                split_of[g] = "test"
                continue
            b = self._bucket(seed, f"{g[0]}|{g[1]}")
            split_of[g] = "train" if b < ratios[0] else ("val" if b < ratios[0] + ratios[1] else "test")
        cond = {r[0]: r[1] for r in self.con.execute("SELECT source_version_id, condition FROM image_condition")}
        stats: dict = {"images": len(members), "groups": len(groups), "duplicates_dropped": dupes, "skipped": skipped,
                       "frames": sum(1 for m in members if m[3] is not None),
                       "recordings": len({m[0]["id"] for m in members if m[3] is not None}),
                       "splits": {s: {"images": 0, "groups": 0, "objects": 0, "full": 0, "partial": 0} for s in SPLITS},
                       "by_condition": {}, "by_view": {}}
        for g in groups:
            stats["splits"][split_of[g]]["groups"] += 1
        self.con.execute("BEGIN")
        try:
            did = self.con.execute(
                "INSERT INTO training_dataset(name,task,params,stats,frozen_test_from,author,created_at) VALUES (?,?,?,?,?,?,?)",
                (name.strip(), "vehicle_detection", json.dumps({"ratios": list(ratios), "block_days": block_days, "seed": seed,
                                                                "classes": list(TRAINABLE),
                                                                "review_marks": self._review_marks(),
                                                                **({"frames": {"per_event": FRAMES_PER_EVENT,
                                                                               "background_every_s": BACKGROUND_EVERY_S,
                                                                               "min_gap_s": FRAME_MIN_GAP_S,
                                                                               "per_recording": FRAMES_PER_RECORDING}}
                                                                   if self.video else {})}),
                 "{}", prev[0] if prev else None, author, utcnow())).lastrowid
            for r, lab, g, frame_t, sha in members:
                sp = split_of[g]
                st = stats["splits"][sp]
                st["images"] += 1
                st["objects"] += len(lab.positives)
                st[lab.review] += 1
                c = cond.get(r["id"]) or "unknown"
                stats["by_condition"].setdefault(c, {s: 0 for s in SPLITS})[sp] += 1
                stats["by_view"].setdefault(g[0], {s: 0 for s in SPLITS})[sp] += 1
                self.con.execute(
                    "INSERT INTO dataset_member(dataset_id,source_version_id,frame_t,sha256,view_key,block,split,review,"
                    "condition,labels,last_revision_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (did, r["id"], frame_t, sha, g[0], g[1], sp, lab.review, cond.get(r["id"]),
                     json.dumps({"positives": lab.positives, "ignore_boxes": lab.ignore_boxes}), lab.last_revision_id))
            self.con.execute("UPDATE training_dataset SET stats=? WHERE id=?", (json.dumps(stats), did))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return self.dataset(did)

    def datasets(self) -> list[dict]:
        return [self._ds_row(r) for r in self.con.execute("SELECT * FROM training_dataset ORDER BY id DESC")]

    @staticmethod
    def _ds_row(r) -> dict:
        return {"id": r["id"], "name": r["name"], "task": r["task"], "params": json.loads(r["params"]),
                "stats": json.loads(r["stats"]), "frozen_test_from": r["frozen_test_from"], "author": r["author"],
                "created_at": r["created_at"]}

    def dataset(self, dataset_id: int) -> dict:
        r = self.con.execute("SELECT * FROM training_dataset WHERE id=?", (dataset_id,)).fetchone()
        if r is None:
            raise LearningError("No such dataset version.")
        return self._ds_row(r) | {"leakage": self.leakage(dataset_id)}

    def leakage(self, dataset_id: int) -> list[str]:
        """Problems that would let test images leak into training. An empty list means none found."""
        issues = []
        for r in self.con.execute(
                "SELECT view_key, block, GROUP_CONCAT(DISTINCT split) s FROM dataset_member WHERE dataset_id=? "
                "GROUP BY view_key, block HAVING COUNT(DISTINCT split) > 1", (dataset_id,)):
            issues.append(f"{r['view_key']} on {r['block']} is in several splits ({r['s']})")
        for r in self.con.execute(
                "SELECT sha256, COUNT(*) n FROM dataset_member WHERE dataset_id=? GROUP BY sha256 HAVING n > 1", (dataset_id,)):
            issues.append(f"identical image bytes appear {r['n']} times ({r['sha256'][:12]}…)")
        prev_test = self.con.execute(
            "SELECT DISTINCT d.view_key, d.block FROM dataset_member d WHERE d.split='test' AND d.dataset_id<?", (dataset_id,))
        for r in prev_test:
            bad = self.con.execute("SELECT COUNT(*) FROM dataset_member WHERE dataset_id=? AND view_key=? AND block=? "
                                   "AND split!='test'", (dataset_id, r["view_key"], r["block"])).fetchone()[0]
            if bad:
                issues.append(f"{r['view_key']} on {r['block']} was in an earlier test set but is now used for {bad} "
                              "training or validation image(s)")
        return issues

    def members(self, dataset_id: int, split: str | None = None) -> list[dict]:
        q = "SELECT d.*, f.rel_path FROM dataset_member d JOIN source_version v ON v.id=d.source_version_id " \
            "JOIN source_file f ON f.id=v.source_file_id WHERE d.dataset_id=?"
        args: list = [dataset_id]
        if split:
            q += " AND d.split=?"
            args.append(split)
        return [dict(r) | {"labels": json.loads(r["labels"])} for r in self.con.execute(q + " ORDER BY d.source_version_id, d.frame_t", args)]

    # ---------- several dataset versions as one ----------
    def _open(self, mission_id: str) -> "LearningService":
        if mission_id == self.m.id:
            return self
        cache = self.__dict__.setdefault("_others", {})
        if mission_id not in cache:
            try:
                m = self.m.root.open_mission(mission_id)
            except KeyError:
                raise LearningError("A chosen dataset version belongs to a mission that no longer exists.") from None
            cache[mission_id] = LearningService(m)
        return cache[mission_id]

    def all_datasets(self) -> list[dict]:
        """Dataset versions of every mission on this computer (still images and video), this mission's first."""
        out = [d | {"mission_id": self.m.id, "mission": self.m.config().name} for d in self.datasets()]
        for r in self.m.root.missions():
            if r["id"] == self.m.id or r["state"] != "active":
                continue
            try:
                ls = self._open(r["id"])
                out += [d | {"mission_id": r["id"], "mission": r["name"]} for d in ls.datasets()]
            except Exception:          # a mission that cannot be opened just offers nothing
                continue
        return out

    def combine(self, refs, split: str | None = None, strict: bool = False) -> tuple[list[dict], list[dict]]:
        """The members of several dataset versions as one set, with the leakage rules kept across them.

        - A camera-view-and-day group that is in a test set in any chosen version is test in the combination.
        - Otherwise a group takes its split from the newest chosen version that holds it.
        - Identical image bytes are one sample, with the labels of the newest version; test if any copy is test.
        Returns (items, sources): items are {"ls", "row", "split", "mission_id"}.
        """
        refs = _refs(refs)
        if not refs:
            raise LearningError("Choose at least one dataset version.")
        rows, sources = [], []
        for mid, did in refs:
            ls = self._open(mid)
            ds = ls.dataset(did)
            if strict and ds["leakage"]:
                raise LearningError(f"Dataset version “{ds['name']}” has leakage problems; it cannot be used: "
                                    + "; ".join(ds["leakage"]))
            sources.append({"mission_id": mid, "mission": ls.m.config().name, "dataset_id": did, "name": ds["name"]})
            rows += [(ds["created_at"], mid, did, ls, r) for r in ls.members(did)]
        rows.sort(key=lambda x: (x[0], x[1], x[2]))       # oldest first: later versions override
        group: dict[tuple, str] = {}
        for _, mid, _, _, r in rows:
            g = (mid, r["view_key"], r["block"])
            if group.get(g) != "test":
                group[g] = r["split"]
        test_sha = {r["sha256"] for _, mid, _, _, r in rows if group[(mid, r["view_key"], r["block"])] == "test"}
        newest: dict[str, tuple] = {}
        for item in rows:
            newest[item[4]["sha256"]] = item
        items = []
        for _, mid, _, ls, r in newest.values():
            sp = "test" if r["sha256"] in test_sha else group[(mid, r["view_key"], r["block"])]
            if split is None or sp == split:
                items.append({"ls": ls, "row": r, "split": sp, "mission_id": mid})
        items.sort(key=lambda it: (it["mission_id"] != self.m.id, it["mission_id"], it["row"]["source_version_id"]))
        return items, sources

    def export_combined(self, refs, out_dir: Path | None = None) -> Path:
        """Export several dataset versions as one dataset (for training on all of them together)."""
        refs = _refs(refs)
        if len(refs) == 1 and refs[0][0] == self.m.id:
            return self.export_dataset(refs[0][1], out_dir)
        items, sources = self.combine(refs, strict=True)
        name = " + ".join(s["name"] if s["mission_id"] == self.m.id else f"{s['mission']}: {s['name']}" for s in sources)
        stats = {"images": len(items), "splits": {sp: {"images": sum(1 for i in items if i["split"] == sp),
                                                       "objects": sum(len(i["row"]["labels"]["positives"]) for i in items
                                                                      if i["split"] == sp)} for sp in SPLITS}}
        return self._write_export(items, Path(out_dir or self.m.config().output_dir) / "datasets",
                                  f"{self.m.config().name}_combined_{len(sources)}",
                                  {"mission_id": self.m.id, "dataset_id": None, "datasets": sources, "name": name,
                                   "params": {"combined": True}, "stats": stats, "created_at": utcnow()})

    def export_dataset(self, dataset_id: int, out_dir: Path | None = None) -> Path:
        """Write images and COCO-style annotations per split for training on another machine. Never overwrites."""
        ds = self.dataset(dataset_id)
        if ds["leakage"]:
            raise LearningError("This dataset version has leakage problems; it is not exported: " + "; ".join(ds["leakage"]))
        items = [{"ls": self, "row": r, "split": r["split"], "mission_id": self.m.id} for r in self.members(dataset_id)]
        return self._write_export(items, Path(out_dir or self.m.config().output_dir) / "datasets",
                                  f"{self.m.config().name}_ds{dataset_id}_{ds['name']}",
                                  {"mission_id": self.m.id, "dataset_id": dataset_id, "name": ds["name"],
                                   "params": ds["params"], "stats": ds["stats"], "created_at": ds["created_at"]})

    def _write_export(self, items: list[dict], base: Path, stem: str, manifest: dict) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
        final, n = base / stem, 2
        while final.exists() or final.with_name(final.name + ".partial").exists():
            final, n = base / f"{stem}_{n}", n + 1
        tmp = final.with_name(final.name + ".partial")
        (tmp / "images").mkdir(parents=True)
        cats = [{"id": i + 1, "name": c} for i, c in enumerate(TRAINABLE)]
        cid = {c["name"]: c["id"] for c in cats}
        files = []
        multi = len({it["mission_id"] for it in items}) > 1
        ann_id = 0
        for split in SPLITS:
            images, anns = [], []
            for it in (i for i in items if i["split"] == split):
                ls, mrow = it["ls"], it["row"]
                vid = mrow["source_version_id"]
                frame_t = mrow.get("frame_t")
                img_id = vid if not multi and frame_t is None else len(images) + 1 + 10_000_000 * (SPLITS.index(split) + 1)
                src = ls.views._path(vid)
                prefix = f"{ls.m.id[:8]}_" if multi else ""
                if frame_t is None:
                    name = prefix + f"{vid}_{mrow['sha256'][:12]}{src.suffix.lower()}"
                    shutil.copy2(src, tmp / "images" / name)
                    img = read_image(src)
                else:                                   # a frame of a recording, written as a still image
                    import cv2
                    img = ls.image_of(mrow)
                    name = prefix + f"{vid}_{int(round(frame_t * 1000)):09d}ms_{mrow['sha256'][:12]}.png"
                    ok, buf = cv2.imencode(".png", img)
                    if not ok:
                        raise LearningError(f"Could not write the frame at {frame_t:.2f} s of {mrow['rel_path']}.")
                    (tmp / "images" / name).write_bytes(buf.tobytes())
                view = ls.views.current_view(vid)
                regions = RegionSet.load(ls.con, view) if view is not None else RegionSet([])
                images.append({"id": img_id, "file_name": f"images/{name}", "width": int(img.shape[1]),
                               "height": int(img.shape[0]), "mission_id": ls.m.id, "source_version_id": vid,
                               "sha256": mrow["sha256"], "source_file": mrow["rel_path"], "frame_t": frame_t,
                               "view": mrow["view_key"],
                               "date_block": mrow["block"], "condition": mrow["condition"], "review": mrow["review"],
                               "unlabelled_area": "background inside included regions" if mrow["review"] == "full" else "ignore",
                               "include_regions": [r.polygon for r in regions.include],
                               "exclude_regions": [r.polygon for r in regions.exclude],
                               "ignore_boxes": mrow["labels"]["ignore_boxes"]})
                for p in mrow["labels"]["positives"]:
                    x1, y1, x2, y2 = p["box"]
                    ann_id += 1
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": cid.get(trainable(p["cls"]) or "car", cid["car"]),
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                 "label_source": p["source"]})
            f = tmp / f"annotations_{split}.json"
            f.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}, ensure_ascii=False),
                         encoding="utf-8")
            files.append(f.name)
        manifest = manifest | {"exported_at": utcnow(), "files": files, "rules": self.eligibility()["rules"]}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if final.exists():
            raise LearningError(f"{final.name} appeared while writing; nothing was overwritten.")
        tmp.rename(final)
        return final

    # ---------- evaluation ----------
    def evaluate(self, dataset_id: int | None, detector, author: str, split: str = "test", iou: float = 0.5,
                 control: RunControl | None = None, progress: Callable[[int, int], None] | None = None,
                 datasets=None) -> dict:
        """Run a detector on one split of one dataset version, or of several together (datasets: references
        (mission_id, dataset_id), from this mission or others), and score it against the labels."""
        if split not in SPLITS:
            raise LearningError(f"Split must be one of {SPLITS}.")
        refs = _refs(datasets) if datasets else [(self.m.id, int(dataset_id))]
        mem, sources = self.combine(refs, split)
        if not mem:
            raise LearningError(f"The {split} split of {'this dataset version' if len(sources) == 1 else 'these dataset versions'} "
                                "is empty.")
        multi = len({s["mission_id"] for s in sources}) > 1
        home = next((s["dataset_id"] for s in sources if s["mission_id"] == self.m.id), None)
        control = control or RunControl()
        desc = detector.describe()
        eid = self.con.execute(
            "INSERT INTO evaluation(dataset_id,split,model,state,author,created_at,sources) VALUES (?,?,?,?,?,?,?)",
            (home, split, json.dumps(desc), "running", author, utcnow(),
             json.dumps([{"mission_id": s["mission_id"], "dataset_id": s["dataset_id"]} for s in sources]))).lastrowid
        # Labels of a class the detector does not look for (people, for a vehicle-only detector) are neither found
        # nor missed: they are scored as "ignore".
        looks_for = {trainable(c) for c in (getattr(detector, "classes", None) or {}).values()} - {None} \
            or set(VEHICLE_CLASSES)
        per_image, took = [], []
        try:
            for n, it in enumerate(mem, 1):
                control.checkpoint()
                ls, mrow = it["ls"], it["row"] | {"split": it["split"]}
                if multi:
                    mrow["view_key"] = f"{ls.m.config().name}: {mrow['view_key']}"
                vid = mrow["source_version_id"]
                img = ls.image_of(mrow)
                t0 = time.perf_counter()
                found = detector.detect(img)
                took.append(time.perf_counter() - t0)
                dets = [(list(map(float, d.box)), trainable(d.cls) or d.cls, float(d.score)) for d in found]
                view = ls.views.current_view(vid)
                regions = RegionSet.load(ls.con, view) if view is not None and ls.views.regions_applicable(view) else RegionSet([])
                per_image.append(self._score_image(mrow, dets, regions, iou, looks_for))
                if progress:
                    progress(n, len(mem))
        except Exception as exc:
            from .process import RunCancelled
            state = "cancelled" if isinstance(exc, RunCancelled) else "failed"
            self.con.execute("UPDATE evaluation SET state=?, error=?, finished_at=? WHERE id=?",
                             (state, f"{type(exc).__name__}: {exc}"[:1000], utcnow(), eid))
            raise
        warm = took[1:] or took                  # the first picture also loads the model onto the device
        report = {"iou": iou, "split": split, "images": len(per_image),
                  "ms_per_image": round(1000 * sum(warm) / len(warm), 1) if warm else None,
                  "overall": _summarise(per_image),
                  "by_condition": {k: _summarise([p for p in per_image if p["condition"] == k])
                                   for k in sorted({p["condition"] for p in per_image})},
                  "by_view": {k: _summarise([p for p in per_image if p["view"] == k])
                              for k in sorted({p["view"] for p in per_image})},
                  "notes": ["Precision, AP and count error use fully reviewed images only; partial reviews add recall "
                            "for their corrected objects.",
                            "AP is limited by the detector's own score threshold: boxes below it are never seen."]}
        self.con.execute("UPDATE evaluation SET state='completed', report=?, finished_at=? WHERE id=?",
                         (json.dumps(report), utcnow(), eid))
        return self.evaluation(eid)

    @staticmethod
    def _score_image(mrow: dict, dets, regions: RegionSet, iou: float, looks_for: set | None = None) -> dict:
        lab = mrow["labels"]
        full = mrow["review"] == "full"
        looks_for = looks_for or set(VEHICLE_CLASSES)
        refs = [p | {"cls": trainable(p["cls"]) or p["cls"]} for p in lab["positives"]
                if regions.contains_box(tuple(p["box"])) and trainable(p["cls"]) in looks_for]
        ignore = list(lab["ignore_boxes"]) + [p["box"] for p in lab["positives"] if trainable(p["cls"]) not in looks_for]
        dets = [d for d in dets if d[1] in looks_for]
        lab = {**lab, "ignore_boxes": ignore}
        preds = sorted([d for d in dets if regions.contains_box(tuple(d[0]))], key=lambda d: -d[2])
        used, scored, tp, cls_ok = set(), [], 0, 0
        for box, cls, score in preds:
            best, bi = 0.0, None
            for i, r in enumerate(refs):
                if i in used:
                    continue
                v = _iou(box, r["box"])
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= iou:
                used.add(bi)
                tp += 1
                cls_ok += int(cls == refs[bi]["cls"])
                scored.append((score, 1))
                continue
            if any(_iou(box, ig) >= iou or (ig[0] <= (box[0] + box[2]) / 2 <= ig[2] and ig[1] <= (box[1] + box[3]) / 2 <= ig[3])
                   for ig in lab["ignore_boxes"]):
                continue                      # an unresolved night candidate: neither right nor wrong
            if full:
                scored.append((score, 0))     # a false positive only counts where the whole image was reviewed
        return {"id": mrow["source_version_id"], "frame_t": mrow.get("frame_t"), "full": full, "condition": mrow["condition"] or "unknown",
                "view": mrow["view_key"], "refs": len(refs), "tp": tp, "fn": len(refs) - tp,
                "fp": sum(1 for _, t in scored if t == 0), "cls_ok": cls_ok, "scored": scored,
                "pred_count": len([p for p in preds]), "count_error": (len(preds) - len(refs)) if full else None}

    def evaluations(self, dataset_id: int | None = None) -> list[dict]:
        q, a = "SELECT * FROM evaluation", []
        if dataset_id is not None:
            q, a = q + " WHERE dataset_id=?", [dataset_id]
        return [self._ev_row(r) for r in self.con.execute(q + " ORDER BY id DESC", a)]

    def _ev_row(self, r) -> dict:
        sources = json.loads(r["sources"]) if r["sources"] else [{"mission_id": self.m.id, "dataset_id": r["dataset_id"]}]
        report = json.loads(r["report"]) if r["report"] else None
        return {"id": r["id"], "dataset_id": r["dataset_id"], "sources": sources, "split": r["split"],
                "model": json.loads(r["model"]), "state": r["state"], "report": report, "error": r["error"],
                "author": r["author"], "created_at": r["created_at"], "finished_at": r["finished_at"],
                "data_key": data_key(sources, r["split"], (report or {}).get("iou", 0.5))}

    def find_evaluation(self, name: str, version: str, sources: list[dict], split: str, iou: float = 0.5) -> dict | None:
        """The newest completed evaluation of this model version on exactly these images, if there is one."""
        key = data_key(sources, split, iou)
        for ev in self.evaluations():
            md = (ev["model"] or {}).get("model") or {}
            if ev["state"] == "completed" and ev["data_key"] == key and (md.get("name"), md.get("version")) == (name, version):
                return ev
        return None

    # ---------- Run mode "full auto" ----------
    _REVIEW_TABLES = (("review_decision", " AND bulk_approval_id IS NULL"), ("annotation_revision", ""),
                      ("candidate_revision", ""), ("event_revision", ""), ("recording_review", ""))

    def _review_marks(self) -> dict:
        return {t: self.con.execute(f"SELECT COALESCE(MAX(id), 0) FROM {t}").fetchone()[0] for t, _ in self._REVIEW_TABLES}

    def reviews_since_last_dataset(self) -> int:
        """Operator decisions (on images, objects, night candidates, video events and recordings) made since the
        newest dataset version was frozen: the automatic loop starts a round when enough have built up."""
        last = self.con.execute("SELECT params, created_at FROM training_dataset ORDER BY id DESC LIMIT 1").fetchone()
        marks = json.loads(last["params"]).get("review_marks") if last else None
        n = 0
        for table, extra in self._REVIEW_TABLES:
            if marks is not None:
                q, a = f"SELECT COUNT(*) FROM {table} WHERE id > ?{extra}", (marks.get(table, 0),)
            else:
                q, a = f"SELECT COUNT(*) FROM {table} WHERE created_at > ?{extra}", (last["created_at"] if last else "",)
            n += self.con.execute(q, a).fetchone()[0]
        return n

    def auto_log(self, action: str, **detail) -> None:
        self.con.execute("INSERT INTO auto_learning_log(action,detail,created_at) VALUES (?,?,?)",
                         (action, json.dumps(detail, ensure_ascii=False), utcnow()))

    def auto_history(self, limit: int = 30) -> list[dict]:
        return [{"id": r["id"], "action": r["action"], "detail": json.loads(r["detail"]), "created_at": r["created_at"]}
                for r in self.con.execute("SELECT * FROM auto_learning_log ORDER BY id DESC LIMIT ?", (limit,))]

    # ---------- training rounds ----------
    def start_training_run(self, sources, start_from: str, epochs: int, author: str) -> int:
        refs = [{"mission_id": m, "dataset_id": d} for m, d in _refs(sources)]
        return self.con.execute(
            "INSERT INTO training_run(sources,start_from,epochs,state,author,created_at) VALUES (?,?,?,?,?,?)",
            (json.dumps(refs), start_from, int(epochs), "running", author, utcnow())).lastrowid

    def finish_training_run(self, run_id: int, **f) -> None:
        cols = ("state", "model_name", "model_version", "evaluation_id", "baseline_evaluation_id", "improved", "error")
        sets = {k: f[k] for k in cols if k in f}
        if "history" in f:
            sets["history"] = json.dumps(f["history"])
        if sets.get("state") in ("finished", "failed", "cancelled"):
            sets["finished_at"] = utcnow()
        if "improved" in sets and sets["improved"] is not None:
            sets["improved"] = int(bool(sets["improved"]))
        if sets:
            self.con.execute(f"UPDATE training_run SET {', '.join(k + '=?' for k in sets)} WHERE id=?",
                             (*sets.values(), run_id))

    def training_runs(self) -> list[dict]:
        out = []
        for r in self.con.execute("SELECT * FROM training_run ORDER BY id DESC"):
            d = dict(r)
            d["sources"] = json.loads(d["sources"])
            d["history"] = json.loads(d["history"]) if d["history"] else []
            d["improved"] = None if d["improved"] is None else bool(d["improved"])
            out.append(d)
        return out

    def carry_over(self) -> dict:
        """Rounds since the last one that improved and did not improve: their data goes into the next round too."""
        runs = []
        for r in self.training_runs():                    # newest first
            if r["state"] != "finished":
                continue
            if r["improved"]:
                break
            if r["improved"] is False:
                runs.append(r)
        refs: list[dict] = []
        for r in reversed(runs):
            for s in r["sources"]:
                if s not in refs:
                    refs.append(s)
        return {"runs": [r["id"] for r in reversed(runs)], "datasets": refs}

    def evaluation(self, eid: int) -> dict:
        r = self.con.execute("SELECT * FROM evaluation WHERE id=?", (eid,)).fetchone()
        if r is None:
            raise LearningError("No such evaluation.")
        return self._ev_row(r)


def _summarise(items: list[dict]) -> dict:
    full = [p for p in items if p["full"]]
    tp_all = sum(p["tp"] for p in items)
    refs_all = sum(p["refs"] for p in items)
    tp_f, fp_f, refs_f = sum(p["tp"] for p in full), sum(p["fp"] for p in full), sum(p["refs"] for p in full)
    precision = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else None
    recall = tp_all / refs_all if refs_all else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision == 0 or recall == 0 else None)
    scored = sorted((s for p in full for s in p["scored"]), key=lambda x: -x[0])
    ap = None
    if refs_f:
        tps = fps = 0
        pts = []
        for _, t in scored:
            tps += t
            fps += 1 - t
            pts.append((tps / refs_f, tps / (tps + fps)))
        # All-point interpolated AP.
        ap, prev_r = 0.0, 0.0
        for i, (r, _) in enumerate(pts):
            pmax = max(p for _, p in pts[i:])
            ap += (r - prev_r) * pmax
            prev_r = r
    errs = [p["count_error"] for p in full if p["count_error"] is not None]
    return {"images": len(items), "full_images": len(full), "partial_images": len(items) - len(full),
            "references": refs_all, "true_positives": tp_all, "false_positives_full": fp_f,
            "precision": _r(precision), "recall": _r(recall), "f1": _r(f1), "ap50": _r(ap),
            "class_agreement": _r(sum(p["cls_ok"] for p in items) / tp_all) if tp_all else None,
            "count_mae": _r(sum(abs(e) for e in errs) / len(errs)) if errs else None,
            "count_bias": _r(sum(errs) / len(errs)) if errs else None}


def _r(v):
    return None if v is None else round(float(v), 4)


# ------------------------------------------------------------------ production pointer (control.db)

class ModelRegistry:
    """Production model per task. Only an operator's promotion or rollback moves it."""

    def __init__(self, data_root: Path | str):
        self.con = connect(Path(data_root) / "control.db")
        self.con.executescript(CONTROL_SCHEMA)

    def history(self, task: str) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM model_promotion WHERE task=? ORDER BY id DESC", (task,))]

    def promote(self, task: str, name: str, version: str, author: str, mission_id: str | None = None,
                evaluation_id: int | None = None, note: str | None = None, models_root: Path | None = None) -> dict:
        from . import models
        pack = models.find(name, version, models_root)       # verifies the checksum
        if pack.card.get("task") != task:
            raise LearningError(f"{name} {version} is a {pack.card.get('task')} model, not {task}.")
        if evaluation_id is None or mission_id is None:
            raise LearningError("Promote a model only with the evaluation that supports it.")
        self.con.execute(
            "INSERT INTO model_promotion(task,name,version,sha256,action,mission_id,evaluation_id,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (task, name, version, pack.sha256, "promote", mission_id, evaluation_id, author,
                                             note, utcnow()))
        return self.production(task)

    def _stack(self, task: str) -> list[dict]:
        """Replay the history: a promotion pushes a model, a rollback pops back to the one before it."""
        stack: list[dict] = []
        for r in reversed(self.history(task)):
            if r["action"] == "promote":
                stack.append(r)
            elif stack:
                stack.pop()
        return stack

    def production(self, task: str) -> dict | None:
        st = self._stack(task)
        return st[-1] if st else None

    def rollback(self, task: str, author: str, note: str | None = None) -> dict | None:
        st = self._stack(task)
        if not st:
            raise LearningError("Nothing has been promoted for this task; the built-in default is in use.")
        cur = st[-1]
        back = st[-2] if len(st) > 1 else None
        self.con.execute(
            "INSERT INTO model_promotion(task,name,version,sha256,action,mission_id,evaluation_id,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (task, cur["name"], cur["version"], cur["sha256"], "rollback",
                                             cur["mission_id"], cur["evaluation_id"], author, note, utcnow()))
        return back            # None: the built-in default is back in use


# ------------------------------------------------------------------ saved models (control.db)

class ModelLibrary:
    """Models kept by a name of the operator's choosing, frozen, and usable by any mission.

    Saving freezes the model: its files are made read-only, and its checksum is pinned here and checked every time the
    model is loaded. Training from a saved model always writes a new model; the saved one never changes.
    """

    def __init__(self, data_root: Path | str, models_root: Path | None = None):
        self.con = connect(Path(data_root) / "control.db")
        self.con.executescript(CONTROL_SCHEMA)
        self.models_root = models_root

    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
        return d

    def saved(self) -> list[dict]:
        from . import models
        have = {(m["name"], m["version"]): m for m in models.installed(root=self.models_root)}
        out = []
        for r in self.con.execute("SELECT * FROM saved_model ORDER BY created_at DESC, id DESC"):
            d = self._row(r)
            card = have.get((d["model_name"], d["model_version"]))
            d["installed"] = card is not None and card.get("sha256") == d["sha256"]
            d["history"] = (card or {}).get("history") or []
            out.append(d)
        return out

    def get(self, name: str) -> dict | None:
        r = self.con.execute("SELECT * FROM saved_model WHERE name=? COLLATE NOCASE", (name.strip(),)).fetchone()
        return self._row(r) if r else None

    def of_pack(self, model_name: str, model_version: str) -> dict | None:
        r = self.con.execute("SELECT * FROM saved_model WHERE model_name=? AND model_version=?",
                             (model_name, model_version)).fetchone()
        return self._row(r) if r else None

    def save(self, name: str, model_name: str, model_version: str, author: str, description: str | None = None,
             mission_id: str | None = None, evaluation_id: int | None = None, summary: dict | None = None) -> dict:
        from . import models
        name = " ".join((name or "").split())
        if not name:
            raise LearningError("Give the model a name, for example “Night cars, Route 6”.")
        if len(name) > 80:
            raise LearningError("Keep the name under 80 characters.")
        if self.get(name):
            raise LearningError(f"A saved model is already called “{name}”. Choose another name.")
        dup = self.of_pack(model_name, model_version)
        if dup:
            raise LearningError(f"{model_name} {model_version} is already saved as “{dup['name']}”.")
        try:
            pack = models.find(model_name, model_version, self.models_root)     # verifies the checksum
        except models.ModelPackError as exc:
            raise LearningError(str(exc)) from None
        if pack.card.get("evaluation_only"):
            raise LearningError(f"{model_name} {model_version} is for evaluation only and cannot be saved for use.")
        freeze(pack.path)
        self.con.execute(
            "INSERT INTO saved_model(name,description,task,model_name,model_version,sha256,mission_id,evaluation_id,summary,"
            "author,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (name, (description or "").strip() or None, pack.card.get("task") or "vehicle_detection", pack.name, pack.version,
             pack.sha256, mission_id, evaluation_id, json.dumps(summary or {}), author, utcnow()))
        return self.get(name)

    def remove(self, name: str) -> None:
        """Take a model off the saved list. Its files stay where they are (still read-only)."""
        if not self.get(name):
            raise LearningError(f"There is no saved model called “{name}”.")
        self.con.execute("DELETE FROM saved_model WHERE name=? COLLATE NOCASE", (name.strip(),))

    def pack(self, name: str):
        """The model pack of a saved model, checked against the checksum pinned when it was saved."""
        from . import models
        d = self.get(name)
        if d is None:
            raise LearningError(f"There is no saved model called “{name}”.")
        try:
            pack = models.find(d["model_name"], d["model_version"], self.models_root)
        except models.ModelPackError as exc:
            raise LearningError(f"The saved model “{d['name']}” cannot be loaded: {exc}") from None
        if pack.sha256 != d["sha256"]:
            raise LearningError(f"The files of the saved model “{d['name']}” changed after it was saved; it is "
                                "not used. Import the original model again.")
        return pack


def freeze(folder: Path) -> None:
    """Make a model pack's files read-only (best effort: some file systems keep no such flag)."""
    for f in Path(folder).iterdir():
        if f.is_file():
            try:
                os.chmod(f, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except OSError:
                pass
