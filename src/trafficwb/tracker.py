"""Tracking within one recording (VID-02, VID-03).

ByteTrack/BoT-SORT style (MIT-licensed methods, our own code): a Kalman filter
predicts each box, strong boxes are matched first and weak ones then keep
existing tracks alive, and on a moving camera the camera's own motion is taken
out of every track before matching. The earlier IoU tracker with
constant-velocity prediction remains for checkpoints written before. It is
deterministic and its whole state is plain data, so it can be saved at a
checkpoint and resumed with identical results.

Rules:
- A track starts as tentative and is confirmed after `min_hits` matched
  frames. Only confirmed tracks become events, so one-frame false detections
  do not create events.
- Association ignores the class within a family (the detector may call the
  same vehicle car in one frame and truck in the next) but never crosses
  families: a person is not matched to the bicycle they ride. The event class
  is the score-weighted vote, and its share is kept as the class uncertainty.
- Occlusion: a track that is not matched keeps its predicted motion for up to
  `gap_s` seconds. If it is matched again within that time it is the same
  track. After that the track is closed.
- Re-entry: a new track that starts within `reentry_s` of a closed track and
  near where that track was last predicted is flagged "possible duplicate" of
  it. The tracker never merges them silently; an operator decides.
- A scene cut or the end of the recording closes every track. No identity is
  carried across recordings or cameras.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import kalman
from .detect import family


@dataclass
class TrackerParams:
    iou_min: float = 0.25
    min_hits: int = 3
    gap_s: float = 1.5
    reentry_s: float = 3.0
    reentry_dist: float = 2.0      # in box widths
    edge_px: float = 4.0
    velocity_smoothing: float = 0.5
    # Second chance for boxes seen seldom (SAM 3.1 looks once a second): a box that overlaps no track still joins
    # the nearest track of its family whose predicted centre is within this many box sizes per second since it was
    # last seen, if the sizes are within a factor of 2. 0 turns it off (the standard detector sees every frame).
    jump_per_s: float = 0.0
    # "byte" (ByteTrack/BoT-SORT style): a Kalman filter predicts each box; boxes scoring at least high_score are
    # matched first and may start tracks; weaker boxes (down to the detector's threshold) only keep existing tracks
    # alive, at IoU low_iou_min or more. "iou": the earlier tracker (constant velocity), kept for checkpoints
    # written before the change.
    algo: str = "byte"
    high_score: float = 0.30
    low_iou_min: float = 0.5
    frame_s: float = 0.2           # Kalman time step: one analysed frame


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass
class Track:
    key: str
    box: list[float]
    t: float                        # time of the last match
    v: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])   # px/s for x1, y1, x2, y2
    hits: int = 1
    state: str = "tentative"        # tentative | confirmed
    first_t: float = 0.0
    first_box: list[float] = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    best_area: float = 0.0
    best_t: float = 0.0
    best_box: list[float] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    family: str = "vehicle"         # classes this track can match (detect.family); older checkpoints: vehicles
    mean: list | None = None        # Kalman state ("byte"), at time kf_t
    cov: list | None = None
    kf_t: float = 0.0

    def predict(self, t: float) -> list[float]:
        """Constant-velocity prediction from the last match (the "iou" tracker)."""
        dt = t - self.t
        return [b + v * dt for b, v in zip(self.box, self.v)]


def warp_box(box, a) -> list[float]:
    """A box moved by a 2x3 affine (camera motion): its corners mapped, then the enclosing box."""
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float) @ np.asarray(a, float).T
    return [round(float(v), 2) for v in (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())]


@dataclass
class Closed:
    """A track that ended. Confirmed ones become events."""
    track: Track
    reason: str                     # left_frame | lost | scene_cut | end_of_recording
    confirmed: bool


class Tracker:
    def __init__(self, width: int, height: int, params: TrackerParams | None = None):
        self.w, self.h = width, height
        self.p = params or TrackerParams()
        self.tracks: list[Track] = []
        self.recent: list[dict] = []    # recently closed confirmed tracks, for re-entry checks
        self.counter = 0

    # ---------- state ----------
    def state(self) -> dict:
        return {"w": self.w, "h": self.h, "params": asdict(self.p), "counter": self.counter,
                "tracks": [asdict(t) for t in self.tracks], "recent": self.recent}

    @classmethod
    def from_state(cls, s: dict) -> "Tracker":
        params = dict(s["params"])
        params.setdefault("algo", "iou")          # checkpoints written before the Kalman tracker resume as they began
        tr = cls(s["w"], s["h"], TrackerParams(**params))
        tr.counter = s["counter"]
        tr.tracks = [Track(**t) for t in s["tracks"]]
        tr.recent = s["recent"]
        return tr

    # ---------- update ----------
    def _touches_edge(self, box) -> bool:
        e = self.p.edge_px
        return box[0] <= e or box[1] <= e or box[2] >= self.w - e or box[3] >= self.h - e

    def _predict(self, tr: Track, t: float) -> list[float]:
        if self.p.algo == "byte" and tr.mean is not None:
            m, _ = kalman.predict(tr.mean, tr.cov, (t - tr.kf_t) / self.p.frame_s)
            return kalman.z_to_box(m)
        return tr.predict(t)

    def _apply_motion(self, a) -> None:
        """The camera moved by ``a`` (2x3 affine) since the last analysed frame: move every track and every
        recently closed track with it, so only the objects' own motion is left to predict."""
        a = np.asarray(a, float)
        r = a[:, :2]
        for tr in self.tracks:
            tr.box = warp_box(tr.box, a)
            if tr.mean is not None:
                tr.mean, tr.cov = kalman.warp(tr.mean, tr.cov, a)
            vx1, vy1 = r @ np.array(tr.v[:2])
            vx2, vy2 = r @ np.array(tr.v[2:])
            tr.v = [round(float(x), 4) for x in (vx1, vy1, vx2, vy2)]
        for rec in self.recent:
            rec["box"] = warp_box(rec["box"], a)

    def update(self, t: float, dets: list[tuple[list[float], str, float]],
               motion=None) -> tuple[list[tuple[Track, list[float], str, float]], list[Closed], list[Track]]:
        """Feed one analysed frame. dets: (box, cls, score). ``motion``: the camera's motion since the previous
        analysed frame as a 2x3 affine (moving cameras), or None for a fixed camera.

        Returns (matched observations as (track, box, cls, score), closed tracks, newly created tracks).
        """
        byte = self.p.algo == "byte"
        if byte:
            for tr in self.tracks:            # every track advances to this frame, as in ByteTrack
                if tr.mean is not None:
                    m, c = kalman.predict(tr.mean, tr.cov, (t - tr.kf_t) / self.p.frame_s)
                    tr.mean, tr.cov, tr.kf_t = m.tolist(), c.ravel().tolist(), t
        if motion is not None:
            self._apply_motion(motion)
        preds = [self._predict(tr, t) for tr in self.tracks]
        strong = [j for j, d in enumerate(dets) if not byte or d[2] >= self.p.high_score]
        weak = [j for j, d in enumerate(dets) if byte and d[2] < self.p.high_score]
        pairs = []
        for i, pb in enumerate(preds):
            for j in strong:
                db, dcls, _ = dets[j]
                if family(dcls) != self.tracks[i].family:
                    continue
                v = iou(pb, db)
                if v >= self.p.iou_min:
                    pairs.append((-v, i, j))
        pairs.sort()
        used_t, used_d, obs = set(), set(), []
        pairs_matched: list[tuple[int, int]] = []
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            obs.append(self._match(self.tracks[i], dets[j], t))
            pairs_matched.append((i, j))
        n_iou = len(pairs_matched)
        if self.p.jump_per_s > 0:
            near = []
            for i, (tr, pb) in enumerate(zip(self.tracks, preds)):
                if i in used_t:
                    continue
                size = max(pb[2] - pb[0], pb[3] - pb[1], 1.0)
                reach = self.p.jump_per_s * max(t - tr.t, 1e-6) * size
                for j in strong:
                    db, dcls, _ = dets[j]
                    if j in used_d or family(dcls) != tr.family:
                        continue
                    dsize = max(db[2] - db[0], db[3] - db[1], 1.0)
                    dist = (((pb[0] + pb[2]) - (db[0] + db[2])) ** 2 + ((pb[1] + pb[3]) - (db[1] + db[3])) ** 2) ** 0.5 / 2
                    if dist <= reach and 0.5 <= dsize / size <= 2.0:
                        near.append((dist, i, j))
            for _, i, j in sorted(near):
                if i in used_t or j in used_d:
                    continue
                pairs_matched.append((i, j))
                used_t.add(i)
                used_d.add(j)
        if weak:                              # ByteTrack's second pass: weak boxes keep tracks alive
            low = []
            for i, pb in enumerate(preds):
                if i in used_t:
                    continue
                for j in weak:
                    db, dcls, _ = dets[j]
                    if family(dcls) == self.tracks[i].family:
                        v = iou(pb, db)
                        if v >= self.p.low_iou_min:
                            low.append((-v, i, j))
            for _, i, j in sorted(low):
                if i in used_t or j in used_d:
                    continue
                pairs_matched.append((i, j))
                used_t.add(i)
                used_d.add(j)
        for i, j in pairs_matched[n_iou:]:
            obs.append(self._match(self.tracks[i], dets[j], t))
        used_d |= set(weak)                   # a weak box never starts a track
        closed: list[Closed] = []
        keep: list[Track] = []
        for i, tr in enumerate(self.tracks):
            if i in used_t:
                keep.append(tr)
                continue
            if t - tr.t > self.p.gap_s:
                closed.append(self._close(tr, "left_frame" if self._touches_edge(tr.box) else "lost"))
            else:
                keep.append(tr)
        new: list[Track] = []
        for j, (db, cls, score) in enumerate(dets):
            if j in used_d:
                continue
            self.counter += 1
            box = [round(float(x), 2) for x in db]
            area = (db[2] - db[0]) * (db[3] - db[1])
            tr = Track(f"T{self.counter}", box, t, first_t=t, first_box=box, votes={cls: round(score, 4)},
                       best_area=round(area, 2), best_t=t, best_box=box, family=family(cls))
            if byte:
                tr.mean, tr.cov = kalman.initiate(box)
                tr.kf_t = t
            if self.p.min_hits <= 1:
                tr.state = "confirmed"
            for r in self.recent:
                if r.get("family", "vehicle") == tr.family and 0 <= t - r["t"] <= self.p.reentry_s:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    rx, ry = (r["box"][0] + r["box"][2]) / 2, (r["box"][1] + r["box"][3]) / 2
                    width = max(box[2] - box[0], r["box"][2] - r["box"][0], 1.0)
                    if ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5 <= self.p.reentry_dist * width:
                        tr.related.append(r["key"])
            keep.append(tr)
            new.append(tr)
            obs.append((tr, db, cls, score))
        self.tracks = keep
        self.recent = [r for r in self.recent if t - r["t"] <= self.p.reentry_s]
        return obs, closed, new

    def _match(self, tr: Track, det, t: float):
        db, cls, score = det
        dt = max(t - tr.t, 1e-6)
        a = self.p.velocity_smoothing
        new_v = [(n - o) / dt for n, o in zip(db, tr.box)]
        tr.v = [round(a * ov + (1 - a) * nv, 4) for ov, nv in zip(tr.v, new_v)] if tr.hits > 1 else [round(x, 4) for x in new_v]
        tr.box, tr.t = [round(float(x), 2) for x in db], t
        if self.p.algo == "byte" and tr.mean is not None:
            m, c = kalman.update(tr.mean, tr.cov, db)
            tr.mean, tr.cov, tr.kf_t = m.tolist(), c.ravel().tolist(), t
        tr.hits += 1
        tr.votes[cls] = round(tr.votes.get(cls, 0.0) + score, 4)
        area = (db[2] - db[0]) * (db[3] - db[1])
        if area > tr.best_area:
            tr.best_area, tr.best_t, tr.best_box = round(area, 2), t, [round(float(x), 2) for x in db]
        if tr.state == "tentative" and tr.hits >= self.p.min_hits:
            tr.state = "confirmed"
        return (tr, db, cls, score)

    def _close(self, tr: Track, reason: str) -> Closed:
        c = Closed(tr, reason, tr.state == "confirmed")
        if c.confirmed and reason in ("lost", "left_frame"):
            self.recent.append({"key": tr.key, "t": tr.t, "box": self._recent_box(tr),
                                "family": tr.family})
        return c

    def _recent_box(self, tr: Track) -> list[float]:
        """Where a closed track was heading, half a second after it was last seen (for re-entry checks)."""
        ahead = tr.t + min(self.p.gap_s, 0.5)
        if self.p.algo == "byte" and tr.mean is not None:
            m, _ = kalman.predict(tr.mean, tr.cov, (ahead - tr.kf_t) / self.p.frame_s)
            return [round(v, 2) for v in kalman.z_to_box(m)]
        return tr.predict(ahead)

    def close_all(self, reason: str, t: float | None = None) -> list[Closed]:
        """Close every track. At the end of a recording, a track already unseen for a while at the frame edge
        is recorded as having left the frame."""
        out = []
        for tr in self.tracks:
            r = reason
            if reason == "end_of_recording" and t is not None and t - tr.t > 0.5 and self._touches_edge(tr.box):
                r = "left_frame"
            out.append(self._close(tr, r))
        self.tracks = []
        if reason == "scene_cut":
            self.recent = []          # a cut ends all continuity, including re-entry suspicion
        return out
