"""Vehicle tracking within one recording (VID-02, VID-03).

An IoU tracker with constant-velocity prediction. It is deterministic and its
whole state is plain data, so it can be saved at a checkpoint and resumed with
identical results.

Rules:
- A track starts as tentative and is confirmed after `min_hits` matched
  frames. Only confirmed tracks become events, so one-frame false detections
  do not create events.
- Association is class-agnostic (the detector may call the same vehicle car
  in one frame and truck in the next); the event class is the score-weighted
  vote, and its share is kept as the class uncertainty.
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


@dataclass
class TrackerParams:
    iou_min: float = 0.25
    min_hits: int = 3
    gap_s: float = 1.5
    reentry_s: float = 3.0
    reentry_dist: float = 2.0      # in box widths
    edge_px: float = 4.0
    velocity_smoothing: float = 0.5


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

    def predict(self, t: float) -> list[float]:
        dt = t - self.t
        return [b + v * dt for b, v in zip(self.box, self.v)]


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
        tr = cls(s["w"], s["h"], TrackerParams(**s["params"]))
        tr.counter = s["counter"]
        tr.tracks = [Track(**t) for t in s["tracks"]]
        tr.recent = s["recent"]
        return tr

    # ---------- update ----------
    def _touches_edge(self, box) -> bool:
        e = self.p.edge_px
        return box[0] <= e or box[1] <= e or box[2] >= self.w - e or box[3] >= self.h - e

    def update(self, t: float, dets: list[tuple[list[float], str, float]]) -> tuple[list[tuple[Track, list[float], str, float]], list[Closed], list[Track]]:
        """Feed one analysed frame. dets: (box, cls, score).

        Returns (matched observations as (track, box, cls, score), closed tracks, newly created tracks).
        """
        preds = [tr.predict(t) for tr in self.tracks]
        pairs = []
        for i, pb in enumerate(preds):
            for j, (db, _, _) in enumerate(dets):
                v = iou(pb, db)
                if v >= self.p.iou_min:
                    pairs.append((-v, i, j))
        pairs.sort()
        used_t, used_d, obs = set(), set(), []
        for _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            tr, (db, cls, score) = self.tracks[i], dets[j]
            dt = max(t - tr.t, 1e-6)
            a = self.p.velocity_smoothing
            new_v = [(n - o) / dt for n, o in zip(db, tr.box)]
            tr.v = [round(a * ov + (1 - a) * nv, 4) for ov, nv in zip(tr.v, new_v)] if tr.hits > 1 else [round(x, 4) for x in new_v]
            tr.box, tr.t = [round(float(x), 2) for x in db], t
            tr.hits += 1
            tr.votes[cls] = round(tr.votes.get(cls, 0.0) + score, 4)
            area = (db[2] - db[0]) * (db[3] - db[1])
            if area > tr.best_area:
                tr.best_area, tr.best_t, tr.best_box = round(area, 2), t, [round(float(x), 2) for x in db]
            if tr.state == "tentative" and tr.hits >= self.p.min_hits:
                tr.state = "confirmed"
            obs.append((tr, db, cls, score))
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
                       best_area=round(area, 2), best_t=t, best_box=box)
            if self.p.min_hits <= 1:
                tr.state = "confirmed"
            for r in self.recent:
                if 0 <= t - r["t"] <= self.p.reentry_s:
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

    def _close(self, tr: Track, reason: str) -> Closed:
        c = Closed(tr, reason, tr.state == "confirmed")
        if c.confirmed and reason in ("lost", "left_frame"):
            self.recent.append({"key": tr.key, "t": tr.t, "box": tr.predict(tr.t + min(self.p.gap_s, 0.5))})
        return c

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
