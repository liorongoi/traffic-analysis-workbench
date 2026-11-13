"""Parallel video analysis: a recording cut into time segments, analysed side by side, tracks joined at the cuts.

Rules:
- A recording longer than one segment is cut at multiples of ``segment_s`` (5 minutes by default). Each segment
  owns the frames in [start, end); it also analyses ``overlap_s`` seconds before its start and after its end, so a
  target crossing a cut is tracked on both sides of it.
- Segments run in a pool of workers that share the loaded detector (ONNX Runtime and OpenVINO release Python's
  lock while they compute, so threads run in parallel). Each segment has its own tracker and touches no database;
  its result is stored once it is done, so a cancelled run keeps the segments it finished.
- Joining: a track that reaches a cut on the left and one that starts before it on the right are the same target
  when their boxes on the same frames inside the overlap agree (IoU 0.5 or more on most shared frames, and at least
  two of them), and they are of one class family. Pairs are taken most-agreeing first. Each kept track has the
  observations of the segments that own their frames, so every frame is counted once.
- The SAM 3.1 frames are fixed by the time grid (every Nth grid point from the recording start), so a segment calls
  SAM 3.1 on the frames the whole recording would.
- The result is the same kind of tracks the single-worker path makes; events, counts and review do not know which
  path made them. Joins are not guaranteed to equal a single run's tracks (a tracker's history differs at a cut);
  the tests check that targets and counts agree.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .detect import family
from .tracker import Track, Tracker, TrackerParams, iou
from .video import compose, camera_motion, frames, is_scene_cut, motion_frame, scene_signature


@dataclass
class Segment:
    idx: int
    own_a: float          # frames in [own_a, own_b) belong to this segment
    own_b: float
    run_a: float          # analysed from run_a (overlap before) ...
    run_b: float          # ... up to run_b (overlap after)


def plan(duration: float, segment_s: float, overlap_s: float, grid_s: float) -> list[Segment]:
    """Segments covering [0, duration); every start is a grid point, so segments see the frames a whole run would."""
    seg = max(grid_s, round(segment_s / grid_s) * grid_s)
    ov = round(overlap_s / grid_s) * grid_s
    out, a, k = [], 0.0, 0
    while a < duration - 1e-6:
        b = min(duration, a + seg)
        if duration - b < seg * 0.25:                  # no tiny last segment: the last one takes the rest
            b = duration
        out.append(Segment(k, round(a, 6), round(b, 6) if b < duration else float("inf"),
                           round(max(0.0, a - ov), 6), round(b + ov, 6) if b < duration else float("inf")))
        a, k = b, k + 1
    return out


@dataclass
class Piece:
    """One track as one segment saw it."""
    key: str
    family: str
    obs: list = field(default_factory=list)        # (t, [x1,y1,x2,y2], cls, score, found_by)
    reason: str = "segment_end"
    related: list = field(default_factory=list)


@dataclass
class SegmentResult:
    idx: int
    pieces: list[Piece]
    cuts: list[float]                               # scene cuts inside the segment's own range
    frames: int                                     # analysed frames inside the own range
    seconds: float                                  # wall time


def analyse(seg: Segment, path, width: int, height: int, detect: Callable, grid_s: float, params: TrackerParams,
            moving: bool, sam_every: int, checkpoint: Callable[[], None] | None = None) -> SegmentResult:
    """Track one segment. ``detect(img, sam: bool)`` returns detections; ``checkpoint`` may raise to cancel."""
    t0 = time.monotonic()
    tracker = Tracker(width, height, params)
    pieces: dict[str, Piece] = {}
    cuts: list[float] = []
    prev = pm = acc = None
    n_own = 0
    after = seg.run_a - grid_s / 2 if seg.run_a > 0 else None
    for k, (fr, nxt_after) in enumerate(frames(path, grid_s, after_t=after, next_sample_t=seg.run_a)):
        if fr.t >= seg.run_b - 1e-6:
            break
        sig = scene_signature(fr.image)
        cut = is_scene_cut(prev, sig)
        motion = None
        if moving:
            mf = motion_frame(fr.image)
            step, why = camera_motion(pm, mf, width, [t.box for t in tracker.tracks])
            cut = why == "no_match" or (why == "no_texture" and pm is not None and cut)
            motion = compose(acc, step)
            pm = mf
        prev = sig
        if cut:
            for c in tracker.close_all("scene_cut"):
                pieces[c.track.key].reason = "scene_cut"
                pieces[c.track.key].related = list(c.track.related)
            if seg.own_a <= fr.t < seg.own_b:
                cuts.append(round(fr.t, 6))
            motion = None
        grid_idx = int(round((nxt_after - grid_s) / grid_s))
        found_dets = detect(fr.image, sam_every <= 1 or grid_idx % sam_every == 0)
        if found_dets is None:              # SAM 3.1 alone on a moving camera: a frame for the motion only
            acc = motion
            if seg.own_a <= fr.t < seg.own_b:
                n_own += 1
            continue
        acc = None
        dets, found = [], {}
        for d in found_dets:
            box = list(map(float, d.box))
            found[id(box)] = getattr(d, "found_by", None)
            dets.append((box, d.cls, float(d.score)))
        obs, closed, new = tracker.update(fr.t, dets, motion=motion, image=fr.image)
        for tr in new:
            pieces[tr.key] = Piece(tr.key, tr.family)
        for tr, box, cls, score in obs:
            pieces[tr.key].obs.append((round(fr.t, 6), [round(float(b), 2) for b in box], cls, round(score, 4),
                                       found.get(id(box))))
        for c in closed:
            pieces[c.track.key].reason = c.reason
            pieces[c.track.key].related = list(c.track.related)
        if seg.own_a <= fr.t < seg.own_b:
            n_own += 1
        if checkpoint and k % 10 == 9:
            checkpoint()
    for c in tracker.close_all("segment_end"):
        pieces[c.track.key].related = list(c.track.related)
    return SegmentResult(seg.idx, list(pieces.values()), cuts, n_own, round(time.monotonic() - t0, 3))


def make_detect(kind: str, detector) -> Callable:
    """``detect(img, sam)`` for ``analyse``: with both detectors, frames between SAM 3.1 frames use the standard one
    alone; every box says which detector found it. The same on this computer and on another one."""
    from dataclasses import replace

    def found(d, who):
        try:
            return replace(d, found_by=who)
        except TypeError:
            return d

    def detect(img, sam: bool):
        if kind == "both" and not sam:
            return [found(d, "standard") for d in detector.standard.detect(img)]
        if kind == "sam3" and not sam:
            return None                     # a frame for the camera's motion only
        return [d if getattr(d, "found_by", None) else found(d, "sam3" if kind == "sam3" else "standard")
                for d in detector.detect(img)]
    return detect


# ---------- joining ----------

@dataclass
class Joined:
    family: str
    obs: list
    reason: str
    related: list            # (segment idx, piece key)
    ids: list                # (segment idx, piece key) of every piece joined


def _agreement(a: Piece, b: Piece, lo: float, hi: float, iou_min: float) -> float | None:
    """Mean IoU over the frames both pieces saw in [lo, hi), when most of those frames agree."""
    ta = {o[0]: o[1] for o in a.obs if lo - 1e-6 <= o[0] < hi}
    shared = [(ta[o[0]], o[1]) for o in b.obs if o[0] in ta]
    if len(shared) < 2:
        return None
    ious = [iou(x, y) for x, y in shared]
    if sum(v >= iou_min for v in ious) * 2 <= len(ious):
        return None
    return sum(ious) / len(ious)


def join(segs: list[Segment], results: dict[int, SegmentResult], iou_min: float = 0.5) -> list[Joined]:
    """Tracks for the whole recording from the segments' pieces, each frame taken from the segment that owns it."""
    open_: dict[str, Joined] = {}        # pieces of the previous segment still running at the cut
    out: list[Joined] = []
    for s in segs:
        res = results[s.idx]
        own = lambda p: [o for o in p.obs if s.own_a - 1e-6 <= o[0] < s.own_b - 1e-6]   # noqa: E731
        matched: dict[str, Joined] = {}
        if open_ and s.idx > 0:
            lo, hi = s.run_a, segs[s.idx - 1].run_b
            prev = results[s.idx - 1]
            cands = []
            for key, j in open_.items():
                left = next(p for p in prev.pieces if p.key == key)
                for p in res.pieces:
                    if p.family != left.family:
                        continue
                    v = _agreement(left, p, lo, hi, iou_min)
                    if v is not None:
                        cands.append((-v, key, p.key))
            used_l, used_r = set(), set()
            for _, lk, rk in sorted(cands):
                if lk in used_l or rk in used_r:
                    continue
                used_l.add(lk)
                used_r.add(rk)
                matched[rk] = open_[lk]
            for lk, j in open_.items():
                if lk not in used_l:            # reached the cut but nothing on the right agrees: it ended there
                    j.reason = "lost"
        nxt_open: dict[str, Joined] = {}
        for p in res.pieces:
            mine = own(p)
            j = matched.get(p.key)
            if j is not None:
                j.obs += mine
                j.ids.append((s.idx, p.key))
                j.reason = p.reason
                j.related += [(s.idx, r) for r in p.related]
            elif mine:
                j = Joined(p.family, mine, p.reason, [(s.idx, r) for r in p.related], [(s.idx, p.key)])
                out.append(j)
            else:
                continue
            # Still running at this segment's end (it was closed by the segment, not by the scene): may continue.
            if p.reason == "segment_end" and s.own_b != float("inf"):
                nxt_open[p.key] = j
        open_ = nxt_open
    return [j for j in out if j.obs]


def to_track(j: Joined, key: str, min_hits: int) -> tuple[Track, bool]:
    """A tracker Track built from a joined track's kept observations, and whether it is confirmed."""
    first, last = j.obs[0], j.obs[-1]
    votes: dict[str, float] = {}
    best = (0.0, first[0], first[1])
    for t, box, cls, score, _ in j.obs:
        votes[cls] = round(votes.get(cls, 0.0) + score, 4)
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best[0]:
            best = (round(area, 2), t, box)
    tr = Track(key, list(last[1]), last[0], hits=len(j.obs), state="confirmed" if len(j.obs) >= min_hits else "tentative",
               first_t=first[0], first_box=list(first[1]), votes=votes, best_area=best[0], best_t=best[1],
               best_box=list(best[2]), family=j.family or family(first[2]))
    return tr, len(j.obs) >= min_hits
