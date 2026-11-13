"""Video decoding by presentation timestamps (VID-02, TIME rules for video).

Rules:
- Every time is the frame's presentation timestamp (PTS) converted to seconds
  from the stream start. Frame counting is never used, so variable frame rate
  recordings keep correct times.
- Analysis samples frames on a fixed time grid (every `interval_s` seconds):
  for each grid point the first frame whose time is at or after it is used.
  The grid and the interval are recorded with the run.
- Decoding is resumable: it can start after any analysed time and continues
  the same grid, so a run that is interrupted and resumed analyses exactly the
  same frames as one that is not.
- The absolute UTC start comes from a confirmed source: the filename pattern
  or an explicit operator correction. The container's creation time is shown
  but not trusted. Without a confirmed source, absolute times stay empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Probe:
    container: str | None
    codec: str | None
    width: int | None
    height: int | None
    duration_s: float | None
    nominal_fps: float | None
    frame_count: int | None
    variable_frame_rate: bool | None
    creation_time: str | None
    error: str | None = None


def probe(path: Path, scan_frames: int = 300) -> Probe:
    """Read stream facts. Frame intervals of the first `scan_frames` packets decide whether the rate is variable."""
    import av
    try:
        with av.open(str(path)) as c:
            vs = next((s for s in c.streams if s.type == "video"), None)
            if vs is None:
                return Probe(c.format.name, None, None, None, None, None, None, None, None, "no video stream")
            dur = None
            if vs.duration is not None and vs.time_base is not None:
                dur = float(vs.duration * vs.time_base)
            elif c.duration:
                dur = c.duration / 1_000_000
            pts = []
            for pkt in c.demux(vs):
                if pkt.pts is not None:
                    pts.append(pkt.pts)
                if len(pts) >= scan_frames:
                    break
            pts.sort()
            vfr = None
            if len(pts) >= 3:
                d = np.diff(np.array(pts, dtype=np.float64))
                d = d[d > 0]
                if len(d):
                    vfr = bool((d.max() - d.min()) / np.median(d) > 0.10)
            rate = vs.average_rate or vs.guessed_rate
            created = c.metadata.get("creation_time") or vs.metadata.get("creation_time")
            return Probe(c.format.name, vs.codec_context.name, vs.codec_context.width, vs.codec_context.height, dur,
                         float(rate) if rate else None, vs.frames or None, vfr, created)
    except Exception as exc:  # corrupt or unsupported: reported, never raised into the run loop
        return Probe(None, None, None, None, None, None, None, None, None, f"{type(exc).__name__}: {exc}")


@dataclass
class Frame:
    t: float                     # seconds from stream start
    image: np.ndarray            # BGR


def frames(path: Path, interval_s: float, after_t: float | None = None, next_sample_t: float = 0.0,
           until_t: float | None = None) -> Iterator[tuple[Frame, float]]:
    """Yield (frame, next grid time) for frames on the sampling grid.

    after_t: only frames later than this time (resume point); next_sample_t: the grid point to look for next.
    until_t: stop before frames at or after this time (chunking).
    """
    import av
    eps = 1e-6
    with av.open(str(path)) as c:
        vs = next(s for s in c.streams if s.type == "video")
        vs.thread_type = "AUTO"
        tb = vs.time_base
        start = vs.start_time if vs.start_time is not None else 0
        if after_t is not None and after_t > 0:
            target = int((Fraction(after_t).limit_denominator(1_000_000) / tb)) + start
            c.seek(target, stream=vs, backward=True, any_frame=False)
        nxt = next_sample_t
        for fr in c.decode(vs):
            if fr.pts is None:
                continue
            t = float((fr.pts - start) * tb)
            if after_t is not None and t <= after_t + eps:
                continue
            if until_t is not None and t >= until_t - eps:
                return
            if t + eps < nxt:
                continue
            # Advance the grid past this frame's time; skipped grid points mean frames are sparser than the interval.
            if nxt <= t + eps:
                nxt += (int((t + eps - nxt) // interval_s) + 1) * interval_s
            yield Frame(t, fr.to_ndarray(format="bgr24")), nxt


def frame_at(path: Path, t: float) -> Frame | None:
    """The first frame at or after time t (for thumbnails and colour)."""
    for f, _ in frames(path, interval_s=1e9, after_t=max(0.0, t - 1e-3) if t > 0 else None, next_sample_t=t):
        return f
    return None


def start_utc(capture_utc: str | None) -> datetime | None:
    if not capture_utc:
        return None
    return datetime.fromisoformat(capture_utc.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_at(start: datetime | None, t: float) -> str | None:
    if start is None:
        return None
    return (start + timedelta(seconds=t)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def scene_signature(img: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g.astype(np.uint8)


def is_scene_cut(prev: np.ndarray | None, cur: np.ndarray, diff_threshold: float = 45.0) -> bool:
    """A cut changes most of the picture at once; moving vehicles change only small parts."""
    if prev is None:
        return False
    d = np.abs(prev.astype(np.int16) - cur.astype(np.int16))
    return float(d.mean()) > diff_threshold or float((d > 60).mean()) > 0.6


# ---------- camera motion (moving and PTZ cameras) ----------

MOTION_WIDTH = 480          # frames are compared at this width; the motion is returned in full-frame pixels


def motion_frame(img: np.ndarray) -> np.ndarray:
    """The small grey frame camera motion is measured on (kept in the checkpoint for resuming)."""
    import cv2
    h, w = img.shape[:2]
    s = min(1.0, MOTION_WIDTH / w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def compose(first, then):
    """The 2x3 affine of ``first`` followed by ``then`` (either may be None: no motion known)."""
    if first is None:
        return then
    if then is None:
        return first
    a = np.vstack([np.asarray(first, float), [0.0, 0.0, 1.0]])
    b = np.vstack([np.asarray(then, float), [0.0, 0.0, 1.0]])
    return (b @ a)[:2]


def camera_motion(prev: np.ndarray | None, cur: np.ndarray, full_width: int, boxes=(),
                  min_points: int = 30, min_inliers: float = 0.5, min_shared: float = 0.08) -> tuple[np.ndarray | None, str]:
    """How the camera moved from ``prev`` to ``cur`` (small grey frames from ``motion_frame``), as a 2x3 affine in
    full-frame pixels: corners tracked with pyramidal Lucas-Kanade optical flow (kept only when tracking them back
    lands where they started), then a RANSAC fit of rotation, scale and shift. ``boxes`` (full-frame pixels) are
    left out so moving objects do not count as camera motion.
    Returns (motion, reason): "ok"; "no_texture" when the previous frame has too few corners to tell (fog, a blank
    wall), motion None; "unsure" when the two pictures share enough but no single motion explains them (a camera
    moving forward through a deep scene, a crowd), motion None and not a cut; or "no_match" when almost nothing of
    the previous picture is found in this one (under ``min_shared`` of its corners): the picture was replaced, a cut."""
    import cv2
    if prev is None or prev.shape != cur.shape:
        return None, "no_texture"
    s = cur.shape[1] / full_width
    mask = np.full(prev.shape, 255, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v * s)) for v in b)
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 0
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=3, mask=mask)
    if pts is None or len(pts) < min_points:
        return None, "no_texture"
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
    if ok.mean() < min_shared:
        return None, "no_match"
    if ok.sum() < min_points // 2:
        return None, "unsure"
    a, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                         maxIters=2000, confidence=0.99, refineIters=10)
    if a is None or inl is None or inl.mean() < min_inliers:
        return None, "unsure"
    a = a.astype(float)
    a[:, 2] /= s                                  # shift back to full-frame pixels; rotation and scale are unchanged
    return a, "ok"

