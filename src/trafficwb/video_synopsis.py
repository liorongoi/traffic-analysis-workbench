"""A short "summary" video of a recording (phase 6c): the targets of a long video shown together over its empty
road, each with the time it really passed, so an hour can be looked through in a minute or two.

Design (LanePulse's own):
- Each target is a "strip": its boxes, at 10 per second, from its first to its last box (with the operator's boxes
  in place of the tracker's). The picture under all of them is the empty road (``video_maps.background``).
- Strips keep their order: a later target never starts before an earlier one in the summary. Each starts at the
  first moment, at or after the one before it, at which it overlaps the targets already placed in at most a few
  frames. "How busy" sets how much they may overlap and how many may be on screen at once.
- Each frame pastes, over the empty road, the part of the original frame inside each target's box, its edges
  blended so it does not look cut out, and writes the real time beside it (and its event number). Targets that
  overlap are drawn earliest first.
- With the video, an index says which events are on screen at each second of the summary, so the page can list
  them and open the event or play the original moment.
- The summary is made in the background, one at a time per mission; it is kept until the analysis changes.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

RATE = 10.0                      # strip samples and summary frames per second
BUSY = {"calm": (0.02, 4), "normal": (0.08, 8), "busy": (0.2, 16)}   # (overlap allowed per frame, most on screen)
MAX_TARGETS = 600


@dataclass
class Strip:
    event_id: int
    key: str
    cls: str
    t0: float
    times: list[float]
    boxes: list[list[float]]
    start: float = 0.0            # where it begins in the summary
    crops: dict = field(default_factory=dict)   # sample index -> (x, y, image)


def _interp(boxes: list[tuple[float, list[float]]], t: float) -> list[float] | None:
    if not boxes or t < boxes[0][0] - 1e-6 or t > boxes[-1][0] + 1e-6:
        return None
    lo, hi = 0, len(boxes) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if boxes[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, a), (tb, b) = boxes[lo], boxes[hi]
    if tb - ta > 3.0 and min(t - ta, tb - t) > 0.2:
        return None                                       # a long gap: the target was not seen then
    w = 0.0 if tb == ta else (t - ta) / (tb - ta)
    return [a[k] + w * (b[k] - a[k]) for k in range(4)]


def strips(svc, events) -> list[Strip]:
    boxes = svc.boxes_by_event(events)
    out = []
    for e in sorted(events, key=lambda e: e.first_t):
        bs = boxes.get(e.id) or []
        if len(bs) < 2:
            continue
        n = int((bs[-1][0] - bs[0][0]) * RATE) + 1
        times, bx = [], []
        for k in range(n):
            t = bs[0][0] + k / RATE
            b = _interp(bs, t)
            if b is not None:
                times.append(round(t, 3))
                bx.append(b)
        if bx:
            out.append(Strip(e.id, e.key, e.cls, times[0], times, bx))
    return out


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
    return inter / small


def place(ss: list[Strip], busy: str = "normal") -> float:
    """Give each strip its start in the summary; returns the summary's length in seconds."""
    allow, most = BUSY.get(busy, BUSY["normal"])
    frames: dict[int, list[list[float]]] = {}             # summary frame -> boxes already there
    earliest = 0
    end = 0
    for s in ss:
        n = len(s.boxes)
        k = earliest
        while True:
            clash = 0.0
            full = False
            for i in range(n):
                there = frames.get(k + i, ())
                if len(there) >= most:
                    full = True
                    break
                clash += max((_overlap(s.boxes[i], b) for b in there), default=0.0)
            if not full and clash / n <= allow:
                break
            k += 1
        s.start = k / RATE
        for i in range(n):
            frames.setdefault(k + i, []).append(s.boxes[i])
        earliest = k
        end = max(end, k + n)
    return end / RATE


def _grab(svc, version_id: int, ss: list[Strip], progress=None, stop=None) -> None:
    """Cut each strip's boxes out of the original frames, decoding the video once."""
    from .video import frames
    want: dict[int, list[tuple[Strip, int]]] = {}
    for s in ss:
        for i, t in enumerate(s.times):
            want.setdefault(int(round(t * RATE)), []).append((s, i))
    if not want:
        return
    last = max(want)
    for f, _ in frames(svc.path(version_id), interval_s=1.0 / RATE):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        g = int(round(f.t * RATE))
        H, W = f.image.shape[:2]
        for s, i in want.pop(g, []):
            x1, y1, x2, y2 = s.boxes[i]
            pw, ph = (x2 - x1) * 0.08 + 2, (y2 - y1) * 0.08 + 2
            xa, ya = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            xb, yb = min(W, int(x2 + pw)), min(H, int(y2 + ph))
            if xb > xa and yb > ya:
                s.crops[i] = (xa, ya, f.image[ya:yb, xa:xb].copy())
        if progress:
            progress(min(0.6, 0.6 * g / max(last, 1)))
        if g >= last:
            break


def _feather(h: int, w: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (max(1, int(w * 0.48)), max(1, int(h * 0.48))), 0, 0, 360, 1.0, -1)
    k = max(3, (min(h, w) // 6) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]


def clock(start, t: float) -> str:
    from datetime import timedelta
    if start is not None:
        return (start + timedelta(seconds=t)).strftime("%H:%M:%S")
    s = int(t)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render(svc, version_id: int, events, out: Path, busy: str = "normal", progress=None, stop=None) -> dict:
    """Write the summary video to ``out`` and return its index."""
    import av
    import cv2
    from .video_maps import background
    ss = strips(svc, events)[:MAX_TARGETS]
    length = place(ss, busy)
    _grab(svc, version_id, ss, progress, stop)
    bg = background(svc, version_id)
    H, W = bg.shape[:2]
    W2, H2 = W - W % 2, H - H % 2
    start = getattr(events[0], "_start", None) if events else None
    n = max(1, int(round(length * RATE)))
    on: dict[int, list[tuple[Strip, int]]] = {}
    for s in ss:
        k0 = int(round(s.start * RATE))
        for i in range(len(s.times)):
            on.setdefault(k0 + i, []).append((s, i))
    enc = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    index = []
    fs = max(0.35, W / 1600)
    th = max(1, int(round(W / 900)))
    tmp = out.with_suffix(".part.mp4")
    with av.open(str(tmp), "w", format="mp4") as o:
        st = o.add_stream(enc, rate=int(RATE))
        st.width, st.height, st.pix_fmt = W2, H2, "yuv420p"
        st.time_base = Fraction(1, int(RATE))
        if enc == "libx264":
            st.options = {"bf": "0", "crf": "23"}
        for k in range(n):
            if stop is not None and stop.is_set():
                raise InterruptedError("stopped")
            img = bg.copy()
            here = sorted(on.get(k, []), key=lambda si: si[0].t0)
            labels = []
            for s, i in here:
                c = s.crops.get(i)
                if c is None:
                    continue
                x, y, crop = c
                h, w = crop.shape[:2]
                m = _feather(h, w)
                roi = img[y:y + h, x:x + w].astype(np.float32)
                img[y:y + h, x:x + w] = (roi * (1 - m) + crop.astype(np.float32) * m).astype(np.uint8)
                labels.append((s, s.boxes[i]))
            for s, b in labels:
                txt = f"{clock(start, s.t0)} {s.key}"
                org = (int(b[0]), max(int(12 * fs * 2), int(b[1]) - 4))
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:H2, :W2]), format="bgr24")
            fr.pts = k
            for p in st.encode(fr):
                o.mux(p)
            if k % int(RATE) == 0:
                index.append(sorted({s.event_id for s, _ in here}))
            if progress and k % 10 == 0:
                progress(0.6 + 0.4 * k / n)
        for p in st.encode():
            o.mux(p)
    tmp.replace(out)
    return {"length_s": round(n / RATE, 1), "targets": len(ss), "per_second": index,
            "events": {s.event_id: {"key": s.key, "cls": s.cls, "t": s.t0, "at": round(s.start, 1)} for s in ss},
            "original_s": round(max((s.times[-1] for s in ss), default=0) - min((s.t0 for s in ss), default=0), 1),
            "encoder": enc}


# ---------------------------------------------------------------- making it in the background

class Synopses:
    """One summary at a time per mission, made in a background thread; finished ones are kept on disk."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    @staticmethod
    def key(version_id: int, run_item_id: int, ids: list[int], busy: str, revisions: int) -> str:
        h = hashlib.sha1(json.dumps([version_id, run_item_id, sorted(ids), busy, revisions]).encode()).hexdigest()
        return h[:16]

    def start(self, mission_id: str, folder: Path, key: str, work) -> dict:
        """``work(progress, stop) -> index`` writes ``folder/<key>.mp4``."""
        out = folder / f"{key}.mp4"
        meta = folder / f"{key}.json"
        if out.is_file() and meta.is_file():
            return {"state": "done", "key": key} | json.loads(meta.read_text())
        with self.lock:
            j = self.jobs.get(mission_id)
            if j and j["state"] == "running":
                if j["key"] == key:
                    return dict(j)
                j["stop"].set()                         # a newer request replaces the one running
            stop = threading.Event()
            j = {"state": "running", "key": key, "progress": 0.0, "stop": stop}
            self.jobs[mission_id] = j

        def run():
            def progress(p):
                j["progress"] = round(p, 3)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                idx = work(out, progress, stop)
                meta.write_text(json.dumps(idx))
                j["state"] = "done"
            except InterruptedError:
                j["state"] = "stopped"
            except Exception as exc:  # noqa: BLE001
                j["state"], j["error"] = "failed", f"{type(exc).__name__}: {exc}"
        threading.Thread(target=run, name=f"synopsis-{mission_id}", daemon=True).start()
        return self.status(mission_id, key, folder)

    def status(self, mission_id: str, key: str, folder: Path) -> dict:
        meta = folder / f"{key}.json"
        if (folder / f"{key}.mp4").is_file() and meta.is_file():
            return {"state": "done", "key": key} | json.loads(meta.read_text())
        j = self.jobs.get(mission_id)
        if j and j["key"] == key:
            return {k: v for k, v in j.items() if k != "stop"}
        return {"state": "none", "key": key}
