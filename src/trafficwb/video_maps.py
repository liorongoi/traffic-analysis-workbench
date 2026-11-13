"""Heatmaps and path maps of a video (phase 6d).

Rules:
- The picture under the map is the empty road: the median of nine frames spread over the video (moving targets
  vanish in a median). It is made once per video and kept with the mission's evidence.
- A heatmap shows where targets spent time: each tracked box adds the time until the next box at the bottom middle
  of the box (where it touches the ground), spread over a small disc, so a car that waits makes a hot spot and one
  that drives past a faint trail. One hue, from transparent (no time) to strong (the most time in this video);
  the scale says the most seconds at one spot.
- A path map draws each target's path (bottom middle of its boxes) as a thin line, faint so that busy routes stand
  out where many overlap, with a dot where it was first seen. Colour by class (a legend names each) or one colour.
- Only counted events are drawn; the class, direction and the other filters of the events page can narrow them.
  The camera's lines and areas are drawn on top.
"""
from __future__ import annotations

import numpy as np

from .detect import family

HEAT_BGR = (0, 140, 255)           # one hue (orange) whose strength carries the amount
# class colours in a fixed order (a 9th class reuses none: it is drawn in grey)
CLASS_BGR = [(232, 163, 74), (74, 186, 242), (120, 200, 90), (220, 110, 200), (90, 90, 235), (200, 200, 80),
             (160, 120, 240), (80, 200, 200)]
GREY = (170, 170, 170)


def background(svc, version_id: int, w: int | None = None):
    """The empty road: the median of nine frames (cached)."""
    import cv2
    from .video import frame_at
    cache = svc.m.dir / "evidence" / "maps" / f"background_{version_id}.jpg"
    if cache.is_file():
        img = cv2.imread(str(cache))
    else:
        dur = svc.con.execute("SELECT duration_s FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone()
        d = float(dur[0]) if dur and dur[0] else 0.0
        frames = []
        for k in range(9):
            f = frame_at(svc.path(version_id), d * (k + 0.5) / 9 if d else 0.0)
            if f is not None:
                frames.append(f.image)
        if not frames:
            raise ValueError("No frame could be read from the video.")
        h0, w0 = frames[0].shape[:2]
        frames = [f for f in frames if f.shape[:2] == (h0, w0)]
        img = np.median(np.stack(frames), axis=0).astype(np.uint8)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img


def _foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def heat(shape, tracks: list[list[tuple[float, list[float]]]]) -> np.ndarray:
    """Seconds spent per pixel (float32), each box's time spread over a disc a third of its width."""
    import cv2
    h, w = shape[:2]
    acc = np.zeros((h, w), np.float32)
    for boxes in tracks:
        for (t0, b), (t1, _) in zip(boxes, boxes[1:] + [(boxes[-1][0] + 0.2, boxes[-1][1])]):
            dt = max(0.0, min(t1 - t0, 2.0))
            if dt <= 0:
                continue
            x, y = _foot(b)
            r = max(3, int((b[2] - b[0]) / 3))
            disc = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
            cv2.circle(disc, (r, r), r, 1.0, -1)
            disc *= dt                                   # the box's time, at every pixel under it
            x0, y0 = int(x) - r, int(y) - r
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + 2 * r + 1), min(h, y0 + 2 * r + 1)
            if xa < xb and ya < yb:
                acc[ya:yb, xa:xb] += disc[ya - y0:yb - y0, xa - x0:xb - x0]
    return cv2.GaussianBlur(acc, (0, 0), 3)


def render(svc, version_id: int, kind: str, events, zones: list[dict], width: int = 1280) -> tuple[bytes, dict]:
    """A JPEG of the map and what its legend needs."""
    import cv2
    bg = background(svc, version_id)
    img = (bg.astype(np.float32) * 0.75).astype(np.uint8)          # a little darker, so the marks stand out
    boxes = svc.boxes_by_event(events)
    legend: dict = {"kind": kind, "targets": len(events)}
    if kind == "heat":
        acc = heat(img.shape, [boxes[e.id] for e in events if boxes.get(e.id)])
        top = float(acc.max())
        legend["max_s"] = round(top, 2)
        if top > 0:
            a = np.clip(acc / top, 0, 1) ** 0.6                       # faint trails still show
            col = np.zeros_like(img, np.float32)
            col[:] = HEAT_BGR
            img = (img.astype(np.float32) * (1 - a[..., None] * 0.85) + col * (a[..., None] * 0.85)).astype(np.uint8)
    else:
        classes = sorted({e.cls for e in events}, key=lambda c: (family(c) != "vehicle", c))
        legend["classes"] = {c: _hex(CLASS_BGR[i] if i < len(CLASS_BGR) else GREY) for i, c in enumerate(classes)}
        over = img.copy()
        lw = max(1, img.shape[1] // 640)
        for e in events:
            pts = np.array([_foot(b) for _, b in boxes.get(e.id, [])], np.int32)
            if len(pts) < 2:
                continue
            i = classes.index(e.cls)
            c = CLASS_BGR[i] if i < len(CLASS_BGR) else GREY
            cv2.polylines(over, [pts], False, c, lw, cv2.LINE_AA)
            cv2.circle(over, tuple(int(v) for v in pts[0]), lw + 2, c, -1, cv2.LINE_AA)
        alpha = 0.55 if len(events) < 50 else 0.35
        img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    lw = max(1, img.shape[1] // 480)
    for z in zones:
        p = np.array(z["points"], np.int32)
        cv2.polylines(img, [p], z["kind"] == "area", (255, 255, 255), lw, cv2.LINE_AA)
        cv2.putText(img, z["name"], tuple(int(v) for v in p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * lw, (255, 255, 255),
                    lw, cv2.LINE_AA)
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes(), legend


def _hex(bgr) -> str:
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])
