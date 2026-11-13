"""What each video target did and looked like, for searching (phase 6a): direction, speed, time in view, time
stopped, size, colour, and for people the colour of the upper and lower body; and how many were in view at once.

Rules:
- Motion is measured on the bottom middle of the box (where a target touches the ground), from the tracked boxes
  with the operator's boxes in place of the tracker's.
- Direction is where the target went in the picture, from where it was first seen to where it was last seen:
  right, down-right, down, down-left, left, up-left, up, up-right; "stayed" when it moved less than its own size.
- Speed is the median over the track, each step taken over at least half a second (so a box that shakes does not
  look fast); "top" is the 90th percentile. Time stopped adds the steps slower than a tenth of the target's height
  per second (about 0.5 km/h for a car), or 1 km/h when the camera is measured.
- Measuring the camera: the operator clicks two points on the picture and picks the real distance between them from
  a list. That gives one scale (metres per pixel) for the recording and for the other recordings from the same
  folder (the same camera) that have none of their own. One scale for the whole picture is approximate: the two
  points should be on the road, at the depth where targets pass. Without it, speed and size stay in pixels.
- Colour is read once per target, on its sharpest frame (the one shown as its picture), by the still-image colour
  reader (``attributes.read_colour``): observed, uncertain (competitors kept) or not observable. A person's upper
  body is read from 15 % to 50 % of the box height and the lower body from 50 % to 90 %.
- In view at once: per family (vehicles, people, ...), the most counted targets in view at the same moment. A crowd
  alert (the mission's setting) lists the periods with at least that many people in view.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import PurePosixPath

from .detect import family
from .store import utcnow

COLOUR_VERSION = "video-colour-1"
DIRECTIONS = ("right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right")
STEP_S = 0.5
DISTANCES_M = (1, 2, 3, 3.5, 4.5, 5, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)


def _foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def motion(boxes: list[tuple[float, list[float]]], m_per_px: float | None = None) -> dict:
    """Direction, speed, time in view, time stopped and size, from (t, box) by time."""
    out: dict = {"direction": None, "dwell_s": 0.0, "stopped_s": 0.0, "speed": None, "top_speed": None,
                 "unit": "km/h" if m_per_px else "px/s", "w": None, "h": None, "size_unit": "m" if m_per_px else "px"}
    if not boxes:
        return out
    t0, t1 = boxes[0][0], boxes[-1][0]
    out["dwell_s"] = round(t1 - t0, 2)
    ws = sorted(b[2] - b[0] for _, b in boxes)
    hs = sorted(b[3] - b[1] for _, b in boxes)
    w, h = ws[len(ws) // 2], hs[len(hs) // 2]
    scale = m_per_px or 1.0
    out["w"], out["h"] = round(w * scale, 2 if m_per_px else 0), round(h * scale, 2 if m_per_px else 0)
    (x0, y0), (x1, y1) = _foot(boxes[0][1]), _foot(boxes[-1][1])
    dx, dy = x1 - x0, y1 - y0
    if math.hypot(dx, dy) < max(w, h, 1.0):
        out["direction"] = "stayed"
    else:
        ang = math.degrees(math.atan2(dy, dx)) % 360          # image y grows downwards: 90 is "down"
        out["direction"] = DIRECTIONS[int(((ang + 22.5) % 360) // 45)]
    # speed over steps of at least STEP_S
    steps = []
    j = 0
    for i in range(len(boxes)):
        while j < len(boxes) and boxes[j][0] - boxes[i][0] < STEP_S:
            j += 1
        if j >= len(boxes):
            break
        dt = boxes[j][0] - boxes[i][0]
        (ax, ay), (bx, by) = _foot(boxes[i][1]), _foot(boxes[j][1])
        steps.append((boxes[i][0], dt, math.hypot(bx - ax, by - ay) / dt))
    to_unit = (lambda v: v * m_per_px * 3.6) if m_per_px else (lambda v: v)
    if steps:
        sp = sorted(s for _, _, s in steps)
        out["speed"] = round(to_unit(sp[len(sp) // 2]), 1)
        out["top_speed"] = round(to_unit(sp[min(len(sp) - 1, int(len(sp) * 0.9))]), 1)
        slow = (1.0 / (m_per_px * 3.6)) if m_per_px else 0.1 * max(h, 1.0)      # px/s
        # time stopped: each step's own share of the track (steps overlap; count the time to the next step)
        stopped = 0.0
        for k, (t, dt, s) in enumerate(steps):
            nxt = steps[k + 1][0] if k + 1 < len(steps) else t + dt
            if s < slow:
                stopped += nxt - t
        out["stopped_s"] = round(min(stopped, out["dwell_s"]), 2)
    return out


# ---------------------------------------------------------------- measuring the camera

class Calibration:
    def __init__(self, mission):
        self.m = mission
        self.con = mission.con

    def _row(self, version_id: int):
        return self.con.execute(
            "SELECT * FROM video_calibration WHERE source_version_id=? ORDER BY id DESC LIMIT 1", (version_id,)).fetchone()

    def get(self, version_id: int) -> dict | None:
        """This recording's measure, else the newest of another recording in the same folder (the same camera)."""
        r = self._row(version_id)
        if r is not None:
            return None if r["cleared"] else self._dict(r, own=True)
        folder = self._folder(version_id)
        if folder is None:
            return None
        rows = self.con.execute(
            "SELECT c.*, f.rel_path FROM video_calibration c JOIN source_version v ON v.id=c.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id WHERE c.id IN (SELECT MAX(id) FROM video_calibration "
            "GROUP BY source_version_id) ORDER BY c.id DESC").fetchall()
        for r in rows:
            if not r["cleared"] and str(PurePosixPath(r["rel_path"]).parent) == folder:
                return self._dict(r, own=False)
        return None

    def _folder(self, version_id: int) -> str | None:
        r = self.con.execute("SELECT f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                             "WHERE v.id=?", (version_id,)).fetchone()
        return str(PurePosixPath(r[0]).parent) if r else None

    @staticmethod
    def _dict(r, own: bool) -> dict:
        px = math.hypot(r["x2"] - r["x1"], r["y2"] - r["y1"])
        return {"from_version_id": r["source_version_id"], "own": own, "points": [r["x1"], r["y1"], r["x2"], r["y2"]],
                "meters": r["meters"], "m_per_px": r["meters"] / px if px else None, "author": r["author"],
                "created_at": r["created_at"]}

    def set(self, version_id: int, points: list[float], meters: float, author: str) -> dict:
        x1, y1, x2, y2 = (float(v) for v in points)
        if math.hypot(x2 - x1, y2 - y1) < 10:
            raise ValueError("Place the two points further apart (at least 10 pixels).")
        if not 0.2 <= meters <= 1000:
            raise ValueError("Pick a distance between 0.2 m and 1000 m.")
        self.con.execute("INSERT INTO video_calibration(source_version_id,x1,y1,x2,y2,meters,cleared,author,created_at) "
                         "VALUES (?,?,?,?,?,?,0,?,?)", (version_id, x1, y1, x2, y2, float(meters), author, utcnow()))
        return self.get(version_id)

    def clear(self, version_id: int, author: str) -> None:
        self.con.execute("INSERT INTO video_calibration(source_version_id,x1,y1,x2,y2,meters,cleared,author,created_at) "
                         "VALUES (?,0,0,0,0,1,1,?,?)", (version_id, author, utcnow()))


# ---------------------------------------------------------------- colour

def _monochrome(img) -> bool:
    import numpy as np
    small = img[::8, ::8].astype(np.int16)
    return float(np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean()) < 3.0


def read_target_colours(img, box, cls: str) -> dict:
    from .attributes import read_colour
    mono = _monochrome(img)
    if family(cls) == "person" or cls == "person":
        x1, y1, x2, y2 = box
        h = y2 - y1
        up = read_colour(img, [x1, y1 + 0.15 * h, x2, y1 + 0.5 * h], mono).as_dict()
        lo = read_colour(img, [x1, y1 + 0.5 * h, x2, y1 + 0.9 * h], mono).as_dict()
        return {"upper": up, "lower": lo}
    return {"colour": read_colour(img, box, mono).as_dict()}


def stored_colours(con, event_ids: list[int]) -> dict[int, dict]:
    if not event_ids:
        return {}
    out = {}
    for i in range(0, len(event_ids), 500):
        chunk = event_ids[i:i + 500]
        for r in con.execute(f"SELECT event_id, data FROM video_event_colour WHERE version=? AND event_id IN "
                             f"({','.join('?' * len(chunk))})", (COLOUR_VERSION, *chunk)):
            out[r[0]] = json.loads(r[1])
    return out


def read_missing_colours(svc, events, stop: threading.Event | None = None) -> int:
    """Read the colour of the events that have none yet (one frame each). Returns how many were read."""
    from .video import frame_at
    have = stored_colours(svc.con, [e.id for e in events])
    todo = [e for e in events if e.id not in have and e.thumb_box]
    n = 0
    for e in sorted(todo, key=lambda e: (e.source_version_id, e.thumb_t)):
        if stop is not None and stop.is_set():
            break
        try:
            f = frame_at(svc.path(e.source_version_id), e.thumb_t)
            data = read_target_colours(f.image, e.thumb_box, e.cls) if f is not None else {"error": "frame not found"}
        except Exception as exc:  # noqa: BLE001 (a file gone or unreadable: say so, do not stop the others)
            data = {"error": f"{type(exc).__name__}: {exc}"}
        svc.con.execute("INSERT OR REPLACE INTO video_event_colour(event_id, version, data, read_at) VALUES (?,?,?,?)",
                        (e.id, COLOUR_VERSION, json.dumps(data), utcnow()))
        n += 1
    return n


_reading: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def read_colours_soon(mission_id: str, open_service) -> bool:
    """Read missing colours in the background (one reader per mission). True when a reader is running."""
    with _lock:
        t = _reading.get(mission_id)
        if t is not None and t.is_alive():
            return True

        def run():
            svc = open_service()
            try:
                read_missing_colours(svc, svc.events())
            finally:
                try:
                    svc.con.close()
                except Exception:  # noqa: BLE001
                    pass
        t = threading.Thread(target=run, name=f"colours-{mission_id}", daemon=True)
        _reading[mission_id] = t
        t.start()
        return True


def colour_summary(data: dict | None) -> dict:
    """{"colour": ["red"] | ["red", "orange"] (uncertain) | [] (not observable) | None (not read), "upper", "lower"}"""
    def one(d):
        if d is None:
            return None
        return d.get("values", []) if d.get("state") in ("observed", "uncertain") else []
    if data is None:
        return {"colour": None, "upper": None, "lower": None, "colour_state": None}
    if "error" in data:
        return {"colour": [], "upper": None, "lower": None, "colour_state": "not_observable"}
    return {"colour": one(data.get("colour")), "upper": one(data.get("upper")), "lower": one(data.get("lower")),
            "colour_state": (data.get("colour") or data.get("upper") or {}).get("state")}


# ---------------------------------------------------------------- in view at once

def in_view(events, crowd_at: int | None = None, crowd_family: str = "person") -> dict:
    """Per family: the most counted targets in view at once, and when; and the crowd periods (crowd_at or more)."""
    by_fam: dict[str, list[tuple[float, int]]] = {}
    for e in events:
        if not e.counted:
            continue
        for a, b in e.parts:
            by_fam.setdefault(family(e.cls), []).extend([(a, 1), (b, -1)])
    out: dict = {"families": {}, "crowd": [], "crowd_at": crowd_at}
    for fam, edges in by_fam.items():
        edges.sort(key=lambda x: (x[0], x[1]))           # an end before a start at the same moment
        cur = best = 0
        best_t = 0.0
        start = None
        for t, d in edges:
            cur += d
            if cur > best:
                best, best_t = cur, t
            if fam == crowd_family and crowd_at:
                if cur >= crowd_at and start is None:
                    start = t
                    peak = cur
                elif start is not None and cur >= crowd_at:
                    peak = max(peak, cur)
                elif start is not None and cur < crowd_at:
                    out["crowd"].append({"start_s": round(start, 2), "end_s": round(t, 2), "peak": peak})
                    start = None
        out["families"][fam] = {"max": best, "at_s": round(best_t, 2)}
    return out
