"""Lines, areas and paths in video (phase 6b).

Rules:
- The operator draws them on the player: a counting line (two points; its "in" side is to the right of the line
  drawn from its first to its second point, shown with an arrow) or an area (a polygon of three or more points), each
  named from a list. They belong to the camera: every video in the same folder uses them, as with the measure.
- A target crosses a line when the bottom middle of its box (where it touches the ground) passes from one side of
  the line to the other within the line's length. Each crossing has a time and a direction ("in" or "out"). A
  target that wobbles on the line counts once: a crossing back within 1 s cancels the one before.
- A target is in an area while that point is inside the polygon; entries shorter than 0.5 s are ignored (a box that
  shakes across the edge). For each entry: when it entered, when it left (or that it was still inside when last
  seen) and the time inside.
- A target's path is the order of the lines it crossed and the areas it entered. "From A to B" matches a target
  that was at A and later at B (other zones may come between).
- Only counted events are tallied (not rejected, duplicate or merged); boxes the operator drew replace the tracker's.
"""
from __future__ import annotations

import json
import math
import secrets
from pathlib import PurePosixPath

from .store import utcnow

CROSS_BACK_S = 1.0
CLEAR = 0.2
MIN_INSIDE_S = 0.5
LINE_NAMES = ["Stop line", "Entrance", "Exit", "Northbound", "Southbound", "Eastbound", "Westbound", "Crosswalk",
              "Gate", "Bridge", "Lane 1", "Lane 2", "Lane 3", "Count line A", "Count line B"]
AREA_NAMES = ["Junction", "Crosswalk", "Bus stop", "Parking", "No-stopping zone", "Entrance", "Exit", "Sidewalk",
              "Lane 1", "Lane 2", "Lane 3", "Waiting area", "Gate", "Area A", "Area B"]


class ZoneError(ValueError):
    pass


def _side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _cross(p, q, a, b) -> float | None:
    """Where on p->q (0..1) it crosses segment a-b, or None."""
    d1, d2 = _side(p, a, b), _side(q, a, b)
    if (d1 >= 0) == (d2 >= 0):                       # a point on the line counts with the "in" side
        return None
    d3, d4 = _side(a, p, q), _side(b, p, q)
    if (d3 > 0) == (d4 > 0) and d3 != 0 and d4 != 0:
        return None
    return d1 / (d1 - d2) if d1 != d2 else 0.0


def inside(p, poly) -> bool:
    x, y = p
    n = len(poly)
    c = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def crossings(boxes: list[tuple[float, list[float]]], line: list[list[float]]) -> list[dict]:
    """[{"t", "dir": "in"|"out"}] for one target's boxes and one line [[x1, y1], [x2, y2]]."""
    a, b = line
    out: list[dict] = []
    pts = [(t, _foot(bx)) for t, bx in boxes]
    for (t0, p), (t1, q) in zip(pts, pts[1:]):
        f = _cross(p, q, a, b)
        if f is None:
            continue
        # "in": from the left of a->b to its right (image y grows downwards: right of a->b has a positive side)
        d = "in" if _side(q, a, b) >= 0 else "out"
        t = t0 + f * (t1 - t0)
        if out and out[-1]["dir"] != d and t - out[-1]["t"] < CROSS_BACK_S:
            out.pop()                                    # it wobbled back over the line
            continue
        out.append({"t": round(t, 3), "dir": d})
    # A target standing on the line, whose box shakes, must get clear of it: after each crossing it has to go at
    # least a fifth of its own height past the line before it crosses back (or before it is lost).
    seg = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    kept = []
    for k, c in enumerate(out):
        end = out[k + 1]["t"] if k + 1 < len(out) else float("inf")
        far = max((abs(_side(_foot(bx), a, b)) / seg - CLEAR * (bx[3] - bx[1])
                   for t, bx in boxes if c["t"] <= t <= end), default=-1.0)
        if far >= 0:
            kept.append(c)
    return kept


def visits(boxes: list[tuple[float, list[float]]], poly: list[list[float]]) -> list[dict]:
    """[{"enter", "leave", "inside_s", "left": bool}] for one target's boxes and one area."""
    out: list[dict] = []
    cur = None
    last_t = None
    for t, bx in boxes:
        ins = inside(_foot(bx), poly)
        if ins and cur is None:
            cur = t
        elif not ins and cur is not None:
            if t - cur >= MIN_INSIDE_S:
                out.append({"enter": round(cur, 3), "leave": round(t, 3), "inside_s": round(t - cur, 2), "left": True})
            cur = None
        last_t = t
    if cur is not None and last_t is not None and (last_t - cur >= MIN_INSIDE_S or cur == boxes[0][0]):
        out.append({"enter": round(cur, 3), "leave": round(last_t, 3), "inside_s": round(last_t - cur, 2), "left": False})
    return out


def path(boxes, zones: list[dict]) -> dict:
    """For one target: its crossings and visits per zone uid, and its path (zone uids in time order)."""
    hits: list[tuple[float, str]] = []
    per: dict[str, list] = {}
    for z in zones:
        if z["kind"] == "line":
            r = crossings(boxes, z["points"])
            hits += [(c["t"], z["uid"]) for c in r]
        else:
            r = visits(boxes, z["points"])
            hits += [(v["enter"], z["uid"]) for v in r]
        if r:
            per[z["uid"]] = r
    seq: list[str] = []
    for _, uid in sorted(hits):
        if not seq or seq[-1] != uid:
            seq.append(uid)
    return {"zones": per, "path": seq}


def went(p: list[str], a: str, b: str) -> bool:
    """Whether a path visits a, then later b."""
    try:
        i = p.index(a)
    except ValueError:
        return False
    return b in p[i + 1:]


class Zones:
    """A camera's lines and areas, kept by folder (every video from that folder uses them)."""

    def __init__(self, mission):
        self.m = mission
        self.con = mission.con

    def folder(self, version_id: int) -> str:
        r = self.con.execute("SELECT f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                             "WHERE v.id=?", (version_id,)).fetchone()
        if r is None:
            raise ZoneError("That video is not in this mission.")
        return str(PurePosixPath(r[0]).parent)

    def of_folder(self, folder: str) -> list[dict]:
        rows = self.con.execute("SELECT * FROM video_zone WHERE folder=? AND deleted_at IS NULL ORDER BY id", (folder,))
        return [{"uid": r["uid"], "kind": r["kind"], "name": r["name"], "points": json.loads(r["points"]),
                 "author": r["author"], "created_at": r["created_at"]} for r in rows]

    def of(self, version_id: int) -> list[dict]:
        return self.of_folder(self.folder(version_id))

    def add(self, version_id: int, kind: str, name: str, points: list, author: str) -> dict:
        if kind not in ("line", "area"):
            raise ZoneError("A zone is a line or an area.")
        pts = [[float(x), float(y)] for x, y in points]
        if kind == "line" and (len(pts) != 2 or ((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2) < 100):
            raise ZoneError("A line needs two points at least 10 pixels apart.")
        if kind == "area" and len(pts) < 3:
            raise ZoneError("An area needs three points or more.")
        name = (name or "").strip()[:60] or ("Line" if kind == "line" else "Area")
        folder = self.folder(version_id)
        taken = {z["name"] for z in self.of_folder(folder)}
        base, k = name, 2
        while name in taken:
            name, k = f"{base} {k}", k + 1
        uid = secrets.token_hex(6)
        self.con.execute("INSERT INTO video_zone(uid,folder,kind,name,points,author,created_at) VALUES (?,?,?,?,?,?,?)",
                         (uid, folder, kind, name, json.dumps(pts), author, utcnow()))
        return next(z for z in self.of_folder(folder) if z["uid"] == uid)

    def delete(self, uid: str, author: str) -> None:
        n = self.con.execute("UPDATE video_zone SET deleted_at=?, deleted_by=? WHERE uid=? AND deleted_at IS NULL",
                             (utcnow(), author, uid)).rowcount
        if not n:
            raise ZoneError("That line or area does not exist (any more).")


def tally(svc, evs) -> dict:
    """Per video: each zone's counts by class (lines: in and out; areas: entries and time inside), and each event's
    path. Only counted events are tallied."""
    zs = Zones(svc.m)
    by_vid: dict[int, list] = {}
    for e in evs:
        by_vid.setdefault(e.source_version_id, []).append(e)
    boxes = svc.boxes_by_event(evs)
    out: dict = {"recordings": [], "events": {}}
    for vid, vs in sorted(by_vid.items()):
        zones = zs.of(vid)
        if not zones:
            continue
        rec = {"source_version_id": vid, "file": vs[0].file, "zones": []}
        per_zone = {z["uid"]: {"uid": z["uid"], "name": z["name"], "kind": z["kind"], "by_class": {}} for z in zones}
        for e in vs:
            p = path(boxes[e.id], zones)
            out["events"][e.id] = p
            if not e.counted:
                continue
            for uid, hits in p["zones"].items():
                c = per_zone[uid]["by_class"].setdefault(e.cls, {"in": 0, "out": 0} if per_zone[uid]["kind"] == "line"
                                                         else {"entries": 0, "inside_s": 0.0})
                if per_zone[uid]["kind"] == "line":
                    for h in hits:
                        c[h["dir"]] += 1
                else:
                    c["entries"] += len(hits)
                    c["inside_s"] = round(c["inside_s"] + sum(h["inside_s"] for h in hits), 2)
        rec["zones"] = list(per_zone.values())
        out["recordings"].append(rec)
    return out
