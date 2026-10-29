"""Inclusion and exclusion regions per camera view.

- Coordinates are in the view's original image pixels. Display zoom never changes them.
- Precedence: exclusion wins. With no inclusion region, the whole frame is included.
- Membership rule for an object box (recorded on each run):
  bottom_center (default; approximate ground contact), centroid, or overlap
  (share of the box inside included-and-not-excluded area, estimated on a fixed
  11 x 11 grid of points so results are deterministic).
- Every edit is a new revision; deleting a region adds a deleted revision.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

from .store import utcnow

Point = tuple[float, float]
RULES = ("bottom_center", "centroid", "overlap")


def point_in_polygon(x: float, y: float, poly: list[Point]) -> bool:
    """Even-odd rule; points exactly on an edge count as inside."""
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # on-edge test
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1, o2, o3, o4 = orient(p1, p2, p3), orient(p1, p2, p4), orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def validate_polygon(poly: list[Point], width: int, height: int) -> list[str]:
    errs = []
    if len(poly) < 3:
        errs.append("A region needs at least 3 points.")
        return errs
    for x, y in poly:
        if not (0 <= x <= width and 0 <= y <= height):
            errs.append(f"Point ({x:g}, {y:g}) lies outside the {width}x{height} image.")
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segments_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                errs.append("Region edges cross each other; draw the outline without crossing lines.")
                return errs
    area = 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n)))
    if area < 1:
        errs.append("Region has no area.")
    return errs


@dataclass(frozen=True)
class Region:
    uid: str
    kind: str  # include | exclude
    name: str | None
    polygon: list[Point]


class RegionSet:
    def __init__(self, regions: list[Region]):
        self.include = [r for r in regions if r.kind == "include"]
        self.exclude = [r for r in regions if r.kind == "exclude"]

    @classmethod
    def load(cls, con: sqlite3.Connection, view_id: int) -> "RegionSet":
        rows = con.execute(
            "SELECT r.* FROM region_revision r WHERE r.view_id=? AND r.id = "
            "(SELECT MAX(id) FROM region_revision x WHERE x.region_uid=r.region_uid) ORDER BY r.id", (view_id,)).fetchall()
        return cls([Region(r["region_uid"], r["kind"], r["name"], [tuple(p) for p in json.loads(r["polygon"])])
                    for r in rows if not r["deleted"]])

    def mask(self, width: int, height: int):
        """Boolean image of the scan area (included and not excluded), for skipping parts of a frame."""
        import cv2
        import numpy as np
        key = (width, height)
        if getattr(self, "_mask_key", None) != key:
            m = np.zeros((height, width), np.uint8)
            if not self.include:
                m[:] = 1
            for r in self.include:
                cv2.fillPoly(m, [np.round(np.array(r.polygon)).astype(np.int32)], 1)
            for r in self.exclude:
                cv2.fillPoly(m, [np.round(np.array(r.polygon)).astype(np.int32)], 0)
            self._mask, self._mask_key = m.astype(bool), key
        return self._mask

    def vertices(self) -> list[Point]:
        return [p for r in self.include + self.exclude for p in r.polygon]

    def contains_point(self, x: float, y: float) -> bool:
        if any(point_in_polygon(x, y, r.polygon) for r in self.exclude):
            return False
        return not self.include or any(point_in_polygon(x, y, r.polygon) for r in self.include)

    def contains_box(self, box: tuple[float, float, float, float], rule: str = "bottom_center", min_overlap: float = 0.5) -> bool:
        x1, y1, x2, y2 = box
        if rule == "bottom_center":
            return self.contains_point((x1 + x2) / 2, y2)
        if rule == "centroid":
            return self.contains_point((x1 + x2) / 2, (y1 + y2) / 2)
        if rule == "overlap":
            n = 11
            pts = [(x1 + (x2 - x1) * i / (n - 1), y1 + (y2 - y1) * j / (n - 1)) for i in range(n) for j in range(n)]
            return sum(self.contains_point(x, y) for x, y in pts) / len(pts) >= min_overlap
        raise ValueError(f"unknown rule {rule}; use one of {RULES}")


class RegionEditor:
    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def _size(self, view_id: int) -> tuple[int, int]:
        r = self.con.execute("SELECT ref_width, ref_height FROM camera_view WHERE id=?", (view_id,)).fetchone()
        return r["ref_width"], r["ref_height"]

    def save(self, view_id: int, kind: str, polygon: list[Point], author: str, name: str | None = None,
             region_uid: str | None = None) -> str:
        """Create a region (no uid) or a new revision of an existing one."""
        if kind not in ("include", "exclude"):
            raise ValueError("kind must be include or exclude")
        errs = validate_polygon([tuple(map(float, p)) for p in polygon], *self._size(view_id))
        if errs:
            raise ValueError(" ".join(errs))
        uid = region_uid or uuid.uuid4().hex
        self.con.execute(
            "INSERT INTO region_revision(view_id,region_uid,kind,name,polygon,author,created_at) VALUES (?,?,?,?,?,?,?)",
            (view_id, uid, kind, name, json.dumps([list(map(float, p)) for p in polygon]), author, utcnow()))
        return uid

    def delete(self, view_id: int, region_uid: str, author: str) -> None:
        last = self.con.execute("SELECT * FROM region_revision WHERE region_uid=? ORDER BY id DESC LIMIT 1", (region_uid,)).fetchone()
        self.con.execute(
            "INSERT INTO region_revision(view_id,region_uid,kind,name,polygon,deleted,author,created_at) VALUES (?,?,?,?,?,1,?,?)",
            (view_id, region_uid, last["kind"], last["name"], last["polygon"], author, utcnow()))

    def history(self, region_uid: str) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM region_revision WHERE region_uid=? ORDER BY id", (region_uid,))]
