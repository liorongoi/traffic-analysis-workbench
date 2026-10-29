"""Vehicles found as changes against empty-road images of the same camera view (still images).

The operator marks a few images of a view in which the road is empty, by day and by night. Their per-pixel
median becomes the view's empty background for that light condition (day, dusk, night); a frame is compared
only with the background of its own condition, because day and night pixels differ everywhere.

In a frame, what differs from the empty background inside the scan area is a change. Changes close together
(two headlights and their reflection) merge into one. Each change that no detector box or operator box
already covers counts as a vehicle: glare from headlights is accepted as a vehicle. When SAM 3.1 is installed
it looks at a zoomed-in crop of each change; when it sees a car, truck, bus, motorcycle or bicycle there, the
vehicle gets that class and SAM's tighter box, otherwise its class stays "vehicle".

Everything is kept: the empty images chosen, each background built (immutable), and each comparison with its
parameters. Operators correct these vehicles like detector boxes (remove, change class, move).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .night import dumps, measure_condition, png, unpng
from .regions import RegionSet
from .store import Mission, transaction, utcnow

ALGO = "change-v2"
CONDITIONS = ("day", "dusk", "night")
# A frame without a background of its own condition borrows a close one: dusk and night stand in for each other.
FALLBACK = {"day": (), "dusk": ("night",), "night": ("dusk",)}
UNTYPED = "vehicle"
NOT_COUNTED = {"covered_by_detection", "no_lights_at_night"}   # stored for the record, never counted


@dataclass
class ChangeParams:
    blur: int = 5                  # px, Gaussian; evens out sensor noise and JPEG blocks
    min_diff: float = 28.0         # grey levels a pixel must differ by, at the least
    spread_factor: float = 3.0     # ... and this many times the spread between the empty images at that pixel
    open_px: int = 3               # removes speckle
    close_px: int = 9              # fills holes inside one vehicle
    min_area_share: float = 0.00012  # of the frame: about 110 px on 1280 x 720
    merge_gap_share: float = 0.02  # changes closer than this share of the frame width are one vehicle
    inside_share: float = 0.3      # share of a change that must lie in the scan area
    lights_delta: float = 40.0     # brighter than the background by this much on average: lights or glare
    core_level: float = 225.0      # a headlight or glare has a near-white core; light thrown on the road does not
    new_core_px: int = 4           # near-white pixels a change needs where the empty background is not bright already
    lamp_level: float = 200.0      # bright in the empty background: street lamps, lit signs
    lamp_reach_share: float = 0.015  # around them (share of frame width) a halo grows and shrinks with fog: no new lights
    road_light_reach_share: float = 0.06   # soft road light joins bright lights this close (share of frame width)
    sam_threshold: float = 0.35
    sam_context: float = 1.0       # crop margin on each side, in units of the change's own size
    sam_min_crop: int = 320        # px; small far changes are cropped at least this large
    area_prompts: tuple[str, ...] = ("road",)   # with no regions drawn, SAM 3.1 outlines these on the empty background
    area_threshold: float = 0.4

    def record(self) -> dict:
        return asdict(self)


@dataclass
class Change:
    box: tuple[float, float, float, float]
    score: float
    kind: str                      # lights | glow | shape (see _kind)
    parts: int
    changed_px: int
    cls: str = UNTYPED
    sam_cls: str | None = None
    sam_score: float | None = None
    change_box: tuple[float, float, float, float] | None = None
    in_region: int | None = None
    flags: list[str] = field(default_factory=list)


def to_gray(img_bgr: np.ndarray, blur: int) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    g = g.astype(np.float32)
    return cv2.GaussianBlur(g, (blur | 1, blur | 1), 0) if blur > 1 else g


def build_reference(grays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of the empty images, and how much they differ from it (robust spread, 0 with < 3)."""
    stack = np.stack(grays).astype(np.float32)
    med = np.median(stack, axis=0)
    if len(grays) >= 3:
        spread = 1.4826 * np.median(np.abs(stack - med), axis=0)
    else:
        spread = np.zeros_like(med)
    return med, spread


def _merge(boxes: list[tuple[int, int, int, int]], gap: float, height: int | None = None) -> list[list[int]]:
    """Groups of box indexes whose boxes come within `gap` px of each other (transitively). With `height`, the
    gap grows towards the bottom of the frame, where vehicles are closer and look bigger."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
            g = gap if not height else gap * (0.4 + 1.2 * max(a[3], b[3]) / height)
            if max(dx, dy) <= g:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def find_changes(gray: np.ndarray, background: np.ndarray, spread: np.ndarray, area: np.ndarray | None,
                 p: ChangeParams | None = None) -> tuple[list[Change], float]:
    """Changes against the empty background, inside `area` (boolean, None: the whole frame).

    Returns the changes and the overall brightness shift that was taken out first (exposure, wet road)."""
    p = p or ChangeParams()
    h, w = gray.shape
    if background.shape != gray.shape:
        raise ValueError(f"image is {w}x{h} but the empty background is {background.shape[1]}x{background.shape[0]}")
    inside = area if area is not None else np.ones((h, w), bool)
    if not inside.any():
        return [], 0.0
    diff = gray - background
    shift = float(np.median(diff[inside]))          # the frame as a whole is brighter or darker
    diff -= shift
    thr = np.maximum(p.min_diff, p.spread_factor * spread)
    changed = (np.abs(diff) >= thr).astype(np.uint8)
    if p.open_px > 1:
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((p.open_px, p.open_px), np.uint8))
    if p.close_px > 1:
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((p.close_px, p.close_px), np.uint8))
    lamps = _lamp_zone(background, p, w)
    new_core = (gray >= p.core_level) & ~lamps        # lights that are not in the empty road
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    min_area = max(12, int(p.min_area_share * w * h))
    keep, boxes = [], []
    for i in range(1, n):
        x, y, bw, bh, a = (int(v) for v in stats[i])
        if a < min_area:
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        if inside[y:y + bh, x:x + bw][comp].mean() < p.inside_share:
            continue
        keep.append(i)
        boxes.append((x, y, x + bw, y + bh))
    groups = _merge(boxes, p.merge_gap_share * w, h)
    groups = _merge_road_light(groups, boxes, labels, keep, diff, new_core, p, w)
    out = []
    for grp in groups:
        x1 = min(boxes[i][0] for i in grp)
        y1 = min(boxes[i][1] for i in grp)
        x2 = max(boxes[i][2] for i in grp)
        y2 = max(boxes[i][3] for i in grp)
        sel = np.isin(labels[y1:y2, x1:x2], [keep[i] for i in grp])
        d = diff[y1:y2, x1:x2][sel]
        core = new_core[y1:y2, x1:x2] & sel
        kind = _kind(float(d.mean()), int(core.sum()), p)
        score = round(min(1.0, float(np.abs(d).mean()) / 100.0), 3)
        full = (float(x1), float(y1), float(x2), float(y2))
        c = Change(full, score, kind, len(grp), int(sel.sum()))
        if kind == "lights":
            c.box, c.change_box = _vehicle_at_lights(core, x1, y1, full), full
        out.append(c)
    out.sort(key=lambda c: (c.box[1], c.box[0]))
    return out, round(shift, 2)


def _lamp_zone(background: np.ndarray, p: "ChangeParams", w: int) -> np.ndarray:
    """Around every light already in the empty background (street lamps, lit signs): its halo grows and shrinks
    with fog and exposure, so white there is not a new light. Each lamp's zone grows with its own size."""
    bright = (background >= p.lamp_level).astype(np.uint8)
    if not bright.any():
        return np.zeros(background.shape, bool)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    reach = p.lamp_reach_share * w + 2.0 * np.minimum(0.05 * w, np.sqrt(stats[:, cv2.CC_STAT_AREA] / np.pi))
    # distance of every pixel to the nearest bright pixel, and which lamp that pixel belongs to
    dist, near = cv2.distanceTransformWithLabels(1 - bright, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(bright)
    owner = np.zeros(int(near.max()) + 1, np.int32)
    owner[near[ys, xs]] = comp[ys, xs]
    return dist <= reach[owner[near]]


def _vehicle_at_lights(core: np.ndarray, ox: int, oy: int, full) -> tuple[float, float, float, float]:
    """The vehicle around its lights: the lit road ahead and the halo are left out of the box (kept as change_box)."""
    ys, xs = np.nonzero(core)
    cx1, cy1, cx2, cy2 = xs.min() + ox, ys.min() + oy, xs.max() + 1 + ox, ys.max() + 1 + oy
    s = max(cx2 - cx1, cy2 - cy1, 6)
    box = (max(full[0], cx1 - 0.5 * s), max(full[1], cy1 - 0.6 * s), min(full[2], cx2 + 0.5 * s), min(full[3], cy2 + 0.4 * s))
    return tuple(round(float(v), 1) for v in box)


def _kind(brighter: float, new_core: int, p: "ChangeParams") -> str:
    """lights: brighter, with a near-white core where the empty road has none (headlights, glare); glow: brighter
    but soft (light thrown on the road, a lamp's halo); shape: anything else (a vehicle by day, a shadow)."""
    if brighter >= p.lights_delta / 2 and new_core >= p.new_core_px:
        return "lights"
    if brighter >= p.lights_delta / 2:
        return "glow"
    return "shape"


def _merge_road_light(groups: list[list[int]], boxes, labels, keep, diff, new_core, p: "ChangeParams", w: int) -> list[list[int]]:
    """Light a vehicle throws on the road (brighter, but soft, with no near-white core) joins the nearest bright
    lights within a wider reach, so a car and the lit asphalt ahead of it count once."""
    def stats(grp):
        x1 = min(boxes[i][0] for i in grp); y1 = min(boxes[i][1] for i in grp)
        x2 = max(boxes[i][2] for i in grp); y2 = max(boxes[i][3] for i in grp)
        sel = np.isin(labels[y1:y2, x1:x2], [keep[i] for i in grp])
        return (x1, y1, x2, y2), float(diff[y1:y2, x1:x2][sel].mean()), int((new_core[y1:y2, x1:x2] & sel).sum())

    info = [stats(g) for g in groups]
    kinds = [_kind(d, nc, p) for _, d, nc in info]
    core = [k == "lights" for k in kinds]
    soft = [k == "glow" for k in kinds]
    reach = p.road_light_reach_share * w
    merged = {i: list(g) for i, g in enumerate(groups)}
    for i in [k for k, s_ in enumerate(soft) if s_]:
        bi = info[i][0]
        best = None
        for j, c in enumerate(core):
            if not c or j not in merged:
                continue
            bj = info[j][0]
            dx = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            dy = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
            d = max(dx, dy)
            if d <= reach and (best is None or d < best[0]):
                best = (d, j)
        if best is not None and i in merged:
            merged[best[1]] += merged.pop(i)
    return list(merged.values())


def overlaps(a, b, share: float = 0.5) -> bool:
    """Most of the smaller box lies in the other one (a detector box inside a large glare, or the reverse)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return small > 0 and ix * iy / small >= share


def sam_check(engine, img_bgr: np.ndarray, c: Change, p: ChangeParams) -> None:
    """Ask SAM 3.1 about a zoomed-in crop around the change; type the vehicle when it sees one there."""
    from .sam3 import VEHICLE_PROMPTS
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = c.box
    bw, bh = x2 - x1, y2 - y1
    mx = max(bw * p.sam_context, (p.sam_min_crop - bw) / 2)
    my = max(bh * p.sam_context, (p.sam_min_crop - bh) / 2)
    cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = img_bgr[cy1:cy2, cx1:cx2]
    prompts = {q: cls for cls, qs in VEHICLE_PROMPTS.items() for q in qs}
    found = engine.segment(crop, list(prompts), p.sam_threshold)
    best = None
    for q, insts in found.items():
        for inst in insts:
            bx1, by1, bx2, by2 = inst.box
            box = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
            mcx, mcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            ex, ey = bw * 0.25 + 4, bh * 0.25 + 4
            if not (x1 - ex <= mcx <= x2 + ex and y1 - ey <= mcy <= y2 + ey):
                continue          # a vehicle elsewhere in the crop is not this change
            if best is None or inst.score > best[1]:
                best = (prompts[q], inst.score, box)
    if best is None:
        c.flags.append("sam_saw_no_vehicle")
        return
    c.sam_cls, c.sam_score = best[0], round(float(best[1]), 4)
    c.cls = best[0]
    c.change_box = c.box
    c.box = tuple(round(v, 1) for v in best[2])
    c.flags.append("typed_by_sam")


class ChangeService:
    def __init__(self, mission: Mission, params: ChangeParams | None = None):
        self.m = mission
        self.con: sqlite3.Connection = mission.con
        self.p = params or ChangeParams()

    def describe(self) -> dict:
        return {"algo": ALGO, "params": self.p.record(), "conditions": list(CONDITIONS),
                "fallback": {k: list(v) for k, v in FALLBACK.items()}}

    # ---------- the empty images an operator chose ----------
    def _view_of(self, version_id: int) -> int:
        from .views import ViewService
        view_id = ViewService(self.m).current_view(version_id)
        if view_id is None:
            raise ValueError("This image is not in a camera view yet, so it cannot be an empty background.")
        return view_id

    def mark(self, version_id: int, author: str, empty: bool = True) -> dict:
        view_id = self._view_of(version_id)
        if empty and self.condition_of(version_id) is None:
            raise ValueError("This image has not been processed yet. Process it first, then mark it as empty.")
        self.con.execute("INSERT INTO reference_mark(view_id,source_version_id,action,author,created_at) VALUES (?,?,?,?,?)",
                         (view_id, version_id, "marked" if empty else "unmarked", author, utcnow()))
        return self.references(view_id)

    def marked(self, view_id: int) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT m.source_version_id FROM reference_mark m WHERE m.view_id=? AND m.id=(SELECT MAX(id) FROM reference_mark x "
            "WHERE x.view_id=m.view_id AND x.source_version_id=m.source_version_id) AND m.action='marked' "
            "AND EXISTS (SELECT 1 FROM source_version v WHERE v.id=m.source_version_id AND v.superseded_by IS NULL) "
            "ORDER BY m.source_version_id", (view_id,))]

    def is_marked(self, version_id: int) -> bool:
        r = self.con.execute("SELECT action FROM reference_mark WHERE source_version_id=? ORDER BY id DESC LIMIT 1",
                             (version_id,)).fetchone()
        return r is not None and r[0] == "marked"

    def condition_of(self, version_id: int) -> str | None:
        r = self.con.execute("SELECT condition FROM image_condition WHERE source_version_id=?", (version_id,)).fetchone()
        return r[0] if r else None

    def references(self, view_id: int) -> dict:
        """Per light condition: the empty images marked and the background currently built from them."""
        by: dict[str, list[int]] = {c: [] for c in CONDITIONS}
        for v in self.marked(view_id):
            c = self.condition_of(v)
            if c in by:
                by[c].append(v)
        out = {}
        for c in CONDITIONS:
            bg = self.current(view_id, c)
            check = json.loads(bg["params"]).get("empty_check", []) if bg is not None else []
            area = None
            if bg is not None:
                mask, aid, source = self.area_for(view_id, bg)
                area = {"source": source, "share": round(float(mask.mean()), 4) if mask is not None else 1.0}
                ar = self.area_row(bg["id"])
                if source == "sam" and ar is not None:
                    check = json.loads(ar["empty_check"])
            out[c] = {"images": by[c], "background_id": bg["id"] if bg is not None else None, "empty_check": check,
                      "area": area,
                      "built_from": json.loads(bg["input_ids"]) if bg is not None else [],
                      "up_to_date": (bg is not None and json.loads(bg["input_ids"]) == by[c]) or (bg is None and not by[c])}
        return {"view_id": view_id, "conditions": out, "fallback": {k: list(v) for k, v in FALLBACK.items()}}

    # ---------- backgrounds ----------
    def current(self, view_id: int, condition: str) -> sqlite3.Row | None:
        r = self.con.execute("SELECT * FROM reference_background WHERE view_id=? AND condition=? ORDER BY id DESC LIMIT 1",
                             (view_id, condition)).fetchone()
        return r if r is not None and json.loads(r["input_ids"]) else None

    def ensure(self, view_id: int, condition: str) -> sqlite3.Row | None:
        """The view's background for this condition, rebuilt when the marked images changed."""
        from .views import ViewService
        want = [v for v in self.marked(view_id) if self.condition_of(v) == condition]
        cur = self.con.execute("SELECT * FROM reference_background WHERE view_id=? AND condition=? ORDER BY id DESC LIMIT 1",
                               (view_id, condition)).fetchone()
        if cur is not None and json.loads(cur["input_ids"]) == want:
            return cur if want else None
        if not want:
            if cur is None:
                return None
            # Every empty image was unmarked: record that the background no longer applies.
            self.con.execute("INSERT INTO reference_background(view_id,condition,input_ids,image_count,width,height,median,spread,"
                             "params,built_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (view_id, condition, "[]", 0, 0, 0, b"", b"", dumps(self.p.record()), utcnow()))
            return None
        vs = ViewService(self.m)
        grays, used = [], []
        size = None
        for v in want:
            try:
                g = to_gray(vs.image(v), self.p.blur)
            except Exception:
                continue          # the file is gone; build from the others
            if size is None:
                size = g.shape
            if g.shape == size:
                grays.append(g)
                used.append(v)
        if not grays:
            return None
        med, spread = build_reference(grays)
        # Each empty image against the others: changes found there are false alarms (rain, lamps, camera shake).
        regions = RegionSet.load(self.con, view_id) if vs.regions_applicable(view_id) else None
        area = regions.mask(size[1], size[0]) if regions is not None else None
        check = self._empty_check(grays, used, area, condition)
        rid = self.con.execute(
            "INSERT INTO reference_background(view_id,condition,input_ids,image_count,width,height,median,spread,params,built_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (view_id, condition, dumps(want), len(used), size[1], size[0], png(np.clip(med, 0, 255).astype(np.uint8)),
             png(np.clip(spread, 0, 255).astype(np.uint8)), dumps(self.p.record() | {"used_ids": used, "empty_check": check}), utcnow())).lastrowid
        return self.con.execute("SELECT * FROM reference_background WHERE id=?", (rid,)).fetchone()

    def _empty_check(self, grays: list[np.ndarray], used: list[int], area, condition: str) -> list[dict]:
        """Each empty image against a background of the others: what would count there is a false alarm."""
        if len(grays) < 3:
            return []
        dark = condition in ("night", "dusk")
        out = []
        for i, v in enumerate(used):
            m_, s_ = build_reference(grays[:i] + grays[i + 1:])
            ch = find_changes(grays[i], m_, s_, area, self.p)[0]
            out.append({"source_version_id": v, "changes": sum(1 for c in ch if not dark or c.kind == "lights")})
        return out

    # ---------- where changes are looked for ----------
    def area_row(self, background_id: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM reference_area WHERE background_id=? ORDER BY id DESC LIMIT 1",
                                (background_id,)).fetchone()

    def _auto_area(self, view_id: int, bg: sqlite3.Row, engine, sam_model_id: int | None, exclude) -> sqlite3.Row:
        """SAM 3.1 outlines the road on the empty background itself: no vehicles hide it, and day and night each
        get their own outline."""
        from .sam3 import find_areas
        from .views import ViewService
        med = unpng(bg["median"])
        found = find_areas(engine, cv2.cvtColor(med, cv2.COLOR_GRAY2BGR), list(self.p.area_prompts), "include",
                           self.p.area_threshold)
        polys = [{"kind": a.kind, "polygon": [list(pt) for pt in a.polygon]} for a in found]
        mask = self._poly_mask(polys, bg["width"], bg["height"], exclude)
        vs = ViewService(self.m)
        used = json.loads(bg["params"]).get("used_ids", json.loads(bg["input_ids"]))
        grays, ids = [], []
        for v in used:
            try:
                g = to_gray(vs.image(v), self.p.blur)
            except Exception:
                continue
            if g.shape == med.shape:
                grays.append(g)
                ids.append(v)
        check = self._empty_check(grays, ids, mask, bg["condition"]) if mask is not None else []
        share = float(mask.mean()) if mask is not None else 1.0
        rid = self.con.execute(
            "INSERT INTO reference_area(background_id,sam_model_id,prompts,polygons,area_share,empty_check,created_at) "
            "VALUES (?,?,?,?,?,?,?)", (bg["id"], sam_model_id, dumps(list(self.p.area_prompts)), dumps(polys),
                                       round(share, 4), dumps(check), utcnow())).lastrowid
        return self.con.execute("SELECT * FROM reference_area WHERE id=?", (rid,)).fetchone()

    @staticmethod
    def _poly_mask(polys: list[dict], w: int, h: int, exclude=None):
        if not any(p_["kind"] == "include" for p_ in polys):
            return None
        m = np.zeros((h, w), np.uint8)
        for k in ("include", "exclude"):
            for p_ in polys:
                if p_["kind"] == k:
                    cv2.fillPoly(m, [np.round(np.array(p_["polygon"])).astype(np.int32)], 1 if k == "include" else 0)
        m = m.astype(bool)
        return m & exclude if exclude is not None else m

    def area_for(self, view_id: int, bg: sqlite3.Row, engine=None, sam_model_id: int | None = None):
        """Where changes are looked for: the view's include regions; without any, the road SAM 3.1 found on the
        empty background (found now when SAM is at hand); otherwise the whole frame. Exclusions always apply.

        Returns (mask or None, area row id or None, source: regions | sam | whole_frame)."""
        from .views import ViewService
        w, h = bg["width"], bg["height"]
        regions = RegionSet.load(self.con, view_id) if ViewService(self.m).regions_applicable(view_id) else None
        if regions is not None and regions.include:
            return regions.mask(w, h), None, "regions"
        exclude = regions.mask(w, h) if regions is not None and regions.exclude else None
        row = self.area_row(bg["id"])
        if row is None and engine is not None:
            row = self._auto_area(view_id, bg, engine, sam_model_id, exclude)
        if row is None:
            return exclude, None, "whole_frame"
        mask = self._poly_mask(json.loads(row["polygons"]), w, h, exclude)
        return (mask, row["id"], "sam") if mask is not None else (exclude, row["id"], "whole_frame")

    def background_for(self, view_id: int, condition: str | None) -> sqlite3.Row | None:
        if condition is None:
            return None
        for c in (condition, *FALLBACK.get(condition, ())):
            bg = self.ensure(view_id, c)
            if bg is not None:
                return bg
        return None

    @staticmethod
    def arrays(bg: sqlite3.Row) -> tuple[np.ndarray, np.ndarray]:
        return unpng(bg["median"]).astype(np.float32), unpng(bg["spread"]).astype(np.float32)

    # ---------- comparing images ----------
    def latest(self, version_id: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM change_analysis WHERE source_version_id=? ORDER BY id DESC LIMIT 1",
                                (version_id,)).fetchone()

    def vehicles(self, version_id: int) -> list[sqlite3.Row]:
        a = self.latest(version_id)
        if a is None or a["reference_id"] is None:
            return []
        return self.con.execute("SELECT * FROM change_vehicle WHERE analysis_id=? ORDER BY id", (a["id"],)).fetchall()

    def _region_revision(self, view_id: int) -> int | None:
        return self.con.execute("SELECT MAX(id) FROM region_revision WHERE view_id=?", (view_id,)).fetchone()[0]

    def analyse(self, version_id: int, view_id: int, bg: sqlite3.Row | None, img: np.ndarray | None = None,
                engine=None, sam_model_id: int | None = None, area=None) -> int:
        """Compare one image with its view's empty background and store the result. Returns the vehicles kept."""
        from .review import ReviewService
        from .views import ViewService
        vs = ViewService(self.m)
        region_rev = self._region_revision(view_id)
        if bg is None or self.is_marked(version_id):
            # No background applies, or the image is one of the empty ones: an empty result, so nothing old counts.
            self.con.execute("INSERT INTO change_analysis(source_version_id,reference_id,region_revision_id,algo,params,created_at) "
                             "VALUES (?,?,?,?,?,?)", (version_id, None, region_rev, ALGO, dumps(self.p.record()), utcnow()))
            return 0
        img = img if img is not None else vs.image(version_id)
        gray = to_gray(img, self.p.blur)
        med, spread = self.arrays(bg)
        mask, area_id, _ = area if area is not None else self.area_for(view_id, bg, engine, sam_model_id)
        if gray.shape != med.shape:
            raise ValueError("This image's size differs from its view's empty background.")
        changes, shift = find_changes(gray, med, spread, mask, self.p)
        own_view = vs.current_view(version_id) == view_id
        others = [o.box for o in ReviewService(self.m).objects(version_id, include_removed=False) if o.origin != "change"]
        dark = self.condition_of(version_id) in ("night", "dusk")
        for c in changes:
            if any(overlaps(c.box, b) for b in others):
                c.flags.append("covered_by_detection")
                continue          # already counted; SAM is not asked
            if engine is not None:
                try:
                    sam_check(engine, img, c, self.p)
                except Exception as exc:  # a SAM failure must not lose the change
                    c.flags.append(f"sam_failed: {type(exc).__name__}")
            else:
                c.flags.append("sam_not_installed")
            if not own_view:
                c.flags.append("background_from_nearest_view")
            if dark and c.kind != "lights" and c.sam_cls is None:
                # At night a vehicle shows its lights; a soft or dark change alone is light on the road or a shadow.
                c.flags.append("no_lights_at_night")
            if mask is not None:
                bx = int(min(max((c.box[0] + c.box[2]) / 2, 0), mask.shape[1] - 1))
                by = int(min(max(c.box[3] - 1, 0), mask.shape[0] - 1))
                c.in_region = int(mask[by, bx])
        aid = self.con.execute(
            "INSERT INTO change_analysis(source_version_id,reference_id,region_revision_id,algo,params,sam_model_id,shift,"
            "area_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (version_id, bg["id"], region_rev, ALGO, dumps(self.p.record()), sam_model_id if engine is not None else None,
             shift, area_id, utcnow())).lastrowid
        kept = 0
        for c in changes:
            self.con.execute(
                "INSERT INTO change_vehicle(analysis_id,source_version_id,x1,y1,x2,y2,change_box,cls,score,kind,parts,changed_px,"
                "sam_cls,sam_score,in_region,flags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (aid, version_id, *c.box, dumps(list(c.change_box or c.box)), c.cls, c.score, c.kind, c.parts, c.changed_px,
                 c.sam_cls, c.sam_score, c.in_region, dumps(c.flags)))
            kept += not set(c.flags) & NOT_COUNTED
        return kept

    def _images(self, view_id: int) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT v.id FROM source_version v JOIN view_assignment a ON a.source_version_id=v.id AND a.superseded_by IS NULL "
            "WHERE a.view_id=? AND v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id "
            "AND i.processing_state='done') ORDER BY v.capture_utc, v.id", (view_id,))]

    def refresh(self, engine=None, sam_model_id: int | None = None, control=None, progress=None) -> dict:
        """Bring every processed image of every view with empty images up to date.

        An image is compared again when its view's background or regions changed, or when SAM 3.1 became available."""
        stats = {"views": 0, "images_compared": 0, "vehicles": 0}
        views = [r[0] for r in self.con.execute("SELECT DISTINCT view_id FROM reference_mark")]
        # Images left out of every view (bright lights, rain on the lens) borrow the most similar view's background.
        from .night_service import NightService
        ns = NightService(self.m)
        borrowed: dict[int, list[int]] = {}
        for (v,) in self.con.execute(
                "SELECT v.id FROM source_version v WHERE v.superseded_by IS NULL AND NOT EXISTS (SELECT 1 FROM view_assignment a "
                "WHERE a.source_version_id=v.id AND a.superseded_by IS NULL AND a.view_id IS NOT NULL) AND EXISTS "
                "(SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND i.processing_state='done')").fetchall():
            near = ns.nearest_view(v)
            if near in views:
                borrowed.setdefault(near, []).append(v)
        work = []
        for view_id in views:
            with transaction(self.con):
                bgs = {c: self.background_for(view_id, c) for c in CONDITIONS}
            with transaction(self.con):
                areas = {bg["id"]: self.area_for(view_id, bg, engine, sam_model_id) for bg in bgs.values() if bg is not None}
            region_rev = self._region_revision(view_id)
            stats["views"] += 1
            for v in self._images(view_id) + borrowed.get(view_id, []):
                cond = self.condition_of(v)
                bg = bgs.get(cond) if cond else None
                if bg is None and cond is None:
                    continue
                last = self.latest(v)
                want_ref = None if (bg is None or self.is_marked(v)) else bg["id"]
                area = areas.get(want_ref) if want_ref is not None else None
                if last is not None and last["reference_id"] == want_ref and last["algo"] == ALGO and (
                        want_ref is None or (last["region_revision_id"] == region_rev and last["area_id"] == area[1]
                                             and (engine is None or last["sam_model_id"] is not None))):
                    continue
                if last is None and want_ref is None:
                    continue
                work.append((v, view_id, bg, area))
        for n, (v, view_id, bg, area) in enumerate(work, 1):
            if control is not None:
                control.checkpoint()
            with transaction(self.con):
                stats["vehicles"] += self.analyse(v, view_id, bg, engine=engine, sam_model_id=sam_model_id, area=area)
            stats["images_compared"] += 1
            if progress:
                progress(n, len(work))
        return stats

    def suggest_empty(self, view_id: int, limit: int = 12) -> list[dict]:
        """Images of the view with the fewest vehicles, per light condition: likely candidates for empty images."""
        from .review import ReviewService
        rs = ReviewService(self.m)
        rows = []
        for v in self._images(view_id):
            cond = self.condition_of(v)
            if cond is None:
                continue
            objs = [o for o in rs.objects(v, include_removed=False) if o.origin != "change"]
            lights = self.con.execute("SELECT COUNT(*) FROM light_group WHERE source_version_id=? AND is_candidate=1",
                                      (v,)).fetchone()[0]
            rows.append({"source_version_id": v, "condition": cond, "vehicles": len(objs), "night_candidates": lights,
                         "marked": self.is_marked(v)})
        rows.sort(key=lambda r: (r["vehicles"] + r["night_candidates"], r["source_version_id"]))
        out, per = [], {}
        for r in rows:
            if per.get(r["condition"], 0) < limit:
                per[r["condition"]] = per.get(r["condition"], 0) + 1
                out.append(r)
        return out


__all__ = ["ALGO", "Change", "ChangeParams", "ChangeService", "build_reference", "find_changes", "measure_condition"]
