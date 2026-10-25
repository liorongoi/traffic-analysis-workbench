"""Night analysis for still images: conditions, lights, persistent-light background, grouping, candidates.

Evidence levels (NGT-01), kept separate all the way to export:
  light component  - one bright region not explained by the background. Never counted.
  light group      - components that may belong together: single, pair, cluster. Never counted.
  vehicle candidate - an unexplained group on the road with no detector box. Counted only as
                     "unresolved", as a range (count_min..count_max), never as vehicles.
  detected vehicle - detector box (Phase 3). Lights inside it support it; they are not counted again.
  confirmed vehicle - an operator confirmed a candidate or drew a vehicle.

Everything works on original pixels. Nothing is enhanced, so no enhanced copy can pose as evidence.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

COND_ALGO = "cond-v1"

# Proposed thresholds, chosen from the Fintraffic sample; operator-adjustable later.
MONO_SATURATION = 5.0      # mean HSV saturation below this: infrared / monochrome
NIGHT_LUMA = 60.0          # colour image this dark: night
NIGHT_DARK_PCT = 40.0
DUSK_LUMA = 85.0
DUSK_DARK_PCT = 15.0
GLARE_CLIPPED_PCT = 2.0
LOW_DETAIL = 2.0           # median |Laplacian|; rain or fog on the lens measured 2.0 on the sample


@dataclass(frozen=True)
class Condition:
    label: str                 # day | dusk | night
    monochrome: bool
    mean_luma: float
    saturation: float
    clipped_pct: float
    dark_pct: float
    detail: float
    flags: tuple[str, ...]


def measure_condition(img_bgr: np.ndarray, ignore_top: int = 0) -> Condition:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[ignore_top:]
    sat = float(hsv[..., 1].mean())
    luma = float(g.mean())
    clipped = float((g >= 250).mean() * 100)
    dark = float((g < 40).mean() * 100)
    gh, gw = g.shape
    core = g[gh // 10: gh - gh // 10, gw // 10: gw - gw // 10]   # burned-in captions sit at the edges
    detail = float(np.median(np.abs(cv2.Laplacian(core, cv2.CV_32F))))
    mono = sat < MONO_SATURATION
    if mono or luma < NIGHT_LUMA or dark > NIGHT_DARK_PCT:
        label = "night"
    elif luma < DUSK_LUMA and dark > DUSK_DARK_PCT:
        label = "dusk"
    else:
        label = "day"
    flags = []
    if clipped > GLARE_CLIPPED_PCT:
        flags.append("glare")
    if detail <= LOW_DETAIL:
        flags.append("low_detail")
    return Condition(label, mono, round(luma, 2), round(sat, 2), round(clipped, 3), round(dark, 2), round(detail, 2),
                     tuple(flags))


# ---------- light components ----------

@dataclass
class LightParams:
    min_brightness: int = 235      # 0-255 grey level
    min_tophat: int = 40           # brighter than the surroundings by at least this
    tophat_kernels: tuple[int, ...] = (25, 61)
    close_px: int = 5              # join fragments of one light
    min_area: int = 6
    glare_level: int = 250         # a large clipped region is a light too (headlight glare of a close vehicle),
    glare_min_area: int = 400      # which the top-hat misses because it is wider than its kernels

    def record(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Light:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 (exclusive)
    cx: float
    cy: float
    area: int
    peak: float
    mean: float
    elongation: float                # height / width of the bounding box
    red_share: float | None = None
    id: int | None = None


def light_mask(img_bgr: np.ndarray, p: LightParams | None = None, ignore: np.ndarray | None = None) -> np.ndarray:
    p = p or LightParams()
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    th = np.zeros_like(g)
    for k in p.tophat_kernels:
        th = np.maximum(th, cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
    m = ((g >= p.min_brightness) & (th >= p.min_tophat)).astype(np.uint8)
    clipped = (g >= p.glare_level).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(clipped)
    big = [i for i in range(1, n) if st[i][4] >= p.glare_min_area]
    if big:
        m[np.isin(lab, big)] = 1
    if p.close_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((p.close_px, p.close_px), np.uint8))
    if ignore is not None:
        m[ignore > 0] = 0
    return m


def light_components(img_bgr: np.ndarray, p: LightParams | None = None, ignore: np.ndarray | None = None) -> list[Light]:
    p = p or LightParams()
    m = light_mask(img_bgr, p, ignore)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    colour = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()) >= MONO_SATURATION
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < p.min_area:
            continue
        sel = lab[y:y + h, x:x + w] == i
        vals = g[y:y + h, x:x + w][sel]
        red = None
        if colour:
            px = img_bgr[y:y + h, x:x + w][sel].astype(int)
            red = round(float(((px[:, 2] > px[:, 1] + 40) & (px[:, 2] > px[:, 0] + 40)).mean()), 3)
        out.append(Light((x, y, x + w, y + h), round(float(cen[i][0]), 1), round(float(cen[i][1]), 1), a,
                         float(vals.max()), round(float(vals.mean()), 1), round(h / max(1, w), 2), red))
    return out


# ---------- persistent-light background ----------

@dataclass
class BackgroundParams:
    persistent_share: float = 0.4   # lit in at least this share of images: persistent (a lamp measured 0.45 on the sample)
    min_images: int = 8             # fewer images: no model, nothing is explained away
    recent_images: int = 12         # window for "new persistent light?"
    dilate_px: int = 5              # tolerance for small jitter between captures
    explained_share: float = 0.5    # component explained when this share of it lies in the mask
    attach_px: float = 3.0          # a light this close to a persistent light is part of it (its glow varies)
    near_px: int = 4                # neighbourhood in which a light's history at this place is read
    recurring_share: float = 0.2    # lit here in at least this share of images: a background light not always lit
    occasional_share: float = 0.1   # lit here this often: weaker sign of background
    recurring_factor: float = 0.4   # candidate score multipliers
    occasional_factor: float = 0.7

    def record(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Background:
    persistent: np.ndarray           # uint8 0/1
    frequency: np.ndarray            # float 0..1
    image_count: int
    new_lights: list[tuple[int, int, int, int]] = field(default_factory=list)

    def explains(self, light: Light, share: float = 0.5) -> bool:
        x1, y1, x2, y2 = (int(round(v)) for v in light.box)
        win = self.persistent[y1:y2, x1:x2]
        return win.size > 0 and float(win.mean()) >= share

    def _distance(self) -> np.ndarray:
        if getattr(self, "_dist", None) is None:
            free = (self.persistent == 0).astype(np.uint8)
            self._dist = cv2.distanceTransform(free, cv2.DIST_L2, 3) if self.persistent.any() else \
                np.full(self.persistent.shape, 1e6, np.float32)
        return self._dist

    def recurrence(self, light: Light, near: int = 4) -> float:
        """Highest share of images in which the light's neighbourhood was lit."""
        x1, y1, x2, y2 = (int(round(v)) for v in light.box)
        h, w = self.frequency.shape
        win = self.frequency[max(0, y1 - near):min(h, y2 + near), max(0, x1 - near):min(w, x2 + near)]
        return float(win.max()) if win.size else 0.0

    def reading(self, light: Light, p: "BackgroundParams") -> tuple[str, float]:
        """How the background explains a light: persistent, attached (to a persistent light), recurring,
        occasional or rare. Returns the state and the light's recurrence at its place."""
        if self.explains(light, p.explained_share):
            return "persistent", 1.0
        x1, y1, x2, y2 = (int(round(v)) for v in light.box)
        h, w = self.persistent.shape
        d = self._distance()[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)]
        # Glare (a large clipped blob) next to a lamp is usually a vehicle's headlights near the lamp, not the
        # lamp's own glow, which the persistent mask already covers; so glare is never explained by touching.
        if d.size and float(d.min()) <= p.attach_px and not is_glare(light):
            return "attached", 1.0
        r = round(self.recurrence(light, p.near_px), 3)
        if r >= p.recurring_share:
            return "recurring", r
        if r >= p.occasional_share:
            return "occasional", r
        return "rare", r


def build_background(masks_in_time_order: list[np.ndarray], vehicle_boxes: list[list[tuple]] | None = None,
                     labels: list[tuple[str, list]] | None = None, p: BackgroundParams | None = None) -> Background | None:
    """Persistent-light model from many images of one view and condition (stills).

    Pixels inside vehicle boxes (detected or confirmed) are left out of that image's
    contribution, so a stopped vehicle never becomes background (NGT-03). Lights that
    are frequent only in the most recent images are reported as possible new
    persistent lights for the operator, not absorbed. Operator labels win.
    """
    p = p or BackgroundParams()
    n = len(masks_in_time_order)
    if n < p.min_images:
        return None
    h, w = masks_in_time_order[0].shape
    acc = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)     # images where the pixel was observable (not under a vehicle)
    recent_acc = np.zeros((h, w), np.float32)
    recent_seen = np.zeros((h, w), np.float32)
    k = np.ones((p.dilate_px, p.dilate_px), np.uint8)
    for idx, m in enumerate(masks_in_time_order):
        lit = cv2.dilate(m, k).astype(np.float32)
        obs = np.ones((h, w), np.float32)
        for (x1, y1, x2, y2) in (vehicle_boxes[idx] if vehicle_boxes else []):
            obs[int(max(0, y1)):int(min(h, y2)), int(max(0, x1)):int(min(w, x2))] = 0
        acc += lit * obs
        seen += obs
        if idx >= n - p.recent_images:
            recent_acc += lit * obs
            recent_seen += obs
    freq = np.divide(acc, seen, out=np.zeros_like(acc), where=seen > 0)
    persistent = ((freq >= p.persistent_share) & (seen >= p.min_images)).astype(np.uint8)
    recent = np.divide(recent_acc, recent_seen, out=np.zeros_like(acc), where=recent_seen > 0)
    new = ((recent >= p.persistent_share) & (persistent == 0) & (recent_seen >= min(p.recent_images, n) / 2)).astype(np.uint8)
    for kind, poly in labels or []:
        pm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(pm, [np.array(poly, np.int32)], 1)
        if kind == "persistent":
            persistent[pm > 0] = 1
            freq[pm > 0] = 1.0
        else:                       # "not persistent": the history at this place no longer counts against a light
            persistent[pm > 0] = 0
            freq[pm > 0] = 0.0
        new[pm > 0] = 0
    nn, _, st, _ = cv2.connectedComponentsWithStats(new)
    new_boxes = [(int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3]))
                 for i in range(1, nn) if st[i][4] >= 6]
    return Background(persistent, freq, n, new_boxes)


# ---------- perspective: expected vehicle width by image row ----------

@dataclass(frozen=True)
class WidthMap:
    a: float
    b: float
    samples: int
    source: str   # daytime_detections | operator

    def width(self, y: float) -> float:
        return max(4.0, self.a * y + self.b)

    @classmethod
    def fit(cls, boxes: list[tuple[float, float, float, float]], min_samples: int = 20) -> "WidthMap | None":
        """Vehicle width as a linear function of the box's bottom row, from daytime car detections in this view."""
        if len(boxes) < min_samples:
            return None
        arr = np.array(boxes, dtype=float)
        wv, yv = arr[:, 2] - arr[:, 0], arr[:, 3]
        if np.ptp(yv) < 20:
            return None
        a, b = np.polyfit(yv, wv, 1)
        if a <= 0:
            return None
        return cls(round(float(a), 5), round(float(b), 3), len(boxes), "daytime_detections")


# ---------- distance bands ----------

@dataclass
class BandParams:
    """What a vehicle looks like at night depends on its distance, read from the expected vehicle width at its row.

    far:    its two headlights merge into one light with a glow
    middle: one light or two lights close together
    close:  its shape is visible, so the detector should find it; lights alone there are suspect
    """
    far_below: float = 24.0          # expected vehicle width in px under this: far
    close_from: float = 60.0         # from this width: close

    def record(self) -> dict:
        return dict(self.__dict__)


def band_at(wm: "WidthMap | None", row: float, bp: BandParams) -> str | None:
    if wm is None:
        return None
    w = wm.width(row)
    return "far" if w < bp.far_below else "close" if w >= bp.close_from else "middle"


def band_rows(wm: "WidthMap | None", bp: BandParams) -> dict | None:
    """Image rows where the bands change (for drawing them)."""
    if wm is None:
        return None
    return {"far_until_row": round((bp.far_below - wm.b) / wm.a, 1), "close_from_row": round((bp.close_from - wm.b) / wm.a, 1)}


# ---------- grouping ----------

@dataclass
class GroupingParams:
    pair_min: float = 0.35           # light spacing / expected vehicle width
    pair_max: float = 1.15
    max_dy: float = 0.25             # vertical offset / expected width
    max_area_ratio: float = 3.0
    cluster_gap: float = 0.20        # lights closer than this share of a vehicle width form one cluster
    reflection_dx: float = 0.35
    reflection_dy: float = 2.0

    def record(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Group:
    kind: str                        # single | pair | cluster
    members: list[int]               # indices into the light list
    box: tuple[float, float, float, float]
    score: float
    count_min: int
    count_max: int
    alternatives: list[str]
    flags: list[str]
    reflection: float = 0.0
    band: str | None = None


def _box_of(lights: list[Light], idx: list[int]) -> tuple[float, float, float, float]:
    return (float(min(lights[i].box[0] for i in idx)), float(min(lights[i].box[1] for i in idx)),
            float(max(lights[i].box[2] for i in idx)), float(max(lights[i].box[3] for i in idx)))


def reflection_score(i: int, lights: list[Light], wm: WidthMap | None, p: GroupingParams) -> float:
    """A vertically elongated light below a brighter one, as on a wet road."""
    li = lights[i]
    w = wm.width(li.cy) if wm else max(8.0, 3 * (li.box[2] - li.box[0]))
    for j, lj in enumerate(lights):
        if j == i or lj.cy >= li.cy or lj.peak < li.peak:
            continue
        if abs(lj.cx - li.cx) <= p.reflection_dx * w and 0 < li.cy - lj.cy <= p.reflection_dy * w:
            return 0.7 if li.elongation >= 1.5 else 0.3
    return 0.0


def is_glare(li: Light, lp: LightParams | None = None) -> bool:
    lp = lp or LightParams()
    return li.area >= lp.glare_min_area and li.peak >= lp.glare_level


def group_lights(lights: list[Light], wm: WidthMap | None, p: GroupingParams | None = None,
                 bands: BandParams | None = None) -> list[Group]:
    p = p or GroupingParams()
    bands = bands or BandParams()
    n = len(lights)
    if n == 0:
        return []
    # 1. clusters: lights almost touching (multi-lamp fronts, fragments)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            w = wm.width((lights[i].cy + lights[j].cy) / 2) if wm else 10.0
            gx = max(0, max(lights[i].box[0], lights[j].box[0]) - min(lights[i].box[2], lights[j].box[2]))
            gy = max(0, max(lights[i].box[1], lights[j].box[1]) - min(lights[i].box[3], lights[j].box[3]))
            if math.hypot(gx, gy) <= p.cluster_gap * w:
                parent[find(i)] = find(j)
    units: dict[int, list[int]] = {}
    for i in range(n):
        units.setdefault(find(i), []).append(i)
    unit_list = list(units.values())

    def centre(u):
        return (sum(lights[i].cx * lights[i].area for i in u) / sum(lights[i].area for i in u),
                sum(lights[i].cy * lights[i].area for i in u) / sum(lights[i].area for i in u))

    def area(u):
        return sum(lights[i].area for i in u)

    # 2. pair hypotheses between units, scored by how well they match a vehicle at that row
    cands = []
    if wm is not None:
        for a in range(len(unit_list)):
            for b in range(a + 1, len(unit_list)):
                (xa, ya), (xb, yb) = centre(unit_list[a]), centre(unit_list[b])
                w = wm.width((ya + yb) / 2)
                spacing = abs(xa - xb) / w
                dy = abs(ya - yb) / w
                ar = max(area(unit_list[a]), area(unit_list[b])) / max(1, min(area(unit_list[a]), area(unit_list[b])))
                if p.pair_min <= spacing <= p.pair_max and dy <= p.max_dy and ar <= p.max_area_ratio:
                    s = 1.0 - abs(spacing - 0.7) / 0.7 * 0.5 - dy - (ar - 1) / p.max_area_ratio * 0.3
                    cands.append((round(s, 3), a, b))
    cands.sort(reverse=True)
    partners: dict[int, int] = {}
    options: dict[int, int] = {}
    for _, a, b in cands:
        options[a] = options.get(a, 0) + 1
        options[b] = options.get(b, 0) + 1
    groups: list[Group] = []
    used = set()
    for s, a, b in cands:
        if a in used or b in used:
            continue
        used |= {a, b}
        partners[a], partners[b] = b, a
        members = unit_list[a] + unit_list[b]
        alts = ["two vehicles with one light each"]
        flags = []
        if options[a] > 1 or options[b] > 1:
            alts.append("lights may pair differently with a neighbour")
            flags.append("ambiguous_pairing")
        groups.append(Group("pair", sorted(members), _box_of(lights, members), round(min(1.0, 0.5 + 0.4 * s), 3),
                            1, 2, alts, flags))
    for u_idx, u in enumerate(unit_list):
        if u_idx in used:
            continue
        if len(u) > 1:
            groups.append(Group("cluster", sorted(u), _box_of(lights, u), 0.5, 1, min(len(u), 3),
                                ["several vehicles close together"], []))
        elif wm is None:
            groups.append(Group("single", u, _box_of(lights, u), 0.3, 0, 1,
                                ["one light of a vehicle whose other light belongs to another single"], ["no_width_map"]))
        else:
            groups.append(Group("single", u, _box_of(lights, u), 0.3, 1, 1,
                                ["motorcycle or a vehicle with one visible light"], []))
    for g in groups:
        if wm is not None and (g.box[2] - g.box[0]) > 1.5 * wm.width(g.box[3]):
            # wider than a vehicle at that row: lit road surface, a light trail or glare; may still hide a vehicle
            g.flags.append("larger_than_vehicle")
            g.count_min = 0
        _apply_band(g, lights, wm, bands)
        g.reflection = max(reflection_score(i, lights, wm, p) for i in g.members)
        if g.reflection:
            g.score = round(g.score * (1 - 0.5 * g.reflection), 3)
            g.flags.append("possible_reflection")
    return groups


def _apply_band(g: Group, lights: list[Light], wm: WidthMap | None, bp: BandParams) -> None:
    """Judge the group by what a vehicle looks like at its distance. Groups inside a detector box are not
    candidates anyway, so 'close' here means close lights without a detected vehicle shape."""
    g.band = band_at(wm, g.box[3], bp)
    lined = [i for i in g.members if not (0.25 < lights[i].elongation < 4.0)]
    if lined:
        # a headlight is roughly round; a long thin light is a lit railing, pole, lane marking or a light trail
        g.flags.append("line_shaped")
        g.score = round(g.score * 0.5, 3)
        g.alternatives.append("a lit railing, pole or road marking")
        return
    glare = any(is_glare(lights[i]) for i in g.members)
    if glare:
        g.flags.append("glare")
        g.alternatives.insert(0, "glare from the headlights of a vehicle facing the camera")
        g.score = round(max(g.score, 0.5), 3)
        return
    if "larger_than_vehicle" in g.flags:
        return
    if g.band == "far":
        if g.kind == "single":
            g.count_min, g.count_max = 1, 1
            g.score = round(max(g.score, 0.4), 3)
            g.alternatives = ["one vehicle whose two lights merge into one glow at this distance",
                              "a light that is not a vehicle"]
    elif g.band == "middle":
        if g.kind == "single":
            g.count_min, g.count_max = 1, 1
            g.score = round(max(g.score, 0.35), 3)
            g.alternatives = ["one vehicle with its two lights merged, or with one light visible",
                              "a light that is not a vehicle"]
        elif g.kind == "pair" and "ambiguous_pairing" not in g.flags:
            g.count_min, g.count_max = 1, 1          # two lights close together: one vehicle at this distance
            g.alternatives = ["two vehicles with one light each (unlikely at this distance)"]
    elif g.band == "close":
        g.flags.append("close_without_shape")
        g.count_min = 0
        g.score = round(g.score * 0.5, 3)
        g.alternatives.insert(0, "a close vehicle should show its shape; lights alone here are often reflections "
                                 "or lit road furniture")


def covered_by(box, det_boxes: list[tuple[int, tuple]], expand: float = 0.15, share: float = 0.5) -> int | None:
    """Detection id whose (slightly expanded) box contains most of this group."""
    x1, y1, x2, y2 = box
    ga = max(1.0, (x2 - x1) * (y2 - y1))
    for did, (a1, b1, a2, b2) in det_boxes:
        ex, ey = (a2 - a1) * expand, (b2 - b1) * expand
        ix = max(0.0, min(x2, a2 + ex) - max(x1, a1 - ex))
        iy = max(0.0, min(y2, b2 + ey) - max(y1, b1 - ey))
        if ix * iy / ga >= share:
            return did
    return None


def png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def unpng(blob: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)
