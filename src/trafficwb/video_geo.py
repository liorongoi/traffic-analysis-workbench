"""A fixed camera on the map (geo-registration of a camera, kept by folder like its lines and its measure).

Rules:
- The operator picks a map file on this computer: a GeoTIFF in latitude and longitude (EPSG:4326), or any picture
  with the latitude and longitude of its top-left and bottom-right corners. Nothing is downloaded.
- Then four or more points, each clicked on a frame of the camera and on the map. The points give the camera's
  ground homography (``georeg.fit_points``). Every point's error when it is left out is shown, and a point the
  others disagree with is named, so a wrong click shows before it is saved.
- It belongs to the camera: every video in the same folder uses it. The newest save wins; clearing it is a save too.
- Then each counted target's path (the bottom middle of its box, where it touches the road) is a line on the map.
- Ground heights are optional (a terrain file, see ``terrain`` and ``map_sources``). With them, each point shows the
  ground's height and points far apart in height are named: the fit takes the ground as flat.
- The frame is laid on the map with its footprint: the outline green when the camera is placed by points on the
  map, or when its flight data was lined up with the map by matching; red when it is placed by flight data only
  (no map, or the match failed). The upper part of the image and its top edge are yellow.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .georeg import GeoMap, LocalFrame, apply_h, fit_points, frame_on_ground, ground_parts, pose_from_telemetry
from .store import utcnow

MAP_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
PREVIEW_SIDE = 4096
PATH_RATE = 2.0                     # map points a second along a target's path


class GeoError(ValueError):
    pass


# ------------------------------------------------------------------ map files

_cache: dict = {}
_lock = threading.Lock()


def _bounds_of_tiff(path: Path):
    import tifffile
    with tifffile.TiffFile(str(path)) as t:
        p = t.pages[0]
        scale, tie, keys = p.tags.get(33550), p.tags.get(33922), p.tags.get(34735)
        if scale is None or tie is None:
            return None
        if keys is not None and 3072 in tuple(keys.value)[4::4]:
            raise GeoError("This GeoTIFF is in a projected grid (metres). Save it in latitude and longitude "
                           "(EPSG:4326), or give the corners of the picture.")
        sx, sy = scale.value[:2]
        i, j, _, lon, lat, _ = tie.value[:6]
        h, w = p.shape[:2]
    lon0, lat0 = lon - i * sx, lat + j * sy
    if not (-180 <= lon0 <= 180 and -90 <= lat0 <= 90 and sx < 1 and sy < 1):
        raise GeoError("This GeoTIFF does not look like latitude and longitude.")
    return (lat0, lon0), (lat0 - h * sy, lon0 + w * sx)


def load_map(path: str, corners: list[float] | None = None) -> GeoMap:
    """The map at ``path`` (cached while the file is unchanged)."""
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in MAP_SUFFIXES:
        raise GeoError("Pick a map file: a GeoTIFF, or a JPEG or PNG picture with its corners.")
    key = (str(p), p.stat().st_mtime, tuple(corners or ()))
    with _lock:
        if key in _cache:
            return _cache[key]
    if corners:
        if len(corners) != 4:
            raise GeoError("Corners are four numbers: top latitude, left longitude, bottom latitude, right longitude.")
        a, b, c, d = (float(v) for v in corners)
        if not (-90 <= c < a <= 90 and -180 <= b < d <= 180):
            raise GeoError("The top must be north of the bottom and the left west of the right.")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise GeoError("That picture cannot be read.")
        gm = GeoMap(img, (a, b), (c, d))
    else:
        if p.suffix.lower() not in (".tif", ".tiff"):
            raise GeoError("A JPEG or PNG map needs the latitude and longitude of its corners.")
        b = _bounds_of_tiff(p)
        if b is None:
            raise GeoError("This TIFF has no map coordinates. Give the corners of the picture.")
        gm = GeoMap.from_geotiff(p)
    with _lock:
        _cache.clear()                                       # one map at a time in memory
        _cache[key] = gm
    return gm


def map_info(gm: GeoMap) -> dict:
    h, w = gm.image.shape[:2]
    (top, left), (bottom, right) = [gm.frame.to_latlon(*apply_h(gm.A, [xy])[0]) for xy in ((0, 0), (w, h))]
    return {"width": w, "height": h, "top": float(top), "left": float(left), "bottom": float(bottom),
            "right": float(right), "m_per_px": round(gm.m_per_px, 3)}


def preview(gm: GeoMap, side: int = PREVIEW_SIDE) -> bytes:
    img = gm.image
    s = min(1.0, side / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def map_files(folder: str, limit: int = 200) -> list[dict]:
    p = Path(folder)
    if not p.is_dir():
        raise GeoError("That folder does not exist on this computer.")
    out = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in MAP_SUFFIXES:
            out.append({"name": f.name, "path": str(f), "size": f.stat().st_size,
                        "geotiff": f.suffix.lower() in (".tif", ".tiff")})
            if len(out) >= limit:
                break
    return out


# ------------------------------------------------------------------ a camera on the map

class CameraGeo:
    def __init__(self, mission):
        self.m = mission
        self.con = mission.con

    def folder(self, version_id: int) -> str:
        r = self.con.execute("SELECT f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                             "WHERE v.id=?", (version_id,)).fetchone()
        if r is None:
            raise GeoError("That video is not in this mission.")
        return str(PurePosixPath(r[0]).parent)

    def get(self, version_id: int) -> dict | None:
        r = self.con.execute("SELECT * FROM camera_geo WHERE folder=? ORDER BY id DESC LIMIT 1",
                             (self.folder(version_id),)).fetchone()
        if r is None or r["cleared"]:
            return None
        fit = json.loads(r["fit"])
        return {"points": json.loads(r["points"]), "map_path": r["map_path"],
                "corners": json.loads(r["map_corners"]) if r["map_corners"] else None, "fit": fit,
                "dtm_path": r["dtm_path"] if "dtm_path" in r.keys() else None,
                "author": r["author"], "created_at": r["created_at"], "folder": r["folder"]}

    @staticmethod
    def check(points: list, dtm_path: str | None = None) -> dict:
        """The fit of these points, without saving: errors, the left-out errors and the suspect."""
        pts = [[float(v) for v in p] for p in points]
        if len(pts) < 4:
            raise GeoError("Pick at least four points, each on the frame and on the map.")
        if any(len(p) != 4 or not (-90 <= p[2] <= 90 and -180 <= p[3] <= 180) for p in pts):
            raise GeoError("Each point is x, y on the frame and latitude, longitude on the map.")
        try:
            f = fit_points([p[:2] for p in pts], [p[2:] for p in pts])
        except ValueError as exc:
            raise GeoError(str(exc)) from None
        except cv2.error:
            raise GeoError("The points do not fit a flat ground (three of them in a line?).") from None
        out = {"homography": f.G.tolist(), "origin": [f.frame.lat0, f.frame.lon0], "residual_m": f.residual_m,
               "leave_one_out_m": f.leave_one_out_m, "suspect": f.suspect,
               "rms_m": round(float(np.sqrt(np.mean(np.square(f.residual_m)))), 2)}
        if dtm_path:
            out |= heights_of(dtm_path, [p[2:] for p in pts])
        return out

    def set(self, version_id: int, points: list, map_path: str | None, corners: list | None, author: str,
            dtm_path: str | None = None) -> dict:
        fit = self.check(points, dtm_path)
        if map_path:
            load_map(map_path, corners)                              # refuse a map that cannot be read
        self.con.execute("INSERT INTO camera_geo(folder,points,map_path,map_corners,fit,cleared,author,created_at,"
                         "dtm_path) VALUES (?,?,?,?,?,0,?,?,?)",
                         (self.folder(version_id), json.dumps(points), map_path,
                          json.dumps(corners) if corners else None, json.dumps(fit), author, utcnow(),
                          dtm_path or None))
        return self.get(version_id)

    def clear(self, version_id: int, author: str) -> None:
        self.con.execute("INSERT INTO camera_geo(folder,points,map_path,map_corners,fit,cleared,author,created_at) "
                         "VALUES (?,'[]',NULL,NULL,'{}',1,?,?)", (self.folder(version_id), author, utcnow()))

    def to_latlon(self, version_id: int, px) -> np.ndarray | None:
        g = self.get(version_id)
        if g is None:
            return None
        G = np.array(g["fit"]["homography"])
        frame = LocalFrame(*g["fit"]["origin"])
        en = apply_h(G, px)
        lat, lon = frame.to_latlon(en[:, 0], en[:, 1])
        return np.c_[lat, lon]

    def paths(self, svc, version_id: int) -> list[dict]:
        """Each counted target of this video as a line on the map: [t, lat, lon] at most ``PATH_RATE`` a second."""
        g = self.get(version_id)
        if g is None:
            return []
        evs = [e for e in svc.events(version_id, None, None) if e.counted]
        boxes = svc.boxes_by_event(evs)
        G = np.array(g["fit"]["homography"])
        frame = LocalFrame(*g["fit"]["origin"])
        out = []
        for e in evs:
            pts, last = [], -1e9
            for t, b in boxes[e.id]:
                if t - last >= 1.0 / PATH_RATE - 1e-6:
                    pts.append((t, (b[0] + b[2]) / 2, b[3]))
                    last = t
            if not pts:
                continue
            en = apply_h(G, [[x, y] for _, x, y in pts])
            lat, lon = frame.to_latlon(en[:, 0], en[:, 1])
            line = [[round(t, 2), round(float(a), 7), round(float(o), 7)] for (t, _, _), a, o in zip(pts, lat, lon)]
            d = sum(frame.distance(p[1], p[2], q[1], q[2]) for p, q in zip(line, line[1:]))
            out.append({"event_id": e.id, "key": e.key, "cls": e.cls, "first_t": e.first_t, "last_t": e.last_t,
                        "path": line, "length_m": round(d, 1)})
        return out



# ------------------------------------------------------------------ ground heights

FLAT_SPAN_M = 3.0                   # points further apart in height than this: the flat-ground fit is warned about


def heights_of(dtm_path: str, latlon: list) -> dict:
    """The ground's height at each point from a terrain file, and a warning when they are far from one level."""
    from .terrain import TerrainError, load_terrain
    try:
        t = load_terrain(dtm_path)
    except TerrainError as exc:
        raise GeoError(str(exc)) from None
    ll = np.asarray(latlon, np.float64).reshape(-1, 2)
    hs = t.height_at(ll[:, 0], ll[:, 1])
    known = hs[np.isfinite(hs)]
    span = float(known.max() - known.min()) if known.size else None
    return {"heights_m": [None if not np.isfinite(v) else round(float(v), 1) for v in hs],
            "height_span_m": None if span is None else round(span, 1),
            "not_flat": bool(span is not None and span > FLAT_SPAN_M)}


# ------------------------------------------------------------------ the frame and its footprint on the map

GOOD_RMS_M = 3.0


_placed: dict = {}
_placed_lock = threading.Lock()


def placement(m, svc, version_id: int, t: float, dtm_path: str | None = None, map_path: str | None = None,
              corners: list | None = None):
    """Where the frame at ``t`` is on the ground: (G, LocalFrame, w, h, how, registered, note) or None.

    Points picked on the map come first (registered: green). Otherwise the recording's flight data, if it has any:
    with a map, the frame is then matched to the map, which lines it up (registered: green, ``how`` "map-match");
    when the match fails it stays where the flight data put it (not registered: red). A terrain file gives the
    ground's height under the camera for flight data that only knows the height above sea level."""
    key = (m.id, version_id, round(t, 1), dtm_path, map_path, tuple(corners or ()),
           m.con.execute("SELECT MAX(id) FROM camera_geo").fetchone()[0])
    with _placed_lock:
        if key in _placed:
            return _placed[key]
    out = _placement(m, svc, version_id, t, dtm_path, map_path, corners)
    with _placed_lock:
        if len(_placed) > 64:
            _placed.clear()
        _placed[key] = out
    return out


def _placement(m, svc, version_id, t, dtm_path, map_path, corners):
    r = m.con.execute("SELECT width, height FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone()
    if r is None or not r[0]:
        raise GeoError("This recording has not been read yet.")
    w, h = int(r[0]), int(r[1])
    g = CameraGeo(m).get(version_id)
    if g is not None:
        rms = g["fit"].get("rms_m")
        good = rms is not None and rms <= GOOD_RMS_M
        return (np.array(g["fit"]["homography"]), LocalFrame(*g["fit"]["origin"]), w, h,
                "points" if good else "points-poor", good, None)
    from .video_meta import stored
    try:
        path = svc.path(version_id)
    except Exception:  # noqa: BLE001 - the file has gone: what was read before is still used
        path = None
    tel = stored(m.con, version_id, path)
    s = tel.at(t) if tel is not None else None
    if not s or s.get("lat") is None:
        return None
    frame = LocalFrame(float(s["lat"]), float(s["lon"]))
    ground = None
    if dtm_path:
        from .terrain import TerrainError, load_terrain
        try:
            v = load_terrain(dtm_path).height_at(s["lat"], s["lon"])
            ground = float(v) if np.isfinite(v) else None
        except TerrainError:
            ground = None
    G = pose_from_telemetry(s, w, h, frame, None, ground)
    if G is None:
        return None
    if not map_path or path is None:
        return G, frame, w, h, "flight-data", False, None
    # line the frame up with the map: draw the map as the flight data says the camera sees it, and match
    from .georeg import snap_to_map
    from .video import frame_at
    try:
        gm = load_map(map_path, corners)
    except GeoError:
        return G, frame, w, h, "flight-data", False, None
    f = frame_at(path, t)
    if f is None:
        return G, frame, w, h, "flight-data", False, None
    img = f.image if f.image.shape[:2] == (h, w) else cv2.resize(f.image, (w, h), interpolation=cv2.INTER_AREA)
    T = np.eye(3)
    e, n = gm.frame.to_en(frame.lat0, frame.lon0)
    T[0, 2], T[1, 2] = float(e), float(n)
    snap = snap_to_map(img, T @ G, gm, _matcher())
    if snap.G is None:
        return G, frame, w, h, "flight-data", False, f"The map match did not work here ({snap.reason})."
    return snap.G, gm.frame, w, h, "map-match", True, f"Lined up with the map: moved {snap.shift_m:.1f} m."


_m = None


def _matcher():
    global _m
    if _m is None:
        from .georeg import Matcher
        _m = Matcher()
    return _m


def footprint_of(m, svc, version_id: int, t: float, dtm_path: str | None = None, map_path: str | None = None,
                 corners: list | None = None) -> dict | None:
    """The footprint at ``t`` in latitude and longitude: near and far parts, the box around them, and how it was
    placed (``registered`` true only for a good fit to points on the map)."""
    p = placement(m, svc, version_id, t, dtm_path, map_path, corners)
    if p is None:
        return None
    G, frame, w, h, how, ok, note = p
    parts = ground_parts(G, w, h)
    if parts is None:
        return {"how": how, "registered": ok, "note": note, "near": None, "far": None, "bounds": None}

    def ll(q):
        lat, lon = frame.to_latlon(q[:, 0], q[:, 1])
        return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]

    near, far = ll(parts["near"]), ll(parts["far"])
    a = np.array(near + far)
    return {"how": how, "registered": ok, "note": note, "near": near, "far": far, "top_edge": far[:2],
            "top_row": round(parts["top_row"], 1), "cut": parts["top_row"] > 0, "frame_w": w, "frame_h": h,
            "bounds": [float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 0].min()), float(a[:, 1].max())]}


def frame_overlay(m, svc, version_id: int, t: float, dtm_path: str | None = None, map_path: str | None = None,
                  corners: list | None = None, side: int = 1600) -> bytes:
    """The frame at ``t`` laid on the ground over its footprint's box, as a PNG with transparency (no black)."""
    from .video import frame_at
    p = placement(m, svc, version_id, t, dtm_path, map_path, corners)
    fp = footprint_of(m, svc, version_id, t, dtm_path, map_path, corners)
    if p is None or fp is None or fp["bounds"] is None:
        raise GeoError("This frame is not on the map.")
    G, frame, w, h = p[:4]
    f = frame_at(svc.path(version_id), t)
    if f is None:
        raise GeoError("No frame at that time.")
    img = f.image
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    top, left, bottom, right = fp["bounds"]
    wm = frame.distance(top, left, top, right)
    hm = frame.distance(top, left, bottom, left)
    k = max(wm, hm) / side
    size = (max(8, round(wm / k)), max(8, round(hm / k)))
    lay = frame_on_ground(img, G, frame, (top, left, bottom, right), size)
    ok, buf = cv2.imencode(".png", lay)
    return buf.tobytes()
