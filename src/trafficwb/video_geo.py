"""A fixed camera on the map (geo-registration of a camera, kept by folder like its lines and its measure).

Rules:
- The operator picks a map file on this computer: a GeoTIFF in latitude and longitude (EPSG:4326), or any picture
  with the latitude and longitude of its top-left and bottom-right corners. Nothing is downloaded.
- Then four or more points, each clicked on a frame of the camera and on the map. The points give the camera's
  ground homography (``georeg.fit_points``). Every point's error when it is left out is shown, and a point the
  others disagree with is named, so a wrong click shows before it is saved.
- It belongs to the camera: every video in the same folder uses it. The newest save wins; clearing it is a save too.
- Then each counted target's path (the bottom middle of its box, where it touches the road) is a line on the map.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .georeg import GeoMap, LocalFrame, apply_h, fit_points
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
                "author": r["author"], "created_at": r["created_at"], "folder": r["folder"]}

    @staticmethod
    def check(points: list) -> dict:
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
        return {"homography": f.G.tolist(), "origin": [f.frame.lat0, f.frame.lon0], "residual_m": f.residual_m,
                "leave_one_out_m": f.leave_one_out_m, "suspect": f.suspect,
                "rms_m": round(float(np.sqrt(np.mean(np.square(f.residual_m)))), 2)}

    def set(self, version_id: int, points: list, map_path: str | None, corners: list | None, author: str) -> dict:
        fit = self.check(points)
        if map_path:
            load_map(map_path, corners)                              # refuse a map that cannot be read
        self.con.execute("INSERT INTO camera_geo(folder,points,map_path,map_corners,fit,cleared,author,created_at) "
                         "VALUES (?,?,?,?,?,0,?,?)",
                         (self.folder(version_id), json.dumps(points), map_path,
                          json.dumps(corners) if corners else None, json.dumps(fit), author, utcnow()))
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

