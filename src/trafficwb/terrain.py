"""Ground heights (a terrain model, DTM): read from a file, sampled at a point, and drawn as shaded relief.

Rules:
- A terrain file is a GeoTIFF of heights in metres in latitude and longitude (EPSG:4326), one band, or an SRTM
  ``.hgt`` tile (named like ``N32E034.hgt``). A GeoTIFF in a projected grid is refused with what to do instead.
- Heights the file marks as missing (its no-data value, or anything below -1000 m) are not a number, never zero.
- A height at a point is read between the four samples around it (bilinear); outside the file it is not known.
- Shaded relief is the ground lit from the north-west, for drawing over the map, so hills and cuts show.
"""
from __future__ import annotations

import math
import re
import threading
from pathlib import Path

import cv2
import numpy as np

TERRAIN_SUFFIXES = {".tif", ".tiff", ".hgt"}
M_PER_DEG = math.radians(1) * 6378137.0


class TerrainError(ValueError):
    pass


class Terrain:
    """Heights (float32, NaN where unknown) on a latitude and longitude grid: ``top``/``left`` is the outer corner of
    the first sample, ``dlat``/``dlon`` the sample size in degrees."""

    def __init__(self, data: np.ndarray, top: float, left: float, dlat: float, dlon: float, name: str = ""):
        self.data = np.asarray(data, np.float32)
        self.top, self.left, self.dlat, self.dlon = float(top), float(left), float(dlat), float(dlon)
        self.name = name

    @property
    def bottom(self) -> float:
        return self.top - self.data.shape[0] * self.dlat

    @property
    def right(self) -> float:
        return self.left + self.data.shape[1] * self.dlon

    def _rc(self, lat, lon):
        return ((self.top - np.asarray(lat, np.float64)) / self.dlat - 0.5,
                (np.asarray(lon, np.float64) - self.left) / self.dlon - 0.5)

    def height_at(self, lat, lon):
        """Height in metres at each point (NaN outside the file or where it has no value)."""
        r, c = self._rc(lat, lon)
        scalar = np.ndim(r) == 0
        r, c = np.atleast_1d(r), np.atleast_1d(c)
        h, w = self.data.shape
        out = np.full(r.shape, np.nan, np.float64)
        e = 1e-6
        ok = (r >= -0.5 - e) & (r <= h - 0.5 + e) & (c >= -0.5 - e) & (c <= w - 0.5 + e)
        rr, cc = np.clip(r[ok], 0, h - 1), np.clip(c[ok], 0, w - 1)
        r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
        r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
        fr, fc = rr - r0, cc - c0
        d = self.data
        out[ok] = (d[r0, c0] * (1 - fr) * (1 - fc) + d[r0, c1] * (1 - fr) * fc
                   + d[r1, c0] * fr * (1 - fc) + d[r1, c1] * fr * fc)
        return float(out[0]) if scalar else out

    def grid(self, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
        """Heights over ``bounds`` (top, left, bottom, right) at ``size`` (width, height) samples, NaN where unknown."""
        top, left, bottom, right = bounds
        W, H = size
        lat = top - (np.arange(H) + 0.5) * (top - bottom) / H
        lon = left + (np.arange(W) + 0.5) * (right - left) / W
        r, c = self._rc(lat, lon)
        mx, my = np.meshgrid(c.astype(np.float32), r.astype(np.float32))
        filled = np.where(np.isfinite(self.data), self.data, np.nanmean(self.data) if np.isfinite(self.data).any()
                          else 0).astype(np.float32)
        g = cv2.remap(filled, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        bad = cv2.remap(np.isnan(self.data).astype(np.float32), mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE)
        h, w = self.data.shape
        outside = (my < -0.5) | (my > h - 0.5) | (mx < -0.5) | (mx > w - 0.5)
        g[(bad > 0.5) | outside] = np.nan
        return g

    def info(self) -> dict:
        v = self.data[np.isfinite(self.data)]
        mid = (self.top + self.bottom) / 2
        return {"name": self.name, "width": int(self.data.shape[1]), "height": int(self.data.shape[0]),
                "top": self.top, "left": self.left, "bottom": self.bottom, "right": self.right,
                "m_per_sample": round(math.sqrt(self.dlat * M_PER_DEG * self.dlon * M_PER_DEG
                                                * math.cos(math.radians(mid))), 2),
                "min_m": round(float(v.min()), 1) if v.size else None,
                "max_m": round(float(v.max()), 1) if v.size else None}


def hillshade(heights: np.ndarray, m_per_px_x: float, m_per_px_y: float, azimuth: float = 315.0,
              altitude: float = 45.0) -> np.ndarray:
    """Shaded relief as BGRA: dark slopes away from the light, bright toward it, transparent where unknown and on
    flat ground (so the map under it shows), stronger the steeper the ground."""
    z = np.where(np.isfinite(heights), heights, np.nanmean(heights) if np.isfinite(heights).any() else 0)
    gy, gx = np.gradient(z.astype(np.float64), m_per_px_y, m_per_px_x)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)                    # rows run south: north is up
    az, alt = math.radians(azimuth), math.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    flat = math.sin(alt)
    light = np.clip((shade - flat) * 3.0, -1, 1)    # 0 on flat ground
    out = np.zeros(z.shape + (4,), np.uint8)
    out[..., :3] = np.where(light[..., None] > 0, 255, 0).astype(np.uint8)
    out[..., 3] = (np.clip(np.abs(light) * 1.4, 0, 1) * 210).astype(np.uint8)
    out[~np.isfinite(heights), 3] = 0
    return out


def shade_over(t: Terrain, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """Shaded relief of ``t`` over the map area ``bounds`` at ``size`` (BGRA)."""
    top, left, bottom, right = bounds
    W, H = size
    g = t.grid(bounds, size)
    mid = math.radians((top + bottom) / 2)
    mx, my = (right - left) / W * M_PER_DEG * math.cos(mid), (top - bottom) / H * M_PER_DEG
    # smooth over the file's own sample size (at least 10 m): no steps where coarse heights were upsampled
    sigma = max(10.0, t.info()["m_per_sample"]) / max(min(mx, my), 1e-6) / 2
    if sigma > 0.6 and np.isfinite(g).any():
        nan = ~np.isfinite(g)
        g = cv2.GaussianBlur(np.where(nan, np.nanmean(g), g).astype(np.float32), (0, 0), min(sigma, 60))
        g[nan] = np.nan
    return hillshade(g, mx, my)


# ------------------------------------------------------------------ files

_cache: dict = {}
_lock = threading.Lock()


def _from_hgt(p: Path) -> Terrain:
    m = re.match(r"([NS])(\d{2})([EW])(\d{3})", p.stem.upper())
    if not m:
        raise TerrainError("An SRTM .hgt file is named by its corner, like N32E034.hgt.")
    lat = int(m[2]) * (1 if m[1] == "N" else -1)
    lon = int(m[4]) * (1 if m[3] == "E" else -1)
    raw = np.fromfile(str(p), ">i2")
    n = int(round(math.sqrt(raw.size)))
    if n * n != raw.size or n < 2:
        raise TerrainError("This .hgt file is not a square SRTM tile.")
    d = 1.0 / (n - 1)
    data = raw.reshape(n, n).astype(np.float32)
    data[data < -1000] = np.nan
    return Terrain(data, lat + 1 + d / 2, lon - d / 2, d, d, p.name)


def _from_geotiff(p: Path) -> Terrain:
    import tifffile
    with tifffile.TiffFile(str(p)) as tf:
        page = tf.pages[0]
        scale, tie, keys = page.tags.get(33550), page.tags.get(33922), page.tags.get(34735)
        nodata = page.tags.get(42113)
        if scale is None or tie is None:
            raise TerrainError("This TIFF has no map coordinates: it is not a terrain (DTM) GeoTIFF.")
        if keys is not None and 3072 in tuple(keys.value)[4::4]:
            raise TerrainError("This terrain GeoTIFF is in a projected grid (metres). Save it in latitude and "
                               "longitude (EPSG:4326), for example with QGIS: Raster, Projections, Warp.")
        kv = tuple(keys.value) if keys is not None else ()
        point = any(kv[k] == 1025 and kv[k + 3] == 2 for k in range(4, len(kv) - 3, 4))
        arr = page.asarray()
        sx, sy = scale.value[:2]
        i, j, _, lon, lat, _ = tie.value[:6]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4) and arr.dtype == np.uint8:
            raise TerrainError("This GeoTIFF is a picture, not heights. A terrain file has one band of heights.")
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
    data = arr.astype(np.float32)
    if nodata is not None:
        try:
            nd = float(str(nodata.value).strip("\x00 "))
            data[data == nd] = np.nan
        except ValueError:
            pass
    data[data < -1000] = np.nan
    left, top = lon - i * sx, lat + j * sy
    if point:                                       # PixelIsPoint: the tie point is a sample centre
        left, top = left - sx / 2, top + sy / 2
    if not (-180 <= left <= 180 and -90 <= top <= 90 and sx < 1 and sy < 1):
        raise TerrainError("This GeoTIFF does not look like latitude and longitude.")
    return Terrain(data, top, left, sy, sx, p.name)


def load_terrain(path: str | Path) -> Terrain:
    """The terrain file at ``path`` (cached while unchanged)."""
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in TERRAIN_SUFFIXES:
        raise TerrainError("Pick a terrain file: a GeoTIFF of heights (DTM) or an SRTM .hgt tile.")
    key = (str(p), p.stat().st_mtime)
    with _lock:
        if key in _cache:
            return _cache[key]
    t = _from_hgt(p) if p.suffix.lower() == ".hgt" else _from_geotiff(p)
    if not np.isfinite(t.data).any():
        raise TerrainError("This terrain file has no heights in it.")
    with _lock:
        if len(_cache) > 4:
            _cache.clear()
        _cache[key] = t
    return t


def terrain_files(folder: str, limit: int = 300) -> list[dict]:
    p = Path(folder)
    if not p.is_dir():
        raise TerrainError("That folder does not exist on this computer.")
    out = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in TERRAIN_SUFFIXES:
            out.append({"name": f.name, "path": str(f), "size": f.stat().st_size})
            if len(out) >= limit:
                break
    return out


def _geokeys():
    # GeoKeyDirectory: version 1.1.0, 3 keys: model type geographic, raster type pixel-is-area, WGS 84
    return (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)


def write_geotiff(path: Path | str, data: np.ndarray, bounds: tuple[float, float, float, float],
                  description: str = "", nodata: float | None = None) -> Path:
    """Write a GeoTIFF in latitude and longitude: a BGR picture (saved as RGB) or one band of heights."""
    import tifffile
    top, left, bottom, right = bounds
    h, w = data.shape[:2]
    sx, sy = (right - left) / w, (top - bottom) / h
    tags = [(33550, "d", 3, (sx, sy, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, left, top, 0.0)),
            (34735, "H", 16, _geokeys())]
    if nodata is not None:
        tags.append((42113, "s", 0, str(nodata), False))
    arr = np.ascontiguousarray(data[:, :, ::-1]) if data.ndim == 3 else data
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tifffile.imwrite(str(tmp), arr, photometric="rgb" if data.ndim == 3 else "minisblack", compression="zlib",
                     description=description or None, extratags=tags)
    tmp.replace(p)
    return p
