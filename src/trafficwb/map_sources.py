"""Where the map picture and the ground heights come from: an online service, files, or a server on the network.

Rules:
- Three ways in. Files on this computer (a GeoTIFF or a picture with its corners for the map; a terrain GeoTIFF or
  an SRTM .hgt tile for heights). An online service from a short built-in list (only when the computer is online).
  A server on the operator's own network: WMTS, WMS, a tile address with {z}/{x}/{y} (REST), an ArcGIS REST map or
  image service, or an Open-Elevation style height service, each with an optional sign-in (user name and password,
  or an API key sent in a header or in the address).
- What is fetched is saved as a GeoTIFF in latitude and longitude in ``<data>/maps`` and used from there, exactly
  like a file the operator picked. So an area fetched once works with no network afterwards, and a camera keeps its
  map even when the server is gone.
- The servers are kept in ``<data>/map_servers.json``, shared by every mission. Passwords and keys stay on this
  computer: the browser only ever sees that one is set. Each save is in the audit log (without the secrets).
- One fetch is at most ``MAX_TILES`` tiles or ``MAX_SIDE`` pixels a side: a bigger area comes at less detail, and
  the answer says what detail it got.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .terrain import M_PER_DEG, write_geotiff

FILE = "map_servers.json"
MAPS_DIR = "maps"
MAX_TILES = 324
MAX_SIDE = 6144
CHUNK = 2048                        # WMS and ArcGIS pictures are asked for in pieces of at most this many pixels
TIMEOUT_S = 25
UA = "LanePulse map client"
MERC = 20037508.342789244
SECRET = "••••••"   # what the browser sees for a password or key that is set

KINDS = {"wmts": "WMTS", "wms": "WMS", "xyz": "Tile address {z}/{x}/{y} (REST)", "arcgis": "ArcGIS REST service",
         "elevation-api": "Height service (Open-Elevation style REST)"}
ENCODINGS = {"terrarium": "Terrarium (red, green, blue)", "terrain-rgb": "Mapbox Terrain-RGB"}

PRESETS = [
    {"id": "esri-imagery", "name": "Esri World Imagery (satellite and aerial)", "role": "map", "kind": "xyz",
     "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "max_zoom": 19, "attribution": "Esri, Maxar, Earthstar Geographics and the GIS user community"},
    {"id": "osm", "name": "OpenStreetMap (streets)", "role": "map", "kind": "xyz",
     "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png", "max_zoom": 19,
     "attribution": "© OpenStreetMap contributors"},
    {"id": "aws-terrain", "name": "Terrain Tiles on AWS (heights, about 30 m, worldwide)", "role": "terrain",
     "kind": "xyz", "encoding": "terrarium", "max_zoom": 15,
     "url": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
     "attribution": "Mapzen terrain tiles: SRTM, GMTED, NED, ETOPO1 and others"},
]


class MapSourceError(ValueError):
    pass


# ------------------------------------------------------------------ the saved servers

_lock = threading.Lock()


def _path(data_root) -> Path:
    return Path(data_root) / FILE


def _load(data_root) -> list[dict]:
    p = _path(data_root)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("servers", [])
    except (OSError, ValueError):
        return []


def _public(s: dict) -> dict:
    out = {k: v for k, v in s.items() if k != "auth"}
    a = dict(s.get("auth") or {"type": "none"})
    for k in ("password", "key"):
        if a.get(k):
            a[k] = SECRET
    out["auth"] = a
    return out


def servers(data_root) -> list[dict]:
    """The operator's servers, with passwords and keys hidden."""
    return [_public(s) for s in _load(data_root)]


def catalogue(data_root) -> dict:
    return {"presets": PRESETS, "servers": servers(data_root), "kinds": KINDS, "encodings": ENCODINGS,
            "maps_folder": str(maps_folder(data_root))}


def maps_folder(data_root) -> Path:
    p = Path(data_root) / MAPS_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_server(s: dict, old: dict | None = None) -> dict:
    """Check one server as the operator described it; a hidden secret keeps the one saved before."""
    name = (s.get("name") or "").strip()
    kind = s.get("kind")
    role = s.get("role")
    url = (s.get("url") or "").strip()
    if not name:
        raise MapSourceError("Give the server a name.")
    if kind not in KINDS:
        raise MapSourceError("Pick the kind of server.")
    if role not in ("map", "terrain"):
        raise MapSourceError("Pick whether the server gives the map picture or ground heights.")
    if kind == "elevation-api" and role != "terrain":
        raise MapSourceError("A height service gives ground heights, not a map picture.")
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise MapSourceError("The address starts with http:// or https://, then the server's name or address.")
    if kind == "xyz" and not all(k in url for k in ("{z}", "{x}", "{y}")):
        raise MapSourceError("A tile address has {z}, {x} and {y} in it, like http://server/tiles/{z}/{x}/{y}.png.")
    a = s.get("auth") or {}
    at = a.get("type") or "none"
    if at not in ("none", "basic", "header", "query"):
        raise MapSourceError("Pick how to sign in to the server.")
    oa = (old or {}).get("auth") or {}
    auth = {"type": at}
    if at == "basic":
        auth["user"] = (a.get("user") or "").strip()
        auth["password"] = oa.get("password", "") if a.get("password") == SECRET else (a.get("password") or "")
        if not auth["user"]:
            raise MapSourceError("Give the user name for the server.")
    elif at in ("header", "query"):
        auth["name"] = (a.get("name") or ("X-API-Key" if at == "header" else "apikey")).strip()
        auth["key"] = oa.get("key", "") if a.get("key") == SECRET else (a.get("key") or "")
        if not auth["key"]:
            raise MapSourceError("Give the API key for the server.")
    enc = s.get("encoding") or None
    if role == "terrain" and kind == "xyz" and enc not in ENCODINGS:
        raise MapSourceError("Pick how the terrain tiles hold the heights.")
    out = {"id": s.get("id") or uuid.uuid4().hex[:12], "name": name, "role": role, "kind": kind, "url": url,
           "layer": (s.get("layer") or "").strip() or None, "style": (s.get("style") or "").strip() or None,
           "matrix_set": (s.get("matrix_set") or "").strip() or None, "format": (s.get("format") or "").strip() or None,
           "encoding": enc if role == "terrain" else None, "auth": auth,
           "max_zoom": int(s["max_zoom"]) if s.get("max_zoom") not in (None, "") else None}
    if kind in ("wms", "wmts") and not out["layer"]:
        raise MapSourceError("Connect to the server and pick a layer.")
    return out


def save_server(root, s: dict, author: str) -> dict:
    """Add or change one server (``root`` is a DataRoot)."""
    with _lock:
        items = _load(root.path)
        old = next((x for x in items if x["id"] == s.get("id")), None)
        new = clean_server(s, old)
        items = [new if x["id"] == new["id"] else x for x in items] if old else items + [new]
        _write(root, items, author, "map_server_saved", new)
    return _public(new)


def delete_server(root, server_id: str, author: str) -> None:
    with _lock:
        items = _load(root.path)
        gone = next((x for x in items if x["id"] == server_id), None)
        if gone is None:
            raise MapSourceError("That server is not in the list.")
        _write(root, [x for x in items if x["id"] != server_id], author, "map_server_removed", gone)


def _write(root, items, author, action, which) -> None:
    p = _path(root.path)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"servers": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)                       # passwords and keys: only this computer's user reads them
    except OSError:
        pass
    os.replace(tmp, p)
    root.con.execute("INSERT INTO audit_log(at,author,action,detail) VALUES (?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), author, action,
                      json.dumps(_public(which), ensure_ascii=False)))


def find_source(data_root, source_id: str) -> dict:
    s = next((x for x in PRESETS if x["id"] == source_id), None) or \
        next((x for x in _load(data_root) if x["id"] == source_id), None)
    if s is None:
        raise MapSourceError("That map source is not in the list any more.")
    return s


# ------------------------------------------------------------------ talking to a server

def _with_auth(src: dict, url: str) -> urllib.request.Request:
    a = src.get("auth") or {}
    if a.get("type") == "query":
        u = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qsl(u.query, keep_blank_values=True) + [(a["name"], a["key"])]
        url = urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(q)))
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    if a.get("type") == "basic":
        tok = base64.b64encode(f"{a['user']}:{a.get('password', '')}".encode()).decode()
        req.add_header("Authorization", f"Basic {tok}")
    elif a.get("type") == "header":
        req.add_header(a["name"], a["key"])
    return req


def _get(src: dict, url: str, missing_ok: bool = False) -> bytes | None:
    host = urllib.parse.urlparse(url).netloc
    try:
        with urllib.request.urlopen(_with_auth(src, url), timeout=TIMEOUT_S) as r:
            return r.read()
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code in (204, 404):
            return None
        if exc.code in (401, 403):
            raise MapSourceError(f"{host} refused the sign-in (error {exc.code}). Check the user name, password "
                                 "or key.") from None
        raise MapSourceError(f"{host} answered with error {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        why = getattr(exc, "reason", exc)
        raise MapSourceError(f"Cannot reach {host} ({why}). If this computer is offline, use a map file or a "
                             "server on your own network.") from None


def _xml(data: bytes) -> ET.Element:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        raise MapSourceError("The server's answer is not a capabilities document. Check the address.") from None
    if _local(root.tag) in ("ServiceExceptionReport", "ExceptionReport"):
        msg = " ".join(t.strip() for t in root.itertext() if t.strip())
        raise MapSourceError(f"The server says: {msg[:300]}")
    return root


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _kids(e: ET.Element, name: str):
    return [c for c in e if _local(c.tag) == name]


def _kid(e: ET.Element, name: str):
    return next((c for c in e if _local(c.tag) == name), None)


def _text(e, name, default=None):
    c = _kid(e, name) if e is not None else None
    return c.text.strip() if c is not None and c.text else default


def _with_query(url: str, **params) -> str:
    u = urllib.parse.urlparse(url)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True)
         if k.upper() not in {p.upper() for p in params}]
    q += list(params.items())
    return urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(q, safe=",:/{}")))


# WMS ---------------------------------------------------------------

def wms_layers(src: dict) -> dict:
    root = _xml(_get(src, _with_query(src["url"], SERVICE="WMS", REQUEST="GetCapabilities")))
    cap = _kid(root, "Capability")
    if cap is None:
        raise MapSourceError("This is not a WMS server (no capabilities). Check the address and the kind.")
    req = _kid(cap, "Request")
    gm = _kid(req, "GetMap") if req is not None else None
    formats = [f.text.strip() for f in _kids(gm, "Format")] if gm is not None else []
    out = []

    def walk(e, crs):
        crs = crs | {c.text.strip().upper() for c in _kids(e, "CRS") + _kids(e, "SRS") if c.text}
        name = _text(e, "Name")
        if name:
            out.append({"name": name, "title": _text(e, "Title", name),
                        "ok": bool({"EPSG:4326", "CRS:84"} & crs) or not crs})
        for c in _kids(e, "Layer"):
            walk(c, crs)

    for lay in _kids(cap, "Layer"):
        walk(lay, set())
    return {"layers": out, "formats": formats, "version": root.attrib.get("version", "1.1.1")}


def _wms_piece(src, bbox, w, h, terrain):
    top, left, bottom, right = bbox
    fmt = src.get("format") or ("image/tiff" if terrain else "image/png")
    url = _with_query(src["url"], SERVICE="WMS", VERSION="1.1.1", REQUEST="GetMap", LAYERS=src["layer"],
                      STYLES=src.get("style") or "", SRS="EPSG:4326", BBOX=f"{left},{bottom},{right},{top}",
                      WIDTH=str(w), HEIGHT=str(h), FORMAT=fmt, TRANSPARENT="FALSE")
    return _decode(_get(src, url), terrain, (w, h))


# ArcGIS REST -------------------------------------------------------

def arcgis_about(src: dict) -> dict:
    data = _get(src, _with_query(src["url"].rstrip("/"), f="json"))
    try:
        j = json.loads(data)
    except ValueError:
        raise MapSourceError("This address does not answer like an ArcGIS REST service.") from None
    if "error" in j:
        raise MapSourceError(f"The server says: {j['error'].get('message', 'error')}")
    name = j.get("name") or j.get("mapName") or j.get("serviceDescription") or "ArcGIS service"
    return {"layers": [], "title": str(name)[:120]}


def _arcgis_piece(src, bbox, w, h, terrain):
    top, left, bottom, right = bbox
    base = src["url"].rstrip("/")
    image = "/ImageServer" in base
    params = {"bbox": f"{left},{bottom},{right},{top}", "bboxSR": "4326", "imageSR": "4326", "size": f"{w},{h}",
              "f": "image"}
    if terrain:
        params |= {"format": "tiff", "pixelType": "F32", "noData": "-32768"}
    else:
        params["format"] = "jpgpng" if image else "png32"
    url = _with_query(base + ("/exportImage" if image else "/export"), **params)
    return _decode(_get(src, url), terrain, (w, h))


# WMTS --------------------------------------------------------------

def _wmts_caps_url(url: str) -> str:
    if url.lower().endswith(".xml") or "getcapabilities" in url.lower():
        return url
    if "{" in url:
        raise MapSourceError("Give the WMTS server's address (or its WMTSCapabilities.xml), not a tile template.")
    return _with_query(url, SERVICE="WMTS", REQUEST="GetCapabilities", VERSION="1.0.0")


def _wmts_caps(src: dict) -> ET.Element:
    return _xml(_get(src, _wmts_caps_url(src["url"])))


def _crs_kind(crs: str) -> str | None:
    c = crs.upper()
    if any(k in c for k in ("3857", "900913", "102100", "GOOGLEMAPSCOMPATIBLE")):
        return "3857"
    if "CRS84" in c:
        return "crs84"
    if c.endswith(":4326") or c.endswith("/4326") or c.endswith("::4326"):
        return "4326"
    return None


def wmts_layers(src: dict) -> dict:
    root = _wmts_caps(src)
    contents = _kid(root, "Contents")
    if contents is None:
        raise MapSourceError("This is not a WMTS server (no contents). Check the address and the kind.")
    sets = {(_text(s, "Identifier")): _text(s, "SupportedCRS", "") for s in _kids(contents, "TileMatrixSet")}
    out = []
    for lay in _kids(contents, "Layer"):
        ident = _text(lay, "Identifier")
        links = [_text(lk, "TileMatrixSet") for lk in _kids(lay, "TileMatrixSetLink")]
        usable = [m for m in links if _crs_kind(sets.get(m, "") or m or "")]
        out.append({"name": ident, "title": _text(lay, "Title", ident), "ok": bool(usable),
                    "matrix_sets": usable or links, "formats": [f.text.strip() for f in _kids(lay, "Format") if f.text]})
    return {"layers": out}


def _wmts_grid(src: dict):
    root = _wmts_caps(src)
    contents = _kid(root, "Contents")
    lay = next((x for x in _kids(contents, "Layer") if _text(x, "Identifier") == src["layer"]), None)
    if lay is None:
        raise MapSourceError(f"The server has no layer {src['layer']} any more.")
    links = [_text(lk, "TileMatrixSet") for lk in _kids(lay, "TileMatrixSetLink")]
    sets = {_text(s, "Identifier"): s for s in _kids(contents, "TileMatrixSet")}
    want = src.get("matrix_set")
    cands = [want] if want else links
    tms = None
    for m in cands:
        s = sets.get(m)
        if s is not None and _crs_kind(_text(s, "SupportedCRS", "") or m):
            tms, mid = s, m
            break
    if tms is None:
        raise MapSourceError("This layer has no tile grid in Web Mercator or latitude and longitude, which LanePulse "
                             "reads. Ask for EPSG:3857 or EPSG:4326 tiles on the server.")
    crs = _crs_kind(_text(tms, "SupportedCRS", "") or mid)
    style = src.get("style") or next((_text(st, "Identifier") for st in _kids(lay, "Style")
                                      if st.attrib.get("isDefault") == "true"), None) \
        or next((_text(st, "Identifier") for st in _kids(lay, "Style")), "default")
    fmt = src.get("format") or next((f.text.strip() for f in _kids(lay, "Format") if f.text), "image/png")
    template = next((r.attrib.get("template") for r in _kids(lay, "ResourceURL")
                     if r.attrib.get("resourceType") == "tile"), None)
    unit = 1.0 if crs == "3857" else 2 * math.pi * 6378137.0 / 360
    levels = []
    for tm in _kids(tms, "TileMatrix"):
        a, b = (float(v) for v in _text(tm, "TopLeftCorner").split()[:2])
        x0, y0 = (b, a) if crs == "4326" else (a, b)            # EPSG:4326 lists latitude first
        res = float(_text(tm, "ScaleDenominator")) * 0.28e-3 / unit
        levels.append({"id": _text(tm, "Identifier"), "x0": x0, "y0": y0, "res": res,
                       "tw": int(_text(tm, "TileWidth", "256")), "th": int(_text(tm, "TileHeight", "256")),
                       "mw": int(_text(tm, "MatrixWidth", "1")), "mh": int(_text(tm, "MatrixHeight", "1"))})
    levels.sort(key=lambda lv: -lv["res"])
    base = src["url"].split("?")[0]

    def tile_url(lv, col, row):
        if template:
            return (template.replace("{TileMatrixSet}", mid).replace("{TileMatrix}", lv["id"])
                    .replace("{TileRow}", str(row)).replace("{TileCol}", str(col)).replace("{Style}", style))
        return _with_query(base, SERVICE="WMTS", REQUEST="GetTile", VERSION="1.0.0", LAYER=src["layer"], STYLE=style,
                           TILEMATRIXSET=mid, TILEMATRIX=lv["id"], TILEROW=str(row), TILECOL=str(col), FORMAT=fmt)

    return ("3857" if crs == "3857" else "4326"), levels, tile_url


def _xyz_grid(src: dict):
    levels = []
    for z in range(0, (src.get("max_zoom") or 19) + 1):
        n = 2 ** z
        levels.append({"id": str(z), "x0": -MERC, "y0": MERC, "res": 2 * MERC / (256 * n), "tw": 256, "th": 256,
                       "mw": n, "mh": n})

    def tile_url(lv, col, row):
        z = int(lv["id"])
        u = src["url"].replace("{z}", str(z)).replace("{x}", str(col)).replace("{y}", str(row))
        return u.replace("{-y}", str(2 ** z - 1 - row)).replace("{s}", "abc"[(col + row) % 3])

    return "3857", levels, tile_url


# ------------------------------------------------------------------ fetching an area

def _to_crs(crs, lat, lon):
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if crs == "4326":
        return lon, lat
    x = lon * MERC / 180
    y = np.log(np.tan(np.radians(90 + np.clip(lat, -85.05, 85.05)) / 2)) * MERC / math.pi
    return x, y


def area(lat: float, lon: float, side_m: float) -> tuple[float, float, float, float]:
    """Top, left, bottom, right of a square ``side_m`` across centred on a point."""
    if not (-85 <= lat <= 85 and -180 <= lon <= 180):
        raise MapSourceError("The centre must be a latitude between -85 and 85 and a longitude between -180 and 180.")
    if not (50 <= side_m <= 20000):
        raise MapSourceError("The area is between 50 m and 20 km across.")
    dlat = side_m / 2 / M_PER_DEG
    dlon = dlat / max(0.05, math.cos(math.radians(lat)))
    return lat + dlat, lon - dlon, lat - dlat, lon + dlon


def _decode(data: bytes | None, terrain: bool, size=None, encoding: str | None = None) -> np.ndarray | None:
    if data is None:
        return None
    if terrain and encoding is None:                     # heights as a float TIFF
        import tifffile
        try:
            a = tifffile.imread(io.BytesIO(data)).astype(np.float32)
        except Exception:  # noqa: BLE001 - any unreadable answer
            if data[:1] == b"<" or data[:1] == b"{":
                raise MapSourceError("The server sent a message instead of heights: "
                                     + data[:200].decode("utf-8", "replace")) from None
            raise MapSourceError("The server sent a picture, not heights. Pick a layer of heights (a DTM) and a "
                                 "TIFF format.") from None
        if a.ndim == 3:
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        a[a < -1000] = np.nan
        return a
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        txt = data[:300].decode("utf-8", "replace")
        if "<" in txt[:5] or "{" in txt[:5]:
            msg = re.sub(r"<[^>]+>", " ", txt)
            raise MapSourceError("The server sent a message instead of a picture: " + " ".join(msg.split())[:200])
        raise MapSourceError("The server's picture cannot be read.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if terrain:
        b, g, r = (img[..., k].astype(np.float64) for k in range(3))
        h = (r * 256 + g + b / 256 - 32768) if encoding == "terrarium" else (-10000 + (r * 65536 + g * 256 + b) * 0.1)
        if img.shape[2] == 4:
            h[img[..., 3] == 0] = np.nan
        return h.astype(np.float32)
    if img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255
        img = (img[..., :3] * a).astype(np.uint8)
    return img


def _fetch_tiled(src, bbox, want_m, terrain, progress):
    crs, levels, tile_url = _xyz_grid(src) if src["kind"] == "xyz" else _wmts_grid(src)
    top, left, bottom, right = bbox
    xs, ys = _to_crs(crs, [top, bottom], [left, right])
    x_min, x_max, y_min, y_max = float(xs[0]), float(xs[1]), float(ys[1]), float(ys[0])
    mid = math.radians((top + bottom) / 2)
    m_per_unit = math.cos(mid) if crs == "3857" else M_PER_DEG * math.cos(mid)
    pick = None
    for lv in levels:                                    # coarse to fine: the finest that fits
        c0, c1 = int((x_min - lv["x0"]) // (lv["res"] * lv["tw"])), int((x_max - lv["x0"]) // (lv["res"] * lv["tw"]))
        r0, r1 = int((lv["y0"] - y_max) // (lv["res"] * lv["th"])), int((lv["y0"] - y_min) // (lv["res"] * lv["th"]))
        n = (c1 - c0 + 1) * (r1 - r0 + 1)
        if n > MAX_TILES or max((c1 - c0 + 1) * lv["tw"], (r1 - r0 + 1) * lv["th"]) > MAX_SIDE * 1.5:
            break
        pick = (lv, c0, c1, r0, r1)
        if lv["res"] * m_per_unit <= want_m:
            break
    if pick is None:
        raise MapSourceError("The area is too big for this source. Choose a smaller area.")
    lv, c0, c1, r0, r1 = pick
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, lv["mw"] - 1), min(r1, lv["mh"] - 1)
    tw, th = lv["tw"], lv["th"]
    enc = src.get("encoding") if terrain else None
    if terrain and enc is None and src["kind"] == "xyz":
        enc = "terrarium"
    if terrain:
        mosaic = np.full(((r1 - r0 + 1) * th, (c1 - c0 + 1) * tw), np.nan, np.float32)
    else:
        mosaic = np.zeros(((r1 - r0 + 1) * th, (c1 - c0 + 1) * tw, 3), np.uint8)
    jobs = [(c, r) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    done = {"n": 0, "got": 0}

    def one(cr):
        c, r = cr
        t = _decode(_get(src, tile_url(lv, c, r), missing_ok=True), terrain, None, enc)
        done["n"] += 1
        if progress:
            progress(done["n"], len(jobs))
        if t is None:
            return
        if t.shape[:2] != (th, tw):
            t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_LINEAR)
        mosaic[(r - r0) * th:(r - r0 + 1) * th, (c - c0) * tw:(c - c0 + 1) * tw] = t
        done["got"] += 1

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    if done["got"] == 0:
        raise MapSourceError("The server has no tiles for this area.")
    # resample onto a latitude and longitude grid at about the tiles' detail
    m_px = lv["res"] * m_per_unit
    W = int(min(MAX_SIDE, max(16, round((right - left) * M_PER_DEG * math.cos(mid) / m_px))))
    H = int(min(MAX_SIDE, max(16, round((top - bottom) * M_PER_DEG / m_px))))
    lat = top - (np.arange(H) + 0.5) * (top - bottom) / H
    lon = left + (np.arange(W) + 0.5) * (right - left) / W
    gx, _ = _to_crs(crs, np.full_like(lon, lat[0]), lon)
    _, gy = _to_crs(crs, lat, np.full_like(lat, lon[0]))
    mx = ((gx - lv["x0"]) / lv["res"] - c0 * tw - 0.5).astype(np.float32)
    my = ((lv["y0"] - gy) / lv["res"] - r0 * th - 0.5).astype(np.float32)
    MX, MY = np.meshgrid(mx, my)
    if terrain:
        filled = np.nan_to_num(mosaic, nan=float(np.nanmean(mosaic)))
        out = cv2.remap(filled, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        holes = cv2.remap(np.isnan(mosaic).astype(np.float32), MX, MY, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        out[holes > 0.5] = np.nan
    else:
        out = cv2.remap(mosaic, MX, MY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out, {"tiles": len(jobs), "tiles_found": done["got"], "level": lv["id"]}


def _fetch_pictures(src, bbox, want_m, terrain, progress):
    top, left, bottom, right = bbox
    mid = math.radians((top + bottom) / 2)
    wm = (right - left) * M_PER_DEG * math.cos(mid)
    hm = (top - bottom) * M_PER_DEG
    k = max(want_m, max(wm, hm) / MAX_SIDE)
    W, H = max(16, round(wm / k)), max(16, round(hm / k))
    piece = _wms_piece if src["kind"] == "wms" else _arcgis_piece
    out = np.full((H, W), np.nan, np.float32) if terrain else np.zeros((H, W, 3), np.uint8)
    jobs = [(x, y) for y in range(0, H, CHUNK) for x in range(0, W, CHUNK)]
    for i, (x, y) in enumerate(jobs):
        w, h = min(CHUNK, W - x), min(CHUNK, H - y)
        b = (top - y / H * (top - bottom), left + x / W * (right - left),
             top - (y + h) / H * (top - bottom), left + (x + w) / W * (right - left))
        img = piece(src, b, w, h, terrain)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        out[y:y + h, x:x + w] = img
        if progress:
            progress(i + 1, len(jobs))
    return out, {"pieces": len(jobs)}


def _fetch_heights_api(src, bbox, want_m, progress):
    top, left, bottom, right = bbox
    n = 48                                              # 48 x 48 heights: a height service is slow per point
    lat = top - (np.arange(n) + 0.5) * (top - bottom) / n
    lon = left + (np.arange(n) + 0.5) * (right - left) / n
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    pts = list(zip(LA.ravel(), LO.ravel()))
    out = np.full(len(pts), np.nan, np.float32)
    step = 100
    for i in range(0, len(pts), step):
        part = pts[i:i + step]
        url = _with_query(src["url"], locations="|".join(f"{a:.6f},{b:.6f}" for a, b in part))
        try:
            j = json.loads(_get(src, url))
            res = j.get("results") or j.get("elevations") or []
            vals = [r.get("elevation") if isinstance(r, dict) else r for r in res]
        except (ValueError, AttributeError):
            raise MapSourceError("The height service's answer is not like Open-Elevation's "
                                 "({\"results\": [{\"elevation\": ...}]}).") from None
        if len(vals) != len(part):
            raise MapSourceError("The height service gave back a different number of heights than asked for.")
        out[i:i + len(part)] = [np.nan if v is None else float(v) for v in vals]
        if progress:
            progress(i + len(part), len(pts))
    return out.reshape(n, n), {"points": len(pts)}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:40] or "map"


def fetch(data_root, source: dict, lat: float, lon: float, side_m: float, want_m: float = 0.3,
          progress=None) -> dict:
    """Fetch an area from a source and save it in the maps folder; returns the saved file and what it holds."""
    terrain = source["role"] == "terrain"
    bbox = area(lat, lon, side_m)
    if source["kind"] in ("xyz", "wmts"):
        img, how = _fetch_tiled(source, bbox, want_m, terrain, progress)
    elif source["kind"] in ("wms", "arcgis"):
        img, how = _fetch_pictures(source, bbox, want_m, terrain, progress)
    elif source["kind"] == "elevation-api":
        img, how = _fetch_heights_api(source, bbox, want_m, progress)
    else:
        raise MapSourceError("This kind of source cannot be fetched.")
    if terrain:
        if not np.isfinite(img).any():
            raise MapSourceError("The source has no heights for this area.")
    elif int(img.max()) == 0:
        raise MapSourceError("The source sent only empty pictures for this area.")
    top, left, bottom, right = bbox
    size = f"{side_m / 1000:g}km" if side_m >= 1000 else f"{side_m:g}m"
    name = f"{_slug(source['name'].split('(')[0])}_{lat:.5f}_{lon:.5f}_{size}{'_heights' if terrain else ''}.tif"
    path = maps_folder(data_root) / name
    desc = f"LanePulse: {source['name']}; {source.get('attribution', '')}; fetched {datetime.now(timezone.utc):%Y-%m-%d}"
    if terrain:
        write_geotiff(path, np.nan_to_num(img, nan=-32768.0).astype(np.float32), bbox, desc, -32768.0)
    else:
        write_geotiff(path, img, bbox, desc)
    mid = math.radians(lat)
    return {"path": str(path), "name": name, "role": source["role"], "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "m_per_px": round((right - left) * M_PER_DEG * math.cos(mid) / img.shape[1], 2),
            "attribution": source.get("attribution"), **how}


def connect(source: dict) -> dict:
    """Check a server and list what it offers (layers for WMS and WMTS)."""
    k = source["kind"]
    if k == "wms":
        return wms_layers(source)
    if k == "wmts":
        return wmts_layers(source)
    if k == "arcgis":
        return arcgis_about(source)
    if k == "xyz":
        u = source["url"].replace("{z}", "0").replace("{x}", "0").replace("{y}", "0").replace("{-y}", "0") \
            .replace("{s}", "a")
        data = _get(source, u, missing_ok=True)
        if data is not None:
            _decode(data, False)
        return {"layers": [], "title": "The tile server answers."}
    if k == "elevation-api":
        j = _get(source, _with_query(source["url"], locations="0,0"))
        try:
            json.loads(j)
        except ValueError:
            raise MapSourceError("The height service's answer is not JSON.") from None
        return {"layers": [], "title": "The height service answers."}
    raise MapSourceError("Pick the kind of server.")
