"""Maps and ground heights from files, online services and servers on the network (WMTS, WMS, tile addresses,
ArcGIS REST, height services), with sign-in; and the frame laid on the map with its footprint."""
import base64
import math
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from trafficwb import map_sources as ms
from trafficwb.terrain import load_terrain, shade_over, write_geotiff

LAT, LON = 32.0765, 34.7835
WORLD = None


def _world():
    """A known picture of the whole area, in latitude and longitude: a grid of coloured squares, so a fetched map
    can be checked pixel by pixel against where it should be."""
    global WORLD
    if WORLD is None:
        img = np.zeros((1000, 1000, 3), np.uint8)
        for i in range(0, 1000, 50):
            for j in range(0, 1000, 50):
                img[i:i + 50, j:j + 50] = ((i * 7) % 256, (j * 5) % 256, ((i + j) * 3) % 256)
        WORLD = img
    return WORLD


W_TOP, W_LEFT, W_BOTTOM, W_RIGHT = LAT + 0.02, LON - 0.02, LAT - 0.02, LON + 0.02


def _world_at(lat, lon):
    img = _world()
    r = (W_TOP - lat) / (W_TOP - W_BOTTOM) * img.shape[0]
    c = (lon - W_LEFT) / (W_RIGHT - W_LEFT) * img.shape[1]
    return r, c


def _height(lat, lon):
    return 100 + (lat - LAT) * 20000 + (lon - LON) * 5000                    # a tilted plane, metres


def _render(top, left, bottom, right, w, h, heights=False):
    lat = top - (np.arange(h) + 0.5) * (top - bottom) / h
    lon = left + (np.arange(w) + 0.5) * (right - left) / w
    LO, LA = np.meshgrid(lon, lat)
    if heights:
        return _height(LA, LO).astype(np.float32)
    r, c = _world_at(LA, LO)
    return cv2.remap(_world(), c.astype(np.float32) - 0.5, r.astype(np.float32) - 0.5, cv2.INTER_NEAREST)


def _merc_tile_bounds(z, x, y):
    n = 2 ** z
    lon0, lon1 = x / n * 360 - 180, (x + 1) / n * 360 - 180
    lat = lambda yy: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))   # noqa: E731
    return lat(y), lon0, lat(y + 1), lon1


def _merc_tile(z, x, y, heights=False):
    top, left, bottom, right = _merc_tile_bounds(z, x, y)
    rows = np.arange(256) + 0.5
    n = 2 ** z
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + rows / 256) / n))))
    lon = left + (np.arange(256) + 0.5) / 256 * (right - left)
    LO, LA = np.meshgrid(lon, lat)
    if heights:
        hgt = _height(LA, LO) + 32768
        r = np.floor(hgt / 256)
        g = np.floor(hgt - r * 256)
        b = np.round((hgt - r * 256 - g) * 256)
        return np.dstack([b, g, r]).astype(np.uint8)
    rr, cc = _world_at(LA, LO)
    return cv2.remap(_world(), cc.astype(np.float32) - 0.5, rr.astype(np.float32) - 0.5, cv2.INTER_NEAREST)


WMTS_CAPS = """<?xml version="1.0"?>
<Capabilities xmlns="http://www.opengis.net/wmts/1.0" xmlns:ows="http://www.opengis.net/ows/1.1" version="1.0.0">
 <Contents>
  <Layer><ows:Title>Orthophoto 2026</ows:Title><ows:Identifier>ortho</ows:Identifier>
   <Style isDefault="true"><ows:Identifier>default</ows:Identifier></Style><Format>image/png</Format>
   <TileMatrixSetLink><TileMatrixSet>GoogleMapsCompatible</TileMatrixSet></TileMatrixSetLink>
  </Layer>
  <TileMatrixSet><ows:Identifier>GoogleMapsCompatible</ows:Identifier>
   <ows:SupportedCRS>urn:ogc:def:crs:EPSG::3857</ows:SupportedCRS>
   {matrices}
  </TileMatrixSet>
 </Contents>
</Capabilities>"""


def _wmts_caps():
    ms_ = []
    for z in range(0, 19):
        ms_.append(f"<TileMatrix><ows:Identifier>{z}</ows:Identifier>"
                   f"<ScaleDenominator>{559082264.0287178 / 2 ** z}</ScaleDenominator>"
                   f"<TopLeftCorner>-20037508.3427892 20037508.3427892</TopLeftCorner><TileWidth>256</TileWidth>"
                   f"<TileHeight>256</TileHeight><MatrixWidth>{2 ** z}</MatrixWidth><MatrixHeight>{2 ** z}</MatrixHeight>"
                   f"</TileMatrix>")
    return WMTS_CAPS.replace("{matrices}", "".join(ms_))


WMS_CAPS = """<?xml version="1.0"?>
<WMT_MS_Capabilities version="1.1.1"><Capability>
 <Request><GetMap><Format>image/png</Format><Format>image/tiff</Format></GetMap></Request>
 <Layer><Title>Local server</Title><SRS>EPSG:4326</SRS>
  <Layer><Name>ortho</Name><Title>Orthophoto</Title></Layer>
  <Layer><Name>dtm</Name><Title>Terrain model</Title></Layer>
 </Layer>
</Capability></WMT_MS_Capabilities>"""


class Fake(BaseHTTPRequestHandler):
    seen: list = []
    need_auth = False

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="image/png", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = {k.upper(): v for k, v in urllib.parse.parse_qsl(u.query)}
        Fake.seen.append(self.path)
        if Fake.need_auth and self.headers.get("Authorization") != "Basic " + base64.b64encode(b"op:pw").decode():
            return self._send(b"no", "text/plain", 401)
        if u.path.startswith("/xyz/") or u.path.startswith("/terrarium/"):
            z, x, y = (int(v.split(".")[0]) for v in u.path.split("/")[2:5])
            img = _merc_tile(z, x, y, heights=u.path.startswith("/terrarium/"))
            return self._send(cv2.imencode(".png", img)[1].tobytes())
        if u.path == "/wmts":
            if q.get("REQUEST") == "GetCapabilities":
                return self._send(_wmts_caps().encode(), "text/xml")
            img = _merc_tile(int(q["TILEMATRIX"]), int(q["TILECOL"]), int(q["TILEROW"]))
            return self._send(cv2.imencode(".png", img)[1].tobytes())
        if u.path == "/wms":
            if q.get("REQUEST") == "GetCapabilities":
                return self._send(WMS_CAPS.encode(), "text/xml")
            left, bottom, right, top = (float(v) for v in q["BBOX"].split(","))
            w, h = int(q["WIDTH"]), int(q["HEIGHT"])
            if q["LAYERS"] == "dtm":
                import io
                import tifffile
                b = io.BytesIO()
                tifffile.imwrite(b, _render(top, left, bottom, right, w, h, heights=True))
                return self._send(b.getvalue(), "image/tiff")
            return self._send(cv2.imencode(".png", _render(top, left, bottom, right, w, h))[1].tobytes())
        if u.path.startswith("/arcgis/rest/services/Ortho/MapServer"):
            if u.path.endswith("/export"):
                left, bottom, right, top = (float(v) for v in q["BBOX"].split(","))
                w, h = (int(v) for v in q["SIZE"].split(","))
                return self._send(cv2.imencode(".png", _render(top, left, bottom, right, w, h))[1].tobytes())
            return self._send(b'{"mapName": "Ortho"}', "application/json")
        if u.path == "/api/v1/lookup":
            pts = [p.split(",") for p in q["LOCATIONS"].split("|")]
            res = [{"latitude": float(a), "longitude": float(b), "elevation": _height(float(a), float(b))}
                   for a, b in pts]
            import json
            return self._send(json.dumps({"results": res}).encode(), "application/json")
        self._send(b"", "text/plain", 404)


@pytest.fixture()
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    Fake.seen, Fake.need_auth = [], False
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _check_map(path, tol_px=2.5):
    """The fetched map matches the known world where it says it is."""
    from trafficwb.video_geo import load_map, map_info
    gm = load_map(path)
    info = map_info(gm)
    want = _render(info["top"], info["left"], info["bottom"], info["right"], info["width"], info["height"])
    diff = np.abs(gm.image.astype(int) - want.astype(int)).max(axis=2)
    assert (diff < 10).mean() > 0.9, (diff < 10).mean()
    return info


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_maps_from_every_kind_of_local_server(server, tmp_path):
    for kind, url, layer in (("xyz", server + "/xyz/{z}/{x}/{y}.png", None), ("wmts", server + "/wmts", "ortho"),
                             ("wms", server + "/wms", "ortho"),
                             ("arcgis", server + "/arcgis/rest/services/Ortho/MapServer", None)):
        src = ms.clean_server({"name": f"Local {kind}", "role": "map", "kind": kind, "url": url, "layer": layer})
        got = ms.connect(src)
        if kind in ("wms", "wmts"):
            assert "ortho" in [x["name"] for x in got["layers"]]
        r = ms.fetch(tmp_path, src, LAT, LON, 600, 1.0)
        info = _check_map(r["path"])
        assert info["top"] == pytest.approx(LAT + 300 / ms.M_PER_DEG, abs=1e-6)
        assert r["m_per_px"] <= 1.3


def test_terrain_from_tiles_wms_and_a_height_service(server, tmp_path):
    for kind, url, layer, enc in (("xyz", server + "/terrarium/{z}/{x}/{y}.png", None, "terrarium"),
                                  ("wms", server + "/wms", "dtm", None),
                                  ("elevation-api", server + "/api/v1/lookup", None, None)):
        src = ms.clean_server({"name": f"T {kind}", "role": "terrain", "kind": kind, "url": url, "layer": layer,
                               "encoding": enc})
        r = ms.fetch(tmp_path, src, LAT, LON, 800, 5.0)
        t = load_terrain(r["path"])
        for dlat, dlon in ((0, 0), (0.002, -0.003), (-0.003, 0.002)):
            assert t.height_at(LAT + dlat, LON + dlon) == pytest.approx(_height(LAT + dlat, LON + dlon), abs=1.0)
        assert np.isnan(t.height_at(LAT + 1, LON))


def test_sign_in_and_secrets_stay_on_this_computer(server, tmp_path):
    from trafficwb.store import DataRoot
    Fake.need_auth = True
    root = DataRoot(tmp_path / "data")
    with pytest.raises(ms.MapSourceError, match="refused the sign-in"):
        ms.connect(ms.clean_server({"name": "L", "role": "map", "kind": "wms", "url": server + "/wms", "layer": "x"}))
    saved = ms.save_server(root, {"name": "City WMS", "role": "map", "kind": "wms", "url": server + "/wms",
                                  "layer": "ortho", "auth": {"type": "basic", "user": "op", "password": "pw"}}, "ana")
    assert saved["auth"]["password"] == ms.SECRET
    assert "pw" not in str(ms.catalogue(root.path))
    # a change with the hidden password keeps the real one
    ms.save_server(root, {**saved, "name": "City WMS 2"}, "ana")
    src = ms.find_source(root.path, saved["id"])
    assert src["auth"]["password"] == "pw" and src["name"] == "City WMS 2"
    assert ms.fetch(root.path, src, LAT, LON, 300, 1.0)["path"].endswith(".tif")
    log = root.con.execute("SELECT detail FROM audit_log WHERE action='map_server_saved'").fetchall()
    assert len(log) == 2 and all('"pw"' not in r[0] for r in log)
    ms.delete_server(root, saved["id"], "ana")
    assert ms.servers(root.path) == []


def test_unreachable_server_says_what_to_do(tmp_path):
    src = ms.clean_server({"name": "Gone", "role": "map", "kind": "xyz", "url": "http://127.0.0.1:9/{z}/{x}/{y}.png"})
    with pytest.raises(ms.MapSourceError, match="Cannot reach"):
        ms.fetch(tmp_path, src, LAT, LON, 300, 1.0)


def test_bad_server_descriptions_are_refused():
    for bad, msg in (({"kind": "xyz", "url": "http://x/tiles"}, "{z}"), ({"kind": "wms", "url": "ftp://x"}, "http"),
                     ({"kind": "wms", "url": "http://x/wms"}, "layer"),
                     ({"kind": "elevation-api", "url": "http://x/api"}, "heights")):
        with pytest.raises(ms.MapSourceError, match=msg):
            ms.clean_server({"name": "x", "role": "map", **bad})


def test_terrain_files_hgt_and_geotiff(tmp_path):
    n = 121
    a = (np.arange(n * n).reshape(n, n) % 500).astype(">i2")
    a[5, 5] = -32768
    a.tofile(tmp_path / "N32E034.hgt")
    t = load_terrain(tmp_path / "N32E034.hgt")
    assert t.height_at(33.0, 34.0) == pytest.approx(0, abs=1e-3)                      # the top-left sample sits on the corner
    assert t.height_at(32.0, 35.0) == pytest.approx(float(a[-1, -1]), abs=1e-3)
    assert np.isnan(t.data[5, 5])
    data = np.full((50, 60), 40.0, np.float32)
    data[:, 30:] = 60.0
    write_geotiff(tmp_path / "dtm.tif", data, (32.1, 34.7, 32.0, 34.82), nodata=-32768.0)
    t2 = load_terrain(tmp_path / "dtm.tif")
    assert t2.height_at(32.05, 34.71) == pytest.approx(40) and t2.height_at(32.05, 34.81) == pytest.approx(60)
    shade = shade_over(t2, (32.1, 34.7, 32.0, 34.82), (120, 100))
    assert shade.shape == (100, 120, 4) and shade[..., 3].max() > 0 and shade[:, :20, 3].max() == 0  # only the step
    pic = np.zeros((10, 10, 3), np.uint8)
    write_geotiff(tmp_path / "pic.tif", pic, (32.1, 34.7, 32.0, 34.82))
    with pytest.raises(ValueError, match="picture, not heights"):
        load_terrain(tmp_path / "pic.tif")


def test_footprint_parts_and_the_frame_on_the_ground_has_no_black():
    from trafficwb.georeg import LocalFrame, apply_h, frame_on_ground, ground_parts, pose_homography
    w, h = 640, 360
    # an oblique camera 8 m up looking 12 degrees down: the horizon is in the frame
    G = pose_homography(w, h, 70, 0.0, 0.0, 8.0, 0.0, -12.0)
    parts = ground_parts(G, w, h, max_range_m=300)
    assert parts is not None and parts["top_row"] > 0
    far_n = parts["far"][:, 1].max()
    near_n = parts["near"][:, 1].max()
    assert far_n > near_n > 0                                            # the far part is further north (ahead)
    base = apply_h(G, [[w / 2, h]])[0]
    assert np.hypot(*(parts["far"][:2] - base).T).max() <= 300 + 1
    frame = np.full((h, w, 3), 128, np.uint8)
    frame[:, :40] = 0                                                    # a black bar on the left
    fr = LocalFrame(LAT, LON)
    a = np.r_[parts["far"], parts["near"]]
    lat, lon = fr.to_latlon(a[:, 0], a[:, 1])
    lay = frame_on_ground(frame, G, fr, (lat.max(), lon.min(), lat.min(), lon.max()), (400, 400))
    seen = lay[..., 3] > 0
    assert seen.mean() > 0.1
    assert lay[seen][:, :3].min() > 100                                   # no black pixel is drawn
    # straight down: no horizon, the whole frame is ground
    G2 = pose_homography(w, h, 70, 0.0, 0.0, 50.0, 0.0, -90.0)
    p2 = ground_parts(G2, w, h)
    assert p2["top_row"] == 0
    assert apply_h(G2, [[w / 2, h / 2]])[0] == pytest.approx([0, 0], abs=1e-6)


def test_the_api_fetches_and_lists(server, tmp_path):
    from fastapi.testclient import TestClient
    from trafficwb.web.app import create_app
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    hd = {"X-TWB": "1"}
    cat = c.get("/api/map-sources", headers=hd).json()
    assert {p["id"] for p in cat["presets"]} >= {"esri-imagery", "aws-terrain"}
    got = c.post("/api/map-sources/connect", json={"server": {"name": "W", "role": "map", "kind": "wmts",
                                                              "url": server + "/wmts"}}, headers=hd)
    assert got.status_code == 200 and got.json()["layers"][0]["name"] == "ortho"
    sv = c.post("/api/map-sources/servers", json={"name": "W", "role": "map", "kind": "wmts", "url": server + "/wmts",
                                                  "layer": "ortho"}, headers=hd).json()["server"]
    r = c.post("/api/map-sources/fetch", json={"source_id": sv["id"], "lat": LAT, "lon": LON, "side_m": 400,
                                               "m_per_px": 1.0}, headers=hd)
    assert r.status_code == 200, r.text
    path = r.json()["path"]
    assert path.startswith(cat["maps_folder"])
    files = c.get("/api/map-files", params={"folder": cat["maps_folder"]}, headers=hd).json()["files"]
    assert [f["path"] for f in files] == [path]
    tr = c.post("/api/map-sources/servers", json={"name": "T", "role": "terrain", "kind": "elevation-api",
                                                  "url": server + "/api/v1/lookup"}, headers=hd).json()["server"]
    d = c.post("/api/map-sources/fetch", json={"source_id": tr["id"], "lat": LAT, "lon": LON, "side_m": 400},
               headers=hd).json()
    info = c.get("/api/terrain-info", params={"path": d["path"]}, headers=hd).json()
    assert info["min_m"] < 100 < info["max_m"]
    png = c.get("/api/terrain-shade", params={"path": d["path"], "top": info["top"], "left": info["left"],
                                               "bottom": info["bottom"], "right": info["right"], "w": 200}, headers=hd)
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    bad = c.post("/api/map-sources/fetch", json={"source_id": "nope", "lat": LAT, "lon": LON}, headers=hd)
    assert bad.status_code == 400


def test_drone_map_view_paints_green_when_registered_and_red_on_flight_data_only():
    from trafficwb.georeg import (REGISTERED_BGR, UNREGISTERED_BGR, FrameFix, GeoMap, LocalFrame, is_registered,
                                  map_view, pose_homography)
    gm = GeoMap(np.full((400, 400, 3), 90, np.uint8), (LAT + 0.002, LON - 0.002), (LAT - 0.002, LON + 0.002),
                LocalFrame(LAT, LON))
    G = pose_homography(320, 180, 60, 0.0, 0.0, 60.0, 0.0, -50.0)
    frame = np.full((180, 320, 3), 200, np.uint8)
    assert is_registered(FrameFix(G, "map", 0.0)) and is_registered(FrameFix(G, "chain", 2.0))
    assert not is_registered(FrameFix(G, "flight-data", None)) and not is_registered(FrameFix(G, "chain", 9.0))
    for ok, col in ((True, REGISTERED_BGR), (False, UNREGISTERED_BGR)):
        v = map_view(gm, frame, G, ok, 300)
        assert (np.abs(v.astype(int) - np.array(col)).sum(axis=2) < 30).sum() > 50
