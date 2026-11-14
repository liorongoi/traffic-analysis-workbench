"""A fixed camera on the map: map files, points, the fit and its checks, and each target's path on the map."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video_attributes import painted_cars  # noqa: E402
from test_video import mission, run  # noqa: E402

av = pytest.importorskip("av")


def _geotiff(path: Path, top=32.0805, left=34.7795, size=800, deg=1e-5):
    import tifffile
    img = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    tifffile.imwrite(str(path), img, extratags=[
        (33550, "d", 3, (deg, deg, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, left, top, 0.0)),
        (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326))])


def test_a_camera_on_the_map_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from trafficwb.georeg import LocalFrame, apply_h, pose_homography
    from trafficwb.video_review import VideoService
    from trafficwb.web.app import create_app
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (7.0, painted_cars, False),
                           "CAM1_20261021T121000Z.mp4": (4.0, painted_cars, False)})
    run(m)
    svc = VideoService(m)
    evs = svc.events()
    vid = evs[0].source_version_id
    other = next(e.source_version_id for e in evs if e.source_version_id != vid)
    maps = tmp_path / "maps"
    maps.mkdir()
    _geotiff(maps / "ortho.tif")
    cv2.imwrite(str(maps / "plain.png"), np.zeros((100, 100, 3), np.uint8))
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    h = {"X-TWB": "1"}
    files = c.get("/api/map-files", params={"folder": str(maps)}, headers=h).json()["files"]
    assert [(f["name"], f["geotiff"]) for f in files] == [("ortho.tif", True), ("plain.png", False)]
    info = c.get("/api/map-info", params={"path": str(maps / "ortho.tif")}, headers=h).json()
    assert info["top"] == pytest.approx(32.0805) and info["right"] == pytest.approx(34.7875) and info["width"] == 800
    assert c.get("/api/map-info", params={"path": str(maps / "plain.png")}, headers=h).status_code == 400   # no corners
    ok = c.get("/api/map-info", params={"path": str(maps / "plain.png"), "corners": "32.1,34.7,32.0,34.8"}, headers=h)
    assert ok.status_code == 200 and ok.json()["bottom"] == pytest.approx(32.0)
    img = c.get("/api/map-image", params={"path": str(maps / "ortho.tif")}, headers=h)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    # a 320x180 camera on a 10 m pole: the points a person would pick, with the truth
    f = LocalFrame(32.0765, 34.7835)
    G = pose_homography(320, 180, 60, 0, 0, 10, 20, -30)
    px = [[20, 170], [300, 170], [160, 120], [40, 100], [280, 100], [160, 160]]
    en = apply_h(G, px)
    lat, lon = f.to_latlon(en[:, 0], en[:, 1])
    pts = [[x, y, float(a), float(o)] for (x, y), a, o in zip(px, lat, lon)]
    url = f"/api/missions/{m.id}/recordings/{vid}/geo"
    assert c.get(url, headers=h).json()["geo"] is None
    bad = [p[:] for p in pts]
    bad[2][2] += 10 / 111320                                          # one point clicked 10 m off
    chk = c.put(url, json={"points": bad, "check_only": True}, headers=h).json()["fit"]
    assert chk["suspect"] == 2
    assert c.get(url, headers=h).json()["geo"] is None                # a check saves nothing
    assert c.put(url, json={"points": pts[:3]}, headers=h).status_code == 400
    r = c.put(url, json={"points": pts, "map_path": str(maps / "ortho.tif")}, headers=h)
    assert r.status_code == 200, r.text
    g = r.json()["geo"]
    assert g["fit"]["rms_m"] < 0.05 and g["map_path"].endswith("ortho.tif")
    assert c.get(f"/api/missions/{m.id}/recordings/{other}/geo", headers=h).json()["geo"]["points"] == pts   # same camera
    paths = c.get(f"/api/missions/{m.id}/recordings/{vid}/map-paths", headers=h).json()["paths"]
    assert paths and all(len(p["path"]) >= 2 for p in paths)
    p0 = paths[0]["path"][0]
    assert abs(p0[1] - 32.0765) < 0.01 and abs(p0[2] - 34.7835) < 0.01
    assert all(p["length_m"] > 0 for p in paths)
    assert c.delete(url, headers=h).json()["geo"] is None
    assert c.get(url, headers=h).json()["geo"] is None and c.get(
        f"/api/missions/{m.id}/recordings/{vid}/map-paths", headers=h).json()["paths"] == []


def test_a_projected_geotiff_is_refused(tmp_path):
    import tifffile

    from trafficwb.video_geo import GeoError, load_map
    p = tmp_path / "utm.tif"
    tifffile.imwrite(str(p), np.zeros((10, 10, 3), np.uint8), extratags=[
        (33550, "d", 3, (0.5, 0.5, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, 500000.0, 3550000.0, 0.0)),
        (34735, "H", 12, (1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32636))])
    with pytest.raises(GeoError, match="projected"):
        load_map(str(p))
