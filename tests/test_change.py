"""Vehicles found as changes against empty-road images of a camera view."""
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import sam3_fake  # noqa: E402

from trafficwb import models, sam3  # noqa: E402
from trafficwb.change import ChangeParams, build_reference, find_changes  # noqa: E402
from trafficwb.web.app import create_app  # noqa: E402

H1 = {"X-TWB": "1"}
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
W, H = 320, 240


def night_scene(seed=0, noise=3):
    """A dark road with two fixed street lamps, in grey (an infrared night camera)."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W), 18, np.float32)
    cv2.fillPoly(base, [np.array([[120, 60], [200, 60], [320, 240], [0, 240]], np.int32)], 32)
    tex = np.random.default_rng(99).integers(0, 30, (60, W)).astype(np.float32)   # same skyline in every frame
    base[:60] += tex
    for x in (60, 260):                                # lamp posts and lamps
        cv2.line(base, (x, 40), (x, 120), 120, 3)
        cv2.circle(base, (x, 40), 6, 255, -1)
    for k in range(6):                                 # lane lines, a bridge, sign gantries: the fixed scene
        cv2.line(base, (130 + 12 * k, 60), (20 + 56 * k, 240), 90, 2)
    cv2.rectangle(base, (0, 95), (319, 108), 110, -1)
    for x in range(0, W, 16):
        cv2.line(base, (x, 95), (x, 108), 40, 1)
    cv2.rectangle(base, (230, 62), (300, 80), 140, 2)
    img = base + rng.normal(0, noise, base.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def with_car_lights(img, x, y, glow=True):
    out = img.copy()
    cv2.circle(out, (x - 9, y), 5, 255, -1)          # two headlights
    cv2.circle(out, (x + 9, y), 5, 255, -1)
    if glow:                                           # the light they throw on the road ahead
        cv2.ellipse(out, (x, y + 34), (22, 9), 0, 0, 360, 95, -1)
    return out


def g(img):
    return cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0)


def test_one_car_with_lights_and_road_glow_is_one_change():
    bg, spread = build_reference([g(night_scene(s)) for s in range(4)])
    frame = with_car_lights(night_scene(10), 170, 150)
    ch, shift = find_changes(g(frame), bg, spread, None, ChangeParams())
    assert len(ch) == 1, ch
    c = ch[0]
    assert c.kind == "lights" and c.parts >= 2
    assert c.box[1] < 150 < c.box[3] and c.box[3] > 175     # the road glow is inside the vehicle's change
    assert abs(shift) < 3


def test_brightness_change_alone_and_lamps_are_not_changes():
    bg, spread = build_reference([g(night_scene(s)) for s in range(4)])
    brighter = np.clip(night_scene(11).astype(np.int16) + 20, 0, 255).astype(np.uint8)   # exposure, wet road
    ch, shift = find_changes(g(brighter), bg, spread, None)
    assert ch == [] and 15 < shift < 25


def test_changes_outside_the_scan_area_are_left_out():
    bg, spread = build_reference([g(night_scene(s)) for s in range(4)])
    frame = with_car_lights(night_scene(12), 170, 150, glow=False)
    area = np.zeros((H, W), bool)
    area[:, :100] = True
    assert find_changes(g(frame), bg, spread, area)[0] == []
    area[:] = True
    assert len(find_changes(g(frame), bg, spread, area)[0]) == 1


def test_two_far_apart_cars_stay_two():
    bg, spread = build_reference([g(night_scene(s)) for s in range(4)])
    frame = with_car_lights(with_car_lights(night_scene(13), 60, 200, glow=False), 250, 200, glow=False)
    assert len(find_changes(g(frame), bg, spread, None)[0]) == 2


def test_soft_light_alone_is_a_glow():
    bg, spread = build_reference([g(night_scene(s)) for s in range(4)])
    frame = night_scene(14)
    cv2.ellipse(frame, (160, 200), (25, 10), 0, 0, 360, 90, -1)
    ch = find_changes(g(frame), bg, spread, None)[0]
    assert [c.kind for c in ch] == ["glow"]


# ---------- in a mission, through the web API ----------

class NoDetections:
    def describe(self):
        return {"model": {"name": "fake", "version": "1", "sha256": "0" * 64}, "preprocessing": {}}

    def detect(self, img):
        return []


def wait(c, url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(url).json().get("job")
        if j and j["state"] not in ("running", "paused"):
            return j
        time.sleep(0.1)
    raise AssertionError("timed out")


def write(src, k, img):
    t = (datetime(2026, 10, 21, 20, tzinfo=timezone.utc) + timedelta(minutes=10 * k)).strftime("%Y%m%dT%H%M%SZ")
    p = src / f"CAM7_{t}.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    os.utime(p, (OLD, OLD))
    return p.name


@pytest.fixture
def mission(tmp_path, request):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    names = [write(src, k, night_scene(k)) for k in range(3)]                       # empty road
    names.append(write(src, 3, with_car_lights(night_scene(3), 170, 150)))           # one car, lights and glow
    names.append(write(src, 4, with_car_lights(night_scene(4), 60, 205, glow=False)))
    dark = night_scene(5)
    cv2.rectangle(dark, (150, 170), (190, 200), 0, -1)                               # a dark shape, no lights
    names.append(write(src, 5, dark))
    kw = {}
    if getattr(request, "param", None) == "sam":
        root = tmp_path / "packs"
        sam3_fake.build(root)
        eng = sam3.Sam3Engine(models.find(sam3.PACK_NAME, root=root))
        kw = {"sam_factory": lambda: eng}
    app = create_app(tmp_path / "data", detector_factory=lambda: (NoDetections(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0, **kw)
    c = TestClient(app)
    body = {"name": "Night road", "country": "IL", "iana_zone": "Asia/Jerusalem", "source_dir": str(src),
            "include_subdirs": False, "processing": "one_time", "output_dir": str(out), "export_formats": ["csv"]}
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    c.post(f"/api/missions/{mid}/job", headers=H1)
    assert wait(c, f"/api/missions/{mid}")["state"] == "finished"
    rows = c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]
    ids = {r["filename"]: r["source_version_id"] for r in rows}
    return c, mid, [ids[n] for n in names], rows


def test_empty_backgrounds_count_vehicles_from_their_lights(mission):
    c, mid, ids, rows = mission
    assert {r["condition"] for r in rows} == {"night"}
    views = c.get(f"/api/missions/{mid}/views").json()["views"]
    assert len(views) == 1, views
    vid = views[0]["id"]
    st = c.get(f"/api/missions/{mid}/views/{vid}/backgrounds").json()
    assert st["conditions"]["night"]["images"] == [] and st["suggestions"]
    assert c.post(f"/api/missions/{mid}/backgrounds/compare", headers=H1).status_code == 400   # nothing marked yet
    for v in ids[:3]:
        r = c.put(f"/api/missions/{mid}/items/{v}/empty", json={"empty": True}, headers=H1)
        assert r.status_code == 200, r.text
    assert c.get(f"/api/missions/{mid}/views/{vid}/backgrounds/night.jpg").status_code == 404   # built on compare
    c.post(f"/api/missions/{mid}/backgrounds/compare", headers=H1)
    job = wait(c, f"/api/missions/{mid}")
    assert job["state"] == "finished", job
    assert job["result"]["images_compared"] == 3, job
    st = c.get(f"/api/missions/{mid}/views/{vid}/backgrounds").json()["conditions"]["night"]
    assert st["images"] == ids[:3] and st["up_to_date"]
    assert [e["changes"] for e in st["empty_check"]] == [0, 0, 0]     # an empty image differs from no other
    bg = c.get(f"/api/missions/{mid}/views/{vid}/backgrounds/night.jpg")
    assert bg.status_code == 200 and bg.headers["content-type"] == "image/jpeg"

    counts = {r["source_version_id"]: r["vehicle_count"]
              for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]}
    assert [counts[v] for v in ids] == [0, 0, 0, 1, 1, 0], counts       # the dark shape at night is not counted
    item = c.post(f"/api/missions/{mid}/items/{ids[3]}", json={}, headers=H1).json()
    car = [o for o in item["objects"] if o["origin"] == "change"]
    assert len(car) == 1 and car[0]["cls"] == "vehicle" and car[0]["counted"]
    assert not any(cd["status"] == "unresolved" and cd["box"][0] > 140 and cd["box"][2] < 200 for cd in item["candidates"])
    assert c.post(f"/api/missions/{mid}/items/{ids[0]}", json={}, headers=H1).json()["empty_background"] is True

    # the operator corrects it like a detector box
    uid = car[0]["uid"]
    assert c.patch(f"/api/missions/{mid}/items/{ids[3]}/objects/{uid}", json={"action": "reclass", "cls": "car"}, headers=H1).status_code == 200
    item = c.post(f"/api/missions/{mid}/items/{ids[3]}", json={}, headers=H1).json()
    assert [o["cls"] for o in item["objects"] if o["uid"] == uid] == ["car"]

    # export: its own evidence level
    files = c.post(f"/api/missions/{mid}/export", json={"formats": ["csv"]}, headers=H1).json()["files"]
    objects = Path(next(f for f in files if f.endswith("_objects.csv"))).read_text(encoding="utf-8-sig")
    assert objects.count("vehicle_from_change") == 2

    # unmarking every empty image takes the vehicles away again
    for v in ids[:3]:
        c.put(f"/api/missions/{mid}/items/{v}/empty", json={"empty": False}, headers=H1)
    c.post(f"/api/missions/{mid}/backgrounds/compare", headers=H1)
    wait(c, f"/api/missions/{mid}")
    counts = [r["vehicle_count"] for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]]
    assert counts == [0] * 6


@pytest.mark.parametrize("mission", ["sam"], indirect=True)
def test_sam_looks_at_each_change(mission):
    c, mid, ids, rows = mission
    for v in ids[:3]:
        c.put(f"/api/missions/{mid}/items/{v}/empty", json={"empty": True}, headers=H1)
    c.post(f"/api/missions/{mid}/backgrounds/compare", headers=H1)
    assert wait(c, f"/api/missions/{mid}")["state"] == "finished"
    item = c.post(f"/api/missions/{mid}/items/{ids[3]}", json={}, headers=H1).json()
    change = [o for o in item["objects"] if o["origin"] == "change"]
    assert len(change) == 1
    assert change[0]["cls"] in ("vehicle", "car", "truck", "bus", "motorcycle", "bicycle")
    # the comparison is not repeated while nothing changed
    c.post(f"/api/missions/{mid}/backgrounds/compare", headers=H1)
    assert wait(c, f"/api/missions/{mid}")["result"]["images_compared"] == 0
