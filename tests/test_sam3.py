"""SAM 3.1: the engine around the model, scan areas from words, the detector, and the screens' API.

The real model is several hundred MB, so these tests use a tiny stand-in with the same graph inputs and
outputs (tests/sam3_fake.py) and run it through OpenVINO exactly as the real one is run.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("openvino")
pytest.importorskip("onnx")
pytest.importorskip("tokenizers")

sys.path.insert(0, str(Path(__file__).parent))
import sam3_fake  # noqa: E402

from trafficwb import models, sam3  # noqa: E402
from trafficwb.models import ModelPackError  # noqa: E402
from trafficwb.regions import Region, RegionSet, validate_polygon  # noqa: E402


@pytest.fixture(scope="module")
def pack_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    sam3_fake.build(root)
    return root


@pytest.fixture(scope="module")
def eng(pack_root):
    return sam3.Sam3Engine(models.find(sam3.PACK_NAME, root=pack_root))


def road_image(w=320, h=240, island=True):
    img = np.zeros((h, w, 3), np.uint8)
    img[h // 2:, :] = 255                                      # the bright lower half is the "road"
    if island:
        cv2.rectangle(img, (140, 160), (190, 200), (0, 0, 0), -1)   # a dark traffic island inside it
    return img


def test_segment_scores_and_scales_to_the_image(eng):
    found = eng.segment(road_image(), ["road", "nothing"])
    assert found["nothing"] == []                              # the presence of a thing decides, not just its shape
    (inst,) = found["road"]
    assert 0.9 < inst.score < 1
    assert inst.box == (80.0, 120.0, 240.0, 240.0)             # normalised xyxy scaled to 320 x 240
    m = inst.mask()
    assert m.shape == (240, 320) and m[210:, :].all() and not m[:100, :].any() and not m[175, 165]


def test_tokens_are_padded_and_cut_to_the_text_length(eng):
    eng.load()
    ids, mask = eng.tokens("road")
    assert ids.shape == (1, 32) and mask[0, :3].tolist() == [1, 1, 1] and mask[0, 3:].sum() == 0
    assert ids[0, 3:].tolist() == [eng.pad_id] * 29
    ids, mask = eng.tokens(" ".join(["road"] * 50))
    assert mask.sum() == 32 and ids[0, -1] == 1                # end-of-text kept last


def test_find_areas_outlines_the_road_and_excludes_its_gaps(eng):
    areas = sam3.find_areas(eng, road_image(), ["road"])
    inc = [a for a in areas if a.kind == "include"]
    exc = [a for a in areas if a.kind == "exclude"]
    assert len(inc) == 1 and len(exc) == 1
    assert 0.45 < inc[0].area_share < 0.55 and exc[0].prompt == "road (gap)"
    for a in areas:
        assert not validate_polygon(a.polygon, 320, 240)       # every outline is a valid region as it stands
    rs = RegionSet([Region("a", a.kind, a.prompt, a.polygon) for a in areas])
    assert rs.contains_point(20, 220) and not rs.contains_point(165, 180) and not rs.contains_point(20, 40)


def test_find_areas_for_an_exclusion_and_for_nothing(eng):
    areas = sam3.find_areas(eng, road_image(island=False), ["road", "nothing", "  "], kind="exclude")
    assert [a.kind for a in areas] == ["exclude"]
    with pytest.raises(ValueError):
        sam3.find_areas(eng, road_image(), ["road"], kind="maybe")


def test_mask_to_polygons_drops_specks_and_caps_corners():
    m = np.zeros((400, 400), bool)
    m[2:4, 2:4] = True                                         # a speck
    yy, xx = np.mgrid[:400, :400]
    m |= (xx - 200) ** 2 + (yy - 200) ** 2 < 150 ** 2          # a disc needs many corners
    polys = sam3.mask_to_polygons(m, max_points=24)
    assert len(polys) == 1 and 3 <= len(polys[0][0]) <= 24


def test_detector_reports_classes_and_skips_tiles_outside_the_scan_area(eng):
    det = sam3.Sam3Detector(eng, sam3.Sam3DetectorSettings(tiles=(2, 2), min_tile_side=100))
    img = road_image(640, 480)
    calls = []
    real = eng.encode_image
    eng.encode_image = lambda im: (calls.append(im.shape), real(im))[1]
    try:
        dets = det.detect(img)
        assert len(calls) == 5                                 # the whole frame and four tiles
        assert dets and set(dets[0].class_scores) == {"car", "truck", "bus", "motorcycle", "bicycle"}
        calls.clear()
        top_left = RegionSet([Region("r", "include", None, [(0, 0), (200, 0), (200, 150), (0, 150)])])
        det.detect(img, area=top_left)
        assert len(calls) == 2                                 # the whole frame and the one tile that holds the area
    finally:
        eng.encode_image = real
    d = det.describe()
    assert d["model"]["name"] == sam3.PACK_NAME and d["runtime"]["engine"] == "openvino"
    assert d["preprocessing"]["prompts"]["car"] == ["car", "van"]


def test_every_file_of_the_pack_is_checked(tmp_path):
    sam3_fake.build(tmp_path)
    d = tmp_path / sam3.PACK_NAME / "test"
    assert models.find(sam3.PACK_NAME, root=tmp_path).card["task"] == "segmentation"
    with open(d / "prompt-decoder.onnx", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ModelPackError, match="prompt-decoder.onnx"):
        models.find(sam3.PACK_NAME, root=tmp_path)
    (d / "prompt-decoder.onnx").unlink()
    with pytest.raises(ModelPackError, match="missing"):
        models.find(sam3.PACK_NAME, root=tmp_path)
    with pytest.raises(sam3.Sam3Unavailable, match="not installed"):
        sam3.find_pack(tmp_path / "none")
    st = sam3.available(tmp_path / "none")
    assert st["installed"] is False and "not installed" in st["reason"]


def test_openvino_never_reports_usage(tmp_path, pack_root):
    """Importing OpenVINO and running SAM 3.1 opens no network connection and writes no telemetry files."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import socket, sys\n"
        "def no(*a, **k):\n    print('CONNECT', a[0]); raise OSError('blocked')\n"
        "socket.create_connection = no\n"
        "_real = socket.socket.connect\n"
        "def no2(self, addr):\n    print('CONNECT', addr); raise OSError('blocked')\n"
        "socket.socket.connect = no2\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import numpy as np, time\n"
        "from pathlib import Path\n"
        "from trafficwb import sam3, models\n"
        f"e = sam3.Sam3Engine(models.find(sam3.PACK_NAME, root=Path({str(pack_root)!r})))\n"
        "img = np.zeros((64, 64, 3), np.uint8); img[32:] = 255\n"
        "print(len(e.segment(img, ['road'])['road']))\n"
        "time.sleep(1)\n", encoding="utf-8")
    env = {**os.environ, "HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path), "USERPROFILE": str(tmp_path)}
    env.pop("CI", None)
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "CONNECT" not in r.stdout and r.stdout.strip().endswith("1")
    assert not (tmp_path / "intel").exists()


# ---------- through the web API ----------

from fastapi.testclient import TestClient  # noqa: E402

from trafficwb.detect import Detection  # noqa: E402
from trafficwb.web.app import create_app  # noqa: E402

H1 = {"X-TWB": "1"}
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()


class StandardFake:
    def describe(self):
        return {"model": {"name": "fake", "version": "1", "sha256": "0" * 64}, "preprocessing": {}}

    def detect(self, img):
        return [Detection((10.0, 10.0, 40.0, 30.0), "car", 0.9, {"car": 0.9})]


@pytest.fixture
def api(tmp_path, eng):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    rng = np.random.default_rng(3)
    for k in range(4):
        img = road_image(320, 240)
        img[:100] = rng.integers(0, 90, (100, 320, 3), dtype=np.uint8)      # some texture so views can be told apart
        t = (datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(minutes=10 * k)).strftime("%Y%m%dT%H%M%SZ")
        p = src / f"CAM7_{t}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
        os.utime(p, (OLD, OLD))
    state = {"installed": True}
    app = create_app(tmp_path / "data", detector_factory=lambda: (StandardFake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0, sam_factory=lambda: eng,
                     sam_status=lambda: {"installed": state["installed"], "runtime": True, "loaded": eng.loaded,
                                         "pack": {"name": sam3.PACK_NAME}, "reason": None if state["installed"] else
                                         "The SAM 3.1 model pack is not installed."})
    c = TestClient(app)
    body = {"name": "Ring road", "country": "IL", "iana_zone": "Asia/Jerusalem", "source_dir": str(src),
            "include_subdirs": False, "processing": "one_time", "output_dir": str(out), "export_formats": ["csv"]}
    return c, body, state


def wait(c, url, key="state", timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(url).json()
        j = r if key == "state" else r.get(key)
        if j and j["state"] not in ("running", "paused"):
            return j
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_mission_with_sam3_detector(api):
    c, body, state = api
    assert c.get("/api/reference").json()["sam3"]["installed"] is True
    s = c.get("/api/sam3").json()
    assert "road" in s["suggestions"]["include"] and s["vehicle_prompts"]["car"] == ["car", "van"]
    bad = c.post("/api/missions/validate", json={**body, "detector": "magic"}, headers=H1).json()
    assert any("Detector" in e for e in bad["errors"])
    vid = c.post("/api/missions/validate", json={**body, "input_type": "video", "mode": "object_search",
                                                 "detector": "sam3"}, headers=H1).json()
    assert any("still images" in e for e in vid["errors"])
    mid = c.post("/api/missions", json={**body, "detector": "sam3"}, headers=H1).json()["id"]
    m = c.get(f"/api/missions/{mid}").json()
    assert m["config"]["detector"] == "sam3"
    c.post(f"/api/missions/{mid}/job", headers=H1)
    job = wait(c, f"/api/missions/{mid}", key="job")
    assert job["state"] == "finished", job
    rows = c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()
    assert rows["total"] == 4
    # switch back to the standard detector; refused while a job runs, and unknown names are refused
    assert c.put(f"/api/missions/{mid}/detector", json={"detector": "nope"}, headers=H1).status_code == 400
    assert c.put(f"/api/missions/{mid}/detector", json={"detector": "standard"}, headers=H1).json()["detector"] == "standard"
    assert c.get(f"/api/missions/{mid}").json()["config"]["detector"] == "standard"
    state["installed"] = False
    r = c.put(f"/api/missions/{mid}/detector", json={"detector": "sam3"}, headers=H1)
    assert r.status_code == 400 and r.json()["error"] == "sam3_missing"
    r = c.post("/api/missions", json={**body, "name": "Other", "detector": "sam3"}, headers=H1)
    assert r.status_code == 400 and "not installed" in r.json()["message"]


def test_find_areas_then_save_them_as_regions(api):
    c, body, state = api
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait(c, f"/api/missions/{mid}", key="job")
    view = c.get(f"/api/missions/{mid}/views").json()["views"][0]
    vid = view["id"]
    reg = c.get(f"/api/missions/{mid}/views/{vid}/regions").json()
    image_id = next(iter(reg["representatives"].values()))[0]
    url = f"/api/missions/{mid}/views/{vid}/find-areas"
    assert c.post(url, json={"prompts": ["  "], "image_id": image_id}, headers=H1).status_code == 400
    assert c.post(url, json={"prompts": ["road"], "image_id": 999999}, headers=H1).status_code == 404
    assert c.post(f"/api/missions/{mid}/views/999/find-areas", json={"prompts": ["road"], "image_id": image_id},
                  headers=H1).status_code == 404
    t = c.post(url, json={"prompts": ["road"], "image_id": image_id}, headers=H1).json()
    done = wait(c, f"/api/missions/{mid}/find-areas/{t['id']}")
    assert done["state"] == "finished", done
    assert done["size"] == [320, 240]
    kinds = sorted(p["kind"] for p in done["proposals"])
    assert kinds == ["exclude", "include"]
    assert c.get(f"/api/missions/{mid}/find-areas/nope").status_code == 404
    regions = [{"kind": p["kind"], "polygon": p["polygon"], "name": f"{p['prompt']} (SAM 3.1)"} for p in done["proposals"]]
    bad = c.post(f"/api/missions/{mid}/views/{vid}/regions/batch", headers=H1,
                 json={"regions": regions + [{"kind": "include", "polygon": [[0, 0], [1, 0]]}]})
    assert bad.status_code == 400
    assert c.get(f"/api/missions/{mid}/views/{vid}/regions").json()["regions"] == []   # all or nothing
    ok = c.post(f"/api/missions/{mid}/views/{vid}/regions/batch", headers=H1, json={"regions": regions})
    assert ok.status_code == 200 and len(ok.json()["uids"]) == 2
    saved = c.get(f"/api/missions/{mid}/views/{vid}/regions").json()["regions"]
    assert sorted(r["kind"] for r in saved) == ["exclude", "include"]
    state["installed"] = False
    r = c.post(url, json={"prompts": ["road"], "image_id": image_id}, headers=H1)
    assert r.status_code == 400 and r.json()["error"] == "sam3_missing"
