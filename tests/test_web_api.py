import os
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from trafficwb.detect import Detection
from trafficwb.web.app import create_app

W, H = 640, 360
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
H1 = {"X-TWB": "1"}


def scene(k):
    rng = np.random.default_rng(1)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(60):
        p1 = tuple(int(v) for v in rng.integers(0, [W, H]))
        p2 = tuple(int(v) for v in rng.integers(0, [W, H]))
        cv2.line(img, p1, p2, tuple(int(c) for c in rng.integers(0, 200, 3)), 2)
    img = cv2.add(img, np.random.default_rng(k).integers(0, 6, img.shape, dtype=np.uint8))
    cv2.rectangle(img, (100, 250), (130, 270), (0, 0, 255), -1)
    cv2.rectangle(img, (400, 100), (430, 120), (0, 0, 255), -1)
    return img


class Fake:
    def __init__(self, delay=0.0):
        self.delay = delay

    def describe(self):
        return {"model": {"name": "fake", "version": "1", "sha256": "0" * 64}, "preprocessing": {}}

    def detect(self, img):
        time.sleep(self.delay)
        mask = np.all(np.abs(img.astype(int) - (0, 0, 255)) <= 8, axis=2).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "car", 0.9, {"car": 0.9})
                for x, y, w, h, a in st[1:] if a > 30]


def ts(i):
    return (datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(minutes=10 * i)).strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture
def env(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for k in range(6):
        p = src / f"CAM7_{ts(k)}.jpg"
        cv2.imwrite(str(p), scene(k), [cv2.IMWRITE_JPEG_QUALITY, 97])
        os.utime(p, (OLD, OLD))
    (src / "notes.jpg").write_bytes(b"x")
    return tmp_path, src, out


def client(tmp_path, delay=0.0):
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(delay), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    return TestClient(app)


def local_operator(c):
    assert c.get("/api/session").json()["user"]["username"]
    return c


def mission(c, src, out, processing="one_time"):
    body = {"name": "Route 6", "country": "IL", "iana_zone": "Asia/Jerusalem", "source_dir": str(src),
            "include_subdirs": False, "processing": processing, "output_dir": str(out), "export_formats": ["csv"]}
    v = c.post("/api/missions/validate", json=body, headers=H1).json()
    assert v["errors"] == []
    return c.post("/api/missions", json=body, headers=H1).json()["id"]


def wait_job(c, mid, timeout=30):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get(f"/api/missions/{mid}").json()["job"]
        if j and j["state"] in ("finished", "cancelled", "failed"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_no_sign_in_required_but_mutations_need_application_header(env):
    tmp, src, out = env
    c = client(tmp)
    assert c.get("/api/missions").status_code == 200
    assert c.post("/api/login", json={"username": "x", "password": "y"}, headers=H1).status_code == 404
    assert c.get("/api/missions").status_code == 200
    assert c.post("/api/missions/validate", json={}, headers={}).status_code == 403


def test_offline_timezone_preview_uses_capture_date_rules(env):
    tmp, _, _ = env
    c = client(tmp)
    winter = c.post("/api/time-preview", json={"iana_zone": "Europe/Riga", "utc": "2026-01-01T12:00:00Z"}, headers=H1).json()
    summer = c.post("/api/time-preview", json={"iana_zone": "Europe/Riga", "utc": "2026-07-01T12:00:00Z"}, headers=H1).json()
    assert winter["offset"] == "UTC+02:00" and winter["is_dst"] is False
    assert summer["offset"] == "UTC+03:00" and summer["is_dst"] is True
    assert c.post("/api/time-preview", json={"iana_zone": "No/Such_Zone", "utc": "2026-07-01T12:00:00Z"},
                  headers=H1).status_code == 400


def test_wizard_preview_create_process_review_export(env):
    tmp, src, out = env
    c = local_operator(client(tmp))
    prev = c.post("/api/source-preview", json={"source_dir": str(src), "include_subdirs": False}, headers=H1).json()
    ok = [f for f in prev["files"] if f["ok"]]
    bad = [f for f in prev["files"] if not f["ok"]]
    assert len(ok) == 6 and ok[0]["precision"] == "second" and bad[0]["file"] == "notes.jpg"
    ref = c.get("/api/reference").json()
    assert ref["country_zones"]["IL"] == "Asia/Jerusalem" and "Asia/Tehran" in ref["zones"]
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    job = wait_job(c, mid)
    assert job["state"] == "finished" and job["done"] == 6
    res = c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()
    assert res["total"] == 6 and all(r["vehicle_count"] == 2 for r in res["rows"])
    vid = res["rows"][0]["source_version_id"]
    item = c.post(f"/api/missions/{mid}/items/{vid}", json={}, headers=H1).json()
    assert item["position"] == {"index": 1, "total": 6, "prev": None, "next": res["rows"][1]["source_version_id"]}
    assert item["local"]["time"].endswith("+03:00") and len(item["objects"]) == 2
    uid = item["objects"][0]["uid"]
    assert c.patch(f"/api/missions/{mid}/items/{vid}/objects/{uid}", json={"action": "remove"}, headers=H1).status_code == 200
    r = c.post(f"/api/missions/{mid}/items/{vid}/objects", json={"box": [500, 300, 540, 330], "cls": "truck"}, headers=H1)
    assert r.status_code == 200
    bad = c.post(f"/api/missions/{mid}/items/{vid}/objects", json={"box": [600, 300, 900, 330], "cls": "car"}, headers=H1)
    assert bad.status_code == 400 and "outside" in bad.json()["message"]
    assert c.post(f"/api/missions/{mid}/items/{vid}/decision/approve", headers=H1).json()["review_status"] == "approved"
    p = c.post(f"/api/missions/{mid}/approve-all/preview", json={}, headers=H1).json()
    assert p["count"] == 5
    assert c.post(f"/api/missions/{mid}/approve-all/commit", json={"token": p["token"]}, headers=H1).status_code == 200
    assert c.post(f"/api/missions/{mid}/approve-all/commit", json={"token": p["token"]}, headers=H1).status_code == 409
    statuses = {r["review_status"] for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]}
    assert statuses == {"approved", "bulk_approved_unreviewed"}
    img = c.get(f"/api/missions/{mid}/images/{vid}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    th = c.get(f"/api/missions/{mid}/thumbs/{vid}?w=160")
    assert th.status_code == 200 and cv2.imdecode(np.frombuffer(th.content, np.uint8), 1).shape[1] == 160
    ex = c.post(f"/api/missions/{mid}/export", json={}, headers=H1).json()
    assert len(ex["files"]) == 2 and all(os.path.isfile(f) for f in ex["files"])


def test_job_pause_resume_cancel(env):
    tmp, src, out = env
    c = local_operator(client(tmp, delay=0.3))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    time.sleep(0.8)
    assert c.post(f"/api/missions/{mid}/job/pause", headers=H1).json()["state"] == "paused"
    done = c.get(f"/api/missions/{mid}").json()["job"]["done"]
    time.sleep(0.8)
    assert c.get(f"/api/missions/{mid}").json()["job"]["done"] <= done + 1    # at most the image in progress finishes
    c.post(f"/api/missions/{mid}/job/resume", headers=H1)
    c.post(f"/api/missions/{mid}/job/cancel", headers=H1)
    j = wait_job(c, mid)
    assert j["state"] == "cancelled"
    m = c.get(f"/api/missions/{mid}").json()
    assert m["runs"][0]["state"] == "cancelled"
    assert m["processing_counts"].get("cancelled", 0) >= 1


def test_views_and_regions(env):
    tmp, src, out = env
    c = local_operator(client(tmp))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait_job(c, mid)
    views = c.get(f"/api/missions/{mid}/views").json()["views"]
    assert len(views) == 1
    vid = views[0]["id"]
    assert c.post(f"/api/missions/{mid}/views/{vid}/actions/rename", json={"name": "North gantry"}, headers=H1).status_code == 200
    r = c.post(f"/api/missions/{mid}/views/{vid}/regions", headers=H1,
               json={"kind": "exclude", "polygon": [[380, 80], [460, 80], [460, 140], [380, 140]], "name": "parking"})
    assert r.status_code == 200
    reg = c.get(f"/api/missions/{mid}/views/{vid}/regions").json()
    assert reg["width"] == W and reg["regions"][0]["name"] == "parking"
    bad = c.post(f"/api/missions/{mid}/views/{vid}/regions", headers=H1,
                 json={"kind": "include", "polygon": [[0, 0], [10, 0]]})
    assert bad.status_code == 400
    assert c.delete(f"/api/missions/{mid}/views/{vid}/regions/{r.json()['uid']}", headers=H1).status_code == 200
    assert c.get(f"/api/missions/{mid}/views/{vid}/regions").json()["regions"] == []


class Named(Fake):
    def __init__(self, name, version):
        super().__init__()
        self.n, self.v = name, version

    def describe(self):
        return {"model": {"name": self.n, "version": self.v, "sha256": "1" * 64}, "preprocessing": {}}


def test_learning_dataset_evaluate_promote_rollback(env, monkeypatch):
    import json
    from trafficwb.models import sha256_file
    tmp, src, out = env
    models = tmp / "models"
    d = models / "det" / "2"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"w2")
    (d / "model.json").write_text(json.dumps({"name": "det", "version": "2", "task": "vehicle_detection",
                                              "license": "MIT", "sha256": sha256_file(d / "model.onnx")}))
    monkeypatch.setenv("TWB_MODELS", str(models))
    app = create_app(tmp / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0, model_loader=lambda n, v: Named(n, v))
    c = local_operator(TestClient(app))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait_job(c, mid)
    s = c.get(f"/api/missions/{mid}/learning").json()
    assert s["eligibility"]["eligible"] == 0 and s["production"] is None and s["models"][0]["name"] == "det"
    for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]:
        c.post(f"/api/missions/{mid}/items/{r['source_version_id']}/decision/approve", headers=H1)
    ds = c.post(f"/api/missions/{mid}/learning/datasets", json={"name": "v1", "ratios": [0, 0, 1]}, headers=H1).json()
    assert ds["stats"]["images"] == 6 and ds["leakage"] == []
    bad = c.post(f"/api/missions/{mid}/learning/datasets", json={"name": "x", "ratios": [1, 1, 1]}, headers=H1)
    assert bad.status_code == 400
    ex = c.post(f"/api/missions/{mid}/learning/datasets/{ds['id']}/export", headers=H1).json()
    assert os.path.isdir(ex["folder"])
    c.post(f"/api/missions/{mid}/learning/evaluate", json={"dataset_id": ds["id"], "name": "det", "version": "2"}, headers=H1)
    job = wait_job(c, mid)
    assert job["kind"] == "evaluate" and job["state"] == "finished", job
    ev = c.get(f"/api/missions/{mid}/learning/evaluations/{job['result']['evaluation']}").json()
    assert ev["report"]["overall"]["recall"] == 1.0 and ev["model"]["model"]["name"] == "det"
    # The evaluation must be of the model being promoted.
    wrong = c.post("/api/models/promote", json={"name": "det", "version": "3", "mission_id": mid,
                                                "evaluation_id": ev["id"]}, headers=H1)
    assert wrong.status_code == 400
    p = c.post("/api/models/promote", json={"name": "det", "version": "2", "mission_id": mid, "evaluation_id": ev["id"]},
               headers=H1).json()
    assert p["production"]["version"] == "2"
    assert c.get(f"/api/missions/{mid}/learning").json()["production"]["name"] == "det"
    assert c.post("/api/models/rollback", json={}, headers=H1).json()["production"] is None
def test_system_backup_and_delete_mission(env):
    tmp, src, out = env
    app = create_app(tmp / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0, backup_dir=tmp / "bk")
    c = local_operator(TestClient(app))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait_job(c, mid)
    s = c.get("/api/system").json()
    assert s["backup"]["folder"] == str(tmp / "bk") and s["backup"]["last"] is None
    b = c.post("/api/admin/backup", headers=H1).json()
    assert os.path.isfile(os.path.join(b["folder"], "manifest.json"))
    assert c.get("/api/system").json()["backup"]["last"]["detail"]["folder"] == b["folder"]
    bad = c.post(f"/api/missions/{mid}/delete", json={"confirm_name": "route 6"}, headers=H1)
    assert bad.status_code == 400
    assert c.post(f"/api/missions/{mid}/delete", json={"confirm_name": "Route 6"}, headers=H1).status_code == 200
    assert c.get(f"/api/missions/{mid}").status_code == 404
    assert c.get("/api/system").json()["audit"][0]["action"] == "mission_deleted"
    assert len(list(src.glob("*.jpg"))) == 7
