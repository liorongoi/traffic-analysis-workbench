"""Training that belongs to no mission: dataset versions of any missions, evaluated and put to use where chosen."""
import os
import time
from datetime import datetime, timedelta, timezone

import cv2
from fastapi.testclient import TestClient

from test_web_api import H1, Fake, direct_access, scene
from trafficwb.web.app import create_app


def _mission(c, tmp, name, days):
    src, out = tmp / f"src-{name}", tmp / f"out-{name}"
    src.mkdir()
    out.mkdir()
    old = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    for k in range(days):
        t = (datetime(2026, 10, 10, 9, tzinfo=timezone.utc) + timedelta(days=k)).strftime("%Y%m%dT%H%M%SZ")
        p = src / f"CAM7_{t}.jpg"
        cv2.imwrite(str(p), scene(k))
        os.utime(p, (old, old))
    body = {"name": name, "country": "IL", "iana_zone": "Asia/Jerusalem", "source_dir": str(src), "include_subdirs": False,
            "processing": "one_time", "output_dir": str(out), "export_formats": ["csv"]}
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    c.post(f"/api/missions/{mid}/job", headers=H1)
    for _ in range(300):
        j = c.get(f"/api/missions/{mid}").json()["job"]
        if j and j["state"] in ("finished", "failed", "cancelled"):
            break
        time.sleep(0.1)
    for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]:
        c.post(f"/api/missions/{mid}/items/{r['source_version_id']}/decision/approve", headers=H1)
    ds = c.post(f"/api/missions/{mid}/learning/datasets", json={"name": f"{name} v1", "ratios": [0.5, 0.17, 0.33]},
                headers=H1).json()
    return mid, ds["id"]


def _wait(c, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get("/api/missions/training/learning").json()["job"]
        if j and j["state"] not in ("running", "paused"):
            return j
        time.sleep(0.1)
    raise AssertionError("workspace job did not end")


def test_training_page_uses_any_missions_and_belongs_to_none(tmp_path):
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    a, da = _mission(c, tmp_path, "Route 6", 6)
    b, db = _mission(c, tmp_path, "Junction", 6)

    s = c.get("/api/missions/training/learning").json()
    assert s["workspace"] is True and s["datasets"] == [] and s["job"] is None
    assert {(d["mission_id"], d["input_type"]) for d in s["all_datasets"]} == {(a, "images"), (b, "images")}
    assert {m["id"] for m in s["missions"]} == {a, b}
    assert all(r["id"] != "training" for r in c.get("/api/missions").json()["missions"])   # never a mission
    assert c.get("/api/missions/training").status_code == 404                               # nor its other pages
    assert c.post("/api/missions/training/job", headers=H1).status_code == 404

    refs = [{"mission_id": a, "dataset_id": da}, {"mission_id": b, "dataset_id": db}]
    j = c.post("/api/missions/training/learning/evaluate", json={"datasets": refs, "split": "test"}, headers=H1).json()
    assert j["kind"] == "evaluate"
    assert _wait(c)["state"] == "finished"
    s = c.get("/api/missions/training/learning").json()
    ev = s["evaluations"][0]
    assert ev["state"] == "completed" and {x["mission_id"] for x in ev["sources"]} == {a, b}
    # kept on the Training page, not in either mission
    assert c.get(f"/api/missions/{a}/learning").json()["evaluations"] == []

    # Pause, continue and stop work here too.
    c.post("/api/missions/training/learning/evaluate", json={"datasets": refs, "split": "train"}, headers=H1)
    r = c.post("/api/missions/training/job/cancel", headers=H1)
    assert r.status_code in (200, 409)
    assert _wait(c)["state"] in ("cancelled", "finished")


def test_use_a_saved_model_in_chosen_missions(tmp_path, monkeypatch):
    from trafficwb import learning
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    a, _ = _mission(c, tmp_path, "Route 6", 3)
    b, _ = _mission(c, tmp_path, "Junction", 3)
    assert c.post("/api/models/saved/use", json={"saved": "nope", "missions": [a]}, headers=H1).status_code == 400
    monkeypatch.setattr(learning.ModelLibrary, "get", lambda self, name: {"name": name} if name == "Night cars" else None)
    r = c.post("/api/models/saved/use", json={"saved": "Night cars", "missions": [a, "gone"]}, headers=H1).json()
    assert r["updated"] == ["Route 6"] and r["missing"] == ["gone"] and r["busy"] == []
    assert c.get(f"/api/missions/{a}").json()["config"]["vehicle_model"] == "Night cars"
    assert c.get(f"/api/missions/{b}").json()["config"]["vehicle_model"] is None          # not ticked: unchanged
