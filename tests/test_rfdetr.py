"""RF-DETR as a detector choice: which model a mission uses, the picker's grades, and training it on the learning page."""
import json
import time
import zipfile

import cv2
import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from test_detectors import S, _pack  # noqa: E402
from onnx import TensorProto  # noqa: E402

from trafficwb import detectors, models  # noqa: E402
from trafficwb.detect import DetrDetector  # noqa: E402


def _rfdetr_pack(root, name, card=None):
    logits = np.full((1, 1, 91), -9.0, np.float32)
    logits[0, 0, 3] = 3.0                                        # one car
    return _pack(root, name, {"family": "detr", "outputs": "boxes_logits", "class_ids": "coco91",
                              "architecture": "rf-detr"} | (card or {}),
                 [("input", TensorProto.FLOAT, [1, 3, S, S])],
                 {"dets": np.array([[[0.5, 0.5, 0.2, 0.2]]], np.float32), "labels": logits})


def test_which_rfdetr_a_mission_uses(tmp_path):
    with pytest.raises(models.ModelPackError, match="not installed"):
        detectors.rfdetr_pack(models_root=tmp_path)
    assert detectors.rfdetr_info(models_root=tmp_path)["installed"] is False
    _rfdetr_pack(tmp_path, "rf-detr-nano")
    assert detectors.rfdetr_pack(models_root=tmp_path).name == "rf-detr-nano"
    _rfdetr_pack(tmp_path, "rf-detr-small")                     # Small is preferred when both are there
    assert detectors.rfdetr_info(models_root=tmp_path)["model"]["name"] == "rf-detr-small"
    assert detectors.is_rfdetr({"name": "custom", "architecture": "rf-detr"})
    assert detectors.is_rfdetr({"name": "rf-detr-small-tuned"}) and not detectors.is_rfdetr({"name": "yolox-s"})


def test_grades_cover_every_choice():
    g = detectors.grades()
    assert {s["key"] for s in g["scenes"]} == {"street", "drone", "thermal"}
    for k in ("standard", "rfdetr", "sam3", "both"):
        d = g["detectors"][k]
        assert set(d["ap50"]) == {"street", "drone", "thermal"} and set(d["seconds"]) == set(d["ap50"])
    # What the benchmarks showed: RF-DETR finds more vehicles than YOLOX-s everywhere, and is far faster than SAM 3.1.
    std, rf, sam = (g["detectors"][k] for k in ("standard", "rfdetr", "sam3"))
    assert all(rf["ap50"][s] > std["ap50"][s] for s in rf["ap50"])
    assert all(rf["seconds"][s] < sam["seconds"][s] / 5 for s in rf["seconds"])


def test_prepare_greys_unreviewed_areas_and_keeps_coco_ids(tmp_path):
    pytest.importorskip("torch")                                # training needs PyTorch (the "training" CI job)
    from trafficwb import rfdetr_train as rt
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    img = np.full((60, 80, 3), 200, np.uint8)
    cv2.imwrite(str(ds / "images" / "a.png"), img)
    cv2.imwrite(str(ds / "images" / "b.png"), img)
    images = [{"id": 1, "file_name": "images/a.png", "width": 80, "height": 60, "review": "partial"},
              {"id": 2, "file_name": "images/b.png", "width": 80, "height": 60, "review": "full",
               "exclude_regions": [[[0, 0], [20, 0], [20, 20], [0, 20]]]}]
    anns = [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [40, 30, 10, 10]},      # a truck
            {"id": 2, "image_id": 2, "category_id": 6, "bbox": [50, 30, 10, 10]}]      # a person
    (ds / "annotations_train.json").write_text(json.dumps({"images": images, "annotations": anns}))
    counts = rt.prepare(ds, tmp_path / "coco")
    assert counts["train"] == {"images": 2, "boxes": 2} and counts["val"]["images"] == 0
    a = json.loads((tmp_path / "coco" / "annotations" / "instances_train2017.json").read_text())
    assert sorted(x["category_id"] for x in a["annotations"]) == [1, 8]                # COCO person, truck
    assert (tmp_path / "coco" / "val2017").is_dir()                                     # validates on train if needed
    pa = cv2.imread(str(tmp_path / "coco" / "train2017" / "0000001.jpg"))
    assert abs(int(pa[5, 5, 0]) - rt.GREY) < 6 and abs(int(pa[35, 45, 0]) - 200) < 6   # only the box stays
    pb = cv2.imread(str(tmp_path / "coco" / "train2017" / "0000002.jpg"))
    assert abs(int(pb[5, 5, 0]) - rt.GREY) < 6 and abs(int(pb[50, 10, 0]) - 200) < 6   # excluded corner only


def test_mission_chooses_rfdetr(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission
    from trafficwb.web.app import create_app

    root = tmp_path / "models"
    monkeypatch.setenv("TWB_MODELS", str(root))
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    mid = mission(c, src, out)
    info = c.get("/api/detectors").json()
    assert info["rfdetr"]["installed"] is False and info["grades"]["detectors"]["rfdetr"]
    r = c.put(f"/api/missions/{mid}/detector", json={"detector": "rfdetr"}, headers=H1)
    assert r.status_code == 400 and r.json()["error"] == "rfdetr_missing"
    _rfdetr_pack(root, "rf-detr-small")
    assert c.get("/api/reference").json()["rfdetr"]["model"]["name"] == "rf-detr-small"
    assert c.put(f"/api/missions/{mid}/detector", json={"detector": "rfdetr"}, headers=H1).status_code == 200
    m = app.state.jobs  # the mission's detector is RF-DETR now; the standard one stays the fake
    from trafficwb.store import DataRoot
    det, card = m.detector_for(DataRoot(tmp_path / "data").open_mission(mid))
    assert isinstance(det, DetrDetector) and card["name"] == "rf-detr-small"
    found = det.detect(np.zeros((S, S, 3), np.uint8))
    assert [d.cls for d in found] == ["car"]
    from trafficwb.workers import remote_spec                     # other computers fetch the same pack
    assert remote_spec(m, DataRoot(tmp_path / "data").open_mission(mid))["standard"]["name"] == "rf-detr-small"


def test_train_rfdetr_on_the_learning_page(tmp_path, monkeypatch):
    """Train with arch rf-detr: RF-DETR's training runs (stand-in here), the pack is added and evaluated."""
    pytest.importorskip("torch")
    import os
    from datetime import datetime, timedelta, timezone

    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission, scene, wait_job
    from trafficwb import rfdetr_train, training_env
    from trafficwb.web.app import create_app

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    old = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    for k in range(6):
        t = (datetime(2026, 10, 10, 9, tzinfo=timezone.utc) + timedelta(days=k)).strftime("%Y%m%dT%H%M%SZ")
        p = src / f"CAM7_{t}.jpg"
        cv2.imwrite(str(p), scene(k), [cv2.IMWRITE_JPEG_QUALITY, 97])
        os.utime(p, (old, old))
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("TWB_MODELS", str(root))
    monkeypatch.setattr(training_env, "_state", {"state": "ready", "device_type": "cpu", "device": "CPU",
                                                 "coco_weights": False, "rfdetr": True, "rfdetr_reason": None,
                                                 "rfdetr_weights": ["small"], "suggested_epochs": 20})
    seen = {}

    def fake_train(dataset, out_dir, *, init, epochs, size, on_epoch=None, progress=None, **kw):
        seen.update(init=init, epochs=epochs, size=size, files=sorted(p.name for p in dataset.iterdir()))
        on_epoch(1, epochs, {"epoch": 1, "train_loss": 1.0, "val_loss": None, "val_ap50": 0.5})
        pack = _rfdetr_pack(tmp_path / "built", "rf-detr-small-tuned", {"torch_weights": None})
        z = out_dir / "p.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            for f in pack.path.iterdir():
                zf.write(f, f"rf-detr-small-tuned/1/{f.name}")
        return z

    monkeypatch.setattr(rfdetr_train, "train", fake_train)
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait_job(c, mid)
    for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]:
        c.post(f"/api/missions/{mid}/items/{r['source_version_id']}/decision/approve", headers=H1)
    ds = c.post(f"/api/missions/{mid}/learning/datasets", json={"name": "v1", "ratios": [0.5, 0.17, 0.33]},
                headers=H1).json()
    no = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "start_from": "coco",
                                                               "arch": "rf-detr", "size": "medium"}, headers=H1)
    assert no.status_code == 400 and no.json()["error"] == "no_weights"          # only Small is shipped here
    j = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "epochs": 3, "start_from": "coco",
                                                              "arch": "rf-detr"}, headers=H1).json()
    assert j["kind"] == "train"
    t0 = time.time()
    while time.time() - t0 < 60:
        job = c.get(f"/api/missions/{mid}/learning").json()["training"]["job"]
        if job["state"] not in ("running", "paused"):
            break
        time.sleep(0.2)
    assert job["state"] == "finished", job
    assert seen["init"] == "coco" and seen["epochs"] == 3 and seen["size"] == "small"
    assert "manifest.json" in seen["files"]
    assert job["result"]["model"]["name"] == "rf-detr-small-tuned" and "evaluation" in job["result"]
    saved = c.get("/api/models/saved").json()["saved"]
    assert saved == []                                                             # added, not saved or promoted


def test_rfdetr_together_with_sam(tmp_path, monkeypatch):
    """The picker's tick boxes: RF-DETR and SAM 3.1 ticked together run merged, as "rfdetr+sam3"."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission
    from trafficwb.combined import CombinedDetector
    from trafficwb.store import DataRoot
    from trafficwb.web.app import create_app

    assert detectors.uses_sam("rfdetr+sam3") and detectors.fast_of("rfdetr+sam3") == "rfdetr"
    assert detectors.uses_sam("both") and detectors.fast_of("both") == "standard"
    assert detectors.fast_of("sam3") is None and not detectors.uses_sam("rfdetr")
    root = tmp_path / "models"
    monkeypatch.setenv("TWB_MODELS", str(root))
    _rfdetr_pack(root, "rf-detr-small")
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    mid = mission(c, src, out)
    jobs = app.state.jobs
    eng = SimpleNamespace(pack=SimpleNamespace(card={"name": "sam3.1", "license": "SAM License"}))
    jobs.sam_factory = lambda: eng
    m = DataRoot(tmp_path / "data").open_mission(mid)
    m.update_config("operator", "test", detector="rfdetr+sam3")
    det, card = jobs.detector_for(m)
    assert isinstance(det, CombinedDetector) and isinstance(det.standard, DetrDetector)
    assert "sam" in card["license"].lower()


def test_train_dfine_and_rtdetr_on_the_learning_page(tmp_path, monkeypatch):
    """Train with arch d-fine (or rt-detr): detr_train runs (stand-in here) with that arch and size, and a trained one
    continues as its own kind."""
    pytest.importorskip("torch")
    import os
    from datetime import datetime, timedelta, timezone

    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission, scene, wait_job
    from trafficwb import detr_train, training_env
    from trafficwb.web.app import create_app

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    old = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    for k in range(6):
        t = (datetime(2026, 10, 10, 9, tzinfo=timezone.utc) + timedelta(days=k)).strftime("%Y%m%dT%H%M%SZ")
        p = src / f"CAM7_{t}.jpg"
        cv2.imwrite(str(p), scene(k), [cv2.IMWRITE_JPEG_QUALITY, 97])
        os.utime(p, (old, old))
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("TWB_MODELS", str(root))
    detr = {a: {"title": t, "ok": True, "reason": None, "sizes": s, "default": s[0], "weights": [s[0]]}
            for a, t, s in (("d-fine", "D-FINE", ["small", "medium"]), ("rt-detr", "RT-DETRv2", ["r18", "r50"]))}
    monkeypatch.setattr(training_env, "_state", {"state": "ready", "device_type": "cpu", "device": "CPU",
                                                 "coco_weights": False, "rfdetr": False, "rfdetr_weights": [],
                                                 "detr": detr, "suggested_epochs": 20})
    seen = []

    def fake_train(dataset, out_dir, *, init, epochs, arch, size=None, on_epoch=None, **kw):
        seen.append({"init": init, "arch": arch, "size": size})
        on_epoch(1, epochs, {"epoch": 1, "train_loss": 1.0, "val_loss": 0.5})
        name = f"{arch}-tuned"
        pack = _pack(tmp_path / f"built{len(seen)}", name,
                     {"family": "detr", "outputs": "boxes_logits", "class_ids": "coco80", "architecture": arch,
                      "size": size, "torch_weights": "model.safetensors"},
                     [("input", TensorProto.FLOAT, [1, 3, S, S])],
                     {"dets": np.array([[[0.5, 0.5, 0.2, 0.2]]], np.float32),
                      "labels": np.full((1, 1, 80), -9.0, np.float32)})
        (pack.path / "model.safetensors").write_bytes(b"weights")
        z = out_dir / "p.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            for f in pack.path.iterdir():
                zf.write(f, f"{name}/{len(seen)}/{f.name}")
        return z

    monkeypatch.setattr(detr_train, "train", fake_train)
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    mid = mission(c, src, out)
    c.post(f"/api/missions/{mid}/job", headers=H1)
    wait_job(c, mid)
    for r in c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]:
        c.post(f"/api/missions/{mid}/items/{r['source_version_id']}/decision/approve", headers=H1)
    ds = c.post(f"/api/missions/{mid}/learning/datasets", json={"name": "v1", "ratios": [0.5, 0.17, 0.33]},
                headers=H1).json()

    def run(body):
        j = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "epochs": 2} | body,
                   headers=H1)
        assert j.status_code == 200, j.text
        t0 = time.time()
        while time.time() - t0 < 60:
            job = c.get(f"/api/missions/{mid}/learning").json()["training"]["job"]
            if job["state"] not in ("running", "paused"):
                return job
            time.sleep(0.2)

    no = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "start_from": "coco",
                                                               "arch": "rt-detr", "size": "r50"}, headers=H1)
    assert no.status_code == 400 and no.json()["error"] == "no_weights"          # only r18 is shipped here
    bad = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "arch": "yolo-x9"}, headers=H1)
    assert bad.status_code == 400
    job = run({"start_from": "coco", "arch": "d-fine"})
    assert job["state"] == "finished", job
    assert seen[-1] == {"init": "coco", "arch": "d-fine", "size": "small"}
    m = job["result"]["model"]
    job = run({"start_from": f"{m['name']}/{m['version']}"})                    # continues as D-FINE, same size
    assert job["state"] == "finished", job
    assert seen[-1]["arch"] == "d-fine" and seen[-1]["size"] == "small" and seen[-1]["init"].endswith(m["version"])
