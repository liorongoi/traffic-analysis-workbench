"""Training the vehicle detector (needs PyTorch; skipped where it is not installed) and importing packs."""
import json
import zipfile

import numpy as np
import pytest

from trafficwb import models
from trafficwb.model_import import import_pack
from trafficwb.models import ModelPackError

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from trafficwb import train as tr  # noqa: E402
from trafficwb.yolox_torch import YOLOX, yolox_loss  # noqa: E402


def _dataset(tmp_path, n=4):
    """A tiny exported dataset: grey frames with dark rectangles as cars."""
    import cv2
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    files = []
    for split, count in (("train", n), ("val", 2), ("test", 1)):
        images, anns = [], []
        for i in range(count):
            vid = len(files) * 100 + i + 1
            im = np.full((360, 640, 3), 150, np.uint8)
            boxes = [[60 + 40 * i, 100, 140 + 40 * i, 160], [400, 200, 520, 280]]
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(im, (x1, y1), (x2, y2), (40, 40, 40), -1)
            name = f"{vid}.png"
            cv2.imwrite(str(ds / "images" / name), im)
            images.append({"id": vid, "file_name": f"images/{name}", "width": 640, "height": 360,
                           "review": "full" if i % 2 == 0 else "partial", "include_regions": [],
                           "exclude_regions": [[[0, 0], [100, 0], [100, 50], [0, 50]]], "ignore_boxes": [[300, 20, 340, 60]]})
            for k, (x1, y1, x2, y2) in enumerate(boxes):
                anns.append({"id": vid * 1000 + k, "image_id": vid, "category_id": 2 if k == 0 else 5,
                             "bbox": [x1, y1, x2 - x1, y2 - y1]})
        (ds / f"annotations_{split}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
        files.append(f"annotations_{split}.json")
    (ds / "manifest.json").write_text(json.dumps({"mission_id": "m", "dataset_id": 1, "name": "tiny", "stats": {},
                                                  "exported_at": "2026-10-27T00:00:00Z", "files": files}))
    return ds


def test_background_only_where_someone_looked():
    full = {"review": "full", "include_regions": [[[0, 0], [50, 0], [50, 50], [0, 50]]],
            "exclude_regions": [[[0, 0], [10, 0], [10, 10], [0, 10]]], "ignore_boxes": [[20, 20, 30, 30]]}
    m = tr.background_mask(full, 100, 100)
    assert m[40, 40] == 1           # inside the included region
    assert m[80, 80] == 0           # outside it
    assert m[5, 5] == 0             # excluded
    assert m[25, 25] == 0           # an unresolved candidate
    assert tr.background_mask({"review": "partial"}, 100, 100).max() == 0


def test_letterbox_matches_the_app_and_maps_classes(tmp_path):
    ds = tr.ExportedDataset(_dataset(tmp_path), "val", augment=False)
    x, mask, t = ds[0]
    assert x.shape == (3, 640, 640) and float(x[0, 639, 0]) == 114       # top-left, pad 114 below
    assert sorted(t[:, 0].tolist()) == [2, 7]                               # car -> COCO 2, truck -> COCO 7
    cx, cy, w, h = t[t[:, 0] == 2][0, 1:].tolist()
    assert (cx, cy, w, h) == pytest.approx((100, 130, 80, 60))             # scale 1 for a 640-wide frame
    assert mask[30, 30] == 0 and mask[300, 600] == 1


def test_cut_boxes_become_unknown_not_background():
    im = np.zeros((100, 100, 3), np.uint8)
    mask = np.ones((100, 100), np.uint8)
    tr.random.seed(1)
    for _ in range(20):
        _, m, t = tr.place(im, mask, np.array([[2, 0, 0, 100, 100]], np.float32), augment=True)
        assert len(t) == 1 or m.sum() < 640 * 640                            # dropped box -> its area is unknown


def test_unreviewed_areas_never_teach_background():
    torch.manual_seed(0)
    m = YOLOX("nano", 80).eval()
    raw, grid, stride = m(torch.rand(1, 3, 128, 128) * 255)
    none = [torch.zeros((0, 5))]
    zero, _ = yolox_loss(raw, grid, stride, none, torch.zeros(raw.shape[:2]), 80)
    some, _ = yolox_loss(raw, grid, stride, none, torch.ones(raw.shape[:2]), 80)
    assert float(zero) == 0 and float(some) > 0


def test_train_export_import_and_continue(tmp_path):
    ds = _dataset(tmp_path)
    z = tr.train(ds, tmp_path / "out", init="none", epochs=2, batch=2, workers=0, device="cpu", size="nano",
                 version="t1", log=lambda *_: None)
    assert z.name == "yolox-s-tuned-t1.zip"
    with zipfile.ZipFile(z) as f:
        assert sorted(f.namelist()) == [f"yolox-s-tuned/t1/{n}" for n in ("LICENSE", "model.json", "model.onnx", "model.pt")]
    card = json.loads((tmp_path / "out/yolox-s-tuned/t1/model.json").read_text())
    assert card["task"] == "vehicle_detection" and card["classes"] == "coco80" and card["input_size"] == 640
    assert card["training"]["dataset"]["name"] == "tiny" and card["training"]["export_check_max_difference"] < 1e-2

    root = tmp_path / "models"
    r = import_pack(z, root)
    assert not r["already"] and (root / "yolox-s-tuned/t1/model.onnx").is_file()
    assert import_pack(z, root)["already"]
    pack = models.find("yolox-s-tuned", "t1", root)

    from trafficwb.detect import YoloxDetector                               # the app runs it unchanged
    dets = YoloxDetector(pack).detect(np.full((360, 640, 3), 150, np.uint8))
    assert isinstance(dets, list)

    # continue training from the imported pack
    z2 = tr.train(ds, tmp_path / "out", init=str(z), epochs=1, batch=2, workers=0, device="cpu", size="nano",
                  version="t2", log=lambda *_: None)
    card2 = json.loads((tmp_path / "out/yolox-s-tuned/t2/model.json").read_text())
    assert card2["training"]["init"] == {"name": "yolox-s-tuned", "version": "t1", "sha256": card["sha256"]}
    assert z2.is_file()


def test_import_refuses_damaged_or_conflicting_packs(tmp_path):
    ds = _dataset(tmp_path, n=2)
    z = tr.train(ds, tmp_path / "out", init="none", epochs=1, batch=2, workers=0, device="cpu", size="nano",
                 version="t1", log=lambda *_: None)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            dst.writestr(n, data[:-10] + b"0" * 10 if n.endswith("model.pt") else data)
    root = tmp_path / "models"
    with pytest.raises(ModelPackError, match="damaged"):
        import_pack(bad, root)
    assert not (root / "yolox-s-tuned").exists()
    import_pack(z, root)
    other = tmp_path / "out2"
    z2 = tr.train(ds, other, init="none", epochs=1, batch=2, workers=0, device="cpu", size="nano", version="t1",
                  seed=5, log=lambda *_: None)
    with pytest.raises(ModelPackError, match="different weights"):
        import_pack(z2, root)


def test_train_on_the_learning_page(tmp_path, monkeypatch):
    """Train button: export, train on this computer, add the new model, evaluate it; nothing is promoted."""
    import os
    import time
    from datetime import datetime, timedelta, timezone

    import cv2
    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission, scene, wait_job
    from trafficwb import training_env
    from trafficwb.web.app import create_app

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    old = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    for k in range(6):                               # one image a day: six groups, so every split gets some
        t = (datetime(2026, 10, 10, 9, tzinfo=timezone.utc) + timedelta(days=k)).strftime("%Y%m%dT%H%M%SZ")
        p = src / f"CAM7_{t}.jpg"
        cv2.imwrite(str(p), scene(k), [cv2.IMWRITE_JPEG_QUALITY, 97])
        os.utime(p, (old, old))
    # Stand-in starting weights (random YOLOX-s), installed next to Python as the installers do.
    share = tmp_path / "share"
    share.mkdir()
    torch.manual_seed(0)
    torch.save({"model": YOLOX("s", 80).state_dict()}, share / "yolox_s.pth")
    monkeypatch.setenv("LANEPULSE_CACHE", str(share))
    monkeypatch.setitem(tr.COCO_WEIGHTS, "sha256", tr._sha256(share / "yolox_s.pth"))
    models_root = tmp_path / "models"
    models_root.mkdir()
    monkeypatch.setenv("TWB_MODELS", str(models_root))
    training_env.reset()

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
    sp = ds["stats"]["splits"]
    assert sp["train"]["images"] and sp["test"]["images"], sp

    for _ in range(300):                             # PyTorch is checked in the background
        s = c.get(f"/api/missions/{mid}/learning").json()["training"]
        if s["state"] != "checking":
            break
        time.sleep(0.1)
    assert s["state"] == "ready" and s["coco_weights"] and s["device_type"] in ("cpu", "cuda", "mps"), s

    bad = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "start_from": "nope/1"}, headers=H1)
    assert bad.status_code == 400
    j = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "epochs": 1, "start_from": "coco"},
               headers=H1).json()
    assert j["kind"] == "train"
    # Training has its own slot: the mission can still process meanwhile.
    assert c.get(f"/api/missions/{mid}").json()["job"]["kind"] != "train"
    again = c.post(f"/api/missions/{mid}/learning/train", json={"dataset_id": ds["id"], "epochs": 1}, headers=H1)
    assert again.status_code in (200, 409)
    t0 = time.time()
    while time.time() - t0 < 600:
        job = c.get(f"/api/missions/{mid}/learning").json()["training"]["job"]
        if job["state"] not in ("running", "paused"):
            break
        time.sleep(0.5)
    assert job["state"] == "finished", job
    res = job["result"]
    assert res["model"]["name"] == "yolox-s-tuned" and res["history"] and res["device"]
    assert "evaluation" in res, res
    s = c.get(f"/api/missions/{mid}/learning").json()
    new = [md for md in s["models"] if md["name"] == "yolox-s-tuned"]
    assert len(new) == 1 and new[0]["trainable"] and new[0]["trained_on"] == "v1"
    assert s["production"] is None                   # added and evaluated, never promoted by itself
    ev = c.get(f"/api/missions/{mid}/learning/evaluations/{res['evaluation']}").json()
    assert ev["state"] == "completed" and ev["model"]["model"]["version"] == res["model"]["version"]
    assert not (out / ".training").exists() or not any((out / ".training").iterdir())   # temporary copy removed


def test_training_can_be_stopped(tmp_path):
    from trafficwb.process import RunControl
    ds = _dataset(tmp_path, n=4)
    ctl = RunControl()
    steps = []

    def progress(step, total):
        steps.append(step)
        ctl.cancel()

    with pytest.raises(tr.TrainCancelled):
        tr.train(ds, tmp_path / "out", init="none", epochs=3, batch=2, workers=0, device="cpu", size="nano",
                 version="t1", log=lambda *_: None, progress=progress, control=ctl)
    assert steps == [1] and not (tmp_path / "out" / "yolox-s-tuned-t1.zip").exists()


def test_no_download_when_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("LANEPULSE_CACHE", str(tmp_path))
    monkeypatch.setattr(tr, "shipped_weights_dirs", lambda: [tmp_path])
    with pytest.raises(tr.TrainError, match="not installed"):
        tr.coco_weights(download=False)
