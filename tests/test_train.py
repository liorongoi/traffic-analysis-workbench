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
