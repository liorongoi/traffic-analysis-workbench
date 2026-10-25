import json
from datetime import datetime, timedelta, timezone

import pytest

from test_detect_review import BLUE, FakeDetector, frame, write
from trafficwb.detect import Detection
from trafficwb.ingest import cycle
from trafficwb.learning import LearningError, LearningService, ModelRegistry
from trafficwb.models import sha256_file
from trafficwb.process import Processor
from trafficwb.review import ReviewService
from trafficwb.store import DataRoot, MissionConfig

DAYS = 6


def name(day, k):
    t = datetime(2026, 10, 10, 9, tzinfo=timezone.utc) + timedelta(days=day, minutes=10 * k)
    return f"0045A_{t.strftime('%Y%m%dT%H%M%SZ')}.jpg"


@pytest.fixture
def mission(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(
        name="Learn", input_type="images", mode="general_traffic", country="IL", iana_zone="Asia/Jerusalem",
        source_dir=str(src), include_subdirs=False, processing="continuous", output_dir=str(out),
        export_formats=["csv"]), "test")
    n = 0
    for d in range(DAYS):
        for k in range(2):
            n += 1
            write(src, name(d, k), frame(n, cars=[(100, 250), (400, 100)], trucks=[(250, 200)]))
    cycle(m)
    Processor(m, FakeDetector()).process_all()
    return m, root, out


def vids(m):
    return [r[0] for r in m.con.execute("SELECT id FROM source_version ORDER BY capture_utc")]


def approve_all_individually(m):
    rs = ReviewService(m)
    for v in vids(m):
        rs.approve(v, "lior")


def test_eligibility_full_partial_bulk(mission):
    m, _, _ = mission
    rs, ls = ReviewService(m), LearningService(m)
    v = vids(m)
    rs.approve(v[0], "lior")
    car = next(o for o in rs.objects(v[1]) if o.cls == "car")
    rs.reclass(v[1], car.uid, "bus", "lior")
    e = ls.eligibility()
    assert e["images"] == {"full": 1, "partial": 1, "bulk": 0, "none": len(v) - 2}
    assert e["eligible"] == 2
    # Full review: every object is a label. Partial: only the corrected one.
    assert len(ls.labels(v[0]).positives) == 3
    lab = ls.labels(v[1])
    assert lab.review == "partial" and [p["cls"] for p in lab.positives] == ["bus"]
    # Bulk approval covers the corrected image too, and bulk approved images are never used.
    rs.approve_all_commit(rs.approve_all_preview(), "lior")
    assert ls.eligibility()["images"] == {"full": 1, "partial": 0, "bulk": len(v) - 1, "none": 0}
    ds = ls.create_dataset("v1", "lior", ratios=(0, 0, 1))
    assert ds["stats"]["images"] == 1 and ds["stats"]["skipped"]["bulk"] == len(v) - 1


def test_dataset_splits_group_wise_and_test_stays_frozen(mission):
    m, _, _ = mission
    approve_all_individually(m)
    ls = LearningService(m)
    ds = ls.create_dataset("v1", "lior", seed=1)
    assert ds["leakage"] == []
    mem = ls.members(ds["id"])
    assert len(mem) == 2 * DAYS
    splits_per_group = {}
    for r in mem:
        splits_per_group.setdefault((r["view_key"], r["block"]), set()).add(r["split"])
    assert len(splits_per_group) == DAYS          # one group per local day for this one view
    assert all(len(s) == 1 for s in splits_per_group.values())
    st = ds["stats"]["splits"]
    assert sum(st[s]["images"] for s in st) == 2 * DAYS
    test1 = {(r["view_key"], r["block"]) for r in mem if r["split"] == "test"}
    assert test1, "seed 1 puts at least one day in test"
    # A later version with other ratios and seed keeps the earlier test groups in test.
    ds2 = ls.create_dataset("v2", "lior", ratios=(0.9, 0.1, 0.0), seed=99)
    assert ds2["leakage"] == [] and ds2["frozen_test_from"] == ds["id"]
    test2 = {(r["view_key"], r["block"]) for r in ls.members(ds2["id"]) if r["split"] == "test"}
    assert test1 <= test2
    # Leakage is detected if someone moves an earlier test group into training.
    if test1:
        g = next(iter(test1))
        m.con.execute("UPDATE dataset_member SET split='train' WHERE dataset_id=? AND view_key=? AND block=?",
                      (ds2["id"], *g))
        assert any("earlier test set" in x for x in ls.dataset(ds2["id"])["leakage"])
        with pytest.raises(LearningError, match="leakage"):
            ls.export_dataset(ds2["id"])


def test_dataset_rejects_bad_params(mission):
    ls = LearningService(mission[0])
    with pytest.raises(LearningError):
        ls.create_dataset("x", "lior", ratios=(0.5, 0.5, 0.5))
    with pytest.raises(LearningError):
        ls.create_dataset(" ", "lior")


def test_export_coco_never_overwrites(mission):
    m, _, out = mission
    approve_all_individually(m)
    ls = LearningService(m)
    ds = ls.create_dataset("v1", "lior")
    p1 = ls.export_dataset(ds["id"])
    p2 = ls.export_dataset(ds["id"])
    assert p1 != p2 and p1.exists() and p2.exists() and not list(out.glob("datasets/*.partial"))
    total_imgs = total_anns = 0
    for s in ("train", "val", "test"):
        coco = json.loads((p1 / f"annotations_{s}.json").read_text(encoding="utf-8"))
        total_imgs += len(coco["images"])
        total_anns += len(coco["annotations"])
        for im in coco["images"]:
            assert (p1 / im["file_name"]).exists()
            assert im["unlabelled_area"] == "background inside included regions"
    assert (total_imgs, total_anns) == (2 * DAYS, 6 * DAYS)
    assert json.loads((p1 / "manifest.json").read_text(encoding="utf-8"))["dataset_id"] == ds["id"]


class NoTrucks(FakeDetector):
    def detect(self, img):
        return [d for d in super().detect(img) if d.cls != "truck"]


class ExtraBox(FakeDetector):
    def detect(self, img):
        return super().detect(img) + [Detection((10.0, 10.0, 40.0, 30.0), "car", 0.3, {"car": 0.3})]


def test_evaluation_metrics(mission):
    m, _, _ = mission
    approve_all_individually(m)
    ls = LearningService(m)
    ds = ls.create_dataset("all-test", "lior", ratios=(0, 0, 1))
    ev = ls.evaluate(ds["id"], FakeDetector(), "lior")
    o = ev["report"]["overall"]
    assert ev["state"] == "completed"
    assert (o["precision"], o["recall"], o["ap50"], o["count_mae"], o["class_agreement"]) == (1.0, 1.0, 1.0, 0.0, 1.0)
    o = ls.evaluate(ds["id"], NoTrucks(), "lior")["report"]["overall"]
    assert o["precision"] == 1.0 and o["recall"] == pytest.approx(2 / 3, abs=1e-3) and o["count_mae"] == 1.0
    o = ls.evaluate(ds["id"], ExtraBox(), "lior")["report"]["overall"]
    assert o["recall"] == 1.0 and o["precision"] == 0.75 and o["ap50"] == 1.0   # the extra box scores lowest
    assert set(ls.evaluate(ds["id"], FakeDetector(), "lior")["report"]["by_condition"])
    assert len(ls.evaluations(ds["id"])) == 4
    with pytest.raises(LearningError, match="empty"):
        ls.evaluate(ds["id"], FakeDetector(), "lior", split="train")


def test_partial_review_never_counts_false_positives(mission):
    m, _, _ = mission
    rs, ls = ReviewService(m), LearningService(m)
    v = vids(m)[0]
    truck = next(o for o in rs.objects(v) if o.cls == "truck")
    rs.move(v, truck.uid, (250, 200, 301, 231), "lior")
    ds = ls.create_dataset("partial", "lior", ratios=(0, 0, 1))
    o = ls.evaluate(ds["id"], FakeDetector(), "lior")["report"]["overall"]
    assert o["full_images"] == 0 and o["recall"] == 1.0 and o["precision"] is None and o["false_positives_full"] == 0


def _pack(root, name, version, task="vehicle_detection"):
    d = root / name / version
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(f"{name}{version}".encode())
    (d / "model.json").write_text(json.dumps({"name": name, "version": version, "task": task, "license": "MIT",
                                              "sha256": sha256_file(d / "model.onnx")}))


def test_promotion_and_rollback(tmp_path):
    models = tmp_path / "models"
    _pack(models, "det", "1")
    _pack(models, "det", "2")
    _pack(models, "ocr", "1", task="plate_ocr")
    reg = ModelRegistry(tmp_path)
    assert reg.production("vehicle_detection") is None
    with pytest.raises(LearningError, match="evaluation"):
        reg.promote("vehicle_detection", "det", "1", "lior", models_root=models)
    with pytest.raises(LearningError, match="plate_ocr"):
        reg.promote("vehicle_detection", "ocr", "1", "lior", "m1", 1, models_root=models)
    reg.promote("vehicle_detection", "det", "1", "lior", "m1", 1, models_root=models)
    reg.promote("vehicle_detection", "det", "2", "lior", "m1", 2, models_root=models)
    assert reg.production("vehicle_detection")["version"] == "2"
    assert reg.rollback("vehicle_detection", "lior")["version"] == "1"
    assert reg.production("vehicle_detection")["version"] == "1"
    # A second rollback goes further back (to the built-in default), not forward again.
    assert reg.rollback("vehicle_detection", "lior") is None
    assert reg.production("vehicle_detection") is None
    with pytest.raises(LearningError):
        reg.rollback("vehicle_detection", "lior")
    assert [h["action"] for h in reg.history("vehicle_detection")] == ["rollback", "rollback", "promote", "promote"]
