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


# ---------- several dataset versions, training rounds, saved models ----------

def second_mission(tmp_path, root, days=(0, 1, 2), cam="0099B"):
    src = tmp_path / f"src-{cam}"
    src.mkdir()
    m = root.create_mission(MissionConfig(
        name=f"Learn {cam}", input_type="images", mode="general_traffic", country="IL", iana_zone="Asia/Jerusalem",
        source_dir=str(src), include_subdirs=False, processing="continuous", output_dir=str(tmp_path / "out"),
        export_formats=["csv"]), "test")
    n = 100
    for d in days:
        n += 1
        write(src, name(d, 0).replace("0045A", cam), frame(n, cars=[(100, 250)], trucks=[]))
    cycle(m)
    Processor(m, FakeDetector()).process_all()
    approve_all_individually(m)
    return m


def test_datasets_of_several_missions_train_and_evaluate_together(mission, tmp_path):
    m, root, _ = mission
    approve_all_individually(m)
    ls = LearningService(m)
    a = ls.create_dataset("here", "lior", seed=1)
    m2 = second_mission(tmp_path, root)
    b = LearningService(m2).create_dataset("there", "lior", ratios=(0, 0, 1))
    refs = [{"mission_id": m.id, "dataset_id": a["id"]}, {"mission_id": m2.id, "dataset_id": b["id"]}]
    listed = {(d["mission_id"], d["id"]) for d in ls.all_datasets()}
    assert {(m.id, a["id"]), (m2.id, b["id"])} <= listed
    items, sources = ls.combine(refs)
    assert len(items) == 2 * DAYS + 3 and [s["name"] for s in sources] == ["here", "there"]
    assert all(i["split"] == "test" for i in items if i["mission_id"] == m2.id)
    # Exported as one dataset: every image once, ids unique across missions.
    folder = ls.export_combined(refs, tmp_path / "exp")
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert man["name"] == "here + Learn 0099B: there" and len(man["datasets"]) == 2
    ids = []
    for sp in ("train", "val", "test"):
        coco = json.loads((folder / f"annotations_{sp}.json").read_text(encoding="utf-8"))
        ids += [im["id"] for im in coco["images"]]
        assert all((folder / im["file_name"]).exists() for im in coco["images"])
    assert len(ids) == len(set(ids)) == 2 * DAYS + 3
    # Evaluated on both test sets together; the views of each mission are kept apart.
    ev = ls.evaluate(None, FakeDetector(), "lior", datasets=refs)
    assert ev["state"] == "completed" and ev["dataset_id"] == a["id"] and len(ev["sources"]) == 2
    assert any(k.startswith("Learn 0099B: ") for k in ev["report"]["by_view"])
    assert ev["report"]["overall"]["recall"] == 1.0
    found = ls.find_evaluation("fake", "1", list(reversed(ev["sources"])), "test")
    assert found is None or found["id"] == ev["id"]          # same images in any order is the same data


def test_a_test_group_stays_test_when_versions_are_combined(mission):
    m, _, _ = mission
    approve_all_individually(m)
    ls = LearningService(m)
    a = ls.create_dataset("a", "lior", seed=1)
    b = ls.create_dataset("b", "lior", seed=1)
    # Someone (an older version) had a group in training that the other has in test: test wins.
    g = m.con.execute("SELECT view_key, block FROM dataset_member WHERE dataset_id=? AND split='test' LIMIT 1",
                      (a["id"],)).fetchone()
    m.con.execute("UPDATE dataset_member SET split='train' WHERE dataset_id=? AND view_key=? AND block=?", (b["id"], *g))
    items, _ = ls.combine([(m.id, a["id"]), (m.id, b["id"])])
    mine = [i for i in items if (i["row"]["view_key"], i["row"]["block"]) == tuple(g)]
    assert mine and all(i["split"] == "test" for i in mine)
    assert len(items) == 2 * DAYS                              # identical images count once


def test_rounds_that_did_not_improve_hand_their_data_on(mission):
    m, _, _ = mission
    approve_all_individually(m)
    ls = LearningService(m)
    a = ls.create_dataset("a", "lior")
    b = ls.create_dataset("b", "lior")
    assert ls.carry_over() == {"runs": [], "datasets": []}
    r1 = ls.start_training_run([(m.id, a["id"])], "coco", 5, "lior")
    ls.finish_training_run(r1, state="finished", improved=False, history=[{"epoch": 1, "train_loss": 2.0, "val_loss": 2.5}])
    r2 = ls.start_training_run([(m.id, b["id"])], "coco", 5, "lior")
    ls.finish_training_run(r2, state="finished", improved=False)
    co = ls.carry_over()
    assert co["runs"] == [r1, r2]
    assert co["datasets"] == [{"mission_id": m.id, "dataset_id": a["id"]}, {"mission_id": m.id, "dataset_id": b["id"]}]
    r3 = ls.start_training_run([(m.id, a["id"]), (m.id, b["id"])], "coco", 5, "lior")
    ls.finish_training_run(r3, state="failed")                 # a failed round neither hands on nor resets
    assert ls.carry_over()["runs"] == [r1, r2]
    r4 = ls.start_training_run([(m.id, b["id"])], "coco", 5, "lior")
    ls.finish_training_run(r4, state="finished", improved=True)
    assert ls.carry_over() == {"runs": [], "datasets": []}
    runs = ls.training_runs()
    assert runs[-1]["history"][0]["val_loss"] == 2.5 and runs[0]["improved"] is True and runs[-1]["finished_at"]


def test_improved_compares_f1_then_ap():
    from trafficwb.learning import improved
    assert improved({"f1": 0.8, "ap50": 0.7}, {"f1": 0.7, "ap50": 0.9}) is True
    assert improved({"f1": 0.7, "ap50": 0.7}, {"f1": 0.8, "ap50": 0.6}) is False
    assert improved({"f1": 0.8, "ap50": 0.75}, {"f1": 0.8, "ap50": 0.7}) is True
    assert improved({"f1": 0.8, "ap50": 0.7}, {"f1": 0.8, "ap50": 0.7}) is False
    assert improved(None, {"f1": 0.8}) is None


def test_saved_models_are_frozen_and_checked(tmp_path):
    import os
    from trafficwb.learning import ModelLibrary
    models = tmp_path / "models"
    _pack(models, "det", "1")
    _pack(models, "det", "2")
    lib = ModelLibrary(tmp_path, models_root=models)
    s = lib.save("Night cars", "det", "1", "lior", "cars at night, Route 6")
    assert s["name"] == "Night cars" and s["model_version"] == "1"
    assert not os.access(models / "det" / "1" / "model.onnx", os.W_OK) or os.name == "nt" or os.geteuid() == 0
    with pytest.raises(LearningError, match="already called"):
        lib.save("night CARS", "det", "2", "lior")
    with pytest.raises(LearningError, match="already saved as"):
        lib.save("Other", "det", "1", "lior")
    with pytest.raises(LearningError, match="not installed"):
        lib.save("Ghost", "det", "9", "lior")
    assert lib.pack("night cars").version == "1"
    assert [x["name"] for x in lib.saved()] == ["Night cars"] and lib.saved()[0]["installed"]
    # The files changed after saving: the model is not used.
    lib.con.execute("UPDATE saved_model SET sha256=? WHERE name=?", ("0" * 64, "Night cars"))
    with pytest.raises(LearningError, match="changed"):
        lib.pack("Night cars")
    assert not lib.saved()[0]["installed"]
    lib.remove("Night cars")
    assert lib.saved() == [] and (models / "det" / "1" / "model.onnx").exists()


# ------------------------------------------------------------------ SAM 3.1 as the teacher

class Teacher(FakeDetector):
    """The red and blue boxes, plus an unsure box and one of a class no detector here learns, like SAM 3.1 would."""

    def detect(self, img):
        out = super().detect(img)
        return out + [Detection((10.0, 10.0, 40.0, 40.0), "car", 0.4, {"car": 0.4}),
                      Detection((500.0, 10.0, 560.0, 60.0), "tractor", 0.8, {"tractor": 0.8})]


def test_sam_labels_what_nobody_reviewed_and_reviews_win(mission, monkeypatch):
    m, _, _ = mission
    rs, ls = ReviewService(m), LearningService(m)
    v = vids(m)
    rs.approve(v[0], "lior")
    truck = next(o for o in rs.objects(v[1]) if o.cls == "truck")
    rs.reclass(v[1], truck.uid, "bus", "lior")
    car = next(o for o in rs.objects(v[1]) if o.cls == "car")
    rs.remove(v[1], car.uid, "lior")
    assert ls.taught("sam3") == {"samples": 0, "sources": 0, "untaught": len(v) - 1}

    # Everything not approved one by one is labelled once per teacher version and words.
    out = ls.teach(Teacher(), "sam3", "k1")
    assert out["labelled"] == len(v) - 1 and out["left"] == 0
    assert ls.teach(Teacher(), "sam3", "k1")["labelled"] == 0
    part = ls.teach(Teacher(), "sam3", "k2", limit=3)
    assert part["labelled"] == 3 and part["left"] == len(v) - 4
    ls.teach(Teacher(), "sam3", "k2")
    assert ls.taught("sam3") == {"samples": len(v) - 1, "sources": len(v) - 1, "untaught": 0}

    # Without the teacher nothing changes: reviewed images only.
    plain = ls.create_dataset("plain", "lior", ratios=(1, 0, 0))
    assert plain["stats"]["images"] == 2 and "teacher" not in plain["stats"]

    ds = ls.create_dataset("taught", "lior", ratios=(1, 0, 0), teacher="sam3")
    assert ds["stats"]["images"] == len(v) and ds["stats"]["teacher"]["samples"] == len(v) - 1
    mem = {r["source_version_id"]: r for r in ls.members(ds["id"])}
    assert mem[v[0]]["review"] == "full"                        # an approved image keeps its own labels
    lab = mem[v[1]]["labels"]
    assert mem[v[1]]["review"] == "sam"
    # The operator's bus stays a bus, the car they removed is not brought back, and the other car is SAM's.
    assert sorted((p["cls"], p["source"]) for p in lab["positives"]) == [("bus", "human"), ("car", "sam")]
    assert not any(p["box"] == [float(x) for x in car.box] for p in lab["positives"])
    # Unsure boxes and classes nobody here learns are ignored, not taught as background.
    assert [10.0, 10.0, 40.0, 40.0] in lab["ignore_boxes"] and [500.0, 10.0, 560.0, 60.0] in lab["ignore_boxes"]
    lab = mem[v[5]]["labels"]
    assert sorted(p["cls"] for p in lab["positives"]) == ["car", "car", "truck"]
    assert {p["source"] for p in lab["positives"]} == {"sam"}


def test_sam_labels_never_reach_the_test_split(mission):
    m, _, _ = mission
    ls = LearningService(m)
    ReviewService(m).approve(vids(m)[0], "lior")
    ls.teach(Teacher(), "sam3", "k1")
    ds = ls.create_dataset("taught", "lior", ratios=(0.4, 0.2, 0.4), seed=3, teacher="sam3")
    mem = ls.members(ds["id"])
    assert {r["split"] for r in mem if r["review"] == "sam"} <= {"train", "val"}
    test_groups = {(r["view_key"], r["block"]) for r in mem if r["split"] == "test"}
    assert ds["stats"]["teacher"]["kept_out_of_test"] > 0 or not test_groups
    assert all(r["review"] == "full" for r in mem if r["split"] == "test")
    with pytest.raises(LearningError, match="teacher"):
        ls.create_dataset("x", "lior", teacher="gpt")
