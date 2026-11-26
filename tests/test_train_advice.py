"""Advice after a training round: what the curves say, and replace or keep (recall first)."""
from trafficwb.train_advice import curve_advice, swap_advice


def _hist(train, val):
    return [{"epoch": i + 1, "train_loss": t, "val_loss": v} for i, (t, v) in enumerate(zip(train, val))]


def kinds(a):
    return [x["kind"] for x in a["items"]]


def test_overfitting_is_seen_and_more_epochs_are_not_advised():
    train = [3.0 - 0.1 * i for i in range(20)]
    val = [2.5, 2.2, 2.0, 1.9, 1.85, 1.8, 1.82, 1.86, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45]
    a = curve_advice(_hist(train, val), 20, val_before=2.6, train_images=400, val_images=80)
    assert a["verdict"] == "overfitting" and kinds(a)[0] == "overfitting"
    item = a["items"][0]
    assert "epoch 6" in item["title"] and item["suggested_epochs"] < 20 and "varied" in item["text"]


def test_still_falling_at_the_end_advises_more_epochs():
    train = [3.0 - 0.1 * i for i in range(10)]
    val = [2.5 - 0.1 * i for i in range(10)]
    a = curve_advice(_hist(train, val), 10, val_before=2.6, train_images=400, val_images=80)
    assert a["verdict"] == "still_learning"
    it = next(x for x in a["items"] if x["kind"] == "still_learning")
    assert it["suggested_epochs"] >= 20


def test_levelled_off_is_healthy():
    train = [3.0, 2.5, 2.2, 2.0, 1.9, 1.85, 1.82, 1.8, 1.79, 1.79]
    val = [2.4, 2.1, 1.95, 1.9, 1.88, 1.87, 1.87, 1.865, 1.866, 1.866]
    a = curve_advice(_hist(train, val), 10, val_before=2.6, train_images=400, val_images=80)
    assert a["verdict"] == "converged" and kinds(a) == ["converged"]


def test_no_gain_no_validation_small_data_and_rare_class():
    train = [3.0, 2.9, 2.8, 2.7]
    val = [2.6, 2.62, 2.61, 2.63]
    a = curve_advice(_hist(train, val), 4, val_before=2.6, train_images=30, val_images=5,
                     class_counts={"car": 300, "bus": 4, "truck": 40})
    assert a["verdict"] == "no_gain"
    assert {"no_gain", "few_images", "few_validation", "rare_class"} <= set(kinds(a))
    assert "bus" in next(x for x in a["items"] if x["kind"] == "rare_class")["title"]
    b = curve_advice([{"epoch": i, "train_loss": 2.0 - 0.1 * i, "val_loss": None} for i in range(1, 6)], 5)
    assert "no_validation" in kinds(b)
    assert curve_advice([], 0)["items"] == []


def test_evaluation_points_to_missed_vehicles_and_the_weak_condition():
    train = [3.0, 2.5, 2.2, 2.0, 1.9, 1.85, 1.82, 1.8, 1.79, 1.79]
    val = [2.4, 2.1, 1.95, 1.9, 1.88, 1.87, 1.87, 1.865, 1.866, 1.866]
    a = curve_advice(_hist(train, val), 10, train_images=400, val_images=80,
                     overall={"recall": 0.7, "precision": 0.9, "true_positives": 70, "references": 100},
                     by_condition={"day": {"recall": 0.85, "references": 60}, "night": {"recall": 0.45, "references": 40}})
    assert "missed" in kinds(a)
    cond = next(x for x in a["items"] if x["kind"] == "condition")
    assert cond["condition"] == "night" and "night" in cond["text"]


def _ov(tp, refs, p, fp=None):
    return {"true_positives": tp, "references": refs, "recall": tp / refs, "precision": p,
            "f1": 2 * p * (tp / refs) / (p + tp / refs), "ap50": 0.5, "false_positives_full": fp}


def test_more_vehicles_found_means_replace_even_with_more_false_alarms():
    s = swap_advice(_ov(180, 200, 0.70, 77), _ov(160, 200, 0.85, 28))
    assert s["verdict"] == "replace" and "20 more" in s["headline"]
    assert any("wrong boxes" in c for c in s["cautions"])


def test_fewer_vehicles_found_means_keep_even_with_better_precision():
    s = swap_advice(_ov(150, 200, 0.95), _ov(160, 200, 0.80), current_label="the saved model “Route 6”")
    assert s["verdict"] == "keep" and "misses 10 more" in s["headline"] and "Route 6" in s["headline"]


def test_same_recall_precision_decides():
    assert swap_advice(_ov(160, 200, 0.90), _ov(159, 200, 0.80))["verdict"] == "replace"
    assert swap_advice(_ov(160, 200, 0.70), _ov(161, 200, 0.80))["verdict"] == "keep"
    assert swap_advice(_ov(160, 200, 0.80), _ov(160, 200, 0.80))["verdict"] == "keep"


def test_condition_drop_is_a_caution_and_tiny_tests_are_unclear():
    s = swap_advice(_ov(180, 200, 0.8), _ov(160, 200, 0.8),
                    new_by_condition={"night": {"recall": 0.5, "references": 40}},
                    current_by_condition={"night": {"recall": 0.7, "references": 40}})
    assert s["verdict"] == "replace" and any("night" in c for c in s["cautions"])
    assert swap_advice(_ov(9, 10, 0.9), _ov(5, 10, 0.9))["verdict"] == "unclear"
    assert swap_advice(None, _ov(5, 10, 0.9))["verdict"] == "unclear"


def test_round_keeps_its_advice_and_the_operator_answer(tmp_path):
    """The advice is stored with the round; answering it is a record only and never changes the mission's model."""
    from fastapi.testclient import TestClient

    from test_web_api import H1, Fake, direct_access, mission
    from trafficwb.learning import LearningService
    from trafficwb.store import DataRoot
    from trafficwb.web.app import create_app

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    app = create_app(tmp_path / "data", detector_factory=lambda: (Fake(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = direct_access(TestClient(app))
    mid = mission(c, src, out)
    ls = LearningService(DataRoot(tmp_path / "data").open_mission(mid))
    rid = ls.start_training_run([{"mission_id": mid, "dataset_id": 1}], "coco", 10, "lior")
    advice = {"curves": curve_advice(_hist([3, 2.8, 2.6], [2.5, 2.4, 2.3]), 3),
              "swap": swap_advice(_ov(180, 200, 0.8, 5), _ov(160, 200, 0.8, 5))}
    ls.finish_training_run(rid, state="finished", advice=advice, current_evaluation_id=7)
    none = ls.start_training_run([{"mission_id": mid, "dataset_id": 1}], "coco", 10, "lior")
    run = c.get(f"/api/missions/{mid}/learning").json()["training_runs"][1]
    assert run["advice"]["swap"]["verdict"] == "replace" and run["current_evaluation_id"] == 7

    bad = c.post(f"/api/missions/{mid}/learning/rounds/{rid}/decision", json={"decision": "maybe"}, headers=H1)
    assert bad.status_code == 400
    assert c.post(f"/api/missions/{mid}/learning/rounds/{none}/decision", json={"decision": "kept"},
                  headers=H1).status_code == 400
    r = c.post(f"/api/missions/{mid}/learning/rounds/{rid}/decision", json={"decision": "kept"}, headers=H1).json()
    assert r["advice"]["decision"]["choice"] == "kept" and r["advice"]["swap"]["verdict"] == "replace"
    assert c.get(f"/api/missions/{mid}").json()["config"].get("vehicle_model") is None
