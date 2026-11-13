"""Following a target the operator marked, through a recording (learning a target SAM 3.1 does not know)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
av = pytest.importorskip("av")
from test_video import H, W, make_video  # noqa: E402
from trafficwb.follow import SURE, follow  # noqa: E402

_rng = np.random.default_rng(5)
BG = cv2.GaussianBlur(_rng.integers(60, 200, (H, W, 3), dtype=np.uint8), (0, 0), 3)
PATTERN = cv2.resize(np.random.default_rng(8).integers(0, 255, (6, 8, 3), dtype=np.uint8), (64, 48),
                     interpolation=cv2.INTER_NEAREST)


def where(t: float) -> list[float]:
    """The target enters on the left at 1 s, comes closer (grows from 32 to about 56 px wide), leaves on the
    right at about 7 s."""
    if t < 1.0:
        return []
    w = 32 + 4 * (t - 1)
    x = -w + 60 * (t - 1)
    return [x, 70.0 - w * 0.375, x + w, 70.0 + w * 0.375]


def scene(t: float) -> np.ndarray:
    img = BG.copy()
    b = where(t)
    if b:
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        p = cv2.resize(PATTERN, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        a, c = max(0, x1), min(W, x2)
        if c > a:
            img[y1:y2, a:c] = p[:, a - x1:c - x1]
    return img


def test_a_marked_target_is_followed_both_ways_and_its_range_suggested(tmp_path):
    path = tmp_path / "CAM1_20261021T120000Z.mp4"
    make_video(path, 8.0, scene)
    t0 = 4.0
    res = follow(path, [{"t": t0, "box": where(t0)}])
    sure = [s for s in res.steps if s.score >= SURE]
    assert res.start_t is not None and res.start_t <= 2.0          # found back to near where it entered
    assert res.end_t is not None and res.end_t >= 6.4               # and on to near where it left
    for s in sure:                                                  # where it really is, through its growth
        x1, y1, x2, y2 = where(s.t) or [0, 0, 0, 0]
        cx = (max(x1, 0) + min(x2, W)) / 2
        assert abs((s.box[0] + s.box[2]) / 2 - cx) < 12, (s.t, s.box, where(s.t))
    grown = [s for s in sure if s.t > 5.5 and s.box[2] < W - 2]
    assert grown and grown[-1].box[2] - grown[-1].box[0] > 40      # bigger than when marked (44 px)
    assert set(res.ended.values()) <= {"lost", "left_frame", "limit"}


def test_a_box_too_small_is_refused(tmp_path):
    path = tmp_path / "CAM1_20261021T120000Z.mp4"
    make_video(path, 2.0, scene)
    with pytest.raises(ValueError, match="too small"):
        follow(path, [{"t": 1.5, "box": [10, 10, 13, 13]}])


def test_mark_follow_confirm_and_teach_a_new_kind_of_target(tmp_path, monkeypatch):
    """The whole flow on a recording: the operator boxes a target the detectors do not know, the app follows it,
    the operator corrects the range and confirms, and its frames become labels of a new class, category 7, that a
    trained model calls by its own name."""
    from fastapi.testclient import TestClient

    from test_video import RedDetector, mission
    from trafficwb import own_classes as tr
    from trafficwb.learning import LearningService
    from trafficwb.video_process import VideoProcessor
    from trafficwb.web.app import create_app
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (8.0, scene, False)})
    VideoProcessor(m, RedDetector(), {}).process_all()            # finds nothing: the target is not red
    vid = m.con.execute("SELECT id FROM source_version").fetchone()[0]
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    h = {"X-TWB": "1"}
    base = f"/api/missions/{m.id}"
    r = c.post(f"{base}/recordings/{vid}/marks", json={"cls": "Drone", "t": 4.0, "box": where(4.0)}, headers=h)
    assert r.status_code == 200, r.text
    tg = r.json()
    assert tg["cls"] == "drone" and tg["state"] == "suggested" and tg["start_t"] <= 2.0 and tg["end_t"] >= 6.4
    assert "drone" in (m.config().sam3_words or {})                # the mission now lists the new class
    # A second box (a correction) follows it again from both; a range that ends before it starts is refused.
    tg = c.post(f"{base}/marks/{tg['id']}/correct", json={"t": 6.0, "box": where(6.0)}, headers=h).json()
    assert len(tg["marks"]) == 2
    assert c.put(f"{base}/marks/{tg['id']}", json={"start_t": 5.0, "end_t": 3.0}, headers=h).status_code == 400
    assert c.put(f"{base}/marks/{tg['id']}", json={"start_t": 1.6, "end_t": 6.4, "state": "confirmed"},
                 headers=h).json()["state"] == "confirmed"
    assert c.post(f"{base}/recordings/{vid}/marks", json={"cls": "drone", "t": 4.0, "box": [1, 1, 3, 3]},
                  headers=h).status_code == 400

    ls = LearningService(m)
    monkeypatch.setattr(LearningService, "_bucket", staticmethod(lambda seed, key: 0.1))     # the group: train
    ds = ls.create_dataset("drones", "lior")
    mem = ls.members(ds["id"])
    assert ds["stats"]["splits"]["train"]["marked"] == len(mem) >= 8
    assert all(r["labels"]["positives"][0]["cls"] == "drone" for r in mem)
    assert all(1.6 - 1e-6 <= r["frame_t"] <= 6.4 + 1e-6 for r in mem)
    folder = ls.export_dataset(ds["id"])
    import json
    ann = json.loads((folder / "annotations_train.json").read_text(encoding="utf-8"))
    assert {"id": 7, "name": "drone"} in ann["categories"] and {a["category_id"] for a in ann["annotations"]} == {7}
    assert ann["images"][0]["unlabelled_area"] == "ignore"       # only the marked box teaches on its frames
    cmap, own = tr.category_map(folder)
    assert cmap[7] == tr.OWN_SLOTS[0] and own == {tr.OWN_SLOTS[0]: "drone"}
    # A model that already knows "drone" keeps it where it is, and a new kind goes to the next free place.
    assert tr.category_map(folder, {str(tr.OWN_SLOTS[3]): "drone"})[1] == {tr.OWN_SLOTS[3]: "drone"}
    assert c.delete(f"{base}/marks/{tg['id']}", headers=h).json()["ok"]
    assert ls.create_dataset("none", "lior")["stats"]["images"] == 0


def test_a_trained_model_finds_marked_targets_by_their_own_names(tmp_path):
    from trafficwb.detect import YoloxDetector
    from trafficwb.models import ModelPack
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    d = tmp_path / "pack"
    d.mkdir()
    out = np.zeros((1, 1, 85), np.float32)
    out[0, 0, :4] = [32, 32, 4.0, 4.0]          # one box in the middle; YOLOX decodes grid-relative later, enough here
    g = helper.make_graph([helper.make_node("Identity", ["c"], ["output"])], "g",
                          [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
                          [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 85])],
                          [numpy_helper.from_array(out, "c")])
    mdl = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    mdl.ir_version = 8
    onnx.save(mdl, str(d / "model.onnx"))
    card = {"name": "t", "version": "1", "task": "vehicle_detection", "weights": "model.onnx", "input_size": 64,
            "classes": "coco80", "own_classes": {"4": "drone"}}
    det = YoloxDetector(ModelPack(d, card))
    assert det.classes[4] == "drone" and det.classes[2] == "car"
