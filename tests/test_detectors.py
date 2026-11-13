"""Detector families (phase 1c): transformer detectors exported to ONNX run through the same interface as YOLOX."""
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from trafficwb.detect import DetectorSettings, DetrDetector, YoloxDetector, make_detector  # noqa: E402
from trafficwb.models import ModelPack, sha256_file  # noqa: E402

S = 64


def _const_model(path, inputs, outputs: dict):
    """An ONNX model that ignores its picture and always answers ``outputs``."""
    nodes = [helper.make_node("Constant", [], [k], value=numpy_helper.from_array(v, k)) for k, v in outputs.items()]
    ins = [helper.make_tensor_value_info(n, t, shp) for n, t, shp in inputs]
    outs = [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), list(v.shape))
            for k, v in outputs.items()]
    g = helper.make_graph(nodes, "const", ins, outs)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def _pack(tmp_path, name, card, inputs, outputs):
    d = tmp_path / name / "1"
    d.mkdir(parents=True)
    _const_model(d / "model.onnx", inputs, outputs)
    (d / "model.json").write_text(json.dumps({"name": name, "version": "1", "task": "vehicle_detection",
                                              "license": "Apache-2.0", "input_size": S,
                                              "sha256": sha256_file(d / "model.onnx")} | card))
    return ModelPack.load(d)


def test_dfine_layout_labels_boxes_scores(tmp_path):
    # Two queries: a car (COCO category id 3) and a person (id 1, not looked for), boxes in input pixels.
    pack = _pack(tmp_path, "dfine-s", {"family": "detr", "class_ids": "coco91"},
                 [("images", TensorProto.FLOAT, [1, 3, S, S]), ("orig_target_sizes", TensorProto.INT64, [1, 2])],
                 {"labels": np.array([[3, 1, 3]], np.int64),
                  "boxes": np.array([[[8, 16, 24, 48], [0, 0, 10, 10], [30, 30, 40, 40]]], np.float32),
                  "scores": np.array([[0.9, 0.95, 0.1]], np.float32)})
    det = make_detector(pack, DetectorSettings(tiles=(1, 1)))
    assert isinstance(det, DetrDetector) and det.describe()["model"]["family"] == "detr"
    img = np.zeros((S // 2, S * 2, 3), np.uint8)               # stretched to the input: x * 2, y / 2
    out = det.detect(img)
    assert [(d.cls, d.box, d.score) for d in out] == [("car", (16.0, 8.0, 48.0, 24.0), 0.9)]
    # The same model, told to look for people too, finds the person.
    both = det.with_classes({0: "person", 2: "car"}).detect(img)
    assert sorted(d.cls for d in both) == ["car", "person"]


def test_rfdetr_layout_boxes_and_logits(tmp_path):
    logits = np.full((1, 2, 91), -9.0, np.float32)
    logits[0, 0, 8] = 3.0                                        # COCO category 8: truck
    logits[0, 1, 3] = -2.0                                       # a weak car, under the threshold
    pack = _pack(tmp_path, "rfdetr-b", {"family": "detr", "outputs": "boxes_logits", "class_ids": "coco91"},
                 [("input", TensorProto.FLOAT, [1, 3, S, S])],
                 {"dets": np.array([[[0.5, 0.5, 0.5, 0.25], [0.1, 0.1, 0.1, 0.1]]], np.float32), "labels": logits})
    det = make_detector(pack, DetectorSettings(tiles=(1, 1)))
    assert det.describe()["preprocessing"]["mean"] == pytest.approx([0.485, 0.456, 0.406])
    out = det.detect(np.zeros((100, 200, 3), np.uint8))
    assert len(out) == 1 and out[0].cls == "truck" and out[0].box == (50.0, 37.5, 150.0, 62.5)
    assert out[0].score == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-4)


def test_family_picks_the_detector(tmp_path):
    yolox = _pack(tmp_path, "yx", {}, [("images", TensorProto.FLOAT, [1, 3, S, S])],
                  {"out": np.zeros((1, 84, 85), np.float32)})
    assert isinstance(make_detector(yolox), YoloxDetector)
    odd = _pack(tmp_path, "odd", {"family": "something"}, [("images", TensorProto.FLOAT, [1, 3, S, S])],
                {"out": np.zeros((1, 1), np.float32)})
    with pytest.raises(ValueError, match="cannot run"):
        make_detector(odd)


def test_wrap_an_exported_detector_into_a_pack(tmp_path):
    from trafficwb.model_import import import_pack, main
    src = tmp_path / "dfine_s_coco.onnx"
    _const_model(src, [("images", TensorProto.FLOAT, [1, 3, S, S]), ("orig_target_sizes", TensorProto.INT64, [1, 2])],
                 {"labels": np.array([[3]], np.int64), "boxes": np.array([[[1, 1, 20, 20]]], np.float32),
                  "scores": np.array([[0.8]], np.float32)})
    assert main(["wrap", str(src), "--name", "dfine-s", "--version", "coco", "--license", "Apache-2.0",
                 "--input-size", str(S), "--out", str(tmp_path / "packs")]) == 0
    r = import_pack(tmp_path / "packs" / "dfine-s" / "coco", tmp_path / "models")
    pack = ModelPack.load(tmp_path / "models" / "dfine-s" / "coco")
    assert r["name"] == "dfine-s" and pack.card["family"] == "detr" and pack.card["class_ids"] == "coco91"
    assert [d.cls for d in make_detector(pack, DetectorSettings(tiles=(1, 1))).detect(np.zeros((S, S, 3), np.uint8))] == ["car"]
    assert main(["wrap", str(src), "--name", "dfine-s", "--version", "coco", "--license", "x",
                 "--out", str(tmp_path / "packs")]) == 1                         # never replaced
