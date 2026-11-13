"""D-FINE and RT-DETRv2 training (detr_train.py) on a small exported dataset, with tiny models made here (no download)."""
import json
import zipfile

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from trafficwb import detr_train as dt  # noqa: E402
from trafficwb.detect import DetectorSettings, make_detector  # noqa: E402
from trafficwb.models import ModelPack  # noqa: E402
from trafficwb.train import TrainError  # noqa: E402

if not dt.available()["ok"]:
    pytest.skip(dt.available()["reason"], allow_module_level=True)


def tiny_weights(arch, folder):
    """A small randomly initialised model of the architecture, saved like the released weights."""
    if arch == "d-fine":
        from transformers import DFineConfig, DFineForObjectDetection, HGNetV2Config
        bb = HGNetV2Config(downsample_in_bottleneck=False, downsample_in_first_stage=False, embedding_size=32,
                           hidden_sizes=[128, 256, 512, 1024], layer_type="basic", out_features=["stage3", "stage4"],
                           out_indices=[3, 4], stage_in_channels=[16, 64, 256, 512], stage_mid_channels=[16, 32, 64, 128],
                           stage_num_blocks=[1, 1, 1, 1], stage_numb_of_layers=[1, 1, 1, 1],
                           stage_out_channels=[64, 256, 512, 1024], stem_channels=[3, 16, 16],
                           use_learnable_affine_block=True)
        cfg = DFineConfig(backbone_config=bb, d_model=64, decoder_ffn_dim=128, decoder_in_channels=[64, 64], decoder_layers=2,
                          decoder_n_points=[2, 2], depth_mult=0.34, encode_proj_layers=[1], encoder_ffn_dim=128,
                          encoder_hidden_dim=64, encoder_in_channels=[512, 1024], feat_strides=[16, 32],
                          hidden_expansion=0.34, num_feature_levels=2, num_queries=20, num_denoising=10, num_labels=80)
        model = DFineForObjectDetection(cfg)
    else:
        from transformers import RTDetrResNetConfig, RTDetrV2Config, RTDetrV2ForObjectDetection
        bb = RTDetrResNetConfig(depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512], layer_type="basic",
                                out_features=["stage2", "stage3", "stage4"], out_indices=[2, 3, 4])
        cfg = RTDetrV2Config(backbone_config=bb, decoder_layers=2, encoder_in_channels=[128, 256, 512], hidden_expansion=0.5,
                             d_model=64, encoder_hidden_dim=64, decoder_in_channels=[64, 64, 64], encoder_ffn_dim=128,
                             decoder_ffn_dim=128, num_queries=20, num_denoising=10, num_labels=80)
        model = RTDetrV2ForObjectDetection(cfg)
    model.save_pretrained(str(folder))
    return folder


def dataset(folder):
    """Three training and one validation picture: cars (category 2) and a taught target (category 7, a drone).
    The first picture is a partial review with an ignore box."""
    (folder / "images").mkdir(parents=True)
    cats = [{"id": i, "name": n} for i, n in enumerate(["bicycle", "car", "motorcycle", "bus", "truck", "person"], 1)]
    cats.append({"id": 7, "name": "drone"})
    rng = np.random.default_rng(0)
    for split, n in (("train", 3), ("val", 1)):
        imgs, anns = [], []
        for k in range(1, n + 1):
            im = rng.integers(80, 120, (240, 320, 3), dtype=np.uint8)
            cv2.rectangle(im, (40, 120), (120, 170), (30, 30, 200), -1)
            cv2.circle(im, (250, 50), 18, (0, 215, 255), -1)
            name = f"images/{split}{k}.png"
            cv2.imwrite(str(folder / name), im)
            imgs.append({"id": k, "file_name": name, "width": 320, "height": 240,
                         "review": "partial" if (split, k) == ("train", 1) else "full",
                         "ignore_boxes": [[0, 0, 30, 30]] if (split, k) == ("train", 1) else []})
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": 2, "bbox": [40, 120, 80, 50], "area": 4000,
                         "iscrowd": 0})
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": 7, "bbox": [232, 32, 36, 36], "area": 1296,
                         "iscrowd": 0})
        (folder / f"annotations_{split}.json").write_text(json.dumps({"images": imgs, "annotations": anns,
                                                                      "categories": cats}), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({"name": "test", "files": ["annotations_train.json"]}),
                                          encoding="utf-8")
    return folder


def test_unlooked_areas_go_grey_and_labelled_boxes_stay():
    im = np.full((100, 100, 3), 200, np.uint8)
    out = dt.grey_unlooked(im, {"review": "partial"}, [[10, 10, 20, 20]])
    assert (out[50, 50] == dt.GREY).all() and (out[20, 20] == 200).all()
    out = dt.grey_unlooked(im, {"review": "sam", "ignore_boxes": [[60, 60, 90, 90]]}, [])
    assert (out[70, 70] == dt.GREY).all() and (out[20, 20] == 200).all()      # SAM: unsure boxes only
    assert dt.grey_unlooked(im, {"review": "full"}, []) is im


@pytest.mark.parametrize("arch", ["d-fine", "rt-detr"])
def test_train_writes_a_pack_the_app_runs_with_a_taught_target(arch, tmp_path):
    w = tiny_weights(arch, tmp_path / "start")
    ds = dataset(tmp_path / "ds")
    seen = []
    z = dt.train(ds, tmp_path / "out", arch=arch, init=str(w), epochs=2, batch=2, device="cpu", name="t", version="1",
                 on_epoch=lambda ep, n, row: seen.append(row), log=lambda *a: None)
    assert len(seen) == 2 and seen[0]["val_loss"] is not None
    pack = ModelPack.load(tmp_path / "out" / "t" / "1")
    card = pack.card
    assert card["architecture"] == arch and card["family"] == "detr" and card["outputs"] == "boxes_logits"
    assert card["own_classes"] == {"4": "drone"}                       # the first free COCO class
    assert {"model.onnx", "model.safetensors", "config.json", "LICENSE"} <= {p.name for p in pack.path.iterdir()}
    assert zipfile.ZipFile(z).namelist()
    det = make_detector(pack, DetectorSettings(tiles=(1, 1), score_threshold=0.0))
    assert "drone" in det.classes.values() and "car" in det.classes.values()
    found = det.detect(cv2.imread(str(ds / "images" / "val1.png")))
    assert found and all(np.isfinite(d.box).all() for d in found)
    # The ONNX model gives what the PyTorch one gives.
    import onnxruntime as ort
    from transformers import AutoModelForObjectDetection
    m = AutoModelForObjectDetection.from_pretrained(str(pack.path)).eval()
    x = np.random.default_rng(1).random((1, 3, dt.INPUT, dt.INPUT), dtype=np.float32)
    with torch.no_grad():
        ref = m(pixel_values=torch.from_numpy(x))
    boxes, logits = ort.InferenceSession(str(pack.weights)).run(None, {"images": x})
    # (an untrained tiny model's queries nearly tie, so they are compared as sets: best score per query, sorted)
    best = lambda v: np.sort(v[0].max(1))                                   # noqa: E731
    assert np.abs(best(logits) - best(ref.logits.numpy())).max() < 1e-2 and boxes.shape == ref.pred_boxes.shape
    # Training goes on from the pack, keeps the taught target's class, and refuses another architecture's pack.
    dt.train(ds, tmp_path / "out", arch=arch, init=str(z), epochs=1, batch=2, device="cpu", name="t", version="2",
             log=lambda *a: None)
    assert ModelPack.load(tmp_path / "out" / "t" / "2").card["own_classes"] == {"4": "drone"}
    other = "rt-detr" if arch == "d-fine" else "d-fine"
    with pytest.raises(TrainError, match="is not a"):
        dt.train(ds, tmp_path / "out", arch=other, init=str(z), epochs=1, device="cpu", log=lambda *a: None)


def test_sizes_and_offline_weights(tmp_path, monkeypatch):
    with pytest.raises(TrainError, match="size must be one of"):
        dt._check_arch("d-fine", "huge")
    with pytest.raises(TrainError, match="Unknown detector"):
        dt._check_arch("yolo-whatever", None)
    monkeypatch.setattr(dt, "shipped_weights_dirs", lambda: [tmp_path])
    assert dt.find_coco_weights("rt-detr", "r18") is None
    with pytest.raises(TrainError, match="not installed here"):
        dt.coco_weights("rt-detr", "r18", download=False)
    tiny_weights("rt-detr", tmp_path / dt.weights_folder("rt-detr", "r18"))
    assert dt.find_coco_weights("rt-detr", "r18") == tmp_path / "PekingU__rtdetr_v2_r18vd"
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"                          # never contacts a model hub when training
