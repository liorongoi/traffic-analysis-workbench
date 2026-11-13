"""Staging-machine only: download model packs and write their model cards.

Run on a connected machine, then copy the resulting models/ folder to the
offline workstation. The application itself never downloads anything.

    python scripts/fetch_models.py [--dest models]

It also fetches SAM 3.1 (sam3.1-w8a16, about 880 MB; --no-sam3 skips it), RF-DETR Small (exported to ONNX here; needs
PyTorch and the rfdetr package; --no-rfdetr skips it, --rfdetr-sizes picks others) and the YOLOX-s and RF-DETR training
starting weights (yolox_s.pth, rf-detr-small-coco.pth) into training_weights/; the installers put
them next to their own Python (share/lanepulse), so training on the learning page never goes online.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PACKS = [
    {
        "name": "yolox-s",
        "version": "0.1.1rc0",
        "task": "vehicle_detection",
        "weights_url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
        "license": "Apache-2.0 (code and released weights, per the YOLOX repository)",
        "license_url": "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE",
        "data_note": "Weights trained on COCO 2017. COCO annotations are CC BY 4.0; the underlying Flickr images carry their own terms.",
        "source": "https://github.com/Megvii-BaseDetection/YOLOX",
        "input_size": 640,
        "input_format": "BGR uint8 values as float32, letterboxed top-left, pad value 114, no normalisation",
        "classes": "coco80",
    },
    {
        "name": "plate-yolov9-t-640",
        "version": "oim-assets-1",
        "task": "plate_detection",
        "weights_url": "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx",
        "sha256": "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
        "license": "MIT (open-image-models repository). Training images and their rights are not stated: evaluation only.",
        "license_url": "https://raw.githubusercontent.com/ankandrew/open-image-models/main/LICENSE",
        "data_note": "Trained by the open-image-models author on plate datasets that are not documented. The YOLOv9 architecture comes from a GPL-3.0 repository; whether that affects these exported weights is not settled. Verify before operational use.",
        "source": "https://github.com/ankandrew/open-image-models",
        "input_size": 640,
        "input_format": "RGB float32 0..1, letterboxed centred, pad value 114; output rows [batch, x1, y1, x2, y2, class, score] with NMS inside the model",
        "evaluation_only": True,
    },
    {
        "name": "plate-ocr-cct-s-v2-global",
        "version": "arg-plates-1",
        "task": "plate_ocr",
        "weights_url": "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_s_v2_global.onnx",
        "sha256": "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b",
        "license": "MIT (fast-plate-ocr repository). Training images and their rights are not stated: evaluation only.",
        "license_url": "https://raw.githubusercontent.com/ankandrew/fast-plate-ocr/master/LICENSE",
        "data_note": "Global plate dataset of the fast-plate-ocr author, not documented. Latin letters and digits only: it cannot read Persian or Arabic characters.",
        "source": "https://github.com/ankandrew/fast-plate-ocr",
        "input_size": [64, 128],
        "input_format": "RGB uint8, resized to 128x64 without keeping aspect ratio (linear)",
        "alphabet": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
        "pad_char": "_",
        "max_slots": 10,
        "region_labels": ['Albania', 'Andorra', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Gibraltar', 'Greece', 'Guernsey', 'Hungary', 'Iceland', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malaysia', 'Malta', 'Mexico', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'New Zealand', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Qatar', 'Romania', 'San Marino', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Turkey', 'United States', 'Ukraine', 'United Kingdom', 'Vietnam', 'Unknown'],
        "evaluation_only": True,
    },
]


# Starting weights for training the vehicle detector (PyTorch checkpoint of the same released YOLOX-s).
TRAINING_WEIGHTS = {
    "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "sha256": "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    "file": "yolox_s.pth",
}


# SAM 3.1 (Meta's Segment Anything 3.1): the 8-bit weight-compressed (W8A16) OpenVINO export by
# rajeshgangireddy, pinned to one commit, with every file's SHA-256. The .bin hashes are the ones Hugging Face
# lists for those files; the .xml graphs and the tokenizer were hashed when the pin was chosen (29 Oct 2026).
# The license is Meta's SAM License, which allows redistribution with the license text: it is shipped from
# packaging/licenses (the Hugging Face repository only links to it).
SAM3_COMMIT = "9e742f7509733b3cec1b1c902db562abff877b59"
SAM3_BASE = f"https://huggingface.co/rajeshgangireddy/SAM3_OpenVINO/resolve/{SAM3_COMMIT}/"
SAM3_FILES = {
    "vision-encoder.xml": ("openvino-int8_w8a16/vision-encoder.xml", "415f47ec986f350c6fc35cc61d2eb1e45f695a71190ff95c00fb391c9de64d78"),
    "vision-encoder.bin": ("openvino-int8_w8a16/vision-encoder.bin", "f973e2dcce4a76e7edc181578d35ed3d0a5e3bdefef2b4b8a40739c81a309f19"),
    "text-encoder.xml": ("openvino-int8_w8a16/text-encoder.xml", "cec21de907c0ee54241690b5c8f56618cd564c788b306f0fbb63b3f02578847c"),
    "text-encoder.bin": ("openvino-int8_w8a16/text-encoder.bin", "a486e1d22badae9150bee4f03c00bf75296b433cd6a4d087af7bc91b2fc01496"),
    "prompt-decoder.xml": ("openvino-int8_w8a16/prompt-decoder.xml", "68db349e37c459adfe974d823255fada2178c2fc97524063e055ba367ddc97e6"),
    "prompt-decoder.bin": ("openvino-int8_w8a16/prompt-decoder.bin", "7c30ef7afc461912b3fae9e37d14b33bc462c24df4b06035f0479771bd3b77cb"),
    "tokenizer.json": ("tokenizer.json", "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"),
}
SAM3_LICENSE_SHA256 = "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be"   # Meta's text, 19 Nov 2025
SAM3_CARD = {
    "name": "sam3.1-w8a16",
    "version": "ov-9e742f7",
    "task": "segmentation",
    "license": "SAM License (Meta, 19 November 2025): use, copy, change and redistribute with this license included; "
               "not for military, warfare, nuclear, espionage or ITAR-controlled uses",
    "license_url": "https://github.com/facebookresearch/sam3/blob/main/LICENSE",
    "source": "https://huggingface.co/rajeshgangireddy/SAM3_OpenVINO (OpenVINO export of facebook/sam3.1, "
              f"commit {SAM3_COMMIT}); original model https://github.com/facebookresearch/sam3",
    "data_note": "Trained by Meta on its SA-Co data. Nothing in LanePulse retrains it; it runs as published.",
    "precision": "INT8 weights, 16-bit activations (W8A16, NNCF weight compression)",
    "runtime": "openvino",
    "weights": "vision-encoder.xml",
    "graphs": {"vision": "vision-encoder.xml", "text": "text-encoder.xml", "decoder": "prompt-decoder.xml"},
    "tokenizer": "tokenizer.json",
    "input_size": 1008,
    "input_format": "RGB, resized to 1008x1008 without keeping aspect ratio, (x/255 - 0.5) / 0.5, NCHW float32",
    "text_length": 32,
    "pad_token_id": 49407,
    "box_format": "xyxy",
    "mask_size": 288,
    "queries": 200,
}


def fetch(url: str, dest: Path, tries: int = 4) -> None:
    import time
    part = dest.with_name(dest.name + ".partial")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.replace(dest)
            return
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def fetch_sam3(dest: Path) -> int:
    d = dest / SAM3_CARD["name"] / SAM3_CARD["version"]
    d.mkdir(parents=True, exist_ok=True)
    for name, (path, want) in SAM3_FILES.items():
        f = d / name
        if f.exists() and sha256_file(f) == want:
            continue
        print(f"downloading {SAM3_BASE}{path}")
        fetch(SAM3_BASE + path, f)
        got = sha256_file(f)
        if got != want:
            f.unlink()
            print(f"error: checksum mismatch for SAM 3.1 {name}: {got}", file=sys.stderr)
            return 1
    lic = Path(__file__).resolve().parents[1] / "packaging" / "licenses" / "SAM-LICENSE.txt"
    if sha256_file(lic) != SAM3_LICENSE_SHA256:
        print(f"error: {lic} is not Meta's SAM License text as pinned", file=sys.stderr)
        return 1
    (d / "LICENSE").write_bytes(lic.read_bytes())
    card = dict(SAM3_CARD)
    card["sha256"] = SAM3_FILES[card["weights"]][1]
    card["files"] = {n: w for n, (_, w) in SAM3_FILES.items() if n != card["weights"]} | {"LICENSE": SAM3_LICENSE_SHA256}
    (d / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"ok {d}")
    return 0


# RF-DETR (Roboflow, Apache-2.0): the released COCO weights, checked against the MD5 the rfdetr package publishes, are
# exported to ONNX here (needs PyTorch and the rfdetr package on this staging machine: pip install "rfdetr[onnx]"),
# and kept without their optimizer state as training starting weights (training_weights/rf-detr-<size>-coco.pth).
RFDETR_SIZES_DEFAULT = "small"
RFDETR_LICENSE_URL = "https://raw.githubusercontent.com/roboflow/rf-detr/develop/LICENSE"


def fetch_rfdetr(dest: Path, weights_dir: Path, sizes: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        import rfdetr  # noqa: F401
        from trafficwb import rfdetr_train as rt
    except ImportError as exc:
        print(f"error: RF-DETR needs PyTorch and the rfdetr package here ({exc}); pip install \"rfdetr[train,onnx]\", "
              "or pass --no-rfdetr", file=sys.stderr)
        return 1
    weights_dir.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        info = rt.SIZES[size]
        name = f"rf-detr-{size}"
        slim = weights_dir / rt.weights_name(size)
        full = weights_dir / f"{name}-released.part"
        if not slim.exists():
            print(f"downloading {info['url']}")
            fetch(info["url"], full)
            got = hashlib.md5(full.read_bytes()).hexdigest()
            if got != info["md5"]:
                full.unlink()
                print(f"error: checksum mismatch for RF-DETR {size}: {got}", file=sys.stderr)
                return 1
            rt.slim_checkpoint(full, slim)
            full.unlink()
        print(f"ok {slim}")
        from importlib.metadata import version as _v
        version = _v("rfdetr")
        d = dest / name / version
        if (d / "model.json").exists():
            print(f"ok {d}")
            continue
        tmp = d.with_name(version + ".partial")
        tmp.mkdir(parents=True, exist_ok=True)
        model = rt._model_class(size)(pretrain_weights=str(slim.resolve()), device="cpu")
        model.export(output_dir=str(tmp / "export"), verbose=False)
        (tmp / "export").joinpath(next(p.name for p in (tmp / "export").glob("*.onnx"))).replace(tmp / "model.onnx")
        import shutil
        shutil.rmtree(tmp / "export", ignore_errors=True)
        fetch(RFDETR_LICENSE_URL, tmp / "LICENSE")
        res = info["resolution"]
        card = {"name": name, "version": version, "task": "vehicle_detection", "sha256": sha256_file(tmp / "model.onnx"),
                "license": "Apache-2.0 (RF-DETR code and the released Nano, Small and Medium weights; DINOv2 backbone "
                           "Apache-2.0)",
                "license_url": RFDETR_LICENSE_URL,
                "data_note": "Weights trained on COCO 2017 (after Objects365 pre-training). COCO annotations are CC BY 4.0; "
                             "the underlying Flickr images carry their own terms.",
                "source": f"https://github.com/roboflow/rf-detr ({info['url']}, MD5 {info['md5']}), exported to ONNX "
                          f"with rfdetr {version}",
                "family": "detr", "architecture": "rf-detr", "size": size, "outputs": "boxes_logits",
                "class_ids": "coco91", "input_size": res,
                "input": {"colour": "RGB", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "input_format": f"RGB, stretched to {res}x{res}, ImageNet mean and std",
                "weights": "model.onnx", "files": {"LICENSE": sha256_file(tmp / "LICENSE")}}
        (tmp / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        tmp.replace(d)
        print(f"ok {d}")
    return 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--training-weights", default=str(Path(__file__).resolve().parents[1] / "training_weights"))
    ap.add_argument("--no-sam3", action="store_true", help="skip SAM 3.1 (about 880 MB), e.g. for unit tests")
    ap.add_argument("--no-rfdetr", action="store_true", help="skip RF-DETR (needs PyTorch and rfdetr on this machine)")
    ap.add_argument("--rfdetr-sizes", default=RFDETR_SIZES_DEFAULT,
                    help="comma-separated RF-DETR sizes to fetch: nano, small, medium (default small)")
    a = ap.parse_args(argv)
    for p in PACKS:
        d = Path(a.dest) / p["name"] / p["version"]
        d.mkdir(parents=True, exist_ok=True)
        w = d / "model.onnx"
        if not w.exists() or hashlib.sha256(w.read_bytes()).hexdigest() != p["sha256"]:
            print(f"downloading {p['weights_url']}")
            fetch(p["weights_url"], w)
        got = hashlib.sha256(w.read_bytes()).hexdigest()
        if got != p["sha256"]:
            print(f"error: checksum mismatch for {p['name']}: {got}", file=sys.stderr)
            return 1
        fetch(p["license_url"], d / "LICENSE")
        card = {k: v for k, v in p.items() if k not in ("weights_url",)}
        card["weights"] = "model.onnx"
        (d / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        print(f"ok {d}")
    if not a.no_sam3 and fetch_sam3(Path(a.dest)):
        return 1
    t = Path(a.training_weights)
    t.mkdir(parents=True, exist_ok=True)
    w = t / TRAINING_WEIGHTS["file"]
    if not w.exists() or hashlib.sha256(w.read_bytes()).hexdigest() != TRAINING_WEIGHTS["sha256"]:
        print(f"downloading {TRAINING_WEIGHTS['url']}")
        fetch(TRAINING_WEIGHTS["url"], w)
    got = hashlib.sha256(w.read_bytes()).hexdigest()
    if got != TRAINING_WEIGHTS["sha256"]:
        w.unlink()
        print(f"error: checksum mismatch for {w.name}: {got}", file=sys.stderr)
        return 1
    print(f"ok {w}")
    if not a.no_rfdetr and fetch_rfdetr(Path(a.dest), t, [x.strip() for x in a.rfdetr_sizes.split(",") if x.strip()]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
