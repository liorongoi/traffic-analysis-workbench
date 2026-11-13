"""Staging-machine only: download model packs and write their model cards.

Run on a connected machine, then copy the resulting models/ folder to the
offline workstation. The application itself never downloads anything.

    python scripts/fetch_models.py [--dest models]

Every size of the four fast detectors is fetched by default, so the app can choose the one that suits each computer
(hardware.py): YOLOX nano, tiny, s, m, l, x; RF-DETR nano, small, medium (exported to ONNX here; needs PyTorch and the
rfdetr package; --no-rfdetr skips it); D-FINE nano to xlarge and RT-DETRv2 R18 to R101 (need PyTorch and transformers;
--no-detr skips them). --yolox-sizes, --rfdetr-sizes and --detr-sizes pick fewer; --minimal keeps one of each.
It also fetches SAM 3.1 (sam3.1-w8a16, about 880 MB; --no-sam3 skips it) and the training starting weights
(yolox_<size>.pth, rf-detr-<size>-coco.pth, d-fine-<size>-coco.pth, rt-detr-<size>-coco.pth) into training_weights/;
the installers put them next to their own Python (share/lanepulse), so training on the learning page never goes
online. Files already in --dest and --training-weights are checked and kept, so with everything staged it runs
without a network.
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
        "architecture": "yolox",
        "size": "s",
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

# The other released YOLOX sizes (same release, license and data as YOLOX-s): ONNX for detection, PyTorch checkpoint
# for training. Nano and tiny take 416x416 pictures. Nano (depthwise layers) runs but is not trained here.
YOLOX_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
YOLOX_OTHER = {     # size: (onnx sha256, input size, pth sha256 or None)
    "nano": ("c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d", 416, None),
    "tiny": ("427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7", 416,
             "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0"),
    "m": ("21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f", 640,
          "60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6"),
    "l": ("7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411", 640,
          "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917"),
    "x": ("c892d7aaf1c4746d8a4d675bec669a4db4f434b4ee1efb654bc9b353379c7c55", 640,
          "5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf"),
}
YOLOX_ALL = ["nano", "tiny", "s", "m", "l", "x"]


def yolox_pack(size: str) -> dict:
    if size == "s":
        return PACKS[0]
    sha, res, _ = YOLOX_OTHER[size]
    return dict(PACKS[0], name=f"yolox-{size}", weights_url=f"{YOLOX_RELEASE}yolox_{size}.onnx", sha256=sha,
                input_size=res, size=size)


# Starting weights for training the vehicle detector (PyTorch checkpoint of the same released YOLOX-s).
TRAINING_WEIGHTS = {
    "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "sha256": "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    "file": "yolox_s.pth",
}


def fetch_yolox_weights(weights_dir: Path, size: str) -> int:
    """yolox_<size>.pth for tiny, m, l, x: the released checkpoint, checked by its SHA-256, then kept without its
    optimizer state when PyTorch is here (half the size; trainer and loader read only its "model")."""
    _, _, want = YOLOX_OTHER[size]
    w = weights_dir / f"yolox_{size}.pth"
    if want is None or w.is_file():
        return 0
    part = weights_dir / f"yolox_{size}.released.part"
    print(f"downloading {YOLOX_RELEASE}yolox_{size}.pth")
    fetch(f"{YOLOX_RELEASE}yolox_{size}.pth", part)
    got = sha256_file(part)
    if got != want:
        part.unlink()
        print(f"error: checksum mismatch for yolox_{size}.pth: {got}", file=sys.stderr)
        return 1
    try:
        import torch
        ck = torch.load(part, map_location="cpu", weights_only=True)
        torch.save({"model": ck["model"]}, w)
        part.unlink()
    except ImportError:
        part.replace(w)
    print(f"ok {w}")
    return 0


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
RFDETR_SIZES_DEFAULT = "all"
RFDETR_ALL = ["nano", "small", "medium"]


def _staged(dest: Path, weights_dir: Path, name: str, weights: str) -> bool:
    """A pack and its training weights already here (copied in for an offline build): nothing to fetch or export."""
    return (weights_dir / weights).is_file() and any((dest / name).glob("*/model.json"))
RFDETR_LICENSE_URL = "https://raw.githubusercontent.com/roboflow/rf-detr/develop/LICENSE"


def fetch_rfdetr(dest: Path, weights_dir: Path, sizes: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sizes = [s for s in sizes if not _staged(dest, weights_dir, f"rf-detr-{s}", f"rf-detr-{s}-coco.pth")]
    if not sizes:
        return 0
    try:
        import rfdetr  # noqa: F401
        from trafficwb import rfdetr_train as rt
    except ImportError as exc:
        # Exporting RF-DETR needs PyTorch and rfdetr on this machine. Without them the other packs are still written and
        # the installation offers RF-DETR as "not installed here"; --require-rfdetr makes this an error instead.
        print(f"warning: RF-DETR skipped: it needs PyTorch and the rfdetr package here ({exc}); "
              "pip install \"rfdetr[train]\" onnx", file=sys.stderr)
        return 2
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


# D-FINE and RT-DETRv2 (Apache-2.0): the released COCO weights at a fixed commit, checked by SHA-256 (detr_train.ARCHS),
# kept as training starting weights (training_weights/<arch>-<size>-coco.pth) and exported to ONNX as packs the app
# runs (models/<arch>-<size>/coco). Needs PyTorch and transformers on this staging machine.
DETR_SIZES_DEFAULT = "all"
DETR_ALL = ["d-fine:nano", "d-fine:small", "d-fine:medium", "d-fine:large", "d-fine:xlarge",
            "rt-detr:r18", "rt-detr:r34", "rt-detr:r50", "rt-detr:r101"]


def fetch_detr(dest: Path, weights_dir: Path, wanted: list[str]) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    wanted = [w for w in wanted if not _staged(dest, weights_dir, w.replace(":", "-"), w.replace(":", "-") + "-coco.pth")]
    if not wanted:
        return 0
    try:
        from trafficwb import detr_train as dt
        if not dt.available()["ok"]:
            raise ImportError(dt.available()["reason"])
    except ImportError as exc:
        print(f"warning: D-FINE and RT-DETRv2 skipped: they need PyTorch and transformers here ({exc}); "
              "pip install torch transformers onnx", file=sys.stderr)
        return 2
    for item in wanted:
        arch, _, size = item.partition(":")
        if arch not in dt.ARCHS or size not in dt.ARCHS[arch]["sizes"]:
            print(f"error: unknown D-FINE / RT-DETRv2 size {item}", file=sys.stderr)
            return 1
        try:
            w = dt.fetch_weights(arch, size, weights_dir)
        except dt.TrainError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"ok {w}")
        print(f"ok {dt.released_pack(arch, size, w, dest)}")
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
    ap.add_argument("--require-rfdetr", action="store_true", help="fail when RF-DETR cannot be exported here")
    ap.add_argument("--rfdetr-sizes", default=RFDETR_SIZES_DEFAULT,
                    help="comma-separated RF-DETR sizes to fetch: nano, small, medium, or all (the default)")
    ap.add_argument("--yolox-sizes", default="all",
                    help="comma-separated YOLOX sizes to fetch: nano, tiny, s, m, l, x, or all (the default)")
    ap.add_argument("--minimal", action="store_true",
                    help="only YOLOX-s, RF-DETR Small, D-FINE Small and RT-DETRv2 R18 (quick test builds)")
    ap.add_argument("--no-detr", action="store_true", help="skip D-FINE and RT-DETRv2 (need PyTorch and transformers)")
    ap.add_argument("--require-detr", action="store_true", help="fail when D-FINE and RT-DETRv2 cannot be fetched here")
    ap.add_argument("--detr-sizes", default=DETR_SIZES_DEFAULT,
                    help="comma-separated arch:size to fetch, e.g. d-fine:small,rt-detr:r18, or all (the default)")
    a = ap.parse_args(argv)
    if a.minimal:
        a.yolox_sizes, a.rfdetr_sizes, a.detr_sizes = "s", "small", "d-fine:small,rt-detr:r18"
    every = {"yolox": YOLOX_ALL, "rfdetr": RFDETR_ALL, "detr": DETR_ALL}

    def sizes(text: str, kind: str) -> list[str]:
        items = [x.strip() for x in text.split(",") if x.strip()]
        return list(every[kind]) if items == ["all"] else items

    yolox = sizes(a.yolox_sizes, "yolox")
    bad = [x for x in yolox if x not in YOLOX_ALL]
    if bad:
        print(f"error: unknown YOLOX size {', '.join(bad)}", file=sys.stderr)
        return 1
    for p in [*(yolox_pack(z) for z in yolox if z != "s"), *PACKS]:
        d = Path(a.dest) / p["name"] / p["version"]
        d.mkdir(parents=True, exist_ok=True)
        w = d / "model.onnx"
        if not w.exists() or sha256_file(w) != p["sha256"]:
            print(f"downloading {p['weights_url']}")
            fetch(p["weights_url"], w)
        got = sha256_file(w)
        if got != p["sha256"]:
            print(f"error: checksum mismatch for {p['name']}: {got}", file=sys.stderr)
            return 1
        if not (d / "LICENSE").is_file():                 # staged already: no network needed
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
    if not w.exists() or sha256_file(w) != TRAINING_WEIGHTS["sha256"]:
        print(f"downloading {TRAINING_WEIGHTS['url']}")
        fetch(TRAINING_WEIGHTS["url"], w)
    got = sha256_file(w)
    if got != TRAINING_WEIGHTS["sha256"]:
        w.unlink()
        print(f"error: checksum mismatch for {w.name}: {got}", file=sys.stderr)
        return 1
    print(f"ok {w}")
    for z in yolox:
        if z != "s" and fetch_yolox_weights(t, z):
            return 1
    if not a.no_rfdetr:
        r = fetch_rfdetr(Path(a.dest), t, sizes(a.rfdetr_sizes, "rfdetr"))
        if r == 1 or (r == 2 and a.require_rfdetr):
            return 1
    if not a.no_detr:
        r = fetch_detr(Path(a.dest), t, sizes(a.detr_sizes, "detr"))
        if r == 1 or (r == 2 and a.require_detr):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
