"""Staging-machine only: download model packs and write their model cards.

Run on a connected machine, then copy the resulting models/ folder to the
offline workstation. The application itself never downloads anything.

    python scripts/fetch_models.py [--dest models]

It also fetches the YOLOX-s training starting weights (yolox_s.pth) into training_weights/; the installers put
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


def fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
    ap.add_argument("--training-weights", default=str(Path(__file__).resolve().parents[1] / "training_weights"))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
