"""Staging-machine only: download model packs and write their model cards.

Run on a connected machine, then copy the resulting models/ folder to the
offline workstation. The application itself never downloads anything.

    python scripts/fetch_models.py [--dest models]
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
]


def fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
