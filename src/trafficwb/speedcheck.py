"""Speed check on this workstation (HW-01, P9 budgets): how fast the models run on each backend.

    trafficwb-speedcheck [--images <folder>] [--n 20] [--out <report.txt>]

For every backend ONNX Runtime offers here (CUDA, DirectML, CPU), the vehicle model is
loaded on that backend and run on the same images: your own when --images is given
(the first n), otherwise generated road-like test pictures at 1920x1080. It reports
seconds per image (median and slowest 5%), images per hour, and what that means for a
day of images and for video at 5 analysed frames per second. With the plate packs
installed it also times plate finding and reading per vehicle. Nothing is changed
in any mission; nothing is downloaded. The report is printed and written to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def test_pictures(n: int, w: int = 1920, h: int = 1080) -> list[np.ndarray]:
    """Road-like pictures: grey road, lane lines and car-sized boxes, different in each picture."""
    import cv2
    out = []
    for k in range(n):
        rng = np.random.default_rng(k)
        img = np.full((h, w, 3), 90, np.uint8)
        img += rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
        for x in range(200, w, 400):
            cv2.line(img, (x, h), (w // 2 + (x - w // 2) // 4, h // 3), (210, 210, 210), 6)
        for _ in range(12):
            x, y = int(rng.integers(0, w - 200)), int(rng.integers(h // 3, h - 120))
            s = 0.4 + 1.2 * (y - h / 3) / (h * 2 / 3)
            cv2.rectangle(img, (x, y), (x + int(160 * s), y + int(90 * s)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
        out.append(img)
    return out


def load_images(folder: str | None, n: int) -> tuple[list[np.ndarray], str]:
    if not folder:
        return test_pictures(n), f"{n} generated test pictures, 1920x1080"
    from .views import read_image
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)[:n]
    if not files:
        raise SystemExit(f"No images in {folder}.")
    imgs = [read_image(p) for p in files]
    sizes = sorted({f"{i.shape[1]}x{i.shape[0]}" for i in imgs})
    return imgs, f"{len(imgs)} images from {folder} ({', '.join(sizes)})"


def available_backends() -> list[str]:
    import onnxruntime as ort
    have = ort.get_available_providers()
    return [name for name, prov in (("cuda", "CUDAExecutionProvider"), ("directml", "DmlExecutionProvider"),
                                    ("cpu", "CPUExecutionProvider")) if prov in have]


def time_detector(backend: str, imgs: list[np.ndarray]) -> dict:
    from . import models
    from .detect import YoloxDetector
    t0 = time.perf_counter()
    det = YoloxDetector(models.find("yolox-s"), accelerator=backend)
    load = time.perf_counter() - t0
    rt = det.runtime.record()
    det.detect(imgs[0])                                   # warm-up: the first run includes setup work
    per = []
    found = 0
    for img in imgs:
        t = time.perf_counter()
        found += len(det.detect(img))
        per.append(time.perf_counter() - t)
    q = sorted(per)
    med = statistics.median(q)
    return {"backend": backend, "runs_on": rt["accelerator"], "notes": rt["notes"], "model_load_s": round(load, 2),
            "s_per_image_median": round(med, 3), "s_per_image_p95": round(q[int(0.95 * (len(q) - 1))], 3),
            "images_per_hour": int(3600 / med) if med else None, "vehicles_found": found}


def time_plates(backend: str, imgs: list[np.ndarray]) -> dict | None:
    """Plate finding and reading on vehicle-sized crops (one per image)."""
    try:
        from .plates import default_plate_models
        finder, reader = default_plate_models(accelerator=backend)
    except Exception as exc:  # packs not installed
        return {"skipped": f"plate packs not available ({type(exc).__name__})"}
    crops = [img[img.shape[0] // 2: img.shape[0] // 2 + 240, img.shape[1] // 3: img.shape[1] // 3 + 320] for img in imgs]
    finder.find(crops[0])
    per = []
    for c in crops:
        t = time.perf_counter()
        boxes = finder.find(c)
        plate = c[150:180, 100:220]
        if boxes:
            x1, y1, x2, y2 = (int(v) for v in boxes[0].box)
            if x2 > x1 and y2 > y1:
                plate = c[y1:y2, x1:x2]
        reader.read(plate)
        per.append(time.perf_counter() - t)
    med = statistics.median(per)
    return {"s_per_vehicle_median": round(med, 4), "vehicles_per_hour": int(3600 / med) if med else None}


def report_text(r: dict) -> str:
    L = [f"LanePulse speed check, {r['at']}",
         f"Computer: {r['machine']['system']}, {r['machine']['cpu']}, {r['machine']['cpu_count']} logical CPUs",
         f"Pictures: {r['pictures']}", ""]
    for b in r["backends"]:
        if "error" in b:
            L.append(f"{b['backend']:>9}: could not run ({b['error']})")
            continue
        L.append(f"{b['backend']:>9}: {b['s_per_image_median']:.2f} s per image (slowest 5%: {b['s_per_image_p95']:.2f} s), "
                 f"about {b['images_per_hour']:,} images per hour" + ("" if b["runs_on"] == b["backend"] else
                                                                      f"  [actually ran on {b['runs_on']}]"))
        for n in b["notes"]:
            L.append(f"           note: {n}")
        if b.get("plates") and "skipped" not in b["plates"]:
            L.append(f"           plates: {b['plates']['s_per_vehicle_median'] * 1000:.0f} ms per vehicle")
    best = r.get("fastest")
    if best:
        b = next(x for x in r["backends"] if x["backend"] == best)
        per_day = b["images_per_hour"] * 24
        video = 5 * b["s_per_image_median"]
        L += ["", f"Fastest: {best}. The app picks it by itself (set TWB_ACCELERATOR in start.cmd to force one).",
              f"- Still images: about {per_day:,} images in 24 hours of processing.",
              f"- Video at 5 analysed frames per second, frames of this size: {video:.1f} s of processing per second of video"
              + (" (faster than real time)." if video < 1 else f" (one hour of video takes about {video:.1f} hours).")]
    L += ["", "Times include tiling (3x2 tiles on large pictures), as in normal processing.",
          "Your own pictures give a truer figure: run with --images <folder with ~20 typical images>."]
    return "\n".join(L)


def main(argv=None) -> int:
    from . import safe_console
    safe_console()
    ap = argparse.ArgumentParser(prog="trafficwb-speedcheck", description="Measure model speed on this computer.")
    ap.add_argument("--images", help="folder with typical images (the first --n are used)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", help="write the report here as text, and as .json next to it")
    ap.add_argument("--backends", help="comma-separated subset of cuda,directml,cpu")
    a = ap.parse_args(argv)
    imgs, desc = load_images(a.images, max(3, a.n))
    backends = a.backends.split(",") if a.backends else available_backends()
    r = {"at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "pictures": desc,
         "machine": {"system": f"{platform.system()} {platform.release()}", "cpu": platform.processor() or platform.machine(),
                     "cpu_count": os.cpu_count(), "python": platform.python_version()}, "backends": []}
    for b in backends:
        print(f"Measuring {b}...", file=sys.stderr, flush=True)
        try:
            res = time_detector(b, imgs)
            res["plates"] = time_plates(b, imgs)
        except Exception as exc:
            res = {"backend": b, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        r["backends"].append(res)
    ok = [b for b in r["backends"] if "error" not in b and b["runs_on"] == b["backend"]]
    r["fastest"] = min(ok, key=lambda b: b["s_per_image_median"])["backend"] if ok else None
    text = report_text(r)
    print(text)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\nSaved to {out}")
    return 0 if r["fastest"] else 1


if __name__ == "__main__":
    raise SystemExit(main())