"""Benchmarks on public datasets: detection, tracking and unique counts, run with LanePulse's own code.

Development and staging machines only; the application never downloads anything. Needs the `bench` extra
(pycocotools, trackeval) and the model packs (YOLOX-s, and SAM 3.1 for the sam3 and both detectors).

    python scripts/benchmark.py detect --suite coco --data <dir> --detector standard [--limit 300] [--out docs/benchmarks/results]
    python scripts/benchmark.py track  --data <MOT17 train dir> --detector standard --rate app
    python scripts/benchmark.py report --out docs/benchmarks/results

Suites and where their files come from (see docs/benchmarks/README.md for licences and links):

    coco      COCO val2017: images/val2017/*.jpg and annotations/instances_val2017.json
    visdrone  VisDrone2019-DET val: images/*.jpg and annotations/*.txt (drone, top-down and oblique)
    hituav    HIT-UAV thermal (drone, infrared): normal_json/test/*.jpg and normal_json/annotations/test.json
    mot17     MOT17 train sequences (people, fixed and moving cameras): <seq>/img1, <seq>/gt/gt.txt, <seq>/seqinfo.ini

Detectors are built the way a mission builds them: the standard detector (YOLOX-s, score 0.30, 3x2 tiles,
cross-class merge), SAM 3.1 with the default vehicle words, or both merged. People are looked for as a mission's own
"person" class. ``--detector published`` runs YOLOX-s the way its authors evaluate it (all 80 COCO classes, score
0.001, no tiles), only to check that the model and its preprocessing reproduce the published COCO figure.

Every result is written as JSON, with the dataset, the image or sequence list, the settings and the time taken,
so a run can be repeated and compared later.
"""
from __future__ import annotations

import argparse
import configparser
import contextlib
import io
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# The classes evaluated, with COCO's own category ids, so every suite shares one scheme.
CATS = {"person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "bus": 6, "truck": 8}
VEHICLES = ("bicycle", "car", "motorcycle", "bus", "truck")
VEHICLE_ID = 100                      # every vehicle class as one ("vehicle, any class")
MAX_DETS = {"coco": 100, "visdrone": 500, "hituav": 100}


# ---------------------------------------------------------------- ground truth

def coco_gt(data: Path) -> dict:
    """COCO val2017 restricted to the evaluated classes (all images kept, crowd boxes kept as crowd)."""
    d = json.loads((data / "annotations" / "instances_val2017.json").read_text())
    keep = set(CATS.values())
    return {"images": [{"id": i["id"], "file": str(data / "images" / "val2017" / i["file_name"]),
                        "width": i["width"], "height": i["height"]} for i in d["images"]],
            "annotations": [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
                             "iscrowd": a["iscrowd"], "area": a["area"]}
                            for a in d["annotations"] if a["category_id"] in keep],
            "coco80": {c["id"]: c["name"] for c in d["categories"]}}


# VisDrone categories: 0 ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
# 8 awning-tricycle, 9 bus, 10 motor, 11 others. Vans count as cars, as COCO and LanePulse count them.
VISDRONE_MAP = {1: "person", 2: "person", 3: "bicycle", 4: "car", 5: "car", 6: "truck", 9: "bus", 10: "motorcycle"}
VISDRONE_IGNORE_VEHICLE = {7, 8, 11}   # not one of LanePulse's classes: a box on one is neither right nor wrong


def visdrone_gt(data: Path) -> dict:
    images, anns = [], []
    for k, ip in enumerate(sorted((data / "images").glob("*.jpg")), 1):
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        images.append({"id": k, "file": str(ip), "width": w, "height": h})
        anns += visdrone_annotations(k, (data / "annotations" / (ip.stem + ".txt")).read_text())
    return {"images": images, "annotations": anns}


def visdrone_annotations(image_id: int, text: str) -> list[dict]:
    out = []
    for line in text.strip().splitlines():
        v = [int(x) for x in line.strip().rstrip(",").split(",")[:6]]
        x, y, w, h, _score, cat = v
        if w <= 0 or h <= 0:
            continue
        box = [float(x), float(y), float(w), float(h)]
        if cat in VISDRONE_MAP:
            out.append({"image_id": image_id, "category_id": CATS[VISDRONE_MAP[cat]], "bbox": box, "iscrowd": 0,
                        "area": w * h})
        elif cat == 0 or cat in VISDRONE_IGNORE_VEHICLE:
            # An ignored region hides every class; tricycles and "others" only excuse vehicle boxes.
            for name in (CATS if cat == 0 else VEHICLES):
                out.append({"image_id": image_id, "category_id": CATS[name], "bbox": box, "iscrowd": 1,
                            "area": w * h})
    return out


# HIT-UAV categories: 0 Person, 1 Car, 2 Bicycle, 3 OtherVehicle, 4 DontCare.
HITUAV_MAP = {0: "person", 1: "car", 2: "bicycle"}


def hituav_gt(data: Path, split: str = "test") -> dict:
    d = json.loads((data / "normal_json" / "annotations" / f"{split}.json").read_text())
    images = [{"id": i["id"], "file": str(data / "normal_json" / split / i["filename"]), "width": i["width"],
               "height": i["height"], **hituav_conditions(i["filename"])} for i in d["images"]]
    return {"images": images, "annotations": hituav_annotations(d["annotation"])}


def hituav_annotations(raw: list[dict]) -> list[dict]:
    out = []
    for a in raw:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        c = a["category_id"]
        if c in HITUAV_MAP:
            out.append({"image_id": a["image_id"], "category_id": CATS[HITUAV_MAP[c]], "bbox": [x, y, w, h],
                        "iscrowd": 0, "area": w * h})
        elif c == 3:
            # "Other vehicle" (vans, trucks, buses together): any vehicle class may claim it without penalty, and it
            # counts in full in the any-vehicle score.
            out.append({"image_id": a["image_id"], "category_id": VEHICLE_ID, "bbox": [x, y, w, h], "iscrowd": 0,
                        "area": w * h, "other_vehicle": True})
        elif c == 4:
            for name in CATS:
                out.append({"image_id": a["image_id"], "category_id": CATS[name], "bbox": [x, y, w, h],
                            "iscrowd": 1, "area": w * h})
    return out


def hituav_conditions(filename: str) -> dict:
    """HIT-UAV names encode <0 day | 1 night>_<altitude m>_<camera pitch deg>_..."""
    parts = Path(filename).stem.split("_")
    try:
        return {"night": parts[0] == "1", "altitude_m": int(parts[1]), "pitch_deg": int(parts[2])}
    except (IndexError, ValueError):
        return {}


# ---------------------------------------------------------------- evaluation

def _coco_api(images: list[dict], anns: list[dict], cats: dict[str, int]):
    from pycocotools.coco import COCO
    c = COCO()
    c.dataset = {"images": [{"id": i["id"], "width": i["width"], "height": i["height"]} for i in images],
                 "annotations": [{**a, "id": k} for k, a in enumerate(anns, 1)],
                 "categories": [{"id": v, "name": n} for n, v in cats.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def class_view(anns: list[dict]) -> list[dict]:
    """Ground truth for per-class AP: an "other vehicle" box becomes a crowd box of every vehicle class."""
    out = []
    for a in anns:
        if a.get("other_vehicle"):
            out += [{**a, "category_id": CATS[n], "iscrowd": 1} for n in VEHICLES]
        else:
            out.append(a)
    return out


def vehicle_view(anns: list[dict]) -> list[dict]:
    """Ground truth for "vehicle, any class": every vehicle box under one id; vehicle crowd regions kept once."""
    out, seen = [], set()
    for a in anns:
        if a["category_id"] == VEHICLE_ID or a["category_id"] in {CATS[n] for n in VEHICLES}:
            if a["iscrowd"]:
                key = (a["image_id"], tuple(a["bbox"]))
                if key in seen:
                    continue
                seen.add(key)
            out.append({**a, "category_id": VEHICLE_ID})
    return out


def coco_eval(images, anns, dets, cats: dict[str, int], max_dets: int) -> dict:
    """AP (0.5:0.95), AP50, AP by size and AR at ``max_dets``, overall and per class."""
    from pycocotools.cocoeval import COCOeval
    gt = _coco_api(images, anns, cats)
    ids = set(cats.values())
    d = [x for x in dets if x["category_id"] in ids]
    if not d:
        return {"AP": 0.0, "AP50": 0.0, "per_class": {n: {"AP": 0.0, "AP50": 0.0} for n in cats}}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(d)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [i["id"] for i in images]
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]                      # [T, R, K, A, M]; M = -1 is max_dets

    def ap(t=slice(None), area=0):
        p = prec[t, :, :, area, -1]
        return float(np.mean(p[p > -1])) if (p > -1).any() else -1.0
    rec = ev.eval["recall"][:, :, 0, -1]
    # pycocotools' own summary reads AP (0.5:0.95) and AP by size at 100 boxes per image whatever maxDets says, so
    # they are taken from the accumulated tables at max_dets (VisDrone's crowded frames use 500).
    out = {"AP": ap(), "AP50": ap(0), "AP75": ap(5), "AP_small": ap(area=1), "AP_medium": ap(area=2),
           "AP_large": ap(area=3), "AR": float(np.mean(rec[rec > -1])) if (rec > -1).any() else -1.0,
           "max_dets": max_dets, "per_class": {}}
    for k, cid in enumerate(ev.params.catIds):
        name = next(n for n, v in cats.items() if v == cid)
        p = prec[:, :, k, 0, -1]
        p50 = prec[0, :, k, 0, -1]
        n_gt = sum(1 for a in anns if a["category_id"] == cid and not a["iscrowd"])
        out["per_class"][name] = {"AP": float(np.mean(p[p > -1])) if (p > -1).any() else None,
                                  "AP50": float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else None,
                                  "gt": n_gt}
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and v >= 0 else v) for k, v in out.items()}


def operating_point(images, anns, dets, cat: int, score_min: float, iou: float = 0.5) -> dict:
    """Precision, recall and F1 at IoU ``iou`` for boxes the application keeps (score >= score_min): what an
    operator sees in review. A box on a crowd or ignored region is neither right nor wrong."""
    by_img_gt: dict[int, list] = {}
    for a in anns:
        if a["category_id"] == cat:
            by_img_gt.setdefault(a["image_id"], []).append(a)
    tp = fp = 0
    n_gt = sum(1 for a in anns if a["category_id"] == cat and not a["iscrowd"])
    by_img_dt: dict[int, list] = {}
    for d in dets:
        if d["category_id"] == cat and d["score"] >= score_min:
            by_img_dt.setdefault(d["image_id"], []).append(d)
    for img in images:
        gts = by_img_gt.get(img["id"], [])
        real = [g for g in gts if not g["iscrowd"]]
        crowd = [g for g in gts if g["iscrowd"]]
        used = set()
        for d in sorted(by_img_dt.get(img["id"], []), key=lambda d: -d["score"]):
            best, bj = iou, None
            for j, g in enumerate(real):
                if j in used:
                    continue
                v = _iou_xywh(d["bbox"], g["bbox"])
                if v >= best:
                    best, bj = v, j
            if bj is not None:
                used.add(bj)
                tp += 1
            elif any(_ioa_xywh(d["bbox"], g["bbox"]) >= 0.5 for g in crowd):
                continue
            else:
                fp += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"score_min": score_min, "iou": iou, "gt": n_gt, "tp": tp, "fp": fp, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _iou_xywh(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def _ioa_xywh(a, region) -> float:
    ix = max(0.0, min(a[0] + a[2], region[0] + region[2]) - max(a[0], region[0]))
    iy = max(0.0, min(a[1] + a[3], region[1] + region[3]) - max(a[1], region[1]))
    return ix * iy / (a[2] * a[3]) if a[2] * a[3] > 0 else 0.0


# ---------------------------------------------------------------- detectors

def build_detector(kind: str, models_root: Path | None, people: bool = True, coco80: dict | None = None,
                   words: dict | None = None, standard_pack: str = "yolox-s"):
    """A detector as a mission builds it. ``people``: also look for a "person" class of the mission's own."""
    from trafficwb import models, sam3
    from trafficwb.combined import CombinedDetector
    from trafficwb.detect import DetectorSettings, YoloxDetector, standard_classes

    own = ["person"] if people else []
    std = sam = None
    if kind in ("standard", "both", "published"):
        pack = models.find(standard_pack if kind != "published" else "yolox-s", root=models_root)
        if kind == "published":
            names = [coco80[k] for k in sorted(coco80)]
            std = YoloxDetector(pack, DetectorSettings(score_threshold=0.001, nms_iou=0.65, cross_class_iou=1.01,
                                                       tiles=(1, 1)))
            std = std.with_classes(dict(enumerate(names)))
            return std, std.describe()
        from trafficwb import detect as _detect
        make = getattr(_detect, "make_detector", None)     # picks the class from the pack's family, where present
        std = (make(pack) if make else YoloxDetector(pack)).with_classes(standard_classes(own))
    if kind in ("sam3", "both"):
        prompts = dict(sam3.vehicle_prompts(words))
        if people:
            prompts["person"] = ("person",)
        sam = sam3.Sam3Detector(sam3.engine(models_root), prompts=prompts)
    if kind == "standard":
        return std, std.describe()
    if kind == "sam3":
        return sam, sam.describe()
    comb = CombinedDetector(std, std.pack.card, sam, sam.pack.card)
    return comb, comb.describe()


def detections_to_coco(image_id: int, dets, name_to_id: dict[str, int]) -> list[dict]:
    out = []
    for d in dets:
        cid = name_to_id.get(d.cls)
        if cid is None:
            continue
        x1, y1, x2, y2 = d.box
        out.append({"image_id": image_id, "category_id": cid, "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(d.score), "found_by": d.found_by})
    return out


def subset(images: list[dict], anns: list[dict], limit: int | None, seed: int) -> list[dict]:
    """A repeatable sample of images holding at least one vehicle (all images when ``limit`` is None)."""
    if not limit or limit >= len(images):
        return images
    veh = {CATS[n] for n in VEHICLES} | {VEHICLE_ID}
    has = {a["image_id"] for a in anns if a["category_id"] in veh and not a["iscrowd"]}
    pool = sorted((i for i in images if i["id"] in has), key=lambda i: i["id"])
    return sorted(random.Random(seed).sample(pool, min(limit, len(pool))), key=lambda i: i["id"])


def run_detect(a) -> dict:
    data = Path(a.data)
    gt = {"coco": coco_gt, "visdrone": visdrone_gt, "hituav": hituav_gt}[a.suite](data)
    images = subset(gt["images"], gt["annotations"], a.limit, a.seed)
    det, desc = build_detector(a.detector, Path(a.models) if a.models else None, coco80=gt.get("coco80"),
                               standard_pack=a.standard_pack)
    names = {v: k for k, v in gt["coco80"].items()} if a.detector == "published" else dict(CATS)
    if a.detector != "published":
        names.update({"van": CATS["car"]})
    dets, secs = [], []
    t_all = time.monotonic()
    for k, im in enumerate(images, 1):
        img = cv2.imread(im["file"])
        t0 = time.monotonic()
        out = det.detect(img)
        secs.append(time.monotonic() - t0)
        dets += detections_to_coco(im["id"], out, names)
        if k % 25 == 0 or k == len(images):
            print(f"  {a.suite}/{a.detector}: {k}/{len(images)} images, {np.mean(secs):.2f} s each", flush=True)
    ids = {i["id"] for i in images}
    anns = [x for x in gt["annotations"] if x["image_id"] in ids]
    res: dict = {"kind": "detect", "suite": a.suite, "detector": a.detector, "images": len(images),
                 "image_ids": [i["id"] for i in images] if a.limit else "all", "seed": a.seed,
                 "model": desc.get("model"), "preprocessing": desc.get("preprocessing"),
                 "seconds_per_image": round(float(np.mean(secs)), 3), "wall_s": round(time.monotonic() - t_all, 1),
                 "machine": machine()}
    md = MAX_DETS[a.suite]
    if a.detector == "published":
        coco80 = gt["coco80"]
        full = json.loads((data / "annotations" / "instances_val2017.json").read_text())["annotations"]
        res["all80"] = coco_eval(images, [x for x in full if x["image_id"] in ids], dets, {n: i for i, n in coco80.items()}, 100)
        res["all80"].pop("per_class", None)
    res["classes"] = coco_eval(images, class_view(anns), dets, CATS, md)
    vdets = [{**d, "category_id": VEHICLE_ID} for d in dets if d["category_id"] in {CATS[n] for n in VEHICLES}]
    res["vehicle_any"] = coco_eval(images, vehicle_view(anns), vdets, {"vehicle": VEHICLE_ID}, md)
    res["vehicle_any"].pop("per_class", None)
    score_min = 0.30 if a.detector in ("standard", "published") else 0.0
    res["operating_point"] = {
        "vehicle_any": operating_point(images, vehicle_view(anns), vdets, VEHICLE_ID, score_min),
        "person": operating_point(images, class_view(anns), dets, CATS["person"], score_min)}
    if a.detector == "both":
        res["found_by"] = {k: sum(1 for d in dets if d.get("found_by") == k) for k in ("standard", "sam3", "both")}
    if a.suite == "hituav":
        res["conditions"] = {}
        for label, pick in (("day", lambda i: not i.get("night")), ("night", lambda i: i.get("night")),
                            ("altitude <= 80 m", lambda i: i.get("altitude_m", 999) <= 80),
                            ("altitude >= 100 m", lambda i: i.get("altitude_m", 0) >= 100)):
            sub = [i for i in images if pick(i)]
            ids = {i["id"] for i in sub}
            if not sub:
                continue
            v = coco_eval(sub, [x for x in vehicle_view(anns) if x["image_id"] in ids],
                          [d for d in vdets if d["image_id"] in ids], {"vehicle": VEHICLE_ID}, md)
            p = coco_eval(sub, [x for x in class_view(anns) if x["image_id"] in ids],
                          [d for d in dets if d["image_id"] in ids], {"person": CATS["person"]}, md)
            res["conditions"][label] = {"images": len(sub), "vehicle_AP50": v["AP50"], "person_AP50": p["AP50"]}
    return res


# ---------------------------------------------------------------- tracking (MOT17)

MOT17_MOVING = {"MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"}   # camera on a person or a vehicle


def seqinfo(seq: Path) -> dict:
    c = configparser.ConfigParser()
    c.read(seq / "seqinfo.ini")
    s = c["Sequence"]
    n_img = len(list((seq / "img1").glob("*.jpg")))
    # The folder name is the sequence's name for TrackEval (MOT17 folders may drop the detector suffix of "name").
    return {"name": seq.name, "fps": float(s["frameRate"]), "width": int(s["imWidth"]), "height": int(s["imHeight"]),
            "frames": min(int(s["seqLength"]), n_img)}


def sample_frames(n: int, fps: float, grid_s: float | None) -> list[int]:
    """Frame numbers (1-based) the application analyses: the first frame at or after each grid point, as
    trafficwb.video.frames picks them. ``grid_s`` None: every frame."""
    if grid_s is None:
        return list(range(1, n + 1))
    out, nxt, eps = [], 0.0, 1e-6
    for f in range(1, n + 1):
        t = (f - 1) / fps
        if t + eps < nxt:
            continue
        out.append(f)
        nxt += (int((t + eps - nxt) // grid_s) + 1) * grid_s
    return out


def read_gt(seq: Path) -> np.ndarray:
    return np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)


def write_gt_subset(seq: Path, info: dict, keep: list[int], dest: Path) -> dict:
    """The ground truth restricted to the analysed frames, renumbered 1..N, in TrackEval's MOT layout."""
    gt = read_gt(seq)
    idx = {f: k for k, f in enumerate(keep, 1)}
    rows = gt[np.isin(gt[:, 0].astype(int), keep)].copy()
    rows[:, 0] = [idx[int(f)] for f in rows[:, 0]]
    d = dest / info["name"]
    (d / "gt").mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "gt" / "gt.txt", rows, delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.6f"])
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={info['name']}\nimDir=img1\nframeRate={info['fps']}\nseqLength={len(keep)}\n"
        f"imWidth={info['width']}\nimHeight={info['height']}\nimExt=.jpg\n")
    people = rows[(rows[:, 7] == 1) & (rows[:, 6] == 1)]
    return {"gt_people": int(len(np.unique(people[:, 1]))), "gt_boxes": int(len(people))}


def track_sequence(seq: Path, info: dict, keep: list[int], kind: str, detector, grid_s: float, params,
                   sam_every: int) -> tuple[list, dict]:
    """LanePulse's own segment analysis (trafficwb.video_segments.analyse) over an image sequence, as one segment
    covering the whole recording. Returns MOT rows (frame index in ``keep``, id, box) of confirmed tracks."""
    from trafficwb import video_segments as vs
    from trafficwb.video import Frame

    paths = sorted((seq / "img1").glob("*.jpg"))
    fps = info["fps"]

    def frames(_path, interval_s, after_t=None, next_sample_t=0.0, until_t=None):
        for k, f in enumerate(keep):
            # analyse() derives the grid index (which frames SAM 3.1 looks at) from the next grid point
            yield Frame((f - 1) / fps, cv2.imread(str(paths[f - 1]))), (k + 1) * interval_s

    seg = vs.Segment(0, 0.0, math.inf, 0.0, math.inf)
    orig = vs.frames
    vs.frames = frames
    try:
        t0 = time.monotonic()
        res = vs.analyse(seg, seq, info["width"], info["height"], vs.make_detect(kind, detector), grid_s, params,
                         info["name"][:8] in MOT17_MOVING, sam_every)
        wall = time.monotonic() - t0
    finally:
        vs.frames = orig
    t_to_idx = {round((f - 1) / fps, 6): k for k, f in enumerate(keep, 1)}
    rows, n_conf, dup = [], 0, 0
    for tid, p in enumerate(res.pieces, 1):
        if p.family != "person" or len(p.obs) < params.min_hits:
            continue
        n_conf += 1
        dup += bool(p.related)
        for t, box, _cls, _score, _fb in p.obs:
            x1, y1, x2, y2 = box
            rows.append((t_to_idx[round(t, 6)], tid, x1, y1, x2 - x1, y2 - y1))
    return rows, {"tracks_confirmed": n_conf, "flagged_possible_duplicate": dup, "wall_s": round(wall, 1),
                  "seconds_per_frame": round(wall / max(1, len(keep)), 3)}


def trackeval_mot(gt_dir: Path, trk_dir: Path, seqs: list[str], name: str) -> dict:
    import trackeval
    cfg = trackeval.Evaluator.get_default_eval_config()
    cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False,
                "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "USE_PARALLEL": False, "DISPLAY_LESS_PROGRESS": True})
    seqmap = gt_dir / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({"GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trk_dir), "TRACKERS_TO_EVAL": [name],
                 "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "SKIP_SPLIT_FOL": True, "SEQMAP_FILE": str(seqmap),
                 "TRACKER_SUB_FOLDER": "data", "PRINT_CONFIG": False})
    with contextlib.redirect_stdout(io.StringIO()):
        ev = trackeval.Evaluator(cfg)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    with contextlib.redirect_stdout(io.StringIO()):
        out, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)], metrics)   # quiet: it prints its config
    r = out["MotChallenge2DBox"][name]

    def pick(m):
        return {"HOTA": float(np.mean(m["HOTA"]["HOTA"])), "DetA": float(np.mean(m["HOTA"]["DetA"])),
                "AssA": float(np.mean(m["HOTA"]["AssA"])), "MOTA": float(m["CLEAR"]["MOTA"]),
                "IDF1": float(m["Identity"]["IDF1"]), "IDSW": int(m["CLEAR"]["IDSW"]),
                "MT": int(m["CLEAR"]["MT"]), "ML": int(m["CLEAR"]["ML"]), "FP": int(m["CLEAR"]["CLR_FP"]),
                "FN": int(m["CLEAR"]["CLR_FN"]), "Frag": int(m["CLEAR"]["Frag"])}
    per = {s: pick(r[s]["pedestrian"]) for s in seqs}
    return {"combined": pick(r["COMBINED_SEQ"]["pedestrian"]), "per_sequence": per}


def run_track(a) -> dict:
    import tempfile
    from trafficwb.tracker import TrackerParams
    from trafficwb.video_process import VideoSettings, _low_threshold

    data = Path(a.data)
    seqs = sorted(p for p in data.iterdir() if (p / "seqinfo.ini").is_file())
    if a.sequences:
        seqs = [s for s in seqs if s.name in a.sequences.split(",")]
    det, desc = build_detector(a.detector, Path(a.models) if a.models else None, standard_pack=a.standard_pack)
    vset = VideoSettings()
    sam_s = 1.0
    grid_s = vset.interval_s if a.rate == "app" else None
    params = TrackerParams()
    if a.detector == "sam3":
        grid_s = max(vset.interval_s, sam_s) if a.rate == "app" else None
    if a.detector != "sam3":
        det = _low_threshold(det, vset.low_score)
    if a.detector != "standard":
        params = replace(params, jump_per_s=3.0)
    work = Path(tempfile.mkdtemp(prefix="twb-mot-"))
    gt_dir, trk = work / "gt", work / "trackers" / "lanepulse" / "data"
    trk.mkdir(parents=True)
    counts, timing = {}, {}
    for seq in seqs:
        info = seqinfo(seq)
        g = grid_s if grid_s is not None else 1.0 / info["fps"]
        keep = sample_frames(info["frames"], info["fps"], grid_s)
        p = params if grid_s is not None else replace(params, frame_s=1.0 / info["fps"])
        sam_every = max(1, round(sam_s / g)) if a.detector == "both" else 1
        gtinfo = write_gt_subset(seq, info, keep, gt_dir)
        rows, st = track_sequence(seq, info, keep, a.detector, det, g, p, sam_every)
        with open(trk / f"{info['name']}.txt", "w") as f:
            for fr, tid, x, y, w, h in sorted(rows):
                f.write(f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        counts[info["name"]] = {**gtinfo, "counted": st["tracks_confirmed"],
                                "flagged_possible_duplicate": st["flagged_possible_duplicate"],
                                "frames_analysed": len(keep), "frames_in_subset": info["frames"],
                                "moving_camera": info["name"][:8] in MOT17_MOVING}
        timing[info["name"]] = st["seconds_per_frame"]
        print(f"  {info['name']}: {len(keep)} frames, {st['tracks_confirmed']} people counted "
              f"(ground truth {gtinfo['gt_people']}), {st['seconds_per_frame']} s/frame", flush=True)
    names = [s.name for s in seqs]
    m = trackeval_mot(gt_dir, work / "trackers", names, "lanepulse")
    tot_gt = sum(c["gt_people"] for c in counts.values())
    tot = sum(c["counted"] for c in counts.values())
    return {"kind": "track", "suite": "mot17", "detector": a.detector, "rate": a.rate,
            "grid_s": grid_s, "sequences": names, "metrics": m, "counts": counts,
            "count_total": {"gt_people": tot_gt, "counted": tot, "ratio": round(tot / tot_gt, 3) if tot_gt else None},
            "tracker": asdict(params), "model": desc.get("model"), "seconds_per_frame": timing,
            "machine": machine()}


# ---------------------------------------------------------------- report

def machine() -> dict:
    import os
    return {"os": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()}


def result_name(r: dict) -> str:
    pack = r.get("standard_pack", "yolox-s")
    det = r["detector"] if pack == "yolox-s" or r["detector"] in ("sam3", "published") else f"{r['detector']}-{pack}"
    if r["kind"] == "track":
        return f"track-{r['suite']}-{det}-{r['rate']}.json"
    n = "all" if r["image_ids"] == "all" else f"n{r['images']}"
    return f"detect-{r['suite']}-{det}-{n}.json"


def pct(v) -> str:
    return "–" if v is None or (isinstance(v, float) and v < 0) else f"{100 * v:.1f}"


def run_report(a) -> str:
    rs = [json.loads(p.read_text()) for p in sorted(Path(a.out).glob("*.json"))]
    lines = ["| Suite | Images | Detector | Vehicle AP50 (any class) | Vehicle AP | Car AP50 | Person AP50 | "
             "Vehicle P / R at app threshold | s per image |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        if r["kind"] != "detect":
            continue
        c, v, op = r["classes"]["per_class"], r["vehicle_any"], r["operating_point"]["vehicle_any"]
        lines.append(f"| {r['suite']} | {r['images']} | {r['detector']} | {pct(v['AP50'])} | {pct(v['AP'])} | "
                     f"{pct(c['car']['AP50'])} | {pct(c['person']['AP50'])} | {pct(op['precision'])} / "
                     f"{pct(op['recall'])} | {r['seconds_per_image']} |")
    lines += ["", "| Detector | Rate | HOTA | MOTA | IDF1 | IDSW | People counted / in ground truth |",
              "|---|---|---|---|---|---|---|"]
    for r in rs:
        if r["kind"] != "track":
            continue
        m, ct = r["metrics"]["combined"], r["count_total"]
        lines.append(f"| {r['detector']} | {r['rate']} | {pct(m['HOTA'])} | {pct(m['MOTA'])} | {pct(m['IDF1'])} | "
                     f"{m['IDSW']} | {ct['counted']} / {ct['gt_people']} |")
    text = "\n".join(lines)
    print(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--suite", choices=["coco", "visdrone", "hituav"], required=True)
    d.add_argument("--data", required=True)
    d.add_argument("--detector", choices=["standard", "sam3", "both", "published"], default="standard")
    d.add_argument("--limit", type=int, help="a repeatable sample of this many images that hold a vehicle")
    d.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("track")
    t.add_argument("--data", required=True, help="MOT17 train folder (one folder per sequence)")
    t.add_argument("--detector", choices=["standard", "sam3", "both"], default="standard")
    t.add_argument("--rate", choices=["app", "full"], default="app",
                   help="app: 5 analysed frames per second as LanePulse runs; full: every frame, as published")
    t.add_argument("--sequences", help="comma-separated sequence names (default: all)")
    r = sub.add_parser("report")
    for p in (d, t, r):
        p.add_argument("--out", default=str(ROOT / "docs" / "benchmarks" / "results"))
    for p in (d, t):
        p.add_argument("--models", help="model packs folder (default: TWB_MODELS or <repo>/models)")
        p.add_argument("--standard-pack", default="yolox-s",
                       help="the standard detector's model pack, e.g. a trained or imported one (default yolox-s)")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        run_report(a)
        return 0
    res = run_detect(a) if a.cmd == "detect" else run_track(a)
    res["standard_pack"] = a.standard_pack
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / result_name(res)
    p.write_text(json.dumps(res, indent=1))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
