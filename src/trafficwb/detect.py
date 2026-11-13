"""Vehicle detection with an ONNX model pack (YOLOX family, COCO classes).

Inference runs at the model's input size on the whole frame and, optionally, on
overlapping tiles at native resolution so small and distant vehicles are not
lost to downscaling. Boxes from all passes are merged with per-class
non-maximum suppression. Every resize and tile is written to the preprocessing
record that is stored with the run (HW-06).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .models import ModelPack

# COCO ids kept for traffic analysis and their names in this application.
COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = tuple(COCO_VEHICLES.values())

# Other COCO classes the standard detector can also look for, when a mission adds a class of its own by one of
# these names (or an alias below). Classes not listed here are found by SAM 3.1 only.
COCO_OTHER = {0: "person", 6: "train", 8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign",
              12: "parking meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
              19: "cow", 24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase"}
COCO_ALIASES = {"pedestrian": "person", "people": "person", "human": "person", "man": "person", "woman": "person"}


def coco_id(cls: str) -> int | None:
    """The COCO class id the standard detector knows a mission class by, or None (SAM 3.1 only)."""
    name = COCO_ALIASES.get(cls, cls)
    for i, n in {**COCO_VEHICLES, **COCO_OTHER}.items():
        if n == name:
            return i
    return None


def standard_classes(own: list[str] | tuple[str, ...] | None) -> dict[int, str]:
    """COCO id to class name for the standard detector: the five vehicle classes plus a mission's own classes that
    COCO knows (a "pedestrian" class is found as COCO "person")."""
    out = dict(COCO_VEHICLES)
    for cls in own or ():
        i = coco_id(cls)
        if i is not None and i not in out:
            out[i] = cls
    return out


def family(cls: str) -> str:
    """Classes that may be one object: the vehicle classes (the detector may call one vehicle a car, then a truck).
    Every other class is a family of its own, so a person is never merged or tracked with the bicycle they ride."""
    return "vehicle" if cls in VEHICLE_CLASSES else cls


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    cls: str
    score: float
    class_scores: dict[str, float]
    found_by: str | None = None             # with two detectors: "standard", "sam3" or "both"


@dataclass
class DetectorSettings:
    score_threshold: float = 0.30
    nms_iou: float = 0.45
    cross_class_iou: float = 0.50     # one vehicle labelled both car and truck counts once
    tiles: tuple[int, int] = (3, 2)   # columns, rows; (1, 1) disables tiling
    tile_overlap: float = 0.20
    min_tile_side: int = 360          # skip tiling when tiles would be smaller than this

    def record(self) -> dict:
        return asdict(self)


class Detector(Protocol):
    # Detectors may also take area=<RegionSet>: parts of the frame outside the scan area are then not searched.
    def detect(self, img_bgr: np.ndarray) -> list[Detection]: ...
    def describe(self) -> dict: ...


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    out = np.full((size, size, 3), 114, np.uint8)
    rs = cv2.resize(img, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_LINEAR)
    out[: rs.shape[0], : rs.shape[1]] = rs
    return out, r


def tile_grid(w: int, h: int, cols: int, rows: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles covering the frame, as (x1, y1, x2, y2)."""
    tw = w / (cols - (cols - 1) * overlap)
    th = h / (rows - (rows - 1) * overlap)
    out = []
    for j in range(rows):
        for i in range(cols):
            x1 = round(i * tw * (1 - overlap))
            y1 = round(j * th * (1 - overlap))
            out.append((x1, y1, min(w, round(x1 + tw)), min(h, round(y1 + th))))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _merge_across_classes(dets: list[Detection], iou: float) -> list[Detection]:
    """Keep the higher-scoring box when two boxes of different classes of one family cover the same object."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(family(d.cls) != family(k.cls) or _iou(d.box, k.box) < iou for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: (-d.score, d.box))


def _nms_per_class(dets: list[Detection], iou: float) -> list[Detection]:
    kept: list[Detection] = []
    for cls in sorted({d.cls for d in dets}):
        group = [d for d in dets if d.cls == cls]
        rects = [[d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]] for d in group]
        idx = cv2.dnn.NMSBoxes(rects, [d.score for d in group], 0.0, iou)
        kept += [group[i] for i in np.array(idx, dtype=int).flatten()]
    return sorted(kept, key=lambda d: (-d.score, d.box))


class _OnnxDetector:
    """What every ONNX detector shares: the classes looked for, the score threshold, tiling and merging. A family
    only says how one picture goes through its model (``_run``)."""
    family = "?"

    def __init__(self, pack: ModelPack, settings: DetectorSettings | None = None, accelerator: str | None = None):
        from .runtime import make_session  # imports onnxruntime here so the rest of the app starts without it

        self.pack = pack
        self.classes = dict(COCO_VEHICLES)     # COCO id -> class name looked for
        self.settings = settings or DetectorSettings()
        self.session, self.runtime = make_session(pack.weights, accelerator)
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(pack.card.get("input_size", 640))

    def with_classes(self, classes: dict[int, str]):
        """The same loaded model looking for other COCO classes (it shares the session; nothing is loaded again)."""
        if classes == self.classes:
            return self
        other = copy.copy(self)
        other.classes = dict(classes)
        return other

    def with_threshold(self, score: float):
        """The same loaded model reporting boxes down to ``score`` (video's tracker uses the weak ones)."""
        if score == self.settings.score_threshold:
            return self
        from dataclasses import replace
        other = copy.copy(self)
        other.settings = replace(self.settings, score_threshold=score)
        return other

    def _preprocessing(self) -> dict:
        return {"letterbox": self.size, "pad": 114, "colour": "BGR"}

    def describe(self) -> dict:
        pre = self._preprocessing() | {"tiles": self.settings.record()}
        if self.classes != COCO_VEHICLES:
            pre["classes"] = {str(k): v for k, v in sorted(self.classes.items())}
        return {
            "model": {"name": self.pack.name, "version": self.pack.version, "sha256": self.pack.sha256,
                      "family": self.family},
            "providers": self.session.get_providers(),
            "runtime": self.runtime.record(),
            "preprocessing": pre,
        }

    def _detection(self, ids: list[int], scores: np.ndarray, box, w: int, h: int, ox: int, oy: int) -> Detection | None:
        """One box (x1, y1, x2, y2 in the picture's pixels) with its scores over ``ids``, clipped to the picture."""
        best = int(scores.argmax())
        x1, y1 = max(0.0, float(box[0])), max(0.0, float(box[1]))
        x2, y2 = min(float(w), float(box[2])), min(float(h), float(box[3]))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        # Class scores within the detection's own family: a person's box says nothing about vehicle classes.
        fam = family(self.classes[ids[best]])
        cs = {self.classes[i]: round(float(scores[j]), 4) for j, i in enumerate(ids) if family(self.classes[i]) == fam}
        out = tuple(round(float(v), 1) for v in (x1 + ox, y1 + oy, x2 + ox, y2 + oy))
        return Detection(out, self.classes[ids[best]], round(float(scores[best]), 4), cs)

    def _run(self, img: np.ndarray, ox: int, oy: int) -> list[Detection]:
        raise NotImplementedError

    def detect(self, img_bgr: np.ndarray, area=None) -> list[Detection]:
        """``area``: optional RegionSet. Tiles holding none of its scan area are skipped: a vehicle found only
        there could not count, since its box would lie wholly outside the area."""
        h, w = img_bgr.shape[:2]
        dets = self._run(img_bgr, 0, 0)
        cols, rows = self.settings.tiles
        if (cols, rows) != (1, 1):
            tiles = tile_grid(w, h, cols, rows, self.settings.tile_overlap)
            if min(min(t[2] - t[0], t[3] - t[1]) for t in tiles) >= self.settings.min_tile_side:
                cover = area.mask(w, h) if area is not None and hasattr(area, "mask") else None
                for x1, y1, x2, y2 in tiles:
                    if cover is not None and not cover[y1:y2, x1:x2].any():
                        continue
                    dets += [d for d in self._run(img_bgr[y1:y2, x1:x2], x1, y1)
                             if not _touches_inner_edge(d.box, (x1, y1, x2, y2), (w, h))]
        return _merge_across_classes(_nms_per_class(dets, self.settings.nms_iou), self.settings.cross_class_iou)


class YoloxDetector(_OnnxDetector):
    family = "yolox"

    def __init__(self, pack: ModelPack, settings: DetectorSettings | None = None, accelerator: str | None = None):
        super().__init__(pack, settings, accelerator)
        grids, strides = [], []
        for st in (8, 16, 32):
            g = self.size // st
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), st))
        self._grids = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def _run(self, img: np.ndarray, ox: int, oy: int) -> list[Detection]:
        if not self.classes:
            return []
        x, r = _letterbox(img, self.size)
        out = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * self._strides
        scores = out[:, 4:5] * out[:, 5:]
        ids = sorted(self.classes)
        veh = scores[:, ids]
        best = veh.argmax(1)
        conf = veh.max(1)
        keep = np.where(conf >= self.settings.score_threshold)[0]
        dets = []
        h, w = img.shape[:2]
        for k in keep:
            cx, cy = xy[k] / r
            bw, bh = wh[k] / r
            d = self._detection(ids, veh[k], (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), w, h, ox, oy)
            if d is not None:
                dets.append(d)
        return dets


# COCO's 91 category ids (with gaps) in the order of the 80 classes most detectors are trained on.
COCO91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
          34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
          62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class DetrDetector(_OnnxDetector):
    """Transformer detectors exported to ONNX (D-FINE, RT-DETR, RT-DETRv2, RF-DETR). The model card says how:

    - ``outputs``: "labels_boxes_scores" (the exports of D-FINE and RT-DETR, which take ``orig_target_sizes`` too and
      give each query's class, box in pixels and score) or "boxes_logits" (RF-DETR: boxes as centre and size in
      0..1, and one logit per class).
    - ``class_ids``: "coco80" (classes numbered 0..79) or "coco91" (COCO's category ids, 1..90 with gaps).
    - ``input``: {"colour": "RGB", "mean": [...], "std": [...]} (values in 0..1 before the mean and std; RF-DETR
      uses the ImageNet ones). The picture is stretched to the input size, as these models were trained.
    """
    family = "detr"

    def __init__(self, pack: ModelPack, settings: DetectorSettings | None = None, accelerator: str | None = None):
        super().__init__(pack, settings, accelerator)
        card = pack.card
        self.outputs = card.get("outputs", "labels_boxes_scores")
        if self.outputs not in ("labels_boxes_scores", "boxes_logits"):
            raise ValueError(f"Model card output layout '{self.outputs}' is not known.")
        self.coco91 = card.get("class_ids", "coco80") == "coco91"
        inp = card.get("input") or {}
        self.rgb = inp.get("colour", "RGB").upper() == "RGB"
        default_norm = self.outputs == "boxes_logits"
        self.mean = np.array(inp.get("mean", IMAGENET_MEAN if default_norm else (0, 0, 0)), np.float32)
        self.std = np.array(inp.get("std", IMAGENET_STD if default_norm else (1, 1, 1)), np.float32)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocessing(self) -> dict:
        return {"resize": self.size, "colour": "RGB" if self.rgb else "BGR", "mean": self.mean.tolist(),
                "std": self.std.tolist(), "outputs": self.outputs, "class_ids": "coco91" if self.coco91 else "coco80"}

    def _cls_index(self, label: int) -> int | None:
        """The 80-class COCO index (as ``classes`` uses) of a model's label."""
        if not self.coco91:
            return int(label)
        try:
            return COCO91.index(int(label))
        except ValueError:
            return None

    def _run(self, img: np.ndarray, ox: int, oy: int) -> list[Detection]:
        if not self.classes:
            return []
        h, w = img.shape[:2]
        x = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            x = x[..., ::-1]
        x = ((x.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None]
        feed = {self.input_names[0]: np.ascontiguousarray(x, np.float32)}
        ids = sorted(self.classes)
        found: list[tuple[np.ndarray, list[float]]] = []        # (scores over ids, box in picture pixels)
        if self.outputs == "labels_boxes_scores":
            if len(self.input_names) > 1:
                feed[self.input_names[1]] = np.array([[self.size, self.size]], np.int64)
            res = dict(zip(self.output_names, self.session.run(None, feed)))
            labels = res.get("labels", list(res.values())[0])[0]
            boxes = res.get("boxes", list(res.values())[1])[0]
            scores = res.get("scores", list(res.values())[2])[0]
            sx, sy = w / self.size, h / self.size
            for lab, box, sc in zip(labels, boxes, scores):
                k = self._cls_index(int(lab))
                if k is None or k not in self.classes or sc < self.settings.score_threshold:
                    continue
                v = np.zeros(len(ids), np.float32)
                v[ids.index(k)] = sc
                found.append((v, [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]))
        else:
            res = dict(zip(self.output_names, self.session.run(None, feed)))
            boxes = res.get("dets", list(res.values())[0])[0]
            logits = res.get("labels", list(res.values())[1])[0]
            prob = 1.0 / (1.0 + np.exp(-logits))
            cols = [COCO91[i] if self.coco91 else i for i in ids]
            ok = [c < prob.shape[1] for c in cols]
            sel = np.zeros((prob.shape[0], len(ids)), np.float32)
            for j, (c, good) in enumerate(zip(cols, ok)):
                if good:
                    sel[:, j] = prob[:, c]
            for q in np.where(sel.max(1) >= self.settings.score_threshold)[0]:
                cx, cy, bw, bh = boxes[q]
                found.append((sel[q], [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        dets = []
        for v, box in found:
            d = self._detection(ids, v, box, w, h, ox, oy)
            if d is not None:
                dets.append(d)
        return dets


FAMILIES = {"yolox": YoloxDetector, "detr": DetrDetector}


def make_detector(pack: ModelPack, settings: DetectorSettings | None = None, accelerator: str | None = None):
    """The detector for a model pack, by the family its card names (YOLOX when it names none)."""
    fam = pack.card.get("family", "yolox")
    if fam not in FAMILIES:
        raise ValueError(f"Model {pack.name} is of a family LanePulse cannot run ({fam}).")
    return FAMILIES[fam](pack, settings, accelerator)


def accepts_area(detector) -> bool:
    """Whether a detector can be told the scan area (test doubles and older detectors cannot)."""
    import inspect
    try:
        return "area" in inspect.signature(detector.detect).parameters
    except (TypeError, ValueError):
        return False


def _touches_inner_edge(box, tile, frame, margin: float = 2.0) -> bool:
    """Drop tile boxes cut by a tile edge that is not a frame edge; the neighbouring tile or full frame covers them."""
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    fw, fh = frame
    return ((tx1 > 0 and x1 <= tx1 + margin) or (ty1 > 0 and y1 <= ty1 + margin)
            or (tx2 < fw and x2 >= tx2 - margin) or (ty2 < fh and y2 >= ty2 - margin))
