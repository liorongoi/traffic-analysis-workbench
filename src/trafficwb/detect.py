"""Vehicle detection with an ONNX model pack (YOLOX family, COCO classes).

Inference runs at the model's input size on the whole frame and, optionally, on
overlapping tiles at native resolution so small and distant vehicles are not
lost to downscaling. Boxes from all passes are merged with per-class
non-maximum suppression. Every resize and tile is written to the preprocessing
record that is stored with the run (HW-06).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .models import ModelPack

# COCO ids kept for traffic analysis and their names in this application.
COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = tuple(COCO_VEHICLES.values())


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels
    cls: str
    score: float
    class_scores: dict[str, float]


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
    """Keep the higher-scoring box when two boxes of different classes cover the same vehicle."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(_iou(d.box, k.box) < iou for k in kept):
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


class YoloxDetector:
    def __init__(self, pack: ModelPack, settings: DetectorSettings | None = None, providers: list[str] | None = None):
        import onnxruntime as ort  # imported here so the rest of the app starts without it

        self.pack = pack
        self.settings = settings or DetectorSettings()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(pack.weights), opts, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = int(pack.card.get("input_size", 640))
        grids, strides = [], []
        for st in (8, 16, 32):
            g = self.size // st
            yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), st))
        self._grids = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def describe(self) -> dict:
        return {
            "model": {"name": self.pack.name, "version": self.pack.version, "sha256": self.pack.sha256},
            "providers": self.session.get_providers(),
            "preprocessing": {"letterbox": self.size, "pad": 114, "colour": "BGR", "tiles": self.settings.record()},
        }

    def _run(self, img: np.ndarray, ox: int, oy: int) -> list[Detection]:
        x, r = _letterbox(img, self.size)
        out = self.session.run(None, {self.input_name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * self._strides
        scores = out[:, 4:5] * out[:, 5:]
        ids = list(COCO_VEHICLES)
        veh = scores[:, ids]
        best = veh.argmax(1)
        conf = veh.max(1)
        keep = np.where(conf >= self.settings.score_threshold)[0]
        dets = []
        h, w = img.shape[:2]
        for k in keep:
            cx, cy = xy[k] / r
            bw, bh = wh[k] / r
            x1, y1 = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
            x2, y2 = min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cs = {COCO_VEHICLES[i]: round(float(veh[k, j]), 4) for j, i in enumerate(ids)}
            box = tuple(round(float(v), 1) for v in (x1 + ox, y1 + oy, x2 + ox, y2 + oy))
            dets.append(Detection(box,
                                  COCO_VEHICLES[ids[best[k]]], round(float(conf[k]), 4), cs))
        return dets

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        dets = self._run(img_bgr, 0, 0)
        cols, rows = self.settings.tiles
        if (cols, rows) != (1, 1):
            tiles = tile_grid(w, h, cols, rows, self.settings.tile_overlap)
            if min(min(t[2] - t[0], t[3] - t[1]) for t in tiles) >= self.settings.min_tile_side:
                for x1, y1, x2, y2 in tiles:
                    dets += [d for d in self._run(img_bgr[y1:y2, x1:x2], x1, y1)
                             if not _touches_inner_edge(d.box, (x1, y1, x2, y2), (w, h))]
        return _merge_across_classes(_nms_per_class(dets, self.settings.nms_iou), self.settings.cross_class_iou)


def _touches_inner_edge(box, tile, frame, margin: float = 2.0) -> bool:
    """Drop tile boxes cut by a tile edge that is not a frame edge; the neighbouring tile or full frame covers them."""
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    fw, fh = frame
    return ((tx1 > 0 and x1 <= tx1 + margin) or (ty1 > 0 and y1 <= ty1 + margin)
            or (tx2 < fw and x2 >= tx2 - margin) or (ty2 < fh and y2 >= ty2 - margin))
