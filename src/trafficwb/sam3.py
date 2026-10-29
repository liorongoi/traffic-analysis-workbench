"""SAM 3.1 (Meta's Segment Anything 3.1): find and outline anything named in words, on still images.

LanePulse uses it in two places:

- as a vehicle detector (``Sam3Detector``), an accurate but slow alternative to YOLOX-s that a mission
  chooses when it is created; and
- to draw a camera view's scan areas (``find_areas``): "road", "parking lot" and so on become include or
  exclude regions that the operator reviews before they are saved.

The model pack is the 8-bit weight-compressed OpenVINO build (W8A16) of SAM 3.1 published by
rajeshgangireddy/SAM3_OpenVINO, which its author measured as matching FP16 accuracy on CPU. It runs
on the CPU of any Windows, Mac or Linux computer through OpenVINO, with no network. Five graphs:

    vision-encoder   pixel_values [1, 3, S, S]          -> fpn_feat_0..2, fpn_pos_2
    text-encoder     input_ids, attention_mask [1, 32]  -> text_features [1, 32, 256], text_mask
    prompt-decoder   fpn features + prompt features     -> pred_masks [1, Q, M, M], pred_boxes [1, Q, 4],
                                                           pred_logits [1, Q], presence_logits [1, 1]
    (geometry-encoder and geometry-encoder-exemplar take box and point prompts; not used yet)

An instance's score is sigmoid(pred_logit) x sigmoid(presence_logit): the presence head says whether the
named thing is in the image at all, which is what keeps "bus" from outlining a car. The image is
resized to S x S without keeping its aspect ratio and normalised to (x - 0.5) / 0.5, as SAM 3 was trained.

The models are loaded on first use, shared by every mission, and unloaded after some minutes without use
(``IDLE_UNLOAD_S``), since they hold about 2 GB of memory while loaded.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detect import (COCO_VEHICLES, Detection, VEHICLE_CLASSES, _iou, _merge_across_classes, _nms_per_class,
                     _touches_inner_edge, tile_grid)
from .models import ModelPack, ModelPackError

PACK_NAME = "sam3.1-w8a16"
TASK = "segmentation"
IDLE_UNLOAD_S = 600
MIN_MEMORY = 12 * 2**30        # below this much memory in the computer, the screens warn before SAM 3.1 is used

# Words SAM 3.1 is asked for each vehicle class of this application. "van" is counted as a car, as COCO does.
VEHICLE_PROMPTS: dict[str, tuple[str, ...]] = {
    "car": ("car", "van"),
    "truck": ("truck",),
    "bus": ("bus",),
    "motorcycle": ("motorcycle",),
    "bicycle": ("bicycle",),
}
assert set(VEHICLE_PROMPTS) == set(COCO_VEHICLES.values())

# Suggestions on the region editor; any other words can be typed.
AREA_SUGGESTIONS = {"include": ["road", "highway lane", "intersection", "bridge"],
                    "exclude": ["parking lot", "sidewalk", "building", "sky", "vegetation"]}


class Sam3Unavailable(RuntimeError):
    """The SAM 3.1 pack or OpenVINO is not installed; the message says what to do."""


def import_openvino():
    """OpenVINO, with its usage reporting switched off.

    Importing OpenVINO also imports its model converter, which sends usage events to Google Analytics through
    the ``openvino_telemetry`` package unless a consent file says no. LanePulse never sends anything anywhere,
    so the package is hidden before OpenVINO is first imported; OpenVINO then uses its own silent stand-in.
    """
    import sys
    if "openvino" not in sys.modules:
        for name in ("openvino_telemetry", "openvino_telemetry.backend", "openvino_telemetry.backend.backend_ga4"):
            sys.modules[name] = None        # makes "import openvino_telemetry" raise ImportError
    import openvino
    return openvino


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


@dataclass
class Instance:
    """One thing SAM 3.1 found for one prompt. ``mask_logits`` are at the model's mask size; ``mask()`` scales them."""
    prompt: str
    score: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 in the image's pixels
    mask_logits: np.ndarray = field(repr=False)
    size: tuple[int, int] = (0, 0)               # image width, height

    def mask(self) -> np.ndarray:
        w, h = self.size
        return cv2.resize(self.mask_logits, (w, h), interpolation=cv2.INTER_LINEAR) > 0


class Sam3Engine:
    """The five SAM 3.1 graphs on OpenVINO's CPU device. Thread-safe; one image at a time."""

    def __init__(self, pack: ModelPack, device: str = "CPU"):
        self.pack = pack
        self.card = pack.card
        self.device = device
        self.image_size = int(self.card.get("input_size", 1008))
        self.text_len = int(self.card.get("text_length", 32))
        self.pad_id = int(self.card.get("pad_token_id", 49407))
        self.box_format = self.card.get("box_format", "xyxy")   # checked on the real model: normalised x1, y1, x2, y2
        self._lock = threading.RLock()
        self._models: dict | None = None
        self._tokenizer = None
        self._text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_used = time.monotonic()
        self.load_seconds: float | None = None

    # ---------- loading ----------
    def _graph(self, part: str) -> Path:
        return self.pack.path / self.card["graphs"][part]

    def load(self) -> None:
        with self._lock:
            if self._models is not None:
                return
            try:
                ov = import_openvino()
            except ImportError as exc:      # the wheel is a core dependency, so this means a damaged install
                raise Sam3Unavailable("OpenVINO is not installed, so SAM 3.1 cannot run. Repair LanePulse with its "
                                      "installer.") from exc
            from tokenizers import Tokenizer
            t0 = time.monotonic()
            core = ov.Core()
            models = {}
            for part in ("vision", "text", "decoder"):
                model = core.read_model(str(self._graph(part)))
                if part == "vision":
                    # One image at a time. A fixed batch of 1 is also what lets the image encoder's transposed
                    # convolutions run on Apple-silicon (ARM) CPUs, which refuse them with an open batch size.
                    model.reshape({"pixel_values": [1, 3, self.image_size, self.image_size]})
                # OpenVINO's default precision for the device (f16 on ARM, f32 or bf16 on x86) is kept on purpose:
                # forcing f32 on ARM brings the same refusal back.
                models[part] = core.compile_model(model, self.device)
            self._models = models
            self._tokenizer = Tokenizer.from_file(str(self.pack.path / self.card["tokenizer"]))
            self.load_seconds = round(time.monotonic() - t0, 2)
            self.last_used = time.monotonic()

    def release(self) -> None:
        with self._lock:
            self._models = None
            self._text_cache.clear()

    @property
    def loaded(self) -> bool:
        return self._models is not None

    def describe(self) -> dict:
        try:
            version = import_openvino().get_version()
        except ImportError:
            version = None
        return {"model": {"name": self.pack.name, "version": self.pack.version, "sha256": self.pack.sha256},
                "runtime": {"engine": "openvino", "version": version, "device": self.device,
                            "precision": self.card.get("precision"), "accelerator": "cpu"},
                "preprocessing": {"resize": [self.image_size, self.image_size], "keep_aspect": False,
                                  "colour": "RGB", "normalise": "(x/255 - 0.5) / 0.5"}}

    # ---------- inference ----------
    def _run(self, part: str, feed: dict) -> dict:
        res = self._models[part](feed)
        return {o.get_any_name(): res[o] for o in self._models[part].outputs}

    def encode_image(self, img_bgr: np.ndarray) -> dict:
        """Image features, reused by every prompt asked about the same image."""
        self.load()
        h, w = img_bgr.shape[:2]
        s = self.image_size
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (s, s), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        with self._lock:
            out = self._run("vision", {"pixel_values": np.ascontiguousarray(x)})
            self.last_used = time.monotonic()
        out["_size"] = (w, h)
        return out

    def tokens(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self._tokenizer.encode(text.strip().lower()).ids
        if len(ids) > self.text_len:            # keep the end-of-text token last
            ids = ids[: self.text_len - 1] + [ids[-1]]
        mask = [1] * len(ids) + [0] * (self.text_len - len(ids))
        ids = ids + [self.pad_id] * (self.text_len - len(ids))
        return np.array([ids], np.int64), np.array([mask], np.int64)

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        self.load()
        key = text.strip().lower()
        with self._lock:
            if key not in self._text_cache:
                ids, mask = self.tokens(key)
                out = self._run("text", {"input_ids": ids, "attention_mask": mask})
                self._text_cache[key] = (out["text_features"], out["text_mask"].astype(bool))
            return self._text_cache[key]

    def find(self, features: dict, text: str, threshold: float = 0.5) -> list[Instance]:
        """Every instance of ``text`` in the encoded image with a score at or above ``threshold``."""
        prompt, pmask = self.encode_text(text)
        feed = {k: features[k] for k in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2")}
        feed["prompt_features"] = prompt
        feed["prompt_mask"] = pmask
        with self._lock:
            out = self._run("decoder", feed)
            self.last_used = time.monotonic()
        scores = _sigmoid(out["pred_logits"][0]) * _sigmoid(out["presence_logits"][0, 0])
        w, h = features["_size"]
        found = []
        for q in np.argsort(-scores):
            if scores[q] < threshold:
                break
            found.append(Instance(text, round(float(scores[q]), 4), self._box(out["pred_boxes"][0, q], w, h),
                                  out["pred_masks"][0, q].astype(np.float32), (w, h)))
        return found

    def _box(self, b: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
        b = [float(v) for v in b]
        if self.box_format == "cxcywh":
            b = [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in b)   # normalised to the image
        return (round(x1 * w, 1), round(y1 * h, 1), round(x2 * w, 1), round(y2 * h, 1))

    def segment(self, img_bgr: np.ndarray, prompts: list[str], threshold: float = 0.5) -> dict[str, list[Instance]]:
        feats = self.encode_image(img_bgr)
        return {p: self.find(feats, p, threshold) for p in prompts}


# ---------- the shared engine ----------

_shared: dict[str, Sam3Engine] = {}
_shared_lock = threading.Lock()
_reaper: threading.Thread | None = None


def find_pack(root: Path | None = None) -> ModelPack:
    from . import models
    try:
        return models.find(PACK_NAME, root=root)
    except ModelPackError as exc:
        raise Sam3Unavailable(f"SAM 3.1 is not installed ({exc}). It comes with the full LanePulse installer; "
                              "a SAM 3.1 model pack can also be added with the Import Model tool.") from exc


def available(root: Path | None = None) -> dict:
    """What the screens show about SAM 3.1, without loading it (the checksum is checked on first use)."""
    from . import models
    cards = [c for c in models.installed(TASK, root) if c["name"] == PACK_NAME]
    try:
        import_openvino()
        runtime = True
    except ImportError:
        runtime = False
    eng = _shared.get(cards[-1]["version"]) if cards else None
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:
        total = None
    return {"installed": bool(cards) and runtime, "pack": cards[-1] if cards else None, "runtime": runtime,
            "loaded": bool(eng and eng.loaded), "memory_gb": round(total / 2**30, 1) if total else None,
            # one image needs about 7 GB while SAM 3.1 works on it
            "memory_note": None if not total or total >= MIN_MEMORY else
            f"This computer has {total / 2**30:.0f} GB of memory; SAM 3.1 needs about 8 GB free while it works, "
            "so it may be very slow or stop here.",
            "reason": None if cards and runtime else
            ("OpenVINO is missing from this installation." if cards else "The SAM 3.1 model pack is not installed.")}


def engine(root: Path | None = None) -> Sam3Engine:
    """The one engine of this process (models load on first use and unload when idle)."""
    global _reaper
    pack = find_pack(root)                 # verifies every file's checksum
    with _shared_lock:
        eng = _shared.get(pack.version)
        if eng is None or eng.pack.path != pack.path:
            eng = _shared[pack.version] = Sam3Engine(pack)
        if _reaper is None:
            _reaper = threading.Thread(target=_unload_idle, name="sam3-idle", daemon=True)
            _reaper.start()
    return eng


def _unload_idle() -> None:
    while True:
        time.sleep(30)
        for eng in list(_shared.values()):
            if eng.loaded and time.monotonic() - eng.last_used > IDLE_UNLOAD_S and eng._lock.acquire(blocking=False):
                try:
                    eng.release()
                finally:
                    eng._lock.release()


# ---------- vehicle detection ----------

@dataclass
class Sam3DetectorSettings:
    score_threshold: float = 0.45
    cross_class_iou: float = 0.50
    nms_iou: float = 0.50
    # Large frames are also searched in overlapping tiles so distant vehicles are not lost to the 1008-pixel input.
    tiles: tuple[int, int] = (2, 2)
    tile_overlap: float = 0.20
    min_tile_side: int = 700

    def record(self) -> dict:
        return asdict(self)


class Sam3Detector:
    """Vehicle detection by asking SAM 3.1 for each vehicle class by name. Slower than YOLOX, and needs no training."""

    def __init__(self, eng: Sam3Engine, settings: Sam3DetectorSettings | None = None):
        self.engine = eng
        self.pack = eng.pack
        self.settings = settings or Sam3DetectorSettings()

    def describe(self) -> dict:
        d = self.engine.describe()
        d["preprocessing"] = {**d["preprocessing"], "prompts": {k: list(v) for k, v in VEHICLE_PROMPTS.items()},
                              "settings": self.settings.record()}
        d["providers"] = ["OpenVINO CPU"]
        return d

    def _pass(self, img: np.ndarray, ox: int, oy: int) -> list[Detection]:
        feats = self.engine.encode_image(img)
        raw: list[tuple[str, float, tuple]] = []
        for cls, words in VEHICLE_PROMPTS.items():
            for word in words:
                for inst in self.engine.find(feats, word, self.settings.score_threshold):
                    x1, y1, x2, y2 = inst.box
                    if x2 - x1 >= 2 and y2 - y1 >= 2:
                        raw.append((cls, inst.score, (x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
        dets = []
        for cls, score, box in raw:
            # What each class said about this box, as YOLOX's class scores do; the box's own class is its score.
            cs = {c: 0.0 for c in VEHICLE_CLASSES}
            for c2, s2, b2 in raw:
                if _iou(box, b2) >= 0.5:
                    cs[c2] = max(cs[c2], round(s2, 4))
            cs[cls] = max(cs[cls], round(score, 4))
            dets.append(Detection(tuple(round(v, 1) for v in box), cls, round(score, 4), cs))
        return dets

    def detect(self, img_bgr: np.ndarray, area=None) -> list[Detection]:
        """``area``: optional RegionSet; tiles that hold none of its scan area are not searched."""
        h, w = img_bgr.shape[:2]
        dets = self._pass(img_bgr, 0, 0)
        cols, rows = self.settings.tiles
        if (cols, rows) != (1, 1):
            tiles = tile_grid(w, h, cols, rows, self.settings.tile_overlap)
            if min(min(t[2] - t[0], t[3] - t[1]) for t in tiles) >= self.settings.min_tile_side:
                cover = area.mask(w, h) if area is not None and hasattr(area, "mask") else None
                for x1, y1, x2, y2 in tiles:
                    if cover is not None and not cover[y1:y2, x1:x2].any():
                        continue
                    dets += [d for d in self._pass(img_bgr[y1:y2, x1:x2], x1, y1)
                             if not _touches_inner_edge(d.box, (x1, y1, x2, y2), (w, h))]
        return _merge_across_classes(_nms_per_class(dets, self.settings.nms_iou), self.settings.cross_class_iou)


# ---------- scan areas from words ----------

@dataclass
class AreaProposal:
    prompt: str
    kind: str                          # include | exclude
    polygon: list[tuple[float, float]]
    area_share: float                  # of the whole image
    score: float                       # best instance score inside it


def mask_to_polygons(mask: np.ndarray, min_share: float = 0.003, max_points: int = 120,
                     max_outlines: int = 24) -> list[tuple[list, list[list]]]:
    """Outer outlines of a binary mask, each with its large holes: [(outline, [hole, ...]), ...].

    Small specks (under ``min_share`` of the image) are dropped, edges are smoothed, and every polygon is
    simplified until it has at most ``max_points`` corners and does not cross itself.
    """
    from .regions import validate_polygon
    h, w = mask.shape
    m = mask.astype(np.uint8)
    k = max(3, int(round(min(w, h) / 200)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    min_area = min_share * w * h

    def simplify(c) -> list | None:
        peri = cv2.arcLength(c, True)
        for f in (0.002, 0.004, 0.007, 0.012, 0.02, 0.035):
            p = cv2.approxPolyDP(c, f * peri, True).reshape(-1, 2)
            if len(p) < 3:
                return None
            pts = [(float(min(max(x, 0), w)), float(min(max(y, 0), h))) for x, y in p]
            if len(pts) <= max_points and not validate_polygon(pts, w, h):
                return pts
        hull = cv2.convexHull(c).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in hull] if len(hull) >= 3 else None

    out = []
    outer = sorted((i for i, c in enumerate(contours) if hier[i][3] == -1 and cv2.contourArea(c) >= min_area),
                   key=lambda i: -cv2.contourArea(contours[i]))[:max_outlines]   # holes go with their outline
    for i in outer:
        c = contours[i]
        outline = simplify(c)
        if outline is None:
            continue
        holes, j = [], hier[i][2]
        while j != -1:
            if cv2.contourArea(contours[j]) >= min_area:
                hp = simplify(contours[j])
                if hp:
                    holes.append(hp)
            j = hier[j][0]
        out.append((outline, holes))
    return out


def find_areas(eng: Sam3Engine, img_bgr: np.ndarray, prompts: list[str], kind: str = "include",
               threshold: float = 0.4) -> list[AreaProposal]:
    """Outline every area named by ``prompts``. Holes in an include area (a traffic island) become exclusions."""
    if kind not in ("include", "exclude"):
        raise ValueError("kind must be include or exclude")
    h, w = img_bgr.shape[:2]
    feats = eng.encode_image(img_bgr)
    out: list[AreaProposal] = []
    for p in prompts:
        p = p.strip()
        if not p:
            continue
        found = eng.find(feats, p, threshold)
        if not found:
            continue
        union = np.zeros((h, w), bool)
        best = np.zeros((h, w), np.float32)
        for inst in found:
            m = inst.mask()
            union |= m
            best[m] = np.maximum(best[m], inst.score)
        for outline, holes in mask_to_polygons(union):
            poly_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(poly_mask, [np.array(outline, np.int32)], 1)
            inside = poly_mask.astype(bool)
            score = float(best[inside].max()) if inside.any() else 0.0
            out.append(AreaProposal(p, kind, [(round(x, 1), round(y, 1)) for x, y in outline],
                                    round(float(inside.mean()), 4), round(score, 3)))
            if kind == "include":
                for hole in holes:
                    hm = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(hm, [np.array(hole, np.int32)], 1)
                    out.append(AreaProposal(f"{p} (gap)", "exclude", [(round(x, 1), round(y, 1)) for x, y in hole],
                                            round(float(hm.mean()), 4), round(score, 3)))
    return sorted(out, key=lambda a: (a.prompt, -a.area_share))
