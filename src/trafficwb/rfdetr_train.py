"""Fine-tune RF-DETR on a dataset exported from LanePulse, the same way trafficwb-train fine-tunes YOLOX-s.

RF-DETR (Roboflow, Apache-2.0, https://github.com/roboflow/rf-detr) is a transformer detector with a DINOv2 backbone.
Its released COCO weights are the starting point; training keeps their 91-slot COCO class head (a car stays COCO
category 3), so the trained model runs in the app exactly like the released one (detect.DetrDetector, "boxes_logits")
and still knows the COCO classes the mission did not train.

    trafficwb-train run --arch rf-detr --dataset <exported dataset folder> --out <folder>

The training loop is RF-DETR's own (PyTorch Lightning, through the rfdetr package); this module turns the exported
dataset into the layout it reads, reports progress, pauses and stops between steps, and writes a model pack:
model.onnx (what the app runs), model.pth (to train further), model.json and LICENSE, and a .zip of the folder.

Areas nobody reviewed (a partial review, excluded regions, ignore boxes) must not teach "nothing is here". YOLOX
training leaves them out of its loss; RF-DETR's loss cannot, so those areas are painted plain grey in the training
copy of the picture (the boxes the operator drew stay visible).

Starting weights: --init coco (the released RF-DETR weights of the chosen size; the installers ship them next to
their own Python, in <prefix>/share/lanepulse, as rf-detr-<size>-coco.pth) or an earlier RF-DETR pack (folder or .zip).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import contextlib
import hashlib
import json
import math
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .train import TrainCancelled, TrainError, _open_pack, _sha256, background_mask, describe_device, pick_device, \
    shipped_weights_dirs

ARCH = "rf-detr"
# Dataset categories (1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck, 6 person) to COCO category ids, the slots of
# RF-DETR's class head.
CAT_TO_COCO91 = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8, 6: 1}
COCO_NAMES = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
GREY = 114
# The released COCO weights, as RF-DETR publishes them (MD5 from the rfdetr package's own list). fetch_models.py keeps
# only the weights ("model" and "args"; the optimizer state is two thirds of the file) as rf-detr-<size>-coco.pth.
SIZES = {
    "nano": {"cls": "RFDETRNano", "resolution": 384, "md5": "fb6504cce7fbdc783f7a46991f07639f",
             "url": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth"},
    "small": {"cls": "RFDETRSmall", "resolution": 512, "md5": "fb37061c1af7bace359c91b723a8d5c1",
              "url": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth"},
    "medium": {"cls": "RFDETRMedium", "resolution": 576, "md5": "7223f764a87b863f02eb8d52bf0ce2ee",
               "url": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth"},
}
DEFAULT_SIZE = "small"
LICENSE_NOTE = """This model was fine-tuned from RF-DETR (Roboflow, https://github.com/roboflow/rf-detr).
RF-DETR code and the released Nano, Small and Medium weights: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
Its backbone is DINOv2 (Meta, Apache License 2.0).
The starting weights were trained on COCO 2017 (annotations CC BY 4.0; the Flickr images carry their own terms).
Fine-tuning used the mission's own reviewed images, exported from LanePulse.
"""


def available() -> dict:
    """Whether RF-DETR can be trained here (the rfdetr package with its training extras)."""
    import importlib.util
    if importlib.util.find_spec("rfdetr") is None:
        return {"ok": False, "reason": "RF-DETR training is not part of this installation (the rfdetr package is missing). "
                                      "Install the current LanePulse version."}
    if importlib.util.find_spec("pytorch_lightning") is None:
        return {"ok": False, "reason": "RF-DETR's training tools (PyTorch Lightning) are missing here. Install the current "
                                      "LanePulse version."}
    return {"ok": True, "reason": None}


def weights_name(size: str) -> str:
    return f"rf-detr-{size}-coco.pth"


def find_coco_weights(size: str = DEFAULT_SIZE) -> Path | None:
    for d in shipped_weights_dirs():
        p = d / weights_name(size)
        if p.is_file():
            return p
    return None


def slim_checkpoint(src: Path, dest: Path) -> Path:
    """Keep only what training starts from (weights and their settings), not the optimizer state."""
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "args": ck.get("args")}, dest)
    return dest


def coco_weights(size: str = DEFAULT_SIZE, log=print, download: bool = True) -> Path:
    found = find_coco_weights(size)
    if found:
        return found
    if not download:
        raise TrainError(f"The RF-DETR {size} starting weights are not installed here ({weights_name(size)} in "
                         f"{shipped_weights_dirs()[-2]}). Reinstall LanePulse, or continue from a trained RF-DETR model.")
    info = SIZES[size]
    cache = Path(os.environ.get("LANEPULSE_CACHE", Path.home() / ".cache" / "lanepulse"))
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"rf-detr-{size}.part"
    log(f"Fetching the RF-DETR {size} starting weights (about 400 MB) into {cache} ...")
    h = hashlib.md5()
    with urllib.request.urlopen(info["url"], timeout=60) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != info["md5"]:
        tmp.unlink(missing_ok=True)
        raise TrainError("The downloaded RF-DETR weights do not match their published checksum; nothing was used.")
    out = slim_checkpoint(tmp, cache / weights_name(size))
    tmp.unlink(missing_ok=True)
    return out


# ------------------------------------------------------------------ data

def _grey_unreviewed(im: np.ndarray, img: dict, boxes: list[list[float]]) -> np.ndarray:
    """Paint where nobody looked plain grey, keeping every box the operator drew."""
    h, w = im.shape[:2]
    keep = background_mask(img, h, w).astype(bool)
    if keep.all():
        return im
    for x, y, bw, bh in boxes:
        x1, y1 = max(0, int(math.floor(x)) - 2), max(0, int(math.floor(y)) - 2)
        x2, y2 = min(w, int(math.ceil(x + bw)) + 2), min(h, int(math.ceil(y + bh)) + 2)
        keep[y1:y2, x1:x2] = True
    out = im.copy()
    out[~keep] = GREY
    return out


def class_map(dataset: Path, init: str = "coco") -> tuple[dict[int, int], dict[int, str]]:
    """(dataset category -> COCO category id, the slot of RF-DETR's head; COCO 80-class index -> name of a target the
    operator marked). Targets of their own (categories 7 and up) take over COCO classes traffic analysis never looks
    for, the same ones YOLOX training uses (own_classes.py), so either model reports them by their own names."""
    try:
        from .own_classes import OwnClassError, _init_own, category_map
    except ImportError:                                  # an installation without marked targets
        return dict(CAT_TO_COCO91), {}
    from .detect import COCO91
    try:
        m80, own = category_map(Path(dataset), _init_own(str(init)))
    except OwnClassError as exc:
        raise TrainError(str(exc)) from None
    return {c: COCO91[i] for c, i in m80.items()}, own


def prepare(dataset: Path, work: Path, cmap: dict[int, int] | None = None, own: dict[int, str] | None = None) -> dict:
    """The exported dataset in the COCO layout RF-DETR reads (train2017, val2017, annotations/instances_*.json), with
    COCO category ids. Returns counts per split."""
    from .detect import COCO91
    dataset = Path(dataset)
    cmap = cmap or CAT_TO_COCO91
    names = dict(COCO_NAMES) | {COCO91[i]: n for i, n in (own or {}).items()}
    cats = [{"id": i, "name": n, "supercategory": "object"} for i, n in sorted(names.items())]
    counts = {}
    (work / "annotations").mkdir(parents=True, exist_ok=True)
    for split, folder in (("train", "train2017"), ("val", "val2017")):
        f = dataset / f"annotations_{split}.json"
        data = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"images": [], "annotations": []}
        (work / folder).mkdir(parents=True, exist_ok=True)
        by_img: dict[int, list] = {}
        for a in data["annotations"]:
            if a["bbox"][2] > 1 and a["bbox"][3] > 1:
                by_img.setdefault(a["image_id"], []).append(a)
        images, anns = [], []
        for k, img in enumerate(data["images"], 1):
            raw = np.fromfile(str(dataset / img["file_name"]), dtype=np.uint8)
            im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if im is None:
                raise TrainError(f"Cannot read {img['file_name']}.")
            mine = by_img.get(img["id"], [])
            im = _grey_unreviewed(im, img, [a["bbox"] for a in mine])
            name = f"{k:07d}.jpg"
            cv2.imwrite(str(work / folder / name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            images.append({"id": k, "file_name": name, "width": int(im.shape[1]), "height": int(im.shape[0])})
            for a in mine:
                x, y, bw, bh = a["bbox"]
                anns.append({"id": len(anns) + 1, "image_id": k, "category_id": cmap.get(a["category_id"], 3),
                             "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
        (work / "annotations" / f"instances_{folder}.json").write_text(
            json.dumps({"images": images, "annotations": anns, "categories": cats}), encoding="utf-8")
        counts[split] = {"images": len(images), "boxes": len(anns)}
    if not counts["val"]["images"]:                      # RF-DETR validates every epoch: use the training images
        shutil.rmtree(work / "val2017")
        shutil.copytree(work / "train2017", work / "val2017")
        shutil.copy(work / "annotations" / "instances_train2017.json", work / "annotations" / "instances_val2017.json")
    return counts


# ------------------------------------------------------------------ training

def _model_class(size: str):
    import rfdetr
    return getattr(rfdetr, SIZES[size]["cls"])


def _init_weights(init: str, size: str, work: Path, log, download: bool) -> tuple[Path, dict, str]:
    """(checkpoint to start from, what it was, model size)."""
    if init == "coco":
        p = coco_weights(size, log, download)
        return p, {"name": f"rf-detr-{size}", "version": "coco", "file": p.name, "sha256": _sha256(p)}, size
    p = Path(init)
    if p.suffix == ".pth" and p.is_file():
        return p, {"file": p.name, "sha256": _sha256(p)}, size
    pack = _open_pack(p, work)
    card = json.loads((pack / "model.json").read_text(encoding="utf-8"))
    if card.get("architecture") != ARCH:
        raise TrainError(f"{card.get('name')} is not an RF-DETR model; start RF-DETR training from the released "
                         "RF-DETR weights or from an RF-DETR model trained earlier.")
    pt = pack / (card.get("torch_weights") or "")
    if not card.get("torch_weights") or not pt.is_file():
        raise TrainError(f"{p} has no training weights (model.pth); only RF-DETR packs written by LanePulse can be "
                         "trained further. Start from the released RF-DETR weights instead.")
    expect = (card.get("files") or {}).get(pt.name)
    if expect and _sha256(pt) != expect:
        raise TrainError(f"{pt.name} in {p} does not match its model card; copy the pack again.")
    return pt, {"name": card["name"], "version": card["version"], "sha256": card["sha256"]}, card.get("size", size)


@contextlib.contextmanager
def _with_callback(cb):
    """RF-DETR builds its own Lightning trainer; add our callback (progress, pause and stop) to it."""
    import rfdetr.training as rt
    orig = rt.build_trainer

    def build(*a, **kw):
        kw["extra_callbacks"] = list(kw.get("extra_callbacks") or ()) + [cb]
        return orig(*a, **kw)

    rt.build_trainer = build
    try:
        yield
    finally:
        rt.build_trainer = orig


def _callback(epochs: int, check, progress, on_epoch, log, hist: list, t0: float):
    from pytorch_lightning import Callback

    class Report(Callback):
        def __init__(self):
            self.step = 0
            self.run = 0.0
            self.n = 0

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            check()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            self.step += 1
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None:
                self.run += float(loss.detach() if hasattr(loss, "detach") else loss)
                self.n += 1
            if progress:
                per = max(int(trainer.num_training_batches or 1), 1)
                progress(self.step, per * epochs)

        def on_validation_epoch_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            ep = int(trainer.current_epoch) + 1
            m = trainer.callback_metrics
            def num(*keys):
                for k in keys:
                    if k in m:
                        try:
                            return round(float(m[k]), 4)
                        except (TypeError, ValueError):
                            pass
                return None
            row = {"epoch": ep, "train_loss": round(self.run / self.n, 4) if self.n else None,
                   "val_loss": num("val/loss", "val_loss"),
                   "val_ap50": num("val/ema_mAP_50", "val/mAP_50", "val/map_50"),
                   "val_ap": num("val/ema_mAP_50_95", "val/mAP_50_95", "val/map")}
            self.run, self.n = 0.0, 0
            hist.append(row)
            log(f"Epoch {ep}/{epochs}: training loss {row['train_loss']}"
                + ("" if row["val_loss"] is None else f", validation loss {row['val_loss']:.3f}")
                + ("" if row["val_ap50"] is None else f", validation AP50 {row['val_ap50']:.3f}")
                + f", {(time.time() - t0) / ep:.0f} s per epoch")
            if on_epoch:
                on_epoch(ep, epochs, row)

    return Report()


def train(dataset: Path, out: Path, *, init: str = "coco", epochs: int = 20, batch: int | None = None,
          lr: float | None = None, device: str = "auto", workers: int | None = None, name: str | None = None,
          version: str | None = None, seed: int = 0, size: str = DEFAULT_SIZE, log=print, progress=None, control=None,
          download: bool = True, on_epoch=None, mp_context: str | None = None, **_ignored) -> Path:
    """Train and write the model pack; returns the .zip path. Same arguments and callbacks as train.train."""
    import torch

    def check():
        if control is not None:
            from .process import RunCancelled
            try:
                control.checkpoint()
            except RunCancelled:
                raise TrainCancelled("Training was stopped; no model was written.") from None

    ok = available()
    if not ok["ok"]:
        raise TrainError(ok["reason"])
    if size not in SIZES:
        raise TrainError(f"RF-DETR size must be one of {', '.join(SIZES)}.")
    dataset, out = Path(dataset), Path(out)
    manifest_f = dataset / "manifest.json"
    if not manifest_f.is_file():
        raise TrainError(f"{dataset} is not an exported dataset (no manifest.json). Export one on the learning page.")
    manifest = json.loads(manifest_f.read_text(encoding="utf-8"))
    dev = pick_device(device)
    batch = batch or (8 if dev.type == "cuda" else 4)
    lr = lr if lr is not None else 1e-4
    log(f"Device: {describe_device(dev)}")
    torch.manual_seed(seed)
    work = Path(tempfile.mkdtemp(prefix="lanepulse-rfdetr-"))
    try:
        start, init_src, size = _init_weights(init, size, work, log, download)
        name = name or f"rf-detr-{size}-tuned"
        data = work / "coco"
        cmap, own = class_map(dataset, init)
        if own:
            log("Marked targets learnt as their own classes: " + ", ".join(own.values()))
        counts = prepare(dataset, data, cmap, own)
        if not counts["train"]["images"]:
            raise TrainError("The dataset has no training images.")
        log(f"Training images: {counts['train']['images']} ({counts['train']['boxes']} vehicles), "
            f"validation images: {counts['val']['images']}")
        check()
        model = _model_class(size)(pretrain_weights=str(Path(start).resolve()), num_classes=90, device=dev.type)
        hist: list[dict] = []
        t0 = time.time()
        cb = _callback(epochs, check, progress, on_epoch, log, hist, t0)
        run_dir = work / "run"
        if workers is None:
            workers = min(4, os.cpu_count() or 1) if dev.type == "cuda" else 0
        with _with_callback(cb):
            try:
                model.train(dataset_dir=str(data), dataset_file="coco", output_dir=str(run_dir), epochs=epochs,
                            batch_size=batch, grad_accum_steps=max(1, 16 // batch), lr=lr, num_workers=workers,
                            device=dev.type, tensorboard=False, progress_bar=None, checkpoint_interval=max(epochs, 1),
                            seed=seed, run_test=False)
            except TrainCancelled:
                raise
            except Exception as exc:
                cause = exc
                while cause is not None and not isinstance(cause, TrainCancelled):
                    cause = cause.__cause__ or cause.__context__
                if isinstance(cause, TrainCancelled):
                    raise cause from None
                raise
        best = next((run_dir / f for f in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth",
                                            "checkpoint_best_regular.pth") if (run_dir / f).is_file()), None)
        if best is None:
            raise TrainError("RF-DETR training ended without writing a model.")
        log(f"Keeping {best.name.replace('checkpoint_', '').replace('.pth', '').replace('_', ' ')} weights.")
        final = _model_class(size)(pretrain_weights=str(best.resolve()), num_classes=90, device="cpu")
        record = {"dataset": {k: manifest.get(k) for k in ("mission_id", "dataset_id", "name", "stats", "exported_at", "datasets")},
                  "dataset_files": {f: _sha256(dataset / f) for f in manifest.get("files", []) if (dataset / f).is_file()},
                  "init": init_src, "epochs": epochs, "batch": batch, "lr": lr, "size": size,
                  "device": describe_device(dev), "torch": torch.__version__, "seed": seed, "history": hist,
                  "seconds": round(time.time() - t0), "own_classes": {str(k): v for k, v in sorted(own.items())}}
        try:
            from importlib.metadata import version as _version
            record["rfdetr"] = _version("rfdetr")
        except Exception:
            pass
        return write_pack(final, best, out, name, version or datetime.now().strftime("%Y%m%d-%H%M"), size, record, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ model pack

def write_pack(model, checkpoint: Path, out: Path, name: str, version: str, size: str, record: dict, log=print) -> Path:
    """model.onnx (what the app runs), model.pth (to train further), model.json, LICENSE; and a .zip of it."""
    pack = out / name / version
    if pack.exists():
        raise TrainError(f"{pack} exists already; nothing was overwritten.")
    tmp = pack.with_name(version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    exp = tmp / "export"
    model.export(output_dir=str(exp), verbose=False)
    onnx_files = sorted(exp.glob("*.onnx"))
    if not onnx_files:
        raise TrainError("RF-DETR did not write an ONNX file.")
    shutil.move(str(onnx_files[0]), tmp / "model.onnx")
    shutil.rmtree(exp, ignore_errors=True)
    slim_checkpoint(checkpoint, tmp / "model.pth")
    (tmp / "LICENSE").write_text(LICENSE_NOTE, encoding="utf-8")
    res = SIZES[size]["resolution"]
    own = record.get("own_classes") or {}
    card = {"name": name, "version": version, "task": "vehicle_detection", "sha256": _sha256(tmp / "model.onnx"),
            "license": "Apache-2.0 (RF-DETR code and starting weights); fine-tuned on the mission's own reviewed images",
            "source": f"trafficwb-train (fine-tuned RF-DETR {size})", "family": "detr", "architecture": ARCH,
            "size": size, "outputs": "boxes_logits", "class_ids": "coco91", "input_size": res,
            "input": {"colour": "RGB", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "input_format": f"RGB, stretched to {res}x{res}, ImageNet mean and std",
            "weights": "model.onnx", "torch_weights": "model.pth", **({"own_classes": own} if own else {}),
            "files": {"model.pth": _sha256(tmp / "model.pth"), "LICENSE": _sha256(tmp / "LICENSE")},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "training": record}
    (tmp / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(pack)
    zpath = out / f"{name}-{version}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pack.iterdir()):
            z.write(f, f"{name}/{version}/{f.name}")
    log(f"Model pack: {pack}")
    return zpath
