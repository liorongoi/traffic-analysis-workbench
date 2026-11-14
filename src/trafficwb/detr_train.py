"""Fine-tune D-FINE or RT-DETRv2 on a dataset exported from LanePulse, the same way trafficwb-train fine-tunes YOLOX-s.

Both are real-time transformer detectors (D-FINE: USTC, Apache-2.0, https://github.com/Peterande/D-FINE; RT-DETRv2:
Baidu/PekingU, Apache-2.0, https://github.com/lyuwenyu/RT-DETR). Training uses their Hugging Face ``transformers``
implementations and the released COCO weights as the starting point. The 80-class COCO head is kept (a car stays
COCO class 2), so the trained model still knows the classes the mission did not train, and a target the operator
taught ("Teach a new target", categories 7+ of the export) takes a COCO class traffic analysis never looks for, as in
YOLOX training (own_classes.py); the pack names it.

    trafficwb-train run --arch d-fine --size small --dataset <exported dataset folder> --out <folder>
    trafficwb-train run --arch rt-detr --size r18 --dataset <folder> --out <folder>

Every label the export holds teaches, whoever made it: the operator's reviews, SAM 3.1's labels on what nobody
reviewed (reviews win where they overlap; learning.py), and confirmed marked targets. Areas nobody looked at (a
partial review, excluded regions, unsure SAM boxes and other ignore boxes) must not teach "nothing is here"; a
transformer's set loss cannot leave them out, so they are painted plain grey in the training copy of the picture
(every labelled box stays visible).

Starting weights: --init coco (the released weights of the chosen size, shipped next to the installation's Python
as <arch>-<size>-coco.pth, e.g. d-fine-small-coco.pth) or an earlier D-FINE / RT-DETRv2 pack (folder or .zip).

The pack: model.onnx (what the app runs: boxes as centre and size in 0..1 and one logit per class, "boxes_logits"),
config.json and model.safetensors (to train further; the pack folder loads with ``from_pretrained``), model.json,
LICENSE; and a .zip of the folder.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Never reach any model hub at run time: no account, token, telemetry or online check. Weights are local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import math
import random
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .own_classes import OwnClassError, _init_own, category_map
from .train import TrainCancelled, TrainError, _open_pack, _sha256, background_mask, describe_device, pick_device, \
    shipped_weights_dirs

GREY = 114
# Released COCO weights (Apache-2.0), as published by the authors' Hugging Face organisations: repository, commit and
# SHA-256 of model.safetensors, so a fetch is always the same file. Fetched only when installers are built (or with
# scripts/fetch_models.py); training itself never goes online.
ARCHS = {
    "d-fine": {"title": "D-FINE", "default": "small", "sizes": {
        "nano": ("ustc-community/dfine-nano-coco", "066438d3d8f0da137a37b38fdf3368fd4afceced",
                 "19e06bdc873da819920a8d373b879721a5b9759d822f8213220bb09abbdab58b"),
        "small": ("ustc-community/dfine-small-coco", "f79e65b5fbb33ceb9d3ebba042955d7410c608f8",
                  "a144baff3a27bfd538fe37b11fa8d4fca5744b8c054b5301e7c699a1379cd18f"),
        "medium": ("ustc-community/dfine-medium-coco", "4ab9f5e466432f4fcf9a9a023da6aa5ecc9c9829",
                   "5a53154fb87af91ad1ca3081599f21423a5681c46b613e15f2ff556837b02634"),
        "large": ("ustc-community/dfine-large-coco", "aa8f8211545e4cff5adf52ea512f9ac74a9824ec",
                  "4e9d77ba937ab6eb8537b183c1c05372262daaa552e7682d721417042e24a8dc"),
        "xlarge": ("ustc-community/dfine-xlarge-coco", "ea4f6be7350bbe3c199ec6febc74168346cb5a68",
                   "3ac6ea37c6c995ed65a10beeda703249e97aac1bb40bda18cfaa1c2f6cdca4c5")},
        "license": "D-FINE (USTC, https://github.com/Peterande/D-FINE): code and released weights under the "
                   "Apache License 2.0."},
    "rt-detr": {"title": "RT-DETRv2", "default": "r18", "sizes": {
        "r18": ("PekingU/rtdetr_v2_r18vd", "5650961749fa93567c0d46fc7f43ea4f9e914107",
                "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115"),
        "r34": ("PekingU/rtdetr_v2_r34vd", "5d60d8fb5eb2a3ee5575bf200ecbce4e2cfa420c",
                "d67528a6c4bbaece03d65feefbba94052894aa6520aa1f65c083cd0f7b92d612"),
        "r50": ("PekingU/rtdetr_v2_r50vd", "282494075698cab9faa1096ae26856890030c817",
                "3331d977dbc0c7a6cdae9ec0b0b6ad156eb6720d65b7cf0fa710dcc541d88d71"),
        "r101": ("PekingU/rtdetr_v2_r101vd", "2c5dbbd2d4d8c8814827a3b42737ba1afce3cf2a",
                 "fa2d79aa50064eb2d41abdc1acaa5ca20fc9d95347851c93ab6aad30a8974ee0")},
        "license": "RT-DETRv2 (Baidu, https://github.com/lyuwenyu/RT-DETR): code and released weights under the "
                   "Apache License 2.0."},
}
DEFAULT_SHIPPED = {"d-fine": ["small"], "rt-detr": ["r18"]}   # what the installers carry unless told otherwise
INPUT = 640


def available() -> dict:
    """Whether D-FINE and RT-DETRv2 can be trained here (the transformers package, new enough for D-FINE)."""
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        return {"ok": False, "reason": "D-FINE and RT-DETR training are not part of this installation (the transformers "
                                      "package is missing). Install the current LanePulse version."}
    try:
        from transformers import DFineForObjectDetection, RTDetrV2ForObjectDetection  # noqa: F401
    except ImportError:
        return {"ok": False, "reason": "The transformers package here is too old for D-FINE. Install the current "
                                      "LanePulse version."}
    return {"ok": True, "reason": None}


def _check_arch(arch: str, size: str | None) -> str:
    if arch not in ARCHS:
        raise TrainError(f"Unknown detector to train: {arch}. Choose one of {', '.join(ARCHS)}.")
    size = size or ARCHS[arch]["default"]
    if size not in ARCHS[arch]["sizes"]:
        raise TrainError(f"{ARCHS[arch]['title']} size must be one of {', '.join(ARCHS[arch]['sizes'])}.")
    return size


def weights_name(arch: str, size: str) -> str:
    """The starting weights as one file, shipped next to the installation's Python like yolox_s.pth."""
    return f"{arch}-{size}-coco.pth"


def find_coco_weights(arch: str, size: str | None = None) -> Path | None:
    """The released weights shipped with the installation (<prefix>/share/lanepulse/<arch>-<size>-coco.pth)."""
    size = size or ARCHS[arch]["default"]
    for d in shipped_weights_dirs():
        p = d / weights_name(arch, size)
        if p.is_file():
            return p
    return None


def save_start(model, path: Path) -> Path:
    """One file holding the model's settings and weights: {"arch", "config", "model"}."""
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    torch.save({"arch": "d-fine" if model.config.model_type == "d_fine" else "rt-detr",
                "config": model.config.to_dict(), "model": model.state_dict()}, tmp)
    os.replace(tmp, path)
    return path


def coco_weights(arch: str, size: str, log=print, download: bool = True) -> Path:
    found = find_coco_weights(arch, size)
    if found:
        return found
    if not download:
        raise TrainError(f"The {ARCHS[arch]['title']} {size} starting weights are not installed here "
                         f"({weights_name(arch, size)} in {shipped_weights_dirs()[-1]}). Reinstall LanePulse, or "
                         "continue from a trained model of the same kind.")
    cache = Path(os.environ.get("LANEPULSE_CACHE", Path.home() / ".cache" / "lanepulse"))
    log(f"Fetching the {ARCHS[arch]['title']} {size} starting weights (public files, no account) ...")
    return fetch_weights(arch, size, cache)


def fetch_weights(arch: str, size: str, dest_dir: Path) -> Path:
    """Download the released weights as plain public files at a fixed commit (no account, token or hub library),
    check their SHA-256 and keep them as ``dest_dir/<arch>-<size>-coco.pth``. Installers do this when they are built."""
    import hashlib
    import urllib.request
    repo, commit, sha = ARCHS[arch]["sizes"][size]
    dest = Path(dest_dir) / weights_name(arch, size)
    if dest.is_file():
        return dest
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lanepulse-fetch-") as t:
        for f in ("config.json", "model.safetensors"):
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{commit}/{f}", timeout=60) as r, \
                    open(Path(t) / f, "wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
        h = hashlib.sha256()
        with open(Path(t) / "model.safetensors", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            raise TrainError(f"The downloaded {ARCHS[arch]['title']} {size} weights do not match their published "
                             "checksum; nothing was kept.")
        save_start(_model(arch, Path(t), {}), dest)
    return dest


def _init_weights(arch: str, init: str, size: str, work: Path, log, download: bool) -> tuple[Path, dict, str, dict]:
    """(file or folder to load from, what it was, size, its own classes)."""
    if init == "coco":
        p = coco_weights(arch, size, log, download)
        return p, {"name": ARCHS[arch]["sizes"][size][0], "version": "coco", "sha256": _sha256(p)}, size, {}
    p = Path(init)
    if p.suffix == ".pth" and p.is_file():
        return p, {"file": p.name, "sha256": _sha256(p)}, size, {}
    if p.is_dir() and not (p / "model.json").is_file() and (p / "config.json").is_file():
        return p, {"file": str(p), "sha256": _sha256(p / "model.safetensors")}, size, {}
    pack = _open_pack(p, work)
    card = json.loads((pack / "model.json").read_text(encoding="utf-8"))
    if card.get("architecture") != arch:
        raise TrainError(f"{card.get('name')} is not a {ARCHS[arch]['title']} model; start from the released "
                         f"{ARCHS[arch]['title']} weights or from a {ARCHS[arch]['title']} model trained earlier.")
    if not (pack / "model.safetensors").is_file() or not (pack / "config.json").is_file():
        raise TrainError(f"{p} has no training weights (model.safetensors); only packs written by LanePulse can be "
                         "trained further.")
    expect = (card.get("files") or {}).get("model.safetensors")
    if expect and _sha256(pack / "model.safetensors") != expect:
        raise TrainError(f"model.safetensors in {p} does not match its model card; copy the pack again.")
    return pack, {"name": card["name"], "version": card["version"], "sha256": card["sha256"]}, \
        card.get("size", size), card.get("own_classes") or {}


# ------------------------------------------------------------------ data

def grey_unlooked(im: np.ndarray, img: dict, boxes: list[list[float]]) -> np.ndarray:
    """Paint where nobody looked plain grey, keeping every labelled box (x, y, w, h) visible."""
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


def load_split(dataset: Path, split: str, cmap: dict[int, int]) -> list[dict]:
    """One split as [{file, img (the export's image entry), boxes: [[x, y, w, h]], labels: [COCO index]}]."""
    f = Path(dataset) / f"annotations_{split}.json"
    if not f.is_file():
        return []
    data = json.loads(f.read_text(encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in data.get("annotations", []):
        if a["bbox"][2] > 1 and a["bbox"][3] > 1 and a["category_id"] in cmap:
            by_img.setdefault(a["image_id"], []).append(a)
    return [{"file": Path(dataset) / img["file_name"], "img": img,
             "boxes": [a["bbox"] for a in by_img.get(img["id"], [])],
             "labels": [cmap[a["category_id"]] for a in by_img.get(img["id"], [])]} for img in data.get("images", [])]


def picture(item: dict, augment: bool, rng: random.Random):
    """(CHW float tensor in 0..1 at INPUT x INPUT, target {class_labels, boxes as centre and size in 0..1})."""
    import torch
    raw = np.fromfile(str(item["file"]), dtype=np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if im is None:
        raise TrainError(f"Cannot read {item['file'].name}.")
    im = grey_unlooked(im, item["img"], item["boxes"])
    h, w = im.shape[:2]
    boxes = np.array(item["boxes"], np.float32).reshape(-1, 4)
    if augment and rng.random() < 0.5:
        im = im[:, ::-1]
        boxes[:, 0] = w - boxes[:, 0] - boxes[:, 2]
    x = cv2.resize(np.ascontiguousarray(im), (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)[..., ::-1]
    x = x.astype(np.float32) / 255.0
    if augment:
        x = np.clip(x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08), 0, 1)
    cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2] / 2) / w, (boxes[:, 1] + boxes[:, 3] / 2) / h,
                       boxes[:, 2] / w, boxes[:, 3] / h], 1) if len(boxes) else np.zeros((0, 4), np.float32)
    return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            {"class_labels": torch.tensor(item["labels"], dtype=torch.long),
             "boxes": torch.from_numpy(np.clip(cxcywh, 0, 1).astype(np.float32))})


# ------------------------------------------------------------------ training

def _model(arch: str, src: Path, own: dict):
    """The model from a weights file (save_start) or a folder (config.json and model.safetensors)."""
    from transformers import (AutoModelForObjectDetection, DFineConfig, DFineForObjectDetection, RTDetrV2Config,
                              RTDetrV2ForObjectDetection)
    src = Path(src)
    if src.is_file():
        import torch
        ck = torch.load(src, map_location="cpu", weights_only=True)
        if ck.get("arch") != arch:
            raise TrainError(f"{src.name} holds {ck.get('arch')} weights, not {ARCHS[arch]['title']}.")
        cfg_cls, model_cls = (DFineConfig, DFineForObjectDetection) if arch == "d-fine" else \
            (RTDetrV2Config, RTDetrV2ForObjectDetection)
        model = model_cls(cfg_cls.from_dict(ck["config"]))
        model.load_state_dict(ck["model"])
    else:
        model = AutoModelForObjectDetection.from_pretrained(str(src), local_files_only=True)
    if own:                                         # the names a marked target's class shows under
        model.config.id2label = {**model.config.id2label, **{int(k): v for k, v in own.items()}}
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return model


def train(dataset: Path, out: Path, *, arch: str = "d-fine", size: str | None = None, init: str = "coco",
          epochs: int = 20, batch: int | None = None, lr: float | None = None, device: str = "auto",
          workers: int | None = None, name: str | None = None, version: str | None = None, seed: int = 0, log=print,
          progress=None, control=None, download: bool = True, on_epoch=None, **_ignored) -> Path:
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
    size = _check_arch(arch, size)
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
    rng = random.Random(seed)
    work = Path(tempfile.mkdtemp(prefix="lanepulse-detr-"))
    try:
        start, init_src, size, keep = _init_weights(arch, init, size, work, log, download)
        name = name or f"{arch}-{size}-tuned"
        try:
            cmap, own = category_map(dataset, keep or _init_own(init))
        except OwnClassError as exc:
            raise TrainError(str(exc)) from None
        tr_items, va_items = load_split(dataset, "train", cmap), load_split(dataset, "val", cmap)
        if not tr_items:
            raise TrainError("The dataset has no training images.")
        n_boxes = sum(len(i["boxes"]) for i in tr_items)
        log(f"Training images: {len(tr_items)} ({n_boxes} labelled targets), validation images: {len(va_items)}"
            + (f"; taught targets: {', '.join(own.values())}" if own else ""))
        check()
        model = _model(arch, start, own).to(dev)
        backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        opt = torch.optim.AdamW([{"params": backbone, "lr": lr / 10}, {"params": rest, "lr": lr}], weight_decay=1e-4)
        steps_per = max(1, math.ceil(len(tr_items) / batch))
        total = steps_per * epochs
        warm = min(100, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
        hist: list[dict] = []
        best, best_loss = work / "best", None
        t0, step = time.time(), 0

        def run_loss(items, train_mode: bool) -> float | None:
            nonlocal step
            if not items:
                return None
            model.train(train_mode)
            order = list(range(len(items)))
            if train_mode:
                rng.shuffle(order)
            tot, n = 0.0, 0
            for k in range(0, len(order), batch):
                check()
                pairs = [picture(items[i], train_mode, rng) for i in order[k:k + batch]]
                x = torch.stack([p[0] for p in pairs]).to(dev)
                labels = [{kk: v.to(dev) for kk, v in p[1].items()} for p in pairs]
                with torch.set_grad_enabled(train_mode):
                    loss = model(pixel_values=x, labels=labels).loss
                if train_mode:
                    if not torch.isfinite(loss):
                        raise TrainError("Training diverged (the loss is not a number). Try a lower learning rate.")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    opt.step()
                    sched.step()
                    step += 1
                    if progress:
                        progress(step, total)
                tot += float(loss.detach()) * len(pairs)
                n += len(pairs)
            return tot / max(n, 1)

        for ep in range(1, epochs + 1):
            tl = run_loss(tr_items, True)
            vl = run_loss(va_items, False)
            row = {"epoch": ep, "train_loss": round(tl, 4) if tl is not None else None,
                   "val_loss": round(vl, 4) if vl is not None else None}
            hist.append(row)
            score = vl if vl is not None else tl
            if best_loss is None or score < best_loss:
                best_loss = score
                model.save_pretrained(str(best), safe_serialization=True)
            log(f"Epoch {ep}/{epochs}: training loss {row['train_loss']}"
                + ("" if vl is None else f", validation loss {row['val_loss']}")
                + f", {(time.time() - t0) / ep:.0f} s per epoch")
            if on_epoch:
                on_epoch(ep, epochs, row)
        log(f"Keeping the weights with the lowest {'validation' if va_items else 'training'} loss ({best_loss:.4f}).")
        final = _model(arch, best, {}).to("cpu").eval()
        record = {"dataset": {k: manifest.get(k) for k in ("mission_id", "dataset_id", "name", "stats", "exported_at", "datasets")},
                  "dataset_files": {f: _sha256(dataset / f) for f in manifest.get("files", []) if (dataset / f).is_file()},
                  "init": init_src, "epochs": epochs, "batch": batch, "lr": lr, "size": size,
                  "device": describe_device(dev), "torch": torch.__version__, "seed": seed, "history": hist,
                  "own_classes": own, "seconds": round(time.time() - t0)}
        try:
            from importlib.metadata import version as _version
            record["transformers"] = _version("transformers")
        except Exception:
            pass
        return write_pack(final, best, out, arch, name, version or datetime.now().strftime("%Y%m%d-%H%M"), size, own,
                          record, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ model pack

def export_onnx(model, path: Path) -> None:
    """model.onnx: one RGB picture in 0..1 at INPUT x INPUT in; boxes (centre and size, 0..1) and class logits out."""
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, images):
            o = self.m(pixel_values=images)
            return o.pred_boxes, o.logits

    x = torch.rand(1, 3, INPUT, INPUT)
    with torch.no_grad():
        torch.onnx.export(Wrap(model.eval()), (x,), str(path), input_names=["images"], output_names=["dets", "labels"],
                          opset_version=17, dynamo=False)


def write_pack(model, weights: Path, out: Path, arch: str, name: str, version: str, size: str, own: dict, record: dict,
               log=print) -> Path:
    pack = out / name / version
    if pack.exists():
        raise TrainError(f"{pack} exists already; nothing was overwritten.")
    tmp = pack.with_name(version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    log("Writing model.onnx ...")
    export_onnx(model, tmp / "model.onnx")
    for f in ("config.json", "model.safetensors"):
        shutil.copy(weights / f, tmp / f)
    a = ARCHS[arch]
    (tmp / "LICENSE").write_text(
        f"This model was fine-tuned from {a['title']}.\n{a['license']}\nhttps://www.apache.org/licenses/LICENSE-2.0\n"
        "The starting weights were trained on COCO 2017 (annotations CC BY 4.0; the Flickr images carry their own "
        "terms).\nFine-tuning used the mission's own labelled images, exported from LanePulse.\n", encoding="utf-8")
    card = {"name": name, "version": version, "task": "vehicle_detection", "sha256": _sha256(tmp / "model.onnx"),
            "license": f"Apache-2.0 ({a['title']} code and starting weights); fine-tuned on the mission's own images",
            "source": f"trafficwb-train (fine-tuned {a['title']} {size})", "family": "detr", "architecture": arch,
            "size": size, "outputs": "boxes_logits", "class_ids": "coco80", "input_size": INPUT,
            "input": {"colour": "RGB", "mean": [0, 0, 0], "std": [1, 1, 1]},
            "input_format": f"RGB in 0..1, stretched to {INPUT}x{INPUT}",
            "weights": "model.onnx", "torch_weights": "model.safetensors",
            "files": {f: _sha256(tmp / f) for f in ("config.json", "model.safetensors", "LICENSE")},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "training": record}
    if own:
        card["own_classes"] = {str(k): v for k, v in own.items()}
    (tmp / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(pack)
    zpath = out / f"{name}-{version}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pack.iterdir()):
            z.write(f, f"{name}/{version}/{f.name}")
    log(f"Model pack: {pack}")
    return zpath


def released_pack(arch: str, size: str, weights: Path, models_dir: Path, source_version: str = "coco") -> Path:
    """The released model as a pack the app runs (models/<arch>-<size>/coco): for the detector comparison, and as the
    model a trained one is compared with afterwards. No training weights inside; the training starting point is the
    shipped <arch>-<size>-coco.pth."""
    name = f"{arch}-{size}"
    pack = Path(models_dir) / name / source_version
    if (pack / "model.json").is_file():
        return pack
    tmp = pack.with_name(source_version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    export_onnx(_model(arch, weights, {}), tmp / "model.onnx")
    a = ARCHS[arch]
    repo, commit, sha = a["sizes"][size]
    (tmp / "LICENSE").write_text(f"{a['title']}\n{a['license']}\nhttps://www.apache.org/licenses/LICENSE-2.0\n",
                                 encoding="utf-8")
    card = {"name": name, "version": source_version, "task": "vehicle_detection", "sha256": _sha256(tmp / "model.onnx"),
            "license": f"Apache-2.0 ({a['title']} code and released weights)",
            "data_note": "Weights trained on COCO 2017. COCO annotations are CC BY 4.0; the underlying Flickr images "
                         "carry their own terms.",
            "source": f"https://huggingface.co/{repo} (commit {commit}, model.safetensors SHA-256 {sha}), exported to "
                      "ONNX by LanePulse",
            "family": "detr", "architecture": arch, "size": size, "outputs": "boxes_logits", "class_ids": "coco80",
            "input_size": INPUT, "input": {"colour": "RGB", "mean": [0, 0, 0], "std": [1, 1, 1]},
            "input_format": f"RGB in 0..1, stretched to {INPUT}x{INPUT}", "weights": "model.onnx",
            "files": {"LICENSE": _sha256(tmp / "LICENSE")}}
    (tmp / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp.rename(pack)
    return pack
