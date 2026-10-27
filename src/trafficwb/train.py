"""Fine-tune the vehicle detector on a dataset exported from LanePulse (LRN-02 vehicle detector adapter).

Runs on a computer with PyTorch: an NVIDIA GPU (CUDA), an Apple-silicon Mac (Metal, "mps"), or the CPU
(slow; for trying it out). The result is a model pack, the same kind of folder the app already uses, also
written as one .zip file. Copy the .zip to any installation and import it there:

    trafficwb-models import yolox-s-tuned-20261027-1530.zip

Then evaluate it on the learning page and promote it if it is better. Nothing here changes a model in use.

    trafficwb-train run --dataset <exported dataset folder> --out <folder>
    trafficwb-train device              which accelerator would be used

Starting weights: --init coco (the released YOLOX-s weights, fetched once and checked against their
SHA-256, then kept in ~/.cache/lanepulse), a .pth file, or an earlier trained pack (folder or .zip) to
continue from it.
"""
from __future__ import annotations

import os

# Apple GPUs: run the few operations Metal lacks on the CPU instead of failing. Read when torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .yolox_torch import YOLOX, ExportWrapper, load_checkpoint, yolox_loss

SIZE = 640
PAD = 114
# Dataset categories (1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck) to the model's COCO class index.
CAT_TO_COCO = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7}
COCO_WEIGHTS = {
    "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "sha256": "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30",
    "name": "yolox-s", "version": "0.1.1rc0",
}
LICENSE_NOTE = """This model was fine-tuned from YOLOX-s (Megvii, https://github.com/Megvii-BaseDetection/YOLOX).
YOLOX code and released weights: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
The starting weights were trained on COCO 2017 (annotations CC BY 4.0; the Flickr images carry their own terms).
Fine-tuning used the mission's own reviewed images, exported from LanePulse.
"""


class TrainError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ device

def pick_device(want: str = "auto") -> torch.device:
    if want != "auto":
        if want == "cuda" and not torch.cuda.is_available():
            raise TrainError("No NVIDIA GPU is usable here (PyTorch was built without CUDA, or the driver is too old).")
        if want == "mps" and not torch.backends.mps.is_available():
            raise TrainError("The Apple GPU is not usable here (needs an Apple-silicon Mac and macOS 14 or later).")
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return f"NVIDIA {p.name}, {p.total_memory / 2**30:.0f} GB (CUDA)"
    if dev.type == "mps":
        return "Apple GPU (Metal)"
    return f"CPU, {os.cpu_count()} threads (slow: for trying it out)"


# ------------------------------------------------------------------ data

def _polygon_mask(shape, polys) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(p, np.float32)).astype(np.int32)], 1)
    return m


def background_mask(img: dict, h: int, w: int) -> np.ndarray:
    """1 where nothing unlabelled is a vehicle (may learn "background"), 0 where nobody looked."""
    if img.get("review") != "full":
        return np.zeros((h, w), np.uint8)                   # partial review: only the corrected boxes teach
    m = _polygon_mask((h, w), img.get("include_regions") or []) if img.get("include_regions") else np.ones((h, w), np.uint8)
    if img.get("exclude_regions"):
        m[_polygon_mask((h, w), img["exclude_regions"]) > 0] = 0
    for x1, y1, x2, y2 in img.get("ignore_boxes") or []:
        m[max(0, int(y1)):max(0, int(math.ceil(y2))), max(0, int(x1)):max(0, int(math.ceil(x2)))] = 0
    return m


class ExportedDataset(torch.utils.data.Dataset):
    """One split of a dataset exported by the learning page (annotations_<split>.json + images/)."""

    def __init__(self, folder: Path, split: str, augment: bool):
        self.folder, self.augment = Path(folder), augment
        f = self.folder / f"annotations_{split}.json"
        data = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"images": [], "annotations": []}
        by_img: dict[int, list] = {}
        for a in data["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        self.items = []
        for img in data["images"]:
            boxes = []
            for a in by_img.get(img["id"], []):
                x, y, bw, bh = a["bbox"]
                if bw > 1 and bh > 1:
                    boxes.append([CAT_TO_COCO.get(a["category_id"], 2), x, y, x + bw, y + bh])
            self.items.append((img, np.asarray(boxes, np.float32).reshape(-1, 5)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, boxes = self.items[i]
        data = np.fromfile(str(self.folder / img["file_name"]), dtype=np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if im is None:
            raise TrainError(f"Cannot read {img['file_name']}.")
        h, w = im.shape[:2]
        mask = background_mask(img, h, w)
        canvas, cmask, out = place(im, mask, boxes.copy(), self.augment)
        x = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float()
        return x, torch.from_numpy(cmask), torch.from_numpy(out)


def place(im, mask, boxes, augment: bool):
    """Letterbox to SIZE x SIZE as the app does (top-left, pad 114); with augment, also random scale,
    position, mirror and brightness. Boxes cut too much by the edge become "nobody looked" there."""
    h, w = im.shape[:2]
    r = min(SIZE / h, SIZE / w)
    ox = oy = 0
    flip = False
    if augment and random.random() > 0.25:
        r *= random.uniform(0.6, 1.6)
        nw, nh = max(1, round(w * r)), max(1, round(h * r))
        ox = random.randint(min(0, SIZE - nw), max(0, SIZE - nw))
        oy = random.randint(min(0, SIZE - nh), max(0, SIZE - nh))
        flip = random.random() < 0.5
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    rs = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rm = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((SIZE, SIZE, 3), PAD, np.uint8)
    cmask = np.zeros((SIZE, SIZE), np.uint8)
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(SIZE, ox + nw), min(SIZE, oy + nh)
    canvas[y0:y1, x0:x1] = rs[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    cmask[y0:y1, x0:x1] = rm[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if augment:
        gain = random.uniform(0.75, 1.25)
        bias = random.uniform(-20, 20)
        canvas = np.clip(canvas.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    out = []
    for c, bx1, by1, bx2, by2 in boxes:
        b = np.array([bx1 * r + ox, by1 * r + oy, bx2 * r + ox, by2 * r + oy])
        cb = np.clip(b, 0, SIZE)
        full = (b[2] - b[0]) * (b[3] - b[1])
        vis = (cb[2] - cb[0]) * (cb[3] - cb[1])
        if vis <= 0:
            continue
        if vis < 0.4 * full or cb[2] - cb[0] < 3 or cb[3] - cb[1] < 3:
            cmask[int(cb[1]):int(math.ceil(cb[3])), int(cb[0]):int(math.ceil(cb[2]))] = 0
            continue
        out.append([c, cb[0], cb[1], cb[2], cb[3]])
    if flip:
        canvas, cmask = canvas[:, ::-1], cmask[:, ::-1]
        for o in out:
            o[1], o[3] = SIZE - o[3], SIZE - o[1]
    t = np.asarray(out, np.float32).reshape(-1, 5)
    if len(t):                                           # to [cls, cx, cy, w, h]
        t = np.stack([t[:, 0], (t[:, 1] + t[:, 3]) / 2, (t[:, 2] + t[:, 4]) / 2, t[:, 3] - t[:, 1], t[:, 4] - t[:, 2]], 1)
    return np.ascontiguousarray(canvas), np.ascontiguousarray(cmask), t


def collate(batch):
    xs, ms, ts = zip(*batch)
    return torch.stack(xs), torch.stack(ms), list(ts)


def anchor_weights(masks: torch.Tensor, grid: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    """The background mask read at every anchor's centre: (N, A)."""
    c = ((grid + 0.5) * stride[:, None]).long().clamp(0, SIZE - 1)
    return masks[:, c[:, 1], c[:, 0]].float()


# ------------------------------------------------------------------ starting weights

def _cache_dir() -> Path:
    return Path(os.environ.get("LANEPULSE_CACHE", Path.home() / ".cache" / "lanepulse"))


def coco_weights(log=print) -> Path:
    path = _cache_dir() / "yolox_s.pth"
    if path.is_file() and _sha256(path) == COCO_WEIGHTS["sha256"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Fetching the YOLOX-s starting weights (about 70 MB) into {path.parent} ...")
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(COCO_WEIGHTS["url"], timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    if _sha256(tmp) != COCO_WEIGHTS["sha256"]:
        tmp.unlink(missing_ok=True)
        raise TrainError("The downloaded starting weights do not match their published checksum; nothing was used.")
    tmp.replace(path)
    return path


def _open_pack(src: Path, work: Path) -> Path:
    """A pack folder, or a .zip of one unpacked into work; returns the folder with model.json."""
    if src.is_dir():
        hits = [src] if (src / "model.json").is_file() else [p.parent for p in src.rglob("model.json")]
    else:
        with zipfile.ZipFile(src) as z:
            z.extractall(work)
        hits = [p.parent for p in work.rglob("model.json")]
    if len(hits) != 1:
        raise TrainError(f"{src} does not hold exactly one model pack.")
    return hits[0]


def load_init(model: YOLOX, init: str, work: Path, log=print) -> dict:
    """Load starting weights; returns what they were, for the model card."""
    if init == "none":                                     # random weights: only for tests
        return {"file": None}
    if init == "coco":
        p = coco_weights(log)
        missing = load_checkpoint(model, str(p))
        src = {"name": COCO_WEIGHTS["name"], "version": COCO_WEIGHTS["version"], "file": "yolox_s.pth",
               "sha256": COCO_WEIGHTS["sha256"]}
    else:
        p = Path(init)
        if p.suffix == ".pth" and p.is_file():
            missing = load_checkpoint(model, str(p))
            src = {"file": p.name, "sha256": _sha256(p)}
        else:
            pack = _open_pack(p, work)
            card = json.loads((pack / "model.json").read_text(encoding="utf-8"))
            pt = pack / (card.get("torch_weights") or "")
            if not card.get("torch_weights") or not pt.is_file():
                raise TrainError(f"{p} has no training weights (model.pt); only packs made by trafficwb-train can be "
                                 "trained further. Use --init coco to start from the released weights.")
            expect = (card.get("files") or {}).get(pt.name)
            if expect and _sha256(pt) != expect:
                raise TrainError(f"{pt.name} in {p} does not match its model card; copy the pack again.")
            sd = torch.load(pt, map_location="cpu", weights_only=True)
            own = model.state_dict()
            fit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            model.load_state_dict(fit, strict=False)
            missing = sorted(set(own) - set(fit))
            src = {"name": card["name"], "version": card["version"], "sha256": card["sha256"]}
    if missing:
        raise TrainError(f"The starting weights do not fit YOLOX-s ({len(missing)} missing, e.g. {missing[0]}).")
    return src


# ------------------------------------------------------------------ training

class EMA:
    """Moving average of the weights; it is what gets exported (steadier than the last step)."""

    def __init__(self, model, decay=0.9998, tau=2000):
        self.m = copy.deepcopy(model).eval()
        self.decay, self.tau, self.updates = decay, tau, 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.m.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _param_groups(model, wd):
    decay, no_decay = [], []
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            (no_decay if isinstance(mod, torch.nn.BatchNorm2d) or name == "bias" else decay).append(p)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": wd}]


def _loss(model, x, masks, targets, dev, amp):
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
        raw, grid, stride = model(x)
    nw = anchor_weights(masks, grid, stride)
    return yolox_loss(raw.float(), grid, stride, [t.to(dev) for t in targets], nw, raw.shape[-1] - 5)


@torch.no_grad()
def validate(model, loader, dev, amp) -> float | None:
    if loader is None:
        return None
    model.eval()
    tot, n = 0.0, 0
    for x, masks, targets in loader:
        loss, _ = _loss(model, x.to(dev), masks.to(dev), targets, dev, amp)
        tot += float(loss) * len(x)
        n += len(x)
    return tot / max(n, 1)


def train(dataset: Path, out: Path, *, init: str = "coco", epochs: int = 40, batch: int | None = None,
          lr: float | None = None, device: str = "auto", workers: int | None = None, name: str = "yolox-s-tuned",
          version: str | None = None, seed: int = 0, freeze_bn: bool | None = None, size: str = "s",
          log=print) -> Path:
    """Train and write the model pack; returns the .zip path."""
    dataset, out = Path(dataset), Path(out)
    manifest_f = dataset / "manifest.json"
    if not manifest_f.is_file():
        raise TrainError(f"{dataset} is not an exported dataset (no manifest.json). Export one on the learning page.")
    manifest = json.loads(manifest_f.read_text(encoding="utf-8"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = pick_device(device)
    amp = dev.type == "cuda"
    batch = batch or (16 if dev.type == "cuda" else 8)
    lr = lr if lr is not None else 0.01 * batch / 64
    freeze_bn = batch < 32 if freeze_bn is None else freeze_bn
    workers = workers if workers is not None else min(4, os.cpu_count() or 1)
    log(f"Device: {describe_device(dev)}")

    tr = ExportedDataset(dataset, "train", augment=True)
    va = ExportedDataset(dataset, "val", augment=False)
    if len(tr) == 0:
        raise TrainError("The dataset has no training images.")
    nbox = sum(len(b) for _, b in tr.items)
    log(f"Training images: {len(tr)} ({nbox} vehicles), validation images: {len(va)}")
    dl_kw = dict(num_workers=workers, collate_fn=collate, persistent_workers=workers > 0)
    tl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, drop_last=len(tr) >= batch, **dl_kw)
    vl = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, **dl_kw) if len(va) else None

    work = Path(tempfile.mkdtemp(prefix="lanepulse-train-"))
    try:
        model = YOLOX(size, 80)
        init_src = load_init(model, init, work, log)
        model.to(dev)
        ema = EMA(model)
        opt = torch.optim.SGD(_param_groups(model, 5e-4), lr=lr, momentum=0.9, nesterov=True)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        steps = epochs * len(tl)
        warm = min(len(tl) * 2, max(steps // 10, 1))
        best, best_sd, best_epoch, step, hist = math.inf, None, 0, 0, []
        base_val = validate(ema.m, vl, dev, amp)
        if base_val is not None:
            log(f"Validation loss before training: {base_val:.3f}")
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            if freeze_bn:                                # small batches: keep the learned image statistics
                for mod in model.modules():
                    if isinstance(mod, torch.nn.BatchNorm2d):
                        mod.eval()
            run, parts = 0.0, {"iou": 0.0, "obj": 0.0, "cls": 0.0}
            for x, masks, targets in tl:
                f = step / warm if step < warm else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps - warm, 1)))
                for g in opt.param_groups:
                    g["lr"] = lr * f
                loss, p = _loss(model, x.to(dev, non_blocking=True), masks.to(dev), targets, dev, amp)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                step += 1
                run += float(loss.detach())
                for k in parts:
                    parts[k] += p[k]
            n = max(len(tl), 1)
            v = validate(ema.m, vl, dev, amp)
            hist.append({"epoch": ep, "train_loss": round(run / n, 4), "val_loss": None if v is None else round(v, 4)})
            log(f"Epoch {ep}/{epochs}: training loss {run / n:.3f} (box {parts['iou'] / n:.3f}, object {parts['obj'] / n:.3f}, "
                f"class {parts['cls'] / n:.3f})" + ("" if v is None else f", validation loss {v:.3f}")
                + f", {(time.time() - t0) / ep:.0f} s per epoch")
            score = v if v is not None else -ep
            if score < best:
                best, best_epoch = score, ep
                best_sd = {k: t.detach().to("cpu").clone() for k, t in ema.m.state_dict().items()}
        log(f"Keeping epoch {best_epoch}" + (" (lowest validation loss)." if vl is not None else " (the last)."))
        final = YOLOX(size, 80)
        final.load_state_dict(best_sd)
        final.eval()
        record = {"dataset": {k: manifest.get(k) for k in ("mission_id", "dataset_id", "name", "stats", "exported_at")},
                  "dataset_files": {f: _sha256(dataset / f) for f in manifest.get("files", []) if (dataset / f).is_file()},
                  "init": init_src, "epochs": epochs, "best_epoch": best_epoch, "batch": batch, "lr": lr, "freeze_bn": freeze_bn,
                  "device": describe_device(dev), "torch": torch.__version__, "seed": seed,
                  "val_loss_before": None if base_val is None else round(base_val, 4),
                  "val_loss_after": None if vl is None else round(best, 4), "history": hist,
                  "seconds": round(time.time() - t0)}
        return write_pack(final, out, name, version or datetime.now().strftime("%Y%m%d-%H%M"), record, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ model pack

def write_pack(model: YOLOX, out: Path, name: str, version: str, record: dict, log=print) -> Path:
    """model.onnx (what the app runs), model.pt (to train further), model.json, LICENSE; and a .zip of it."""
    pack = out / name / version
    if pack.exists():
        raise TrainError(f"{pack} exists already; nothing was overwritten.")
    tmp = pack.with_name(version + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model = model.float().cpu()
    wrap = ExportWrapper(model).eval()                     # a new module starts in training mode; export keeps it
    onnx_path = tmp / "model.onnx"
    torch.onnx.export(wrap, torch.zeros(1, 3, SIZE, SIZE), str(onnx_path), opset_version=11,
                      input_names=["images"], output_names=["output"], dynamo=False)
    check = None
    try:                                                    # same answers from the exported file?
        import onnxruntime as ort
        x = torch.rand(1, 3, SIZE, SIZE) * 255
        wrap.eval()
        with torch.no_grad():
            want = wrap(x).numpy()
        got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
        check = float(np.abs(want - got).max())
        if check > 1e-2:
            raise TrainError(f"The exported model gives different answers (difference {check:.4f}); not written.")
    except ImportError:
        log("onnxruntime is not installed here, so the exported file was not double-checked.")
    torch.save(model.state_dict(), tmp / "model.pt")
    (tmp / "LICENSE").write_text(LICENSE_NOTE, encoding="utf-8")
    record["export_check_max_difference"] = check
    card = {"name": name, "version": version, "task": "vehicle_detection", "sha256": _sha256(onnx_path),
            "license": "Apache-2.0 (YOLOX code and starting weights); fine-tuned on the mission's own reviewed images",
            "source": "trafficwb-train (fine-tuned YOLOX-s)", "input_size": SIZE,
            "input_format": "BGR uint8 values as float32, letterboxed top-left, pad value 114, no normalisation",
            "classes": "coco80", "weights": "model.onnx", "torch_weights": "model.pt",
            "files": {"model.pt": _sha256(tmp / "model.pt"), "LICENSE": _sha256(tmp / "LICENSE")},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "training": record}
    (tmp / "model.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(pack)
    zpath = out / f"{name}-{version}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pack.iterdir()):
            z.write(f, f"{name}/{version}/{f.name}")
    log(f"Model pack: {pack}")
    log(f"To use it on another installation, copy {zpath.name} there and import it "
        f"(drag it onto import-model.cmd, or: trafficwb-models import {zpath.name}).")
    return zpath


# ------------------------------------------------------------------ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="trafficwb-train", description="Fine-tune the vehicle detector on an exported dataset.")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="train and write a model pack")
    r.add_argument("--dataset", required=True, help="folder exported from the learning page")
    r.add_argument("--out", required=True, help="where to write the model pack and its .zip")
    r.add_argument("--init", default="coco", help="coco (released YOLOX-s weights), a .pth file, or an earlier trained pack")
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--batch", type=int, help="images per step (default 16 on NVIDIA, 8 otherwise)")
    r.add_argument("--lr", type=float, help="learning rate (default 0.01 x batch / 64)")
    r.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    r.add_argument("--workers", type=int, help="image loading processes (default up to 4)")
    r.add_argument("--name", default="yolox-s-tuned", help="model name")
    r.add_argument("--version", help="model version (default: date and time)")
    r.add_argument("--seed", type=int, default=0)
    sub.add_parser("device", help="show which accelerator training would use")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            train(Path(args.dataset), Path(args.out), init=args.init, epochs=args.epochs, batch=args.batch, lr=args.lr,
                  device=args.device, workers=args.workers, name=args.name, version=args.version, seed=args.seed)
        elif args.cmd == "device":
            print(describe_device(pick_device()))
        else:
            ap.print_help()
            return 2
    except TrainError as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
