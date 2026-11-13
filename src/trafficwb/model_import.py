"""Add a model pack to this installation: a .zip written by trafficwb-train, or a pack folder.

    trafficwb-models list
    trafficwb-models import <pack.zip or folder>
    trafficwb-models wrap <model.onnx> --name dfine-s --version 1 --family detr --class-ids coco91 --license Apache-2.0

"wrap" makes a pack folder from a detector exported to ONNX elsewhere (D-FINE's, RT-DETR's or RF-DETR's own export
scripts), so it can be imported and compared with the others on the learning page.

The pack is checked (model card, SHA-256 of every file) before it is copied into the models folder,
and an installed version is never replaced. Importing does not change the model in use: evaluate the
new model on the learning page and promote it there. Needs no PyTorch and no network.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from . import models
from .models import ModelPack, ModelPackError, sha256_file

TASKS = ("vehicle_detection", "plate_detection", "plate_ocr", "segmentation")


def _unpack(src: Path, work: Path) -> Path:
    if src.is_dir():
        hits = [src] if (src / "model.json").is_file() else [p.parent for p in src.rglob("model.json")]
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.startswith(("/", "\\")) or ".." in Path(n).parts:
                    raise ModelPackError(f"{src.name} has an unsafe path inside ({n}); not imported.")
            z.extractall(work)
        hits = [p.parent for p in work.rglob("model.json")]
    else:
        raise ModelPackError(f"{src} is neither a model pack folder nor a .zip of one.")
    if len(hits) != 1:
        raise ModelPackError(f"{src.name} holds {len(hits)} model packs; expected one.")
    return hits[0]


def import_pack(src: Path | str, root: Path | None = None) -> dict:
    """Check and copy a pack into <root>/<name>/<version>; returns its card summary."""
    src, root = Path(src), Path(root or models.models_root())
    if not src.exists():
        raise ModelPackError(f"{src} does not exist.")
    with tempfile.TemporaryDirectory(prefix="lanepulse-import-") as w:
        folder = _unpack(src, Path(w))
        pack = ModelPack.load(folder)                     # checks the card and the weights' SHA-256
        card = pack.card
        if card.get("task") not in TASKS:
            raise ModelPackError(f"Unknown model task '{card.get('task')}'.")
        for name, want in (card.get("files") or {}).items():
            f = folder / name
            if not f.is_file() or sha256_file(f) != want:
                raise ModelPackError(f"{name} in the pack is missing or damaged; copy the pack again.")
        for part in (pack.name, pack.version):
            if not part or any(c in part for c in '\\/:*?"<>|') or part in (".", ".."):
                raise ModelPackError(f"The model name or version '{part}' cannot be used as a folder name.")
        dest = root / pack.name / pack.version
        if dest.exists():
            have = json.loads((dest / "model.json").read_text(encoding="utf-8")).get("sha256") if (dest / "model.json").is_file() else None
            if have == card["sha256"]:
                return {**_summary(card), "already": True, "path": str(dest)}
            raise ModelPackError(f"{pack.name} {pack.version} is installed already with different weights; "
                                 "nothing was replaced. Give the new model another version.")
        tmp = dest.with_name(dest.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, tmp)
        ModelPack.load(tmp)                               # the copy is intact
        tmp.rename(dest)
    return {**_summary(card), "already": False, "path": str(dest)}


def wrap(onnx_path: Path | str, out: Path | str, name: str, version: str, license: str, family: str = "detr",
         outputs: str = "labels_boxes_scores", class_ids: str = "coco91", input_size: int = 640,
         source: str | None = None) -> Path:
    """A vehicle-detection pack folder ``out/<name>/<version>`` holding a copy of ``onnx_path`` and its card."""
    src = Path(onnx_path)
    if not src.is_file():
        raise ModelPackError(f"{src} does not exist.")
    if family not in ("yolox", "detr"):
        raise ModelPackError("The family is yolox or detr.")
    if outputs not in ("labels_boxes_scores", "boxes_logits") or class_ids not in ("coco80", "coco91"):
        raise ModelPackError("Outputs are labels_boxes_scores or boxes_logits; class ids coco80 or coco91.")
    dest = Path(out) / name / version
    if dest.exists():
        raise ModelPackError(f"{dest} exists already; nothing was replaced.")
    dest.mkdir(parents=True)
    shutil.copyfile(src, dest / "model.onnx")
    card = {"name": name, "version": version, "task": "vehicle_detection", "license": license, "family": family,
            "input_size": int(input_size), "sha256": sha256_file(dest / "model.onnx"), "source": source or src.name}
    if family == "detr":
        card |= {"outputs": outputs, "class_ids": class_ids}
    (dest / "model.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    ModelPack.load(dest)
    return dest


def _summary(card: dict) -> dict:
    t = card.get("training") or {}
    return {"name": card["name"], "version": card["version"], "task": card["task"],
            "dataset": (t.get("dataset") or {}).get("name"), "device": t.get("device")}


def main(argv=None) -> int:
    from trafficwb import safe_console
    safe_console()
    ap = argparse.ArgumentParser(prog="trafficwb-models", description="List or import model packs.")
    ap.add_argument("--models", help="models folder (default: TWB_MODELS, or the installation's models folder)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the installed model packs")
    i = sub.add_parser("import", help="check a model pack (.zip or folder) and add it")
    i.add_argument("pack")
    w = sub.add_parser("wrap", help="make a vehicle-detection pack folder from an exported ONNX detector")
    w.add_argument("onnx")
    w.add_argument("--name", required=True)
    w.add_argument("--version", required=True)
    w.add_argument("--license", required=True, help="the licence of the code and weights, e.g. Apache-2.0")
    w.add_argument("--family", default="detr", choices=("detr", "yolox"))
    w.add_argument("--outputs", default="labels_boxes_scores", choices=("labels_boxes_scores", "boxes_logits"))
    w.add_argument("--class-ids", default="coco91", choices=("coco91", "coco80"))
    w.add_argument("--input-size", type=int, default=640)
    w.add_argument("--out", default="packs", help="folder to write the pack into (default: packs)")
    args = ap.parse_args(argv)
    if args.cmd == "wrap":
        try:
            d = wrap(args.onnx, args.out, args.name, args.version, args.license, args.family, args.outputs,
                     args.class_ids, args.input_size)
        except (ModelPackError, OSError) as exc:
            print(f"Not made: {exc}")
            return 1
        print(f"Made {d}. Import it with: trafficwb-models import {d}")
        return 0
    root = Path(args.models) if args.models else None
    if args.cmd == "list":
        for m in models.installed(root=root):
            print(f"{m['task']:<18} {m['name']} {m['version']}" + ("  (evaluation only)" if m["evaluation_only"] else ""))
        return 0
    if args.cmd == "import":
        try:
            r = import_pack(args.pack, root)
        except (ModelPackError, OSError, zipfile.BadZipFile, ValueError) as exc:
            print(f"Not imported: {exc}")
            return 1
        if r["already"]:
            print(f"{r['name']} {r['version']} is already installed (same weights).")
        else:
            print(f"Imported {r['name']} {r['version']} into {r['path']}.")
        print("Next: on the learning page, evaluate it on a dataset, and promote it if it is better than the model in use.")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
