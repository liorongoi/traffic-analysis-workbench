"""Targets the operator marked and followed, as classes of a trained model (no PyTorch needed here)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

# Dataset categories (1 bicycle, 2 car, 3 motorcycle, 4 bus, 5 truck, 6 person) to the model's COCO class index.
CAT_TO_COCO = {1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 0}


class OwnClassError(ValueError):
    pass


# Targets the operator marked and followed (categories 7, 8, ...: a drone, a tractor) take over COCO classes that
# traffic analysis never looks for, in this order (airplane, kite, frisbee, baseball bat, surfboard, tennis racket,
# elephant, bear). The model pack names them (card "own_classes"), so the app finds them by their own names.
OWN_SLOTS = (4, 33, 29, 34, 37, 38, 20, 21)


def category_map(folder: Path, keep: dict | None = None) -> tuple[dict[int, int], dict[int, str]]:
    """(dataset category id -> COCO index, COCO index -> name of a marked target) of an exported dataset. ``keep``:
    the starting model's own classes ({slot: name}); a target it already knows keeps its place."""
    cats = []
    for split in ("train", "val", "test"):
        f = Path(folder) / f"annotations_{split}.json"
        if f.is_file():
            cats = json.loads(f.read_text(encoding="utf-8")).get("categories") or cats
            if cats:
                break
    m, own = dict(CAT_TO_COCO), {}
    known = {v: int(k) for k, v in (keep or {}).items()}
    extra = sorted((c for c in cats if c["id"] not in CAT_TO_COCO), key=lambda c: (c["name"] not in known, c["id"]))
    free = [s for s in OWN_SLOTS if s not in known.values()]
    for c in extra:
        slot = known.get(c["name"])
        if slot is None:
            if not free:
                raise OwnClassError(f"A model can learn at most {len(OWN_SLOTS)} marked kinds of target; this dataset "
                                 "and the starting model have more.")
            slot = free.pop(0)
        m[c["id"]], own[slot] = slot, c["name"]
    return m, own


def _init_own(init: str) -> dict:
    """The own classes ({slot: name}) of a starting model pack (folder or .zip), without unpacking it."""
    if init in ("none", "coco") or init.endswith(".pth"):
        return {}
    p = Path(init)
    try:
        if p.is_dir():
            card = json.loads((p / "model.json").read_text(encoding="utf-8"))
        else:
            with zipfile.ZipFile(p) as z:
                name = next(n for n in z.namelist() if n.endswith("model.json"))
                card = json.loads(z.read(name).decode("utf-8"))
    except (OSError, StopIteration, ValueError, zipfile.BadZipFile):
        return {}
    return card.get("own_classes") or {}
