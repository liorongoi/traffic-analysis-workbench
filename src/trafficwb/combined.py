"""Both vehicle detectors on one still image: the standard detector (YOLOX-s, or the promoted model) and SAM 3.1.

They miss different vehicles, so together they find more. A vehicle both of them found is kept once: the box and
class of the more confident one, with each class's best score from either. A vehicle only one of them found is kept
and says which. Every detection is stored with the model that found it; one found by both is stored with the
combined model (named after both), so exports and the review screen show where each vehicle came from.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

import numpy as np

from .detect import Detection, _iou, accepts_area


@dataclass
class CombinedSettings:
    same_vehicle_iou: float = 0.5      # boxes overlapping this much are one vehicle
    same_vehicle_inside: float = 0.8   # ... or when this share of the smaller box lies in the larger one

    def record(self) -> dict:
        return asdict(self)


def _inside(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def merge(standard: list[Detection], sam: list[Detection], s: CombinedSettings | None = None) -> list[Detection]:
    """One list from both detectors' lists; pairs are matched greedily, most overlapping first."""
    s = s or CombinedSettings()
    pairs = []
    for i, a in enumerate(standard):
        for j, b in enumerate(sam):
            iou = _iou(a.box, b.box)
            if iou >= s.same_vehicle_iou or _inside(a.box, b.box) >= s.same_vehicle_inside:
                pairs.append((iou, i, j))
    used_a, used_b, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        a, b = standard[i], sam[j]
        best = a if a.score >= b.score else b
        scores = {c: max(a.class_scores.get(c, 0.0), b.class_scores.get(c, 0.0))
                  for c in set(a.class_scores) | set(b.class_scores)}
        out.append(Detection(best.box, best.cls, max(a.score, b.score), scores, "both"))
    out += [replace(a, found_by="standard") for i, a in enumerate(standard) if i not in used_a]
    out += [replace(b, found_by="sam3") for j, b in enumerate(sam) if j not in used_b]
    return sorted(out, key=lambda d: (-d.score, d.box))


class CombinedDetector:
    """Runs the standard detector, then SAM 3.1, and merges what they found."""

    def __init__(self, standard, standard_card: dict, sam, sam_card: dict, settings: CombinedSettings | None = None):
        self.standard, self.sam = standard, sam
        self.standard_card, self.sam_card = standard_card, sam_card
        self.settings = settings or CombinedSettings()
        self._std_area = accepts_area(standard)
        self._sam_area = accepts_area(sam)

    def parts(self) -> dict[str, tuple[dict, dict]]:
        """Each detector's own record and model card, so each detection can name the model that found it."""
        return {"standard": (self.standard.describe(), self.standard_card), "sam3": (self.sam.describe(), self.sam_card)}

    def card(self) -> dict:
        return {"task": "vehicle_detection",
                "license": f"{self.standard_card.get('license', 'unknown')}; {self.sam_card.get('license', 'unknown')}",
                "parts": [self.standard_card, self.sam_card]}

    def describe(self) -> dict:
        a, b = self.standard.describe(), self.sam.describe()
        ma, mb = a["model"], b["model"]
        sha = hashlib.sha256(f"{ma['sha256']}+{mb['sha256']}".encode()).hexdigest()
        return {"model": {"name": f"{ma['name']} + {mb['name']}", "version": f"{ma['version']} + {mb['version']}",
                          "sha256": sha},
                "preprocessing": {"standard": a.get("preprocessing", {}), "sam3": b.get("preprocessing", {}),
                                  "merge": self.settings.record()},
                "providers": [*a.get("providers", []), *b.get("providers", [])]}

    def detect(self, img_bgr: np.ndarray, area=None) -> list[Detection]:
        a = self.standard.detect(img_bgr, area=area) if area is not None and self._std_area else self.standard.detect(img_bgr)
        b = self.sam.detect(img_bgr, area=area) if area is not None and self._sam_area else self.sam.detect(img_bgr)
        return merge(a, b, self.settings)
