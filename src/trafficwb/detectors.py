"""The vehicle detectors a mission can choose, and how each one scored on the public benchmarks.

- "standard": YOLOX-s, or the model promoted on the learning page (fast; trainable).
- "rfdetr": RF-DETR (Roboflow, Apache-2.0), a transformer detector; more accurate than YOLOX-s, a little slower;
  trainable on the learning page as well. The mission uses the RF-DETR model promoted for RF-DETR missions if there is
  one, else the released RF-DETR model installed here (Small first, then Medium, then Nano).
- "sam3": SAM 3.1, which finds objects by name; much slower, strongest from the air and in thermal.
- "both": the standard detector and SAM 3.1, merged; "rfdetr+sam3": RF-DETR and SAM 3.1, merged. The picker shows
  these as tick boxes: one fast detector (YOLOX-s or RF-DETR), SAM 3.1, or a fast one together with SAM 3.1.

GRADES are the measured results (docs/benchmarks/README.md): vehicle AP50 per kind of scene, and seconds per image
on 4 CPU cores. The detector picker shows them so the choice is an informed one.
"""
from __future__ import annotations

import json
from pathlib import Path

ARCH_RFDETR = "rf-detr"
# Each choice: (fast detector or None, whether SAM 3.1 runs too).
CHOICES = {"standard": ("standard", False), "rfdetr": ("rfdetr", False), "sam3": (None, True),
           "both": ("standard", True), "rfdetr+sam3": ("rfdetr", True)}


def uses_sam(choice: str | None) -> bool:
    return CHOICES.get(choice or "standard", ("standard", False))[1]


def fast_of(choice: str | None) -> str | None:
    """The fast detector of a choice: "standard" (YOLOX-s or the promoted model), "rfdetr", or None (SAM 3.1 alone)."""
    return CHOICES.get(choice or "standard", ("standard", False))[0]


RFDETR_PREFERENCE = ("rf-detr-small", "rf-detr-medium", "rf-detr-nano")


def is_rfdetr(card: dict | None) -> bool:
    card = card or {}
    return card.get("architecture") == ARCH_RFDETR or str(card.get("name", "")).startswith("rf-detr")


def rfdetr_pack(data_root: Path | str | None = None, models_root: Path | None = None):
    """The RF-DETR model an "rfdetr" mission uses without a saved model of its own: the one promoted on the learning
    page if it is an RF-DETR, else the first released RF-DETR installed. Raises ModelPackError when there is none."""
    from . import models
    if data_root is not None:
        from .learning import ModelRegistry
        prod = ModelRegistry(data_root).production("vehicle_detection")
        if prod:
            try:
                pack = models.find(prod["name"], prod["version"], models_root)
                if is_rfdetr(pack.card):
                    return pack
            except models.ModelPackError:
                pass
    for name in RFDETR_PREFERENCE:
        try:
            return models.find(name, root=models_root)
        except models.ModelPackError:
            continue
    others = [m for m in models.installed("vehicle_detection", models_root) if is_rfdetr(m)]
    if others:
        return models.find(others[-1]["name"], others[-1]["version"], models_root)
    raise models.ModelPackError("RF-DETR is not installed on this computer. Install the current LanePulse version, or "
                                "add an RF-DETR model pack (trafficwb-models import).")


def rfdetr_info(data_root: Path | str | None = None, models_root: Path | None = None) -> dict:
    """For the detector picker: whether RF-DETR can be chosen here, and which model it would be."""
    from . import models
    try:
        pack = rfdetr_pack(data_root, models_root)
    except models.ModelPackError as exc:
        return {"installed": False, "reason": str(exc), "model": None}
    return {"installed": True, "reason": None, "model": {"name": pack.name, "version": pack.version}}


def grades() -> dict:
    """Benchmark results per detector (see GRADES_FILE); empty when the file is missing."""
    try:
        return json.loads(GRADES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


GRADES_FILE = Path(__file__).with_name("detector_grades.json")
