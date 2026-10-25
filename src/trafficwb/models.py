"""Model packs: versioned folders with weights, a model card, license text and a checksum.

Layout: <models_root>/<name>/<version>/
    model.onnx    weights
    model.json    card: name, version, task, sha256, licenses, source, classes, input size
    LICENSE*      license texts as shipped by the authors

Packs are prepared on a connected staging machine (scripts/fetch_models.py) and
copied to the workstation. Nothing is ever downloaded at run time. Loading a
pack checks the SHA-256 of the weights against the card and refuses on mismatch.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


class ModelPackError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModelPack:
    path: Path
    card: dict

    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def version(self) -> str:
        return self.card["version"]

    @property
    def sha256(self) -> str:
        return self.card["sha256"]

    @property
    def weights(self) -> Path:
        return self.path / self.card.get("weights", "model.onnx")

    @classmethod
    def load(cls, path: Path | str, verify: bool = True) -> "ModelPack":
        path = Path(path)
        card_path = path / "model.json"
        if not card_path.is_file():
            raise ModelPackError(f"No model card (model.json) in {path}.")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for key in ("name", "version", "task", "sha256", "license"):
            if key not in card:
                raise ModelPackError(f"Model card in {path} lacks '{key}'.")
        pack = cls(path, card)
        if not pack.weights.is_file():
            raise ModelPackError(f"Weights file {pack.weights.name} is missing from {path}.")
        if verify:
            actual = sha256_file(pack.weights)
            if actual != card["sha256"]:
                raise ModelPackError(
                    f"Weights in {path} do not match the model card (expected {card['sha256'][:12]}…, found {actual[:12]}…). "
                    "The pack may be damaged or altered; copy it again from the staging machine.")
        return pack


def models_root() -> Path:
    """TWB_MODELS overrides; otherwise <repo or app>/models."""
    env = os.environ.get("TWB_MODELS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"


def find(name: str, version: str | None = None, root: Path | None = None) -> ModelPack:
    base = (root or models_root()) / name
    if not base.is_dir():
        raise ModelPackError(f"Model '{name}' is not installed under {base.parent}.")
    versions = sorted(p.name for p in base.iterdir() if (p / "model.json").is_file())
    if not versions:
        raise ModelPackError(f"Model '{name}' has no installed versions.")
    v = version or versions[-1]
    if v not in versions:
        raise ModelPackError(f"Model '{name}' version {v} is not installed (have: {', '.join(versions)}).")
    return ModelPack.load(base / v)


def installed(task: str | None = None, root: Path | None = None) -> list[dict]:
    """Model packs on disk (cards only, weights not checked here), optionally for one task."""
    base = root or models_root()
    out = []
    if not base.is_dir():
        return out
    for card_path in sorted(base.glob("*/*/model.json")):
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if task and card.get("task") != task:
            continue
        out.append({"name": card.get("name"), "version": card.get("version"), "task": card.get("task"),
                    "license": card.get("license"), "evaluation_only": bool(card.get("evaluation_only")),
                    "sha256": card.get("sha256")})
    return out
