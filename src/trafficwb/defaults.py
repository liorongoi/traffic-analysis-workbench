"""Settings for new missions: what is chosen once, before analysis, and not on each mission.

Three parts, as on the Settings page: what applies to still images and video alike, what applies to still images only
and what applies to video only. A new mission takes these; the operator then only chooses the input folder and a
name. The file is ``<data>/mission_defaults.json``; a missing or damaged file gives the built-in values.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .store import CAMERAS, DETECTORS, EXPORT_FORMATS, MODES, RUN_MODES, TEACHERS

FILE = "mission_defaults.json"

BUILT_IN: dict = {
    "both": {
        "results_folder": None,          # None: a new "<input> LanePulse exports" folder beside the input folder
        "export_formats": ["xlsx", "csv"],
        "iana_zone": None,               # None: the zone of the latest mission (Asia/Jerusalem before any)
        "country": None,
        "watch": False,                  # keep watching the input folder for new files (continuous processing)
        "run_mode": "manual",            # learning: manual | auto
        "teacher": "none",               # learning: none | sam3
        "auto_min_reviews": 20,
        "auto_epochs": 30,
    },
    "images": {"detector": "standard", "mode": "general_traffic", "vehicle_model": None},
    "video": {"detector": "standard", "mode": "object_search", "sam3_interval_s": 1.0, "camera": "fixed",
              "video_workers": 1, "analyse_new": False},
}


def load(data_root: Path | str) -> dict:
    out = json.loads(json.dumps(BUILT_IN))
    try:
        saved = json.loads((Path(data_root) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for part, values in out.items():
        got = saved.get(part) if isinstance(saved, dict) else None
        if isinstance(got, dict):
            values.update({k: v for k, v in got.items() if k in values})
    return out


def check(d: dict) -> list[str]:
    errors: list[str] = []
    b, im, vi = d["both"], d["images"], d["video"]
    if b["results_folder"] is not None and not Path(str(b["results_folder"])).expanduser().is_dir():
        errors.append(f"{b['results_folder']} is not a folder on this workstation.")
    if not b["export_formats"] or not set(b["export_formats"]) <= EXPORT_FORMATS:
        errors.append("Export formats must be Excel, CSV, or both.")
    if b["iana_zone"] is not None:
        try:
            ZoneInfo(b["iana_zone"])
        except (ZoneInfoNotFoundError, ValueError):
            errors.append(f"'{b['iana_zone']}' is not a known time zone.")
    if not isinstance(b["watch"], bool):
        errors.append("Keep watching must be on or off.")
    if b["run_mode"] not in RUN_MODES:
        errors.append(f"Run mode must be one of {sorted(RUN_MODES)}.")
    if b["teacher"] not in TEACHERS:
        errors.append(f"Learn from must be one of {sorted(TEACHERS)}.")
    if not isinstance(b["auto_min_reviews"], int) or not 1 <= b["auto_min_reviews"] <= 100000:
        errors.append("Reviews before an automatic round must be 1 to 100000.")
    if not isinstance(b["auto_epochs"], int) or not 1 <= b["auto_epochs"] <= 500:
        errors.append("Epochs per automatic round must be 1 to 500.")
    if not isinstance(vi["analyse_new"], bool):
        errors.append("Analyse new videos must be on or off.")
    for part, kind in ((im, "images"), (vi, "video")):
        if part["detector"] not in DETECTORS:
            errors.append(f"Detector must be one of {sorted(DETECTORS)}.")
        if part["mode"] not in MODES[kind]:
            errors.append(f"For {kind}, what to find must be one of {sorted(MODES[kind])}.")
    if vi["camera"] not in CAMERAS:
        errors.append(f"Camera must be one of {sorted(CAMERAS)}.")
    try:
        if not 0.2 <= float(vi["sam3_interval_s"]) <= 10:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("SAM 3.1 in video runs every 0.2 to 10 seconds of the recording.")
    if not isinstance(vi["video_workers"], int) or not 1 <= vi["video_workers"] <= 32:
        errors.append("Parts analysed side by side must be 1 to 32.")
    return errors


def save(data_root: Path | str, changes: dict) -> dict:
    """Merge ``changes`` ({part: {key: value}}) into the saved settings, check them and write them atomically."""
    cur = load(data_root)
    for part, values in (changes or {}).items():
        if part not in cur or not isinstance(values, dict):
            raise ValueError(f"Unknown settings part '{part}'.")
        for k, v in values.items():
            if k not in cur[part]:
                raise ValueError(f"Unknown setting '{part}.{k}'.")
            if k == "results_folder" and isinstance(v, str):
                v = v.strip() or None
            cur[part][k] = v
    errors = check(cur)
    if errors:
        raise ValueError(" ".join(errors))
    root = Path(data_root)
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".defaults-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=2)
    os.replace(tmp, root / FILE)
    return cur


def mission_fields(d: dict, input_type: str) -> dict:
    """The part of a new mission's configuration that comes from these settings."""
    b, part = d["both"], d["video" if input_type == "video" else "images"]
    out = {"mode": part["mode"], "detector": part["detector"], "export_formats": list(b["export_formats"]),
           "processing": "continuous" if b["watch"] else "one_time", "run_mode": b["run_mode"], "teacher": b["teacher"],
           "auto_min_reviews": int(b["auto_min_reviews"]), "auto_epochs": int(b["auto_epochs"])}
    if input_type == "video":
        out.update(sam3_interval_s=float(part["sam3_interval_s"]), camera=part["camera"],
                   video_workers=int(part["video_workers"]), analyse_new=bool(part["analyse_new"]))
    else:
        out["vehicle_model"] = part["vehicle_model"]
    return out
