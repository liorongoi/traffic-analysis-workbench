"""Runs of a still-image mission, and what found their vehicles.

Every run keeps its own detections (they are never overwritten), so a mission can be run again with another
detector and the runs compared side by side. The results show each image's latest finished run; earlier runs stay
in the database and are summarised here.
"""
from __future__ import annotations

import json
from collections import Counter

DETECTOR_LABEL = {
    "standard": "YOLOX-s", "rfdetr": "RF-DETR", "sam3": "SAM 3.1",
    "both": "YOLOX-s + SAM 3.1", "rfdetr+sam3": "RF-DETR + SAM 3.1",
}
KIND_LABEL = {"initial": "First run", "continuous_batch": "New files", "reprocess": "Run again"}


def detector_of(config_snapshot: str | None) -> dict:
    """The detector setting a run was created with, and the saved model it used, if any."""
    try:
        cfg = json.loads(config_snapshot or "{}")
    except ValueError:
        cfg = {}
    key = cfg.get("detector") or "standard"
    model = cfg.get("vehicle_model") if key != "sam3" else None
    label = DETECTOR_LABEL.get(key, key) + (f" with “{model}”" if model else "")
    return {"key": key, "saved_model": model, "label": label}


def _model(model_versions: str | None) -> str | None:
    """The model the run actually loaded, as name and version (written when the run starts)."""
    try:
        m = (json.loads(model_versions or "{}") or {}).get("vehicle_detection") or {}
    except ValueError:
        return None
    return f"{m['name']} {m.get('version', '')}".strip() if m.get("name") else None


def summaries(con, limit: int = 20) -> list[dict]:
    """The latest runs, newest first: detector, images, and what was found by class and by model."""
    runs = con.execute("SELECT id, kind, state, config_snapshot, model_versions, created_at, started_at, finished_at "
                       "FROM run ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    if not runs:
        return []
    ids = [r["id"] for r in runs]
    marks = ",".join("?" * len(ids))
    items: dict[int, Counter] = {i: Counter() for i in ids}
    for r in con.execute(f"SELECT run_id, processing_state, COUNT(*) FROM run_item WHERE run_id IN ({marks}) "
                         "GROUP BY run_id, processing_state", ids):
        items[r[0]][r[1]] = r[2]
    by_class: dict[int, Counter] = {i: Counter() for i in ids}
    by_model: dict[int, Counter] = {i: Counter() for i in ids}
    for r in con.execute(
            f"SELECT i.run_id, d.cls, m.name, COUNT(*) FROM detection d JOIN run_item i ON i.id=d.run_item_id "
            f"JOIN model_version m ON m.id=d.model_version_id WHERE i.run_id IN ({marks}) "
            "GROUP BY i.run_id, d.cls, m.name", ids):
        by_class[r[0]][r[1]] += r[3]
        by_model[r[0]][r[2]] += r[3]
    shown = Counter(r[0] for r in con.execute(
        f"SELECT run_id FROM run_item WHERE id IN (SELECT MAX(id) FROM run_item WHERE processing_state='done' "
        f"GROUP BY source_version_id) AND run_id IN ({marks})", ids))
    out = []
    for r in runs:
        n = items[r["id"]]
        out.append({
            "id": r["id"], "kind": r["kind"], "kind_label": KIND_LABEL.get(r["kind"], r["kind"]), "state": r["state"],
            "created_at": r["created_at"], "started_at": r["started_at"], "finished_at": r["finished_at"],
            "detector": detector_of(r["config_snapshot"]), "model": _model(r["model_versions"]),
            "images": sum(n.values()), "done": n["done"], "failed": n["failed"],
            "detections": sum(by_class[r["id"]].values()),
            "by_class": dict(by_class[r["id"]].most_common()), "by_model": dict(by_model[r["id"]].most_common()),
            "shown": shown[r["id"]],      # images whose results on the Results tab come from this run
        })
    return out


def image_runs(con, version_id: int) -> list[dict]:
    """Every finished run of one image, newest first, with what each found. The first one is shown."""
    out = []
    for r in con.execute(
            "SELECT i.id AS item_id, r.id, r.config_snapshot, r.model_versions, r.finished_at, r.started_at, "
            " (SELECT COUNT(*) FROM detection d WHERE d.run_item_id=i.id) AS n "
            "FROM run_item i JOIN run r ON r.id=i.run_id WHERE i.source_version_id=? AND i.processing_state='done' "
            "ORDER BY i.id DESC", (version_id,)):
        classes = Counter(dict(con.execute("SELECT cls, COUNT(*) FROM detection WHERE run_item_id=? GROUP BY cls",
                                           (r["item_id"],)).fetchall()))
        out.append({"run_id": r["id"], "detector": detector_of(r["config_snapshot"]), "model": _model(r["model_versions"]),
                    "detections": r["n"], "by_class": dict(classes.most_common()),
                    "at": r["finished_at"] or r["started_at"]})
    return out


def model_names(con) -> dict[int, str]:
    """Model id to a short name, for saying which model found each box."""
    return {r[0]: f"{r[1]} {r[2]}".strip() for r in con.execute("SELECT id, name, version FROM model_version")}


def rerun_versions(con) -> list[int]:
    """Images to run again: every current image that has finished, or failed, at least once."""
    return [r[0] for r in con.execute(
        "SELECT v.id FROM source_version v WHERE v.superseded_by IS NULL AND v.integrity='ok' "
        "AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND i.processing_state IN "
        "('done','failed','cancelled')) "
        "AND NOT EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND i.processing_state IN "
        "('queued','running')) ORDER BY v.id")]
