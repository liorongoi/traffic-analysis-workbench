"""Exports of still-image results to CSV and Excel.

- Two tables: one row per image, one row per object (including removed ones,
  so false detections stay traceable).
- Every row carries mission, run and source-version ids, the file's SHA-256,
  capture time in UTC with its precision, and local time with offset and zone.
- Machine predictions, human corrections and approval type are separate
  columns, never merged.
- Text is written as text: Unicode preserved (UTF-8 with BOM for CSV so Excel
  opens it correctly) and identifiers keep leading zeros.
- Files are never overwritten. A new export gets a new name; the write goes to
  a temporary file first and failures are reported and recorded.
- Excel's row limit is handled by continuing on numbered sheets, never by
  dropping rows.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .localtime import to_local
from .review import Filter, ReviewService
from .store import Mission, utcnow

XLSX_MAX_ROWS = 1_048_576

IMAGE_COLUMNS = ["mission_id", "run_id", "source_version_id", "file", "sha256", "capture_utc", "capture_precision",
                 "capture_local", "utc_offset", "time_zone", "view", "condition", "vehicles_in_image",
                 "confirmed_from_candidates", "unresolved_candidates_min", "unresolved_candidates_max",
                 "processing_state", "review_status", "flags"]
OBJECT_COLUMNS = ["mission_id", "run_id", "source_version_id", "file", "sha256", "capture_utc", "capture_local", "view",
                  "object_id", "evidence_level", "origin", "final_status", "final_class", "final_x1", "final_y1", "final_x2", "final_y2",
                  "machine_class", "machine_score", "machine_x1", "machine_y1", "machine_x2", "machine_y2",
                  "correction", "in_region", "counted", "model", "review_status",
                  "light_group_kind", "light_ids", "candidate_count_min", "candidate_count_max", "alternatives",
                  "night_distance_band", "lit_here_share", "night_flags"]


class ExportError(RuntimeError):
    pass


# Evidence level of an object in the export, by where it came from.
EVIDENCE = {"machine": "detected", "change": "vehicle_from_change", "human": "human_added"}

def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def build_tables(mission: Mission, flt: Filter | None = None) -> tuple[list[list[str]], list[list[str]]]:
    rs = ReviewService(mission)
    con = mission.con
    models = {r["id"]: f"{r['name']} {r['version']}" for r in con.execute("SELECT id, name, version FROM model_version")}
    det_model = {r["id"]: r["model_version_id"] for r in con.execute("SELECT id, model_version_id FROM detection")}
    images, objects = [], []
    for row in rs.results(flt, sort="capture_utc"):
        v = con.execute("SELECT sha256, capture_precision FROM source_version WHERE id=?", (row.source_version_id,)).fetchone()
        offset = ""
        if row.capture_utc:
            utc = datetime.fromisoformat(row.capture_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
            lt = to_local(utc, mission.zone)
            secs = int(lt.utc_offset.total_seconds())
            offset = f"{'+' if secs >= 0 else '-'}{abs(secs) // 3600:02d}:{abs(secs) % 3600 // 60:02d}"
        images.append([_fmt(x) for x in [mission.id, row.run_id, row.source_version_id, row.filename, v["sha256"],
                                         row.capture_utc, v["capture_precision"], row.capture_local, offset, mission.zone,
                                         row.view, row.condition, row.vehicle_count, row.confirmed_from_candidates,
                                         row.candidate_count, row.candidate_count_max, row.processing_state,
                                         row.review_status, ";".join(row.flags)]])
        if row.processing_state != "done":
            continue
        for o in rs.objects(row.source_version_id):
            mb = o.machine_box or (None, None, None, None)
            objects.append([_fmt(x) for x in [
                mission.id, row.run_id, row.source_version_id, row.filename, v["sha256"], row.capture_utc, row.capture_local,
                row.view, o.uid, EVIDENCE.get(o.origin, "human_added"), o.origin, o.status, o.cls,
                *o.box, o.machine_cls, o.score, *mb, o.correction,
                {1: "yes", 0: "no", None: "not applied"}[o.in_region], "yes" if o.counted else "no",
                models.get(det_model.get(o.detection_id)) if o.detection_id else
                ("change against the empty background" if o.origin == "change" else ""), row.review_status,
                "", "", "", "", "", "", "", ""]])
        for c in rs.candidates(row.source_version_id):
            level = {"confirmed": "confirmed_vehicle_from_lights", "unresolved": "unresolved_candidate"}.get(
                c.status, "candidate_" + c.status)
            objects.append([_fmt(x) for x in [
                mission.id, row.run_id, row.source_version_id, row.filename, v["sha256"], row.capture_utc, row.capture_local,
                row.view, c.uid, level, "lights" if c.origin == "machine" else "lights_grouped_by_operator", c.status,
                c.cls or "", *c.box, "", c.score, "", "", "", "", "", "", "yes" if c.status == "confirmed" else "no",
                "lights (night analysis)" if c.origin == "machine" else "", row.review_status,
                c.kind, ";".join(str(i) for i in c.member_ids), c.count_min if c.status == "unresolved" else "",
                c.count_max if c.status == "unresolved" else "", "; ".join(c.alternatives), c.band or "",
                c.recurrence if c.recurrence is not None else "", ";".join(c.flags)]])
    return images, objects


def _unique_path(out_dir: Path, stem: str, ext: str) -> Path:
    p = out_dir / f"{stem}.{ext}"
    n = 2
    while p.exists() or p.with_name(p.name + ".partial").exists():
        p = out_dir / f"{stem}_{n}.{ext}"
        n += 1
    return p


def _finish(tmp: Path, final: Path) -> None:
    if final.exists():
        raise ExportError(f"{final.name} appeared while exporting; nothing was overwritten. Export again.")
    os.rename(tmp, final)  # Windows refuses to rename onto an existing file; checked above for POSIX


def write_sheets_csv(out_dir: Path, stem: str, sheets: list[tuple[str, list[str], list[list[str]]]]) -> list[Path]:
    paths = []
    for name, cols, rows in sheets:
        final = _unique_path(out_dir, f"{stem}_{name}", "csv")
        tmp = final.with_name(final.name + ".partial")
        with open(tmp, "x", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(cols)
            w.writerows(rows)
        _finish(tmp, final)
        paths.append(final)
    return paths


def write_sheets_xlsx(out_dir: Path, stem: str, sheets: list[tuple[str, list[str], list[list[str]]]],
                      max_rows: int = XLSX_MAX_ROWS) -> list[Path]:
    from openpyxl import Workbook

    final = _unique_path(out_dir, stem, "xlsx")
    tmp = final.with_name(final.name + ".partial")
    wb = Workbook(write_only=True)
    per_sheet = max_rows - 1
    for name, cols, rows in sheets:
        chunks = [rows[i:i + per_sheet] for i in range(0, len(rows), per_sheet)] or [[]]
        for k, chunk in enumerate(chunks, 1):
            ws = wb.create_sheet(name if k == 1 else f"{name}_{k}")
            ws.append(cols)
            for r in chunk:
                ws.append(r)  # all values are strings, so Excel keeps leading zeros and exact text
    with open(tmp, "xb") as f:
        wb.save(f)
    _finish(tmp, final)
    return [final]


def write_csv(out_dir: Path, stem: str, images, objects) -> list[Path]:
    return write_sheets_csv(out_dir, stem, [("images", IMAGE_COLUMNS, images), ("objects", OBJECT_COLUMNS, objects)])


def write_xlsx(out_dir: Path, stem: str, images, objects, max_rows: int = XLSX_MAX_ROWS) -> list[Path]:
    return write_sheets_xlsx(out_dir, stem, [("images", IMAGE_COLUMNS, images), ("objects", OBJECT_COLUMNS, objects)],
                             max_rows)


def export(mission: Mission, author: str, formats: list[str] | None = None, flt: Filter | None = None,
           out_dir: Path | str | None = None) -> list[Path]:
    cfg = mission.config()
    formats = formats or cfg.export_formats
    out = Path(out_dir or cfg.output_dir)
    scope = json.dumps((flt or Filter()).describe())
    stem = f"{cfg.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    paths: list[Path] = []
    try:
        images, objects = build_tables(mission, flt)
        out.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            if fmt == "csv":
                new = write_csv(out, stem, images, objects)
            elif fmt == "xlsx":
                new = write_xlsx(out, stem, images, objects)
            else:
                raise ExportError(f"Unknown export format '{fmt}'.")
            paths += new
            mission.con.execute(
                "INSERT INTO export(format,path,scope,rows,status,author,created_at) VALUES (?,?,?,?,?,?,?)",
                (fmt, ";".join(str(p) for p in new), scope, len(images) + len(objects), "completed", author, utcnow()))
    except Exception as exc:
        mission.con.execute("INSERT INTO export(format,path,scope,rows,status,error,author,created_at) VALUES (?,?,?,?,?,?,?,?)",
                            (",".join(formats), None, scope, None, "failed", str(exc)[:500], author, utcnow()))
        mission.notify("error", "export_failed", f"Export failed: {exc}")
        raise
    mission.notify("info", "export_completed", f"Exported {len(images)} images to {out}.", paths=[str(p) for p in paths])
    return paths
