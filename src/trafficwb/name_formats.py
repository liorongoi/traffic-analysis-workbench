"""Filename formats the operator defines, kept in the data folder and shared by every mission.

Stored in <data>/filename_formats.json. Each save is recorded in the audit log. After a save, the next scan
of each mission reads again the names it could not read before (see ingest), so files already in a source
folder are picked up without copying them again.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePath
from zoneinfo import ZoneInfo

from .timeparse import (DEFAULT_PATTERNS, FormatError, MalformedName, TemplatePattern, compile_template,
                        parse_capture_time, set_custom_patterns)

FILE = "filename_formats.json"


def _path(data_root: Path | str) -> Path:
    return Path(data_root) / FILE


def compile_all(items: list[dict]) -> list[TemplatePattern]:
    """Compile a list of {name, template, zone}; FormatError names the format that is wrong."""
    out, names = [], {p.name for p in DEFAULT_PATTERNS}
    for i, it in enumerate(items, 1):
        label = (it.get("name") or "").strip() or f"format {i}"
        try:
            pat = compile_template(it.get("name", ""), it.get("template", ""), it.get("zone") or "UTC")
        except FormatError as exc:
            raise FormatError(f"{label}: {exc}") from None
        if pat.name in names:
            raise FormatError(f"{label}: another format already has this name.")
        names.add(pat.name)
        out.append(pat)
    return out


def load(data_root: Path | str) -> list[dict]:
    p = _path(data_root)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("formats", [])
    except (OSError, ValueError):
        return []


def register(data_root: Path | str) -> list[TemplatePattern]:
    """Make the saved formats active in this process. A format that no longer compiles is left out."""
    pats = []
    for it in load(data_root):
        try:
            pats.append(compile_template(it.get("name", ""), it.get("template", ""), it.get("zone") or "UTC"))
        except FormatError:
            continue
    set_custom_patterns(pats)
    return pats


def save(root, items: list[dict], author: str) -> list[dict]:
    """Validate, write atomically, record in the audit log and activate. `root` is a DataRoot."""
    clean = [{"name": (it.get("name") or "").strip(), "template": (it.get("template") or "").strip(),
              "zone": it.get("zone") or "UTC"} for it in items]
    pats = compile_all(clean)
    before = load(root.path)
    p = _path(root.path)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"formats": clean}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
    root.con.execute("INSERT INTO audit_log(at,author,action,detail) VALUES (?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), author, "filename_formats_saved",
                      json.dumps({"before": before, "after": clean}, ensure_ascii=False)))
    set_custom_patterns(pats)
    return clean


def try_format(template: str, zone: str, names: list[str]) -> dict:
    """How a template reads each name: the parts it found and the UTC time, or why it does not match."""
    try:
        pat = compile_template("try", template, zone or "UTC")
    except FormatError as exc:
        return {"ok": False, "error": str(exc), "results": []}
    results = []
    for n in names:
        name = PurePath(n.replace("\\", "/")).name
        stem = name.rsplit(".", 1)[0] if "." in name else name
        row: dict = {"name": name}
        m = pat.match(stem)
        if m is None:
            row.update(ok=False, reason="does not match this format")
        else:
            row["parts"] = {k: v for k, v in m.groupdict().items() if v is not None}
            try:
                ct = parse_capture_time(name, (pat,))
                row.update(ok=True, utc=ct.utc.isoformat().replace("+00:00", "Z"), precision=ct.precision.value,
                           fields=ct.fields)
                if pat.zone != "UTC":
                    row["local"] = ct.utc.astimezone(ZoneInfo(pat.zone)).strftime("%Y-%m-%d %H:%M:%S %Z")
            except MalformedName as exc:
                row.update(ok=False, reason=exc.reason)
        results.append(row)
    return {"ok": True, "precision": pat.precision.value, "results": results}
