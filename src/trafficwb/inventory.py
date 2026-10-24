"""Sample-data inventory.

Scans a folder read-only and reports formats, dimensions, EXIF presence,
checksums, filename-time parsing and capture intervals. Used in Phase 0 to
define supported formats from real samples.

Usage: trafficwb-inventory <folder> [--recursive] [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .timeparse import MalformedName, parse_capture_time

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v", ".wmv", ".asf", ".dav", ".h264", ".h265"}


@dataclass
class ImageInfo:
    format: str | None
    width: int | None
    height: int | None
    has_exif: bool
    progressive: bool | None
    error: str | None = None


def read_image_header(path: Path) -> ImageInfo:
    with path.open("rb") as f:
        head = f.read(64 * 1024)
    if head[:2] == b"\xff\xd8":
        return _jpeg(path)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return ImageInfo("png", w, h, b"eXIf" in head, None)
    if head[:2] == b"BM":
        w, h = struct.unpack("<ii", head[18:26])
        return ImageInfo("bmp", w, abs(h), False, None)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ImageInfo("tiff", None, None, False, None)
    return ImageInfo(None, None, None, False, None, "unrecognized image signature")


def _jpeg(path: Path) -> ImageInfo:
    data = path.read_bytes()
    i, exif = 2, False
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return ImageInfo("jpeg", None, None, exif, None, f"corrupt marker at byte {i}")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        if marker == 0xE1 and data[i + 4 : i + 10] == b"Exif\x00\x00":
            exif = True
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            truncated = not data.rstrip(b"\x00").endswith(b"\xff\xd9")
            return ImageInfo("jpeg", w, h, exif, marker in (0xC2, 0xC6, 0xCA, 0xCE),
                             "missing end-of-image marker (possibly partial copy)" if truncated else None)
        i += 2 + length
    return ImageInfo("jpeg", None, None, exif, None, "no frame header found")


def probe_video(path: Path) -> dict[str, Any]:
    exe = shutil.which("ffprobe")
    if not exe:
        return {"error": "ffprobe not available"}
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    if r.returncode != 0:
        return {"error": r.stderr.strip()[:500]}
    j = json.loads(r.stdout)
    fmt = j.get("format", {})
    vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    v = vs[0] if vs else {}
    return {
        "container": fmt.get("format_name"),
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "creation_time": (fmt.get("tags") or {}).get("creation_time"),
        "codec": v.get("codec_name"),
        "width": v.get("width"),
        "height": v.get("height"),
        "r_frame_rate": v.get("r_frame_rate"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "time_base": v.get("time_base"),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(root: Path, recursive: bool = False) -> dict[str, Any]:
    files = sorted(root.rglob("*") if recursive else root.glob("*"))
    files = [p for p in files if p.is_file()]
    rows: list[dict[str, Any]] = []
    for p in files:
        ext = p.suffix.lower()
        kind = "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None
        if kind is None:
            continue
        row: dict[str, Any] = {"path": str(p.relative_to(root)), "kind": kind, "bytes": p.stat().st_size, "sha256": sha256(p)}
        try:
            ct = parse_capture_time(p.name)
            row.update(capture_utc=ct.utc.isoformat().replace("+00:00", "Z"), precision=ct.precision.value,
                       pattern=ct.pattern, name_fields=ct.fields)
        except MalformedName as exc:
            row.update(capture_utc=None, parse_error=exc.reason)
        if kind == "image":
            row["image"] = asdict(read_image_header(p))
        else:
            row["video"] = probe_video(p)
        rows.append(row)
    return {"root": str(root), "recursive": recursive, "files": rows, "summary": summarize(rows)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    imgs = [r for r in rows if r["kind"] == "image"]
    by_cam: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r.get("capture_utc"):
            by_cam[(r.get("name_fields") or {}).get("camera", "(none)")].append(r["capture_utc"])
    intervals: dict[str, dict[str, int]] = {}
    from datetime import datetime

    for cam, ts in by_cam.items():
        t = sorted(datetime.fromisoformat(x.replace("Z", "+00:00")) for x in ts)
        gaps = Counter(f"{round((b - a).total_seconds() / 60)} min" for a, b in zip(t, t[1:]))
        intervals[cam] = dict(gaps.most_common())
    sha_counts = Counter(r["sha256"] for r in rows)
    return {
        "images": len(imgs),
        "videos": sum(r["kind"] == "video" for r in rows),
        "patterns": dict(Counter(r.get("pattern", "malformed") for r in rows)),
        "malformed": [r["path"] for r in rows if r.get("capture_utc") is None][:50],
        "image_formats": dict(Counter(r["image"]["format"] for r in imgs)),
        "dimensions": dict(Counter(f'{r["image"]["width"]}x{r["image"]["height"]}' for r in imgs)),
        "with_exif": sum(r["image"]["has_exif"] for r in imgs),
        "image_errors": [{"path": r["path"], "error": r["image"]["error"]} for r in imgs if r["image"]["error"]],
        "identical_content_groups": sum(1 for c in sha_counts.values() if c > 1),
        "capture_range_utc": [min(x for v in by_cam.values() for x in v), max(x for v in by_cam.values() for x in v)] if by_cam else None,
        "interval_histogram_by_camera": intervals,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inventory a sample folder (read-only).")
    ap.add_argument("folder")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args(argv)
    rep = inventory(Path(a.folder), a.recursive)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
