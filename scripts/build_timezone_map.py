"""Simplify a timezone-boundary-builder GeoJSON release into an offline SVG path map.

The source archive is not needed at runtime. See docs/design/time-and-schedules.md for
the pinned release, checksum, attribution, and regeneration command.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

WIDTH, HEIGHT = 1000, 500


def project(point: list[float]) -> tuple[float, float]:
    lon, lat = point[:2]
    return ((lon + 180.0) * WIDTH / 360.0, (90.0 - lat) * HEIGHT / 180.0)


def distance_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return ((p[0] - (a[0] + t * dx)) ** 2 + (p[1] - (a[1] + t * dy)) ** 2) ** 0.5


def simplify(points: list[tuple[float, float]], tolerance: float = 0.7) -> list[tuple[float, float]]:
    if len(points) <= 4:
        return points
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        farthest, largest = -1, tolerance
        for i in range(first + 1, last):
            distance = distance_to_segment(points[i], points[first], points[last])
            if distance > largest:
                farthest, largest = i, distance
        if farthest >= 0:
            keep.add(farthest)
            stack.extend(((first, farthest), (farthest, last)))
    result = [points[i] for i in sorted(keep)]
    return result if len(result) >= 4 else points[:4]


def ring_path(ring: list[list[float]]) -> str:
    pts = [project(p) for p in ring]
    # Split polygon segments crossing the antimeridian to avoid drawing a chord across the map.
    parts: list[list[tuple[float, float]]] = [[]]
    previous_lon: float | None = None
    for raw, xy in zip(ring, pts):
        lon = raw[0]
        if previous_lon is not None and abs(lon - previous_lon) > 180:
            if parts[-1]:
                parts.append([])
        parts[-1].append(xy)
        previous_lon = lon
    commands: list[str] = []
    for part in parts:
        if len(part) < 4:
            continue
        points = simplify(part)
        if len(points) < 4:
            continue
        commands.append("M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z")
    return " ".join(commands)


def feature_path(geometry: dict) -> str:
    geom_type, coords = geometry["type"], geometry["coordinates"]
    polygons = [coords] if geom_type == "Polygon" else coords if geom_type == "MultiPolygon" else []
    return " ".join(ring_path(ring) for polygon in polygons for ring in polygon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="timezone-boundary-builder *-now.geojson.zip release archive")
    parser.add_argument("--out", type=Path, default=Path("ui/src/data/timezones.json"))
    args = parser.parse_args()
    with zipfile.ZipFile(args.source) as archive:
        candidates = [n for n in archive.namelist() if n.endswith(".json")]
        if len(candidates) != 1:
            parser.error(f"expected one GeoJSON file inside the archive, found {len(candidates)}")
        with archive.open(candidates[0]) as source:
            geojson = json.load(source)
    features = []
    for feature in geojson.get("features", []):
        zone = feature.get("properties", {}).get("tzid")
        if zone:
            path = feature_path(feature["geometry"])
            if path:
                features.append({"zone": zone, "path": path})
    if not features:
        parser.error("no timezone polygons found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"width": WIDTH, "height": HEIGHT, "features": features}, separators=(",", ":")),
                        encoding="utf-8")
    print(f"Wrote {len(features)} timezone shapes to {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
