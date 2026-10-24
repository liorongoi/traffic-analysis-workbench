"""Measure processing speed, memory and storage on a folder of images (P9 budgets).

    python scripts/measure.py --images <folder> --work <empty folder> [--mode general_traffic|plate_recognition]
                              [--tiles 3x2] [--accelerator auto|cpu|cuda|directml]

Creates a one-time mission over a copy-free view of the folder, runs the whole
pipeline (scan, vehicle detection, night analysis, plates for plate missions), and
reports images per minute, seconds per image (median and 95th percentile), peak
process memory, and database and evidence size per image. Prints JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path


def peak_rss_mb() -> float | None:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 if sys.platform != "darwin" else r / 1e6
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6
        except Exception:
            return None


def size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0


def main(argv=None) -> int:
    from trafficwb import safe_console
    safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--mode", default="general_traffic")
    ap.add_argument("--country", default="LV")
    ap.add_argument("--zone", default="Europe/Riga")
    ap.add_argument("--tiles", default="3x2")
    ap.add_argument("--accelerator")
    a = ap.parse_args(argv)
    if a.accelerator:
        os.environ["TWB_ACCELERATOR"] = a.accelerator
    from trafficwb import models
    from trafficwb.detect import DetectorSettings, YoloxDetector
    from trafficwb.ingest import cycle
    from trafficwb.process import Processor
    from trafficwb.store import DataRoot, MissionConfig

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=False)
    (work / "out").mkdir()
    root = DataRoot(work / "data")
    m = root.create_mission(MissionConfig(
        name="measure", input_type="images", mode=a.mode, country=a.country, iana_zone=a.zone, source_dir=a.images,
        include_subdirs=False, processing="one_time", output_dir=str(work / "out"), export_formats=["csv"]), "measure")
    t0 = time.perf_counter()
    cycle(m, stability_seconds=0)
    t_scan = time.perf_counter() - t0
    cols, rows = (int(x) for x in a.tiles.split("x"))
    pack = models.find("yolox-s")
    t1 = time.perf_counter()
    det = YoloxDetector(pack, DetectorSettings(tiles=(cols, rows)))
    t_load = time.perf_counter() - t1
    stamps = []
    last = [time.perf_counter()]

    def progress(n, total):
        now = time.perf_counter()
        stamps.append(now - last[0])
        last[0] = now

    plates = None
    if a.mode == "plate_recognition":
        from trafficwb.plates import default_plate_models
        pm = default_plate_models()
        plates = lambda: pm  # noqa: E731
    t2 = time.perf_counter()
    rep = Processor(m, det, pack.card, plate_models=plates).process_all(progress)
    t_proc = time.perf_counter() - t2
    n = m.con.execute("SELECT COUNT(*) FROM source_version").fetchone()[0]
    dets = m.con.execute("SELECT COUNT(*) FROM detection").fetchone()[0]
    m.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    img_mb = sum(p.stat().st_size for p in Path(a.images).iterdir() if p.is_file()) / 1e6
    q = sorted(stamps)
    out = {
        "machine": {"system": platform.system(), "cpu": platform.processor() or platform.machine(),
                    "cpu_count": os.cpu_count(), "python": platform.python_version()},
        "runtime": det.runtime.record(), "tiles": a.tiles, "mode": a.mode,
        "images": n, "detections": dets, "source_mb": round(img_mb, 1),
        "scan_s": round(t_scan, 2), "model_load_s": round(t_load, 2), "processing_s": round(t_proc, 1),
        "images_per_minute": round(60 * n / t_proc, 1) if t_proc else None,
        "s_per_image_median": round(statistics.median(q), 2) if q else None,
        "s_per_image_p95": round(q[int(0.95 * (len(q) - 1))], 2) if q else None,
        "peak_rss_mb": round(peak_rss_mb() or 0, 0),
        "db_kb_per_image": round(1000 * size_mb(m.dir / "mission.db") / n, 1) if n else None,
        "evidence_kb_per_image": round(1000 * size_mb(m.dir / "evidence") / n, 1) if n else None,
        "runs": [{k: v for k, v in r.__dict__.items() if k in ("done", "failed", "detections", "night", "plates")} for r in rep],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
