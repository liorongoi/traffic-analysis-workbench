"""Command-line access to missions until the web UI exists.

trafficwb-mission create --data DIR --name N --source DIR --output DIR --country IL --zone Asia/Jerusalem [...]
trafficwb-mission scan   --data DIR --mission ID [--stability 30]
trafficwb-mission status --data DIR [--mission ID]
trafficwb-mission detect --data DIR --mission ID [--model yolox-s | --model sam3] [--tiles 3x2]
trafficwb-mission results --data DIR --mission ID [--limit 50]
trafficwb-mission export --data DIR --mission ID [--formats csv,xlsx]
"""
from __future__ import annotations

import argparse
import json
import sys

from .ingest import DEFAULT_STABILITY_SECONDS, cycle
from .store import DataRoot, MissionConfig, MissionValidationError, validate


def main(argv: list[str] | None = None) -> int:
    from trafficwb import safe_console
    safe_console()
    ap = argparse.ArgumentParser(prog="trafficwb-mission")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--data", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--input-type", default="images", choices=["images", "video"])
    c.add_argument("--mode", default="general_traffic")
    c.add_argument("--country", required=True)
    c.add_argument("--zone", required=True, help="IANA time zone, e.g. Asia/Jerusalem")
    c.add_argument("--source", required=True)
    c.add_argument("--no-subdirs", action="store_true")
    c.add_argument("--processing", default="one_time", choices=["one_time", "continuous"])
    c.add_argument("--output", required=True)
    c.add_argument("--formats", default="xlsx,csv")
    c.add_argument("--author", default="operator")
    c.add_argument("--detector", default="standard", choices=["standard", "sam3"],
                   help="vehicle detector for still images: standard (YOLOX-s) or sam3 (SAM 3.1)")
    s = sub.add_parser("scan")
    s.add_argument("--data", required=True)
    s.add_argument("--mission", required=True)
    s.add_argument("--stability", type=int, default=DEFAULT_STABILITY_SECONDS)
    st = sub.add_parser("status")
    st.add_argument("--data", required=True)
    st.add_argument("--mission")
    d = sub.add_parser("detect")
    d.add_argument("--data", required=True)
    d.add_argument("--mission", required=True)
    d.add_argument("--model", help="yolox-s (default), another vehicle detector pack, or sam3 (SAM 3.1); default: the "
                   "mission's own choice")
    d.add_argument("--model-version")
    d.add_argument("--tiles", default="3x2", help="columns x rows, 1x1 to disable tiling")
    d.add_argument("--rule", default="bottom_center", choices=["bottom_center", "centroid", "overlap"])
    r = sub.add_parser("results")
    r.add_argument("--data", required=True)
    r.add_argument("--mission", required=True)
    r.add_argument("--limit", type=int, default=50)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--mission", required=True)
    e.add_argument("--formats")
    e.add_argument("--author", default="operator")
    a = ap.parse_args(argv)

    root = DataRoot(a.data)
    if a.cmd == "create":
        cfg = MissionConfig(name=a.name, input_type=a.input_type, mode=a.mode, country=a.country, iana_zone=a.zone,
                            source_dir=a.source, include_subdirs=not a.no_subdirs, processing=a.processing,
                            output_dir=a.output, export_formats=a.formats.split(","), detector=a.detector)
        val = validate(cfg, root.path)
        for w in val.warnings:
            print(f"warning: {w}", file=sys.stderr)
        try:
            m = root.create_mission(cfg, a.author)
        except MissionValidationError as exc:
            for e in exc.validation.errors:
                print(f"error: {e}", file=sys.stderr)
            return 2
        print(m.id)
        return 0
    if a.cmd == "scan":
        m = root.open_mission(a.mission)
        res, run_id = cycle(m, stability_seconds=a.stability)
        out = {k: (len(v) if isinstance(v, list) and k != "malformed" and k != "corrupt" else v)
               for k, v in res.__dict__.items()}
        out["run_id"] = run_id
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    if a.cmd == "status":
        ids = [a.mission] if a.mission else [r["id"] for r in root.missions()]
        print(json.dumps({i: root.open_mission(i).status() for i in ids}, indent=2))
        return 0
    if a.cmd == "detect":
        from . import models
        from .detect import DetectorSettings, YoloxDetector
        from .process import Processor
        cols, rows = (int(x) for x in a.tiles.lower().split("x"))
        m = root.open_mission(a.mission)
        model = a.model or ("sam3" if m.input_type == "images" and m.config().detector == "sam3" else "yolox-s")
        if model in ("sam3", "sam3.1"):
            from .sam3 import Sam3Detector, engine
            eng = engine()
            det, pack = Sam3Detector(eng), eng.pack
        else:
            pack = models.find(model, a.model_version)
            det = YoloxDetector(pack, DetectorSettings(tiles=(cols, rows)))
        if m.input_type == "video":
            from .video_process import VideoProcessor
            proc = VideoProcessor(m, det, pack.card)
        else:
            proc = Processor(m, det, pack.card, region_rule=a.rule)
        reps = proc.process_all(
            progress=lambda n, t: print(f"\r{n}/{t}", end="", file=sys.stderr, flush=True))
        print(file=sys.stderr)
        print(json.dumps([r.__dict__ for r in reps], indent=2, ensure_ascii=False))
        return 0
    if a.cmd == "results":
        from .review import ReviewService
        m = root.open_mission(a.mission)
        for row in ReviewService(m).results(limit=a.limit):
            print(f"{row.filename}\t{row.capture_local}\t{row.view or '-'}\t{row.vehicle_count}\t"
                  f"{row.processing_state}\t{row.review_status}\t{','.join(row.flags)}")
        return 0
    if a.cmd == "export":
        from .export import export
        m = root.open_mission(a.mission)
        if m.input_type == "video":
            from .video_review import VideoService
            for p in VideoService(m).export(a.author, a.formats.split(",") if a.formats else None):
                print(p)
            return 0
        for p in export(m, a.author, a.formats.split(",") if a.formats else None):
            print(p)
        if m.config().mode == "plate_recognition":
            from .plates import PlateService
            for p in PlateService(m).export(a.author, a.formats.split(",") if a.formats else None):
                print(p)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
