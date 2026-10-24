"""Command-line access to missions until the web UI exists.

trafficwb-mission create --data DIR --name N --source DIR --output DIR --country IL --zone Asia/Jerusalem [...]
trafficwb-mission scan   --data DIR --mission ID [--stability 30]
trafficwb-mission status --data DIR [--mission ID]
"""
from __future__ import annotations

import argparse
import json
import sys

from .ingest import DEFAULT_STABILITY_SECONDS, cycle
from .store import DataRoot, MissionConfig, MissionValidationError, validate


def main(argv: list[str] | None = None) -> int:
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
    s = sub.add_parser("scan")
    s.add_argument("--data", required=True)
    s.add_argument("--mission", required=True)
    s.add_argument("--stability", type=int, default=DEFAULT_STABILITY_SECONDS)
    st = sub.add_parser("status")
    st.add_argument("--data", required=True)
    st.add_argument("--mission")
    a = ap.parse_args(argv)

    root = DataRoot(a.data)
    if a.cmd == "create":
        cfg = MissionConfig(name=a.name, input_type=a.input_type, mode=a.mode, country=a.country, iana_zone=a.zone,
                            source_dir=a.source, include_subdirs=not a.no_subdirs, processing=a.processing,
                            output_dir=a.output, export_formats=a.formats.split(","))
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
