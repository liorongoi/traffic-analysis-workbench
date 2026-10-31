"""FastAPI application: JSON API under /api and the built UI as static files.

Browsers never touch database files (ARC-02); every read and write goes through here.
Each request opens its own database connections, so request threads never share one.
Mutating requests need the X-TWB header, which blocks cross-site form posts.
"""
from __future__ import annotations

import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import zoneinfo
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..detect import VEHICLE_CLASSES
from ..change import CONDITIONS, ChangeService
from ..export import export as run_export
from ..localtime import to_local
from ..regions import RegionEditor, RegionSet
from ..review import Filter, ReviewService
from ..store import (COUNTRY_ZONES, MODES, TARGET_COUNTRIES, DataRoot, Mission, MissionConfig, MissionValidationError,
                     connect, utcnow, validate)
from ..timeparse import MalformedName, parse_capture_time
from ..views import ViewService, read_image
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, action: str | None = None):
        super().__init__(status, {"error": code, "message": message, "action": action})


@dataclass
class User:
    id: int
    username: str
    display_name: str
    role: str


# ---------- request bodies ----------

class MissionBody(BaseModel):
    name: str
    input_type: str = "images"
    mode: str = "general_traffic"
    country: str
    iana_zone: str
    source_dir: str
    include_subdirs: bool = True
    processing: str = "one_time"
    output_dir: str
    export_formats: list[str] = Field(default_factory=lambda: ["xlsx", "csv"])
    latitude: float | None = None
    longitude: float | None = None
    detector: str = "standard"
    sam3_words: dict[str, list[str]] | None = None


class DetectorBody(BaseModel):
    detector: str


class WordsBody(BaseModel):
    words: dict[str, list[str]] | None = None     # None: back to the default words


class FindAreasBody(BaseModel):
    prompts: list[str] = Field(min_length=1, max_length=8)
    kind: str = "include"
    image_id: int
    threshold: float = Field(0.4, ge=0.05, le=0.95)


class EmptyBody(BaseModel):
    empty: bool = True


class TimePreviewBody(BaseModel):
    iana_zone: str
    utc: datetime


class ObjectBody(BaseModel):
    box: list[float] | None = None
    cls: str | None = None
    action: str | None = None   # remove | restore | reclass | move
    note: str | None = None


class CandidateBody(BaseModel):
    action: str                 # confirm | reject | restore | persistent
    box: list[float] | None = None
    cls: str = "car"
    note: str | None = None


class RegroupBody(BaseModel):
    light_ids: list[int]


class FilterBody(BaseModel):
    search: str | None = None
    view_id: int | None = None
    review_status: str | None = None
    processing_state: str | None = None
    local_date_from: str | None = None
    local_date_to: str | None = None
    min_vehicles: int | None = None


class CommitBody(BaseModel):
    token: str


class ViewBody(BaseModel):
    name: str | None = None
    absorb: int | None = None
    version_ids: list[int] = Field(default_factory=list, max_length=200)
    target_view: int | None = None


class RegionBody(BaseModel):
    kind: str
    polygon: list[list[float]]
    name: str | None = None
    region_uid: str | None = None


class RegionBatchBody(BaseModel):
    regions: list[RegionBody] = Field(min_length=1, max_length=60)


class ExportBody(BaseModel):
    formats: list[str] | None = None
    filter: FilterBody | None = None


class AskBody(BaseModel):
    text: str


class ConfirmBody(BaseModel):
    spec: dict | None = None
    ignore_unknown: bool = False


class HitDecisionBody(BaseModel):
    decision: str
    note: str | None = None


class ColourBody(BaseModel):
    value: str | None = None
    note: str | None = None


class ReviseBody(BaseModel):
    action: str
    note: str | None = None
    of: str | None = None
    first_t: float | None = None
    last_t: float | None = None
    cls: str | None = None
    target: int | None = None
    t: float | None = None


class PlateReviseBody(BaseModel):
    action: str
    text: str | None = None
    box: list[float] | None = None
    note: str | None = None


class EnableBody(BaseModel):
    enabled: bool
    note: str | None = None


class ClipBody(BaseModel):
    before_s: float | None = None
    after_s: float | None = None
    method: str = "copy"


class VideoStartBody(BaseModel):
    action: str = "set"             # set | clear
    utc: datetime | None = None
    precision: str | None = None     # minute | second
    note: str | None = None


class MonitoringBody(BaseModel):
    enabled: bool


class PreviewBody(BaseModel):
    source_dir: str
    include_subdirs: bool = True
    limit: int = 20


class BandsBody(BaseModel):
    far_below: float
    close_from: float


class DeleteBody(BaseModel):
    confirm_name: str


class DatasetBody(BaseModel):
    name: str
    ratios: list[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    block_days: int = 1
    seed: int = 0


class EvaluateBody(BaseModel):
    dataset_id: int
    split: str = "test"
    name: str | None = None           # model pack; empty means the model in production use
    version: str | None = None
    iou: float = Field(0.5, ge=0.1, le=0.95)


class TrainBody(BaseModel):
    dataset_id: int
    epochs: int = Field(20, ge=1, le=300)
    start_from: str | None = None     # "coco" (the released YOLOX-s) or "<name>/<version>" of a trained model


class PromoteBody(BaseModel):
    task: str = "vehicle_detection"
    name: str
    version: str
    mission_id: str
    evaluation_id: int
    note: str | None = None


class RollbackBody(BaseModel):
    task: str = "vehicle_detection"
    note: str | None = None


def default_detector_factory(data_root: Path | str | None = None):
    """The vehicle detector in production use: the last promoted pack, else the built-in default."""
    from .. import models
    from ..detect import YoloxDetector
    pack = None
    if data_root is not None:
        from ..learning import ModelRegistry
        prod = ModelRegistry(data_root).production("vehicle_detection")
        if prod:
            pack = models.find(prod["name"], prod["version"])
    pack = pack or models.find("yolox-s")
    return YoloxDetector(pack), pack.card


def create_app(data_root: Path | str, detector_factory: Callable | None = None,
               monitor_interval: float = 60, stability_seconds: int = 30, start_monitoring: bool = True,
               plate_factory: Callable | None = None, model_loader: Callable | None = None,
               backup_dir: Path | str | None = None, backup_at: str = "02:00",
               sam_factory: Callable | None = None, sam_status: Callable[[], dict] | None = None) -> FastAPI:
    data_root = Path(data_root)
    DataRoot(data_root)  # create folders and control.db
    jobs = JobManager(data_root, detector_factory or (lambda: default_detector_factory(data_root)), monitor_interval,
                      stability_seconds, plate_factory, model_loader, sam_factory)

    def sam_info() -> dict:
        if sam_status:
            return sam_status()
        from ..sam3 import available
        return available()
    previews: dict[str, tuple[float, dict]] = {}
    from ..admin import BackupScheduler
    backups = BackupScheduler(data_root, backup_dir, backup_at) if backup_dir else None

    @asynccontextmanager
    async def lifespan(_app):
        if start_monitoring:
            jobs.resume_monitoring()  # monitoring survives restarts (MIS-07)
        if backups:
            backups.start()
        yield
        if backups:
            backups.stop()
        jobs.shutdown()

    app = FastAPI(title="LanePulse", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.exception_handler(HTTPException)
    def _http_error(_req, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) else {"error": "error", "message": str(exc.detail)}
        return JSONResponse(body, status_code=exc.status_code)

    @app.middleware("http")
    async def _csrf(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method in ("POST", "PUT", "PATCH", "DELETE") \
                and request.headers.get("x-twb") != "1":
            return JSONResponse({"error": "missing_header", "message": "Requests must come from the application."},
                                status_code=403)
        return await call_next(request)

    # ---------- dependencies ----------
    def control_con():
        con = connect(data_root / "control.db")
        try:
            yield con
        finally:
            con.close()

    def user() -> User:
        # LanePulse is a local workstation app. Edits remain attributed to its OS user,
        # without an additional account, password, session timeout, or sign-in screen.
        import getpass
        name = getpass.getuser() or "local"
        return User(0, name, name, "admin")

    def mission(mission_id: str, _u: User = Depends(user)) -> Mission:
        try:
            m = DataRoot(data_root).open_mission(mission_id)
        except KeyError:
            raise ApiError(404, "no_mission", "That mission does not exist.") from None
        return m

    def flt(b: FilterBody | None) -> Filter:
        return Filter(**b.model_dump()) if b else Filter()

    # A neutral health probe keeps installers and supervisors independent of any user or session API.
    @app.get("/api/health")
    def health():
        import sys
        # The installed build this server runs from (its environment folder), so a launcher can tell an old one.
        return {"status": "ok", "version": __version__, "install": Path(sys.prefix).name}

    @app.post("/api/time-preview")
    def time_preview(body: TimePreviewBody):
        if body.utc.tzinfo is None or body.utc.utcoffset() is None:
            raise ApiError(400, "naive_time", "Enter a UTC date and time with a Z suffix.")
        try:
            local = to_local(body.utc.astimezone(timezone.utc), body.iana_zone)
        except (ValueError, zoneinfo.ZoneInfoNotFoundError):
            raise ApiError(400, "bad_zone", f"'{body.iana_zone}' is not a known IANA time zone.") from None
        offset_minutes = int(local.utc_offset.total_seconds() // 60)
        sign = "+" if offset_minutes >= 0 else "-"
        hours, minutes = divmod(abs(offset_minutes), 60)
        return {"utc": local.utc.isoformat().replace("+00:00", "Z"), "local": local.local.isoformat(),
                "zone": local.zone, "offset": f"UTC{sign}{hours:02d}:{minutes:02d}", "is_dst": local.is_dst,
                "abbreviation": local.abbreviation}

    # ---------- reference data ----------
    @app.get("/api/reference")
    def reference(_u: User = Depends(user)):
        return {"countries": TARGET_COUNTRIES, "country_zones": COUNTRY_ZONES, "modes": {k: sorted(v) for k, v in MODES.items()},
                "zones": sorted(zoneinfo.available_timezones()), "classes": list(VEHICLE_CLASSES), "version": __version__,
                "sam3": sam_info()}

    @app.get("/api/sam3")
    def sam3_status(_u: User = Depends(user)):
        from ..sam3 import AREA_SUGGESTIONS, VEHICLE_PROMPTS
        return {**sam_info(), "suggestions": AREA_SUGGESTIONS, "vehicle_prompts": VEHICLE_PROMPTS}

    @app.post("/api/source-preview")
    def source_preview(body: PreviewBody, _u: User = Depends(user)):
        """First files in a source folder with their parsed capture time, for the wizard."""
        root = Path(body.source_dir)
        if not root.is_dir():
            raise ApiError(400, "no_folder", f"{root} is not a folder on this workstation.")
        it = root.rglob("*") if body.include_subdirs else root.glob("*")
        out = []
        for p in it:
            if not p.is_file():
                continue
            row: dict[str, Any] = {"file": str(p.relative_to(root))}
            try:
                ct = parse_capture_time(p.name)
                row.update(ok=True, utc=ct.utc.isoformat().replace("+00:00", "Z"), precision=ct.precision.value,
                           pattern=ct.pattern, fields=ct.fields)
            except MalformedName as exc:
                row.update(ok=False, reason=exc.reason)
            out.append(row)
            if len(out) >= body.limit:
                break
        return {"files": out}

    # ---------- missions ----------
    @app.get("/api/missions")
    def missions(_u: User = Depends(user)):
        root = DataRoot(data_root)
        out = []
        for r in root.missions():
            m = root.open_mission(r["id"])
            cfg = m.config()
            out.append({"id": r["id"], "name": r["name"], "input_type": r["input_type"], "mode": r["mode"],
                        "processing": cfg.processing, "monitoring": cfg.monitoring_enabled, "created_at": r["created_at"],
                        "status": m.status(), "job": jobs.current(r["id"]).snapshot() if jobs.current(r["id"]) else None})
        return {"missions": out}

    @app.post("/api/missions/validate")
    def mission_validate(body: MissionBody, _u: User = Depends(user)):
        v = validate(MissionConfig(**body.model_dump()), data_root)
        return {"errors": v.errors, "warnings": v.warnings}

    @app.post("/api/missions")
    def mission_create(body: MissionBody, u: User = Depends(user)):
        if body.detector in ("sam3", "both") and not sam_info()["installed"]:
            raise ApiError(400, "sam3_missing", "SAM 3.1 is not installed on this computer: " + (sam_info()["reason"] or ""),
                           "Choose the standard detector, or install the full LanePulse package.")
        try:
            m = DataRoot(data_root).create_mission(MissionConfig(**body.model_dump()), u.username)
        except MissionValidationError as exc:
            raise ApiError(400, "invalid", " ".join(exc.validation.errors)) from None
        return {"id": m.id}

    @app.get("/api/missions/{mission_id}")
    def mission_get(m: Mission = Depends(mission)):
        cfg = m.config()
        job = jobs.current(m.id)
        counts = dict(m.con.execute(
            "SELECT processing_state, COUNT(*) FROM run_item i WHERE i.id IN (SELECT MAX(id) FROM run_item GROUP BY "
            "source_version_id) GROUP BY processing_state").fetchall())
        runs = [dict(r) for r in m.con.execute("SELECT id, kind, state, created_at, started_at, finished_at FROM run "
                                               "ORDER BY id DESC LIMIT 10")]
        notes = [dict(r) for r in m.con.execute("SELECT id, created_at, severity, code, message, acknowledged FROM "
                                                "notification ORDER BY id DESC LIMIT 30")]
        return {"id": m.id, "config": asdict(cfg), "status": m.status(), "processing_counts": counts, "runs": runs,
                "job": job.snapshot() if job else None, "notifications": notes, "monitor_running": m.id in jobs.monitors}

    @app.post("/api/missions/{mission_id}/job")
    def job_start(m: Mission = Depends(mission)):
        return jobs.start(m.id).snapshot()

    @app.post("/api/missions/{mission_id}/job/{action}")
    def job_control(action: str, m: Mission = Depends(mission)):
        job = jobs.current(m.id)
        if job is None or job.state not in ("running", "paused"):
            raise ApiError(409, "no_job", "Nothing is running for this mission.")
        if action == "pause":
            job.control.pause()
        elif action == "resume":
            job.control.resume()
        elif action == "cancel":
            job.control.cancel()
        else:
            raise ApiError(400, "bad_action", "Use pause, resume or cancel.")
        return job.snapshot()

    @app.put("/api/missions/{mission_id}/monitoring")
    def monitoring(body: MonitoringBody, m: Mission = Depends(mission), u: User = Depends(user)):
        cfg = m.config()
        if cfg.processing != "continuous":
            raise ApiError(400, "one_time", "Monitoring applies to continuous missions only.")
        m.update_config(u.username, "monitoring " + ("on" if body.enabled else "off"), monitoring_enabled=body.enabled)
        jobs.set_monitoring(m.id, body.enabled)
        return {"enabled": body.enabled}

    @app.put("/api/missions/{mission_id}/sam3-words")
    def sam3_words_set(body: WordsBody, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..sam3 import VEHICLE_PROMPTS, vehicle_prompts
        try:
            prompts = vehicle_prompts(body.words)
        except ValueError as exc:
            raise ApiError(400, "invalid", str(exc)) from None
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Processing is running. Change the words when it has finished or is cancelled.")
        words = None if prompts == dict(VEHICLE_PROMPTS) else {k: list(v) for k, v in prompts.items()}
        if words != getattr(m.config(), "sam3_words", None):
            m.update_config(u.username, "words for SAM 3.1 changed" if words else "words for SAM 3.1 back to the defaults",
                            sam3_words=words)
        return {"words": {k: list(v) for k, v in prompts.items()}, "default": words is None}

    @app.put("/api/missions/{mission_id}/detector")
    def detector_set(body: DetectorBody, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..store import DETECTORS
        cfg = m.config()
        if body.detector not in DETECTORS:
            raise ApiError(400, "invalid", f"Detector must be one of {sorted(DETECTORS)}.")
        if body.detector in ("sam3", "both") and cfg.input_type != "images":
            raise ApiError(400, "invalid", "SAM 3.1 detection is for still images; video missions use the standard detector.")
        if body.detector in ("sam3", "both") and not sam_info()["installed"]:
            raise ApiError(400, "sam3_missing", "SAM 3.1 is not installed on this computer: " + (sam_info()["reason"] or ""))
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Processing is running. Change the detector when it has finished or is cancelled.")
        if body.detector != getattr(cfg, "detector", "standard"):
            m.update_config(u.username, f"detector set to {body.detector}", detector=body.detector)
        return {"detector": body.detector}

    @app.post("/api/missions/{mission_id}/notifications/ack")
    def ack(m: Mission = Depends(mission)):
        m.con.execute("UPDATE notification SET acknowledged=1 WHERE acknowledged=0")
        return {"ok": True}

    # ---------- results ----------
    @app.post("/api/missions/{mission_id}/results")
    def results(body: FilterBody, m: Mission = Depends(mission), sort: str = "capture_utc", desc: bool = False,
                offset: int = 0, limit: int = Query(50, le=500)):
        rs = ReviewService(m)
        try:
            rows = rs.results(flt(body), sort, desc)
        except ValueError as exc:
            raise ApiError(400, "bad_query", str(exc)) from None
        return {"total": len(rows), "rows": [r.__dict__ for r in rows[offset: offset + limit]]}

    def _vid(m: Mission, version_id: int) -> int:
        if m.con.execute("SELECT 1 FROM source_version WHERE id=?", (version_id,)).fetchone() is None:
            raise ApiError(404, "no_image", "That image is not part of this mission.")
        return version_id

    @app.get("/api/missions/{mission_id}/images/{version_id}")
    def image(version_id: int, m: Mission = Depends(mission)):
        path = ViewService(m)._path(_vid(m, version_id))
        if not path.is_file():
            raise ApiError(410, "source_missing", f"{path.name} is no longer in the source folder.")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/missions/{mission_id}/thumbs/{version_id}")
    def thumb(version_id: int, m: Mission = Depends(mission), w: int = Query(320, ge=64, le=800)):
        sha = m.con.execute("SELECT sha256 FROM source_version WHERE id=?", (_vid(m, version_id),)).fetchone()[0]
        cache = m.dir / "evidence" / "thumbs" / f"{sha}_{w}.jpg"
        if not cache.is_file():
            img = read_image(ViewService(m)._path(version_id))
            h = round(img.shape[0] * w / img.shape[1])
            cache.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
            tmp = cache.with_suffix(".tmp")
            tmp.write_bytes(buf.tobytes())
            os.replace(tmp, cache)
        return FileResponse(cache, headers={"Cache-Control": "private, max-age=86400"})

    def _item(m: Mission, version_id: int, body: FilterBody | None, sort: str, desc: bool) -> dict:
        from ..night_service import NightService
        rs = ReviewService(m)
        vs = ViewService(m)
        r = m.con.execute("SELECT v.*, f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                          "WHERE v.id=?", (version_id,)).fetchone()
        sig = m.con.execute("SELECT width, height FROM view_signature WHERE source_version_id=?", (version_id,)).fetchone()
        cond = m.con.execute("SELECT * FROM image_condition WHERE source_version_id=?", (version_id,)).fetchone()
        view_id = vs.current_view(version_id)
        local = None
        if r["capture_utc"]:
            from datetime import datetime, timezone
            lt = to_local(datetime.fromisoformat(r["capture_utc"].replace("Z", "+00:00")).astimezone(timezone.utc), m.zone)
            local = {"time": lt.local.isoformat(), "abbreviation": lt.abbreviation, "dst": lt.is_dst}
        regions = RegionSet.load(m.con, view_id) if view_id else RegionSet([])
        item = rs._latest_item(version_id)
        return {
            "id": version_id, "file": r["rel_path"], "sha256": r["sha256"], "capture_utc": r["capture_utc"],
            "capture_precision": r["capture_precision"], "local": local, "zone": m.zone,
            "width": sig["width"] if sig else None, "height": sig["height"] if sig else None,
            "view": {"id": view_id, "name": m.con.execute("SELECT name FROM camera_view WHERE id=?", (view_id,)).fetchone()[0]
                     if view_id else None, "regions_applied": vs.regions_applicable(view_id) if view_id else None},
            "view_assignment_history": vs.history(version_id),
            "condition": dict(cond) if cond else None,
            "processing_state": item["processing_state"] if item else "not_queued",
            "error": item["error_detail"] if item else None,
            "review_status": rs.review_status(version_id),
            "empty_background": ChangeService(m).is_marked(version_id),
            "objects": [{**o.__dict__, "counted": o.counted} for o in rs.objects(version_id)],
            "candidates": [c.__dict__ for c in rs.candidates(version_id)],
            "lights": [li.__dict__ for li in NightService(m).lights(version_id)],
            "bands": NightService(m).bands(view_id) if cond and cond["condition"] in ("night", "dusk") else None,
            "regions": [{"uid": x.uid, "kind": x.kind, "name": x.name, "polygon": x.polygon}
                        for x in regions.include + regions.exclude],
            "position": rs.position(version_id, flt(body), sort, desc),
        }

    @app.post("/api/missions/{mission_id}/items/{version_id}")
    def item_get(version_id: int, body: FilterBody | None = None, sort: str = "capture_utc", desc: bool = False,
                 m: Mission = Depends(mission)):
        return _item(m, _vid(m, version_id), body, sort, desc)

    def _edit(fn):
        try:
            fn()
        except (ValueError, KeyError) as exc:
            raise ApiError(400, "invalid_edit", str(exc).strip("'")) from None

    @app.post("/api/missions/{mission_id}/items/{version_id}/objects")
    def object_add(version_id: int, body: ObjectBody, m: Mission = Depends(mission), u: User = Depends(user)):
        rs = ReviewService(m)
        out = {}
        _edit(lambda: out.update(uid=rs.add(_vid(m, version_id), body.box, body.cls or "car", u.username, body.note)))
        return out

    @app.patch("/api/missions/{mission_id}/items/{version_id}/objects/{uid}")
    def object_edit(version_id: int, uid: str, body: ObjectBody, m: Mission = Depends(mission), u: User = Depends(user)):
        rs = ReviewService(m)
        v = _vid(m, version_id)
        acts = {"remove": lambda: rs.remove(v, uid, u.username, body.note),
                "restore": lambda: rs.restore(v, uid, u.username),
                "reclass": lambda: rs.reclass(v, uid, body.cls or "", u.username, body.note),
                "move": lambda: rs.move(v, uid, body.box, u.username)}
        if body.action not in acts:
            raise ApiError(400, "bad_action", "Use remove, restore, reclass or move.")
        _edit(acts[body.action])
        return {"ok": True}

    @app.post("/api/missions/{mission_id}/items/{version_id}/candidates/{uid}")
    def candidate_edit(version_id: int, uid: str, body: CandidateBody, m: Mission = Depends(mission), u: User = Depends(user)):
        rs = ReviewService(m)
        v = _vid(m, version_id)
        acts = {"confirm": lambda: rs.confirm_candidate(v, uid, u.username, body.box, body.cls, body.note),
                "reject": lambda: rs.reject_candidate(v, uid, u.username, body.note),
                "restore": lambda: rs.restore_candidate(v, uid, u.username),
                "persistent": lambda: rs.mark_persistent_light(v, uid, u.username)}
        if body.action not in acts:
            raise ApiError(400, "bad_action", "Use confirm, reject, restore or persistent.")
        _edit(acts[body.action])
        return {"ok": True}

    @app.post("/api/missions/{mission_id}/items/{version_id}/regroup")
    def regroup(version_id: int, body: RegroupBody, m: Mission = Depends(mission), u: User = Depends(user)):
        out = {}
        _edit(lambda: out.update(uid=ReviewService(m).regroup(_vid(m, version_id), body.light_ids, u.username)))
        return out

    @app.post("/api/missions/{mission_id}/items/{version_id}/decision/{decision}")
    def decide(version_id: int, decision: str, m: Mission = Depends(mission), u: User = Depends(user)):
        rs = ReviewService(m)
        v = _vid(m, version_id)
        if decision == "approve":
            _edit(lambda: rs.approve(v, u.username))
        elif decision == "reopen":
            rs.reopen(v, u.username)
        else:
            raise ApiError(404, "unknown", "Unknown decision.")
        return {"review_status": rs.review_status(v)}

    @app.post("/api/missions/{mission_id}/approve-all/preview")
    def approve_all_preview(body: FilterBody, m: Mission = Depends(mission)):
        p = ReviewService(m).approve_all_preview(flt(body))
        token = secrets.token_urlsafe(16)
        now = time.time()
        for k in [k for k, (t, _) in previews.items() if now - t > 600]:
            previews.pop(k, None)
        previews[token] = (now, {**p, "mission_id": m.id})
        return {"token": token, "count": p["count"], "scope": p["scope"], "excluded_not_processed": p["excluded_not_processed"]}

    @app.post("/api/missions/{mission_id}/approve-all/commit")
    def approve_all_commit(body: CommitBody, m: Mission = Depends(mission), u: User = Depends(user)):
        entry = previews.pop(body.token, None)
        if entry is None or entry[1]["mission_id"] != m.id or time.time() - entry[0] > 600:
            raise ApiError(409, "preview_expired", "The approval preview expired. Check the count again before approving.")
        bid = ReviewService(m).approve_all_commit(entry[1], u.username)
        return {"bulk_approval_id": bid}

    # ---------- views and regions ----------
    @app.get("/api/missions/{mission_id}/views")
    def views(m: Mission = Depends(mission)):
        vs = ViewService(m)
        from ..night_service import NightService
        ns = NightService(m)
        return {"views": [v | {"night_bands": ns.bands(v["id"])} for v in vs.views()], "unassigned": vs.unassigned()}

    @app.post("/api/missions/{mission_id}/views/{view_id}/actions/{action}")
    def view_action(view_id: int, action: str, body: ViewBody, m: Mission = Depends(mission), u: User = Depends(user)):
        vs = ViewService(m)
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=? AND merged_into IS NULL", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        if action == "rename" and body.name:
            vs.rename(view_id, body.name, u.username)
        elif action == "confirm":
            vs.confirm(view_id, u.username)
        elif action == "merge" and body.absorb:
            vs.merge(view_id, body.absorb, u.username)
        elif action == "split":
            if not body.version_ids:
                raise ApiError(400, "empty_selection", "Select at least one image to split into a new view.")
            current = set(vs.members(view_id))
            if len(set(body.version_ids)) != len(body.version_ids) or not set(body.version_ids) <= current:
                raise ApiError(400, "invalid_selection", "Every selected image must currently belong to this view.")
            out = {}
            _edit(lambda: out.update(view_id=vs.split(view_id, body.version_ids, u.username, body.name)))
            return out
        elif action == "reassign":
            if not body.version_ids:
                raise ApiError(400, "empty_selection", "Select at least one image to reassign.")
            if len(set(body.version_ids)) != len(body.version_ids):
                raise ApiError(400, "invalid_selection", "The selection contains duplicate images.")
            if body.target_view is not None and m.con.execute(
                    "SELECT 1 FROM camera_view WHERE id=? AND merged_into IS NULL", (body.target_view,)).fetchone() is None:
                raise ApiError(404, "no_view", "The destination camera view does not exist.")
            marks = ",".join("?" for _ in body.version_ids)
            known = {r[0] for r in m.con.execute(f"SELECT id FROM source_version WHERE id IN ({marks})", body.version_ids)}
            if known != set(body.version_ids):
                raise ApiError(404, "no_image", "One or more selected images no longer exist.")
            _edit(lambda: vs.reassign(body.version_ids, body.target_view, u.username))
        else:
            raise ApiError(400, "bad_action", "Use rename, confirm, merge, split, or reassign.")
        return {"ok": True}

    @app.get("/api/missions/{mission_id}/views/unassigned/members")
    def unassigned_view_members(m: Mission = Depends(mission), offset: int = Query(0, ge=0),
                                limit: int = Query(60, ge=1, le=200)):
        ids = ViewService(m).unassigned()
        return {"total": len(ids), "offset": offset, "limit": limit, "ids": ids[offset:offset + limit]}

    @app.get("/api/missions/{mission_id}/views/{view_id}/members")
    def view_members(view_id: int, m: Mission = Depends(mission), offset: int = Query(0, ge=0),
                     limit: int = Query(60, ge=1, le=200)):
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=? AND merged_into IS NULL", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        ids = ViewService(m).members(view_id)
        return {"total": len(ids), "offset": offset, "limit": limit, "ids": ids[offset:offset + limit]}

    @app.get("/api/missions/{mission_id}/views/{view_id}/bands")
    def bands_get(view_id: int, m: Mission = Depends(mission)):
        from ..night_service import NightService
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=?", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        return NightService(m).bands(view_id)

    @app.put("/api/missions/{mission_id}/views/{view_id}/bands")
    def bands_put(view_id: int, body: BandsBody, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..night_service import NightService
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=?", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        ns = NightService(m)
        out = {}
        _edit(lambda: out.update(ns.set_bands(view_id, body.far_below, body.close_from, u.username)))
        out["reanalysed"] = ns.refresh(with_decisions=True)["images_analysed"]   # night candidates follow at once
        return out

    @app.post("/api/missions/{mission_id}/night/apply-decisions")
    def night_apply_decisions(m: Mission = Depends(mission), u: User = Depends(user)):
        """Re-check night images so operators' decisions at a place apply to the view's other images."""
        from ..night_service import NightService
        return {"reanalysed": NightService(m).refresh(with_decisions=True)["images_analysed"]}

    @app.get("/api/missions/{mission_id}/night/queue")
    def night_queue(m: Mission = Depends(mission), view: int | None = None, band: str | None = None,
                    flag: str | None = None, status: str = "open", offset: int = Query(0, ge=0),
                    limit: int = Query(60, ge=1, le=300)):
        """Night candidates from every night or dusk image, most uncertain first, for fast review."""
        from ..night_review import NightReview
        try:
            return NightReview(m).queue(view, band or None, flag or None, status, offset, limit)
        except ValueError as exc:
            raise ApiError(400, "bad_filter", str(exc)) from None

    @app.get("/api/missions/{mission_id}/night/accuracy")
    def night_accuracy(m: Mission = Depends(mission), view: int | None = None):
        """Share of answered night candidates that were vehicles, overall and by band, kind, score and flag."""
        from ..night_review import NightReview
        return NightReview(m).accuracy(view)

    @app.get("/api/missions/{mission_id}/views/{view_id}/regions")
    def regions_get(view_id: int, m: Mission = Depends(mission)):
        r = m.con.execute("SELECT ref_width, ref_height FROM camera_view WHERE id=?", (view_id,)).fetchone()
        if r is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        rs = RegionSet.load(m.con, view_id)
        rep = ViewService(m).representatives(view_id, 1)
        return {"width": r["ref_width"], "height": r["ref_height"], "representatives": rep,
                "regions": [{"uid": x.uid, "kind": x.kind, "name": x.name, "polygon": x.polygon}
                            for x in rs.include + rs.exclude]}

    @app.post("/api/missions/{mission_id}/views/{view_id}/regions")
    def regions_save(view_id: int, body: RegionBody, m: Mission = Depends(mission), u: User = Depends(user)):
        out = {}
        _edit(lambda: out.update(uid=RegionEditor(m.con).save(view_id, body.kind, [tuple(p) for p in body.polygon],
                                                              u.username, body.name, body.region_uid)))
        from ..night_service import NightService
        NightService(m).refresh()  # night candidates follow the new regions at once
        return out

    @app.post("/api/missions/{mission_id}/views/{view_id}/regions/batch")
    def regions_save_batch(view_id: int, body: RegionBatchBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Several regions at once (areas found by SAM 3.1): all are saved, or none if one is not valid."""
        uids = []
        ed = RegionEditor(m.con)
        m.con.execute("BEGIN")
        try:
            for r in body.regions:
                uids.append(ed.save(view_id, r.kind, [tuple(p) for p in r.polygon], u.username, r.name, r.region_uid))
        except (ValueError, KeyError) as exc:
            m.con.execute("ROLLBACK")
            raise ApiError(400, "invalid_edit", str(exc).strip("'")) from None
        m.con.execute("COMMIT")
        from ..night_service import NightService
        NightService(m).refresh()
        return {"uids": uids}

    @app.post("/api/missions/{mission_id}/views/{view_id}/find-areas")
    def find_areas_start(view_id: int, body: FindAreasBody, m: Mission = Depends(mission), _u: User = Depends(user)):
        if body.kind not in ("include", "exclude"):
            raise ApiError(400, "invalid", "kind must be include or exclude.")
        prompts = [p.strip()[:60] for p in body.prompts if p.strip()]
        if not prompts:
            raise ApiError(400, "invalid", "Name at least one thing to look for, such as road.")
        info = sam_info()
        if not info["installed"]:
            raise ApiError(400, "sam3_missing", "SAM 3.1 is not installed on this computer: " + (info["reason"] or ""),
                           "Draw the regions by hand, or install the full LanePulse package.")
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=?", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        _vid(m, body.image_id)
        return jobs.find_areas(m.id, view_id, body.image_id, prompts, body.kind, body.threshold).snapshot()

    # ---------- empty backgrounds: vehicles found as changes ----------
    def _view(m: Mission, view_id: int) -> int:
        if m.con.execute("SELECT 1 FROM camera_view WHERE id=? AND merged_into IS NULL", (view_id,)).fetchone() is None:
            raise ApiError(404, "no_view", "That camera view does not exist.")
        return view_id

    @app.get("/api/missions/{mission_id}/views/{view_id}/backgrounds")
    def backgrounds(view_id: int, m: Mission = Depends(mission)):
        cs = ChangeService(m)
        out = cs.references(_view(m, view_id))
        out["suggestions"] = cs.suggest_empty(view_id)
        out["busy"] = jobs.busy(m.id)
        return out

    @app.put("/api/missions/{mission_id}/items/{version_id}/empty")
    def mark_empty(version_id: int, body: EmptyBody, m: Mission = Depends(mission), u: User = Depends(user)):
        out = {}
        _edit(lambda: out.update(ChangeService(m).mark(_vid(m, version_id), u.username, body.empty)))
        return out

    @app.get("/api/missions/{mission_id}/views/{view_id}/backgrounds/{condition}.jpg")
    def background_image(view_id: int, condition: str, m: Mission = Depends(mission)):
        if condition not in CONDITIONS:
            raise ApiError(404, "no_background", "Use day, dusk or night.")
        cs = ChangeService(m)
        bg = cs.current(_view(m, view_id), condition)
        if bg is None:
            raise ApiError(404, "no_background", f"This view has no empty {condition} background yet.")
        med, _ = cs.arrays(bg)
        ok, buf = cv2.imencode(".jpg", med.astype("uint8"), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/missions/{mission_id}/backgrounds/compare")
    def backgrounds_compare(m: Mission = Depends(mission), _u: User = Depends(user)):
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "The mission is processing. The comparison runs at the end of it anyway.")
        if not m.con.execute("SELECT 1 FROM reference_mark LIMIT 1").fetchone():
            raise ApiError(400, "no_empty_images", "Mark a few empty-road images of a camera view first.")
        return jobs.start(m.id, "compare_backgrounds").snapshot()

    @app.get("/api/missions/{mission_id}/find-areas/{task_id}")
    def find_areas_get(task_id: str, m: Mission = Depends(mission)):
        t = jobs.area_tasks.get(task_id)
        if t is None or t.mission_id != m.id:
            raise ApiError(404, "no_task", "That search has ended; start it again.")
        return t.snapshot()

    @app.delete("/api/missions/{mission_id}/views/{view_id}/regions/{uid}")
    def regions_delete(view_id: int, uid: str, m: Mission = Depends(mission), u: User = Depends(user)):
        RegionEditor(m.con).delete(view_id, uid, u.username)
        from ..night_service import NightService
        NightService(m).refresh()
        return {"ok": True}

    # ---------- export ----------
    @app.post("/api/missions/{mission_id}/export")
    def export_(body: ExportBody, m: Mission = Depends(mission), u: User = Depends(user)):
        try:
            paths = run_export(m, u.username, body.formats, flt(body.filter))
        except Exception as exc:
            raise ApiError(500, "export_failed", f"Export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None
        return {"files": [str(p) for p in paths]}

    # ---------- object search ----------
    def _search(m: Mission):
        from ..search import SearchService
        return SearchService(m)

    def _search_call(fn):
        from ..search import SearchError
        try:
            return fn()
        except SearchError as exc:
            raise ApiError(400, "search", str(exc)) from None

    @app.get("/api/missions/{mission_id}/search")
    def search_home(m: Mission = Depends(mission)):
        from ..interpret import CATEGORIES, COLOURS, CONDITIONS, SUBTYPES, WEEKDAYS
        svc = _search(m)
        return {"requests": svc.requests(), "region_names": list(svc.context().region_names),
                "vocabulary": {"categories": CATEGORIES, "subtypes": list(SUBTYPES), "colours": COLOURS,
                               "conditions": CONDITIONS, "weekdays": WEEKDAYS},
                "mode": m.con.execute("SELECT mode FROM mission").fetchone()[0]}

    @app.post("/api/missions/{mission_id}/search/ask")
    def search_ask(body: AskBody, m: Mission = Depends(mission), u: User = Depends(user)):
        return _search_call(lambda: _search(m).ask(body.text, u.username))

    @app.post("/api/missions/{mission_id}/search/{request_id}/confirm")
    def search_confirm(request_id: int, body: ConfirmBody, m: Mission = Depends(mission), u: User = Depends(user)):
        return _search_call(lambda: _search(m).confirm(request_id, u.username, body.spec, body.ignore_unknown))

    @app.post("/api/missions/{mission_id}/search/{request_id}/run")
    def search_run(request_id: int, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _search(m)
        req = _search_call(lambda: svc.request(request_id))
        if not req["confirmed"]:
            raise ApiError(400, "search", "Confirm the search before running it.")
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Another job is running for this mission. Wait for it or cancel it.")
        return jobs.start(m.id, "search", request_id=request_id, author=u.username).snapshot()

    @app.get("/api/missions/{mission_id}/search-runs/{run_id}")
    def search_run_get(run_id: int, m: Mission = Depends(mission), status: str | None = None,
                       decision: str | None = None, offset: int = 0, limit: int = Query(100, ge=1, le=500)):
        svc = _search(m)
        info = _search_call(lambda: svc.run_info(run_id))
        hits, total = svc.hits(run_id, status, decision, offset, limit)
        return info | {"hits": [h.__dict__ for h in hits], "total": total}

    @app.post("/api/missions/{mission_id}/search-hits/{hit_id}/decision")
    def search_hit_decide(hit_id: int, body: HitDecisionBody, m: Mission = Depends(mission), u: User = Depends(user)):
        _search_call(lambda: _search(m).decide(hit_id, body.decision, u.username, body.note))
        return {"ok": True}

    @app.put("/api/missions/{mission_id}/items/{version_id}/objects/{uid}/colour")
    def object_colour(version_id: int, uid: str, body: ColourBody, m: Mission = Depends(mission), u: User = Depends(user)):
        _search_call(lambda: _search(m).set_colour(_vid(m, version_id), uid, body.value, u.username, body.note))
        return {"ok": True}

    @app.post("/api/missions/{mission_id}/search-runs/{run_id}/export")
    def search_export(run_id: int, body: ExportBody, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..search import SearchError, export_search
        try:
            paths = export_search(m, run_id, u.username, body.formats)
        except SearchError as exc:
            raise ApiError(400, "search", str(exc)) from None
        except Exception as exc:
            raise ApiError(500, "export_failed", f"Export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None
        return {"files": [str(p) for p in paths]}

    @app.get("/api/missions/{mission_id}/crops/{version_id}")
    def crop(version_id: int, box: str, m: Mission = Depends(mission), h: int = Query(110, ge=32, le=400),
             context: float = Query(0.0, ge=0.0, le=60.0), mark: bool = False):
        """context: extra margin around the box in pixels of the source image (at least this much on each side);
        mark draws the original box, so a small light stays findable in a wider view."""
        try:
            x1, y1, x2, y2 = (float(v) for v in box.split(","))
        except ValueError:
            raise ApiError(400, "bad_box", "box must be x1,y1,x2,y2") from None
        bx = (x1, y1, x2, y2)
        path = ViewService(m)._path(_vid(m, version_id))
        if not path.is_file():
            raise ApiError(410, "source_missing", f"{path.name} is no longer in the source folder.")
        img = read_image(path)
        H, W = img.shape[:2]
        pw, ph = max((x2 - x1) * 0.15 + 4, context), max((y2 - y1) * 0.15 + 4, context)
        a, b = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        c, d = min(W, int(x2 + pw)), min(H, int(y2 + ph))
        if c <= a or d <= b:
            raise ApiError(400, "bad_box", "box is outside the image")
        part = img[b:d, a:c]
        scale = h / part.shape[0]
        part = cv2.resize(part, (max(1, int(part.shape[1] * scale)), h),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        if mark:
            m_ = 3  # outline drawn just outside the box so the light itself stays visible
            p1 = (int((bx[0] - a) * scale) - m_, int((bx[1] - b) * scale) - m_)
            p2 = (int((bx[2] - a) * scale) + m_, int((bx[3] - b) * scale) + m_)
            cv2.rectangle(part, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    # ---------- plates ----------
    def _plates_svc(m: Mission):
        from ..plates import PlateService
        if m.config().mode != "plate_recognition":
            raise ApiError(400, "not_plates", "Plates are read only in plate recognition missions.")
        return PlateService(m)

    def _plate_call(fn):
        from ..plates import PlateError
        try:
            return fn()
        except PlateError as exc:
            raise ApiError(400, "plates", str(exc)) from None

    @app.get("/api/missions/{mission_id}/plates")
    def plates_list(m: Mission = Depends(mission), outcome: str | None = None, readability: str | None = None,
                    review: str | None = None, search: str | None = None, version_id: int | None = None,
                    offset: int = 0, limit: int = Query(100, ge=1, le=500)):
        from collections import Counter
        svc = _plates_svc(m)
        everything = svc.plates(version_id)
        shown = svc.plates(version_id, outcome or None, readability or None, review or None, search or None)
        return {"plates": [p.__dict__ for p in shown[offset: offset + limit]], "total": len(shown),
                "counts": {"outcome": Counter(p.outcome for p in everything),
                           "readability": Counter(p.readability or "none" for p in everything),
                           "review": Counter(p.review for p in everything)},
                "country": svc.country_status()}

    @app.get("/api/missions/{mission_id}/plates/{plate_id}/crop")
    def plate_crop(plate_id: int, m: Mission = Depends(mission), h: int = Query(64, ge=16, le=300)):
        svc = _plates_svc(m)
        row = m.con.execute(f"SELECT crop_path FROM {svc.table} WHERE id=?", (plate_id,)).fetchone()
        if row is None or not row[0] or not Path(row[0]).is_file():
            raise ApiError(404, "no_crop", "No plate crop for this vehicle.")
        img = cv2.imread(row[0])
        s = h / img.shape[0]
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_NEAREST if s > 3 else cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", img)
        return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/missions/{mission_id}/plates/{plate_id}/revise")
    def plate_revise(plate_id: int, body: PlateReviseBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _plates_svc(m)
        return _plate_call(lambda: svc.revise(plate_id, body.action, u.username, body.text, body.box, body.note)).__dict__

    @app.post("/api/missions/{mission_id}/plates/country/evaluate")
    def plate_evaluate(m: Mission = Depends(mission), u: User = Depends(user)):
        return _plates_svc(m).evaluate(u.username)

    @app.post("/api/missions/{mission_id}/plates/country/enable")
    def plate_enable(body: EnableBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _plates_svc(m)
        return _plate_call(lambda: svc.set_enabled(body.enabled, u.username, body.note))

    @app.post("/api/missions/{mission_id}/plates/export")
    def plates_export(body: ExportBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _plates_svc(m)
        try:
            paths = svc.export(u.username, body.formats)
        except Exception as exc:
            raise ApiError(500, "export_failed", f"Export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None
        return {"files": [str(p) for p in paths]}

    # ---------- system and administration ----------
    @app.get("/api/system")
    def system(u: User = Depends(user)):
        from .. import admin
        root = DataRoot(data_root)
        last = root.con.execute("SELECT at, author, detail FROM audit_log WHERE action IN ('backup','backup_failed') "
                                "ORDER BY id DESC LIMIT 1").fetchone()
        det = jobs._detector
        rt = det[0].describe().get("runtime") if det and hasattr(det[0], "describe") else None
        return {"version": __version__, "runtime": rt,
                "backup": {"folder": str(backup_dir) if backup_dir else None, "at": backup_at if backup_dir else None,
                           "last": dict(last) | {"detail": json.loads(last["detail"])} if last else None,
                           "last_error": backups.last_error if backups else None},
                "audit": admin.audit_log(root, 50) if u.role == "admin" else None}

    @app.post("/api/admin/backup")
    def backup_now(u: User = Depends(user)):
        from ..admin import AdminError, backup
        _admin_only(u)
        if not backup_dir:
            raise ApiError(400, "no_backup_folder", "No backup folder is set.",
                           "Start the server with --backup-dir pointing to another drive.")
        try:
            return {"folder": str(backup(data_root, backup_dir, u.username))}
        except (AdminError, OSError) as exc:
            raise ApiError(500, "backup_failed", f"Backup failed: {exc}",
                           "Check that the backup folder exists, is writable and has free space.") from None

    @app.post("/api/missions/{mission_id}/delete")
    def mission_delete(mission_id: str, body: DeleteBody, u: User = Depends(user)):
        from ..admin import AdminError, delete_mission
        _admin_only(u)
        if jobs.busy(mission_id):
            raise ApiError(409, "busy", "A job is running for this mission. Cancel it first.")
        jobs.set_monitoring(mission_id, False)
        jobs.jobs.pop(mission_id, None)
        try:
            return delete_mission(data_root, mission_id, u.username, body.confirm_name)
        except AdminError as exc:
            raise ApiError(400, "delete", str(exc)) from None

    def _admin_only(u: User):
        if u.role != "admin":
            raise ApiError(403, "admin_only", "Only an administrator can do this.")

    # ---------- learning ----------
    def _learn(m: Mission):
        from ..learning import LearningService
        if m.input_type != "images":
            raise ApiError(400, "not_images", "Training datasets are built from reviewed still images only, for now.")
        return LearningService(m)

    def _learn_call(fn):
        from ..learning import LearningError
        try:
            return fn()
        except LearningError as exc:
            raise ApiError(400, "learning", str(exc)) from None

    @app.get("/api/missions/{mission_id}/learning")
    def learning_summary(m: Mission = Depends(mission)):
        from .. import models
        from ..learning import ADAPTERS, ModelRegistry
        ls = _learn(m)
        reg = ModelRegistry(data_root)
        prod = reg.production("vehicle_detection")
        default = [x for x in models.installed("vehicle_detection") if x["name"] == "yolox-s"]
        in_use = ({"name": prod["name"], "version": prod["version"]} if prod else
                  ({"name": "yolox-s", "version": default[-1]["version"]} if default else None))
        from .. import training_env
        tj = jobs.training_busy()
        return {"in_use": in_use, "eligibility": ls.eligibility(), "datasets": ls.datasets(), "evaluations": ls.evaluations(),
                "adapters": ADAPTERS, "models": models.installed("vehicle_detection"),
                "production": prod, "history": reg.history("vehicle_detection")[:20],
                "training": {**training_env.status(), "busy_mission": tj.mission_id if tj else None,
                             "job": (lambda j: j.snapshot() if j else None)(jobs.training_job(m.id))}}

    @app.post("/api/missions/{mission_id}/learning/datasets")
    def dataset_create(body: DatasetBody, m: Mission = Depends(mission), u: User = Depends(user)):
        ls = _learn(m)
        return _learn_call(lambda: ls.create_dataset(body.name, u.username, tuple(body.ratios), body.block_days, body.seed))

    @app.get("/api/missions/{mission_id}/learning/datasets/{dataset_id}")
    def dataset_get(dataset_id: int, m: Mission = Depends(mission)):
        ls = _learn(m)
        return _learn_call(lambda: ls.dataset(dataset_id))

    @app.post("/api/missions/{mission_id}/learning/datasets/{dataset_id}/export")
    def dataset_export(dataset_id: int, m: Mission = Depends(mission)):
        ls = _learn(m)
        try:
            return {"folder": str(_learn_call(lambda: ls.export_dataset(dataset_id)))}
        except OSError as exc:
            raise ApiError(500, "export_failed", f"Export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None

    @app.post("/api/missions/{mission_id}/learning/evaluate")
    def evaluate_start(body: EvaluateBody, m: Mission = Depends(mission), u: User = Depends(user)):
        ls = _learn(m)
        _learn_call(lambda: ls.dataset(body.dataset_id))
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Another job is running for this mission. Wait for it or cancel it first.")
        if bool(body.name) != bool(body.version):
            raise ApiError(400, "model", "Give both the model name and version, or neither for the model in use.")
        return jobs.start(m.id, kind="evaluate", dataset_id=body.dataset_id, split=body.split, name=body.name,
                          version=body.version, iou=body.iou, author=u.username).snapshot()

    @app.post("/api/missions/{mission_id}/learning/train")
    def train_start(body: TrainBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Fine-tune the vehicle detector here, on this computer's GPU or CPU. The new model is added, not promoted."""
        from .. import models, training_env
        ls = _learn(m)
        ds = _learn_call(lambda: ls.dataset(body.dataset_id))
        if ds["leakage"]:
            raise ApiError(400, "learning", "This dataset version has leakage problems; it cannot be used for training.")
        if not ((ds.get("stats") or {}).get("splits") or {}).get("train", {}).get("images"):
            raise ApiError(400, "learning", "This dataset version has no training images.")
        env = training_env.status(wait=True)
        if env["state"] != "ready":
            raise ApiError(400, "no_training", env.get("reason") or "Training is not available here.")
        start = body.start_from or "coco"
        if start == "coco":
            if not env.get("coco_weights"):
                raise ApiError(400, "no_weights", "The YOLOX-s starting weights are not installed here. Install the current "
                               "LanePulse version, or start from a model trained earlier.")
            init = "coco"
        else:
            name, _, version = start.partition("/")
            try:
                pack = models.find(name, version or None)
            except models.ModelPackError as exc:
                raise ApiError(400, "model", str(exc)) from None
            if not (pack.card.get("torch_weights") and (pack.path / pack.card["torch_weights"]).is_file()):
                raise ApiError(400, "model", f"{name} {pack.version} cannot be trained further (no PyTorch weights). "
                               "Start from the released YOLOX-s instead.")
            init = str(pack.path)
        other = jobs.training_busy()
        if other:
            raise ApiError(409, "training_busy", "A model is already training on this computer"
                           + ("" if other.mission_id == m.id else " (for another mission)") + ". Wait for it to finish.")
        work = Path(m.config().output_dir) / ".training"
        try:
            work.mkdir(parents=True, exist_ok=True)
        except OSError:
            work = None
        return jobs.start(m.id, kind="train", dataset_id=body.dataset_id, epochs=body.epochs, init=init,
                          start_from=start, author=u.username, work_dir=str(work) if work else None).snapshot()

    @app.post("/api/missions/{mission_id}/learning/train/{action}")
    def train_control(action: str, m: Mission = Depends(mission)):
        job = jobs.training_job(m.id)
        if job is None or job.state not in ("running", "paused"):
            raise ApiError(409, "no_job", "Nothing is training for this mission.")
        if action == "pause":
            job.control.pause()
        elif action == "resume":
            job.control.resume()
        elif action == "cancel":
            job.control.cancel()
        else:
            raise ApiError(400, "bad_action", "Use pause, resume or cancel.")
        return job.snapshot()

    @app.get("/api/missions/{mission_id}/learning/evaluations/{evaluation_id}")
    def evaluation_get(evaluation_id: int, m: Mission = Depends(mission)):
        ls = _learn(m)
        return _learn_call(lambda: ls.evaluation(evaluation_id))

    def _admin(u: User):
        if u.role != "admin":
            raise ApiError(403, "admin_only", "Only an administrator can change the model in production use.")

    @app.post("/api/models/promote")
    def model_promote(body: PromoteBody, u: User = Depends(user)):
        from .. import models
        from ..learning import LearningService, ModelRegistry
        _admin(u)
        try:
            m = DataRoot(data_root).open_mission(body.mission_id)
        except KeyError:
            raise ApiError(404, "no_mission", "That mission does not exist.") from None
        ev = _learn_call(lambda: LearningService(m).evaluation(body.evaluation_id))
        mdl = ev["model"].get("model", {})
        if ev["state"] != "completed" or (mdl.get("name"), mdl.get("version")) != (body.name, body.version):
            raise ApiError(400, "evaluation", "That evaluation is not a completed evaluation of this model version.")
        try:
            prod = _learn_call(lambda: ModelRegistry(data_root).promote(body.task, body.name, body.version, u.username,
                                                                        body.mission_id, body.evaluation_id, body.note))
        except models.ModelPackError as exc:
            raise ApiError(400, "model", str(exc)) from None
        jobs.reset_detector()
        return {"production": prod}

    @app.post("/api/models/rollback")
    def model_rollback(body: RollbackBody, u: User = Depends(user)):
        from ..learning import ModelRegistry
        _admin(u)
        prod = _learn_call(lambda: ModelRegistry(data_root).rollback(body.task, u.username, body.note))
        jobs.reset_detector()
        return {"production": prod}

    # ---------- video ----------
    def _video(m: Mission):
        from ..video_review import VideoService
        if m.input_type != "video":
            raise ApiError(400, "not_video", "This mission analyses still images, not video.")
        return VideoService(m)

    def _video_call(fn):
        from ..video_review import VideoError
        try:
            return fn()
        except VideoError as exc:
            raise ApiError(400, "video", str(exc)) from None

    def _event_dict(e) -> dict:
        d = {k: v for k, v in e.__dict__.items() if not k.startswith("_")}
        d["parts"] = e.parts
        return d

    @app.get("/api/missions/{mission_id}/recordings")
    def recordings(m: Mission = Depends(mission)):
        return {"recordings": _video(m).recordings()}

    @app.put("/api/missions/{mission_id}/recordings/{version_id}/start-time")
    def recording_start_time(version_id: int, body: VideoStartBody, m: Mission = Depends(mission), u: User = Depends(user)):
        return _video_call(lambda: _video(m).revise_start(version_id, body.action, u.username, body.utc,
                                                           body.precision, body.note))

    @app.get("/api/missions/{mission_id}/videos/{version_id}")
    def video_file(version_id: int, m: Mission = Depends(mission)):
        path = _video(m).path(_vid(m, version_id))
        if not path.is_file():
            raise ApiError(410, "source_missing", f"{path.name} is no longer in the source folder.")
        return FileResponse(path, headers={"Cache-Control": "private, no-transform"})

    @app.get("/api/missions/{mission_id}/videos/{version_id}/frame")
    def video_frame(version_id: int, t: float = Query(..., ge=0), m: Mission = Depends(mission),
                    w: int = Query(1280, ge=64, le=3840)):
        """One decoded frame, for browsers that cannot play the recording's codec. X-Frame-Time is its exact time."""
        from ..video import frame_at
        path = _video(m).path(_vid(m, version_id))
        if not path.is_file():
            raise ApiError(410, "source_missing", f"{path.name} is no longer in the source folder.")
        f = frame_at(path, t)
        if f is None:
            raise ApiError(404, "no_frame", "No frame at or after that time.")
        img = f.image
        if img.shape[1] > w:
            img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={"X-Frame-Time": f"{f.t:.6f}", "Cache-Control": "private, max-age=3600"})

    @app.get("/api/missions/{mission_id}/events")
    def events(m: Mission = Depends(mission), version_id: int | None = None, status: str | None = None,
               flag: str | None = None):
        return {"events": [_event_dict(e) for e in _video(m).events(version_id, status, flag)]}

    @app.get("/api/missions/{mission_id}/events/{event_id}")
    def event_get(event_id: int, m: Mission = Depends(mission)):
        svc = _video(m)
        e = _video_call(lambda: svc.event(event_id))
        return _event_dict(e) | {"observations": svc.observations(event_id), "clips": svc.clips(event_id)}

    @app.get("/api/missions/{mission_id}/events/{event_id}/thumb")
    def event_thumb(event_id: int, m: Mission = Depends(mission), h: int = Query(120, ge=32, le=400)):
        svc = _video(m)
        data = _video_call(lambda: svc.thumbnail(event_id, h))
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/missions/{mission_id}/events/{event_id}/revise")
    def event_revise(event_id: int, body: ReviseBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _video(m)
        payload = {k: v for k, v in body.model_dump().items() if k not in ("action", "note") and v is not None}
        try:
            e = _video_call(lambda: svc.revise(event_id, body.action, u.username, body.note, **payload))
        except KeyError as exc:
            raise ApiError(400, "video", f"Missing field: {exc.args[0]}") from None
        return _event_dict(e)

    @app.post("/api/missions/{mission_id}/events/{event_id}/clip")
    def event_clip(event_id: int, body: ClipBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _video(m)
        try:
            return _video_call(lambda: svc.clip(event_id, body.before_s, body.after_s, body.method, u.username))
        except ApiError:
            raise
        except Exception as exc:
            raise ApiError(500, "clip_failed", f"Clip export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None

    @app.post("/api/missions/{mission_id}/events/export")
    def events_export(body: ExportBody, m: Mission = Depends(mission), u: User = Depends(user)):
        svc = _video(m)
        try:
            paths = svc.export(u.username, body.formats)
        except Exception as exc:
            raise ApiError(500, "export_failed", f"Export failed: {exc}",
                           "Check that the output folder exists, is writable and has free space.") from None
        return {"files": [str(p) for p in paths]}

    # ---------- UI ----------
    if (STATIC_DIR / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise ApiError(404, "not_found", "No such API endpoint.")
            return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        def no_ui():
            return HTMLResponse("<h1>LanePulse</h1><p>The UI has not been built. "
                                "Run <code>python scripts/build_ui.py</code> on the staging machine.</p>")
    return app


def main(argv=None) -> int:
    from trafficwb import safe_console
    safe_console()
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(prog="trafficwb-web")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--data", required=True)
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--lan", action="store_true", help="accept connections from other computers on the network")
    s.add_argument("--backup-dir", help="folder for the nightly database backup, ideally on another drive")
    s.add_argument("--backup-at", default="02:00", help="local time of the nightly backup (HH:MM)")
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        host = "0.0.0.0" if args.lan else "127.0.0.1"
        uvicorn.run(create_app(args.data, backup_dir=args.backup_dir, backup_at=args.backup_at), host=host, port=args.port, log_level="info")
        return 0
    ap.print_help()
    return 1
