"""FastAPI application: JSON API under /api and the built UI as static files.

Browsers never touch database files (ARC-02); every read and write goes through here.
Each request opens its own database connections, so request threads never share one.
Mutating requests need the X-TWB header, which blocks cross-site form posts.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
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
from ..detectors import fast_of, grades as detector_grades, is_rfdetr, rfdetr_info, uses_sam
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
    source_dir: str = ""
    source_dirs: list[str] | None = None   # several source folders; source_dir is the first
    include_subdirs: bool = True
    processing: str = "one_time"
    output_dir: str
    export_formats: list[str] = Field(default_factory=lambda: ["xlsx", "csv"])
    latitude: float | None = None
    longitude: float | None = None
    detector: str = "standard"
    sam3_words: dict[str, list[str]] | None = None
    vehicle_model: str | None = None       # a saved model's name; None: the model in production use
    sam3_interval_s: float | None = None   # video: seconds between SAM 3.1 frames (None: 1 s)
    camera: str = "fixed"                  # video: fixed | moving (PTZ, vehicle, drone, body-worn)
    video_workers: int | None = None       # video: segments analysed side by side (None: 1)
    run_mode: str = "manual"               # learning: manual decisions | auto
    teacher: str = "none"                  # learning: none | sam3
    auto_min_reviews: int = 20
    auto_epochs: int = 30
    analyse_new: bool = False              # video: new videos analysed as they arrive

    def config(self) -> MissionConfig:
        d = self.model_dump()
        dirs = [x.strip() for x in (d.pop("source_dirs") or []) if x and x.strip()]
        first = d["source_dir"].strip() or (dirs[0] if dirs else "")
        rest = [x for x in dirs if x != first]
        return MissionConfig(**d | {"source_dir": first, "source_dirs": [first, *rest] if rest else None})


class DetectorBody(BaseModel):
    detector: str
    sam3_interval_s: float | None = None   # video: seconds between SAM 3.1 frames; None leaves it as it is
    camera: str | None = None              # video: fixed | moving; None leaves it as it is
    video_workers: int | None = None       # video: segments analysed side by side; None leaves it as it is
    track_min_frames: int | None = None    # video: analysed frames before a target counts; 0 back to the default,
                                           # None leaves it as it is


class SizeBody(BaseModel):
    family: str
    size: str | None = None                # None: automatic (by this computer's hardware)


class RunModeBody(BaseModel):
    run_mode: str                          # manual | auto
    auto_min_reviews: int | None = Field(None, ge=1, le=100000)
    auto_epochs: int | None = Field(None, ge=1, le=500)
    analyse_new: bool | None = None        # video: analyse new videos in the input folder with the mission's model
    teacher: str | None = None             # none | sam3: SAM 3.1 labels what nobody reviewed, for training


class AnalyseBody(BaseModel):
    version_ids: list[int]                 # the videos ticked in the catalog


class MarkBody(BaseModel):
    cls: str                               # what the target is (a class of the mission, or a new one)
    t: float                               # the frame the box was drawn on
    box: list[float]                       # x1, y1, x2, y2 in the recording's pixels


class MarkCorrectBody(BaseModel):
    t: float
    box: list[float]


class MarkUpdateBody(BaseModel):
    start_t: float | None = None
    end_t: float | None = None
    state: str | None = None               # suggested | confirmed
    cls: str | None = None


class RecordingReviewBody(BaseModel):
    action: str                            # approve | confirm_rest_and_approve | reopen


class VehicleModelBody(BaseModel):
    saved: str | None = None               # a saved model's name; None: the model in production use


class SourcesBody(BaseModel):
    add: list[str] = Field(min_length=1, max_length=50)


class WordsBody(BaseModel):
    words: dict[str, list[str]] | None = None     # None: back to the default words
    uncounted: list[str] = Field(default_factory=list)   # classes of the mission's own not counted as vehicles


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
    min_score: float | None = Field(None, ge=0, le=1)


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
    min_score: float | None = Field(None, ge=0, le=1)   # events: the machine's events scoring under this are left out


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
    box: list[float] | None = None


class SynopsisBody(BaseModel):
    event_ids: list[int] | None = Field(None, max_length=100000)
    busy: str = Field("normal", pattern="^(calm|normal|busy)$")
    status_only: bool = False


class MapBody(BaseModel):
    kind: str = Field("heat", pattern="^(heat|paths)$")
    event_ids: list[int] | None = Field(None, max_length=100000)
    zones: bool = True


class ZoneBody(BaseModel):
    kind: str = Field(..., pattern="^(line|area)$")
    name: str = Field("", max_length=60)
    points: list[list[float]] = Field(..., min_length=2, max_length=64)


class CrowdAlertBody(BaseModel):
    people: int | None = Field(None, ge=2, le=10000)


class GeoBody(BaseModel):
    points: list[list[float]] = Field(..., min_length=1, max_length=100)    # [x, y, lat, lon]
    map_path: str | None = None
    corners: list[float] | None = Field(None, min_length=4, max_length=4)
    check_only: bool = False


class CalibrationBody(BaseModel):
    points: list[float] = Field(..., min_length=4, max_length=4)
    meters: float


class WorkerSwitchBody(BaseModel):
    enabled: bool


class WorkerCodeBody(BaseModel):
    label: str = Field("worker", max_length=60)


class WorkerHello(BaseModel):
    key: str = Field(..., min_length=32, max_length=128)
    name: str = Field(..., min_length=1, max_length=80)
    info: dict = {}


class WorkerAnswerBody(BaseModel):
    allow: bool


class WorkerCall(BaseModel):
    code: str
    name: str = Field(..., min_length=1, max_length=80)
    can: list[str] = ["standard"]
    info: dict = {}
    result: dict | None = None
    error: str | None = None


class AddTargetBody(BaseModel):
    t: float = Field(..., ge=0)
    box: list[float]
    cls: str
    note: str | None = None


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


class FormatItem(BaseModel):
    name: str = ""
    template: str = ""
    zone: str = "UTC"


class FormatsBody(BaseModel):
    formats: list[FormatItem]


class FormatTryBody(BaseModel):
    template: str = ""
    zone: str = "UTC"
    examples: list[str] = []
    folder: str | None = None       # also try every image or video name in this folder (first 5000)
    include_subdirs: bool = True


class PreviewBody(BaseModel):
    source_dir: str
    include_subdirs: bool = True
    limit: int = 20


class QuickBody(BaseModel):
    source_dir: str
    include_subdirs: bool = True


class LearnFormatBody(BaseModel):
    folder: str
    include_subdirs: bool = True
    zone: str | None = None     # the time zone the names are in; None: UTC
    save: bool = False
    name: str | None = None


class FolderBody(BaseModel):
    path: str


class BandsBody(BaseModel):
    far_below: float
    close_from: float


class DeleteBody(BaseModel):
    confirm_name: str


class DatasetRef(BaseModel):
    mission_id: str
    dataset_id: int


class SaveModelBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = Field(None, max_length=500)
    model_name: str
    model_version: str
    mission_id: str | None = None
    evaluation_id: int | None = None


class DatasetBody(BaseModel):
    name: str
    ratios: list[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    block_days: int = 1
    seed: int = 0
    teacher: str = "none"             # sam3: SAM 3.1 first labels what nobody reviewed (a background job)


class EvaluateBody(BaseModel):
    dataset_id: int | None = None
    datasets: list[DatasetRef] | None = None   # several dataset versions, from this mission or others
    split: str = "test"
    name: str | None = None           # model pack; empty means the model in production use
    version: str | None = None
    iou: float = Field(0.5, ge=0.1, le=0.95)


class BenchmarkBody(BaseModel):
    dataset_id: int | None = None
    datasets: list[DatasetRef] | None = None
    split: str = "test"
    iou: float = Field(0.5, ge=0.1, le=0.95)
    models: list[str] | None = None   # "<name>/<version>"; empty means every installed vehicle detector


class TrainBody(BaseModel):
    dataset_id: int | None = None
    datasets: list[DatasetRef] | None = None   # several dataset versions, from this mission or others
    epochs: int = Field(20, ge=1, le=300)
    start_from: str | None = None     # "coco" (the released weights) or "<name>/<version>" of a trained model
    arch: str = "yolox"               # what to train from the released weights: yolox, rf-detr, d-fine or rt-detr
    size: str | None = None           # RF-DETR only: nano, small or medium (default small)


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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".mp4", ".mkv", ".avi", ".mov", ".ts"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".ts"}
# Folder names that say nothing about the camera: a mission is then named after the folder above.
GENERIC_FOLDERS = {"images", "image", "img", "photos", "pictures", "videos", "video", "recordings", "frames", "data", "src",
                   "source", "input", "export", "exports", "dcim"}


def quick_start(source_dir: str, include_subdirs: bool, data_root: Path, limit: int = 5000) -> dict:
    """What a new mission for this folder most likely is, so it can be created on one screen: a name, still images or
    video, how many names give a capture time and over what span, the time zone of the latest mission, and an
    exports folder beside the source."""
    root = Path(source_dir.strip()).expanduser()
    if not root.is_dir():
        raise ApiError(400, "no_folder", f"{root} is not a folder on this workstation.")
    images = videos = readable = 0
    first = last = None
    pattern = None
    unreadable: list[dict] = []
    more = False
    for p in (root.rglob("*") if include_subdirs else root.glob("*")):
        if p.name.startswith(".") or p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        if images + videos >= limit:
            more = True
            break
        if p.suffix.lower() in VIDEO_EXTS:
            videos += 1
        else:
            images += 1
        try:
            ct = parse_capture_time(p.name)
        except MalformedName as exc:
            if len(unreadable) < 3:
                unreadable.append({"file": p.name, "reason": exc.reason})
            continue
        readable += 1
        pattern = pattern or ct.pattern
        first = ct.utc if first is None or ct.utc < first else first
        last = ct.utc if last is None or ct.utc > last else last
    named = root.parent if root.name.lower() in GENERIC_FOLDERS and root.parent != root else root
    words = named.name.replace("_", " ").replace("-", " ").split()
    if first is not None:
        # A date already in the folder name is said once, in words, at the end.
        words = [w for w in words if not (w.isdigit() and len(w) == 8 and w.startswith(("19", "20")))]
    name = " ".join(words) or "New mission"
    if first is not None:
        name += f", {first.day} {first:%b %Y}"
    from .. import defaults
    settings = defaults.load(data_root)
    both = settings["both"]
    zone, country = "Asia/Jerusalem", "IL"
    zone_from = None
    dr = DataRoot(data_root)
    rows = sorted(dr.missions(), key=lambda r: r["created_at"])
    if both["iana_zone"]:
        zone, zone_from = both["iana_zone"], "settings"
        country = both["country"] or next((c for c, z in COUNTRY_ZONES.items() if z == zone), country)
    elif rows:
        cfg = dr.open_mission(rows[-1]["id"]).config()
        zone, country, zone_from = cfg.iana_zone, cfg.country, rows[-1]["name"]
    # Results go in the results folder chosen in Settings, or else in a new folder beside the input.
    base = Path(both["results_folder"]).expanduser() if both["results_folder"] else (root.parent if root.parent != root else root)
    out = base / f"{root.name} LanePulse exports"
    input_type = "video" if videos > images else "images"
    return {
        "source_dir": str(root), "name": name, "input_type": input_type,
        "images": images, "videos": videos, "readable": readable, "more": more, "pattern": pattern,
        "first_utc": first.isoformat().replace("+00:00", "Z") if first else None,
        "last_utc": last.isoformat().replace("+00:00", "Z") if last else None,
        "unreadable": unreadable, "iana_zone": zone, "country": country, "zone_from": zone_from,
        "output_dir": str(out), "output_exists": out.is_dir(),
        "settings": defaults.mission_fields(settings, input_type),
    }


def folder_roots() -> list[dict]:
    """Where browsing starts: the drives on Windows, the home folder and / elsewhere, and mounted volumes."""
    out: list[dict] = []
    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            p = Path(f"{letter}:\\")
            try:
                if p.exists():
                    out.append({"name": f"{letter}:", "path": str(p)})
            except OSError:
                pass
    else:
        home = Path.home()
        out.append({"name": "Home", "path": str(home)})
        for base in (Path("/Volumes"), Path("/mnt"), Path("/media")):
            try:
                for d in sorted(base.iterdir()) if base.is_dir() else []:
                    if d.is_dir() and not d.name.startswith("."):
                        out.append({"name": d.name, "path": str(d)})
            except OSError:
                pass
        out.append({"name": "/", "path": "/"})
    return out


def list_folders(path: str | None, limit: int = 500) -> dict:
    """The subfolders of path (hidden ones left out) and how many images or videos sit directly in it."""
    if not path or not path.strip():
        return {"path": None, "parent": None, "folders": [], "files": 0, "roots": folder_roots()}
    p = Path(path.strip()).expanduser()
    if not p.is_dir():
        raise ApiError(400, "no_folder", f"{p} is not a folder on this workstation.")
    folders, files, more = [], 0, False
    try:
        entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
    except OSError as exc:
        raise ApiError(400, "no_access", f"{p} cannot be opened: {exc.strerror or exc}.") from None
    for e in entries:
        if e.name.startswith((".", "$")):
            continue
        try:
            if e.is_dir():
                if len(folders) < limit:
                    folders.append({"name": e.name, "path": str(Path(e.path))})
                else:
                    more = True
            elif Path(e.name).suffix.lower() in IMAGE_EXTS:
                files += 1
        except OSError:
            continue
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "folders": folders, "more": more, "files": files, "roots": folder_roots()}


def default_detector_factory(data_root: Path | str | None = None):
    """The vehicle detector in production use: the last promoted pack, else the built-in default."""
    from .. import models
    from ..detect import make_detector
    pack = None
    if data_root is not None:
        from ..learning import ModelRegistry
        prod = ModelRegistry(data_root).production("vehicle_detection")
        if prod:
            pack = models.find(prod["name"], prod["version"])
    if pack is None:                                   # the YOLOX size that suits this computer (hardware.py)
        from ..hardware import yolox_pack
        pack = yolox_pack(data_root)
    return make_detector(pack), pack.card


ACCESS_COOKIE = "twb_access"


class AccessBody(BaseModel):
    key: str


def access_key_for(data_root: Path | str, new: bool = False) -> str:
    """The network access key, kept in <data>/access-key (readable by the server's account only), made when missing."""
    p = Path(data_root) / "access-key"
    if not new:
        try:
            k = p.read_text(encoding="utf-8").strip()
            if k:
                return k
        except FileNotFoundError:
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    k = "-".join(secrets.token_hex(3) for _ in range(5))          # e.g. 3f9a1c-07be42-…: easy to read out and type
    tmp = p.with_name("access-key.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(k + "\n")
    os.replace(tmp, p)
    return k


def _access_token(key: str) -> str:
    """What the browser keeps in its cookie: derived from the key, so the cookie is not the key itself."""
    return hashlib.sha256(("lanepulse-access:" + key).encode()).hexdigest()


def _host_allowlist(listening: dict) -> Callable[[str], bool] | None:
    """The Host header check for a server started by `serve` (None: not started that way, e.g. inside tests).

    On this computer only: localhost and loopback addresses. On the network (--lan): also any IP address, this
    computer's own names, and the names in TWB_ALLOWED_HOSTS (comma separated). Names an attacker controls are
    refused, which is what stops DNS rebinding.
    """
    import ipaddress
    import os
    import socket
    if not listening.get("port"):
        return None
    names = {"localhost"}
    if listening.get("lan"):
        try:
            h = socket.gethostname().lower()
            names |= {h, h.split(".")[0], h.split(".")[0] + ".local", socket.getfqdn().lower()}
        except OSError:
            pass
        names |= {n.strip().lower() for n in os.environ.get("TWB_ALLOWED_HOSTS", "").split(",") if n.strip()}

    def ok(host: str) -> bool:
        host = host.strip().lower()
        if host.startswith("["):                       # [IPv6]:port
            name = host[1:host.find("]")] if "]" in host else ""
        else:
            name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        name = name.rstrip(".")
        if name in names:
            return True
        try:
            ip = ipaddress.ip_address(name)
        except ValueError:
            return False
        return bool(listening.get("lan")) or ip.is_loopback

    return ok


def create_app(data_root: Path | str, detector_factory: Callable | None = None,
               monitor_interval: float = 60, stability_seconds: int = 30, start_monitoring: bool = True,
               plate_factory: Callable | None = None, model_loader: Callable | None = None,
               backup_dir: Path | str | None = None, backup_at: str = "02:00",
               sam_factory: Callable | None = None, sam_status: Callable[[], dict] | None = None,
               listening: dict | None = None) -> FastAPI:
    data_root = Path(data_root)
    listening = dict(listening or {})       # how the server was started (for the Workers screen)
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

    discovery = None

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal discovery
        if listening.get("lan") and listening.get("discovery", True):
            # Helping computers on this network find the server without an address typed.
            from ..discovery import PORT, Responder
            from ..workers import Requests
            discovery = Responder(listening.get("port", 8765), Requests(data_root).server_id(), __version__,
                                  listening.get("discovery_port", PORT)).start()
        if start_monitoring:
            jobs.resume_monitoring()  # monitoring survives restarts (MIS-07)
            jobs.start_auto_loop()    # missions in Run mode "full auto" learn on their own
        if backups:
            backups.start()
        yield
        if backups:
            backups.stop()
        if discovery:
            discovery.stop()
        jobs.shutdown()

    app = FastAPI(title="LanePulse", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.exception_handler(HTTPException)
    def _http_error(_req, exc: HTTPException):
        body = exc.detail if isinstance(exc.detail, dict) else {"error": "error", "message": str(exc.detail)}
        return JSONResponse(body, status_code=exc.status_code)

    allowed_host = _host_allowlist(listening)

    @app.middleware("http")
    async def _host_check(request: Request, call_next):
        # DNS rebinding: a web page whose own domain is switched to this computer's address would otherwise be
        # same-origin with the API. Only names that really mean this server are answered.
        if allowed_host is not None and not allowed_host(request.headers.get("host", "")):
            return JSONResponse({"error": "bad_host", "message": "This address is not one LanePulse answers on. "
                                 "Open it by its IP address, or add the name to TWB_ALLOWED_HOSTS."}, status_code=421)
        return await call_next(request)

    access_key = access_key_for(data_root) if listening.get("lan") else None
    access_token = _access_token(access_key) if access_key else None
    OPEN_PATHS = ("/api/health", "/api/access")

    def has_access(request: Request) -> bool:
        if access_token is None:
            return True
        if hmac.compare_digest(request.cookies.get(ACCESS_COOKIE, ""), access_token):
            return True
        auth = request.headers.get("authorization", "")
        return auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip(), access_key)

    @app.middleware("http")
    async def _access(request: Request, call_next):
        # On the network the API needs the access key (made at install, `trafficwb-web key`). Workers keep their own
        # pairing codes on /api/worker/; the screens themselves are not data and load without it, to ask for the key.
        path = request.url.path
        if access_token is not None and path.startswith("/api/") and path not in OPEN_PATHS \
                and not path.startswith("/api/worker/") and not has_access(request):
            return JSONResponse({"error": "need_key", "message": "Enter the LanePulse access key."}, status_code=401)
        return await call_next(request)

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

    @app.get("/api/access")
    def access_state(request: Request):
        return {"required": access_token is not None, "ok": has_access(request)}

    @app.post("/api/access")
    def access_enter(body: AccessBody, request: Request):
        if access_token is None:
            return {"required": False, "ok": True}
        if not hmac.compare_digest(body.key.strip(), access_key):
            time.sleep(1)                                  # slows guessing; the key is long and random anyway
            raise ApiError(403, "wrong_key", "That is not the access key. Ask whoever installed the server, "
                                             "or run: sudo lanepulse key")
        r = JSONResponse({"required": True, "ok": True})
        r.set_cookie(ACCESS_COOKIE, access_token, max_age=400 * 86400, httponly=True, samesite="strict",
                     secure=request.url.scheme == "https")
        return r

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
                "sam3": sam_info(), "rfdetr": rfdetr_info(data_root), "grades": detector_grades()}

    @app.get("/api/detectors")
    def detectors(_u: User = Depends(user)):
        """For the detector picker: what RF-DETR would be here, and how each detector scored on the benchmarks."""
        return {"rfdetr": rfdetr_info(data_root), "grades": detector_grades()}

    @app.get("/api/sam3")
    def sam3_status(_u: User = Depends(user)):
        from ..sam3 import AREA_SUGGESTIONS, VEHICLE_PROMPTS
        from ..detect import COCO_ALIASES, COCO_OTHER
        return {**sam_info(), "suggestions": AREA_SUGGESTIONS, "vehicle_prompts": VEHICLE_PROMPTS,
                # class names the standard detector also finds when a mission adds them (a pedestrian is a person)
                "standard_knows": sorted({*COCO_OTHER.values(), *COCO_ALIASES})}

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

    @app.post("/api/quick-start")
    def quick_start_get(body: QuickBody, _u: User = Depends(user)):
        return quick_start(body.source_dir, body.include_subdirs, data_root)

    @app.get("/api/settings/missions")
    def mission_defaults_get(_u: User = Depends(user)):
        """What new missions take: for still images and video alike, for still images only, for video only."""
        from .. import defaults
        return defaults.load(data_root)

    @app.put("/api/settings/missions")
    def mission_defaults_put(body: dict[str, Any], u: User = Depends(user)):
        from .. import defaults
        try:
            out = defaults.save(data_root, body)
        except ValueError as exc:
            raise ApiError(400, "invalid", str(exc)) from None
        from ..admin import audit
        audit(DataRoot(data_root), u.username, "settings_changed", parts=sorted(body))
        return out

    @app.post("/api/quick-start/output")
    def quick_start_output(body: FolderBody, _u: User = Depends(user)):
        """Make the suggested exports folder: one new folder inside a folder that already exists, nothing deeper."""
        p = Path(body.path.strip()).expanduser()
        if not p.parent.is_dir():
            raise ApiError(400, "no_folder", f"{p.parent} is not a folder on this workstation.")
        try:
            p.mkdir(exist_ok=True)
        except OSError as exc:
            raise ApiError(400, "no_access", f"{p} could not be made: {exc.strerror or exc}.") from None
        return {"path": str(p)}

    def _saved_or_none(name: str):
        from ..learning import ModelLibrary
        return ModelLibrary(data_root).get(name)

    @app.get("/api/folders")
    def folders(path: str | None = None, _u: User = Depends(user)):
        """Folders on this computer, for the folder picker: the subfolders of path, or the places to start from."""
        return list_folders(path)

    # ---------- filename formats ----------
    @app.get("/api/name-formats")
    def name_formats_get(_u: User = Depends(user)):
        from ..name_formats import load
        from ..timeparse import DEFAULT_PATTERNS, TOKEN_LABELS
        return {"formats": load(data_root),
                "builtin": [{"name": p.name, "description": p.description} for p in DEFAULT_PATTERNS],
                "tokens": [{"token": "{" + t + "}", "label": lbl} for t, lbl in TOKEN_LABELS.items()]}

    @app.post("/api/name-formats/suggest")
    def name_formats_suggest(body: FormatTryBody, _u: User = Depends(user)):
        from ..timeparse import suggest_template
        name = (body.examples or [""])[0]
        return {"template": suggest_template(name)}

    @app.post("/api/name-formats/try")
    def name_formats_try(body: FormatTryBody, _u: User = Depends(user)):
        from ..inventory import IMAGE_EXT, VIDEO_EXT
        from ..name_formats import try_format
        out = try_format(body.template, body.zone, [e for e in body.examples if e.strip()])
        if body.folder and out["ok"]:
            root = Path(body.folder)
            if not root.is_dir():
                out["folder"] = {"error": f"{root} is not a folder on this workstation."}
            else:
                names = []
                for p in (root.rglob("*") if body.include_subdirs else root.glob("*")):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT | VIDEO_EXT:
                        names.append(p.name)
                        if len(names) >= 5000:
                            break
                rows = try_format(body.template, body.zone, names)["results"]
                bad = [r for r in rows if not r["ok"]]
                out["folder"] = {"checked": len(rows), "matched": len(rows) - len(bad),
                                 "not_matched_examples": [r["name"] for r in bad[:10]],
                                 "matched_examples": [r for r in rows if r["ok"]][:3]}
        return out

    @app.post("/api/name-formats/learn")
    def name_formats_learn(body: LearnFormatBody, u: User = Depends(user)):
        """A new kind of file name in an input folder: guess its format from the names that cannot be read now and
        check it on all of them. With save, remember it with the other formats, so every mission reads such names
        from then on."""
        from ..inventory import IMAGE_EXT, VIDEO_EXT
        from ..name_formats import load, save, try_format
        from ..timeparse import FormatError, suggest_template
        root = Path(body.folder.strip()).expanduser()
        if not root.is_dir():
            raise ApiError(400, "no_folder", f"{root} is not a folder on this workstation.")
        unread: list[str] = []
        for p in (root.rglob("*") if body.include_subdirs else root.glob("*")):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXT | VIDEO_EXT:
                try:
                    parse_capture_time(p.name)
                except MalformedName:
                    unread.append(p.name)
                if len(unread) >= 2000:
                    break
        if not unread:
            return {"template": None, "total": 0, "matched": 0, "examples": []}
        template = suggest_template(unread[0])
        if not template:
            return {"template": None, "total": len(unread), "matched": 0, "examples": [],
                    "reason": "No date and time could be found in names like " + unread[0] + "."}
        zone = body.zone or "UTC"
        rows = try_format(template, zone, unread)
        good = [r for r in rows.get("results", []) if r.get("ok")]
        out = {"template": template, "zone": zone, "total": len(unread), "matched": len(good),
               "examples": [{"name": r["name"], "utc": r["utc"]} for r in good[:3]],
               "not_matched": [r["name"] for r in rows.get("results", []) if not r.get("ok")][:3]}
        if body.save:
            if not good:
                raise ApiError(400, "bad_format", "This format reads none of the names, so it was not kept.")
            have = load(data_root)
            if any(f.get("template") == template and (f.get("zone") or "UTC") == zone for f in have):
                return out | {"saved": True}
            names = {f.get("name") for f in have}
            base = (body.name or "").strip() or f"Learned from {root.name}"
            name, n = base, 2
            while name in names:
                name, n = f"{base} ({n})", n + 1
            try:
                save(DataRoot(data_root), [*have, {"name": name, "template": template, "zone": zone}], u.username)
            except FormatError as exc:
                raise ApiError(400, "bad_format", str(exc)) from None
            out |= {"saved": True, "name": name}
        return out

    @app.put("/api/name-formats")
    def name_formats_put(body: FormatsBody, u: User = Depends(user)):
        from ..name_formats import save
        from ..timeparse import FormatError
        try:
            saved = save(DataRoot(data_root), [f.model_dump() for f in body.formats], u.username)
        except FormatError as exc:
            raise ApiError(400, "bad_format", str(exc))
        return {"formats": saved}

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
                        "status": m.status(), "job": jobs.current(r["id"]).snapshot() if jobs.current(r["id"]) else None,
                        "review": _review_summary(m) if r["input_type"] == "images" else None})
        return {"missions": out}

    def _review_summary(m: Mission) -> dict:
        """For the missions list: how many processed images wait for review, the first of them, and the latest image."""
        rows = ReviewService(m).results(Filter(), "capture_utc", False)
        waiting = [r.source_version_id for r in rows if r.processing_state == "done" and r.review_status == "unreviewed"]
        done = [r for r in rows if r.processing_state == "done"]
        return {"unreviewed": len(waiting), "first_unreviewed": waiting[0] if waiting else None,
                "approved": sum(1 for r in rows if r.review_status != "unreviewed"), "processed": len(done),
                "latest": rows[-1].source_version_id if rows else None}

    @app.post("/api/missions/validate")
    def mission_validate(body: MissionBody, _u: User = Depends(user)):
        v = validate(body.config(), data_root)
        if body.vehicle_model and not _saved_or_none(body.vehicle_model):
            v.errors.append(f"There is no saved model named “{body.vehicle_model}”.")
        return {"errors": v.errors, "warnings": v.warnings}

    @app.post("/api/missions")
    def mission_create(body: MissionBody, u: User = Depends(user)):
        if uses_sam(body.detector) and not sam_info()["installed"]:
            raise ApiError(400, "sam3_missing", "SAM 3.1 is not installed on this computer: " + (sam_info()["reason"] or ""),
                           "Choose the standard detector, or install the full LanePulse package.")
        if fast_of(body.detector) == "rfdetr" and not body.vehicle_model and not rfdetr_info(data_root)["installed"]:
            raise ApiError(400, "rfdetr_missing", rfdetr_info(data_root)["reason"], "Choose another detector.")
        if body.vehicle_model and not _saved_or_none(body.vehicle_model):
            raise ApiError(400, "invalid", f"There is no saved model named “{body.vehicle_model}”.")
        try:
            m = DataRoot(data_root).create_mission(body.config(), u.username)
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
        jobs.watch_new_videos(m.id)
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
        own = [k for k in prompts if k not in VEHICLE_PROMPTS]
        bad = sorted(set(body.uncounted) - set(own))
        if bad:
            raise ApiError(400, "invalid", f"Only a class of your own can be left out of the vehicle count, not {', '.join(bad)}.")
        uncounted = sorted(set(body.uncounted)) or None
        cfg = m.config()
        if words != getattr(cfg, "sam3_words", None) or uncounted != getattr(cfg, "sam3_uncounted", None):
            m.update_config(u.username, "words for SAM 3.1 changed" if words else "words for SAM 3.1 back to the defaults",
                            sam3_words=words, sam3_uncounted=uncounted)
        return {"words": {k: list(v) for k, v in prompts.items()}, "default": words is None,
                "own_classes": own, "uncounted": uncounted or []}

    @app.put("/api/missions/{mission_id}/detector")
    def detector_set(body: DetectorBody, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..store import DETECTORS
        cfg = m.config()
        if body.detector not in DETECTORS:
            raise ApiError(400, "invalid", f"Detector must be one of {sorted(DETECTORS)}.")
        if body.camera is not None and body.camera not in ("fixed", "moving"):
            raise ApiError(400, "invalid", "Camera must be fixed or moving.")
        if body.sam3_interval_s is not None and not (0.2 <= body.sam3_interval_s <= 10):
            raise ApiError(400, "invalid", "SAM 3.1 in video runs every 0.2 to 10 seconds of the recording.")
        if body.video_workers is not None and not (1 <= body.video_workers <= 32):
            raise ApiError(400, "invalid", "Segments side by side must be 1 to 32.")
        if body.track_min_frames is not None and not (0 <= body.track_min_frames <= 30):
            raise ApiError(400, "invalid", "Frames before a target counts must be 1 to 30.")
        if uses_sam(body.detector) and not sam_info()["installed"]:
            raise ApiError(400, "sam3_missing", "SAM 3.1 is not installed on this computer: " + (sam_info()["reason"] or ""))
        if fast_of(body.detector) == "rfdetr" and not cfg.vehicle_model and not rfdetr_info(data_root)["installed"]:
            raise ApiError(400, "rfdetr_missing", rfdetr_info(data_root)["reason"])
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Processing is running. Change the detector when it has finished or is cancelled.")
        if body.detector != getattr(cfg, "detector", "standard"):
            m.update_config(u.username, f"detector set to {body.detector}", detector=body.detector)
        if body.sam3_interval_s is not None and body.sam3_interval_s != cfg.sam3_interval_s:
            m.update_config(u.username, f"SAM 3.1 every {body.sam3_interval_s:g} s of video",
                            sam3_interval_s=body.sam3_interval_s)
        if body.camera is not None and body.camera != cfg.camera:
            m.update_config(u.username, f"camera set to {body.camera}", camera=body.camera)
        if body.video_workers is not None and body.video_workers != cfg.video_workers:
            m.update_config(u.username, f"{body.video_workers} segments side by side", video_workers=body.video_workers)
        if body.track_min_frames is not None and (body.track_min_frames or None) != cfg.track_min_frames:
            n = body.track_min_frames or None
            m.update_config(u.username, f"a target counts after {n} analysed frames" if n else
                            "frames before a target counts back to the default", track_min_frames=n)
        c2 = m.config()
        return {"detector": body.detector, "sam3_interval_s": c2.sam3_interval_s, "camera": c2.camera,
                "video_workers": c2.video_workers, "track_min_frames": c2.track_min_frames}

    @app.post("/api/missions/{mission_id}/sources")
    def sources_add(body: SourcesBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Add source folders to a mission; the next scan reads them too. Folders are never removed."""
        cfg = m.config()
        have = cfg.sources()
        new = [d.strip() for d in body.add if d.strip() and d.strip() not in have]
        if not new:
            raise ApiError(400, "invalid", "Those folders are already sources of this mission.")
        trial = MissionConfig(**asdict(cfg) | {"source_dirs": [*have, *new]})
        v = validate(trial, data_root, min_free_bytes=0)
        if v.errors:
            raise ApiError(400, "invalid", " ".join(v.errors))
        for d in new:
            m.con.execute("INSERT OR IGNORE INTO source_root(path,recursive) VALUES (?,?)",
                          (str(Path(d).resolve()), int(cfg.include_subdirs)))
        m.update_config(u.username, "source folders added: " + ", ".join(new), source_dirs=[*have, *new])
        if m.id in jobs.monitors:                     # watch the new folders too
            jobs.set_monitoring(m.id, False)
            jobs.set_monitoring(m.id, True)
        return {"sources": [*have, *new]}

    @app.put("/api/missions/{mission_id}/vehicle-model")
    def vehicle_model_set(body: VehicleModelBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Which model the standard detector uses for this mission: a saved model, or the one in production use."""
        if body.saved and not _saved_or_none(body.saved):
            raise ApiError(400, "invalid", f"There is no saved model named “{body.saved}”.")
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Processing is running. Change the model when it has finished or is cancelled.")
        if (body.saved or None) != getattr(m.config(), "vehicle_model", None):
            m.update_config(u.username, f"model set to {body.saved}" if body.saved else "model set to the one in production use",
                            vehicle_model=body.saved or None)
        return {"vehicle_model": body.saved or None}

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
            "classes": rs.classes(),
            "uncounted_classes": sorted(rs.uncounted()),
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
        return LearningService(m)

    def _auto_soon(mission_id: str) -> None:
        """After a review decision, let a mission in full auto check whether a round is due (in the background)."""
        def go():
            try:
                jobs.auto_learn(mission_id)
            except Exception:
                pass
        threading.Thread(target=go, name="auto-learning-check", daemon=True).start()

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
        in_use = {"name": prod["name"], "version": prod["version"]} if prod else None
        if in_use is None:
            from ..hardware import yolox_pack
            try:
                d = yolox_pack(data_root)
                in_use = {"name": d.name, "version": d.version}
            except models.ModelPackError:
                pass
        from .. import training_env
        tj = jobs.training_busy()
        from ..learning import ModelLibrary
        saved = ModelLibrary(data_root).saved()
        by_pack = {(x["model_name"], x["model_version"]): x["name"] for x in saved}
        installed = [md | {"saved_as": by_pack.get((md["name"], md["version"]))} for md in models.installed("vehicle_detection")]
        cfg = m.config()
        auto = {"run_mode": cfg.run_mode, "auto_min_reviews": cfg.auto_min_reviews, "auto_epochs": cfg.auto_epochs,
                "analyse_new": cfg.analyse_new, "teacher": cfg.teacher, "new_reviews": ls.reviews_since_last_dataset(),
                "history": ls.auto_history()}
        return {"in_use": in_use, "eligibility": ls.eligibility(), "datasets": ls.datasets(), "evaluations": ls.evaluations(),
                "auto": auto, "video": m.input_type == "video",
                "all_datasets": ls.all_datasets(), "training_runs": ls.training_runs(), "carry_over": ls.carry_over(),
                "saved_models": saved, "mission_model": getattr(m.config(), "vehicle_model", None),
                "adapters": ADAPTERS, "models": installed,
                "production": prod, "history": reg.history("vehicle_detection")[:20],
                "training": {**training_env.status(), "busy_mission": tj.mission_id if tj else None,
                             "job": (lambda j: j.snapshot() if j else None)(jobs.training_job(m.id))}}

    @app.put("/api/missions/{mission_id}/run-mode")
    def run_mode_set(body: RunModeBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Manual decisions (the operator builds datasets, trains and chooses the model) or full auto (the loop does
        it and adopts a new model for this mission only when it did better)."""
        from ..store import RUN_MODES, TEACHERS
        if body.run_mode not in RUN_MODES:
            raise ApiError(400, "invalid", "Run mode must be manual or auto.")
        if body.teacher is not None and body.teacher not in TEACHERS:
            raise ApiError(400, "invalid", "Learn from must be none or sam3.")
        cfg = m.config()
        changes = {k: v for k, v in (("run_mode", body.run_mode), ("auto_min_reviews", body.auto_min_reviews),
                                     ("auto_epochs", body.auto_epochs), ("analyse_new", body.analyse_new),
                                     ("teacher", body.teacher))
                   if v is not None and getattr(cfg, k) != v}
        if changes:
            what = ("run mode set to " + ("full auto" if body.run_mode == "auto" else "manual decisions")
                    if "run_mode" in changes else "automatic learning settings changed")
            m.update_config(u.username, what, **changes)
            from ..learning import LearningService
            LearningService(m).auto_log("mode", run_mode=body.run_mode, author=u.username, **{
                k: v for k, v in changes.items() if k != "run_mode"})
        status = jobs.auto_learn(m.id) if body.run_mode == "auto" else {"state": "manual"}
        jobs.watch_new_videos(m.id)
        c2 = m.config()
        return {"run_mode": c2.run_mode, "auto_min_reviews": c2.auto_min_reviews, "auto_epochs": c2.auto_epochs,
                "analyse_new": c2.analyse_new, "teacher": c2.teacher,
                "status": {k: v for k, v in status.items() if k != "job"}}

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/review")
    def recording_review(version_id: int, body: RecordingReviewBody, m: Mission = Depends(mission),
                         u: User = Depends(user)):
        """Approve a recording for learning (every target is an event and every event is decided), or reopen it."""
        ls = _learn(m)
        if body.action in ("approve", "confirm_rest_and_approve"):
            out = _learn_call(lambda: ls.approve_recording(version_id, u.username,
                                                           confirm_rest=body.action == "confirm_rest_and_approve"))
        elif body.action == "reopen":
            out = _learn_call(lambda: ls.reopen_recording(version_id, u.username))
        else:
            raise ApiError(400, "invalid", "Action must be approve, confirm_rest_and_approve or reopen.")
        _auto_soon(m.id)
        return out

    @app.post("/api/missions/{mission_id}/learning/datasets")
    def dataset_create(body: DatasetBody, m: Mission = Depends(mission), u: User = Depends(user)):
        ls = _learn(m)
        if body.teacher not in ("none", "sam3"):
            raise ApiError(400, "invalid", "Learn from must be none or sam3.")
        if body.teacher == "none":
            return _learn_call(lambda: ls.create_dataset(body.name, u.username, tuple(body.ratios), body.block_days,
                                                         body.seed))
        if not body.name.strip():
            raise ApiError(400, "invalid", "Give the dataset version a name.")
        if len(body.ratios) != 3 or any(x < 0 for x in body.ratios) or abs(sum(body.ratios) - 1) > 1e-6:
            raise ApiError(400, "invalid", "Split ratios must be three non-negative numbers that add up to 1.")
        try:
            jobs._sam_if_available()
        except Exception as exc:  # noqa: BLE001
            raise ApiError(409, "sam3_unavailable", f"SAM 3.1 cannot label here: {exc}") from None
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Another job of this mission is running. Try again when it ends.")
        job = jobs.start(m.id, kind="teach", teacher=body.teacher, author=u.username,
                         dataset={"name": body.name.strip(), "ratios": list(body.ratios), "block_days": body.block_days,
                                  "seed": body.seed})
        return {"teaching": True, "job": job.snapshot()}

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
        refs = _refs_of(m, body.dataset_id, body.datasets)
        if not _learn_call(lambda: ls.combine(refs, body.split)[0]):
            raise ApiError(400, "learning", f"The {body.split} split of the chosen dataset versions is empty.")
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Another job is running for this mission. Wait for it or cancel it first.")
        if bool(body.name) != bool(body.version):
            raise ApiError(400, "model", "Give both the model name and version, or neither for the model in use.")
        return jobs.start(m.id, kind="evaluate", dataset_id=body.dataset_id, datasets=refs, split=body.split,
                          name=body.name, version=body.version, iou=body.iou, author=u.username).snapshot()

    @app.post("/api/missions/{mission_id}/learning/benchmark")
    def benchmark_start(body: BenchmarkBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Compare detectors (phase 1c): every installed vehicle detector, or the chosen ones, on the same images."""
        from .. import models
        ls = _learn(m)
        refs = _refs_of(m, body.dataset_id, body.datasets)
        if not refs:
            raise ApiError(400, "learning", "Choose the dataset version to compare the detectors on.")
        if not _learn_call(lambda: ls.combine(refs, body.split)[0]):
            raise ApiError(400, "learning", f"The {body.split} split of the chosen dataset versions is empty.")
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "Another job is running for this mission. Wait for it or cancel it first.")
        have = [(x["name"], x["version"]) for x in models.installed("vehicle_detection")]
        if body.models:
            want = [tuple(x.split("/", 1)) for x in body.models if "/" in x]
            missing = [f"{a} {b}" for a, b in want if (a, b) not in have]
            if missing:
                raise ApiError(400, "model", "Not installed here: " + ", ".join(missing) + ".")
        else:
            want = have
        if len(want) < 1:
            raise ApiError(400, "model", "No vehicle detector is installed here.")
        return jobs.start(m.id, kind="benchmark", datasets=refs, split=body.split, iou=body.iou,
                          candidates=[list(x) for x in want], author=u.username).snapshot()

    @app.post("/api/missions/{mission_id}/learning/train")
    def train_start(body: TrainBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Fine-tune the vehicle detector here, on this computer's GPU or CPU. The new model is added, not promoted."""
        from .. import models, training_env
        ls = _learn(m)
        refs = _refs_of(m, body.dataset_id, body.datasets)
        items, _src = _learn_call(lambda: ls.combine(refs, strict=True))
        if not any(i["split"] == "train" for i in items):
            raise ApiError(400, "learning", "The chosen dataset versions have no training images.")
        env = training_env.status(wait=True)
        if env["state"] != "ready":
            raise ApiError(400, "no_training", env.get("reason") or "Training is not available here.")
        start = body.start_from or "coco"
        detr = env.get("detr") or {}
        arch = "rf-detr" if body.arch in ("rf-detr", "rfdetr") else body.arch
        if arch not in ("yolox", "rf-detr", *detr):
            raise ApiError(400, "invalid", "The model to train is YOLOX (yolox), RF-DETR (rf-detr), D-FINE (d-fine) or "
                           "RT-DETRv2 (rt-detr).")
        # the size this computer suits (training_env.recommended), unless the operator chose one
        rec = (env.get("recommended") or {}).get("sizes") or {}
        size = body.size or rec.get(arch) or (detr[arch]["default"] if arch in detr else
                                              {"yolox": "s", "rf-detr": "small"}[arch])
        if arch == "rf-detr" and not env.get("rfdetr"):
            raise ApiError(400, "no_training", env.get("rfdetr_reason") or "RF-DETR cannot be trained here.")
        if arch in detr and not detr[arch]["ok"]:
            raise ApiError(400, "no_training", detr[arch]["reason"] or f"{detr[arch]['title']} cannot be trained here.")
        if start == "coco":
            if arch == "yolox" and size not in (env.get("yolox_weights") or (["s"] if env.get("coco_weights") else [])):
                raise ApiError(400, "no_weights", f"The YOLOX-{size} starting weights are not installed here. Install the "
                               "current LanePulse version, or start from a model trained earlier.")
            if arch == "rf-detr" and size not in (env.get("rfdetr_weights") or []):
                raise ApiError(400, "no_weights", f"The RF-DETR {size} starting weights are not installed here. Install the "
                               "current LanePulse version, or start from an RF-DETR model trained earlier.")
            if arch in detr and size not in detr[arch]["weights"]:
                raise ApiError(400, "no_weights", f"The {detr[arch]['title']} {size} starting weights are not installed "
                               f"here. Install the current LanePulse version, or start from a {detr[arch]['title']} "
                               "model trained earlier.")
            init = "coco"
        else:
            name, _, version = start.partition("/")
            try:
                pack = models.find(name, version or None)
            except models.ModelPackError as exc:
                raise ApiError(400, "model", str(exc)) from None
            if not (pack.card.get("torch_weights") and (pack.path / pack.card["torch_weights"]).is_file()):
                raise ApiError(400, "model", f"{name} {pack.version} cannot be trained further (no PyTorch weights). "
                               "Start from the released weights instead.")
            init = str(pack.path)
            # training continues the model's own kind
            arch = "rf-detr" if is_rfdetr(pack.card) else (pack.card.get("architecture")
                                                           if pack.card.get("architecture") in detr else "yolox")
            size = pack.card.get("size") or ("s" if arch == "yolox" else size)
            if arch == "rf-detr" and not env.get("rfdetr"):
                raise ApiError(400, "no_training", env.get("rfdetr_reason") or "RF-DETR cannot be trained here.")
            if arch in detr and not detr[arch]["ok"]:
                raise ApiError(400, "no_training", detr[arch]["reason"] or f"{detr[arch]['title']} cannot be trained here.")
        other = jobs.training_busy()
        if other:
            raise ApiError(409, "training_busy", "A model is already training on this computer"
                           + ("" if other.mission_id == m.id else " (for another mission)") + ". Wait for it to finish.")
        work = Path(m.config().output_dir) / ".training"
        try:
            work.mkdir(parents=True, exist_ok=True)
        except OSError:
            work = None
        if start == "coco":
            # The released model of that kind and size, when installed, is what the new model is compared with afterwards.
            try:
                rel = models.find(f"{arch}-{size}")
                start = f"{rel.name}/{rel.version}"
            except models.ModelPackError:
                pass
        return jobs.start(m.id, kind="train", dataset_id=body.dataset_id, datasets=refs, epochs=body.epochs, init=init,
                          start_from=start, author=u.username, work_dir=str(work) if work else None, arch=arch,
                          size=size).snapshot()

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

    def _refs_of(m: Mission, dataset_id: int | None, datasets) -> list[dict]:
        refs = [{"mission_id": d.mission_id, "dataset_id": d.dataset_id} for d in datasets or []]
        if not refs and dataset_id is not None:
            refs = [{"mission_id": m.id, "dataset_id": dataset_id}]
        if not refs:
            raise ApiError(400, "learning", "Choose at least one dataset version.")
        return refs

    # ---------- saved models ----------
    @app.get("/api/models/saved")
    def saved_models(_u: User = Depends(user)):
        from .. import models
        from ..learning import ModelLibrary
        rows = ModelLibrary(data_root).saved()
        cards = {(c["name"], c["version"]): c for c in models.installed("vehicle_detection")}
        for r in rows:                      # which detector choice the model belongs to (YOLOX-s or RF-DETR)
            r["rfdetr"] = is_rfdetr(cards.get((r["model_name"], r["model_version"])) or {"name": r["model_name"]})
        return {"saved": rows}

    @app.post("/api/models/saved")
    def saved_model_add(body: SaveModelBody, u: User = Depends(user)):
        """Keep a model by a name of your own, frozen, so any mission can use it."""
        from ..learning import LearningService, ModelLibrary
        summary: dict[str, Any] = {}
        if body.mission_id and body.evaluation_id:
            try:
                ev = LearningService(DataRoot(data_root).open_mission(body.mission_id)).evaluation(body.evaluation_id)
            except KeyError:
                raise ApiError(404, "no_mission", "That mission does not exist.") from None
            md = (ev["model"] or {}).get("model") or {}
            if (md.get("name"), md.get("version")) == (body.model_name, body.model_version) and ev["report"]:
                summary = {"overall": ev["report"]["overall"], "split": ev["split"], "sources": ev["sources"]}
        from .. import models
        card = next((x for x in models.installed() if (x["name"], x["version"]) == (body.model_name, body.model_version)), None)
        if card:
            summary["trained_on"] = card.get("trained_on")
        return _learn_call(lambda: ModelLibrary(data_root).save(body.name, body.model_name, body.model_version, u.username,
                                                                 body.description, body.mission_id, body.evaluation_id,
                                                                 summary))

    @app.delete("/api/models/saved/{name}")
    def saved_model_remove(name: str, u: User = Depends(user)):
        from ..learning import ModelLibrary
        _admin(u)
        _learn_call(lambda: ModelLibrary(data_root).remove(name))
        return {"removed": name}

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

    # ---------- detector sizes for this computer ----------
    def _hardware() -> dict:
        from .. import hardware, training_env
        env = training_env.status()
        tr = None
        if env.get("state") == "ready":
            tr = {"yolox": env.get("yolox_weights") or [], "rf-detr": env.get("rfdetr_weights") or [],
                  **{a: v.get("weights") or [] for a, v in (env.get("detr") or {}).items()}}
        out = hardware.summary(data_root, trainable=tr)
        out["training"] = env.get("state")
        return out

    @app.get("/api/hardware")
    def hardware_get(_u: User = Depends(user)):
        """What this computer has, and per detector the size it runs and trains (chosen from that, or by the operator)."""
        return _hardware()

    @app.put("/api/hardware/sizes")
    def hardware_size(body: SizeBody, u: User = Depends(user)):
        from .. import hardware
        _admin_only(u)
        try:
            hardware.set_choice(data_root, body.family, body.size)
        except ValueError as exc:
            raise ApiError(400, "invalid", str(exc)) from None
        jobs.reset_detector()                         # the next job loads the newly chosen size
        return _hardware()

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

    # ---------- video catalog ----------
    def _catalog(m: Mission):
        from ..video_catalog import VideoCatalog
        if m.input_type != "video":
            raise ApiError(400, "not_video", "The catalog lists the videos of a video mission.")
        return VideoCatalog(m)

    @app.get("/api/missions/{mission_id}/catalog")
    def catalog_list(refresh: bool = True, m: Mission = Depends(mission)):
        """Every video in the source folders with its details; looks for new files first unless a job is running."""
        cat = _catalog(m)
        if refresh and not jobs.busy(m.id):
            cat.refresh(stability_seconds=stability_seconds)
        return cat.overview() | {"busy": jobs.busy(m.id)}

    @app.post("/api/missions/{mission_id}/catalog/merge")
    def catalog_merge(body: AnalyseBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Merge videos that are parts of one recording: one job, a target crossing a cut counted once, one tab."""
        from ..video_catalog import CatalogError
        try:
            return _catalog(m).merge(body.version_ids, u.username)
        except CatalogError as exc:
            raise ApiError(400, "catalog", str(exc)) from None

    @app.delete("/api/missions/{mission_id}/catalog/groups/{group_id}")
    def catalog_separate(group_id: int, m: Mission = Depends(mission)):
        """Separate a merged recording into its videos again (their analyses stay)."""
        from ..video_catalog import CatalogError
        try:
            _catalog(m).separate(group_id)
        except CatalogError as exc:
            raise ApiError(400, "catalog", str(exc)) from None
        return {"ok": True}

    @app.post("/api/missions/{mission_id}/catalog/analyse")
    def catalog_analyse(body: AnalyseBody, m: Mission = Depends(mission)):
        """Analyse exactly the ticked videos, in one job."""
        from ..video_catalog import CatalogError
        if jobs.busy(m.id):
            raise ApiError(409, "busy", "A job is running for this mission. Wait for it or cancel it first.")
        cat = _catalog(m)
        try:
            ids = cat.queue(body.version_ids)
        except CatalogError as exc:
            raise ApiError(400, "catalog", str(exc)) from None
        return jobs.start(m.id, kind="process", only=ids).snapshot()

    # ---------- workers on other computers (phase 5) ----------
    import socket
    from ..workers import Pairing, Requests

    def _lan_address() -> list[str]:
        import socket
        out = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.connect(("10.255.255.255", 1))            # no packet is sent; this picks the outgoing address
                out.append(sk.getsockname()[0])
        except OSError:
            pass
        return out

    @app.get("/api/workers")
    def workers_status():
        """The switch, the pairing codes, the computers asking to help, those helping and the segments waiting."""
        p = Pairing(data_root)
        return jobs.hub.status() | {"codes": p.codes(), "asking": Requests(data_root).asking(),
                                    "lan": listening.get("lan", False), "port": listening.get("port", 8765),
                                    "addresses": _lan_address(),
                                    "discovery": None if discovery is None else (discovery.error or "on")}

    @app.put("/api/workers/asking/{request_id}")
    def workers_answer(request_id: int, body: WorkerAnswerBody, u: User = Depends(user)):
        """Allow or refuse a computer that asks to help."""
        try:
            Requests(data_root).decide(request_id, body.allow, u.username)
        except KeyError:
            raise ApiError(404, "no_request", "That computer is no longer asking.") from None
        return workers_status()

    @app.post("/api/worker/hello")
    def worker_hello(body: WorkerHello, request: Request):
        """A computer that found this server asks to help, with its own key. It helps once the operator allows it."""
        r = Requests(data_root)
        state = r.ask(body.key, body.name, body.info, request.client.host if request.client else None)
        if state == "allowed" and not jobs.hub.enabled:
            state = "off"
        return {"state": state, "server_id": r.server_id(), "name": socket.gethostname()}

    @app.put("/api/workers/switch")
    def workers_switch(body: WorkerSwitchBody, u: User = Depends(user)):
        Pairing(data_root).set_enabled(body.enabled, u.username)
        jobs.hub.enabled = body.enabled
        return workers_status()

    @app.post("/api/workers/codes")
    def workers_code(body: WorkerCodeBody, u: User = Depends(user)):
        """A new pairing code. It is shown once; only its hash is kept."""
        return Pairing(data_root).new_code(body.label, u.username)

    @app.delete("/api/workers/codes/{code_id}")
    def workers_revoke(code_id: int):
        Pairing(data_root).revoke(code_id)
        jobs.hub.forget_code(code_id)
        return {"ok": True}

    def _worker(code: str | None) -> int:
        if not jobs.hub.enabled:
            raise ApiError(403, "workers_off", "This server does not take help from other computers now. Turn it on "
                                               "in Settings, Other computers.")
        cid = Pairing(data_root).check(code)
        if cid is None:
            raise ApiError(401, "bad_code", "That pairing code is not valid (or was revoked). Make a new one on the "
                                            "server's Workers screen.")
        return cid

    def _own_lease(lease_id: str, cid: int):
        ls = jobs.hub.lease(lease_id)
        if ls is None:
            return None
        h = jobs.hub.helpers.get(ls.worker)
        if not ls.remote or h is None or h.code_id != cid:
            raise ApiError(403, "not_yours", "That segment was given to another worker.")
        return ls

    @app.post("/api/worker/next")
    def worker_next(body: WorkerCall):
        cid = _worker(body.code)
        jobs.hub.seen(body.name, cid, body.info)
        ls = jobs.hub.take(body.name, remote=True, can=set(body.can))
        if ls is None:
            return Response(status_code=204)
        inf = float("inf")
        s = ls.seg
        return {"lease": ls.id, "mission_id": ls.job.mission_id, "version_id": ls.job.version_id,
                "segment": {"idx": s.idx, "own_a": s.own_a, "own_b": None if s.own_b == inf else s.own_b,
                            "run_a": s.run_a, "run_b": None if s.run_b == inf else s.run_b},
                "spec": ls.job.spec, "lease_s": jobs.hub.lease_s}

    @app.get("/api/worker/lease/{lease_id}/video")
    def worker_video(lease_id: str, code: str):
        """The recording of a leased segment; the worker reads only the range it needs (HTTP range requests)."""
        cid = _worker(code)
        ls = jobs.hub.lease(lease_id)
        if ls is None or (jobs.hub.helpers.get(ls.worker) and jobs.hub.helpers[ls.worker].code_id != cid):
            raise ApiError(410, "lease_ended", "That segment is no longer yours.")
        m = DataRoot(data_root).open_mission(ls.job.mission_id)
        path = _video(m).path(ls.job.version_id)
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.post("/api/worker/lease/{lease_id}/renew")
    def worker_renew(lease_id: str, body: WorkerCall):
        cid = _worker(body.code)
        return {"ok": _own_lease(lease_id, cid) is not None and jobs.hub.renew(lease_id)}

    @app.post("/api/worker/lease/{lease_id}/result")
    def worker_result(lease_id: str, body: WorkerCall):
        from ..video_process import _result
        cid = _worker(body.code)
        if _own_lease(lease_id, cid) is None or body.result is None:
            return {"ok": False, "reason": "the lease ended; the segment went to another worker"}
        try:
            res = _result(body.result)
        except (KeyError, TypeError, ValueError) as exc:
            jobs.hub.give_back(lease_id, f"bad result: {exc}")
            raise ApiError(400, "bad_result", "The segment's result could not be read.") from None
        return {"ok": jobs.hub.complete(lease_id, res)}

    @app.post("/api/worker/lease/{lease_id}/give-back")
    def worker_give_back(lease_id: str, body: WorkerCall):
        cid = _worker(body.code)
        if _own_lease(lease_id, cid) is not None:
            jobs.hub.give_back(lease_id, body.error)
        return {"ok": True}

    @app.get("/api/worker/model/{sha}")
    def worker_model(sha: str, code: str):
        """A model pack an open job uses, packed for a worker that does not have it (fetched once)."""
        import io
        import tarfile
        _worker(code)
        path = jobs.hub_packs.get(sha)
        if path is None or not path.is_dir():
            raise ApiError(404, "no_model", "No open job uses that model.")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for f in sorted(path.iterdir()):
                if f.is_file():
                    t.add(f, arcname=f.name)
        return Response(buf.getvalue(), media_type="application/gzip")

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/thumb")
    def recording_thumb(version_id: int, h: int = 96, m: Mission = Depends(mission)):
        from ..video_catalog import CatalogError
        cat = _catalog(m)
        try:
            data = cat.thumbnail(version_id, max(48, min(int(h), 360)))
        except (CatalogError, TypeError) as exc:
            raise ApiError(404, "no_thumb", str(exc)) from None
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/missions/{mission_id}/recordings")
    def recordings(m: Mission = Depends(mission)):
        from ..learning import LearningService
        from ..video_groups import Groups
        ls = LearningService(m)
        return {"recordings": [r | {"review": ls.review_kind(r["id"]), "approved": ls.recording_approved(r["id"]),
                                    "pending": sum(1 for e in ls._events(r["id"]) if e.status == "pending")}
                               for r in _video(m).recordings()],
                "groups": Groups(m).all()}

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
               flag: str | None = None, group_id: int | None = None):
        svc = _video(m)
        evs = _video_call(lambda: svc.events(version_id, status, flag, group_id=group_id))
        attrs = svc.attributes(evs)
        reading = False
        if any(a["colour"] is None for a in attrs.values()):
            # Colours are read once per target, one frame each, in the background; the list fills in as they come.
            from ..video_attributes import read_colours_soon
            from ..video_review import VideoService
            reading = read_colours_soon(m.id, lambda: VideoService(DataRoot(data_root).open_mission(m.id)))
        from ..video_zones import tally
        zt = tally(svc, evs)
        zone_names = {z["uid"]: z["name"] for r in zt["recordings"] for z in r["zones"]}
        for e in evs:
            p = zt["events"].get(e.id)
            attrs[e.id]["path"] = p["path"] if p else []
            # per zone: a line's crossing directions in order, or "inside" for an area it entered
            attrs[e.id]["zones"] = {u: ([h["dir"] for h in hits] if "dir" in hits[0] else ["inside"])
                                    for u, hits in (p["zones"] if p else {}).items()}
        return {"events": [_event_dict(e) | {"attrs": attrs[e.id]} for e in evs], "classes": svc.classes(),
                "reading_colours": reading, "zones": zone_names}

    @app.get("/api/missions/{mission_id}/zones")
    def zones_tally(m: Mission = Depends(mission), version_id: int | None = None, group_id: int | None = None):
        """Each line's crossings (in and out) and each area's entries and time inside, by class, per video."""
        from ..video_zones import tally
        svc = _video(m)
        evs = _video_call(lambda: [e for e in svc.events(version_id, group_id=group_id) if e.continues is None])
        out = tally(svc, evs)
        out.pop("events")
        return out

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/map")
    def recording_map(version_id: int, body: MapBody, m: Mission = Depends(mission)):
        """A heatmap (where targets spent time) or a path map of one video, over its empty road, as a JPEG data URL.
        ``event_ids`` narrows it to the events the events page shows; otherwise every counted event."""
        import base64
        from ..video_maps import render
        from ..video_zones import Zones
        svc = _video(m)
        evs = [e for e in _video_call(lambda: svc.events(version_id)) if e.counted]
        if body.event_ids is not None:
            keep = set(body.event_ids)
            evs = [e for e in evs if e.id in keep]
        try:
            data, legend = render(svc, version_id, body.kind, evs, Zones(m).of(version_id) if body.zones else [])
        except ValueError as exc:
            raise ApiError(400, "map", str(exc)) from None
        return {"image": "data:image/jpeg;base64," + base64.b64encode(data).decode(), "legend": legend}

    from ..video_synopsis import Synopses
    synopses = Synopses()

    def _synopsis_args(m: Mission, version_id: int, body: "SynopsisBody"):
        svc = _video(m)
        evs = [e for e in _video_call(lambda: svc.events(version_id)) if e.counted]
        if body.event_ids is not None:
            keep = set(body.event_ids)
            evs = [e for e in evs if e.id in keep]
        rid = max((e.run_item_id or 0 for e in evs), default=0)
        revs = sum(e.revisions for e in evs)
        key = Synopses.key(version_id, rid, [e.id for e in evs], body.busy, revs)
        return svc, evs, key, m.dir / "evidence" / "synopsis"

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/synopsis")
    def synopsis_start(version_id: int, body: SynopsisBody, m: Mission = Depends(mission)):
        """Make (or find) the summary video of these events: the targets together over the empty road, each with
        its real time. Made in the background; ask again (same body) for its progress."""
        from ..video_review import VideoService
        from ..video_synopsis import render
        svc, evs, key, folder = _synopsis_args(m, version_id, body)
        if not evs:
            raise ApiError(400, "synopsis", "No counted event to show in a summary.")
        if body.status_only:
            return synopses.status(m.id, key, folder)
        ids = [e.id for e in evs]
        mid = m.id

        def work(out, progress, stop):
            s2 = VideoService(DataRoot(data_root).open_mission(mid))
            keep = set(ids)
            ev2 = [e for e in s2.events(version_id) if e.id in keep]
            return render(s2, version_id, ev2, out, body.busy, progress, stop)
        return synopses.start(m.id, folder, key, work)

    @app.get("/api/missions/{mission_id}/synopsis/{key}.mp4")
    def synopsis_file(key: str, m: Mission = Depends(mission)):
        if not key.isalnum():
            raise ApiError(404, "synopsis", "No such summary.")
        path = m.dir / "evidence" / "synopsis" / f"{key}.mp4"
        if not path.is_file():
            raise ApiError(404, "synopsis", "That summary is not made (yet).")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-transform"})

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/zones")
    def zones_get(version_id: int, m: Mission = Depends(mission)):
        from ..video_zones import AREA_NAMES, LINE_NAMES, ZoneError, Zones
        _video(m)
        try:
            return {"zones": Zones(m).of(version_id), "line_names": LINE_NAMES, "area_names": AREA_NAMES}
        except ZoneError as exc:
            raise ApiError(404, "zone", str(exc)) from None

    # ---------- a fixed camera on the map (video_geo) ----------
    def _geo_call(fn):
        from ..video_geo import GeoError
        try:
            return fn()
        except GeoError as exc:
            raise ApiError(400, "geo", str(exc)) from None

    @app.get("/api/map-files")
    def map_files_get(folder: str, _u: User = Depends(user)):
        """Map pictures in a folder on this computer: GeoTIFFs, and JPEG or PNG pictures (they need their corners)."""
        from ..video_geo import map_files
        return {"files": _geo_call(lambda: map_files(folder))}

    @app.get("/api/map-info")
    def map_info_get(path: str, corners: str | None = None, _u: User = Depends(user)):
        """Where a map file is: its corners in latitude and longitude, its size and metres per pixel."""
        from ..video_geo import load_map, map_info
        cs = [float(v) for v in corners.split(",")] if corners else None
        return _geo_call(lambda: map_info(load_map(path, cs)))

    @app.get("/api/map-image")
    def map_image_get(path: str, corners: str | None = None, _u: User = Depends(user)):
        """The map file as a JPEG (at most 4096 pixels a side), for the browser to draw on."""
        from ..video_geo import load_map, preview
        cs = [float(v) for v in corners.split(",")] if corners else None
        data = _geo_call(lambda: preview(load_map(path, cs)))
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/geo")
    def geo_get(version_id: int, m: Mission = Depends(mission)):
        """This camera on the map (points, map file and fit), or null."""
        from ..video_geo import CameraGeo
        _video(m)
        return {"geo": _geo_call(lambda: CameraGeo(m).get(version_id))}

    @app.put("/api/missions/{mission_id}/recordings/{version_id}/geo")
    def geo_set(version_id: int, body: GeoBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Put the camera on the map from points picked on a frame and on the map (``check_only``: fit, do not save)."""
        from ..video_geo import CameraGeo
        _video(m)
        cg = CameraGeo(m)
        if body.check_only:
            return {"fit": _geo_call(lambda: cg.check(body.points))}
        return {"geo": _geo_call(lambda: cg.set(version_id, body.points, body.map_path, body.corners, u.username))}

    @app.delete("/api/missions/{mission_id}/recordings/{version_id}/geo")
    def geo_clear(version_id: int, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..video_geo import CameraGeo
        _video(m)
        _geo_call(lambda: CameraGeo(m).clear(version_id, u.username))
        return {"geo": None}

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/map-paths")
    def geo_paths(version_id: int, m: Mission = Depends(mission)):
        """Each counted target of this video as a line on the map (latitude and longitude, two points a second)."""
        from ..video_geo import CameraGeo
        svc = _video(m)
        return {"paths": _geo_call(lambda: _video_call(lambda: CameraGeo(m).paths(svc, version_id)))}

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/telemetry")
    def telemetry_get(version_id: int, m: Mission = Depends(mission)):
        """Where a drone or aircraft camera was and looked (DJI subtitle file or KLV), when the recording has it."""
        from ..video_meta import stored
        svc = _video(m)
        if m.con.execute("SELECT 1 FROM source_version WHERE id=?", (version_id,)).fetchone() is None:
            raise ApiError(404, "no_recording", "That recording is not in this mission.")
        try:
            path = svc.path(version_id)
        except Exception:  # noqa: BLE001 - the file has gone: what was read before is still shown
            path = None
        tel = stored(m.con, version_id, path)
        if tel is None:
            return {"source": None}
        return {"source": tel.source, "summary": tel.summary(), "samples": tel.samples}

    # ---------- targets the operator marks and the app follows (learning a target SAM 3.1 does not know) ----------
    def _marks(m: Mission):
        from ..follow import MarkedTargets
        svc = _video(m)
        return MarkedTargets(m, svc.path)

    def _mark_call(fn):
        from ..follow import MarkError
        try:
            return fn()
        except MarkError as exc:
            raise ApiError(400, "mark", str(exc)) from None

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/marks")
    def marks_list(version_id: int, m: Mission = Depends(mission)):
        return {"targets": _marks(m).list(version_id)}

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/marks")
    def marks_add(version_id: int, body: MarkBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """A box around a target on one frame: the app follows it both ways and suggests where it is in view."""
        mt = _marks(m)
        if m.con.execute("SELECT 1 FROM source_version WHERE id=?", (version_id,)).fetchone() is None:
            raise ApiError(404, "no_recording", "That recording is not in this mission.")
        out = _mark_call(lambda: mt.add(version_id, body.cls, body.t, body.box, u.username))
        own = getattr(m.config(), "sam3_words", None) or {}
        from ..detect import VEHICLE_CLASSES
        name = out["cls"]
        if name not in VEHICLE_CLASSES and name != "person" and name not in own:
            # a new kind of target: the mission lists it (SAM 3.1 is asked for it by name too)
            try:
                m.update_config(u.username, f"class {name} added by marking a target",
                                sam3_words={**own, name: [name]})
            except Exception:  # noqa: BLE001 - a name SAM 3.1 cannot take still works for marking and training
                pass
        return out

    @app.post("/api/missions/{mission_id}/marks/{target_id}/correct")
    def marks_correct(target_id: int, body: MarkCorrectBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Another box of the same target where following went wrong or lost it: followed again from every box."""
        return _mark_call(lambda: _marks(m).correct(target_id, body.t, body.box, u.username))

    @app.put("/api/missions/{mission_id}/marks/{target_id}")
    def marks_update(target_id: int, body: MarkUpdateBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Set where the target is in view (start, end), confirm it for training, or rename what it is."""
        out = _mark_call(lambda: _marks(m).update(target_id, u.username, body.start_t, body.end_t, body.state,
                                                  body.cls))
        if body.state == "confirmed":
            _auto_soon(m.id)
        return out

    @app.delete("/api/missions/{mission_id}/marks/{target_id}")
    def marks_delete(target_id: int, m: Mission = Depends(mission), u: User = Depends(user)):
        _mark_call(lambda: _marks(m).remove(target_id, u.username))
        return {"ok": True}

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/zones")
    def zones_add(version_id: int, body: ZoneBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """A counting line or an area on this camera's picture (every video in the same folder uses it)."""
        from ..video_zones import ZoneError, Zones
        _video(m)
        try:
            return Zones(m).add(version_id, body.kind, body.name, body.points, u.username)
        except ZoneError as exc:
            raise ApiError(400, "zone", str(exc)) from None

    @app.delete("/api/missions/{mission_id}/zones/{uid}")
    def zones_delete(uid: str, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..video_zones import ZoneError, Zones
        _video(m)
        try:
            Zones(m).delete(uid, u.username)
        except ZoneError as exc:
            raise ApiError(404, "zone", str(exc)) from None
        return {"ok": True}

    @app.get("/api/missions/{mission_id}/in-view")
    def in_view(m: Mission = Depends(mission), version_id: int | None = None, group_id: int | None = None):
        """The most targets in view at once, per family, and the crowd periods (the mission's crowd alert)."""
        svc = _video(m)
        return _video_call(lambda: svc.in_view(version_id, group_id))

    @app.put("/api/missions/{mission_id}/crowd-alert")
    def crowd_alert(body: CrowdAlertBody, m: Mission = Depends(mission), u: User = Depends(user)):
        _video(m)
        m.update_config(u.username, f"crowd alert at {body.people}" if body.people else "crowd alert off",
                        crowd_alert=body.people)
        return {"crowd_alert": body.people}

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/calibration")
    def calibration_get(version_id: int, m: Mission = Depends(mission)):
        from ..video_attributes import DISTANCES_M, Calibration
        _video(m)
        return {"calibration": Calibration(m).get(version_id), "distances_m": DISTANCES_M}

    @app.put("/api/missions/{mission_id}/recordings/{version_id}/calibration")
    def calibration_set(version_id: int, body: CalibrationBody, m: Mission = Depends(mission), u: User = Depends(user)):
        """Measure the camera: two points on the picture and the real distance between them."""
        from ..video_attributes import Calibration
        _video(m)
        try:
            return {"calibration": Calibration(m).set(version_id, body.points, body.meters, u.username)}
        except ValueError as exc:
            raise ApiError(400, "calibration", str(exc)) from None

    @app.delete("/api/missions/{mission_id}/recordings/{version_id}/calibration")
    def calibration_clear(version_id: int, m: Mission = Depends(mission), u: User = Depends(user)):
        from ..video_attributes import Calibration
        _video(m)
        Calibration(m).clear(version_id, u.username)
        return {"calibration": Calibration(m).get(version_id)}

    @app.get("/api/missions/{mission_id}/counts")
    def counts(m: Mission = Depends(mission), version_id: int | None = None,
               bucket_s: int = Query(60, ge=10, le=3600), group_id: int | None = None):
        """Unique targets per class (one per track), per recording and for the mission, before and after review."""
        svc = _video(m)
        return _video_call(lambda: svc.counts(version_id, bucket_s, group_id=group_id))

    @app.get("/api/missions/{mission_id}/events/{event_id}")
    def event_get(event_id: int, m: Mission = Depends(mission)):
        svc = _video(m)
        e = _video_call(lambda: svc.event(event_id))
        return _event_dict(e) | {"observations": svc.observations(event_id), "clips": svc.clips(event_id),
                                 "attrs": svc.attributes([e])[e.id]}

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
        if getattr(m.config(), "run_mode", "manual") == "auto":
            _auto_soon(m.id)
        return _event_dict(e)

    @app.get("/api/missions/{mission_id}/recordings/{version_id}/overlay")
    def recording_overlay(version_id: int, m: Mission = Depends(mission)):
        """The player's data: every event of the recording's newest analysis with its boxes by time."""
        svc = _video(m)
        out = _video_call(lambda: svc.overlay(version_id))
        from ..learning import LearningService
        ls = LearningService(m)
        out["recording"] |= {"review": ls.review_kind(version_id), "approved": ls.recording_approved(version_id),
                             "pending": sum(1 for e in out["events"] if e["status"] == "pending")}
        return out

    @app.post("/api/missions/{mission_id}/recordings/{version_id}/targets")
    def recording_add_target(version_id: int, body: AddTargetBody, m: Mission = Depends(mission),
                             u: User = Depends(user)):
        """A target the analysis missed, drawn on one frame: a new confirmed event of the operator's."""
        svc = _video(m)
        e = _video_call(lambda: svc.add_target(version_id, body.t, body.box, body.cls, u.username, body.note))
        if getattr(m.config(), "run_mode", "manual") == "auto":
            _auto_soon(m.id)
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
            paths = svc.export(u.username, body.formats, min_score=body.min_score)
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
    k = sub.add_parser("key", help="show the access key the browser asks for on the network (--lan)")
    k.add_argument("--data", required=True)
    k.add_argument("--new", action="store_true", help="make a new key (browsers must enter it again)")
    args = ap.parse_args(argv)
    if args.cmd == "key":
        print(access_key_for(args.data, new=args.new))
        return 0
    if args.cmd == "serve":
        host = "0.0.0.0" if args.lan else "127.0.0.1"
        uvicorn.run(create_app(args.data, backup_dir=args.backup_dir, backup_at=args.backup_at,
                               listening={"lan": args.lan, "port": args.port}),
                    host=host, port=args.port, log_level="info")
        return 0
    ap.print_help()
    return 1
