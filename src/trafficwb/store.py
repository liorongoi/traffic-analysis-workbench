"""Mission registry and per-mission databases.

- control.db (in the data root) lists missions.
- Each mission gets its own SQLite database, created when the mission is
  committed, before any run. Missions never share analytical tables.
- Machine and human records are append-only where the design requires history.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 1
TARGET_COUNTRIES = {"IR": "Iran", "LB": "Lebanon", "IL": "Israel", "LV": "Latvia"}
INPUT_TYPES = {"images", "video"}
MODES = {
    "images": {"general_traffic", "object_search", "plate_recognition"},
    "video": {"object_search"},
}
EXPORT_FORMATS = {"xlsx", "csv"}

CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mission (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  db_path TEXT NOT NULL,
  input_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'active'
);
"""

MISSION_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE mission (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  input_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  country TEXT NOT NULL,
  iana_zone TEXT NOT NULL,
  latitude REAL, longitude REAL,
  created_at TEXT NOT NULL
);
-- Every configuration change is a new revision; current = highest seq per kind.
CREATE TABLE config_revision (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL,
  reason TEXT
);
CREATE TABLE source_root (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  recursive INTEGER NOT NULL
);
-- Logical identity of a path under a source root.
CREATE TABLE source_file (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES source_root(id),
  rel_path TEXT NOT NULL,
  first_seen TEXT NOT NULL,
  missing_since TEXT,
  last_size INTEGER,
  last_mtime_ns INTEGER,
  UNIQUE(root_id, rel_path)
);
-- One row per distinct content observed at a path. Never deleted or overwritten.
CREATE TABLE source_version (
  id INTEGER PRIMARY KEY,
  source_file_id INTEGER NOT NULL REFERENCES source_file(id),
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  capture_utc TEXT,
  capture_precision TEXT,
  name_pattern TEXT,
  name_fields TEXT,
  parse_status TEXT NOT NULL,      -- ok | malformed
  parse_detail TEXT,
  integrity TEXT NOT NULL,         -- ok | corrupt | unsupported
  integrity_detail TEXT,
  first_seen TEXT NOT NULL,
  superseded_by INTEGER REFERENCES source_version(id),
  possible_rename_of INTEGER REFERENCES source_version(id),
  UNIQUE(source_file_id, sha256, size, mtime_ns)
);
CREATE INDEX ix_version_capture ON source_version(capture_utc);
CREATE INDEX ix_version_sha ON source_version(sha256);
-- Files seen but not yet stable (possibly still being copied).
CREATE TABLE pending_file (
  root_id INTEGER NOT NULL,
  rel_path TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  unchanged_since TEXT NOT NULL,
  PRIMARY KEY(root_id, rel_path)
);
CREATE TABLE scan (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,           -- running | completed | source_unavailable | failed
  summary TEXT
);
CREATE TABLE run (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,             -- initial | continuous_batch | reprocess
  state TEXT NOT NULL,            -- queued | running | paused | completed | cancelled | failed
  config_snapshot TEXT NOT NULL,
  model_versions TEXT NOT NULL DEFAULT '{}',
  preprocessing TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT
);
-- Processing status only; review status is stored separately.
CREATE TABLE run_item (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES run(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  processing_state TEXT NOT NULL DEFAULT 'queued',
  stage TEXT, error_code TEXT, error_detail TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  UNIQUE(run_id, source_version_id)
);
CREATE TABLE notification (
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  severity TEXT NOT NULL,         -- info | warning | error
  code TEXT NOT NULL,
  message TEXT NOT NULL,
  context TEXT,
  acknowledged INTEGER NOT NULL DEFAULT 0
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, isolation_level=None)  # explicit transactions
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=5000")
    return con


@dataclass
class MissionConfig:
    name: str
    input_type: str
    mode: str
    country: str
    iana_zone: str
    source_dir: str
    include_subdirs: bool
    processing: str  # one_time | continuous
    output_dir: str
    export_formats: list[str]
    latitude: float | None = None
    longitude: float | None = None
    monitoring_enabled: bool = field(default=False)


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate(cfg: MissionConfig, data_root: Path, min_free_bytes: int = 1 << 30) -> Validation:
    v = Validation()
    if not cfg.name.strip():
        v.errors.append("Mission name is empty.")
    if cfg.input_type not in INPUT_TYPES:
        v.errors.append(f"Input type must be one of {sorted(INPUT_TYPES)}.")
    elif cfg.mode not in MODES[cfg.input_type]:
        v.errors.append(f"Mode '{cfg.mode}' is not available for {cfg.input_type}; choose one of {sorted(MODES[cfg.input_type])}.")
    if cfg.country not in TARGET_COUNTRIES:
        v.warnings.append(
            f"Country '{cfg.country}' is outside the supported set ({', '.join(TARGET_COUNTRIES)}); "
            "plate recognition will not be available."
        )
        if cfg.mode == "plate_recognition":
            v.errors.append("Plate recognition needs one of the supported countries.")
    try:
        ZoneInfo(cfg.iana_zone)
    except (ZoneInfoNotFoundError, ValueError):
        v.errors.append(f"'{cfg.iana_zone}' is not a known IANA time zone.")
    if cfg.processing not in {"one_time", "continuous"}:
        v.errors.append("Processing must be one_time or continuous.")
    if not cfg.export_formats or not set(cfg.export_formats) <= EXPORT_FORMATS:
        v.errors.append("Export formats must be xlsx, csv, or both.")

    src = Path(cfg.source_dir)
    if not src.is_dir():
        v.errors.append(f"Source folder {src} does not exist or is not a folder on this workstation.")
    elif not os.access(src, os.R_OK | os.X_OK):
        v.errors.append(f"Source folder {src} cannot be read by the application account.")

    out = Path(cfg.output_dir)
    if not out.is_dir():
        v.errors.append(f"Output folder {out} does not exist.")
    else:
        try:
            with tempfile.NamedTemporaryFile(dir=out, prefix=".trafficwb-write-test-"):
                pass
        except OSError as exc:
            v.errors.append(f"Output folder {out} is not writable: {exc.strerror or exc}.")
        free = shutil.disk_usage(out).free
        if free < min_free_bytes:
            v.warnings.append(f"Only {free // (1 << 20)} MiB free on the output volume.")
    if src.is_dir() and out.is_dir() and _inside(out, src) and cfg.include_subdirs:
        v.errors.append("Output folder is inside the source folder and subfolders are included; exports would be re-ingested.")
    if src.is_dir() and _inside(data_root, src) and cfg.include_subdirs:
        v.errors.append("The application data folder is inside the source folder; databases would be re-ingested.")
    return v


class DataRoot:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "missions").mkdir(exist_ok=True)
        self.con = connect(self.path / "control.db")
        self.con.executescript(CONTROL_SCHEMA)
        if self.con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            self.con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))

    def missions(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM mission ORDER BY created_at").fetchall()

    def create_mission(self, cfg: MissionConfig, author: str) -> "Mission":
        """Validate and commit a mission: its database exists before any run."""
        val = validate(cfg, self.path)
        if not val.ok:
            raise MissionValidationError(val)
        if self.con.execute("SELECT 1 FROM mission WHERE name=?", (cfg.name,)).fetchone():
            raise MissionValidationError(Validation(errors=[f"A mission named '{cfg.name}' already exists."]))
        mid = str(uuid.uuid4())
        mdir = self.path / "missions" / mid
        mdir.mkdir()
        tmp = mdir / "mission.db.creating"
        con = connect(tmp)
        now = utcnow()
        con.executescript(MISSION_SCHEMA)
        con.execute("BEGIN")
        con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        con.execute(
            "INSERT INTO mission VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, cfg.name, cfg.input_type, cfg.mode, cfg.country, cfg.iana_zone, cfg.latitude, cfg.longitude, now),
        )
        con.execute(
            "INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
            (json.dumps(asdict(cfg)), author, now, "mission created"),
        )
        con.execute(
            "INSERT INTO source_root(path,recursive) VALUES (?,?)",
            (str(Path(cfg.source_dir).resolve()), int(cfg.include_subdirs)),
        )
        con.execute("COMMIT")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        final = mdir / "mission.db"
        os.replace(tmp, final)
        self.con.execute(
            "INSERT INTO mission(id,name,db_path,input_type,mode,created_at) VALUES (?,?,?,?,?,?)",
            (mid, cfg.name, str(final.relative_to(self.path)), cfg.input_type, cfg.mode, now),
        )
        return Mission(self, mid)

    def open_mission(self, mission_id: str) -> "Mission":
        return Mission(self, mission_id)


class MissionValidationError(ValueError):
    def __init__(self, validation: Validation):
        super().__init__("; ".join(validation.errors))
        self.validation = validation


class Mission:
    def __init__(self, root: DataRoot, mission_id: str):
        row = root.con.execute("SELECT * FROM mission WHERE id=?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(mission_id)
        self.root = root
        self.id = mission_id
        self.dir = root.path / "missions" / mission_id
        self.con = connect(root.path / row["db_path"])

    # --- configuration ---
    def config(self) -> MissionConfig:
        row = self.con.execute(
            "SELECT payload FROM config_revision WHERE kind='mission' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return MissionConfig(**json.loads(row["payload"]))

    def update_config(self, author: str, reason: str, **changes: Any) -> MissionConfig:
        cfg = self.config()
        for k, val in changes.items():
            if k in ("input_type", "mode"):
                raise ValueError("A mission's input type and mode cannot change after creation.")
            if not hasattr(cfg, k):
                raise AttributeError(k)
            setattr(cfg, k, val)
        self.con.execute(
            "INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
            (json.dumps(asdict(cfg)), author, utcnow(), reason),
        )
        return cfg

    @property
    def zone(self) -> str:
        return self.con.execute("SELECT iana_zone FROM mission").fetchone()[0]

    @property
    def input_type(self) -> str:
        return self.con.execute("SELECT input_type FROM mission").fetchone()[0]

    def notify(self, severity: str, code: str, message: str, **context: Any) -> None:
        self.con.execute(
            "INSERT INTO notification(created_at,severity,code,message,context) VALUES (?,?,?,?,?)",
            (utcnow(), severity, code, message, json.dumps(context, ensure_ascii=False)),
        )

    # --- runs ---
    def create_run(self, kind: str, version_ids: list[int]) -> int | None:
        """Create a run over the given source versions. Returns None if there is nothing to process."""
        if not version_ids:
            return None
        self.con.execute("BEGIN")
        cur = self.con.execute(
            "INSERT INTO run(kind,state,config_snapshot,created_at) VALUES (?,?,?,?)",
            (kind, "queued", json.dumps(asdict(self.config())), utcnow()),
        )
        run_id = cur.lastrowid
        self.con.executemany(
            "INSERT INTO run_item(run_id,source_version_id) VALUES (?,?)", [(run_id, v) for v in version_ids]
        )
        self.con.execute("COMMIT")
        return run_id

    def status(self) -> dict[str, Any]:
        q = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "source_files": q("SELECT COUNT(*) FROM source_file"),
            "source_versions": q("SELECT COUNT(*) FROM source_version"),
            "malformed": q("SELECT COUNT(*) FROM source_version WHERE parse_status='malformed' AND superseded_by IS NULL"),
            "corrupt": q("SELECT COUNT(*) FROM source_version WHERE integrity!='ok' AND superseded_by IS NULL"),
            "pending_stability": q("SELECT COUNT(*) FROM pending_file"),
            "missing": q("SELECT COUNT(*) FROM source_file WHERE missing_since IS NOT NULL"),
            "runs": q("SELECT COUNT(*) FROM run"),
            "queued_items": q("SELECT COUNT(*) FROM run_item WHERE processing_state='queued'"),
            "unacknowledged_notifications": q("SELECT COUNT(*) FROM notification WHERE acknowledged=0"),
        }
