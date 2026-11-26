"""Training that belongs to no mission: one place on the computer to train and compare vehicle detectors on dataset
versions from any missions, still images and video alike, chosen freely.

The datasets themselves still come from missions (that is where images are reviewed and versions frozen). The
workspace keeps only what training makes: its training rounds, evaluations and their advice, in a database shaped
like a mission's (so the learning code runs unchanged) at <data root>/training/training.db. It holds no images,
never processes anything and never appears in the mission list. A model trained here is used by a mission only when
the operator chooses it for that mission (or several).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from pathlib import Path

from .store import (MISSION_MIGRATIONS, MISSION_SCHEMA, SCHEMA_VERSION, DataRoot, Mission, MissionConfig, _statements,
                    connect, ensure_indexes, migrate, utcnow)

WORKSPACE_ID = "training"
_create_lock = threading.Lock()


def is_workspace(mission_id: str | None) -> bool:
    return mission_id == WORKSPACE_ID


class TrainingWorkspace(Mission):
    """A Mission in shape only: its config names it "Training", it has no source folder and no images."""

    def __init__(self, root: DataRoot):            # noqa: D107 - not registered as a mission on purpose
        self.root = root
        self.id = WORKSPACE_ID
        self.dir = root.path / "training"
        db = self.dir / "training.db"
        with _create_lock:
            if not db.exists():
                self._create(db)
        self.con = connect(db)
        migrate(db, self.con)
        ensure_indexes(db, self.con)

    def _create(self, db: Path) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = MissionConfig(name="Training", input_type="images", mode="general_traffic", country="", iana_zone="UTC",
                            source_dir="", include_subdirs=False, processing="one_time",
                            output_dir=str(self.dir / "work"), export_formats=[])
        tmp = db.with_name(db.name + ".creating")
        tmp.unlink(missing_ok=True)
        con = connect(tmp)
        now = utcnow()
        con.executescript(MISSION_SCHEMA)
        con.execute("BEGIN")
        con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        con.execute("INSERT INTO mission VALUES (?,?,?,?,?,?,?,?,?)",
                    (WORKSPACE_ID, cfg.name, cfg.input_type, cfg.mode, cfg.country, cfg.iana_zone, None, None, now))
        con.execute("INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
                    (json.dumps(asdict(cfg)), "system", now, "training workspace created"))
        con.execute("COMMIT")
        for v in sorted(MISSION_MIGRATIONS):
            con.execute("BEGIN")
            for stmt in _statements(MISSION_MIGRATIONS[v]):
                con.execute(stmt)
            con.execute("INSERT INTO schema_version VALUES (?)", (v,))
            con.execute("COMMIT")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        os.replace(tmp, db)


def open_scope(root: DataRoot, mission_id: str) -> Mission:
    """The training workspace for its id, else the mission (KeyError when there is none)."""
    return TrainingWorkspace(root) if is_workspace(mission_id) else root.open_mission(mission_id)
