import json
import sqlite3

import pytest

from test_detect_review import FakeDetector, populate
from trafficwb.admin import AdminError, audit_log, backup, delete_mission, restore, verify
from trafficwb.ingest import cycle
from trafficwb.process import Processor
from trafficwb.store import MISSION_MIGRATIONS, DataRoot, MissionConfig


@pytest.fixture
def env(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(
        name="Route 1 – בדיקה", input_type="images", mode="general_traffic", country="IL", iana_zone="Asia/Jerusalem",
        source_dir=str(src), include_subdirs=False, processing="one_time", output_dir=str(out),
        export_formats=["csv"]), "test")
    populate(src, 3)
    cycle(m)
    Processor(m, FakeDetector()).process_all()
    return tmp_path, root, m, src


def test_backup_verify_restore(env):
    tmp, root, m, _ = env
    b = backup(root.path, tmp / "backups", "lior")
    assert verify(b) == [] and not list((tmp / "backups").glob("*.partial"))
    man = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    assert man["missions"][0]["name"] == "Route 1 – בדיקה"
    assert man["files"][f"missions/{m.id}/mission.db"]["schema"] == max(MISSION_MIGRATIONS)
    # A second backup in the same second gets its own folder.
    assert backup(root.path, tmp / "backups") != b
    restored = DataRoot(tmp / "restored")
    restored.con.close()
    with pytest.raises(AdminError, match="not empty"):
        restore(b, tmp / "restored")
    restore(b, tmp / "restored2", "lior")
    r = DataRoot(tmp / "restored2")
    m2 = r.open_mission(m.id)
    assert m2.con.execute("SELECT COUNT(*) FROM detection").fetchone()[0] == 9
    assert audit_log(r)[0]["action"] == "restore"
    # A damaged backup is refused.
    with open(b / "missions" / m.id / "mission.db", "r+b") as f:
        f.seek(200)
        f.write(b"\xff" * 64)
    assert verify(b) and "checksum" in verify(b)[0]
    with pytest.raises(AdminError, match="did not verify"):
        restore(b, tmp / "restored3")


def test_backup_refuses_inside_data_root(env):
    tmp, root, _, _ = env
    with pytest.raises(AdminError, match="outside"):
        backup(root.path, root.path / "bk")


def test_update_backs_up_before_migrating_and_rollback_restores_old_schema(env):
    tmp, root, m, _ = env
    db = m.dir / "mission.db"
    newest = max(MISSION_MIGRATIONS)
    m.con.close()
    # Make the mission look like it was written by the previous version (one migration behind).
    con = sqlite3.connect(db)
    assert newest == 12, "update this test to undo the newest migration"
    con.execute("ALTER TABLE light_group DROP COLUMN features")
    con.execute("ALTER TABLE light_group DROP COLUMN model_id")
    con.execute("DROP TABLE night_light_model")
    con.execute("DELETE FROM schema_version WHERE version=?", (newest,))
    con.commit()
    con.close()
    pre = backup(root.path, tmp / "backups", "lior")         # the step before an update
    m = root.open_mission(m.id)                              # the new version opens it: migrates
    assert m.con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == newest
    auto = list((m.dir / "backups").glob(f"mission-v{newest - 1}-*.db"))
    assert len(auto) == 1                                    # automatic copy taken before migrating
    assert m.con.execute("SELECT COUNT(*) FROM detection").fetchone()[0] == 9
    # Rollback: restore the pre-update backup into a new folder; it keeps the old schema for the old version.
    restore(pre, tmp / "rollback")
    con = sqlite3.connect(tmp / "rollback" / "missions" / m.id / "mission.db")
    assert con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == newest - 1
    con.close()


def test_delete_mission_is_explicit_and_logged(env):
    tmp, root, m, src = env
    m.con.close()
    with pytest.raises(AdminError, match="name"):
        delete_mission(root.path, m.id, "lior", "Route 1")
    d = delete_mission(root.path, m.id, "lior", "Route 1 – בדיקה")
    assert d["rows"]["detection"] == 9 and "folder_left" not in d
    assert not (root.path / "missions" / m.id).exists()
    assert len(list(src.glob("*.jpg"))) == 3                 # source images untouched
    assert [r["id"] for r in root.missions()] == []
    e = audit_log(root)[0]
    assert e["action"] == "mission_deleted" and e["author"] == "lior" and e["detail"]["name"] == "Route 1 – בדיקה"
    with pytest.raises(AdminError):
        delete_mission(root.path, m.id, "lior", "Route 1 – בדיקה")


def test_backup_due_once_a_day():
    from datetime import datetime
    from trafficwb.admin import backup_due
    assert not backup_due(datetime(2026, 10, 24, 1, 59), None, "02:00")
    assert backup_due(datetime(2026, 10, 24, 2, 0), None, "02:00")
    assert not backup_due(datetime(2026, 10, 24, 9, 0), datetime(2026, 10, 24, 2, 1), "02:00")
    assert backup_due(datetime(2026, 10, 25, 2, 5), datetime(2026, 10, 24, 2, 1), "02:00")
    # Server was off at 02:00: the backup runs when it next checks that day.
    assert backup_due(datetime(2026, 10, 24, 14, 0), datetime(2026, 10, 23, 2, 0), "02:00")


def test_scheduler_records_scheduled_backup(env):
    from datetime import datetime
    from trafficwb.admin import BackupScheduler
    tmp, root, _, _ = env
    s = BackupScheduler(root.path, tmp / "nightly", "02:00")
    p = s.run_if_due(datetime.now().replace(hour=23, minute=59))
    assert p is not None and verify(p) == []
    assert s.run_if_due(datetime.now().replace(hour=23, minute=59)) is None   # already done today
