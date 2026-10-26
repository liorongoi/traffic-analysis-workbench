import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trafficwb.ingest import Ingestor, Monitor, cycle, resume_enabled
from trafficwb.store import DataRoot, MissionConfig, MissionValidationError
from test_inventory import jpeg_bytes

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
OLD = NOW - timedelta(hours=1)


def put(path: Path, data: bytes, mtime: datetime = OLD) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


@pytest.fixture
def env(tmp_path):
    src, out, data = tmp_path / "src", tmp_path / "out", tmp_path / "data"
    src.mkdir()
    out.mkdir()
    return src, out, DataRoot(data)


def cfg(src, out, **kw):
    base = dict(name="Test", input_type="images", mode="general_traffic", country="IL", iana_zone="Asia/Jerusalem",
                source_dir=str(src), include_subdirs=True, processing="continuous", output_dir=str(out),
                export_formats=["xlsx", "csv"])
    base.update(kw)
    return MissionConfig(**base)


def versions(m):
    return m.con.execute("SELECT * FROM source_version ORDER BY id").fetchall()


# ---------- mission creation ----------

def test_mission_db_created_on_commit_before_any_run(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), author="lior")
    assert (m.dir / "mission.db").exists()
    assert not (m.dir / "mission.db.creating").exists()
    assert m.con.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
    assert [r["name"] for r in root.missions()] == ["Test"]
    assert m.zone == "Asia/Jerusalem"


def test_missions_are_isolated_databases(env):
    src, out, root = env
    a = root.create_mission(cfg(src, out, name="A"), "lior")
    b = root.create_mission(cfg(src, out, name="B"), "lior")
    assert a.dir != b.dir
    put(src / "17052026-1604-x.jpg", jpeg_bytes(64, 48))
    Ingestor(a).scan(NOW)
    assert len(versions(a)) == 1 and len(versions(b)) == 0


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(name=" "), "name is empty"),
        (dict(iana_zone="Mars/Olympus"), "not a known IANA time zone"),
        (dict(input_type="video", mode="general_traffic"), "not available for video"),
        (dict(export_formats=["pdf"]), "xlsx, csv"),
        (dict(country="FI", mode="plate_recognition"), "supported countries"),
        (dict(latitude=32.0), "both latitude and longitude"),
        (dict(latitude=91.0, longitude=0.0), "latitude −90..90"),
    ],
)
def test_validation_errors(env, change, message):
    src, out, root = env
    with pytest.raises(MissionValidationError) as e:
        root.create_mission(cfg(src, out, **change), "lior")
    assert message in str(e.value)


def test_output_inside_recursive_source_rejected(env):
    src, out, root = env
    inner = src / "exports"
    inner.mkdir()
    with pytest.raises(MissionValidationError) as e:
        root.create_mission(cfg(src, inner), "lior")
    assert "re-ingested" in str(e.value)


def test_non_target_country_is_a_warning_not_an_error(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out, country="FI", iana_zone="Europe/Helsinki"), "lior")
    assert m.config().country == "FI"


def test_missing_source_folder(env, tmp_path):
    _, out, root = env
    with pytest.raises(MissionValidationError) as e:
        root.create_mission(cfg(tmp_path / "nope", out), "lior")
    assert "does not exist" in str(e.value)


def test_input_type_and_mode_immutable(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    with pytest.raises(ValueError):
        m.update_config("lior", "try", mode="object_search")
    m.update_config("lior", "turn on", monitoring_enabled=True)
    assert m.config().monitoring_enabled is True
    assert m.con.execute("SELECT COUNT(*) FROM config_revision").fetchone()[0] == 2  # history kept


# ---------- ingestion ----------

def test_repeated_scans_do_not_duplicate(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    r1 = Ingestor(m).scan(NOW)
    r2 = Ingestor(m).scan(NOW + timedelta(minutes=1))
    r3 = Ingestor(m).scan(NOW + timedelta(minutes=2))
    assert len(r1.new_versions) == 1 and r2.new_versions == [] and r3.new_versions == []
    assert len(versions(m)) == 1


def test_identical_bytes_at_different_times_stay_distinct(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    data = jpeg_bytes(64, 48)
    put(src / "17052026-1604-a.jpg", data)
    put(src / "17052026-1614-a.jpg", data)
    r = Ingestor(m).scan(NOW)
    assert len(r.new_versions) == 2
    caps = sorted(v["capture_utc"] for v in versions(m))
    assert caps == ["2026-05-17T16:04:00Z", "2026-05-17T16:14:00Z"]


def test_recent_file_waits_for_stability(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    p = put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48), mtime=NOW - timedelta(seconds=5))
    r1 = Ingestor(m, stability_seconds=30).scan(NOW)
    assert r1.new_versions == [] and r1.waiting_for_stability == 1
    r2 = Ingestor(m, stability_seconds=30).scan(NOW + timedelta(seconds=40))
    assert len(r2.new_versions) == 1 and r2.waiting_for_stability == 0


def test_file_growing_between_scans_is_not_ingested_until_it_stops(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    full = jpeg_bytes(64, 48)
    p = put(src / "17052026-1604-a.jpg", full[:20], mtime=NOW - timedelta(seconds=5))
    assert Ingestor(m).scan(NOW).new_versions == []
    put(p, full[:40], mtime=NOW + timedelta(seconds=30))
    assert Ingestor(m).scan(NOW + timedelta(seconds=35)).new_versions == []
    put(p, full, mtime=NOW + timedelta(seconds=60))
    assert Ingestor(m).scan(NOW + timedelta(seconds=65)).new_versions == []  # changed since last look
    r = Ingestor(m).scan(NOW + timedelta(seconds=120))
    assert len(r.new_versions) == 1
    assert versions(m)[0]["size"] == len(full)


def test_truncated_jpeg_with_old_mtime_waits_then_becomes_corrupt(env):
    # Copy tools can preserve the original mtime while still writing.
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48, complete=False))
    r1 = Ingestor(m).scan(NOW)
    assert r1.new_versions == [] and r1.corrupt == [] and r1.waiting_for_stability == 1
    r2 = Ingestor(m).scan(NOW + timedelta(minutes=5))
    assert r2.corrupt == ["17052026-1604-a.jpg"] and r2.new_versions == []
    assert versions(m)[0]["integrity"] == "corrupt"


def test_malformed_names_recorded_not_queued_and_notified(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "IMG_0001.jpg", jpeg_bytes(64, 48))
    put(src / "31022026-1200-x.jpg", jpeg_bytes(64, 48))
    r = Ingestor(m).scan(NOW)
    assert sorted(r.malformed) == ["31022026-1200-x.jpg", "IMG_0001.jpg"]
    assert r.new_versions == []
    assert all(v["parse_status"] == "malformed" and v["capture_utc"] is None for v in versions(m))
    assert m.con.execute("SELECT COUNT(*) FROM notification WHERE code='malformed_names'").fetchone()[0] == 1


def test_no_filesystem_time_fallback(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "photo.jpg", jpeg_bytes(64, 48))
    Ingestor(m).scan(NOW)
    assert versions(m)[0]["capture_utc"] is None


def test_replaced_file_creates_new_version_and_keeps_old(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    p = put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    Ingestor(m).scan(NOW)
    put(p, jpeg_bytes(128, 96), mtime=NOW + timedelta(minutes=1))
    r = Ingestor(m).scan(NOW + timedelta(minutes=10))
    vs = versions(m)
    assert len(vs) == 2 and len(r.new_versions) == 1
    assert vs[0]["superseded_by"] == vs[1]["id"] and vs[1]["superseded_by"] is None
    assert vs[0]["sha256"] != vs[1]["sha256"]


def test_touched_file_with_same_content_is_not_a_new_version(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    data = jpeg_bytes(64, 48)
    p = put(src / "17052026-1604-a.jpg", data)
    Ingestor(m).scan(NOW)
    put(p, data, mtime=NOW + timedelta(minutes=1))
    r = Ingestor(m).scan(NOW + timedelta(minutes=10))
    assert r.new_versions == [] and len(versions(m)) == 1


def test_deleted_file_marked_missing_and_rename_linked(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    data = jpeg_bytes(64, 48)
    p = put(src / "17052026-1604-a.jpg", data)
    Ingestor(m).scan(NOW)
    p.unlink()
    r = Ingestor(m).scan(NOW + timedelta(minutes=1))
    assert r.newly_missing == ["17052026-1604-a.jpg"]
    assert len(versions(m)) == 1  # history kept
    put(src / "renamed" / "17052026-1604-a.jpg", data)
    Ingestor(m).scan(NOW + timedelta(minutes=2))
    vs = versions(m)
    assert vs[1]["possible_rename_of"] == vs[0]["id"]


def test_subfolders_respected(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out, include_subdirs=False), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    put(src / "sub" / "17052026-1605-a.jpg", jpeg_bytes(64, 48))
    assert len(Ingestor(m).scan(NOW).new_versions) == 1


def test_app_data_folder_inside_source_is_never_scanned(tmp_path):
    # Non-recursive missions may keep data under the source; recursive ones are refused at validation.
    src = tmp_path / "src"
    out = tmp_path / "out"
    out.mkdir()
    root = DataRoot(src / "appdata")
    m = root.create_mission(cfg(src, out, include_subdirs=False), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    m.update_config("lior", "test", include_subdirs=True)
    m.con.execute("UPDATE source_root SET recursive=1")
    r = Ingestor(m).scan(NOW)
    assert len(r.new_versions) == 1
    rels = [r[0] for r in m.con.execute("SELECT rel_path FROM source_file")]
    assert rels == ["17052026-1604-a.jpg"]


def test_unsupported_extensions_ignored_and_counted(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    put(src / "manifest.json", b"{}")
    r = Ingestor(m).scan(NOW)
    assert r.ignored_by_extension == {".json": 1}


def test_unavailable_source_marks_nothing_missing(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    Ingestor(m).scan(NOW)
    src.rename(src.with_name("src-offline"))
    r = Ingestor(m).scan(NOW + timedelta(minutes=1))
    assert r.status == "source_unavailable" and r.newly_missing == []
    assert m.status()["missing"] == 0


def test_restart_resumes_from_database(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    put(src / "17052026-1605-a.jpg", jpeg_bytes(64, 48), mtime=NOW - timedelta(seconds=5))
    Ingestor(m).scan(NOW)
    mid, path = m.id, root.path
    m.con.close()
    root.con.close()
    root2 = DataRoot(path)  # "restart"
    m2 = root2.open_mission(mid)
    r = Ingestor(m2).scan(NOW + timedelta(minutes=5))
    assert len(r.new_versions) == 1  # only the file that was still settling
    assert len(versions(m2)) == 2


# ---------- runs ----------

def test_one_time_mission_gets_single_initial_run(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out, processing="one_time"), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    res, run = cycle(m, NOW)
    assert run is not None
    put(src / "17052026-1605-a.jpg", jpeg_bytes(64, 48))
    res2, run2 = cycle(m, NOW + timedelta(minutes=5))
    assert run2 is None and res2.status == "skipped_one_time_complete"
    assert m.con.execute("SELECT kind FROM run").fetchall()[0][0] == "initial"


def test_one_time_waits_for_complete_snapshot(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out, processing="one_time"), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    put(src / "17052026-1605-a.jpg", jpeg_bytes(64, 48), mtime=NOW - timedelta(seconds=5))
    _, run = cycle(m, NOW)
    assert run is None
    _, run = cycle(m, NOW + timedelta(minutes=5))
    assert m.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=?", (run,)).fetchone()[0] == 2


def test_continuous_mission_creates_batches_and_notifies(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    _, r1 = cycle(m, NOW)
    _, none = cycle(m, NOW + timedelta(minutes=1))
    put(src / "17052026-1614-a.jpg", jpeg_bytes(64, 48))
    _, r2 = cycle(m, NOW + timedelta(minutes=2))
    assert none is None
    kinds = [r[0] for r in m.con.execute("SELECT kind FROM run ORDER BY id")]
    assert kinds == ["initial", "continuous_batch"] and r2 != r1
    assert m.con.execute("SELECT COUNT(*) FROM notification WHERE code='new_data'").fetchone()[0] == 2
    assert m.con.execute("SELECT processing_state FROM run_item").fetchall()[0][0] == "queued"


def test_monitor_runs_in_background_and_resumes_after_restart(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    m.update_config("lior", "enable", monitoring_enabled=True)
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    mons = resume_enabled(root, interval_seconds=0.05, stability_seconds=0)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and m.status()["runs"] == 0:
            time.sleep(0.05)
    finally:
        for mon in mons:
            mon.stop()
    assert len(mons) == 1
    assert m.status()["runs"] == 1


def test_filesystem_notification_wakes_monitor_before_poll_interval(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    m.update_config("lior", "enable", monitoring_enabled=True)
    completed_cycle = threading.Event()
    mon = Monitor(m, interval_seconds=30, stability_seconds=0,
                  on_cycle=lambda _result, _run_id: completed_cycle.set())
    mon.start()
    try:
        assert mon._observer is not None, "watchdog observer must be available in the installed app"
        assert completed_cycle.wait(5), "initial reconciliation did not complete"
        completed_cycle.clear()
        put(src / "17052026-1604-notified.jpg", jpeg_bytes(64, 48))
        assert completed_cycle.wait(5), "filesystem event did not wake the long-polling monitor"
        assert m.status()["runs"] == 1
        assert m.con.execute("SELECT COUNT(*) FROM source_version").fetchone()[0] == 1
    finally:
        mon.stop()


def test_monitor_not_started_when_disabled(env):
    src, out, root = env
    root.create_mission(cfg(src, out), "lior")
    assert resume_enabled(root, interval_seconds=0.05) == []


def test_files_ingested_before_an_interrupted_cycle_are_still_queued(env):
    src, out, root = env
    m = root.create_mission(cfg(src, out), "lior")
    put(src / "17052026-1604-a.jpg", jpeg_bytes(64, 48))
    Ingestor(m).scan(NOW)  # scan finished, then the app stopped before creating a run
    _, run = cycle(m, NOW + timedelta(minutes=1))
    assert m.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=?", (run,)).fetchone()[0] == 1
