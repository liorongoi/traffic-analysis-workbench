import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from trafficwb.ingest import cycle
from trafficwb.regions import RegionEditor, RegionSet, point_in_polygon, validate_polygon
from trafficwb.store import DataRoot, MissionConfig
from trafficwb.views import ViewService, signature

W, H = 640, 360
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()


def scene(seed: int) -> np.ndarray:
    """A textured 'fixed camera view': random lines, poles and blocks."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(60):
        p1 = tuple(int(v) for v in rng.integers(0, [W, H]))
        p2 = tuple(int(v) for v in rng.integers(0, [W, H]))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, p1, p2, col, int(rng.integers(1, 4)))
    for _ in range(25):
        x, y = (int(v) for v in rng.integers(0, [W - 40, H - 40]))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 40)), y + int(rng.integers(10, 40))),
                      tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return img


def day(base, k):
    rng = np.random.default_rng(1000 + k)
    out = cv2.convertScaleAbs(base, alpha=1.0 + 0.05 * (k % 3), beta=int(rng.integers(-15, 15)))
    return cv2.add(out, rng.integers(0, 8, out.shape, dtype=np.uint8))


def night(base, k):
    """Monochrome, dark, inverted-ish contrast with bright spots: a different look, same structure."""
    rng = np.random.default_rng(2000 + k)
    g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g = cv2.convertScaleAbs(255 - g, alpha=0.6, beta=10)
    for _ in range(8):
        cv2.circle(g, tuple(int(v) for v in rng.integers(0, [W, H])), int(rng.integers(3, 9)), 255, -1)
    g = cv2.add(g, rng.integers(0, 10, g.shape, dtype=np.uint8))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def shifted(img, dx=40, dy=0):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def write(src: Path, name: str, img):
    p = src / name
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    os.utime(p, (OLD, OLD))
    return p


def ts(i):
    return (datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(minutes=10 * i)).strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture
def mission(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(
        name="v", input_type="images", mode="general_traffic", country="IL", iana_zone="Asia/Jerusalem",
        source_dir=str(src), include_subdirs=False, processing="continuous", output_dir=str(out),
        export_formats=["csv"]), "test")
    return m, src


def name_of(m, vid):
    return m.con.execute("SELECT f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id WHERE v.id=?",
                         (vid,)).fetchone()[0]


def populate(src, a, b, n=6):
    i = 0
    for base, cam in ((a, "CAMA"), (b, "CAMB")):
        for k in range(n):
            write(src, f"{cam}_{ts(i)}.jpg", day(base, k)); i += 1
            write(src, f"{cam}_{ts(i)}.jpg", night(base, k)); i += 1


# ---------- signatures ----------

def test_signature_condition_measured_from_pixels():
    base = scene(1)
    assert signature(day(base, 0)).condition == "colour"
    assert signature(night(base, 0)).condition == "monochrome"


def test_same_view_more_similar_than_other_view_across_lighting():
    a, b = scene(1), scene(2)
    sa_day, sa_night, sb_night = signature(day(a, 0)), signature(night(a, 1)), signature(night(b, 1))
    assert float(sa_day.vec @ sa_night.vec) > float(sa_day.vec @ sb_night.vec) + 0.1


# ---------- grouping ----------

def test_two_views_each_spanning_day_and_night(mission):
    m, src = mission
    populate(src, scene(1), scene(2))
    cycle(m)
    vs = ViewService(m)
    stats = vs.assign_or_propose()
    views = vs.views()
    assert stats["new_views"] == 2 and len(views) == 2
    for v in views:
        assert v["conditions"] == {"colour": 6, "monochrome": 6}
        cams = {name_of(m, x)[:4] for x in vs.members(v["id"])}
        assert len(cams) == 1  # no mixing of physical views
        assert v["link_confirmed"] == 0 and "structure" in v["link_note"]  # day/night join awaits operator


def test_rerun_does_not_override_existing_assignments(mission):
    m, src = mission
    populate(src, scene(1), scene(2))
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    before = m.con.execute("SELECT COUNT(*) FROM view_assignment").fetchone()[0]
    assert vs.assign_or_propose() == {"assigned_existing": 0, "new_views": 0, "linked_by_geometry": 0,
                                      "linked_by_structure": 0, "unassigned": 0}
    assert m.con.execute("SELECT COUNT(*) FROM view_assignment").fetchone()[0] == before


def test_new_images_join_existing_view(mission):
    m, src = mission
    a, b = scene(1), scene(2)
    populate(src, a, b)
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    write(src, f"CAMA_{ts(100)}.jpg", day(a, 9))
    write(src, f"CAMB_{ts(101)}.jpg", night(b, 9))
    cycle(m)
    stats = vs.assign_or_propose()
    assert stats["assigned_existing"] == 2 and stats["new_views"] == 0
    for v in vs.views():
        assert len({name_of(m, x)[:4] for x in vs.members(v["id"])}) == 1


def test_small_groups_stay_unassigned(mission):
    m, src = mission
    populate(src, scene(1), scene(2))
    write(src, f"ODD_{ts(200)}.jpg", day(scene(77), 0))
    cycle(m)
    vs = ViewService(m)
    stats = vs.assign_or_propose(min_size=3)
    assert stats["unassigned"] == 1
    assert [name_of(m, x)[:3] for x in vs.unassigned()] == ["ODD"]


def test_resolution_mismatch_never_shares_a_view(mission):
    m, src = mission
    a = scene(1)
    for k in range(4):
        write(src, f"CAMA_{ts(k)}.jpg", day(a, k))
        write(src, f"CAMA_{ts(10 + k)}.jpg", cv2.resize(day(a, k), (1280, 720)))
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    sizes = sorted((v["ref_width"], v["ref_height"]) for v in vs.views())
    assert sizes == [(640, 360), (1280, 720)]


# ---------- operator actions ----------

def test_manual_reassign_split_merge_rename_keep_history(mission):
    m, src = mission
    populate(src, scene(1), scene(2))
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    v1, v2 = [v["id"] for v in vs.views()]
    item = vs.members(v1)[0]
    vs.reassign([item], v2, "lior", "wrong group")
    assert vs.current_view(item) == v2
    hist = vs.history(item)
    assert [h["method"] for h in hist] == ["auto", "manual"] and hist[1]["author"] == "lior"

    night_items = [x for x in vs.members(v2) if m.con.execute(
        "SELECT condition FROM view_signature WHERE source_version_id=?", (x,)).fetchone()[0] == "monochrome"]
    v3 = vs.split(v2, night_items, "lior", name="B night")
    assert set(vs.members(v3)) == set(night_items)
    vs.merge(v2, v3, "lior")
    assert vs.members(v3) == [] and set(night_items) <= set(vs.members(v2))
    vs.rename(v2, "Bridge north", "lior")
    assert {v["name"] for v in vs.views()} >= {"Bridge north"}
    kinds = [r[0] for r in m.con.execute("SELECT kind FROM view_event WHERE view_id=? ORDER BY id", (v2,))]
    assert kinds[-3:] == ["split", "merged", "renamed"]
    vs.confirm(v2, "lior")
    assert m.con.execute("SELECT state, link_confirmed FROM camera_view WHERE id=?", (v2,)).fetchone()[:] == ("active", 1)


# ---------- view-change detection ----------

def test_drift_flags_view_after_consecutive_verified_shifts(mission):
    m, src = mission
    a = scene(1)
    for k in range(4):
        write(src, f"CAMA_{ts(k)}.jpg", day(a, k))
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    vid = vs.views()[0]["id"]
    ok_item = [x for x in vs.members(vid) if x != m.con.execute(
        "SELECT source_version_id FROM view_reference WHERE view_id=?", (vid,)).fetchone()[0]][0]
    assert vs.check_drift(ok_item) == "ok"

    write(src, f"CAMA_{ts(50)}.jpg", shifted(day(a, 5), 40))
    write(src, f"CAMA_{ts(51)}.jpg", shifted(day(a, 6), 40))
    write(src, f"CAMA_{ts(52)}.jpg", cv2.GaussianBlur(day(a, 7), (61, 61), 30))  # lens obstructed
    cycle(m)
    new = sorted(set(r[0] for r in m.con.execute("SELECT id FROM source_version")) - set(vs.members(vid)))
    vs.reassign(new, vid, "lior")  # operator puts them in the view regardless of their look
    r = [vs.check_drift(x) for x in new]
    assert r[0] == "shifted" and r[1] == "shifted" and r[2] == "unverified"
    assert vs.regions_applicable(vid) is False
    assert m.con.execute("SELECT COUNT(*) FROM notification WHERE code='view_changed'").fetchone()[0] == 1


def test_single_shift_does_not_flag(mission):
    m, src = mission
    a = scene(1)
    for k in range(3):
        write(src, f"CAMA_{ts(k)}.jpg", day(a, k))
    write(src, f"CAMA_{ts(9)}.jpg", shifted(day(a, 5), 40))
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose(min_size=1)
    vid = vs.current_view(m.con.execute("SELECT MIN(id) FROM source_version").fetchone()[0])
    last = m.con.execute("SELECT MAX(id) FROM source_version").fetchone()[0]
    vs.reassign([last], vid, "lior")
    assert vs.check_drift(last) == "shifted"
    assert vs.regions_applicable(vid) is True


# ---------- regions ----------

SQUARE = [(100, 100), (300, 100), (300, 300), (100, 300)]


def test_point_in_polygon_including_edges():
    assert point_in_polygon(200, 200, SQUARE)
    assert point_in_polygon(100, 200, SQUARE)  # on edge
    assert not point_in_polygon(50, 200, SQUARE)


@pytest.mark.parametrize("poly, msg", [
    ([(0, 0), (10, 10)], "at least 3"),
    ([(0, 0), (700, 0), (0, 100)], "outside"),
    ([(0, 0), (100, 100), (100, 0), (0, 100)], "cross"),
    ([(0, 0), (10, 0), (20, 0)], "no area"),
])
def test_polygon_validation(poly, msg):
    assert any(msg in e for e in validate_polygon(poly, 640, 360))


def test_exclusion_wins_and_no_inclusion_means_whole_frame():
    from trafficwb.regions import Region
    rs = RegionSet([])
    assert rs.contains_point(5, 5)
    rs = RegionSet([Region("i", "include", None, SQUARE), Region("e", "exclude", None, [(150, 150), (250, 150), (250, 250), (150, 250)])])
    assert rs.contains_point(120, 120)
    assert not rs.contains_point(200, 200)  # inside both: exclusion wins
    assert not rs.contains_point(400, 200)


def test_box_membership_rules():
    from trafficwb.regions import Region
    rs = RegionSet([Region("i", "include", None, SQUARE)])
    box = (250, 50, 350, 150)  # bottom-centre (300,150) on edge, centroid (300,100) on corner, quarter overlap
    assert rs.contains_box(box, "bottom_center")
    assert rs.contains_box(box, "centroid")
    assert not rs.contains_box(box, "overlap", 0.5)
    assert rs.contains_box(box, "overlap", 0.2)


def test_region_revisions_and_delete(mission):
    m, src = mission
    populate(src, scene(1), scene(2), n=3)
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    vid = vs.views()[0]["id"]
    ed = RegionEditor(m.con)
    uid = ed.save(vid, "include", SQUARE, "lior", name="carriageway")
    ed.save(vid, "include", [(100, 100), (320, 100), (320, 300), (100, 300)], "lior", region_uid=uid)
    rs = RegionSet.load(m.con, vid)
    assert rs.include[0].polygon[1] == (320.0, 100.0)
    ed.delete(vid, uid, "lior")
    assert RegionSet.load(m.con, vid).include == []
    assert len(ed.history(uid)) == 3
    with pytest.raises(ValueError):
        ed.save(vid, "include", [(0, 0), (900, 0), (0, 100)], "lior")  # beyond the 640x360 view


# ---------- optional: real sample ----------

@pytest.mark.skipif(not os.environ.get("TWB_SAMPLE_DIR"), reason="set TWB_SAMPLE_DIR to a folder of sample images")
def test_real_sample_single_preset_forms_one_view(tmp_path):
    import shutil
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for p in Path(os.environ["TWB_SAMPLE_DIR"]).glob("*.jpg"):
        shutil.copy2(p, src)
        os.utime(src / p.name, (OLD, OLD))
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(
        name="s", input_type="images", mode="general_traffic", country="FI", iana_zone="Europe/Helsinki",
        source_dir=str(src), include_subdirs=False, processing="one_time", output_dir=str(out), export_formats=["csv"]), "t")
    cycle(m)
    vs = ViewService(m)
    vs.assign_or_propose()
    views = vs.views()
    assert len(views) == 1
    assert set(views[0]["conditions"]) == {"colour", "monochrome"}
