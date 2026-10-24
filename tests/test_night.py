import json
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from trafficwb.detect import Detection
from trafficwb.export import IMAGE_COLUMNS, OBJECT_COLUMNS, build_tables
from trafficwb.ingest import cycle
from trafficwb.night import (BackgroundParams, Light, WidthMap, build_background, covered_by, group_lights,
                             light_components, measure_condition)
from trafficwb.night_service import NightService
from trafficwb.process import Processor
from trafficwb.regions import RegionEditor
from trafficwb.review import ReviewService
from trafficwb.store import DataRoot, MissionConfig
from trafficwb.views import ViewService

W, H = 640, 360
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
LAMPS = [(60, 40), (300, 30), (560, 45)]          # street lamps: lit in every night image


def scene(seed=3):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 70, np.uint8)
    for _ in range(50):
        p1 = tuple(int(v) for v in rng.integers(0, [W, H]))
        p2 = tuple(int(v) for v in rng.integers(0, [W, H]))
        cv2.line(img, p1, p2, tuple(int(c) for c in rng.integers(20, 160, 3)), 2)
    return img


def day_frame(k):
    """Colour daytime image with car-like boxes whose width grows with the row (perspective)."""
    rng = np.random.default_rng(10 + k)
    img = cv2.add(scene(), rng.integers(0, 6, (H, W, 3), dtype=np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = 90
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for y in (120, 200, 280, 330):
        w = int(0.1 * y + 6)
        x = int(rng.integers(50, 500))
        cv2.rectangle(img, (x, y - int(w * 0.6)), (x + w, y), (0, 0, 255), -1)
    return img


def night_frame(k, pairs=(), singles=()):
    """Monochrome night image: lamps always, plus headlight pairs at given (x_centre, y)."""
    rng = np.random.default_rng(50 + k)
    g = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    g = cv2.add((g * 0.6).astype(np.uint8), rng.integers(0, 6, g.shape, dtype=np.uint8))
    for (x, y) in LAMPS:
        cv2.circle(g, (x, y), 5, 255, -1)
    for (x, y) in pairs:
        half = (0.1 * y + 6) * 0.35
        for dx in (-half, half):
            cv2.circle(g, (int(x + dx), y), 3, 255, -1)
    for (x, y) in singles:
        cv2.circle(g, (x, y), 3, 255, -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class RedBoxDetector:
    def describe(self):
        return {"model": {"name": "fake", "version": "1", "sha256": "0" * 64}, "preprocessing": {}}

    def detect(self, img):
        mask = np.all(np.abs(img.astype(int) - (0, 0, 255)) <= 8, axis=2).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "car", 0.9, {"car": 0.9})
                for x, y, w, h, a in st[1:] if a > 30]


def ts(i):
    return (datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(minutes=10 * i)).strftime("%Y%m%dT%H%M%SZ")


def write(src, name, img):
    p = src / name
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    os.utime(p, (OLD, OLD))


@pytest.fixture
def mission(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    root = DataRoot(tmp_path / "data")
    m = root.create_mission(MissionConfig(
        name="n", input_type="images", mode="general_traffic", country="LV", iana_zone="Europe/Riga",
        source_dir=str(src), include_subdirs=False, processing="continuous", output_dir=str(out),
        export_formats=["csv"]), "test")
    return m, src


def vid_of(m, name):
    return m.con.execute("SELECT v.id FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                         "WHERE f.rel_path=?", (name,)).fetchone()[0]


# ---------- pure functions ----------

def test_condition_from_content_not_clock():
    assert measure_condition(day_frame(0)).label == "day"
    n = measure_condition(night_frame(0))
    assert n.label == "night" and n.monochrome
    dark_colour = cv2.convertScaleAbs(day_frame(1), alpha=0.3)
    assert measure_condition(dark_colour).label == "night"
    flat = np.full((H, W, 3), (90, 110, 130), np.uint8)
    assert "low_detail" in measure_condition(flat).flags


def test_light_components_found_with_features():
    lights = light_components(night_frame(0, pairs=[(320, 300)]))
    assert len(lights) == 5   # 3 lamps + 2 headlights
    assert all(li.area >= 6 and li.peak >= 235 for li in lights)


def test_width_map_fit_and_refusal():
    boxes = [(0, y - 10, 0.1 * y + 6, y) for y in range(100, 340, 10)]
    wm = WidthMap.fit(boxes)
    assert wm and abs(wm.width(300) - 36) < 0.5
    assert WidthMap.fit(boxes[:5]) is None                       # too few samples


def _L(cx, cy, r=3, peak=255.0):
    return Light((cx - r, cy - r, cx + r, cy + r), cx, cy, (2 * r) ** 2, peak, 250.0, 1.0)


def test_pairing_uses_expected_width_at_row():
    wm = WidthMap(0.1, 6, 50, "test")                             # width 36 px at row 300
    lights = [_L(300, 300), _L(326, 300), _L(500, 120)]          # spacing 26/36 = 0.72: a pair
    groups = group_lights(lights, wm)
    kinds = sorted(g.kind for g in groups)
    assert kinds == ["pair", "single"]
    pair = next(g for g in groups if g.kind == "pair")
    assert (pair.count_min, pair.count_max) == (1, 2)          # ambiguity kept: one vehicle or two
    far = [_L(300, 300), _L(380, 300)]                            # 80 px apart at a 36 px row: not one vehicle
    assert sorted(g.kind for g in group_lights(far, wm)) == ["single", "single"]


def test_no_width_map_means_no_pairs_and_wide_range():
    groups = group_lights([_L(300, 300), _L(326, 300)], None)
    assert [g.kind for g in groups] == ["single", "single"]
    assert sum(g.count_min for g in groups) == 0 and sum(g.count_max for g in groups) == 2
    assert all("no_width_map" in g.flags for g in groups)


def test_ambiguous_pairing_is_flagged():
    wm = WidthMap(0.1, 6, 50, "test")
    lights = [_L(290, 300), _L(316, 300), _L(342, 300)]         # middle light pairs with either side
    groups = group_lights(lights, wm)
    pair = next(g for g in groups if g.kind == "pair")
    assert "ambiguous_pairing" in pair.flags


def test_reflection_lowers_score():
    wm = WidthMap(0.1, 6, 50, "test")
    lamp = _L(300, 200)
    refl = Light((297, 215, 303, 240), 300, 227, 120, 240.0, 230.0, 4.0)   # tall streak below a brighter light
    g = [x for x in group_lights([lamp, refl], wm) if 1 in x.members][0]
    assert "possible_reflection" in g.flags and g.score < 0.3


def test_background_keeps_stopped_vehicle_out_and_reports_new_lights():
    masks = []
    for k in range(20):
        m = np.zeros((100, 200), np.uint8)
        m[10:14, 10:14] = 1                     # lamp: always lit
        m[50:54, 100:104] = 1                   # stopped car's light: always lit, but under a vehicle box
        if k >= 14:
            m[80:84, 150:154] = 1               # light that appeared only recently
        masks.append(m)
    boxes = [[(95, 45, 110, 60)] for _ in masks]
    bg = build_background(masks, boxes, p=BackgroundParams(min_images=8, recent_images=8))
    assert bg.persistent[12, 12] == 1
    assert bg.persistent[52, 102] == 0          # never absorbed
    assert bg.persistent[82, 152] == 0
    assert any(b[0] <= 152 <= b[2] and b[1] <= 82 <= b[3] for b in bg.new_lights)
    assert build_background(masks[:5]) is None  # too few images: no model


def test_operator_labels_override_statistics():
    masks = [np.zeros((50, 50), np.uint8) for _ in range(10)]
    for m in masks[:3]:
        m[20:24, 20:24] = 1
    bg = build_background(masks, labels=[("persistent", [[18, 18], [26, 18], [26, 26], [18, 26]])])
    assert bg.persistent[22, 22] == 1


def test_covered_by_detection():
    assert covered_by((100, 100, 120, 105), [(7, (95, 90, 125, 110))]) == 7
    assert covered_by((300, 100, 320, 105), [(7, (95, 90, 125, 110))]) is None


# ---------- through processing and review ----------

def _night_mission(mission, n_day=6, n_night=10):
    m, src = mission
    for k in range(n_day):
        write(src, f"CAM1_{ts(k)}.jpg", day_frame(k))
    for k in range(n_night):
        pairs = [(200 + 20 * k, 300)] if k % 2 == 0 else []
        write(src, f"CAM1_{ts(100 + k)}.jpg", night_frame(k, pairs=pairs))
    cycle(m)
    ViewService(m).assign_or_propose()
    return m, src


def test_night_pipeline_lamps_explained_headlights_become_candidates(mission):
    m, src = _night_mission(mission)
    vs = ViewService(m)
    views = vs.views()
    assert len(views) == 1 and set(views[0]["conditions"]) == {"colour", "monochrome"}
    rep = Processor(m, RedBoxDetector()).process_all()[0]
    assert rep.night["backgrounds_built"] == 1
    bg = m.con.execute("SELECT * FROM background_model").fetchone()
    assert bg["state"] == "ready" and bg["image_count"] == 10
    rs = ReviewService(m)
    with_car = rs.results()[6]                       # first night image has one headlight pair
    assert with_car.condition == "night"
    assert with_car.vehicle_count == 0               # lights are never counted as vehicles
    assert (with_car.candidate_count, with_car.candidate_count_max) == (1, 2)
    cands = rs.candidates(with_car.source_version_id)
    assert len(cands) == 1 and cands[0].kind == "pair"
    empty = rs.results()[7]
    assert (empty.candidate_count, empty.candidate_count_max) == (0, 0)   # lamps explained by the background
    run = m.con.execute("SELECT preprocessing FROM run").fetchone()[0]
    assert json.loads(run)["night"]["enhancement"].startswith("none")


def test_candidate_decisions_and_counts(mission):
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    rs = ReviewService(m)
    rows = [r for r in rs.results() if r.candidate_count]
    a, b, c = rows[0].source_version_id, rows[1].source_version_id, rows[2].source_version_id
    rs.approve(a, "lior")
    rs.confirm_candidate(a, rs.candidates(a)[0].uid, "lior")
    assert rs.review_status(a) == "unreviewed"      # a decision reopens an approved image
    ra = next(r for r in rs.results() if r.source_version_id == a)
    assert ra.vehicle_count == 1 and ra.confirmed_from_candidates == 1 and ra.candidate_count == 0
    rs.reject_candidate(b, rs.candidates(b)[0].uid, "lior", note="reflection on the barrier")
    rb = next(r for r in rs.results() if r.source_version_id == b)
    assert rb.vehicle_count == 0 and rb.candidate_count == 0
    # Regroup: the operator says the two lights are two motorcycles.
    lights = [li.id for li in NightService(m).lights(c) if li.cy > 250]
    u1 = rs.regroup(c, lights[:1], "lior")
    u2 = rs.regroup(c, lights[1:], "lior")
    rs.confirm_candidate(c, u1, "lior", cls="motorcycle")
    rs.confirm_candidate(c, u2, "lior", cls="motorcycle")
    rc = next(r for r in rs.results() if r.source_version_id == c)
    assert rc.vehicle_count == 2 and rc.candidate_count == 0
    assert m.con.execute("SELECT COUNT(*) FROM light_group").fetchone()[0] > 0
    before = m.con.execute("SELECT COUNT(*) FROM light_component").fetchone()[0]
    images, objects = build_tables(m)
    assert all(len(r) == len(IMAGE_COLUMNS) for r in images)
    assert all(len(r) == len(OBJECT_COLUMNS) for r in objects)
    levels = {r[OBJECT_COLUMNS.index("evidence_level")] for r in objects}
    assert {"confirmed_vehicle_from_lights", "candidate_rejected", "unresolved_candidate"} <= levels
    assert m.con.execute("SELECT COUNT(*) FROM light_component").fetchone()[0] == before   # machine output untouched


def test_mark_persistent_light_rebuilds_background(mission):
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    rs = ReviewService(m)
    v = next(r for r in rs.results() if r.candidate_count).source_version_id
    uid = rs.candidates(v)[0].uid
    rs.mark_persistent_light(v, uid, "lior")
    assert m.con.execute("SELECT COUNT(*) FROM light_label").fetchone()[0] == 1
    assert m.con.execute("SELECT COUNT(*) FROM background_model").fetchone()[0] == 2
    assert rs.candidates(v)[0].status == "persistent_light"
    assert next(r for r in rs.results() if r.source_version_id == v).candidate_count == 0


def test_stopped_vehicle_confirmed_is_not_absorbed(mission):
    """A car parked with lights on all night: once confirmed, rebuilding never turns it into background."""
    m, src = mission
    for k in range(6):
        write(src, f"CAM1_{ts(k)}.jpg", day_frame(k))
    for k in range(10):
        write(src, f"CAM1_{ts(100 + k)}.jpg", night_frame(k, pairs=[(400, 300)]))
    cycle(m)
    ViewService(m).assign_or_propose()
    Processor(m, RedBoxDetector()).process_all()
    rs = ReviewService(m)
    night_rows = [r for r in rs.results() if r.condition == "night"]
    # The parked car is lit in every image, so the statistics alone call it background.
    assert all(r.candidate_count == 0 for r in night_rows)
    # The operator labels it as not background and confirms it in every image.
    view = ViewService(m).views()[0]["id"]
    m.con.execute("INSERT INTO light_label(view_id,condition,kind,polygon,author,created_at) VALUES (?,?,?,?,?,?)",
                  (view, "night", "not_persistent", json.dumps([[370, 285], [430, 285], [430, 315], [370, 315]]), "lior",
                   "2026-10-24T00:00:00Z"))
    NightService(m).refresh()
    for r in night_rows:
        cands = rs.candidates(r.source_version_id)
        assert len(cands) == 1
        rs.confirm_candidate(r.source_version_id, cands[0].uid, "lior")
    NightService(m).ensure_background(view, "night", force=True)
    NightService(m).refresh()
    assert all(r.vehicle_count == 1 for r in rs.results() if r.condition == "night")


# ---------- the real sample ----------

@pytest.mark.skipif(not os.environ.get("TWB_SAMPLE_DIR"), reason="set TWB_SAMPLE_DIR")
def test_real_sample_night_condition():
    from trafficwb.views import read_image
    from pathlib import Path
    d = Path(os.environ["TWB_SAMPLE_DIR"])
    assert measure_condition(read_image(d / "C0450701_20261020T194513Z.jpg")).label == "night"
    assert measure_condition(read_image(d / "C0450701_20261021T071528Z.jpg")).label == "day"
    assert "low_detail" in measure_condition(read_image(d / "C0450701_20261021T034516Z.jpg")).flags


def test_region_edit_triggers_reanalysis(mission):
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    rs = ReviewService(m)
    before = sum(r.candidate_count or 0 for r in rs.results())
    assert before > 0
    view = ViewService(m).views()[0]["id"]
    RegionEditor(m.con).save(view, "exclude", [(0, 250), (640, 250), (640, 360), (0, 360)], "lior")  # the headlight row
    stats = NightService(m).refresh()
    assert stats["images_analysed"] == 10 and stats["backgrounds_built"] == 0
    assert sum(r.candidate_count or 0 for r in rs.results()) == 0
