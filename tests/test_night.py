import json
import os
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from trafficwb.detect import Detection
from trafficwb.export import IMAGE_COLUMNS, OBJECT_COLUMNS, build_tables
from trafficwb.ingest import cycle
from trafficwb.night import (BackgroundParams, BandParams, Light, WidthMap, band_at, band_rows, build_background,
                             covered_by, group_lights, is_glare, light_components, measure_condition, off_vehicle_map,
                             on_vehicle_map, vehicle_map, map_distance)
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
    # 36 px is the middle band: two lights close together are one vehicle there
    assert pair.band == "middle" and (pair.count_min, pair.count_max) == (1, 1)
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
    assert (with_car.candidate_count, with_car.candidate_count_max) == (1, 1)   # a middle-distance pair: one vehicle
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


# ---------- distance bands and background readings ----------

def test_distance_bands_judge_what_a_vehicle_looks_like():
    wm = WidthMap(0.1, 6, 50, "test")                 # 16 px wide at row 100, 36 at row 300, 76 at row 700
    bp = BandParams(far_below=24, close_from=60)
    assert [band_at(wm, y, bp) for y in (100, 300, 700)] == ["far", "middle", "close"]
    assert band_at(None, 300, bp) is None
    assert band_rows(wm, bp) == {"far_until_row": 180.0, "close_from_row": 540.0}
    far, mid, close = (group_lights([_L(400, y)], wm, bands=bp)[0] for y in (100, 300, 700))
    # far: one light with a glow is one vehicle; middle: one light is one vehicle
    assert (far.band, far.count_min, far.count_max) == ("far", 1, 1) and far.score >= 0.4
    assert (mid.band, mid.count_min, mid.count_max) == ("middle", 1, 1)
    # close: a vehicle's shape should be visible, so lights alone are suspect and may count as nothing
    assert close.band == "close" and "close_without_shape" in close.flags and close.count_min == 0 and close.score < 0.2
    # A pair of close-together lights in the middle band is one vehicle.
    pair = next(g for g in group_lights([_L(300, 300), _L(326, 300)], wm, bands=bp) if g.kind == "pair")
    assert (pair.count_min, pair.count_max) == (1, 1)
    # Without a width map nothing is judged by distance.
    assert group_lights([_L(400, 300)], None, bands=bp)[0].band is None


def test_long_thin_lights_are_scored_down():
    wm = WidthMap(0.1, 6, 50, "test")
    rail = Light((300, 298, 360, 302), 330, 300, 240, 255.0, 250.0, 4 / 60)   # lit railing: 60 x 4 px
    g = group_lights([rail], wm)[0]
    assert "line_shaped" in g.flags and g.score < 0.2


def test_glare_of_close_headlights_is_found():
    g = np.full((H, W), 40, np.uint8)
    cv2.circle(g, (320, 250), 45, 255, -1)            # wider than the top-hat kernels: missed before
    lights = light_components(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    big = [li for li in lights if li.area > 3000]
    assert len(big) == 1 and is_glare(big[0])
    grp = group_lights(big, WidthMap(0.1, 6, 50, "test"))[0]
    assert "glare" in grp.flags and grp.score >= 0.5


def test_background_reading_persistent_attached_recurring_occasional_rare():
    masks = []
    for k in range(20):
        m = np.zeros((100, 200), np.uint8)
        m[10:16, 10:16] = 1                           # lamp: always lit
        if k % 3 == 0:
            m[50:54, 50:54] = 1                       # lit in 7 of 20 images (35%): recurring
        if k % 7 == 0:
            m[50:54, 150:154] = 1                     # 3 of 20 (15%): occasional
        masks.append(m)
    bp = BackgroundParams(min_images=8)
    bg = build_background(masks, p=bp)

    def light(x1, y1, x2, y2):
        return Light((x1, y1, x2, y2), (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1), 255.0, 250.0, 1.0)

    assert bg.reading(light(10, 10, 16, 16), bp)[0] == "persistent"
    assert bg.reading(light(18, 10, 22, 14), bp)[0] == "attached"          # a fragment of the lamp's glow
    assert bg.reading(light(50, 50, 54, 54), bp) == ("recurring", 0.35)
    assert bg.reading(light(150, 50, 154, 54), bp)[0] == "occasional"
    assert bg.reading(light(100, 80, 104, 84), bp) == ("rare", 0.0)
    # Large glare touching the lamp is not explained by touching it: it is usually headlights near the lamp.
    glare = Light((17, 5, 60, 45), 38, 25, 1600, 255.0, 252.0, 0.93)
    assert is_glare(glare) and bg.reading(glare, bp)[0] != "attached"
    # The operator says the recurring spot is not a background light: its history stops counting against lights.
    bg2 = build_background(masks, labels=[("not_persistent", [[48, 48], [56, 48], [56, 56], [48, 56]])], p=bp)
    assert bg2.reading(light(50, 50, 54, 54), bp)[0] == "rare"


def test_band_setting_triggers_reanalysis(mission):
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    ns = NightService(m)
    view = ViewService(m).views()[0]["id"]
    b = ns.bands(view)
    assert b["operator_set"] is False and b["width_map"] is not None and b["rows"] is not None
    with pytest.raises(ValueError):
        ns.set_bands(view, 80, 40, "lior")
    ns.set_bands(view, 10, 20, "lior")                # everything below row 140 is now close
    assert ns.refresh()["images_analysed"] == 10
    groups = [g for v in ns._versions(view, "night") for g in ns.current_groups(v) if g["is_candidate"]]
    assert groups and all(g["band"] == "close" and g["setting_id"] is not None for g in groups)
    assert all("close_without_shape" in json.loads(g["flags"]) for g in groups)
    assert ns.refresh()["images_analysed"] == 0      # nothing changed: nothing re-analysed


def test_vehicle_map_from_daytime_boxes():
    vm = vehicle_map([(100, 100, 140, 124), (300, 200, 360, 236)], (360, 640), dilate_px=15)
    assert on_vehicle_map(vm, (110, 105, 130, 115)) and not on_vehicle_map(vm, (500, 20, 510, 30))
    d = map_distance(vm)
    wm = WidthMap(0.1, 6, 50, "test")                 # 36 px wide at row 300
    assert not off_vehicle_map(d, (370, 290, 380, 300), wm)        # just outside: the map has gaps
    assert off_vehicle_map(d, (560, 290, 570, 300), wm)            # far from any place a vehicle was seen


def test_operator_decisions_at_a_place_apply_to_other_images(mission):
    m, src = mission
    for k in range(6):
        write(src, f"CAM1_{ts(k)}.jpg", day_frame(k))
    lit = (1, 5, 9)                                    # a fixed light (a reflection on a sign) in 3 of 10 night images
    for k in range(10):
        write(src, f"CAM1_{ts(100 + k)}.jpg", night_frame(k, singles=[(500, 250)] if k in lit else []))
    cycle(m)
    ViewService(m).assign_or_propose()
    Processor(m, RedBoxDetector()).process_all()
    # Not penalised for recurring here, so it is offered as a candidate in each image.
    bp = BackgroundParams(recurring_share=0.95, occasional_share=0.95)
    ns = NightService(m, bg_params=bp)
    view = ViewService(m).views()[0]["id"]
    ids = [vid_of(m, f"CAM1_{ts(100 + k)}.jpg") for k in lit]
    bg = ns.current_background(view, "night")
    for v in ids:
        m.con.execute("BEGIN"); ns.analyse(v, bg); m.con.execute("COMMIT")
    rs = ReviewService(m)

    def at_place(v):
        return [c for c in rs.candidates(v) if c.box[0] <= 500 <= c.box[2] and c.box[1] <= 250 <= c.box[3]]

    assert all(at_place(v) and at_place(v)[0].status == "unresolved" for v in ids)
    for v in ids[:2]:
        rs.reject_candidate(v, at_place(v)[0].uid, "lior", note="reflection on the sign")
    stats = NightService(m, bg_params=bp).refresh(with_decisions=True)
    assert stats["images_analysed"] == 10              # the view's latest decision changed
    assert at_place(ids[2]) == []                       # no longer offered as a candidate
    g = next(g for g in NightService(m).current_groups(ids[2]) if g["x1"] <= 500 <= g["x2"] and g["y1"] <= 250 <= g["y2"])
    assert "rejected_here_before" in json.loads(g["flags"]) and "2 times" in g["alternatives"]
    # The rejected images keep their own decision; it is not counted against themselves.
    assert [c.status for c in rs.candidates(ids[0], include_explained=True) if c.box[0] <= 500 <= c.box[2]] == ["rejected"]
    assert NightService(m, bg_params=bp).refresh(with_decisions=True)["images_analysed"] == 0


def test_nearest_view_for_images_left_out_of_views(mission):
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    ns = NightService(m)
    v = vid_of(m, f"CAM1_{ts(100)}.jpg")
    view = ViewService(m).views()[0]["id"]
    assert ns.nearest_view(v) == view
    assert ns.nearest_view(v, min_similarity=1.01) is None


def test_night_review_queue_and_accuracy(mission):
    from trafficwb.night_review import NightReview
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    nr = NightReview(m)
    q = nr.queue()
    assert q["total"] >= 3 and all(it["status"] == "unresolved" for it in q["items"])
    dist = [round(abs((it["score"] or 0) - 0.5), 2) for it in q["items"]]
    assert dist == sorted(dist)                                # most uncertain first
    assert all(it["condition"] in ("night", "dusk") for it in q["items"])
    acc = nr.accuracy()
    assert acc["overall"]["answered"] == 0 and acc["overall"]["vehicle_share"] is None
    assert acc["overall"]["open"] == q["total"]
    rs = ReviewService(m)
    a, b, c = q["items"][:3]
    rs.confirm_candidate(a["version_id"], a["uid"], "lior")
    rs.confirm_candidate(b["version_id"], b["uid"], "lior")
    rs.reject_candidate(c["version_id"], c["uid"], "lior")
    acc = nr.accuracy()
    assert (acc["overall"]["vehicle"], acc["overall"]["not_vehicle"]) == (2, 1)
    assert acc["overall"]["vehicle_share"] == round(2 / 3, 3)
    assert acc["overall"]["open"] == q["total"] - 3
    assert sum(r["answered"] for r in acc["by_band"].values()) == 3
    assert nr.queue()["total"] == q["total"] - 3
    answered = nr.queue(status="answered")
    assert answered["total"] == 3 and {it["status"] for it in answered["items"]} == {"confirmed", "rejected"}
    band = a["band"] or "none"
    assert all((it["band"] or "none") == band for it in nr.queue(band=band, status="all")["items"])
    assert nr.queue(flag="no_such_flag")["total"] == 0
    with pytest.raises(ValueError):
        nr.queue(status="maybe")


def test_night_review_api(mission, tmp_path):
    from fastapi.testclient import TestClient
    from trafficwb.web.app import create_app
    m, src = _night_mission(mission)
    Processor(m, RedBoxDetector()).process_all()
    h = {"X-TWB": "1"}
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False, stability_seconds=0))
    c.post("/api/setup", json={"username": "lior", "display_name": "Lior", "password": "longpassword"}, headers=h)
    c.post("/api/login", json={"username": "lior", "password": "longpassword"}, headers=h)
    r = c.get(f"/api/missions/{m.id}/night/queue?limit=5", headers=h)
    assert r.status_code == 200, r.text
    q = r.json()
    assert q["total"] >= 3 and len(q["items"]) <= 5
    it = q["items"][0]
    box = ",".join(f"{v:.1f}" for v in it["box"])
    crop = c.get(f"/api/missions/{m.id}/crops/{it['version_id']}?box={box}&h=120&context=40&mark=true", headers=h)
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    r = c.post(f"/api/missions/{m.id}/items/{it['version_id']}/candidates/{it['uid']}", json={"action": "confirm"}, headers=h)
    assert r.status_code == 200, r.text
    acc = c.get(f"/api/missions/{m.id}/night/accuracy", headers=h).json()
    assert acc["overall"]["vehicle"] == 1
    assert c.get(f"/api/missions/{m.id}/night/queue?status=nope", headers=h).status_code == 400
