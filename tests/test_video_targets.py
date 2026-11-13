"""Video like still images: the standard detector, SAM 3.1 or both, and targets of the mission's own (people)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import H, RedDetector, W, mission, summary  # noqa: E402

from trafficwb.combined import CombinedDetector, merge  # noqa: E402
from trafficwb.detect import (Detection, _merge_across_classes, coco_id, family,  # noqa: E402
                              standard_classes)
from trafficwb.process import RunCancelled, RunControl  # noqa: E402
from trafficwb.store import MissionConfig, validate  # noqa: E402
from trafficwb.tracker import Tracker  # noqa: E402
from trafficwb.video_process import VideoProcessor, VideoSettings  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")


def d(box, cls, score=0.8):
    return Detection(box, cls, score, {cls: score})


# ------------------------------------------------------------------ classes and families

def test_people_are_a_class_the_standard_detector_knows():
    assert coco_id("person") == 0 and coco_id("pedestrian") == 0 and coco_id("tractor") is None
    assert standard_classes(None) == {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    cl = standard_classes(["pedestrian", "tractor"])
    assert cl[0] == "pedestrian" and "tractor" not in cl.values()       # a tractor is found by SAM 3.1 only
    assert family("car") == family("truck") == "vehicle" and family("pedestrian") == "pedestrian"


def test_a_person_on_a_bicycle_stays_two_objects():
    rider, bike = d((10, 0, 40, 60), "pedestrian", 0.7), d((8, 20, 42, 62), "bicycle", 0.9)
    assert len(_merge_across_classes([rider, bike], 0.5)) == 2
    assert len(_merge_across_classes([d((8, 20, 42, 62), "car", 0.6), bike], 0.5)) == 1   # one vehicle, two names
    out = merge([bike], [rider])
    assert sorted(x.found_by for x in out) == ["sam3", "standard"]


def test_tracks_never_cross_families_and_old_checkpoints_still_load():
    tr = Tracker(W, H)
    for k in range(4):
        t = k * 0.2
        tr.update(t, [([10 + k, 20, 40 + k, 80], "bicycle", 0.9), ([12 + k, 10, 38 + k, 78], "pedestrian", 0.8)])
    fams = sorted(t.family for t in tr.tracks)
    assert fams == ["pedestrian", "vehicle"] and all(t.state == "confirmed" for t in tr.tracks)
    assert all(set(t.votes) == ({"bicycle"} if t.family == "vehicle" else {"pedestrian"}) for t in tr.tracks)
    st = tr.state()
    for t in st["tracks"]:
        t.pop("family")                               # a checkpoint written before families existed
    old = Tracker.from_state(st)
    assert all(t.family == "vehicle" for t in old.tracks)


def test_video_missions_may_choose_sam_and_how_often_it_looks(tmp_path):
    cfg = MissionConfig(name="v", input_type="video", mode="object_search", country="IL", iana_zone="Asia/Jerusalem",
                        source_dir=str(tmp_path), include_subdirs=False, processing="one_time",
                        output_dir=str(tmp_path), export_formats=["csv"], detector="both", sam3_interval_s=0.5)
    assert validate(cfg, tmp_path, min_free_bytes=0).ok
    cfg.sam3_interval_s = 30
    assert any("0.2 to 10 seconds" in e for e in validate(cfg, tmp_path, min_free_bytes=0).errors)
    cfg.sam3_interval_s, cfg.video_workers = 1.0, 40
    assert any("1 to 32" in e for e in validate(cfg, tmp_path, min_free_bytes=0).errors)


# ------------------------------------------------------------------ both detectors in a recording

def scene(t: float) -> np.ndarray:
    """A red car crossing fast and a blue pedestrian walking slowly."""
    img = np.full((H, W, 3), 100, np.uint8)
    x = int(-40 + 70 * t)
    cv2.rectangle(img, (x, 40), (x + 40, 64), (0, 0, 255), -1)
    px = int(120 + 8 * t)
    cv2.rectangle(img, (px, 100), (px + 12, 130), (255, 0, 0), -1)
    return img


class BlueSam:
    """Stands in for SAM 3.1 asked for "pedestrian": finds the blue boxes, and counts the frames it saw."""

    def __init__(self):
        self.calls = 0

    def describe(self):
        return {"model": {"name": "sam-fake", "version": "1", "sha256": "s" * 64}, "preprocessing": {}}

    def detect(self, img):
        self.calls += 1
        m = np.all(np.abs(img.astype(int) - (255, 0, 0)) <= 60, axis=2).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "pedestrian", 0.8, {"pedestrian": 0.8})
                for x, y, w, h, a in st[1:] if a > 80]


def both(sam=None):
    return CombinedDetector(RedDetector(), {"license": "test"}, sam or BlueSam(), {"license": "test"})


def test_both_detectors_in_video_find_cars_and_people_and_say_who_found_each(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, scene, False)})
    sam = BlueSam()
    rep = VideoProcessor(m, both(sam), {}).process_all()
    frames = rep[0].frames
    assert frames == 30 and sam.calls == 6                 # SAM 3.1 once a second, the standard detector 5 times
    evs = VideoService(m).events()
    assert sorted(e.cls for e in evs) == ["car", "pedestrian"]
    by = dict(m.con.execute("SELECT o.found_by, COUNT(*) FROM track_observation o GROUP BY o.found_by"))
    assert by.get("sam3") == 6 and by.get("standard", 0) > 10
    walker = next(e for e in evs if e.cls == "pedestrian")
    obs = VideoService(m).observations(walker.id)
    assert {o["found_by"] for o in obs} == {"sam3"} and len(obs) == 6
    pre = m.con.execute("SELECT preprocessing FROM run").fetchone()[0]
    assert '"sam3_every_frames": 5' in pre and '"detector": "both"' in pre


def test_sam_alone_spaces_the_grid_and_people_still_become_events(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, scene, False)})

    class SamOnly(BlueSam):
        prompts = {"pedestrian": ("pedestrian",)}
        engine = object()

    sam = SamOnly()
    rep = VideoProcessor(m, sam, {}, VideoSettings(sam3_interval_s=0.5)).process_all()
    assert rep[0].frames == 12 == sam.calls
    (e,) = VideoService(m).events()
    assert e.cls == "pedestrian"


def test_resume_calls_sam_on_the_same_frames(tmp_path):
    ref = mission(tmp_path / "a", {"CAM1_20261021T120000Z.mp4": (6.0, scene, False)})
    VideoProcessor(ref, both(), {}, VideoSettings(chunk_frames=50)).process_all()

    class CancelOnce(RunControl):
        def __init__(self):
            super().__init__()
            self.n = 0

        def checkpoint(self):
            self.n += 1
            if self.n == 2:
                raise RunCancelled()

    m = mission(tmp_path / "b", {"CAM1_20261021T120000Z.mp4": (6.0, scene, False)})
    VideoProcessor(m, both(), {}, VideoSettings(chunk_frames=7), CancelOnce()).process_all()
    VideoProcessor(m, both(), {}, VideoSettings(chunk_frames=7)).process_all()
    assert summary(m) == summary(ref) and len(summary(ref)) == 2
    q = "SELECT found_by, COUNT(*), ROUND(SUM(t), 3) FROM track_observation GROUP BY found_by ORDER BY found_by"
    assert m.con.execute(q).fetchall() == ref.con.execute(q).fetchall()


def test_review_can_set_a_class_of_the_missions_own(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, scene, False)})
    m.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    VideoProcessor(m, both(), {}).process_all()
    svc = VideoService(m)
    car = next(e for e in svc.events() if e.cls == "car")
    assert "pedestrian" in svc.classes()
    assert svc.revise(car.id, "set_class", "lior", cls="pedestrian").cls == "pedestrian"


# ------------------------------------------------------------------ moving cameras and the Kalman tracker

_rng = np.random.default_rng(7)
WORLD = cv2.GaussianBlur(_rng.integers(0, 255, (H, 1000, 3), dtype=np.uint8), (0, 0), 1.2)
# Another scene after the cut: a coarser texture (fine noise on both sides of a cut matches by chance in 20 % of corners).
WORLD2 = cv2.GaussianBlur(np.random.default_rng(99).integers(0, 255, (H, 1000, 3), dtype=np.uint8), (0, 0), 4)


def panning(t: float, cut_at: float | None = None) -> np.ndarray:
    """A camera panning back and forth fast (up to 250 px/s) over a textured scene, with a car driving slowly."""
    off = int(round(200 + 80 * np.sin(np.pi * t)))
    world = (WORLD2 if cut_at is not None and t >= cut_at else WORLD).copy()
    x = int(300 + 10 * t)
    cv2.rectangle(world, (x, 60), (x + 40, 84), (0, 0, 255), -1)
    return np.ascontiguousarray(world[:, off:off + W])


def test_a_moving_camera_keeps_one_track_where_a_fixed_one_breaks_it(tmp_path):
    fixed = mission(tmp_path / "f", {"CAM1_20261021T120000Z.mp4": (4.0, panning, False)})
    VideoProcessor(fixed, RedDetector(), {}, VideoSettings(camera="fixed")).process_all()
    moving = mission(tmp_path / "m", {"CAM1_20261021T120000Z.mp4": (4.0, panning, False)})
    VideoProcessor(moving, RedDetector(), {}, VideoSettings(camera="moving")).process_all()
    n_tracks = lambda m: m.con.execute("SELECT COUNT(*) FROM track").fetchone()[0]
    assert n_tracks(fixed) > 3                              # the car jumps 50 px a frame: the fixed tracker loses it
    assert n_tracks(moving) == 1 and len(VideoService(moving).events()) == 1
    assert moving.con.execute("SELECT COUNT(*) FROM video_marker").fetchone()[0] == 0   # a pan is not a cut
    assert '"camera": "moving"' in moving.con.execute("SELECT preprocessing FROM run").fetchone()[0]


def test_a_real_cut_still_closes_tracks_on_a_moving_camera(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (4.0, lambda t: panning(t, cut_at=2.0), False)})
    VideoProcessor(m, RedDetector(), {}, VideoSettings(camera="moving")).process_all()
    assert m.con.execute("SELECT COUNT(*) FROM video_marker WHERE kind='scene_cut'").fetchone()[0] == 1


def test_moving_camera_resume_is_identical(tmp_path):
    ref = mission(tmp_path / "a", {"CAM1_20261021T120000Z.mp4": (4.0, panning, False)})
    VideoProcessor(ref, RedDetector(), {}, VideoSettings(camera="moving")).process_all()

    class CancelOnce(RunControl):
        def __init__(self):
            super().__init__()
            self.n = 0

        def checkpoint(self):
            self.n += 1
            if self.n == 1:
                raise RunCancelled()

    m = mission(tmp_path / "b", {"CAM1_20261021T120000Z.mp4": (4.0, panning, False)})
    VideoProcessor(m, RedDetector(), {}, VideoSettings(camera="moving", chunk_frames=6), CancelOnce()).process_all()
    VideoProcessor(m, RedDetector(), {}, VideoSettings(camera="moving", chunk_frames=6)).process_all()
    q = "SELECT COUNT(*), ROUND(SUM(t), 4), ROUND(SUM(x1 + y2), 2) FROM track_observation"
    assert m.con.execute(q).fetchone() == ref.con.execute(q).fetchone()
    assert summary(m) == summary(ref)


def test_weak_boxes_keep_a_track_alive_but_never_start_one():
    tr = Tracker(W, H)
    for k in range(3):
        tr.update(k * 0.2, [([10 + 4 * k, 20, 50 + 4 * k, 44], "car", 0.9)])
    (t1,) = tr.tracks
    obs, _, new = tr.update(0.6, [([22, 20, 62, 44], "car", 0.15), ([200, 100, 240, 124], "car", 0.15)])
    assert not new and len(obs) == 1 and obs[0][0] is t1 and t1.hits == 4      # second pass kept it; no new track
    old = Tracker.from_state({**tr.state(), "params": {k: v for k, v in tr.state()["params"].items()
                                                      if k not in ("algo", "high_score", "low_iou_min", "frame_s", "first_jump")}})
    assert old.p.algo == "iou"                                    # checkpoints from before resume the old way


# ------------------------------------------------------------------ unique counts

def crowd(t: float) -> np.ndarray:
    """Three cars cross one after another (each seen for many frames) and a pedestrian walks."""
    img = np.full((H, W, 3), 100, np.uint8)
    for k, start in enumerate((0.0, 1.5, 3.0)):
        x = int(-40 + 140 * (t - start))
        if -40 < x < W:
            cv2.rectangle(img, (x, 30 + 40 * k), (x + 40, 54 + 40 * k), (0, 0, 255), -1)
    px = int(120 + 8 * t)
    cv2.rectangle(img, (px, 140), (px + 12, 170), (255, 0, 0), -1)
    return img


def test_unique_counts_count_each_target_once_and_follow_review(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    m.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]}, sam3_uncounted=["pedestrian"])
    rep = VideoProcessor(m, both(), {}).process_all()
    svc = VideoService(m)
    c = svc.counts()
    assert c["total"]["machine"] == {"car": 3, "pedestrian": 1}
    assert c["total"]["counted"] == {"car": 3, "pedestrian": 1} and c["total"]["vehicles_counted"] == 3
    assert rep[0].frames == 30                                   # 30 frames, yet 4 targets: one per track
    (r,) = c["recordings"]
    assert sum(b["counts"].get("car", 0) for b in r["per_bucket"]) == 3 and c["uncounted"] == ["pedestrian"]
    cars = [e for e in svc.events() if e.cls == "car"]
    svc.revise(cars[0].id, "reject", "lior")
    svc.revise(cars[1].id, "duplicate", "lior", of=cars[2].key)
    svc.revise(cars[2].id, "split_at", "lior", t=(cars[2].first_t + cars[2].last_t) / 2)
    c = svc.counts()
    assert c["total"]["machine"]["car"] == 3 and c["total"]["counted"]["car"] == 2   # rejected 1, duplicate 1, split +1
    files = svc.export("lior", ["csv"])
    counts_csv = next(p for p in files if p.name.endswith("_counts.csv")).read_text(encoding="utf-8-sig")
    assert '"(all recordings)","","car","yes","3","2"' in counts_csv
    assert any(p.name.endswith("_counts_per_minute.csv") for p in files)


# ------------------------------------------------------------------ parallel segments

def test_plan_cuts_on_the_grid_and_the_last_segment_takes_a_short_tail():
    from trafficwb.video_segments import plan
    segs = plan(6.0, 2.0, 0.6, 0.2)
    assert [(s.own_a, s.own_b) for s in segs] == [(0.0, 2.0), (2.0, 4.0), (4.0, float("inf"))]
    assert segs[0].run_a == 0.0 and segs[1].run_a == 1.4 and segs[1].run_b == 4.6 and segs[2].run_b == float("inf")
    assert [(s.own_a, s.own_b) for s in plan(6.3, 2.0, 0.6, 0.2)] == [(0.0, 2.0), (2.0, 4.0), (4.0, float("inf"))]
    assert len(plan(1.0, 300.0, 3.0, 0.2)) == 1
    assert all(round(s.own_a / 0.2, 6).is_integer() for s in plan(17.0, 2.1, 0.5, 0.2))


def segmented(**kw):
    return VideoSettings(workers=3, segment_s=2.0, overlap_s=0.6, **kw)


def test_segments_side_by_side_find_the_same_targets_and_counts_as_one_worker(tmp_path):
    files = {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)}
    ref = mission(tmp_path / "a", files)
    ref.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    one = VideoProcessor(ref, both(), {}).process_all()
    m = mission(tmp_path / "b", files)
    m.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    par = VideoProcessor(m, both(), {}, segmented()).process_all()
    assert m.con.execute("SELECT COUNT(*) FROM video_segment WHERE state='done'").fetchone()[0] == 3
    assert par[0].frames == one[0].frames == 30
    a, b = VideoService(ref).counts()["total"], VideoService(m).counts()["total"]
    assert b["machine"] == a["machine"] == {"car": 3, "pedestrian": 1}
    ev_a = sorted((e.cls, round(e.first_t, 1), round(e.last_t, 1)) for e in VideoService(ref).events())
    ev_b = sorted((e.cls, round(e.first_t, 1), round(e.last_t, 1)) for e in VideoService(m).events())
    assert ev_b == ev_a
    q = "SELECT found_by, COUNT(*) FROM track_observation GROUP BY found_by ORDER BY found_by"
    assert m.con.execute(q).fetchall() == ref.con.execute(q).fetchall()    # every frame once, SAM on the same frames
    assert par[0].realtime_factor


def test_a_cancelled_segmented_run_keeps_the_segments_it_finished(tmp_path):
    files = {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)}
    ref = mission(tmp_path / "a", files)
    VideoProcessor(ref, both(), {}, segmented()).process_all()

    class CancelLate(RunControl):
        def checkpoint(self):                        # cancel once the first segment is stored
            import threading
            if threading.current_thread() is threading.main_thread():
                raise RunCancelled()

    m = mission(tmp_path / "b", files)
    VideoProcessor(m, both(), {}, VideoSettings(workers=2, segment_s=2.0, overlap_s=0.6), CancelLate()).process_all()
    assert not VideoService(m).events()
    kept = m.con.execute("SELECT COUNT(*) FROM video_segment WHERE state='done'").fetchone()[0]
    assert 0 < kept < 3
    VideoProcessor(m, both(), {}, segmented()).process_all()
    assert summary(m) == summary(ref)
