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
