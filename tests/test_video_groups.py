"""Videos that are parts of one recording: suggested, merged with one step, analysed as one job, and a target that
crosses from one part into the next counted once."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import car_scene, mission, RedDetector  # noqa: E402

from trafficwb.video_catalog import CatalogError, VideoCatalog  # noqa: E402
from trafficwb.video_groups import part_of  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")

SLOW = ((-10, 80, 50),)           # one car crossing slowly, still in the picture when the first part ends


def test_part_numbers_in_names():
    assert part_of("a/drive_001.mp4") == ("a/drive", 1) and part_of("a/drive-2.mp4") == ("a/drive", 2)
    assert part_of("drive (3).mp4") == ("./drive", 3) and part_of("trip_part4.mov") == ("./trip", 4)
    assert part_of("GH020123.MP4") == ("./GH0123", 2)
    assert part_of("CAM1_20261021T120000.mp4") is None and part_of("plain.mp4") is None


def test_parts_are_suggested_merged_analysed_together_and_a_crossing_target_counts_once(tmp_path):
    files = {"CAM1_20261021T120000Z.mp4": (3.0, lambda t: car_scene(t, SLOW), False),
             "CAM1_20261021T120003Z.mp4": (3.0, lambda t: car_scene(t + 3.0, SLOW), False),
             "CAM1_20261021T130000Z.mp4": (3.0, car_scene, False)}
    m = mission(tmp_path, files)
    cat = VideoCatalog(m)
    ov = cat.overview()
    a, b, c = [v["id"] for v in ov["videos"]]
    assert ov["suggestions"] == [{"members": [a, b], "reason": "each starts where the one before ends"}]
    with pytest.raises(CatalogError, match="two or more"):
        cat.merge([a], "lior")
    g = cat.merge([b, a], "lior")
    assert g["members"] == [a, b] and g["name"].endswith("(2 parts)")
    ov = cat.overview()
    assert not ov["suggestions"] and {v["id"]: v["part"] for v in ov["videos"]} == {a: 1, b: 2, c: None}
    with pytest.raises(CatalogError, match="Already part"):
        cat.merge([b, c], "lior")

    ids = cat.queue([b])                              # ticking one part analyses the whole recording, in order
    assert ids == [a, b]
    VideoProcessor(m, RedDetector(), {}, only=set(ids)).process_all()
    svc = VideoService(m)
    evs = svc.events(group_id=g["id"])
    assert len(evs) == 2 and {e.source_version_id for e in evs} == {a, b}
    later = next(e for e in evs if e.source_version_id == b)
    assert later.continues == next(e for e in evs if e.source_version_id == a).id
    assert svc.counts(group_id=g["id"])["total"]["counted"] == {"car": 1}

    cat.separate(g["id"])                              # separated: each part counts on its own again
    assert svc.counts()["total"]["counted"] == {"car": 2}
    assert cat.overview()["suggestions"][0]["members"] == [a, b]
