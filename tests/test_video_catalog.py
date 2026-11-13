"""The video catalog: list the videos of the source folders, analyse only the ticked ones, analyse one again."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import RedDetector, car_scene, mission  # noqa: E402

from trafficwb.video_catalog import CatalogError, VideoCatalog  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")

FILES = {f"CAM1_2026102{d}T120000Z.mp4": (3.0, car_scene, False) for d in (1, 2, 3)}


def test_the_catalog_lists_every_video_and_ticks_the_new_ones(tmp_path):
    m = mission(tmp_path, FILES)
    cat = VideoCatalog(m)
    vids = cat.items()
    assert [v["rel_path"] for v in vids] == sorted(FILES)
    assert all(v["state"] == "queued" and v["selected"] and not v["analysed"] for v in vids)
    assert all(abs(v["duration_s"] - 3.0) < 0.1 and v["width"] == 320 for v in vids)
    assert cat.thumbnail(vids[0]["id"])[:2] == b"\xff\xd8"                  # a JPEG
    with pytest.raises(CatalogError, match="at least one"):
        cat.queue([])


def test_only_the_ticked_videos_are_analysed_and_one_can_be_analysed_again(tmp_path):
    m = mission(tmp_path, FILES)
    cat = VideoCatalog(m)
    first, second, third = [v["id"] for v in cat.items()]
    ids = cat.queue([first, third])
    VideoProcessor(m, RedDetector(), {}, only=set(ids)).process_all()
    st = {v["id"]: v for v in cat.items()}
    assert st[first]["analysed"] and st[third]["analysed"] and st[second]["state"] == "queued"
    assert st[first]["events"] == 1 and not st[first]["selected"] and st[second]["selected"]
    assert {e.source_version_id for e in VideoService(m).events()} == {first, third}

    # Ticking an analysed video analyses it again; its newest analysis is the one that counts.
    ids = cat.queue([first, second])
    VideoProcessor(m, RedDetector(), {}, only=set(ids)).process_all()
    st = {v["id"]: v for v in cat.items()}
    assert all(v["analysed"] for v in st.values())
    c = VideoService(m).counts()
    assert c["total"]["machine"] == {"car": 3}
