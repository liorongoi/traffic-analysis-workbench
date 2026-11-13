"""Phase 6c: the summary video (targets of a long video together over its empty road, with their real times)."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import H, W, mission, run  # noqa: E402

from trafficwb.video_review import VideoService  # noqa: E402
from trafficwb.video_synopsis import Strip, place, render, strips  # noqa: E402

av = pytest.importorskip("av")


def one_by_one(t):
    """Four cars, each alone, 4 s apart: the summary can show them together."""
    import cv2
    import numpy as np
    img = np.full((H, W, 3), 100, np.uint8)
    for k in range(4):
        x = int(-40 + 100 * (t - 4 * k))
        if -40 <= x <= W:
            cv2.rectangle(img, (x, 20 + 40 * k), (x + 40, 44 + 40 * k), (0, 0, 255), -1)
    return img


def test_placing_keeps_order_and_avoids_overlap():
    a = Strip(1, "E-1", "car", 0, [i / 10 for i in range(20)], [[0, 0, 10, 10]] * 20)
    b = Strip(2, "E-2", "car", 50, [50 + i / 10 for i in range(20)], [[0, 0, 10, 10]] * 20)      # same place
    c = Strip(3, "E-3", "car", 90, [90 + i / 10 for i in range(20)], [[100, 0, 110, 10]] * 20)  # elsewhere
    length = place([a, b, c], "calm")
    assert a.start == 0 and b.start >= 1.9 and c.start == b.start and length < 5   # 100 s into under 5 s


def test_summary_video_of_a_recording(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (16.0, one_by_one, False)})
    run(m)
    svc = VideoService(m)
    evs = svc.events()
    assert len(evs) == 4 and len(strips(svc, evs)) == 4
    out = tmp_path / "s.mp4"
    idx = render(svc, evs[0].source_version_id, evs, out)
    assert idx["targets"] == 4 and idx["length_s"] < 8 and idx["original_s"] > 14
    assert max(len(s) for s in idx["per_second"]) >= 3                # shown together
    with av.open(str(out)) as c:
        n = sum(1 for _ in c.decode(video=0))
    assert abs(n - idx["length_s"] * 10) <= 2
    # over the API, in the background
    from fastapi.testclient import TestClient

    from trafficwb.web.app import create_app
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    h = {"X-TWB": "1"}
    url = f"/api/missions/{m.id}/recordings/{evs[0].source_version_id}/synopsis"
    r = c.post(url, json={"busy": "busy"}, headers=h).json()
    for _ in range(200):
        if r["state"] != "running":
            break
        time.sleep(0.1)
        r = c.post(url, json={"busy": "busy", "status_only": True}, headers=h).json()
    assert r["state"] == "done", r
    v = c.get(f"/api/missions/{m.id}/synopsis/{r['key']}.mp4")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    assert c.post(url, json={"busy": "busy"}, headers=h).json()["state"] == "done"      # kept: made once


def test_a_busy_road_never_makes_the_summary_longer():
    from trafficwb.video_synopsis import Strip, place
    # 40 cars, one every second, all on the same spot for 5 s: they cannot avoid each other.
    ss = [Strip(i, f"E-{i}", "car", float(i), [i + k / 10 for k in range(50)], [[10, 10, 50, 40]] * 50)
          for i in range(40)]
    length = place(ss, "calm")
    assert length <= 39 + 5 + 0.1
    assert [s.start for s in ss] == sorted(s.start for s in ss)
