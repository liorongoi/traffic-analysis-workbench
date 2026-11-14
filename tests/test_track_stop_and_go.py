"""Stop and go, and the operator's "frames before a target counts": a target that drives, stops and drives on keeps
one ID, also when something passes in front of it while it stands; the player gets each box's detector score."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import H, W, RedDetector, car_scene, mission  # noqa: E402

from trafficwb.tracker import Tracker, TrackerParams  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")


def stop_and_go(t: float) -> list[float]:
    """Drives right for 2 s, stands 5 s, drives on."""
    x = 40 + 40 * min(t, 2.0) + 40 * max(0.0, t - 7.0)
    return [x, 60, x + 50, 90]


def frame_with(boxes) -> np.ndarray:
    img = np.full((H, W, 3), 90, np.uint8)
    for (x1, y1, x2, y2), colour in boxes:
        img[int(y1):int(y2), int(x1):int(x2)] = colour
        img[int(y1) + 4:int(y1) + 10, int(x1) + 4:int(x2) - 4] = (20, 20, 20)
    return img


def test_a_target_that_stops_and_drives_on_keeps_its_id():
    tr = Tracker(W, H, TrackerParams(min_hits=3))
    keys = set()
    for i in range(60):                                       # 12 s at 5 analysed frames a second
        t = round(i * 0.2, 3)
        b = stop_and_go(t)
        obs, closed, _ = tr.update(t, [(b, "car", 0.9)], image=frame_with([(b, (40, 40, 200))]))
        keys |= {o[0].key for o in obs}
        assert not closed
    assert len(keys) == 1


@pytest.mark.parametrize("hold", [6.0, 0.0])
def test_a_standing_target_hidden_by_a_passing_truck_is_taken_up_where_it_stood(hold):
    """Hidden for 4 s while it waits (longer than the 1.5 s gap and the 1.5 s re-link time): held where it stood, it
    is the same target when it shows again and drives on. With the hold off it would be a new target."""
    tr = Tracker(W, H, TrackerParams(min_hits=3, still_hold_s=hold))
    keys = []
    for i in range(60):
        t = round(i * 0.2, 3)
        b = stop_and_go(t)
        hidden = 3.0 <= t < 7.0
        img = frame_with([] if hidden else [(b, (40, 40, 200))])
        obs, _, _ = tr.update(t, [] if hidden else [(b, "car", 0.9)], image=img)
        keys += [o[0].key for o in obs]
    assert (len(set(keys)) == 1) == (hold > 0)


def test_a_moving_target_lost_for_long_is_not_held():
    """The hold is for targets standing still: one driving away is closed after the usual time."""
    tr = Tracker(W, H, TrackerParams(min_hits=3))
    box = lambda t: [20 + 30 * t, 60, 70 + 30 * t, 90]
    for i in range(8):
        t = i * 0.2
        tr.update(t, [(box(t), "car", 0.9)], image=frame_with([(box(t), (40, 40, 200))]))
    assert not tr.still(tr.tracks[0])
    closed = []
    for t in np.arange(1.6, 5.0, 0.2):
        closed += tr.update(round(float(t), 3), [], image=frame_with([]))[1]
    assert len(closed) == 1 and closed[0].track.t < 1.6


def test_old_checkpoints_resume_without_the_hold():
    tr = Tracker(W, H)
    s = tr.state()
    del s["params"]["still_hold_s"]
    assert Tracker.from_state(s).p.still_hold_s == 0.0


def test_frames_before_a_target_counts_is_the_missions_choice_and_the_player_gets_scores(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (3.0, car_scene, False)})
    assert VideoProcessor(m, RedDetector(), {}).s.tracker.min_hits == 5
    m.update_config("lior", "test", track_min_frames=12)
    assert VideoProcessor(m, RedDetector(), {}).s.tracker.min_hits == 12
    m.update_config("lior", "test", track_min_frames=None)
    VideoProcessor(m, RedDetector(), {}).process_all()
    svc = VideoService(m)
    (vid,) = [r["id"] for r in svc.recordings()]
    ev = svc.overlay(vid)["events"][0]
    assert len(ev["scores"]) == len(ev["boxes"]) and all(s == 0.9 for s in ev["scores"])
    t, *box = ev["boxes"][2]
    svc.revise(ev["id"], "set_box", "lior", t=t, box=[box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3])
    ev = svc.overlay(vid)["events"][0]
    i = [round(b[0], 3) for b in ev["boxes"]].index(round(t, 3))
    assert ev["scores"][i] is None and len(ev["scores"]) == len(ev["boxes"])   # a box you drew has no score


def test_the_setting_through_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from trafficwb.web.app import create_app
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (1.0, car_scene, False)})
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    h = {"X-TWB": "1"}
    url = f"/api/missions/{m.id}/detector"
    r = c.put(url, json={"detector": "standard", "track_min_frames": 8}, headers=h)
    assert r.status_code == 200 and r.json()["track_min_frames"] == 8
    assert c.get(f"/api/missions/{m.id}", headers=h).json()["config"]["track_min_frames"] == 8
    assert c.put(url, json={"detector": "standard", "track_min_frames": 40}, headers=h).status_code == 400
    assert c.put(url, json={"detector": "standard", "track_min_frames": 0}, headers=h).json()["track_min_frames"] is None


def test_one_box_per_target_on_a_frame():
    from trafficwb.tracker import fuse
    van, half, side = [100, 50, 300, 250], [150, 52, 290, 120], [240, 120, 300, 250]
    other_car, person = [400, 100, 500, 180], [120, 60, 160, 200]
    d = [(van, "car", 0.9), (half, "car", 0.4), (side, "truck", 0.35), (other_car, "car", 0.8), (person, "person", 0.9)]
    kept = fuse(d, 0.7)
    assert [k[0] for k in kept] == [van, other_car, person]           # a person is never fused into a car
    assert kept[0][0] is van                                          # the very box given
    # a weak box over two cars gives way to the sure car inside it
    assert [k[0] for k in fuse([([0, 0, 200, 100], "car", 0.3), ([0, 10, 90, 100], "car", 0.9)], 0.7)] == [[0, 10, 90, 100]]
    # a car and a truck on one van: the surer stays
    assert [k[1] for k in fuse([([0, 0, 100, 100], "truck", 0.5), ([2, 2, 100, 98], "car", 0.8)], 0.7)] == ["car"]
