"""Reviewing in the player: every event's boxes over the recording, fixing a box on a frame, adding a target the
analysis missed, and analysing new videos with the mission's model as they arrive."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import RedDetector, car_scene, mission  # noqa: E402

from trafficwb.learning import LearningService  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoError, VideoService  # noqa: E402

av = pytest.importorskip("av")


def analysed(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (3.0, car_scene, False)})
    VideoProcessor(m, RedDetector(), {}).process_all()
    return m


def test_the_overlay_has_every_event_with_its_boxes_and_a_fixed_box_replaces_the_machines(tmp_path):
    m = analysed(tmp_path)
    svc = VideoService(m)
    (vid,) = [r["id"] for r in svc.recordings()]
    ov = svc.overlay(vid)
    assert ov["recording"]["width"] == 320 and len(ov["events"]) == 1
    ev = ov["events"][0]
    assert ev["origin"] == "machine" and len(ev["boxes"]) > 5 and all(len(b) == 5 for b in ev["boxes"])
    t, *box = ev["boxes"][3]

    fixed = [box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5]
    e = svc.revise(ev["id"], "set_box", "lior", t=t, box=fixed)
    assert len(e.fixes) == 1 and e.fixes[0]["box"] == fixed
    boxes = {round(b[0], 3): b[1:] for b in svc.overlay(vid)["events"][0]["boxes"]}
    assert boxes[round(t, 3)] == fixed and len(boxes) == len(ev["boxes"])      # replaced, not added
    assert any(o["found_by"] == "operator" for o in svc.observations(ev["id"]))
    # A box fix is a review of the event: its frame teaches with the operator's box.
    kind, frames = LearningService(m).video_frames(vid)
    assert kind == "partial" and any(p["box"] == fixed for f in frames for p in f["positives"])

    with pytest.raises(VideoError, match="too small"):
        svc.revise(ev["id"], "set_box", "lior", t=t, box=[10, 10, 11, 11])
    assert not svc.revise(ev["id"], "clear_box", "lior", t=t).fixes
    with pytest.raises(VideoError, match="no box of yours"):
        svc.revise(ev["id"], "clear_box", "lior", t=t)


def test_a_missed_target_is_added_confirmed_counted_and_taught(tmp_path):
    m = analysed(tmp_path)
    svc = VideoService(m)
    (vid,) = [r["id"] for r in svc.recordings()]
    with pytest.raises(VideoError, match="Class"):
        svc.add_target(vid, 1.0, [10, 10, 60, 40], "spaceship", "lior")
    h = svc.add_target(vid, 1.0, [10, 10, 60, 40], "truck", "lior")
    assert h.key == "H-0001" and h.origin == "human" and h.status == "confirmed" and h.cls == "truck"
    h = svc.revise(h.id, "set_box", "lior", t=2.0, box=[60, 10, 110, 40])          # a box on a later frame extends it
    assert h.last_t >= 1.99 and len(h.fixes) == 2
    assert svc.counts(vid)["total"]["counted"] == {"car": 1, "truck": 1}
    # Approving: the operator's target teaches on its own frames; between them it is neither label nor background.
    ls = LearningService(m)
    ls.approve_recording(vid, "lior", confirm_rest=True)
    _, frames = ls.video_frames(vid)
    trucks = [p for f in frames for p in f["positives"] if p["cls"] == "truck"]
    assert len(trucks) == 2
    mid = [f for f in frames if 1.2 < f["t"] < 1.8]
    assert all(any(b[0] > 10 and b[2] < 110 and b[1] == 10 for b in f["ignore_boxes"]) for f in mid)


def test_new_videos_are_analysed_with_the_missions_model_when_set(tmp_path):
    from trafficwb.web.jobs import JobManager
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (3.0, car_scene, False)})
    jobs = JobManager(tmp_path / "data", detector_factory=lambda: (RedDetector(), {}), monitor_interval=3600)
    try:
        assert not jobs.watch_new_videos(m.id) and m.id not in jobs.monitors
        m.update_config("lior", "analyse new", analyse_new=True)
        assert jobs.watch_new_videos(m.id) and m.id in jobs.monitors
        m.update_config("lior", "wait", analyse_new=False)
        assert not jobs.watch_new_videos(m.id) and m.id not in jobs.monitors
        m.update_config("lior", "auto", run_mode="auto")                # full auto always analyses new videos
        assert jobs.watch_new_videos(m.id) and jobs.resume_monitoring() == [m.id]
    finally:
        jobs.shutdown()
