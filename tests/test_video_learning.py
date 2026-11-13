"""Learning from video (phase 3): reviewed events give training frames, approved recordings teach background,
people are learnt as COCO person, and Run mode "full auto" runs rounds on its own."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_learning import _pack  # noqa: E402
from test_video import RedDetector, W  # noqa: E402
from test_video import mission as video_mission  # noqa: E402
from test_video_targets import both, crowd  # noqa: E402

from trafficwb.learning import FRAMES_PER_EVENT, LearningError, LearningService, trainable  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")


def analysed(tmp_path, files):
    m = video_mission(tmp_path, files)
    m.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    VideoProcessor(m, both(), {}).process_all()
    return m


def test_people_are_learnt_as_coco_person_and_sam_only_classes_are_not():
    assert trainable("car") == "car" and trainable("pedestrian") == "person" and trainable("person") == "person"
    assert trainable("tractor") is None and trainable(None) is None


def test_reviewed_events_give_frames_and_an_approved_recording_teaches_background(tmp_path):
    m = analysed(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    ls, svc = LearningService(m), VideoService(m)
    (vid,) = [r["id"] for r in svc.recordings()]
    assert ls.review_kind(vid) == "none" and ls.video_frames(vid) == ("none", [])
    cars = sorted((e for e in svc.events() if e.cls == "car"), key=lambda e: e.first_t)
    walker = next(e for e in svc.events() if e.cls == "pedestrian")
    assert len(cars) == 3

    # One event confirmed: a partial review. Only its boxes are labels, on its best frame and a few more.
    svc.revise(cars[0].id, "confirm", "lior")
    kind, frames = ls.video_frames(vid)
    assert kind == "partial" and len(frames) == FRAMES_PER_EVENT
    assert all(len(f["positives"]) == 1 and f["positives"][0]["cls"] == "car" and not f["ignore_boxes"] for f in frames)
    assert any(abs(f["t"] - cars[0].thumb_t) < 1e-3 for f in frames)

    # Approving needs every event decided.
    with pytest.raises(LearningError, match="not decided"):
        ls.approve_recording(vid, "lior")
    svc.revise(cars[1].id, "reject", "lior")
    svc.revise(cars[2].id, "confirm", "lior")
    svc.revise(walker.id, "confirm", "lior")
    assert ls.approve_recording(vid, "lior")["review"] == "full"
    kind, frames = ls.video_frames(vid)
    assert kind == "full"
    times = [f["t"] for f in frames]
    assert times == sorted(times) and all(b - a >= 0.4 - 1e-6 for a, b in zip(times, times[1:]))
    labels = [p for f in frames for p in f["positives"]]
    assert {p["cls"] for p in labels} == {"car", "pedestrian"}
    assert not any(p["uid"].startswith(cars[1].key + "@") for p in labels)        # rejected: background
    assert any(abs(t) < 0.3 for t in times) and any(abs(t - 5.0) < 0.3 for t in times)   # empty road too

    ds = ls.create_dataset("v1", "lior")
    assert ds["stats"]["frames"] == len(frames) and ds["stats"]["recordings"] == 1 and not ds["leakage"]
    mem = ls.members(ds["id"])
    assert len(mem) == len(frames) and len({r["sha256"] for r in mem}) == len(mem)
    assert len({r["split"] for r in mem}) == 1                  # frames of one recording and day never split apart

    folder = ls.export_dataset(ds["id"])
    ann = json.loads((folder / f"annotations_{mem[0]['split']}.json").read_text(encoding="utf-8"))
    assert {"id": 6, "name": "person"} in ann["categories"]
    assert {a["category_id"] for a in ann["annotations"]} == {2, 6}
    img = ann["images"][0]
    assert img["frame_t"] is not None and img["width"] == W and (folder / img["file_name"]).is_file()
    assert img["unlabelled_area"] == "background inside included regions"

    # Evaluation on frames. The red-box detector looks for vehicles only, so the people are neither found nor
    # missed; it finds the rejected car too, which counts against it where the recording is approved.
    ev = ls.evaluate(ds["id"], RedDetector(), "lior", split=mem[0]["split"])
    o = ev["report"]["overall"]
    assert o["recall"] == 1.0 and o["false_positives_full"] > 0 and o["precision"] < 1.0

    # Reopening the recording makes it a partial review again.
    assert ls.reopen_recording(vid, "lior")["review"] == "partial"


def test_one_step_confirms_the_events_left_and_approves(tmp_path):
    m = analysed(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    ls, svc = LearningService(m), VideoService(m)
    (vid,) = [r["id"] for r in svc.recordings()]
    first = svc.events()[0]
    svc.revise(first.id, "reject", "lior")
    assert ls.approve_recording(vid, "lior", confirm_rest=True)["review"] == "full"
    st = {e.id: e.status for e in svc.events()}
    assert st.pop(first.id) == "rejected" and set(st.values()) == {"confirmed"}


def test_video_datasets_combine_with_still_image_datasets(tmp_path):
    m = analysed(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    svc, ls = VideoService(m), LearningService(m)
    for e in svc.events():
        svc.revise(e.id, "confirm", "lior")
    ds = ls.create_dataset("v1", "lior")
    assert any(d["mission_id"] == m.id for d in ls.all_datasets())
    items, sources = ls.combine([(m.id, ds["id"])])
    assert len(items) == ds["stats"]["images"] and sources[0]["name"] == "v1"


# ------------------------------------------------------------------ Run mode "full auto"

class FakeJob:
    def __init__(self, result):
        self.result = result


def test_full_auto_starts_a_round_when_enough_reviews_have_built_up(tmp_path, monkeypatch):
    from trafficwb import training_env
    from trafficwb.web.jobs import JobManager
    m = analysed(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    jobs = JobManager(tmp_path / "data", detector_factory=lambda: (RedDetector(), {}))
    assert jobs.auto_learn(m.id) == {"state": "manual"}
    m.update_config("lior", "run mode", run_mode="auto", auto_min_reviews=3, auto_epochs=2)
    svc = VideoService(m)
    evs = svc.events()
    svc.revise(evs[0].id, "confirm", "lior")
    st = jobs.auto_learn(m.id)
    assert st["state"] == "waiting" and st["new_reviews"] == 1 and st["needed"] == 3

    for e in evs[1:]:
        svc.revise(e.id, "confirm", "lior")
    # Training not available here: the loop waits, and says why once.
    monkeypatch.setattr(training_env, "status", lambda wait=False: {"state": "missing", "reason": "no PyTorch"})
    assert jobs.auto_learn(m.id)["state"] == "blocked"
    assert jobs.auto_learn(m.id)["reason"] == "no PyTorch"
    ls = LearningService(m)
    assert [h["action"] for h in ls.auto_history()] == ["waiting"]

    started = {}
    monkeypatch.setattr(training_env, "status", lambda wait=False: {"state": "ready", "coco_weights": True})
    monkeypatch.setattr(jobs, "start", lambda mid, kind, **p: started.update(p, kind=kind) or FakeJob({}))
    FakeJob.snapshot = lambda self: {}
    monkeypatch.setattr(LearningService, "_bucket", staticmethod(lambda seed, key: 0.1))     # the group: train
    st = jobs.auto_learn(m.id)
    assert st["state"] == "training" and started["kind"] == "train" and started["auto"] and started["epochs"] == 2
    assert started["start_from"] == "coco" and started["datasets"][-1]["dataset_id"] == st["dataset_id"]
    assert ls.reviews_since_last_dataset() == 0                 # the new dataset version took them in
    assert jobs.auto_learn(m.id)["state"] == "waiting"


def test_full_auto_adopts_a_better_model_for_the_mission_only(tmp_path):
    from trafficwb.web.jobs import JobManager
    m = analysed(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)})
    models = tmp_path / "models"
    _pack(models, "yolox-s-tuned", "a1")
    _pack(models, "yolox-s-tuned", "a2")
    jobs = JobManager(tmp_path / "data", detector_factory=lambda: (RedDetector(), {}))
    ls = LearningService(m)
    p = {"models_root": str(models), "previous_model": None}
    jobs._auto_decide(m, ls, FakeJob({"improved": False}), 1, {"name": "yolox-s-tuned", "version": "a1"}, p)
    assert m.config().vehicle_model is None and ls.auto_history()[0]["action"] == "kept"
    job = FakeJob({"improved": True})
    jobs._auto_decide(m, ls, job, 2, {"name": "yolox-s-tuned", "version": "a2"}, p)
    assert m.config().vehicle_model == job.result["adopted"] == "v auto 2"
    h = ls.auto_history()[0]
    assert h["action"] == "adopted" and h["detail"]["saved_as"] == "v auto 2" and h["detail"]["previous_model"] is None
