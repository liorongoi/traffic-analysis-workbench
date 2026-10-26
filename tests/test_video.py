"""Video: timestamps, tracking, events, review, clips, export and search (VID-01..04)."""
import csv
import os
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from trafficwb.detect import Detection
from trafficwb.ingest import cycle
from trafficwb.process import RunCancelled, RunControl
from trafficwb.store import DataRoot, MissionConfig
from trafficwb.tracker import Tracker, TrackerParams
from trafficwb.video import frames, probe
from trafficwb.video_process import VideoProcessor, VideoSettings
from trafficwb.video_review import VideoError, VideoService
from trafficwb.web.app import create_app

av = pytest.importorskip("av")

OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
H1 = {"X-TWB": "1"}
W, H = 320, 180


def car_scene(t: float, cars=((-40, 80, 60),), bg=100) -> np.ndarray:
    """Grey road; each car (x0, y, speed px/s) is a red box 40x24 moving right."""
    img = np.full((H, W, 3), bg, np.uint8)
    for x0, y, v in cars:
        x = int(x0 + v * t)
        cv2.rectangle(img, (x, y), (x + 40, y + 24), (0, 0, 255), -1)
    return img


def make_video(path: Path, dur: float, draw=car_scene, vfr: bool = False) -> list[float]:
    """Write an H.264 file with millisecond timestamps. Returns the frame times written."""
    c = av.open(str(path), "w")
    s = c.add_stream("libx264", rate=25)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.time_base = s.codec_context.time_base = Fraction(1, 1000)
    s.options = {"g": "25", "bf": "0", "crf": "18"}
    t, times = 0.0, []
    while t < dur - 1e-9:
        ms = int(round(t * 1000))
        fr = av.VideoFrame.from_ndarray(draw(ms / 1000), format="bgr24")
        fr.pts, fr.time_base = ms, Fraction(1, 1000)
        for p in s.encode(fr):
            c.mux(p)
        times.append(ms / 1000)
        t += (0.02 if int(t * 2) % 2 else 0.06) if vfr else 0.04
    for p in s.encode():
        c.mux(p)
    c.close()
    os.utime(path, (OLD, OLD))
    return times


class RedDetector:
    """Finds the red boxes. Stands in for the vehicle model so tracking can be checked exactly."""

    def describe(self):
        return {"model": {"name": "red-boxes", "version": "1", "sha256": "0" * 64}, "preprocessing": {}}

    def detect(self, img):
        m = np.all(np.abs(img.astype(int) - (0, 0, 255)) <= 60, axis=2).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(m)
        return [Detection((float(x), float(y), float(x + w), float(y + h)), "car", 0.9, {"car": 0.9})
                for x, y, w, h, a in st[1:] if a > 80]


def mission(tmp_path, files: dict, mode="object_search"):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    for name, (dur, draw, vfr) in files.items():
        make_video(src / name, dur, draw, vfr)
    dr = DataRoot(tmp_path / "data")
    cfg = MissionConfig(name="v", input_type="video", mode=mode, country="IL", iana_zone="Asia/Jerusalem",
                        source_dir=str(src), include_subdirs=False, processing="one_time", output_dir=str(out),
                        export_formats=["csv"])
    m = dr.create_mission(cfg, "lior")
    cycle(m, stability_seconds=0)
    return m


def run(m, settings=None, control=None):
    return VideoProcessor(m, RedDetector(), {}, settings, control).process_all()


def summary(m):
    return [(e.key, e.first_t, e.last_t, e.cls, tuple(e.flags), e.end_reason) for e in VideoService(m).events()]


# ------------------------------------------------------------------ timestamps

def test_vfr_times_are_presentation_timestamps(tmp_path):
    p = tmp_path / "vfr.mp4"
    written = make_video(p, 6.0, vfr=True)
    pr = probe(p)
    assert pr.variable_frame_rate is True and pr.width == W
    got = [f.t for f, _ in frames(p, 0.2)]
    # Reference: for every grid point, the first written frame at or after it.
    want, g = [], 0.0
    for t in written:
        if t + 1e-6 >= g:
            want.append(t)
            g += (int((t + 1e-6 - g) // 0.2) + 1) * 0.2
    assert got == pytest.approx(want, abs=1e-6)
    # Frame counting would be wrong: 6 s at the nominal 25 fps is 150 frames, but a different number was written.
    assert len(written) != 150 and len(written) / 25 != pytest.approx(6.0, abs=0.5)


def test_constant_rate_and_resume_point(tmp_path):
    p = tmp_path / "cfr.mp4"
    make_video(p, 4.0)
    assert probe(p).variable_frame_rate is False
    full = [(round(f.t, 6), n) for f, n in frames(p, 0.2)]
    k = 7
    after, nxt = full[k - 1][0], full[k - 1][1]
    rest = [(round(f.t, 6), n) for f, n in frames(p, 0.2, after_t=after, next_sample_t=nxt)]
    assert full[k:] == rest


# ------------------------------------------------------------------ tracker

def test_min_hits_discards_blips():
    tr = Tracker(W, H, TrackerParams(min_hits=3, gap_s=0.5))
    tr.update(0.0, [([100, 50, 140, 74], "car", 0.9)])
    tr.update(0.2, [])
    _, closed, _ = tr.update(1.0, [])
    assert len(closed) == 1 and not closed[0].confirmed


def test_occlusion_within_gap_keeps_identity_and_reentry_is_flagged():
    p = TrackerParams(min_hits=3, gap_s=1.5, reentry_s=3.0)
    tr = Tracker(W, H, p)
    box = lambda t: [40 + 50 * t, 60, 80 + 50 * t, 84]
    for i in range(5):
        tr.update(i * 0.2, [(box(i * 0.2), "car", 0.9)])
    key = tr.tracks[0].key
    for t in (1.0, 1.2, 1.4, 1.6):                      # hidden for 0.8 s
        tr.update(t, [])
    tr.update(1.8, [(box(1.8), "car", 0.9)])
    assert [t.key for t in tr.tracks] == [key]           # same vehicle after occlusion
    # Now hidden longer than the gap: the track closes, and the reappearance is a new, flagged track.
    closed = []
    for t in np.arange(2.0, 3.6, 0.2):
        closed += tr.update(round(float(t), 3), [])[1]
    assert len(closed) == 1 and closed[0].confirmed and closed[0].reason == "lost"
    _, _, new = tr.update(3.6, [(box(3.6), "car", 0.9)])
    assert new[0].key != key and new[0].related == [key]


def test_tracker_state_roundtrip():
    tr = Tracker(W, H)
    for i in range(4):
        tr.update(i * 0.2, [([10 + i * 8, 50, 50 + i * 8, 74], "car", 0.9)])
    tr2 = Tracker.from_state(tr.state())
    a = tr.update(1.0, [([42, 50, 82, 74], "truck", 0.8)])
    b = tr2.update(1.0, [([42, 50, 82, 74], "truck", 0.8)])
    assert tr.state() == tr2.state() and len(a[0]) == len(b[0])


# ------------------------------------------------------------------ processing

def test_one_event_per_crossing_with_utc(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, lambda t: car_scene(t, cars=((-40, 80, 100),)), True)})
    rep = run(m)
    assert rep[0].done == 1 and rep[0].events == 1
    (e,) = VideoService(m).events()
    assert e.key == "E-0001" and e.cls == "car"
    assert e.end_reason == "left_frame" and "appeared_inside_frame" not in e.flags
    assert e.first_utc.startswith("2026-10-21T12:00:0") and e.utc_precision
    # The car enters at t=0 and is fully out (x0 + 100 t > 320) at t=3.6.
    assert e.first_t < 0.5 and 3.0 < e.last_t < 3.7


def test_chunk_boundaries_and_cancel_resume_give_identical_events(tmp_path):
    cars = lambda t: car_scene(t, cars=((-40, 40, 70), (-200, 110, 90)))
    ref = mission(tmp_path / "a", {"CAM1_20261021T120000Z.mp4": (7.0, cars, True)})
    run(ref, VideoSettings(chunk_frames=50))

    class CancelOnce(RunControl):
        def __init__(self):
            super().__init__()
            self.n = 0

        def checkpoint(self):
            self.n += 1
            if self.n == 2:
                raise RunCancelled()

    m = mission(tmp_path / "b", {"CAM1_20261021T120000Z.mp4": (7.0, cars, True)})
    run(m, VideoSettings(chunk_frames=4), CancelOnce())
    state = m.con.execute("SELECT processing_state FROM run_item").fetchone()[0]
    assert state == "queued"
    cp = m.con.execute("SELECT frames_analysed FROM video_checkpoint").fetchone()[0]
    assert cp == 8
    run(m, VideoSettings(chunk_frames=4))
    assert summary(m) == summary(ref) and len(summary(ref)) == 2
    n_obs = lambda mm: mm.con.execute("SELECT COUNT(*), SUM(t) FROM track_observation").fetchone()
    assert n_obs(m)[0] == n_obs(ref)[0] and n_obs(m)[1] == pytest.approx(n_obs(ref)[1])


def test_scene_cut_closes_tracks(tmp_path):
    def cut(t):
        return car_scene(t, cars=((60, 80, 10),), bg=100 if t < 3.0 else 230)
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (6.0, cut, False)})
    run(m)
    evs = VideoService(m).events()
    assert len(evs) == 2
    assert evs[0].end_reason == "scene_cut" and "cut_by_scene_change" in evs[0].flags and evs[0].last_t < 3.0
    assert evs[1].first_t >= 3.0 and evs[1].end_reason == "end_of_recording"
    assert m.con.execute("SELECT COUNT(*) FROM video_marker WHERE kind='scene_cut'").fetchone()[0] == 1


def test_unknown_start_time_leaves_utc_empty(tmp_path):
    m = mission(tmp_path, {"parking_camera.mp4": (3.0, lambda t: car_scene(t, cars=((100, 80, 5),)), False)})
    run(m)
    (e,) = VideoService(m).events()
    assert e.first_utc is None and e.utc_precision is None
    files = VideoService(m).export("lior")
    row = next(csv.DictReader(open(files[0], encoding="utf-8-sig")))
    assert row["start_utc"] == "" and row["utc_precision"] == "unknown" and row["start"].startswith("00:00:00")


# ------------------------------------------------------------------ review, clips, export

@pytest.fixture
def two_cars(tmp_path):
    cars = lambda t: car_scene(t, cars=((-40, 40, 70), (-200, 110, 90)))
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (7.0, cars, False)})
    run(m)
    return m, VideoService(m)


def test_revisions_are_append_only_and_snap_to_frames(two_cars):
    m, vs = two_cars
    a, b = vs.events()
    e = vs.revise(a.id, "set_times", "lior", first_t=0.501, last_t=2.0)
    assert e.first_t == pytest.approx(0.52) and e.last_t == pytest.approx(2.0)   # 25 fps grid: 0.52 is the next frame
    assert e.machine_first_t == a.first_t                                         # machine output kept
    with pytest.raises(VideoError):
        vs.revise(a.id, "set_times", "lior", first_t=3.0, last_t=2.0)
    e = vs.revise(a.id, "split_at", "lior", t=1.0)
    assert len(e.parts) == 2 and e.parts[0][1] == pytest.approx(1.0)
    vs.revise(b.id, "merge_into", "lior", target=a.id)
    a2, b2 = vs.events()
    assert b2.status == "merged" and a2.absorbed == [b.key] and a2.last_t >= b.last_t
    vs.revise(b.id, "reopen", "lior")
    assert vs.event(b.id).status == "pending"
    with pytest.raises(VideoError):
        vs.revise(a.id, "duplicate", "lior", of="E-9999")
    vs.revise(b.id, "duplicate", "lior", of=a.key)
    assert vs.event(b.id).duplicate_of == a.key
    vs.revise(a.id, "set_class", "lior", cls="truck")
    assert vs.event(a.id).cls == "truck" and vs.event(a.id).machine_cls == "car"
    n = m.con.execute("SELECT COUNT(*) FROM event_revision").fetchone()[0]
    assert n == 6


def test_clips_copy_and_reencode(two_cars):
    m, vs = two_cars
    a = vs.events()[0]
    with pytest.raises(VideoError):
        vs.clip(a.id, None, 1.0, "copy", "lior")          # context has no preset
    c1 = vs.clip(a.id, 1.5, 0.5, "copy", "lior")
    req_start = max(0.0, a.first_t - 1.5)
    assert c1["actual"][0] <= req_start + 1e-6             # copy starts at a keyframe at or before the request
    c2 = vs.clip(a.id, 1.5, 0.5, "reencode", "lior")
    assert c2["actual"][0] == pytest.approx(req_start, abs=0.041)
    assert c2["actual"][1] == pytest.approx(a.last_t + 0.5, abs=0.081)
    c3 = vs.clip(a.id, 1.5, 0.5, "reencode", "lior")
    assert c3["path"] != c2["path"] and Path(c2["path"]).is_file()   # never overwritten
    for c in (c1, c2):
        assert next(f for f, _ in frames(Path(c["path"]), 0.2)) is not None
    assert len(vs.clips(a.id)) == 3


def test_export_one_row_per_part(two_cars):
    m, vs = two_cars
    a, b = vs.events()
    vs.revise(a.id, "split_at", "lior", t=1.0)
    vs.revise(b.id, "reject", "lior")
    files = vs.export("lior")
    rows = list(csv.DictReader(open(files[0], encoding="utf-8-sig")))
    assert [r["event_id"] for r in rows] == ["E-0001", "E-0001", "E-0002"]
    assert [r["part"] for r in rows] == ["1", "2", ""]
    assert rows[2]["review_status"] == "rejected" and rows[0]["start_utc"].startswith("2026-10-21T12:00:0")
    assert vs.export("lior")[0] != files[0]


# ------------------------------------------------------------------ web

@pytest.fixture
def api(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    make_video(src / "CAM1_20261021T120000Z.mp4", 7.0, lambda t: car_scene(t, cars=((-40, 40, 70), (-200, 110, 90))))
    app = create_app(tmp_path / "data", detector_factory=lambda: (RedDetector(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = TestClient(app)
    def make(mode="object_search"):
        body = {"name": "Vid", "input_type": "video", "mode": mode, "country": "IL", "iana_zone": "Asia/Jerusalem",
                "source_dir": str(src), "include_subdirs": False, "processing": "one_time", "output_dir": str(out),
                "export_formats": ["csv"]}
        r = c.post("/api/missions", json=body, headers=H1)
        assert r.status_code == 200, r.text
        mid = r.json()["id"]
        c.post(f"/api/missions/{mid}/job", headers=H1)
        wait(c, mid)
        return mid
    return c, make


def wait(c, mid, timeout=90):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get(f"/api/missions/{mid}").json()["job"]
        if j and j["state"] in ("finished", "cancelled", "failed"):
            assert j["state"] == "finished", j
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_video_api_review_flow(api):
    c, make = api
    mid = make()
    recs = c.get(f"/api/missions/{mid}/recordings").json()["recordings"]
    assert len(recs) == 1 and recs[0]["events"] == 2 and recs[0]["processing_state"] == "done"
    vid = recs[0]["id"]
    assert recs[0]["start_source"] == "filename"
    manual = c.put(f"/api/missions/{mid}/recordings/{vid}/start-time", headers=H1,
                   json={"action": "set", "utc": "2026-10-21T13:15:00Z", "precision": "minute",
                         "note": "Confirmed against recording log"})
    assert manual.status_code == 200, manual.text
    assert manual.json()["start_utc"] == "2026-10-21T13:15:00Z"
    assert manual.json()["start_source"] == "manual" and manual.json()["start_revision_author"]
    assert c.get(f"/api/missions/{mid}/events").json()["events"][0]["first_utc"].startswith("2026-10-21T13:15:00")
    naive = c.put(f"/api/missions/{mid}/recordings/{vid}/start-time", headers=H1,
                  json={"action": "set", "utc": "2026-10-21T13:15:00", "precision": "second"})
    assert naive.status_code == 400
    cleared = c.put(f"/api/missions/{mid}/recordings/{vid}/start-time", headers=H1,
                    json={"action": "clear", "note": "Restore filename timestamp"})
    assert cleared.status_code == 200 and cleared.json()["start_source"] == "filename"
    assert cleared.json()["start_utc"] == "2026-10-21T12:00:00Z"
    assert c.get(f"/api/missions/{mid}/events").json()["events"][0]["first_utc"].startswith("2026-10-21T12:00:00")
    r = c.get(f"/api/missions/{mid}/videos/{vid}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    evs = c.get(f"/api/missions/{mid}/events").json()["events"]
    assert [e["key"] for e in evs] == ["E-0001", "E-0002"]
    one = c.get(f"/api/missions/{mid}/events/{evs[0]['id']}").json()
    assert len(one["observations"]) >= 3 and one["clips"] == []
    th = c.get(f"/api/missions/{mid}/events/{evs[0]['id']}/thumb")
    assert th.status_code == 200 and th.headers["content-type"] == "image/jpeg"
    r = c.post(f"/api/missions/{mid}/events/{evs[0]['id']}/revise", json={"action": "confirm"}, headers=H1)
    assert r.json()["status"] == "confirmed"
    bad = c.post(f"/api/missions/{mid}/events/{evs[0]['id']}/revise", json={"action": "split_at", "t": 99}, headers=H1)
    assert bad.status_code == 400
    assert c.get(f"/api/missions/{mid}/events?status=confirmed").json()["events"][0]["key"] == "E-0001"
    noctx = c.post(f"/api/missions/{mid}/events/{evs[0]['id']}/clip", json={"method": "copy"}, headers=H1)
    assert noctx.status_code == 400
    clip = c.post(f"/api/missions/{mid}/events/{evs[0]['id']}/clip",
                  json={"before_s": 1, "after_s": 1, "method": "reencode"}, headers=H1).json()
    assert Path(clip["path"]).is_file()
    files = c.post(f"/api/missions/{mid}/events/export", json={}, headers=H1).json()["files"]
    assert files and Path(files[0]).is_file()


def test_video_search_finds_events(api):
    c, make = api
    mid = make()
    r = c.post(f"/api/missions/{mid}/search/ask", json={"text": "cars"}, headers=H1).json()
    c.post(f"/api/missions/{mid}/search/{r['id']}/confirm", json={}, headers=H1)
    assert c.post(f"/api/missions/{mid}/search/{r['id']}/run", headers=H1).status_code == 200
    j = wait(c, mid)
    res = c.get(f"/api/missions/{mid}/search-runs/{j['result']['search_run']}").json()
    assert res["total"] == 2 and all(h["object_uid"].startswith("ev") for h in res["hits"])
    assert all(h["status"] == "match" for h in res["hits"])
    assert all(h["event_t"] is not None and h["capture_local"].startswith("2026-10-21T15:00:0") for h in res["hits"])
    h = res["hits"][0]
    ok = c.put(f"/api/missions/{mid}/items/{h['source_version_id']}/objects/{h['object_uid']}/colour",
               json={"value": "red"}, headers=H1)
    assert ok.status_code == 200
    # A rejected event drops out of the next search.
    eid = int(h["object_uid"][2:])
    c.post(f"/api/missions/{mid}/events/{eid}/revise", json={"action": "reject"}, headers=H1)
    r2 = c.post(f"/api/missions/{mid}/search/ask", json={"text": "cars"}, headers=H1).json()
    c.post(f"/api/missions/{mid}/search/{r2['id']}/confirm", json={}, headers=H1)
    c.post(f"/api/missions/{mid}/search/{r2['id']}/run", headers=H1)
    j2 = wait(c, mid)
    assert c.get(f"/api/missions/{mid}/search-runs/{j2['result']['search_run']}").json()["total"] == 1


def test_image_mission_has_no_video_endpoints(api, tmp_path):
    c, _ = api
    body = {"name": "Img", "mode": "general_traffic", "country": "IL", "iana_zone": "Asia/Jerusalem",
            "source_dir": str(tmp_path / "src"), "include_subdirs": False, "processing": "one_time",
            "output_dir": str(tmp_path / "out"), "export_formats": ["csv"]}
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    assert c.get(f"/api/missions/{mid}/events").status_code == 400


# ------------------------------------------------------------------ plates in video

def plate_car_scene(t: float) -> np.ndarray:
    """One large red car moving right, with a yellow plate on it."""
    img = np.full((H, W, 3), 100, np.uint8)
    x = int(20 + 40 * t)
    cv2.rectangle(img, (x, 60), (x + 100, 120), (0, 0, 255), -1)
    cv2.rectangle(img, (x + 25, 100), (x + 75, 114), (0, 220, 255), -1)
    return img


def test_combine_frame_reads_votes_and_never_mixes():
    import json as _json
    from trafficwb.plates import combine_frame_reads
    o = [{"t": float(i)} for i in range(4)]
    full = lambda text, s=0.9: {"outcome": "read", "readability": "full", "normalized_text": text, "det_score": s, "reasons": "[]"}
    t, r = combine_frame_reads([(o[0], full("1234567")), (o[1], full("1234567", 0.95)), (o[2], {"outcome": "no_plate_found", "reasons": "[]"})])
    assert r["normalized_text"] == "1234567" and r["readability"] == "full" and t["t"] == 1.0
    assert "2 frames agree" in _json.loads(r["reasons"]) and "plate read on 2 of 3 frames" in _json.loads(r["reasons"])
    t, r = combine_frame_reads([(o[0], full("1234567")), (o[1], full("1234561"))])
    assert r["readability"] == "partial" and any("disagree" in x for x in _json.loads(r["reasons"]))
    part = lambda text, s: {"outcome": "read", "readability": "partial", "normalized_text": text, "det_score": s, "reasons": "[]"}
    t, r = combine_frame_reads([(o[0], part("12??567", 0.9)), (o[1], part("123?567", 0.6))])
    assert r["normalized_text"] == "123?567"                     # fewest unsure characters; nothing filled in
    t, r = combine_frame_reads([(o[0], {"outcome": "vehicle_too_small", "reasons": "[]"}),
                                (o[1], {"outcome": "plate_too_small", "readability": "unreadable", "reasons": "[]"})])
    assert r["outcome"] == "plate_too_small"


def test_plates_read_on_several_frames_of_each_event(tmp_path):
    from test_plates import PAD, YellowFinder, slots
    from trafficwb.plates import OcrOut, PlateService

    class Reader:
        def __init__(self, texts):
            self.texts, self.calls = texts, 0

        def describe(self):
            return {"model": {"name": "list-reader", "version": "1", "sha256": "3" * 64}, "preprocessing": {}}

        def read(self, plate):
            self.calls += 1
            return OcrOut(slots(self.texts[(self.calls - 1) % len(self.texts)]), PAD, "Israel", 0.9)

    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (5.0, plate_car_scene, False)}, mode="plate_recognition")
    reader = Reader(["1234567"])
    rep = VideoProcessor(m, RedDetector(), {}, plate_models=lambda: (YellowFinder(), reader)).process_all()
    assert rep[0].events == 1 and rep[0].plates["events"] == 1 and rep[0].plates["frames"] == 4
    svc = PlateService(m)
    (p,) = svc.plates()
    assert p.object_uid.startswith("e") and p.event_key == "E-0001" and p.t is not None
    assert (p.outcome, p.machine_text, p.readability) == ("read", "1234567", "full")
    assert "4 frames agree" in p.reasons and p.capture_utc.startswith("2026-10-21T12:00:0")
    # Operator review works as for still images, and the export names the event and time.
    svc.revise(p.id, "confirm", "lior")
    assert svc.plates()[0].review == "confirmed"
    paths = svc.export("lior", ["csv"])
    rows = list(csv.DictReader(open(paths[0], encoding="utf-8-sig")))
    assert rows[0]["video_event"] == "E-0001" and rows[0]["plate_text"] == "1234567"
    ev = list(csv.DictReader(open(VideoService(m).export("lior", ["csv"])[0], encoding="utf-8-sig")))
    assert ev[0]["plate_text"] == "1234567" and ev[0]["plate_review"] == "confirmed"
    # Nothing new to read: a second refresh reads nothing.
    assert PlateService(m, YellowFinder(), reader).refresh()["events"] == 0
    # Frames that disagree give a partial read, with both texts in the reasons.
    m2 = mission(tmp_path / "b", {"CAM1_20261021T120000Z.mp4": (5.0, plate_car_scene, False)}, mode="plate_recognition")
    VideoProcessor(m2, RedDetector(), {}, plate_models=lambda: (YellowFinder(), Reader(["1234567", "7654321"]))).process_all()
    (p,) = PlateService(m2).plates()
    assert p.readability == "partial" and any("disagree" in r for r in p.reasons)
