"""Workers on other computers (phase 5): pairing, the switch, a worker sharing a recording over HTTP with the same
results, and a lost worker's segment going back to the queue."""
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import mission  # noqa: E402
from test_video_targets import both, crowd, segmented  # noqa: E402

from trafficwb.segment_hub import SegmentHub  # noqa: E402
from trafficwb.video_process import VideoProcessor  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402
from trafficwb.video_segments import Segment  # noqa: E402
from trafficwb.workers import Pairing, Worker, WorkerStopped  # noqa: E402

av = pytest.importorskip("av")
FILES = {"CAM1_20261021T120000Z.mp4": (6.0, crowd, False)}


def test_a_lease_not_renewed_ends_and_the_segment_goes_back():
    hub = SegmentHub(lease_s=0.05)
    hub.enabled = True
    segs = [Segment(0, 0, 2, 0, 3), Segment(1, 2, 4, 1, 5)]
    job = hub.publish("m", 1, 1, segs, {"needs": ["standard"]})
    hub.seen("pc2", 7, {})
    assert hub.take("pc2", remote=True, can={"sam3"}) is None          # it cannot run this job's detector
    gone = hub.take("pc2", remote=True, can={"standard"})
    assert gone.seg.idx == 0
    time.sleep(0.1)
    again = hub.take("this computer", job=job)                          # lost: back at the front of the queue
    assert again.seg.idx == 0 and hub.status()["helpers"][0]["lost"] == 1
    assert not hub.complete(gone.id, object())                           # a late result is ignored
    assert hub.complete(again.id, object()) and [s.idx for s, _, _ in hub.collect(job, 0)] == [0]


def test_pairing_codes_are_kept_as_hashes_and_can_be_revoked(tmp_path):
    p = Pairing(tmp_path)
    assert not p.enabled()
    p.set_enabled(True, "lior")
    c = p.new_code("laptop", "lior")
    assert len(c["code"]) == 14 and p.check(c["code"].lower()) == c["id"] and p.check("nope") is None
    assert c["code"] not in str(p.con.execute("SELECT * FROM worker_code").fetchall())
    p.revoke(c["id"])
    assert p.check(c["code"]) is None and p.enabled()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_another_computer_shares_a_recording_with_the_same_results(tmp_path):
    import uvicorn
    from fastapi.testclient import TestClient
    from trafficwb.web.app import create_app
    ref = mission(tmp_path / "a", FILES)
    ref.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    VideoProcessor(ref, both(), {}).process_all()

    m = mission(tmp_path / "b", FILES)
    m.update_config("lior", "words", sam3_words={"pedestrian": ["pedestrian"]})
    app = create_app(tmp_path / "b" / "data", detector_factory=lambda: (both(), {}), start_monitoring=False)
    api = TestClient(app, headers={"X-TWB": "1"})
    code = api.post("/api/workers/codes", json={"label": "pc2"}).json()["code"]
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    try:
        w = Worker(f"http://127.0.0.1:{port}", code, "pc2", detector_factory=lambda spec: both(), log=lambda s: None)
        with pytest.raises(WorkerStopped, match="does not take help"):     # the switch is off by default
            w.once()
        assert api.put("/api/workers/switch", json={"enabled": True}).json()["enabled"]
        assert w.once() is False                                           # nothing to do yet, but now it is known
        hub = app.state.jobs.hub
        assert hub.helpers_ready()
        stop = threading.Event()
        t = threading.Thread(target=w.run, kwargs={"stop": stop, "idle_s": 0.05}, daemon=True)
        t.start()
        settings = segmented()
        from dataclasses import replace
        par = VideoProcessor(m, both(), {}, replace(settings, workers=1), hub=hub,
                             remote_spec={"detector": "both", "words": None, "standard": None}).process_all()
        stop.set()
        t.join(10)
        rows = m.con.execute("SELECT result FROM video_segment WHERE state='done'").fetchall()
        assert len(rows) == 3 and par[0].frames == 30
        assert w.segments >= 1 and any('"worker": "pc2"' in r[0] for r in rows)   # the other computer did some
        a, b = VideoService(ref).counts()["total"], VideoService(m).counts()["total"]
        assert b["machine"] == a["machine"] == {"car": 3, "pedestrian": 1}
        ev = lambda mm: sorted((e.cls, round(e.first_t, 1), round(e.last_t, 1)) for e in VideoService(mm).events())  # noqa: E731
        assert ev(m) == ev(ref)
        st = api.get("/api/workers").json()
        assert st["enabled"] and st["helpers"][0]["name"] == "pc2" and st["helpers"][0]["segments"] == w.segments
        # A revoked code stops its worker.
        api.delete(f"/api/workers/codes/{st['codes'][0]['id']}")
        with pytest.raises(WorkerStopped, match="not valid"):
            w.once()
    finally:
        server.should_exit = True
