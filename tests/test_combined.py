"""Both vehicle detectors on one still image: the standard one and SAM 3.1, merged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_sam3 import api, eng, pack_root, wait, H1  # noqa: E402,F401  (fixtures)

from trafficwb.combined import CombinedDetector, merge  # noqa: E402
from trafficwb.detect import Detection  # noqa: E402


def d(box, cls, score):
    return Detection(box, cls, score, {cls: score})


def test_a_vehicle_both_found_counts_once_with_the_more_confident_answer():
    std = [d((10, 10, 50, 40), "car", 0.6), d((200, 50, 240, 80), "truck", 0.8)]
    sam = [d((12, 11, 49, 41), "car", 0.9), d((100, 100, 130, 120), "bus", 0.5)]
    out = merge(std, sam)
    assert len(out) == 3
    both = [x for x in out if x.found_by == "both"]
    assert len(both) == 1 and both[0].box == (12, 11, 49, 41) and both[0].score == 0.9
    assert {x.found_by for x in out} == {"both", "standard", "sam3"}


def test_a_small_box_inside_a_larger_one_is_the_same_vehicle_and_classes_keep_their_best_scores():
    std = [Detection((0, 0, 100, 60), "truck", 0.7, {"truck": 0.7, "car": 0.2})]
    sam = [Detection((10, 5, 90, 55), "car", 0.5, {"car": 0.5})]      # IoU about 0.67 ... and inside
    (one,) = merge(std, sam)
    assert one.found_by == "both" and one.cls == "truck" and one.class_scores == {"truck": 0.7, "car": 0.5}
    far = merge(std, [Detection((300, 300, 340, 330), "car", 0.5, {"car": 0.5})])
    assert sorted(x.found_by for x in far) == ["sam3", "standard"]


class Fake:
    def __init__(self, name, dets):
        self.name, self.dets = name, dets

    def describe(self):
        return {"model": {"name": self.name, "version": "1", "sha256": self.name[0] * 64}, "preprocessing": {"p": self.name}}

    def detect(self, img):
        return self.dets


def test_the_combined_model_is_named_after_both_and_keeps_each_record():
    c = CombinedDetector(Fake("yolox", []), {"license": "Apache-2.0"}, Fake("sam", []), {"license": "SAM License"})
    desc = c.describe()
    assert desc["model"]["name"] == "yolox + sam" and len(desc["model"]["sha256"]) == 64
    assert desc["preprocessing"]["standard"] == {"p": "yolox"} and "merge" in desc["preprocessing"]
    assert c.card()["license"] == "Apache-2.0; SAM License"
    assert set(c.parts()) == {"standard", "sam3"}


def test_mission_with_both_detectors(api):
    c, body, state = api
    vid = c.post("/api/missions/validate", json={**body, "input_type": "video", "mode": "object_search",
                                                 "detector": "both"}, headers=H1).json()
    assert any("still images" in e for e in vid["errors"])
    mid = c.post("/api/missions", json={**body, "detector": "both"}, headers=H1).json()["id"]
    assert c.get(f"/api/missions/{mid}").json()["config"]["detector"] == "both"
    c.post(f"/api/missions/{mid}/job", headers=H1)
    assert wait(c, f"/api/missions/{mid}", key="job")["state"] == "finished"
    rows = c.post(f"/api/missions/{mid}/results", json={}, headers=H1).json()["rows"]
    assert rows and all(r["vehicle_count"] >= 2 for r in rows), rows    # the standard car and SAM's vehicle on the road
    files = c.post(f"/api/missions/{mid}/export", json={"formats": ["csv"]}, headers=H1).json()["files"]
    objects = Path(next(f for f in files if f.endswith("_objects.csv"))).read_text(encoding="utf-8-sig")
    assert "fake 1" in objects and "sam3.1" in objects                  # each detection names the model that found it
    state["installed"] = False
    r = c.put(f"/api/missions/{mid}/detector", json={"detector": "both"}, headers=H1)
    assert r.status_code == 400 and r.json()["error"] == "sam3_missing"


def test_words_to_look_for_are_checked_and_used():
    import pytest
    from trafficwb.sam3 import VEHICLE_PROMPTS, vehicle_prompts
    assert vehicle_prompts(None) == VEHICLE_PROMPTS
    w = vehicle_prompts({"car": ["Car", " police  car ", "car"], "truck": ["truck", "tractor"]})
    assert w["car"] == ("car", "police car") and w["truck"] == ("truck", "tractor") and w["bus"] == ("bus",)
    assert vehicle_prompts({"bicycle": []})["bicycle"] == ()
    for bad in ({"tank": ["tank"]}, {"car": ["x" * 41]}, {"car": ["van"], "truck": ["van"]},
                {c: [] for c in VEHICLE_PROMPTS}):
        with pytest.raises(ValueError):
            vehicle_prompts(bad)


def test_mission_words_reach_sam(api, tmp_path):
    c, body, state = api
    mid = c.post("/api/missions", json={**body, "detector": "sam3"}, headers=H1).json()["id"]
    r = c.put(f"/api/missions/{mid}/sam3-words", json={"words": {"truck": ["truck", "tractor"]}}, headers=H1)
    assert r.status_code == 200 and r.json()["words"]["truck"] == ["truck", "tractor"] and not r.json()["default"]
    assert c.get(f"/api/missions/{mid}").json()["config"]["sam3_words"]["truck"] == ["truck", "tractor"]
    bad = c.put(f"/api/missions/{mid}/sam3-words", json={"words": {"tank": ["tank"]}}, headers=H1)
    assert bad.status_code == 400 and "tank" in bad.json()["message"]
    c.post(f"/api/missions/{mid}/job", headers=H1)
    assert wait(c, f"/api/missions/{mid}", key="job")["state"] == "finished"
    from trafficwb.store import DataRoot
    m = DataRoot(tmp_path / "data").open_mission(mid)
    pre = m.con.execute("SELECT preprocessing FROM run ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert '"tractor"' in pre                                           # the run records the words SAM was asked
    m.con.close()
    r = c.put(f"/api/missions/{mid}/sam3-words", json={"words": None}, headers=H1)
    assert r.json()["default"] and c.get(f"/api/missions/{mid}").json()["config"]["sam3_words"] is None
