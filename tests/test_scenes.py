"""Sorting still images into indoor and outdoor: processing, the operator's corrections, learning and copying out."""
import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from trafficwb import scenes
from trafficwb.web.app import create_app

H1 = {"X-TWB": "1"}
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()


class FakeEmbedder:
    """Stands in for CLIP: bright images lean outdoor, dark ones indoor; 'tricky' ones (a mid grey) lean the wrong
    way, the same way every time, so the operator has something to correct and learning something to learn."""

    model = "fake/1"

    class pack:
        name, version, sha256 = "fake", "1", "0" * 64

    def __init__(self):
        p = scenes.prompts()
        self.i, self.o = np.asarray(p["indoor"]), np.asarray(p["outdoor"])
        rng = np.random.default_rng(3)
        self.tricky = rng.normal(size=512)
        self.tricky /= np.linalg.norm(self.tricky)

    def embed(self, images):
        out = []
        for img in images:
            m = float(img.mean())
            noise = np.random.default_rng(int(m * 10)).normal(scale=0.02, size=512)
            if 110 < m < 150:          # tricky: looks a little indoor to the prompts, but the operator says outdoor
                v = 0.75 * self.i + 0.25 * self.o + 0.5 * self.tricky + noise
            else:
                v = (self.o if m >= 150 else self.i) + noise
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "models" / scenes.PACK_NAME / "test"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"stand-in")
    (d / "model.json").write_text(json.dumps({"name": scenes.PACK_NAME, "version": "test", "task": scenes.TASK,
                                              "sha256": "0" * 64, "license": "MIT"}))
    monkeypatch.setenv("TWB_MODELS", str(tmp_path / "models"))
    return d


def photos(src, n_in=4, n_out=4, n_tricky=3):
    src.mkdir()
    names = []
    for k in range(n_in + n_out + n_tricky):
        kind, val = ("in", 40 + k) if k < n_in else ("out", 200 + k) if k < n_in + n_out else ("tricky", 125 + k)
        p = src / f"holiday {kind} {k}.jpg"           # no capture time in the name: sorting does not need one
        cv2.imwrite(str(p), np.full((90, 160, 3), val, np.uint8))
        os.utime(p, (OLD, OLD))
        names.append(p.name)
    return names


def app_client(tmp_path):
    app = create_app(tmp_path / "data", detector_factory=lambda: (_ for _ in ()).throw(AssertionError("no detector")),
                     start_monitoring=False, stability_seconds=0)
    app.state.jobs.scene_factory = FakeEmbedder
    return TestClient(app)


def make_mission(c, src, out, name="Bundle"):
    body = {"name": name, "input_type": "images", "mode": "indoor_outdoor", "country": "IL", "iana_zone": "Asia/Jerusalem",
            "source_dir": str(src), "include_subdirs": True, "processing": "one_time", "output_dir": str(out),
            "export_formats": ["csv"]}
    v = c.post("/api/missions/validate", json=body, headers=H1).json()
    assert v["errors"] == [], v
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    c.post(f"/api/missions/{mid}/job", headers=H1)
    for _ in range(300):
        j = c.get(f"/api/missions/{mid}").json()["job"]
        if j and j["state"] not in ("running", "paused"):
            break
        time.sleep(0.05)
    assert j["state"] == "finished", j
    return mid


def test_mode_needs_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TWB_MODELS", str(tmp_path / "none"))
    c = app_client(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "out").mkdir()
    body = {"name": "B", "input_type": "images", "mode": "indoor_outdoor", "country": "IL", "iana_zone": "Asia/Jerusalem",
            "source_dir": str(tmp_path / "src"), "include_subdirs": True, "processing": "one_time",
            "output_dir": str(tmp_path / "out"), "export_formats": ["csv"]}
    errs = c.post("/api/missions/validate", json=body, headers=H1).json()["errors"]
    assert any("indoor/outdoor model" in e for e in errs)
    assert "indoor_outdoor" in c.get("/api/reference").json()["modes"]["images"]


def test_sort_correct_learn_and_copy(tmp_path, models_dir):
    c = app_client(tmp_path)
    src, out = tmp_path / "src", tmp_path / "out"
    out.mkdir()
    photos(src)
    mid = make_mission(c, src, out)

    r = c.get(f"/api/missions/{mid}/scenes").json()
    items = {i["file"]: i for i in r["items"]}
    assert len(items) == 11                                  # names without a capture time are sorted too
    assert all(items[f"holiday in {k}.jpg"]["label"] == "indoor" for k in range(4))
    assert all(items[f"holiday out {k}.jpg"]["label"] == "outdoor" for k in range(4, 8))
    tricky = [i for f, i in items.items() if "tricky" in f]
    assert all(i["machine"] == "indoor" for i in tricky)     # LanePulse gets these wrong
    assert r["counts"] == {"indoor": 7, "outdoor": 4, "moved": 0, "confirmed": 0, "unsure": r["counts"]["unsure"]}

    # The operator moves two of the three tricky ones; the decision wins over LanePulse's choice.
    moved = [tricky[0]["version_id"], tricky[1]["version_id"]]
    d = c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": moved, "label": "outdoor"}, headers=H1).json()
    assert d["counts"]["outdoor"] == 6 and d["counts"]["moved"] == 2
    # ... and can take a decision back.
    c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": [moved[1]], "label": None}, headers=H1)
    after = {i["version_id"]: i for i in c.get(f"/api/missions/{mid}/scenes").json()["items"]}
    assert after[moved[0]]["label"] == "outdoor" and after[moved[0]]["decided_by"]
    assert after[moved[1]]["label"] == "indoor" and after[moved[1]]["decided"] is None
    assert c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": moved, "label": "attic"},
                  headers=H1).status_code == 400

    # Learning needs two placed images; then it learns from the moves and re-sorts the similar image nobody moved.
    first = c.post(f"/api/missions/{mid}/scenes/learn", headers=H1).json()
    assert first["kept"] is False and "at least" in first["reason"]
    ins = [i["version_id"] for i in r["items"] if i["file"].startswith("holiday in")][:2]
    outs = [i["version_id"] for i in r["items"] if i["file"].startswith("holiday out")][:1]
    c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": ins, "label": "indoor"}, headers=H1)
    c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": outs + [moved[1]], "label": "outdoor"}, headers=H1)
    res = c.post(f"/api/missions/{mid}/scenes/learn", headers=H1).json()
    assert res["kept"] is True and res["right_after"] >= res["right_before"], res
    now = {i["version_id"]: i for i in c.get(f"/api/missions/{mid}/scenes").json()["items"]}
    assert now[tricky[2]["version_id"]]["machine"] == "outdoor"     # learned from the two the operator moved
    assert now[tricky[2]["version_id"]]["decided"] is None and now[tricky[2]["version_id"]]["learned"] is True
    assert all(now[i["version_id"]]["label"] == "indoor" for i in r["items"] if i["file"].startswith("holiday in"))
    assert c.get("/api/scenes/status").json()["learner"]["in_use"]["version"] == res["version"]

    # A second bundle starts with what was learned.
    src2 = tmp_path / "src2"
    photos(src2, n_in=1, n_out=1, n_tricky=1)
    mid2 = make_mission(c, src2, out, name="Bundle 2")
    tr2 = [i for i in c.get(f"/api/missions/{mid2}/scenes").json()["items"] if "tricky" in i["file"]][0]
    assert tr2["label"] == "outdoor"

    # Going back to the original sorter.
    assert c.post(f"/api/missions/{mid}/scenes/forget", headers=H1).json()["learner"]["in_use"] is None
    assert {i["version_id"]: i for i in c.get(f"/api/missions/{mid}/scenes").json()["items"]}[
        tricky[2]["version_id"]]["machine"] == "indoor"
    c.post(f"/api/missions/{mid}/scenes/learn", headers=H1)

    # Copy out: two folders and a list; the source files are untouched.
    before = {p.name: p.stat().st_mtime for p in src.iterdir()}
    cp = c.post(f"/api/missions/{mid}/scenes/copy", headers=H1).json()
    dest = out / "Bundle - indoor and outdoor"
    assert cp["copied"] == 11 and cp["folder"] == str(dest)
    assert len(list((dest / "indoor").iterdir())) == cp["indoor"] and len(list((dest / "outdoor").iterdir())) == cp["outdoor"]
    assert (dest / "outdoor" / f"holiday tricky {tricky[0]['file'].split()[-1]}").is_file()
    assert "decided_by" in (dest / "sorting.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    # A later move takes the copy out of the other folder.
    c.post(f"/api/missions/{mid}/scenes/decide", json={"version_ids": [moved[0]], "label": "indoor"}, headers=H1)
    c.post(f"/api/missions/{mid}/scenes/copy", headers=H1)
    assert (dest / "indoor" / tricky[0]["file"]).is_file() and not (dest / "outdoor" / tricky[0]["file"]).exists()
    assert {p.name: p.stat().st_mtime for p in src.iterdir()} == before


def test_other_missions_have_no_sorting(tmp_path, models_dir):
    c = app_client(tmp_path)
    src, out = tmp_path / "src", tmp_path / "out"
    out.mkdir()
    photos(src, 1, 1, 0)
    mid = make_mission(c, src, out)
    from trafficwb.store import DataRoot
    root = DataRoot(tmp_path / "data")
    root.con.execute("UPDATE mission SET mode='general_traffic' WHERE id=?", (mid,))
    m = root.open_mission(mid)
    cfg = json.loads(m.con.execute("SELECT payload FROM config_revision ORDER BY seq DESC LIMIT 1").fetchone()[0])
    cfg["mode"] = "general_traffic"
    m.con.execute("INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
                  (json.dumps(cfg), "t", "2026-01-01T00:00:00Z", "t"))
    assert c.get(f"/api/missions/{mid}/scenes").status_code == 400


def test_preprocess_matches_clip():
    img = np.zeros((300, 600, 3), np.uint8)
    img[:, :, 2] = 255                                   # BGR red
    x = scenes.preprocess(img)
    assert x.shape == (3, 224, 224)
    assert x[0].mean() == pytest.approx((1 - 0.48145466) / 0.26862954, abs=1e-3)   # red channel first (RGB)
