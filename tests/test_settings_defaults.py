"""Settings for new missions (both / still images / video) and file-name formats learned from an input folder."""
import json

import pytest

from trafficwb import defaults
from test_web_api import H1, client, direct_access, env  # noqa: F401  (env is a fixture)


def test_built_in_values_without_a_file(tmp_path):
    d = defaults.load(tmp_path)
    assert d["both"]["results_folder"] is None and d["both"]["export_formats"] == ["xlsx", "csv"]
    assert d["images"]["mode"] == "general_traffic" and d["video"]["mode"] == "object_search"


def test_save_merges_checks_and_survives_a_damaged_file(tmp_path):
    out = defaults.save(tmp_path, {"images": {"detector": "rfdetr"}, "video": {"camera": "moving", "video_workers": 4}})
    assert out["images"]["detector"] == "rfdetr" and out["video"]["camera"] == "moving"
    assert defaults.load(tmp_path)["video"]["video_workers"] == 4
    for bad in ({"video": {"mode": "general_traffic"}}, {"both": {"export_formats": []}}, {"both": {"iana_zone": "Mars/Base"}},
                {"images": {"nope": 1}}, {"nope": {}}, {"both": {"results_folder": str(tmp_path / "missing")}}):
        with pytest.raises(ValueError):
            defaults.save(tmp_path, bad)
    assert defaults.load(tmp_path)["images"]["detector"] == "rfdetr"      # a refused change leaves the file as it was
    (tmp_path / defaults.FILE).write_text("{not json", encoding="utf-8")
    assert defaults.load(tmp_path) == json.loads(json.dumps(defaults.BUILT_IN))


def test_mission_fields_per_kind(tmp_path):
    d = defaults.save(tmp_path, {"both": {"watch": True}, "video": {"sam3_interval_s": 2}})
    img, vid = defaults.mission_fields(d, "images"), defaults.mission_fields(d, "video")
    assert img["processing"] == "continuous" and "camera" not in img and img["mode"] == "general_traffic"
    assert vid["sam3_interval_s"] == 2.0 and vid["camera"] == "fixed" and "vehicle_model" not in vid


def test_quick_start_takes_settings(env):  # noqa: F811
    tmp, src, out = env
    c = direct_access(client(tmp))
    r = c.put("/api/settings/missions", json={"both": {"results_folder": str(out), "iana_zone": "Europe/Riga"}}, headers=H1)
    assert r.status_code == 200 and r.json()["both"]["iana_zone"] == "Europe/Riga"
    assert c.put("/api/settings/missions", json={"video": {"camera": "sideways"}}, headers=H1).status_code == 400
    q = c.post("/api/quick-start", json={"source_dir": str(src)}, headers=H1).json()
    assert q["output_dir"] == str(out / "src LanePulse exports") and q["iana_zone"] == "Europe/Riga" and q["country"] == "LV"
    assert q["settings"]["mode"] == "general_traffic" and q["settings"]["processing"] == "one_time"
    body = q["settings"] | {k: q[k] for k in ("name", "input_type", "country", "iana_zone", "source_dir", "output_dir")}
    c.post("/api/quick-start/output", json={"path": q["output_dir"]}, headers=H1)
    mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
    assert c.get(f"/api/missions/{mid}").json()["config"]["output_dir"] == q["output_dir"]


def test_a_new_kind_of_file_name_is_learned_and_remembered(env):  # noqa: F811
    tmp, src, out = env
    new = tmp / "gate"
    new.mkdir()
    for i in range(3):
        (new / f"GateA__2{i}.04.2024_13-28-3{i}.jpg").write_bytes(b"x")
    c = direct_access(client(tmp))
    g = c.post("/api/name-formats/learn", json={"folder": str(new)}, headers=H1).json()
    assert g["template"] == "{camera}__{DD}.{MM}.{YYYY}_{hh}-{mm}-{ss}" and g["matched"] == g["total"] == 3
    assert c.post("/api/quick-start", json={"source_dir": str(new)}, headers=H1).json()["readable"] == 0
    s = c.post("/api/name-formats/learn", json={"folder": str(new), "save": True}, headers=H1).json()
    assert s["saved"] and s["name"] == "Learned from gate"
    assert c.post("/api/quick-start", json={"source_dir": str(new)}, headers=H1).json()["readable"] == 3
    # Asking again keeps one copy, and a folder whose names are all known has nothing to learn.
    c.post("/api/name-formats/learn", json={"folder": str(new), "save": True}, headers=H1)
    assert len(c.get("/api/name-formats").json()["formats"]) == 1
    assert c.post("/api/name-formats/learn", json={"folder": str(new)}, headers=H1).json()["template"] is None
