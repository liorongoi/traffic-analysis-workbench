"""Phase 6b: counting lines, areas and paths in video."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video_attributes import painted_cars  # noqa: E402
from test_video import mission, run  # noqa: E402

from trafficwb.video_review import VideoService  # noqa: E402
from trafficwb.video_zones import ZoneError, Zones, crossings, path, tally, visits, went  # noqa: E402

av = pytest.importorskip("av")


def _walk(xs, y=50, dt=0.1):
    return [(i * dt, [x - 10, y - 20, x + 10, y]) for i, x in enumerate(xs)]


def test_line_crossings_have_a_direction_and_a_wobble_counts_once():
    line = [[100, 0], [100, 200]]                      # drawn downwards: "in" is to its left on screen (x < 100)
    right = _walk(range(50, 160, 10))
    assert [c["dir"] for c in crossings(right, line)] == ["out"]
    left = _walk(range(160, 50, -10))
    assert [c["dir"] for c in crossings(left, line)] == ["in"]
    wobble = _walk([80, 90, 105, 95, 110, 130], dt=0.2)   # over, back within 1 s, over again
    assert [c["dir"] for c in crossings(wobble, line)] == ["out"]
    assert crossings(_walk(range(50, 160, 10), y=300), line) == []      # passes beyond the line's end
    shaking = _walk([97, 103] * 10, dt=0.6)                              # stands on the line, its box shaking
    assert crossings(shaking, line) == []


def test_area_visits_and_paths():
    area = [[100, 0], [200, 0], [200, 100], [100, 100]]
    b = _walk(range(50, 260, 5))                       # 5 px per 0.1 s: inside from x=100 to 200, 2 s
    v = visits(b, area)
    assert len(v) == 1 and v[0]["left"] and 1.9 <= v[0]["inside_s"] <= 2.1
    assert visits(_walk([95, 105, 95]), area) == []    # a shake over the edge
    zones = [{"uid": "A", "kind": "line", "points": [[60, 0], [60, 200]]}, {"uid": "B", "kind": "area", "points": area},
             {"uid": "C", "kind": "line", "points": [[240, 0], [240, 200]]}]
    p = path(b, zones)
    assert p["path"] == ["A", "B", "C"] and went(p["path"], "A", "C") and not went(p["path"], "C", "A")


def test_zones_count_a_camera_and_reach_the_api_and_export(tmp_path):
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (7.0, painted_cars, False),
                           "CAM1_20261021T121000Z.mp4": (4.0, painted_cars, False)})
    run(m)
    svc = VideoService(m)
    evs = svc.events()
    vid = evs[0].source_version_id
    z = Zones(m)
    with pytest.raises(ZoneError):
        z.add(vid, "line", "Stop line", [[0, 0], [3, 3]], "lior")
    line = z.add(vid, "line", "Stop line", [[160, 200], [160, 0]], "lior")     # drawn upwards: "in" is to its right
    z.add(vid, "area", "Junction", [[200, 0], [320, 0], [320, 180], [200, 180]], "lior")
    assert z.add(vid, "line", "Stop line", [[10, 0], [10, 100]], "lior")["name"] == "Stop line 2"
    other = next(e.source_version_id for e in evs if e.source_version_id != vid)
    assert [x["name"] for x in z.of(other)] == ["Stop line", "Junction", "Stop line 2"]     # same camera folder
    t = tally(svc, evs)
    rec = next(r for r in t["recordings"] if r["source_version_id"] == vid)
    stop = next(x for x in rec["zones"] if x["name"] == "Stop line")
    assert stop["by_class"]["car"]["in"] == 2 and stop["by_class"]["car"]["out"] == 0
    junction = next(x for x in rec["zones"] if x["name"] == "Junction")
    assert junction["by_class"]["car"]["entries"] >= 1
    z.delete(line["uid"], "lior")
    with pytest.raises(ZoneError):
        z.delete(line["uid"], "lior")
    assert [x["name"] for x in z.of(vid)] == ["Junction", "Stop line 2"]
    import csv
    files = svc.export("lior")
    sheet = next(p for p in files if p.name.endswith("_lines_and_areas.csv"))
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert {r["zone"] for r in rows} >= {"Junction"}
    ev_rows = list(csv.DictReader(next(p for p in files if p.name.endswith("_events.csv")).open(encoding="utf-8-sig")))
    assert any("Junction" in r["path"] for r in ev_rows)


def test_heatmap_and_path_map(tmp_path):
    import base64

    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from trafficwb.video_maps import background, heat
    from trafficwb.web.app import create_app
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (7.0, painted_cars, False)})
    run(m)
    svc = VideoService(m)
    vid = svc.events()[0].source_version_id
    bg = background(svc, vid)
    assert bg.shape[:2] == (180, 320) and float(np.abs(bg.astype(int) - 100).mean()) < 6   # the cars are gone
    # a target that waits 3 s makes the hottest spot
    acc = heat(bg.shape, [[(t * 0.1, [100, 80, 140, 100]) for t in range(31)], [(t * 0.1, [t * 10, 20, t * 10 + 40, 40]) for t in range(30)]])
    y, x = np.unravel_index(int(acc.argmax()), acc.shape)
    assert abs(x - 120) <= 3 and abs(y - 100) <= 3
    c = TestClient(create_app(tmp_path / "data", start_monitoring=False))
    h = {"X-TWB": "1"}
    for kind in ("heat", "paths"):
        r = c.post(f"/api/missions/{m.id}/recordings/{vid}/map", json={"kind": kind}, headers=h)
        assert r.status_code == 200, r.text
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r.json()["image"].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (180, 320)
    assert r.json()["legend"]["classes"] == {"car": "#4aa3e8"}
    one = svc.events()[0].id
    assert c.post(f"/api/missions/{m.id}/recordings/{vid}/map", json={"kind": "heat", "event_ids": [one]},
                  headers=h).json()["legend"]["targets"] == 1
