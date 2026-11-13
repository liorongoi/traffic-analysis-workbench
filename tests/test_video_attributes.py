"""Phase 6a: what each video target did and looked like (direction, speed, time in view and stopped, size, colour),
measuring the camera, and how many were in view at once."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_video import H, W, mission, run  # noqa: E402

from trafficwb.video_attributes import Calibration, in_view, motion, read_missing_colours  # noqa: E402
from trafficwb.video_review import VideoService  # noqa: E402

av = pytest.importorskip("av")


def painted_cars(t):
    """Two cars moving right at 70 and 90 px/s, painted a red the colour reader takes for paint, not a lamp."""
    import cv2
    import numpy as np
    img = np.full((H, W, 3), 100, np.uint8)
    for x0, y, v in ((-40, 40, 70), (-200, 110, 90)):
        x = int(x0 + v * t)
        cv2.rectangle(img, (x, y), (x + 40, y + 24), (50, 50, 205), -1)
    return img


def test_motion_from_boxes():
    # 40 px wide, 20 px high, moving right at 60 px/s for 4 s, then stopped 3 s
    boxes = [(t / 10, [t * 6.0, 50, t * 6.0 + 40, 70]) for t in range(41)]
    boxes += [(4 + t / 10, [240, 50, 280, 70]) for t in range(1, 31)]
    a = motion(boxes)
    assert a["direction"] == "right" and a["unit"] == "px/s" and a["dwell_s"] == 7.0
    assert 55 <= a["top_speed"] <= 65 and 2.5 <= a["stopped_s"] <= 3.5 and (a["w"], a["h"]) == (40, 20)
    k = motion(boxes, m_per_px=0.1)                       # 60 px/s = 6 m/s = 21.6 km/h
    assert k["unit"] == "km/h" and 20 <= k["top_speed"] <= 23 and (k["w"], k["h"]) == (4.0, 2.0)
    up_left = [(t, [100 - 20 * t, 100 - 20 * t, 120 - 20 * t, 120 - 20 * t]) for t in range(4)]
    assert motion(up_left)["direction"] == "up-left"
    assert motion([(0, [0, 0, 10, 10]), (5, [3, 2, 13, 12])])["direction"] == "stayed"


def test_cars_get_direction_speed_colour_and_a_measured_camera(tmp_path):
    cars = painted_cars
    m = mission(tmp_path, {"CAM1_20261021T120000Z.mp4": (7.0, cars, False), "CAM1_20261021T121000Z.mp4": (4.0, cars, False)})
    run(m)
    svc = VideoService(m)
    evs = svc.events()
    first, second = sorted({e.source_version_id for e in evs})
    a = svc.attributes(evs)
    one = [e for e in evs if e.source_version_id == first]
    assert {a[e.id]["direction"] for e in one} == {"right"} and all(a[e.id]["unit"] == "px/s" for e in one)
    assert sorted(round(a[e.id]["speed"] / 10) for e in one) == [7, 9]
    assert all(a[e.id]["colour"] is None for e in evs)                     # not read yet
    assert read_missing_colours(svc, evs) == len(evs) and read_missing_colours(svc, evs) == 0
    a = svc.attributes(evs)
    assert {tuple(a[e.id]["colour"]) for e in evs} == {("red",)}
    # measure the camera on the first recording: 40 px (a car) is 4.5 m; the other one, same folder, uses it too
    cal = Calibration(m)
    with pytest.raises(ValueError):
        cal.set(first, [0, 0, 3, 3], 4.5, "lior")
    c = cal.set(first, [100, 60, 140, 60], 4.5, "lior")
    assert c["own"] and abs(c["m_per_px"] - 4.5 / 40) < 1e-9
    assert cal.get(second)["own"] is False
    a = svc.attributes(evs)
    kmh = sorted(a[e.id]["speed"] for e in one)
    assert a[one[0].id]["unit"] == "km/h" and 25 <= kmh[0] <= 32 and 33 <= kmh[1] <= 40   # 70 and 90 px/s
    assert abs(a[one[0].id]["w"] - 4.5) < 0.6
    cal.clear(second, "lior")
    assert cal.get(second) is None and cal.get(first)["own"]
    # the export carries them
    import csv
    path = next(p for p in svc.export("lior") if p.name.endswith("_events.csv"))
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert {r["direction"] for r in rows} == {"right"} and {r["colour"] for r in rows} == {"red"}
    assert {r["speed_unit"] for r in rows} == {"km/h", "px/s"}


class _E:
    def __init__(self, cls, a, b, counted=True):
        self.cls, self.parts, self.counted = cls, [(a, b)], counted


def test_in_view_at_once_and_crowd_periods():
    evs = [_E("person", 0, 10), _E("person", 2, 8), _E("person", 3, 5), _E("person", 4, 6, counted=False),
           _E("car", 0, 1), _E("truck", 0.5, 2)]
    r = in_view(evs, crowd_at=3)
    assert r["families"]["person"] == {"max": 3, "at_s": 3} and r["families"]["vehicle"]["max"] == 2
    assert r["crowd"] == [{"start_s": 3, "end_s": 5, "peak": 3}]
    assert in_view(evs)["crowd"] == []
