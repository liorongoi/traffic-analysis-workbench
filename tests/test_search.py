"""Object search: interpretation, spec validation, colour reading and search runs (IMG-03)."""
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from trafficwb.attributes import read_colour
from trafficwb.detect import Detection
from trafficwb.interpret import Context, SpecError, apply_edits, describe, empty_spec, interpret, target, validate_spec
from trafficwb.search import time_ok
from trafficwb.web.app import create_app

# ------------------------------------------------------------------ interpretation

CTX = Context(("parking lot", "bridge"))

# A written request set: request -> expected fields (only the fields named are checked).
REQUESTS = [
    ("Find red pickup trucks.", {"targets": [target("truck", "pickup", ["red"])]}),
    ("red cars", {"targets": [target("car", None, ["red"])]}),
    ("show me all buses", {"targets": [target("bus")]}),
    ("white vans", {"targets": [target("vehicle", "van", ["white"])]}),
    ("red or blue cars", {"targets": [target("car", None, ["red", "blue"])]}),
    ("red cars and blue buses", {"targets": [target("car", None, ["red"]), target("bus", None, ["blue"])]}),
    ("cars that are not red", {"targets": [target("car", None, [], ["red"])]}),
    ("trucks, not white", {"targets": [target("truck", None, [], ["white"])]}),
    ("non-white trucks", {"targets": [target("truck", None, [], ["white"])]}),
    ("vehicles except buses", {"targets": [target("vehicle")], "exclude": {"categories": ["bus"], "subtypes": []}}),
    ("cars but no taxis", {"targets": [target("car")], "exclude": {"categories": [], "subtypes": ["taxi"]}}),
    ("silver SUVs", {"targets": [target("car", "suv", ["grey"])]}),
    ("gray semi-trailers", {"targets": [target("truck", "semi_trailer", ["grey"])]}),
    ("motorbikes and bicycles", {"targets": [target("motorcycle"), target("bicycle")]}),
    ("green", {"targets": [target("vehicle", None, ["green"])]}),
    ("anything", {"targets": [target("vehicle")], "not_understood": ["anything"]}),
    ("trucks at night", {"time.conditions": ["night"]}),
    ("cars in daylight", {"time.conditions": ["day"]}),
    ("buses between 17:00 and 05:00",
     {"time.windows": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"], "start": "17:00", "end": "05:00"}]}),
    ("buses from 7am to 9:30am", {"time.windows": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                                                    "start": "07:00", "end": "09:30"}]}),
    ("cars 22:00-02:00 on sunday", {"time.windows": [{"days": ["sun"], "start": "22:00", "end": "02:00"}]}),
    ("trucks from monday to friday after 22:00",
     {"time.windows": [{"days": ["mon", "tue", "wed", "thu", "fri"], "start": "22:00", "end": "24:00"}]}),
    ("trucks before 6:00", {"time.windows": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                                              "start": "00:00", "end": "06:00"}]}),
    ("buses on saturday", {"time.windows": [{"days": ["sat"], "start": "00:00", "end": "24:00"}]}),
    ("friday to sunday buses", {"time.windows": [{"days": ["fri", "sat", "sun"], "start": "00:00", "end": "24:00"}]}),
    ("cars from 2026-10-01 to 2026-10-21", {"time.date_from": "2026-10-01", "time.date_to": "2026-10-21"}),
    ("cars on 21.10.2026", {"time.date_from": "2026-10-21", "time.date_to": "2026-10-21"}),
    ("at least 3 trucks", {"min_count": 3}),
    ("5 or more cars", {"min_count": 5}),
    ("red cars, only certain ones", {"uncertain": "exclude"}),
    ("red cars, no uncertain results", {"uncertain": "exclude"}),
    ("white vans in parking lot", {"regions": {"include": ["parking lot"], "exclude": []}}),
    ("cars outside the bridge", {"regions": {"include": [], "exclude": ["bridge"]}}),
    ("cars in region harbour", {"regions": {"include": [], "exclude": []}}),
]


def _get(spec, dotted):
    v = spec
    for k in dotted.split("."):
        v = v[k]
    return v


@pytest.mark.parametrize("text,expected", REQUESTS, ids=[r[0] for r in REQUESTS])
def test_request_set(text, expected):
    spec = interpret(text, CTX)
    assert validate_spec(spec) == []
    for k, v in expected.items():
        got = _get(spec, k)
        if k == "not_understood":
            assert all(any(x in g for g in got) for x in v), got
        elif k == "targets":
            assert got == v, got
        else:
            assert got == v, (k, got)


def test_interpretation_accuracy_report():
    """The share of the written request set interpreted exactly (all expected fields), reported for the record."""
    ok = 0
    for text, expected in REQUESTS:
        spec = interpret(text, CTX)
        good = True
        for k, v in expected.items():
            got = _get(spec, k)
            good &= (all(any(x in g for g in got) for x in v) if k == "not_understood" else got == v)
        ok += good
    assert ok == len(REQUESTS)


def test_unknown_words_are_listed_not_dropped():
    spec = interpret("red parked cars near the exit", CTX)
    assert spec["targets"] == [target("car", None, ["red"])]
    assert {"parked", "near", "exit"} <= set(spec["not_understood"])


def test_country_dependent_words_are_not_guessed():
    for w in ("weekends", "weekdays", "working days"):
        spec = interpret(f"buses on {w}", CTX)
        assert spec["time"]["windows"] == []
        assert any("differ by country" in x for x in spec["not_understood"])


def test_unknown_region_name_is_reported():
    spec = interpret("cars in region harbour", CTX)
    assert any("harbour" in x for x in spec["not_understood"])


def test_subtype_note_and_description():
    spec = interpret("red pickup trucks", CTX)
    assert any("not identified by the current detector" in n for n in spec["notes"])
    assert any("colour cannot be seen in infrared" in n.lower() for n in spec["notes"])
    assert describe(spec)[0].startswith("Find red pickups")


def test_code_like_text_is_only_words():
    for text in ("__import__('os').system('rm -rf /')", "cars; DROP TABLE detection; --", "{{7*7}} ${jndi:x} `ls`"):
        spec = interpret(text, CTX)
        assert validate_spec(spec) == []
        assert spec["not_understood"], text
        assert json.loads(json.dumps(spec)) == spec   # plain data only


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: s.update(extra=1), "unknown keys"),
    (lambda s: s.pop("time"), "missing keys"),
    (lambda s: s.update(targets=[]), "at least one target"),
    (lambda s: s["targets"][0].update(category="tank"), "unknown category"),
    (lambda s: s["targets"][0].update(subtype="limousine"), "unknown subtype"),
    (lambda s: s["targets"][0].update(colours=["mauve"]), "colours must be"),
    (lambda s: s["targets"][0].update(colours=["red"], exclude_colours=["red"]), "both required and excluded"),
    (lambda s: s["targets"][0].update(code="print(1)"), "must have category"),
    (lambda s: s["exclude"].update(categories=["vehicle"]), "detector classes"),
    (lambda s: s["time"].update(windows=[{"days": ["mon"], "start": "25:00", "end": "05:00"}]), "start must be HH:MM"),
    (lambda s: s["time"].update(windows=[{"days": ["mon"], "start": "05:00", "end": "05:00"}]), "must differ"),
    (lambda s: s["time"].update(windows=[{"days": [], "start": "01:00", "end": "05:00"}]), "days must be"),
    (lambda s: s["time"].update(date_from="2026-10-21", date_to="2026-10-01"), "after"),
    (lambda s: s["time"].update(conditions=["fog"]), "conditions"),
    (lambda s: s.update(min_count=0), "min_count"),
    (lambda s: s.update(min_count=True), "min_count"),
    (lambda s: s.update(uncertain="guess"), "uncertain"),
    (lambda s: s["regions"].update(include=[""]), "region names"),
])
def test_invalid_specs_are_rejected(mutate, msg):
    spec = interpret("red cars", CTX)
    mutate(spec)
    errs = validate_spec(spec)
    assert any(msg in e for e in errs), errs
    with pytest.raises(SpecError):
        apply_edits(interpret("red cars", CTX), spec)


def test_validate_never_raises_on_garbage():
    for junk in (None, 3, "x", [], {"schema": 1}, {k: None for k in empty_spec()}):
        assert validate_spec(junk)


def test_time_rule_matches_schedules():
    spec = interpret("cars between 17:00 and 05:00", CTX)
    sec = timedelta(seconds=1)
    assert time_ok(spec, datetime(2026, 10, 21, 17, 0, 0), sec)
    assert time_ok(spec, datetime(2026, 10, 22, 4, 59, 59), sec)
    assert not time_ok(spec, datetime(2026, 10, 22, 5, 0, 0), sec)
    assert time_ok(spec, datetime(2026, 10, 22, 4, 59), timedelta(minutes=1))     # the whole minute ends at 05:00
    whole = interpret("cars on saturday", CTX)
    assert time_ok(whole, datetime(2026, 10, 24, 23, 59, 0), timedelta(minutes=1))   # Saturday 23:59 minute
    assert not time_ok(whole, datetime(2026, 10, 25, 0, 0, 0), sec)
    late = interpret("cars after 22:00", CTX)
    assert time_ok(late, datetime(2026, 10, 21, 23, 59, 59), sec)
    assert not time_ok(late, datetime(2026, 10, 22, 0, 0, 0), sec)
    dates = interpret("cars from 2026-10-01 to 2026-10-21", CTX)
    assert time_ok(dates, datetime(2026, 10, 21, 23, 59), timedelta(minutes=1))
    assert not time_ok(dates, datetime(2026, 10, 22, 0, 0), sec)


# ------------------------------------------------------------------ colour

ROAD = (112, 110, 108)


def road(h=240, w=320, cast=(0, 0, 0)):
    rng = np.random.default_rng(3)
    img = np.full((h, w, 3), ROAD, np.int16) + np.array(cast, np.int16)
    img += rng.integers(-6, 7, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def car(img, box, bgr, glass=True):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
    if glass:
        cv2.rectangle(img, (x1 + 6, y1 + 3), (x2 - 6, y1 + (y2 - y1) // 4), (35, 35, 35), -1)
    return img


@pytest.mark.parametrize("bgr,name", [((40, 40, 200), "red"), ((235, 235, 235), "white"), ((190, 90, 30), "blue"),
                                      ((40, 160, 40), "green"), ((30, 200, 230), "yellow"), ((112, 112, 112), "grey"),
                                      ((20, 20, 20), "black")])
def test_colour_observed_on_plain_paint(bgr, name):
    img = car(road(), (100, 80, 200, 160), bgr, glass=False)
    r = read_colour(img, (100, 80, 200, 160), monochrome=False, condition="day")
    assert r.state == "observed" and r.values == [name], (r.values, r.shares)


def test_colour_cast_from_light_is_removed():
    cast = (25, 5, -15)        # bluish morning light on grey paint and grey road alike
    img = road(cast=cast)
    car(img, (100, 80, 200, 160), tuple(int(c) for c in np.clip(np.array((120, 120, 120)) + cast, 0, 255)), glass=False)
    r = read_colour(img, (100, 80, 200, 160), False, "day")
    assert r.values == ["grey"], r.shares


def test_lit_brake_lamps_are_not_paint():
    img = car(road(), (100, 80, 200, 160), (110, 110, 110), glass=False)
    for x in (104, 180):
        cv2.rectangle(img, (x, 120), (x + 16, 132), (40, 40, 255), -1)
    r = read_colour(img, (100, 80, 200, 160), False, "day")
    assert "red" not in r.values, r.shares


def test_colour_not_observable_cases():
    img = car(road(), (100, 80, 200, 160), (40, 40, 200))
    mono = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert read_colour(mono, (100, 80, 200, 160), True).state == "not_observable"
    assert read_colour(img, (100, 80, 108, 86), False).reason == "vehicle too small in the image"
    dark = (img * 0.15).astype(np.uint8)
    assert read_colour(dark, (100, 80, 200, 160), False).reason == "too dark"


def test_poor_light_is_never_observed():
    img = car(road(), (100, 80, 200, 160), (40, 40, 200), glass=False)
    r = read_colour(img, (100, 80, 200, 160), False, condition="dusk")
    assert r.state == "uncertain" and "red" in r.values


def test_competing_colours_are_uncertain():
    img = road()
    cv2.rectangle(img, (100, 80), (150, 160), (40, 40, 200), -1)
    cv2.rectangle(img, (150, 80), (200, 160), (190, 90, 30), -1)
    r = read_colour(img, (100, 80, 200, 160), False, "day")
    assert r.state == "uncertain" and set(r.values) >= {"red", "blue"}


# ------------------------------------------------------------------ search runs through the API

W, H = 640, 360
OLD = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
H1 = {"X-TWB": "1"}
BOXES = [((100, 200, 170, 250), "car", (40, 40, 200)),        # red car
         ((300, 200, 370, 250), "car", (235, 235, 235)),      # white car
         ((450, 140, 540, 220), "truck", (190, 90, 30)),      # blue truck
         ((600, 100, 608, 106), "car", (40, 40, 200))]        # too small for colour
MONO = 5


def search_scene(k):
    img = road(H, W)
    for (x1, y1, x2, y2), _, bgr in BOXES:
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
    # structure so that view grouping has something to hold on to
    cv2.line(img, (0, 300), (W, 280), (200, 200, 200), 3)
    cv2.line(img, (0, 120), (W, 100), (60, 60, 60), 3)
    if k == MONO:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


class FixedDetector:
    def describe(self):
        return {"model": {"name": "fixed", "version": "1", "sha256": "1" * 64}, "preprocessing": {}}

    def detect(self, img):
        return [Detection(tuple(float(v) for v in b), cls, 0.9, {cls: 0.9}) for b, cls, _ in BOXES]


def ts(i):
    return (datetime(2026, 10, 21, tzinfo=timezone.utc) + timedelta(minutes=10 * i)).strftime("%Y%m%dT%H%M%SZ")


@pytest.fixture
def api(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for k in range(6):
        p = src / f"CAM9_{ts(k)}.png"
        cv2.imwrite(str(p), search_scene(k))
        os.utime(p, (OLD, OLD))
    app = create_app(tmp_path / "data", detector_factory=lambda: (FixedDetector(), {"task": "vehicle_detection"}),
                     start_monitoring=False, stability_seconds=0)
    c = TestClient(app)
    def make(mode):
        body = {"name": f"S {mode}", "mode": mode, "country": "IL", "iana_zone": "Asia/Jerusalem", "source_dir": str(src),
                "include_subdirs": False, "processing": "one_time", "output_dir": str(out), "export_formats": ["csv"]}
        mid = c.post("/api/missions", json=body, headers=H1).json()["id"]
        c.post(f"/api/missions/{mid}/job", headers=H1)
        wait(c, mid)
        return mid
    return c, make, out


def wait(c, mid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        j = c.get(f"/api/missions/{mid}").json()["job"]
        if j and j["state"] in ("finished", "cancelled", "failed"):
            assert j["state"] == "finished", j
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def search(c, mid, text, ignore=False, spec=None):
    r = c.post(f"/api/missions/{mid}/search/ask", json={"text": text}, headers=H1)
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    body = {"ignore_unknown": ignore}
    if spec is not None:
        body["spec"] = spec(r.json()["spec"])
    cr = c.post(f"/api/missions/{mid}/search/{rid}/confirm", json=body, headers=H1)
    assert cr.status_code == 200, cr.text
    assert c.post(f"/api/missions/{mid}/search/{rid}/run", headers=H1).status_code == 200
    j = wait(c, mid)
    run = j["result"]["search_run"]
    return c.get(f"/api/missions/{mid}/search-runs/{run}?limit=500").json()


def test_search_end_to_end(api):
    c, make, out = api
    mid = make("object_search")

    red = search(c, mid, "red cars")
    s = red["summary"]
    assert s["images_searched"] == 6 and s["objects_checked"] == 24
    matches = [h for h in red["hits"] if h["status"] == "match"]
    possible = [h for h in red["hits"] if h["status"] == "possible"]
    assert len(matches) == 5 and all(h["box"][0] == 100 for h in matches)          # the red car, colour images
    # the tiny car everywhere, plus both cars in the infrared image
    assert len(possible) == 8
    assert all(any("not observable" in r for r in h["reasons"]) for h in possible)
    assert s["colour"]["not_observable"] == 8

    certain = search(c, mid, "red cars, only certain ones")
    assert certain["summary"]["hits_possible"] == 0 and certain["summary"]["possible_left_out"] == 8

    trucks = search(c, mid, "trucks, not white")
    assert [h["status"] for h in trucks["hits"]].count("match") == 5
    assert [h["status"] for h in trucks["hits"]].count("possible") == 1

    pickups = search(c, mid, "red pickup trucks")
    assert pickups["summary"]["hits_match"] == 0
    assert pickups["hits"] and all(any("pickup is not identified" in r for r in h["reasons"]) for h in pickups["hits"])

    early = search(c, mid, "red cars between 03:00 and 03:30")      # local time, Asia/Jerusalem is UTC+3 on this date
    assert early["summary"]["images_searched"] == 3 and early["summary"]["images_outside_time"] == 3

    night = search(c, mid, "cars at night")
    assert night["summary"]["images_searched"] == 1 and night["summary"]["images_other_condition"] == 5

    busy = search(c, mid, "at least 4 vehicles")
    assert busy["summary"]["images_with_hits"] == 6
    none = search(c, mid, "at least 5 vehicles")
    assert none["summary"]["images_with_hits"] == 0 and none["summary"]["images_below_min_count"] == 6


def test_search_needs_confirmation_and_known_words(api):
    c, make, _ = api
    mid = make("object_search")
    r = c.post(f"/api/missions/{mid}/search/ask", json={"text": "red parked cars"}, headers=H1).json()
    assert "parked" in r["spec"]["not_understood"]
    run = c.post(f"/api/missions/{mid}/search/{r['id']}/run", headers=H1)
    assert run.status_code == 400 and "Confirm" in run.json()["message"]
    bad = c.post(f"/api/missions/{mid}/search/{r['id']}/confirm", json={}, headers=H1)
    assert bad.status_code == 400 and "not understood" in bad.json()["message"]
    edited = dict(r["spec"], not_understood=[], extra="x")
    bad2 = c.post(f"/api/missions/{mid}/search/{r['id']}/confirm", json={"spec": edited}, headers=H1)
    assert bad2.status_code == 400 and "not valid" in bad2.json()["message"]
    ok = c.post(f"/api/missions/{mid}/search/{r['id']}/confirm", json={"ignore_unknown": True}, headers=H1)
    assert ok.status_code == 200 and ok.json()["confirmed"]
    again = c.post(f"/api/missions/{mid}/search/{r['id']}/confirm", json={"ignore_unknown": True}, headers=H1)
    assert again.status_code == 400
    # the operator's edit is stored next to the machine interpretation
    r2 = c.post(f"/api/missions/{mid}/search/ask", json={"text": "red cars"}, headers=H1).json()
    spec = r2["spec"]
    spec["targets"][0]["colours"] = ["white"]
    done = c.post(f"/api/missions/{mid}/search/{r2['id']}/confirm", json={"spec": spec}, headers=H1).json()
    assert done["machine_spec"]["targets"][0]["colours"] == ["red"]
    assert done["spec"]["targets"][0]["colours"] == ["white"] and "targets" in done["edits"]
    assert c.post(f"/api/missions/{mid}/search/ask", json={"text": "x"}).status_code == 403   # header required


def test_search_only_in_object_search_missions(api):
    c, make, _ = api
    mid = make("general_traffic")
    r = c.post(f"/api/missions/{mid}/search/ask", json={"text": "red cars"}, headers=H1)
    assert r.status_code == 400 and "object search missions only" in r.json()["message"]


def test_verdicts_colour_corrections_and_export(api):
    c, make, out = api
    mid = make("object_search")
    first = search(c, mid, "red cars")
    hit = next(h for h in first["hits"] if h["status"] == "match")
    assert c.post(f"/api/missions/{mid}/search-hits/{hit['id']}/decision", json={"decision": "correct"},
                  headers=H1).status_code == 200
    assert c.post(f"/api/missions/{mid}/search-hits/{hit['id']}/decision", json={"decision": "maybe"},
                  headers=H1).status_code == 400
    again = c.get(f"/api/missions/{mid}/search-runs/{first['id']}?decision=correct").json()
    assert again["total"] == 1

    # The operator says the white car in the first image is red: the next search follows.
    vid = hit["source_version_id"]
    white_uid = next(o["uid"] for o in c.post(f"/api/missions/{mid}/items/{vid}", json={}, headers=H1).json()["objects"]
                     if o["box"][0] == 300)
    assert c.put(f"/api/missions/{mid}/items/{vid}/objects/{white_uid}/colour", json={"value": "red"},
                 headers=H1).status_code == 200
    assert c.put(f"/api/missions/{mid}/items/{vid}/objects/{white_uid}/colour", json={"value": "mauve"},
                 headers=H1).status_code == 400
    second = search(c, mid, "red cars")
    assert second["summary"]["hits_match"] == 6
    corrected = next(h for h in second["hits"] if h["object_uid"] == white_uid and h["source_version_id"] == vid)
    assert corrected["colour"]["algo"] == "operator"
    assert first["summary"]["hits_match"] == 5                              # earlier results are kept as run

    files = c.post(f"/api/missions/{mid}/search-runs/{first['id']}/export", json={}, headers=H1).json()["files"]
    hits_csv = next(f for f in files if f.endswith("_hits.csv"))
    with open(hits_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == first["total"]
    assert {"match", "possible"} == {r["status"] for r in rows}
    assert any(r["operator_decision"] == "correct" for r in rows)
    spec_csv = next(f for f in files if f.endswith("_search.csv"))
    with open(spec_csv, encoding="utf-8-sig") as f:
        meta = list(csv.DictReader(f))[0]
    assert meta["request"] == "red cars" and json.loads(meta["spec_json"])["targets"][0]["colours"] == ["red"]
    files2 = c.post(f"/api/missions/{mid}/search-runs/{first['id']}/export", json={}, headers=H1).json()["files"]
    assert set(files2).isdisjoint(files)                                     # never overwritten
    crop = c.get(f"/api/missions/{mid}/crops/{vid}?box=100,200,170,250")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"


def test_named_regions(api):
    c, make, _ = api
    mid = make("object_search")
    views = c.get(f"/api/missions/{mid}/views").json()["views"]
    view = next(v for v in views if v.get("size", 0) >= 5 or v.get("count", 0) >= 5) if len(views) > 1 else views[0]
    r = c.post(f"/api/missions/{mid}/views/{view['id']}/regions", headers=H1,
               json={"kind": "include", "name": "left lane", "polygon": [[80, 180], [200, 180], [200, 270], [80, 270]]})
    assert r.status_code == 200, r.text
    res = search(c, mid, "cars in left lane")
    assert res["hits"] and all(h["box"][0] == 100 for h in res["hits"])
    out = search(c, mid, "cars outside left lane")
    assert out["hits"] and all(h["box"][0] != 100 for h in out["hits"])
