"""Object search over a still-image mission (IMG-03).

Flow: typed request -> interpret() -> spec shown to the operator -> confirm
(optionally edited) -> run -> hits the operator can mark correct or not a match.

Rules:
- Only a confirmed spec runs. If the interpreter did not understand some words,
  the operator must edit them away or explicitly choose to ignore them.
- The search works on the current objects of each image: machine detections
  with operator corrections applied, removed objects left out, plus night
  candidates an operator confirmed as vehicles.
- Each object gets one of: match, possible (with the reasons), or no match.
  "Possible" is used whenever a required fact cannot be established: a subtype
  the detector does not identify, colour uncertain or not observable, or
  regions not applied because the view needs review. When the spec says
  "only certain matches", possible hits are left out and counted.
- Unresolved night light candidates are not searched: type and colour cannot
  be known from lights. The summary says how many there were.
- Where: if the spec names regions, the object's bottom-centre point must lie
  in one of the named regions of its view (whatever that region's kind), and
  outside any named exclusion. Otherwise the view's own analysis regions apply.
- When: local time windows and dates in the mission zone, using the capture's
  full stated interval (same rule as scanning schedules); day/dusk/night is the
  measured image condition.
- Operator colour corrections override the machine reading in later searches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from .attributes import COLOUR_ALGO, ColourReading, read_colour
from .interpret import (INTERPRETER_VERSION, COLOURS, Context, SpecError, apply_edits, describe, detector_classes,
                        interpret, validate_spec)
from .localtime import to_local
from .regions import RegionSet, point_in_polygon
from .review import ReviewService
from .schedule import WEEKDAYS, Schedule, Window
from .store import Mission, utcnow
from .timeparse import Precision

DECISIONS = ("correct", "not_a_match", "undecided")


class SearchError(ValueError):
    pass


@dataclass
class Hit:
    id: int
    source_version_id: int
    filename: str
    capture_local: str | None
    object_uid: str
    status: str
    reasons: list[str]
    target: int
    cls: str
    box: tuple[float, float, float, float]
    colour: dict
    decision: str = "undecided"
    capture_utc: str | None = None          # for a video event: the event's start
    event_t: float | None = None            # for a video event: seconds from the start of the recording


@dataclass
class RunControlLite:
    cancelled: bool = False
    stage: str = ""

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class _Decision:
    status: str | None              # match | possible | None (no match)
    reasons: list[str] = field(default_factory=list)


def time_ok(spec: dict, local: datetime, span: timedelta) -> bool:
    """True if the naive local interval [local, local+span) satisfies the spec's dates and windows.

    A window ending at 24:00 ends at midnight; 00:00-24:00 is the whole day.
    """
    tm = spec["time"]
    df = date.fromisoformat(tm["date_from"]) if tm["date_from"] else None
    dt = date.fromisoformat(tm["date_to"]) if tm["date_to"] else None
    if not tm["windows"]:
        d = local.date()
        return (df is None or d >= df) and (dt is None or d <= dt)
    for w in tm["windows"]:
        days = frozenset(WEEKDAYS.index(d) for d in w["days"])
        if w["start"] == "00:00" and w["end"] == "24:00":
            d = local.date()
            if d.weekday() in days and (df is None or d >= df) and (dt is None or d <= dt) \
                    and local + span <= datetime.combine(d + timedelta(days=1), time(0, 0)):
                return True
            continue
        end = time(0, 0) if w["end"] == "24:00" else time.fromisoformat(w["end"])
        if Schedule("search", (Window(days, time.fromisoformat(w["start"]), end),), df, dt).contains_local(local, span):
            return True
    return False


class SearchService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.rs = ReviewService(mission)

    # ---------- requests ----------
    def context(self) -> Context:
        names = sorted({r["name"] for r in self.con.execute(
            "SELECT r.name, r.deleted FROM region_revision r WHERE r.id=(SELECT MAX(id) FROM region_revision x "
            "WHERE x.region_uid=r.region_uid)") if r["name"] and not r["deleted"]})
        return Context(tuple(names))

    def _check_mode(self) -> None:
        mode = self.con.execute("SELECT mode FROM mission").fetchone()[0]
        if mode != "object_search":
            raise SearchError("Object search runs in object search missions only. Create an object search mission "
                              "over the same source folder.")

    def ask(self, text: str, author: str) -> dict:
        self._check_mode()
        text = (text or "").strip()
        if not text:
            raise SearchError("Type what to look for, for example 'red pickup trucks'.")
        if len(text) > 1000:
            raise SearchError("The request is too long (1000 characters at most).")
        spec = interpret(text, self.context())
        cur = self.con.execute(
            "INSERT INTO search_request(chat_text,interpreter_version,spec_json,created_by,created_at) VALUES (?,?,?,?,?)",
            (text, INTERPRETER_VERSION, json.dumps(spec), author, utcnow()))
        return self.request(cur.lastrowid)

    def request(self, request_id: int) -> dict:
        r = self.con.execute("SELECT * FROM search_request WHERE id=?", (request_id,)).fetchone()
        if r is None:
            raise SearchError("No such search request.")
        spec = json.loads(r["confirmed_spec_json"] or r["spec_json"])
        return {"id": r["id"], "text": r["chat_text"], "interpreter": r["interpreter_version"],
                "machine_spec": json.loads(r["spec_json"]), "spec": spec, "description": describe(spec),
                "confirmed": r["confirmed_at"] is not None, "confirmed_by": r["confirmed_by"],
                "confirmed_at": r["confirmed_at"], "edits": json.loads(r["edits_json"] or "{}"),
                "region_names": list(self.context().region_names), "created_at": r["created_at"]}

    def confirm(self, request_id: int, author: str, spec: dict | None = None, ignore_unknown: bool = False) -> dict:
        self._check_mode()
        r = self.con.execute("SELECT * FROM search_request WHERE id=?", (request_id,)).fetchone()
        if r is None:
            raise SearchError("No such search request.")
        if r["confirmed_at"]:
            raise SearchError("This request is already confirmed. Ask again to change it.")
        machine = json.loads(r["spec_json"])
        try:
            final, diff = apply_edits(machine, spec if spec is not None else machine)
        except SpecError as exc:
            raise SearchError(f"The search spec is not valid: {exc}") from exc
        unknown = final["not_understood"]
        if unknown and not ignore_unknown:
            raise SearchError("Some words were not understood: " + "; ".join(unknown)
                              + ". Edit the search, or choose to ignore them.")
        names = set(self.context().region_names)
        missing = [n for n in final["regions"]["include"] + final["regions"]["exclude"] if n not in names]
        if missing:
            raise SearchError(f"No region named {', '.join(missing)} in this mission.")
        self.con.execute(
            "UPDATE search_request SET confirmed_spec_json=?, edits_json=?, ignored_words=?, confirmed_by=?, confirmed_at=? "
            "WHERE id=?", (json.dumps(final), json.dumps(diff), json.dumps(unknown if ignore_unknown else []), author,
                           utcnow(), request_id))
        return self.request(request_id)

    def requests(self, limit: int = 50) -> list[dict]:
        out = []
        for r in self.con.execute("SELECT id FROM search_request ORDER BY id DESC LIMIT ?", (limit,)):
            q = self.request(r["id"])
            q["runs"] = [dict(x) | {"summary": json.loads(x["summary"] or "{}")} for x in self.con.execute(
                "SELECT id, state, started_at, finished_at, summary FROM search_run WHERE request_id=? ORDER BY id DESC",
                (r["id"],))]
            out.append(q)
        return out

    # ---------- attributes ----------
    def colour(self, version_id: int, obj_uid: str, box, img_loader, condition: str, monochrome: bool) -> ColourReading:
        human = self.con.execute(
            "SELECT value, author, created_at FROM attribute_revision WHERE source_version_id=? AND object_uid=? "
            "AND attribute='colour' ORDER BY id DESC LIMIT 1", (version_id, obj_uid)).fetchone()
        if human is not None and human["value"] is not None:
            if human["value"] == "not_observable":
                return ColourReading("not_observable", reason=f"set by {human['author']}", algo="operator")
            return ColourReading("observed", [human["value"]], reason=f"set by {human['author']}", algo="operator")
        key = json.dumps([round(float(v), 1) for v in box])
        row = self.con.execute(
            "SELECT reading FROM object_attribute WHERE source_version_id=? AND object_uid=? AND box=? AND attribute='colour' "
            "AND algo=?", (version_id, obj_uid, key, COLOUR_ALGO)).fetchone()
        if row:
            return ColourReading.from_dict(json.loads(row["reading"]))
        reading = read_colour(img_loader(), box, monochrome, condition)
        self.con.execute(
            "INSERT OR IGNORE INTO object_attribute(source_version_id,object_uid,box,attribute,reading,algo,created_at) "
            "VALUES (?,?,?,?,?,?,?)", (version_id, obj_uid, key, "colour", json.dumps(reading.as_dict()), COLOUR_ALGO, utcnow()))
        return reading

    def set_colour(self, version_id: int, obj_uid: str, value: str | None, author: str, note: str | None = None) -> None:
        if value is not None and value not in COLOURS and value != "not_observable":
            raise SearchError(f"Colour must be one of {', '.join(COLOURS)}, or not_observable.")
        if obj_uid.startswith("ev") and obj_uid[2:].isdigit():
            known = {f"ev{r[0]}" for r in self.con.execute(
                "SELECT id FROM video_event WHERE source_version_id=?", (version_id,))}
        else:
            known = {o.uid for o in self.rs.objects(version_id)} | {c.uid for c in self.rs.candidates(version_id)}
        if obj_uid not in known:
            raise SearchError("No such object in this image.")
        self.con.execute(
            "INSERT INTO attribute_revision(source_version_id,object_uid,attribute,value,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?)", (version_id, obj_uid, "colour", value, author, note, utcnow()))

    # ---------- matching ----------
    @staticmethod
    def _colour_test(reading: ColourReading, want: list[str], avoid: list[str]) -> _Decision:
        if not want and not avoid:
            return _Decision("match")
        if reading.state == "not_observable":
            return _Decision("possible", [f"colour not observable ({reading.reason})"])
        vals = reading.values
        if reading.state == "observed":
            c = vals[0]
            if want and c not in want:
                return _Decision(None)
            if c in avoid:
                return _Decision(None)
            return _Decision("match")
        # uncertain
        if want and not set(vals) & set(want):
            return _Decision(None)
        if avoid and set(vals) <= set(avoid):
            return _Decision(None)
        return _Decision("possible", [f"colour uncertain: {' or '.join(vals)}" + (f" ({reading.reason})" if reading.reason else "")])

    def _match(self, spec: dict, cls: str, colour_of) -> tuple[_Decision, int]:
        """Best decision over the spec's targets for one object, and the target index."""
        if cls in spec["exclude"]["categories"]:
            return _Decision(None), -1
        best, best_i = _Decision(None), -1
        for i, t in enumerate(spec["targets"]):
            if cls not in detector_classes(t):
                continue
            d = _Decision("match")
            if t["subtype"]:
                d = _Decision("possible", [f"{t['subtype'].replace('_', '-')} is not identified by the detector "
                                           f"(detected as {cls})"])
            for ex in spec["exclude"]["subtypes"]:
                from .interpret import SUBTYPES
                if cls in SUBTYPES[ex]:
                    d = _Decision("possible", d.reasons + [f"cannot rule out {ex.replace('_', '-')}"])
            if t["colours"] or t["exclude_colours"]:
                c = self._colour_test(colour_of(), t["colours"], t["exclude_colours"])
                if c.status is None:
                    continue
                if c.status == "possible":
                    d = _Decision("possible", d.reasons + c.reasons)
            if best.status is None or (best.status == "possible" and d.status == "match"):
                best, best_i = d, i
        return best, best_i

    # ---------- run ----------
    def run(self, request_id: int, author: str, control=None, progress=None) -> int:
        self._check_mode()
        req = self.con.execute("SELECT * FROM search_request WHERE id=?", (request_id,)).fetchone()
        if req is None or not req["confirmed_at"]:
            raise SearchError("Confirm the search before running it.")
        spec = json.loads(req["confirmed_spec_json"])
        errs = validate_spec(spec)
        if errs:
            raise SearchError("; ".join(errs))
        control = control or RunControlLite()
        run_id = self.con.execute(
            "INSERT INTO search_run(request_id,spec_json,state,attribute_algo,started_by,started_at) VALUES (?,?,?,?,?,?)",
            (request_id, json.dumps(spec), "running", COLOUR_ALGO, author, utcnow())).lastrowid
        try:
            summary = self._execute(run_id, spec, control, progress)
            state = "cancelled" if control.cancelled else "finished"
            self.con.execute("UPDATE search_run SET state=?, summary=?, finished_at=? WHERE id=?",
                             (state, json.dumps(summary), utcnow(), run_id))
            self.m.notify("info", "search_finished",
                          f"Search {'cancelled' if control.cancelled else 'finished'}: {summary['hits_match']} matches, "
                          f"{summary['hits_possible']} possible in {summary['images_with_hits']} images.", search_run=run_id)
        except Exception as exc:
            self.con.execute("UPDATE search_run SET state='failed', error=?, finished_at=? WHERE id=?",
                             (f"{type(exc).__name__}: {exc}"[:500], utcnow(), run_id))
            raise
        return run_id

    def _images(self) -> list:
        return self.con.execute(
            "SELECT v.id, f.rel_path, v.capture_utc, v.capture_precision, "
            " (SELECT a.view_id FROM view_assignment a WHERE a.source_version_id=v.id AND a.superseded_by IS NULL "
            "  ORDER BY a.id DESC LIMIT 1) AS view_id "
            "FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id) "
            "ORDER BY v.capture_utc, v.id").fetchall()

    def _execute(self, run_id: int, spec: dict, control, progress) -> dict:
        if self.m.input_type == "video":
            return self._execute_video(run_id, spec, control, progress)
        from .views import ViewService
        vs = ViewService(self.m)
        zone = self.m.zone
        named_in, named_out = spec["regions"]["include"], spec["regions"]["exclude"]
        region_cache: dict[int | None, RegionSet] = {}
        s = {"images_total": 0, "images_searched": 0, "images_not_processed": 0, "images_outside_time": 0,
             "images_other_condition": 0, "images_condition_unknown": 0, "images_without_named_region": 0,
             "objects_checked": 0, "hits_match": 0, "hits_possible": 0, "possible_left_out": 0,
             "images_with_hits": 0, "images_below_min_count": 0, "unresolved_night_candidates_not_searched": 0,
             "colour": {"observed": 0, "uncertain": 0, "not_observable": 0}}
        rows = self._images()
        s["images_total"] = len(rows)
        for n, r in enumerate(rows, 1):
            if hasattr(control, "checkpoint"):
                control.checkpoint()      # waits while paused
            if control.cancelled:
                break
            control.stage = "searching"
            vid = r["id"]
            try:
                item = self.rs._latest_item(vid)
                if item is None or item["processing_state"] != "done":
                    s["images_not_processed"] += 1
                    continue
                if (spec["time"]["windows"] or spec["time"]["date_from"] or spec["time"]["date_to"]):
                    if not r["capture_utc"]:
                        s["images_outside_time"] += 1
                        continue
                    utc = datetime.fromisoformat(r["capture_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
                    local = to_local(utc, zone).local.replace(tzinfo=None)
                    if not time_ok(spec, local, Precision(r["capture_precision"] or "second").span):
                        s["images_outside_time"] += 1
                        continue
                cond = self.con.execute("SELECT condition, monochrome FROM image_condition WHERE source_version_id=?",
                                        (vid,)).fetchone()
                if spec["time"]["conditions"]:
                    if cond is None:
                        s["images_condition_unknown"] += 1
                        continue
                    if cond["condition"] not in spec["time"]["conditions"]:
                        s["images_other_condition"] += 1
                        continue
                view_id = r["view_id"]
                if view_id not in region_cache:
                    region_cache[view_id] = RegionSet.load(self.con, view_id) if view_id is not None else RegionSet([])
                regions = region_cache[view_id]
                polys_in = [x.polygon for x in regions.include + regions.exclude if x.name in named_in]
                polys_out = [x.polygon for x in regions.include + regions.exclude if x.name in named_out]
                if named_in and not polys_in:
                    s["images_without_named_region"] += 1
                    continue
                s["images_searched"] += 1
                img_cache: dict = {}

                def load(v=vid):
                    if "img" not in img_cache:
                        img_cache["img"] = vs.image(v)
                    return img_cache["img"]

                condition = cond["condition"] if cond else "day"
                mono = bool(cond["monochrome"]) if cond else False
                objs = [(o.uid, o.cls, o.box, o.in_region) for o in self.rs.objects(vid, include_removed=False)]
                cands = self.rs.candidates(vid)
                objs += [(c.uid, c.cls or "car", c.box, 1) for c in cands if c.status == "confirmed"]
                s["unresolved_night_candidates_not_searched"] += sum(c.status == "unresolved" for c in cands)
                hits = []
                for uid, cls, box, in_region in objs:
                    s["objects_checked"] += 1
                    bx, by = (box[0] + box[2]) / 2, box[3]
                    reasons: list[str] = []
                    if named_in:
                        if not any(point_in_polygon(bx, by, p) for p in polys_in):
                            continue
                    elif in_region == 0:
                        continue
                    elif in_region is None:
                        reasons.append("analysis regions not applied (camera view needs review)")
                    if any(point_in_polygon(bx, by, p) for p in polys_out):
                        continue
                    memo: dict = {}

                    def colour_of(u=uid, b=box):
                        if "c" not in memo:
                            memo["c"] = self.colour(vid, u, b, load, condition, mono)
                            s["colour"][memo["c"].state] += 1
                        return memo["c"]

                    d, ti = self._match(spec, cls, colour_of)
                    if d.status is None:
                        continue
                    status = d.status if not reasons else "possible"
                    reasons = d.reasons + reasons
                    reading = memo.get("c")
                    hits.append((uid, status, reasons, ti, cls, box, reading.as_dict() if reading else None))
                if spec["uncertain"] == "exclude":
                    s["possible_left_out"] += sum(h[1] == "possible" for h in hits)
                    hits = [h for h in hits if h[1] == "match"]
                n_match = sum(h[1] == "match" for h in hits)
                if len(hits) < spec["min_count"]:
                    if hits:
                        s["images_below_min_count"] += 1
                    continue
                s["images_with_hits"] += 1
                s["hits_match"] += n_match
                s["hits_possible"] += len(hits) - n_match
                self.con.executemany(
                    "INSERT INTO search_hit(search_run_id,source_version_id,object_uid,status,reasons,target,cls,x1,y1,x2,y2,colour) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(run_id, vid, uid, st, json.dumps(rs), ti, cls, *[float(v) for v in box],
                      json.dumps(col or {"state": "not_needed"})) for uid, st, rs, ti, cls, box, col in hits])
            finally:
                if progress:
                    progress(n, len(rows))
        return s

    def _execute_video(self, run_id: int, spec: dict, control, progress) -> dict:
        """Search video events: one object per event, read on its best frame. Regions are not applied to video yet."""
        from .night import measure_condition
        from .video import frame_at
        from .video_review import VideoService
        vsv = VideoService(self.m)
        zone = self.m.zone
        s = {"events_total": 0, "events_searched": 0, "events_not_counted": 0, "events_outside_time": 0,
             "events_time_unknown": 0, "events_other_condition": 0, "hits_match": 0, "hits_possible": 0,
             "possible_left_out": 0, "recordings_with_hits": 0, "recordings_below_min_count": 0,
             "objects_checked": 0, "colour": {"observed": 0, "uncertain": 0, "not_observable": 0},
             "note": "Analysis regions are not applied to video in this version."}
        events = vsv.events()
        s["events_total"] = len(events)
        timed = bool(spec["time"]["windows"] or spec["time"]["date_from"] or spec["time"]["date_to"])
        by_rec: dict[int, list] = {}
        for n, e in enumerate(events, 1):
            if hasattr(control, "checkpoint"):
                control.checkpoint()
            if control.cancelled:
                break
            control.stage = "searching"
            try:
                if e.status in ("rejected", "duplicate", "merged"):
                    s["events_not_counted"] += 1
                    continue
                if timed:
                    if not e.first_utc:
                        s["events_time_unknown"] += 1
                        continue
                    utc = datetime.fromisoformat(e.first_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
                    local = to_local(utc, zone).local.replace(tzinfo=None)
                    span = Precision(e.utc_precision or "second").span
                    if not time_ok(spec, local, span):
                        s["events_outside_time"] += 1
                        continue
                cache: dict = {}

                def frame(ev=e):
                    if "f" not in cache:
                        f = frame_at(vsv.path(ev.source_version_id), ev.thumb_t)
                        if f is None:
                            raise SearchError(f"Frame {ev.thumb_t:.3f} s not found in {ev.file}.")
                        cache["f"] = f.image
                        cache["cond"] = measure_condition(f.image)
                    return cache["f"]

                if spec["time"]["conditions"]:
                    frame()
                    if cache["cond"].label not in spec["time"]["conditions"]:
                        s["events_other_condition"] += 1
                        continue
                s["events_searched"] += 1
                s["objects_checked"] += 1
                memo: dict = {}

                def colour_of(ev=e):
                    if "c" not in memo:
                        img = frame()
                        c = cache["cond"]
                        memo["c"] = self.colour(ev.source_version_id, f"ev{ev.id}", ev.thumb_box, lambda: img,
                                                c.label, c.monochrome)
                        s["colour"][memo["c"].state] += 1
                    return memo["c"]

                d, ti = self._match(spec, e.cls, colour_of)
                if d.status is None:
                    continue
                reasons = list(d.reasons)
                if "class_uncertain" in e.flags:
                    reasons.append(f"class uncertain over the track ({round(e.cls_share * 100)}% {e.cls})")
                    d = _Decision("possible", reasons)
                reading = memo.get("c")
                by_rec.setdefault(e.source_version_id, []).append(
                    (f"ev{e.id}", d.status, reasons, ti, e.cls, e.thumb_box, reading.as_dict() if reading else None))
            finally:
                if progress:
                    progress(n, len(events))
        for vid, hits in by_rec.items():
            if spec["uncertain"] == "exclude":
                s["possible_left_out"] += sum(h[1] == "possible" for h in hits)
                hits = [h for h in hits if h[1] == "match"]
            if len(hits) < spec["min_count"]:
                if hits:
                    s["recordings_below_min_count"] += 1
                continue
            if not hits:
                continue
            s["recordings_with_hits"] += 1
            nm = sum(h[1] == "match" for h in hits)
            s["hits_match"] += nm
            s["hits_possible"] += len(hits) - nm
            self.con.executemany(
                "INSERT INTO search_hit(search_run_id,source_version_id,object_uid,status,reasons,target,cls,x1,y1,x2,y2,colour) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run_id, vid, uid, st, json.dumps(rs), ti, cls, *[float(v) for v in box],
                  json.dumps(col or {"state": "not_needed"})) for uid, st, rs, ti, cls, box, col in hits])
        s["images_with_hits"] = s["recordings_with_hits"]
        s["images_searched"] = s["events_searched"]
        return s

    # ---------- results ----------
    def run_info(self, run_id: int) -> dict:
        r = self.con.execute("SELECT * FROM search_run WHERE id=?", (run_id,)).fetchone()
        if r is None:
            raise SearchError("No such search run.")
        spec = json.loads(r["spec_json"])
        return {"id": r["id"], "request_id": r["request_id"], "state": r["state"], "spec": spec,
                "description": describe(spec), "summary": json.loads(r["summary"] or "{}"), "error": r["error"],
                "attribute_algo": r["attribute_algo"], "started_by": r["started_by"], "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "text": self.con.execute("SELECT chat_text FROM search_request WHERE id=?", (r["request_id"],)).fetchone()[0]}

    def hits(self, run_id: int, status: str | None = None, decision: str | None = None,
             offset: int = 0, limit: int | None = None) -> tuple[list[Hit], int]:
        zone = self.m.zone
        rows = self.con.execute(
            "SELECT h.*, f.rel_path, v.capture_utc, "
            " (SELECT d.decision FROM search_hit_decision d WHERE d.search_hit_id=h.id ORDER BY d.id DESC LIMIT 1) AS decision "
            "FROM search_hit h JOIN source_version v ON v.id=h.source_version_id JOIN source_file f ON f.id=v.source_file_id "
            "WHERE h.search_run_id=? ORDER BY (h.status='match') DESC, v.capture_utc, h.id", (run_id,)).fetchall()
        ev_start: dict[int, tuple[float, str | None]] = {}
        if any(r["object_uid"].startswith("ev") for r in rows):
            from .video_review import VideoService
            ev_start = {e.id: (e.first_t, e.first_utc) for e in VideoService(self.m).events()}
        out = []
        for r in rows:
            dec = r["decision"] or "undecided"
            if status and r["status"] != status:
                continue
            if decision and dec != decision:
                continue
            local, cap, et = None, r["capture_utc"], None
            uid = r["object_uid"]
            if uid.startswith("ev") and uid[2:].isdigit() and int(uid[2:]) in ev_start:
                et, cap = ev_start[int(uid[2:])]
            if cap:
                utc = datetime.fromisoformat(cap.replace("Z", "+00:00")).astimezone(timezone.utc)
                local = to_local(utc, zone).local.isoformat(timespec="milliseconds" if et is not None else "seconds")
            out.append(Hit(r["id"], r["source_version_id"], r["rel_path"], local, uid, r["status"],
                           json.loads(r["reasons"]), r["target"], r["cls"], (r["x1"], r["y1"], r["x2"], r["y2"]),
                           json.loads(r["colour"]), dec, cap, et))
        total = len(out)
        return out[offset: offset + limit if limit else None], total

    def decide(self, hit_id: int, decision: str, author: str, note: str | None = None) -> None:
        if decision not in DECISIONS:
            raise SearchError(f"Decision must be one of {DECISIONS}.")
        if self.con.execute("SELECT 1 FROM search_hit WHERE id=?", (hit_id,)).fetchone() is None:
            raise SearchError("No such hit.")
        self.con.execute("INSERT INTO search_hit_decision(search_hit_id,decision,author,note,created_at) VALUES (?,?,?,?,?)",
                         (hit_id, decision, author, note, utcnow()))


HIT_COLUMNS = ["mission_id", "search_run_id", "request", "source_version_id", "file", "sha256", "capture_utc",
               "capture_local", "view", "object_id", "status", "reasons", "target", "detected_class",
               "x1", "y1", "x2", "y2", "colour_state", "colour_values", "colour_source", "operator_decision", "event_start_s"]
SPEC_COLUMNS = ["search_run_id", "request", "interpreter", "confirmed_by", "confirmed_at", "operator_edits",
                "spec_json", "plain_english", "summary_json", "colour_algorithm"]


def export_search(mission: Mission, run_id: int, author: str, formats: list[str] | None = None,
                  out_dir=None) -> list:
    """Hits and the exact spec that produced them. Never overwrites files."""
    from pathlib import Path

    from .export import _fmt, write_sheets_csv, write_sheets_xlsx

    svc = SearchService(mission)
    info = svc.run_info(run_id)
    req = mission.con.execute("SELECT * FROM search_request WHERE id=?", (info["request_id"],)).fetchone()
    cfg = mission.config()
    formats = formats or cfg.export_formats
    out = Path(out_dir or cfg.output_dir)
    hits, _ = svc.hits(run_id)
    meta = {r["id"]: r for r in mission.con.execute(
        "SELECT v.id, v.sha256, v.capture_utc, (SELECT c.name FROM view_assignment a JOIN camera_view c ON c.id=a.view_id "
        " WHERE a.source_version_id=v.id AND a.superseded_by IS NULL ORDER BY a.id DESC LIMIT 1) AS view "
        "FROM source_version v WHERE v.id IN (SELECT source_version_id FROM search_hit WHERE search_run_id=?)", (run_id,))}
    rows = []
    for h in hits:
        mrow = meta[h.source_version_id]
        rows.append([_fmt(x) for x in (
            mission.id, run_id, info["text"], h.source_version_id, h.filename, mrow["sha256"], h.capture_utc,
            h.capture_local, mrow["view"], h.object_uid, h.status, "; ".join(h.reasons), h.target + 1, h.cls,
            *[round(v, 1) for v in h.box], h.colour.get("state"), "/".join(h.colour.get("values") or []),
            h.colour.get("algo"), h.decision, "" if h.event_t is None else round(h.event_t, 3))])
    spec_rows = [[_fmt(x) for x in (run_id, info["text"], req["interpreter_version"], req["confirmed_by"],
                                     req["confirmed_at"], json.loads(req["edits_json"] or "{}"), info["spec"],
                                     " ".join(info["description"]), info["summary"], info["attribute_algo"])]]
    stem = "".join(c if c.isalnum() or c in "-_." else "_"
                   for c in f"{cfg.name}_search{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    sheets = [("hits", HIT_COLUMNS, rows), ("search", SPEC_COLUMNS, spec_rows)]
    paths = []
    out.mkdir(parents=True, exist_ok=True)
    scope = json.dumps({"search_run": run_id})
    for fmt in formats:
        if fmt == "csv":
            new = write_sheets_csv(out, stem, sheets)
        elif fmt == "xlsx":
            new = write_sheets_xlsx(out, stem, sheets)
        else:
            raise SearchError(f"Unknown export format '{fmt}'.")
        paths += new
        mission.con.execute(
            "INSERT INTO export(format,path,scope,rows,status,author,created_at) VALUES (?,?,?,?,?,?,?)",
            (fmt, ";".join(str(p) for p in new), scope, len(rows), "completed", author, utcnow()))
    return paths
