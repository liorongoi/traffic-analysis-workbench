"""Review of still-image results: results table, corrections and approval.

- Machine detections are never changed. Every human decision is an appended
  annotation_revision; the latest revision per object is current.
- Review status (unreviewed, approved, bulk approved) is kept apart from
  processing status.
- Editing an approved image reopens it, so an approval always refers to what
  the operator last saw.
- "Approve all" works on a fixed list of images taken when the operator sees
  the count. It is recorded as a bulk approval, never as a human review, and
  never covers images that arrive later.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .detect import VEHICLE_CLASSES
from .localtime import to_local
from .regions import RegionSet
from .store import Mission, utcnow

REVIEW_LABELS = {None: "unreviewed", "reopened": "unreviewed", "approved": "approved",
                 "bulk_approved": "bulk_approved_unreviewed"}
SORT_KEYS = ("capture_utc", "filename", "vehicle_count", "view", "review_status", "processing_state")


@dataclass
class Obj:
    uid: str
    source_version_id: int
    origin: str                     # machine | human | change (differs from the empty background)
    status: str                     # present | removed
    box: tuple[float, float, float, float]
    cls: str
    in_region: int | None
    detection_id: int | None = None
    machine_cls: str | None = None
    machine_box: tuple | None = None
    score: float | None = None
    correction: str = "none"        # none | reclassed | moved | reclassed_moved | removed | added
    run_item_id: int | None = None

    @property
    def counted(self) -> bool:
        return self.status == "present" and self.in_region != 0


@dataclass
class Filter:
    search: str | None = None             # substring of the filename
    view_id: int | None = None
    review_status: str | None = None      # unreviewed | approved | bulk_approved_unreviewed
    processing_state: str | None = None   # queued | running | done | failed
    local_date_from: str | None = None    # YYYY-MM-DD, mission time zone
    local_date_to: str | None = None
    min_vehicles: int | None = None

    def describe(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ResultRow:
    source_version_id: int
    filename: str
    capture_utc: str | None
    capture_local: str | None
    view_id: int | None
    view: str | None
    vehicle_count: int | None
    candidate_count: int | None          # unresolved night candidates, lower bound
    processing_state: str
    review_status: str
    run_id: int | None = None
    flags: list[str] = field(default_factory=list)
    candidate_count_max: int | None = None
    condition: str | None = None
    confirmed_from_candidates: int = 0


@dataclass
class Candidate:
    uid: str
    source_version_id: int
    origin: str                     # machine | human (operator regrouped lights)
    kind: str                       # single | pair | cluster
    member_ids: list[int]
    box: tuple[float, float, float, float]
    score: float | None
    status: str                     # unresolved | confirmed | rejected | persistent_light | superseded
    count_min: int
    count_max: int
    alternatives: list[str]
    flags: list[str]
    light_group_id: int | None = None
    cls: str | None = None          # for confirmed candidates
    band: str | None = None         # far | middle | close (night distance band at the group's row)
    recurrence: float | None = None # share of the view's images lit at this place


class ReviewService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con

    # ---------- objects ----------
    def _latest_item(self, version_id: int):
        return self.con.execute(
            "SELECT i.* FROM run_item i WHERE i.source_version_id=? ORDER BY (i.processing_state='done') DESC, i.id DESC LIMIT 1",
            (version_id,)).fetchone()

    def _rule(self, run_id: int) -> tuple[str, float]:
        pre = json.loads(self.con.execute("SELECT preprocessing FROM run WHERE id=?", (run_id,)).fetchone()[0] or "{}")
        r = pre.get("region_rule") or {}
        return r.get("rule") or "bottom_center", r.get("min_overlap") or 0.5

    def _membership(self, version_id: int, box, run_id: int | None) -> int | None:
        from .views import ViewService
        vs = ViewService(self.m)
        view_id = vs.current_view(version_id)
        if view_id is not None and not vs.regions_applicable(view_id):
            return None
        rule, ov = self._rule(run_id) if run_id else ("bottom_center", 0.5)
        regions = RegionSet.load(self.con, view_id) if view_id is not None else RegionSet([])
        return int(regions.contains_box(box, rule, ov))

    def objects(self, version_id: int, include_removed: bool = True) -> list[Obj]:
        item = self._latest_item(version_id)
        objs: dict[str, Obj] = {}
        if item is not None and item["processing_state"] == "done":
            for d in self.con.execute("SELECT * FROM detection WHERE run_item_id=? ORDER BY id", (item["id"],)):
                box = (d["x1"], d["y1"], d["x2"], d["y2"])
                objs[f"d{d['id']}"] = Obj(f"d{d['id']}", version_id, "machine", "present", box, d["cls"], d["in_region"],
                                          d["id"], d["cls"], box, d["score"], run_item_id=item["id"])
            # Vehicles found as changes against the view's empty background; those a detector box covers are not
            # counted twice.
            from .change import NOT_COUNTED, ChangeService, overlaps
            dboxes = [o.box for o in objs.values()]
            for c in ChangeService(self.m).vehicles(version_id):
                box = (c["x1"], c["y1"], c["x2"], c["y2"])
                if set(json.loads(c["flags"])) & NOT_COUNTED or any(overlaps(box, b) for b in dboxes):
                    continue
                objs[f"c{c['id']}"] = Obj(f"c{c['id']}", version_id, "change", "present", box, c["cls"], c["in_region"],
                                          None, c["cls"], box, c["score"], run_item_id=item["id"])
        revs = self.con.execute(
            "SELECT * FROM annotation_revision r WHERE source_version_id=? AND id=(SELECT MAX(id) FROM annotation_revision x "
            "WHERE x.object_uid=r.object_uid) ORDER BY id", (version_id,)).fetchall()
        run_id = self._run_of(item)
        for r in revs:
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            o = objs.get(r["object_uid"])
            if o is None:
                if r["detection_id"] is not None or r["object_uid"].startswith("c"):
                    continue  # correction to a machine object from an older run or comparison; superseded
                objs[r["object_uid"]] = Obj(r["object_uid"], version_id, "human", r["status"], box, r["cls"],
                                            self._membership(version_id, box, run_id), correction="added")
                continue
            o.status = r["status"]
            moved = box != o.machine_box
            o.box, o.cls = box, r["cls"]
            if moved:
                o.in_region = self._membership(version_id, box, run_id)
            if o.status == "removed":
                o.correction = "removed"
            else:
                o.correction = {(True, True): "reclassed_moved", (True, False): "reclassed",
                                (False, True): "moved"}.get((o.cls != o.machine_cls, moved), "none")
        out = list(objs.values())
        return out if include_removed else [o for o in out if o.status == "present"]

    @staticmethod
    def _run_of(item) -> int | None:
        return item["run_id"] if item is not None else None

    def _revise(self, version_id: int, uid: str, detection_id: int | None, action: str, status: str, box, cls: str,
                author: str, note: str | None) -> None:
        self.con.execute(
            "INSERT INTO annotation_revision(object_uid,source_version_id,detection_id,action,status,x1,y1,x2,y2,cls,author,note,"
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (uid, version_id, detection_id, action, status, *map(float, box), cls, author, note, utcnow()))
        self._reopen_if_approved(version_id, author)

    def _reopen_if_approved(self, version_id: int, author: str) -> None:
        if self.review_status(version_id) != "unreviewed":
            self._decide(version_id, "reopened", author)

    def _get(self, version_id: int, uid: str) -> Obj:
        for o in self.objects(version_id):
            if o.uid == uid:
                return o
        raise KeyError(f"object {uid} not found on this image")

    def remove(self, version_id: int, uid: str, author: str, note: str | None = None) -> None:
        o = self._get(version_id, uid)
        self._revise(version_id, uid, o.detection_id, "removed", "removed", o.box, o.cls, author, note)

    def restore(self, version_id: int, uid: str, author: str) -> None:
        o = self._get(version_id, uid)
        self._revise(version_id, uid, o.detection_id, "restored", "present", o.box, o.cls, author, None)

    def reclass(self, version_id: int, uid: str, cls: str, author: str, note: str | None = None) -> None:
        self._check_cls(cls)
        o = self._get(version_id, uid)
        self._revise(version_id, uid, o.detection_id, "reclassed", o.status, o.box, cls, author, note)

    def move(self, version_id: int, uid: str, box, author: str) -> None:
        o = self._get(version_id, uid)
        self._revise(version_id, uid, o.detection_id, "moved", o.status, self._check_box(version_id, box), o.cls, author, None)

    def add(self, version_id: int, box, cls: str, author: str, note: str | None = None) -> str:
        """Add a vehicle the detector missed, drawn as a rectangle in original pixels."""
        self._check_cls(cls)
        uid = "h" + uuid.uuid4().hex[:16]
        self._revise(version_id, uid, None, "added", "present", self._check_box(version_id, box), cls, author, note)
        return uid

    @staticmethod
    def _check_cls(cls: str) -> None:
        if cls not in VEHICLE_CLASSES:
            raise ValueError(f"Unknown class '{cls}'. Use one of: {', '.join(VEHICLE_CLASSES)}.")

    def _check_box(self, version_id: int, box) -> tuple:
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("The rectangle is too small.")
        sig = self.con.execute("SELECT width, height FROM view_signature WHERE source_version_id=?", (version_id,)).fetchone()
        if sig and (x1 < 0 or y1 < 0 or x2 > sig["width"] or y2 > sig["height"]):
            raise ValueError(f"The rectangle extends outside the {sig['width']}x{sig['height']} image.")
        return (x1, y1, x2, y2)

    # ---------- night candidates ----------
    def candidates(self, version_id: int, include_explained: bool = False) -> list[Candidate]:
        """Current light groups that are vehicle candidates, with the operator's decisions applied."""
        from .night_service import NightService, group_uid
        groups = NightService(self.m).current_groups(version_id)
        out: dict[str, Candidate] = {}
        for g in groups:
            if not g["is_candidate"] and not include_explained:
                continue
            ids = json.loads(g["member_ids"])
            uid = group_uid(ids)
            out[uid] = Candidate(uid, version_id, "machine", g["kind"], ids, (g["x1"], g["y1"], g["x2"], g["y2"]),
                                 g["score"], "unresolved" if g["is_candidate"] else "not_candidate",
                                 g["count_min"], g["count_max"], json.loads(g["alternatives"]), json.loads(g["flags"]), g["id"],
                                 band=g["band"], recurrence=g["recurrence"])
        revs = self.con.execute(
            "SELECT * FROM candidate_revision r WHERE source_version_id=? AND id=(SELECT MAX(id) FROM candidate_revision x "
            "WHERE x.group_uid=r.group_uid) ORDER BY id", (version_id,)).fetchall()
        for r in revs:
            c = out.get(r["group_uid"])
            box = (r["x1"], r["y1"], r["x2"], r["y2"]) if r["x1"] is not None else None
            if c is None:
                if r["member_ids"] is not None:  # the group is no longer formed; keep the decision visible
                    ids = json.loads(r["member_ids"])
                    c = Candidate(r["group_uid"], version_id, "human" if r["group_uid"].startswith("H") else "machine",
                                  "single" if len(ids) == 1 else ("pair" if len(ids) == 2 else "cluster"), ids, box,
                                  None, "unresolved", 1, 1, [], [] if r["group_uid"].startswith("H") else ["group_no_longer_formed"])
                    out[r["group_uid"]] = c
                else:
                    continue
            status = {"confirmed": "confirmed", "rejected": "rejected", "persistent_light": "persistent_light",
                      "superseded": "superseded", "regrouped": "unresolved", "restored": "unresolved"}[r["action"]]
            if status == "unresolved" and "group_no_longer_formed" in c.flags:
                status = "superseded"
            c.status = status
            if box is not None:
                c.box = box
            c.cls = r["cls"]
        # Lights that a vehicle found as a change already counts are not open candidates any more.
        changes = [o.box for o in self.objects(version_id, include_removed=False) if o.origin == "change"]
        if changes:
            from .change import overlaps
            for c in out.values():
                if c.status == "unresolved" and c.origin == "machine" and any(overlaps(c.box, b) for b in changes):
                    c.status = "not_candidate"
        return [c for c in out.values() if c.status != "not_candidate"]

    def _cand_rev(self, version_id: int, c: Candidate, action: str, author: str, box=None, cls=None, note=None) -> None:
        b = box or c.box
        self.con.execute(
            "INSERT INTO candidate_revision(group_uid,source_version_id,light_group_id,action,member_ids,x1,y1,x2,y2,cls,author,"
            "note,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (c.uid, version_id, c.light_group_id, action, json.dumps(c.member_ids), *(float(v) for v in b), cls, author,
             note, utcnow()))
        self._reopen_if_approved(version_id, author)

    def _cand(self, version_id: int, uid: str) -> Candidate:
        for c in self.candidates(version_id):
            if c.uid == uid:
                return c
        raise KeyError(f"candidate {uid} not found on this image")

    def confirm_candidate(self, version_id: int, uid: str, author: str, box=None, cls: str = "car",
                          note: str | None = None) -> None:
        """The operator says these lights are one vehicle. Optionally with the vehicle's outline."""
        self._check_cls(cls)
        c = self._cand(version_id, uid)
        self._cand_rev(version_id, c, "confirmed", author, self._check_box(version_id, box) if box else None, cls, note)

    def reject_candidate(self, version_id: int, uid: str, author: str, note: str | None = None) -> None:
        self._cand_rev(version_id, self._cand(version_id, uid), "rejected", author, note=note)

    def restore_candidate(self, version_id: int, uid: str, author: str) -> None:
        self._cand_rev(version_id, self._cand(version_id, uid), "restored", author)

    def mark_persistent_light(self, version_id: int, uid: str, author: str, margin: float = 4.0) -> int:
        """The lights are fixed background (street lamp, sign). Adds an authoritative label for the view."""
        from .night_service import NightService
        from .views import ViewService
        c = self._cand(version_id, uid)
        view_id = ViewService(self.m).current_view(version_id)
        if view_id is None:
            raise ValueError("Assign this image to a camera view before marking background lights.")
        cond = self.con.execute("SELECT condition FROM image_condition WHERE source_version_id=?", (version_id,)).fetchone()[0]
        x1, y1, x2, y2 = c.box
        poly = [[x1 - margin, y1 - margin], [x2 + margin, y1 - margin], [x2 + margin, y2 + margin], [x1 - margin, y2 + margin]]
        lid = self.con.execute("INSERT INTO light_label(view_id,condition,kind,polygon,author,created_at) VALUES (?,?,?,?,?,?)",
                               (view_id, cond, "persistent", json.dumps(poly), author, utcnow())).lastrowid
        self._cand_rev(version_id, c, "persistent_light", author, note=f"light_label {lid}")
        NightService(self.m).refresh()  # the label changes this view's background; regroup its images
        return lid

    def regroup(self, version_id: int, light_ids: list[int], author: str, note: str | None = None) -> str:
        """Form one candidate from the chosen lights; groups that shared any of them are superseded."""
        from .night_service import NightService
        lights = {li.id: li for li in NightService(self.m).lights(version_id)}
        if not light_ids or any(i not in lights for i in light_ids):
            raise ValueError("Choose lights that belong to this image.")
        chosen = set(light_ids)
        for c in self.candidates(version_id):
            if c.status in ("unresolved", "confirmed", "rejected") and chosen & set(c.member_ids):
                self._cand_rev(version_id, c, "superseded", author, note="regrouped by operator")
        box = (min(lights[i].box[0] for i in chosen), min(lights[i].box[1] for i in chosen),
               max(lights[i].box[2] for i in chosen), max(lights[i].box[3] for i in chosen))
        uid = "H" + uuid.uuid4().hex[:12]
        c = Candidate(uid, version_id, "human", "single", sorted(chosen), box, None, "unresolved", 1, 1, [], [])
        self._cand_rev(version_id, c, "regrouped", author, note=note)
        return uid

    # ---------- review status ----------
    def review_status(self, version_id: int) -> str:
        r = self.con.execute("SELECT decision FROM review_decision WHERE source_version_id=? ORDER BY id DESC LIMIT 1",
                             (version_id,)).fetchone()
        return REVIEW_LABELS[r[0] if r else None]

    def _decide(self, version_id: int, decision: str, author: str, bulk_id: int | None = None) -> None:
        self.con.execute("INSERT INTO review_decision(source_version_id,decision,bulk_approval_id,author,created_at) "
                         "VALUES (?,?,?,?,?)", (version_id, decision, bulk_id, author, utcnow()))

    def approve(self, version_id: int, author: str) -> None:
        item = self._latest_item(version_id)
        if item is None or item["processing_state"] != "done":
            raise ValueError("Only processed images can be approved.")
        self._decide(version_id, "approved", author)

    def reopen(self, version_id: int, author: str) -> None:
        self._decide(version_id, "reopened", author)

    def approve_all_preview(self, flt: Filter | None = None) -> dict:
        """What 'Approve all' would cover: processed, not yet approved images in the current filter."""
        rows = [r for r in self.results(flt) if r.processing_state == "done" and r.review_status == "unreviewed"]
        return {"scope": (flt or Filter()).describe(), "count": len(rows), "version_ids": [r.source_version_id for r in rows],
                "excluded_not_processed": sum(1 for r in self.results(flt) if r.processing_state != "done")}

    def approve_all_commit(self, preview: dict, author: str) -> int:
        """Record a bulk approval for exactly the images in the preview the operator saw."""
        ids = preview["version_ids"]
        self.con.execute("BEGIN")
        bid = self.con.execute("INSERT INTO bulk_approval(scope,item_count,author,created_at) VALUES (?,?,?,?)",
                               (json.dumps(preview["scope"]), len(ids), author, utcnow())).lastrowid
        n = 0
        for v in ids:
            if self.review_status(v) == "unreviewed":
                self._decide(v, "bulk_approved", author, bid)
                n += 1
        self.con.execute("UPDATE bulk_approval SET item_count=? WHERE id=?", (n, bid))
        self.con.execute("COMMIT")
        return bid

    # ---------- results table ----------
    def results(self, flt: Filter | None = None, sort: str = "capture_utc", descending: bool = False,
                offset: int = 0, limit: int | None = None) -> list[ResultRow]:
        flt = flt or Filter()
        if sort not in SORT_KEYS:
            raise ValueError(f"sort must be one of {SORT_KEYS}")
        zone = self.m.zone
        rows = self.con.execute(
            "SELECT v.id, f.rel_path, v.capture_utc, "
            " (SELECT a.view_id FROM view_assignment a WHERE a.source_version_id=v.id AND a.superseded_by IS NULL "
            "  ORDER BY a.id DESC LIMIT 1) AS view_id "
            "FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id)").fetchall()
        names = {r["id"]: r["name"] for r in self.con.execute("SELECT id, name FROM camera_view")}
        states = {r["id"]: r["state"] for r in self.con.execute("SELECT id, state FROM camera_view")}
        out = []
        for r in rows:
            vid = r["id"]
            if flt.search and flt.search.lower() not in r["rel_path"].lower():
                continue
            if flt.view_id is not None and r["view_id"] != flt.view_id:
                continue
            item = self._latest_item(vid)
            pstate = item["processing_state"] if item else "not_queued"
            if flt.processing_state and pstate != flt.processing_state:
                continue
            rstatus = self.review_status(vid)
            if flt.review_status and rstatus != flt.review_status:
                continue
            local = None
            if r["capture_utc"]:
                utc = datetime.fromisoformat(r["capture_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
                local = to_local(utc, zone).local.isoformat()
                day = local[:10]
                if (flt.local_date_from and day < flt.local_date_from) or (flt.local_date_to and day > flt.local_date_to):
                    continue
            count = None
            cmin = cmax = None
            confirmed = 0
            flags = []
            cond = self.con.execute("SELECT condition, flags FROM image_condition WHERE source_version_id=?", (vid,)).fetchone()
            if pstate == "done":
                objs = self.objects(vid, include_removed=False)
                cands = self.candidates(vid)
                confirmed = sum(c.status == "confirmed" for c in cands)
                count = sum(o.counted for o in objs) + confirmed
                open_c = [c for c in cands if c.status == "unresolved"]
                cmin, cmax = sum(c.count_min for c in open_c), sum(c.count_max for c in open_c)
                if any(o.in_region is None for o in objs):
                    flags.append("regions_not_applied")
                if cond and cond["condition"] != "day" and not cands and not self.con.execute(
                        "SELECT 1 FROM light_group WHERE source_version_id=? LIMIT 1", (vid,)).fetchone() \
                        and self.con.execute("SELECT 1 FROM light_component WHERE source_version_id=? LIMIT 1",
                                             (vid,)).fetchone():
                    flags.append("night_analysis_pending")
            if cond:
                flags += json.loads(cond["flags"])
            if flt.min_vehicles is not None and (count or 0) < flt.min_vehicles:
                continue
            if r["view_id"] is None:
                flags.append("no_view")
            elif states.get(r["view_id"]) == "needs_review":
                flags.append("view_needs_review")
            out.append(ResultRow(vid, r["rel_path"], r["capture_utc"], local, r["view_id"], names.get(r["view_id"]),
                                 count, cmin, pstate, rstatus, item["run_id"] if item else None, flags, cmax,
                                 cond["condition"] if cond else None, confirmed))
        key = {"capture_utc": lambda x: (x.capture_utc or "", x.filename), "filename": lambda x: x.filename,
               "vehicle_count": lambda x: (x.vehicle_count if x.vehicle_count is not None else -1, x.capture_utc or ""),
               "view": lambda x: (x.view or "", x.capture_utc or ""), "review_status": lambda x: (x.review_status, x.filename),
               "processing_state": lambda x: (x.processing_state, x.filename)}[sort]
        out.sort(key=key, reverse=descending)
        return out[offset: offset + limit if limit else None]

    def position(self, version_id: int, flt: Filter | None = None, sort: str = "capture_utc", descending: bool = False) -> dict:
        """Index of total plus previous and next image for the investigation screen."""
        ids = [r.source_version_id for r in self.results(flt, sort, descending)]
        if version_id not in ids:
            return {"index": None, "total": len(ids), "prev": None, "next": None}
        i = ids.index(version_id)
        return {"index": i + 1, "total": len(ids), "prev": ids[i - 1] if i else None,
                "next": ids[i + 1] if i + 1 < len(ids) else None}
