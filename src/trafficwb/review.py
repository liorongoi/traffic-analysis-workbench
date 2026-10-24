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
    origin: str                     # machine | human
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
    candidate_count: int | None
    processing_state: str
    review_status: str
    run_id: int | None = None
    flags: list[str] = field(default_factory=list)


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
        revs = self.con.execute(
            "SELECT * FROM annotation_revision r WHERE source_version_id=? AND id=(SELECT MAX(id) FROM annotation_revision x "
            "WHERE x.object_uid=r.object_uid) ORDER BY id", (version_id,)).fetchall()
        run_id = self._run_of(item)
        for r in revs:
            box = (r["x1"], r["y1"], r["x2"], r["y2"])
            o = objs.get(r["object_uid"])
            if o is None:
                if r["detection_id"] is not None:
                    continue  # correction to a detection from an older run; superseded by reprocessing
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
            flags = []
            if pstate == "done":
                objs = self.objects(vid, include_removed=False)
                count = sum(o.counted for o in objs)
                if any(o.in_region is None for o in objs):
                    flags.append("regions_not_applied")
            if flt.min_vehicles is not None and (count or 0) < flt.min_vehicles:
                continue
            if r["view_id"] is None:
                flags.append("no_view")
            elif states.get(r["view_id"]) == "needs_review":
                flags.append("view_needs_review")
            out.append(ResultRow(vid, r["rel_path"], r["capture_utc"], local, r["view_id"], names.get(r["view_id"]),
                                 count, None, pstate, rstatus,  # night candidates arrive in Phase 4
                                 item["run_id"] if item else None, flags))
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
