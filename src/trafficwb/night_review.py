"""Night review across images, and accuracy of night candidates from operator answers (NGT-06).

- The queue lists night candidates from every processed dusk or night image, most uncertain first
  (score closest to 0.5), so a few minutes of answers teach the most. Answers are the normal candidate
  decisions (confirm, reject, fixed light); nothing new is stored here.
- Accuracy is measured only on candidates an operator answered: the share that were vehicles,
  overall and by distance band, kind and flag. Unanswered candidates are counted, never guessed.
  This is the "candidate acceptance rate" of the night evaluation plan; it says nothing about vehicles
  that produced no candidate (missed vehicles need operator-drawn boxes and are counted separately).
"""
from __future__ import annotations

from datetime import datetime, timezone

from .localtime import to_local
from .review import Candidate, ReviewService
from .store import Mission
from .views import ViewService

DECIDED = ("confirmed", "rejected", "persistent_light")


class NightReview:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.rs = ReviewService(mission)
        self.vs = ViewService(mission)

    def _images(self, view_id: int | None = None) -> list:
        rows = self.con.execute(
            "SELECT v.id, f.rel_path, v.capture_utc, c.condition FROM source_version v "
            "JOIN source_file f ON f.id=v.source_file_id JOIN image_condition c ON c.source_version_id=v.id "
            "WHERE c.condition IN ('night','dusk') AND v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i "
            "WHERE i.source_version_id=v.id AND i.processing_state='done') ORDER BY v.capture_utc, v.id").fetchall()
        if view_id is not None:
            rows = [r for r in rows if self.vs.current_view(r["id"]) == view_id]
        return rows

    def _local(self, utc: str | None) -> str | None:
        if not utc:
            return None
        t = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        return to_local(t, self.m.zone).local.replace(tzinfo=None).isoformat()

    def _all(self, view_id: int | None = None) -> list[tuple[dict, Candidate]]:
        out = []
        for r in self._images(view_id):
            img = {"version_id": r["id"], "file": r["rel_path"], "capture_local": self._local(r["capture_utc"]),
                   "condition": r["condition"], "view_id": self.vs.current_view(r["id"])}
            for c in self.rs.candidates(r["id"]):
                if c.origin == "machine" and c.status in ("unresolved", *DECIDED):
                    out.append((img, c))
        return out

    def queue(self, view_id: int | None = None, band: str | None = None, flag: str | None = None,
              status: str = "open", offset: int = 0, limit: int = 60) -> dict:
        """status: open (not answered yet) | answered | all."""
        if status not in ("open", "answered", "all"):
            raise ValueError("status must be open, answered or all")
        items = []
        for img, c in self._all(view_id):
            if status == "open" and c.status != "unresolved":
                continue
            if status == "answered" and c.status not in DECIDED:
                continue
            if band and (c.band or "none") != band:
                continue
            if flag and flag not in c.flags:
                continue
            items.append((img, c))
        if status == "open":
            # Most uncertain first. Among equal uncertainty, take one per image in turn (then a second per image, ...)
            # so one busy image or one kind of light does not fill the first page.
            unc = lambda x: round(abs((x[1].score or 0.0) - 0.5), 2)
            items.sort(key=lambda x: (unc(x), x[1].uid))
            turn, seen = {}, {}
            for x in items:
                k = (unc(x), x[0]["version_id"])
                turn[id(x)] = seen[k] = seen.get(k, -1) + 1
            items.sort(key=lambda x: (unc(x), turn[id(x)], x[0]["capture_local"] or "", x[1].uid))
        page = items[offset: offset + limit]
        return {"total": len(items), "items": [img | {
            "uid": c.uid, "kind": c.kind, "box": list(c.box), "score": c.score, "status": c.status, "band": c.band,
            "recurrence": c.recurrence, "flags": c.flags, "alternatives": c.alternatives,
            "count_min": c.count_min, "count_max": c.count_max, "cls": c.cls} for img, c in page]}

    def found(self, view_id: int | None = None) -> dict:
        """How many vehicles the night analysis found, on night images you approved one by one.

        Approving an image says every vehicle in it is marked, so a vehicle you drew there is one the
        detector and the light candidates both missed. Found = by the detector + confirmed light candidates."""
        from .night import band_at
        from .night_service import NightService
        ns = NightService(self.m)
        wms: dict = {}
        tot = {"detector": 0, "lights": 0, "drawn": 0}
        by_band: dict[str, dict] = {}
        images = 0
        for r in self._images(view_id):
            v = r["id"]
            if self.rs.review_status(v) != "approved":
                continue
            images += 1
            view = self.vs.current_view(v)
            if view not in wms:
                wms[view] = (ns.width_map(view), ns.band_setting(view)[1])
            wm, bp = wms[view]
            for o in self.rs.objects(v, include_removed=False):
                if not o.counted:
                    continue
                k = "detector" if o.origin == "machine" else "drawn"
                tot[k] += 1
                by_band.setdefault(band_at(wm, o.box[3], bp) or "none", {"detector": 0, "lights": 0, "drawn": 0})[k] += 1
            for c in self.rs.candidates(v):
                if c.status == "confirmed":
                    tot["lights"] += 1
                    by_band.setdefault(c.band or "none", {"detector": 0, "lights": 0, "drawn": 0})["lights"] += 1

        def share(d):
            n = d["detector"] + d["lights"] + d["drawn"]
            return d | {"vehicles": n, "found_share": round((d["detector"] + d["lights"]) / n, 3) if n else None}

        return {"images": images, **share(tot), "by_band": {k: share(v) for k, v in sorted(by_band.items())},
                "note": "Counted only on night images you approved one by one. A vehicle you drew there counts as missed."}

    def accuracy(self, view_id: int | None = None) -> dict:
        """Share of answered candidates that were vehicles, overall and by band, kind and flag."""
        def blank():
            return {"vehicle": 0, "not_vehicle": 0, "open": 0}

        overall, by_band, by_kind, by_flag, by_score = blank(), {}, {}, {}, {}
        for _, c in self._all(view_id):
            key = "open" if c.status == "unresolved" else "vehicle" if c.status == "confirmed" else "not_vehicle"
            s = c.score or 0.0
            score_bin = "under 0.3" if s < 0.3 else "0.3–0.5" if s < 0.5 else "0.5–0.7" if s < 0.7 else "0.7 and over"
            for table, k in ((by_band, c.band or "none"), (by_kind, c.kind), (by_score, score_bin)):
                table.setdefault(k, blank())[key] += 1
            for f in c.flags or ["no_flags"]:
                by_flag.setdefault(f, blank())[key] += 1
            overall[key] += 1

        def rate(d):
            n = d["vehicle"] + d["not_vehicle"]
            return d | {"answered": n, "vehicle_share": round(d["vehicle"] / n, 3) if n else None}

        views = [view_id] if view_id is not None else [v["id"] for v in self.vs.views()]
        models = []
        for vid in views:
            r = self.con.execute("SELECT * FROM night_light_model WHERE view_id=? ORDER BY id DESC LIMIT 1", (vid,)).fetchone()
            if r is not None:
                models.append({k: r[k] for k in ("view_id", "created_at", "n_pos", "n_neg", "auc_model", "auc_rules",
                                                  "applied", "reason")})
        return {"learned": models, "found": self.found(view_id),
                "overall": rate(overall),
                "by_band": {k: rate(v) for k, v in sorted(by_band.items())},
                "by_kind": {k: rate(v) for k, v in sorted(by_kind.items())},
                "by_score": {k: rate(v) for k, v in by_score.items()},
                "by_flag": {k: rate(v) for k, v in sorted(by_flag.items(), key=lambda x: -sum(x[1].values()))},
                "note": "Measured only on candidates you answered. Vehicles that produced no candidate are not "
                        "counted here; draw them on the image to count them."}
