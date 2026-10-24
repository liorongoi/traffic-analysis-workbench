"""Night analysis stored per mission: conditions, light components, backgrounds, groups.

Flow inside a processing run:
  per image (same transaction as detection): measure condition; at dusk or night store light components.
  after the images: per view and condition, build or reuse the persistent-light background,
  then group each image's lights against it and decide which groups are vehicle candidates.

Backgrounds are rebuilt from stored light components (no image is re-read), only when
the image set grew by REBUILD_GROWTH or the operator's light labels changed. Group ids
are derived from their member lights, so operator decisions survive a rebuild that
groups the same lights the same way.
"""
from __future__ import annotations

import json
import sqlite3

import cv2
import numpy as np

from .night import (COND_ALGO, Background, BackgroundParams, Condition, GroupingParams, Light, LightParams, WidthMap,
                    build_background, covered_by, dumps, group_lights, light_components, png, unpng)
from .regions import RegionSet
from .store import Mission, utcnow

NIGHT_CONDITIONS = ("night", "dusk")
REBUILD_GROWTH = 0.10
MIN_CANDIDATE_SCORE = 0.2


def group_uid(member_ids: list[int]) -> str:
    return "L" + "-".join(str(i) for i in sorted(member_ids))


class NightService:
    def __init__(self, mission: Mission, light_params: LightParams | None = None, bg_params: BackgroundParams | None = None,
                 grouping: GroupingParams | None = None):
        self.m = mission
        self.con: sqlite3.Connection = mission.con
        self.lp = light_params or LightParams()
        self.bp = bg_params or BackgroundParams()
        self.gp = grouping or GroupingParams()

    def describe(self) -> dict:
        return {"condition": COND_ALGO, "lights": self.lp.record(), "background": self.bp.record(),
                "grouping": self.gp.record(), "candidate_min_score": MIN_CANDIDATE_SCORE,
                "enhancement": "none (original pixels only)"}

    # ---------- per image ----------
    def record_item(self, run_item_id: int, version_id: int, img: np.ndarray, cond: Condition) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO image_condition VALUES (?,?,?,?,?,?,?,?,?,?)",
            (version_id, cond.label, int(cond.monochrome), cond.mean_luma, cond.saturation, cond.clipped_pct,
             cond.dark_pct, cond.detail, json.dumps(list(cond.flags)), COND_ALGO))
        if cond.label not in NIGHT_CONDITIONS:
            return
        for li in light_components(img, self.lp):
            self.con.execute(
                "INSERT INTO light_component(run_item_id,source_version_id,x1,y1,x2,y2,cx,cy,area,peak,mean,elongation,red_share)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_item_id, version_id, *li.box, li.cx, li.cy, li.area, li.peak, li.mean, li.elongation, li.red_share))

    def condition(self, version_id: int) -> str | None:
        r = self.con.execute("SELECT condition FROM image_condition WHERE source_version_id=?", (version_id,)).fetchone()
        return r[0] if r else None

    def lights(self, version_id: int) -> list[Light]:
        item = self.con.execute("SELECT id FROM run_item WHERE source_version_id=? AND processing_state='done' "
                                "ORDER BY id DESC LIMIT 1", (version_id,)).fetchone()
        if item is None:
            return []
        return [Light((r["x1"], r["y1"], r["x2"], r["y2"]), r["cx"], r["cy"], r["area"], r["peak"], r["mean"],
                      r["elongation"], r["red_share"], r["id"])
                for r in self.con.execute("SELECT * FROM light_component WHERE run_item_id=? ORDER BY id", (item[0],))]

    # ---------- perspective ----------
    def width_map(self, view_id: int | None) -> WidthMap | None:
        if view_id is None:
            return None
        boxes = self.con.execute(
            "SELECT d.x1,d.y1,d.x2,d.y2 FROM detection d JOIN image_condition c ON c.source_version_id=d.source_version_id "
            "WHERE d.view_id=? AND d.cls='car' AND c.condition='day' AND d.score>=0.5", (view_id,)).fetchall()
        return WidthMap.fit([tuple(b) for b in boxes])

    # ---------- background ----------
    def _versions(self, view_id: int, condition: str) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT v.id FROM source_version v JOIN image_condition c ON c.source_version_id=v.id "
            "JOIN view_assignment a ON a.source_version_id=v.id AND a.superseded_by IS NULL "
            "WHERE a.view_id=? AND c.condition=? AND v.superseded_by IS NULL AND EXISTS "
            "(SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND i.processing_state='done') "
            "ORDER BY v.capture_utc, v.id", (view_id, condition))]

    def _labels(self, view_id: int, condition: str) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM light_label WHERE view_id=? AND condition=? AND deleted=0 ORDER BY id",
                                (view_id, condition)).fetchall()

    def _vehicle_boxes(self, version_id: int) -> list[tuple]:
        """Areas holding a vehicle in this image: machine detections, human boxes, confirmed candidates."""
        from .review import ReviewService
        rs = ReviewService(self.m)
        boxes = [o.box for o in rs.objects(version_id, include_removed=False)]
        boxes += [c.box for c in rs.candidates(version_id) if c.status == "confirmed"]
        return boxes

    def current_background(self, view_id: int, condition: str) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM background_model WHERE view_id=? AND condition=? ORDER BY id DESC LIMIT 1",
                                (view_id, condition)).fetchone()

    def ensure_background(self, view_id: int, condition: str, force: bool = False) -> sqlite3.Row | None:
        versions = self._versions(view_id, condition)
        if not versions:
            return None
        labels = self._labels(view_id, condition)
        label_key = [r["id"] for r in labels]
        cur = self.current_background(view_id, condition)
        if cur is not None and not force:
            params = json.loads(cur["params"])
            prev = json.loads(cur["input_ids"])
            grew = len(versions) - len(prev)
            if params.get("label_ids") == label_key and (
                    grew <= 0 or (cur["state"] == "ready" and grew < REBUILD_GROWTH * len(prev))):
                return cur
        size = self.con.execute("SELECT ref_width, ref_height FROM camera_view WHERE id=?", (view_id,)).fetchone()
        w, h = size["ref_width"], size["ref_height"]
        masks, vboxes = [], []
        for v in versions:
            m = np.zeros((h, w), np.uint8)
            for li in self.lights(v):
                x1, y1, x2, y2 = (int(c) for c in li.box)
                cv2.ellipse(m, (((x1 + x2) / 2, (y1 + y2) / 2), (max(1, x2 - x1), max(1, y2 - y1)), 0), 1, -1)
            masks.append(m)
            vboxes.append(self._vehicle_boxes(v))
        bg = build_background(masks, vboxes, [(r["kind"], json.loads(r["polygon"])) for r in labels], self.bp)
        params = dict(self.bp.record(), label_ids=label_key, light_mask="filled ellipses from stored light components")
        if bg is None:
            rid = self.con.execute(
                "INSERT INTO background_model(view_id,condition,image_count,state,params,input_ids,built_at) "
                "VALUES (?,?,?,?,?,?,?)", (view_id, condition, len(versions), "insufficient", dumps(params),
                                           dumps(versions), utcnow())).lastrowid
        else:
            rid = self.con.execute(
                "INSERT INTO background_model(view_id,condition,image_count,state,persistent_mask,frequency,new_lights,"
                "params,input_ids,built_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (view_id, condition, bg.image_count, "ready", png(bg.persistent * 255),
                 png((bg.frequency * 255).astype(np.uint8)), dumps(bg.new_lights), dumps(params), dumps(versions),
                 utcnow())).lastrowid
            prev_new = json.loads(cur["new_lights"]) if cur is not None and cur["new_lights"] else []
            if bg.new_lights and len(bg.new_lights) != len(prev_new):
                self.m.notify("info", "new_persistent_light",
                              f"{len(bg.new_lights)} light(s) became frequent only recently in view {view_id} ({condition}). "
                              "Mark them as persistent background or leave them as possible vehicles.",
                              view_id=view_id, condition=condition, boxes=bg.new_lights)
        return self.con.execute("SELECT * FROM background_model WHERE id=?", (rid,)).fetchone()

    @staticmethod
    def _background(row) -> Background | None:
        if row is None or row["state"] != "ready":
            return None
        return Background((unpng(row["persistent_mask"]) > 0).astype(np.uint8), unpng(row["frequency"]) / 255.0,
                          row["image_count"], [tuple(b) for b in json.loads(row["new_lights"])])

    # ---------- groups ----------
    def analyse(self, version_id: int, bg_row) -> int:
        from .review import ReviewService
        from .views import ViewService
        vs = ViewService(self.m)
        view_id = vs.current_view(version_id)
        lights = self.lights(version_id)
        bg = self._background(bg_row)
        wm = self.width_map(view_id)
        explained = [bool(bg and bg.explains(li, self.bp.explained_share)) for li in lights]
        live = [li for li, e in zip(lights, explained) if not e]
        regions = None
        if view_id is not None and vs.regions_applicable(view_id):
            regions = RegionSet.load(self.con, view_id)
        elif view_id is None:
            regions = RegionSet([])
        dets = [(o.detection_id, o.box) for o in ReviewService(self.m).objects(version_id, include_removed=False)
                if o.detection_id is not None]
        now = utcnow()
        bg_id = bg_row["id"] if bg_row is not None else None
        self._region_rev = self._region_revision(view_id)
        rows = 0
        for li, e in zip(lights, explained):
            if e:
                self._insert(version_id, bg_id, "single", [li.id], li.box, 0.0, 0.0, 1, None, None, 0, 0, 0, [], [], now)
                rows += 1
        for g in group_lights(live, wm, self.gp):
            ids = [live[i].id for i in g.members]
            cx, cy = (g.box[0] + g.box[2]) / 2, (g.box[1] + g.box[3]) / 2
            inside = int(regions.contains_point(cx, cy)) if regions is not None else None
            cov = covered_by(g.box, dets)
            flags = list(g.flags) + (["no_view"] if view_id is None else []) + \
                (["regions_not_applied"] if inside is None else []) + (["no_background"] if bg is None else [])
            cand = int(cov is None and inside != 0 and g.score >= MIN_CANDIDATE_SCORE)
            self._insert(version_id, bg_id, g.kind, ids, g.box, g.score, g.reflection, 0, inside, cov, cand,
                         g.count_min if cand else 0, g.count_max if cand else 0, g.alternatives, flags, now)
            rows += 1
        return rows

    def _insert(self, version_id, bg_id, kind, ids, box, score, refl, explained, inside, cov, cand, cmin, cmax, alts,
                flags, now):
        self.con.execute(
            "INSERT INTO light_group(source_version_id,background_model_id,region_revision_id,kind,member_ids,x1,y1,x2,y2,score,reflection,"
            "explained_by_background,in_region,covered_by_detection,is_candidate,count_min,count_max,alternatives,flags,"
            "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version_id, bg_id, self._region_rev, kind, dumps(sorted(ids)), *(float(v) for v in box), float(score), float(refl), explained,
             inside, cov, cand, cmin, cmax, dumps(alts), dumps(flags), now))

    def _region_revision(self, view_id: int | None) -> int | None:
        if view_id is None:
            return None
        return self.con.execute("SELECT MAX(id) FROM region_revision WHERE view_id=?", (view_id,)).fetchone()[0]

    def current_groups(self, version_id: int) -> list[sqlite3.Row]:
        """Groups from the most recent analysis of this image."""
        last = self.con.execute("SELECT MAX(created_at) FROM light_group WHERE source_version_id=?", (version_id,)).fetchone()[0]
        if last is None:
            return []
        latest = self.con.execute(
            "SELECT background_model_id, region_revision_id FROM light_group WHERE source_version_id=? ORDER BY id DESC LIMIT 1",
            (version_id,)).fetchone()
        return self.con.execute(
            "SELECT * FROM light_group WHERE source_version_id=? AND background_model_id IS ? AND region_revision_id IS ? "
            "AND created_at=? ORDER BY id", (version_id, latest[0], latest[1], last)).fetchall()

    def refresh(self) -> dict:
        """Bring backgrounds and groups up to date for all processed dusk and night images."""
        from .views import ViewService
        vs = ViewService(self.m)
        stats = {"backgrounds_built": 0, "images_analysed": 0}
        pairs = self.con.execute(
            "SELECT DISTINCT a.view_id, c.condition FROM image_condition c JOIN view_assignment a "
            "ON a.source_version_id=c.source_version_id AND a.superseded_by IS NULL "
            "WHERE c.condition IN ('night','dusk') AND a.view_id IS NOT NULL").fetchall()
        for view_id, cond in pairs:
            before = self.current_background(view_id, cond)
            bg = self.ensure_background(view_id, cond)
            if bg is None:
                continue
            if before is None or bg["id"] != before["id"]:
                stats["backgrounds_built"] += 1
            region_rev = self._region_revision(view_id)
            for v in self._versions(view_id, cond):
                done = self.con.execute("SELECT 1 FROM light_group WHERE source_version_id=? AND background_model_id=? "
                                        "AND region_revision_id IS ? LIMIT 1", (v, bg["id"], region_rev)).fetchone()
                if done or not self.lights(v):
                    continue
                self.con.execute("BEGIN")
                self.analyse(v, bg)
                self.con.execute("COMMIT")
                stats["images_analysed"] += 1
        # Dusk or night images without a view: analysed without background or regions, and flagged.
        for (v,) in self.con.execute(
                "SELECT c.source_version_id FROM image_condition c WHERE c.condition IN ('night','dusk') AND NOT EXISTS "
                "(SELECT 1 FROM light_group g WHERE g.source_version_id=c.source_version_id) AND EXISTS (SELECT 1 FROM "
                "run_item i WHERE i.source_version_id=c.source_version_id AND i.processing_state='done')").fetchall():
            if vs.current_view(v) is None and self.lights(v):
                self.con.execute("BEGIN")
                self.analyse(v, None)
                self.con.execute("COMMIT")
                stats["images_analysed"] += 1
        return stats
