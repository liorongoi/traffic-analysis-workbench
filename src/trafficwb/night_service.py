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

from .night import (COND_ALGO, GROUP_ALGO, is_soft_glow, Background, BackgroundParams, BandParams, Condition, GroupingParams, Light, LightParams,
                    WidthMap, band_rows, build_background, covered_by, dumps, group_lights, light_components,
                    map_distance, off_vehicle_map, on_vehicle_map, png, unpng, vehicle_map)
from .night_learn import NightLearner, measure, vector
from .regions import RegionSet
from .store import Mission, transaction, utcnow

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
        self.learner = NightLearner(self.con)

    def describe(self) -> dict:
        return {"condition": COND_ALGO, "groups": GROUP_ALGO, "lights": self.lp.record(), "background": self.bp.record(),
                "grouping": self.gp.record(), "bands": BandParams().record() | {"per_view": "view_night_setting"}, "candidate_min_score": MIN_CANDIDATE_SCORE,
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

    def nearest_view(self, version_id: int, min_similarity: float = 0.6) -> int | None:
        """For an image left out of every view (rain, fog on the lens): the view of the same size whose
        signature it resembles most, if the resemblance is at least `min_similarity` (the colour-to-infrared
        link level). Used only to borrow that view's background, width map and bands; it does not assign."""
        from .views import ViewService
        vs = ViewService(self.m)
        rows = vs._sig_rows([version_id])
        if not rows:
            return None
        r = rows[0]
        x = vs._vec(r)
        sizes = {v["id"]: (v["ref_width"], v["ref_height"]) for v in self.con.execute(
            "SELECT * FROM camera_view WHERE merged_into IS NULL")}
        best = max(((float(c @ x), vid) for (vid, _), c in vs._centroids().items()
                    if sizes.get(vid) == (r["width"], r["height"])), default=(0.0, None))
        return best[1] if best[0] >= min_similarity else None

    # ---------- what operators decided ----------
    def decision_rev(self, view_id: int | None) -> int | None:
        if view_id is None:
            return None
        return self.con.execute(
            "SELECT MAX(r.id) FROM candidate_revision r JOIN view_assignment a ON a.source_version_id=r.source_version_id "
            "AND a.superseded_by IS NULL WHERE a.view_id=?", (view_id,)).fetchone()[0]

    def decision_places(self, view_id: int | None) -> list[tuple[int, str, tuple]]:
        """Current operator decisions on night candidates in the view: (image, confirmed | rejected, box)."""
        if view_id is None:
            return []
        cache = self.__dict__.setdefault("_decisions", {})
        if view_id not in cache:
            cache[view_id] = [(r["source_version_id"], r["action"], (r["x1"], r["y1"], r["x2"], r["y2"])) for r in self.con.execute(
                "SELECT r.* FROM candidate_revision r JOIN view_assignment a ON a.source_version_id=r.source_version_id "
                "AND a.superseded_by IS NULL WHERE a.view_id=? AND r.x1 IS NOT NULL AND r.id=(SELECT MAX(id) FROM "
                "candidate_revision x WHERE x.group_uid=r.group_uid AND x.source_version_id=r.source_version_id) "
                "AND r.action IN ('confirmed','rejected')", (view_id,))]
        return cache[view_id]

    @staticmethod
    def decided_here(places, version_id: int, box) -> tuple[int, int]:
        """Rejections and confirmations whose (slightly enlarged) box holds this box's centre, from other images."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rej = conf = 0
        for vid, action, (x1, y1, x2, y2) in places:
            if vid == version_id:
                continue
            px, py = (x2 - x1) * 0.25 + 4, (y2 - y1) * 0.25 + 4
            if x1 - px <= cx <= x2 + px and y1 - py <= cy <= y2 + py:
                if action == "rejected":
                    rej += 1
                else:
                    conf += 1
        return rej, conf

    # ---------- where vehicles drive ----------
    def vehicle_map(self, view_id: int | None) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Union of daytime vehicle boxes in the view, or None with too few of them. Cached per service."""
        if view_id is None:
            return None
        cache = self.__dict__.setdefault("_vmaps", {})
        if view_id not in cache:
            size = self.con.execute("SELECT ref_width, ref_height FROM camera_view WHERE id=?", (view_id,)).fetchone()
            boxes = [tuple(b) for b in self.con.execute(
                "SELECT d.x1,d.y1,d.x2,d.y2 FROM detection d JOIN image_condition c ON c.source_version_id=d.source_version_id "
                "WHERE d.view_id=? AND c.condition='day' AND d.score>=0.4", (view_id,))]
            if size and len(boxes) >= self.bp.vmap_min_boxes:
                vm = vehicle_map(boxes, (size["ref_height"], size["ref_width"]), self.bp.vmap_dilate_px)
                cache[view_id] = (vm, map_distance(vm), len(boxes))
            else:
                cache[view_id] = None
        return cache[view_id]

    # ---------- distance bands ----------
    def band_setting(self, view_id: int | None) -> tuple[int | None, BandParams]:
        """The view's distance bands (latest operator setting, or the defaults) and the setting's id."""
        if view_id is None:
            return None, BandParams()
        r = self.con.execute("SELECT * FROM view_night_setting WHERE view_id=? ORDER BY id DESC LIMIT 1",
                             (view_id,)).fetchone()
        return (r["id"], BandParams(r["far_below"], r["close_from"])) if r else (None, BandParams())

    def set_bands(self, view_id: int, far_below: float, close_from: float, author: str) -> dict:
        if not 4 <= far_below < close_from <= 2000:
            raise ValueError("Give widths in pixels with far below close, for example far under 24 and close from 60.")
        self.con.execute("INSERT INTO view_night_setting(view_id,far_below,close_from,author,created_at) VALUES (?,?,?,?,?)",
                         (view_id, float(far_below), float(close_from), author, utcnow()))
        return self.bands(view_id)

    def bands(self, view_id: int | None) -> dict:
        sid, bp = self.band_setting(view_id)
        wm = self.width_map(view_id)
        return {"far_below": bp.far_below, "close_from": bp.close_from, "operator_set": sid is not None,
                "width_map": None if wm is None else {"a": wm.a, "b": wm.b, "samples": wm.samples, "source": wm.source},
                "rows": band_rows(wm, bp)}

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
        borrowed = False
        if view_id is None:                 # e.g. rain on the lens: borrow the most similar view of this camera
            view_id = self.nearest_view(version_id)
            borrowed = view_id is not None
            if borrowed and bg_row is None:
                cond = self.condition(version_id)
                bg_row = self.ensure_background(view_id, cond) if cond else None
        lights = self.lights(version_id)
        bg = self._background(bg_row)
        wm = self.width_map(view_id)
        self._setting_id, bp = self.band_setting(view_id)
        vmap = self.vehicle_map(view_id)
        self._decision_rev = self.decision_rev(view_id)
        places = self.decision_places(view_id)
        readings = [bg.reading(li, self.bp) if bg else ("no_background", 0.0) for li in lights]
        explained = [r[0] in ("persistent", "attached") for r in readings]
        live = [li for li, e in zip(lights, explained) if not e]
        live_reading = [r for r, e in zip(readings, explained) if not e]
        regions = None
        if view_id is not None and not borrowed and vs.regions_applicable(view_id):
            regions = RegionSet.load(self.con, view_id)
        elif view_id is None:
            regions = RegionSet([])
        dets = [(o.detection_id, o.box) for o in ReviewService(self.m).objects(version_id, include_removed=False)
                if o.detection_id is not None]
        now = utcnow()
        bg_id = bg_row["id"] if bg_row is not None else None
        self._region_rev = self._region_revision(view_id)
        rows = 0
        for li, e, r in zip(lights, explained, readings):
            if e:
                fl = ["attached_to_background_light"] if r[0] == "attached" else []
                self._insert(version_id, bg_id, "single", [li.id], li.box, 0.0, 0.0, 1, None, None, 0, 0, 0, [], fl, now,
                             None, r[1])
                rows += 1
        gray = None   # read only when a group needs it
        model = None if borrowed else self.learner.model_for(view_id, self._decision_rev)
        for g in group_lights(live, wm, self.gp, bp):
            ids = [live[i].id for i in g.members]
            cx, cy = (g.box[0] + g.box[2]) / 2, (g.box[1] + g.box[3]) / 2
            inside = int(regions.contains_point(cx, cy)) if regions is not None else None
            cov = covered_by(g.box, dets)
            # A group is as background-like as its least background-like light.
            state, rec = min((live_reading[i] for i in g.members), key=lambda r: r[1])
            # Where vehicles drive, lights recur because of traffic, so recurring there is not held against a
            # light (a light lit in most images is persistent anyway). Elsewhere it is a lamp or a sign.
            on_road = vmap is not None and on_vehicle_map(vmap[0], g.box)
            if state == "recurring" and not on_road:
                g.score = round(g.score * self.bp.recurring_factor, 3)
                g.flags.append("recurring_light_here")
                g.alternatives.append(f"a background light lit here in {round(rec * 100)}% of images")
            elif state == "occasional" and not on_road:
                g.score = round(g.score * self.bp.occasional_factor, 3)
                g.flags.append("occasional_light_here")
            elif state in ("recurring", "occasional"):
                g.flags.append("often_lit_where_vehicles_drive")
            if vmap is not None and "glare" not in g.flags and off_vehicle_map(vmap[1], g.box, wm):
                g.score = round(g.score * self.bp.off_road_factor, 3)
                g.flags.append("no_vehicle_seen_here_by_day")
                g.alternatives.append("no vehicle has been seen at this place in daylight")
            if gray is None:
                gray = self._gray(version_id)
            if "glare" not in g.flags:
                if gray is not None and is_soft_glow(gray, g.box, self.bp):
                    g.score = round(g.score * self.bp.soft_factor, 3)
                    g.flags.append("soft_glow_no_bright_core")
                    g.alternatives.insert(0, "light on the road surface, a lit lane line or a drop on the lens")
            road = None
            if vmap is not None:
                h_, w_ = vmap[1].shape
                road = float(vmap[1][min(h_ - 1, max(0, int(cy))), min(w_ - 1, max(0, int(cx)))])
            feats = measure(gray, g.box, [live[i] for i in g.members], wm, rec if bg else None, road, g.kind, g.band)
            model_id = None
            if model is not None:
                # The view's learned model replaces the rule score (it was checked against the rules on your answers).
                model_id = model[0]
                g.score = round(float(model[1].predict([[np.nan if v is None else float(v) for v in vector(feats)]])[0]), 3)
                g.flags.append("learned_score")
            rej, conf = self.decided_here(places, version_id, g.box)
            if conf >= self.bp.confirmed_min and conf >= rej:
                g.score = round(max(g.score, 0.4), 3)
                g.flags.append("vehicles_confirmed_here_before")
            elif rej >= self.bp.rejected_min and rej > 2 * conf:
                g.score = round(g.score * self.bp.rejected_factor, 3)
                g.flags.append("rejected_here_before")
                g.alternatives.insert(0, f"an operator said 'not a vehicle' at this place {rej} times")
            flags = list(g.flags) + (["no_view"] if view_id is None else []) + \
                (["background_from_nearest_view"] if borrowed else []) + \
                (["regions_not_applied"] if inside is None else []) + (["no_background"] if bg is None else [])
            cand = int(cov is None and inside != 0 and g.score >= MIN_CANDIDATE_SCORE)
            self._insert(version_id, bg_id, g.kind, ids, g.box, g.score, g.reflection, 0, inside, cov, cand,
                         g.count_min if cand else 0, g.count_max if cand else 0, g.alternatives, flags, now, g.band, rec if bg else None,
                         feats, model_id)
            rows += 1
        return rows

    def _gray(self, version_id: int) -> np.ndarray | None:
        """Grey image for measurements; None when the source file is gone (the check is then skipped)."""
        from .views import ViewService, read_image
        path = ViewService(self.m)._path(version_id)
        if not path.is_file():
            return None
        try:
            return cv2.cvtColor(read_image(path), cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    def _insert(self, version_id, bg_id, kind, ids, box, score, refl, explained, inside, cov, cand, cmin, cmax, alts,
                flags, now, band=None, recurrence=None, features=None, model_id=None):
        self.con.execute(
            "INSERT INTO light_group(source_version_id,background_model_id,region_revision_id,kind,member_ids,x1,y1,x2,y2,score,reflection,"
            "explained_by_background,in_region,covered_by_detection,is_candidate,count_min,count_max,alternatives,flags,"
            "created_at,band,recurrence,setting_id,decision_rev,algo,features,model_id) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version_id, bg_id, self._region_rev, kind, dumps(sorted(ids)), *(float(v) for v in box), float(score), float(refl), explained,
             inside, cov, cand, cmin, cmax, dumps(alts), dumps(flags), now, band,
             None if recurrence is None else round(float(recurrence), 3), self._setting_id, self._decision_rev, GROUP_ALGO,
             dumps(features) if features is not None else None, model_id))

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
            "SELECT background_model_id, region_revision_id, setting_id, decision_rev, algo FROM light_group "
            "WHERE source_version_id=? ORDER BY id DESC LIMIT 1", (version_id,)).fetchone()
        return self.con.execute(
            "SELECT * FROM light_group WHERE source_version_id=? AND background_model_id IS ? AND region_revision_id IS ? "
            "AND setting_id IS ? AND decision_rev IS ? AND algo IS ? AND created_at=? ORDER BY id",
            (version_id, latest[0], latest[1], latest[2], latest[3], latest[4], last)).fetchall()

    def refresh(self, with_decisions: bool = False) -> dict:
        """Bring backgrounds and groups up to date for all processed dusk and night images.

        with_decisions: also re-analyse images analysed before the view's latest operator decision on a
        candidate, so what operators decided at a place is applied to the other images. Processing runs do
        this; a single decision on the investigation screen does not, to stay fast."""
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
            setting = self.band_setting(view_id)[0]
            decided = self.decision_rev(view_id) if with_decisions else None
            for v in self._versions(view_id, cond):
                if with_decisions:
                    done = self.con.execute("SELECT 1 FROM light_group WHERE source_version_id=? AND background_model_id=? "
                                            "AND region_revision_id IS ? AND setting_id IS ? AND decision_rev IS ? AND algo IS ? LIMIT 1",
                                            (v, bg["id"], region_rev, setting, decided, GROUP_ALGO)).fetchone()
                else:
                    done = self.con.execute("SELECT 1 FROM light_group WHERE source_version_id=? AND background_model_id=? "
                                            "AND region_revision_id IS ? AND setting_id IS ? AND algo IS ? LIMIT 1",
                                            (v, bg["id"], region_rev, setting, GROUP_ALGO)).fetchone()
                if done or not self.lights(v):
                    continue
                self._analyse_tx(v, bg)
                stats["images_analysed"] += 1
        # Dusk or night images without a view: analysed with the background of the most similar view when
        # there is one (flagged), otherwise without background or regions (flagged).
        for (v,) in self.con.execute(
                "SELECT c.source_version_id FROM image_condition c WHERE c.condition IN ('night','dusk') AND NOT EXISTS "
                "(SELECT 1 FROM light_group g WHERE g.source_version_id=c.source_version_id AND g.algo IS ?) AND EXISTS (SELECT 1 FROM "
                "run_item i WHERE i.source_version_id=c.source_version_id AND i.processing_state='done')", (GROUP_ALGO,)).fetchall():
            if vs.current_view(v) is None and self.lights(v):
                self._analyse_tx(v, None)
                stats["images_analysed"] += 1
        return stats

    def _analyse_tx(self, version_id: int, bg) -> None:
        """Analyse one image in its own transaction. On an error nothing of it is kept, including a model
        fitted along the way, so the cached model is dropped too."""
        try:
            with transaction(self.con):
                self.analyse(version_id, bg)
        except BaseException:
            self.learner._cache.clear()
            raise
