"""Plate finding and reading for plate recognition missions (IMG-04, IMG-05).

For each vehicle the detector found, the plate finder looks inside the vehicle
box; the best plate is cropped and read. Everything the machine produced is
stored as it came out: the box, the crop, the raw text with a confidence per
character, the normalised text, the layout and readability. Operators confirm
or correct plates in append-only revisions.

A country is **evaluation only** until an evaluation on that mission's
reviewed plates passes the thresholds and an operator enables it. Evaluation
only reads are shown and exported with that label. No owner lookup exists.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .detect import VEHICLE_CLASSES
from .models import ModelPack
from .plate_layout import COUNTRIES, READABILITY, ReadRules, assess, cer, normalize
from .process import RunControl, register_model
from .store import Mission, utcnow
from .views import ViewService, read_image


class PlateError(ValueError):
    pass


# ------------------------------------------------------------------ models

@dataclass(frozen=True)
class PlateBox:
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class OcrOut:
    slots: list[tuple[str, float]]
    pad: str
    region: str | None
    region_conf: float | None


class PlateFinder(Protocol):
    def find(self, crop_bgr: np.ndarray) -> list[PlateBox]: ...
    def describe(self) -> dict: ...


class PlateReader(Protocol):
    def read(self, plate_bgr: np.ndarray) -> OcrOut: ...
    def describe(self) -> dict: ...


def _session(pack: ModelPack, accelerator=None):
    from .runtime import make_session
    return make_session(pack.weights, accelerator)[0]


class YoloPlateFinder:
    """End-to-end YOLO plate detector (NMS inside the model). Input: RGB 0..1, centred letterbox."""

    def __init__(self, pack: ModelPack, score_threshold: float = 0.25, accelerator=None):
        self.pack, self.thr = pack, score_threshold
        self.s = _session(pack, accelerator)
        self.inp = self.s.get_inputs()[0].name
        self.size = int(pack.card.get("input_size", 640))

    def describe(self) -> dict:
        return {"model": {"name": self.pack.name, "version": self.pack.version, "sha256": self.pack.sha256},
                "preprocessing": {"letterbox": self.size, "centred": True, "pad": 114, "colour": "RGB 0..1",
                                  "score_threshold": self.thr}}

    def find(self, crop: np.ndarray) -> list[PlateBox]:
        h, w = crop.shape[:2]
        r = min(self.size / h, self.size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        im = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (self.size - nw) / 2, (self.size - nh) / 2
        im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)),
                                int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = (im.transpose(2, 0, 1)[::-1] / 255.0).astype(np.float32)[None]
        out = self.s.run(None, {self.inp: x})[0]
        res = []
        for row in out:
            if float(row[6]) < self.thr:
                continue
            x1, y1 = (row[1] - dw) / r, (row[2] - dh) / r
            x2, y2 = (row[3] - dw) / r, (row[4] - dh) / r
            x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
            if x2 > x1 and y2 > y1:
                res.append(PlateBox((float(x1), float(y1), float(x2), float(y2)), float(row[6])))
        return sorted(res, key=lambda b: -b.score)


class SlotPlateReader:
    """Fixed-slot plate reader: one character class per slot, plus a country head."""

    def __init__(self, pack: ModelPack, accelerator=None):
        c = pack.card
        self.pack = pack
        self.alphabet, self.pad, self.slots = c["alphabet"], c["pad_char"], int(c["max_slots"])
        self.h, self.w = c["input_size"]
        self.regions = c.get("region_labels") or []
        self.s = _session(pack, accelerator)
        self.inp = self.s.get_inputs()[0].name

    def describe(self) -> dict:
        return {"model": {"name": self.pack.name, "version": self.pack.version, "sha256": self.pack.sha256},
                "preprocessing": {"resize": [self.w, self.h], "colour": "RGB uint8", "keep_aspect": False},
                "alphabet": self.alphabet}

    def read(self, plate: np.ndarray) -> OcrOut:
        x = cv2.resize(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        outs = self.s.run(None, {self.inp: x[None].astype(np.uint8)})
        p = outs[0].reshape(self.slots, len(self.alphabet))
        slots = [(self.alphabet[int(i)], float(v)) for i, v in zip(p.argmax(-1), p.max(-1))]
        region = conf = None
        if len(outs) > 1 and self.regions:
            r = outs[1].reshape(-1)
            k = int(r.argmax())
            region, conf = (self.regions[k] if k < len(self.regions) else str(k)), float(r[k])
        return OcrOut(slots, self.pad, region, conf)


def default_plate_models():
    """Load the installed plate finder and reader packs (evaluation only)."""
    from . import models
    return (YoloPlateFinder(models.find("plate-yolov9-t-640")), SlotPlateReader(models.find("plate-ocr-cct-s-v2-global")))


# ------------------------------------------------------------------ service

PLATE_VEHICLES = tuple(c for c in VEHICLE_CLASSES if c != "bicycle")
ACTIONS = ("confirm", "correct", "unreadable", "no_plate", "reopen")


@dataclass
class PlateSettings:
    margin: float = 0.05             # around the vehicle box
    min_vehicle_side_px: int = 24
    upscale_to: int = 320            # small vehicle crops are enlarged before the finder sees them
    rules: ReadRules = field(default_factory=ReadRules)

    def record(self) -> dict:
        d = asdict(self)
        d["rules"] = self.rules.record()
        return d


@dataclass
class PlateView:
    id: int | None
    source_version_id: int
    file: str
    capture_utc: str | None
    object_uid: str
    vehicle_cls: str
    vehicle_box: list[float]
    outcome: str
    box: list[float] | None
    det_score: float | None
    raw_text: str | None
    chars: list
    machine_text: str | None
    machine_display: str | None
    machine_readability: str | None
    layout: str | None
    region_label: str | None
    region_conf: float | None
    reasons: list[str]
    country: str
    evaluation_only: bool
    text: str | None = None              # what stands after operator revisions
    readability: str | None = None
    review: str = "pending"              # pending | confirmed | corrected | unreadable | no_plate
    revisions: int = 0
    operator_box: list[float] | None = None


class PlateService:
    def __init__(self, mission: Mission, finder: PlateFinder | None = None, reader: PlateReader | None = None,
                 settings: PlateSettings | None = None):
        self.m = mission
        self.con = mission.con
        self.finder, self.reader = finder, reader
        self.s = settings or PlateSettings()
        self.views = ViewService(mission)

    @property
    def country(self) -> str:
        return self.m.config().country

    # ---------- machine stage ----------
    def describe(self) -> dict:
        return {"finder": self.finder.describe() if self.finder else None,
                "reader": self.reader.describe() if self.reader else None, "settings": self.s.record()}

    def refresh(self, control: RunControl | None = None, progress: Callable[[int, int], None] | None = None) -> dict:
        """Read plates of every current vehicle detection not yet read with this finder and reader."""
        control = control or RunControl()
        fd, rd = self.finder.describe(), self.reader.describe()
        fid = register_model(self.con, fd, getattr(getattr(self.finder, "pack", None), "card", {"task": "plate_detection"}))
        rid = register_model(self.con, rd, getattr(getattr(self.reader, "pack", None), "card", {"task": "plate_ocr"}))
        q = ",".join("?" * len(PLATE_VEHICLES))
        rows = self.con.execute(
            f"SELECT d.*, f.rel_path FROM detection d JOIN run_item i ON i.id=d.run_item_id "
            f"JOIN source_version v ON v.id=d.source_version_id JOIN source_file f ON f.id=v.source_file_id "
            f"WHERE i.processing_state='done' AND v.superseded_by IS NULL AND d.cls IN ({q}) "
            f"AND i.id=(SELECT MAX(j.id) FROM run_item j WHERE j.source_version_id=d.source_version_id) "
            f"AND NOT EXISTS (SELECT 1 FROM plate_read p WHERE p.detection_id=d.id AND p.detector_model_id=? "
            f"AND p.reader_model_id=?) ORDER BY d.source_version_id, d.id", (*PLATE_VEHICLES, fid, rid)).fetchall()
        by_img: dict[int, list] = {}
        for r in rows:
            by_img.setdefault(r["source_version_id"], []).append(r)
        stats = {"images": 0, "vehicles": 0, "read": 0, "no_plate_found": 0, "too_small": 0, "failed": 0}
        crops = self.m.dir / "evidence" / "plates"
        crops.mkdir(parents=True, exist_ok=True)
        rules = json.dumps(self.s.record())
        country = self.country
        for n, (vid, dets) in enumerate(by_img.items(), 1):
            control.checkpoint()                # cancel or pause only between images
            try:
                img = read_image(self.views._path(vid))
            except Exception as exc:
                stats["failed"] += 1
                self.m.notify("warning", "plate_read_failed", f"Could not read {dets[0]['rel_path']}: {exc}",
                              source_version_id=vid)
                continue
            H, W = img.shape[:2]
            self.con.execute("BEGIN")
            try:
                for d in dets:
                    stats["vehicles"] += 1
                    row = self._read_one(img, W, H, d, country, crops)
                    stats[{"read": "read", "no_plate_found": "no_plate_found"}.get(row["outcome"], "too_small")] += 1
                    cols = ["detection_id", "source_version_id", "object_uid", *row.keys(), "detector_model_id",
                            "reader_model_id", "rules", "created_at"]
                    vals = [d["id"], vid, f"d{d['id']}", *row.values(), fid, rid, rules, utcnow()]
                    self.con.execute(f"INSERT INTO plate_read({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise
            stats["images"] += 1
            if progress:
                progress(n, len(by_img))
        return stats

    def _read_one(self, img, W, H, d, country: str, crops: Path) -> dict:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < self.s.min_vehicle_side_px:
            return {"outcome": "vehicle_too_small", "reasons": json.dumps([f"vehicle box {bw:.0f}x{bh:.0f} px"])}
        mx, my = bw * self.s.margin, bh * self.s.margin
        a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
        c, e = min(W, int(np.ceil(x2 + mx))), min(H, int(np.ceil(y2 + my)))
        crop = img[b:e, a:c]
        scale = max(1.0, self.s.upscale_to / max(crop.shape[:2]))
        view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
        found = self.finder.find(view)
        # The plate must sit inside the vehicle box (its centre), not on a neighbour caught by the margin.
        found = [f for f in found if x1 <= a + (f.box[0] + f.box[2]) / 2 / scale <= x2
                 and y1 <= b + (f.box[1] + f.box[3]) / 2 / scale <= y2]
        if not found:
            return {"outcome": "no_plate_found", "reasons": json.dumps([])}
        best = found[0]
        px1, py1 = a + best.box[0] / scale, b + best.box[1] / scale
        px2, py2 = a + best.box[2] / scale, b + best.box[3] / scale
        row = {"x1": round(px1, 2), "y1": round(py1, 2), "x2": round(px2, 2), "y2": round(py2, 2),
               "det_score": round(best.score, 4)}
        pc = img[max(0, int(py1)):min(H, int(np.ceil(py2))), max(0, int(px1)):min(W, int(np.ceil(px2)))]
        if pc.size == 0:
            return {"outcome": "no_plate_found", "reasons": json.dumps(["empty plate box"])}
        path = crops / f"det{d['id']}_{d['source_version_id']}.png"
        cv2.imwrite(str(path), pc)
        row["crop_path"] = str(path)
        if py2 - py1 < self.s.rules.min_plate_height_px:
            return {"outcome": "plate_too_small", **row, "readability": "unreadable",
                    "reasons": json.dumps([f"plate is {py2 - py1:.0f} px high; at least "
                                           f"{self.s.rules.min_plate_height_px} px are needed to read it"])}
        cp = COUNTRIES.get(country)
        if cp is not None and not cp.reader_covers:
            return {"outcome": "read", **row, "readability": "unreadable", "country_hyp": country,
                    "reasons": json.dumps([f"no reader for {cp.name} plates ({cp.script}) is installed"])}
        o = self.reader.read(pc)
        rd = assess(o.slots, o.pad, country, self.s.rules)
        # The reader is trained on single-line plates. A square box (a sign, a route number, a two-line plate) or a
        # box the finder is unsure of can still produce confident characters, so such reads are never "full".
        if rd.readability != "unreadable":
            aspect = (px2 - px1) / max(py2 - py1, 1e-6)
            if aspect < self.s.rules.min_aspect:
                rd.readability = "partial"
                rd.reasons.append(f"box is not plate-shaped (width/height {aspect:.1f}): a two-line plate, or not a plate")
            if best.score < self.s.rules.finder_sure:
                rd.readability = "partial"
                rd.reasons.append(f"the plate finder is unsure this is a plate (score {best.score:.2f})")
        return {"outcome": "read", **row, "raw_text": rd.raw_text, "raw_chars": json.dumps(rd.chars, ensure_ascii=False),
                "normalized_text": rd.normalized, "display_text": rd.display, "country_hyp": country,
                "layout_hyp": rd.layout, "region_label": o.region,
                "region_conf": None if o.region_conf is None else round(o.region_conf, 4),
                "readability": rd.readability, "reasons": json.dumps(rd.reasons)}

    # ---------- reading ----------
    def country_status(self, country: str | None = None) -> dict:
        country = country or self.country
        last = self.con.execute("SELECT * FROM plate_country_status WHERE country=? AND action IN ('enable','disable') "
                                "ORDER BY id DESC LIMIT 1", (country,)).fetchone()
        ev = self.con.execute("SELECT * FROM plate_country_status WHERE country=? AND action='evaluate' "
                              "ORDER BY id DESC LIMIT 1", (country,)).fetchone()
        cp = COUNTRIES.get(country)
        return {"country": country, "name": cp.name if cp else country, "script": cp.script if cp else None,
                "reader_covers": bool(cp and cp.reader_covers),
                "layouts": [{"name": l.name, "note": l.note} for l in cp.layouts] if cp else [],
                "enabled": bool(last and last["action"] == "enable"),
                "enabled_by": last["author"] if last and last["action"] == "enable" else None,
                "last_evaluation": json.loads(ev["result"]) | {"passed": bool(ev["passed"]), "at": ev["created_at"],
                                                              "by": ev["author"]} if ev else None}

    def plates(self, version_id: int | None = None, outcome: str | None = None, readability: str | None = None,
               review: str | None = None, search: str | None = None) -> list[PlateView]:
        where, args = "", []
        if version_id is not None:
            where, args = "AND p.source_version_id=?", [version_id]
        rows = self.con.execute(
            "SELECT p.*, d.cls AS vcls, d.x1 AS vx1, d.y1 AS vy1, d.x2 AS vx2, d.y2 AS vy2, f.rel_path, v.capture_utc "
            "FROM plate_read p JOIN detection d ON d.id=p.detection_id JOIN source_version v ON v.id=p.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL AND p.id IN (SELECT MAX(id) FROM plate_read GROUP BY detection_id) "
            f"AND d.run_item_id=(SELECT MAX(j.id) FROM run_item j WHERE j.source_version_id=d.source_version_id) {where} "
            "ORDER BY v.capture_utc, f.rel_path, p.id", args).fetchall()
        enabled = self.country_status()["enabled"]
        out: list[PlateView] = []
        for r in rows:
            out.append(PlateView(
                r["id"], r["source_version_id"], r["rel_path"], r["capture_utc"], r["object_uid"], r["vcls"],
                [r["vx1"], r["vy1"], r["vx2"], r["vy2"]], r["outcome"],
                [r["x1"], r["y1"], r["x2"], r["y2"]] if r["x1"] is not None else None, r["det_score"], r["raw_text"],
                json.loads(r["raw_chars"] or "[]"), r["normalized_text"], r["display_text"], r["readability"],
                r["layout_hyp"], r["region_label"], r["region_conf"], json.loads(r["reasons"]),
                r["country_hyp"] or self.country, not enabled, r["normalized_text"], r["readability"]))
        self._apply(out)
        if outcome:
            out = [p for p in out if p.outcome == outcome]
        if readability:
            out = [p for p in out if p.readability == readability]
        if review:
            out = [p for p in out if p.review == review]
        if search:
            key = normalize(search)
            out = [p for p in out if p.text and key in p.text]
        return out

    def _apply(self, plates: list[PlateView]) -> None:
        by = {(p.source_version_id, p.object_uid): p for p in plates}
        if not by:
            return
        vids = sorted({k[0] for k in by})
        q = ",".join("?" * len(vids))
        for r in self.con.execute(f"SELECT * FROM plate_revision WHERE source_version_id IN ({q}) ORDER BY id", vids):
            p = by.get((r["source_version_id"], r["object_uid"]))
            if p is None:
                continue
            p.revisions += 1
            a = r["action"]
            if a == "confirm":
                p.review, p.text, p.readability, p.operator_box = "confirmed", p.machine_text, "full", None
            elif a == "correct":
                p.review, p.text, p.readability = "corrected", r["normalized_text"], "full"
                if r["box"]:
                    p.operator_box = json.loads(r["box"])
            elif a == "unreadable":
                p.review, p.text, p.readability = "unreadable", None, "unreadable"
            elif a == "no_plate":
                p.review, p.text, p.readability = "no_plate", None, None
            elif a == "reopen":
                p.review, p.text, p.readability, p.operator_box = "pending", p.machine_text, p.machine_readability, None

    def plate(self, plate_id: int) -> PlateView:
        r = self.con.execute("SELECT source_version_id, detection_id FROM plate_read WHERE id=?", (plate_id,)).fetchone()
        if r is None:
            raise PlateError("No such plate.")
        for p in self.plates(r["source_version_id"]):
            if p.object_uid == f"d{r['detection_id']}":
                return p
        raise PlateError("That plate belongs to an older processing run.")

    # ---------- operator ----------
    def revise(self, plate_id: int, action: str, author: str, text: str | None = None, box: list[float] | None = None,
               note: str | None = None) -> PlateView:
        if action not in ACTIONS:
            raise PlateError(f"Action must be one of {', '.join(ACTIONS)}.")
        p = self.plate(plate_id)
        norm = None
        if action == "confirm":
            # Confirming says the machine read is right, whatever was decided before.
            if p.machine_readability != "full" or not p.machine_text or "?" in p.machine_text:
                raise PlateError("Only a fully read plate can be confirmed as it is. Type the text you can read, "
                                 "or mark it unreadable.")
            text, norm = p.machine_text, p.machine_text
        elif action == "correct":
            if not text or not text.strip():
                raise PlateError("Type the plate text, or mark the plate unreadable.")
            norm = normalize(text.strip())
            if "?" in norm:
                raise PlateError("A corrected plate must be fully readable. Mark it unreadable instead of guessing.")
            if box is not None:
                box = self._check_box(p, box)
            text = text.strip()
        else:
            text, box = None, None
        self.con.execute(
            "INSERT INTO plate_revision(source_version_id,object_uid,action,text,normalized_text,box,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (p.source_version_id, p.object_uid, action, text, norm, json.dumps(box) if box else None, author, note, utcnow()))
        return self.plate(plate_id)

    def _check_box(self, p: PlateView, box) -> list[float]:
        """An operator's plate box: inside the image, and its centre on the vehicle (10% margin)."""
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise PlateError("The plate box must be x1,y1,x2,y2 with x2>x1 and y2>y1.")
        box = [round(float(v), 1) for v in box]
        sig = self.con.execute("SELECT width, height FROM view_signature WHERE source_version_id=?",
                               (p.source_version_id,)).fetchone()
        if sig and (box[0] < 0 or box[1] < 0 or box[2] > sig["width"] or box[3] > sig["height"]):
            raise PlateError("The plate box is outside the image.")
        if box[2] - box[0] < 4 or box[3] - box[1] < 2:
            raise PlateError("The plate box is too small to be a plate.")
        x1, y1, x2, y2 = p.vehicle_box
        mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if not (x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my):
            raise PlateError("The plate box is not on this vehicle. Select the vehicle the plate belongs to.")
        return box

    # ---------- evaluation gate ----------
    def evaluate(self, author: str, min_plates: int = 100, min_exact: float = 0.90, max_cer: float = 0.05,
                 country: str | None = None) -> dict:
        """Compare machine reads with operator-reviewed plates of this mission.

        Only plates an operator confirmed, corrected or marked unreadable count. A machine read the operator marked
        unreadable is a guess and is counted against the reader.
        """
        country = country or self.country
        reviewed = [p for p in self.plates() if p.country == country and p.review in ("confirmed", "corrected", "unreadable")
                    and p.outcome in ("read", "plate_too_small")]
        per: dict[str, dict] = {}
        guesses = 0
        for p in reviewed:
            key = p.layout or "no layout"
            st = per.setdefault(key, {"plates": 0, "exact": 0, "cer_sum": 0.0})
            if p.review == "unreadable":
                if p.machine_readability == "full":
                    guesses += 1
                continue
            st["plates"] += 1
            st["exact"] += int((p.machine_text or "") == p.text)
            st["cer_sum"] += cer(p.text or "", p.machine_text or "")
        n = sum(v["plates"] for v in per.values())
        exact = sum(v["exact"] for v in per.values()) / n if n else 0.0
        cer_all = sum(v["cer_sum"] for v in per.values()) / n if n else 1.0
        missed = sum(1 for p in self.plates() if p.country == country and p.review == "corrected" and p.operator_box
                     and p.outcome in ("no_plate_found", "vehicle_too_small"))
        result = {"country": country, "reviewed_plates": n, "finder_missed": missed, "operator_unreadable": sum(p.review == "unreadable" for p in reviewed),
                  "machine_guesses_on_unreadable": guesses, "exact_match": round(exact, 4), "cer": round(cer_all, 4),
                  "per_layout": {k: {"plates": v["plates"], "exact_match": round(v["exact"] / v["plates"], 4) if v["plates"] else None,
                                     "cer": round(v["cer_sum"] / v["plates"], 4) if v["plates"] else None} for k, v in per.items()},
                  "thresholds": {"min_plates": min_plates, "min_exact": min_exact, "max_cer": max_cer,
                                 "machine_guesses_on_unreadable": 0}}
        passed = n >= min_plates and exact >= min_exact and cer_all <= max_cer and guesses == 0
        rid = self._reader_id()
        self.con.execute("INSERT INTO plate_country_status(country,action,reader_model_id,result,passed,author,created_at) "
                         "VALUES (?,?,?,?,?,?,?)", (country, "evaluate", rid, json.dumps(result), int(passed), author, utcnow()))
        return result | {"passed": passed}

    def _reader_id(self) -> int | None:
        r = self.con.execute("SELECT reader_model_id FROM plate_read ORDER BY id DESC LIMIT 1").fetchone()
        return r[0] if r else None

    def set_enabled(self, enabled: bool, author: str, note: str | None = None, country: str | None = None) -> dict:
        country = country or self.country
        st = self.country_status(country)
        if enabled:
            ev = st["last_evaluation"]
            if not ev or not ev["passed"]:
                raise PlateError("Run an evaluation on reviewed plates first; the country can be enabled only after it passes.")
            if not st["reader_covers"]:
                raise PlateError("The installed reader cannot read this country's plates.")
        self.con.execute("INSERT INTO plate_country_status(country,action,reader_model_id,result,passed,author,note,created_at) "
                         "VALUES (?,?,?,?,?,?,?,?)", (country, "enable" if enabled else "disable", self._reader_id(), "{}",
                                                      None, author, note, utcnow()))
        return self.country_status(country)

    # ---------- export ----------
    def export(self, author: str, formats: list[str] | None = None, out_dir=None) -> list[Path]:
        from datetime import datetime, timezone

        from .export import _fmt, write_sheets_csv, write_sheets_xlsx
        cfg = self.m.config()
        formats = formats or cfg.export_formats
        out = Path(out_dir or cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sha = {r[0]: r[1] for r in self.con.execute("SELECT id, sha256 FROM source_version")}
        models = {r[0]: f"{r[1]} {r[2]}" for r in self.con.execute("SELECT id, name, version FROM model_version")}
        ids = {r["id"]: (r["detector_model_id"], r["reader_model_id"]) for r in
               self.con.execute("SELECT id, detector_model_id, reader_model_id FROM plate_read")}
        rows = []
        for p in self.plates():
            fid, rid = ids.get(p.id, (None, None))
            rows.append([_fmt(x) for x in (
                self.m.id, p.source_version_id, p.file, sha.get(p.source_version_id), p.capture_utc, p.object_uid,
                p.vehicle_cls, p.outcome, *(p.box or ["", "", "", ""]), p.det_score,
                *(p.operator_box or ["", "", "", ""]), p.raw_text,
                " ".join(f"{c}:{v:.2f}" for c, v in p.chars), p.machine_text, p.machine_display, p.machine_readability,
                p.layout, p.country, p.region_label, p.region_conf, "; ".join(p.reasons), p.review, p.text, p.readability,
                p.revisions, "yes" if p.evaluation_only else "no", models.get(fid, ""), models.get(rid, ""))])
        cols = ["mission_id", "source_version_id", "file", "sha256", "capture_utc", "vehicle_id", "vehicle_class",
                "outcome", "plate_x1", "plate_y1", "plate_x2", "plate_y2", "plate_score",
                "operator_x1", "operator_y1", "operator_x2", "operator_y2", "raw_text", "char_confidences",
                "machine_text", "machine_display", "machine_readability", "layout", "country", "reader_country_guess",
                "reader_country_conf", "reasons", "review", "plate_text", "readability", "operator_revisions",
                "evaluation_only", "plate_finder", "plate_reader"]
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                       for ch in f"{cfg.name}_plates_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        paths = []
        for fmt in formats:
            new = (write_sheets_csv if fmt == "csv" else write_sheets_xlsx)(out, stem, [("plates", cols, rows)])
            paths += new
            self.con.execute("INSERT INTO export(format,path,scope,rows,status,author,created_at) VALUES (?,?,?,?,?,?,?)",
                             (fmt, ";".join(str(x) for x in new), json.dumps({"plates": True}), len(rows), "completed",
                              author, utcnow()))
        return paths


__all__ = ["PlateService", "PlateSettings", "PlateView", "PlateError", "YoloPlateFinder", "SlotPlateReader",
           "default_plate_models", "PlateBox", "OcrOut", "READABILITY"]
