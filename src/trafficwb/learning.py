"""Learning, part 1: eligible evidence, dataset versions, evaluation and model promotion (LRN-01..04).

Rules (design package, deliverable 07):
- Only explicit reviewed labels are used. An image an operator approved
  individually is a **full** review: every vehicle inside the included regions
  is marked, so unmarked areas there are background. An image with corrections
  but no approval is a **partial** review: only the corrected objects are
  labels, everything else is ignored, never treated as background. Bulk
  approved and unreviewed images are never used.
- A dataset version is a frozen list of images, labels and splits. Splits are
  group-wise by camera view and date block, so near-duplicate frames of one
  view and day never sit in two splits. Once an image group is in a test set,
  later dataset versions keep it in test.
- Evaluation runs a detector on a split and compares it with the labels:
  precision, recall, F1, AP at IoU 0.5 and per-image count error, overall, by
  light condition and by view. Precision and count error use full reviews
  only; partial reviews add recall for their corrected objects.
- No training adapter is implemented yet; each algorithm declares its planned
  adapter (LRN-02) and nothing is shown as trainable.
- The production model per task changes only when an operator promotes a
  version; rollback moves the pointer back. Nothing is promoted automatically.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from .detect import VEHICLE_CLASSES
from .localtime import to_local
from .process import RunControl
from .regions import RegionSet
from .review import ReviewService
from .store import CONTROL_SCHEMA, Mission, connect, utcnow
from .views import ViewService, read_image


class LearningError(ValueError):
    pass


# LRN-02: every algorithm declares its learning adapter. "planned" means designed, not implemented.
ADAPTERS = [
    {"algorithm": "Vehicle detector", "adapter": "fine-tune on this computer from the learning page (NVIDIA or Apple GPU, or the CPU)",
     "status": "available"},
    {"algorithm": "Vehicle detector thresholds", "adapter": "thresholds per view", "status": "planned"},
    {"algorithm": "Colour reader", "adapter": "small classifier trained on corrected colours", "status": "planned"},
    {"algorithm": "Night light grouping", "adapter": "pair classifier on reviewed groups", "status": "planned"},
    {"algorithm": "Persistent-light background", "adapter": "background refit", "status": "planned"},
    {"algorithm": "Plate finder", "adapter": "fine-tune per country (GPU)", "status": "planned"},
    {"algorithm": "Plate reader", "adapter": "fine-tune per country and layout (GPU)", "status": "planned"},
    {"algorithm": "Video tracker", "adapter": "association parameters", "status": "planned"},
    {"algorithm": "View grouping", "adapter": "merge thresholds", "status": "planned"},
    {"algorithm": "Search interpreter", "adapter": "none: vocabulary revisions only", "status": "not_applicable"},
]

SPLITS = ("train", "val", "test")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class ItemLabels:
    source_version_id: int
    review: str                       # full | partial | bulk | none
    positives: list[dict]             # {"box": [...], "cls": str, "uid": str, "source": machine|human|candidate}
    ignore_boxes: list[list[float]]   # unresolved night candidates: never background
    last_revision_id: int | None


class LearningService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.rs = ReviewService(mission)
        self.views = ViewService(mission)
        self.zone = mission.config().iana_zone

    # ---------- eligibility ----------
    def review_kind(self, vid: int) -> str:
        st = self.rs.review_status(vid)
        if st == "approved":
            return "full"
        if st == "bulk_approved_unreviewed":
            return "bulk"
        touched = self.con.execute(
            "SELECT (SELECT COUNT(*) FROM annotation_revision WHERE source_version_id=?) + "
            "(SELECT COUNT(*) FROM candidate_revision WHERE source_version_id=?)", (vid, vid)).fetchone()[0]
        return "partial" if touched else "none"

    def labels(self, vid: int) -> ItemLabels:
        kind = self.review_kind(vid)
        pos: list[dict] = []
        ignore: list[list[float]] = []
        if kind in ("full", "partial"):
            for o in self.rs.objects(vid, include_removed=False):
                if kind == "partial" and o.correction in (None, "none"):
                    continue            # untouched machine output is not a label in a partial review
                pos.append({"box": [round(float(v), 2) for v in o.box], "cls": o.cls, "uid": o.uid,
                            "source": "human" if o.origin == "human" or o.correction not in (None, "none") else "machine"})
            for c in self.rs.candidates(vid):
                if c.status == "confirmed":
                    pos.append({"box": [round(float(v), 2) for v in c.box], "cls": c.cls or "car", "uid": c.uid,
                                "source": "candidate"})
                elif c.status == "unresolved" and kind == "full":
                    ignore.append([round(float(v), 2) for v in c.box])
        last = self.con.execute(
            "SELECT MAX(id) FROM annotation_revision WHERE source_version_id=?", (vid,)).fetchone()[0]
        return ItemLabels(vid, kind, pos, ignore, last)

    def _images(self) -> list:
        return self.con.execute(
            "SELECT v.id, v.sha256, v.capture_utc, f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL AND EXISTS (SELECT 1 FROM run_item i WHERE i.source_version_id=v.id AND "
            "i.processing_state='done') ORDER BY v.capture_utc, v.id").fetchall()

    def eligibility(self) -> dict:
        counts = {"full": 0, "partial": 0, "bulk": 0, "none": 0}
        for r in self._images():
            counts[self.review_kind(r["id"])] += 1
        return {"images": counts, "eligible": counts["full"] + counts["partial"],
                "rules": {"full": "approved individually: positives and background inside included regions",
                          "partial": "corrected but not approved: corrected objects only, the rest ignored",
                          "bulk": "never used", "none": "never used"}}

    # ---------- datasets ----------
    def _group(self, r, block_days: int) -> tuple[str, str]:
        vid = r["id"]
        view = self.views.current_view(vid)
        view_key = f"view-{view}" if view is not None else "unassigned"
        if r["capture_utc"]:
            utc = datetime.fromisoformat(r["capture_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
            d = to_local(utc, self.zone).local.date()
            start = date.fromordinal(d.toordinal() - (d.toordinal() % block_days))
            block = start.isoformat()
        else:
            block = "no-date"
        return view_key, block

    @staticmethod
    def _bucket(seed: int, key: str) -> float:
        h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64

    def create_dataset(self, name: str, author: str, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                       block_days: int = 1, seed: int = 0) -> dict:
        if not name.strip():
            raise LearningError("Give the dataset version a name.")
        if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios) - 1) > 1e-6:
            raise LearningError("Split ratios must be three non-negative numbers that add up to 1.")
        if block_days < 1 or block_days > 366:
            raise LearningError("A date block is 1 to 366 days.")
        prev = self.con.execute("SELECT id FROM training_dataset ORDER BY id DESC LIMIT 1").fetchone()
        frozen = {}
        if prev:
            # Groups already in a test set stay in test (the test set is frozen across versions).
            for r in self.con.execute("SELECT DISTINCT view_key, block FROM dataset_member WHERE split='test'"):
                frozen[(r["view_key"], r["block"])] = "test"
            for r in self.con.execute("SELECT DISTINCT d.source_version_id FROM dataset_member d WHERE d.split='test'"):
                frozen[("vid", str(r[0]))] = "test"
        members, seen_sha, dupes, skipped = [], {}, 0, {"bulk": 0, "none": 0, "no_labels": 0}
        for r in self._images():
            lab = self.labels(r["id"])
            if lab.review in ("bulk", "none"):
                skipped[lab.review] += 1
                continue
            if lab.review == "partial" and not lab.positives:
                skipped["no_labels"] += 1
                continue
            if r["sha256"] in seen_sha:
                dupes += 1            # identical bytes are one sample, kept with the first copy
                continue
            seen_sha[r["sha256"]] = r["id"]
            g = self._group(r, block_days)
            members.append((r, lab, g))
        groups = sorted({g for _, _, g in members})
        split_of = {}
        for g in groups:
            if g in frozen:
                split_of[g] = "test"
                continue
            b = self._bucket(seed, f"{g[0]}|{g[1]}")
            split_of[g] = "train" if b < ratios[0] else ("val" if b < ratios[0] + ratios[1] else "test")
        cond = {r[0]: r[1] for r in self.con.execute("SELECT source_version_id, condition FROM image_condition")}
        stats: dict = {"images": len(members), "groups": len(groups), "duplicates_dropped": dupes, "skipped": skipped,
                       "splits": {s: {"images": 0, "groups": 0, "objects": 0, "full": 0, "partial": 0} for s in SPLITS},
                       "by_condition": {}, "by_view": {}}
        for g in groups:
            stats["splits"][split_of[g]]["groups"] += 1
        self.con.execute("BEGIN")
        try:
            did = self.con.execute(
                "INSERT INTO training_dataset(name,task,params,stats,frozen_test_from,author,created_at) VALUES (?,?,?,?,?,?,?)",
                (name.strip(), "vehicle_detection", json.dumps({"ratios": list(ratios), "block_days": block_days, "seed": seed,
                                                                "classes": list(VEHICLE_CLASSES)}),
                 "{}", prev[0] if prev else None, author, utcnow())).lastrowid
            for r, lab, g in members:
                sp = split_of[g]
                st = stats["splits"][sp]
                st["images"] += 1
                st["objects"] += len(lab.positives)
                st[lab.review] += 1
                c = cond.get(r["id"]) or "unknown"
                stats["by_condition"].setdefault(c, {s: 0 for s in SPLITS})[sp] += 1
                stats["by_view"].setdefault(g[0], {s: 0 for s in SPLITS})[sp] += 1
                self.con.execute(
                    "INSERT INTO dataset_member(dataset_id,source_version_id,sha256,view_key,block,split,review,condition,labels,"
                    "last_revision_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (did, r["id"], r["sha256"], g[0], g[1], sp, lab.review, cond.get(r["id"]),
                     json.dumps({"positives": lab.positives, "ignore_boxes": lab.ignore_boxes}), lab.last_revision_id))
            self.con.execute("UPDATE training_dataset SET stats=? WHERE id=?", (json.dumps(stats), did))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return self.dataset(did)

    def datasets(self) -> list[dict]:
        return [self._ds_row(r) for r in self.con.execute("SELECT * FROM training_dataset ORDER BY id DESC")]

    @staticmethod
    def _ds_row(r) -> dict:
        return {"id": r["id"], "name": r["name"], "task": r["task"], "params": json.loads(r["params"]),
                "stats": json.loads(r["stats"]), "frozen_test_from": r["frozen_test_from"], "author": r["author"],
                "created_at": r["created_at"]}

    def dataset(self, dataset_id: int) -> dict:
        r = self.con.execute("SELECT * FROM training_dataset WHERE id=?", (dataset_id,)).fetchone()
        if r is None:
            raise LearningError("No such dataset version.")
        return self._ds_row(r) | {"leakage": self.leakage(dataset_id)}

    def leakage(self, dataset_id: int) -> list[str]:
        """Problems that would let test images leak into training. An empty list means none found."""
        issues = []
        for r in self.con.execute(
                "SELECT view_key, block, GROUP_CONCAT(DISTINCT split) s FROM dataset_member WHERE dataset_id=? "
                "GROUP BY view_key, block HAVING COUNT(DISTINCT split) > 1", (dataset_id,)):
            issues.append(f"{r['view_key']} on {r['block']} is in several splits ({r['s']})")
        for r in self.con.execute(
                "SELECT sha256, COUNT(*) n FROM dataset_member WHERE dataset_id=? GROUP BY sha256 HAVING n > 1", (dataset_id,)):
            issues.append(f"identical image bytes appear {r['n']} times ({r['sha256'][:12]}…)")
        prev_test = self.con.execute(
            "SELECT DISTINCT d.view_key, d.block FROM dataset_member d WHERE d.split='test' AND d.dataset_id<?", (dataset_id,))
        for r in prev_test:
            bad = self.con.execute("SELECT COUNT(*) FROM dataset_member WHERE dataset_id=? AND view_key=? AND block=? "
                                   "AND split!='test'", (dataset_id, r["view_key"], r["block"])).fetchone()[0]
            if bad:
                issues.append(f"{r['view_key']} on {r['block']} was in an earlier test set but is now used for {bad} "
                              "training or validation image(s)")
        return issues

    def members(self, dataset_id: int, split: str | None = None) -> list[dict]:
        q = "SELECT d.*, f.rel_path FROM dataset_member d JOIN source_version v ON v.id=d.source_version_id " \
            "JOIN source_file f ON f.id=v.source_file_id WHERE d.dataset_id=?"
        args: list = [dataset_id]
        if split:
            q += " AND d.split=?"
            args.append(split)
        return [dict(r) | {"labels": json.loads(r["labels"])} for r in self.con.execute(q + " ORDER BY d.source_version_id", args)]

    def export_dataset(self, dataset_id: int, out_dir: Path | None = None) -> Path:
        """Write images and COCO-style annotations per split for training on another machine. Never overwrites."""
        ds = self.dataset(dataset_id)
        if ds["leakage"]:
            raise LearningError("This dataset version has leakage problems; it is not exported: " + "; ".join(ds["leakage"]))
        base = Path(out_dir or self.m.config().output_dir) / "datasets"
        base.mkdir(parents=True, exist_ok=True)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{self.m.config().name}_ds{dataset_id}_{ds['name']}")
        final, n = base / stem, 2
        while final.exists() or final.with_name(final.name + ".partial").exists():
            final, n = base / f"{stem}_{n}", n + 1
        tmp = final.with_name(final.name + ".partial")
        (tmp / "images").mkdir(parents=True)
        cats = [{"id": i + 1, "name": c} for i, c in enumerate(VEHICLE_CLASSES)]
        cid = {c["name"]: c["id"] for c in cats}
        files = []
        for split in SPLITS:
            images, anns = [], []
            for mrow in self.members(dataset_id, split):
                vid = mrow["source_version_id"]
                src = self.views._path(vid)
                ext = src.suffix.lower()
                name = f"{vid}_{mrow['sha256'][:12]}{ext}"
                shutil.copy2(src, tmp / "images" / name)
                img = read_image(src)
                view = self.views.current_view(vid)
                regions = RegionSet.load(self.con, view) if view is not None else RegionSet([])
                images.append({"id": vid, "file_name": f"images/{name}", "width": int(img.shape[1]), "height": int(img.shape[0]),
                               "sha256": mrow["sha256"], "source_file": mrow["rel_path"], "view": mrow["view_key"],
                               "date_block": mrow["block"], "condition": mrow["condition"], "review": mrow["review"],
                               "unlabelled_area": "background inside included regions" if mrow["review"] == "full" else "ignore",
                               "include_regions": [r.polygon for r in regions.include],
                               "exclude_regions": [r.polygon for r in regions.exclude],
                               "ignore_boxes": mrow["labels"]["ignore_boxes"]})
                for k, p in enumerate(mrow["labels"]["positives"]):
                    x1, y1, x2, y2 = p["box"]
                    anns.append({"id": vid * 1000 + k, "image_id": vid, "category_id": cid.get(p["cls"], cid["car"]),
                                 "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                 "label_source": p["source"]})
            f = tmp / f"annotations_{split}.json"
            f.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}, ensure_ascii=False),
                         encoding="utf-8")
            files.append(f.name)
        manifest = {"mission_id": self.m.id, "dataset_id": dataset_id, "name": ds["name"], "params": ds["params"],
                    "stats": ds["stats"], "created_at": ds["created_at"], "exported_at": utcnow(), "files": files,
                    "rules": self.eligibility()["rules"]}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if final.exists():
            raise LearningError(f"{final.name} appeared while writing; nothing was overwritten.")
        tmp.rename(final)
        return final

    # ---------- evaluation ----------
    def evaluate(self, dataset_id: int, detector, author: str, split: str = "test", iou: float = 0.5,
                 control: RunControl | None = None, progress: Callable[[int, int], None] | None = None) -> dict:
        if split not in SPLITS:
            raise LearningError(f"Split must be one of {SPLITS}.")
        self.dataset(dataset_id)
        mem = self.members(dataset_id, split)
        if not mem:
            raise LearningError(f"The {split} split of this dataset version is empty.")
        control = control or RunControl()
        desc = detector.describe()
        eid = self.con.execute("INSERT INTO evaluation(dataset_id,split,model,state,author,created_at) VALUES (?,?,?,?,?,?)",
                               (dataset_id, split, json.dumps(desc), "running", author, utcnow())).lastrowid
        per_image = []
        try:
            for n, mrow in enumerate(mem, 1):
                control.checkpoint()
                vid = mrow["source_version_id"]
                img = read_image(self.views._path(vid))
                dets = [(list(map(float, d.box)), d.cls, float(d.score)) for d in detector.detect(img)]
                view = self.views.current_view(vid)
                regions = RegionSet.load(self.con, view) if view is not None and self.views.regions_applicable(view) else RegionSet([])
                per_image.append(self._score_image(mrow, dets, regions, iou))
                if progress:
                    progress(n, len(mem))
        except Exception as exc:
            from .process import RunCancelled
            state = "cancelled" if isinstance(exc, RunCancelled) else "failed"
            self.con.execute("UPDATE evaluation SET state=?, error=?, finished_at=? WHERE id=?",
                             (state, f"{type(exc).__name__}: {exc}"[:1000], utcnow(), eid))
            raise
        report = {"iou": iou, "split": split, "images": len(per_image),
                  "overall": _summarise(per_image),
                  "by_condition": {k: _summarise([p for p in per_image if p["condition"] == k])
                                   for k in sorted({p["condition"] for p in per_image})},
                  "by_view": {k: _summarise([p for p in per_image if p["view"] == k])
                              for k in sorted({p["view"] for p in per_image})},
                  "notes": ["Precision, AP and count error use fully reviewed images only; partial reviews add recall "
                            "for their corrected objects.",
                            "AP is limited by the detector's own score threshold: boxes below it are never seen."]}
        self.con.execute("UPDATE evaluation SET state='completed', report=?, finished_at=? WHERE id=?",
                         (json.dumps(report), utcnow(), eid))
        return self.evaluation(eid)

    @staticmethod
    def _score_image(mrow: dict, dets, regions: RegionSet, iou: float) -> dict:
        lab = mrow["labels"]
        full = mrow["review"] == "full"
        refs = [p for p in lab["positives"] if regions.contains_box(tuple(p["box"]))]
        preds = sorted([d for d in dets if regions.contains_box(tuple(d[0]))], key=lambda d: -d[2])
        used, scored, tp, cls_ok = set(), [], 0, 0
        for box, cls, score in preds:
            best, bi = 0.0, None
            for i, r in enumerate(refs):
                if i in used:
                    continue
                v = _iou(box, r["box"])
                if v > best:
                    best, bi = v, i
            if bi is not None and best >= iou:
                used.add(bi)
                tp += 1
                cls_ok += int(cls == refs[bi]["cls"])
                scored.append((score, 1))
                continue
            if any(_iou(box, ig) >= iou or (ig[0] <= (box[0] + box[2]) / 2 <= ig[2] and ig[1] <= (box[1] + box[3]) / 2 <= ig[3])
                   for ig in lab["ignore_boxes"]):
                continue                      # an unresolved night candidate: neither right nor wrong
            if full:
                scored.append((score, 0))     # a false positive only counts where the whole image was reviewed
        return {"id": mrow["source_version_id"], "full": full, "condition": mrow["condition"] or "unknown",
                "view": mrow["view_key"], "refs": len(refs), "tp": tp, "fn": len(refs) - tp,
                "fp": sum(1 for _, t in scored if t == 0), "cls_ok": cls_ok, "scored": scored,
                "pred_count": len([p for p in preds]), "count_error": (len(preds) - len(refs)) if full else None}

    def evaluations(self, dataset_id: int | None = None) -> list[dict]:
        q, a = "SELECT * FROM evaluation", []
        if dataset_id is not None:
            q, a = q + " WHERE dataset_id=?", [dataset_id]
        return [self._ev_row(r) for r in self.con.execute(q + " ORDER BY id DESC", a)]

    @staticmethod
    def _ev_row(r) -> dict:
        return {"id": r["id"], "dataset_id": r["dataset_id"], "split": r["split"], "model": json.loads(r["model"]),
                "state": r["state"], "report": json.loads(r["report"]) if r["report"] else None, "error": r["error"],
                "author": r["author"], "created_at": r["created_at"], "finished_at": r["finished_at"]}

    def evaluation(self, eid: int) -> dict:
        r = self.con.execute("SELECT * FROM evaluation WHERE id=?", (eid,)).fetchone()
        if r is None:
            raise LearningError("No such evaluation.")
        return self._ev_row(r)


def _summarise(items: list[dict]) -> dict:
    full = [p for p in items if p["full"]]
    tp_all = sum(p["tp"] for p in items)
    refs_all = sum(p["refs"] for p in items)
    tp_f, fp_f, refs_f = sum(p["tp"] for p in full), sum(p["fp"] for p in full), sum(p["refs"] for p in full)
    precision = tp_f / (tp_f + fp_f) if (tp_f + fp_f) else None
    recall = tp_all / refs_all if refs_all else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if precision == 0 or recall == 0 else None)
    scored = sorted((s for p in full for s in p["scored"]), key=lambda x: -x[0])
    ap = None
    if refs_f:
        tps = fps = 0
        pts = []
        for _, t in scored:
            tps += t
            fps += 1 - t
            pts.append((tps / refs_f, tps / (tps + fps)))
        # All-point interpolated AP.
        ap, prev_r = 0.0, 0.0
        for i, (r, _) in enumerate(pts):
            pmax = max(p for _, p in pts[i:])
            ap += (r - prev_r) * pmax
            prev_r = r
    errs = [p["count_error"] for p in full if p["count_error"] is not None]
    return {"images": len(items), "full_images": len(full), "partial_images": len(items) - len(full),
            "references": refs_all, "true_positives": tp_all, "false_positives_full": fp_f,
            "precision": _r(precision), "recall": _r(recall), "f1": _r(f1), "ap50": _r(ap),
            "class_agreement": _r(sum(p["cls_ok"] for p in items) / tp_all) if tp_all else None,
            "count_mae": _r(sum(abs(e) for e in errs) / len(errs)) if errs else None,
            "count_bias": _r(sum(errs) / len(errs)) if errs else None}


def _r(v):
    return None if v is None else round(float(v), 4)


# ------------------------------------------------------------------ production pointer (control.db)

class ModelRegistry:
    """Production model per task. Only an operator's promotion or rollback moves it."""

    def __init__(self, data_root: Path | str):
        self.con = connect(Path(data_root) / "control.db")
        self.con.executescript(CONTROL_SCHEMA)

    def history(self, task: str) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM model_promotion WHERE task=? ORDER BY id DESC", (task,))]

    def promote(self, task: str, name: str, version: str, author: str, mission_id: str | None = None,
                evaluation_id: int | None = None, note: str | None = None, models_root: Path | None = None) -> dict:
        from . import models
        pack = models.find(name, version, models_root)       # verifies the checksum
        if pack.card.get("task") != task:
            raise LearningError(f"{name} {version} is a {pack.card.get('task')} model, not {task}.")
        if evaluation_id is None or mission_id is None:
            raise LearningError("Promote a model only with the evaluation that supports it.")
        self.con.execute(
            "INSERT INTO model_promotion(task,name,version,sha256,action,mission_id,evaluation_id,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (task, name, version, pack.sha256, "promote", mission_id, evaluation_id, author,
                                             note, utcnow()))
        return self.production(task)

    def _stack(self, task: str) -> list[dict]:
        """Replay the history: a promotion pushes a model, a rollback pops back to the one before it."""
        stack: list[dict] = []
        for r in reversed(self.history(task)):
            if r["action"] == "promote":
                stack.append(r)
            elif stack:
                stack.pop()
        return stack

    def production(self, task: str) -> dict | None:
        st = self._stack(task)
        return st[-1] if st else None

    def rollback(self, task: str, author: str, note: str | None = None) -> dict | None:
        st = self._stack(task)
        if not st:
            raise LearningError("Nothing has been promoted for this task; the built-in default is in use.")
        cur = st[-1]
        back = st[-2] if len(st) > 1 else None
        self.con.execute(
            "INSERT INTO model_promotion(task,name,version,sha256,action,mission_id,evaluation_id,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", (task, cur["name"], cur["version"], cur["sha256"], "rollback",
                                             cur["mission_id"], cur["evaluation_id"], author, note, utcnow()))
        return back            # None: the built-in default is back in use
