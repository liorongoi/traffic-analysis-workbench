"""Sort still images into indoor and outdoor (the indoor_outdoor mission mode).

Two groups, as the operator defines them: indoor is a camera inside a location (a house, a store, a restaurant, a
mosque) that sees only the inside; outdoor is a camera outside a building, or a camera inside that sees the outside
(through a window or a door). The second kind of outdoor is marked apart: "seen from inside" (label "window").

Each image is turned into a CLIP ViT-B/32 image embedding (512 numbers; the model pack clip-vit-b32-image, MIT) and
kept in the mission's database. Its label is decided against three text embeddings (scene_prompts.json), so no
training is needed to start.

The operator can move any image to another label. Those decisions are append-only (scene_decision): the newest
for an image wins, and "back to LanePulse's choice" is a decision too. A decision always beats the machine.

Learning: the images people placed, in every sorting mission on this workstation, become a memory. An image nobody
checked whose nearest images (by CLIP similarity) include ones a person placed takes the side most of those neighbours
are on, so a move carries over to similar shots, here and in later bundles. Each memory is checked by cross-validation
on the placed images and used only when it gets at least as many of them right as LanePulse did alone. Every version
is kept in the data folder (scene_learner/), and going back to sorting without it is one press. Learning re-reads no
image: it takes a second.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .store import DataRoot, Mission, utcnow

MODE = "indoor_outdoor"
PACK_NAME = "clip-vit-b32-image"
TASK = "scene_embedding"
LABELS = ("indoor", "window", "outdoor")     # window: outdoor, seen from inside
GROUP = {"indoor": "indoor", "window": "outdoor", "outdoor": "outdoor"}
TITLE = {"indoor": "indoor", "window": "outdoor, seen from inside", "outdoor": "outdoor"}
FOLDER = {"indoor": "indoor", "window": "outdoor/seen from inside", "outdoor": "outdoor"}
UNSURE_BELOW = 0.8        # chance of the chosen label under which an image is shown as "not sure"
_PROMPTS = Path(__file__).with_name("scene_prompts.json")
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_result (       -- one row per image: its CLIP embedding (machine; immutable)
  source_version_id INTEGER PRIMARY KEY,
  run_item_id INTEGER,
  embedding BLOB NOT NULL,                      -- 512 float16, unit length
  model TEXT NOT NULL,                          -- pack name/version
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scene_decision (     -- the operator's choices, append-only; newest per image wins
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL,
  label TEXT,                                   -- indoor | window | outdoor | NULL (back to LanePulse's choice)
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_scene_decision ON scene_decision(source_version_id, id);
"""


def ensure_schema(con) -> None:
    con.executescript(SCHEMA)


def prompts() -> dict:
    return json.loads(_PROMPTS.read_text(encoding="utf-8"))


def base_scores(x: np.ndarray) -> np.ndarray:
    """Scores of embeddings x (n x 512) for each label in LABELS order (n x 3); the highest wins."""
    p = prompts()
    by = {c["label"]: c for c in p["classes"]}
    c = np.stack([np.asarray(by[k]["embedding"], np.float64) for k in LABELS])
    b = np.asarray([float(by[k].get("bias", 0.0)) for k in LABELS])
    return p["logit_scale"] * (np.atleast_2d(x) @ c.T) + b


def base_labels(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LanePulse's own label index for each embedding, and how sure it is (softmax of the scores, 1/3 to 1)."""
    z = base_scores(x)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    pr = e / e.sum(axis=1, keepdims=True)
    return pr.argmax(axis=1), pr.max(axis=1)


def available(root: Path | None = None) -> dict:
    from . import models
    try:
        cards = [c for c in models.installed(TASK, root) if c["name"] == PACK_NAME]
    except OSError:
        cards = []
    if not cards:
        return {"installed": False, "reason": "The indoor/outdoor model (clip-vit-b32-image) is not installed. Copy the "
                                              "model packs from the staging machine (scripts/fetch_models.py)."}
    return {"installed": True, "reason": None, "version": cards[-1]["version"], "license": cards[-1]["license"]}


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """CLIP's own preparation: shorter side to 224 (bicubic), centre crop, CLIP mean and std. Returns 3x224x224."""
    import cv2
    h, w = img_bgr.shape[:2]
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    s = 224 / min(h, w)
    nw, nh = max(224, round(w * s)), max(224, round(h * s))
    small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x, y = (nw - 224) // 2, (nh - 224) // 2
    rgb = small[y:y + 224, x:x + 224, ::-1].astype(np.float32) / 255.0
    return ((rgb - _MEAN) / _STD).transpose(2, 0, 1)


class Embedder:
    """The CLIP image encoder on the best backend (runtime.make_session)."""

    def __init__(self, root: Path | None = None, setting: str | None = None):
        from . import models
        from .runtime import make_session
        self.pack = models.find(PACK_NAME, root=root)
        self.session, self.runtime = make_session(self.pack.weights, setting)
        self.input = self.session.get_inputs()[0].name

    @property
    def model(self) -> str:
        return f"{self.pack.name}/{self.pack.version}"

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        # One image per call: the 8-bit model scales its numbers over the whole batch, so an image's embedding would
        # otherwise depend on the images beside it. Batches are no faster on the CPU here.
        e = np.concatenate([self.session.run(None, {self.input: preprocess(i)[None]})[0] for i in images]).astype(np.float64)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)


# ---------- what LanePulse learned (one memory per workstation, shared by every sorting mission) ----------
#
# Learning is a memory of the images people placed: an image nobody checked whose nearest images (CLIP similarity at
# least NEAR, the K nearest) include one a person placed takes the label those neighbours vote for, closer ones
# counting for more and LanePulse's own label for the image counting a little (SELF). The neighbours also include
# images nobody checked, at LanePulse's own label, so a single move does not pull a whole crowd over. Measured in
# docs/decisions/indoor-outdoor.md.

NEAR = 0.8
K = 5
SELF = 0.1
MAX_UNCHECKED = 3000      # images nobody checked kept in the memory, in all


def _learner_dir(data_root: Path) -> Path:
    return Path(data_root) / "scene_learner"


def learner_state(data_root: Path) -> dict:
    try:
        return json.loads((_learner_dir(data_root) / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"current": None, "versions": []}


def _save_state(data_root: Path, state: dict) -> None:
    p = _learner_dir(data_root) / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, p)


@dataclass
class Memory:
    emb: np.ndarray          # n x 512
    label: np.ndarray        # n, index into LABELS
    human: np.ndarray        # n, bool: placed by a person (otherwise LanePulse's own label, unchecked)

    def vote(self, x: np.ndarray, base: np.ndarray, skip_self: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Labels for embeddings x (base: LanePulse's own). Returns (labels, changed by the memory)."""
        out = base.copy()
        hit = np.zeros(len(x), bool)
        if not len(self.emb) or not self.human.any() or not len(x):
            return out, hit
        for s in range(0, len(x), 512):
            sim = x[s:s + 512] @ self.emb.T
            if skip_self:      # the image itself is in the memory: it is not its own neighbour
                sim[sim > 0.9995] = -1
            top = np.argsort(-sim, axis=1)[:, :K]
            for i, row in enumerate(top):
                nb = row[sim[i, row] >= NEAR]
                if not len(nb) or not self.human[nb].any():
                    continue
                # Closer neighbours count for more.
                v = np.bincount(self.label[nb], weights=sim[i, nb] - NEAR + 0.05, minlength=len(LABELS))
                v[base[s + i]] += SELF
                if (v == v.max()).sum() == 1:
                    out[s + i] = int(v.argmax())
        hit = out != base
        return out, hit


def load_memory(data_root: Path) -> tuple[Memory | None, dict | None]:
    st = learner_state(data_root)
    v = next((x for x in st["versions"] if x["version"] == st.get("current")), None)
    if v is None:
        return None, None
    try:
        z = np.load(_learner_dir(data_root) / v["file"])
        return Memory(z["emb"].astype(np.float64), z["label"], z["human"]), v
    except (OSError, KeyError, ValueError):
        return None, None


def _everything(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Embeddings of every sorting mission here: (embeddings, the person's label index or -1, unused, missions with
    decisions)."""
    dr = DataRoot(data_root)
    xs, ys, missions = [], [], 0
    for row in dr.missions():
        if row["mode"] != MODE or row["state"] != "active":
            continue
        m = dr.open_mission(row["id"])
        ensure_schema(m.con)
        found = False
        for r in m.con.execute(
                "SELECT s.embedding, d.label FROM scene_result s LEFT JOIN scene_decision d ON d.id=(SELECT MAX(id) "
                "FROM scene_decision WHERE source_version_id=s.source_version_id)"):
            xs.append(np.frombuffer(r[0], np.float16).astype(np.float64))
            ys.append(-1 if r[1] not in LABELS else LABELS.index(r[1]))
            found = found or r[1] is not None
        missions += found
        m.con.close()
    if not xs:
        return np.zeros((0, 512)), np.zeros(0, int), np.zeros(0), 0
    return np.stack(xs), np.asarray(ys), np.zeros(0), missions


def learn(data_root: Path, author: str) -> dict:
    """Make a new memory from every image people placed on this workstation; keep it only if it does not do worse."""
    x, y, _, n_missions = _everything(data_root)
    placed = y >= 0
    per = {k: int((y == i).sum()) for i, k in enumerate(LABELS)}
    if placed.sum() < 2:
        return {"kept": False, "decisions": int(placed.sum()), **per,
                "reason": "Move or mark at least two images first; LanePulse learns from the images you placed."}
    base = base_labels(x)[0] if len(x) else np.zeros(0, int)
    hx, hy = x[placed], y[placed]
    rest = np.where(~placed)[0]
    rest = np.random.default_rng(0).permutation(rest)[:MAX_UNCHECKED]
    ux, uy = x[rest], base[rest]
    # Cross-validation: each placed image is sorted by a memory that does not hold it.
    order = np.random.default_rng(0).permutation(len(hy))
    learned_ok = 0
    for f in np.array_split(order, min(5, len(hy))):
        tr = np.setdiff1d(order, f)
        mem = Memory(np.concatenate([hx[tr], ux]), np.concatenate([hy[tr], uy]),
                     np.concatenate([np.ones(len(tr), bool), np.zeros(len(ux), bool)]))
        got, _ = mem.vote(hx[f], base[placed][f], skip_self=False)
        learned_ok += int((got == hy[f]).sum())
    prompt_ok = int((base[placed] == hy).sum())
    mem = Memory(np.concatenate([hx, ux]), np.concatenate([hy, uy]),
                 np.concatenate([np.ones(len(hy), bool), np.zeros(len(ux), bool)]))
    changed = int(mem.vote(x[~placed], base[~placed])[1].sum())
    st = learner_state(data_root)
    version = max([v["version"] for v in st["versions"]], default=0) + 1
    kept = learned_ok >= prompt_ok
    entry = {"version": version, "file": f"memory-{version}.npz", "created_at": utcnow(), "author": author,
             "decisions": int(placed.sum()), **per, "missions": n_missions,
             "right_before": prompt_ok, "right_after": learned_ok, "kept": kept, "unchecked_changed": changed}
    d = _learner_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / (entry["file"] + ".tmp"), "wb") as fh:
        np.savez_compressed(fh, emb=mem.emb.astype(np.float16), label=mem.label.astype(np.int8), human=mem.human)
    os.replace(d / (entry["file"] + ".tmp"), d / entry["file"])
    st["versions"].append(entry)
    if kept:
        st["current"] = version
    _save_state(data_root, st)
    return entry | {"reason": None if kept else
                    "What LanePulse learned did not get more of your images right than before, so it is not used."}


def reset_learning(data_root: Path) -> dict:
    st = learner_state(data_root)
    st["current"] = None
    _save_state(data_root, st)
    return {"current": None}


def learner_summary(data_root: Path) -> dict:
    st = learner_state(data_root)
    v = next((x for x in st["versions"] if x["version"] == st.get("current")), None)
    return {"in_use": None if v is None else {k: v.get(k, 0) for k in ("version", "created_at", "decisions", *LABELS,
                                                                        "missions", "right_before", "right_after")},
            "versions": len(st["versions"])}


# ---------- one mission ----------

@dataclass
class SceneItem:
    version_id: int
    file: str
    label: str            # the final label: the operator's when they chose, otherwise LanePulse's
    group: str            # indoor | outdoor (window is outdoor, seen from inside)
    machine: str          # LanePulse's label (with what it learned)
    sure: float           # how sure the prompts are of their own label, 1/3 to 1
    decided: str | None   # the operator's label, when they chose one
    decided_by: str | None
    learned: bool         # LanePulse's label comes from images people placed, not from the prompts


class SceneService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        ensure_schema(self.con)

    def items(self) -> list[SceneItem]:
        rows = self.con.execute(
            "SELECT s.source_version_id, s.embedding, f.rel_path, d.label, d.author FROM scene_result s "
            "JOIN source_version v ON v.id=s.source_version_id JOIN source_file f ON f.id=v.source_file_id "
            "LEFT JOIN scene_decision d ON d.id=(SELECT MAX(id) FROM scene_decision WHERE source_version_id="
            "s.source_version_id) WHERE v.superseded_by IS NULL ORDER BY f.rel_path").fetchall()
        if not rows:
            return []
        x = np.stack([np.frombuffer(r[1], np.float16).astype(np.float64) for r in rows])
        base, sure = base_labels(x)
        mem, _ = load_memory(self.m.root.path)
        lab, learned = mem.vote(x, base) if mem is not None else (base, np.zeros(len(x), bool))
        out = []
        for r, li, lr, pi in zip(rows, lab, learned, sure):
            machine = LABELS[int(li)]
            label = r[3] if r[3] in LABELS else machine
            out.append(SceneItem(r[0], r[2], label, GROUP[label], machine, float(pi), r[3],
                                 r[4] if r[3] else None, bool(lr)))
        return out

    def decide(self, version_ids: list[int], label: str | None, author: str) -> int:
        if label is not None and label not in LABELS:
            raise ValueError("Choose indoor, outdoor, or outdoor seen from inside (window).")
        known = {r[0] for r in self.con.execute("SELECT source_version_id FROM scene_result")}
        bad = [v for v in version_ids if v not in known]
        if bad:
            raise KeyError(f"Image {bad[0]} has not been sorted in this mission.")
        now = utcnow()
        self.con.execute("BEGIN")
        self.con.executemany("INSERT INTO scene_decision(source_version_id,label,author,created_at) VALUES (?,?,?,?)",
                             [(v, label, author, now) for v in version_ids])
        self.con.execute("COMMIT")
        return len(version_ids)

    def counts(self, items: list[SceneItem] | None = None) -> dict:
        items = self.items() if items is None else items
        return {"indoor": sum(i.group == "indoor" for i in items), "outdoor": sum(i.group == "outdoor" for i in items),
                "window": sum(i.label == "window" for i in items),
                "moved": sum(i.decided is not None and i.decided != i.machine for i in items),
                "confirmed": sum(i.decided is not None and i.decided == i.machine for i in items),
                "unsure": sum(i.decided is None and i.sure < UNSURE_BELOW for i in items)}

    def copy_out(self, dest: Path, progress: Callable[[int, int], None] | None = None) -> dict:
        """Copy every sorted image into dest/indoor and dest/outdoor, the outdoor ones seen from inside in
        dest/outdoor/seen from inside (keeping subfolders), with a list of them. The source files are only read."""
        import csv
        from .views import ViewService
        items = self.items()
        vs = ViewService(self.m)
        dest = Path(dest)
        for lab in LABELS:
            (dest / FOLDER[lab]).mkdir(parents=True, exist_ok=True)
        copied = missing = 0
        listing = dest / "sorting.csv"
        with open(listing.with_suffix(".tmp"), "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "group", "seen_from_inside", "decided_by", "lanepulse_choice", "lanepulse_sure_pct"])
            for n, it in enumerate(items, 1):
                src = vs._path(it.version_id)
                target = dest / FOLDER[it.label] / Path(it.file)
                for lab in LABELS:      # moved since the last copy
                    other = dest / FOLDER[lab] / Path(it.file)
                    if lab != it.label and other.is_file():
                        other.unlink()
                if src.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not (target.is_file() and target.stat().st_size == src.stat().st_size):
                        shutil.copy2(src, target)
                    copied += 1
                else:
                    missing += 1
                wr.writerow([it.file, it.group, "yes" if it.label == "window" else "", it.decided_by or "LanePulse",
                             TITLE[it.machine], round(100 * it.sure)])
                if progress:
                    progress(n, len(items))
        os.replace(listing.with_suffix(".tmp"), listing)
        c = self.counts(items)
        return {"folder": str(dest), "copied": copied, "missing": missing, "indoor": c["indoor"], "outdoor": c["outdoor"],
                "window": c["window"], "list": str(listing)}


class SceneProcessor:
    """Processing runs of a sorting mission: one embedding per image, committed one image at a time (resumable)."""

    BATCH = 8

    def __init__(self, mission: Mission, embedder_factory: Callable[[], Embedder], control=None):
        from .process import RunControl
        self.m = mission
        self.con = mission.con
        self.factory = embedder_factory
        self.control = control or RunControl()
        ensure_schema(self.con)

    def pending_runs(self) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT id FROM run WHERE state IN ('queued','running') OR id IN "
            "(SELECT run_id FROM run_item WHERE processing_state IN ('queued','running')) ORDER BY id")]

    def process_all(self, progress: Callable[[int, int], None] | None = None) -> list[dict]:
        runs = self.pending_runs()
        if not runs:
            return []
        self.control.stage = "loading the indoor/outdoor model"
        emb = self.factory()
        return [self.process_run(r, emb, progress) for r in runs]

    def process_run(self, run_id: int, emb: Embedder, progress=None) -> dict:
        from .process import MAX_ATTEMPTS, RunCancelled
        from .views import ViewService, read_image
        rep = {"run_id": run_id, "done": 0, "failed": 0, "errors": []}
        self.con.execute("UPDATE run SET state='running', started_at=COALESCE(started_at,?), model_versions=? WHERE id=?",
                         (utcnow(), json.dumps({TASK: {"name": emb.pack.name, "version": emb.pack.version,
                                                       "sha256": emb.pack.sha256}}), run_id))
        items = self.con.execute(
            "SELECT i.*, f.rel_path FROM run_item i JOIN source_version v ON v.id=i.source_version_id "
            "JOIN source_file f ON f.id=v.source_file_id WHERE i.run_id=? AND i.processing_state IN ('queued','running') "
            "ORDER BY i.id", (run_id,)).fetchall()
        vs = ViewService(self.m)
        self.control.stage = "sorting indoor and outdoor"
        total = len(items)
        n = 0
        for start in range(0, total, self.BATCH):
            try:
                self.control.checkpoint()
            except RunCancelled:
                self.con.execute("UPDATE run_item SET processing_state='cancelled' WHERE run_id=? AND "
                                 "processing_state IN ('queued','running')", (run_id,))
                self.con.execute("UPDATE run SET state='cancelled', finished_at=? WHERE id=?", (utcnow(), run_id))
                self.control.stage = "cancelled"
                return rep
            batch, imgs = [], []
            for it in items[start:start + self.BATCH]:
                n += 1
                if it["attempts"] >= MAX_ATTEMPTS:
                    self.con.execute("UPDATE run_item SET processing_state='failed', error_code='too_many_attempts' "
                                     "WHERE id=?", (it["id"],))
                    rep["failed"] += 1
                    continue
                self.con.execute("UPDATE run_item SET processing_state='running', stage='sort', attempts=attempts+1 "
                                 "WHERE id=?", (it["id"],))
                try:
                    imgs.append(read_image(vs._path(it["source_version_id"])))
                    batch.append(it)
                except Exception as exc:  # one unreadable image must not stop the run
                    self._failed(it, exc, run_id, rep)
            if batch:
                try:
                    e = emb.embed(imgs)
                except Exception as exc:
                    for it in batch:
                        self._failed(it, exc, run_id, rep)
                    continue
                now = utcnow()
                self.con.execute("BEGIN")
                for it, v in zip(batch, e):
                    self.con.execute("INSERT OR REPLACE INTO scene_result(source_version_id,run_item_id,embedding,model,"
                                     "created_at) VALUES (?,?,?,?,?)",
                                     (it["source_version_id"], it["id"], v.astype(np.float16).tobytes(), emb.model, now))
                    self.con.execute("UPDATE run_item SET processing_state='done', stage=NULL, error_code=NULL, "
                                     "error_detail=NULL WHERE id=?", (it["id"],))
                self.con.execute("COMMIT")
                rep["done"] += len(batch)
            if progress:
                for _ in range(len(items[start:start + self.BATCH])):
                    progress(n, total)
        failed = self.con.execute("SELECT COUNT(*) FROM run_item WHERE run_id=? AND processing_state='failed'",
                                  (run_id,)).fetchone()[0]
        self.con.execute("UPDATE run SET state=?, finished_at=? WHERE id=?",
                         ("completed" if not failed else "completed_with_errors", utcnow(), run_id))
        self.m.notify("info" if not failed else "warning", "run_finished",
                      f"Run {run_id} finished: {rep['done']} sorted, {failed} failed.", run_id=run_id)
        self.control.stage = "finished"
        return rep

    def _failed(self, it, exc: Exception, run_id: int, rep: dict) -> None:
        self.con.execute("UPDATE run_item SET processing_state='failed', error_code=?, error_detail=? WHERE id=?",
                         (type(exc).__name__, str(exc)[:500], it["id"]))
        self.m.notify("warning", "processing_failed", f"Could not sort {it['rel_path']}: {exc}", run_id=run_id,
                      source_version_id=it["source_version_id"])
        rep["failed"] += 1
        rep["errors"].append(f"{it['rel_path']}: {exc}")
