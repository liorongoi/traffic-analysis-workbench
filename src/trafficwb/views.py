"""Camera-view grouping, assignment history and view-change detection.

A camera location can have several views (shooting angles). Views are proposed
automatically and confirmed or corrected by the operator. Lighting alone never
makes a new view: clusters found separately in colour and monochrome (night)
images are joined when the evidence supports it, and a join supported only by
structure (not by geometric matching) is marked for the operator to confirm.

See docs/design/camera-views.md for the method and thresholds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .store import Mission, utcnow

SIG_ALGO = "hog-u8-320x180-c10-v1"
SIG_SIZE = (320, 180)
CELL = 10
BINS = 8
MONO_SATURATION = 5.0  # mean HSV saturation (0-255) below which an image is treated as monochrome

# Proposed starting values; to be tuned on target-camera samples (see design doc).
TAU_MEMBER = 0.70  # member-to-centroid similarity to join a cluster within one condition
TAU_LINK = 0.60    # colour/monochrome centroid similarity to propose the same view
GEO_MIN_INLIERS = 40
GEO_MAX_SHIFT_FRAC = 0.0125  # 8 px at 640 px width
DRIFT_TOLERANCE_PX = 12.0    # in original pixels; operator-adjustable per view
DRIFT_CONSECUTIVE = 2

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


@dataclass
class Signature:
    vec: np.ndarray  # float32, unit length, zero mean
    width: int
    height: int
    mean_luma: float
    saturation: float

    @property
    def condition(self) -> str:
        return "monochrome" if self.saturation < MONO_SATURATION else "colour"


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return img


def signature(img_bgr: np.ndarray) -> Signature:
    """Gradient-orientation histogram on a coarse grid.

    Unsigned orientation (0-180 degrees) and local contrast equalisation make it
    largely insensitive to brightness and contrast inversion between day and night.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    s = _clahe.apply(cv2.resize(gray, SIG_SIZE, interpolation=cv2.INTER_AREA))
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    ang = np.mod(ang, np.pi)
    b = np.minimum((ang / np.pi * BINS).astype(np.int32), BINS - 1)
    gw, gh = SIG_SIZE[0] // CELL, SIG_SIZE[1] // CELL
    out = np.zeros((gh, gw, BINS), np.float32)
    for k in range(BINS):
        out[:, :, k] = cv2.resize(mag * (b == k), (gw, gh), interpolation=cv2.INTER_AREA)
    out = np.sqrt(out)
    out /= np.linalg.norm(out, axis=2, keepdims=True) + 1e-6
    v = out.ravel()
    v = v - v.mean()
    v /= np.linalg.norm(v) + 1e-9
    return Signature(v.astype(np.float32), w, h, float(gray.mean()), sat)


# ---------------------------------------------------------------- geometry

_orb = cv2.ORB_create(2000)


def _orb_features(img_bgr: np.ndarray):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scale = 640 / g.shape[1]
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.resize(g, (640, int(round(g.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    kp, des = _orb.detectAndCompute(g, None)
    return kp, des, scale


def match_geometry(a_bgr: np.ndarray, b_bgr: np.ndarray, points: np.ndarray | None = None) -> tuple[int, float | None]:
    """ORB + RANSAC homography from a to b.

    Returns (inliers, max displacement in a's original pixels of `points`, or of
    the frame corners when no points are given). Displacement is None when no
    homography could be estimated.
    """
    ka, da, sa = _orb_features(a_bgr)
    kb, db, sb = _orb_features(b_bgr)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0, None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 8:
        return len(good), None
    p = np.float32([ka[g.queryIdx].pt for g in good]) / sa
    q = np.float32([kb[g.trainIdx].pt for g in good]) / sb
    H, mask = cv2.findHomography(p, q, cv2.RANSAC, 3.0 / sa)
    if H is None:
        return 0, None
    if points is None:
        h, w = a_bgr.shape[:2]
        points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts = np.float32(points).reshape(-1, 1, 2)
    disp = float(np.linalg.norm(cv2.perspectiveTransform(pts, H) - pts, axis=2).max())
    return int(mask.sum()), disp


def geometrically_same(a_bgr: np.ndarray, b_bgr: np.ndarray) -> bool:
    inl, disp = match_geometry(a_bgr, b_bgr)
    return inl >= GEO_MIN_INLIERS and disp is not None and disp <= GEO_MAX_SHIFT_FRAC * a_bgr.shape[1]


# ---------------------------------------------------------------- clustering

def _cluster(vecs: np.ndarray, tau: float) -> list[list[int]]:
    """Leader clustering, then centroid merging, then one reassignment pass. Deterministic for a given order."""
    cents: list[np.ndarray] = []
    groups: list[list[int]] = []
    for i, v in enumerate(vecs):
        if cents:
            sims = np.stack(cents) @ v
            j = int(np.argmax(sims))
            if sims[j] >= tau:
                groups[j].append(i)
                c = vecs[groups[j]].mean(0)
                cents[j] = c / np.linalg.norm(c)
                continue
        cents.append(v.copy())
        groups.append([i])
    merged = True
    while merged and len(cents) > 1:
        merged = False
        C = np.stack(cents)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        if S[a, b] >= tau:
            a, b = min(a, b), max(a, b)
            groups[a] += groups.pop(b)
            cents.pop(b)
            c = vecs[groups[a]].mean(0)
            cents[a] = c / np.linalg.norm(c)
            merged = True
    C = np.stack(cents)
    best = np.argmax(vecs @ C.T, axis=1)
    out = [[i for i in range(len(vecs)) if best[i] == k] for k in range(len(cents))]
    return [g for g in out if g]


def _centroid(vs: np.ndarray) -> np.ndarray:
    c = vs.mean(0)
    return c / (np.linalg.norm(c) + 1e-9)


# ---------------------------------------------------------------- mission-level service

class ViewService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con

    # --- signatures ---
    def _path(self, version_id: int) -> Path:
        r = self.con.execute(
            "SELECT r.path, f.rel_path FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "JOIN source_root r ON r.id=f.root_id WHERE v.id=?", (version_id,)).fetchone()
        return Path(r["path"]) / r["rel_path"]

    def image(self, version_id: int) -> np.ndarray:
        return read_image(self._path(version_id))

    def compute_signatures(self) -> int:
        rows = self.con.execute(
            "SELECT v.id FROM source_version v LEFT JOIN view_signature s ON s.source_version_id=v.id "
            "WHERE s.source_version_id IS NULL AND v.superseded_by IS NULL AND v.integrity='ok'").fetchall()
        n = 0
        for r in rows:
            try:
                sig = signature(self.image(r["id"]))
            except (ValueError, OSError) as exc:
                self.m.notify("warning", "signature_failed", f"Could not read image for view grouping: {exc}",
                              source_version_id=r["id"])
                continue
            self.con.execute(
                "INSERT INTO view_signature VALUES (?,?,?,?,?,?,?,?)",
                (r["id"], SIG_ALGO, sig.vec.astype(np.float16).tobytes(), sig.width, sig.height,
                 sig.mean_luma, sig.saturation, sig.condition))
            n += 1
        return n

    def _sig_rows(self, version_ids: list[int] | None = None):
        q = "SELECT * FROM view_signature"
        rows = self.con.execute(q).fetchall()
        if version_ids is not None:
            want = set(version_ids)
            rows = [r for r in rows if r["source_version_id"] in want]
        return rows

    @staticmethod
    def _vec(row) -> np.ndarray:
        return np.frombuffer(row["vec"], dtype=np.float16).astype(np.float32)

    # --- current state ---
    def current_view(self, version_id: int) -> int | None:
        r = self.con.execute("SELECT view_id FROM view_assignment WHERE source_version_id=? AND superseded_by IS NULL",
                             (version_id,)).fetchone()
        return r["view_id"] if r else None

    def members(self, view_id: int) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT source_version_id FROM view_assignment WHERE view_id=? AND superseded_by IS NULL ORDER BY source_version_id",
            (view_id,))]

    def views(self) -> list[dict]:
        out = []
        for v in self.con.execute("SELECT * FROM camera_view WHERE merged_into IS NULL ORDER BY id"):
            mem = self.members(v["id"])
            conds = dict(self.con.execute(
                "SELECT s.condition, COUNT(*) FROM view_signature s JOIN view_assignment a ON a.source_version_id=s.source_version_id "
                "WHERE a.view_id=? AND a.superseded_by IS NULL GROUP BY s.condition", (v["id"],)).fetchall())
            out.append({**dict(v), "size": len(mem), "conditions": conds, "representatives": self.representatives(v["id"])})
        return out

    def unassigned(self) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT s.source_version_id FROM view_signature s LEFT JOIN view_assignment a "
            "ON a.source_version_id=s.source_version_id AND a.superseded_by IS NULL "
            "WHERE a.id IS NULL OR a.view_id IS NULL ORDER BY s.source_version_id")]

    def representatives(self, view_id: int, per_condition: int = 2) -> dict[str, list[int]]:
        """Members closest to the view's centroid, per lighting condition."""
        rows = self.con.execute(
            "SELECT s.* FROM view_signature s JOIN view_assignment a ON a.source_version_id=s.source_version_id "
            "WHERE a.view_id=? AND a.superseded_by IS NULL", (view_id,)).fetchall()
        out: dict[str, list[int]] = {}
        for cond in sorted({r["condition"] for r in rows}):
            rs = [r for r in rows if r["condition"] == cond]
            V = np.stack([self._vec(r) for r in rs])
            order = np.argsort(-(V @ _centroid(V)))
            out[cond] = [rs[i]["source_version_id"] for i in order[:per_condition]]
        return out

    def _centroids(self) -> dict[tuple[int, str], np.ndarray]:
        cents: dict[tuple[int, str], list[np.ndarray]] = {}
        for r in self.con.execute(
                "SELECT s.*, a.view_id FROM view_signature s JOIN view_assignment a ON a.source_version_id=s.source_version_id "
                "JOIN camera_view v ON v.id=a.view_id WHERE a.superseded_by IS NULL AND v.merged_into IS NULL"):
            cents.setdefault((r["view_id"], r["condition"]), []).append(self._vec(r))
        return {k: _centroid(np.stack(v)) for k, v in cents.items()}

    # --- writes ---
    def _assign(self, version_ids: list[int], view_id: int | None, method: str, author: str, reason: str,
                scores: dict[int, float] | None = None) -> None:
        now = utcnow()
        for vid in version_ids:
            new = self.con.execute(
                "INSERT INTO view_assignment(source_version_id,view_id,method,score,author,reason,created_at) VALUES (?,?,?,?,?,?,?)",
                (vid, view_id, method, (scores or {}).get(vid), author, reason, now)).lastrowid
            self.con.execute("UPDATE view_assignment SET superseded_by=? WHERE source_version_id=? AND superseded_by IS NULL AND id<>?",
                             (new, vid, new))

    def _new_view(self, name: str | None, width: int, height: int, author: str, note: str | None = None,
                  confirmed: bool = True) -> int:
        n = self.con.execute("SELECT COUNT(*) FROM camera_view").fetchone()[0] + 1
        vid = self.con.execute(
            "INSERT INTO camera_view(name,ref_width,ref_height,link_note,link_confirmed,created_at) VALUES (?,?,?,?,?,?)",
            (name or f"View {n}", width, height, note, int(confirmed), utcnow())).lastrowid
        self._event(vid, "created", note, author)
        return vid

    def _event(self, view_id: int, kind: str, detail, author: str) -> None:
        self.con.execute("INSERT INTO view_event(view_id,kind,detail,author,created_at) VALUES (?,?,?,?,?)",
                         (view_id, kind, json.dumps(detail) if not isinstance(detail, str) and detail is not None else detail,
                          author, utcnow()))

    # --- automatic grouping ---
    def assign_or_propose(self, min_size: int = 3, author: str = "system") -> dict:
        """Assign new images to existing views, cluster the rest into proposed views.

        Never overrides a current assignment; manual decisions are kept.
        """
        self.compute_signatures()
        todo = [r for r in self._sig_rows() if self.current_view(r["source_version_id"]) is None
                and not self.con.execute("SELECT 1 FROM view_assignment WHERE source_version_id=? AND superseded_by IS NULL",
                                         (r["source_version_id"],)).fetchone()]
        stats = {"assigned_existing": 0, "new_views": 0, "linked_by_geometry": 0, "linked_by_structure": 0, "unassigned": 0}
        if not todo:
            return stats
        self.con.execute("BEGIN")
        try:
            # 1. existing views, same condition and resolution
            cents = self._centroids()
            sizes = {v["id"]: (v["ref_width"], v["ref_height"]) for v in self.con.execute("SELECT * FROM camera_view WHERE merged_into IS NULL")}
            rest = []
            for r in todo:
                v = self._vec(r)
                cands = [(float(c @ v), vid) for (vid, cond), c in cents.items()
                         if cond == r["condition"] and sizes.get(vid) == (r["width"], r["height"])]
                best = max(cands, default=(0.0, None))
                if best[0] >= TAU_MEMBER:
                    self._assign([r["source_version_id"]], best[1], "auto", author, "nearest existing view", {r["source_version_id"]: best[0]})
                    stats["assigned_existing"] += 1
                else:
                    rest.append(r)
            # 2. cluster the rest per (condition, resolution)
            clusters: list[dict] = []
            keys = sorted({(r["condition"], r["width"], r["height"]) for r in rest})
            for key in keys:
                rs = [r for r in rest if (r["condition"], r["width"], r["height"]) == key]
                V = np.stack([self._vec(r) for r in rs])
                for g in _cluster(V, TAU_MEMBER):
                    ids = [rs[i]["source_version_id"] for i in g]
                    clusters.append({"cond": key[0], "size": key[1:], "ids": ids, "cent": _centroid(V[g]),
                                     "scores": {rs[i]["source_version_id"]: float(V[i] @ _centroid(V[g])) for i in g}})
            # 3. join clusters showing the same geometry (any condition, same resolution)
            parent = list(range(len(clusters)))

            def find(i):
                while parent[i] != i:
                    i = parent[i]
                return i

            how: dict[int, list[str]] = {}
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    a, b = clusters[i], clusters[j]
                    if find(i) == find(j) or a["size"] != b["size"] or float(a["cent"] @ b["cent"]) < 0.4:
                        continue
                    ra = self._medoid(a)
                    rb = self._medoid(b)
                    if geometrically_same(self.image(ra), self.image(rb)):
                        parent[find(j)] = find(i)
                        how.setdefault(find(i), []).append(f"geometry {ra}~{rb}")
                        stats["linked_by_geometry"] += 1
            # 4. colour <-> monochrome by structure: mutual best, above TAU_LINK -> needs confirmation
            unconfirmed: set[int] = set()
            col = [i for i, c in enumerate(clusters) if c["cond"] == "colour"]
            mono = [i for i, c in enumerate(clusters) if c["cond"] == "monochrome"]
            if col and mono:
                S = np.array([[float(clusters[i]["cent"] @ clusters[j]["cent"]) if clusters[i]["size"] == clusters[j]["size"] else -1
                               for j in mono] for i in col])
                for ci, i in enumerate(col):
                    mj = int(np.argmax(S[ci]))
                    if S[ci, mj] >= TAU_LINK and int(np.argmax(S[:, mj])) == ci and find(i) != find(mono[mj]):
                        root = find(i)
                        parent[find(mono[mj])] = root
                        how.setdefault(root, []).append(f"structure similarity {S[ci, mj]:.2f} between colour and monochrome images")
                        unconfirmed.add(root)
                        stats["linked_by_structure"] += 1
            # 5. write views; small groups stay unassigned
            groups: dict[int, list[int]] = {}
            for i in range(len(clusters)):
                groups.setdefault(find(i), []).append(i)
            for root, idx in groups.items():
                ids = [x for i in idx for x in clusters[i]["ids"]]
                scores = {k: v for i in idx for k, v in clusters[i]["scores"].items()}
                if len(ids) < min_size:
                    self._assign(ids, None, "auto", author, f"group of {len(ids)} below minimum size {min_size}")
                    stats["unassigned"] += len(ids)
                    continue
                w, h = clusters[idx[0]]["size"]
                note = "; ".join(how.get(root, [])) or None
                vid = self._new_view(None, w, h, author, note, confirmed=root not in unconfirmed)
                self._assign(ids, vid, "auto", author, "proposed grouping", scores)
                for i in idx:
                    self._set_reference(vid, clusters[i]["cond"], self._medoid(clusters[i]), author)
                stats["new_views"] += 1
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return stats

    def _medoid(self, cluster: dict) -> int:
        return max(cluster["scores"], key=cluster["scores"].get)

    def _set_reference(self, view_id: int, condition: str, version_id: int, author: str) -> None:
        self.con.execute(
            "INSERT INTO view_reference VALUES (?,?,?,?,?) ON CONFLICT(view_id,condition) DO UPDATE SET "
            "source_version_id=excluded.source_version_id, set_by=excluded.set_by, created_at=excluded.created_at",
            (view_id, condition, version_id, author, utcnow()))

    # --- operator actions (all keep history) ---
    def reassign(self, version_ids: list[int], view_id: int | None, author: str, reason: str = "manual reassignment") -> None:
        self._assign(version_ids, view_id, "manual", author, reason)

    def rename(self, view_id: int, name: str, author: str) -> None:
        old = self.con.execute("SELECT name FROM camera_view WHERE id=?", (view_id,)).fetchone()["name"]
        self.con.execute("UPDATE camera_view SET name=? WHERE id=?", (name, view_id))
        self._event(view_id, "renamed", {"from": old, "to": name}, author)

    def merge(self, keep: int, absorb: int, author: str) -> None:
        mem = self.members(absorb)
        self._assign(mem, keep, "manual", author, f"merged view {absorb} into {keep}")
        self.con.execute("UPDATE camera_view SET merged_into=?, state='retired' WHERE id=?", (keep, absorb))
        self._event(keep, "merged", {"absorbed": absorb, "items": len(mem)}, author)

    def split(self, view_id: int, version_ids: list[int], author: str, name: str | None = None) -> int:
        src = self.con.execute("SELECT * FROM camera_view WHERE id=?", (view_id,)).fetchone()
        new = self._new_view(name, src["ref_width"], src["ref_height"], author, f"split from view {view_id}")
        self._assign(version_ids, new, "manual", author, f"split from view {view_id}")
        self._event(view_id, "split", {"new_view": new, "items": len(version_ids)}, author)
        return new

    def confirm(self, view_id: int, author: str) -> None:
        """Operator confirms the grouping (including any day/night link); the view becomes active."""
        self.con.execute("UPDATE camera_view SET link_confirmed=1, state='active' WHERE id=?", (view_id,))
        self._event(view_id, "confirmed_link", None, author)

    def history(self, version_id: int) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT view_id, method, author, reason, created_at FROM view_assignment WHERE source_version_id=? ORDER BY id",
            (version_id,))]

    # --- view-change detection ---
    def check_drift(self, version_id: int, author: str = "system", tolerance_px: float = DRIFT_TOLERANCE_PX) -> str:
        """Compare an image with its view's reference for the same lighting condition.

        'shifted' needs enough matched features and a displacement above tolerance.
        Poor visibility gives 'unverified', which is recorded but never flags the view.
        After DRIFT_CONSECUTIVE shifted results in a row the view goes to needs_review.
        """
        view_id = self.current_view(version_id)
        if view_id is None:
            return "unassigned"
        row = self.con.execute("SELECT condition FROM view_signature WHERE source_version_id=?", (version_id,)).fetchone()
        if row is None:
            self.compute_signatures()
            row = self.con.execute("SELECT condition FROM view_signature WHERE source_version_id=?", (version_id,)).fetchone()
            if row is None:
                return "unverified"
        cond = row["condition"]
        ref = self.con.execute("SELECT source_version_id FROM view_reference WHERE view_id=? AND condition=?", (view_id, cond)).fetchone()
        if ref is None or ref[0] == version_id:
            return "no_reference"
        from .regions import RegionSet
        pts = RegionSet.load(self.con, view_id).vertices()
        inl, disp = match_geometry(self.image(ref[0]), self.image(version_id), np.float32(pts) if len(pts) else None)
        if inl < GEO_MIN_INLIERS or disp is None:
            result = "unverified"
        else:
            result = "shifted" if disp > tolerance_px else "ok"
        self.con.execute("INSERT OR REPLACE INTO view_drift VALUES (?,?,?,?,?,?)",
                         (version_id, view_id, inl, disp, result, utcnow()))
        if result == "shifted":
            recent = [r[0] for r in self.con.execute(
                "SELECT d.result FROM view_drift d JOIN source_version v ON v.id=d.source_version_id "
                "WHERE d.view_id=? AND d.result<>'unverified' ORDER BY v.capture_utc DESC LIMIT ?", (view_id, DRIFT_CONSECUTIVE))]
            if len(recent) == DRIFT_CONSECUTIVE and all(x == "shifted" for x in recent):
                self.con.execute("UPDATE camera_view SET state='needs_review' WHERE id=?", (view_id,))
                self._event(view_id, "needs_review", {"version": version_id, "displacement_px": disp}, author)
                self.m.notify("warning", "view_changed",
                              f"View {view_id} appears to have moved by about {disp:.0f} px. Its regions and background "
                              "are not applied until you review it.", view_id=view_id)
        return result

    def regions_applicable(self, view_id: int) -> bool:
        st = self.con.execute("SELECT state FROM camera_view WHERE id=?", (view_id,)).fetchone()["state"]
        return st != "needs_review"
