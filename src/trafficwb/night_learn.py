"""Learned night scoring per camera view, from the operator's answers on night candidates.

- Every analysed light group stores a few measurements (`light_group.features`): brightness and how much it
  stands out, clipped share, size against the expected vehicle width, shape, how often the place is lit,
  distance to where vehicles drive by day, kind and distance band.
- When a view has enough answers (confirmed = vehicle; rejected or fixed light = not a vehicle), a small
  logistic model is fitted on those measurements. It is checked by cross-validation, with folds split by
  image so an image never helps grade itself, against the hand-made rule score on the same answers.
- The model is used only when it ranks the answered candidates better than the rules (AUC higher by at least
  `min_gain`). Otherwise the rules stay, and the reason is recorded. Each fit is stored with its numbers.
- Nothing is learned from statistics alone: unanswered candidates are never treated as either class.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np

from .store import utcnow

KINDS = ("single", "pair", "cluster")
BANDS = ("far", "middle", "close")
NUMERIC = ("peak", "contrast", "clipped_share", "log_area", "n_lights", "size_ratio", "aspect", "recurrence",
           "road_distance", "row")
FEATURES = NUMERIC + tuple(f"kind_{k}" for k in KINDS) + tuple(f"band_{b}" for b in BANDS)


@dataclass
class LearnParams:
    min_each: int = 15          # answers of each class needed before a model is fitted
    min_gain: float = 0.02      # cross-validated AUC must beat the rules by this much
    folds: int = 5
    l2: float = 1.0

    def record(self) -> dict:
        return dict(self.__dict__)


def measure(gray: np.ndarray | None, box, members, wm, recurrence: float | None, road_dist: float | None,
            kind: str, band: str | None) -> dict:
    """Measurements of one light group. `members` are its Light objects."""
    x1, y1, x2, y2 = box
    width = wm.width(y2) if wm else None
    out = {"peak": max(li.peak for li in members), "contrast": None, "clipped_share": None,
           "log_area": float(np.log1p(sum(li.area for li in members))), "n_lights": len(members),
           "size_ratio": round((x2 - x1) / width, 3) if width else None,
           "aspect": round((y2 - y1 + 1) / (x2 - x1 + 1), 3), "recurrence": recurrence,
           "road_distance": round(min(road_dist / width, 5.0), 3) if road_dist is not None and width else None,
           "row": None, "kind": kind, "band": band}
    if gray is not None:
        from .night import glow_contrast
        H, W = gray.shape[:2]
        peak, contrast = glow_contrast(gray, box)
        a, b = max(0, int(x1)), max(0, int(y1))
        c, d = min(W, max(a + 1, int(round(x2)))), min(H, max(b + 1, int(round(y2))))
        part = gray[b:d, a:c]
        lit = int((part >= 235).sum())
        out.update(contrast=round(contrast, 1), clipped_share=round(float((part >= 250).sum()) / max(1, lit), 3),
                   row=round(y2 / H, 3))
    return out


def vector(f: dict) -> list[float | None]:
    num = [f.get(k) for k in NUMERIC]
    return num + [1.0 if f.get("kind") == k else 0.0 for k in KINDS] + [1.0 if f.get("band") == b else 0.0 for b in BANDS]


def auc(scores, y) -> float | None:
    """Probability that a random vehicle scores higher than a random non-vehicle (ties count half)."""
    s, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):                                  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@dataclass
class LightModel:
    fill: list[float]          # value used for a missing measurement (training median)
    mean: list[float]
    std: list[float]
    w: list[float]
    b: float

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        nan = np.isnan(X)
        X[nan] = np.take(np.asarray(self.fill), np.nonzero(nan)[1])
        return (X - np.asarray(self.mean)) / np.asarray(self.std)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float).reshape(-1, len(self.w))
        z = self._prep(X) @ np.asarray(self.w) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def record(self) -> dict:
        return {"features": list(FEATURES), "fill": self.fill, "mean": self.mean, "std": self.std, "w": self.w, "b": self.b}

    @classmethod
    def from_record(cls, r: dict) -> "LightModel":
        return cls(r["fill"], r["mean"], r["std"], r["w"], r["b"])


def fit(X, y, l2: float = 1.0, iters: int = 50) -> LightModel:
    """L2-regularised logistic regression by Newton steps. Missing values are filled with the training median."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    with warnings.catch_warnings():                      # a measurement missing everywhere: filled with 0
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmedian(X, axis=0)
    fill = np.where(np.isnan(fill), 0.0, fill)
    Xf = np.where(np.isnan(X), fill, X)
    mean, std = Xf.mean(axis=0), Xf.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = np.hstack([(Xf - mean) / std, np.ones((len(Xf), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0                                        # the intercept is not shrunk
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return LightModel(fill.tolist(), mean.tolist(), std.tolist(), w[:-1].tolist(), float(w[-1]))


def cross_validate(X, y, groups, rule, p: LearnParams) -> tuple[float | None, float | None]:
    """AUC of the model and of the rule score on held-out answers; folds keep each image's answers together."""
    X, y, rule = np.asarray(X, float), np.asarray(y, int), np.asarray(rule, float)
    ids = sorted(set(groups))
    rng = np.random.default_rng(7)
    rng.shuffle(ids)
    fold_of = {g: i % p.folds for i, g in enumerate(ids)}
    folds = np.array([fold_of[g] for g in groups])
    pred = np.full(len(y), np.nan)
    for k in range(p.folds):
        test = folds == k
        train = ~test
        if not test.any() or len(set(y[train])) < 2:
            continue
        pred[test] = fit(X[train], y[train], p.l2).predict(X[test])
    ok = ~np.isnan(pred)
    return auc(pred[ok], y[ok]), auc(rule[ok], y[ok])


class NightLearner:
    def __init__(self, con, params: LearnParams | None = None):
        self.con = con
        self.p = params or LearnParams()
        self._cache: dict = {}

    def answers(self, view_id: int) -> list[tuple[int, dict, float, int]]:
        """(image, measurements, rule score, 1 vehicle / 0 not) for the view's answered machine candidates."""
        rows = self.con.execute(
            "SELECT r.source_version_id, r.action, g.features, g.score FROM candidate_revision r "
            "JOIN view_assignment a ON a.source_version_id=r.source_version_id AND a.superseded_by IS NULL "
            "JOIN light_group g ON g.id=r.light_group_id "
            "WHERE a.view_id=? AND g.features IS NOT NULL AND r.id=(SELECT MAX(id) FROM candidate_revision x "
            "WHERE x.group_uid=r.group_uid AND x.source_version_id=r.source_version_id) "
            "AND r.action IN ('confirmed','rejected','persistent_light')", (view_id,)).fetchall()
        return [(r[0], json.loads(r[2]), float(r[3]), 1 if r[1] == "confirmed" else 0) for r in rows]

    def train(self, view_id: int, decision_rev: int | None) -> dict:
        """Fit and check a model for the view at this decision revision; stored either way, with the reason."""
        data = self.answers(view_id)
        y = [d[3] for d in data]
        n_pos, n_neg = sum(y), len(y) - sum(y)
        rec = {"view_id": view_id, "decision_rev": decision_rev, "n_pos": n_pos, "n_neg": n_neg,
               "auc_model": None, "auc_rules": None, "applied": 0, "model": None}
        if min(n_pos, n_neg) < self.p.min_each:
            rec["reason"] = f"needs at least {self.p.min_each} answers of each kind (vehicles {n_pos}, not vehicles {n_neg})"
        else:
            X = [[np.nan if v is None else float(v) for v in vector(d[1])] for d in data]
            a_m, a_r = cross_validate(X, y, [d[0] for d in data], [d[2] for d in data], self.p)
            rec.update(auc_model=a_m, auc_rules=a_r)
            if a_m is None or a_r is None:
                rec["reason"] = "answers too few to check the model"
            elif a_m < a_r + self.p.min_gain:
                rec["reason"] = f"does not rank your answers better than the rules ({a_m:.2f} against {a_r:.2f})"
            else:
                rec.update(applied=1, model=fit(X, y, self.p.l2).record(),
                           reason=f"ranks your answers better than the rules ({a_m:.2f} against {a_r:.2f})")
        self.con.execute(
            "INSERT INTO night_light_model(view_id,decision_rev,created_at,n_pos,n_neg,auc_model,auc_rules,applied,reason,"
            "params,model) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (view_id, decision_rev, utcnow(), n_pos, n_neg, rec["auc_model"], rec["auc_rules"], rec["applied"],
             rec["reason"], json.dumps(self.p.record()), json.dumps(rec["model"]) if rec["model"] else None))
        return rec

    def latest(self, view_id: int | None) -> dict | None:
        if view_id is None:
            return None
        r = self.con.execute("SELECT * FROM night_light_model WHERE view_id=? ORDER BY id DESC LIMIT 1", (view_id,)).fetchone()
        return dict(r) if r else None

    def model_for(self, view_id: int | None, decision_rev: int | None) -> tuple[int, LightModel] | None:
        """The applied model fitted at this decision revision, training it first when there is none yet."""
        if view_id is None or decision_rev is None:
            return None
        key = (view_id, decision_rev)
        if key not in self._cache:
            r = self.con.execute("SELECT id, applied, model FROM night_light_model WHERE view_id=? AND decision_rev=? "
                                 "ORDER BY id DESC LIMIT 1", (view_id, decision_rev)).fetchone()
            if r is None:
                self.train(view_id, decision_rev)
                r = self.con.execute("SELECT id, applied, model FROM night_light_model WHERE view_id=? AND decision_rev=? "
                                     "ORDER BY id DESC LIMIT 1", (view_id, decision_rev)).fetchone()
            self._cache[key] = (r[0], LightModel.from_record(json.loads(r[2]))) if r[1] else None
        return self._cache[key]
