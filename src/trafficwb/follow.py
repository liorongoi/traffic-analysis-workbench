"""Following one target the operator marked, through a recording (learning a target SAM 3.1 does not know).

The operator draws a box around the target on one frame (or a few). From there the target is followed forwards
and backwards on the analysis grid by matching what it looks like (normalised cross-correlation of its picture,
at three sizes, so a target coming closer or moving away is kept), until it is no longer found. The frames where it
was found surely give the suggested range: where it enters and where it leaves. The operator corrects the range or
draws another box where the follow went wrong, confirms, and the frames inside the range become training labels.

Our own code, OpenCV only; deterministic for a given recording and box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

STEP_S = 0.2              # one analysed frame
MAX_S = 60.0              # at most this far from a marked frame, each way
SURE = 0.55               # a match this good is the target
LOST_STEPS = 3            # this many poor matches in a row: it has left (or is hidden)
SCALES = (0.93, 1.0, 1.07)
CHUNK_S = 5.0             # going backwards, the recording is read in pieces this long


@dataclass
class Followed:
    t: float
    box: list[float]
    score: float


@dataclass
class FollowResult:
    steps: list[Followed] = field(default_factory=list)        # in time order, the marked frames included
    start_t: float | None = None                               # suggested range: first and last sure frame
    end_t: float | None = None
    ended: dict = field(default_factory=dict)                  # why each direction stopped

    def plain(self) -> dict:
        return {"steps": [{"t": s.t, "box": s.box, "score": s.score} for s in self.steps],
                "start_t": self.start_t, "end_t": self.end_t, "ended": self.ended}


def _grey(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


class TemplateFollower:
    """One target from one marked box: where it is on each next frame, and how sure."""

    def __init__(self, img: np.ndarray, box, min_side: int = 8):
        g = _grey(img)
        h, w = g.shape
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        if x2 - x1 < min_side or y2 - y1 < min_side:
            raise ValueError("The box is too small to follow; draw it around the whole target.")
        self.box = [x1, y1, x2, y2]
        self.key = g[int(y1):int(y2), int(x1):int(x2)].copy()     # what it looked like where it was marked
        self.cur = self.key.copy()                                # what it looked like last
        self.v = (0.0, 0.0)                                       # centre speed, px per step

    def step(self, img: np.ndarray) -> tuple[list[float], float]:
        import cv2
        g = _grey(img)
        H, W = g.shape
        x1, y1, x2, y2 = self.box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2 + self.v[0], (y1 + y2) / 2 + self.v[1]
        best = (-1.0, None)
        for s in SCALES:
            tw, th = max(4, int(round(bw * s))), max(4, int(round(bh * s)))
            # search a window of the target's size around where it was heading
            ax, ay = int(max(0, cx - tw)), int(max(0, cy - th))
            bx, by = int(min(W, cx + tw)), int(min(H, cy + th))
            win = g[ay:by, ax:bx]
            if win.shape[0] <= th or win.shape[1] <= tw:
                continue
            for tpl in (self.cur, self.key):
                t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                if float(t.std()) < 1e-3:
                    continue
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(r)
                if mv > best[0]:
                    best = (float(mv), [ax + ml[0], ay + ml[1], ax + ml[0] + tw, ay + ml[1] + th])
        score, nb = best
        if nb is None:
            return list(self.box), 0.0
        if score >= SURE:
            ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
            ocx, ocy = (x1 + x2) / 2, (y1 + y2) / 2
            self.v = (0.5 * self.v[0] + 0.5 * (ncx - ocx), 0.5 * self.v[1] + 0.5 * (ncy - ocy))
            self.box = [float(v) for v in nb]
            patch = g[int(nb[1]):int(nb[3]), int(nb[0]):int(nb[2])]
            if patch.size and patch.shape[0] >= 4 and patch.shape[1] >= 4:
                # keep up with a slow change of look, never forgetting the marked one (it is matched too)
                self.cur = cv2.addWeighted(cv2.resize(self.cur, (patch.shape[1], patch.shape[0])), 0.7, patch, 0.3, 0)
        return [round(v, 1) for v in nb], round(score, 3)


def _leaving(box, w: int, h: int) -> bool:
    x1, y1, x2, y2 = box
    inside = max(0.0, min(x2, w) - max(x1, 0)) * max(0.0, min(y2, h) - max(y1, 0))
    return inside < 0.5 * max((x2 - x1) * (y2 - y1), 1e-6)


def _run(follower: TemplateFollower, frames, w: int, h: int) -> tuple[list[Followed], str]:
    out, poor = [], 0
    for t, img in frames:
        box, score = follower.step(img)
        out.append(Followed(round(t, 6), box, score))
        if score < SURE:
            poor += 1
            if poor >= LOST_STEPS:
                return out, "lost"
        else:
            poor = 0
            if _leaving(box, w, h):
                return out, "left_frame"
    return out, "limit"


def follow(path: Path, marks: list[dict], step_s: float = STEP_S, max_s: float = MAX_S,
           duration: float | None = None, checkpoint: Callable[[], None] | None = None) -> FollowResult:
    """Follow a target from the operator's marks ([{"t", "box"}], one or more) forwards and backwards.

    Each mark starts its own follow both ways; between two marks the later mark's backward follow and the earlier
    one's forward follow meet, and the surer of the two is kept for each frame. A marked frame is always sure.
    """
    from .video import frame_at, frames as read

    if not marks:
        raise ValueError("Mark the target on a frame first.")
    marks = sorted(marks, key=lambda m: m["t"])
    best: dict[float, Followed] = {}
    ended = {}

    def keep(f: Followed):
        k = round(f.t, 3)
        if k not in best or f.score > best[k].score:
            best[k] = f

    for n, mk in enumerate(marks):
        fr = frame_at(path, float(mk["t"]))
        if fr is None:
            raise ValueError(f"The frame at {mk['t']:.2f} s cannot be read.")
        h, w = fr.image.shape[:2]
        keep(Followed(round(fr.t, 6), [round(float(v), 1) for v in mk["box"]], 1.0))
        t0 = fr.t
        # forwards, up to the next mark (it takes over from there) or max_s
        stop = marks[n + 1]["t"] if n + 1 < len(marks) else t0 + max_s
        fw = TemplateFollower(fr.image, mk["box"])
        steps, why = _run(fw, ((f.t, f.image) for f, _ in read(path, step_s, after_t=t0, next_sample_t=t0 + step_s,
                                                                     until_t=stop)), w, h)
        for s in steps:
            keep(s)
        ended[f"after {t0:.2f}"] = why
        if checkpoint:
            checkpoint()
        # backwards, down to the previous mark or max_s, read a few seconds at a time
        low = marks[n - 1]["t"] if n > 0 else max(0.0, t0 - max_s)
        bw = TemplateFollower(fr.image, mk["box"])

        def back():
            b = t0
            while b > low + 1e-6:
                a = max(low, b - CHUNK_S)
                chunk = [(f.t, f.image) for f, _ in read(path, step_s, after_t=a - 1e-3 if a > 0 else None,
                                                         next_sample_t=a, until_t=b)]
                for item in reversed(chunk):
                    if item[0] < t0 - 1e-6:
                        yield item
                b = a
        steps, why = _run(bw, back(), w, h)
        for s in steps:
            keep(s)
        ended[f"before {t0:.2f}"] = why
        if checkpoint:
            checkpoint()
    steps = sorted(best.values(), key=lambda s: s.t)
    sure = [s.t for s in steps if s.score >= SURE]
    res = FollowResult(steps, min(sure) if sure else None, max(sure) if sure else None, ended)
    return res


# ---------- marked targets of a mission ----------

FRAMES_PER_TARGET = 60      # training frames a confirmed target gives, spread over its range
MARK_MIN_GAP_S = 0.4


class MarkError(ValueError):
    pass


class MarkedTargets:
    """The targets the operator marked on a mission's recordings: follow, correct, set the range, confirm."""

    def __init__(self, mission, path_of: Callable[[int], Path]):
        self.m = mission
        self.con = mission.con
        self.path_of = path_of

    @staticmethod
    def _row(r) -> dict:
        return {"id": r["id"], "source_version_id": r["source_version_id"], "cls": r["cls"],
                "marks": json.loads(r["marks"]), "steps": json.loads(r["steps"]), "start_t": r["start_t"],
                "end_t": r["end_t"], "state": r["state"], "author": r["author"], "created_at": r["created_at"],
                "updated_at": r["updated_at"]}

    def list(self, vid: int | None = None, state: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM marked_target WHERE deleted_at IS NULL", []
        if vid is not None:
            q, a = q + " AND source_version_id=?", a + [vid]
        if state:
            q, a = q + " AND state=?", a + [state]
        return [self._row(r) for r in self.con.execute(q + " ORDER BY source_version_id, id", a)]

    def get(self, tid: int) -> dict:
        r = self.con.execute("SELECT * FROM marked_target WHERE id=? AND deleted_at IS NULL", (tid,)).fetchone()
        if r is None:
            raise MarkError("That marked target is not in this mission.")
        return self._row(r)

    @staticmethod
    def _box(box) -> list[float]:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise MarkError("A box is four numbers: left, top, right, bottom.")
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            raise MarkError("Draw the box from one corner to the opposite one.")
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    def _follow(self, vid: int, marks: list[dict]) -> FollowResult:
        try:
            return follow(self.path_of(vid), marks)
        except ValueError as exc:
            raise MarkError(str(exc)) from None

    def add(self, vid: int, cls: str, t: float, box, author: str) -> dict:
        from .store import utcnow
        cls = (cls or "").strip().lower()
        if not cls:
            raise MarkError("Choose what the target is.")
        marks = [{"t": round(float(t), 6), "box": self._box(box)}]
        res = self._follow(vid, marks)
        now = utcnow()
        tid = self.con.execute(
            "INSERT INTO marked_target(source_version_id,cls,marks,steps,start_t,end_t,state,author,created_at,"
            "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (vid, cls, json.dumps(marks), json.dumps(res.plain()["steps"]), res.start_t, res.end_t, "suggested",
             author, now, now)).lastrowid
        return self.get(tid)

    def correct(self, tid: int, t: float, box, author: str) -> dict:
        """Another box of the same target (where the follow went wrong, or where it was lost): followed again from
        every mark. A mark within a frame of an earlier one replaces it."""
        from .store import utcnow
        cur = self.get(tid)
        marks = [mk for mk in cur["marks"] if abs(mk["t"] - float(t)) > STEP_S / 2]
        marks.append({"t": round(float(t), 6), "box": self._box(box)})
        res = self._follow(cur["source_version_id"], marks)
        self.con.execute("UPDATE marked_target SET marks=?, steps=?, start_t=?, end_t=?, state='suggested', "
                         "author=?, updated_at=? WHERE id=?",
                         (json.dumps(sorted(marks, key=lambda m: m["t"])), json.dumps(res.plain()["steps"]),
                          res.start_t, res.end_t, author, utcnow(), tid))
        return self.get(tid)

    def update(self, tid: int, author: str, start_t: float | None = None, end_t: float | None = None,
               state: str | None = None, cls: str | None = None) -> dict:
        from .store import utcnow
        cur = self.get(tid)
        a = cur["start_t"] if start_t is None else float(start_t)
        b = cur["end_t"] if end_t is None else float(end_t)
        if a is None or b is None or b < a:
            raise MarkError("The range must start before it ends.")
        if state is not None and state not in ("suggested", "confirmed"):
            raise MarkError("A marked target is suggested or confirmed.")
        if cls is not None and not cls.strip():
            raise MarkError("Choose what the target is.")
        self.con.execute("UPDATE marked_target SET start_t=?, end_t=?, state=?, cls=?, author=?, updated_at=? "
                         "WHERE id=?", (a, b, state or cur["state"], (cls or cur["cls"]).strip(), author, utcnow(),
                                        tid))
        return self.get(tid)

    def remove(self, tid: int, author: str) -> None:
        from .store import utcnow
        self.get(tid)
        self.con.execute("UPDATE marked_target SET deleted_at=?, author=? WHERE id=?", (utcnow(), author, tid))

    @staticmethod
    def frames(target: dict, limit: int = FRAMES_PER_TARGET) -> list[dict]:
        """A confirmed target's training frames: its marks, then frames inside its range where it was surely found,
        spread out and at least 0.4 s apart. [{"t", "box"}]"""
        a, b = target["start_t"], target["end_t"]
        if a is None or b is None:
            return []
        inside = lambda t: a - 1e-6 <= t <= b + 1e-6               # noqa: E731
        must = [{"t": mk["t"], "box": mk["box"]} for mk in target["marks"] if inside(mk["t"])]
        sure = [{"t": s["t"], "box": s["box"]} for s in target["steps"] if s["score"] >= SURE and inside(s["t"])]
        chosen = list(must)
        rest = [s for s in sure if all(abs(s["t"] - c["t"]) >= MARK_MIN_GAP_S for c in must)]
        if rest:
            room = max(0, limit - len(chosen))
            step = max(1.0, len(rest) / max(room, 1))
            for k in range(min(room, len(rest))):
                s = rest[int(k * step)]
                if all(abs(s["t"] - c["t"]) >= MARK_MIN_GAP_S for c in chosen):
                    chosen.append(s)
        return sorted(chosen, key=lambda c: c["t"])
