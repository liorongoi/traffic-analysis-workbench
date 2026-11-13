"""Videos that are parts of one recording: suggesting them, grouping them, and joining a target across the cut.

Rules:
- Two videos are suggested as consecutive parts when, in the same folder, the second starts where the first ends
  (within 2 s, from the file-name start time and the length), or their names differ only by a part number
  ("_001", "-2", " (3)", "part4", or a GoPro chapter such as GH01xxxx, GH02xxxx). Suggestions are only offered;
  the operator confirms them with one click, or ticks videos and merges them.
- A group is analysed as one job, its parts in order. A video belongs to one active group at most.
- A target that crosses from one part into the next is joined: an event that ends within 1 s of the end of a part
  and an event that starts within 1 s of the start of the next part, of the same family (vehicles together), whose
  boxes overlap or nearly touch, are one target. The later event "continues" the earlier one and is not counted
  again. Each event can continue one event and be continued by one.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath

from .detect import family
from .store import Mission, utcnow

TIME_GAP_S = 2.0
EDGE_S = 1.0
_PART = re.compile(r"^(?P<key>.*?)(?:[ _.\-]*(?:part|pt|seg|segment|chunk|clip|p)?[ _.\-]*\(?(?P<n>\d{1,3})\)?)$", re.I)
_GOPRO = re.compile(r"^(?P<pre>G[HXOP])(?P<n>\d{2})(?P<id>\d{4})$", re.I)


class GroupError(ValueError):
    pass


def part_of(rel_path: str) -> tuple[str, int] | None:
    """(the name without its part number, the part number), or None when the name carries none."""
    p = PurePosixPath(rel_path)
    stem = p.stem
    g = _GOPRO.match(stem)
    if g:
        return f"{p.parent}/{g['pre'].upper()}{g['id']}", int(g["n"])
    m = _PART.match(stem)
    if not m or not m["key"] or m["key"] == stem:
        return None
    sep = stem[len(m["key"]):]
    if sep.isdigit() and m["key"][-1:].isdigit():
        return None                  # digits run on from the name (a time or an id), not a part number
    return f"{p.parent}/{m['key'].lower()}", int(m["n"])


def _start(v: dict) -> float | None:
    s = v.get("start_utc")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def suggest(items: list[dict], grouped: set[int]) -> list[dict]:
    """Runs of videos that look like consecutive parts of one recording: [{"members": [ids], "reason"}]."""
    free = [v for v in items if v["id"] not in grouped and not v.get("error")]
    out: list[dict] = []
    taken: set[int] = set()

    def runs(seq, linked, reason):
        cur = [seq[0]] if seq else []
        for a, b in zip(seq, seq[1:]):
            if linked(a, b):
                cur.append(b)
                continue
            if len(cur) > 1:
                out.append({"members": [v["id"] for v in cur], "reason": reason})
            cur = [b]
        if len(cur) > 1:
            out.append({"members": [v["id"] for v in cur], "reason": reason})

    # by name: the same name with part numbers 1, 2, 3 ... in one folder
    by_key: dict[str, list[tuple[int, dict]]] = {}
    for v in free:
        k = part_of(v["rel_path"])
        if k:
            by_key.setdefault(k[0], []).append((k[1], v))
    for parts in by_key.values():
        parts.sort(key=lambda x: x[0])
        nums = {v["id"]: n for n, v in parts}
        runs([v for _, v in parts], lambda a, b: nums[b["id"]] == nums[a["id"]] + 1, "part numbers in the names")
    taken = {i for s in out for i in s["members"]}
    # by time: in one folder, each video starting where the one before ended
    timed = sorted((v for v in free if v["id"] not in taken and _start(v) is not None and v.get("duration_s")),
                   key=lambda v: (str(PurePosixPath(v["rel_path"]).parent), _start(v)))
    runs(timed, lambda a, b: PurePosixPath(a["rel_path"]).parent == PurePosixPath(b["rel_path"]).parent
         and abs(_start(b) - (_start(a) + a["duration_s"])) <= TIME_GAP_S, "each starts where the one before ends")
    return out


class Groups:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con

    def all(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT g.id, g.name, g.author, g.created_at, m.source_version_id, m.position FROM recording_group g "
            "JOIN recording_group_member m ON m.group_id=g.id WHERE g.dissolved_at IS NULL ORDER BY g.id, m.position")
        out: dict[int, dict] = {}
        for r in rows:
            g = out.setdefault(r["id"], {"id": r["id"], "name": r["name"], "author": r["author"],
                                         "created_at": r["created_at"], "members": []})
            g["members"].append(r["source_version_id"])
        return list(out.values())

    def of(self, version_id: int) -> dict | None:
        return next((g for g in self.all() if version_id in g["members"]), None)

    def get(self, group_id: int) -> dict:
        g = next((g for g in self.all() if g["id"] == group_id), None)
        if g is None:
            raise GroupError("That merged recording does not exist (any more).")
        return g

    def create(self, items: list[dict], version_ids: list[int], author: str) -> dict:
        """Merge videos into one recording. The order is by start time, else part number, else name."""
        ids = list(dict.fromkeys(int(v) for v in version_ids))
        if len(ids) < 2:
            raise GroupError("Tick two or more videos to merge them into one recording.")
        known = {v["id"]: v for v in items}
        if any(i not in known for i in ids):
            raise GroupError("Some ticked videos are no longer in the source folders. Refresh the list.")
        busy = [known[i]["rel_path"] for i in ids if self.of(i)]
        if busy:
            raise GroupError("Already part of a merged recording: " + ", ".join(busy) + ". Separate it first.")

        def order(i):
            v = known[i]
            p = part_of(v["rel_path"])
            return (_start(v) if _start(v) is not None else float("inf"), p[1] if p else 0, v["rel_path"])
        ids.sort(key=order)
        first = PurePosixPath(known[ids[0]]["rel_path"]).stem
        p = part_of(known[ids[0]]["rel_path"])
        name = (PurePosixPath(p[0]).name if p else first) + f" ({len(ids)} parts)"
        gid = self.con.execute("INSERT INTO recording_group(name,author,created_at) VALUES (?,?,?)",
                               (name, author, utcnow())).lastrowid
        self.con.executemany("INSERT INTO recording_group_member(group_id,source_version_id,position) VALUES (?,?,?)",
                             [(gid, v, k) for k, v in enumerate(ids, 1)])
        return self.get(gid)

    def dissolve(self, group_id: int) -> None:
        self.get(group_id)
        self.con.execute("UPDATE recording_group SET dissolved_at=? WHERE id=?", (utcnow(), group_id))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _near(a, b) -> float:
    """How well two boxes line up across a cut: their overlap, or, for boxes that nearly touch, a small score."""
    o = _iou(a, b)
    if o > 0:
        return o
    ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    return 0.05 if d <= size else 0.0


def links(svc, group: dict) -> dict[int, int]:
    """Across each cut of a merged recording: {event id in the later part: the event it continues in the part before}."""
    dur = {r[0]: r[1] for r in svc.con.execute("SELECT source_version_id, duration_s FROM video_probe")}
    by_vid = {v: [e for e in svc._events_plain(v) if e.status not in ("rejected", "merged", "duplicate")]
              for v in group["members"]}
    out: dict[int, int] = {}
    for a, b in zip(group["members"], group["members"][1:]):
        end = dur.get(a) or 0.0
        ending = [(e, svc._edge_box(e, last=True)) for e in by_vid[a] if e.last_t >= end - EDGE_S]
        starting = [(e, svc._edge_box(e, last=False)) for e in by_vid[b] if e.first_t <= EDGE_S]
        pairs = sorted(((_near(x, y), ea, eb) for ea, x in ending for eb, y in starting
                        if x and y and family(ea.cls) == family(eb.cls)), key=lambda p: -p[0])
        used_a, used_b = set(), set()
        for score, ea, eb in pairs:
            if score <= 0 or ea.id in used_a or eb.id in used_b:
                continue
            used_a.add(ea.id)
            used_b.add(eb.id)
            out[eb.id] = ea.id
    return out
