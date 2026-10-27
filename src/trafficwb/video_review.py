"""Video events: review, timing edits, merge, split, clips and export (VID-04, REV for video).

Rules:
- Machine events never change. Every operator decision is an appended
  event_revision; the current state of an event is derived from its
  revisions in order.
- Timing edits snap to the presentation time of a real frame.
- Merge: the merged event points at the event it joins; that event's interval
  becomes the union. Split: the event is shown and exported as two parts cut
  at a frame time; both parts keep the same machine event as their origin.
- Clips: context before and after is an operator setting with no preset value
  (Q15). Two methods: "copy" (fast, starts at the keyframe at or before the
  requested start, so it may start earlier; the actual start is recorded) and
  "reencode" (frame-exact). Clip files are never overwritten.
- Absolute UTC appears only when the recording start is known, with its
  precision; otherwise only times within the recording are given.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .store import Mission, utcnow
from .video import frame_at, start_utc, utc_at
from .views import ViewService

ACTIONS = ("confirm", "reject", "duplicate", "set_times", "set_class", "merge_into", "split_at", "reopen")
CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle")


class VideoError(ValueError):
    pass


def fmt_t(t: float | None) -> str:
    if t is None:
        return ""
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class Event:
    id: int
    key: str
    source_version_id: int
    file: str
    run_id: int
    machine_first_t: float
    machine_last_t: float
    first_t: float
    last_t: float
    machine_cls: str
    cls: str
    cls_share: float
    flags: list[str]
    related: list[str]
    end_reason: str
    entry: list[float]
    exit: list[float]
    status: str = "pending"            # pending | confirmed | rejected | duplicate | merged
    duplicate_of: str | None = None
    merged_into: int | None = None
    absorbed: list[str] = field(default_factory=list)
    split_at: list[float] = field(default_factory=list)
    first_utc: str | None = None
    last_utc: str | None = None
    utc_precision: str | None = None
    thumb_t: float = 0.0
    thumb_box: list[float] = field(default_factory=list)
    revisions: int = 0

    @property
    def parts(self) -> list[tuple[float, float]]:
        cuts = [c for c in sorted(self.split_at) if self.first_t < c < self.last_t]
        edges = [self.first_t, *cuts, self.last_t]
        return list(zip(edges[:-1], edges[1:]))


class VideoService:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.views = ViewService(mission)

    def path(self, version_id: int) -> Path:
        return self.views._path(version_id)

    # ---------- reading ----------
    def _base(self, where: str = "", args: tuple = ()) -> list[Event]:
        rows = self.con.execute(
            "SELECT e.*, f.rel_path, i.run_id, "
            "CASE WHEN sr.action='set' THEN sr.start_utc ELSE p.start_utc END AS effective_start_utc, "
            "CASE WHEN sr.action='set' THEN sr.precision ELSE p.start_precision END AS effective_start_precision "
            "FROM video_event e "
            "JOIN source_version v ON v.id=e.source_version_id JOIN source_file f ON f.id=v.source_file_id "
            "JOIN run_item i ON i.id=e.run_item_id LEFT JOIN video_probe p ON p.source_version_id=e.source_version_id "
            "LEFT JOIN video_start_revision sr ON sr.id=(SELECT MAX(r.id) FROM video_start_revision r "
            "WHERE r.source_version_id=e.source_version_id) "
            f"{where} ORDER BY f.rel_path, e.first_t, e.id", args).fetchall()
        out = []
        for r in rows:
            out.append(Event(r["id"], r["event_key"], r["source_version_id"], r["rel_path"], r["run_id"], r["first_t"],
                             r["last_t"], r["first_t"], r["last_t"], r["cls"], r["cls"], r["cls_share"],
                             json.loads(r["flags"]), json.loads(r["related"]), r["end_reason"], json.loads(r["entry"]),
                             json.loads(r["exit"]), utc_precision=r["effective_start_precision"], thumb_t=r["thumb_t"],
                             thumb_box=json.loads(r["thumb_box"])))
            out[-1]._start = start_utc(r["effective_start_utc"])  # type: ignore[attr-defined]
        return out

    def _apply(self, events: list[Event]) -> list[Event]:
        by_id = {e.id: e for e in events}
        if not by_id:
            return events
        q = ",".join("?" * len(by_id))
        revs = self.con.execute(f"SELECT * FROM event_revision WHERE event_id IN ({q}) ORDER BY id", tuple(by_id)).fetchall()
        for r in revs:
            e = by_id[r["event_id"]]
            p = json.loads(r["payload"])
            e.revisions += 1
            a = r["action"]
            if a == "confirm":
                e.status = "confirmed"
            elif a == "reject":
                e.status = "rejected"
            elif a == "duplicate":
                e.status, e.duplicate_of = "duplicate", p["of"]
            elif a == "set_times":
                e.first_t, e.last_t = p["first_t"], p["last_t"]
            elif a == "set_class":
                e.cls = p["cls"]
            elif a == "merge_into":
                e.status, e.merged_into = "merged", p["target"]
            elif a == "split_at":
                e.split_at = sorted(set(e.split_at) | {p["t"]})
            elif a == "reopen":
                e.status, e.duplicate_of, e.merged_into = "pending", None, None
        # Merges extend the target (only when both are loaded; a single-event read re-queries below).
        for e in events:
            if e.status == "merged" and e.merged_into in by_id:
                tgt = by_id[e.merged_into]
                tgt.first_t, tgt.last_t = min(tgt.first_t, e.first_t), max(tgt.last_t, e.last_t)
                tgt.absorbed.append(e.key)
        for e in events:
            st = getattr(e, "_start", None)
            e.first_utc, e.last_utc = utc_at(st, e.first_t), utc_at(st, e.last_t)
        return events

    def events(self, version_id: int | None = None, status: str | None = None, flag: str | None = None) -> list[Event]:
        where, args = "", ()
        if version_id is not None:
            where, args = "WHERE e.source_version_id=?", (version_id,)
        evs = self._apply(self._base(where, args))
        if status:
            evs = [e for e in evs if e.status == status]
        if flag:
            evs = [e for e in evs if flag in e.flags]
        return evs

    def event(self, event_id: int) -> Event:
        r = self.con.execute("SELECT source_version_id FROM video_event WHERE id=?", (event_id,)).fetchone()
        if r is None:
            raise VideoError("No such event.")
        return next(e for e in self.events(r[0]) if e.id == event_id)

    def observations(self, event_id: int) -> list[dict]:
        return [dict(r) for r in self.con.execute(
            "SELECT o.t, o.x1, o.y1, o.x2, o.y2, o.cls, o.score, o.appearance FROM track_observation o "
            "JOIN video_event e ON e.track_id=o.track_id WHERE e.id=? ORDER BY o.t", (event_id,))]

    def recordings(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT v.id, f.rel_path, p.*, (SELECT processing_state FROM run_item i WHERE i.source_version_id=v.id "
            " ORDER BY i.id DESC LIMIT 1) AS processing_state, "
            "(SELECT COUNT(*) FROM video_event e WHERE e.source_version_id=v.id) AS events, "
            "sr.id AS start_revision_id, sr.action AS start_revision_action, sr.start_utc AS revised_start_utc, "
            "sr.precision AS revised_start_precision, sr.author AS start_revision_author, "
            "sr.note AS start_revision_note, sr.created_at AS start_revision_created_at "
            "FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "LEFT JOIN video_probe p ON p.source_version_id=v.id "
            "LEFT JOIN video_start_revision sr ON sr.id=(SELECT MAX(r.id) FROM video_start_revision r "
            "WHERE r.source_version_id=v.id) "
            "WHERE v.superseded_by IS NULL ORDER BY f.rel_path").fetchall()
        out = []
        for row in rows:
            d = dict(row) | {"id": row["id"]}
            filename_start, filename_precision = d.get("start_utc"), d.get("start_precision")
            if d["start_revision_id"] is not None:
                if d["start_revision_action"] == "set":
                    d["start_utc"] = d["revised_start_utc"]
                    d["start_precision"] = d["revised_start_precision"]
                    d["start_source"] = "manual"
                else:
                    d["start_utc"], d["start_precision"] = filename_start, filename_precision
                    d["start_source"] = "filename" if filename_start else "none"
            else:
                d["start_source"] = "filename" if d.get("start_utc") else "none"
            return_fields = ("start_revision_id", "start_revision_action", "revised_start_utc", "revised_start_precision")
            for field in return_fields:
                d.pop(field, None)
            out.append(d)
        return out

    def revise_start(self, version_id: int, action: str, author: str, start: datetime | None = None,
                     precision: str | None = None, note: str | None = None) -> dict:
        """Append an operator-confirmed UTC start or clear the latest manual override."""
        if self.m.config().input_type != "video":
            raise VideoError("Recording start times can only be corrected in a video mission.")
        if self.con.execute("SELECT 1 FROM source_version WHERE id=?", (version_id,)).fetchone() is None:
            raise VideoError("That recording does not exist in this mission.")
        if action not in ("set", "clear"):
            raise VideoError("Action must be set or clear.")
        if action == "set":
            if start is None or start.tzinfo is None or start.utcoffset() is None:
                raise VideoError("Enter a start time with an explicit UTC offset, such as Z.")
            if precision not in ("minute", "second"):
                raise VideoError("Precision must be minute or second.")
            normalized = start.astimezone(timezone.utc)
            if precision == "minute":
                normalized = normalized.replace(second=0, microsecond=0)
            value = normalized.isoformat(timespec="seconds").replace("+00:00", "Z")
        else:
            value = None
            precision = None
        self.con.execute(
            "INSERT INTO video_start_revision(source_version_id,action,start_utc,precision,author,note,created_at) "
            "VALUES (?,?,?,?,?,?,?)", (version_id, action, value, precision, author, note, utcnow()))
        return next(r for r in self.recordings() if r["id"] == version_id)

    # ---------- decisions ----------
    def revise(self, event_id: int, action: str, author: str, note: str | None = None, **payload) -> Event:
        if action not in ACTIONS:
            raise VideoError(f"Action must be one of {', '.join(ACTIONS)}.")
        e = self.event(event_id)
        if action == "duplicate":
            other = payload.get("of")
            if not other or not self.con.execute(
                    "SELECT 1 FROM video_event WHERE event_key=? AND source_version_id=?", (other, e.source_version_id)).fetchone():
                raise VideoError("Name the event this one duplicates (same recording).")
            payload = {"of": other}
        elif action == "set_times":
            a, b = float(payload["first_t"]), float(payload["last_t"])
            a, b = self.snap(e.source_version_id, a), self.snap(e.source_version_id, b)
            if b <= a:
                raise VideoError("The end must be after the start.")
            payload = {"first_t": a, "last_t": b}
        elif action == "set_class":
            if payload.get("cls") not in CLASSES:
                raise VideoError(f"Class must be one of {', '.join(CLASSES)}.")
            payload = {"cls": payload["cls"]}
        elif action == "merge_into":
            tgt = int(payload["target"])
            row = self.con.execute("SELECT source_version_id FROM video_event WHERE id=?", (tgt,)).fetchone()
            if row is None or row[0] != e.source_version_id or tgt == event_id:
                raise VideoError("Merge only with another event of the same recording.")
            if self.event(tgt).status == "merged":
                raise VideoError("That event is itself merged; merge into the event it joined.")
            payload = {"target": tgt}
        elif action == "split_at":
            t = self.snap(e.source_version_id, float(payload["t"]))
            if not e.first_t < t < e.last_t:
                raise VideoError("Split inside the event's interval.")
            payload = {"t": t}
        else:
            payload = {}
        self.con.execute("INSERT INTO event_revision(event_id,action,payload,author,note,created_at) VALUES (?,?,?,?,?,?)",
                         (event_id, action, json.dumps(payload), author, note, utcnow()))
        return self.event(event_id)

    def snap(self, version_id: int, t: float) -> float:
        """The presentation time of the first frame at or after t (or the last frame)."""
        f = frame_at(self.path(version_id), max(0.0, t))
        if f is not None:
            return round(f.t, 6)
        last = self.con.execute("SELECT duration_s FROM video_probe WHERE source_version_id=?", (version_id,)).fetchone()
        return round(min(t, last[0] or t), 6) if last else round(t, 6)

    # ---------- thumbnails ----------
    def thumbnail(self, event_id: int, height: int = 120) -> bytes:
        import cv2
        e = self.event(event_id)
        cache = self.m.dir / "evidence" / "thumbs" / f"event_{event_id}_{height}.jpg"
        if cache.is_file():
            return cache.read_bytes()
        f = frame_at(self.path(e.source_version_id), e.thumb_t)
        if f is None:
            raise VideoError("Frame not found in the recording.")
        x1, y1, x2, y2 = e.thumb_box
        H, W = f.image.shape[:2]
        pw, ph = (x2 - x1) * 0.2 + 4, (y2 - y1) * 0.2 + 4
        crop = f.image[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
        s = height / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), height))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(buf.tobytes())
        return buf.tobytes()

    # ---------- clips ----------
    def clip(self, event_id: int, before_s: float, after_s: float, method: str, author: str,
             out_dir: Path | None = None) -> dict:
        if method not in ("copy", "reencode"):
            raise VideoError("Method must be copy or reencode.")
        if before_s is None or after_s is None or before_s < 0 or after_s < 0 or before_s > 600 or after_s > 600:
            raise VideoError("Give the context before and after the event in seconds (0 to 600).")
        e = self.event(event_id)
        src = self.path(e.source_version_id)
        cfg = self.m.config()
        out = Path(out_dir or cfg.output_dir) / "clips"
        out.mkdir(parents=True, exist_ok=True)
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{Path(e.file).stem}_{e.key}_{method}")
        path = out / f"{stem}.mp4"
        n = 2
        while path.exists() or path.with_name(path.name + ".partial").exists():
            path = out / f"{stem}_{n}.mp4"
            n += 1
        start, end = max(0.0, e.first_t - before_s), e.last_t + after_s
        tmp = path.with_name(path.name + ".partial")
        if method == "copy":
            a, b, enc = _copy_clip(src, tmp, start, end)
        else:
            a, b, enc = _reencode_clip(src, tmp, start, end)
        if path.exists():
            raise VideoError(f"{path.name} appeared while writing; nothing was overwritten.")
        tmp.rename(path)
        cid = self.con.execute(
            "INSERT INTO clip(event_id,path,method,encoder,context_before_s,context_after_s,requested_start_t,requested_end_t,"
            "actual_start_t,actual_end_t,author,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, str(path), method, enc, before_s, after_s, start, end, a, b, author, utcnow())).lastrowid
        return {"id": cid, "path": str(path), "method": method, "encoder": enc, "requested": [start, end], "actual": [a, b]}

    def clips(self, event_id: int) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM clip WHERE event_id=? ORDER BY id", (event_id,))]

    # ---------- export ----------
    def export(self, author: str, formats: list[str] | None = None, out_dir=None) -> list[Path]:
        from .export import _fmt, write_sheets_csv, write_sheets_xlsx
        cfg = self.m.config()
        formats = formats or cfg.export_formats
        out = Path(out_dir or cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sha = {r[0]: r[1] for r in self.con.execute("SELECT id, sha256 FROM source_version")}
        models = {r["id"]: json.loads(r["model_versions"] or "{}") for r in self.con.execute("SELECT id, model_versions FROM run")}
        plates = {}
        if cfg.mode == "plate_recognition":
            from .plates import PlateService
            plates = {p.object_uid: p for p in PlateService(self.m).plates()}
        rows = []
        for e in self.events():
            clips = ";".join(c["path"] for c in self.clips(e.id))
            pl = plates.get(f"e{e.id}")
            for k, (a, b) in enumerate(e.parts, 1):
                st = getattr(e, "_start", None)
                rows.append([_fmt(x) for x in (
                    self.m.id, e.run_id, e.source_version_id, e.file, sha.get(e.source_version_id), e.key,
                    k if len(e.parts) > 1 else "", fmt_t(a), fmt_t(b), round(a, 3), round(b, 3), utc_at(st, a), utc_at(st, b),
                    e.utc_precision or ("unknown" if st is None else ""), e.cls, e.machine_cls, e.cls_share, e.status,
                    e.duplicate_of or "", ",".join(e.absorbed), ",".join(e.flags), ",".join(e.related), e.end_reason,
                    e.entry[0], e.entry[1], e.exit[0], e.exit[1], fmt_t(e.machine_first_t), fmt_t(e.machine_last_t),
                    e.revisions, clips,
                    (models.get(e.run_id, {}).get("vehicle_detection") or {}).get("name", ""),
                    pl.text if pl else "", pl.readability if pl else "", pl.review if pl else "")])
        cols = ["mission_id", "run_id", "source_version_id", "file", "sha256", "event_id", "part", "start", "end",
                "start_s", "end_s", "start_utc", "end_utc", "utc_precision", "class", "machine_class", "class_share",
                "review_status", "duplicate_of", "merged_events", "flags", "possible_duplicate_of", "end_reason",
                "entry_x", "entry_y", "exit_x", "exit_y", "machine_start", "machine_end", "operator_revisions", "clips",
                "model", "plate_text", "plate_readability", "plate_review"]
        stem = "".join(c if c.isalnum() or c in "-_." else "_"
                       for c in f"{cfg.name}_events_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        paths = []
        for fmt in formats:
            new = (write_sheets_csv if fmt == "csv" else write_sheets_xlsx)(out, stem, [("events", cols, rows)])
            paths += new
            self.con.execute("INSERT INTO export(format,path,scope,rows,status,author,created_at) VALUES (?,?,?,?,?,?,?)",
                             (fmt, ";".join(str(p) for p in new), json.dumps({"video_events": True}), len(rows),
                              "completed", author, utcnow()))
        return paths


def _copy_clip(src: Path, dst: Path, start: float, end: float) -> tuple[float | None, float | None, str | None]:
    import av
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="mp4") as out:
        vs = next(s for s in inp.streams if s.type == "video")
        os_ = out.add_stream_from_template(vs)
        tb, st0 = vs.time_base, vs.start_time or 0
        inp.seek(int(Fraction(start).limit_denominator(1_000_000) / tb) + st0, stream=vs, backward=True, any_frame=False)
        first_pts = first = last = None
        for pkt in inp.demux(vs):
            if pkt.pts is None or pkt.dts is None:
                continue
            t = float((pkt.pts - st0) * tb)
            if first_pts is None:
                if not pkt.is_keyframe:
                    continue
                first_pts, first = pkt.pts, t
            if t > end:
                break
            pkt.pts -= first_pts
            pkt.dts -= first_pts
            pkt.stream = os_
            out.mux(pkt)
            last = t if last is None else max(last, t)
    return first, last, None


def _reencode_clip(src: Path, dst: Path, start: float, end: float) -> tuple[float | None, float | None, str]:
    import av
    from .video import frames
    enc = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    first = last = None
    with av.open(str(dst), "w", format="mp4") as out:
        stream = None
        for f, _ in frames(src, interval_s=1e-6, after_t=start - 1e-6 if start > 0 else None, next_sample_t=start):
            if f.t > end:
                break
            if stream is None:
                stream = out.add_stream(enc, rate=30)
                h, w = f.image.shape[:2]
                stream.width, stream.height = w - w % 2, h - h % 2
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(1, 1000)
                stream.codec_context.time_base = Fraction(1, 1000)
                if enc == "libx264":
                    stream.options = {"bf": "0"}
                first = f.t
            img = f.image[:stream.height, :stream.width]
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts = int(round((f.t - first) * 1000))
            fr.time_base = Fraction(1, 1000)
            for p in stream.encode(fr):
                out.mux(p)
            last = f.t
        if stream is not None:
            for p in stream.encode():
                out.mux(p)
    if first is None:
        dst.unlink(missing_ok=True)
        raise VideoError("No frames in the requested interval.")
    return first, last, enc
