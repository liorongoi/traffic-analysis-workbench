"""The video catalog: every recording in a mission's source folders, with a thumbnail and its details, and which of
them to analyse.

Rules:
- Opening the catalog looks for new files (the same scan as processing; new files are queued, nothing is analysed).
- A recording not analysed yet (queued, cancelled or failed) is ticked by default, so analysing what is new is one
  click. A recording already analysed can be ticked to analyse it again: it gets a new run item and its newest
  analysis is the one that counts, as for any re-analysis.
- Analysing the ticked recordings runs one job over exactly those; the others stay as they are.
- Videos that are parts of one recording can be merged (see video_groups): ticking one part analyses all of its
  parts, in order, in the same job.
"""
from __future__ import annotations

from .ingest import cycle
from .store import Mission
from .video_groups import GroupError, Groups, suggest
from .video_process import probe_version
from .views import ViewService

NOT_ANALYSED = ("queued", "cancelled", "failed", None)


class CatalogError(ValueError):
    pass


class VideoCatalog:
    def __init__(self, mission: Mission):
        self.m = mission
        self.con = mission.con
        self.views = ViewService(mission)

    def refresh(self, stability_seconds: int = 30) -> None:
        cycle(self.m, stability_seconds=stability_seconds)

    def items(self) -> list[dict]:
        rows = self.con.execute(
            "SELECT v.id, v.sha256, v.capture_utc, f.rel_path, f.last_size AS size_bytes, "
            "(SELECT i.processing_state FROM run_item i WHERE i.source_version_id=v.id ORDER BY i.id DESC LIMIT 1) AS state, "
            "(SELECT i.id FROM run_item i WHERE i.source_version_id=v.id ORDER BY i.id DESC LIMIT 1) AS item_id "
            "FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
            "WHERE v.superseded_by IS NULL ORDER BY f.rel_path").fetchall()
        out = []
        for r in rows:
            try:
                p = probe_version(self.con, self.views, r["id"])
            except Exception as exc:          # a file that cannot be read is listed with its reason
                p = {"error": str(exc)}
            events = self.con.execute("SELECT COUNT(*) FROM video_event WHERE run_item_id=?", (r["item_id"],)).fetchone()[0] \
                if r["item_id"] else 0
            state = r["state"]
            out.append({
                "id": r["id"], "rel_path": r["rel_path"], "size_bytes": r["size_bytes"],
                "duration_s": p.get("duration_s"), "width": p.get("width"), "height": p.get("height"),
                "nominal_fps": p.get("nominal_fps"), "codec": p.get("codec"), "start_utc": p.get("start_utc"),
                "error": p.get("error"), "state": state or "new", "events": events,
                "analysed": state == "done", "selected": state in NOT_ANALYSED and not p.get("error"),
            })
        part = {v: (g["id"], k) for g in Groups(self.m).all() for k, v in enumerate(g["members"], 1)}
        for v in out:
            v["group_id"], v["part"] = part.get(v["id"], (None, None))
        return out

    def overview(self) -> dict:
        """The videos, the merged recordings, and runs of videos that look like parts of one recording."""
        vids = self.items()
        groups = Groups(self.m).all()
        grouped = {v for g in groups for v in g["members"]}
        return {"videos": vids, "groups": groups, "suggestions": suggest(vids, grouped)}

    def merge(self, version_ids: list[int], author: str) -> dict:
        try:
            return Groups(self.m).create(self.items(), version_ids, author)
        except GroupError as exc:
            raise CatalogError(str(exc)) from None

    def separate(self, group_id: int) -> None:
        try:
            Groups(self.m).dissolve(group_id)
        except GroupError as exc:
            raise CatalogError(str(exc)) from None

    def queue(self, version_ids: list[int]) -> list[int]:
        """Make the ticked recordings ready to analyse: a new run for those already analysed, cancelled or failed;
        those still queued stay in their run. Returns the recordings to analyse."""
        ids = sorted({int(v) for v in version_ids})
        if not ids:
            raise CatalogError("Tick at least one video to analyse.")
        groups = Groups(self.m)
        for v in list(ids):                       # a part of a merged recording brings all of its parts, in order
            g = groups.of(v)
            if g:
                ids = [i for i in ids if i not in g["members"]] + g["members"]
        ids = list(dict.fromkeys(ids))
        known = {r["id"]: r for r in self.items()}
        missing = [v for v in ids if v not in known]
        if missing:
            raise CatalogError("Some ticked videos are no longer in the source folders. Refresh the list.")
        bad = [known[v]["rel_path"] for v in ids if known[v]["error"]]
        if bad:
            raise CatalogError("These videos cannot be decoded: " + ", ".join(bad))
        again = [v for v in ids if known[v]["state"] not in ("queued", "running")]
        if again:
            self.m.create_run("reprocess", again)
        return ids

    def thumbnail(self, version_id: int, height: int = 96) -> bytes:
        import cv2
        from .video import frame_at
        cache = self.m.dir / "evidence" / "thumbs" / f"recording_{version_id}_{height}.jpg"
        if cache.is_file():
            return cache.read_bytes()
        p = probe_version(self.con, self.views, version_id)
        dur = p.get("duration_s") or 0
        f = frame_at(self.views._path(version_id), min(1.0, dur * 0.1))
        if f is None:
            raise CatalogError("No frame could be read from this video.")
        img = f.image
        w = max(1, round(img.shape[1] * height / img.shape[0]))
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA),
                               [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise CatalogError("The thumbnail could not be written.")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(buf.tobytes())
        return buf.tobytes()
