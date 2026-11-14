"""Drone and aircraft video on the map: platform data the operator types, and the whole video registered ahead.

Rules:
- Platform data can be typed for a recording: position, height above sea level or above the ground, heading,
  camera angle (pitch, roll) and field of view. A typed value wins over the same value in the video's own flight
  data (the operator knows better); what is not typed comes from the flight data. A height above sea level is
  turned into a height above the ground with the terrain file (DTM) under the platform; with no terrain the ground
  is taken at sea level and the page says so.
- A square on the map can be drawn for a recording: when the video has no position at all, the first frame is
  searched for on the map inside that square only (``georeg.locate``), which is faster and far less often wrong.
- "Prepare" registers the whole video ahead, frame after frame (``georeg.DroneRegistrar``, waiting for every map
  match: the offline, more accurate way), and keeps the result in the mission's evidence folder
  (``evidence/georeg/<recording>.json``). Playing it back is then instant, frame by frame, at the video's own pace.
  A result belongs to the map it was made with; another map needs another preparation.
- Every prepared moment says how it was placed: lined up with the map (registered, green) or flight data only
  (red), or lost (no outline).
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from .store import utcnow

PLATFORM_FIELDS = {"lat": (-90, 90), "lon": (-180, 180), "alt_asl": (-500, 30000), "agl": (1, 30000),
                   "heading": (-360, 360), "pitch": (-90, 0), "roll": (-90, 90), "hfov": (1, 170)}
PREPARE_FPS = 5.0
RELOCATE_S = 2.0                    # with no fix, search the square again this often


class PlatformError(ValueError):
    pass


# ------------------------------------------------------------------ typed platform data

def platform_get(con, version_id: int) -> dict | None:
    r = con.execute("SELECT data, author, created_at FROM video_platform WHERE source_version_id=? ORDER BY id DESC "
                    "LIMIT 1", (version_id,)).fetchone()
    if r is None:
        return None
    d = json.loads(r[0])
    return None if not d else d | {"author": r[1], "created_at": r[2]}


def clean_platform(data: dict) -> dict:
    out = {}
    for k, (lo, hi) in PLATFORM_FIELDS.items():
        v = data.get(k)
        if v is None or v == "":
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise PlatformError(f"{k} must be a number.") from None
        if not lo <= v <= hi:
            raise PlatformError(f"{k} must be between {lo} and {hi}.")
        out[k] = v
    if ("lat" in out) != ("lon" in out):
        raise PlatformError("Give both the latitude and the longitude, or neither.")
    if "alt_asl" in out and "agl" in out:
        raise PlatformError("Give the height above sea level or above the ground, not both.")
    box = data.get("search_box")
    if box:
        try:
            top, left, bottom, right = (float(v) for v in box)
        except (TypeError, ValueError):
            raise PlatformError("The search square is top, left, bottom, right.") from None
        top, bottom = max(top, bottom), min(top, bottom)
        left, right = min(left, right), max(left, right)
        if not (-90 <= bottom < top <= 90 and -180 <= left < right <= 180):
            raise PlatformError("The search square is not on the earth.")
        out["search_box"] = [top, left, bottom, right]
    return out


def platform_set(con, version_id: int, data: dict, author: str) -> dict | None:
    clean = clean_platform(data)
    con.execute("INSERT INTO video_platform(source_version_id,data,author,created_at) VALUES (?,?,?,?)",
                (version_id, json.dumps(clean), author, utcnow()))
    return platform_get(con, version_id)


def _terrain_at(dtm_path, lat, lon):
    if not dtm_path:
        return None
    from .terrain import TerrainError, load_terrain
    try:
        v = load_terrain(dtm_path).height_at(lat, lon)
    except TerrainError:
        return None
    return float(v) if np.isfinite(v) else None


def sample_at(con, tel, typed: dict | None, t: float, dtm_path: str | None = None) -> tuple[dict | None, float | None]:
    """The flight data at ``t`` with the typed platform data over it, and the ground's height under the platform
    (from the terrain file), or (None, None) when there is no position."""
    s = dict(tel.at(t) or {}) if tel is not None else {}
    if typed:
        for k in ("lat", "lon", "heading", "pitch", "roll", "hfov"):
            if typed.get(k) is not None:
                s[k] = typed[k]
        if typed.get("agl") is not None:
            s["rel_alt"] = typed["agl"]
            s.pop("alt", None)
            s.pop("agl", None)
            s.pop("center_lat", None)                  # a typed height: the corners and centre no longer agree
        if typed.get("alt_asl") is not None:
            s["alt"] = typed["alt_asl"]
            s.pop("rel_alt", None)
            s.pop("agl", None)
        if any(typed.get(k) is not None for k in ("lat", "heading", "pitch", "hfov", "agl", "alt_asl")):
            for i in range(1, 5):                      # KLV corners would ignore what was typed
                s.pop(f"corner{i}_lat", None)
                s.pop(f"corner{i}_lon", None)
    if s.get("lat") is None or s.get("lon") is None:
        return None, None
    if s.get("pitch") is None and s.get("rel_alt") is not None:
        s["pitch"] = -90.0
    return s, _terrain_at(dtm_path, s["lat"], s["lon"])


# ------------------------------------------------------------------ the whole video, ahead

_jobs: dict = {}
_jobs_lock = threading.Lock()


def track_path(m, version_id: int) -> Path:
    return m.dir / "evidence" / "georeg" / f"{version_id}.json"


def load_track(m, version_id: int, map_path: str | None = None) -> dict | None:
    p = track_path(m, version_id)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if map_path is not None and d.get("map_path") != map_path:
        return None
    return d


def job_status(mission_id: str, version_id: int) -> dict | None:
    with _jobs_lock:
        j = _jobs.get((mission_id, version_id))
        return dict(j) if j else None


def start(open_mission, mission_id: str, version_id: int, map_path: str, corners: list | None,
          dtm_path: str | None, author: str) -> dict:
    """Prepare in the background (``open_mission()`` gives the mission on the worker's own connection)."""
    key = (mission_id, version_id)
    with _jobs_lock:
        if _jobs.get(key, {}).get("state") == "running":
            return dict(_jobs[key])
        _jobs[key] = {"state": "running", "done": 0, "total": None, "started_at": utcnow(), "error": None,
                      "author": author}

    def go():
        try:
            m = open_mission()
            prepare(m, version_id, map_path, corners, dtm_path,
                    progress=lambda d, n: _jobs[key].update(done=d, total=n))
            _jobs[key].update(state="done")
        except Exception as exc:  # noqa: BLE001 - shown to the operator
            _jobs[key].update(state="failed", error=str(exc))

    threading.Thread(target=go, daemon=True, name=f"georeg-{version_id}").start()
    return dict(_jobs[key])


def prepare(m, version_id: int, map_path: str, corners: list | None, dtm_path: str | None,
            fps: float = PREPARE_FPS, progress=None) -> dict:
    """Register every ``1/fps`` seconds of the recording to the map and keep the result."""
    from .georeg import DroneRegistrar, Matcher, is_registered, locate
    from .video import frames
    from .video_geo import load_map
    from .video_meta import stored
    from .video_review import VideoService
    svc = VideoService(m)
    path = svc.path(version_id)
    if not path.is_file():
        raise PlatformError(f"{path.name} is no longer in the source folder.")
    r = m.con.execute("SELECT width, height, duration_s FROM video_probe WHERE source_version_id=?",
                      (version_id,)).fetchone()
    w, h, dur = int(r[0]), int(r[1]), float(r[2] or 0)
    gm = load_map(map_path, corners)
    tel = stored(m.con, version_id, path)
    typed = platform_get(m.con, version_id)
    box = (typed or {}).get("search_box")
    region = None
    if box:
        e0, n0 = gm.frame.to_en(box[0], box[1])
        e1, n1 = gm.frame.to_en(box[2], box[3])
        region = (float(min(e0, e1)), float(min(n0, n1)), float(max(e0, e1)), float(max(n0, n1)))
    matcher = Matcher()
    reg = DroneRegistrar(gm, matcher, snap_every_s=1.0, async_snap=False)
    total = max(1, int(dur * fps)) if dur else None
    out, started, last_try, t0 = [], False, -1e9, time.time()
    for f, _ in frames(path, 1.0 / fps):
        t = f.t
        img = f.image
        s, ground = sample_at(m.con, tel, typed, t, dtm_path)
        reg.ground_alt = ground
        if not started:
            G0 = None
            if s is None and region is not None and t - last_try >= RELOCATE_S:
                last_try = t
                agl, hf = (typed or {}).get("agl"), (typed or {}).get("hfov")
                mpp = 2 * agl * math.tan(math.radians(hf) / 2) / w if agl and hf else 0.3
                snap = locate(img, gm, matcher, mpp, region)
                G0 = snap.G
            if s is None and G0 is None:
                out.append({"t": round(t, 3), "G": None, "how": "lost", "registered": False})
                if progress:
                    progress(len(out), total)
                continue
            fix = reg.start(img, G0, s, t)
            started = fix.G is not None
            if not started:
                reg.G_key = None
        else:
            fix = reg.step(img, t, s)
        ok = is_registered(fix, reg.trust_flight_data_s)
        how = "lost" if fix.G is None else ("map-match" if ok else "flight-data")
        out.append({"t": round(t, 3), "G": None if fix.G is None else [round(float(v), 9) for v in fix.G.ravel()],
                    "how": how, "registered": ok})
        if progress:
            progress(len(out), total)
    reg.close()
    res = {"map_path": map_path, "corners": corners, "dtm_path": dtm_path, "origin": [gm.frame.lat0, gm.frame.lon0],
           "width": w, "height": h, "fps": fps, "made_at": utcnow(), "seconds": round(time.time() - t0, 1),
           "matcher": matcher.kind, "summary": {k: sum(x["how"] == k for x in out) for k in
                                                ("map-match", "flight-data", "lost")},
           "samples": out}
    p = track_path(m, version_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(p)
    return res


def track_sample(track: dict, t: float) -> dict | None:
    """The prepared moment nearest ``t`` (within one step), or None."""
    s = track.get("samples") or []
    if not s:
        return None
    ts = [x["t"] for x in s]
    i = int(np.searchsorted(ts, t))
    best = min((j for j in (i - 1, i) if 0 <= j < len(s)), key=lambda j: abs(ts[j] - t))
    return s[best] if abs(ts[best] - t) <= 1.0 / track.get("fps", PREPARE_FPS) + 1e-6 else None


def track_outlines(track: dict) -> list[dict]:
    """Each prepared moment as an outline on the map, for playing back in the browser."""
    from .georeg import LocalFrame, ground_parts
    from .video_geo import VIEW_RANGE_M
    fr = LocalFrame(*track["origin"])
    w, h = track["width"], track["height"]
    out = []
    for x in track.get("samples") or []:
        row = {"t": x["t"], "how": x["how"], "registered": x["registered"], "near": None, "far": None}
        if x["G"] is not None:
            parts = ground_parts(np.array(x["G"]).reshape(3, 3), w, h, VIEW_RANGE_M, max_px_m=None)
            if parts is not None:
                for k in ("near", "far"):
                    lat, lon = fr.to_latlon(parts[k][:, 0], parts[k][:, 1])
                    row[k] = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]
        out.append(row)
    return out
