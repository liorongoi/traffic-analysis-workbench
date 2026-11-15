"""A copy of a recording that any browser can play, for videos whose format the browser cannot play itself.

Analysis never needs this: the bundled FFmpeg (PyAV) decodes AVI, WMV, MKV, H.265 and the rest directly. The player,
however, runs in the browser, which plays only a few formats (H.264 MP4, WebM VP8/VP9/AV1). For any other recording
the player asks for this copy: H.264 in MP4 (VP9 in WebM when this FFmpeg has no H.264 encoder), the same frame size
and the same timestamps as the original, so boxes and events line up with the picture exactly. It is made once, in
the background, kept in the mission's evidence folder and served from there. The original file is never changed.
"""
from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from typing import Callable

TB = Fraction(1, 90000)          # output clock: fine enough for any frame rate, so timestamps survive unchanged

_lock = threading.Lock()
_jobs: dict[str, dict] = {}      # destination path -> {"progress": 0..1, "error": str | None, "thread": Thread}


def encoder() -> tuple[str, str, str]:
    """(codec, container format, file extension) of the copy."""
    import av
    if "libx264" in av.codecs_available:
        return "libx264", "mp4", ".mp4"
    return "libvpx-vp9", "webm", ".webm"


def target(cache_dir: Path, version_id: int) -> Path:
    return Path(cache_dir) / f"recording_{version_id}{encoder()[2]}"


def make(src: Path, dst: Path, progress: Callable[[float], None] | None = None,
         stop: threading.Event | None = None) -> Path:
    """Write the playable copy of ``src`` to ``dst`` (through a temporary file, so a half-made copy is never served)."""
    import av
    codec, fmt, _ = encoder()
    tmp = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(src)) as inp:
        vs = next(s for s in inp.streams if s.type == "video")
        vs.thread_type = "AUTO"
        tb = vs.time_base
        start = vs.start_time if vs.start_time is not None else 0
        dur = float(vs.duration * tb) if vs.duration is not None and tb else (inp.duration or 0) / 1_000_000
        w, h = vs.codec_context.width, vs.codec_context.height
        w, h = w - w % 2, h - h % 2                 # 4:2:0 needs even sides
        with av.open(str(tmp), "w", format=fmt, options={"movflags": "+faststart"} if fmt == "mp4" else {}) as out:
            os_ = out.add_stream(codec, rate=vs.average_rate or vs.guessed_rate or 25)
            os_.width, os_.height, os_.pix_fmt = w, h, "yuv420p"
            os_.time_base = TB
            os_.codec_context.time_base = TB
            if codec == "libx264":
                os_.options = {"preset": "veryfast", "crf": "23"}
            else:
                os_.options = {"deadline": "realtime", "cpu-used": "8", "crf": "32", "b": "0"}
            last = -1
            for fr in inp.decode(vs):
                if stop is not None and stop.is_set():
                    raise RuntimeError("stopped")
                if fr.pts is None:
                    continue
                t = float((fr.pts - start) * tb)
                pts = round(t / TB)
                if pts <= last:                     # duplicate or out-of-order timestamps: keep them increasing
                    pts = last + 1
                last = pts
                g = fr.reformat(width=w, height=h, format="yuv420p")
                g.pts, g.time_base = pts, TB
                for p in os_.encode(g):
                    out.mux(p)
                if progress and dur:
                    progress(min(1.0, t / dur))
            for p in os_.encode():
                out.mux(p)
    tmp.replace(dst)
    if progress:
        progress(1.0)
    return dst


def status(src: Path, dst: Path) -> dict:
    """{"ready": bool, "progress": 0..1, "error": str | None}; starts making the copy if it is not there yet."""
    if dst.is_file():
        return {"ready": True, "progress": 1.0, "error": None}
    key = str(dst)
    with _lock:
        job = _jobs.get(key)
        if job is None or (job["error"] and not job["thread"].is_alive()):
            job = {"progress": 0.0, "error": None}

            def run(j=job):
                try:
                    make(src, dst, lambda f: j.__setitem__("progress", f))
                except Exception as exc:  # noqa: BLE001 (reported to the player, which keeps the frame view)
                    j["error"] = f"{type(exc).__name__}: {exc}"
                    try:
                        dst.with_name(dst.name + ".part").unlink(missing_ok=True)
                    except OSError:
                        pass

            job["thread"] = threading.Thread(target=run, name=f"playable-{dst.stem}", daemon=True)
            _jobs[key] = job
            job["thread"].start()
    if dst.is_file():
        return {"ready": True, "progress": 1.0, "error": None}
    return {"ready": False, "progress": round(job["progress"], 3), "error": job["error"]}
