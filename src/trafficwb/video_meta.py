"""Where an airborne camera was and what it looked at (phase 7): drone and aircraft video metadata, found on its own.

Rules:
- Two kinds are read, with nothing to set: a DJI subtitle file beside the video (the same name, ending .srt or .SRT)
  and KLV in the video itself (STANAG 4609: MISB ST 0601, the "UAS Datalink Local Set" of military and police
  video, with ST 0102 security marking and ST 0903 moving targets; decoded in ``misb``).
- A subtitle file is telemetry only when most of its entries carry a position. Ordinary subtitles (speech, "[Music]")
  are not telemetry and are ignored.
- KLV packets are checked (their key, their length and their checksum when they carry one); a damaged packet is
  skipped, not guessed at.
- Each sample has its video time (seconds from the start). Between samples a value is interpolated; outside them
  there is none.
- What is read: the platform's position, height and heading, and where the camera looked (the frame centre) and
  how wide (field of view), when the source has them; and the time of day (KLV's own clock, or the DJI date), so a
  recording whose file name carries no time still gets one. KLV adds the frame's four corners on the ground, the
  sensor's angles, the height above ground, the security marking (shown, never acted on) and the moving targets the
  aircraft's own tracker reported (pixel boxes and, when sent, ground positions).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import misb

UAS_LS_KEY = misb.UAS_LS_KEY                                     # MISB ST 0601 UAS Datalink Local Set
FIELDS = ("lat", "lon", "alt", "rel_alt", "agl", "heading", "pitch", "roll", "hfov", "vfov", "center_lat",
          "center_lon", "center_elev", "slant_range", "focal_mm", "zoom", "sensor_az", "sensor_el", "sensor_roll",
          "ground_speed", "course",
          "corner1_lat", "corner1_lon", "corner2_lat", "corner2_lon", "corner3_lat", "corner3_lon",
          "corner4_lat", "corner4_lon")   # corners: KLV's frame corners, 1 top left, then clockwise


@dataclass
class Telemetry:
    source: str                                   # "dji-srt" | "klv-0601"
    samples: list[dict] = field(default_factory=list)   # {"t": seconds, field: value, ...}, in time order
    start_utc: str | None = None                  # KLV: time at video time 0, ISO 8601 with Z
    start_local: str | None = None                # DJI: the drone's own clock at the first sample (no zone given)
    platform: str | None = None                   # KLV platform designation / mission id, when given
    skipped: int = 0                              # packets or entries that could not be read
    security: dict | None = None                  # KLV ST 0102 marking (classification, countries), as sent
    targets: list[dict] = field(default_factory=list)   # KLV ST 0903: {"t", "targets": [...], "frame_width", ...}

    def at(self, t: float) -> dict | None:
        """Every field known at video time ``t``, interpolated between the samples around it."""
        s = self.samples
        if not s or t < s[0]["t"] - 0.5 or t > s[-1]["t"] + 0.5:
            return None
        lo, hi = 0, len(s) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if s[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid
        a, b = s[lo], s[hi]
        if t <= a["t"] or a is b:
            return dict(a)
        if t >= b["t"]:
            return dict(b)
        w = (t - a["t"]) / (b["t"] - a["t"])
        out = {"t": round(t, 3)}
        for k in FIELDS:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None:
                v = va if va is not None else vb
            elif k == "heading":
                d = (vb - va + 180) % 360 - 180               # the short way round
                v = (va + w * d) % 360
            else:
                v = va + w * (vb - va)
            if v is not None:
                out[k] = v
        return out

    def summary(self) -> dict:
        s = self.samples
        alts = [x["alt"] for x in s if x.get("alt") is not None]
        rel = [x["rel_alt"] for x in s if x.get("rel_alt") is not None]
        pos = [x for x in s if x.get("lat") is not None]
        return {"source": self.source, "samples": len(s), "skipped": self.skipped, "start_utc": self.start_utc,
                "start_local": self.start_local, "platform": self.platform,
                "from_t": s[0]["t"] if s else None, "to_t": s[-1]["t"] if s else None,
                "first": {"lat": pos[0]["lat"], "lon": pos[0]["lon"]} if pos else None,
                "alt_m": [round(min(alts), 1), round(max(alts), 1)] if alts else None,
                "rel_alt_m": [round(min(rel), 1), round(max(rel), 1)] if rel else None,
                "looks_at": bool(any(x.get("center_lat") is not None for x in s)),
                "corners": bool(any(x.get("corner1_lat") is not None for x in s)),
                "security": self.security, "target_reports": len(self.targets),
                "fields": sorted({k for x in s for k in x if k != "t"})}


# ------------------------------------------------------------------ DJI subtitle files

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_NUM = r"(-?\d+(?:\.\d+)?)"
_DJI = {                                          # newer DJI drones: [key: value] pairs
    "lat": re.compile(r"\[\s*latitude\s*:\s*" + _NUM, re.I),
    "lon": re.compile(r"\[\s*longt?itude\s*:\s*" + _NUM, re.I),
    "rel_alt": re.compile(r"rel_alt\s*:\s*" + _NUM, re.I),
    "alt": re.compile(r"abs_alt\s*:\s*" + _NUM, re.I),
    "focal_mm": re.compile(r"focal_len\s*:\s*" + _NUM, re.I),
    "zoom": re.compile(r"dzoom_ratio\s*:\s*" + _NUM, re.I),
    "heading": re.compile(r"gb_yaw\s*:\s*" + _NUM, re.I),
    "pitch": re.compile(r"gb_pitch\s*:\s*" + _NUM, re.I),
    "roll": re.compile(r"gb_roll\s*:\s*" + _NUM, re.I),
}
_GPS_OLD = re.compile(r"GPS\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*(?:,\s*" + _NUM + r")?\s*\)", re.I)   # lon, lat, alt
_H_OLD = re.compile(r"\bH\s*" + _NUM + r"\s*m", re.I)
_DATE = re.compile(r"(\d{4})[-.](\d{2})[-.](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?")


def sidecar(video: Path) -> Path | None:
    """The subtitle file beside a video: the same name ending .srt (any case)."""
    for p in video.parent.glob(video.stem + ".*"):
        if p.suffix.lower() == ".srt" and p.is_file():
            return p
    return None


def read_srt(path: Path) -> Telemetry | None:
    """DJI telemetry from a subtitle file, or None when the file is ordinary subtitles."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    tel = Telemetry("dji-srt")
    entries = 0
    for b in blocks:
        m = _TIME.search(b)
        if not m:
            continue
        entries += 1
        h, mi, se, ms = (int(x) for x in m.groups()[:4])
        t = h * 3600 + mi * 60 + se + int(m.group(4).ljust(3, "0")[:3]) / 1000
        body = b[m.end():]
        smp: dict = {"t": round(t, 3)}
        for k, rx in _DJI.items():
            g = rx.search(body)
            if g:
                smp[k] = float(g.group(1))
        g = _GPS_OLD.search(body)
        if g and "lat" not in smp:
            smp["lon"], smp["lat"] = float(g.group(1)), float(g.group(2))
            if g.group(3) is not None:
                smp["alt"] = float(g.group(3))
        g = _H_OLD.search(body)
        if g and "rel_alt" not in smp:
            smp["rel_alt"] = float(g.group(1))
        if smp.get("lat") is None or smp.get("lon") is None or not (-90 <= smp["lat"] <= 90 and -180 <= smp["lon"] <= 180) \
                or (smp["lat"] == 0 and smp["lon"] == 0):
            tel.skipped += 1
            continue                                      # no position (or no GPS fix yet): not a sample
        if tel.start_local is None:
            d = _DATE.search(body)
            if d:
                y, mo, dd, hh, mm, ss = (int(x) for x in d.groups()[:6])
                frac = int((d.group(7) or "0").ljust(6, "0")[:6])
                # DJI writes the drone's local time and no zone: kept as it is, for the mission's zone to place.
                start = datetime(y, mo, dd, hh, mm, ss, frac)
                tel.start_local = (start - timedelta(seconds=smp["t"])).isoformat(timespec="milliseconds")
        tel.samples.append(smp)
    if entries == 0 or len(tel.samples) < max(2, entries // 2):
        return None                                       # captions, or too little to be telemetry
    tel.samples.sort(key=lambda s: s["t"])
    return tel


# ------------------------------------------------------------------ KLV (MISB ST 0601)

def checksum(packet: bytes) -> int:
    """MISB ST 0601 checksum: a 16-bit running sum of the bytes, each even byte in the high half."""
    return misb.checksum(packet)


# ST 0601 names (``misb``) to the short names every flight-data source shares
_SHORT = {"sensor_latitude": "lat", "sensor_longitude": "lon", "sensor_true_altitude": "alt",
          "altitude_agl": "agl", "platform_heading": "heading", "platform_pitch": "pitch", "platform_roll": "roll",
          "sensor_hfov": "hfov", "sensor_vfov": "vfov", "frame_center_latitude": "center_lat",
          "frame_center_longitude": "center_lon", "frame_center_elevation": "center_elev", "slant_range": "slant_range",
          "sensor_relative_azimuth": "sensor_az", "sensor_relative_elevation": "sensor_el",
          "sensor_relative_roll": "sensor_roll", "platform_ground_speed": "ground_speed",
          "platform_course_angle": "course", "mission_id": "mission", "platform_designation": "platform"}


def parse_0601(packet: bytes) -> dict | None:
    """The values of one UAS Datalink Local Set packet (key included) under the short names, or None when it is not
    one or is damaged. ``misb.decode_0601`` gives every item under its ST 0601 name."""
    v = misb.decode_0601(packet)
    if v is None:
        return None
    out = {short: v[name] for name, short in _SHORT.items() if v.get(name) is not None}
    if v.get("platform_pitch_full") is not None:
        out["pitch"] = v["platform_pitch_full"]
    if v.get("platform_roll_full") is not None:
        out["roll"] = v["platform_roll_full"]
    if v.get("precision_time"):
        t = datetime.fromisoformat(v["precision_time"].replace("Z", "+00:00"))
        out["unix_us"] = round(t.timestamp() * 1e6)
    for k in range(1, 5):
        if v.get(f"corner_latitude_{k}") is not None and v.get(f"corner_longitude_{k}") is not None:
            out[f"corner{k}_lat"], out[f"corner{k}_lon"] = v[f"corner_latitude_{k}"], v[f"corner_longitude_{k}"]
    if v.get("security"):
        out["security"] = v["security"]
    if v.get("vmti"):
        out["vmti"] = v["vmti"]
    return out


def read_klv(path: Path, limit_s: float | None = None) -> Telemetry | None:
    """KLV metadata from the video's own data stream, or None when it has none."""
    import av
    try:
        c = av.open(str(path))
    except Exception:  # noqa: BLE001 - not a container PyAV can read: no KLV
        return None
    tel = Telemetry("klv-0601")
    try:
        data = [s for s in c.streams if s.type == "data"]
        if not data:
            return None
        video = c.streams.video[0] if c.streams.video else None
        t0 = None
        if video is not None and video.start_time is not None and video.time_base:
            t0 = float(video.start_time * video.time_base)
        for pkt in c.demux(*data):
            if pkt.size == 0:
                continue
            pts = pkt.pts if pkt.pts is not None else pkt.dts
            t = float(pts * pkt.time_base) if pts is not None and pkt.time_base else None
            for kind, raw in misb.split(bytes(pkt)):
                if kind == "st0601":
                    v = parse_0601(raw)
                elif kind == "st0903":
                    vm = misb.decode_vmti(raw)
                    v = None if vm is None else {"vmti": vm}
                else:
                    sec = misb.decode_security(raw)
                    v = None if sec is None else {"security": sec}
                if v is None:
                    tel.skipped += 1
                    continue
                if t is None:
                    continue
                if t0 is None:
                    t0 = t
                if limit_s is not None and t - t0 > limit_s:
                    break
                smp = {"t": round(max(0.0, t - t0), 3)} | {k: v[k] for k in FIELDS if k in v}
                if v.get("security") and tel.security is None:
                    tel.security = v["security"]
                if v.get("vmti"):
                    vm = v["vmti"]
                    tel.targets.append({"t": smp["t"]} | {k: vm[k] for k in ("frame_width", "frame_height",
                                                                              "frame_number", "targets") if k in vm})
                if tel.start_utc is None and v.get("unix_us"):
                    start = datetime.fromtimestamp(v["unix_us"] / 1e6 - smp["t"], tz=timezone.utc)
                    tel.start_utc = start.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                if tel.platform is None and (v.get("platform") or v.get("mission")):
                    tel.platform = v.get("platform") or v.get("mission")
                if len(smp) > 1:
                    tel.samples.append(smp)
    finally:
        c.close()
    if not tel.samples and not tel.targets:
        return None
    tel.samples.sort(key=lambda s: s["t"])
    return tel


def find(video: Path) -> Telemetry | None:
    """The metadata of a video, from whichever source it has: KLV inside it first, then a DJI subtitle file."""
    video = Path(video)
    tel = read_klv(video)
    if tel is not None:
        return tel
    srt = sidecar(video)
    return read_srt(srt) if srt else None


# ------------------------------------------------------------------ writing KLV (tests and samples)

def _ber(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _signed(v: float, lo: float, hi: float, n: int) -> bytes:
    return int(round(v * ((1 << (8 * n - 1)) - 1) / ((hi - lo) / 2))).to_bytes(n, "big", signed=True)


def _unsigned(v: float, lo: float, hi: float, n: int) -> bytes:
    return int(round((v - lo) * ((1 << (8 * n)) - 1) / (hi - lo))).to_bytes(n, "big")


def make_0601(unix_us: int, lat: float, lon: float, alt: float, heading: float = 0.0, hfov: float | None = None,
              center: tuple[float, float] | None = None, platform: str | None = None,
              corners: list[tuple[float, float]] | None = None, center_elev: float | None = None,
              extra: list[tuple[int, bytes]] | None = None) -> bytes:
    """One UAS Datalink Local Set packet with its checksum (used by the tests to make KLV video)."""
    items = [(2, unix_us.to_bytes(8, "big"))]
    if platform:
        items.append((10, platform.encode()))
    items += [(5, _unsigned(heading, 0, 360, 2)), (13, _signed(lat, -90, 90, 4)), (14, _signed(lon, -180, 180, 4)),
              (15, _unsigned(alt, -900, 19000, 2))]
    if hfov is not None:
        items.append((16, _unsigned(hfov, 0, 180, 2)))
    if center:
        items += [(23, _signed(center[0], -90, 90, 4)), (24, _signed(center[1], -180, 180, 4))]
    if center_elev is not None:
        items.append((25, _unsigned(center_elev, -900, 19000, 2)))
    if center and corners:                                    # as offsets from the centre (tags 26 to 33)
        for k, (clat, clon) in enumerate(corners):
            items += [(26 + 2 * k, _signed(clat - center[0], -0.075, 0.075, 2)),
                      (27 + 2 * k, _signed(clon - center[1], -0.075, 0.075, 2))]
    items += extra or []                                      # e.g. (48, security set), (74, VMTI set)
    body = b"".join(bytes([t]) + _ber(len(v)) + v for t, v in items)
    body_len = len(body) + 4                                  # with the checksum item (tag, length, 2 bytes)
    head = UAS_LS_KEY + _ber(body_len) + body + bytes([1, 2])
    return head + struct.pack(">H", checksum(head))


# ------------------------------------------------------------------ kept with the mission

MAX_RATE = 5.0                                     # samples a second kept


def thin(samples: list[dict], rate: float = MAX_RATE) -> list[dict]:
    out, last = [], None
    for x in samples:
        if last is None or x["t"] - last >= 1.0 / rate - 1e-6:
            out.append({k: (round(v, 7) if isinstance(v, float) else v) for k, v in x.items()})
            last = x["t"]
    if samples and out[-1]["t"] != samples[-1]["t"]:
        out.append(samples[-1])
    return out


def stored(con, version_id: int, path: Path | None = None) -> Telemetry | None:
    """A recording's metadata, read from its file once and then kept in ``video_telemetry``."""
    import json
    from .store import utcnow
    row = con.execute("SELECT * FROM video_telemetry WHERE source_version_id=?", (version_id,)).fetchone()
    if row is None:
        if path is None:
            return None
        try:
            tel = find(path)
        except Exception:  # noqa: BLE001 - unreadable metadata never stops the analysis
            tel = None
        con.execute("INSERT OR REPLACE INTO video_telemetry(source_version_id,source,summary,samples,read_at) "
                    "VALUES (?,?,?,?,?)",
                    (version_id, tel.source if tel else None, json.dumps(tel.summary()) if tel else None,
                     json.dumps(thin(tel.samples)) if tel else None, utcnow()))
        if tel is not None and tel.targets:
            summ = tel.summary() | {"vmti": thin(tel.targets)}
            con.execute("UPDATE video_telemetry SET summary=? WHERE source_version_id=?", (json.dumps(summ), version_id))
        return tel
    if not row["source"]:
        return None
    summ = json.loads(row["summary"])
    return Telemetry(row["source"], json.loads(row["samples"]), summ.get("start_utc"), summ.get("start_local"),
                     summ.get("platform"), summ.get("skipped", 0), summ.get("security"), summ.get("vmti") or [])
