"""Where an airborne camera was and what it looked at (phase 7): drone and aircraft video metadata, found on its own.

Rules:
- Two kinds are read, with nothing to set: a DJI subtitle file beside the video (the same name, ending .srt or .SRT)
  and KLV in the video itself (MISB ST 0601, the "UAS Datalink Local Set" of military and police video).
- A subtitle file is telemetry only when most of its entries carry a position. Ordinary subtitles (speech, "[Music]")
  are not telemetry and are ignored.
- KLV packets are checked (their key, their length and their checksum when they carry one); a damaged packet is
  skipped, not guessed at.
- Each sample has its video time (seconds from the start). Between samples a value is interpolated; outside them
  there is none.
- What is read: the platform's position, height and heading, and where the camera looked (the frame centre) and
  how wide (field of view), when the source has them; and the time of day (KLV's own clock, or the DJI date), so a
  recording whose file name carries no time still gets one.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

UAS_LS_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")   # MISB ST 0601 UAS Datalink Local Set
FIELDS = ("lat", "lon", "alt", "rel_alt", "heading", "pitch", "roll", "hfov", "vfov", "center_lat", "center_lon",
          "center_elev", "slant_range", "focal_mm", "zoom")


@dataclass
class Telemetry:
    source: str                                   # "dji-srt" | "klv-0601"
    samples: list[dict] = field(default_factory=list)   # {"t": seconds, field: value, ...}, in time order
    start_utc: str | None = None                  # KLV: time at video time 0, ISO 8601 with Z
    start_local: str | None = None                # DJI: the drone's own clock at the first sample (no zone given)
    platform: str | None = None                   # KLV platform designation / mission id, when given
    skipped: int = 0                              # packets or entries that could not be read

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

def _ber_len(buf: bytes, i: int) -> tuple[int, int]:
    """A BER length at ``buf[i]``: (length, index after it)."""
    b = buf[i]
    if b < 0x80:
        return b, i + 1
    n = b & 0x7F
    if n == 0 or n > 8 or i + 1 + n > len(buf):
        raise ValueError("bad BER length")
    return int.from_bytes(buf[i + 1:i + 1 + n], "big"), i + 1 + n


def _ber_oid(buf: bytes, i: int) -> tuple[int, int]:
    """A BER-OID tag at ``buf[i]``: (tag, index after it)."""
    v = 0
    for _ in range(4):
        b = buf[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, i
    raise ValueError("bad tag")


def checksum(packet: bytes) -> int:
    """MISB ST 0601 checksum: a 16-bit running sum of the bytes, each even byte in the high half."""
    s = 0
    for k, b in enumerate(packet):
        s += b << (8 * ((k + 1) % 2))
    return s & 0xFFFF


def _scale(raw: bytes, lo: float, hi: float, signed: bool = False) -> float | None:
    n = len(raw)
    v = int.from_bytes(raw, "big", signed=signed)
    if signed:
        if v == -(1 << (8 * n - 1)):
            return None                                   # the "error" value of signed fields
        return v * (hi - lo) / 2 / ((1 << (8 * n - 1)) - 1)
    return lo + v * (hi - lo) / ((1 << (8 * n)) - 1)


def parse_0601(packet: bytes) -> dict | None:
    """The values of one UAS Datalink Local Set packet (key included), or None when it is not one or is damaged."""
    if not packet.startswith(UAS_LS_KEY):
        return None
    try:
        n, i = _ber_len(packet, 16)
    except (ValueError, IndexError):
        return None
    end = i + n
    if end > len(packet):
        return None
    out: dict = {}
    try:
        while i < end:
            tag, i = _ber_oid(packet, i)
            ln, i = _ber_len(packet, i)
            v = packet[i:i + ln]
            if len(v) != ln:
                return None
            if tag == 1:                                  # checksum, over everything up to its own value
                if ln != 2 or checksum(packet[:i]) != int.from_bytes(v, "big"):
                    return None
            elif tag == 2 and ln == 8:
                out["unix_us"] = int.from_bytes(v, "big")
            elif tag == 3:
                out["mission"] = v.decode("utf-8", "replace").strip()
            elif tag == 5 and ln == 2:
                out["heading"] = _scale(v, 0, 360)
            elif tag == 6 and ln == 2:
                out["pitch"] = _scale(v, -20, 20, True)
            elif tag == 7 and ln == 2:
                out["roll"] = _scale(v, -50, 50, True)
            elif tag == 10:
                out["platform"] = v.decode("utf-8", "replace").strip()
            elif tag == 13 and ln == 4:
                out["lat"] = _scale(v, -90, 90, True)
            elif tag == 14 and ln == 4:
                out["lon"] = _scale(v, -180, 180, True)
            elif tag == 15 and ln == 2:
                out["alt"] = _scale(v, -900, 19000)
            elif tag == 16 and ln == 2:
                out["hfov"] = _scale(v, 0, 180)
            elif tag == 17 and ln == 2:
                out["vfov"] = _scale(v, 0, 180)
            elif tag == 21 and ln == 4:
                out["slant_range"] = _scale(v, 0, 5_000_000)
            elif tag == 23 and ln == 4:
                out["center_lat"] = _scale(v, -90, 90, True)
            elif tag == 24 and ln == 4:
                out["center_lon"] = _scale(v, -180, 180, True)
            elif tag == 25 and ln == 2:
                out["center_elev"] = _scale(v, -900, 19000)
            i += ln
    except (ValueError, IndexError):
        return None
    return {k: v for k, v in out.items() if v is not None}


def _split_klv(buf: bytes):
    """Every UAS Local Set packet in a data stream's bytes (a PES may hold several, or a packet may be cut short)."""
    i = 0
    while True:
        j = buf.find(UAS_LS_KEY, i)
        if j < 0:
            return
        try:
            n, k = _ber_len(buf, j + 16)
        except (ValueError, IndexError):
            return
        yield buf[j:k + n]
        i = k + n if k + n > j else j + 1


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
            for raw in _split_klv(bytes(pkt)):
                v = parse_0601(raw)
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
                if tel.start_utc is None and v.get("unix_us"):
                    start = datetime.fromtimestamp(v["unix_us"] / 1e6 - smp["t"], tz=timezone.utc)
                    tel.start_utc = start.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                if tel.platform is None and (v.get("platform") or v.get("mission")):
                    tel.platform = v.get("platform") or v.get("mission")
                if len(smp) > 1:
                    tel.samples.append(smp)
    finally:
        c.close()
    if not tel.samples:
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
              center: tuple[float, float] | None = None, platform: str | None = None) -> bytes:
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
        return tel
    if not row["source"]:
        return None
    summ = json.loads(row["summary"])
    return Telemetry(row["source"], json.loads(row["samples"]), summ.get("start_utc"), summ.get("start_local"),
                     summ.get("platform"), summ.get("skipped", 0))
