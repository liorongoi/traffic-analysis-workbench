"""STANAG 4609 metadata: the MISB KLV sets that motion imagery from military, police and survey aircraft carries.

Read here, from the bytes of a video's KLV data stream:
- MISB ST 0601 (UAS Datalink Local Set), every item of ST 0601.17 that is a number, text or time, including the
  extended items encoded with ST 1201 (IMAPB); nested sets are decoded where a reader can use them, otherwise kept
  as their raw bytes in hex.
- MISB ST 0102 (Security Metadata), nested in ST 0601 (item 48) or on its own.
- MISB ST 0903 (VMTI, moving targets), nested in ST 0601 (item 74) or on its own: each target's pixel box and,
  when sent, its ground position.
- MISB ST 1201 (IMAPB) and BER / BER-OID encodings, which the sets above use.

Rules: a packet whose key, length or checksum is wrong is refused whole; an item that cannot be read is left out
and counted, not guessed at; the "out of range" values of signed items read as nothing.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

UAS_LS_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")       # ST 0601
SECURITY_LS_KEY = bytes.fromhex("060E2B34020301010E01030302000000")  # ST 0102 local set
VMTI_LS_KEY = bytes.fromhex("060E2B34020B01010E01030306000000")      # ST 0903
KEYS = {UAS_LS_KEY: "st0601", SECURITY_LS_KEY: "st0102", VMTI_LS_KEY: "st0903"}


# ------------------------------------------------------------------ encodings

def ber_len(buf: bytes, i: int) -> tuple[int, int]:
    b = buf[i]
    if b < 0x80:
        return b, i + 1
    n = b & 0x7F
    if n == 0 or n > 8 or i + 1 + n > len(buf):
        raise ValueError("bad BER length")
    return int.from_bytes(buf[i + 1:i + 1 + n], "big"), i + 1 + n


def ber_oid(buf: bytes, i: int) -> tuple[int, int]:
    v = 0
    for _ in range(5):
        b = buf[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if b < 0x80:
            return v, i
    raise ValueError("bad BER-OID")


def ber_len_bytes(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def ber_oid_bytes(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F))
        v >>= 7
    return bytes(reversed(out))


def checksum(packet: bytes) -> int:
    """ST 0601 / ST 0903 checksum: a 16-bit running sum, each even byte in the high half."""
    s = 0
    for k, b in enumerate(packet):
        s += b << (8 * ((k + 1) % 2))
    return s & 0xFFFF


def uint_map(raw: bytes, lo: float, hi: float) -> float:
    return lo + int.from_bytes(raw, "big") * (hi - lo) / ((1 << (8 * len(raw))) - 1)


def int_map(raw: bytes, lo: float, hi: float) -> float | None:
    n = len(raw)
    v = int.from_bytes(raw, "big", signed=True)
    if v == -(1 << (8 * n - 1)):
        return None                                   # "out of range" / error
    return v * (hi - lo) / 2 / ((1 << (8 * n - 1)) - 1)


def imapb(raw: bytes, a: float, b: float) -> float | None:
    """ST 1201 IMAPB: a float in [a, b] from ``len(raw)`` bytes; special values (top bit set) read as nothing."""
    L = len(raw)
    if L == 0:
        return None
    x = int.from_bytes(raw, "big")
    if raw[0] & 0x80:
        return None
    b_pow = math.ceil(math.log2(b - a))
    d_pow = 8 * L - 1
    s_r = 2.0 ** (b_pow - d_pow)
    s_f = 2.0 ** (d_pow - b_pow)
    z = 0.0
    if a < 0 and s_f * a != math.floor(s_f * a):
        z = s_f * a - math.floor(s_f * a)
    return s_r * (x - z) + a


def imapb_bytes(v: float, a: float, b: float, L: int) -> bytes:
    b_pow = math.ceil(math.log2(b - a))
    d_pow = 8 * L - 1
    s_f = 2.0 ** (d_pow - b_pow)
    z = 0.0
    if a < 0 and s_f * a != math.floor(s_f * a):
        z = s_f * a - math.floor(s_f * a)
    return int(math.floor(s_f * (v - a) + z)).to_bytes(L, "big")


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace").rstrip("\x00").strip()


def _time(raw: bytes) -> str | None:
    if len(raw) != 8:
        return None
    us = int.from_bytes(raw, "big")
    try:
        return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (OverflowError, ValueError, OSError):
        return None


# ------------------------------------------------------------------ ST 0601

# tag: (name, kind, lo, hi). kinds: u (unsigned mapped), s (signed mapped), i (IMAPB), n (plain unsigned),
# z (plain signed), t (text), T (microsecond time), x (kept as hex).
U, S, I, N, Z, TXT, TIME, X = "u", "s", "i", "n", "z", "t", "T", "x"
ST0601: dict[int, tuple] = {
    2: ("precision_time", TIME), 3: ("mission_id", TXT), 4: ("platform_tail_number", TXT),
    5: ("platform_heading", U, 0, 360), 6: ("platform_pitch", S, -20, 20), 7: ("platform_roll", S, -50, 50),
    8: ("platform_true_airspeed", N), 9: ("platform_indicated_airspeed", N), 10: ("platform_designation", TXT),
    11: ("image_source_sensor", TXT), 12: ("image_coordinate_system", TXT),
    13: ("sensor_latitude", S, -90, 90), 14: ("sensor_longitude", S, -180, 180),
    15: ("sensor_true_altitude", U, -900, 19000), 16: ("sensor_hfov", U, 0, 180), 17: ("sensor_vfov", U, 0, 180),
    18: ("sensor_relative_azimuth", U, 0, 360), 19: ("sensor_relative_elevation", S, -180, 180),
    20: ("sensor_relative_roll", U, 0, 360), 21: ("slant_range", U, 0, 5_000_000), 22: ("target_width", U, 0, 10_000),
    23: ("frame_center_latitude", S, -90, 90), 24: ("frame_center_longitude", S, -180, 180),
    25: ("frame_center_elevation", U, -900, 19000),
    26: ("offset_corner_latitude_1", S, -0.075, 0.075), 27: ("offset_corner_longitude_1", S, -0.075, 0.075),
    28: ("offset_corner_latitude_2", S, -0.075, 0.075), 29: ("offset_corner_longitude_2", S, -0.075, 0.075),
    30: ("offset_corner_latitude_3", S, -0.075, 0.075), 31: ("offset_corner_longitude_3", S, -0.075, 0.075),
    32: ("offset_corner_latitude_4", S, -0.075, 0.075), 33: ("offset_corner_longitude_4", S, -0.075, 0.075),
    34: ("icing_detected", N), 35: ("wind_direction", U, 0, 360), 36: ("wind_speed", U, 0, 100),
    37: ("static_pressure", U, 0, 5000), 38: ("density_altitude", U, -900, 19000), 39: ("outside_air_temperature", Z),
    40: ("target_location_latitude", S, -90, 90), 41: ("target_location_longitude", S, -180, 180),
    42: ("target_location_elevation", U, -900, 19000), 43: ("target_track_gate_width", N),
    44: ("target_track_gate_height", N), 45: ("target_error_ce90", U, 0, 4095), 46: ("target_error_le90", U, 0, 4095),
    47: ("generic_flag_data", N), 49: ("differential_pressure", U, 0, 5000), 50: ("platform_angle_of_attack", S, -20, 20),
    51: ("platform_vertical_speed", S, -180, 180), 52: ("platform_sideslip_angle", S, -20, 20),
    53: ("airfield_barometric_pressure", U, 0, 5000), 54: ("airfield_elevation", U, -900, 19000),
    55: ("relative_humidity", U, 0, 100), 56: ("platform_ground_speed", N), 57: ("ground_range", U, 0, 5_000_000),
    58: ("platform_fuel_remaining", U, 0, 10_000), 59: ("platform_call_sign", TXT), 60: ("weapon_load", N),
    61: ("weapon_fired", N), 62: ("laser_prf_code", N), 63: ("sensor_fov_name", N),
    64: ("platform_magnetic_heading", U, 0, 360), 65: ("uas_ls_version", N),
    67: ("alternate_platform_latitude", S, -90, 90), 68: ("alternate_platform_longitude", S, -180, 180),
    69: ("alternate_platform_altitude", U, -900, 19000), 70: ("alternate_platform_name", TXT),
    71: ("alternate_platform_heading", U, 0, 360), 72: ("event_start_time", TIME), 73: ("rvt_local_set", X),
    75: ("sensor_ellipsoid_height", U, -900, 19000), 76: ("alternate_platform_ellipsoid_height", U, -900, 19000),
    77: ("operational_mode", N), 78: ("frame_center_height_above_ellipsoid", U, -900, 19000),
    79: ("sensor_north_velocity", S, -327, 327), 80: ("sensor_east_velocity", S, -327, 327),
    81: ("image_horizon_pixel_pack", X),
    82: ("corner_latitude_1", S, -90, 90), 83: ("corner_longitude_1", S, -180, 180),
    84: ("corner_latitude_2", S, -90, 90), 85: ("corner_longitude_2", S, -180, 180),
    86: ("corner_latitude_3", S, -90, 90), 87: ("corner_longitude_3", S, -180, 180),
    88: ("corner_latitude_4", S, -90, 90), 89: ("corner_longitude_4", S, -180, 180),
    90: ("platform_pitch_full", S, -90, 90), 91: ("platform_roll_full", S, -90, 90),
    92: ("platform_angle_of_attack_full", S, -90, 90), 93: ("platform_sideslip_angle_full", S, -180, 180),
    94: ("miis_core_identifier", X), 95: ("sar_motion_imagery_local_set", X),
    96: ("target_width_extended", I, 0, 1_500_000), 97: ("range_image_local_set", X),
    98: ("geo_registration_local_set", X), 99: ("composite_imaging_local_set", X), 100: ("segment_local_set", X),
    101: ("amend_local_set", X), 102: ("sdcc_flp", X),
    103: ("density_altitude_extended", I, -900, 40_000), 104: ("sensor_ellipsoid_height_extended", I, -900, 40_000),
    105: ("alternate_platform_ellipsoid_height_extended", I, -900, 40_000), 106: ("stream_designator", TXT),
    107: ("operational_base", TXT), 108: ("broadcast_source", TXT),
    109: ("range_to_recovery_location", I, 0, 21_000), 110: ("time_airborne", N), 111: ("propulsion_unit_speed", N),
    112: ("platform_course_angle", I, 0, 360), 113: ("altitude_agl", I, -900, 40_000),
    114: ("radar_altimeter", I, -900, 40_000), 115: ("control_command", X), 116: ("control_command_verification", X),
    117: ("sensor_azimuth_rate", I, -1000, 1000), 118: ("sensor_elevation_rate", I, -1000, 1000),
    119: ("sensor_roll_rate", I, -1000, 1000), 120: ("onboard_storage_percent_full", I, 0, 100),
    121: ("active_wavelength_list", X), 122: ("country_codes", X), 123: ("navsats_in_view", N),
    124: ("positioning_method_source", N), 125: ("platform_status", N), 126: ("sensor_control_mode", N),
    127: ("sensor_frame_rate_pack", X), 128: ("wavelengths_list", X), 129: ("target_id", TXT),
    130: ("airbase_locations", X), 131: ("takeoff_time", TIME), 132: ("transmission_frequency", I, 1, 99_999),
    133: ("onboard_storage_capacity", N), 134: ("zoom_percentage", I, 0, 100), 135: ("communications_method", TXT),
    136: ("leap_seconds", Z), 137: ("correction_offset", Z), 138: ("payload_list", X), 139: ("active_payloads", X),
    140: ("weapons_stores", X), 141: ("waypoint_list", X), 142: ("view_domain", X),
    143: ("metadata_substream_id", X),
}
SECURITY_CLASSES = {1: "UNCLASSIFIED", 2: "RESTRICTED", 3: "CONFIDENTIAL", 4: "SECRET", 5: "TOP SECRET"}


def _value(spec: tuple, raw: bytes):
    kind = spec[1]
    if kind == U:
        return uint_map(raw, spec[2], spec[3]) if raw else None
    if kind == S:
        return int_map(raw, spec[2], spec[3]) if raw else None
    if kind == I:
        return imapb(raw, spec[2], spec[3])
    if kind == N:
        return int.from_bytes(raw, "big") if raw else None
    if kind == Z:
        return int.from_bytes(raw, "big", signed=True) if raw else None
    if kind == TXT:
        return _text(raw)
    if kind == TIME:
        return _time(raw)
    return raw.hex()


def _items(buf: bytes, i: int, end: int):
    while i < end:
        tag, i = ber_oid(buf, i)
        ln, i = ber_len(buf, i)
        if i + ln > end:
            raise ValueError("item runs past the set")
        yield tag, buf[i:i + ln], i
        i += ln


def _set_body(packet: bytes, key: bytes) -> tuple[int, int]:
    if not packet.startswith(key):
        raise ValueError("wrong key")
    n, i = ber_len(packet, 16)
    if i + n > len(packet):
        raise ValueError("cut short")
    return i, i + n


def decode_0601(packet: bytes, strict: bool = True) -> dict | None:
    """Every ST 0601 item of one packet (key included) by name, or None when the packet is not one or is damaged.

    ``strict=False`` reads a packet whose checksum is wrong (and says so in ``_bad_checksum``); for inspection only.
    Nested sets come back decoded: ``security`` (ST 0102) and ``vmti`` (ST 0903). Offset corners are turned into
    full corners (``corner_latitude_1`` ...) when the packet has no full ones. ``_unread`` counts items not read.
    """
    try:
        i, end = _set_body(packet, UAS_LS_KEY)
        out: dict = {}
        unread = 0
        for tag, raw, at in _items(packet, i, end):
            if tag == 1:
                if len(raw) != 2 or checksum(packet[:at]) != int.from_bytes(raw, "big"):
                    if strict:
                        return None
                    out["_bad_checksum"] = True
                continue
            if tag == 48:
                out["security"] = decode_security_items(raw)
                continue
            if tag == 74:
                out["vmti"] = decode_vmti_items(raw)
                continue
            spec = ST0601.get(tag)
            if spec is None:
                unread += 1
                continue
            try:
                v = _value(spec, raw)
            except (ValueError, OverflowError):
                v = None
            if v is None:
                unread += 1
            else:
                out[spec[0]] = v
    except (ValueError, IndexError):
        return None
    if "target_track_gate_width" in out:
        out["target_track_gate_width"] *= 2
    if "target_track_gate_height" in out:
        out["target_track_gate_height"] *= 2
    clat, clon = out.get("frame_center_latitude"), out.get("frame_center_longitude")
    if clat is not None and clon is not None:
        for k in range(1, 5):
            dlat, dlon = out.get(f"offset_corner_latitude_{k}"), out.get(f"offset_corner_longitude_{k}")
            if dlat is not None and dlon is not None and out.get(f"corner_latitude_{k}") is None:
                out[f"corner_latitude_{k}"], out[f"corner_longitude_{k}"] = clat + dlat, clon + dlon
        v = out.get("vmti")
        if v:
            for t in v.get("targets", []):
                if t.get("offset_latitude") is not None and t.get("latitude") is None:
                    t["latitude"], t["longitude"] = clat + t["offset_latitude"], clon + t["offset_longitude"]
    if unread:
        out["_unread"] = unread
    return out


# ------------------------------------------------------------------ ST 0102

ST0102 = {1: "classification", 2: "classifying_country_coding", 3: "classifying_country", 4: "sci_shi",
          5: "caveats", 6: "releasing_instructions", 7: "classified_by", 8: "derived_from",
          9: "classification_reason", 10: "declassification_date", 11: "classification_system",
          12: "object_country_coding", 13: "object_country_codes", 14: "classification_comments",
          16: "stream_id", 17: "transport_stream_id", 21: "item_designator_id", 22: "version",
          23: "classifying_country_coding_date", 24: "object_country_coding_date"}


def decode_security_items(body: bytes) -> dict:
    out: dict = {}
    for tag, raw, _ in _items(body, 0, len(body)):
        name = ST0102.get(tag)
        if name is None:
            continue
        if tag == 1:
            out[name] = SECURITY_CLASSES.get(raw[0], str(raw[0])) if raw else None
        elif tag in (2, 12, 16, 17, 22):
            out[name] = int.from_bytes(raw, "big") if raw else None
        elif tag == 13:
            # ST 0102.11 and later: UTF-16; older: ASCII
            txt = raw.decode("utf-16-be", "replace") if len(raw) % 2 == 0 and raw[:1] == b"\x00" else _text(raw)
            out[name] = txt.strip("\x00").strip()
        else:
            out[name] = _text(raw)
    return out


def decode_security(packet: bytes) -> dict | None:
    try:
        i, end = _set_body(packet, SECURITY_LS_KEY)
        return decode_security_items(packet[i:end])
    except (ValueError, IndexError):
        return None


# ------------------------------------------------------------------ ST 0903 (VMTI)

def _pixel(n: int, width: int | None) -> tuple[int, int] | None:
    """ST 0903 pixel number (1 is the top-left pixel) to (column, row) from 0."""
    if not width or n <= 0:
        return None
    return (n - 1) % width, (n - 1) // width


def _target(body: bytes, width: int | None) -> dict:
    tid, i = ber_oid(body, 0)
    t: dict = {"id": tid}
    for tag, raw, _ in _items(body, i, len(body)):
        if tag in (1, 2, 3):
            p = _pixel(int.from_bytes(raw, "big"), width)
            if p is not None:
                t[{1: "centroid", 2: "box_top_left", 3: "box_bottom_right"}[tag]] = list(p)
        elif tag == 4:
            t["priority"] = raw[0] if raw else None
        elif tag == 5:
            t["confidence"] = raw[0] if raw else None
        elif tag == 6:
            t["history"] = int.from_bytes(raw, "big")
        elif tag == 7:
            t["percent_pixels"] = raw[0] if raw else None
        elif tag == 8 and len(raw) == 3:
            t["color"] = "#" + raw.hex()
        elif tag == 9:
            t["intensity"] = int.from_bytes(raw, "big")
        elif tag == 10:
            t["offset_latitude"] = imapb(raw, -19.2, 19.2)
        elif tag == 11:
            t["offset_longitude"] = imapb(raw, -19.2, 19.2)
        elif tag == 12:
            t["height"] = imapb(raw, -900, 19000)
        elif tag == 17 and len(raw) >= 10:
            t["latitude"] = imapb(raw[0:4], -90, 90)
            t["longitude"] = imapb(raw[4:8], -180, 180)
            t["hae"] = imapb(raw[8:10], -900, 19000)
        elif tag == 19:
            t["centroid_row"] = int.from_bytes(raw, "big")
        elif tag == 20:
            t["centroid_column"] = int.from_bytes(raw, "big")
        elif tag == 22:
            t["algorithm_id"] = int.from_bytes(raw, "big")
    if "centroid" not in t and "centroid_row" in t and "centroid_column" in t:
        t["centroid"] = [t["centroid_column"], t["centroid_row"]]
    return {k: v for k, v in t.items() if v is not None}


def decode_vmti_items(body: bytes, check: bytes | None = None) -> dict:
    out: dict = {"targets": []}
    series = None
    for tag, raw, at in _items(body, 0, len(body)):
        if tag == 1 and check is not None:
            if len(raw) != 2 or checksum(check[:len(check) - len(body) + at]) != int.from_bytes(raw, "big"):
                raise ValueError("bad VMTI checksum")
        elif tag == 2:
            out["time"] = _time(raw)
        elif tag == 3:
            out["system_name"] = _text(raw)
        elif tag == 4:
            out["version"] = int.from_bytes(raw, "big")
        elif tag == 5:
            out["targets_detected"] = int.from_bytes(raw, "big")
        elif tag == 6:
            out["targets_reported"] = int.from_bytes(raw, "big")
        elif tag == 7:
            out["frame_number"] = int.from_bytes(raw, "big")
        elif tag == 8:
            out["frame_width"] = int.from_bytes(raw, "big")
        elif tag == 9:
            out["frame_height"] = int.from_bytes(raw, "big")
        elif tag == 10:
            out["source_sensor"] = _text(raw)
        elif tag == 11:
            out["hfov"] = imapb(raw, 0, 180)
        elif tag == 12:
            out["vfov"] = imapb(raw, 0, 180)
        elif tag == 101:
            series = raw
    if series is not None:
        j = 0
        while j < len(series):
            ln, j = ber_len(series, j)
            try:
                out["targets"].append(_target(series[j:j + ln], out.get("frame_width")))
            except (ValueError, IndexError):
                out["skipped_targets"] = out.get("skipped_targets", 0) + 1
            j += ln
    return out


def decode_vmti(packet: bytes) -> dict | None:
    try:
        i, end = _set_body(packet, VMTI_LS_KEY)
        return decode_vmti_items(packet[i:end], packet[:end])
    except (ValueError, IndexError):
        return None


# ------------------------------------------------------------------ a stream of packets

def split(buf: bytes):
    """Each (kind, packet) of the known sets in a KLV data stream's bytes."""
    i = 0
    while True:
        hits = [(buf.find(k, i), k) for k in KEYS]
        hits = [(j, k) for j, k in hits if j >= 0]
        if not hits:
            return
        j, k = min(hits)
        try:
            n, s = ber_len(buf, j + 16)
        except (ValueError, IndexError):
            return
        yield KEYS[k], buf[j:s + n]
        i = s + n if s + n > j else j + 1


def decode(kind: str, packet: bytes) -> dict | None:
    return {"st0601": decode_0601, "st0102": decode_security, "st0903": decode_vmti}[kind](packet)


# ------------------------------------------------------------------ writing (tests and samples)

def _item(tag: int, value: bytes) -> bytes:
    return ber_oid_bytes(tag) + ber_len_bytes(len(value)) + value


def encode_vmti_items(width: int, height: int, targets: list[dict], frame_number: int | None = None) -> bytes:
    """A VMTI body (no key) with targets given as {"id", "centroid": (col, row), "box": (c0, r0, c1, r1),
    "confidence", "lat", "lon", "hae"}."""
    items = [_item(4, bytes([5])), _item(8, width.to_bytes(2, "big")), _item(9, height.to_bytes(2, "big")),
             _item(5, len(targets).to_bytes(1, "big")), _item(6, len(targets).to_bytes(1, "big"))]
    if frame_number is not None:
        items.append(_item(7, frame_number.to_bytes(4, "big")))
    series = b""
    for t in targets:
        pn = lambda c, r: (r * width + c + 1).to_bytes(4, "big")      # noqa: E731
        body = ber_oid_bytes(t["id"]) + _item(1, pn(*t["centroid"]))
        if "box" in t:
            c0, r0, c1, r1 = t["box"]
            body += _item(2, pn(c0, r0)) + _item(3, pn(c1, r1))
        if "confidence" in t:
            body += _item(5, bytes([t["confidence"]]))
        if "lat" in t:
            body += _item(17, imapb_bytes(t["lat"], -90, 90, 4) + imapb_bytes(t["lon"], -180, 180, 4)
                          + imapb_bytes(t.get("hae", 0.0), -900, 19000, 2))
        series += ber_len_bytes(len(body)) + body
    items.append(_item(101, series))
    return b"".join(items)


def encode_security_items(classification: int = 1, country: str = "//USA", version: int = 12) -> bytes:
    return (_item(1, bytes([classification])) + _item(2, bytes([7])) + _item(3, country.encode())
            + _item(12, bytes([7])) + _item(13, country.encode()) + _item(22, version.to_bytes(2, "big")))
