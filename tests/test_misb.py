"""STANAG 4609 metadata: MISB ST 0601 in full, ST 0102 security, ST 0903 moving targets, ST 1201 IMAPB."""
from fractions import Fraction

import numpy as np
import pytest

from trafficwb import misb
from trafficwb.video_meta import make_0601, parse_0601, read_klv

# The ST 0601 example packet with constant items as published with the klvdata project (MIT licence,
# github.com/paretech/klvdata, data/DynamicConstantMISMMSPacketData.bin). Its last two bytes (the checksum) were
# copied from the shorter example and do not match this packet; the test puts the right ones in.
CONSTANT = bytes.fromhex(
    "060e2b34020b01010e0103010100000081d2020800046050584e0180030a4d697373696f6e203132050271c20602fd3d070208b8"
    "0a085072656461746f720b07454f204e6f73650c0e47656f64657469632057475338340d045595b66d0e045b5360c40f02c22110"
    "02cd9c1102d9171204724a0a20130487f84b8614047dc55ece150403830926160212811704f101a229180414bc082b190234f330"
    "1c01010102010703052f2f5553410c01070d060055005300411602000a4101065e220170f592f02373364af8aa9162c00f2eb2da"
    "16b74341000841a0be365b5ab96a36450102aa43")


def _fixed(packet: bytes) -> bytes:
    head = packet[:-2]
    return head + misb.checksum(head).to_bytes(2, "big")


def test_the_published_example_with_every_constant_item():
    assert misb.decode_0601(CONSTANT) is None                       # its checksum is wrong: refused
    loose = misb.decode_0601(CONSTANT, strict=False)
    assert loose["_bad_checksum"] is True
    v = misb.decode_0601(_fixed(CONSTANT))
    assert "_bad_checksum" not in v
    assert v["precision_time"] == "2009-01-12T22:08:22.000000Z"
    assert v["mission_id"] == "Mission 12" and v["platform_designation"] == "Predator"
    assert v["image_source_sensor"] == "EO Nose" and v["image_coordinate_system"] == "Geodetic WGS84"
    expect = {"platform_heading": 159.9744, "platform_pitch": -0.4315, "platform_roll": 3.4059,
              "sensor_latitude": 60.176823, "sensor_longitude": 128.426759, "sensor_true_altitude": 14190.719,
              "sensor_hfov": 144.5713, "sensor_vfov": 152.6436, "sensor_relative_azimuth": 160.7192,
              "sensor_relative_elevation": -168.7923, "sensor_relative_roll": 176.8654, "slant_range": 68590.983,
              "target_width": 722.8199, "frame_center_latitude": -10.542389, "frame_center_longitude": 29.157890,
              "frame_center_elevation": 3216.037}
    for k, want in expect.items():
        assert v[k] == pytest.approx(want, abs=1e-3), k
    assert v["security"]["classification"] == "UNCLASSIFIED" and v["security"]["classifying_country"] == "//USA"
    assert v["security"]["object_country_codes"] == "USA" and v["security"]["version"] == 10
    assert v["uas_ls_version"] == 6 and len(v["miis_core_identifier"]) == 68


def test_imapb_matches_the_standard_example():
    # ST 0601 item 103, density altitude extended: 0x2F921E is 23456.24 m in the standard's example
    assert misb.imapb(bytes.fromhex("2F921E"), -900, 40000) == pytest.approx(23456.23, abs=0.02)
    for v, a, b, L in ((123.456, 0, 360, 3), (-512.25, -900, 40000, 3), (99.5, 0, 100, 2), (-45.0, -90, 90, 4)):
        assert misb.imapb(misb.imapb_bytes(v, a, b, L), a, b) == pytest.approx(v, abs=(b - a) / 2 ** (8 * L - 2))
    assert misb.imapb(b"\x80\x00", 0, 360) is None                 # a special value, not a number


def test_ber_oid_and_lengths_round_trip():
    for n in (0, 5, 127, 128, 300, 70000):
        assert misb.ber_oid(misb.ber_oid_bytes(n), 0) == (n, len(misb.ber_oid_bytes(n)))
        assert misb.ber_len(misb.ber_len_bytes(n), 0) == (n, len(misb.ber_len_bytes(n)))


def test_targets_and_security_nested_in_0601_with_offsets_resolved():
    vm = misb.encode_vmti_items(1280, 720, [
        {"id": 7, "centroid": (640, 360), "box": (600, 330, 680, 390), "confidence": 88, "lat": 32.081, "lon": 34.781,
         "hae": 21.0},
        {"id": 300, "centroid": (10, 700), "confidence": 40}], frame_number=1234)
    pkt = make_0601(1_714_573_800_000_000, 32.07, 34.78, 900.0, 15.0, 20.0, (32.08, 34.78), "TEST",
                    extra=[(48, misb.encode_security_items(1, "//ISR")), (74, vm)])
    v = misb.decode_0601(pkt)
    assert v["security"]["classification"] == "UNCLASSIFIED" and v["security"]["classifying_country"] == "//ISR"
    t = v["vmti"]
    assert t["frame_width"] == 1280 and t["frame_height"] == 720 and t["frame_number"] == 1234
    a, b = t["targets"]
    assert a["id"] == 7 and a["centroid"] == [640, 360] and a["box_top_left"] == [600, 330]
    assert a["box_bottom_right"] == [680, 390] and a["confidence"] == 88
    assert a["latitude"] == pytest.approx(32.081, abs=1e-6) and a["longitude"] == pytest.approx(34.781, abs=1e-6)
    assert b["id"] == 300 and b["centroid"] == [10, 700] and "latitude" not in b
    short = parse_0601(pkt)
    assert short["security"]["classification"] == "UNCLASSIFIED" and len(short["vmti"]["targets"]) == 2


def test_a_standalone_vmti_packet_and_its_checksum():
    body = misb.encode_vmti_items(640, 480, [{"id": 1, "centroid": (5, 6)}])
    head = misb.VMTI_LS_KEY + misb.ber_len_bytes(len(body) + 4) + body + bytes([1, 2])
    pkt = head + misb.checksum(head).to_bytes(2, "big")
    v = misb.decode_vmti(pkt)
    assert v["targets"][0]["centroid"] == [5, 6]
    assert misb.decode_vmti(pkt[:-1] + bytes([pkt[-1] ^ 1])) is None
    assert [k for k, _ in misb.split(b"junk" + pkt + make_0601(1, 1.0, 2.0, 3.0))] == ["st0903", "st0601"]


def test_corners_from_the_full_items_win_over_offsets():
    c = (32.0, 34.8)
    off = [(32.0006, 34.7992), (32.0006, 34.8008), (31.9994, 34.8008), (31.9994, 34.7992)]
    full = b"".join(bytes([82 + 2 * k]) + bytes([4]) + int(round(lat * (2 ** 31 - 1) / 90)).to_bytes(4, "big", signed=True)
                    + bytes([83 + 2 * k]) + bytes([4]) + int(round((lon + 0.0001) * (2 ** 31 - 1) / 180)).to_bytes(4, "big", signed=True)
                    for k, (lat, lon) in enumerate(off))
    pkt = make_0601(1, 31.99, 34.8, 1000.0, 0.0, 10.0, c, None, off,
                    extra=[(t, b) for t, b in _split_items(full)])
    v = misb.decode_0601(pkt)
    assert v["corner_longitude_1"] == pytest.approx(34.7993, abs=1e-6)          # the full item, not the offset


def _split_items(raw: bytes):
    i = 0
    while i < len(raw):
        t, n = raw[i], raw[i + 1]
        yield t, raw[i + 2:i + 2 + n]
        i += 2 + n


def test_targets_and_security_are_read_from_a_video(tmp_path):
    import av
    p = tmp_path / "isr.ts"
    with av.open(str(p), "w", format="mpegts") as o:
        vs = o.add_stream("mpeg2video", rate=10)
        vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
        ds = o.add_data_stream("klv")
        ds.time_base = Fraction(1, 90000)
        for k in range(20):
            f = av.VideoFrame.from_ndarray(np.full((120, 160, 3), 80, np.uint8), format="rgb24")
            f.pts = k
            for q in vs.encode(f):
                o.mux(q)
            if k % 5 == 0:
                vm = misb.encode_vmti_items(160, 120, [{"id": 1, "centroid": (20 + k, 30)}], frame_number=k)
                pkt = av.Packet(make_0601(1_714_573_800_000_000 + k * 100_000, 32.08, 34.78, 500.0, 0.0, 30.0,
                                          (32.081, 34.78), "ISR", extra=[(48, misb.encode_security_items(1)), (74, vm)]))
                pkt.stream, pkt.pts, pkt.dts, pkt.time_base = ds, k * 9000, k * 9000, ds.time_base
                o.mux(pkt)
        for q in vs.encode():
            o.mux(q)
    tel = read_klv(p)
    assert tel is not None and tel.security["classification"] == "UNCLASSIFIED"
    assert [x["targets"][0]["centroid"][0] for x in tel.targets] == [20, 25, 30, 35]
    assert tel.summary()["target_reports"] == 4 and tel.summary()["security"]["classification"] == "UNCLASSIFIED"
