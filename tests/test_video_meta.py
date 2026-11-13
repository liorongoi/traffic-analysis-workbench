"""Drone and aircraft video metadata (phase 7): DJI subtitle files and KLV (MISB ST 0601), found on their own."""
from fractions import Fraction

import numpy as np
import pytest

from trafficwb.video_meta import checksum, find, make_0601, parse_0601, read_klv, read_srt, sidecar

DJI_NEW = """1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2024-05-01 14:30:00.123
[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [color_md : default] [focal_len: 24.00] [latitude: 32.081234] [longitude: 34.781234] [rel_alt: 120.000 abs_alt: 160.500] [ct: 5500] </font>

2
00:00:01,000 --> 00:00:01,033
<font size="28">FrameCnt: 31, DiffTime: 33ms
2024-05-01 14:30:01.123
[iso: 100] [focal_len: 24.00] [latitude: 32.082234] [longitude: 34.782234] [rel_alt: 130.000 abs_alt: 170.500] </font>

3
00:00:02,000 --> 00:00:02,033
[latitude: 0.0] [longitude: 0.0] [rel_alt: 0.000 abs_alt: 0.000]
"""

DJI_OLD = """1
00:00:00,000 --> 00:00:01,000
HOME(34.7812,32.0812) 2019.03.05 10:00:00
GPS(34.7800,32.0800,45) BAROMETER:52.1
ISO:100 Shutter:60 EV:0 Fnum:F2.8

2
00:00:01,000 --> 00:00:02,000
HOME(34.7812,32.0812) 2019.03.05 10:00:01
GPS(34.7810,32.0810,46) BAROMETER:53.0
ISO:100 Shutter:60 EV:0 Fnum:F2.8
"""

CAPTIONS = """1
00:00:06,480 --> 00:00:08,639
Um, they'll have to look in the air

2
00:00:08,639 --> 00:00:11,840
where my drone is

3
00:00:11,840 --> 00:00:14,559
[Music]
"""


def test_dji_subtitles_are_telemetry(tmp_path):
    p = tmp_path / "DJI_0001.SRT"
    p.write_text(DJI_NEW)
    tel = read_srt(p)
    assert tel.source == "dji-srt" and len(tel.samples) == 2 and tel.skipped == 1      # no GPS fix: not a sample
    assert tel.samples[0]["lat"] == pytest.approx(32.081234) and tel.samples[0]["rel_alt"] == 120.0
    assert tel.samples[0]["alt"] == 160.5 and tel.samples[0]["focal_mm"] == 24.0
    assert tel.start_local == "2024-05-01T14:30:00.123"
    mid = tel.at(0.5)
    assert mid["lat"] == pytest.approx(32.081734) and mid["rel_alt"] == pytest.approx(125.0)
    assert tel.at(10.0) is None
    s = tel.summary()
    assert s["rel_alt_m"] == [120.0, 130.0] and s["first"]["lon"] == pytest.approx(34.781234)
    old = tmp_path / "old.srt"
    old.write_text(DJI_OLD)
    t2 = read_srt(old)
    assert t2.samples[1]["lat"] == pytest.approx(32.081) and t2.samples[1]["lon"] == pytest.approx(34.781)
    assert t2.samples[0]["alt"] == 45.0 and t2.start_local == "2019-03-05T10:00:00.000"


def test_ordinary_subtitles_are_not_telemetry(tmp_path):
    p = tmp_path / "talk.srt"
    p.write_text(CAPTIONS)
    assert read_srt(p) is None
    (tmp_path / "talk.mp4").write_bytes(b"")
    assert sidecar(tmp_path / "talk.mp4") == p
    assert find(tmp_path / "talk.mp4") is None


def test_klv_packet_round_trip_and_checksum():
    pkt = make_0601(1_714_573_800_000_000, 32.0812, 34.7812, 1500.0, heading=271.5, hfov=12.5,
                    center=(32.07, 34.79), platform="MQ-1")
    v = parse_0601(pkt)
    assert v["lat"] == pytest.approx(32.0812, abs=1e-6) and v["lon"] == pytest.approx(34.7812, abs=1e-6)
    assert v["alt"] == pytest.approx(1500.0, abs=0.5) and v["heading"] == pytest.approx(271.5, abs=0.01)
    assert v["hfov"] == pytest.approx(12.5, abs=0.01) and v["center_lat"] == pytest.approx(32.07, abs=1e-6)
    assert v["platform"] == "MQ-1" and v["unix_us"] == 1_714_573_800_000_000
    bad = bytearray(pkt)
    bad[30] ^= 0xFF                                    # one damaged byte: the checksum no longer matches
    assert parse_0601(bytes(bad)) is None
    assert parse_0601(b"\x00" * 40) is None
    assert checksum(b"\x01\x02") == 0x0102


def _klv_video(path, seconds=3, rate=10):
    import av
    with av.open(str(path), "w", format="mpegts") as o:
        vs = o.add_stream("mpeg2video", rate=rate)
        vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
        ds = o.add_data_stream("klv")
        ds.time_base = Fraction(1, 90000)
        for k in range(seconds * rate):
            f = av.VideoFrame.from_ndarray(np.full((120, 160, 3), 80, np.uint8), format="rgb24")
            f.pts = k
            for p in vs.encode(f):
                o.mux(p)
            if k % rate == 0:                          # one metadata packet a second
                pkt = av.Packet(make_0601(1_714_573_800_000_000 + k * 100_000, 32.08 + 0.001 * k / rate, 34.78,
                                          1000.0 + 10 * k / rate, platform="TEST"))
                pkt.stream = ds
                pkt.pts = pkt.dts = int(k / rate * 90000)
                pkt.time_base = ds.time_base
                o.mux(pkt)
        for p in vs.encode():
            o.mux(p)


def test_klv_inside_a_video_is_found(tmp_path):
    p = tmp_path / "uav.ts"
    _klv_video(p)
    tel = find(p)
    assert tel is not None and tel.source == "klv-0601" and tel.platform == "TEST"
    assert len(tel.samples) == 3 and tel.skipped == 0
    assert tel.start_utc == "2024-05-01T14:30:00.000Z"
    assert tel.at(1.5)["alt"] == pytest.approx(1015.0, abs=1.0)
    # A video with no data stream has no KLV.
    plain = tmp_path / "plain.ts"
    import av
    with av.open(str(plain), "w", format="mpegts") as o:
        vs = o.add_stream("mpeg2video", rate=10)
        vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
        for k in range(5):
            f = av.VideoFrame.from_ndarray(np.zeros((120, 160, 3), np.uint8), format="rgb24")
            f.pts = k
            for q in vs.encode(f):
                o.mux(q)
        for q in vs.encode():
            o.mux(q)
    assert read_klv(plain) is None and find(plain) is None


def test_a_mission_reads_and_keeps_a_recordings_metadata(tmp_path):
    import shutil
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_video import car_scene, mission, run
    from trafficwb.video_meta import stored
    from trafficwb.video_review import VideoService
    m = mission(tmp_path, {"DRONE_20240501T143000Z.mp4": (2.0, car_scene, False)})
    src = Path(m.config().source_dir)
    (src / "DRONE_20240501T143000Z.SRT").write_text(DJI_NEW)
    run(m)
    vid = m.con.execute("SELECT id FROM source_version").fetchone()[0]
    tel = stored(m.con, vid)                                    # read during the analysis, kept since
    assert tel is not None and tel.source == "dji-srt" and len(tel.samples) == 2
    (src / "DRONE_20240501T143000Z.SRT").unlink()
    assert stored(m.con, vid).samples[1]["rel_alt"] == 130.0    # kept, though the file has gone
    from fastapi.testclient import TestClient
    from trafficwb.web.app import create_app
    from test_web_api import direct_access
    app = create_app(tmp_path / "data", detector_factory=lambda: (None, {}), start_monitoring=False)
    c = direct_access(TestClient(app))
    r = c.get(f"/api/missions/{m.id}/recordings/{vid}/telemetry").json()
    assert r["source"] == "dji-srt" and r["summary"]["rel_alt_m"] == [120.0, 130.0]
    shutil.rmtree(src)


def test_klv_time_dates_a_recording_whose_name_has_none(tmp_path):
    from trafficwb.ingest import cycle
    from trafficwb.store import DataRoot, MissionConfig
    from trafficwb.video_process import probe_version
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "out").mkdir()
    _klv_video(src / "sortie.ts")
    (src / "flight.webm").write_bytes(b"")                     # web video files are taken in too
    cfg = MissionConfig(name="uav", input_type="video", mode="object_search", country="IL", iana_zone="Asia/Jerusalem",
                        source_dir=str(src), include_subdirs=False, processing="one_time",
                        output_dir=str(tmp_path / "out"), export_formats=["csv"])
    m = DataRoot(tmp_path / "data").create_mission(cfg, "lior")
    cycle(m, stability_seconds=0)
    names = {r[0] for r in m.con.execute("SELECT f.rel_path FROM source_file f")}
    assert names == {"sortie.ts", "flight.webm"}
    vid = m.con.execute("SELECT v.id FROM source_version v JOIN source_file f ON f.id=v.source_file_id "
                        "WHERE f.rel_path='sortie.ts'").fetchone()[0]
    from trafficwb.views import ViewService
    p = probe_version(m.con, ViewService(m), vid)
    assert p["start_utc"] == "2024-05-01T14:30:00.000Z" and p["start_source"] == "klv"
