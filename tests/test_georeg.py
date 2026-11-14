"""Geo-registration: geometry, fixed-camera fits, map snapping and the drone chain, on a made-up textured map."""
import math

import cv2
import numpy as np
import pytest

from trafficwb.georeg import (DashcamGround, DroneRegistrar, GeoMap, LocalFrame, Matcher, apply_h, fit_points,
                              footprint, gps_heading, horizon_row, pose_from_telemetry, pose_homography, snap_to_map)
from trafficwb.video_meta import make_0601, parse_0601


def _texture(w=2400, h=2400, seed=0):
    """A map-like picture: fields, roads and buildings at several sizes, so features exist at every scale."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(260):
        x, y = rng.integers(0, w), rng.integers(0, h)
        a, b = rng.integers(30, 260, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + a), int(y + b)), tuple(int(v) for v in rng.integers(40, 220, 3)), -1)
    for _ in range(40):
        p = rng.integers(0, w, 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (200, 200, 200), int(rng.integers(3, 12)))
    for _ in range(1500):
        x, y = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 9)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    noise = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 2)
    return np.clip(img.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def gmap():
    # about 0.5 m a pixel: 2400 px = 1.2 km
    lat0, lon0 = 32.08, 34.78
    dlat = 1200 / 111320.0
    dlon = 1200 / (111320.0 * math.cos(math.radians(lat0)))
    return GeoMap(_texture(), (lat0 + dlat / 2, lon0 - dlon / 2), (lat0 - dlat / 2, lon0 + dlon / 2))


def test_local_frame_round_trip():
    f = LocalFrame(32.0, 34.8)
    e, n = f.to_en(32.001, 34.801)
    lat, lon = f.to_latlon(e, n)
    assert abs(lat - 32.001) < 1e-9 and abs(lon - 34.801) < 1e-9
    assert abs(float(n) - 111.3) < 0.2                     # a thousandth of a degree of latitude is about 111 m


def test_pose_homography_straight_down_and_tilted():
    G = pose_homography(1000, 500, 90, 10, 20, 100, 0, -90)
    c, top, right = apply_h(G, [[500, 250], [500, 0], [1000, 250]])
    assert np.allclose(c, [10, 20]) and np.allclose(top, [10, 70]) and np.allclose(right, [110, 20])
    G = pose_homography(1000, 500, 90, 10, 20, 100, 90, -90)                  # facing east: top of frame is east
    assert np.allclose(apply_h(G, [[500, 0]])[0], [60, 20])
    G = pose_homography(1000, 500, 90, 0, 0, 100, 0, -45)
    assert np.allclose(apply_h(G, [[500, 250]])[0], [0, 100], atol=1e-6)
    assert horizon_row(G, 1000, 500) == 0
    G = pose_homography(1000, 500, 90, 0, 0, 10, 0, -5)                       # nearly level: the horizon is in view
    assert abs(horizon_row(G, 1000, 500) - (250 - 500 * math.tan(math.radians(5)))) < 1e-3
    assert footprint(G, 1000, 500) is None


def test_pose_from_telemetry_corners_centre_and_gimbal():
    f = LocalFrame(32.0, 34.8)
    lat = lambda n: 32.0 + n / 111320.0                                        # noqa: E731
    lon = lambda e: 34.8 + e / (111320.0 * math.cos(math.radians(32.0)))       # noqa: E731
    s = {"corner1_lat": lat(50), "corner1_lon": lon(-100), "corner2_lat": lat(50), "corner2_lon": lon(100),
         "corner3_lat": lat(-50), "corner3_lon": lon(100), "corner4_lat": lat(-50), "corner4_lon": lon(-100)}
    G = pose_from_telemetry(s, 400, 200, f)
    assert np.allclose(apply_h(G, [[200, 100]])[0], [0, 0], atol=0.5)
    # platform 300 m up and 300 m south of where it looks: 45 degrees down, facing north
    s = {"lat": lat(-300), "lon": lon(0), "alt": 350, "center_lat": lat(0), "center_lon": lon(0), "center_elev": 50,
         "hfov": 30}
    G = pose_from_telemetry(s, 400, 200, f)
    assert np.allclose(apply_h(G, [[200, 100]])[0], [0, 0], atol=0.5)
    s = {"lat": lat(0), "lon": lon(0), "rel_alt": 100, "heading": 0, "pitch": -90, "focal_mm": 24}
    G = pose_from_telemetry(s, 400, 300, f)
    fp = footprint(G, 400, 300)
    assert abs((fp[1, 0] - fp[0, 0]) - 2 * 100 * 36 / 48) < 0.5              # 24 mm (35 mm equivalent) over 100 m
    assert pose_from_telemetry({"lat": 1.0, "lon": 1.0}, 400, 300, f) is None


def test_klv_corner_points_round_trip():
    c = (32.0, 34.8)
    corners = [(32.0006, 34.7992), (32.0006, 34.8008), (31.9994, 34.8008), (31.9994, 34.7992)]
    v = parse_0601(make_0601(1_700_000_000_000_000, 31.99, 34.8, 1200, 0, 20.0, c, None, corners, 30.0))
    for k, (a, b) in enumerate(corners, 1):
        assert abs(v[f"corner{k}_lat"] - a) < 5e-6 and abs(v[f"corner{k}_lon"] - b) < 5e-6
    assert abs(v["center_elev"] - 30) < 0.5


def test_fixed_camera_points_and_a_bad_pick():
    f = LocalFrame(32.0, 34.8)
    Gt = pose_homography(1920, 1080, 60, 0, 0, 12, 30, -30)
    px = np.array([[300, 700], [1600, 650], [900, 1000], [200, 1050], [1700, 1000], [1000, 600], [600, 850]], float)
    en = apply_h(Gt, px)
    lat, lon = f.to_latlon(en[:, 0], en[:, 1])
    fit = fit_points(px, np.c_[lat, lon], f)
    assert max(fit.residual_m) < 0.01 and max(fit.leave_one_out_m) < 0.05
    assert np.allclose(fit.to_latlon([[900, 800]])[0], f.to_latlon(*apply_h(Gt, [[900, 800]])[0]), atol=1e-7)
    lat2 = lat.copy()
    lat2[2] += 8 / 111320.0                                                     # one point clicked 8 m off on the map
    bad = fit_points(px, np.c_[lat2, lon], f)
    assert bad.suspect == 2 and bad.leave_one_out_m[2] > 4
    assert fit.suspect is None
    with pytest.raises(ValueError):
        fit_points(px[:3], np.c_[lat, lon][:3], f)


def test_snap_to_map_corrects_a_wrong_guess(gmap):
    w, h = 960, 540
    Gt = pose_homography(w, h, 60, 40, -30, 220, 25, -70)
    frame = gmap.render(gmap.A_inv @ Gt, (w, h))
    frame = np.clip(frame.astype(np.float32) * 0.9 + 12, 0, 255).astype(np.uint8)
    G0 = pose_homography(w, h, 63, 55, -12, 240, 29, -67)                      # 20 m and a few degrees off
    grid = [[x, y] for x in (100, 480, 860) for y in (150, 300, 500)]
    before = np.linalg.norm(apply_h(G0, grid) - apply_h(Gt, grid), axis=1).max()
    s = snap_to_map(frame, G0, gmap, Matcher())
    assert s.G is not None, s.reason
    after = np.linalg.norm(apply_h(s.G, grid) - apply_h(Gt, grid), axis=1).max()
    assert before > 15 and after < 2.0


def test_snap_refuses_the_wrong_place(gmap):
    w, h = 640, 360
    frame = np.random.default_rng(5).integers(0, 255, (h, w, 3), dtype=np.uint8)
    G0 = pose_homography(w, h, 60, 0, 0, 200, 0, -90)
    assert snap_to_map(frame, G0, gmap, Matcher()).G is None


def test_drone_chain_with_flight_data_stays_on_the_map(gmap):
    w, h, fps = 640, 360, 10
    reg = DroneRegistrar(gmap, Matcher(), snap_every_s=1.0, async_snap=False)
    f = gmap.frame
    errs = []
    for i in range(25):
        t = i / fps
        e, n, yaw = -100 + 8 * t, -50 + 5 * t, 10 + 3 * t
        Gt = pose_homography(w, h, 60, e, n, 200, yaw, -80)
        frame = gmap.render(gmap.A_inv @ Gt, (w, h))
        lat, lon = f.to_latlon(e + 12, n - 9)                                   # GPS 15 m off, heading 4 degrees off
        s = {"lat": float(lat), "lon": float(lon), "rel_alt": 185, "heading": yaw + 4, "pitch": -80, "hfov": 60}
        fix = reg.step(frame, t, s)
        errs.append(np.linalg.norm(apply_h(fix.G, [[w / 2, h / 2]]) - apply_h(Gt, [[w / 2, h / 2]])))
    reg.close()
    assert errs[0] < 3 and max(errs) < 3, errs                                  # matched from the first frame on
    assert sum(x["ok"] for x in reg.log) >= 2


def test_dashcam_ground_and_horizon():
    g = DashcamGround(1280, 720, 100, 1.3, -3)
    assert abs(DashcamGround.pitch_from_horizon(g.horizon, 720, 1280, 100) + 3) < 1e-3
    ahead = g.to_vehicle([[640, 700]])[0]
    assert abs(ahead[0]) < 1e-6 and 1 < ahead[1] < 5
    world = g.to_world([[640, 700]], 100, 200, 90)[0]                           # facing east: ahead is east
    assert abs(world[0] - (100 + ahead[1])) < 1e-6 and abs(world[1] - 200) < 1e-6
    hd = gps_heading(np.array([0, 0, 5, 10, 10]), np.array([0, 1, 1, 1, 1]))
    assert abs(hd[-1] - 90) < 1e-6


def test_klv_sensor_angles_and_moving_targets():
    f = LocalFrame(32.0, 34.8)
    from trafficwb.georeg import targets_on_ground
    # a platform heading 90 with the sensor turned 90 more and 45 down: it looks south from 100 m above ground
    s = {"lat": 32.0, "lon": 34.8, "agl": 100.0, "heading": 90.0, "pitch": 0.0, "sensor_az": 90.0, "sensor_el": -45.0,
         "hfov": 30.0}
    G = pose_from_telemetry(s, 640, 480, f)
    assert np.allclose(apply_h(G, [[320, 240]])[0], [0, -100], atol=0.5)
    # a target at the frame centre, and the position the aircraft sent for it 3 m away
    lat, lon = f.to_latlon(3.0, -100.0)
    rep = {"frame_width": 1280, "frame_height": 960, "targets": [
        {"id": 4, "centroid": [640, 480], "latitude": float(lat), "longitude": float(lon)}]}
    t = targets_on_ground(rep, G, f, 640, 480)[0]
    assert t["id"] == 4 and abs(t["apart_m"] - 3.0) < 0.6


def test_command_line_registers_a_klv_video(tmp_path, gmap):
    """The drone command end to end: a video with KLV frame centre and view, a map picture with its corners."""
    import json
    from fractions import Fraction

    import av

    from trafficwb.georeg import main
    f = gmap.frame
    w, h, rate = 320, 240, 10
    video = tmp_path / "flight.ts"
    with av.open(str(video), "w", format="mpegts") as o:
        vs = o.add_stream("mpeg2video", rate=rate)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        ds = o.add_data_stream("klv")
        ds.time_base = Fraction(1, 90000)
        for k in range(15):
            e = -50 + 4 * k
            Gt = pose_homography(w, h, 50, e, 0, 250, 0, -89.9)
            frame = gmap.render(gmap.A_inv @ Gt, (w, h))
            vf = av.VideoFrame.from_ndarray(frame[:, :, ::-1].copy(), format="rgb24")
            vf.pts = k
            for p in vs.encode(vf):
                o.mux(p)
            if k % 5 == 0:
                lat, lon = f.to_latlon(e + 10, 8)                               # 13 m off
                pkt = av.Packet(make_0601(1_700_000_000_000_000 + k * 100_000, float(lat) - 1e-6, float(lon), 250.0,
                                          0.0, 50.0, (float(lat), float(lon)), "T", center_elev=0.0))
                pkt.stream, pkt.pts, pkt.dts, pkt.time_base = ds, k * 9000, k * 9000, ds.time_base
                o.mux(pkt)
        for p in vs.encode():
            o.mux(p)
    mp = tmp_path / "map.png"
    cv2.imwrite(str(mp), gmap.image)
    (lt, ll), (lb, lr) = f.to_latlon(*gmap.A[:2, 2]), f.to_latlon(*apply_h(gmap.A, [[2400, 2400]])[0])
    out = tmp_path / "out.json"
    main(["drone", str(video), "--map", str(mp), "--corners", f"{lt},{ll},{lb},{lr}", "--out", str(out), "--offline"])
    res = json.loads(out.read_text())
    assert res["summary"]["flight_data"] == "klv-0601" and res["summary"]["map_matches_ok"] >= 1
    last = res["frames"][-1]
    e_true = -50 + 4 * 14
    lat, lon = f.to_latlon(e_true, 0)
    assert f.distance(*last["center"], float(lat), float(lon)) < 3.0
