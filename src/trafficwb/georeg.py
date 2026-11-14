"""Geo-registration: where on the ground each pixel of a video frame is (drone, fixed camera, dashcam).

Every camera kind ends in the same thing: a ground homography ``G`` (3x3) from frame pixels to metres east and north
of a local origin. A pixel on the ground (the bottom centre of a vehicle box, say) is then a latitude and longitude.
Ground is taken as flat over one frame's footprint; hills and buildings make errors that grow with their height.

Drone and aircraft video (``DroneRegistrar``), the recipe of real-time UAV video registration:
1. A first guess from the flight data when the video has it (KLV corner points or frame centre and field of view, or
   a DJI subtitle file's position, height and gimbal angles). Flight data stays optional.
2. Frame to frame: corners tracked with optical flow, a homography to the last keyframe, chained. Smooth and fast
   (milliseconds a frame), but it drifts.
3. Every so often (``snap_every_s``), the map (an orthophoto or satellite image with its coordinates) is drawn as the
   camera should see it from the current estimate, and the frame is matched against that drawing with learned
   features (ALIKED + LightGlue, ONNX) or SIFT when no model pack is installed. The match corrects the chain and
   removes the drift. In real time the match runs beside the tracking and its correction is applied when it is ready
   (the chain from the matched frame to the current one is relative, so a late correction is still right).
4. With no flight data, the first fix is a search of the map (``locate``): slower, needs the map to be about the
   right area, and fails more often on featureless ground.

Fixed cameras (``fit_points``): a few points picked on the frame and on the map give the homography directly, with
the error of each point left out of the fit, so a bad pick shows. ``refine_with_map`` can then pull it onto the map
by matching, when the view is steep enough for the map to look like the picture.

Dashcams (``DashcamGround``): the camera's height, pitch and field of view give the ground in front of the car in
metres; the GPS (when the video has it) places the car, and the road's own motion between frames (visual odometry
on the ground plane) fills in between GPS fixes and gives the heading when the car is too slow for GPS to.
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

EARTH_R = 6378137.0


# ------------------------------------------------------------------ coordinates

@dataclass(frozen=True)
class LocalFrame:
    """Metres east and north of an origin (equirectangular, good to well under 0.1% within a few kilometres)."""
    lat0: float
    lon0: float

    def to_en(self, lat, lon):
        k = math.radians(1) * EARTH_R
        return ((np.asarray(lon) - self.lon0) * k * math.cos(math.radians(self.lat0)),
                (np.asarray(lat) - self.lat0) * k)

    def to_latlon(self, e, n):
        k = math.radians(1) * EARTH_R
        return (self.lat0 + np.asarray(n) / k, self.lon0 + np.asarray(e) / (k * math.cos(math.radians(self.lat0))))

    def distance(self, lat1, lon1, lat2, lon2) -> float:
        e1, n1 = self.to_en(lat1, lon1)
        e2, n2 = self.to_en(lat2, lon2)
        return float(np.hypot(e1 - e2, n1 - n2))


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


# ------------------------------------------------------------------ maps

class GeoMap:
    """A north-up orthophoto or satellite image and where it is: ``A`` takes map pixels to local metres."""

    def __init__(self, image: np.ndarray, top_left: tuple[float, float], bottom_right: tuple[float, float],
                 frame: LocalFrame | None = None):
        self.image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = self.image.shape[:2]
        (lat_t, lon_l), (lat_b, lon_r) = top_left, bottom_right
        self.frame = frame or LocalFrame((lat_t + lat_b) / 2, (lon_l + lon_r) / 2)
        e0, n0 = self.frame.to_en(lat_t, lon_l)
        e1, n1 = self.frame.to_en(lat_b, lon_r)
        sx, sy = (e1 - e0) / w, (n1 - n0) / h                          # sy < 0: rows go south
        self.A = np.array([[sx, 0, e0], [0, sy, n0], [0, 0, 1]], np.float64)
        self.A_inv = np.linalg.inv(self.A)
        self.m_per_px = float(math.sqrt(abs(sx * sy)))
        self._pyr = [self.image]
        self._lock = threading.Lock()

    @classmethod
    def from_geotiff(cls, path: Path | str, frame: LocalFrame | None = None) -> "GeoMap":
        """A GeoTIFF in latitude and longitude (EPSG:4326) with a pixel scale and a tie point."""
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            p = t.pages[0]
            scale = p.tags.get(33550)
            tie = p.tags.get(33922)
            if scale is None or tie is None:
                raise ValueError("The GeoTIFF has no pixel scale or tie point.")
            sx, sy = scale.value[:2]
            i, j, _, lon, lat, _ = tie.value[:6]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)            # OpenCV reads every usual TIFF compression
            if img is None:
                img = p.asarray()
            else:
                img = img[:, :, ::-1]                                  # back to RGB, as tifffile gives
        h, w = img.shape[:2]
        lon0, lat0 = lon - i * sx, lat + j * sy
        if img.ndim == 3 and img.shape[2] >= 3:
            img = np.ascontiguousarray(img[:, :, 2::-1])                 # RGB(A) to BGR
        return cls(img, (lat0, lon0), (lat0 - h * sy, lon0 + w * sx), frame)

    def level(self, k: int) -> np.ndarray:
        """The map halved ``k`` times (drawn from far away without aliasing)."""
        with self._lock:
            while len(self._pyr) <= k:
                self._pyr.append(cv2.pyrDown(self._pyr[-1]))
            return self._pyr[k]

    def render(self, T_view_to_map: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """The map as seen through a homography from view pixels to map pixels, at ``size`` (w, h)."""
        w, h = size
        c = np.array([[w / 2, h / 2], [w / 2 + 1, h / 2], [w / 2, h / 2 + 1]])
        m = apply_h(T_view_to_map, c)
        u, v = m[1] - m[0], m[2] - m[0]
        s = math.sqrt(abs(u[0] * v[1] - u[1] * v[0]))           # map pixels per view pixel at the centre
        k = max(0, min(6, int(math.floor(math.log2(max(s, 1e-6))))))
        D = np.diag([0.5 ** k, 0.5 ** k, 1.0])
        D[0, 2] = D[1, 2] = 0.5 ** k * 0.5 - 0.5                          # pyrDown pixel centres
        M = D @ T_view_to_map
        return cv2.warpPerspective(self.level(k), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------------ cameras and poses

def intrinsics(w: int, h: int, hfov_deg: float) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """World (east, north, up) from camera (right, down, forward). Yaw clockwise from north; pitch -90 straight down."""
    p, r, y = (math.radians(v) for v in (pitch_deg, roll_deg, yaw_deg))
    fwd0, down0, right0 = np.array([0, 1.0, 0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0])
    fwd = math.cos(p) * fwd0 - math.sin(p) * down0
    down = math.cos(p) * down0 + math.sin(p) * fwd0
    right = right0
    right, down = math.cos(r) * right + math.sin(r) * down, math.cos(r) * down - math.sin(r) * right
    Rz = np.array([[math.cos(y), math.sin(y), 0], [-math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ np.c_[right, down, fwd]


def pose_homography(w: int, h: int, hfov_deg: float, east: float, north: float, height: float,
                    yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0, K: np.ndarray | None = None) -> np.ndarray:
    """Frame pixels to ground metres, for a camera ``height`` metres above flat ground (``K`` overrides the lens)."""
    K = intrinsics(w, h, hfov_deg) if K is None else np.asarray(K, np.float64)
    R = rotation(yaw_deg, pitch_deg, roll_deg)                         # world from camera
    C = np.array([east, north, height])
    Rcw = R.T
    t = -Rcw @ C
    H_g2i = K @ np.c_[Rcw[:, 0], Rcw[:, 1], t]
    G = np.linalg.inv(H_g2i)
    return G / G[2, 2]


def camera_position(G: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, float]:
    """Where the camera is (metres east, north) and how high, from a ground homography and the lens."""
    H = np.linalg.inv(K) @ np.linalg.inv(G)                         # ground to normalised camera, up to scale
    lam = 1.0 / np.linalg.norm(H[:, 0])
    if (lam * H[:, 2])[2] < 0:
        lam = -lam
    r1, r2, t = lam * H[:, 0], lam * H[:, 1], lam * H[:, 2]
    r2 = r2 - r1 * (r1 @ r2)                                        # nearest rotation
    r2 /= np.linalg.norm(r2)
    R = np.c_[r1, r2, np.cross(r1, r2)]
    C = -R.T @ t
    return C[:2], float(abs(C[2]))


def targets_on_ground(report: dict, G: np.ndarray, frame: LocalFrame, w: int, h: int) -> list[dict]:
    """KLV moving targets (ST 0903) placed with the registration: each target's position from its pixel box, and the
    distance to the position the aircraft sent, when it sent one (a check on both)."""
    fw, fh = report.get("frame_width") or w, report.get("frame_height") or h
    sx, sy = w / fw, h / fh
    out = []
    for t in report.get("targets", []):
        if "box_bottom_right" in t and "box_top_left" in t:
            px = [(t["box_top_left"][0] + t["box_bottom_right"][0]) / 2 * sx, t["box_bottom_right"][1] * sy]
        elif "centroid" in t:
            px = [t["centroid"][0] * sx, t["centroid"][1] * sy]
        else:
            continue
        if horizon_row(G, w, h) >= px[1]:
            continue
        en = apply_h(G, [px])[0]
        lat, lon = frame.to_latlon(en[0], en[1])
        r = {"id": t["id"], "lat": float(lat), "lon": float(lon)}
        if t.get("latitude") is not None and t.get("longitude") is not None:
            r["sent_lat"], r["sent_lon"] = t["latitude"], t["longitude"]
            r["apart_m"] = round(frame.distance(lat, lon, t["latitude"], t["longitude"]), 2)
        out.append(r)
    return out


def horizon_row(G: np.ndarray, w: int, h: int) -> float:
    """The first frame row (from the top) below which every pixel lands on the ground in front of the camera; 0 when
    the whole frame is ground, ``h`` when none of the bottom row is."""
    g = G[2]
    sign = np.sign(g @ np.array([w / 2, h - 1, 1.0]))
    if sign == 0:
        return float(h)
    top = 0.0
    for x in (0.0, float(w)):
        z_top = sign * (g[0] * x + g[2])                         # z at row 0
        dz = sign * g[1]                                         # change per row
        if z_top > 1e-12:
            continue                                             # row 0 already in front at this edge
        if dz <= 0:
            return float(h)
        top = max(top, (-z_top / dz) + 1e-6)
    return float(min(h, top))


def footprint(G: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """The four frame corners on the ground (metres), or None when the top of the frame is above the horizon."""
    if horizon_row(G, w, h) > 0:
        return None
    return apply_h(G, [[0, 0], [w, 0], [w, h], [0, h]])


def ground_parts(G: np.ndarray, w: int, h: int, max_range_m: float = 1500.0, far_share: float = 0.5,
                 max_px_m: float | None = 1.5):
    """The frame's footprint on the ground in two parts (metres): the near part (the lower rows of the frame) and the
    far part (the upper rows that still show ground, toward the horizon), as quads ``[[e, n] x 4]``, plus the first
    frame row used. Rows above the horizon, and rows whose ground is farther than ``max_range_m`` from the ground
    under the bottom of the frame or where one frame pixel covers more than ``max_px_m`` of ground (too coarse to
    show anything), are left out, so an oblique camera does not paint the map to the horizon.
    Returns None when no row of the frame is on the ground."""
    hz = horizon_row(G, w, h)
    if hz >= h - 1:
        return None
    r0 = hz + (0.02 * h if hz > 0 else 0.0)                         # a little under the horizon: far rows explode
    if r0 >= h - 1:
        return None
    base = apply_h(G, [[w / 2, h]])[0]

    def too_far(r):
        q = apply_h(G, [[0, r], [w / 2, r], [w, r], [w / 2, r + 1], [w / 2 + 1, r]])
        if np.hypot(q[:3, 0] - base[0], q[:3, 1] - base[1]).max() > max_range_m:
            return True
        return max_px_m is not None and max(np.hypot(*(q[3] - q[1])), np.hypot(*(q[4] - q[1]))) > max_px_m

    if too_far(r0):
        lo, hi = r0, float(h)                                            # find the first row within the range
        if too_far(hi - 1):
            return None
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if too_far(mid) else (lo, mid)
        r0 = hi
    split = r0 + (h - r0) * far_share
    far = apply_h(G, [[0, r0], [w, r0], [w, split], [0, split]])
    near = apply_h(G, [[0, split], [w, split], [w, h], [0, h]])
    return {"near": near, "far": far, "top_row": float(r0), "split_row": float(split)}


def picture_mask(frame: np.ndarray, dark: int = 12) -> np.ndarray:
    """255 where the frame shows picture, 0 on black bars and borders around it (dark areas touching the edge)."""
    g = frame.max(axis=2) if frame.ndim == 3 else frame
    darkm = (g <= dark).astype(np.uint8)
    n, lab = cv2.connectedComponents(darkm, connectivity=4)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bar = np.isin(lab, edge[edge > 0]) & (darkm > 0)
    # a dark car at the edge is not a bar: only areas that span a whole side, or most of one, count
    keep = np.zeros_like(bar)
    for k in edge[edge > 0]:
        m = lab == k
        ys, xs = np.nonzero(m)
        if (xs.max() - xs.min() + 1) >= 0.8 * g.shape[1] or (ys.max() - ys.min() + 1) >= 0.8 * g.shape[0]:
            keep |= m
    return np.where(keep & bar, 0, 255).astype(np.uint8)


def frame_on_ground(frame: np.ndarray, G: np.ndarray, frame_ll: "LocalFrame", bounds: tuple[float, float, float, float],
                    size: tuple[int, int], max_range_m: float = 1500.0) -> np.ndarray:
    """The frame laid on the ground, as a BGRA picture of the area ``bounds`` (top, left, bottom, right latitude and
    longitude) of ``size`` (width, height) pixels. Transparent where the frame does not reach: outside its footprint,
    above the horizon and on black bars, so only picture lands on the map."""
    h, w = frame.shape[:2]
    W, H = size
    parts = ground_parts(G, w, h, max_range_m)
    out = np.zeros((H, W, 4), np.uint8)
    if parts is None:
        return out
    top, left, bottom, right = bounds
    e0, n0 = frame_ll.to_en(top, left)
    e1, n1 = frame_ll.to_en(bottom, right)
    B = np.array([[(e1 - e0) / W, 0, e0], [0, (n1 - n0) / H, n0], [0, 0, 1]], np.float64)   # output pixel to metres
    M = np.linalg.inv(G) @ B                                                                   # output pixel to frame
    pic = cv2.warpPerspective(frame, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    mask = picture_mask(frame)
    mask[: int(math.ceil(parts["top_row"]))] = 0
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))           # no half-black pixel where a bar is blended in
    a = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT)
    poly = np.r_[parts["far"][:2], parts["near"][2:]]                                          # the whole footprint
    px = apply_h(np.linalg.inv(B), poly)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [np.round(px * 8).astype(np.int32)], 255, shift=3)
    out[:, :, :3] = pic
    out[:, :, 3] = cv2.bitwise_and(a, inside)
    return out


REGISTERED_BGR = (60, 200, 60)        # green: placed by the map (or by points picked on the map)
UNREGISTERED_BGR = (60, 60, 230)      # red: flight data only, or not placed
FAR_BGR = {True: (200, 220, 40), False: (40, 140, 255)}   # the far part: teal when registered, orange when not


def draw_footprint(img: np.ndarray, near_px: np.ndarray, far_px: np.ndarray, registered: bool,
                   thickness: int = 2) -> np.ndarray:
    """Draw a footprint (``ground_parts`` in the picture's pixels) on ``img``: the near part green when the frame is
    registered and red when not, the far part in its own colour, each lightly filled."""
    for poly, col in ((far_px, FAR_BGR[registered]), (near_px, REGISTERED_BGR if registered else UNREGISTERED_BGR)):
        p = np.round(np.asarray(poly) * 8).astype(np.int32)
        fill = img.copy()
        cv2.fillPoly(fill, [p], col, lineType=cv2.LINE_AA, shift=3)
        cv2.addWeighted(fill, 0.18, img, 0.82, 0, dst=img)
        cv2.polylines(img, [p], True, col, thickness, cv2.LINE_AA, shift=3)
    return img


def is_registered(fix: "FrameFix", trust_s: float = 5.0) -> bool:
    """Placed by the map: matched now, or chained from a match a few seconds ago. Flight data alone is not."""
    return fix.G is not None and fix.source in ("map", "chain") and fix.since_map_s is not None \
        and fix.since_map_s <= trust_s


def pose_from_telemetry(s: dict, w: int, h: int, frame: LocalFrame, hfov_deg: float | None = None,
                        ground_alt: float | None = None) -> np.ndarray | None:
    """A first ground homography from one flight-data sample (``video_meta.Telemetry.at``), or None.

    Best first: the four frame corners (KLV), then the platform and the frame centre with the field of view (KLV),
    then the position, height above ground and gimbal angles (DJI). ``hfov_deg`` overrides what the data says.
    """
    corners = [s.get(f"corner{i}_{k}") for i in range(1, 5) for k in ("lat", "lon")]
    if all(v is not None for v in corners):
        en = np.array([frame.to_en(corners[2 * i], corners[2 * i + 1]) for i in range(4)], np.float64)
        img = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
        return cv2.getPerspectiveTransform(img.astype(np.float32), en.astype(np.float32)).astype(np.float64)
    hf = hfov_deg or s.get("hfov")
    if hf is None and s.get("focal_mm"):
        hf = math.degrees(2 * math.atan(36.0 / (2 * s["focal_mm"])))   # DJI gives the 35 mm equivalent
    if hf is None or s.get("lat") is None:
        return None
    e, n = frame.to_en(s["lat"], s["lon"])
    if s.get("center_lat") is not None and s.get("alt") is not None:
        ce, cn = frame.to_en(s["center_lat"], s["center_lon"])
        ground = s.get("center_elev", ground_alt if ground_alt is not None else 0.0)
        height = s["alt"] - ground
        d = math.hypot(ce - e, cn - n)
        if height <= 1:
            return None
        yaw = math.degrees(math.atan2(ce - e, cn - n)) if d > 1 else (s.get("heading") or 0.0)
        pitch = -math.degrees(math.atan2(height, d))
        return pose_homography(w, h, hf, float(e), float(n), height, yaw, pitch, 0.0)
    height = s.get("rel_alt", s.get("agl"))
    if height is None and s.get("alt") is not None and ground_alt is not None:
        height = s["alt"] - ground_alt
    if height is None or height <= 1:
        return None
    if s.get("sensor_az") is not None and s.get("sensor_el") is not None:
        # KLV with no frame centre: the platform's attitude plus the sensor's angles relative to it (roll ignored;
        # the map match removes what is left).
        yaw = (s.get("heading", 0.0) + s["sensor_az"]) % 360
        pitch = s.get("pitch", 0.0) + s["sensor_el"]
        if pitch > -5:
            return None
        return pose_homography(w, h, hf, float(e), float(n), height, yaw, max(pitch, -90.0), 0.0)
    pitch = s.get("pitch", -90.0)
    if pitch > -5:
        return None                                    # looking at the horizon: no ground to place
    return pose_homography(w, h, hf, float(e), float(n), height, s.get("heading", 0.0), pitch, s.get("roll", 0.0))


# ------------------------------------------------------------------ matching

class Matcher:
    """Point matches between two pictures: ALIKED + LightGlue (ONNX model pack) when given, otherwise SIFT."""

    SIZE = 640

    def __init__(self, pack: Path | str | None = None, max_side: int = 640):
        self.max_side = max_side
        self.kind = "sift"
        self._lock = threading.Lock()
        if pack:
            import onnxruntime as ort
            pack = Path(pack)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, (cv2.getNumberOfCPUs() or 2) - 1)
            ext = next(iter(sorted(pack.glob("aliked*.onnx"))), None)
            lg = pack / "lightglue_for_aliked.onnx"
            if ext is None or not lg.is_file():
                raise FileNotFoundError(f"No ALIKED and LightGlue files in {pack}.")
            self.ext = ort.InferenceSession(str(ext), opts, providers=["CPUExecutionProvider"])
            self.lg = ort.InferenceSession(str(lg), opts, providers=["CPUExecutionProvider"])
            self.kind = "aliked-lightglue"
        else:
            self.sift = cv2.SIFT_create(nfeatures=3000)

    def _square(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        s = self.SIZE / max(h, w)
        r = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out = np.zeros((self.SIZE, self.SIZE, 3), np.uint8)
        if r.ndim == 2:
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        out[:r.shape[0], :r.shape[1]] = r
        return out, s

    def features(self, img: np.ndarray, mask: np.ndarray | None = None):
        """Keypoints (pixels of ``img``) and descriptors."""
        if self.kind == "sift":
            g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            s = min(1.0, self.max_side / max(g.shape[:2]))
            gs = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
            ms = None if mask is None else cv2.resize(mask, (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_NEAREST)
            gs = cv2.createCLAHE(2.0, (8, 8)).apply(gs)
            kp, des = self.sift.detectAndCompute(gs, ms)
            if des is None:
                return np.zeros((0, 2), np.float32), None
            des = np.sqrt(des / (des.sum(1, keepdims=True) + 1e-7))      # RootSIFT
            return np.float32([k.pt for k in kp]) / s, des.astype(np.float32)
        sq, s = self._square(img)
        x = sq[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        with self._lock:
            k, d, sc = self.ext.run(None, {"image": x})
        px = (k + 1) * (self.SIZE / 2) - 0.5
        keep = (sc > 0) & (px[:, 0] < img.shape[1] * s) & (px[:, 1] < img.shape[0] * s)
        if mask is not None:
            mi = np.clip((px / s).astype(int), 0, [img.shape[1] - 1, img.shape[0] - 1])
            keep &= mask[mi[:, 1], mi[:, 0]] > 0
        return (px[keep] / s).astype(np.float32), (k[keep], d[keep])

    def match_features(self, fa, fb) -> tuple[np.ndarray, np.ndarray]:
        (pa, da), (pb, db) = fa, fb
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        if self.kind == "sift":
            pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
            good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.85 * m[1].distance]
            return pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]]
        (ka, dsa), (kb, dsb) = da, db
        with self._lock:
            m, sc = self.lg.run(None, {"kpts0": ka[None].astype(np.float32), "kpts1": kb[None].astype(np.float32),
                                       "desc0": dsa[None].astype(np.float32), "desc1": dsb[None].astype(np.float32)})
        m = m[sc > 0.2] if len(m) else m
        return pa[m[:, 0]], pb[m[:, 1]]

    def match(self, a: np.ndarray, b: np.ndarray, mask_a=None, mask_b=None):
        return self.match_features(self.features(a, mask_a), self.features(b, mask_b))


def fit_homography(pa: np.ndarray, pb: np.ndarray, thresh: float = 4.0):
    """A homography from ``pa`` to ``pb`` (MAGSAC), with its inlier mask, or (None, mask)."""
    if len(pa) < 8:
        return None, np.zeros(len(pa), bool)
    H, m = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, thresh, maxIters=5000, confidence=0.999)
    if H is None:
        return None, np.zeros(len(pa), bool)
    return H, m.ravel().astype(bool)


# ------------------------------------------------------------------ snapping a frame to the map

@dataclass
class Snap:
    G: np.ndarray | None                 # corrected frame pixels to metres, or None when the match was refused
    inliers: int
    matches: int
    shift_m: float | None = None         # how far the frame centre moved
    seconds: float = 0.0
    reason: str = ""


def snap_to_map(frame: np.ndarray, G: np.ndarray, gmap: GeoMap, matcher: Matcher, view_side: int = 640,
                min_inliers: int = 25, max_shift_frac: float = 0.6, features=None) -> Snap:
    """Match the frame against the map drawn from the camera's current estimate ``G``; the corrected ``G``."""
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    top = horizon_row(G, w, h)
    if top >= h - 8:
        return Snap(None, 0, 0, reason="no ground in view", seconds=time.perf_counter() - t0)
    top = int(min(h - 8, top + 0.08 * h)) if top > 0 else 0              # leave out the far distance
    s = view_side / max(w, h - top)
    vw, vh = max(16, round(w * s)), max(16, round((h - top) * s))
    V = np.array([[1 / s, 0, 0], [0, 1 / s, top], [0, 0, 1]])           # view pixels to frame pixels
    T = gmap.A_inv @ G @ V                                              # view pixels to map pixels
    drawn = gmap.render(T, (vw, vh))
    valid = (drawn.max(axis=2) > 0).astype(np.uint8)
    if valid.mean() < 0.3:
        return Snap(None, 0, 0, reason="outside the map", seconds=time.perf_counter() - t0)
    view = cv2.resize(frame[top:], (vw, vh), interpolation=cv2.INTER_AREA)
    fv = features if features is not None else matcher.features(view)
    pa, pb = matcher.match_features(fv, matcher.features(drawn, cv2.erode(valid, np.ones((5, 5), np.uint8))))
    Hc, inl = fit_homography(pa, pb, 3.0)
    n = int(inl.sum())
    if Hc is None or n < min_inliers:
        return Snap(None, n, len(pa), reason="too few matches", seconds=time.perf_counter() - t0)
    # The correction must keep the picture a picture: no fold, no wild stretch.
    corners = np.float64([[0, 0], [vw, 0], [vw, vh], [0, vh]])
    cc = apply_h(Hc, corners)
    area0, area1 = vw * vh, abs(cv2.contourArea(cc.astype(np.float32)))
    if not cv2.isContourConvex(cc.astype(np.float32)) or not 0.5 < area1 / area0 < 2.0:
        return Snap(None, n, len(pa), reason="implausible correction", seconds=time.perf_counter() - t0)
    G_new = G @ V @ Hc @ np.linalg.inv(V)
    G_new /= G_new[2, 2]
    c = [[w / 2, (top + h) / 2]]
    shift = float(np.linalg.norm(apply_h(G_new, c) - apply_h(G, c)))
    fp = apply_h(G, [[0, top], [w, top], [w, h], [0, h]])
    size = math.sqrt(abs(cv2.contourArea(fp.astype(np.float32))))
    if shift > max_shift_frac * size:
        return Snap(None, n, len(pa), shift, reason="correction too large", seconds=time.perf_counter() - t0)
    return Snap(G_new, n, len(pa), shift, time.perf_counter() - t0)


def locate(frame: np.ndarray, gmap: GeoMap, matcher: Matcher, m_per_px: float, region=None,
           headings=(0, 90, 180, 270), overlap: float = 0.5, min_inliers: int = 25) -> Snap:
    """Find a roughly straight-down frame on the map with no flight data: a search over map tiles and headings.

    ``m_per_px`` is a guess of the frame's ground size per pixel (height and lens, or a past fix); ``region`` an
    optional (east0, north0, east1, north1) in metres to search in. Slow (seconds to minutes): for the first fix only.
    """
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    fw, fh = w * m_per_px, h * m_per_px                                  # frame footprint, metres
    side = max(fw, fh) * 1.4                                            # a tile holds the frame at any heading
    A = gmap.A
    H0, W0 = gmap.image.shape[:2]
    e0, n0 = apply_h(A, [[0, 0]])[0]
    e1, n1 = apply_h(A, [[W0, H0]])[0]
    if region:
        e0, n1, e1, n0 = max(e0, region[0]), max(n1, region[1]), min(e1, region[2]), min(n0, region[3])
    step = side * (1 - overlap)
    fframe = matcher.features(frame)
    best = Snap(None, 0, 0, reason="nothing matched")
    tile_px = 640
    for cn in np.arange(n0 - side / 2, n1 - side / 2 - step, -step) if n0 > n1 else []:
        for ce in np.arange(e0 + side / 2, e1 - side / 2 + step, step):
            # tile pixels to metres: a north-up square of ``side`` metres centred on (ce, cn)
            k = side / tile_px
            Tm = np.array([[k, 0, ce - side / 2], [0, -k, cn + side / 2], [0, 0, 1]])
            tile = gmap.render(gmap.A_inv @ Tm, (tile_px, tile_px))
            if (tile.max(axis=2) > 0).mean() < 0.5:
                continue
            ft = matcher.features(tile)
            for hd in headings:
                if hd and matcher.kind == "sift":
                    continue                                             # SIFT is rotation invariant already
                if hd:
                    R = cv2.getRotationMatrix2D((tile_px / 2, tile_px / 2), hd, 1.0)
                    rt = cv2.warpAffine(tile, R, (tile_px, tile_px))
                    f2 = matcher.features(rt)
                    Rh = np.vstack([cv2.invertAffineTransform(R), [0, 0, 1]])
                else:
                    f2, Rh = ft, np.eye(3)
                pa, pb = matcher.match_features(fframe, f2)
                Hc, inl = fit_homography(pa, pb, 4.0)
                n = int(inl.sum())
                if Hc is not None and n > best.inliers:
                    G = Tm @ Rh @ Hc
                    best = Snap(G / G[2, 2], n, len(pa))
    best.seconds = time.perf_counter() - t0
    if best.G is not None and best.inliers < min_inliers:
        best = Snap(None, best.inliers, best.matches, reason="too few matches", seconds=best.seconds)
    return best


# ------------------------------------------------------------------ frame to frame

class FlowTracker:
    """Frame-to-frame homography to a keyframe, from corners followed with pyramidal optical flow."""

    def __init__(self, width: int = 640, max_corners: int = 600, min_inliers: int = 40, rekey_frac: float = 0.55):
        self.width = width
        self.max_corners, self.min_inliers, self.rekey_frac = max_corners, min_inliers, rekey_frac
        self.key_pts = self.cur_pts = self.prev = None
        self.scale = 1.0
        self.n0 = 0
        self.orb = cv2.ORB_create(1500)

    def _gray(self, frame):
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.scale = self.width / g.shape[1]
        return cv2.resize(g, (self.width, round(g.shape[0] * self.scale)), interpolation=cv2.INTER_AREA)

    def start(self, frame) -> None:
        g = self._gray(frame)
        self.key_gray = g
        pts = cv2.goodFeaturesToTrack(g, self.max_corners, 0.01, 8, blockSize=7)
        self.key_pts = self.cur_pts = pts if pts is not None else np.zeros((0, 1, 2), np.float32)
        self.n0 = len(self.key_pts)
        self.prev = g

    def _orb_to_key(self, g):
        k1, d1 = self.orb.detectAndCompute(self.key_gray, None)
        k2, d2 = self.orb.detectAndCompute(g, None)
        if d1 is None or d2 is None:
            return None, 0
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
        good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 12:
            return None, len(good)
        a = np.float32([k2[m.queryIdx].pt for m in good])
        b = np.float32([k1[m.trainIdx].pt for m in good])
        H, inl = fit_homography(a, b, 3.0)
        return H, int(inl.sum())

    def step(self, frame) -> tuple[np.ndarray | None, int, bool]:
        """(homography from this frame's pixels to the keyframe's pixels, inliers, whether to start a new keyframe)."""
        g = self._gray(frame)
        H = None
        n = 0
        if len(self.cur_pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.cur_pts, None, winSize=(21, 21), maxLevel=4)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, nxt, None, winSize=(21, 21), maxLevel=4)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - self.cur_pts, axis=2).ravel() < 1.0)
            self.key_pts, self.cur_pts = self.key_pts[ok], nxt[ok]
            if len(self.cur_pts) >= 8:
                H, inl = fit_homography(self.cur_pts.reshape(-1, 2), self.key_pts.reshape(-1, 2), 2.0)
                n = int(inl.sum())
                if H is not None:
                    self.key_pts, self.cur_pts = self.key_pts[inl], self.cur_pts[inl]
        if H is None or n < self.min_inliers:
            H, n = self._orb_to_key(g)                                   # a jump the flow lost: match again
        self.prev = g
        rekey = H is None or n < max(self.min_inliers, self.rekey_frac * self.n0)
        if H is None:
            return None, n, True
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ H @ S, n, rekey


# ------------------------------------------------------------------ drone and aircraft video

@dataclass
class FrameFix:
    G: np.ndarray | None              # frame pixels to metres, or None when the frame is lost
    source: str                       # "flight-data" | "map" | "chain" | "lost"
    since_map_s: float | None         # seconds since the last match to the map
    track_ms: float = 0.0


@dataclass
class DroneRegistrar:
    """Real-time registration of drone or aircraft video: chain frame to frame, snap to the map now and then.

    Call ``step(frame, t, telemetry_sample)`` for each frame in order. With ``async_snap`` the map match runs beside
    the tracking (real time); without it every snap is waited for (offline, a little more accurate).
    """
    gmap: GeoMap
    matcher: Matcher
    snap_every_s: float = 1.0
    async_snap: bool = True
    hfov_deg: float | None = None
    ground_alt: float | None = None
    trust_flight_data_s: float = 5.0        # with no map match for this long, go back to the flight data
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.tracker = FlowTracker()
        self.G_key = None
        self.last_snap_t = -1e9
        self.last_map_t = None
        self.pending: Future | None = None
        self.pending_G = None
        self.pool = ThreadPoolExecutor(1) if self.async_snap else None
        self.t = 0.0
        self.reset_t = -1e9

    def _prior(self, frame, sample):
        if not sample:
            return None
        h, w = frame.shape[:2]
        return pose_from_telemetry(sample, w, h, self.gmap.frame, self.hfov_deg, self.ground_alt)

    def start(self, frame, G: np.ndarray | None = None, sample: dict | None = None, t: float = 0.0) -> FrameFix:
        G = G if G is not None else self._prior(frame, sample)
        self.tracker.start(frame)
        self.G_key, self.t = G, t
        src = "flight-data" if G is not None else "lost"
        if G is not None:
            self._snap(frame, G, t, wait=True)
            if self.last_map_t == t:
                src = "map"
        return FrameFix(self.G_key, src, 0.0 if self.last_map_t == t else None)

    def _apply(self, snap: Snap, G_at_snap: np.ndarray, t: float):
        self.log.append({"t": t, "ok": snap.G is not None, "inliers": snap.inliers, "matches": snap.matches,
                         "shift_m": None if snap.shift_m is None else round(snap.shift_m, 2),
                         "s": round(snap.seconds, 3), "why": snap.reason})
        if snap.G is None or self.reset_t > t:
            return                                                       # refused, or the chain restarted since
        # The correction is metres to metres, so it holds for every later frame of the same chain, keyframes or not.
        F = snap.G @ np.linalg.inv(G_at_snap)
        self.G_key = F @ self.G_key
        self.G_key /= self.G_key[2, 2]
        self.last_map_t = t

    def _snap(self, frame, G, t, wait=False):
        self.last_snap_t = t
        if not wait and self.pool is not None:
            G0 = G.copy()
            self.pending = self.pool.submit(snap_to_map, frame.copy(), G0, self.gmap, self.matcher)
            self.pending_G, self.pending_t = G0, t
            return
        self._apply(snap_to_map(frame, G, self.gmap, self.matcher), G, t)

    def step(self, frame, t: float, sample: dict | None = None) -> FrameFix:
        t0 = time.perf_counter()
        if self.G_key is None:
            fix = self.start(frame, sample=sample, t=t)
            fix.track_ms = (time.perf_counter() - t0) * 1000
            return fix
        if self.pending is not None and self.pending.done():
            self._apply(self.pending.result(), self.pending_G, self.pending_t)
            self.pending = None
        M, n, rekey = self.tracker.step(frame)
        src = "chain"
        stale = self.last_map_t is None or t - self.last_map_t > self.trust_flight_data_s
        prior = self._prior(frame, sample) if (sample and (M is None or stale)) else None
        if M is None:                                                    # lost the picture (a cut, a blur)
            self.tracker.start(frame)
            self.reset_t = t
            self.G_key = prior
            if prior is None:
                return FrameFix(None, "lost", None, (time.perf_counter() - t0) * 1000)
            G, src = prior, "flight-data"
        else:
            G = self.G_key @ M
            G /= G[2, 2]
            if prior is not None:                                        # no map match lately: the flight data rules
                self.G_key, G, src = prior @ np.linalg.inv(M), prior, "flight-data"
            if rekey:
                self.tracker.start(frame)
                self.G_key = G
        if t - self.last_snap_t >= self.snap_every_s and self.pending is None:
            self._snap(frame, G, t, wait=not self.async_snap)
            if not self.async_snap and self.last_map_t == t:
                G = self.G_key @ (np.eye(3) if rekey or M is None else M)
                G /= G[2, 2]
                src = "map"
        elif self.last_map_t is not None and src == "chain" and t - self.last_map_t < 1e-9:
            src = "map"
        since = None if self.last_map_t is None else t - self.last_map_t
        return FrameFix(G, src, since, (time.perf_counter() - t0) * 1000)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)


# ------------------------------------------------------------------ fixed cameras

@dataclass
class PointFit:
    G: np.ndarray                    # frame pixels to metres
    frame: LocalFrame
    residual_m: list[float]          # each point's error with every point in the fit
    leave_one_out_m: list[float]     # each point's error when it is left out of the fit (what a new point would see)
    suspect: int | None = None       # the point most likely picked wrong (the others agree without it), if any

    def to_latlon(self, px) -> np.ndarray:
        en = apply_h(self.G, px)
        lat, lon = self.frame.to_latlon(en[:, 0], en[:, 1])
        return np.c_[lat, lon]


def fit_points(px, latlon, frame: LocalFrame | None = None) -> PointFit:
    """A fixed camera's ground homography from four or more points picked on the frame and on the map."""
    px = np.asarray(px, np.float64)
    ll = np.asarray(latlon, np.float64)
    if len(px) < 4 or len(px) != len(ll):
        raise ValueError("At least four points, each on the frame and on the map.")
    frame = frame or LocalFrame(float(ll[:, 0].mean()), float(ll[:, 1].mean()))
    e, n = frame.to_en(ll[:, 0], ll[:, 1])
    en = np.c_[e, n]

    def fit(i):
        if i.sum() == 4:
            return cv2.getPerspectiveTransform(px[i].astype(np.float32), en[i].astype(np.float32)).astype(np.float64)
        G, _ = cv2.findHomography(px[i], en[i], 0)
        return G

    allm = np.ones(len(px), bool)
    G = fit(allm)
    if G is None:
        raise ValueError("The points do not fit a ground plane (three of them in a line?).")
    res = np.linalg.norm(apply_h(G, px) - en, axis=1).tolist()
    loo, others = [], []
    for k in range(len(px)):
        if len(px) < 5:
            break
        m = allm.copy()
        m[k] = False
        Gk = fit(m)
        loo.append(float(np.linalg.norm(apply_h(Gk, px[k:k + 1]) - en[k])) if Gk is not None else float("nan"))
        if Gk is not None and m.sum() > 4:
            others.append(float(np.sqrt(np.mean(np.sum((apply_h(Gk, px[m]) - en[m]) ** 2, axis=1)))))
        else:
            others.append(float("inf"))
    suspect = None
    if len(others) >= 6:
        k = int(np.argmin(others))
        rms_all = float(np.sqrt(np.mean(np.square(res))))
        # Without it the rest fit at least three times better, and it is far off when left out.
        if others[k] * 3 < rms_all and loo[k] > 1.0:
            suspect = k
    return PointFit(G / G[2, 2], frame, [round(r, 2) for r in res], [round(r, 2) for r in loo], suspect)


def refine_with_map(frame: np.ndarray, fit: PointFit | np.ndarray, gmap: GeoMap, matcher: Matcher, **kw) -> Snap:
    """Pull a fixed camera's hand-made fit onto the map by matching (works when the view is steep enough)."""
    G = fit.G if isinstance(fit, PointFit) else fit
    if isinstance(fit, PointFit) and fit.frame != gmap.frame:
        # bring the fit into the map's metres
        T = np.eye(3)
        e, n = gmap.frame.to_en(fit.frame.lat0, fit.frame.lon0)
        T[0, 2], T[1, 2] = float(e), float(n)
        G = T @ G
    return snap_to_map(frame, G, gmap, matcher, **kw)


# ------------------------------------------------------------------ dashcams

@dataclass
class DashcamGround:
    """The road in front of a dashcam, in metres, and (with GPS) on the map.

    ``height_m`` is the lens above the road, ``pitch_deg`` below 0 when the camera looks down (``pitch_from_horizon``
    gives it from the horizon's row), ``hfov_deg`` the lens's horizontal field of view.
    """
    w: int
    h: int
    hfov_deg: float = 100.0
    height_m: float = 1.3
    pitch_deg: float = -3.0
    roll_deg: float = 0.0
    K: np.ndarray | None = None          # the lens, when known exactly (overrides hfov_deg)

    def __post_init__(self):
        # vehicle frame: x right, y forward, metres from the point on the road under the camera
        self.G = pose_homography(self.w, self.h, self.hfov_deg, 0.0, 0.0, self.height_m, 0.0, self.pitch_deg,
                                 self.roll_deg, self.K)
        self.horizon = horizon_row(self.G, self.w, self.h)

    @staticmethod
    def pitch_from_horizon(row: float, h: int, w: int, hfov_deg: float) -> float:
        """The camera's pitch from the row of the horizon (a row above the centre: looking down, pitch below 0)."""
        f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        return -math.degrees(math.atan((h / 2 - row) / f))

    def to_vehicle(self, px) -> np.ndarray:
        """Frame pixels on the road to metres (right, ahead) of the camera."""
        return apply_h(self.G, px)

    def to_world(self, px, east: float, north: float, heading_deg: float) -> np.ndarray:
        v = self.to_vehicle(px)
        a = math.radians(heading_deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        return v @ R.T + [east, north]

    def road_mask(self, near_m: float = 3.0, far_m: float = 25.0, half_width_m: float = 5.0) -> np.ndarray:
        """Frame pixels of the road ahead, from ``near_m`` to ``far_m`` (where visual odometry looks)."""
        quad = np.float64([[-half_width_m, near_m], [half_width_m, near_m], [half_width_m, far_m],
                           [-half_width_m, far_m]])
        px = apply_h(np.linalg.inv(self.G), quad)
        m = np.zeros((self.h, self.w), np.uint8)
        cv2.fillConvexPoly(m, px.astype(np.int32), 1)
        m[: int(self.horizon) + 2] = 0
        return m


class DashcamOdometry:
    """The car's motion between frames from the road's own movement (ground-plane visual odometry)."""

    def __init__(self, ground: DashcamGround, width: int = 640):
        self.g = ground
        self.scale = width / ground.w
        self.mask = cv2.resize(ground.road_mask(), (width, round(ground.h * self.scale)), interpolation=cv2.INTER_NEAREST)
        self.prev = None
        self.pose = np.zeros(3)                                          # east, north, heading (deg) by odometry

    def step(self, frame) -> tuple[float, float, float, int] | None:
        """(metres forward, metres right, degrees turned right, inliers) since the last frame, or None."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        g = cv2.resize(g, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, g
        if prev is None:
            return None
        pts = cv2.goodFeaturesToTrack(prev, 400, 0.005, 6, mask=self.mask, blockSize=5)
        if pts is None or len(pts) < 10:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            return None
        a = self.g.to_vehicle(pts[ok].reshape(-1, 2) / self.scale)
        b = self.g.to_vehicle(nxt[ok].reshape(-1, 2) / self.scale)
        M, inl = cv2.estimateAffinePartial2D(b, a, method=cv2.RANSAC, ransacReprojThreshold=0.15)
        if M is None or inl.sum() < 8:
            return None
        # M maps road points as seen now to the same points as seen before: the car's motion in its old frame.
        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        dx, dy = M[0, 2], M[1, 2]
        return float(dy), float(dx), float(-ang), int(inl.sum())


def gps_heading(e: np.ndarray, n: np.ndarray, min_move_m: float = 2.0) -> np.ndarray:
    """Heading (deg clockwise from north) along a GPS track; held while the car moves less than ``min_move_m``."""
    out = np.full(len(e), np.nan)
    last = 0
    for i in range(1, len(e)):
        d = math.hypot(e[i] - e[last], n[i] - n[last])
        if d >= min_move_m:
            out[last:i + 1] = math.degrees(math.atan2(e[i] - e[last], n[i] - n[last]))
            last = i
    valid = ~np.isnan(out)
    if valid.any():
        idx = np.where(valid)[0]
        out[: idx[0]] = out[idx[0]]
        for i in range(len(out)):
            if np.isnan(out[i]):
                out[i] = out[i - 1]
    return out


class DashcamTrack:
    """Where the car is and which way it faces, from visual odometry and (when the video has it) GPS.

    Between GPS fixes the car moves by the road's own motion; at each fix the position is pulled toward the GPS by
    ``pos_gain`` and the heading toward the GPS course (once the car has moved ``course_m`` since the fix that
    course is measured from) by ``heading_gain``. With no GPS at all the track is relative to where it started.
    """

    def __init__(self, ground: DashcamGround, pos_gain: float = 0.35, heading_gain: float = 0.5,
                 course_m: float = 15.0):
        self.vo = DashcamOdometry(ground)
        self.ground = ground
        self.pos_gain, self.heading_gain, self.course_m = pos_gain, heading_gain, course_m
        self.e = self.n = 0.0
        self.heading = 0.0
        self.placed = False                     # on the map (a GPS fix seen)
        self.oriented = False                   # heading from GPS (otherwise relative to the start)
        self.fixes: list[tuple[float, float]] = []
        self.vo_since_fix = np.zeros(2)         # odometry's own displacement since the course's first fix
        self.scale = 1.0                        # odometry metres to GPS metres (a wrong height or pitch shows here)

    def step(self, frame, gps_en: tuple[float, float] | None = None) -> tuple[float, float, float]:
        m = self.vo.step(frame)
        if m is not None:
            fwd, right, turn, _ = m
            a = math.radians(self.heading)
            fwd, right = fwd * self.scale, right * self.scale
            de = fwd * math.sin(a) + right * math.cos(a)
            dn = fwd * math.cos(a) - right * math.sin(a)
            self.e += de
            self.n += dn
            self.vo_since_fix += (de, dn)
            self.heading = (self.heading + turn) % 360
        if gps_en is not None:
            ge, gn = gps_en
            if not self.placed:
                self.e, self.n, self.placed = ge, gn, True
            else:
                self.e += self.pos_gain * (ge - self.e)
                self.n += self.pos_gain * (gn - self.n)
            self.fixes.append((ge, gn))
            if len(self.fixes) >= 2:
                (e0, n0) = self.fixes[0]
                d = math.hypot(ge - e0, gn - n0)
                if d >= self.course_m:
                    # The GPS course over the last stretch against the odometry's course over the same stretch.
                    gps_course = math.degrees(math.atan2(ge - e0, gn - n0))
                    vo_course = math.degrees(math.atan2(*self.vo_since_fix))
                    err = (gps_course - vo_course + 180) % 360 - 180
                    gain = 1.0 if not self.oriented else self.heading_gain
                    self.heading = (self.heading + gain * err) % 360
                    vo_d = float(np.hypot(*self.vo_since_fix))
                    if vo_d > 0.5 * self.course_m:
                        self.scale *= float(np.clip(d / vo_d, 0.8, 1.25)) ** 0.3
                    self.oriented = True
                    self.fixes = [(ge, gn)]
                    self.vo_since_fix = np.zeros(2)
        return self.e, self.n, self.heading


# ------------------------------------------------------------------ command line

def load_map(path: str, corners: str | None = None) -> GeoMap:
    """A GeoTIFF in latitude and longitude, or any picture with ``corners`` "lat_top,lon_left,lat_bottom,lon_right"."""
    if corners:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Cannot read the map picture {path}.")
        a, b, c, d = (float(x) for x in corners.split(","))
        return GeoMap(img, (a, b), (c, d))
    return GeoMap.from_geotiff(path)


def _drone(args) -> None:
    import json
    import av
    from .video_meta import find
    gmap = load_map(args.map, args.corners)
    matcher = Matcher(args.model_pack)
    tel = None if args.no_flight_data else find(Path(args.video))
    ground_alt = None
    if args.dtm:
        from .terrain import load_terrain
        c = gmap.frame
        ground_alt = load_terrain(args.dtm).height_at(c.lat0, c.lon0)
        ground_alt = None if ground_alt is None or not np.isfinite(ground_alt) else float(ground_alt)
    reg = DroneRegistrar(gmap, matcher, snap_every_s=args.every, async_snap=not args.offline, hfov_deg=args.hfov,
                         ground_alt=ground_alt)
    out, writer, map_writer = [], None, None
    first_G = None
    with av.open(args.video) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        t_start = None
        for fr in c.decode(vs):
            t = float(fr.pts * fr.time_base) if fr.pts is not None else len(out) / 30.0
            t_start = t if t_start is None else t_start
            t -= t_start
            if args.max_seconds and t > args.max_seconds:
                break
            img = fr.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            sample = tel.at(t) if tel else None
            if not out and sample is None and args.start:
                lat, lon, mpp, hd = (float(x) for x in args.start.split(","))
                e, n = gmap.frame.to_en(lat, lon)
                first_G = pose_homography(w, h, math.degrees(2 * math.atan(w * mpp / 2 / 100)), float(e), float(n),
                                          100.0, hd, -90.0)
            if not out and sample is None and first_G is None:
                s = locate(img, gmap, matcher, args.mpp or 0.2)
                if s.G is None:
                    raise SystemExit(f"Could not find the first frame on the map ({s.reason}). Give --start.")
                first_G = s.G
            fix = reg.start(img, first_G, sample, t) if not out else reg.step(img, t, sample)
            rec = {"t": round(t, 3), "source": fix.source, "ms": round(fix.track_ms, 1),
                   "registered": is_registered(fix, reg.trust_flight_data_s)}
            if fix.G is not None:
                fp = footprint(fix.G, w, h)
                c0 = apply_h(fix.G, [[w / 2, h / 2]])[0] if horizon_row(fix.G, w, h) < h / 2 else None
                if c0 is not None:
                    lat, lon = gmap.frame.to_latlon(c0[0], c0[1])
                    rec["center"] = [round(float(lat), 7), round(float(lon), 7)]
                if fp is not None:
                    lat, lon = gmap.frame.to_latlon(fp[:, 0], fp[:, 1])
                    rec["corners"] = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]
            out.append(rec)
            if args.overlay:
                if writer is None:
                    writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), float(vs.average_rate or 30),
                                             (w // 2 * 2, h))
                view = img.copy()
                if fix.G is not None:
                    T = gmap.A_inv @ fix.G
                    drawn = gmap.render(T, (w, h))
                    edges = cv2.Canny(cv2.cvtColor(drawn, cv2.COLOR_BGR2GRAY), 60, 160)
                    view[edges > 0] = (0, 255, 255)
                cv2.putText(view, f"{t:6.1f}s {fix.source}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                writer.write(view[:, : w // 2 * 2])
            if args.map_overlay:
                side = 1024
                if map_writer is None:
                    map_writer = cv2.VideoWriter(args.map_overlay, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 float(vs.average_rate or 30), (side, side))
                map_writer.write(map_view(gmap, img, fix.G, rec["registered"], side))
    reg.close()
    for wr in (writer, map_writer):
        if wr is not None:
            wr.release()
    ms = [r["ms"] for r in out]
    summary = {"frames": len(out), "flight_data": tel.source if tel else None, "matcher": matcher.kind,
               "map_matches": len(reg.log), "map_matches_ok": sum(x["ok"] for x in reg.log),
               "track_ms_median": float(np.median(ms)) if ms else None,
               "match_s_median": float(np.median([x["s"] for x in reg.log])) if reg.log else None,
               "sources": {k: sum(r["source"] == k for r in out) for k in {r["source"] for r in out}}}
    Path(args.out).write_text(json.dumps({"summary": summary, "frames": out, "matches": reg.log}), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def map_view(gmap: GeoMap, frame: np.ndarray, G: np.ndarray | None, registered: bool, side: int = 1024,
             view_m: float | None = None) -> np.ndarray:
    """The map around the frame's footprint with the frame laid on it (no black edges) and the footprint drawn:
    the near part green when registered and red when not, the far part in its own colour."""
    h, w = frame.shape[:2]
    parts = ground_parts(G, w, h) if G is not None else None
    if parts is None:
        img = gmap.level(0)
        out = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        cv2.putText(out, "not on the map", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, UNREGISTERED_BGR, 2)
        return out
    poly = np.r_[parts["far"], parts["near"]]
    c = (poly.min(axis=0) + poly.max(axis=0)) / 2
    span = view_m or float(np.ptp(poly, axis=0).max() * 1.3 + 20)
    k = span / side
    V = np.array([[k, 0, c[0] - span / 2], [0, -k, c[1] + span / 2], [0, 0, 1]], np.float64)   # view px to metres
    out = gmap.render(gmap.A_inv @ V, (side, side))
    lat_t, lon_l = gmap.frame.to_latlon(c[0] - span / 2, c[1] + span / 2)
    lat_b, lon_r = gmap.frame.to_latlon(c[0] + span / 2, c[1] - span / 2)
    lay = frame_on_ground(frame, G, gmap.frame, (float(lat_t), float(lon_l), float(lat_b), float(lon_r)), (side, side))
    a = lay[:, :, 3:4].astype(np.float32) / 255 * 0.85
    out = (out * (1 - a) + lay[:, :, :3] * a).astype(np.uint8)
    Vi = np.linalg.inv(V)
    draw_footprint(out, apply_h(Vi, parts["near"]), apply_h(Vi, parts["far"]), registered)
    cv2.putText(out, "registered on the map" if registered else "flight data only / not registered", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, REGISTERED_BGR if registered else UNREGISTERED_BGR, 2)
    return out


def _fixed(args) -> None:
    import csv
    import json
    rows = [r for r in csv.reader(open(args.points, encoding="utf-8")) if r and not r[0].startswith("#")]
    px = [[float(r[0]), float(r[1])] for r in rows]
    ll = [[float(r[2]), float(r[3])] for r in rows]
    fit = fit_points(px, ll)
    res = {"points": len(px), "residual_m": fit.residual_m, "leave_one_out_m": fit.leave_one_out_m,
           "homography": fit.G.tolist(), "origin": [fit.frame.lat0, fit.frame.lon0]}
    if args.map and args.frame:
        gmap = load_map(args.map, args.corners)
        frame = cv2.imread(args.frame)
        s = refine_with_map(frame, fit, gmap, Matcher(args.model_pack))
        res["refined"] = {"ok": s.G is not None, "inliers": s.inliers, "moved_m": s.shift_m, "why": s.reason}
        if s.G is not None:
            res["refined"]["homography"] = s.G.tolist()
            res["refined"]["origin"] = [gmap.frame.lat0, gmap.frame.lon0]
    print(json.dumps(res, indent=1))


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="trafficwb-georeg", description="Place video frames on the map.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drone", help="Register a drone or aircraft video to a map, frame by frame.")
    d.add_argument("video")
    d.add_argument("--map", required=True, help="Orthophoto or satellite GeoTIFF (latitude/longitude), or a picture "
                                                "with --corners.")
    d.add_argument("--corners", help="lat_top,lon_left,lat_bottom,lon_right of a map picture that is not a GeoTIFF.")
    d.add_argument("--model-pack", help="Folder with the ALIKED and LightGlue ONNX files (SIFT without it).")
    d.add_argument("--every", type=float, default=1.0, help="Seconds between map matches (default 1).")
    d.add_argument("--hfov", type=float, help="The camera's horizontal field of view, degrees (else from flight data).")
    d.add_argument("--no-flight-data", action="store_true", help="Ignore KLV and DJI flight data.")
    d.add_argument("--start", help="lat,lon,metres_per_pixel,heading of the first frame (no flight data).")
    d.add_argument("--mpp", type=float, help="Guess of metres per pixel for the map search (no flight data).")
    d.add_argument("--offline", action="store_true", help="Wait for every map match (not real time; a little better).")
    d.add_argument("--max-seconds", type=float)
    d.add_argument("--overlay", help="Write a video with the map's edges drawn over each frame.")
    d.add_argument("--map-overlay", help="Write a video of the map with each frame laid on it and its footprint "
                                         "(green: registered on the map; red: flight data only).")
    d.add_argument("--dtm", help="Ground heights (DTM GeoTIFF in latitude/longitude, or SRTM .hgt): the ground under "
                                 "the map is used for flight data that gives height above sea level.")
    d.add_argument("--out", default="georeg.json")
    f = sub.add_parser("fixed", help="A fixed camera from points picked on the frame and on the map.")
    f.add_argument("points", help="CSV: x,y,lat,lon per row (frame pixel, then map position); four or more.")
    f.add_argument("--frame", help="A frame of the camera, to refine against the map.")
    f.add_argument("--map")
    f.add_argument("--corners")
    f.add_argument("--model-pack")
    args = ap.parse_args(argv)
    {"drone": _drone, "fixed": _fixed}[args.cmd](args)


if __name__ == "__main__":
    main()
