"""Kalman filter for tracked boxes, as in SORT, ByteTrack and BoT-SORT (MIT-licensed methods; this is our own code).

State: box centre x, y, aspect ratio a (width / height), height h, and their velocities, per analysed frame.
Noise scales with the box height, so near and far objects are treated alike. Time is counted in analysed frames
(``frame_s`` seconds each), so a gap of two analysed frames predicts twice as far as one.

The state is plain lists of floats, so a tracker checkpoint stores it exactly and a resumed run is identical.
"""
from __future__ import annotations

import numpy as np

W_POS = 1.0 / 20
W_VEL = 1.0 / 160


def box_to_z(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], float)


def z_to_box(z) -> list[float]:
    cx, cy, a, h = (float(v) for v in z[:4])
    h = max(h, 1e-3)
    w = max(a, 1e-3) * h
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def initiate(box) -> tuple[list[float], list[float]]:
    z = box_to_z(box)
    h = z[3]
    mean = np.r_[z, np.zeros(4)]
    std = np.array([2 * W_POS * h, 2 * W_POS * h, 1e-2, 2 * W_POS * h,
                    10 * W_VEL * h, 10 * W_VEL * h, 1e-5, 10 * W_VEL * h])
    return mean.tolist(), np.diag(std ** 2).ravel().tolist()


def initiate_moving(prev_box, box, dt: float) -> tuple[list[float], list[float]]:
    """A filter started on its second sighting, with the speed of its centre between the two (``dt`` in analysed
    frames). Size and shape start still: a box growing as it enters at the frame edge is not a speed."""
    z0, z1 = box_to_z(prev_box), box_to_z(box)
    mean, cov = initiate(box)
    mean[4:6] = ((z1[:2] - z0[:2]) / max(dt, 1e-6)).tolist()
    return mean, cov


def _F(dt: float) -> np.ndarray:
    f = np.eye(8)
    for i in range(4):
        f[i, 4 + i] = dt
    return f


def predict(mean, cov, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``dt`` in analysed frames. Returns new arrays; the inputs are not changed."""
    m, p = np.asarray(mean, float), np.asarray(cov, float).reshape(8, 8)
    if dt <= 0:
        return m.copy(), p.copy()
    h = m[3]
    q = np.r_[W_POS * h, W_POS * h, 1e-2, W_POS * h, W_VEL * h, W_VEL * h, 1e-5, W_VEL * h] ** 2
    f = _F(dt)
    return f @ m, f @ p @ f.T + np.diag(q) * dt


def update(mean, cov, box) -> tuple[np.ndarray, np.ndarray]:
    m, p = np.asarray(mean, float), np.asarray(cov, float).reshape(8, 8)
    z = box_to_z(box)
    h = m[3]
    r = np.diag(np.r_[W_POS * h, W_POS * h, 1e-1, W_POS * h] ** 2)
    s = p[:4, :4] + r
    k = np.linalg.solve(s, p[:4, :]).T           # P H^T S^-1, with H = [I 0]
    m2 = m + k @ (z - m[:4])
    p2 = p - k @ s @ k.T
    return m2, p2


def warp(mean, cov, a: np.ndarray) -> tuple[list[float], list[float]]:
    """Move a track's state by the camera's motion ``a`` (2x3 affine, pixels): the centre is mapped, the velocity
    rotated and scaled, the height scaled (BoT-SORT's camera motion compensation)."""
    m, p = np.asarray(mean, float).copy(), np.asarray(cov, float).reshape(8, 8)
    r, t = a[:, :2], a[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(r)))) or 1.0
    m[0:2] = r @ m[0:2] + t
    m[4:6] = r @ m[4:6]
    m[3] *= s
    m[7] *= s
    j = np.eye(8)
    j[0:2, 0:2] = r
    j[4:6, 4:6] = r
    j[3, 3] = j[7, 7] = s
    return m.tolist(), (j @ p @ j.T).ravel().tolist()
