"""A target that drives out of the picture ends there: its box is not kept floating on the empty road, and a new
target coming in at that edge a moment later is a new target, not the old one taken up again."""
from trafficwb.tracker import Tracker, TrackerParams

W, H = 640, 360


def down(t: float, x0: float = 100) -> list[float]:
    """A car driving down and out through the bottom edge (the detector clips its box at the edge)."""
    y = 150 + 120 * t
    return [x0, y, x0 + 80, min(y + 80, H)]


def run(tr, frames):
    keys, closed = [], []
    for t, dets in frames:
        obs, c, _ = tr.update(t, dets)
        keys += [(t, o[0].key) for o in obs]
        closed += [(t, x) for x in c]
    return keys, closed


def test_a_car_leaving_at_the_bottom_is_closed_on_the_first_frame_it_is_gone():
    tr = Tracker(W, H, TrackerParams(min_hits=3))
    frames = []
    for i in range(20):
        t = round(i * 0.2, 3)
        b = down(t)
        if b[1] < H - 20:                      # still seen (its top part)
            frames.append((t, [(b, "car", 0.9)]))
        else:
            frames.append((t, []))
    keys, closed = run(tr, frames)
    last_seen = max(t for t, _ in keys)
    gone = [(t, c) for t, c in closed if c.confirmed]
    assert len(gone) == 1
    t_closed, c = gone[0]
    assert c.reason == "left_frame"
    assert round(t_closed - last_seen, 3) == 0.2      # not coasted for the 1.5 s gap


def left(t: float, y0: float, x0: float, v: float) -> list[float]:
    """A car driving along at v px/s (negative: to the left), its box clipped at the frame's edges."""
    x = x0 + v * t
    return [max(x, 0.0), y0, min(x + 80, W), y0 + 50]


def test_a_new_car_entering_where_one_left_is_a_new_target_even_with_jumps():
    """SAM 3.1 with the standard detector turned jumps on for every family: a car that drove out at the left edge
    must not take up the next car coming in at that edge a moment later, in the next lane."""
    tr = Tracker(W, H, TrackerParams(min_hits=3, jump_per_s=3.0))
    frames = []
    for i in range(25):
        t = round(i * 0.2, 3)
        dets = []
        b = left(t, 150, 200, -200)
        if b[2] - b[0] > 20:
            dets.append((b, "car", 0.9))
        if t >= 1.6:                           # another car comes in from the left, in the lane below
            dets.append((left(t - 1.6, 205, -60, 150), "car", 0.9))
        frames.append((t, [d for d in dets if d[0][2] - d[0][0] > 20]))
    keys, closed = run(tr, frames)
    first = {k for t, k in keys if t < 1.2}
    second = {k for t, k in keys if t >= 1.6}
    assert len(first) == 1 and len(second) == 1 and first != second
    assert [c.reason for _, c in closed if c.track.key in first] == ["left_frame"]


def test_a_car_hidden_for_a_moment_mid_frame_keeps_its_id():
    tr = Tracker(W, H, TrackerParams(min_hits=3))
    frames = []
    for i in range(25):
        t = round(i * 0.2, 3)
        x = 40 + 80 * t
        hidden = 1.6 <= t < 2.4
        frames.append((t, [] if hidden else [([x, 140, x + 60, 190], "car", 0.9)]))
    keys, closed = run(tr, frames)
    assert len({k for _, k in keys}) == 1
    assert not [c for _, c in closed if c.confirmed]


def test_a_car_standing_at_the_edge_is_not_leaving():
    tr = Tracker(W, H, TrackerParams(min_hits=3))
    frames = [(round(i * 0.2, 3), [] if i in (6, 7) else [([0, 200, 70, 260], "car", 0.9)]) for i in range(15)]
    keys, closed = run(tr, frames)
    assert len({k for _, k in keys}) == 1 and not closed


def test_no_jump_for_a_family_the_standard_detector_sees_every_frame():
    """jump_skip: a lost car does not jump to another car farther away; a person (seen by SAM 3.1 only) still does."""
    def go(fam_cls, skip):
        tr = Tracker(W, H, TrackerParams(min_hits=2, jump_per_s=3.0, jump_skip=skip))
        frames = [(0.0, [([300, 150, 340, 230], fam_cls, 0.9)]), (1.0, [([302, 151, 342, 231], fam_cls, 0.9)]),
                  (2.0, [([380, 160, 420, 240], fam_cls, 0.9)])]
        keys, _ = run(tr, frames)
        return len({k for _, k in keys})
    assert go("car", ["vehicle"]) == 2
    assert go("person", ["vehicle"]) == 1
    assert go("car", []) == 1
