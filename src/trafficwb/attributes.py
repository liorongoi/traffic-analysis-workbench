"""Vehicle colour from the image, with an explicit "not observable" outcome (IMG-03, Deliverable 07).

Rules:
- Colour is read from the central part of the box (roof, bonnet and doors seen
  from a raised camera), pixel by pixel in Lab, and summarised as shares.
- The road around the vehicle is the reference: its median tint is removed
  from the vehicle's pixels first, and black and white are judged against its
  lightness. Asphalt is close to neutral and lit
  like the vehicle, so a blue morning cast or orange street light on grey paint
  is not read as colour. Whether a pixel has colour at all is decided by its
  corrected Lab chroma.
- Lit red and orange lamps (very bright, saturated) and their glow are not
  counted as paint. White headlights are not masked: they look like sunlit white paint.
- Three outcomes:
  - observed: one colour clearly dominates;
  - uncertain: two or more colours compete, or the light is poor (dusk, night
    with a colour camera); the competing colours are kept;
  - not_observable: infrared/monochrome image, box too small, too dark, burned
    out or mostly lamps. No colour is reported at all.
- Silver is reported as grey; the reader does not separate them. Dark glass
  can make a light vehicle look partly black, so "black" needs a larger share.
- The thresholds are starting values checked by eye on the Fintraffic sample,
  not measured accuracy. Each result records the algorithm version.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

COLOUR_ALGO = "colour-lab-7"

MIN_SIDE_PX = 12
MIN_AREA_PX = 300
DARK_LUMA = 35.0           # road around the vehicle this dark (Lab L, 0-255): not observable
BURNED_PCT = 40.0          # share of clipped pixels: not observable
LAMP_MAX = 0.6             # share of lamp pixels above which colour is not observable
BLACK_REL = 0.55           # lightness below this share of the road's: black
WHITE_REL = 1.35           # lightness above this share of the road's: white
ACHROMATIC_CHROMA = 12.0   # corrected Lab chroma below this: black, grey or white
OBSERVED_SHARE = 0.40
OBSERVED_MARGIN = 0.15
BLACK_SHARE = 0.60         # "black" needs more, because windows and shadows are dark too
CANDIDATE_SHARE = 0.20
COLOUR_NAMES = ("black", "white", "grey", "brown", "red", "orange", "yellow", "green", "blue")


@dataclass
class ColourReading:
    state: str                              # observed | uncertain | not_observable
    values: list[str] = field(default_factory=list)   # one value when observed; competitors when uncertain
    shares: dict[str, float] = field(default_factory=dict)
    reason: str | None = None
    algo: str = COLOUR_ALGO

    def as_dict(self) -> dict:
        return {"state": self.state, "values": self.values, "shares": self.shares, "reason": self.reason, "algo": self.algo}

    @classmethod
    def from_dict(cls, d: dict) -> "ColourReading":
        return cls(d["state"], list(d.get("values") or []), dict(d.get("shares") or {}), d.get("reason"),
                   d.get("algo", COLOUR_ALGO))


def _clip_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    H, W = img.shape[:2]
    a, b = max(0, int(round(x1))), max(0, int(round(y1)))
    return a, b, min(W, max(a + 1, int(round(x2)))), min(H, max(b + 1, int(round(y2))))


def body_crop(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a, b, c, d = _clip_box(img, x1 + 0.18 * w, y1 + 0.30 * h, x2 - 0.18 * w, y2 - 0.18 * h)
    return img[b:d, a:c]


def road_reference(img: np.ndarray, box) -> tuple[float, float, float] | None:
    """Median Lab (L, a, b) of a ring around the box: mostly road, lit like the vehicle."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a, b, c, d = _clip_box(img, x1 - 0.4 * w, y1 - 0.3 * h, x2 + 0.4 * w, y2 + 0.3 * h)
    region = img[b:d, a:c]
    mask = np.ones(region.shape[:2], bool)
    ia, ib, ic, id_ = _clip_box(img, x1, y1, x2, y2)
    mask[max(0, ib - b):max(0, id_ - b), max(0, ia - a):max(0, ic - a)] = False
    if mask.sum() < 50:
        return None
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)[mask].astype(np.float32)
    lab = lab[(lab[:, 0] > 30) & (lab[:, 0] < 230)]       # ignore lamps and deep shadow
    if len(lab) < 50:
        return None
    return float(np.median(lab[:, 0])), float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def pixel_shares(crop_bgr: np.ndarray, ref: tuple[float, float, float] | None = None) -> dict[str, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    # Lamp cores are much brighter than paint; bright red paint stays below about 220.
    # White headlight cores are not masked: they cannot be told from sunlit white paint.
    core = (vv >= 225) & (ss >= 100) & ((hh <= 20) | (hh >= 165))
    lamp = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[~lamp]
    rl, ra, rb = ref if ref is not None else (120.0, 128.0, 128.0)
    rl = min(max(rl, 60.0), 180.0)
    L, A, B = lab[:, 0], lab[:, 1] - ra, lab[:, 2] - rb
    n = max(1, len(L))
    chroma = np.hypot(A, B)
    hue = (np.degrees(np.arctan2(B, A)) + 360) % 360
    black = (L < BLACK_REL * rl) & (chroma < 2 * ACHROMATIC_CHROMA)
    achro = ~black & (chroma < ACHROMATIC_CHROMA)
    white = achro & (L >= min(235.0, WHITE_REL * rl))
    grey = achro & ~white
    col = ~black & ~achro
    brown = col & (hue >= 30) & (hue < 80) & (L < 110) & (chroma < 40)
    col2 = col & ~brown
    red = col2 & ((hue < 45) | (hue >= 345))
    orange = col2 & (hue >= 45) & (hue < 70)
    yellow = col2 & (hue >= 70) & (hue < 105)
    green = col2 & (hue >= 105) & (hue < 200)
    blue = col2 & (hue >= 200) & (hue < 300)
    other = col2 & (hue >= 300) & (hue < 345)
    out = {"black": black, "white": white, "grey": grey, "brown": brown, "red": red, "orange": orange,
           "yellow": yellow, "green": green, "blue": blue, "other": other}
    shares = {k: round(float(m.sum()) / n, 3) for k, m in out.items()}
    shares["lamp_pixels"] = round(float(lamp.mean()), 3)
    return shares


def read_colour(img_bgr: np.ndarray, box, monochrome: bool, condition: str = "day") -> ColourReading:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if monochrome:
        return ColourReading("not_observable", reason="infrared or monochrome image")
    if min(w, h) < MIN_SIDE_PX or w * h < MIN_AREA_PX:
        return ColourReading("not_observable", reason="vehicle too small in the image")
    crop = body_crop(img_bgr, box)
    if crop.size == 0:
        return ColourReading("not_observable", reason="vehicle outside the image")
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ref = road_reference(img_bgr, box)
    # Darkness is judged on the surroundings: a black car on a lit road is not "too dark".
    if (ref[0] if ref is not None else float(g.mean())) < DARK_LUMA:
        return ColourReading("not_observable", reason="too dark")
    if float((g >= 250).mean() * 100) > BURNED_PCT:
        return ColourReading("not_observable", reason="burned out by light")
    shares = pixel_shares(crop, ref)
    if shares["lamp_pixels"] > LAMP_MAX:
        return ColourReading("not_observable", shares=shares, reason="mostly lit lamps")
    ranked = sorted(((k, shares[k]) for k in COLOUR_NAMES), key=lambda kv: kv[1], reverse=True)
    (top, ts), (_, ss) = ranked[0], ranked[1]
    competitors = [k for k, v in ranked if v >= CANDIDATE_SHARE] or [top]
    if top not in competitors:
        competitors.insert(0, top)
    if condition != "day":
        return ColourReading("uncertain", competitors, shares, reason=f"poor light ({condition})")
    need = BLACK_SHARE if top == "black" else OBSERVED_SHARE
    if ts >= need and ts - ss >= OBSERVED_MARGIN:
        return ColourReading("observed", [top], shares)
    return ColourReading("uncertain", competitors, shares, reason="no single colour dominates")
