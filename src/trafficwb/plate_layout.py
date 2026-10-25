"""Plate text: normalisation, country layouts and readability (IMG-04, IMG-05).

Rules:
- The raw reader output is stored exactly as emitted, in Unicode and in the
  order read. Nothing here changes the raw text.
- The normalised text removes separators and spaces, upper-cases Latin
  letters, and writes Persian and Eastern Arabic digits as ASCII digits.
  Persian and Arabic letters are kept as they are, in reading order.
  Leading zeros are kept: plate text is always a string.
- A character the reader is not sure of becomes "?". It is never replaced by
  the most likely character for the layout, so an unreadable plate is never
  completed by guessing.
- Layouts are working hypotheses from the design package (section 10). They
  must be checked against official sources and your samples before a country
  is enabled. A text that fits no layout is kept and marked "no layout".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"     # U+06F0..U+06F9
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"      # U+0660..U+0669
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(PERSIAN_DIGITS)} | {ord(c): str(i) for i, c in enumerate(ARABIC_DIGITS)}
SEPARATORS = re.compile(r"[\s\-_.·•|/\u0640]+")   # includes the Arabic tatweel, as in "هـ"
UNSURE = "?"

READABILITY = ("full", "partial", "unreadable")


def normalize(raw: str) -> str:
    """Canonical text for comparison and search; see the module rules."""
    t = SEPARATORS.sub("", raw or "")
    t = t.translate(_DIGIT_MAP)
    return "".join(c.upper() if "a" <= c <= "z" else c for c in t)


@dataclass(frozen=True)
class Layout:
    name: str
    pattern: str                  # regex over the normalised text
    groups: tuple[int, ...] = ()  # display grouping, e.g. (2, 3, 2) -> 12-345-67
    note: str = ""

    def display(self, text: str) -> str:
        if not self.groups or sum(self.groups) != len(text):
            return text
        out, i = [], 0
        for g in self.groups:
            out.append(text[i:i + g])
            i += g
        return "-".join(out)


@dataclass(frozen=True)
class CountryPlates:
    code: str
    name: str
    script: str                    # what the plates carry
    reader_covers: bool            # whether the installed reader's alphabet can read them
    layouts: tuple[Layout, ...] = field(default_factory=tuple)
    min_len: int = 1
    max_len: int = 10


# Sources: English Wikipedia "Vehicle registration plates of Israel / Latvia / Iran / Lebanon" (read 24 Oct 2026)
# and licenseplatemania.com for Lebanon. Not yet checked against official sources or camera samples.
IR_PRIVATE_LETTERS = "بجدسصطقلمنوهی"
IR_OTHER_LETTERS = "اپتثزژشعفکگ"
LB_AREA_LETTERS = "ABYGNOSTKZJRM"

COUNTRIES: dict[str, CountryPlates] = {
    "IL": CountryPlates(
        "IL", "Israel", "digits", True,
        (Layout("private-8", r"\d{8}", (3, 2, 3), "current private (since 2017), 123-45-678"),
         Layout("private-7", r"\d{7}", (2, 3, 2), "1980–2017 private; taxis and buses still, 12-345-67"),
         Layout("legacy-6", r"\d{6}", (3, 3), "older plates, 123-456"),
         Layout("legacy-5", r"\d{5}", (2, 3), "older plates, 12-345")),
        min_len=5, max_len=8),
    "LV": CountryPlates(
        "LV", "Latvia", "Latin letters and digits", True,
        (Layout("standard", r"[A-Z]{2}\d{1,4}", (), "two letters and one to four digits, AB-1234 (taxis TX/TQ, diplomatic CD)"),
         Layout("trailer", r"[A-Z]\d{4}|[A-Z]\d{3}[A-Z]", (), "trailers: A-1234 or A-123B"),
         Layout("personalised", r"(?!.*(.)\1\1\1)[A-Z0-9]{2,8}", (), "personalised: 2 to 8 letters or digits, no symbol more than three times in a row")),
        min_len=2, max_len=8),
    "IR": CountryPlates(
        "IR", "Iran", "Persian digits and letters", False,
        (Layout("private", rf"\d{{2}}[{IR_PRIVATE_LETTERS}]\d{{3}}\d{{2}}", (), "two digits, letter, three digits, two-digit province code"),
         Layout("other", rf"\d{{2}}[{IR_OTHER_LETTERS}]\d{{3}}\d{{2}}", (), "same order with a government, police, taxi or other service letter")),
        min_len=8, max_len=8),
    "LB": CountryPlates(
        "LB", "Lebanon", "Latin area letter and digits (Arabic name on the blue band)", True,
        (Layout("private", rf"[{LB_AREA_LETTERS}]\d{{1,6}}", (), "area letter and up to six digits, B 123456"),
         Layout("rl", r"RL\d{6}", (), "RL and six digits"),
         Layout("parliament", r"(?:MP)?(?:12[0-8]|1[01]\d|\d{1,2})", (), "members of parliament, 1 to 128")),
        min_len=1, max_len=8),
}


def layout_of(text: str, country: str) -> Layout | None:
    cp = COUNTRIES.get(country)
    if cp is None or not text or UNSURE in text:
        return None
    for lay in cp.layouts:
        if re.fullmatch(lay.pattern, text):
            return lay
    return None


@dataclass
class Reading:
    raw_text: str                       # reader output, pad characters removed, nothing else changed
    chars: list[tuple[str, float]]      # every slot the reader emitted, with its confidence
    normalized: str                     # with "?" for unsure characters
    display: str
    layout: str | None
    readability: str                    # full | partial | unreadable
    reasons: list[str]


@dataclass
class ReadRules:
    char_min: float = 0.60              # a character below this confidence becomes "?"
    unreadable_mean: float = 0.40       # mean confidence of the read characters below this: unreadable
    min_plate_height_px: int = 12       # smaller plates are not read at all
    min_aspect: float = 1.8             # width/height; squarer boxes are two-line plates or signs: never "full"
    finder_sure: float = 0.50           # below this plate-finder score a read is never "full"

    def record(self) -> dict:
        return dict(char_min=self.char_min, unreadable_mean=self.unreadable_mean,
                    min_plate_height_px=self.min_plate_height_px, min_aspect=self.min_aspect,
                    finder_sure=self.finder_sure)


def assess(slots: list[tuple[str, float]], pad: str, country: str, rules: ReadRules | None = None) -> Reading:
    """Turn per-slot reader output into a reading. Never fills an unsure character."""
    rules = rules or ReadRules()
    cp = COUNTRIES.get(country)
    # The text ends at the first pad slot the reader is sure of; an unsure pad inside the text is an unsure character.
    chars: list[tuple[str, float]] = []
    for c, p in slots:
        if c == pad and p >= rules.char_min:
            break
        chars.append((c, p))
    raw = "".join(c for c, _ in chars if c != pad)
    shown = "".join((UNSURE if (p < rules.char_min or c == pad) else c) for c, p in chars)
    norm = normalize(shown.replace(UNSURE, "\x00")).replace("\x00", UNSURE)
    reasons: list[str] = []
    confs = [p for _, p in chars]
    if not chars or (sum(confs) / len(confs)) < rules.unreadable_mean or norm.count(UNSURE) == len(norm):
        return Reading(raw, [(c, round(float(p), 4)) for c, p in slots], "", "", None, "unreadable",
                       ["the reader is not confident of any character"])
    lay = layout_of(norm, country)
    readability = "full"
    if UNSURE in norm:
        readability = "partial"
        reasons.append(f"{norm.count(UNSURE)} character(s) not sure")
    if cp is not None and not (cp.min_len <= len(norm) <= cp.max_len):
        readability = "partial"
        reasons.append(f"length {len(norm)} is outside {cp.min_len}–{cp.max_len} for {cp.name}")
    if lay is None and readability == "full":
        reasons.append(f"fits no known {cp.name if cp else country} layout (foreign or special plate?)")
    return Reading(raw, [(c, round(float(p), 4)) for c, p in slots], norm, lay.display(norm) if lay else norm,
                   lay.name if lay else None, readability, reasons)


def cer(truth: str, pred: str) -> float:
    """Character error rate: edit distance over the length of the truth. "?" counts as an error."""
    t, p = normalize(truth), pred
    if not t:
        return 0.0 if not p else 1.0
    prev = list(range(len(p) + 1))
    for i, a in enumerate(t, 1):
        cur = [i]
        for j, b in enumerate(p, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(t)
