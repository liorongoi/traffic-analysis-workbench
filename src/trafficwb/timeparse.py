"""Capture-time parsing from filenames.

Capture timestamps are UTC and come only from the filename. Filesystem times are
never used as a fallback. Each pattern declares its precision, and the precision
is kept: a name that states minutes does not establish the seconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import PurePath
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class Precision(str, Enum):
    MINUTE = "minute"
    SECOND = "second"

    @property
    def span(self) -> timedelta:
        return timedelta(minutes=1) if self is Precision.MINUTE else timedelta(seconds=1)


@dataclass(frozen=True)
class FilenamePattern:
    """A named filename convention that carries a UTC capture time."""

    name: str
    regex: re.Pattern[str]
    precision: Precision
    description: str

    def match(self, stem: str) -> re.Match[str] | None:
        return self.regex.fullmatch(stem)


# DDMMYYYY-HHmm-<suffix>, from the specification. The suffix is opaque.
SPEC_DDMMYYYY_HHMM = FilenamePattern(
    name="ddmmyyyy-hhmm-suffix",
    regex=re.compile(
        r"(?P<day>\d{2})(?P<month>\d{2})(?P<year>\d{4})-(?P<hour>\d{2})(?P<minute>\d{2})-(?P<suffix>.+)"
    ),
    precision=Precision.MINUTE,
    description="DDMMYYYY-HHmm-<suffix>, UTC, minute precision; suffix kept verbatim",
)

# <camera-preset>_YYYYMMDDTHHMMSSZ, seen in the Fintraffic sample (e.g. C0450701_20261021T071528Z).
PRESET_ISO_BASIC_UTC = FilenamePattern(
    name="preset-iso-basic-utc",
    regex=re.compile(
        r"(?P<camera>[A-Za-z0-9]+)_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})T"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z"
    ),
    precision=Precision.SECOND,
    description="<camera>_YYYYMMDDTHHMMSSZ, UTC, second precision; camera id kept verbatim",
)

# YYYY_MM_DDTHH_MM_SS_<camera>_<id>, e.g. 2026_08_26T00_00_18_home_578695735485d7f4 (camera "home").
# The camera part is any text (it may hold underscores); the id is the last underscore-free part.
# The name has no zone marker; like every pattern here it is read as UTC.
UNDERSCORE_ISO_CAMERA_ID = FilenamePattern(
    name="underscore-iso-camera-id",
    regex=re.compile(
        r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})T(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
        r"_(?P<camera>.+)_(?P<suffix>[^_]+)"
    ),
    precision=Precision.SECOND,
    description="YYYY_MM_DDTHH_MM_SS_<camera>_<id>, UTC, second precision; camera and id kept verbatim",
)

DEFAULT_PATTERNS: tuple[FilenamePattern, ...] = (SPEC_DDMMYYYY_HHMM, PRESET_ISO_BASIC_UTC, UNDERSCORE_ISO_CAMERA_ID)


# ---------- formats defined by the operator ----------
#
# A format is a template such as {YYYY}_{MM}_{DD}T{hh}_{mm}_{ss}_{camera}_{id}. Text outside braces must
# appear as written. The time is in UTC unless the format names an IANA zone, in which case the name holds
# local time there and is converted to UTC with the rules in force at that instant.

TOKENS: dict[str, tuple[str, str]] = {          # token -> (regex, group)
    "YYYY": (r"\d{4}", "year"), "YY": (r"\d{2}", "year2"), "MM": (r"\d{2}", "month"), "DD": (r"\d{2}", "day"),
    "hh": (r"\d{2}", "hour"), "mm": (r"\d{2}", "minute"), "ss": (r"\d{2}", "second"),
    "camera": (r".+?", "camera"), "id": (r".+?", "suffix"), "any": (r".*?", None),
}
TOKEN_LABELS = {"YYYY": "year (4 digits)", "YY": "year (2 digits)", "MM": "month", "DD": "day", "hh": "hour (00-23)",
                "mm": "minute", "ss": "second", "camera": "camera name", "id": "id or other text kept",
                "any": "anything (ignored)"}
_TOKEN_RE = re.compile(r"\{([A-Za-z0-9]+)\}")


class FormatError(ValueError):
    """A template that cannot describe a capture time."""


def compile_template(name: str, template: str, zone: str = "UTC") -> "TemplatePattern":
    """Turn an operator template into a pattern. Raises FormatError with a plain reason."""
    name, template = (name or "").strip(), (template or "").strip()
    if not name:
        raise FormatError("Give the format a name.")
    if not template:
        raise FormatError("The format is empty.")
    parts, seen, pos = [], [], 0
    for m in _TOKEN_RE.finditer(template):
        parts.append(re.escape(template[pos:m.start()]))
        tok = m.group(1)
        if tok not in TOKENS:
            raise FormatError(f"{{{tok}}} is not a known part. Known parts: "
                              + ", ".join("{" + t + "}" for t in TOKENS) + ".")
        if tok in seen and tok != "any":
            raise FormatError(f"{{{tok}}} appears more than once.")
        seen.append(tok)
        rx, grp = TOKENS[tok]
        parts.append(f"(?P<{grp}>{rx})" if grp else rx)
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    if "{" in _TOKEN_RE.sub("", template) or "}" in _TOKEN_RE.sub("", template):
        raise FormatError("Braces must hold a part name, for example {YYYY}.")
    if "YYYY" in seen and "YY" in seen:
        raise FormatError("Use {YYYY} or {YY}, not both.")
    missing = [t for t in ("MM", "DD", "hh", "mm") if t not in seen]
    if not ("YYYY" in seen or "YY" in seen):
        missing.insert(0, "YYYY")
    if missing:
        raise FormatError("A capture time needs " + ", ".join("{" + t + "}" for t in missing) + ".")
    for i in range(len(seen) - 1):
        a, b = seen[i], seen[i + 1]
        if TOKENS[a][0].startswith(".") and TOKENS[b][0].startswith(".") and \
                f"{{{a}}}{{{b}}}" in template.replace(" ", ""):
            raise FormatError(f"{{{a}}} and {{{b}}} need something between them, such as _ or -.")
    if zone != "UTC":
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError):
            raise FormatError(f"{zone} is not a known time zone.") from None
    return TemplatePattern(
        name=name, regex=re.compile("".join(parts)),
        precision=Precision.SECOND if "ss" in seen else Precision.MINUTE,
        description=f"{template} ({'UTC' if zone == 'UTC' else 'local time in ' + zone})",
        template=template, zone=zone)


@dataclass(frozen=True)
class TemplatePattern(FilenamePattern):
    template: str = ""
    zone: str = "UTC"


_custom: tuple[FilenamePattern, ...] = ()


def set_custom_patterns(patterns: list[FilenamePattern] | tuple[FilenamePattern, ...]) -> None:
    """Formats the operator defined; tried before the built-in ones by every parse that names no patterns,
    so an operator's choice (for example local time) wins over a built-in format the same names also match."""
    global _custom
    _custom = tuple(patterns)


def active_patterns() -> tuple[FilenamePattern, ...]:
    return _custom + DEFAULT_PATTERNS


def suggest_template(filename: str) -> str:
    """A first guess at the template for an example name, for the editor to start from."""
    stem = PurePath(filename.replace("\\", "/")).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    guesses = [
        (re.compile(r"(\d{4})([-_.]?)(\d{2})\2(\d{2})([T_ -]?)(\d{2})([-_.:]?)(\d{2})(?:\7(\d{2}))?"), "ymd"),
        (re.compile(r"(\d{2})([-_.]?)(\d{2})\2(\d{4})([T_ -]?)(\d{2})([-_.:]?)(\d{2})(?:\7(\d{2}))?"), "dmy"),
    ]

    def plausible(m, order):
        g = m.groups()
        y, mo, d = (g[0], g[2], g[3]) if order == "ymd" else (g[3], g[2], g[0])
        ok = 1990 <= int(y) <= 2100 and 1 <= int(mo) <= 12 and 1 <= int(d) <= 31
        return ok and int(g[5]) < 24 and int(g[7]) < 60 and (g[8] is None or int(g[8]) < 60)

    m = order = None
    for start in range(len(stem)):
        if start and stem[start - 1].isdigit():
            continue                    # a date starts where a run of digits starts
        for rx, o in guesses:
            cand = rx.match(stem, start)
            if cand and plausible(cand, o):
                m, order = cand, o
                break
        if m:
            break
    if m:
        a, s1, b, c, s2, h, s3, mi, se = m.groups()
        date = (f"{{YYYY}}{s1}{{MM}}{s1}{{DD}}" if order == "ymd" else f"{{DD}}{s1}{{MM}}{s1}{{YYYY}}")
        tpl = date + s2 + f"{{hh}}{s3}{{mm}}" + (f"{s3}{{ss}}" if se else "")
        before, after = stem[:m.start()], stem[m.end():]
        if before:
            sep = re.search(r"[-_ .]+$", before)
            tpl = ("{camera}" + sep.group(0) if sep else "{camera}") + tpl if not re.fullmatch(r"[A-Za-z]+[-_ ]?", before) \
                else before + tpl
        if after:
            seps = list(re.finditer(r"[-_ .]+", after))
            if after[0] in "-_ ." and len(seps) >= 2 and "{camera}" not in tpl:
                lead, last = seps[0], seps[-1]
                tpl += lead.group(0) + "{camera}" + last.group(0) + "{id}" if last.end() < len(after) else after
            elif after[0] in "-_ .":
                tpl += seps[0].group(0) + "{id}"
            else:
                tpl += "{id}"
        return tpl
    return ""


@dataclass(frozen=True)
class CaptureTime:
    filename: str
    utc: datetime  # timezone-aware, UTC
    precision: Precision
    pattern: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def interval(self) -> tuple[datetime, datetime]:
        """Half-open UTC interval [start, end) the stated time covers."""
        return self.utc, self.utc + self.precision.span


class MalformedName(ValueError):
    """The filename does not carry a valid capture time under any configured pattern."""

    def __init__(self, filename: str, reason: str):
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


def parse_capture_time(
    filename: str, patterns: tuple[FilenamePattern, ...] | None = None
) -> CaptureTime:
    """Parse the UTC capture time from a filename (path or bare name).

    Raises MalformedName when no pattern matches or the date/time is impossible.
    Impossible values (31 February, hour 24) are rejected, never coerced.
    """
    if patterns is None:
        patterns = active_patterns()
    name = PurePath(filename.replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    for pattern in patterns:
        m = pattern.match(stem)
        if m is None:
            continue
        g = m.groupdict()
        if g.get("year2"):
            g["year"] = str(2000 + int(g["year2"]))
        zone = getattr(pattern, "zone", "UTC")
        try:
            utc = datetime(
                int(g["year"]),
                int(g["month"]),
                int(g["day"]),
                int(g["hour"]),
                int(g["minute"]),
                int(g.get("second") or 0),
                tzinfo=timezone.utc,
            )
        except ValueError as exc:
            raise MalformedName(name, f"pattern {pattern.name} matched but date/time is invalid ({exc})") from None
        if zone != "UTC":
            local = utc.replace(tzinfo=ZoneInfo(zone))
            back = local.astimezone(timezone.utc).astimezone(ZoneInfo(zone))
            if back.replace(tzinfo=None) != local.replace(tzinfo=None):
                raise MalformedName(name, f"pattern {pattern.name} matched but date/time is invalid "
                                          f"(that local time does not exist in {zone}: the clocks moved forward)")
            utc = local.astimezone(timezone.utc)   # a repeated hour (clocks back) is read as its first occurrence
        extra = {k: v for k, v in g.items() if k in ("suffix", "camera")}
        return CaptureTime(filename=name, utc=utc, precision=pattern.precision, pattern=pattern.name, fields=extra)
    raise MalformedName(name, "no configured filename pattern matches")
