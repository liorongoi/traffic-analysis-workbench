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

DEFAULT_PATTERNS: tuple[FilenamePattern, ...] = (SPEC_DDMMYYYY_HHMM, PRESET_ISO_BASIC_UTC)


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
    filename: str, patterns: tuple[FilenamePattern, ...] = DEFAULT_PATTERNS
) -> CaptureTime:
    """Parse the UTC capture time from a filename (path or bare name).

    Raises MalformedName when no pattern matches or the date/time is impossible.
    Impossible values (31 February, hour 24) are rejected, never coerced.
    """
    name = PurePath(filename.replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    for pattern in patterns:
        m = pattern.match(stem)
        if m is None:
            continue
        g = m.groupdict()
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
        extra = {k: v for k, v in g.items() if k in ("suffix", "camera")}
        return CaptureTime(filename=name, utc=utc, precision=pattern.precision, pattern=pattern.name, fields=extra)
    raise MalformedName(name, "no configured filename pattern matches")
