"""UTC to local wall-clock conversion using IANA zone rules per timestamp."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class LocalTime:
    utc: datetime
    zone: str
    local: datetime  # aware, in the zone
    utc_offset: timedelta
    is_dst: bool
    abbreviation: str


def to_local(utc: datetime, zone: str) -> LocalTime:
    """Convert an aware UTC datetime to local time with the rules in force at that instant."""
    if utc.tzinfo is None or utc.utcoffset() != timedelta(0):
        raise ValueError("utc must be a timezone-aware UTC datetime")
    tz = ZoneInfo(zone)
    local = utc.astimezone(tz)
    dst = local.dst()
    return LocalTime(
        utc=utc.astimezone(timezone.utc),
        zone=zone,
        local=local,
        utc_offset=local.utcoffset() or timedelta(0),
        is_dst=bool(dst),
        abbreviation=local.tzname() or "",
    )
