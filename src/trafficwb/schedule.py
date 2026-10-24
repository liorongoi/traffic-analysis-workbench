"""Scanning schedules evaluated in local wall-clock time.

Rules (see docs/design/time-and-schedules.md):
- A window starts on selected weekdays at a local start time and ends at a local
  end time. If end <= start the window crosses midnight and ends the next day.
- Windows are half-open: [start, end).
- A capture is inside only if its whole stated interval is inside. A capture at
  "05:00" with minute precision covers 05:00:00-05:00:59, so it is outside a
  window ending at 05:00 and inside one starting at 05:00.
- Comparison is in local wall-clock time. During the autumn repeated hour both
  occurrences match; window edges inside a spring-forward gap simply have no
  captures at those wall times.
- Optional date range limits the date on which a window occurrence starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

from .localtime import to_local
from .timeparse import CaptureTime

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class Window:
    days: frozenset[int]  # 0=Monday .. 6=Sunday, the day the window starts
    start: time
    end: time

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("window start and end must differ")
        if not self.days or not all(0 <= d <= 6 for d in self.days):
            raise ValueError("days must be a non-empty set of 0..6")
        if self.start.second or self.end.second or self.start.microsecond or self.end.microsecond:
            raise ValueError("window edges must be whole minutes")

    @property
    def crosses_midnight(self) -> bool:
        return self.end <= self.start

    @classmethod
    def parse(cls, days: str, start: str, end: str) -> "Window":
        """Window.parse('sun', '17:00', '05:00') or Window.parse('mon,tue', '07:00', '09:30')."""
        idx = frozenset(WEEKDAYS.index(d.strip().lower()[:3]) for d in days.split(","))
        return cls(idx, time.fromisoformat(start), time.fromisoformat(end))


@dataclass(frozen=True)
class Schedule:
    name: str
    windows: tuple[Window, ...]
    date_from: date | None = None
    date_to: date | None = None  # inclusive

    def _date_ok(self, d: date) -> bool:
        return (self.date_from is None or d >= self.date_from) and (self.date_to is None or d <= self.date_to)

    def contains_local(self, start: datetime, span: timedelta) -> bool:
        """True if naive local interval [start, start+span) lies inside one window occurrence."""
        end = start + span
        for w in self.windows:
            for d in (start.date() - timedelta(days=1), start.date()):
                if d.weekday() not in w.days or not self._date_ok(d):
                    continue
                occ_start = datetime.combine(d, w.start)
                occ_end = datetime.combine(d + timedelta(days=1) if w.crosses_midnight else d, w.end)
                if occ_start <= start and end <= occ_end:
                    return True
        return False


@dataclass(frozen=True)
class ScheduleSet:
    """Captures are eligible if any schedule contains them. An empty set means no time filter."""

    schedules: tuple[Schedule, ...] = field(default_factory=tuple)

    def contains(self, capture: CaptureTime, zone: str) -> bool:
        if not self.schedules:
            return True
        local = to_local(capture.utc, zone).local.replace(tzinfo=None)
        span = capture.precision.span
        return any(s.contains_local(local, span) for s in self.schedules)
