from datetime import date, datetime, time, timezone

import pytest

from trafficwb.schedule import Schedule, ScheduleSet, Window
from trafficwb.timeparse import parse_capture_time

TZ = "Asia/Jerusalem"  # 17 May 2026 is a Sunday; local = UTC+3 in May


def overnight(**kw):
    return ScheduleSet((Schedule("Sun night", (Window.parse("sun", "17:00", "05:00"),), **kw),))


@pytest.mark.parametrize(
    "name, inside",
    [
        ("17052026-1359-x.jpg", False),  # Sun 16:59 local
        ("17052026-1400-x.jpg", True),   # Sun 17:00 local, start is inclusive
        ("17052026-1604-x.jpg", True),   # spec example, Sun 19:04
        ("17052026-2100-x.jpg", True),   # Mon 00:00, after midnight
        ("18052026-0159-x.jpg", True),   # Mon 04:59, whole minute ends at 05:00
        ("18052026-0200-x.jpg", False),  # Mon 05:00, end is exclusive
        ("18052026-1430-x.jpg", False),  # Mon 17:30, Monday is not a start day
        ("16052026-2000-x.jpg", False),  # Sat 23:00
    ],
)
def test_overnight_window_sunday_to_monday(name, inside):
    assert overnight().contains(parse_capture_time(name), TZ) is inside


def test_second_precision_edges():
    s = overnight()
    assert s.contains(parse_capture_time("C1_20260518T015959Z.jpg"), TZ) is True
    assert s.contains(parse_capture_time("C1_20260518T020000Z.jpg"), TZ) is False


def test_window_edges_must_be_whole_minutes():
    with pytest.raises(ValueError):
        Window(frozenset({6}), time(17, 0, 30), time(18, 0))


def test_multiple_windows_per_day():
    s = ScheduleSet((Schedule("Peaks", (Window.parse("mon,tue,wed,thu,fri", "07:00", "09:00"),
                                        Window.parse("mon,tue,wed,thu,fri", "16:00", "18:30"))),))
    # Mon 18 May 2026
    assert s.contains(parse_capture_time("18052026-0500-x.jpg"), TZ) is True   # 08:00
    assert s.contains(parse_capture_time("18052026-0900-x.jpg"), TZ) is False  # 12:00
    assert s.contains(parse_capture_time("18052026-1529-x.jpg"), TZ) is True   # 18:29
    assert s.contains(parse_capture_time("18052026-1530-x.jpg"), TZ) is False  # 18:30


def test_date_range_applies_to_occurrence_start_date():
    s = overnight(date_from=date(2026, 5, 18))
    # Monday-morning part of the occurrence that started Sunday 17 May is excluded.
    assert s.contains(parse_capture_time("18052026-0100-x.jpg"), TZ) is False
    s2 = overnight(date_to=date(2026, 5, 17))
    assert s2.contains(parse_capture_time("18052026-0100-x.jpg"), TZ) is True


def test_autumn_repeated_hour_matches_both_occurrences():
    # 25 Oct 2026 (Sunday): Israel clocks go 02:00 IDT -> 01:00 IST.
    s = ScheduleSet((Schedule("Repeat", (Window.parse("sun", "01:00", "01:30"),)),))
    first = parse_capture_time("C1_20261024T221000Z.jpg")   # 01:10 IDT
    second = parse_capture_time("C1_20261024T231000Z.jpg")  # 01:10 IST
    assert s.contains(first, TZ) and s.contains(second, TZ)


def test_spring_gap_window_edge():
    # 27 Mar 2026 (Friday): 02:00 IST -> 03:00 IDT, wall times 02:00-02:59 do not exist.
    s = ScheduleSet((Schedule("Gap", (Window.parse("fri", "02:00", "03:30"),)),))
    assert s.contains(parse_capture_time("C1_20260326T235900Z.jpg"), TZ) is False  # 01:59 IST
    assert s.contains(parse_capture_time("C1_20260327T000000Z.jpg"), TZ) is True   # 03:00 IDT
    assert s.contains(parse_capture_time("C1_20260327T003000Z.jpg"), TZ) is False  # 03:30 IDT


def test_empty_schedule_set_means_no_filter():
    assert ScheduleSet().contains(parse_capture_time("17052026-1604-x.jpg"), TZ) is True


def test_equal_start_and_end_rejected():
    with pytest.raises(ValueError):
        Window.parse("sun", "05:00", "05:00")
