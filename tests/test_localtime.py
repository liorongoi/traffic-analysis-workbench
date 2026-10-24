from datetime import datetime, timedelta, timezone

import pytest

from trafficwb.localtime import to_local

UTC = timezone.utc
H = timedelta(hours=1)


@pytest.mark.parametrize(
    "zone, utc, offset_h, dst",
    [
        # Israel: DST from Friday before last Sunday of March to last Sunday of October.
        ("Asia/Jerusalem", datetime(2026, 1, 17, 16, 4, tzinfo=UTC), 2, False),
        ("Asia/Jerusalem", datetime(2026, 5, 17, 16, 4, tzinfo=UTC), 3, True),
        ("Asia/Jerusalem", datetime(2026, 3, 26, 23, 59, tzinfo=UTC), 2, False),
        ("Asia/Jerusalem", datetime(2026, 3, 27, 0, 0, tzinfo=UTC), 3, True),
        ("Asia/Jerusalem", datetime(2026, 10, 24, 22, 59, tzinfo=UTC), 3, True),
        ("Asia/Jerusalem", datetime(2026, 10, 24, 23, 0, tzinfo=UTC), 2, False),
        # Latvia and Finland: EU rule, 01:00 UTC on last Sundays of March and October.
        ("Europe/Riga", datetime(2026, 3, 29, 0, 59, tzinfo=UTC), 2, False),
        ("Europe/Riga", datetime(2026, 3, 29, 1, 0, tzinfo=UTC), 3, True),
        ("Europe/Riga", datetime(2026, 10, 25, 0, 59, tzinfo=UTC), 3, True),
        ("Europe/Riga", datetime(2026, 10, 25, 1, 0, tzinfo=UTC), 2, False),
        ("Europe/Helsinki", datetime(2026, 10, 21, 7, 15, 28, tzinfo=UTC), 3, True),
        # Lebanon: winter +2, summer +3.
        ("Asia/Beirut", datetime(2026, 1, 15, 12, 0, tzinfo=UTC), 2, False),
        ("Asia/Beirut", datetime(2026, 7, 15, 12, 0, tzinfo=UTC), 3, True),
    ],
)
def test_offsets_follow_rules_at_each_instant(zone, utc, offset_h, dst):
    lt = to_local(utc, zone)
    assert lt.utc_offset == offset_h * H
    assert lt.is_dst is dst
    assert lt.local.replace(tzinfo=None) == (utc + offset_h * H).replace(tzinfo=None)


@pytest.mark.parametrize("month", [1, 4, 7, 10])
def test_iran_has_fixed_offset_in_2026(month):
    # Iran stopped observing DST in 2022 (tzdata 2022b).
    lt = to_local(datetime(2026, month, 10, 12, 0, tzinfo=UTC), "Asia/Tehran")
    assert lt.utc_offset == timedelta(hours=3, minutes=30)
    assert lt.is_dst is False


def test_historical_archive_uses_rules_of_capture_date_not_today():
    # Same wall-clock UTC time in winter and summer gives different local times.
    w = to_local(datetime(2026, 1, 17, 16, 4, tzinfo=UTC), "Asia/Jerusalem").local
    s = to_local(datetime(2026, 5, 17, 16, 4, tzinfo=UTC), "Asia/Jerusalem").local
    assert (w.hour, w.minute) == (18, 4)
    assert (s.hour, s.minute) == (19, 4)


def test_rejects_naive_input():
    with pytest.raises(ValueError):
        to_local(datetime(2026, 1, 1), "Asia/Jerusalem")
