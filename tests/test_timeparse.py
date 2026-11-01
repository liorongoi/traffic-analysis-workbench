from datetime import datetime, timedelta, timezone

import pytest

from trafficwb.timeparse import MalformedName, Precision, parse_capture_time

UTC = timezone.utc


def test_spec_example_minute_precision():
    ct = parse_capture_time("17052026-1604-dcincincifbvhubuhbv.jpg")
    assert ct.utc == datetime(2026, 5, 17, 16, 4, tzinfo=UTC)
    assert ct.precision is Precision.MINUTE
    assert ct.fields == {"suffix": "dcincincifbvhubuhbv"}
    assert ct.interval == (ct.utc, ct.utc + timedelta(minutes=1))
    assert ct.filename == "17052026-1604-dcincincifbvhubuhbv.jpg"


def test_spec_suffix_kept_verbatim_including_dashes_and_leading_zeros():
    ct = parse_capture_time("01012026-0000-007-A-b.jpg")
    assert ct.fields["suffix"] == "007-A-b"
    assert ct.utc == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def test_fintraffic_sample_name_second_precision():
    ct = parse_capture_time("images/C0450701_20261021T071528Z.jpg")
    assert ct.utc == datetime(2026, 10, 21, 7, 15, 28, tzinfo=UTC)
    assert ct.precision is Precision.SECOND
    assert ct.fields == {"camera": "C0450701"}
    assert ct.filename == "C0450701_20261021T071528Z.jpg"


def test_windows_path_is_reduced_to_name():
    ct = parse_capture_time(r"D:\Feeds\R6N\17052026-1604-x.jpg")
    assert ct.filename == "17052026-1604-x.jpg"


@pytest.mark.parametrize(
    "name",
    [
        "31022026-1200-x.jpg",  # 31 February
        "17132026-1200-x.jpg",  # month 13
        "17052026-2400-x.jpg",  # hour 24
        "17052026-1260-x.jpg",  # minute 60
        "C0450701_20260229T000000Z.jpg",  # 2026 is not a leap year
    ],
)
def test_impossible_values_rejected_not_coerced(name):
    with pytest.raises(MalformedName) as e:
        parse_capture_time(name)
    assert "invalid" in e.value.reason


@pytest.mark.parametrize(
    "name",
    ["IMG_0001.jpg", "1705202-1604-x.jpg", "17052026_1604_x.jpg", "17052026-1604.jpg", "C0450701_20261021T071528.jpg", ""],
)
def test_non_matching_names_are_malformed(name):
    with pytest.raises(MalformedName) as e:
        parse_capture_time(name)
    assert "no configured filename pattern" in e.value.reason


def test_leap_day_accepted_in_leap_year():
    assert parse_capture_time("29022028-0101-x.png").utc == datetime(2028, 2, 29, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("camera", ["home", "Gate-3", "north_road_cam"])
def test_underscore_iso_camera_id_any_camera_name(camera):
    ct = parse_capture_time(f"2026_08_26T00_00_18_{camera}_578695735485d7f4.jpg")
    assert ct.utc == datetime(2026, 8, 26, 0, 0, 18, tzinfo=UTC)
    assert ct.precision is Precision.SECOND
    assert ct.pattern == "underscore-iso-camera-id"
    assert ct.fields == {"camera": camera, "suffix": "578695735485d7f4"}


@pytest.mark.parametrize(
    "name", ["2026_02_29T00_00_18_home_abc.jpg", "2026_08_26T24_00_00_home_abc.jpg"]
)
def test_underscore_iso_camera_id_impossible_values_rejected(name):
    with pytest.raises(MalformedName) as e:
        parse_capture_time(name)
    assert "invalid" in e.value.reason


@pytest.mark.parametrize("name", ["2026_08_26T00_00_18.jpg", "2026_08_26T00_00_18_home.jpg", "2026_08_26_00_00_18_home_x.jpg"])
def test_underscore_iso_camera_id_needs_camera_and_id(name):
    with pytest.raises(MalformedName):
        parse_capture_time(name)
