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


# ---------- operator-defined formats ----------

from trafficwb.timeparse import FormatError, compile_template, set_custom_patterns, suggest_template  # noqa: E402


def test_template_reads_name_with_any_camera_and_keeps_parts():
    p = compile_template("Gate", "{camera}-{YYYY}{MM}{DD}-{hh}{mm}{ss}")
    ct = parse_capture_time("North_Gate-20261021-071528.jpg", (p,))
    assert ct.utc == datetime(2026, 10, 21, 7, 15, 28, tzinfo=UTC)
    assert ct.precision is Precision.SECOND and ct.pattern == "Gate"
    assert ct.fields == {"camera": "North_Gate"}


def test_template_without_seconds_is_minute_precision_and_two_digit_year():
    p = compile_template("Short", "{DD}.{MM}.{YY} {hh}h{mm}")
    ct = parse_capture_time("21.10.26 07h15.png", (p,))
    assert ct.utc == datetime(2026, 10, 21, 7, 15, tzinfo=UTC) and ct.precision is Precision.MINUTE


def test_template_in_local_time_is_converted_with_the_rules_of_that_date():
    p = compile_template("IL", "{YYYY}{MM}{DD}_{hh}{mm}{ss}", zone="Asia/Jerusalem")
    assert parse_capture_time("20260826_031500.jpg", (p,)).utc == datetime(2026, 8, 26, 0, 15, tzinfo=UTC)   # UTC+3
    assert parse_capture_time("20260115_031500.jpg", (p,)).utc == datetime(2026, 1, 15, 1, 15, tzinfo=UTC)   # UTC+2


def test_template_local_time_that_does_not_exist_is_malformed():
    p = compile_template("EU", "{YYYY}{MM}{DD}_{hh}{mm}", zone="Europe/Riga")
    with pytest.raises(MalformedName) as e:
        parse_capture_time("20260329_0330.jpg", (p,))   # clocks go from 03:00 to 04:00
    assert "does not exist" in e.value.reason


@pytest.mark.parametrize("tpl,why", [
    ("{YYYY}{MM}{DD}", "needs"), ("{YYYY}{MM}{DD}{hh}{mm}{foo}", "not a known part"),
    ("{YYYY}{YY}{MM}{DD}{hh}{mm}", "not both"), ("{YYYY}{MM}{DD}{hh}{mm}_{camera}{id}", "something between"),
    ("{YYYY}{MM}{DD}{hh}{mm}{mm}", "more than once"), ("{YYYY}{MM}{DD}{hh}{mm}}", "Braces"), ("", "empty"),
])
def test_bad_templates_say_why(tpl, why):
    with pytest.raises(FormatError) as e:
        compile_template("x", tpl)
    assert why in str(e.value)


def test_bad_zone_rejected():
    with pytest.raises(FormatError):
        compile_template("x", "{YYYY}{MM}{DD}{hh}{mm}", zone="Mars/Base")


def test_custom_patterns_are_used_by_default_parse():
    p = compile_template("Mine", "shot {DD}-{MM}-{YYYY} {hh}-{mm}")
    with pytest.raises(MalformedName):
        parse_capture_time("shot 21-10-2026 07-15.jpg")
    set_custom_patterns([p])
    try:
        assert parse_capture_time("shot 21-10-2026 07-15.jpg").pattern == "Mine"
    finally:
        set_custom_patterns([])


@pytest.mark.parametrize("name,expected", [
    ("2026_08_26T00_00_18_home_578695735485d7f4.jpg", "{YYYY}_{MM}_{DD}T{hh}_{mm}_{ss}_{camera}_{id}"),
    ("CAM3-20261021-071528.jpg", "{camera}-{YYYY}{MM}{DD}-{hh}{mm}{ss}"),
    ("IMG_20261021_071528.jpg", "IMG_{YYYY}{MM}{DD}_{hh}{mm}{ss}"),
    ("17052026-1604-xyz.jpg", "{DD}{MM}{YYYY}-{hh}{mm}-{id}"),
    ("photo.jpg", ""),
])
def test_suggested_templates_read_their_own_example(name, expected):
    tpl = suggest_template(name)
    assert tpl == expected
    if tpl:
        parse_capture_time(name, (compile_template("s", tpl),))


def test_operator_format_wins_over_a_builtin_that_also_matches():
    p = compile_template("Local", "{YYYY}_{MM}_{DD}T{hh}_{mm}_{ss}_{camera}_{id}", zone="Asia/Jerusalem")
    name = "2026_08_26T03_00_18_home_578695735485d7f4.jpg"
    assert parse_capture_time(name).utc == datetime(2026, 8, 26, 3, 0, 18, tzinfo=UTC)   # built-in, UTC
    set_custom_patterns([p])
    try:
        assert parse_capture_time(name).utc == datetime(2026, 8, 26, 0, 0, 18, tzinfo=UTC)
    finally:
        set_custom_patterns([])
