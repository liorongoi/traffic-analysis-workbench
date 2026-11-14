import pytest

from trafficwb.video_georeg import PlatformError, clean_platform, sample_at, track_sample


class Tel:
    def at(self, t):
        return {"lat": 32.0, "lon": 35.0, "rel_alt": 50.0, "heading": 10.0,
                "corner1_lat": 1.0, "corner1_lon": 2.0}


def test_clean_platform_rules():
    assert clean_platform({"lat": "32.1", "lon": "35", "agl": 90}) == {"lat": 32.1, "lon": 35.0, "agl": 90.0}
    with pytest.raises(PlatformError):
        clean_platform({"lat": 32})
    with pytest.raises(PlatformError):
        clean_platform({"alt_asl": 100, "agl": 90})
    with pytest.raises(PlatformError):
        clean_platform({"pitch": 10})
    box = clean_platform({"search_box": [32.0, 35.1, 32.1, 35.0]})["search_box"]
    assert box == [32.1, 35.0, 32.0, 35.1]


def test_typed_values_win_over_flight_data():
    s, ground = sample_at(None, Tel(), {"heading": 90.0, "agl": 120.0}, 1.0)
    assert s["heading"] == 90.0 and s["rel_alt"] == 120.0
    assert "corner1_lat" not in s and s["pitch"] == -90.0 and ground is None
    s, _ = sample_at(None, Tel(), {"alt_asl": 300.0}, 1.0)
    assert s["alt"] == 300.0 and "rel_alt" not in s
    assert sample_at(None, None, {"heading": 5.0}, 0.0) == (None, None)


def test_track_sample_nearest_within_one_step():
    tr = {"fps": 5.0, "samples": [{"t": 0.0}, {"t": 0.2}, {"t": 0.4}]}
    assert track_sample(tr, 0.21)["t"] == 0.2
    assert track_sample(tr, 2.0) is None
