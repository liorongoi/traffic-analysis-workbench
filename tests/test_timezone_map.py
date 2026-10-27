import json
from pathlib import Path


def test_bundled_timezone_map_has_selectable_target_zones():
    path = Path(__file__).parents[1] / "ui" / "src" / "data" / "timezones.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_zone = {feature["zone"]: feature["path"] for feature in data["features"]}
    assert data["width"] == 1000 and data["height"] == 500
    assert {"Asia/Tehran", "Asia/Beirut", "Asia/Jerusalem", "Europe/Riga"} <= by_zone.keys()
    assert len(by_zone) == len(data["features"])
    assert all(path.startswith("M") and path.endswith("Z") for path in by_zone.values())
