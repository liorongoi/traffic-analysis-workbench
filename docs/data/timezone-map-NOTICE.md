# Offline timezone map data notice

`ui/src/data/timezones.json` is a simplified visual derivative of the comprehensive timezone boundary data from [timezone-boundary-builder](https://github.com/evansiroky/timezone-boundary-builder), release **2026e**, source asset `timezones.geojson.zip`.

- Source asset SHA-256: `a75361c760d6de2fc95c63e97e0b1086d5dfe6fdb2bc6d96efeb3a94673aaa01`
- Source data: OpenStreetMap contributors and timezone-boundary-builder contributors
- Data license: [Open Data Commons Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
- Source project's code license: MIT
- Boundaries are approximate and are provided for selecting an operating location and suggesting a time zone. They are not official legal boundaries.
- The selected IANA zone and map location are stored with the mission. UTC conversion uses IANA tzdata; it does not infer historical offsets from these polygons.

To regenerate the simplified data from a downloaded source archive:

```sh
python scripts/build_timezone_map.py timezones.geojson.zip --out ui/src/data/timezones.json
```
