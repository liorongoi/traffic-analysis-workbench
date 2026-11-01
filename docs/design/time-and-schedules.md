# Capture time, local time and schedules

Implements requirements TIME-01, TIME-02, TIME-05, IMG-02 and DER-02 from the design package.

## Filename patterns

Capture time comes only from the filename. Filesystem times are never used as a fallback.

| Pattern name | Example | Precision | Kept verbatim |
|---|---|---|---|
| `ddmmyyyy-hhmm-suffix` | `17052026-1604-dcincincifbvhubuhbv.jpg` → 2026-05-17 16:04 UTC | minute | suffix |
| `preset-iso-basic-utc` | `C0450701_20261021T071528Z.jpg` → 2026-10-21 07:15:28 UTC | second | camera id |
| `underscore-iso-camera-id` | `2026_08_26T00_00_18_home_578695735485d7f4.jpg` → 2026-08-26 00:00:18 UTC | second | camera (any name, here `home`), id |

The first comes from the specification. The second was observed in the Fintraffic sample (camera C04507, preset C0450701). The third carries a camera name (any text) and an id; its names carry no zone marker and are read as UTC like the others.

- Each scan re-reads the names of files recorded as malformed, so a pattern added later picks up files already in the source folder.

## Formats defined by the operator

The **Filename formats** screen (top menu) lets the operator describe how their camera names files. A format is a template: text the names always contain, and parts in braces.

| Part | Meaning |
|---|---|
| `{YYYY}` `{YY}` | year, 4 or 2 digits (`YY` is 20YY) |
| `{MM}` `{DD}` | month, day |
| `{hh}` `{mm}` `{ss}` | hour (00-23), minute, second (optional: without it the time has minute precision) |
| `{camera}` | camera name, any text, kept (it groups the inventory by camera) |
| `{id}` | id or other text, kept |
| `{any}` | any text, ignored |

Example: `{YYYY}_{MM}_{DD}T{hh}_{mm}_{ss}_{camera}_{id}` reads `2026_08_26T00_00_18_gate7_a1b2c3.jpg`. The extension is not part of the format.

- The screen guesses a format from a pasted name, shows the parts and the time it reads for each pasted name, and can check every name in a folder.
- Each format says whether the names hold UTC or local time in an IANA zone. Local time is converted with the rules in force at that instant; a local time that does not exist (clocks moved forward) is malformed, and a repeated hour (clocks moved back) is read as its first occurrence.
- Formats are kept in `<data>/filename_formats.json`, apply to every mission, and are tried before the built-in patterns, so the operator's choice wins when a name matches both. Each save is recorded in the audit log (`filename_formats_saved`, with the formats before and after).
- A format is checked when saved: it needs a year, month, day, hour and minute, each part at most once, known part names, and a separator between two text parts.

- A name that matches a pattern but holds an impossible date or time (31 February, hour 24) is rejected as malformed. It is never coerced.
- A name that matches no pattern is malformed and is listed, not processed.
- A capture stated to the minute covers the half-open interval [hh:mm:00, hh:mm+1:00).

## Local time

Local time is computed from the stored UTC and an IANA zone id, using the rules in force at that instant (`zoneinfo` with the bundled `tzdata` package). It is never stored as the source of truth.

## Choosing a location and time zone

The mission wizard includes a locally bundled, interactive timezone map, a country selector, and search over IANA zone names (for example, `Asia/Tehran`, `Asia/Beirut`, or `Europe/Riga`). Selecting a map area records its approximate latitude/longitude and suggests that area's IANA zone. The zone can be changed directly; that explicit choice is the mission's manual timezone override. The selected zone, map location, and local operator are recorded in the mission's initial configuration revision. Source timestamps remain the original UTC filename timestamps.

The wizard's UTC date/time preview displays the local time, UTC offset, abbreviation, and daylight-saving state for the chosen zone at that particular timestamp. This is a preview; it does not alter timestamps or the stored mission zone.

Map boundaries are approximate, community-maintained time-zone boundaries derived from OpenStreetMap. The offline map data is a simplified derivative of timezone-boundary-builder 2026e `timezones.geojson.zip` (SHA-256 `a75361c760d6de2fc95c63e97e0b1086d5dfe6fdb2bc6d96efeb3a94673aaa01`). It is distributed under ODbL 1.0; attribution and limitations are recorded in `docs/data/timezone-map-NOTICE.md`. Exact capture-time offsets and daylight-saving rules come from the bundled IANA tzdata, not from the map polygons.

If the timestamp embedded in a filename is itself wrong or malformed, LanePulse currently flags it for correction and does not provide a per-file UTC override. That remains an acceptance gap; the mission-level timezone override is available now.

## Schedules

- A window starts on selected weekdays at a local start time and ends at a local end time. If end ≤ start it crosses midnight and ends on the next day. Example: `sun 17:00–05:00` means Sunday 17:00 through Monday 05:00.
- Windows are half-open: start inclusive, end exclusive. Edges are whole minutes.
- A capture is inside only if its whole stated interval is inside. With minute precision, `04:59` is inside a window ending at `05:00`; `05:00` is not.
- Comparison uses local wall-clock time. In the autumn repeated hour both occurrences match. Edges falling in the spring-forward gap have no captures at those wall times.
- An optional date range applies to the date on which a window occurrence starts. A `date_from` of Monday excludes the Monday-morning part of a window that started on Sunday.
- Window length is measured in wall-clock time, so a window spanning a DST change is one hour shorter or longer in real time.
- A mission with no schedules applies no time filter.
