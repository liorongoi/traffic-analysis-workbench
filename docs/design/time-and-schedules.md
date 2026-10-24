# Capture time, local time and schedules

Implements requirements TIME-01, TIME-02, TIME-05, IMG-02 and DER-02 from the design package.

## Filename patterns

Capture time comes only from the filename. Filesystem times are never used as a fallback.

| Pattern name | Example | Precision | Kept verbatim |
|---|---|---|---|
| `ddmmyyyy-hhmm-suffix` | `17052026-1604-dcincincifbvhubuhbv.jpg` → 2026-05-17 16:04 UTC | minute | suffix |
| `preset-iso-basic-utc` | `C0450701_20261021T071528Z.jpg` → 2026-10-21 07:15:28 UTC | second | camera id |

The first comes from the specification. The second was observed in the Fintraffic sample (camera C04507, preset C0450701).

- A name that matches a pattern but holds an impossible date or time (31 February, hour 24) is rejected as malformed. It is never coerced.
- A name that matches no pattern is malformed and is listed, not processed.
- A capture stated to the minute covers the half-open interval [hh:mm:00, hh:mm+1:00).

## Local time

Local time is computed from the stored UTC and an IANA zone id, using the rules in force at that instant (`zoneinfo` with the bundled `tzdata` package). It is never stored as the source of truth.

## Schedules

- A window starts on selected weekdays at a local start time and ends at a local end time. If end ≤ start it crosses midnight and ends on the next day. Example: `sun 17:00–05:00` means Sunday 17:00 through Monday 05:00.
- Windows are half-open: start inclusive, end exclusive. Edges are whole minutes.
- A capture is inside only if its whole stated interval is inside. With minute precision, `04:59` is inside a window ending at `05:00`; `05:00` is not.
- Comparison uses local wall-clock time. In the autumn repeated hour both occurrences match. Edges falling in the spring-forward gap have no captures at those wall times.
- An optional date range applies to the date on which a window occurrence starts. A `date_from` of Monday excludes the Monday-morning part of a window that started on Sunday.
- Window length is measured in wall-clock time, so a window spanning a DST change is one hour shorter or longer in real time.
- A mission with no schedules applies no time filter.
