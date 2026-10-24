# Delivery report: Phase 1, part 1 (missions and ingestion)

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.store`: control database, per-mission SQLite database created on commit (WAL, `synchronous=FULL`), validation, configuration revisions, runs and run items, notifications.
- `trafficwb.ingest`: reconciliation scans with stability rules, source versions, malformed and corrupt handling, missing and rename detection, output and data-folder exclusion, run creation for one-time and continuous missions, background monitor with restart resume.
- `trafficwb-mission` CLI: `create`, `scan`, `status`.
- Rules: `docs/design/ingestion.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| 88 unit tests (`pytest`) | Linux cloud container, Python 3.13.17 | 88 passed |
| Same tests | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| CLI on a copy of the Fintraffic sample (143 JPEGs copied into the container seconds before the first scan) | Linux cloud container | Scan 1: 106 ingested, 37 held back because they had been modified less than 30 s earlier. Scan 2: 1 more. Scan 3 (after 30 s): remaining 36. Scan 4: nothing new. Final: 143 files, 143 versions, 0 malformed, 0 corrupt, 3 runs with 106 + 1 + 36 items, 0 duplicate run items. Capture times, precision and SHA-256 match the sample's `images.csv` for 143/143. |

## Not tested

- Nothing on Windows itself beyond the CI unit tests: no NTFS network shares, no antivirus interference, no real copy tools (Explorer, robocopy).
- `watchdog` notifications are not exercised by tests (polling path only); `watchdog` is not yet a declared dependency.
- No video ingestion on real files.
- No performance test on large archives.

## Known limitations

- Stability interval default (30 s) is a proposal.
- The truncation check covers JPEG only. PNG, BMP and TIFF completeness rely on the modified-time and size rules.
- No web UI yet; missions are created through the CLI.
