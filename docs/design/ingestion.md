# Missions and ingestion

Implements MIS-02, MIS-03, MIS-05 (backend part), MIS-06, MIS-07 (backend part), ING-01 to ING-04, ING-06 and DER-04.

## Missions

- `control.db` in the data root lists missions. Each mission has its own `missions/<id>/mission.db`.
- The mission database is built under a temporary name and renamed into place, so a crash during creation never leaves a half-built `mission.db`. It exists before any run.
- Validation before commit: name, input type and mode (mode must suit the input type), IANA zone, export formats, source folder exists and is readable, output folder exists and is writable (checked with a temporary file), free space, and the output and data folders must not sit inside a recursively scanned source folder.
- A country outside Iran, Lebanon, Israel and Latvia is a warning (plate recognition unavailable), and an error only for a plate-recognition mission.
- Input type and mode cannot change after creation. Every other configuration change is a new revision; history is kept.

## Scans

A reconciliation scan is the authority. Packaged `watchdog` filesystem notifications only trigger the next scan sooner, so a lost notification delays a file but never loses it, and a burst of notifications collapses into one scan. Periodic reconciliation remains enabled as the fallback.

A file is ingested when it is stable:

1. Its last-modified time is at least `stability_seconds` old (default 30), and
2. if it was seen before, its size and modified time have not changed since, and
3. for JPEG, the end-of-image marker is present. A JPEG that stays truncated for a whole stability interval is recorded as corrupt.

Rule 3 exists because some copy tools preserve the original modified time while the file is still being written.

On ingest the file is hashed (SHA-256), its name parsed for capture time, and a source version recorded. Then:

| Situation | Result |
|---|---|
| Same path, same size and modified time | Unchanged, not re-hashed |
| Same path, touched but same content | No new version |
| Same path, new content | New version; the old one is kept and marked superseded |
| Same bytes, different name | Separate file and version (not deduplicated) |
| File disappears | Marked missing; versions and anything attached stay |
| Missing file's content reappears elsewhere | New file, linked as a possible rename |
| Name has no valid capture time | Recorded as malformed, not queued, operator notified |
| Source folder unreachable | Scan reports it, notifies, and marks nothing missing |
| Unsupported extension | Ignored and counted in the scan summary |

## Runs

- One-time mission: a single `initial` run once the snapshot is complete (no files still settling). Later scans do nothing.
- Continuous mission: every scan that leaves eligible versions unqueued creates a `continuous_batch` run and an in-app notification.
- A run queues every eligible current version not yet in any run, so files ingested before a crash are picked up by the next cycle.
- New results stay pending review; no approval is implied.

## Monitoring

`Monitor` runs scans in a background thread, independent of any browser. The on/off flag is part of the mission configuration, and `resume_enabled` restarts monitoring for every continuous mission with the flag on after an application restart. A failed scan is reported as a notification and the monitor keeps running.
