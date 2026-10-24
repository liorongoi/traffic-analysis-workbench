# Traffic Analysis Workbench

Offline road-traffic still-image and video analysis for a single Windows workstation, reachable from browsers on the local network. It is a new, independent application.

Status: **Phase 0 (foundations)**. Only the pieces below exist. Nothing here detects vehicles yet.

| Piece | What it does |
|---|---|
| `trafficwb.timeparse` | Reads UTC capture time from filenames, keeping minute or second precision |
| `trafficwb.localtime` | Converts UTC to local time with the IANA rules in force at each instant |
| `trafficwb.schedule` | Scanning windows by weekday, across midnight, with defined boundary rules |
| `trafficwb-probe` | Reports CPU, RAM, GPU, ONNX Runtime providers and disk space |
| `trafficwb-inventory` | Read-only report on a sample folder: formats, sizes, EXIF, checksums, name parsing, intervals |

## Run

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev,probe]"
pytest
trafficwb-probe --paths D:\ E:\
trafficwb-inventory path\to\sample-folder --json inventory.json
```

## Documents

- `docs/design/design-package.html`: the ten design deliverables and open questions
- `docs/design/time-and-schedules.md`: timestamp and schedule rules
- `docs/traceability.md`: requirement → code → test status
- `docs/delivery/`: per-phase delivery reports (what was built, what was actually tested)
