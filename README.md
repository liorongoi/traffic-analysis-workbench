# Traffic Analysis Workbench

Offline road-traffic still-image and video analysis for a single Windows workstation, reachable from browsers on the local network. It is a new, independent application.

Status: **Phases 0 to 4 done for still images, without screens yet**. Only the pieces below exist. Plates, video and the web UI come later.

| Piece | What it does |
|---|---|
| `trafficwb.timeparse` | Reads UTC capture time from filenames, keeping minute or second precision |
| `trafficwb.localtime` | Converts UTC to local time with the IANA rules in force at each instant |
| `trafficwb.schedule` | Scanning windows by weekday, across midnight, with defined boundary rules |
| `trafficwb-probe` | Reports CPU, RAM, GPU, ONNX Runtime providers and disk space |
| `trafficwb.store` | Mission registry and one SQLite database per mission |
| `trafficwb.ingest` | Folder scanning with stability checks, file versions, runs, background monitoring |
| `trafficwb.views`, `trafficwb.regions` | Camera views (day and night together), camera-shift detection, include and exclude areas |
| `trafficwb.models`, `trafficwb.detect` | Verified offline model packs; vehicle detection (YOLOX-s, Apache-2.0) with tiling |
| `trafficwb.night`, `trafficwb.night_service` | Day/dusk/night from image content, headlights, street-lamp background, vehicle candidates at night |
| `trafficwb.process` | Resumable processing runs |
| `trafficwb.review`, `trafficwb.export` | Results table, corrections, approval, CSV and Excel export |
| `trafficwb-web` | Local web server and screens: missions, wizard, results, investigation, camera views, regions |
| `trafficwb-mission` | Create a mission, scan, detect, list results, export, show status |
| `trafficwb-inventory` | Read-only report on a sample folder: formats, sizes, EXIF, checksums, name parsing, intervals |

## Run

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev,probe]"
python scripts/fetch_models.py    # staging machine only; copy models\ to the workstation
python scripts/build_ui.py        # staging machine only (needs Node.js 20+); builds the web screens
pytest
trafficwb-probe --paths D:\ E:\
trafficwb-inventory path\to\sample-folder --json inventory.json
trafficwb-web serve --data D:\twb-data   # open http://127.0.0.1:8765, create the first account there
trafficwb-mission create --data D:\twb-data --name "Route 6" --country IL --zone Asia/Jerusalem --source D:\Feeds\R6 --output E:\Exports\R6
trafficwb-mission scan --data D:\twb-data --mission <id printed by create>
trafficwb-mission detect --data D:\twb-data --mission <id>
trafficwb-mission results --data D:\twb-data --mission <id>
trafficwb-mission export --data D:\twb-data --mission <id>
```

## Documents

- `docs/design/design-package.html`: the ten design deliverables and open questions
- `docs/design/time-and-schedules.md`: timestamp and schedule rules
- `docs/design/ingestion.md`: mission and ingestion rules
- `docs/design/web-ui.md`: web server, sign-in and screen rules
- `docs/traceability.md`: requirement → code → test status
- `docs/delivery/`: per-phase delivery reports (what was built, what was actually tested)
