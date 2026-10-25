# Traffic Analysis Workbench

Offline road-traffic still-image and video analysis for a single Windows workstation, reachable from browsers on the local network. It is a new, independent application.

Status: **still images (phases 0 to 4), web screens, object search, video part 1, plates part 1, learning part 1 and hardening part 1 (offline install, backups, GPU choice) are done**. Only the pieces below exist. Plate reads are evaluation only until a country passes its own test set.

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
| `trafficwb.night`, `trafficwb.night_service` | Day/dusk/night from image content, headlights and glare, learned background lights (always, often or sometimes lit at each place), distance bands, vehicle candidates at night |
| `trafficwb.process` | Resumable processing runs |
| `trafficwb.interpret`, `trafficwb.attributes`, `trafficwb.search` | Typed object search: request to confirmed spec, vehicle colour with "not observable", match/possible results |
| `trafficwb.video`, `trafficwb.tracker`, `trafficwb.video_process`, `trafficwb.video_review` | Video by frame timestamps in resumable chunks, vehicle tracking, one event per appearance, event review, clips |
| `trafficwb.plates`, `trafficwb.plate_layout` | Plate finding and reading on each vehicle, raw and normalised text, unreadable kept unreadable, review, country evaluation gate |
| `trafficwb.learning` | Which reviewed images can teach a model, frozen dataset versions with leak-free splits, COCO export, model evaluation, promotion and rollback (no training yet) |
| `trafficwb.admin`, `trafficwb.runtime` | Backups, restore, explicit mission deletion with an audit log; CUDA, DirectML or CPU with fallback |
| `trafficwb.review`, `trafficwb.export` | Results table, corrections, approval, CSV and Excel export |
| `trafficwb-web` | Local web server and screens: missions, wizard, results, investigation, camera views, regions, search, video events and player, plates, learning |
| `trafficwb-admin` | Back up, verify, restore, delete a mission, show the audit log |
| `trafficwb-mission` | Create a mission, scan, detect, list results, export, show status |
| `trafficwb-inventory` | Read-only report on a sample folder: formats, sizes, EXIF, checksums, name parsing, intervals |

## Install on the workstation

Build an offline bundle on a connected staging machine with `scripts/make_bundle.py` and install it with `install.ps1`; see `docs/design/deployment.md` and `packaging/INSTALL.txt`.

## Run (development)

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev,probe,cpu]"     # or cuda / directml instead of cpu
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
- `docs/design/object-search.md`: object search rules
- `docs/design/web-ui.md`: web server, sign-in and screen rules
- `docs/design/video.md`: video timing, tracking, events, review and clips
- `docs/design/plates.md`: plate reading rules; `docs/decisions/plate-models.md`: which plate models and why
- `docs/design/learning.md`: training data eligibility, dataset versions, evaluation and promotion
- `docs/design/deployment.md`: offline bundle, install, update, rollback, backups and GPU choice
- `docs/traceability.md`: requirement → code → test status
- `docs/delivery/`: per-phase delivery reports (what was built, what was actually tested)
