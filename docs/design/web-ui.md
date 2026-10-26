# Web screens: rules

## Server

- `trafficwb-web serve --data D:\twb-data` serves on `127.0.0.1:8765`. `--lan` opens it to the local network. Only use it on a closed network.
- Processing and folder monitoring run in server threads, not in the browser. Closing the browser does not stop them. Monitoring that was on resumes when the server starts.
- One background job per mission: processing a source scan or the files that monitoring found. Pause, resume and cancel act between images. The ETA is measured on the running job, and the screen says what it is based on.

## Local access

- LanePulse does not require an account, password, or sign-in. Changes are attributed to the Windows/macOS/Linux account running the server.
- The server listens on localhost by default. `--lan` makes it reachable on the local network without a sign-in, so use that only on a trusted, closed network.
- Every request that changes data must carry the header `X-TWB: 1`, which blocks cross-site form posts.

## Screens

| Screen | Route | What it does |
|---|---|---|
| Missions | `#/` | List, status, notifications |
| New mission | `#/new` | Five steps: name and type, country and zone, source folder with file-name preview (UTC and local time per file), output folder, check and create |
| Overview | `#/m/<id>` | Counts, the current job with progress, pause/resume/cancel, monitoring on/off |
| Results | `#/m/<id>/results` | Filters (name, view, review state, processing state, dates, minimum vehicles), sort, paging, thumbnails, approve all, export |
| Investigation | `#/m/<id>/i/<image>` | Image with boxes, object list, corrections, approval, previous/next within the current filter and sort |
| Camera views | `#/m/<id>/views` | Rename, confirm, merge, representative images, unassigned images |
| Regions | `#/m/<id>/regions/<view>` | Draw include and exclude polygons on a representative image |

## Drawing conventions on the image

- Machine detections: solid line. Operator additions: dashed. Night candidates: dotted. Candidates confirmed as vehicles: dash-dot. The legend is always on the screen, so colour is never the only cue.
- Coordinates are in the image's original pixels. Zoom and window size never change stored values.

## Keyboard (investigation)

`J`/`K` next/previous, `A` approve, `B` draw a missed vehicle, `Del` remove, `C` confirm a candidate, `G` regroup selected lights, `1`–`5` set the class, `H` hide overlays, `Esc` cancel, `?` help.

## Build

The screens are React + TypeScript in `ui/`. They are built with `python scripts/build_ui.py` on the staging machine or in CI, and shipped inside the Python package. The workstation needs no Node.js, and nothing is downloaded at run time (system fonts only).