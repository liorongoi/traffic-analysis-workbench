# Delivery report: web screens, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.web.app`: local FastAPI server (`trafficwb-web serve`), localhost by default, `--lan` on request.
- The app opens directly without accounts, passwords, sessions, or an application sign-in screen. The server attributes changes to the local operating-system account.
- `trafficwb.web.jobs`: background processing and monitoring with pause, resume, cancel and a measured ETA. Monitoring resumes at server start.
- Screens in `ui/`: missions, new-mission wizard, overview, results, investigation, camera views, region drawing. Camera views support selecting members for reassignment/split; investigation shows append-only assignment history. Rules: `docs/design/web-ui.md`.
- `process.Processor` now supports pause and cancel (`RunControl`). A cancelled run marks its remaining images as cancelled and adds a notification.
- CI builds the screens (type check included) before the tests.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full test suite at the time of the initial screen delivery (147 tests), including API coverage for sign-in, take-over, wizard → process → review → export, pause/resume/cancel, views and regions | Linux cloud container, Python 3.13, with the sample | Historical result. Sign-in and take-over behavior was removed later; see current traceability and CI checks. |
| Same tests and the UI build | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Current full test suite after sign-in removal and view assignment controls | GitHub Actions run 48, Windows + Linux, Python 3.11/3.12 | 290 passed, 2 skipped on all four runners; UI builds and CPU/auto offline installer checks passed |
| Browser run on the Fintraffic sample: results sorted by vehicles, day and night investigation, confirm a night candidate, shortcuts help, camera views, drawing a region, wizard file-name preview | Chromium (Playwright) against a local server | Historical check; it included a sign-in flow that was removed later. |

## Not tested

- Automated UI tests in CI. The browser run is a manual check.
- The server as a Windows service, and use over a LAN.
- The newest camera-view reassignment/split controls have not yet had a manual browser interaction check.
- Plates, video and object search: not built yet.
