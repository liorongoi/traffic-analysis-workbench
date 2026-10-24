# Delivery report: web screens, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.web.app`: local FastAPI server (`trafficwb-web serve`), localhost by default, `--lan` on request.
- `trafficwb.web.auth`: accounts, scrypt passwords, one operator at a time with take-over, 30 min idle expiry, sign-in log.
- `trafficwb.web.jobs`: background processing and monitoring with pause, resume, cancel and a measured ETA. Monitoring resumes at server start.
- Screens in `ui/`: missions, new-mission wizard, overview, results, investigation, camera views, region drawing. Rules: `docs/design/web-ui.md`.
- `process.Processor` now supports pause and cancel (`RunControl`). A cancelled run marks its remaining images as cancelled and adds a notification.
- CI builds the screens (type check included) before the tests.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full test suite (147 tests), including 5 API tests: sign-in rules, take-over, wizard → process → review → export, pause/resume/cancel, views and regions | Linux cloud container, Python 3.13, with the sample | All passed |
| Same tests and the UI build | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Browser run on the Fintraffic sample: sign-in, results sorted by vehicles, day and night investigation, confirm a night candidate, shortcuts help, camera views, drawing a region, wizard file-name preview | Chromium (Playwright) against a local server | Worked, no console errors. Checked by eye from screenshots. |

## Not tested

- Automated UI tests in CI. The browser run is a manual check.
- The server as a Windows service, and use over a LAN.
- Several operators trying to sign in at once on real machines (take-over is covered by API tests only).
- Plates, video and object search: not built yet.
