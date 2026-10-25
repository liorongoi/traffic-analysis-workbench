# Delivery report: Phase 2 (camera views and regions)

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- Database migration v2: camera locations, views, signatures, assignments with history, view events, drift references and results, region revisions. Existing mission databases are backed up before migrating.
- `trafficwb.views`: signatures, grouping proposals, colour/monochrome linking, reassign, rename, merge, split, confirm, history, drift checks.
- `trafficwb.regions`: polygon validation, inclusion/exclusion with precedence, three membership rules, revisioned editing.
- Method and thresholds: `docs/design/camera-views.md`.
- New dependencies: `numpy`, `opencv-python-headless`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full test suite including `tests/test_views_regions.py` | Linux cloud container, Python 3.13 | All passed |
| Same tests | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Fintraffic sample (143 JPEGs, one preset) | Linux cloud container | 1 view, 141 images (62 colour, 79 monochrome), linked at structure similarity 0.84 and marked for confirmation; 2 rain-on-lens frames unassigned; 3.4 s |

## Not tested

- A camera with several real presets. The "different view" cases use synthetic scenes and mirrored copies of the sample.
- A real camera that moved. Drift detection is tested on synthetic shifts only.
- Windows performance on large archives.

## Known limitations

- The colour/monochrome link threshold (0.60) is close to what a mirrored image scores (0.62 between cluster centres), so links always need operator confirmation.
- Images with different resolutions are never grouped together automatically.
- No screens yet: views and regions are only reachable through code and tests.
