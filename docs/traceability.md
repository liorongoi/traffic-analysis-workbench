# Requirements traceability

Requirement IDs come from the design package (`docs/design/design-package.html`, section 01).
"Tested" means an automated test in this repository passed in CI or in a recorded local run. Nothing is marked tested on the basis of inspection alone.

| Req | Implementation | Tests | Status |
|---|---|---|---|
| TIME-01 filename UTC, precision kept, suffix opaque | `src/trafficwb/timeparse.py` | `tests/test_timeparse.py` | Tested (unit) |
| TIME-02 no filesystem fallback, malformed handling | `timeparse.parse_capture_time` raises `MalformedName` | `tests/test_timeparse.py` | Tested (unit). Parse preview UI not built. |
| TIME-04 IANA zone id, offset, DST status | `src/trafficwb/localtime.py` | `tests/test_localtime.py` | Conversion tested. Map UI not built. |
| TIME-05 per-timestamp rules | `localtime.to_local` | `tests/test_localtime.py` | Tested for IL, LV, LB, IR, FI. Manual override not built. |
| IMG-02 schedules, overnight, multiple windows, DST | `src/trafficwb/schedule.py` | `tests/test_schedule.py` | Tested (unit) |
| DER-02 minute-precision boundary rule | `schedule.Schedule.contains_local` | `tests/test_schedule.py` | Tested (unit) |
| HW-01 hardware detection | `src/trafficwb/probe.py` | `tests/test_probe.py` | Runs on Linux and CI runners. Not run on the target Windows workstation. GPU backend verification not built. |
| HW-02 RAM tier independent of GPU | `probe.suggest_profile` | `tests/test_probe.py` | Tested (unit). Tier cut-off is a proposal. |
| MIS-06 / ING-02 format definition from samples | `src/trafficwb/inventory.py` | `tests/test_inventory.py` | Tested (unit), run on the Fintraffic sample (see delivery report) |
| ING-03 identical bytes are distinct captures; no duplicate processing | `inventory`, `ingest.Ingestor` | `tests/test_inventory.py`, `tests/test_store_ingest.py` | Tested (unit) |
| MIS-02 per-mission DB created on commit, isolated | `src/trafficwb/store.py` | `tests/test_store_ingest.py` | Tested (unit) |
| MIS-03 run records config snapshot | `store.Mission.create_run` | `tests/test_store_ingest.py` | Config snapshot tested. Model versions not applicable yet. |
| MIS-04 one input type and mode, immutable | `store.Mission.update_config` | `tests/test_store_ingest.py` | Tested (unit) |
| MIS-06 validation of source, output, storage, consistency | `store.validate` | `tests/test_store_ingest.py` | Tested (unit). Wizard UI not built. |
| MIS-07 continuous monitoring independent of browser, resumes after restart | `ingest.Monitor`, `ingest.resume_enabled` | `tests/test_store_ingest.py` | Tested with polling. Filesystem notifications not tested. |
| ING-01 recursive discovery, stability, reconciliation, checkpoints, restart recovery | `src/trafficwb/ingest.py` | `tests/test_store_ingest.py`, sample run | Tested (unit + sample) |
| ING-02 corrupt, partial, replaced, renamed, unsupported, unavailable | `ingest.Ingestor` | `tests/test_store_ingest.py` | Tested (unit) |
| ING-04 identity, checksum, versions, originals untouched, replacements keep history | `ingest.Ingestor._ingest` | `tests/test_store_ingest.py` | Tested (unit) |
| ING-06 / DER-04 outputs and data never re-ingested | `store.validate`, `ingest.Ingestor._walk` | `tests/test_store_ingest.py` | Tested (unit) |

| VIEW-01 location vs view, grouping proposals, reassign, split, merge, rename, history | `src/trafficwb/views.py` (`ViewService`), migration v2 in `store.py` | `tests/test_views_regions.py`, sample run | Tested (unit + sample). Screens not built. |
| VIEW-02 lighting change alone does not create a new view | `views.assign_or_propose` (colour/monochrome link by structure similarity) | `tests/test_views_regions.py`, sample run | Tested. Links found this way need operator confirmation. Background models not built yet. |
| VIEW-03 inclusion/exclusion polygons, precedence, original coordinates, membership rule | `src/trafficwb/regions.py` | `tests/test_views_regions.py` | Tested (unit). Drawing screen not built. |
| VIEW-04 flag a view when it changes | `views.ViewService.check_drift` | `tests/test_views_regions.py` | Tested on synthetic shifts. Not seen on a real moved camera. |

| IMG-01 per-image vehicle detection and counts, never unique vehicles | `src/trafficwb/detect.py`, `process.py`, `review.py` | `tests/test_detect_review.py`, sample run | Tested (unit with a stand-in detector; real detector smoke test; sample run). Accuracy not measured: needs labelled images. |
| HW-06 every resize, tile and threshold recorded on the run | `process.Processor.process_run`, `detect.YoloxDetector.describe` | `tests/test_detect_review.py` | Tested (unit) |
| ENV-05 / ENV-06 no runtime downloads; models prepared on staging | `src/trafficwb/models.py`, `scripts/fetch_models.py` | `tests/test_detect_review.py` | Checksum verification tested. Offline VM proof not done. |
| PRG-01 recover after interruption | `process.Processor` | `tests/test_detect_review.py` | Resume without duplicates tested. Pause, cancel, ETA and live progress screen not built. |
| PRG-02 processing status separate from review status | `run_item.processing_state`, `review_decision` | `tests/test_detect_review.py` | Tested (unit) |
| REV-01 results table, search, sort | `review.ReviewService.results` | `tests/test_detect_review.py` | Logic tested. Screen not built. |
| REV-02 approve all: scope and count, bulk not reviewed, no later data | `review.ReviewService.approve_all_preview/commit` | `tests/test_detect_review.py` | Tested (unit). Screen not built. |
| REV-03 remove false, add missed, correct class, approve, prev/next, index of total | `review.ReviewService` | `tests/test_detect_review.py` | Logic tested for vehicles. Grouping, candidate and plate corrections come with Phases 4 and 5. Screen not built. |
| COR-01 corrections append-only with provenance | `annotation_revision`, `review_decision` | `tests/test_detect_review.py` | Tested for object corrections. Other correction types later. |
| EXP-01 exports: formats, ids, Unicode, leading zeros, prediction vs correction vs approval, no overwrite, failures reported | `src/trafficwb/export.py` | `tests/test_detect_review.py` | Tested (unit, CSV and Excel) |

| NGT-01 light components, groups, candidates and vehicles kept separate; light count never vehicle count | `src/trafficwb/night.py`, `night_service.py`, `review.py` | `tests/test_night.py` | Tested (unit and synthetic pipeline), run on the sample |
| NGT-02 per-view, per-condition background; unexplained lights become candidates | `night.build_background`, `NightService.ensure_background` | `tests/test_night.py` | Tested (unit), run on the sample |
| NGT-03 background never silently absorbs stopped vehicles | `night.build_background` (vehicle areas left out, new lights reported) | `tests/test_night.py` | Tested on synthetic data. Not seen on a real parked vehicle. |
| NGT-04 condition from content; no enhanced evidence; confirmed and unresolved shown apart | `night.measure_condition`, `review.results`, `export` | `tests/test_night.py` | Tested (unit + sample frames) |

All other requirements: not started.

Evidence for each phase is in `docs/delivery/`.
