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
| MIS-06 validation of source, output, storage, consistency | `store.validate`, `ui/src/pages/Wizard.tsx` | `tests/test_store_ingest.py`, `tests/test_web_api.py` | Tested (unit + API). Wizard with file-name preview checked in a browser. |
| MIS-07 continuous monitoring independent of browser, resumes after restart | `ingest.Monitor`, `ingest.resume_enabled` | `tests/test_store_ingest.py` | Tested with polling. Filesystem notifications not tested. |
| ING-01 recursive discovery, stability, reconciliation, checkpoints, restart recovery | `src/trafficwb/ingest.py` | `tests/test_store_ingest.py`, sample run | Tested (unit + sample) |
| ING-02 corrupt, partial, replaced, renamed, unsupported, unavailable | `ingest.Ingestor` | `tests/test_store_ingest.py` | Tested (unit) |
| ING-04 identity, checksum, versions, originals untouched, replacements keep history | `ingest.Ingestor._ingest` | `tests/test_store_ingest.py` | Tested (unit) |
| ING-06 / DER-04 outputs and data never re-ingested | `store.validate`, `ingest.Ingestor._walk` | `tests/test_store_ingest.py` | Tested (unit) |

| VIEW-01 location vs view, grouping proposals, reassign, split, merge, rename, history | `src/trafficwb/views.py` (`ViewService`), migration v2 in `store.py` | `tests/test_views_regions.py`, sample run | Tested (unit + sample). Screens not built. |
| VIEW-02 lighting change alone does not create a new view | `views.assign_or_propose` (colour/monochrome link by structure similarity) | `tests/test_views_regions.py`, sample run | Tested. Links found this way need operator confirmation. Background models not built yet. |
| VIEW-03 inclusion/exclusion polygons, precedence, original coordinates, membership rule | `src/trafficwb/regions.py`, `ui/src/pages/RegionEditor.tsx` | `tests/test_views_regions.py`, `tests/test_web_api.py` | Tested (unit + API). Drawing screen checked in a browser on the sample. |
| VIEW-04 flag a view when it changes | `views.ViewService.check_drift` | `tests/test_views_regions.py` | Tested on synthetic shifts. Not seen on a real moved camera. |

| IMG-01 per-image vehicle detection and counts, never unique vehicles | `src/trafficwb/detect.py`, `process.py`, `review.py` | `tests/test_detect_review.py`, sample run | Tested (unit with a stand-in detector; real detector smoke test; sample run). Accuracy not measured: needs labelled images. |
| HW-06 every resize, tile and threshold recorded on the run | `process.Processor.process_run`, `detect.YoloxDetector.describe` | `tests/test_detect_review.py` | Tested (unit) |
| ENV-05 / ENV-06 no runtime downloads; models prepared on staging | `src/trafficwb/models.py`, `scripts/fetch_models.py` | `tests/test_detect_review.py` | Checksum verification tested. Offline VM proof not done. |
| PRG-01 recover after interruption | `process.Processor` | `tests/test_detect_review.py` | Resume without duplicates tested. Pause, resume, cancel and measured ETA through `web/jobs.py`, tested in `tests/test_web_api.py`; progress card checked in a browser. |
| PRG-02 processing status separate from review status | `run_item.processing_state`, `review_decision` | `tests/test_detect_review.py` | Tested (unit) |
| REV-01 results table, search, sort | `review.ReviewService.results`, `ui/src/pages/Mission.tsx` | `tests/test_detect_review.py`, `tests/test_web_api.py` | Tested; screen checked in a browser on the sample. |
| REV-02 approve all: scope and count, bulk not reviewed, no later data | `review.ReviewService.approve_all_preview/commit` | `tests/test_detect_review.py`, `tests/test_web_api.py` | Tested (unit + API). Dialog shows scope and count; checked in a browser. |
| REV-03 remove false, add missed, correct class, approve, prev/next, index of total | `review.ReviewService` | `tests/test_detect_review.py` | Logic tested for vehicles. Candidate and regroup corrections added in Phase 4; plate corrections not built. Investigation screen (`ui/src/pages/Investigate.tsx`) with keyboard shortcuts checked in a browser on the sample. |
| COR-01 corrections append-only with provenance | `annotation_revision`, `review_decision` | `tests/test_detect_review.py` | Tested for object corrections. Other correction types later. |
| EXP-01 exports: formats, ids, Unicode, leading zeros, prediction vs correction vs approval, no overwrite, failures reported | `src/trafficwb/export.py` | `tests/test_detect_review.py` | Tested (unit, CSV and Excel) |

| NGT-01 light components, groups, candidates and vehicles kept separate; light count never vehicle count | `src/trafficwb/night.py`, `night_service.py`, `review.py` | `tests/test_night.py` | Tested (unit and synthetic pipeline), run on the sample |
| NGT-02 per-view, per-condition background; unexplained lights become candidates | `night.build_background`, `NightService.ensure_background` | `tests/test_night.py` | Tested (unit), run on the sample |
| NGT-03 background never silently absorbs stopped vehicles | `night.build_background` (vehicle areas left out, new lights reported) | `tests/test_night.py` | Tested on synthetic data. Not seen on a real parked vehicle. |
| NGT-04 condition from content; no enhanced evidence; confirmed and unresolved shown apart | `night.measure_condition`, `review.results`, `export` | `tests/test_night.py` | Tested (unit + sample frames) |

All other requirements: not started.

Evidence for each phase is in `docs/delivery/`.
| ARC-02 / ARC-03 local web server, browser UI, work continues without a browser | `src/trafficwb/web/app.py`, `web/jobs.py`, `ui/` | `tests/test_web_api.py` | Tested (API). Jobs and monitoring run in server threads; monitoring resumes at server start. |
| ENV-02 / ENV-03 localhost by default, LAN only on request; sign-in, one operator at a time | `web/app.py` (`--lan`), `web/auth.py` | `tests/test_web_api.py` | Sign-in, take-over, idle expiry, first account from localhost only, header check on changes: tested (API). Windows service install not built. |
| UX-01 screens: missions, wizard, overview, results, investigation, views, regions | `ui/src/pages/*` | Browser run on the sample (see `docs/delivery/P5-web-screens-part1.md`) | Checked by hand in Chromium. No automated UI tests in CI; the UI build (type check) runs in CI. |
| IMG-03 chat request to structured spec shown before execution; no generated code runs | `src/trafficwb/interpret.py`, `search.py`, `ui/src/pages/Search.tsx` | `tests/test_search.py` | Written 34-request set passes; invalid specs refused; not-understood words block the run until edited or ignored. English only (Q11). |
| IMG-03 uncertain or unobservable attributes stay uncertain | `src/trafficwb/attributes.py`, `search.SearchService._match` | `tests/test_search.py`, sample run | Colour: observed / uncertain / not observable (infrared, small, dark, burned, lamps). Subtypes always "possible". Accuracy not measured: checked by eye only. |
| Derived: search hits reviewable, attribute corrections append-only, exports reproducible | `search.decide`, `search.set_colour`, `search.export_search` | `tests/test_search.py` | Tested (API). Export carries the exact spec, edits and reader version. |

