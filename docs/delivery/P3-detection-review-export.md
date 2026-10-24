# Delivery report: Phase 3 (vehicle detection, review and export for still images)

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.models`: versioned model packs (weights, model card, license, SHA-256), verified on load. `scripts/fetch_models.py` builds packs on a connected staging machine.
- `trafficwb.detect`: YOLOX-s through ONNX Runtime, whole frame plus 3x2 native-resolution tiles, per-class NMS and cross-class merge. Decision record: `docs/decisions/vehicle-detector.md`.
- `trafficwb.process`: resumable processing runs that record model and preprocessing, apply view regions and isolate failing images.
- `trafficwb.review`: results table with search, filters, sorting and paging; prev/next with index of total; remove, restore, reclass, move, add; approve, reopen; two-step "Approve all".
- `trafficwb.export`: CSV and Excel, per-image and per-object tables, no overwrite, failures recorded.
- Database migration v3: model versions, detections, annotation revisions, review decisions, bulk approvals, exports.
- CLI: `trafficwb-mission detect | results | export`.
- Rules: `docs/design/detection-review.md`.
- New dependencies: `onnxruntime`, `openpyxl`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full test suite (126 tests, including the real detector on the sample) | Linux cloud container, Python 3.13 | All passed |
| Same tests, including the real detector smoke test after fetching the model pack | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Fintraffic sample, 143 images, `scan` then `detect` | Linux cloud container, 4 cores, CPU only | 143 processed, 0 failed, 1,027 detections (951 car, 67 truck, 9 bus), 2 min 15 s in total (about 0.95 s per image) |
| Same mission: results table, one added car, one approval, "Approve all" for 21 Oct local, export CSV + Excel | Linux cloud container | Table in 0.03 s; preview 134 images; export in 0.64 s with 3 files |

Per-image counts by local hour follow the expected daily pattern: 0 overnight, about 9 at midday, and a peak of 30 to 34 at 16:00 to 18:00 local.

## Visual spot check (not a measurement)

- I looked at 3 daylight images with boxes drawn on them: nearly every vehicle on the near and middle carriageways is found, and false boxes are rare. Misses are very distant cars by the far bridge and cars hidden behind the railing.
- Night infrared images: almost nothing is found, as expected before Phase 4.
- About 10% of boxes were one vehicle labelled both car and truck; the cross-class merge now removes these.

## Not tested

- Accuracy figures (precision, recall). They need labelled images from your cameras. No numbers are claimed.
- GPU (CUDA, DirectML) and the target workstation's speed.
- Archives of tens of thousands of images: the results table is computed on request and will need stored summaries.

## Known limitations

- The detector is a general COCO model. Distant vehicles and partly hidden vehicles are missed more often than near ones.
- The candidate count is empty until night candidates exist (Phase 4).
- No screens yet: everything is available through code and the command line.
