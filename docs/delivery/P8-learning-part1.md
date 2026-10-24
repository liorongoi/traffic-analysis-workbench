# Delivery report: learning, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.learning`:
  - eligibility (full, partial, bulk, none) and labels per image;
  - frozen dataset versions with group-wise splits by camera view and date block, a frozen test set, duplicate removal and a leakage check;
  - COCO export per split with a manifest, never overwriting;
  - evaluation of a detector on a split: precision, recall, F1, AP50, count error and class agreement, overall, by light condition and by view;
  - `ModelRegistry`: production pointer per task in `control.db`, promotion and rollback.
- Database: mission migration v8 (`training_dataset`, `dataset_member`, `evaluation`) and the control table `model_promotion`.
- The web server loads the vehicle detector in production use, and reloads it after a promotion or rollback.
- Web: a Learning tab for image missions, with eligibility counts, dataset versions and export, evaluation (as a background job) with details, the production model with promotion and rollback, and the list of planned training adapters. The Approve button explains what approval means for training.
- Rules: `docs/design/learning.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite. New tests in `tests/test_learning.py`: full, partial and bulk eligibility; group-wise splits; frozen test groups across versions; leakage detection blocking export; bad parameters refused; COCO export and no overwrite; metrics for a perfect detector, one that misses trucks and one with an extra box; partial reviews give no false positives; promotion needs an evaluation and a matching task; rollback steps back to the default | Linux cloud container, Python 3.13 | All passed |
| API: dataset, export, evaluation job with a chosen model, promotion refused for a different version, promotion and rollback, operator refused | `tests/test_web_api.py` | Passed |
| Learning tab on the Finnish sample: 25 images approved, a version frozen, YOLOX-s evaluated on its test split | Chromium via Playwright | Worked, no console errors |

## Not tested

- Any metric on a meaningful test set. The sample has few vehicles per image and only two days, so its test split held a single image.
- Training, which does not exist yet.
- Promotion with a second real detector pack (only stand-in packs in tests).
