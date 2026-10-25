# Delivery report: plates, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.plate_layout`: normalisation (separators, Latin case, Persian and Eastern Arabic digits), country layouts for Israel, Latvia, Iran and Lebanon as working hypotheses, readability rules that turn unsure characters into "?" and never fill them, and the character error rate.
- `trafficwb.plates`:
  - the plate finder and reader packs (evaluation only; see `docs/decisions/plate-models.md`);
  - the plate stage, which runs after vehicle detection in plate recognition missions, with four outcomes per vehicle and the plate crop stored;
  - append-only operator review (confirm, correct, unreadable, no plate, reopen);
  - the evaluation gate: a country stays evaluation only until an evaluation on reviewed plates passes and an operator enables it;
  - the plates export.
- `scripts/fetch_models.py` fetches the two plate packs with pinned checksums and writes their model cards.
- Database migration v7 adds the tables `plate_read`, `plate_revision` and `plate_country_status`.
- Web: a Plates tab for plate recognition missions, with:
  - the country card (evaluation only or enabled, evaluate, enable);
  - counts by outcome;
  - filters and text search;
  - one row per vehicle: vehicle crop, enlarged plate crop, characters shaded by confidence, raw and machine text, reasons and review buttons.
  - Plate text is shown in bidi-isolated spans.
- Rules: `docs/design/plates.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite. New tests in `tests/test_plates.py` cover: normalisation with Persian and Arabic digits and leading zeros; layouts and display; unsure characters never filled; character error rate; the stage outcomes (full, partial, square box, vehicle too small); a country without a reader (found, not read); corrections refusing "?"; Persian digits typed in a correction; the CSV and Excel export keeping a leading zero; the evaluation gate, including a machine read of a plate the operator marked unreadable; the installed reader on rendered Israeli and Latvian plates | Linux cloud container, Python 3.13 | All passed |
| Same tests | GitHub Actions, Windows + Linux | See the PR checks |
| Fintraffic sample as a Latvian plate mission (only to exercise the stage: Finland is not a target country) | same | 1,027 vehicles: 615 too small, 401 no plate found, 10 plates 2–11 px high (unreadable, correctly), and 1 "plate" that is a square yellow road sign, which the reader read as "T1". It is now marked partial because the box is not plate-shaped. The plate stage took about 38 s. |
| Plates screen in Chromium against that mission | Playwright | Filters, review buttons, correction and evaluation worked, with no console errors |

## Not tested

- Any real plate from Israel, Latvia, Lebanon or Iran. There are none in the sample, so **no plate accuracy is known**.
- Persian and Arabic reading: no installed reader supports it.
- Night plates, motorcycle plates, two-line plates.
