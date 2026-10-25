# Delivery report: object search, part 1 (still images)

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.interpret`: an offline request parser that produces a validated JSON spec. It lists words it does not understand, and does not guess words whose meaning depends on the country.
- `trafficwb.attributes`: vehicle colour, read as observed, uncertain or not observable.
- `trafficwb.search`:
  - searches run only after the operator confirms them, over the mission's current objects;
  - each object is a match, possible (with reasons) or no match, and a summary counts everything left out;
  - operators can mark hits correct or not a match, and correct a vehicle's colour;
  - results can be exported.
- Database migration v5 adds the tables `object_attribute`, `attribute_revision`, `search_request`, `search_run`, `search_hit` and `search_hit_decision`.
- Web: a Search tab for object search missions, with:
  - a box for the typed request;
  - the interpretation, with notes and words not understood;
  - a full editor for the spec;
  - a progress bar;
  - results with crops, filters, verdicts, colour correction and export.
  Searches run as background jobs. Cancel works as it does for processing.
- Rules: `docs/design/object-search.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite (224 tests). New tests cover the 34-request written set, spec validation, code-like input, the colour reader on synthetic paint, light cast and lamps, the time rules, and five API flows: end-to-end search, confirmation rules, mode check, verdicts and colour corrections with export, and named regions | Linux cloud container, Python 3.13, with the sample | All passed |
| Same tests and the UI build | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Interpretation on the written request set | same | 34 of 34 as expected. I wrote this set; it does not contain requests from your operators. |
| Fintraffic sample, object search mission (143 images, 1,027 vehicles) | same | The first colour search reads the images and takes about 4 s. Later searches use cached colours and take well under a second. Every vehicle in an infrared night image gets "colour not observable". |
| Colour matches checked by eye on the sample (not a measurement; no labels) | same | See the list below this table. |
| Search screen in Chromium against the sample | Playwright | Ask, interpretation, edit, confirm, progress, results, verdicts and filters all worked, with no console errors |

What the colour searches returned on the sample, checked by eye:
- **"red cars":** 5 matches, all look red.
- **"white cars":** 100 matches. I looked at 48 of them: about 40 are white and the rest are silver or light grey.
- **"blue cars":** 4 matches, 2 look blue.
- **"black cars":** 5 matches, 2 look black. The others are light vehicles seen from behind with dark rear glass.
- Most cars end up as "possible" because their colour is uncertain or not observable. That keeps possible misses visible to the operator.

## Not tested

- Colour accuracy against labelled images. The verdicts operators give on hits are stored, so accuracy can be measured per camera.
- Requests written by your operators, and any language other than English.
- Subtypes. The detector has no pickup, van, SUV, taxi, minibus or semi-trailer classes, so these always come out as "possible". A subtype classifier is planned in the design package's model plan.
- Search in video missions.
