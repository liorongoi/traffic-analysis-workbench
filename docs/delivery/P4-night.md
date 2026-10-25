# Delivery report: Phase 4 (night analysis for still images)

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.night`:
  - Image condition from content (day, dusk, night; `glare` and `low_detail` flags).
  - Light components.
  - Persistent-light background that leaves vehicle areas out and reports new lights.
  - Vehicle width by image row, learned from daytime detections.
  - Light grouping into single, pair and cluster, with count ranges and alternatives.
  - Reflection scoring and detection coverage.
- `trafficwb.night_service`: stores all of the above per mission. It rebuilds backgrounds from stored lights without re-reading images, and re-analyses images when a background or region changes.
- `trafficwb.review`: candidates, and the confirm, reject, restore, mark-as-persistent-light and regroup decisions. Vehicle count, confirmed-from-lights count and unresolved candidates (min–max) are separate.
- `trafficwb.export`: condition and candidate columns per image; candidate rows with evidence level, light ids, count range and alternatives.
- Database migration v4. Processing records the night settings on the run, and a run interrupted after its last image now completes on the next start.
- Rules: `docs/design/night.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full test suite (142 tests), including synthetic night scenes and sample frames | Linux cloud container, Python 3.13 | All passed |
| Same tests | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Fintraffic sample, full processing | Linux cloud container, 4 cores | 2 min 33 s for 143 images. 82 images measured as night (all infrared), 61 as day. One background from 80 night images. Width map fitted from 679 daytime cars: width = 0.090 × row + 5.6 px. 4,891 light components, of which about 70% were explained by the background (street lamps, the caption). |
| Same mission with a "sky and city lights" exclusion drawn | same | Re-analysis of 80 images, no background rebuild; unresolved candidates fell from 1,114–1,400 to 1,046–1,332 |
| Marking 3 recurring lights as persistent on one image | same | The view was regrouped; unresolved candidates over all night images fell from 1,041–1,262 to 866–1,189 |

## What the night results look like (visual check, not a measurement)

- **Busy evening frames:** most candidates are real vehicles, such as headlight pairs on the far roads, single headlights on the bridge, and the glare of a near car.
- **Quiet hours (01:00–05:00 local, 30 images):** 0 to 13 unresolved candidates per image, median about 5. Most are not vehicles: lamps that are lit in only some frames, the lit road surface on the left lane, reflections on the railing, and distant city lights. These are the cases the operator's "persistent light" labels are for. Each label removes the light from every image of the view.
- **Vehicle counts:** the counted-vehicle figure at night stays low (82 over 80 images), because lights are never counted on their own. That is by design (NGT-01).

## Not tested

- Accuracy of candidates against labelled night images. The evaluation plan in the design package (candidate acceptance rate, false alarms per image, recall by appearance) needs labelled images from your cameras.
- Colour cameras at night or at dusk. The sample camera switches to infrared, so the dusk rule has not been checked on real images.
- A real stopped vehicle at night.

## Known limitations

- Unresolved candidates have low precision at quiet hours until persistent lights are labelled. The next step in the design is a light-pair classifier trained on reviewed groups.
- Detections keep the region membership from processing time. A region drawn later applies to candidates at once, but to detections only after reprocessing.
- No screens yet.
