# Night analysis (still images)

## Condition is measured, not taken from the clock (NGT-04)

| Measure | Rule (proposed, from the Fintraffic sample) |
|---|---|
| Mean HSV saturation < 5 | monochrome (infrared): **night** |
| Colour, mean grey < 60 or > 40% of pixels darker than 40 | **night** |
| Colour, mean grey < 85 and > 15% dark pixels | **dusk** |
| Otherwise | **day** |
| > 2% of pixels clipped at 250+ | flag `glare` |
| Median Laplacian of the central 80% ≤ 2 | flag `low_detail` (rain or fog on the lens) |

On the sample the camera's auto-exposure keeps mean brightness near 100 day and night, so brightness alone cannot tell night from day; the switch to infrared can.

## Evidence levels (NGT-01)

| Level | Stored in | Counted? |
|---|---|---|
| Light component | `light_component` | No |
| Light group (single, pair, cluster) | `light_group` | No |
| Vehicle candidate | `light_group.is_candidate = 1` | Only as "unresolved", as a range |
| Detected vehicle | `detection` (Phase 3) | Yes, machine count |
| Confirmed vehicle | `candidate_revision` (confirmed) or `annotation_revision` (added) | Yes, reviewed count |

Lights inside a detector box (expanded by 15%) support that detection and are not counted again.

## Steps

1. **Light components** on the original image: grey level ≥ 235 and at least 40 above the local background (white top-hat at 25 and 61 px), fragments closed at 5 px, at least 6 px in area. Nothing is enhanced, so no enhanced copy can pose as evidence.
2. **Persistent-light background** per view and condition, from many images. Each pixel's share of images in which it is lit; lit in ≥ 40% of images means persistent. It needs at least 8 images; with fewer, nothing is explained away and the images are flagged `no_background`. It is rebuilt from stored components (no image is re-read) when the image set grows by 10% or the operator's light labels change.
3. **Stopped vehicles (NGT-03).** Pixels under a detected, added or confirmed vehicle are left out of that image's contribution, so a confirmed parked car never becomes background. Lights frequent in the last 12 images but not over the long horizon are reported as "new persistent light?" for the operator, not absorbed.
4. **Operator labels** (`light_label`: persistent or not persistent) override the statistics.
5. **Grouping.** The expected vehicle width at each image row is fitted from daytime car detections in the same view (`width = a·row + b`). On the sample: `0.090·row + 5.6` from 679 boxes, median error 16%. Two lights form a pair when they sit at the same height (within 25% of a vehicle width), their spacing is 35–115% of a vehicle width, and their areas differ by less than 3×. Lights closer than 20% of a vehicle width form a cluster. Without a width map no pairs are formed and every light is a single with a count range of 0–1.
6. **Ambiguity is kept.**
   - A pair counts as 1–2 (one vehicle with two lights, or two vehicles with one light each).
   - A cluster counts as 1 to its number of lights, at most 3.
   - A light that could pair with more than one neighbour is flagged `ambiguous_pairing`.
7. **Reflections.** A vertically elongated light below a brighter one scores 0.7 as a possible reflection, otherwise 0.3. Its group's score drops by half of that; the group is not removed.
8. **Candidate.** A group is a candidate when it is not explained by the background, not inside a detection, inside the view's regions (or regions not applied), and its score is at least 0.2.

## Review

- **Decisions:** confirm (optionally with a vehicle outline and class), reject, restore, mark as persistent light (adds a label and regroups the view's images), and regroup. Regroup forms a new candidate from chosen lights and supersedes the groups that shared them.
- **History:** decisions are append-only revisions. Group ids come from their member lights, so a decision survives a rebuild that forms the same group.
- **Reopening:** a decision on an approved image reopens it.
- **Results table:** shows vehicles (detected + added + confirmed from lights), confirmed-from-lights, and unresolved candidates as min–max, each in its own column. The export carries the same columns. In the objects table, every candidate has its own row with an evidence level, light ids, count range and alternatives.

## Re-analysis

Images are regrouped when their view's background or regions change. Older groups stay in the database and only the latest analysis is current.

Detections from Phase 3 keep the region membership they had when processed. A region drawn later applies to night candidates at once, but to detections only after reprocessing (planned).
