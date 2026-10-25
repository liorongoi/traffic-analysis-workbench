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

1. **Light components** on the original image: grey level ≥ 235 and at least 40 above the local background (white top-hat at 25 and 61 px), fragments closed at 5 px, at least 6 px in area. A clipped region (grey ≥ 250) of at least 400 px is also a light: this is the **glare** of a close vehicle's headlights, which the top-hat misses because it is wider than its kernels. Nothing is enhanced, so no enhanced copy can pose as evidence.
2. **Persistent-light background** per view and condition, from many images. Each pixel's share of images in which it is lit; lit in ≥ 40% of images means persistent. It needs at least 8 images; with fewer, nothing is explained away and the images are flagged `no_background`. It is rebuilt from stored components (no image is re-read) when the image set grows by 10% or the operator's light labels change.
3. **Stopped vehicles (NGT-03).** Pixels under a detected, added or confirmed vehicle are left out of that image's contribution, so a confirmed parked car never becomes background. Lights frequent in the last 12 images but not over the long horizon are reported as "new persistent light?" for the operator, not absorbed.
4. **Operator labels** (`light_label`: persistent or not persistent) override the statistics.
4a. **How the background explains each light.** Every light gets one reading:

   | Reading | Rule | Effect |
   |---|---|---|
   | persistent | ≥ 50% of the light lies in the persistent mask | explained, never a candidate |
   | attached | within 3 px of a persistent light (its glow grows and shrinks with fog and exposure) | explained, flagged `attached_to_background_light`. Glare is never explained this way, because large glare next to a lamp is usually a vehicle's headlights near the lamp |
   | recurring | lit at this place (4 px around it) in ≥ 20% of the view's images | score × 0.4, flag `recurring_light_here` |
   | occasional | lit here in 10–20% of images | score × 0.7, flag `occasional_light_here` |
   | rare | less | no change |

   **On the road, recurring is not held against a light.** On a busy lane, oncoming cars light the same spot in many images, so recurrence there is traffic, not a lamp (a light lit in ≥ 40% of images is persistent anyway). The recurring and occasional penalties apply only off the vehicle map (4b); on it the group is flagged `often_lit_where_vehicles_drive` and keeps its score. For glare, recurrence is the average over the blob, not the highest value around it, because a large glare blob can overlap a lamp without being the lamp.

   The share is stored on each group (`recurrence`) and shown on the investigation screen. A group is judged by its least background-like light. An operator label "not persistent" also clears the place's history, so the label wins over the statistics.

   On the Fintraffic sample, lights inside detected vehicles (real car lights) were lit at their place in ≥ 20% of images only 4% of the time, and touched a persistent light 0% of the time. Unexplained lights before this step: 39% and 19%.

4b. **Where vehicles drive.** Each view learns a vehicle map: the union of its daytime vehicle boxes (score ≥ 0.4), enlarged by 15 px. It needs at least 50 daytime vehicles; with fewer there is no map and nothing is judged by it. At night a car's lights lie where its body would be by day, so a light group whose centre is more than 1.5 expected vehicle widths (40 px without a width map) from every place a vehicle was seen has its score multiplied by 0.3 and is flagged `no_vehicle_seen_here_by_day`. This covers the sky, lamp heads, captions and verges. Being just outside the map is not held against a light, because the map has gaps until many days are processed. Glare is not judged by the map.

   On the Fintraffic sample (844 daytime vehicles), 95% of car lights inside detected vehicles lay on the map, against 9% of lights explained as background.

4c. **What operators decided at the same place.** For each light group, the operator's current decisions on candidates in the view's other images are counted when their box (enlarged by 25% + 4 px) holds the group's centre:
   - rejected at least 2 times and more than twice as often as confirmed: score × 0.4, flagged `rejected_here_before`, with "an operator said 'not a vehicle' at this place N times";
   - confirmed at least 2 times and at least as often as rejected: score at least 0.4, flagged `vehicles_confirmed_here_before`.

   An image's own decisions are never counted for itself. Processing runs apply new decisions to the view's other images (`refresh(with_decisions=True)`, using `light_group.decision_rev`); so does the "Apply my night decisions now" button on the Camera views tab. A single decision on the investigation screen does not re-check the other images, so it stays fast.

4d. **Images without a camera view** (for example rain on the lens) borrow the background, width map, vehicle map, bands and decisions of the view whose signature they resemble most, when the resemblance is at least 0.6 and the size matches. They are flagged `background_from_nearest_view`, and regions are not applied to them. On the sample, two such images went from 65 candidates to 33.

5. **Grouping.** The expected vehicle width at each image row is fitted from daytime car detections in the same view (`width = a·row + b`). On the sample: `0.090·row + 5.6` from 679 boxes, median error 16%. Two lights form a pair when they sit at the same height (within 25% of a vehicle width), their spacing is 35–115% of a vehicle width, and their areas differ by less than 3×. Lights closer than 20% of a vehicle width form a cluster. Without a width map no pairs are formed and every light is a single with a count range of 0–1.
6. **Ambiguity is kept.**
   - A pair counts as 1–2 (one vehicle with two lights, or two vehicles with one light each), except in the middle band (see 6a).
   - A cluster counts as 1 to its number of lights, at most 3.
   - A light that could pair with more than one neighbour is flagged `ambiguous_pairing`.
6a. **Distance bands.** What a vehicle looks like at night depends on its distance, read from the expected vehicle width at the group's bottom row:

   | Band | Expected vehicle width (default) | What a vehicle looks like | Rule |
   |---|---|---|---|
   | far | under 24 px | one light with a glow (its two headlights merge) | a single light counts 1–1, score at least 0.4 |
   | middle | 24 to 60 px | one light, or two lights close together | a single counts 1–1; a pair counts 1–1 unless it could pair with a neighbour |
   | close | 60 px and over | its shape is visible, so the detector should find it | lights with no detected vehicle are flagged `close_without_shape`, count 0–1, score halved |

   The two widths are set per camera view on the Camera views tab (`view_night_setting`, append-only). Saving them re-analyses that view's night images at once. The investigation screen draws the band edges as dashed lines. Without a width map (fewer than about 20 daytime cars in the view), lights are not judged by distance.

   Two more shape rules: **glare** is scored at least 0.5 and offered as "glare from the headlights of a vehicle facing the camera". A **long thin** light (height/width ≤ 0.25 or ≥ 4: a lit railing, pole or road marking) has its score halved and is flagged `line_shaped`.

7. **Reflections.** A vertically elongated light below a brighter one scores 0.7 as a possible reflection, otherwise 0.3. Its group's score drops by half of that; the group is not removed.
8. **Candidate.** A group is a candidate when it is not explained by the background, not inside a detection, inside the view's regions (or regions not applied), and its score is at least 0.2.

## Review

- **Decisions:** confirm (optionally with a vehicle outline and class), reject, restore, mark as persistent light (adds a label and regroups the view's images), and regroup. Regroup forms a new candidate from chosen lights and supersedes the groups that shared them.
- **History:** decisions are append-only revisions. Group ids come from their member lights, so a decision survives a rebuild that forms the same group.
- **Reopening:** a decision on an approved image reopens it.
- **Results table:** shows vehicles (detected + added + confirmed from lights), confirmed-from-lights, and unresolved candidates as min–max, each in its own column. The export carries the same columns. In the objects table, every candidate has its own row with an evidence level, light ids, count range and alternatives.

## Night review screen and accuracy (NGT-06, part)

- **Night review tab** (image missions): candidates from every night or dusk image as a grid of crops, most uncertain first (score closest to 0.5), so a few minutes of answers teach the most. Filters: camera view, distance band, flag, and answered or not.
- **Keys:** V vehicle, N or Del not a vehicle, F fixed light, U undo, arrows move, Enter opens the whole image. Each crop shows a wider view than the light itself, with the candidate marked by a thin box.
- **Answers are the normal candidate decisions.** Nothing new is stored, so they feed "decisions at the same place" (step 6a) and the learning dataset exactly like answers given on the investigation screen.
- **Accuracy** is the share of answered candidates that were vehicles (fixed lights count as not vehicles), overall and by distance band, score range, kind and flag. Unanswered candidates are counted as open, never guessed. This measures how often a candidate is right. It does not measure missed vehicles: those need vehicles drawn by the operator on the image.

## Re-analysis

Images are regrouped when their view's background or regions change. Older groups stay in the database and only the latest analysis is current.

Detections from Phase 3 keep the region membership they had when processed. A region drawn later applies to night candidates at once, but to detections only after reprocessing (planned).
