# Delivery report: night still images, background lights, distance bands and learning from decisions

Date: 25 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Why

Two points from Lior:

- The main job is to learn each camera's background lights and to tell car lights apart from them.
- A car looks different depending on its distance:
  - far away it is one light with a glow;
  - in the middle it is one light or two lights close together;
  - close up its shape is visible.

## Implemented

- **Glare is a light.** A large clipped region is now a light component. It is often the headlights of a close car facing the camera. Before this, glare was missed, because the top-hat filter only finds lights narrower than its kernels.
- **The background reads every light** as persistent, attached, recurring, occasional or rare (`Background.reading`):
  - Persistent: the light lies in the persistent-light mask.
  - Attached: the light touches a persistent light (a lamp's changing glow). It is explained, except when it is glare.
  - Recurring and occasional: the light is often lit at that place (lamps that are not always on, signs, bridge lighting). Its score drops.
  - Each group stores its recurrence, and the investigation screen shows it.
- **Distance bands per camera view** (far, middle, close) come from the expected vehicle width at each image row:
  - The rules follow what a vehicle looks like in each band. A close light with no detected shape is marked and may count as nothing.
  - The band edges can be set on the Camera views tab, and saving re-analyses that view's night images.
  - The investigation screen draws the bands.
- **Long thin lights** (railings, poles, road markings) are scored down.
- **Operator labels:**
  - "Not persistent" also clears the history at that place, so the label wins over the statistics.
  - "Persistent" marks the place as always lit.
- **Exports** gain three columns: the band, the share of images lit at that place, and the night flags for each candidate.
- Database migration v9 adds `view_night_setting` and the columns `band`, `recurrence` and `setting_id` on light groups.
- Rules: `docs/design/night.md`, steps 1, 4a and 6a.

### Part 2

- **Where vehicles drive.** Each view learns a map from its daytime detections. A light far from every place a vehicle was seen (sky, lamp heads, the caption, verges) is scored down.
- **Recurring on the road no longer counts as background.** On a busy lane, oncoming cars light the same spot in many images. The first version treated that spot as background and dropped real cars. Now the recurring penalty applies only off the road. For glare, recurrence is the average over the blob.
- **Images with no camera view borrow the most similar view's background** (rain on the lens).
- **Learning from operator decisions.** A place where an operator said "Not a vehicle" at least twice in other images of the view is scored down. A place where vehicles were confirmed at least twice is kept. Decisions are applied at each processing run, or with the "Apply my night decisions now" button on the Camera views tab.
- Database migration v10 adds `decision_rev` on light groups.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite, with new tests in `tests/test_night.py` | Linux cloud container | All passed |

The new tests check:

- bands from the width map, and what counts as a vehicle in each band;
- long thin lights;
- glare found as a light;
- the five background readings, including glare next to a lamp and the "not persistent" label;
- saving band widths re-analyses the view, and only once;
- the vehicle map;
- rejections at a place applied to the view's other images (an image's own decision is not counted for itself);
- borrowing the nearest view.

Fintraffic sample, 143 images, 82 of them night, processed before and after on 4 CPU cores:

Part 1 ("after") and part 2 together:

| What | Before | After part 1 | After part 2 |
|---|---|---|---|
| Candidates per night image | 14.3 | 9.1 | 8.9 |
| Candidates on the two images with no camera view | 65 | 66 | 33 |
| Real cars dropped in four images checked by eye | 0 | 3 (an oncoming car's glare, cars on the far bridge) | 0 |

Part 2 keeps roughly the same number of candidates as part 1. It trades lights far from the road, which are now dropped, for real cars on the road, which part 1 had wrongly dropped.

Part 1 details:

| What | Before | After |
|---|---|---|
| Unexplained light groups offered as candidates, per night image | 14.3 | 9.4 |
| Candidates touching a known lamp | 19% | 5% |
| Candidates at a place lit in ≥ 20% of images | 39% | 13% |
| Car lights inside detected vehicles that the new rules would push down (lit there in ≥ 20% of images) | n/a | 4% |
| Close oncoming cars seen only as glare | missed | found |

## What the results look like (visual check, not a measurement)

I looked at 36 random candidates before and after, and at four whole night images. These are my own judgements from the pictures, not labelled data:

- Many candidates are **real cars that the vehicle detector misses at night**. Examples are oncoming cars seen only as headlights or glare, cars on the far bridge, and trucks with light bars. Before, I judged roughly 40% of a random sample to be vehicles. After, about half.
- Still wrong after the change:
  - pools of light on the road below the bridge (they lie on the road, so the vehicle map cannot help; operator rejections at that place will);
  - a few lit pole segments and railing pieces;
  - lights at the very top edge in rain;
  - part of the burned-in caption.
- The two images with no camera view (rain on the lens) have no background, so every light there is a candidate (about 22 per image).

## Part 3: soft glow on the road (light pools)

Light pools on the road and under gantries, lit lane lines and raindrops on the lens share one trait: they are never clipped, and they are only a little brighter than what is around them. Car lights have a clipped or nearly clipped core. Groups with a peak below 250 and less than 35 grey levels above their surroundings now score × 0.4 (flag `soft_glow_no_bright_core`).

- On the sample, candidates went from 733 to 697 (78 night images): 9.4 → 8.9 per image.
- By eye, 36 of the 38 groups it matched were not vehicles. The other two stay candidates.
- Pools that do clip (peak ≥ 250) are not caught by this rule. Operator rejections at the place still apply to them.
- Older missions are re-checked automatically at the next processing run, because the rules version is now stored on each group.

## Part 4: learned scoring from your answers

The hand-made rules are a starting point. Each camera view now learns its own scoring from the candidates you answer on the Night review tab. A small model weighs the measurements of each light group (how bright, how much it stands out, size against the vehicle width, shape, how often the place is lit, distance to the road). It is used only after at least 15 answers of each kind, and only when it ranks your answers better than the rules on images it did not learn from. The tab shows whether it is in use and why.

- Sample check, with 101 candidates I labelled by eye (55 vehicles): AUC 0.91 for the model against 0.78 for the rules. So the model was used.
- What it learned there: taller-than-wide shapes, small and dim lights are less likely vehicles; pairs, bigger and brighter lights are more likely.
- Without answers nothing changes.

## Not tested

- Accuracy against labelled night images. The Night review tab now measures it from your answers (share of candidates that were vehicles, by distance band and flag), but no answers exist yet. Missed vehicles are counted from vehicles you draw on night images you approve.
- The default band widths (24 px and 60 px) are a first guess for this camera. They should be checked per camera.
- Colour cameras at night (the sample switches to infrared).

## Known limits

- The rules are fixed thresholds, not a learned model. A trained light classifier per view, using operator decisions as labels, is the natural next step once there are reviewed night images.
- Band widths and the background rules apply to new analyses. Older groups stay in the database.
