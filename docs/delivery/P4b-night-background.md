# Delivery report: night still images, background lights and distance bands

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

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite, with new tests in `tests/test_night.py` | Linux cloud container | All passed |

The new tests check:

- bands from the width map, and what counts as a vehicle in each band;
- long thin lights;
- glare found as a light;
- the five background readings, including glare next to a lamp and the "not persistent" label;
- saving band widths re-analyses the view, and only once.

Fintraffic sample, 143 images, 82 of them night, processed before and after on 4 CPU cores:

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
  - pools of light on the road below the bridge;
  - a few lit pole segments and railing pieces;
  - lights at the very top edge in rain;
  - part of the burned-in caption.
- The two images with no camera view (rain on the lens) have no background, so every light there is a candidate (about 22 per image).

## Not tested

- Accuracy against labelled night images. It needs labelled images from your cameras (the night evaluation plan, NGT-06).
- The default band widths (24 px and 60 px) are a first guess for this camera. They should be checked per camera.
- Colour cameras at night (the sample switches to infrared).

## Known limits

- The rules are fixed thresholds, not a learned model. A trained light classifier per view, using operator decisions as labels, is the natural next step once there are reviewed night images.
- Band widths and the background rules apply to new analyses. Older groups stay in the database.
