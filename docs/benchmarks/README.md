# Benchmarks on public datasets

Date: 13 Nov 2026. Code: `main` at `10badef` (video phase 6a). Machine: Linux cloud container, 4 CPU cores, no GPU.
Script: `scripts/benchmark.py`. Raw results, with every setting and the exact image or sequence list: `results/*.json`.

These numbers measure LanePulse as it runs today, with its own code paths: the standard detector (YOLOX-s with
LanePulse's tiling, 0.30 threshold and class merge), SAM 3.1 (the W8A16 OpenVINO pack, default vehicle words plus
"person") and both merged, and for video the segment analysis (`video_segments.analyse`), the tracker and the
"confirmed track = one event" rule. They are not a claim about the operator's own cameras, which no public set
replaces.

## Summary

| | Standard (YOLOX-s) | SAM 3.1 | Both |
|---|---|---|---|
| Ordinary photos (COCO), vehicle AP50, 100 images | 67.1 | 83.1 | 83.0 |
| Drone, daylight (VisDrone), vehicle AP50, 60 images | 48.3 | 72.3 | 72.8 |
| Drone, thermal (HIT-UAV), vehicle AP50, 100 images | 5.7 | 55.9 | 55.9 |
| Seconds per image on 4 CPU cores | 0.07 to 0.43 | 10.6 to 11.8 | 10.8 to 12.6 |
| People tracking (MOT17), HOTA / IDF1, video at 5 frames/s | 45.2 / 53.6 | 38.1 / 39.3 (1 frame/s) | 43.7 / 51.4 |
| People counted vs. really there (MOT17, 7 videos) | 609 vs 331 (1.84x) | 170 vs 317 (0.54x) | 492 vs 331 (1.49x) |

1. **The YOLOX-s model and its preprocessing are right.** Run as its authors evaluate it, it scores 40.5 COCO AP,
   the published figure exactly.
2. **SAM 3.1 finds far more vehicles than YOLOX-s on every set**, most of all from the air and in thermal, where
   YOLOX-s finds almost nothing (3.5% of thermal vehicles at its threshold). It costs about 150 times the time.
3. **"Both" adds almost nothing over SAM 3.1 alone** on still images (within 0.5 AP50), at a slightly lower precision.
4. **Video counts are too high with the standard detector.** One person becomes 1.8 events on average at 5 frames
   per second and 3.2 at every frame: tracks break and restart. About a third of the extra events are already
   flagged "possible duplicate".
5. **SAM 3.1 alone on a moving camera finds nothing.** At one look per second, every frame of a hand-held or
   vehicle-mounted video is taken for a scene cut, so no track lives long enough to become an event (0 events on
   MOT17-11 and -13).

## Datasets

| Set | What it tests | Used | Licence | Where from |
|---|---|---|---|---|
| COCO val2017 | Ordinary photos: people, bicycles, cars, motorcycles, buses, trucks | All 5,000 images (YOLOX); 100 images holding a vehicle (all three detectors) | Annotations CC BY 4.0; images under their Flickr terms | `coco2017val.zip` from the Ultralytics assets release (images.cocodataset.org is blocked here); annotations are COCO's own `instances_val2017.json` |
| VisDrone2019-DET val | Drone video frames, daylight, oblique and top-down, very small and crowded targets | All 548 images (YOLOX); 60 images (all three) | Research and education only (VisDrone terms) | `VisDrone2019-DET-val.zip`, Ultralytics assets release (byte-identical to the official zip) |
| HIT-UAV test | Drone thermal infrared, 60 to 130 m, day and night | All 579 images (YOLOX); 100 images (all three) | CC BY 4.0 | github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset |
| MOT17 train, first half | People tracking, fixed and moving cameras (7 videos, 2,652 frames) | All 7 sequences | CC BY-NC-SA 3.0 | `MOT17-50.zip` (first 50% of each training sequence), BoxMOT release v10.0.83 |

Not reachable from this environment: UA-DETRAC (the vehicle-tracking set), FLIR ADAS (thermal, behind a form), the
MOTChallenge site and Hugging Face. The Mac linked to the project could not reach them either. A vehicle-tracking
benchmark is therefore still missing; see "Not measured".

Class mapping: vans count as cars (as COCO and LanePulse count them); VisDrone "pedestrian" and "people" are both
person; VisDrone tricycles and "others" and HIT-UAV "other vehicle" are excused in the per-class scores and, for HIT-UAV,
counted in "vehicle, any class". Ignored regions are crowd regions. "Vehicle, any class" scores a box as right when it
covers a vehicle, whatever class it was given: it is what a vehicle count needs.

## Detection

### COCO val2017

| Detector | Images | Vehicle AP50 | Vehicle AP | Small / medium / large vehicle AP | Car AP50 | Person AP50 | Vehicle precision / recall at the app's threshold | s per image |
|---|---|---|---|---|---|---|---|---|
| YOLOX-s as published (80 classes, 0.001) | 5,000 | 69.2 | 44.5 | 27.3 / 53.2 / 70.4 | 64.7 | 81.0 | 71.5 / 63.5 | 0.09 |
| Standard, as LanePulse runs it | 5,000 | 59.6 | 39.8 | 22.4 / 48.3 / 66.9 | 54.8 | 71.4 | 79.1 / 62.4 | 0.07 |
| Standard | 100 | 67.1 | 44.1 | 25.0 / 50.9 / 67.5 | 59.3 | 59.6 | 80.4 / 69.2 | 0.07 |
| SAM 3.1 | 100 | 83.1 | 60.0 | 42.1 / 69.2 / 77.1 | 78.4 | 78.3 | 73.6 / 87.4 | 10.6 |
| Both | 100 | 83.0 | 60.5 | 42.6 / 69.7 / 78.0 | 78.2 | 78.2 | 70.5 / 87.9 | 10.8 |

- All 80 classes, YOLOX-s as published: **40.5 AP** (published: 40.5, YOLOX README). The model pack is sound.
- LanePulse's 0.30 threshold cuts the low-score tail that AP rewards, so its AP is lower than the published run while its
  recall at the threshold is the same (62.4% vs 63.5%) and its precision higher (79.1% vs 71.5%). The operating point,
  not AP, is what an operator sees. COCO images are small, so tiling is not used on them.
- SAM 3's paper reports 53.6 box AP on all of COCO (closed vocabulary). That is not the same measure as the 6-class,
  100-image subset here, which is why the table compares the three detectors on the same images only.

### VisDrone2019-DET val (drone, daylight)

| Detector | Images | Vehicle AP50 | Vehicle AP | Small / medium / large vehicle AP | Mean AP50 of 6 classes | Car | Person | Motorcycle | Bicycle | Vehicle P / R | s per image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Standard | 548 | 47.2 | 30.7 | 15.5 / 47.8 / 70.4 | 23.9 | 57.9 | 33.0 | 11.3 | 5.3 | 82.6 / 49.5 | 0.43 |
| Standard | 60 | 48.3 | 30.6 | 16.4 / 47.7 / 72.0 | 23.2 | 60.3 | 31.8 | 12.7 | 9.8 | 82.8 / 50.0 | 0.43 |
| SAM 3.1 | 60 | 72.3 | 43.7 | 28.3 / 60.7 / 80.9 | 52.3 | 81.0 | 53.5 | 47.1 | 33.1 | 83.9 / 75.6 | 11.8 |
| Both | 60 | 72.8 | 43.8 | 28.6 / 60.9 / 80.8 | 52.5 | 81.7 | 54.3 | 46.9 | 34.2 | 79.0 / 76.2 | 12.4 |

Up to 500 boxes per image are scored, as VisDrone does. Published for comparison (VisDrone test-dev, 10 classes, every
model **trained on VisDrone**; "Evaluation of YOLO Models with Sliced Inference for Small Object Detection", arXiv 2203.04799): YOLOX-s 30.3 AP50, 36.3 with sliced training and
inference; best YOLOv5-L 48.8. LanePulse's YOLOX-s, never trained on drone images, reaches 23.9 mean AP50 over its 6
classes, and SAM 3.1 52.3 with no training at all. Small targets are the weak point of both: 15.5 AP for YOLOX-s, 28.3
for SAM 3.1.

### HIT-UAV test (drone, thermal infrared)

| Detector | Images | Vehicle AP50 | Car AP50 | Person AP50 | Bicycle AP50 | Vehicle P / R | Person P / R | s per image |
|---|---|---|---|---|---|---|---|---|
| Standard | 579 | 3.7 | 4.8 | 2.4 | 0.0 | 86.2 / 3.5 | 45.0 / 3.3 | 0.08 |
| Standard | 100 | 5.7 | 5.7 | 1.4 | 0.0 | 92.1 / 5.5 | 26.3 / 4.7 | 0.09 |
| SAM 3.1 | 100 | 55.9 | 78.1 | 18.4 | 0.0 | 68.8 / 58.2 | 17.8 / 33.6 | 11.6 |
| Both | 100 | 55.9 | 78.1 | 18.3 | 0.0 | 68.8 / 58.2 | 17.3 / 33.6 | 12.6 |

By condition (100 images; vehicle AP50, SAM 3.1 / standard): day 79.1 / 11.8, night 43.8 / 2.7, 60 to 80 m 49.3 / 7.9,
100 m and higher 70.4 / 4.6.

Published (HIT-UAV paper, Suo et al. 2023, models **trained on HIT-UAV**): YOLOv4 84.8 mAP50 (person 89.9, car 92.6,
bicycle 86.5); Faster R-CNN 76.8. LanePulse's YOLOX-s is effectively blind in top-down thermal; it was checked by
drawing ground truth and boxes on images (ground truth sits on every car and person, YOLOX-s draws nothing even at
0.05). SAM 3.1 finds cars well by day and half of them at night, but "person" in thermal gives mostly false boxes on
warm spots, and bicycles are never found.

## Tracking and counting (MOT17, people)

The 7 MOT17 training videos, first half of each. Detector, tracker settings and frame spacing are LanePulse's video
defaults: 5 analysed frames per second, ByteTrack-style tracker with weak boxes down to 0.10 for the standard detector,
SAM 3.1 once a second (alone: the whole grid at 1 s; both: on every fifth frame), camera motion taken out on the four
moving-camera videos (05, 10, 11, 13). Scored with TrackEval on the analysed frames only; "counted" is confirmed tracks,
which is the number of events LanePulse would list.

| Detector | Frames | HOTA | DetA | AssA | MOTA | IDF1 | ID switches | Counted / really there | Counted, leaving out "possible duplicate" | s per frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Standard | 5 per s (536) | 45.2 | 41.2 | 50.9 | 26.4 | 53.6 | 290 | 609 / 331 | 421 | 0.09 to 0.56 |
| Standard | every frame (2,652) | 44.0 | 39.9 | 49.9 | 18.6 | 49.7 | 796 | 1,090 / 339 | 584 | 0.09 to 0.66 |
| SAM 3.1 | 1 per s (108) | 38.1 | 32.1 | 46.6 | 14.8 | 39.3 | 144 | 170 / 317 | 166 | 9.7 to 11.1 |
| Both | 5 per s (536) | 43.7 | 40.6 | 48.1 | 22.3 | 51.4 | 362 | 492 / 331 | 447 | 2.3 to 2.7 |

Published, MOT17 validation (the second half of the same videos), detectors **trained on MOT17 and CrowdHuman**:
ByteTrack with YOLOX-X MOTA 76.6, IDF1 79.3 (HOTA 67.9 as BoT-SORT re-measured it); ByteTrack with YOLOX-S MOTA 71.1,
IDF1 73.6 (ByteTrack paper, appendix table 8). The gap is mostly detection (DetA 41): a COCO model misses the many
half-hidden people MOT17 counts, and finds people MOT17 leaves out. Association (AssA 51) is nearer the field.

Per video (standard, 5 per s), counted / really there: 02 115/49, 04 107/69, 05 86/70, 09 66/22, 10 69/36,
11 106/41, 13 60/44. Counting is worst in crowds and when people stop or cross (02, 09, 11).

## Weaknesses and where they belong in the plan

| Finding | Evidence | Where |
|---|---|---|
| YOLOX-s does not work from the air or in thermal | HIT-UAV vehicle recall 3.5%; VisDrone small-vehicle AP 15.5 | Phase 7 (drone EO and thermal): a detector trained on aerial and thermal data is needed; SAM 3.1 is the stop-gap |
| SAM 3.1 "person" in thermal is mostly false boxes; bicycles never found | HIT-UAV person precision 17.8%, bicycle AP 0 | Phase 7; train on thermal people (HIT-UAV is CC BY 4.0, so it may be used) |
| Small targets are missed by every detector | Small-object AP: 15.5 (YOLOX-s, VisDrone), 28.3 (SAM 3.1) | Phase 7 (small targets): finer tiles for high-altitude video, or a small-object detector |
| SAM 3.1 alone on a moving camera finds nothing | 0 events on MOT17-11 and -13; frames 1 s apart are judged scene cuts on all four moving-camera videos (also at 0.2 s on the walking camera of MOT17-05) | Video: measure camera motion on the 0.2 s grid even when SAM 3.1 runs once a second, and confirm a cut by the picture signature before closing tracks |
| Events are over-counted: tracks break and restart | 1.84 events per person at 5 frames/s, 3.2 at every frame | Video: re-join a new track to a recently lost one of the same family automatically when the "possible duplicate" test is strong; tracker settings per frame rate (min_hits and gap are tuned for 5/s) |
| "Both" costs SAM 3.1's time for about SAM 3.1's result on stills | Within 0.5 AP50 on all three sets | Keep "both" for video (where YOLOX fills the frames between SAM looks), and suggest SAM 3.1 alone for stills |
| SAM 3.1 takes 10 to 12 s per image on 4 CPU cores | All sets | Phase 1c (compare detectors on the operator's data) and GPU backends for OpenVINO |
| YOLOX-s at 0.30 is precise but misses a third of vehicles in ordinary photos | COCO vehicle P/R 79.1 / 62.4 | Learning page: training on reviewed images; a stronger standard model compared in phase 1c |

## Not measured

- **Vehicle tracking and vehicle counts.** UA-DETRAC and VisDrone-MOT could not be downloaded here. MOT17 has
  people only. The script reads any MOT-format folder, so a vehicle set can be added when its files are at hand.
- **FLIR ADAS** (vehicle-mounted thermal): behind a form.
- **GPU speed.** Everything ran on 4 CPU cores.
- **The operator's own cameras.** Public sets show strengths and weak points; only reviewed images from the real cameras
  say how LanePulse will do on them (phase 1c).

## Repeating these runs

```
pip install -e ".[cpu,bench]" openvino    # openvino for SAM 3.1
export TWB_MODELS=<folder with yolox-s and sam3.1-w8a16 packs>
python scripts/benchmark.py detect --suite coco --data <coco> --detector published      # checks the model: 40.5 AP
python scripts/benchmark.py detect --suite coco --data <coco> --detector standard
python scripts/benchmark.py detect --suite visdrone --data <VisDrone2019-DET-val> --detector sam3 --limit 60
python scripts/benchmark.py detect --suite hituav --data <HIT-UAV repository> --detector both --limit 100
python scripts/benchmark.py track --data <MOT17/train> --detector standard --rate app     # or full, sam3, both
python scripts/benchmark.py report
```

`--limit N` picks a repeatable sample (seed 0) of N images that hold a vehicle; the image ids are in the result.
`--standard-pack` benchmarks another standard model pack, for example a trained or imported one.

## Sources

- YOLOX-s COCO 40.5 AP: github.com/Megvii-BaseDetection/YOLOX (README, standard models).
- SAM 3, 53.6 COCO box AP: Carion et al., "SAM 3: Segment Anything with Concepts" (ICLR 2026), main results.
- VisDrone: "Evaluation of YOLO Models with Sliced Inference for Small Object Detection" (arXiv 2203.04799), table 1,
  test-dev AP50.
- HIT-UAV: Suo et al., Scientific Data 10, 227 (2023), table 4.
- ByteTrack: Zhang et al., ECCV 2022 (arXiv 2110.06864), tables 1 and 8. BoT-SORT: Aharon et al., arXiv 2206.14651,
  table 1.
