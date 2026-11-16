# YOLO26 compared with YOLOX

Date: 16 Nov 2026. Accuracy: Linux cloud container, 4 CPU cores, `scripts/benchmark.py` with `--standard-pack`, the same
image samples as the earlier runs (same seed and image ids). Speed on the Mac: M5 Pro, ONNX Runtime 1.19.2, 640x640,
median of 10, while a mission was running (absolute times are high; the two families ran under the same load).
Raw results: `results/*yolo26*.json` and `results/*yolox-{nano,tiny,m,l,x}*.json`.

YOLO26 (Ultralytics 8.4.215) was exported to ONNX without end-to-end NMS and runs through `detect.YoloDetector`
(family `yolo`), with the same tiles, 0.30 threshold, NMS and class merge as YOLOX. Licence: AGPL-3.0 (code and
weights); Ultralytics sells an Enterprise licence.

## Vehicle accuracy, released models

Vehicle AP50 / AP (0.5 to 0.95), precision / recall at the app's 0.30 threshold, and seconds per image on 4 cores.

| Model | Street (COCO, 100) | Drone (VisDrone, 60) | Thermal (HIT-UAV, 100) | Precision / recall, street | s per image, street / drone |
|---|---|---|---|---|---|
| YOLOX nano | 43.7 / 26.7 | 28.6 / 16.0 | 2.9 / 1.8 | 81 / 46 | 0.01 / 0.05 |
| YOLOX tiny | 55.6 / 35.4 | 35.8 / 22.2 | 4.7 / 2.5 | 84 / 58 | 0.03 / 0.13 |
| YOLO26 n | 54.1 / 37.9 | 33.6 / 24.0 | 2.0 / 1.4 | 91 / 55 | 0.03 / 0.18 |
| **YOLOX s** (default) | 67.1 / 44.1 | 48.3 / 30.6 | 5.7 / 3.4 | 80 / 69 | 0.07 / 0.38 |
| YOLO26 s | 67.0 / 48.8 | 42.2 / 30.3 | 5.7 / 3.1 | 88 / 68 | 0.07 / 0.37 |
| YOLOX m | 71.7 / 49.6 | 52.1 / 35.4 | 10.1 / 5.5 | 83 / 74 | 0.17 / 1.01 |
| YOLO26 m | 72.0 / 53.5 | 46.1 / 33.4 | 5.9 / 3.4 | 88 / 74 | 0.18 / 1.08 |
| YOLOX l | 75.3 / 53.6 | 53.2 / 36.1 | 11.5 / 6.0 | 83 / 78 | 0.35 / 1.77 |
| YOLO26 l | 70.9 / 53.4 | 45.1 / 33.2 | 13.7 / 7.7 | 89 / 73 | 0.22 / 1.13 |
| YOLOX x | 74.2 / 53.9 | 57.2 / 38.9 | 12.9 / 7.1 | 79 / 76 | 0.62 / 3.35 |
| YOLO26 x | 72.5 / 55.0 | 49.9 / 35.9 | 13.7 / 6.8 | 83 / 75 | 0.46 / 2.43 |

With a 0.15 threshold (fuller curves), size s: street AP50 72.2 (YOLOX) vs 74.3 (YOLO26); drone 53.5 vs 50.7.

- YOLO26 draws fewer false boxes (street precision 88 to 91% against 80 to 84%) and tighter boxes (higher AP).
- It finds fewer of the small, distant vehicles in drone footage at every size (4 to 7 AP50 lower than YOLOX).
- On street photos the two are even at AP50; YOLOX l is the most accurate there.
- Neither finds much in thermal footage without training (2 to 14 AP50).

## Speed on the Mac (ms per frame, processor / Core ML)

| Size | YOLOX | YOLO26 |
|---|---|---|
| nano / n | 10.8 / 3.9 | 22.0 / 11.2 |
| s | 62.9 / 13.2 | 58.1 / 18.8 |
| m | 168.4 / 19.5 | 185.5 / 32.6 |
| l | 348.8 / 29.3 | 240.0 / 35.9 |
| x | 604.8 / 44.2 | 437.7 / 61.1 |

## Decision

YOLOX stays the default. YOLO26 is not better overall: tighter and more precise on street photos, but it misses more
small vehicles in drone footage, is slower on the Apple GPU at every size, and can only be trained with Ultralytics'
own (AGPL) tools, which the learning page does not use.
