# RF-DETR compared with YOLOX-s and SAM 3.1

Date: 13 Nov 2026. Machine: Linux cloud container, 4 CPU cores, no GPU. Script: `scripts/benchmark.py` (as in
[README.md](README.md)), with `--standard-pack rf-detr-nano|rf-detr-small|rf-detr-medium`. Same image samples as the
YOLOX-s and SAM 3.1 runs (same seed, same image ids). Raw results: `results/*rf-detr*.json`.

RF-DETR (Roboflow, Apache-2.0 code and Nano/Small/Medium weights, DINOv2 backbone Apache-2.0) runs in LanePulse as an
ONNX model pack through `detect.DetrDetector`, with the same tiling, 0.30 threshold and class merge as YOLOX-s. Its
weights come from Roboflow's own storage (checked against the MD5 the rfdetr package publishes); nothing comes from
Hugging Face, and it runs and trains with no network.

## Released models, no training

Vehicle AP50 (vehicle, any class), and seconds per image as LanePulse runs each detector.

| Detector | Street (COCO, 100) | Drone daylight (VisDrone, 60) | Drone thermal (HIT-UAV, 100) | s per image (street / drone / thermal) |
|---|---|---|---|---|
| YOLOX-s (standard) | 67.1 | 48.3 | 5.7 | 0.10 / 0.55 / 0.09 |
| RF-DETR Nano | 72.3 | 55.3 | 21.9 | 0.14 / 0.75 / 0.14 |
| **RF-DETR Small** (shipped) | 74.6 | 59.6 | 20.9 | 0.25 / 1.34 / 0.31 |
| RF-DETR Medium | 78.0 | 60.6 | 19.1 | 0.40 / 1.96 / 0.33 |
| SAM 3.1 | 83.1 | 72.3 | 55.9 | 10.6 / 11.8 / 11.6 |
| Both (YOLOX-s + SAM 3.1) | 83.0 | 72.8 | 55.9 | 10.8 / 12.4 / 12.6 |

Vehicle AP (0.5 to 0.95): YOLOX-s 44.1 / 30.6 / 3.4; RF-DETR Small 52.5 / 39.6 / 11.3. Mean AP50 over the 6 classes on
VisDrone: YOLOX-s 23.2, RF-DETR Small 38.3. At the app's 0.30 threshold, RF-DETR Small finds 77% of street vehicles
(YOLOX-s 69%) with 74% precision (YOLOX-s 80%).

- RF-DETR beats YOLOX-s on every set, most on drone footage (+11 AP50) and street photos (+7.5).
- It is 1.5 (Nano) to 4 (Medium) times slower than YOLOX-s on a CPU, and 35 to 80 times faster than SAM 3.1.
- Without training, RF-DETR is still weak in thermal (21 AP50); SAM 3.1 stays the strongest untrained detector there.
- Small is shipped: Medium adds 1 to 3 AP50 for 1.5 times the time; Nano is the choice for slow computers.
- YOLOX-s timings are from this session's rerun (0.07 / 0.43 / 0.09 s in the first benchmark run); accuracy is identical.

## After training (HIT-UAV thermal)

Both detectors trained on the learning page's code path (`trafficwb-train`, the same as the Train button) with 150
HIT-UAV training images and 30 validation images, 6 epochs, CPU only; tested on the same 100 test images as above.

| Detector | Vehicle AP50 | Car AP50 | Person AP50 | Bicycle AP50 | Day / night vehicle AP50 | Training time (4 CPU cores) |
|---|---|---|---|---|---|---|
| YOLOX-s, released | 5.7 | 5.7 | 1.4 | 0.0 | 11.8 / 2.7 | - |
| YOLOX-s, trained | 68.4 | 74.2 | 57.5 | 48.9 | 70.9 / 66.7 | 11 min |
| RF-DETR Nano, trained | **76.5** | 87.2 | 43.3 | 51.5 | 83.5 / 72.9 | 45 min |
| SAM 3.1, untrained | 55.9 | 78.1 | 18.4 | 0.0 | 79.1 / 43.8 | - |

- With 150 labelled images, both fast detectors overtake SAM 3.1 in thermal, at 1% of its time per image.
- RF-DETR finds more vehicles (+8 AP50 over YOLOX-s); YOLOX-s found people better in this run (57.5 vs 43.3 person AP50).
- RF-DETR trains about 4 times slower on a CPU; on a GPU both take minutes.

## Video: people tracking (MOT17, 7 videos, 5 frames per second)

| Detector | HOTA | IDF1 | MOTA | People counted / really there |
|---|---|---|---|---|
| YOLOX-s | 45.2 | 53.6 | 26.4 | 609 / 331 |
| RF-DETR Small | 46.1 | 51.7 | 18.1 | 756 / 331 |

Tracking is about even (HOTA +0.9, IDF1 -1.9). RF-DETR gives more low-confidence boxes above the 0.30 threshold, which
start extra short tracks, so it counts more people too many times. Over-counting in video is a tracker issue that
affects both detectors (see README.md); a higher threshold for RF-DETR in video is worth trying before choosing it for
counting. (Seconds per frame were not comparable in this run: training ran at the same time.)

## Not measured

- Vehicle tracking and counts (UA-DETRAC could not be downloaded here).
- RF-DETR Small trained in thermal (Nano was trained to keep the CPU run under an hour), and training on daylight drone
  footage.
- GPU speeds.
