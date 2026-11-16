# Low resolution and low frame rates

Date: 16 Nov 2026. Linux cloud container, 4 CPU cores. `scripts/benchmark.py` with `--height` (each picture shrunk to
that height, as a lower-resolution camera records it; boxes scored in the original pixels) and `--source-fps` (a
camera recording that many frames a second: every frame analysed, further apart). Same image and video samples as the
other benchmarks. Raw results: `results/*-{480,360,240}p-*.json`, `results/track-*-{2,1}fps*.json`.

## Vehicles found (AP50, vehicle any class)

| Detector | Street, original (~480p) | 360p | 240p | Drone, original (~1080p) | 480p | 360p |
|---|---|---|---|---|---|---|
| YOLOX s (default) | 67.1 | 65.5 | 63.2 | 48.3 | 28.5 | 28.6 |
| YOLOX m | 71.7 | 71.2 | 67.7 | 52.1 | 33.6 | 33.5 |
| YOLO26 s | 67.0 | 64.8 | 61.6 | 42.2 | 27.8 | 27.8 |
| RF-DETR small | 74.6 | 72.7 | 70.7 | 59.6 | 41.3 | 40.4 |

- Street-camera views lose little: 2 to 4 points at 360p and 240p.
- Distant, small vehicles (drone views) lose 20 points. Most of that is the app's tiling switching off: below 1080p
  the 3x2 tiles are under 360 px and are skipped, so the whole frame is all the detector sees.
- Letting tiles go down to 160 px gives YOLOX s back almost everything: drone 480p 28.5 → 48.0, 360p 28.6 → 44.1, at
  about 6 times the time per frame (0.08 → 0.53 s). On street photos it adds 3.4 points (67.1 → 70.5) for the same cost.

## Tracking (MOT17, 7 videos, people; HOTA / IDF1, counted vs really there)

| Detector | Original (5 analysed/s, rerun today) | 480p | 360p | 2 frames/s | 1 frame/s |
|---|---|---|---|---|---|
| YOLOX s | 47.2 / 57.5, 387 of 331 | 45.4 / 55.0, 236 | 45.3 / 55.0, 235 | 43.5 / 51.3, 202 of 325 | 36.5 / 38.0, 77 of 317 |
| YOLO26 s | 48.6 / 57.8, 323 | 44.7 / 52.6, 206 | 44.6 / 52.9, 209 | 44.1 / 49.9, 166 | 35.5 / 35.7, 74 |
| RF-DETR small | 49.2 / 57.1, 425 | 45.3 / 54.1, 279 | 46.6 / 55.7, 275 | 46.9 / 53.1, 216 | 38.2 / 39.3, 96 |

- Lower resolution costs about 2 HOTA and undercounts (about 70% of the people).
- Low frame rates undercounted badly: a target had to be seen on 5 frames, which is 1 s at 5 analysed frames a second
  but 5 s at 1 frame a second. With the tracker stepping by the real frame time and asking for the same time in sight
  (at least 2 frames), YOLOX s counts 333 of 325 at 2 frames/s (was 202) and 369 of 317 at 1 frame/s (was 77), and
  HOTA at 1 frame/s goes from 36.5 to 39.6. This is now what the app does (`VideoProcessor._tracker_for`).

## Speed

Shrinking the picture does not make YOLOX, YOLO26 or RF-DETR faster on street-sized frames (they always see 640 px).
On 1080p drone frames, 480p is 4 to 5 times faster, because the tiles are skipped.

The "original" tracking and RF-DETR figures here are today's reruns (`*-rerun-20261116.json`), made with the current
tracker so every column compares like with like; the earlier files are kept as they were.
