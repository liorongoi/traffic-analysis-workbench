# Decision record: first vehicle detector

Date: 24 Oct 2026. Status: baseline for Phase 3, open to replacement after comparison on your data.

## Chosen

**YOLOX-s**, official ONNX export from release 0.1.1rc0 of Megvii-BaseDetection/YOLOX.

| Item | Value |
|---|---|
| Weights | `yolox_s.onnx`, SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| Code and weights license | Apache-2.0 (LICENSE file in the repository, fetched into the model pack) |
| Training data | COCO 2017. Annotations CC BY 4.0; the underlying Flickr images carry their own terms. |
| Runtime | ONNX Runtime, CPU provider (CUDA and DirectML not verified yet) |
| Classes kept | bicycle, car, motorcycle, bus, truck |

## Why

- Permissive license for code and weights. Ultralytics YOLO was ruled out because it is AGPL-3.0 unless a commercial license is bought.
- An official ONNX file exists, so no PyTorch is needed for inference.
- It is small (36 MB) and runs at about 1 s per 1280x720 image with tiling on 4 CPU cores in the cloud container. That figure is not from the target workstation.

## Not yet compared

RT-DETR, D-FINE and RF-DETR (reported Apache-2.0) and larger YOLOX sizes. The comparison needs labelled images from your cameras. Without them, any accuracy claim would be invented.

## Settings

- Whole frame at 640 px, plus 3x2 overlapping tiles (20% overlap) at native resolution, so distant cars are not lost to downscaling. Tiling is skipped when tiles would be under 360 px.
- Score threshold 0.30, per-class NMS at IoU 0.45, then a cross-class merge at IoU 0.50 so one van labelled both car and truck counts once.
- All settings are recorded on each run.

## Observed on the Fintraffic sample (visual spot check, not a measurement)

- Daylight: nearly every vehicle on the near and middle carriageways is boxed; false boxes are rare. Misses are mostly very distant cars near the far bridge and cars hidden behind the railing.
- Before the cross-class merge, about 10% of boxes were the same vehicle labelled twice.
- Night (infrared, headlights): almost nothing is detected. That is expected; night handling is Phase 4.
