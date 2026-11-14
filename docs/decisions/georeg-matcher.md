# Decision record: feature matcher for geo-registration

Date: 13 Nov 2026. Status: prototype choice, open to replacement.

## Chosen

ALIKED (n16rot, top 2000 points, 640 px) with LightGlue, as ONNX, run with ONNX Runtime on the CPU; SIFT (OpenCV) when
no model pack is given.

| Item | Value |
|---|---|
| Files | `aliked-n16rot-top2k-640.onnx` (SHA-256 `d62b44a5a7b923c60f5351d641e0199bb03fbff49c82eca7de34ec87df0f08eb`), `aliked-n16rot-top1k-640.onnx` (`b46868de0794ea0f2d878985d99dc17d3639dfab85d852994254a4cdf0cdc0ea`), `lightglue_for_aliked.onnx` (`33fffedd24f39f25b139fb66f9090481d276799cef7b0ea56eb6bc0986987c38`) |
| Source of the ONNX files | huggingface.co/bukuroo/ALIKED-LightGlue-ONNX (a third-party export, marked Apache-2.0). Not yet compared against an export we make ourselves |
| Licences of the originals | ALIKED: BSD-3-Clause; LightGlue code and weights: Apache-2.0 |

## Why

- Across seasons, classic features fail: SIFT matched 0 of 158 UAV-VisLoc drone photos to the satellite map, the
  learned matcher 115 (73%).
- Licences allow shipping. SuperPoint, the usual partner of LightGlue, was ruled out: its weights are for
  non-commercial use only.
- No PyTorch at run time, like the detectors.

## Costs

One match takes 4.2 s (2000 points) or 2.0 s (1000 points, 65% matched) on 4 CPU cores. Enough for a map snap every
2 to 4 seconds beside real-time tracking. Before it goes into the installer: export the models ourselves from the
official weights and check the outputs against this export.
