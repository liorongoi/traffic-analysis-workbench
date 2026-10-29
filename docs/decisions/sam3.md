# Decision record: SAM 3.1 for detection by name and for scan areas

Date: 29 Oct 2026. Status: shipped as an option next to YOLOX-s; the standard detector stays the default.

## What it is for

Mainly still images, where time per image matters little:

- **Vehicle detector by name.** A still-image mission can choose SAM 3.1 instead of the standard detector. It is asked for
  "car", "van", "truck", "bus", "motorcycle" and "bicycle" and needs no training on the cameras.
- **Scan areas by name.** On the region editor, "road" (or any other words) becomes outlines of those areas. The operator
  picks which outlines become include or exclude regions; holes in a road outline (a traffic island) become exclusions.
  Parts of the frame outside the regions are then not searched closely (tiles with none of the scan area are skipped).

## Chosen

**SAM 3.1, 8-bit weight-compressed OpenVINO export (W8A16)** published by `rajeshgangireddy/SAM3_OpenVINO`, pinned to
commit `9e742f7509733b3cec1b1c902db562abff877b59`. Meta's own release on Hugging Face (`facebook/sam3`) is gated behind
a form, so a public, ungated export is used.

| Item | Value |
|---|---|
| Files | image encoder, text encoder and prompt decoder (`.xml` graph + `.bin` weights each), CLIP `tokenizer.json`; about 880 MB. Every file's SHA-256 is pinned in `scripts/fetch_models.py` and checked again when the pack is loaded. |
| Format safety | OpenVINO IR and JSON only: no pickle files, so loading the model cannot run code. |
| License | Meta's **SAM License** (19 Nov 2025), shipped as `LICENSE` in the pack (`packaging/licenses/SAM-LICENSE.txt`, SHA-256 `4dea99bf…ae1be`). It allows use, copying, changes and redistribution with the license included. It forbids military and warfare use, nuclear, espionage and weapons work, ITAR-controlled work, and reverse engineering. The export's author states the files are format conversions and weight compression of SAM 3.1, with no change to the architecture. |
| Runtime | OpenVINO on the CPU, on Windows (x86-64), macOS (Apple silicon) and Linux (Ubuntu, Red Hat). No GPU is needed. |
| Accuracy | The author measured W8A16 F1 equal to FP16 on CPU (COCO subsets). On a 1920x1080 test image it agreed with Meta's FP32 ONNX export within 0.007 in score and 0.002 in box position (checked on an Apple M5 Pro). |
| Speed | About 3 s per pass on an Apple M5 Pro and about 5 s on an Intel CPU (author's figure); one detector pass asks six words, and large frames add four tiles. |
| Memory | About 2.5 GB while loaded and about 7 GB at the peak of one image. The models load on first use and unload after 10 minutes unused. Computers under 12 GB get a warning in the screens. |

## Why this export and not the others

| Option | Why not |
|---|---|
| Meta `facebook/sam3` | Gated: needs an account and an accepted form, which an offline install cannot do. |
| `cubert-gmbh/sam3` (byte-identical mirror) | `sam3.pt` is a PyTorch pickle, which can run code when loaded, and needs PyTorch to run (3.45 GB). |
| `AEmotionStudio/sam3`, `1038lab/sam3` | Safe safetensors, but SAM 3 (not 3.1), full precision (3.4 GB), PyTorch needed, no checksums against Meta's file. |
| Same repository, FP32 ONNX | Runs on ONNX Runtime, but 3.4 GB and about 10.5 GB of memory per image. |
| `comicbox/sam3.1-comicapp-onnx` | No text encoder: only a fixed list of concepts, not free words. |
| `embedl/sam3` | Gated, TensorRT on NVIDIA only, and a second license. |
| EfficientSAM3 (distilled) | Promising and Apache-2.0, but research code, pickle weights, no ONNX or OpenVINO export yet. Worth another look later. |

## Things found while checking it

- **Usage reporting.** `import openvino` also imports its model converter, which sends usage events to Google Analytics
  unless a consent file says no. LanePulse hides the `openvino_telemetry` package before OpenVINO is first imported, so
  OpenVINO falls back to its own silent stand-in. A test runs SAM 3.1 with all network connections trapped and checks
  that none is opened and no consent or client-id file is written.
- **Apple silicon.** The image encoder fails on ARM CPUs when its batch size is left open ("Supported executor is not
  found" for its transposed convolutions). LanePulse fixes the input to one image (1 x 3 x 1008 x 1008) before
  compiling, and keeps OpenVINO's default precision, since forcing FP32 on ARM brings the failure back.
- **Boxes** come out as normalised x1, y1, x2, y2 (checked against the masks), scaled to the image's own width and height
  because the image is resized to 1008 x 1008 without keeping its aspect ratio.

## Vehicles found against empty-road backgrounds

At night the detector misses many vehicles that show only as headlight glare. So the operator can mark a few images of a
camera view where the road is empty (on the region editor, or "Empty road" while reviewing). LanePulse builds one
background per light (day, dusk, night) from them: the median of the images and how much each pixel varies between them.
Dusk and night stand in for each other when one is missing.

Each image is then compared with the background of its own light, after removing an overall brightness change (exposure,
wet road). What differs inside the scan area, by more than the usual variation, is a change; nearby parts merge, and the
light a vehicle throws on the road joins the lights that cast it. At night and dusk only changes with a bright light core
count (glare counts as a vehicle), unless SAM 3.1 recognises a vehicle in a close-up of the change; SAM also gives the
type and a tighter box. Changes already covered by a detected or drawn vehicle are not counted twice.

Changes are looked for only on the road: inside the view's include regions, or, when none are drawn, inside the
road SAM 3.1 outlines on the empty background itself ("road", "overpass" and "highway ramp" together; kept with the background). Street lamps and lit signs are
bright in the empty background too; their halos grow and shrink with fog, so a near-white area there is not a new
light (the zone around each lamp grows with its size). A vehicle's box is drawn around its lights; the lit road
ahead stays part of the change but not of the box.

These vehicles show as "V" boxes, are corrected like any detection, and are exported with the evidence level
`vehicle_from_change`. Each empty image is also compared with the others ("empty check"), so a background that would
produce false vehicles shows it before it is used.

## Not done yet

- Point and box prompts ("this one, and things like it"): the geometry encoders exist in the export but are not shipped.
- A measured comparison with YOLOX-s on reviewed images from your cameras.
- Video: SAM 3.1 is offered for still images only; video missions keep the standard detector.
