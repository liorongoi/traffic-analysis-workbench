# Decision record: first plate finder and reader

Date: 24 Oct 2026. Status: **evaluation only**, chosen by Lior on 24 Oct 2026 ("use open plate models as a starting point until plate images arrive"). Every read they produce is labelled evaluation only until a country passes its own test set.

## Chosen

| | Plate finder | Plate reader |
|---|---|---|
| Model | YOLOv9-t, 640 px, end-to-end (NMS inside), from open-image-models | CCT-S v2 "global" fixed-slot reader, from fast-plate-ocr |
| File | `yolo-v9-t-640-license-plates-end2end.onnx`, SHA-256 `c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076` | `cct_s_v2_global.onnx`, SHA-256 `384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b` |
| Repository license | MIT | MIT |
| Training data | Not documented by the author | The author's global plate set, not documented |
| Size | 7.8 MB | 5.3 MB |
| Alphabet | – | `0-9 A-Z`, 10 slots, plus a 66-way country guess |

## Why

- They are the only offline plate models found with a permissive repository license and ready ONNX files, so they run in the existing ONNX Runtime without PyTorch.
- The reader's alphabet covers Israeli plates (digits) and Latvian plates (Latin letters and digits). Its country list includes Israel and Latvia.
- Both are small. On the Finnish sample (93 images, 1,027 vehicles, 412 of them large enough to look for a plate) the plate stage took about 38 s on 4 CPU cores, on top of vehicle detection.

## Risks recorded

- **Image rights**: neither author documents the training images. Before operational use, ask the authors or replace the weights with ones trained on data you hold rights to.
- **YOLOv9 origin**: the architecture comes from a GPL-3.0 repository. Whether that affects weights exported by a third party is not settled.
- **Scripts**: the reader cannot read Persian or Arabic characters. Lebanese and Iranian plates are found but not read, and say so. A reader for them needs your labelled plates (Q13) or a dataset with clear rights (IR-LPR's image rights are still to be checked).
- **Non-plates**: on the Finnish sample the reader read a square yellow road sign as "T1" with high confidence. Boxes that are not plate-shaped (width/height under 1.8) or that the finder is unsure of (score under 0.5) are therefore never marked "full".

## Replacing them

Model packs are versioned, checksum-verified folders. A new finder or reader pack is a new folder; plates are read again for the new pair and earlier reads stay stored with the models that made them.
