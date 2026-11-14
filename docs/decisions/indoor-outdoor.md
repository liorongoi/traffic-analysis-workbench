# Sorting still images into indoor and outdoor

A still-image mission mode, "Sort into indoor and outdoor" (`indoor_outdoor`): a bundle of images goes into two
groups. The operator can move any image to the other group, take a move back, and teach LanePulse from the moves.
The two groups are then copied into `indoor/` and `outdoor/` folders in the mission's exports folder, with
`sorting.csv` listing every image, its group, who decided it and what LanePulse chose. The source images are only read.

## Model

OpenAI CLIP ViT-B/32 image encoder, 8-bit (the transformers.js export by Xenova, pinned to commit `6b7c523`),
89 MB, model pack `clip-vit-b32-image`. MIT licence for the code and the released weights. It is fetched by
`scripts/fetch_models.py` with the other packs, so the Windows, Mac and Linux server installers carry it and the app
never goes online. Each image becomes a 512-number embedding, kept in the mission's database; its side comes from
comparing it with two text embeddings (the mean of five "indoors" and five "outdoors" prompts) that ship in
`src/trafficwb/scene_prompts.json` (made by `scripts/scene_prompts.py`), so the text encoder is not shipped.

Other choices looked at: a Places365 scene classifier with its indoor/outdoor table (the weights' licence is not
clear) and the full-precision CLIP encoder (352 MB, 0.7 points better on the sample below, four times the size).

The 8-bit model scales its numbers over the whole batch, so images are run one at a time; an image's side never
depends on the images processed beside it. About 30 ms per image on the CPU of the build machine, decoding included.

## Measured

1,074 photos from the SUN397 test set (up to 12 of each of 90 scene types whose indoor/outdoor label is given by
the Places365 table; 474 indoor, 600 outdoor), through the app's own code:

| | Right |
|---|---|
| Before any correction (offset chosen on half the photos, checked on the other half) | 98.1% to 98.5% |
| Indoor photos placed indoor | 98.7% |
| Outdoor photos placed outdoor | 97.7% |

Most mistakes are places that are honestly between the two: shopfronts, stadiums, courtyards, covered markets,
boat decks.

## Learning from the operator

What LanePulse learns is a memory of the images people placed, in every sorting mission on the workstation. An image
nobody checked whose nearest images (CLIP similarity at least 0.75, the five nearest) include ones a person placed
takes the side those neighbours are on, closer ones counting for more. The neighbours also include images nobody
checked, at LanePulse's own side, so one move does not pull a crowd over. Such images show "like images you placed".
A person's own decision always wins.

A new memory is checked by cross-validation on the images people placed and used only when it gets at least as many
of them right as LanePulse alone. Every version is kept (`<data>/scene_learner/`), and "Go back to sorting without
what it learned" switches it off.

Measured on the same photos, a person correcting LanePulse's mistakes in a random half and the other half checked
(ten random splits):

| | Before | After learning |
|---|---|---|
| Usual meaning of indoor and outdoor | 98.3% | 98.4% |
| The person's own rule differs (aquariums, hangars, ice rinks and stables called outdoor), overall | 94.6% | 97.5% |
| ... on those four places | 10% | 71% |

A linear correction of the prompt direction (logistic regression from the prompts) was tried first and dropped: from
corrections alone it learned the person's rule but moved too many other images, and on the usual meaning it did
slightly worse than no learning.
