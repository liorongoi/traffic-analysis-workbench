# Sorting still images into indoor and outdoor

A still-image mission mode, "Sort into indoor and outdoor" (`indoor_outdoor`): a bundle of images goes into two
groups, by where the camera is and what it shows:

- **Indoor**: the camera is inside a place (a house, a shop, a restaurant, a mosque) and shows the inside.
- **Outdoor**: the camera is outside the building, **or** it is inside and the outside can be seen (through a window
  or a glass front). This second kind stays in the outdoor group but is marked "seen from inside".

So each image gets one of three labels: `indoor`, `window` (outdoor, seen from inside) and `outdoor`. The operator can
move any image to either of the other two, take a move back, and teach LanePulse from the moves. The groups are then
copied into `indoor/`, `outdoor/` and `outdoor/seen from inside/` in the mission's exports folder, with `sorting.csv`
listing every image, its group, whether it was seen from inside, who decided it and what LanePulse chose. The source
images are only read.

## Model

OpenAI CLIP ViT-B/32 image encoder, 8-bit (the transformers.js export by Xenova, pinned to commit `6b7c523`),
89 MB, model pack `clip-vit-b32-image`. MIT licence for the code and the released weights. It is fetched by
`scripts/fetch_models.py` with the other packs, so the Windows, Mac and Linux server installers carry it and the app
never goes online. Each image becomes a 512-number embedding, kept in the mission's database; its side comes from
comparing it with three text embeddings (each the mean of five prompts: a room, a room with the outside showing
through its windows, an outdoor scene; with a small bias per label) that ship in
`src/trafficwb/scene_prompts.json` (made by `scripts/scene_prompts.py`), so the text encoder is not shipped.

Other choices looked at: a Places365 scene classifier with its indoor/outdoor table (the weights' licence is not
clear) and the full-precision CLIP encoder (352 MB, 0.7 points better on the sample below, four times the size).

The 8-bit model scales its numbers over the whole batch, so images are run one at a time; an image's side never
depends on the images processed beside it. About 30 ms per image on the CPU of the build machine, decoding included.

## Measured

Measured on ordinary photos, not CCTV frames, so treat the figures as a guide for camera images.

**Three labels.** 216 photos labelled by hand to the rule above (135 indoor, 42 seen from inside, 39 outdoor: living
rooms, lobbies, restaurants, mosques, rooms with bow windows, balconies) together with SUN397 outdoor photos; prompt
biases chosen on half and checked on the other half:

| | Right |
|---|---|
| All three labels | 95.4% |
| Indoor placed indoor | 94.3% |
| Seen from inside marked so | 88% |
| Outdoor placed outdoor | 97% |
| Just the two groups (indoor, outdoor) | 96.4% |
| The hand-labelled photos alone, three labels | 89% |

Most mistakes are places that are honestly between the labels: balconies and covered terraces (seen from inside or
outdoor), a room with a small window (indoor or seen from inside), shopfronts and courtyards.

**Two labels, earlier version.** Before "seen from inside" was added, 1,074 SUN397 test photos (474 indoor, 600
outdoor by the Places365 table) were 98.1% to 98.5% right.

## Learning from the operator

What LanePulse learns is a memory of the images people placed, in every sorting mission on the workstation. An image
nobody checked whose nearest images (the five nearest with CLIP similarity at least 0.8) include ones a person placed
takes the label those neighbours have, closer ones counting for more; LanePulse's own label counts as a small extra
vote, so an image needs clear company to move. The neighbours also include images nobody
checked, at LanePulse's own side, so one move does not pull a crowd over. Such images show "like images you placed".
A person's own decision always wins.

A new memory is checked by cross-validation on the images people placed and used only when it gets at least as many
of them right as LanePulse alone. Every version is kept (`<data>/scene_learner/`), and "Go back to sorting without
what it learned" switches it off.

Measured on the three-label set above, a person correcting LanePulse's mistakes in a random half and the other half
checked (ten random splits):

| | Before | After learning |
|---|---|---|
| Usual rule, all three labels | 94.4% | 95.4% |
| ... seen from inside marked so | 86.7% | 82.6% |
| Usual meaning on the SUN397 two-group set | 96.1% | 96.6% |
| The person's own rule differs (some places called the other group) | 92.6% | 93.9% |

Learning trades a little "seen from inside" recall for fewer mistakes overall; the vote threshold (0.8) and
LanePulse's own vote (0.1) were chosen to keep that loss small.

A linear correction of the prompt direction (logistic regression from the prompts) was tried first and dropped: from
corrections alone it learned the person's rule but moved too many other images, and on the usual meaning it did
slightly worse than no learning.
