# Learning: rules (part 1)

Requirements LRN-01 to LRN-04. Part 1 builds everything around training except the training itself: which images can teach a model, frozen dataset versions, evaluation of any installed model on them, and the operator's promotion of a model to production use. Nothing trains, and nothing is promoted, on its own.

## Which images are used

| Review state | Used as | Why |
|---|---|---|
| Approved one by one | **Full review.** Every vehicle the operator left in place is a label, and the rest of the road inside the included regions counts as "no vehicle" | Approving says the image is complete. The Approve button on the investigation screen says so |
| Corrected (moved, re-classed, added, confirmed night candidate) but not approved | **Partial review.** Only the corrected objects are labels; everything else is ignored, never treated as background | The operator may have stopped half way |
| Bulk approved | Never used | Recorded as not reviewed by a person |
| Not reviewed | Never used | Machine output is not a reference |

Unresolved night candidates in a fully reviewed image become "ignore" areas: a detection there is neither right nor wrong. Excluded regions are left out of scoring. Images whose bytes are identical are one sample, kept with the first copy.

## Dataset versions

- Freezing a version stores the list of images, their labels at that moment, and their split. Later corrections do not change it; a new version picks them up.
- Images are grouped by camera view and **date block** (local date in the mission's zone, 1 day by default). A whole group goes to train, validation or test, chosen by a hash of the group and a seed, so the same inputs always give the same split.
- A group that was in a test set in any earlier version stays in test in every later version.
- **Leakage check**: a group in two splits, identical bytes twice, or an earlier test group now used for training or validation. A version with leakage cannot be exported.
- Export writes the images and one COCO-style annotation file per split, plus a manifest with the rules, under `<output folder>/datasets/`. It never overwrites; it writes to a `.partial` folder and renames it at the end.

## Evaluation

- Runs any installed vehicle detector pack (or the one in production use) on one split of a dataset version, as a background job that can be cancelled.
- A detection matches a label at IoU 0.5 or more (configurable), greedily by score.
- Reported overall, by light condition (day, dusk, night) and by camera view: precision, recall, F1, AP at IoU 0.5, count error per image (mean absolute and bias) and class agreement.
- Precision, AP and count error use fully reviewed images only. Partial reviews add recall for their corrected objects.
- AP cannot see boxes below the detector's own score threshold.
- A failed or cancelled evaluation keeps its error.

## Production model

- The vehicle detector in production use is the last model an administrator **promoted**, or the built-in YOLOX-s when none was.
- Promotion needs a completed evaluation of exactly that model name and version. The model pack's checksum is verified and recorded.
- **Roll back** returns to the model in use before the last promotion; rolling back again goes one step further, down to the built-in default.
- Every promotion and rollback is kept with who, when and the note. Images already processed keep the results of the model that processed them.

## Not in this version

- Training. Each algorithm lists its planned adapter on the Learning tab. Train on a GPU machine with the exported dataset, bring the result back as a model pack, evaluate it here, and promote it if it is better.
- Datasets across missions (a reuse policy is needed first).
- Plate and colour models in the dataset and evaluation flow (plates have their own country evaluation, see `plates.md`).
- Video.
