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

### Video (learning from recordings)

| Review state of a recording | Frames used | Labels |
|---|---|---|
| **Approved** (Events tab; needs every event confirmed or rejected) | Each reviewed event's best frame and up to 4 more spread over its track (the frame farthest from those already taken, each time), plus an analysed frame every 5 s | Full review: every kept event's box is a label; rejected detections and unmarked road are background; boxes of tracks that never became events, and an event's interpolated position on a frame it was not matched on, are ignored |
| **Some events reviewed** (confirmed, rejected, duplicate, merged or given another class) | The reviewed events' frames | Partial review: only the reviewed events' boxes; the rest of each frame is ignored |
| Not reviewed | none | none |

- Frames closer than 0.4 s are one; a recording gives at most 300 frames. A frame is identified by the recording's bytes and its time, so the same recording copied twice is one sample.
- Frames are grouped like images (camera view and date block), so the frames of one recording and day never sit on both sides of a split.
- Export writes each frame as a PNG; evaluation decodes the same frame from the recording.

### Classes

- The five vehicle classes and **person** are learnt (category 6, COCO class 0). A mission's own class called person, pedestrian, people, human, man or woman is learnt as person. Other own classes are found by SAM 3.1 only: their boxes are ignored, never labels or background.
- Evaluation scores only the classes the detector looks for: for a vehicle-only detector, people are neither found nor missed.

## Run mode

- **Manual decisions** (default): the operator freezes dataset versions, trains, compares and chooses the model, as below.
- **Full auto**: a loop checks every two minutes and after each review decision. Once a set number of new review decisions (20 by default: image approvals, object corrections, night candidate decisions, video event decisions and recording approvals, counted since the newest dataset version) has built up, and nothing else is processing or training, it freezes a dataset version, trains the model this mission uses (or the released YOLOX-s) for a set number of epochs (30) with the data of rounds that did not improve, and evaluates both on the same test images. The mission switches to the new model, saved by name, **only if it did better** (F1, then AP); otherwise it keeps its model. The production model of other missions never changes by itself.
- Every step is logged with its reason (waiting, round started, adopted, kept, skipped, failed) on the Learning tab; switching back is choosing the earlier model under Saved models.

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

## Several dataset versions together

- Training and evaluation can use several dataset versions at once, from this mission and from any other still-image mission on the computer.
- Combined, a camera-view-and-day group that is in a test set in any chosen version stays in test; otherwise it takes its split from the newest version that holds it. Identical image bytes count once, with the newest labels, and are test if any copy is.
- An evaluation records every dataset version it used (`sources`). Two evaluations are on "the same images" only when their versions, split and match rule are the same.

## Training rounds

- Each training on the learning page is a round, recorded with its dataset versions, starting model, epochs, loss per epoch, the new model and its evaluation.
- After training, the starting model is evaluated on the same test images (or an identical earlier evaluation is reused), then the new model. The round **improved** when the new model's F1 is higher (AP at IoU 0.5 breaks a tie).
- Rounds that did not improve since the last one that did hand their dataset versions on: the next training and evaluation start with that data plus the newest dataset version. The operator can change the choice.
- Loss per epoch is shown as a graph (training, validation or both) and as numbers.

## Saved models

- A model can be saved by a name of the operator's choosing, with a description. Saving freezes it: its files are made read-only, its checksum is kept in `control.db`, and it is checked each time the model is loaded; a changed model is not used.
- Any mission can choose a saved model for its standard detector (wizard, overview or learning page). New images are processed with it; images already processed keep their results.
- Removing a model from the saved list leaves its files in place.

## Production model

- The vehicle detector in production use is the last model an administrator **promoted**, or the built-in YOLOX-s when none was.
- Promotion needs a completed evaluation of exactly that model name and version. The model pack's checksum is verified and recorded.
- **Roll back** returns to the model in use before the last promotion; rolling back again goes one step further, down to the built-in default.
- Every promotion and rollback is kept with who, when and the note. Images already processed keep the results of the model that processed them.

## Not in this version

- Datasets across missions (a reuse policy is needed first).
- Plate and colour models in the dataset and evaluation flow (plates have their own country evaluation, see `plates.md`).
