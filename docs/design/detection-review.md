# Detection, review and export (still images, general traffic)

## Processing

1. A run lists the source versions it covers (Phase 1).
2. Before detecting, new images are grouped into camera views (Phase 2).
3. Each image is read, detected and its view's regions applied. Detections and the item's `done` state are written in one transaction, so an interrupted run resumes without losing or duplicating anything.
4. A failing image is marked `failed` with its error, the operator is notified, and the run continues. After 3 attempts an item stays failed.
5. The run records the model (name, version, SHA-256), every preprocessing step (letterbox size, tiles, thresholds) and the region membership rule.

Region membership per detection: `1` inside, `0` outside, empty when the view needs review (regions not applied). Images with no view use the whole frame.

## Machine output and human decisions

- `detection` rows are never changed or deleted.
- Each correction (remove, restore, reclass, move, add) appends an `annotation_revision`. The latest revision per object is current. History stays.
- Moving a box or adding one recomputes region membership with the run's rule.
- Editing an image that was approved reopens it, so an approval always refers to what the operator last saw.

## Counts

"Vehicles in image" counts current, present objects inside the regions. It is a per-image count, never a count of unique vehicles or traffic flow (IMG-01). The candidate count stays empty until night candidates exist (Phase 4).

## Review status

Kept apart from processing status: `unreviewed`, `approved` (a person looked at it), `bulk_approved_unreviewed` ("Approve all").

"Approve all" has two steps. First the preview shows scope and count and fixes the list of images. Then the commit approves exactly that list. Images that arrive or finish processing afterwards are not included. A bulk approval is never treated as a human review or as ground truth.

## Results table

Filename, capture time (UTC and local), view, vehicles in image, candidate count, processing state, review status, flags (`no_view`, `view_needs_review`, `regions_not_applied`). The table can be searched by filename. It can be filtered by view, status, local date range and minimum count, sorted, and paged. The investigation screen gets index-of-total plus previous and next for the same filter and sort.

The table is computed on request. That is fine for thousands of images. Larger archives will need stored summaries (planned with the web screens).

## Export

- **Formats:** CSV (UTF-8 with BOM, all fields quoted) and/or Excel, as configured in the mission.
- **Tables:** one table per image and one per object. Removed objects are included with `final_status=removed` so false detections stay traceable.
- **Columns:** machine class, score and box, final class and box, the correction type and the approval type, each kept in its own column.
- **Identifiers:** mission, run and source-version ids, SHA-256, capture UTC with precision, and local time with UTC offset and zone.
- **Text safety:** all values are written as text, so leading zeros and Unicode survive.
- **No overwriting:** each export gets a new name. Files are written as `.partial` and renamed when complete.
- **Excel row limit:** rows past Excel's limit continue on `images_2`, `objects_2`, and so on. Rows are never dropped.
- **Records:** every export, including a failed one, is recorded in the mission database, and failures notify the operator.
