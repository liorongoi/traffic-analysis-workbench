# Camera views and regions

## Terms

- **Camera location**: the physical camera (for example `C0450701`). It can have several views.
- **Camera view**: one framing (preset) of that camera. Regions, drift references and, later, background models belong to a view.

## How images are grouped

1. **Signature.** Each image is resized to 320x180, contrast-equalised (CLAHE, clip 3.0) and described by gradient-orientation histograms (10 px cells, 8 unsigned bins). That gives a 4608-number vector, stored as float16. It describes the scene's structure (road edges, poles, signs) and ignores colour.
2. **Lighting condition.** Mean HSV saturation below 5 means monochrome (infrared night mode). Otherwise colour.
3. **Clusters within one condition and resolution.** Leader clustering: an image joins a cluster when its similarity to the cluster centre is at least 0.70.
4. **Geometric join.** Two clusters in the same condition are the same view when ORB features plus a RANSAC homography give at least 40 inliers and a shift of no more than 1.25% of the image width.
5. **Colour/monochrome link.** Features rarely match across the infrared switch, so a colour cluster and a monochrome cluster are linked when each is the other's best match and their centre similarity is at least 0.60. These links are marked "needs confirmation" until the operator confirms them.
6. **Small groups.** Groups with fewer than 3 images stay unassigned for the operator to place.

Existing assignments are never overridden by a later proposal run. Reassign, rename, merge, split and confirm are recorded in `view_event` and assignment history is kept (`superseded_by`).

## Measured on the Fintraffic sample (143 images, one preset)

| Comparison | Similarity |
|---|---|
| Same view, day vs night, image to image | 0.62 to 0.71 |
| Same view, day vs night, cluster centres | 0.84 |
| Mirrored copy (stand-in for a different view), image to image | at most 0.54 |
| Mirrored copy, cluster centres | at most 0.62 |
| Same view panned by 60 px | 0.56 to 0.68 |

Result: one view, 141 images (62 colour, 79 monochrome), link marked for confirmation. The two rain-on-lens frames (03:45Z and 04:45Z) were left unassigned. Grouping took 3.4 s in the cloud container.

The 0.60 link threshold sits close to the mirrored-copy maximum (0.62), which is why the link is never automatic.

## View-change (drift) detection

- Each view keeps one reference image per lighting condition.
- A new image is compared with the reference of its own condition by feature matching.
- **shifted**: enough matches and displacement above the tolerance (12 px in original pixels, adjustable).
- **ok**: enough matches, displacement within tolerance.
- **unverified**: too few matches (rain, fog, glare). Recorded but never flags the view.
- Two consecutive verified shifts set the view to `needs_review`. While a view needs review its regions are not applied.

## Regions

- Polygons in the view's original pixel coordinates.
- Exclusion wins. With no inclusion region the whole frame is included.
- Membership rule per run: `bottom_center` (default, approximate ground contact), `centroid`, or `overlap` (share of the box inside, estimated on an 11x11 grid so results are repeatable).
- Validation: at least 3 points, inside the image, edges must not cross, non-zero area.
- Every edit is a new revision; deleting adds a deleted revision. History is kept.
