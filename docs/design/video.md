# Video: rules

Requirements VID-01 to VID-04. Video missions use the **object search** mode; the design package offers no other mode for video.

## Time

- Every time inside a recording is the frame's **presentation timestamp** (PTS), converted to seconds from the stream start. Frames are never counted, so recordings with a variable frame rate keep correct times. `trafficwb.video.probe` records whether the rate is variable (frame intervals of the first 300 packets differ by more than 10%).
- The **absolute UTC start** comes from a confirmed source: the file name (the same patterns as still images) or an operator's explicit correction. The container's creation time is stored but not trusted.
  - When a start is confirmed, every event gets a UTC start and end: the start plus the PTS.
  - When the name has no time, the recording is still analysed. The operator can set an ISO-8601 UTC start and its minute/second precision on the Recordings screen. The change is append-only and stores the local OS username, timestamp and note. Clearing the correction restores the original filename-derived value, if any.
  - Without a confirmed start, events have relative times only, and absolute time shows as "unknown" on screen and in exports. Search time limits cannot apply to those events, and the search summary counts them.
  - The camera's real time source is still open (Q03). A sample video will show whether names carry the start.
- Time precision follows the name: a name with minutes gives events at minute precision plus the PTS offset, and the export says so.
- A manual start-time correction changes UTC metadata only. It does not shift, re-encode or alter video frames, event-relative PTS, or source file metadata.

## Sampling and resumable processing

- Analysis looks at frames on a fixed time grid, by default every 0.2 s (5 analysed frames per second of video). For each grid point, the first frame at or after it is used. When frames are sparser than the grid, the grid skips ahead; it never analyses the same frame twice.
- Processing is committed in chunks of 50 analysed frames. Each commit stores the observations, closed events, and a checkpoint: the last analysed time, the next grid point, the full tracker state and the previous scene signature.
- Pause and cancel happen only between committed chunks. A cancelled recording returns to the queue with its checkpoint. Resuming seeks to the keyframe before the checkpoint, skips frames already analysed, and continues the same grid, so the result is identical to an uninterrupted run (tested).
- Memory does not grow with recording length: only one frame, the open tracks and the current chunk are held.
- A recording that cannot be decoded is marked failed with the decoder's reason. Other recordings continue.

## Detectors and targets

Video uses the same detector choice, model choice and words as still images (`JobManager.detector_for`, one code path for both):

- **Standard**: YOLOX-s, the promoted model or a saved model, on every analysed frame.
- **SAM 3.1**: the mission's words, on a grid spaced to the SAM 3.1 interval (default 1 s, mission setting 0.2 to 10 s).
- **Both**: the standard detector on every analysed frame and SAM 3.1 on every Nth one (N = SAM interval ÷ 0.2 s), merged as for still images. Which analysed frames get SAM 3.1 depends only on how many frames were analysed before, so a resumed run calls it on the same frames.
- **Targets**: the five vehicle classes plus up to eight classes of the mission's own (people, tractors, animals). A class of the mission's own whose name COCO knows (person, or the aliases pedestrian, people) is also found by the standard detector; any other is found by SAM 3.1 only, and the overview says so. A vehicle class given no words is not looked for by either detector.
- Every tracked box records which detector found it (`track_observation.found_by`: standard, sam3 or both); the event screen shows the totals.

## Tracking (`trafficwb.tracker`)

- Detection runs on each analysed frame with the mission's detector (above). In video the standard detector reports boxes down to score 0.10.
- **Kalman and ByteTrack**: each track's box is predicted by a Kalman filter (centre, aspect, height and their velocities, per analysed frame, as in SORT/ByteTrack/BoT-SORT). Boxes scoring 0.30 or more are matched first, by overlap (IoU ≥ 0.25) with the predicted boxes, and may start new tracks. Weaker boxes are matched next, only to tracks still unmatched and only at IoU ≥ 0.5; they keep a track alive through partial occlusion and never start one. Checkpoints written before this tracker resume with the earlier constant-velocity tracker, so a run never changes method half way.
- **Moving cameras** (mission setting "Camera: moving"): on every analysed frame the camera's own motion is measured on a 480-pixel-wide grey copy (corners tracked with pyramidal Lucas-Kanade optical flow, tracked objects masked out, RANSAC fit of rotation, scale and shift). Every track, its Kalman state and recently closed tracks are moved by it before matching, so only the objects' own motion is predicted. A pan or zoom is never a cut; a textured frame that matches the previous one nowhere is. The previous grey frame is kept in the checkpoint, so a resumed run measures the same motion.
- Tracks are matched by overlap with each track's predicted position. Within the vehicle family the class does not take part in matching, because the detector may call the same vehicle car in one frame and truck in the next. Other classes are each a family of their own: a person is never matched to the bicycle they ride, and the two are never merged into one detection.
- When SAM 3.1 is used, a box that overlaps no track may still join the nearest track of its family whose predicted centre is within 3 box sizes per second since it was last seen (sizes within a factor of 2). SAM 3.1 sees a walking person about once a second, by which time the person has moved several body widths.
- A track becomes **confirmed** after 3 matched frames. Shorter tracks are kept in the database as discarded, and never become events.
- **Occlusion**: an unmatched track keeps its predicted motion for up to 1.5 s. If it is matched again within that time it is the same track.
- **Re-entry**: a new track that starts within 3 s of a closed track, near where that track was heading, is flagged **possible duplicate** of it. The tracker never merges them; the operator decides.
- **Scene cut**: on a fixed camera, when most of the picture changes at once (mean grey difference over 45, or more than 60% of pixels changed strongly), every open track is closed with "cut by a scene change" and a marker is stored. On a moving camera, see above. Re-entry suspicion does not cross a cut.
- **End of recording** closes every track. A track already unseen for more than 0.5 s at the frame edge is recorded as having left the frame. No identity is carried across recordings or cameras.

## Events

- One event per confirmed track, numbered E-0001, E-0002 … within each recording.
- Event fields: file and checksum, event id, first and last time (PTS and UTC), entry and exit position (relative to the frame), class by score-weighted vote with its share, thumbnail (the frame where the vehicle is largest), why the track ended, flags and related events, review status, clips.
- Flags: appeared inside the frame (not at an edge, after the start), lost inside the frame, possible duplicate, class uncertain (vote share under 70%), cut by a scene change, cut by the end of the recording.

## Unique counts

- Each target counts once, however many frames it was seen in: one per confirmed track, never a sum over frames.
- **Machine** count: one per event, by the machine class. **After review**: rejected, duplicate and merged events are left out, a split event counts once per part, and a re-classed event counts in its new class.
- **Maybe the same**: of the counted events, those flagged as a possible duplicate (re-entry) that nobody has reviewed. The true count lies between the count minus these and the count.
- Only the newest analysis of each recording counts. Classes the mission does not count as vehicles are counted and shown, and left out of the vehicle total.
- Shown on the Events tab (per class, per recording, and new targets per minute) and exported as `counts` and `counts_per_minute` sheets beside `events`.
- A target that leaves and comes back later (beyond the re-entry window) counts twice. Re-identification by appearance is planned.

## Parallel segments (`trafficwb.video_segments`)

- A mission chooses how many pieces of a recording are analysed side by side (1 to 32; 1 by default). With more than one, a recording longer than 1.25 segments is cut at multiples of 5 minutes on the time grid; a short tail joins the last segment.
- Each segment owns the frames in [start, end) and also analyses 3 seconds before and after (at least one SAM 3.1 interval, so a target only SAM 3.1 finds is seen twice on each side of the cut). Segments share the loaded detector and each has its own tracker.
- SAM 3.1 frames come from the whole recording's grid, so a segment calls SAM 3.1 on the frames a single run would.
- Joining: a track still running at a cut and a track on the right side that saw the same frames in the overlap are one target when their boxes agree (IoU 0.5 or more on most shared frames, two at least) and their classes are of one family; best agreement first. Each frame's box comes from the segment that owns it, so nothing is counted twice.
- Each finished segment is stored; a cancelled run resumes with the segments still to do. Tracks, observations and events are written once all segments are done, in one transaction.
- A join is not guaranteed to equal a single run's track (the trackers' histories differ at the cut); targets and counts are tested to agree.

## Review

Machine events are never changed. Each operator action is an appended revision, and the event shown is the machine event with its revisions applied:

| Action | Effect |
|---|---|
| confirm, reject, reopen | Review status |
| duplicate of E-nnnn | Marks this event as a duplicate of another event in the same recording |
| set start / end | New times, snapped to the next real frame time |
| set class | Operator class; the machine class stays in the export |
| merge into | This event joins another event of the same recording; the target's interval grows to cover both |
| split at | The event is exported as parts, one row per part; the split time snaps to a frame |
| set box / clear box | The operator's box on one frame replaces the machine's box on that frame (and extends the event's interval if outside it); clear box takes it back |

A target the analysis missed is added as an operator event (key H-nnnn, its track marked `human`), confirmed, with the box drawn on one frame. Boxes drawn on other frames extend it; between two of them its box is interpolated. It counts like any other event. Only the newest analysis of a recording is listed and counted.

For learning, a box the operator drew is a reviewed label and its frame is always taken as a training frame. Between two drawn boxes of an operator event, the interpolated box is ignored (neither label nor background) on an approved recording.

## Clips

- The operator enters the seconds of context before and after the event each time. There is no preset (Q15).
- **Copy**: packets are copied without re-encoding. It is fast and keeps the original quality, but must start at a keyframe, so the clip may start earlier than requested. The actual start is stored and shown.
- **Re-encode**: frame-exact start and end, written as H.264 (or MPEG-4 Part 2 if H.264 is not available in the installed PyAV build).
- Clips are written as `.partial` and renamed, never overwrite an existing file, and are recorded with requested and actual times, method and encoder.

## Search

- A confirmed search runs over events instead of images. Each event is one object: its class, and its colour read from the thumbnail frame with the same colour reader as still images.
- Rejected, duplicate and merged events are left out and counted.
- Time limits and light conditions are judged at the event start. Analysis regions are not applied to video in this version, and the summary says so.

## Catalog

- The mission's overview lists every video of the source folders with a thumbnail, length, picture size, start and status. Opening it scans for new files (nothing is analysed).
- Videos not analysed yet are ticked; "Analyse N videos" runs one job over exactly the ticked ones. Ticking an analysed video analyses it again; its newest analysis is the one that counts.
- Each analysed video has its own results tab on the Events tab (`#/m/<id>/events/rec-<video id>`), next to "All videos".

## Merged recordings (parts of one recording)

- A camera that splits its files, or a recording cut up before, gives several videos of one recording. The catalog suggests them: in one folder, each video starting where the one before ends (within 2 s, from the file-name start and the length), or names that differ only by a part number (`_001`, `-2`, ` (3)`, `part4`, GoPro chapters `GH01xxxx`, `GH02xxxx`). "Merge into one" confirms a suggestion; ticking videos and "Merge the N ticked videos" merges any. "Separate" undoes it; analyses stay.
- The parts are ordered by start time, else part number, else name. Ticking one part analyses all of its parts, in order, in one job.
- Across each cut, an event that ends within 1 s of the end of a part and one that starts within 1 s of the start of the next part, of the same family, with boxes that overlap or nearly touch, are one target: the later event "continues" the earlier one and is not counted again (`video_groups.links`). Each event continues at most one and is continued by at most one.
- A merged recording has one results tab and one count; its events list says which file each event is in.

## New videos

"New videos in the input folder" (Learning tab, Run mode card): analyse them as they arrive, with the mission's model (the saved model chosen for the mission, else the one in production use), or wait until they are ticked in the catalog. Full auto always analyses them as they arrive, so a model it adopts is used for the next videos with no step from the operator. Videos already waiting are analysed when the setting is turned on. The watch resumes after a restart.

## Player

- `#/m/<id>/events/play-<video id>` plays a whole recording with every event's box drawn over it: the box on the nearest analysed frame, blended between frames across short gaps. Line style and a letter say what a box is (M machine, H added or redrawn by the operator; dashed when confirmed, dotted when rejected). A timeline under the video shows each event as a bar; clicking it picks the event and goes to it.
- Reviewing there is the still-image flow: click a box to pick its event; C confirm and X reject (then on to the next event not reviewed), 1–9 class, J/K next or previous event to review, drag on the paused picture to redraw the picked event's box on that frame, B then drag to add a missed target, A approve the recording for learning (confirming the events left), Space play or pause, ←/→ one frame.
- The browser plays the original file (HTTP range requests). The event page draws the tracked box of its event over the video at the nearest analysed frame.
- If the browser cannot play the codec (for example H.265 in some browsers), the page switches to frames decoded on the workstation, one at a time, with a slider and step buttons.

## What targets did and looked like

- Every event shows where it went in the picture (right, down-right, down, down-left, left, up-left, up, up-right, or "stayed" when it moved less than its own size), its speed (median and top), its time in view, its time stopped, its size, and its colour; a person's top and lower clothes are read apart (`video_attributes.py`). The events list filters on each with dropdowns; the export has them as columns.
- Motion is measured on the bottom middle of the box, with the operator's boxes in place of the tracker's. Speed is the median over steps of at least 0.5 s; "top" is the 90th percentile. Stopped time adds the steps slower than a tenth of the target's height per second, or 1 km/h once the camera is measured.
- Measuring the camera (player, "Measure" or M): drag a line between two points on the road whose distance is known and pick it from a list (a car's length, a lane's width, a bus's length, 1 to 100 m). Speeds become km/h and sizes metres, for that video and the other videos from the same folder without a measure of their own. One scale for the whole picture is approximate: the line should be at the depth where targets pass. Without a measure, speed is in pixels per second and the speed filter offers the slowest, middle and fastest third.
- Colour is read once per target, in the background, on the frame shown as its picture, by the still-image colour reader (9 colours; observed, uncertain with the competing colours, or not observable, for example in infrared). A person's top is 15-50 % of the box height and the lower body 50-90 %.
- "Most in view at once": per family (vehicles, people), the most counted targets in view at the same moment and when. The crowd alert (5 to 100 people, or off) lists the periods with that many people in view, with links to play them.
- Not yet: hats, bags and other worn items (they need a model of their own), ground speed from a moving or airborne camera (phase 7), lines and areas in video (phase 6b).

## Lines, areas and paths

- In the player, "Lines and areas" draws a counting line (drag; L) or an area (click each corner, then the first again or Enter), named from a list. They belong to the camera: every video in the same folder uses them. The side panel lists them, with Remove (`video_zones.py`, table `video_zone`, migration 21).
- A target crosses a line when the bottom middle of its box passes from one side to the other within the line's length. "In" is to the right of the line as drawn (an arrow shows it). A crossing back within 1 s cancels the one before, and a crossing counts only once the target is a fifth of its height clear of the line, so a target standing on the line is not counted.
- A target is in an area while that point is inside; entries shorter than 0.5 s are ignored. Its path is the order of the lines it crossed and areas it entered.
- The events page shows "Lines and areas": each line's crossings in and out, and each area's entries and average time inside, by class (counted events only). The filters add "Crossed X (in, out)", "Was in X" and "From X to Y"; the events list has a Path column. The export adds a path column and a `lines_and_areas` sheet.

## Workers on other computers

- Other computers can analyse segments of a long recording. The operator turns "Other computers may help" on (Workers screen). Off by default; while off, nothing is shared and workers are refused.
- A helping computer has the same LanePulse install; nothing else is installed. Its launcher is "LanePulse - help another computer" (Windows Start menu), "Help another computer" in `~/LanePulse` (Mac), `lanepulse worker` (Linux server) or `trafficwb-worker` (pip).
- With no address given, it finds the server: a server started with network access answers on UDP port 8766 with its id, name and port (`discovery.py`). Several servers: it asks which one. It then asks to help with its own key, made once and kept in `~/.lanepulse-worker`; the server keeps only the key's hash. The computer appears under "Asking to help"; the operator answers Allow or Refuse from a dropdown. An allowed computer is listed with the pairing codes and stopped the same way; after a stop it asks again. At most 20 computers wait for an answer at once.
- After it is allowed, it offers to help by itself whenever the computer signs in (Windows Startup folder, Mac LaunchAgent, Linux systemd user service; `--start-at-signin off` undoes it). If the server's address changes, it finds it again.
- Across separate networks or a VPN, where it cannot be found, "Make a pairing code" gives a code (shown once, kept as a hash) and a ready-made command for the chosen system.
- A worker takes one segment at a time, reads only that time range of the recording from the server (HTTP range requests), runs the mission's exact detector (the model checked by its checksum, fetched once if missing), and posts back the tracks. It keeps nothing of the recording. It renews its lease every 20 s; a lease not renewed for 120 s ends and the segment goes back to the front of the queue. A late result is ignored. A worker without SAM 3.1 is not given segments of missions that use it.

## Not in this version

- Night light tracking and analysis regions for video.
- Appearance re-identification: see the phased plan. Learning from video: see `learning.md`.
- Moving cameras: lines and areas drawn on a moving camera, and speed in km/h.
- Plates in video.
- Detecting slow camera movement (pan/zoom) as a view change.
