# Video: rules

Requirements VID-01 to VID-04. Video missions use the **object search** mode; the design package offers no other mode for video.

## Time

- Every time inside a recording is the frame's **presentation timestamp** (PTS), converted to seconds from the stream start. Frames are never counted, so recordings with a variable frame rate keep correct times. `trafficwb.video.probe` records whether the rate is variable (frame intervals of the first 300 packets differ by more than 10%).
- The **absolute UTC start** comes only from the file name (the same patterns as still images). The container's creation time is stored but not trusted.
  - When the name gives the start, every event gets a UTC start and end: the start plus the PTS.
  - When the name has no time, the recording is still analysed. Its events have relative times only, and the absolute time shows as "unknown" on screen and in exports. Search time limits cannot apply to those events, and the search summary counts them.
  - The camera's real time source is still open (Q03). A sample video will show whether names carry the start.
- Time precision follows the name: a name with minutes gives events at minute precision plus the PTS offset, and the export says so.

## Sampling and resumable processing

- Analysis looks at frames on a fixed time grid, by default every 0.2 s (5 analysed frames per second of video). For each grid point, the first frame at or after it is used. When frames are sparser than the grid, the grid skips ahead; it never analyses the same frame twice.
- Processing is committed in chunks of 50 analysed frames. Each commit stores the observations, closed events, and a checkpoint: the last analysed time, the next grid point, the full tracker state and the previous scene signature.
- Pause and cancel happen only between committed chunks. A cancelled recording returns to the queue with its checkpoint. Resuming seeks to the keyframe before the checkpoint, skips frames already analysed, and continues the same grid, so the result is identical to an uninterrupted run (tested).
- Memory does not grow with recording length: only one frame, the open tracks and the current chunk are held.
- A recording that cannot be decoded is marked failed with the decoder's reason. Other recordings continue.

## Tracking (`trafficwb.tracker`)

- Vehicle detection runs on each analysed frame with the same model as still images.
- Tracks are matched by overlap (IoU ≥ 0.25) with each track's predicted position (constant velocity). The class does not take part in matching, because the detector may call the same vehicle car in one frame and truck in the next.
- A track becomes **confirmed** after 3 matched frames. Shorter tracks are kept in the database as discarded, and never become events.
- **Occlusion**: an unmatched track keeps its predicted motion for up to 1.5 s. If it is matched again within that time it is the same track.
- **Re-entry**: a new track that starts within 3 s of a closed track, near where that track was heading, is flagged **possible duplicate** of it. The tracker never merges them; the operator decides.
- **Scene cut**: when most of the picture changes at once (mean grey difference over 45, or more than 60% of pixels changed strongly), every open track is closed with "cut by a scene change" and a marker is stored. Re-entry suspicion does not cross a cut. A slow pan or zoom of a PTZ camera is not detected as a cut in this version.
- **End of recording** closes every track. A track already unseen for more than 0.5 s at the frame edge is recorded as having left the frame. No identity is carried across recordings or cameras.

## Events

- One event per confirmed track, numbered E-0001, E-0002 … within each recording.
- Event fields: file and checksum, event id, first and last time (PTS and UTC), entry and exit position (relative to the frame), class by score-weighted vote with its share, thumbnail (the frame where the vehicle is largest), why the track ended, flags and related events, review status, clips.
- Flags: appeared inside the frame (not at an edge, after the start), lost inside the frame, possible duplicate, class uncertain (vote share under 70%), cut by a scene change, cut by the end of the recording.

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

## Clips

- The operator enters the seconds of context before and after the event each time. There is no preset (Q15).
- **Copy**: packets are copied without re-encoding. It is fast and keeps the original quality, but must start at a keyframe, so the clip may start earlier than requested. The actual start is stored and shown.
- **Re-encode**: frame-exact start and end, written as H.264 (or MPEG-4 Part 2 if H.264 is not available in the installed PyAV build).
- Clips are written as `.partial` and renamed, never overwrite an existing file, and are recorded with requested and actual times, method and encoder.

## Search

- A confirmed search runs over events instead of images. Each event is one object: its class, and its colour read from the thumbnail frame with the same colour reader as still images.
- Rejected, duplicate and merged events are left out and counted.
- Time limits and light conditions are judged at the event start. Analysis regions are not applied to video in this version, and the summary says so.

## Player

- The browser plays the original file (HTTP range requests). The tracked box of the selected event is drawn over the video at the nearest analysed frame.
- If the browser cannot play the codec (for example H.265 in some browsers), the page switches to frames decoded on the workstation, one at a time, with a slider and step buttons.

## Not in this version

- Night light tracking and analysis regions for video.
- Plates in video.
- Detecting slow camera movement (pan/zoom) as a view change.
