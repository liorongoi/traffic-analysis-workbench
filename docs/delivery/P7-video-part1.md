# Delivery report: video, part 1

Date: 24 Oct 2026. Branch: `claude/project-thread-rld21n`.

## Implemented

- `trafficwb.video`: probing (codec, size, length, nominal and variable frame rate) and decoding by presentation timestamps on a fixed time grid, resumable from any analysed time.
- `trafficwb.tracker`: deterministic vehicle tracker with occlusion gap, re-entry flag, scene-cut and end-of-recording rules. Its whole state is saved at each checkpoint.
- `trafficwb.video_process`: processing in committed chunks with checkpoints; pause and cancel between chunks; one event per confirmed track, with flags.
- `trafficwb.video_review`: events with append-only revisions (confirm, reject, duplicate, set times, set class, merge, split, reopen), thumbnails, clip export (copy or re-encode) and the events sheet export (CSV or Excel, one row per split part).
- Object search over video events, including colour from the thumbnail frame.
- Recordings whose names have no start time are analysed; their events have no absolute UTC.
- Database migration v6 adds the tables `video_probe`, `video_checkpoint`, `track`, `track_observation`, `video_marker`, `video_event`, `event_revision` and `clip`.
- Web: video missions show Overview, Search and Events.
  - The Events tab has a recordings table, an events table with thumbnails and filters, and an export button.
  - The event page has a player with the tracked box drawn over the video, review buttons, merge, split and time setting at the playhead, and clip export with context fields that start empty.
  - If the browser cannot play the codec, the page shows frames decoded on the workstation instead.
- Rules: `docs/design/video.md`.

## Actually tested

| What | Where | Result |
|---|---|---|
| Full suite. New tests in `tests/test_video.py`: variable frame rate times against the written timestamps; resume from a mid-point gives the same frames; blips under 3 frames give no event; occlusion keeps identity and a late reappearance is flagged; tracker state round trip; one event per crossing with UTC; cancel after a chunk and resume gives identical events and observations; scene cut closes tracks; unknown start time; revisions and snapping; clip copy and re-encode times; export rows per part; API flows; video search | Linux cloud container, Python 3.13 | All passed |
| Same tests and the UI build | GitHub Actions, Windows + Linux, Python 3.11/3.12 | See the PR checks |
| Events tab, player with overlay, confirm, clip export, video search results, and the frame-by-frame fallback (forced by blocking the video file) | Chromium via Playwright, synthetic recordings | All worked, no console errors |
| Detection time per analysed 1280×720 frame with YOLOX-s on CPU | Cloud container, 4 CPU cores | About 0.96 s. At 5 analysed frames per second of video, that is about 5 times slower than real time on this CPU: one hour of video takes about 5 hours. The workstation's GPU, if ONNX Runtime can use it, should be much faster; this has not been measured. |

All video tests use **synthetic recordings**: red boxes moving across a grey background, with a detector that finds red boxes. They check timing, tracking rules, resumption, review and clips exactly. They say nothing about detection or tracking quality on real traffic.

## Not tested

- Any real camera recording. I have no sample video yet (Q03, Q14).
- Tracking quality in dense traffic, at night, or with long occlusions.
- H.265, MJPEG, or recordings with broken timestamps.
- Analysis speed on the workstation, and the effect of a different sampling interval.
