"""Mission registry and per-mission databases.

- control.db (in the data root) lists missions.
- Each mission gets its own SQLite database, created when the mission is
  committed, before any run. Missions never share analytical tables.
- Machine and human records are append-only where the design requires history.
"""
from __future__ import annotations

import contextlib

import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION = 1  # base schema; later versions are applied by MISSION_MIGRATIONS
TARGET_COUNTRIES = {"IR": "Iran", "LB": "Lebanon", "IL": "Israel", "LV": "Latvia"}
# Suggested zone per country for the wizard; the operator can pick any IANA zone.
COUNTRY_ZONES = {"IR": "Asia/Tehran", "LB": "Asia/Beirut", "IL": "Asia/Jerusalem", "LV": "Europe/Riga", "FI": "Europe/Helsinki"}
INPUT_TYPES = {"images", "video"}
MODES = {
    "images": {"general_traffic", "object_search", "plate_recognition"},
    "video": {"object_search", "plate_recognition"},
}
EXPORT_FORMATS = {"xlsx", "csv"}
# Vehicle detector of a still-image mission: YOLOX-s (or the model promoted on the learning page), or SAM 3.1,
# which finds vehicles by name: slower, and stronger on unusual views. Video always uses the standard detector.
DETECTORS = {"standard", "sam3"}

CONTROL_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mission (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  db_path TEXT NOT NULL,
  input_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'active'
);
-- Administrative actions: backups, restores, mission deletions.
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  author TEXT NOT NULL,
  action TEXT NOT NULL,             -- mission_deleted | backup | backup_failed | restore
  detail TEXT NOT NULL              -- JSON
);
-- Promotions and rollbacks of production models. Append-only; replaying it gives the model in use. Never written automatically.
CREATE TABLE IF NOT EXISTS model_promotion (
  id INTEGER PRIMARY KEY,
  task TEXT NOT NULL,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  action TEXT NOT NULL,             -- promote | rollback
  mission_id TEXT,                  -- where the evaluation behind it lives
  evaluation_id INTEGER,
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
"""

MISSION_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE mission (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  input_type TEXT NOT NULL,
  mode TEXT NOT NULL,
  country TEXT NOT NULL,
  iana_zone TEXT NOT NULL,
  latitude REAL, longitude REAL,
  created_at TEXT NOT NULL
);
-- Every configuration change is a new revision; current = highest seq per kind.
CREATE TABLE config_revision (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL,
  reason TEXT
);
CREATE TABLE source_root (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  recursive INTEGER NOT NULL
);
-- Logical identity of a path under a source root.
CREATE TABLE source_file (
  id INTEGER PRIMARY KEY,
  root_id INTEGER NOT NULL REFERENCES source_root(id),
  rel_path TEXT NOT NULL,
  first_seen TEXT NOT NULL,
  missing_since TEXT,
  last_size INTEGER,
  last_mtime_ns INTEGER,
  UNIQUE(root_id, rel_path)
);
-- One row per distinct content observed at a path. Never deleted or overwritten.
CREATE TABLE source_version (
  id INTEGER PRIMARY KEY,
  source_file_id INTEGER NOT NULL REFERENCES source_file(id),
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  capture_utc TEXT,
  capture_precision TEXT,
  name_pattern TEXT,
  name_fields TEXT,
  parse_status TEXT NOT NULL,      -- ok | malformed
  parse_detail TEXT,
  integrity TEXT NOT NULL,         -- ok | corrupt | unsupported
  integrity_detail TEXT,
  first_seen TEXT NOT NULL,
  superseded_by INTEGER REFERENCES source_version(id),
  possible_rename_of INTEGER REFERENCES source_version(id),
  UNIQUE(source_file_id, sha256, size, mtime_ns)
);
CREATE INDEX ix_version_capture ON source_version(capture_utc);
CREATE INDEX ix_version_sha ON source_version(sha256);
-- Files seen but not yet stable (possibly still being copied).
CREATE TABLE pending_file (
  root_id INTEGER NOT NULL,
  rel_path TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  unchanged_since TEXT NOT NULL,
  PRIMARY KEY(root_id, rel_path)
);
CREATE TABLE scan (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,           -- running | completed | source_unavailable | failed
  summary TEXT
);
CREATE TABLE run (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,             -- initial | continuous_batch | reprocess
  state TEXT NOT NULL,            -- queued | running | paused | completed | cancelled | failed
  config_snapshot TEXT NOT NULL,
  model_versions TEXT NOT NULL DEFAULT '{}',
  preprocessing TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  started_at TEXT, finished_at TEXT
);
-- Processing status only; review status is stored separately.
CREATE TABLE run_item (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES run(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  processing_state TEXT NOT NULL DEFAULT 'queued',
  stage TEXT, error_code TEXT, error_detail TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  UNIQUE(run_id, source_version_id)
);
CREATE TABLE notification (
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  severity TEXT NOT NULL,         -- info | warning | error
  code TEXT NOT NULL,
  message TEXT NOT NULL,
  context TEXT,
  acknowledged INTEGER NOT NULL DEFAULT 0
);
"""


# Forward-only migrations. Each runs once, after a backup of the mission database.
MISSION_MIGRATIONS: dict[int, str] = {
    2: """
CREATE TABLE camera_location (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE camera_view (
  id INTEGER PRIMARY KEY,
  location_id INTEGER REFERENCES camera_location(id),
  name TEXT NOT NULL,
  ref_width INTEGER, ref_height INTEGER,
  state TEXT NOT NULL DEFAULT 'proposed',   -- proposed | active | needs_review | retired
  merged_into INTEGER REFERENCES camera_view(id),
  link_note TEXT,                            -- why clusters were joined, and whether a person confirmed it
  link_confirmed INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);
-- Illumination-robust structural signature per source version.
CREATE TABLE view_signature (
  source_version_id INTEGER PRIMARY KEY REFERENCES source_version(id),
  algo TEXT NOT NULL,
  vec BLOB NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL,
  mean_luma REAL NOT NULL, saturation REAL NOT NULL,
  condition TEXT NOT NULL                    -- colour | monochrome (measured, not from clock time)
);
-- Assignment history; current = latest row per version without superseded_by.
CREATE TABLE view_assignment (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  view_id INTEGER REFERENCES camera_view(id),  -- NULL = unassigned
  method TEXT NOT NULL,                         -- auto | manual
  score REAL,
  author TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL,
  superseded_by INTEGER REFERENCES view_assignment(id)
);
CREATE INDEX ix_assign_current ON view_assignment(source_version_id) WHERE superseded_by IS NULL;
CREATE TABLE view_event (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL,
  kind TEXT NOT NULL,                           -- created | renamed | merged | split | confirmed_link | needs_review | drift_check
  detail TEXT,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
-- Reference image per view and lighting condition, used for drift checks.
CREATE TABLE view_reference (
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  condition TEXT NOT NULL,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  set_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(view_id, condition)
);
CREATE TABLE view_drift (
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  inliers INTEGER NOT NULL,
  displacement_px REAL,          -- max shift of region vertices (or frame corners), original pixels
  result TEXT NOT NULL,          -- ok | shifted | unverified
  checked_at TEXT NOT NULL,
  PRIMARY KEY(source_version_id, view_id)
);
-- Region geometry in the view's original pixel coordinates. Configuration, not training labels.
CREATE TABLE region_revision (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  region_uid TEXT NOT NULL,
  kind TEXT NOT NULL,                           -- include | exclude
  name TEXT,
  polygon TEXT NOT NULL,                        -- JSON [[x,y],...]
  deleted INTEGER NOT NULL DEFAULT 0,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
""",
    3: """
-- Model identity used by runs in this mission (copied from the pack's card).
CREATE TABLE model_version (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  task TEXT NOT NULL,
  license TEXT NOT NULL,
  card TEXT NOT NULL,
  first_used TEXT NOT NULL,
  UNIQUE(name, version, sha256)
);
-- Machine output. Immutable: never updated or deleted; corrections live in annotation_revision.
CREATE TABLE detection (
  id INTEGER PRIMARY KEY,
  run_item_id INTEGER NOT NULL REFERENCES run_item(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  view_id INTEGER REFERENCES camera_view(id),
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,   -- original pixels
  cls TEXT NOT NULL,
  score REAL NOT NULL,
  class_scores TEXT NOT NULL,
  in_region INTEGER,               -- 1 inside, 0 outside, NULL regions not applied (view needs review)
  region_rule TEXT,
  model_version_id INTEGER NOT NULL REFERENCES model_version(id),
  created_at TEXT NOT NULL
);
CREATE INDEX ix_detection_version ON detection(source_version_id);
-- Human decisions on objects. Append-only: the latest revision per object_uid is current.
-- An object is either a machine detection (detection_id set) or added by a person (detection_id NULL).
CREATE TABLE annotation_revision (
  id INTEGER PRIMARY KEY,
  object_uid TEXT NOT NULL,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  detection_id INTEGER REFERENCES detection(id),
  action TEXT NOT NULL,            -- removed | added | reclassed | moved | restored
  status TEXT NOT NULL,            -- present | removed
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  cls TEXT,
  label_role TEXT NOT NULL DEFAULT 'analytical',   -- analytical | human_reference | excluded_from_training
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_annotation_object ON annotation_revision(object_uid);
CREATE INDEX ix_annotation_version ON annotation_revision(source_version_id);
-- Review status per source version, separate from processing status. Append-only.
CREATE TABLE review_decision (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  decision TEXT NOT NULL,          -- approved | bulk_approved | reopened
  bulk_approval_id INTEGER REFERENCES bulk_approval(id),
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_review_version ON review_decision(source_version_id);
-- "Approve all": a fixed list of items at the moment of approval. Not a human review, not ground truth.
CREATE TABLE bulk_approval (
  id INTEGER PRIMARY KEY,
  scope TEXT NOT NULL,             -- JSON description of the filter shown to the operator
  item_count INTEGER NOT NULL,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE export (
  id INTEGER PRIMARY KEY,
  format TEXT NOT NULL,
  path TEXT,
  scope TEXT NOT NULL,
  rows INTEGER,
  status TEXT NOT NULL,            -- completed | failed
  error TEXT,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
""",
    4: """
-- Image condition measured from content (never from clock time).
CREATE TABLE image_condition (
  source_version_id INTEGER PRIMARY KEY REFERENCES source_version(id),
  condition TEXT NOT NULL,          -- day | dusk | night
  monochrome INTEGER NOT NULL,
  mean_luma REAL NOT NULL, saturation REAL NOT NULL,
  clipped_pct REAL NOT NULL, dark_pct REAL NOT NULL, detail REAL NOT NULL,
  flags TEXT NOT NULL,              -- JSON list: glare, low_detail
  algo TEXT NOT NULL
);
-- Light components: one connected bright region each. Machine output, immutable.
CREATE TABLE light_component (
  id INTEGER PRIMARY KEY,
  run_item_id INTEGER NOT NULL REFERENCES run_item(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  cx REAL NOT NULL, cy REAL NOT NULL,
  area INTEGER NOT NULL, peak REAL NOT NULL, mean REAL NOT NULL,
  elongation REAL NOT NULL, red_share REAL
);
CREATE INDEX ix_light_version ON light_component(source_version_id);
-- Persistent-light model per view and condition. Each rebuild is a new version.
CREATE TABLE background_model (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  condition TEXT NOT NULL,
  image_count INTEGER NOT NULL,
  state TEXT NOT NULL,              -- ready | insufficient
  persistent_mask BLOB,             -- PNG, original resolution
  frequency BLOB,                   -- PNG, 0-255 = share of images lit
  new_lights TEXT NOT NULL DEFAULT '[]',   -- JSON boxes lit often recently but not over the long horizon
  params TEXT NOT NULL,
  input_ids TEXT NOT NULL,          -- JSON source version ids used
  built_at TEXT NOT NULL
);
-- Light groups: hypotheses that components belong together. Machine output, immutable.
CREATE TABLE light_group (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  background_model_id INTEGER REFERENCES background_model(id),
  region_revision_id INTEGER,       -- latest region edit of the view when analysed; a newer edit triggers re-analysis
  kind TEXT NOT NULL,               -- single | pair | cluster
  member_ids TEXT NOT NULL,         -- JSON light_component ids
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  score REAL NOT NULL,
  reflection REAL NOT NULL,
  explained_by_background INTEGER NOT NULL,
  in_region INTEGER,
  covered_by_detection INTEGER REFERENCES detection(id),
  is_candidate INTEGER NOT NULL,
  count_min INTEGER NOT NULL, count_max INTEGER NOT NULL,
  alternatives TEXT NOT NULL,       -- JSON list of other readings
  flags TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_group_version ON light_group(source_version_id);
-- Human decisions on candidates. Append-only; the latest per group_uid is current.
CREATE TABLE candidate_revision (
  id INTEGER PRIMARY KEY,
  group_uid TEXT NOT NULL,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  light_group_id INTEGER REFERENCES light_group(id),
  action TEXT NOT NULL,             -- confirmed | rejected | persistent_light | regrouped | superseded | restored
  member_ids TEXT,                  -- for operator-made groups
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  cls TEXT,
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_candidate_version ON candidate_revision(source_version_id);
-- Operator labels: authoritative over the statistics. Configuration, not training labels.
CREATE TABLE light_label (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  condition TEXT NOT NULL,
  kind TEXT NOT NULL,               -- persistent | not_persistent
  polygon TEXT NOT NULL,
  deleted INTEGER NOT NULL DEFAULT 0,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
""",
    5: """
-- Machine attribute readings (colour), cached per object box and algorithm version. Immutable.
CREATE TABLE object_attribute (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,
  box TEXT NOT NULL,                -- JSON [x1,y1,x2,y2] the reading was made on
  attribute TEXT NOT NULL,          -- colour
  reading TEXT NOT NULL,            -- JSON: state, values, shares, reason
  algo TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (source_version_id, object_uid, box, attribute, algo)
);
-- Operator attribute corrections. Append-only; the latest per object and attribute is current.
CREATE TABLE attribute_revision (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,
  attribute TEXT NOT NULL,
  value TEXT,                       -- a colour, 'not_observable', or NULL to return to the machine reading
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_attr_rev ON attribute_revision(source_version_id, object_uid, attribute);
-- Typed search requests, their machine interpretation and the operator's confirmed version.
CREATE TABLE search_request (
  id INTEGER PRIMARY KEY,
  chat_text TEXT NOT NULL,
  interpreter_version TEXT NOT NULL,
  spec_json TEXT NOT NULL,          -- interpreter output, never changed
  confirmed_spec_json TEXT,         -- what the operator confirmed (may be edited)
  edits_json TEXT,                  -- differences between the two
  ignored_words TEXT,               -- JSON list of not-understood words the operator chose to ignore
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  confirmed_by TEXT,
  confirmed_at TEXT
);
CREATE TABLE search_run (
  id INTEGER PRIMARY KEY,
  request_id INTEGER NOT NULL REFERENCES search_request(id),
  spec_json TEXT NOT NULL,          -- the confirmed spec actually executed
  state TEXT NOT NULL,              -- running | finished | cancelled | failed
  attribute_algo TEXT NOT NULL,
  summary TEXT,
  error TEXT,
  started_by TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT
);
CREATE TABLE search_hit (
  id INTEGER PRIMARY KEY,
  search_run_id INTEGER NOT NULL REFERENCES search_run(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,
  status TEXT NOT NULL,             -- match | possible
  reasons TEXT NOT NULL,            -- JSON list of why it is only possible
  target INTEGER NOT NULL,          -- index of the spec target it matched best
  cls TEXT NOT NULL,
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  colour TEXT NOT NULL,             -- JSON reading used, including operator corrections
  UNIQUE (search_run_id, source_version_id, object_uid)
);
CREATE INDEX ix_hit_run ON search_hit(search_run_id, source_version_id);
-- Operator verdicts on hits. Append-only; the latest per hit is current.
CREATE TABLE search_hit_decision (
  id INTEGER PRIMARY KEY,
  search_hit_id INTEGER NOT NULL REFERENCES search_hit(id),
  decision TEXT NOT NULL,           -- correct | not_a_match | undecided
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
""",
    6: """
-- Video: what the decoder found in each recording.
CREATE TABLE video_probe (
  source_version_id INTEGER PRIMARY KEY REFERENCES source_version(id),
  container TEXT, codec TEXT, width INTEGER, height INTEGER,
  duration_s REAL, nominal_fps REAL, frame_count INTEGER,
  variable_frame_rate INTEGER,      -- 1 when frame intervals differ by more than 10%
  creation_time TEXT,               -- container tag, informative only
  start_utc TEXT,                   -- absolute start, from the confirmed source below; NULL when unknown
  start_precision TEXT,
  start_source TEXT,                -- filename | none
  error TEXT,
  probed_at TEXT NOT NULL
);
-- Resumable position inside a recording: decoding restarts after last_t with the saved tracker state.
CREATE TABLE video_checkpoint (
  run_item_id INTEGER PRIMARY KEY REFERENCES run_item(id),
  last_t REAL NOT NULL,             -- seconds from the stream start, last analysed frame
  next_sample_t REAL NOT NULL,
  frames_analysed INTEGER NOT NULL,
  tracker_state TEXT NOT NULL,
  prev_signature BLOB,
  updated_at TEXT NOT NULL
);
CREATE TABLE track (
  id INTEGER PRIMARY KEY,
  run_item_id INTEGER NOT NULL REFERENCES run_item(id),
  track_key TEXT NOT NULL,          -- T1, T2, ... within the recording
  state TEXT NOT NULL,              -- tentative | confirmed | closed | discarded
  closed_reason TEXT,               -- left_frame | lost | scene_cut | end_of_recording | too_short
  UNIQUE (run_item_id, track_key)
);
CREATE TABLE track_observation (
  id INTEGER PRIMARY KEY,
  track_id INTEGER NOT NULL REFERENCES track(id),
  t REAL NOT NULL,                  -- seconds from the stream start (presentation time)
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
  cls TEXT NOT NULL, score REAL NOT NULL,
  appearance TEXT NOT NULL DEFAULT 'body'
);
CREATE INDEX ix_obs_track ON track_observation(track_id, t);
CREATE TABLE video_marker (
  id INTEGER PRIMARY KEY,
  run_item_id INTEGER NOT NULL REFERENCES run_item(id),
  t REAL NOT NULL,
  kind TEXT NOT NULL                -- scene_cut
);
-- One machine event per confirmed track. Immutable; review decisions are appended to event_revision.
CREATE TABLE video_event (
  id INTEGER PRIMARY KEY,
  run_item_id INTEGER NOT NULL REFERENCES run_item(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  event_key TEXT NOT NULL,          -- E-0001 within the recording
  track_id INTEGER NOT NULL REFERENCES track(id),
  first_t REAL NOT NULL, last_t REAL NOT NULL,
  cls TEXT NOT NULL, cls_share REAL NOT NULL,
  class_votes TEXT NOT NULL,
  entry TEXT NOT NULL, exit TEXT NOT NULL,   -- JSON [x, y] relative to frame size
  end_reason TEXT NOT NULL,
  flags TEXT NOT NULL,              -- JSON list: possible_duplicate, touches_edge_at_start, short, ...
  related TEXT NOT NULL,            -- JSON list of event keys this one may duplicate
  thumb_t REAL NOT NULL,
  thumb_box TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (run_item_id, event_key)
);
CREATE TABLE event_revision (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES video_event(id),
  action TEXT NOT NULL,             -- confirm | reject | duplicate | set_times | set_class | merge_into | reopen
  payload TEXT NOT NULL,
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_event_rev ON event_revision(event_id);
CREATE TABLE clip (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES video_event(id),
  path TEXT NOT NULL,
  method TEXT NOT NULL,             -- copy | reencode
  encoder TEXT,
  context_before_s REAL NOT NULL, context_after_s REAL NOT NULL,
  requested_start_t REAL NOT NULL, requested_end_t REAL NOT NULL,
  actual_start_t REAL, actual_end_t REAL,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
""",
    7: """
-- One row per vehicle detection the plate stage looked at, for one detector + reader pair (machine output, never edited).
CREATE TABLE plate_read (
  id INTEGER PRIMARY KEY,
  detection_id INTEGER NOT NULL REFERENCES detection(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,          -- d<detection id>
  outcome TEXT NOT NULL,             -- read | no_plate_found | vehicle_too_small | plate_too_small
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  det_score REAL,
  crop_path TEXT,
  raw_text TEXT,                     -- reader output as emitted (pad removed), Unicode, reading order
  raw_chars TEXT,                    -- JSON [[char, confidence], ...] for every slot
  normalized_text TEXT,              -- '?' for unsure characters; never completed by guessing
  display_text TEXT,
  country_hyp TEXT,                  -- mission country
  layout_hyp TEXT,
  region_label TEXT, region_conf REAL,   -- the reader's own country guess
  readability TEXT,                  -- full | partial | unreadable (NULL when no plate was read)
  reasons TEXT NOT NULL DEFAULT '[]',
  detector_model_id INTEGER REFERENCES model_version(id),
  reader_model_id INTEGER REFERENCES model_version(id),
  rules TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(detection_id, detector_model_id, reader_model_id)
);
CREATE INDEX ix_plate_read_version ON plate_read(source_version_id);
-- Operator corrections, append-only, per vehicle object (so a plate can also be added where none was found).
CREATE TABLE plate_revision (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,
  action TEXT NOT NULL,              -- confirm | correct | unreadable | no_plate | reopen
  text TEXT,                         -- as the operator typed it, Unicode
  normalized_text TEXT,
  box TEXT,                          -- JSON [x1,y1,x2,y2] when the operator drew the plate
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_plate_rev ON plate_revision(source_version_id, object_uid);
-- A country's plates are evaluation only until an evaluation on reviewed plates passes; append-only.
CREATE TABLE plate_country_status (
  id INTEGER PRIMARY KEY,
  country TEXT NOT NULL,
  action TEXT NOT NULL,              -- evaluate | enable | disable
  reader_model_id INTEGER REFERENCES model_version(id),
  result TEXT NOT NULL,              -- JSON: counts, exact match, CER per layout, thresholds
  passed INTEGER,
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
""",
    8: """
-- A dataset version is a frozen list of reviewed images with their labels and split (LRN-03).
CREATE TABLE training_dataset (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  task TEXT NOT NULL,                -- vehicle_detection
  params TEXT NOT NULL,              -- JSON: ratios, block_days, seed, eligibility rules
  stats TEXT NOT NULL,               -- JSON: counts per split, review kind, condition, view
  frozen_test_from INTEGER REFERENCES training_dataset(id),
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE dataset_member (
  dataset_id INTEGER NOT NULL REFERENCES training_dataset(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  sha256 TEXT NOT NULL,
  view_key TEXT NOT NULL,
  block TEXT NOT NULL,               -- local date block, e.g. 2026-10-21
  split TEXT NOT NULL,               -- train | val | test
  review TEXT NOT NULL,              -- full | partial
  condition TEXT,
  labels TEXT NOT NULL,              -- JSON snapshot: positives, ignore boxes and polygons
  last_revision_id INTEGER,          -- newest annotation revision included
  PRIMARY KEY(dataset_id, source_version_id)
);
CREATE INDEX ix_member_split ON dataset_member(dataset_id, split);
CREATE TABLE evaluation (
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL REFERENCES training_dataset(id),
  split TEXT NOT NULL,
  model TEXT NOT NULL,               -- JSON: name, version, sha256, preprocessing
  state TEXT NOT NULL,               -- running | completed | failed | cancelled
  report TEXT,                       -- JSON
  error TEXT,
  author TEXT NOT NULL,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
""",
    9: """
-- Night: per-view distance bands (what a vehicle looks like at each distance). Append-only; the latest row is current.
CREATE TABLE view_night_setting (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  far_below REAL NOT NULL,           -- expected vehicle width (px) under which a vehicle is far
  close_from REAL NOT NULL,          -- expected width from which it is close
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
ALTER TABLE light_group ADD COLUMN band TEXT;              -- far | middle | close; NULL without a width map
ALTER TABLE light_group ADD COLUMN recurrence REAL;        -- share of the view's images lit at this place
ALTER TABLE light_group ADD COLUMN setting_id INTEGER;     -- view_night_setting row used; a newer one triggers re-analysis
""",
    10: """
ALTER TABLE light_group ADD COLUMN decision_rev INTEGER;   -- newest operator candidate decision in the view that was used
""",
    11: """
ALTER TABLE light_group ADD COLUMN algo TEXT;              -- grouping rules version; a newer one triggers re-analysis
""",
    12: """
ALTER TABLE light_group ADD COLUMN features TEXT;          -- measurements used by learned night scoring (JSON)
ALTER TABLE light_group ADD COLUMN model_id INTEGER;       -- night_light_model row that scored it; NULL: rules only
CREATE TABLE night_light_model (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  decision_rev INTEGER,             -- newest operator candidate decision in the view when fitted
  created_at TEXT NOT NULL,
  n_pos INTEGER NOT NULL,           -- answers: vehicle
  n_neg INTEGER NOT NULL,           -- answers: not a vehicle or fixed light
  auc_model REAL,                   -- cross-validated, folds split by image
  auc_rules REAL,                   -- rule score on the same held-out answers
  applied INTEGER NOT NULL,         -- 1 when the model replaces the rule score
  reason TEXT NOT NULL,
  params TEXT NOT NULL,
  model TEXT                        -- coefficients (JSON) when applied
);
CREATE INDEX ix_night_model_view ON night_light_model(view_id, decision_rev);
""",
    13: """
-- Plates of vehicles in video events: read on several frames of the event and combined; machine output, immutable.
CREATE TABLE event_plate_read (
  id INTEGER PRIMARY KEY,
  event_id INTEGER NOT NULL REFERENCES video_event(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  object_uid TEXT NOT NULL,          -- e<event id>
  frame_t REAL,                      -- seconds from the stream start of the frame the result comes from
  vx1 REAL, vy1 REAL, vx2 REAL, vy2 REAL,   -- the vehicle on that frame
  outcome TEXT NOT NULL,             -- read | no_plate_found | vehicle_too_small | plate_too_small
  x1 REAL, y1 REAL, x2 REAL, y2 REAL,
  det_score REAL,
  crop_path TEXT,
  raw_text TEXT, raw_chars TEXT, normalized_text TEXT, display_text TEXT,
  country_hyp TEXT, layout_hyp TEXT, region_label TEXT, region_conf REAL,
  readability TEXT,
  reasons TEXT NOT NULL DEFAULT '[]',
  frames TEXT NOT NULL DEFAULT '[]', -- JSON: every frame read (time, outcome, text, readability, score)
  detector_model_id INTEGER REFERENCES model_version(id),
  reader_model_id INTEGER REFERENCES model_version(id),
  rules TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(event_id, detector_model_id, reader_model_id)
);
CREATE INDEX ix_event_plate_version ON event_plate_read(source_version_id);
""",
    14: """
-- Confirmed manual recording start times. Latest revision controls the UTC start without rewriting probe or source metadata.
CREATE TABLE video_start_revision (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  action TEXT NOT NULL,              -- set | clear
  start_utc TEXT,
  precision TEXT,
  author TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL,
  CHECK ((action='set' AND start_utc IS NOT NULL AND precision IN ('minute','second')) OR
         (action='clear' AND start_utc IS NULL AND precision IS NULL))
);
CREATE INDEX ix_video_start_revision ON video_start_revision(source_version_id, id);
""",
    15: """
-- Empty-road images an operator chose per camera view; latest row per image wins (marked | unmarked).
CREATE TABLE reference_mark (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  action TEXT NOT NULL CHECK (action IN ('marked','unmarked')),
  author TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_reference_mark ON reference_mark(view_id, source_version_id, id);
-- The empty background of a view for one light condition: per-pixel median of its empty images. Immutable.
CREATE TABLE reference_background (
  id INTEGER PRIMARY KEY,
  view_id INTEGER NOT NULL REFERENCES camera_view(id),
  condition TEXT NOT NULL,          -- day | dusk | night
  input_ids TEXT NOT NULL,          -- JSON: source versions it was built from
  image_count INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  median BLOB NOT NULL,             -- PNG, grey, original resolution
  spread BLOB NOT NULL,             -- PNG, per-pixel spread between the empty images (0 with fewer than 3)
  params TEXT NOT NULL,
  built_at TEXT NOT NULL
);
CREATE INDEX ix_reference_background ON reference_background(view_id, condition, id);
-- With no regions drawn, the road SAM 3.1 outlines on an empty background is where changes are looked for. Immutable.
CREATE TABLE reference_area (
  id INTEGER PRIMARY KEY,
  background_id INTEGER NOT NULL REFERENCES reference_background(id),
  sam_model_id INTEGER REFERENCES model_version(id),
  prompts TEXT NOT NULL,            -- JSON: the words asked
  polygons TEXT NOT NULL,           -- JSON: [{kind include|exclude, polygon}]; [] when SAM found none
  area_share REAL NOT NULL,
  empty_check TEXT NOT NULL,        -- JSON: each empty image against the others, inside this area
  created_at TEXT NOT NULL
);
CREATE INDEX ix_reference_area ON reference_area(background_id, id);
-- One comparison of an image with its view's empty background. Machine output, immutable; the newest counts.
CREATE TABLE change_analysis (
  id INTEGER PRIMARY KEY,
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  reference_id INTEGER REFERENCES reference_background(id),   -- NULL: no empty background applies (any more)
  region_revision_id INTEGER,
  algo TEXT NOT NULL,
  params TEXT NOT NULL,
  sam_model_id INTEGER REFERENCES model_version(id),           -- SAM 3.1 pack that typed the vehicles; NULL: not used
  shift REAL,                       -- brightness difference to the background that was taken out
  area_id INTEGER REFERENCES reference_area(id),   -- road found by SAM 3.1 when the view has no regions drawn
  created_at TEXT NOT NULL
);
CREATE INDEX ix_change_analysis ON change_analysis(source_version_id, id);
CREATE TABLE change_vehicle (
  id INTEGER PRIMARY KEY,
  analysis_id INTEGER NOT NULL REFERENCES change_analysis(id),
  source_version_id INTEGER NOT NULL REFERENCES source_version(id),
  x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,   -- original pixels
  change_box TEXT NOT NULL,         -- JSON: the changed area before SAM narrowed it
  cls TEXT NOT NULL,                -- vehicle (type unknown) or the class SAM 3.1 confirmed
  score REAL NOT NULL,              -- how strongly the area differs from the empty background, 0..1
  kind TEXT NOT NULL,               -- lights (a new near-white core: headlights, glare) | glow | shape
  parts INTEGER NOT NULL,           -- changed areas merged into this one (two headlights and a reflection: 3)
  changed_px INTEGER NOT NULL,
  sam_cls TEXT, sam_score REAL,
  in_region INTEGER,
  flags TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX ix_change_vehicle ON change_vehicle(analysis_id);
""",
}


def _statements(sql: str) -> list[str]:
    """Split migration SQL into statements, dropping -- comments (which may contain semicolons)."""
    lines = [l.split("--", 1)[0] for l in sql.splitlines()]
    return [st.strip() for st in "\n".join(lines).split(";") if st.strip()]


def migrate(db_path: Path, con: sqlite3.Connection) -> int:
    """Apply pending migrations, backing up the database first. Returns the resulting version."""
    current = con.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    pending = sorted(v for v in MISSION_MIGRATIONS if v > current)
    if not pending:
        return current
    backups = db_path.parent / "backups"
    backups.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = sqlite3.connect(backups / f"mission-v{current}-{stamp}.db")
    con.backup(dest)
    dest.close()
    for v in pending:
        con.execute("BEGIN")
        try:
            for stmt in _statements(MISSION_MIGRATIONS[v]):
                con.execute(stmt)
            con.execute("INSERT INTO schema_version VALUES (?)", (v,))
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    return pending[-1]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextlib.contextmanager
def transaction(con: sqlite3.Connection):
    """BEGIN ... COMMIT, rolled back when the block raises, so the connection never stays inside a transaction."""
    con.execute("BEGIN")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


def connect(path: Path) -> sqlite3.Connection:
    # One connection is never used by two threads at once, but web requests may open and close it
    # on different worker threads, so the same-thread check is off.
    con = sqlite3.connect(path, isolation_level=None, check_same_thread=False)  # explicit transactions
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=5000")
    return con


@dataclass
class MissionConfig:
    name: str
    input_type: str
    mode: str
    country: str
    iana_zone: str
    source_dir: str
    include_subdirs: bool
    processing: str  # one_time | continuous
    output_dir: str
    export_formats: list[str]
    latitude: float | None = None
    longitude: float | None = None
    monitoring_enabled: bool = field(default=False)
    detector: str = "standard"


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate(cfg: MissionConfig, data_root: Path, min_free_bytes: int = 1 << 30) -> Validation:
    v = Validation()
    if not cfg.name.strip():
        v.errors.append("Mission name is empty.")
    if cfg.input_type not in INPUT_TYPES:
        v.errors.append(f"Input type must be one of {sorted(INPUT_TYPES)}.")
    elif cfg.mode not in MODES[cfg.input_type]:
        v.errors.append(f"Mode '{cfg.mode}' is not available for {cfg.input_type}; choose one of {sorted(MODES[cfg.input_type])}.")
    if cfg.country not in TARGET_COUNTRIES:
        v.warnings.append(
            f"Country '{cfg.country}' is outside the supported set ({', '.join(TARGET_COUNTRIES)}); "
            "plate recognition will not be available."
        )
        if cfg.mode == "plate_recognition":
            v.errors.append("Plate recognition needs one of the supported countries.")
    try:
        ZoneInfo(cfg.iana_zone)
    except (ZoneInfoNotFoundError, ValueError):
        v.errors.append(f"'{cfg.iana_zone}' is not a known IANA time zone.")
    if (cfg.latitude is None) != (cfg.longitude is None):
        v.errors.append("Choose both latitude and longitude, or leave the location unset.")
    elif cfg.latitude is not None and cfg.longitude is not None and not (-90 <= cfg.latitude <= 90 and -180 <= cfg.longitude <= 180):
        v.errors.append("Location must use latitude −90..90 and longitude −180..180.")
    if cfg.processing not in {"one_time", "continuous"}:
        v.errors.append("Processing must be one_time or continuous.")
    if not cfg.export_formats or not set(cfg.export_formats) <= EXPORT_FORMATS:
        v.errors.append("Export formats must be xlsx, csv, or both.")
    if cfg.detector not in DETECTORS:
        v.errors.append(f"Detector must be one of {sorted(DETECTORS)}.")
    elif cfg.detector == "sam3" and cfg.input_type == "video":
        v.errors.append("SAM 3.1 detection is for still images; video missions use the standard detector.")

    src = Path(cfg.source_dir)
    if not src.is_dir():
        v.errors.append(f"Source folder {src} does not exist or is not a folder on this workstation.")
    elif not os.access(src, os.R_OK | os.X_OK):
        v.errors.append(f"Source folder {src} cannot be read by the application account.")

    out = Path(cfg.output_dir)
    if not out.is_dir():
        v.errors.append(f"Output folder {out} does not exist.")
    else:
        try:
            with tempfile.NamedTemporaryFile(dir=out, prefix=".trafficwb-write-test-"):
                pass
        except OSError as exc:
            v.errors.append(f"Output folder {out} is not writable: {exc.strerror or exc}.")
        free = shutil.disk_usage(out).free
        if free < min_free_bytes:
            v.warnings.append(f"Only {free // (1 << 20)} MiB free on the output volume.")
    if src.is_dir() and out.is_dir() and _inside(out, src) and cfg.include_subdirs:
        v.errors.append("Output folder is inside the source folder and subfolders are included; exports would be re-ingested.")
    if src.is_dir() and _inside(data_root, src) and cfg.include_subdirs:
        v.errors.append("The application data folder is inside the source folder; databases would be re-ingested.")
    return v


class DataRoot:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "missions").mkdir(exist_ok=True)
        self.con = connect(self.path / "control.db")
        self.con.executescript(CONTROL_SCHEMA)
        if self.con.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            self.con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))

    def missions(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM mission ORDER BY created_at").fetchall()

    def create_mission(self, cfg: MissionConfig, author: str) -> "Mission":
        """Validate and commit a mission: its database exists before any run."""
        val = validate(cfg, self.path)
        if not val.ok:
            raise MissionValidationError(val)
        if self.con.execute("SELECT 1 FROM mission WHERE name=?", (cfg.name,)).fetchone():
            raise MissionValidationError(Validation(errors=[f"A mission named '{cfg.name}' already exists."]))
        mid = str(uuid.uuid4())
        mdir = self.path / "missions" / mid
        mdir.mkdir()
        tmp = mdir / "mission.db.creating"
        con = connect(tmp)
        now = utcnow()
        con.executescript(MISSION_SCHEMA)
        con.execute("BEGIN")
        con.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        con.execute(
            "INSERT INTO mission VALUES (?,?,?,?,?,?,?,?,?)",
            (mid, cfg.name, cfg.input_type, cfg.mode, cfg.country, cfg.iana_zone, cfg.latitude, cfg.longitude, now),
        )
        con.execute(
            "INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
            (json.dumps(asdict(cfg)), author, now, "mission created"),
        )
        con.execute(
            "INSERT INTO source_root(path,recursive) VALUES (?,?)",
            (str(Path(cfg.source_dir).resolve()), int(cfg.include_subdirs)),
        )
        con.execute("COMMIT")
        for v in sorted(MISSION_MIGRATIONS):  # new databases start at the latest schema, no backup needed
            con.execute("BEGIN")
            for stmt in _statements(MISSION_MIGRATIONS[v]):
                con.execute(stmt)
            con.execute("INSERT INTO schema_version VALUES (?)", (v,))
            con.execute("COMMIT")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        for suffix in ("-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)
        final = mdir / "mission.db"
        os.replace(tmp, final)
        self.con.execute(
            "INSERT INTO mission(id,name,db_path,input_type,mode,created_at) VALUES (?,?,?,?,?,?)",
            (mid, cfg.name, str(final.relative_to(self.path)), cfg.input_type, cfg.mode, now),
        )
        return Mission(self, mid)

    def open_mission(self, mission_id: str) -> "Mission":
        return Mission(self, mission_id)


class MissionValidationError(ValueError):
    def __init__(self, validation: Validation):
        super().__init__("; ".join(validation.errors))
        self.validation = validation


class Mission:
    def __init__(self, root: DataRoot, mission_id: str):
        row = root.con.execute("SELECT * FROM mission WHERE id=?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(mission_id)
        self.root = root
        self.id = mission_id
        self.dir = root.path / "missions" / mission_id
        db_path = root.path / row["db_path"]
        self.con = connect(db_path)
        migrate(db_path, self.con)

    # --- configuration ---
    def config(self) -> MissionConfig:
        row = self.con.execute(
            "SELECT payload FROM config_revision WHERE kind='mission' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return MissionConfig(**json.loads(row["payload"]))

    def update_config(self, author: str, reason: str, **changes: Any) -> MissionConfig:
        cfg = self.config()
        for k, val in changes.items():
            if k in ("input_type", "mode"):
                raise ValueError("A mission's input type and mode cannot change after creation.")
            if not hasattr(cfg, k):
                raise AttributeError(k)
            setattr(cfg, k, val)
        self.con.execute(
            "INSERT INTO config_revision(kind,payload,author,created_at,reason) VALUES ('mission',?,?,?,?)",
            (json.dumps(asdict(cfg)), author, utcnow(), reason),
        )
        return cfg

    @property
    def zone(self) -> str:
        return self.con.execute("SELECT iana_zone FROM mission").fetchone()[0]

    @property
    def input_type(self) -> str:
        return self.con.execute("SELECT input_type FROM mission").fetchone()[0]

    def notify(self, severity: str, code: str, message: str, **context: Any) -> None:
        self.con.execute(
            "INSERT INTO notification(created_at,severity,code,message,context) VALUES (?,?,?,?,?)",
            (utcnow(), severity, code, message, json.dumps(context, ensure_ascii=False)),
        )

    # --- runs ---
    def create_run(self, kind: str, version_ids: list[int]) -> int | None:
        """Create a run over the given source versions. Returns None if there is nothing to process."""
        if not version_ids:
            return None
        self.con.execute("BEGIN")
        cur = self.con.execute(
            "INSERT INTO run(kind,state,config_snapshot,created_at) VALUES (?,?,?,?)",
            (kind, "queued", json.dumps(asdict(self.config())), utcnow()),
        )
        run_id = cur.lastrowid
        self.con.executemany(
            "INSERT INTO run_item(run_id,source_version_id) VALUES (?,?)", [(run_id, v) for v in version_ids]
        )
        self.con.execute("COMMIT")
        return run_id

    def status(self) -> dict[str, Any]:
        q = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "source_files": q("SELECT COUNT(*) FROM source_file"),
            "source_versions": q("SELECT COUNT(*) FROM source_version"),
            "malformed": q("SELECT COUNT(*) FROM source_version WHERE parse_status='malformed' AND superseded_by IS NULL"),
            "corrupt": q("SELECT COUNT(*) FROM source_version WHERE integrity!='ok' AND superseded_by IS NULL"),
            "pending_stability": q("SELECT COUNT(*) FROM pending_file"),
            "missing": q("SELECT COUNT(*) FROM source_file WHERE missing_since IS NOT NULL"),
            "runs": q("SELECT COUNT(*) FROM run"),
            "queued_items": q("SELECT COUNT(*) FROM run_item WHERE processing_state='queued'"),
            "unacknowledged_notifications": q("SELECT COUNT(*) FROM notification WHERE acknowledged=0"),
        }
