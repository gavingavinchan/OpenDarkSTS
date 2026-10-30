# Visual Review File Storage Plan

This document is the storage sub-plan for the manual visual review stage of `docs/training_pipeline_plan.md`.

The training pipeline plan is authoritative for the overall training flow and later dataset selection, training, and evaluation artifacts. This document is authoritative for per-image review JSON storage, collaboration rules, and how reviewed detections are exported or consumed by those later stages.

## Goal

Replace the visual review SQLite database with a file-based review format that works cleanly across multiple collaborators and produces a complete reviewed dataset for YOLO training.

The main requirement is simple collaboration:

- multiple people review different images on different machines
- each person saves work independently
- completed work can be recombined later without database merging
- the final merged review set can be exported directly to YOLO labels

The review format must also preserve per-location false-positive signal. When a reviewer rejects an auto detection with the `neither` label, that detection is not a YOLO class and is not exported as a label, but its position and the fact that the model was wrong at that location must be retained in the review file. This signal informs later dataset curation — deciding which reviewed images to include in the training set. Dataset curation reads per-image review JSONs directly; the export step does not emit a separate sidecar report.

## Decision

Do not use a shared or primary SQLite database for visual review state.

Instead, persist review state as one metadata file per image.

This per-image file becomes the source of truth for:

- review status
- progress/cursor state
- reviewed detections
- manual additions/deletions
- final export to YOLO labels

SQLite is not needed for the long-term design. If any temporary cache remains during migration, it should be treated as disposable implementation detail, not durable review state.

## Storage Model

### 1. Preprocess Manifest

Keep a global preprocess manifest under:

- `workspace/manifests/visual_preprocess_manifest.json`

This manifest defines the shared image universe and reproducibility settings:

- source tile paths
- tile layout
- stitched/cache image paths
- image dimensions
- model weights
- inference settings
- a stable `image_key` per image

The manifest is generated once from preprocessing and is shared with all collaborators.

### 2. One Review File Per Image

Persist one review JSON file per image under:

- `workspace/reviews/<reviewer>/<image_key>.json`

Example:

- `workspace/reviews/gavin/2020-01-04_tile_1_0.json`
- `workspace/reviews/gavin/houston_pass_014.json`

Each review file is self-contained for that image.

It stores:

- image identity
- review stage/status
- detail cursor
- all current detections for that image
- review labels
- manual edits
- provenance needed for merge validation

## Stable Image Identity

Every image must have a stable `image_key` that is consistent across machines.

Recommended rule:

- use a content-hash-based key as the canonical identity, not the filename alone
- include enough render-profile information in the key input to distinguish different review images derived from the same source file, such as true color versus false color renders
- keep source collection, original filename, and original path as metadata for human traceability

This avoids baking tile assumptions or local folder naming into the collaboration model. Some datasets may be tiled, while others may be standalone images with arbitrary filenames that can collide across machines.

The key rule is that collaborators must generate the same `image_key` for the same review image.

Do not use local auto-increment database IDs, local absolute paths, or filename-only stems as the collaboration identity.

## Preprocessing Fingerprint

Each review file records a `preprocess_fingerprint` identifying the exact preprocessing inputs it was produced against. This replaces any notion of a single global manifest version.

Fingerprint rule:

- compute a SHA256 over the canonical JSON of the per-image manifest entry
- inputs to the fingerprint include:
  - source tile paths
  - source tile mtimes or content hashes
  - width and height
  - model weights identifier
  - inference settings
- the resulting value is stored on the review file as `preprocess_fingerprint`

Why per-image and not global:

- adding new unreviewed images to the manifest must not invalidate existing completed reviews
- re-running detection on one image must only affect that image's fingerprint

Merge validation:

- on merge, each review file is checked against the current manifest's fingerprint for its `image_key`
- a mismatch flags the review for reconciliation rather than silently trusting stale state

## Review File Schema

Each file should contain at minimum:

```json
{
  "schema_version": 1,
  "image_key": "img_3f2a9c8eb814d9f1_truecolor",
  "image_filename": "2020-01-04_tile_1_0.png",
  "source_collection": "sentinel2_houston_2020",
  "source_content_hash": "sha256:3f2a9c8eb814d9f1...",
  "reviewer": "gavin",
  "preprocess_fingerprint": "sha256:3f2a9c8e...",
  "date": "2020-01-04",
  "tile_row": 1,
  "tile_col": 0,
  "width": 1576,
  "height": 1476,
  "cache_path": "workspace/cache/sentinel_tiles/2020-01-04_tile_1_0.png",
  "review_status": "detail",
  "detail_cursor": 3,
  "updated_at": "2026-04-22T18:30:00Z",
  "detections": [
    {
      "detection_id": "manual-0001",
      "source_type": "manual",
      "parent_detection_id": null,
      "class_id": 0,
      "class_name": "ship",
      "confidence": null,
      "x_center": 0.51,
      "y_center": 0.44,
      "width": 0.03,
      "height": 0.02,
      "review_label": "ship_non_sts",
      "reviewed": true,
      "reviewed_at": "2026-04-22T18:28:00Z",
      "source_models": []
    }
  ]
}
```

Notes:

- `detection_id` is a simple per-image counter such as `auto-0001` or `manual-0001`, unique only within the image file
- `detection_id` is not stable across re-preprocessing — re-running detection regenerates auto ids
- `source_type` is `auto` (produced by model inference) or `manual` (human added); it is retained only as provenance. Once a bbox has `reviewed = true` and an exportable `review_label`, it is ground truth for training whether it was auto-detected or manually drawn.
- auto detections the human rejects are kept in the file with `review_label = "neither"` as per-location false-positive signal for dataset curation; not exported as a YOLO class
- manual detections the user adds and then removes are deleted from the array outright — no audit record is kept for those
- `reviewed` distinguishes "human has looked at this box" from "human has assigned a label"; a box can be unreviewed with `review_label = null`
- there is no `is_active` flag — inclusion in the array plus `review_label` determine state, using the rules below
- array order is the canonical per-image traversal order, and `detail_cursor` indexes into that array directly
- `review_status` still supports `overview_1`, `detail`, `overview_2`, `completed`
- `detail_cursor` is persisted so reloading resumes the exact review step
- `image_filename` should preserve the actual review image filename for display/export, but it is not the canonical collaboration key
- `source_collection` and `source_content_hash` preserve human traceability and cross-machine identity checks
- `preprocess_fingerprint` is the per-image manifest fingerprint this review was produced against (see Preprocessing Fingerprint)

State rules without `is_active`:

- detection present with `review_label = null` means it is still active in the review queue and has not been labeled yet
- detection present with `review_label = ship_non_sts` or `ships_sts` means it is active and exportable as reviewed ground truth, independent of `source_type`
- auto detection present with `review_label = neither` means the model produced a false positive at that location; the detection is retained as per-location false-positive signal, excluded from the active review queue and from YOLO export
- manual detection removed from the array means it no longer exists and is excluded from both queue and export

## False-Positive Signal

`neither`-labeled detections are retained as a per-location record that the model predicted a ship at that position and a human confirmed it was wrong. They are the primary purpose of the `neither` label, not an audit byproduct.

Intent:

- preserve where and how often each image's model was wrong
- let downstream dataset curation reason about per-image false-positive rate when deciding which images to include in training

Rules:

- `neither` detections keep their original geometry and are never deleted from the review file
- they are not exported as YOLO labels (no YOLO class corresponds to `neither`)
- dataset curation reads the per-image review JSONs directly to assess false-positive profile — the export step does not emit a separate sidecar report

## What Gets Saved

Persist in each image file:

- all current auto and manual detections
- each detection's current `review_label` (including `neither` for auto detections the human rejected)
- each detection's `reviewed` flag
- review progress state (`review_status`, `detail_cursor`)
- geometry edits to any detection
- preprocessing fingerprint so merges can detect staleness

Do not persist only final labels. We need the in-progress state too so interrupted work can resume and be handed off if needed.

Do not split training counts or export eligibility by auto/manual source. For positive labels, the reviewed `review_label` is the ground-truth decision; `source_type` only explains how the bbox entered the review file.

Do not persist an `is_active` flag. Auto detections the human rejects stay in the file with `review_label = "neither"`. Manual detections the user adds then removes are deleted from the array entirely.

Do not persist a `review_order` field. Array order is the single source of truth for per-image traversal order, and `detail_cursor` resumes against that stored order directly.

Do not persist an undo history. Undo is in-memory only (see Undo section).

## Re-run Detections

The review UI exposes a per-image "Re-run detections" button that regenerates auto detections for a single image without affecting any other image's review state.

Behavior when pressed:

- do not silently discard reviewed positive labels. A reviewed `ship_non_sts` or `ships_sts` bbox is ground truth regardless of `source_type`, so re-running detections must either be blocked until the user explicitly discards that review work, or preserve those reviewed positive bboxes as ground-truth annotations before adding fresh auto detections.
- wipe only unreviewed auto detections and auto detections labeled `neither` when regenerating model candidates
- preserve all reviewed positive detections exactly as training labels, whether their `source_type` is `auto` or `manual`
- reset `review_status` to `overview_1`
- reset `detail_cursor` to `0`
- run fresh model inference and populate new auto detections with new `detection_id` values
- recompute and update `preprocess_fingerprint`

This is always an explicit user action. Preprocessing never rewrites review state on its own. Re-running detection on one image is the supported way to pick up new model weights or inference settings for that image.

## Undo

Undo is session-only and in-memory. When the reviewer UI is closed or the user switches away from the current image, the undo stack for that image is discarded.

Rationale:

- most undo use is correcting the most recent mislabel or misedit
- persisting an undo log in the review file would grow the file over time and complicate merges
- worst case a reviewer loses undo history for the single image currently open, which is acceptable

Scope of in-memory undo within a session:

- label changes
- geometry edits
- manual box additions and removals
- detail cursor advances
- review status transitions
- re-run detections on the currently open image

Cross-session undo, cross-image undo, and undo persistence are out of scope.

## Collaboration Workflow

### 1. Shared Inputs

All collaborators must use the same:

- preprocess manifest
- cached image generation recipe
- model weights
- detection merge settings
- schema version

### 2. Split Work

Assign images explicitly using assignment files:

- `workspace/review_assignments/gavin.txt`
- `workspace/review_assignments/alex.txt`

Each file contains one `image_key` per line.

Rules:

- one image belongs to one reviewer at a time
- no overlapping ownership by default
- reassignment means handing off that image file, not merging databases

### 3. Review Locally

The UI loads only assigned images and reads/writes the matching per-image JSON files.

### 4. Recombine Later

To merge collaborator output:

- collect all review JSON files
- validate schema and manifest compatibility
- fail if duplicate completed files exist for the same `image_key`
- combine them into one unified review directory
- export YOLO labels from the merged review files

Step 3 of the training pipeline may show duplicate completed reviews under a `Needs Reconciliation` filter, but duplicate completed reviews for the same `image_key` must not be silently merged into an exportable dataset. The first version should resolve those conflicts manually or by choosing one completed review file.

## Final YOLO Export

The final export step should read directly from per-image review files.

For each image file:

- include only detections whose `review_label` is `ship_non_sts` or `ships_sts`
- exclude detections labeled `neither` or with a null label
- map review labels to YOLO classes exactly as the current code does:
  - `ship_non_sts` -> class `0` (`ship`)
  - `ships_sts` -> class `1` (`ships`)
- write standard YOLO `.txt` labels with lines of the form:
  - `class_id x_center y_center width height`

Output remains:

```text
workspace/datasets/sentinel_reviewed/
  images/
  labels/
  data.yaml
```

Current naming should remain image-key/filename based, not date-only. The existing export code writes files like:

- `images/2020-01-04_tile_1_0.png`
- `labels/2020-01-04_tile_1_0.txt`

In the file-based design, export should produce one PNG and one TXT per reviewed image key, using the image filename directly when it is already unique.

Examples:

- `images/2020-01-04_tile_1_0.png`
- `labels/2020-01-04_tile_1_0.txt`
- `images/houston_pass_014.png`
- `labels/houston_pass_014.txt`

`data.yaml` should continue to define:

- `0: ship`
- `1: ships`

This means the final training dataset is the result of combining reviewed image JSONs and exporting one YOLO image/label pair per reviewed image, keyed by the stable `image_key` derived from the unique filename.

## Why This Is Better Than SQLite

Benefits:

- no database merge conflicts across machines
- one file per image matches the unit of human work
- easy to assign and reassign work
- easy to version in git or copy by folder
- easy to validate duplicates and missing images
- final export path becomes simpler and clearer

Tradeoff:

- the UI must manage file reads/writes directly instead of querying SQLite

That tradeoff is acceptable because collaboration and final YOLO dataset generation matter more than relational querying.

## Migration Plan

### Phase 1. Freeze the Data Model

- finalize the per-image JSON schema
- finalize `image_key` rules
- finalize assignment file format

### Phase 2. Change Preprocessing Output

- stop writing detections to SQLite
- write preprocess manifest plus initial per-image review JSON stubs instead
- each stub contains image metadata and initial merged detections

### Phase 3. Refactor Review UI

- replace DB-backed queue/state reads with file-backed reads
- replace DB mutation endpoints with file writes
- preserve current review workflow and hotkeys
- preserve progress restoration using `review_status` and `detail_cursor` from the image file

### Phase 4. Replace Export Script

- make `export_reviewed_labels.py` read review JSON files instead of SQLite
- export YOLO labels from merged review files only

### Phase 5. Add Collaboration Tools

- add assignment file support
- add validation for duplicate `image_key` reviews
- add optional merge/import helper for collaborator directories

## Immediate Implementation Plan

1. Update the main visual review plan to remove the SQLite database as the primary store.
2. Define the per-image review JSON schema in code.
3. Refactor preprocessing to emit review stubs instead of DB rows.
4. Refactor the reviewer server to load/save image review JSON files.
5. Refactor label export to read from review JSON files.
6. Add assignment filtering so each collaborator sees only their own image list.

## Non-Goals

Not needed for the first file-based version:

- multi-user concurrent editing of the same image
- automatic conflict resolution for duplicate edits
- centralized network database
- cross-session or persistent undo history
- global manifest version strings (replaced by per-image `preprocess_fingerprint`)

The simpler rule is better: one image, one reviewer, one review file.
