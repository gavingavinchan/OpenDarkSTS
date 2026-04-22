# Visual Review File Storage Plan

## Goal

Replace the visual review SQLite database with a file-based review format that works cleanly across multiple collaborators and produces a complete reviewed dataset for YOLO training.

The main requirement is simple collaboration:

- multiple people review different images on different machines
- each person saves work independently
- completed work can be recombined later without database merging
- the final merged review set can be exported directly to YOLO labels

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
- merged raw detections
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
- active/inactive state
- manual edits
- provenance needed for merge validation

## Stable Image Identity

Every image must have a stable `image_key` that is consistent across machines.

Recommended rule:

- use the review image filename itself as the primary identity, typically the filename stem
- examples:
  - `2020-01-04_tile_1_0.png` -> `image_key = 2020-01-04_tile_1_0`
  - `houston_pass_014.png` -> `image_key = houston_pass_014`
  - `sentinel_scene_A17.png` -> `image_key = sentinel_scene_A17`

This avoids baking tile assumptions into the collaboration model. Some datasets may be tiled, while others may be standalone images with arbitrary but unique filenames.

If needed, include a stronger fingerprint in metadata:

- source tile paths
- source tile mtimes or hashes
- width/height
- original image filename

The key rule is that collaborators must generate the same `image_key` for the same review image.

Do not use local auto-increment database IDs as the collaboration identity.

## Review File Schema

Each file should contain at minimum:

```json
{
  "schema_version": 1,
  "image_key": "2020-01-04_tile_1_0",
  "image_filename": "2020-01-04_tile_1_0.png",
  "reviewer": "gavin",
  "manifest_version": "2026-04-22",
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
      "is_active": true,
      "reviewed": true,
      "review_order": 4,
      "reviewed_at": "2026-04-22T18:28:00Z",
      "source_models": []
    }
  ]
}
```

Notes:

- `detection_id` should become a stable per-image string, not a DB row ID
- raw-model provenance can be kept as metadata, but YOLO export will only use active reviewed detections
- `review_status` still supports `overview_1`, `detail`, `overview_2`, `completed`
- `detail_cursor` is persisted so reloading resumes the exact review step
- `image_filename` should preserve the actual unique review image filename even when the dataset is not tile-based

## What Gets Saved

Persist in each image file:

- all active reviewed detections
- all active unreviewed detections
- inactive `neither` detections if we want review history and auditability
- manual additions
- geometry edits
- review progress state

Do not persist only final labels. We need the in-progress state too so interrupted work can resume and be handed off if needed.

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

## Final YOLO Export

The final export step should read directly from per-image review files.

For each image file:

- include only detections with `is_active = true` and review labels `ship_non_sts` or `ships_sts`
- exclude detections labeled `neither`
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

The simpler rule is better: one image, one reviewer, one review file.
