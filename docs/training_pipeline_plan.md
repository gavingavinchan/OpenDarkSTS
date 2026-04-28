# Training Pipeline Plan

## Hierarchy

This document is the authoritative plan for the training side of OpenDarkSTS.

Planning hierarchy:

1. `docs/grand_plan_v2.md` describes the whole project and the split between the training pipeline and the operational detection pipeline.
2. This document describes the training pipeline end to end.
3. `docs/visual_review_plan.md` describes step 2 of this training pipeline in detail.
4. `docs/visual_review_file_storage_plan.md` describes the file-backed review storage used by step 2 and consumed by later training stages.

The operational/deployment pipeline is separate. It consumes promoted trained weights and model metadata produced by this training pipeline, but it does not own import, review, curation, augmentation, training, or model evaluation.

## Goal

Build a reproducible GUI-driven pipeline for creating, curating, training, and evaluating YOLO datasets for ship and STS detection.

The intended flow is:

```text
Import / Download Data
  -> Auto Detect + Manual Review
  -> Reviewed BBox Overview
  -> Dataset Selection
  -> Augmentation + Training Setup
  -> Train YOLO(s)
  -> Evaluate YOLO(s)
```

The GUI should show a clickable top-center progress bar for these stages. Clicking a stage returns the user to that stage while preserving the previous UI state when possible.

## Global GUI Navigation

The training GUI should have one global pipeline progress bar centered at the top of the screen.

Stages:

1. `Import Data`
2. `Manual Review`
3. `BBox Overview`
4. `Dataset Selection`
5. `Training Setup`
6. `Train YOLO`
7. `Evaluate`

Rules:

- The progress bar is clickable.
- The current stage is visually emphasized.
- Completed stages are visually marked.
- Disabled stages are allowed when prerequisites are missing.
- Returning to an earlier stage must not discard later-stage drafts unless the user explicitly chooses to regenerate them.
- The existing bottom-center per-image bbox review progress bar remains scoped only to step 2 manual review.

The top progress bar tracks the whole training pipeline. The bottom review bar tracks bbox progress inside one image.

## Step 1. Import / Download Data

Purpose: prepare just enough source imagery metadata/cache so step 2 manual review can run reliably without modifying original files.

This is not intended to become a large standalone ingestion platform in the first training-pipeline version. Step 2 is already mostly implemented, so step 1 exists mainly to feed step 2 with stable image identity, browser-viewable cache files, and reproducibility metadata.

Inputs may come from:

- read-only local source folders
- Sentinel Hub downloads
- existing GeoTIFF, TIFF, PNG, or JPEG image collections
- future image collections with arbitrary unique filenames

Original images must remain untouched. The import step never edits, renames, moves, or deletes source files.

Normalization means creating workspace artifacts and metadata, not changing source data.

Required minimal normalization:

- assign a stable `image_key` based on the unique review image filename
- record original source path, size, mtime or content hash, image dimensions, and loader settings
- render a browser-friendly RGB image under `workspace/cache/` for review UI use
- preserve the source image's visual geometry: no stretching, rotation, crop, or arbitrary rescale during import
- record band/render choices for GeoTIFFs, such as true color or false color band mapping
- standardize internal bbox coordinates as normalized YOLO-style values: `x_center`, `y_center`, `width`, `height`, all in `0..1`
- store reproducibility metadata in a manifest under `workspace/manifests/`

Allowed workspace transformations:

- GeoTIFF/TIFF to PNG render for browser viewing
- RGB conversion for UI compatibility
- cached thumbnails or preview images
- deterministic tile/image grouping metadata

Disallowed transformations at import:

- editing source files
- overwriting source files
- destructive image cleanup
- lossy resizing as the only retained training source
- baking augmentation into the source dataset

Output artifacts:

- `workspace/manifests/<training_import_manifest>.json`
- `workspace/cache/<imported_images>/...`
- initial per-image review files under `workspace/reviews/<reviewer>/...` after auto detection is run

First-version scope:

- support the image formats and source layouts needed by the existing visual reviewer
- avoid adding download-management complexity unless the current review workflow needs it
- prefer small extensions to `scripts/preprocess_sentinel_tiles.py` and `scripts/visual_review_lib.py` over a separate import subsystem

## Step 2. Auto Detect + Manual Review

Purpose: run one or more YOLO models to create initial boxes, then have a human correct each image.

This is the workflow currently implemented by the visual detection reviewer and documented in `docs/visual_review_plan.md`.

Responsibilities:

- run configured YOLO model(s) on imported review images
- merge duplicate detections using class-aware rules
- write one review JSON file per image
- let the reviewer add, delete, resize, relabel, and reject boxes
- keep `neither` boxes as false-positive signal in review JSONs
- preserve review progress so interrupted work can resume

The reviewer remains the source of truth for corrected bbox labels. Exported YOLO labels are generated later from review JSONs.

### Background Crop Generation

Every time the reviewer finishes reviewing or updating a bbox, the app should enqueue a background crop job for that bbox.

Crop purpose:

- power the step 3 bbox overview grid
- avoid repeatedly cropping the full source image during grid browsing
- give the user fast review of individual corrected boxes

Crop rules:

- crops are generated from the cached review image, not by modifying the original image
- crops are stored under `workspace/cache/review_crops/<image_key>/<detection_id>.png`
- crop metadata is stored separately, including source image key, detection id, bbox geometry, crop padding, source image dimensions, and generation timestamp
- stale crops are regenerated when bbox geometry, class, or source image fingerprint changes
- crop jobs must be safe to rerun

The crop should show the reviewed bbox with enough padding to provide visual context. The exact padding should be configurable.

## Step 3. Reviewed BBox Overview

Purpose: let the human do one more pass over all reviewed bboxes before dataset selection.

This stage shows a grid view similar to Windows Explorer or macOS Finder icon/grid mode.

Grid behavior:

- the grid shows bbox crop images, not full source images
- each tile represents one reviewed bbox crop
- tiles show minimal visual status: class, dataset membership if any, review status, and warning badges
- single-click selects one tile and updates the right sidebar
- double-click opens the original image in step 2 manual review at the corresponding bbox
- a right-sidebar `Edit In Manual Review` button performs the same action as double-click

Right sidebar information:

- source image filename and `image_key`
- detection id
- reviewed class: `ship`, `ships`, or `neither`
- reviewer
- confidence and source model provenance when available
- bbox dimensions and normalized coordinates
- manual vs auto source type
- current dataset memberships
- warnings such as stale crop, incomplete review, missing source cache, or unusual bbox size

Returning from manual review:

- when the user jumps from step 3 to step 2, the GUI stores the step 3 state
- stored state includes filters, sort order, scroll position, selected crop ids, current image, and sidebar selection
- clicking `BBox Overview` in the top progress bar returns to step 3 at the same grid state
- any edited bboxes should regenerate their crop thumbnails in the background

This stage does not decide final training membership by itself. It is a visual quality-control pass before dataset selection.

## Step 4. Dataset Selection

Purpose: choose which reviewed images or bboxes belong to which training dataset versions.

The user may create multiple datasets. A dataset is a named selection plus export settings, not just a folder of copied files.

Selection behavior should follow familiar desktop and mobile conventions:

- single-click selects one item
- `Ctrl`/`Cmd` click toggles individual items
- `Shift` click selects a range
- `Ctrl+A`/`Cmd+A` selects all visible filtered items
- toolbar buttons provide `Select All`, `Clear Selection`, and `Invert Selection`
- touch/mobile mode supports long-press to enter selection mode, tap to toggle, and a visible `Select All` action

Dataset assignment behavior:

- selected items can be added to one or more datasets
- selected items can be removed from one or more datasets
- the UI should make it clear whether assignment is by whole image or by individual bbox
- the first version should prefer image-level assignment, because standard YOLO datasets include one label file per image
- bbox-level inclusion can be represented as an advanced option by filtering labels within selected images

Dataset records should be stored as reproducible selection manifests, for example:

```text
workspace/dataset_selections/<dataset_name>.json
```

Each dataset selection manifest should record:

- dataset name and version
- selected image keys
- optional selected detection ids if bbox-level filtering is enabled
- review manifest fingerprint or per-image preprocess fingerprints
- selection filters used to create the dataset
- reviewer and timestamp
- class inclusion rules
- train/val/test split policy

Exporting a dataset should produce a standard YOLO dataset under:

```text
workspace/datasets/<dataset_name>/
  images/
  labels/
  data.yaml
  dataset_manifest.json
```

The YOLO `images/`, `labels/`, and `data.yaml` remain standard. Reproducibility metadata goes in `dataset_manifest.json`.

## Step 5. Augmentation + Training Setup

Purpose: choose which dataset and training settings will be used for one or more YOLO runs.

The user must select:

- dataset name/version
- model architecture or starting weights
- train/val/test split or existing split manifest
- image size
- epochs
- batch size
- patience
- seed
- device
- workers
- augmentation settings
- output run name

Augmentation should be stored as configuration, not pre-applied to source images. YOLO/Ultralytics can apply augmentation during training. If offline augmentation is later needed, it should write derived images into a new derived dataset with its own manifest.

Training setup output:

```text
workspace/training_configs/<run_group>.json
```

The config should include every parameter needed to reproduce the training command.

## Step 6. Train YOLO(s)

Purpose: run one or more training jobs from the selected dataset and training config.

Training may include multiple models, for example YOLOv8n and YOLO26n.

Required reproducibility metadata:

- exact training config JSON
- dataset manifest path and hash
- data.yaml path and hash
- starting weights path and hash when available
- model architecture name
- package versions, including `ultralytics`, `torch`, Python, CUDA if available
- git commit hash and dirty-worktree flag
- command line used to start training
- start time, end time, and host/device summary
- output run directory

Training outputs should live under `workspace/runs/` or another configured workspace directory, never in the read-only source data folder.

Each completed training run should produce a run metadata file, for example:

```text
workspace/runs/<run_group>/<run_name>/training_metadata.json
```

## Step 7. Evaluate YOLO(s)

Purpose: compare trained models and decide whether any weights should be promoted for downstream use.

Evaluation should include:

- validation metrics
- test metrics
- class-level metrics for `ship` and `ships`
- precision, recall, mAP50, and mAP50-95
- confusion matrix when available
- representative prediction images
- false-positive and false-negative review notes when available
- comparison against current baseline weights

Evaluation outputs:

```text
workspace/reports/<run_group>_evaluation.json
workspace/reports/<run_group>_evaluation.md
```

Promoted weights should be copied or referenced under:

```text
workspace/model_weights/
```

The evaluation report must state why a model was or was not promoted.

## Reproducibility Contract

A training result is reproducible only if these artifacts are present:

- import manifest
- per-image review JSONs used for the dataset
- dataset selection manifest
- exported dataset manifest
- training config
- training metadata
- evaluation report
- final/promoted weight path

Source images remain external inputs and are never modified by the pipeline.

## Relationship To Existing Code

Current code already covers much of step 2 and part of dataset export:

- `scripts/preprocess_sentinel_tiles.py` supports import, render cache, YOLO inference, and review stub creation.
- `scripts/review_visual_detections_ui.py` supports manual bbox review.
- `scripts/visual_review_lib.py` stores review JSON state and provides export helpers.
- `scripts/export_reviewed_labels.py` exports completed review JSONs to YOLO format.
- `scripts/compare_yolo_models.py` trains and evaluates YOLOv8n and YOLO26n, but should eventually consume step 5 training configs and emit the step 6 metadata described here.

Missing higher-level pieces:

- a unified training GUI shell with the top progress bar
- background reviewed-bbox crop generation
- step 3 bbox overview grid
- step 4 dataset selection manifests and multi-dataset assignment UI
- step 5 training setup UI and config writer
- step 6 reproducibility metadata writer
- step 7 evaluation report UI/report generator
