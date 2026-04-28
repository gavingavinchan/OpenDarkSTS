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

## Artifact Layout

New training-pipeline artifacts should live under `workspace/` and follow this layout:

```text
workspace/
  cache/
    review_crops/
      <image_key>/
        <detection_id>.png
        <detection_id>.json

  dataset_selections/
    <dataset_name>.json

  datasets/
    <dataset_name>/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      data.yaml
      dataset_manifest.json

  training_configs/
    <run_group>.json

  runs/
    <run_group>/
      <run_name>/
        training_metadata.json
        ...

  reports/
    <run_group>_evaluation.json
    <run_group>_evaluation.md

  model_weights/
    <promoted_weights>.pt
```

The read-only source data folders remain external inputs and must not be modified.

Every JSON artifact created by the training pipeline should include `schema_version: 1` unless it is a strict third-party format such as YOLO `data.yaml`. This applies to import manifests, crop sidecars, dataset selection manifests, exported dataset manifests, training configs, training metadata, and evaluation reports.

## Label Vocabulary

The training pipeline uses three related label vocabularies. Implementations must use this mapping explicitly instead of comparing display strings ad hoc.

| Internal review label | YOLO class id | YOLO class name | Export behavior |
|---|---:|---|---|
| `ship_non_sts` | `0` | `ship` | Positive YOLO label |
| `ships_sts` | `1` | `ships` | Positive YOLO label |
| `neither` | none | none | Excluded from positive labels; optional background-negative signal |

Keep the existing YOLO names `ship` and `ships` for compatibility with current datasets, weights, docs, and scripts.

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
- Prior stages may be visually marked as already visited, but the UI does not need strict stage-complete semantics.
- Stage actions are gated by prerequisites. For example, `Run Training` stays grayed out until a dataset and valid training config are selected, then turns blue when runnable.
- Disabled stages or actions are allowed when prerequisites are missing.
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

- assign a stable portable `image_key` based on source content and render profile, not filename alone
- record original source path, source collection, original filename, size, mtime, content hash, image dimensions, and loader settings
- render a browser-friendly RGB image under `workspace/cache/` for review UI use
- preserve the source image's visual geometry: no stretching, rotation, crop, or arbitrary rescale during import
- record band/render choices for GeoTIFFs, such as true color or false color band mapping
- standardize internal bbox coordinates as normalized YOLO-style values: `x_center`, `y_center`, `width`, `height`, all in `0..1`
- store reproducibility metadata in a manifest under `workspace/manifests/`

Image identity rule:

- canonical `image_key` should be deterministic across machines and collaborators
- use a content hash as the stable identity base, with enough render-profile information to distinguish different review images made from the same source file, such as true color versus false color renders
- keep human-readable source collection, original filename, and original path as metadata, not as the primary join key
- review JSONs, crop caches, dataset selections, and training manifests should all join on this stable `image_key`

Exact first-version formula:

```text
image_key = img_<sha256(canonical_identity_payload)[:16]>
```

Where `canonical_identity_payload` is canonical JSON containing:

```json
{
  "source_content_hash": "sha256:<hash of original source file bytes>",
  "render_profile": {
    "kind": "true_color",
    "bands": [1, 2, 3],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB"
  }
}
```

Canonical JSON encoding for the identity payload is:

- sorted object keys
- no whitespace, using separators `(',', ':')`
- UTF-8 bytes
- no trailing newline

Python reference implementation:

```python
identity_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
image_key = "img_" + hashlib.sha256(identity_bytes).hexdigest()[:16]
```

For normal RGB PNG/JPEG inputs, the render profile can be `rgb_original`. For GeoTIFF/Sentinel inputs, the render profile distinguishes true color, false color, band choices, and normalization choices.

RGB PNG/JPEG payload example:

```json
{
  "source_content_hash": "sha256:<hash of original source file bytes>",
  "render_profile": {
    "kind": "rgb_original"
  }
}
```

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

- `workspace/manifests/training_import_manifest.json`
- `workspace/cache/<imported_images>/...`
- initial per-image review files under `workspace/reviews/<reviewer>/...` after auto detection is run

First-version scope:

- support the image formats and source layouts needed by the existing visual reviewer
- avoid adding download-management complexity unless the current review workflow needs it
- extend the existing preprocess/review helpers rather than replacing them with a separate import subsystem

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

`detection_id` must be unique within one `image_key` across all auto and manual detections. Auto detections from multiple configured models should be merged first, then assigned monotonically increasing ids such as `auto-0001` through `auto-NNNN`; manual detections use their own monotonic sequence such as `manual-0001`.

### Background Crop Generation

Every time the reviewer finishes reviewing or updating a bbox, the app should enqueue a background crop job for that bbox.

Crop purpose:

- power the step 3 bbox overview grid
- avoid repeatedly cropping the full source image during grid browsing
- give the user fast review of individual corrected boxes

Crop rules:

- crops are generated from the cached review image, not by modifying the original image
- crops are stored under `workspace/cache/review_crops/<image_key>/<detection_id>.png`
- each crop has a sidecar metadata file at `workspace/cache/review_crops/<image_key>/<detection_id>.json`
- crop metadata is stored separately, including source image key, detection id, bbox geometry, crop padding, source image dimensions, and generation timestamp
- stale crop pixels are regenerated when bbox geometry or source image fingerprint changes
- class changes refresh crop metadata and UI badges, but do not require regenerating crop pixels
- crop jobs must be safe to rerun

The crop should show the reviewed bbox with enough padding to provide visual context. The exact padding should be configurable.

Crop metadata sidecar shape:

```json
{
  "schema_version": 1,
  "image_key": "img_abc123",
  "detection_id": "auto-0007",
  "review_file": "workspace/reviews/gavin/img_abc123.json",
  "crop_path": "workspace/cache/review_crops/img_abc123/auto-0007.png",
  "source_cache_path": "workspace/cache/sentinel_tiles/example.png",
  "source_image_width": 1576,
  "source_image_height": 1476,
  "bbox": {
    "x_center": 0.51,
    "y_center": 0.44,
    "width": 0.03,
    "height": 0.02
  },
  "crop_pixels": {
    "x1": 740,
    "y1": 620,
    "x2": 850,
    "y2": 710,
    "padding_px": 48
  },
  "review": {
    "review_label": "ship_non_sts",
    "class_id": 0,
    "class_name": "ship",
    "source_type": "auto",
    "reviewed": true
  },
  "fingerprints": {
    "source_image": "sha256:...",
    "bbox_geometry": "sha256:...",
    "crop_recipe": "sha256:..."
  },
  "generated_at": "2026-04-28T00:00:00Z"
}
```

### Reviewer Reconciliation Before Overview

Step 3 should show one combined set of reviewed boxes without making reviewer identity central in the UI.

First-version reconciliation rules:

- if an `image_key` has exactly one completed review file, use it directly
- if multiple reviewers completed the same `image_key`, flag the image as a reconciliation conflict instead of silently merging boxes
- conflicted images can appear in step 3 under a `Needs Reconciliation` filter, but should not be exportable into a training dataset until resolved
- later versions may add IoU-based merge tools for duplicate boxes, but the first version should avoid silent label corruption

## Step 3. Reviewed BBox Overview

Purpose: let the human do one more pass over all reviewed bboxes before dataset selection.

This stage shows a grid view similar to Windows Explorer or macOS Finder icon/grid mode.

Step 3 should be implemented inside the same local UI/server family as the existing visual reviewer, not as a separate product. The first full-training GUI entry point should be `scripts/training_pipeline_ui.py`, with the existing reviewer reused for step 2 and shared logic kept in helper libraries.

Grid behavior:

- the grid shows bbox crop images, not full source images
- each tile represents one reviewed bbox crop
- tiles show minimal visual status: class, dataset membership if any, review status, and warning badges
- tile size is adjustable, similar to Windows Explorer icon size controls
- large grids must use lazy loading or virtualization so many thousands of bbox crops remain usable
- filters and sorting should apply before tile rendering when possible
- single-click selects one tile and updates the right sidebar
- double-click opens the original image in step 2 manual review at the corresponding bbox
- a right-sidebar `Edit In Manual Review` button performs the same action as double-click

Right sidebar information:

- source image filename and `image_key`
- detection id
- reviewed class: `ship`, `ships`, or `neither`
- reviewer, derived from the review file's `reviewer` field or parent `workspace/reviews/<reviewer>/` directory
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

Training-mode clarification:

- bbox crops shown here are review UI artifacts by default
- standard YOLO training uses whole review images plus per-image YOLO label files
- selecting or inspecting a crop should therefore carry its source image context forward to step 4
- crop-level training images are allowed only as explicit derived-dataset options, mainly for hard-negative mining

## Step 4. Dataset Selection

Purpose: choose which reviewed images or bboxes belong to which training dataset versions.

The user may create multiple datasets. A dataset is a named selection plus export settings, not just a folder of copied files.

The default dataset name should be `dataset_1` until the user changes it.

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
- the first version should prefer image-level assignment, because standard YOLO datasets include one label file per whole image
- bbox-level selection can be represented as an advanced label-filtering option within selected whole images
- step 3 crop selections should carry forward into this stage as candidate source-image selections, not as an automatic switch to crop training
- train/val/test split ownership lives in this stage; later training stages reference the selected dataset and split rather than redefining them

`neither` handling:

- `neither` boxes are excluded from normal YOLO positive label files by default
- step 4 controls how much false-positive signal enters training
- first-version hard-negative support should prefer image-level background negatives: whole images with no exported `ship` or `ships` labels can be included with an empty label file
- crop-level `neither` hard negatives may be added later as an explicit derived-dataset option, where each selected `neither` crop is emitted as a labelless training image

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

`include_labels` enum values for the first version:

- `all_exportable` means every reviewed positive box on this image is exported
- `selected_only` means only detection ids listed in `selected_detection_ids` are exported

Reconciliation conflicts are a hard export blocker. Images marked `Needs Reconciliation` must not be exported into any training dataset until resolved; this is not a per-dataset option.

First-version split rule:

- default to automatic split assignment with `80%` train, `10%` val, and `10%` test
- allow manual override so the user can move images between splits
- persist the final explicit split per image in the dataset selection manifest
- re-exporting a dataset uses the persisted split, not a fresh random split
- when new images are added to an existing dataset, auto-assign only the new images according to the default split ratios and preserve all existing image split assignments and manual overrides
- the UI should note how many new images were auto-split when this happens

Example selection manifest:

```json
{
  "schema_version": 1,
  "dataset_name": "sts_reviewed_v1",
  "dataset_version": 1,
  "created_at": "2026-04-28T00:00:00Z",
  "updated_at": "2026-04-28T00:00:00Z",
  "created_by": "gavin",
  "description": "First reviewed Sentinel STS training set",
  "selection_mode": "image",
  "classes": {
    "0": "ship",
    "1": "ships"
  },
  "sources": {
    "reviews_dir": "workspace/reviews",
    "manifest": "workspace/manifests/training_import_manifest.json"
  },
  "split_policy": {
    "mode": "auto_with_manual_override",
    "default_ratios": {
      "train": 0.8,
      "val": 0.1,
      "test": 0.1
    },
    "seed": 42
  },
  "items": [
    {
      "image_key": "img_abc123",
      "split": "train",
      "include_labels": "all_exportable",
      "selected_detection_ids": null
    }
  ],
  "background_negatives": {
    "enabled": true,
    "mode": "whole_image_empty_label",
    "max_fraction": 0.1,
    "image_keys": ["img_bg001"]
  },
  "export": {
    "output_dir": "workspace/datasets/sts_reviewed_v1",
    "include_incomplete_reviews": false
  }
}
```

Exporting a dataset should produce a standard YOLO dataset under:

```text
workspace/datasets/<dataset_name>/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
  dataset_manifest.json
```

The YOLO `images/`, `labels/`, and `data.yaml` remain standard. Reproducibility metadata goes in `dataset_manifest.json`.

`dataset_manifest.json` should reference the dataset selection manifest and record the concrete exported files and hashes. It should not become a second independent source of selection truth.

First-version export rules:

- export whole review images as YOLO detection images
- write one YOLO label file per exported image
- `ship_non_sts` exports as class `0`
- `ships_sts` exports as class `1`
- `neither` never exports as a positive label
- selected background-negative images export as whole images with empty label files
- if `background_negatives.image_keys` and `max_fraction` are both set, include selected image keys up to the fraction cap where `max_fraction` is the maximum share of exported images that may be background negatives; truncate any oversubscription and record the warning in `dataset_manifest.json`
- bbox crop images are not training inputs unless a later explicit derived-dataset option is enabled

## Step 5. Augmentation + Training Setup

Purpose: choose which dataset and training settings will be used for one or more YOLO runs.

The user must select:

- dataset name/version
- model architecture or starting weights
- image size
- epochs
- batch size
- patience
- seed
- device
- workers
- augmentation settings
- output run name

The selected dataset already owns the train/val/test split. Step 5 may display the split and validate that it exists, but should not define a conflicting split policy.

Default training values depend on the dataset and should be decided later. The first implementation should make the fields explicit in the UI rather than hard-coding final project defaults.

Augmentation should be stored as configuration, not pre-applied to source images. YOLO/Ultralytics can apply augmentation during training. The UI should expose all supported augmentation parameters, with sensible defaults and an advanced section for less common knobs. If offline augmentation is later needed, it should write derived images into a new derived dataset with its own manifest.

Training setup output:

```text
workspace/training_configs/<run_group>.json
```

The config should include every parameter needed to reproduce the training command.

`run_group` rule:

- one `run_group` corresponds to one training config
- a `run_group` may contain multiple `run_name` entries, for example one each for YOLOv8n and YOLO26n
- default `run_group` should be derived from the dataset name and timestamp, unless the user edits it
- `run_name` identifies one concrete model run inside that group

### Run Training Readiness Gate

The `Run Training` button should stay grayed out until the run is ready. When disabled, the UI must show a clear checklist of what is still missing or invalid.

Required before `Run Training` is enabled:

- a dataset selection manifest exists
- the selected dataset has been exported to YOLO format
- exported dataset contains `data.yaml`
- exported dataset contains `images/train/`, `images/val/`, `labels/train/`, and `labels/val/`
- train split has at least one image
- val split has at least one image
- `data.yaml` defines class `0: ship` and class `1: ships`
- no unresolved `Needs Reconciliation` images are included in the selected dataset
- required training config fields are present: dataset name/version, model or starting weights, image size, epochs, batch, device, seed, output run name, and augmentation settings
- output run name is safe and will not overwrite an existing run unless the user explicitly enables overwrite

The test split is recommended but not required to start training. If no test split exists, step 7 evaluation should warn that final test metrics are unavailable.

## Step 6. Train YOLO(s)

Purpose: run one or more training jobs from the selected dataset and training config.

Training may include multiple models, for example YOLOv8n and YOLO26n.

When training starts, it should run as a background process so the browser UI and LLM/terminal session remain usable. The UI should create an active run card below the top progress bar, aligned to the right side of the main content area, that remains visible even if the user navigates to another stage.

Run states:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

While a run is active, the UI should show:

- run name
- model or starting weights
- dataset name/version
- current epoch when available
- elapsed time
- latest metrics when available
- log tail
- live performance/loss plots over epoch

Live plots should be read from the run directory's Ultralytics `results.csv` when available and refreshed every few seconds while the run state is `running`.

Preferred plotted series when available:

- `train/box_loss`
- `train/cls_loss`
- `train/dfl_loss`
- `val/box_loss`
- `val/cls_loss`
- `val/dfl_loss`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

If `results.csv` does not exist yet, the plot area should show `Waiting for first epoch`. If columns differ across YOLO versions, plot whichever recognized columns are available.

Running constraints:

- do not allow editing the dataset/config attached to the active run
- allow safe navigation to other stages while training continues
- write logs under the run directory, such as `workspace/runs/<run_group>/<run_name>/train.log`
- `Stop Training` should attempt best-effort termination of the background process, mark run metadata as `cancelled`, and leave partial run files in place for debugging; the final checkpoint may be partial or unusable
- when training completes, enable the evaluation stage and show a `Go To Evaluation` action

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
- comparison against current baseline weights in `workspace/model_weights/`

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

Baseline weights:

- current baseline weights live under `workspace/model_weights/`
- `docs/progress.md` is the human-readable record of current best weights per architecture
- every promoted-weights decision should update `docs/progress.md` so the human-readable baseline record does not drift
- a machine-readable baseline registry can be added later if the GUI needs automatic baseline selection

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
- `scripts/compare_yolo_models.py` trains and evaluates YOLOv8n and YOLO26n on whole-image YOLO detection datasets, but should eventually consume step 5 training configs and emit the step 6 metadata described here.

Planned first implementation structure:

- `scripts/training_pipeline_ui.py` owns the top progress bar and routes between training stages.
- `scripts/review_visual_detections_ui.py` remains useful for focused step 2 manual-review testing, and the mostly-working step 2 reviewer should be left alone except for integration hooks needed by the training pipeline.
- `scripts/visual_review_lib.py` keeps shared review, bbox, crop, and review-file helpers.
- `scripts/training_pipeline_lib.py` should hold new dataset-selection, export, training-config, training-metadata, and evaluation helpers.

The implementation can attempt the full training flow in one pass, but it should still be built as modular helper functions and stage-specific UI paths so failures are debuggable.

Missing higher-level pieces:

- a unified training GUI shell with the top progress bar
- background reviewed-bbox crop generation
- step 3 bbox overview grid
- step 4 dataset selection manifests and multi-dataset assignment UI
- step 5 training setup UI and config writer
- step 6 reproducibility metadata writer
- step 7 evaluation report UI/report generator
