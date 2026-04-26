# Visual Detection Reviewer Plan

This is the visual reviewer - a manual review tool for YOLO detections on satellite imagery. Its purpose is to generate corrected training data to improve the STS YOLO model.

The AIS candidate reviewer reviews AIS-derived events. This tool reviews satellite image detections directly.

## Input Data

- **Primary source**: `/home/gavin/Downloads/AIS data/Sentinel 2A/sentinel2/`
- **Additional source**: `/home/gavin/Downloads/AIS data/Sentinel 2A/2020/ship_tiles/`
- **Location**: Off the coast of Houston
- **Format**: RGB GeoTIFF tiles, ~1578x1474 px each, ~3 MB per tile
- **Tiling**: Multiple tiles per date, named `{date}_tile_{row}_{col}.tiff`
- **Volume**: The existing `sentinel2/` subset is 700 tiles across 175 dates; there is also a larger `2020/ship_tiles/` collection with 1,257 additional tiles. Tile count per date varies and should not be assumed to stay at a 2x2 grid.
- Future datasets may also arrive as standalone review images with unique filenames rather than tile grids. The reviewer workflow should support both tiled inputs and already-prepared single images.

## Storage Rules

- The source folder in `/home/gavin/Downloads/AIS data/` is read-only. This workflow must never modify source tiles.
- All generated outputs must live under `workspace/`.
- Do not store regenerable intermediates by default. If a stitched review image or merged detection set can be rebuilt from source tiles plus recorded settings, treat it as cache.
- Persist review state as one metadata file per image plus enough provenance metadata to reproduce cached artifacts.
- The authoritative storage, collaboration, merge, and schema rules live in `docs/visual_review_file_storage_plan.md`.

## YOLO Models

Run both available models on each review image:

1. **YOLOv8n** (rectbbox dataset, 150 max epochs, test mAP50 0.800): `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`
2. **YOLO26n** (rectbbox dataset, 150 max epochs, test mAP50 0.830): `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`

Classes: `ship` (class 0, single vessel), `ships` (class 1, STS pair)

Merge detections from both models using class-aware NMS to avoid duplicates.

Merge rules:

- Run NMS separately per class (`ship` vs `ships`) so different classes do not suppress each other automatically.
- If both models predict the same class on the same object, keep one merged detection and record both source detections in provenance.
- If a `ships` box overlaps one or more `ship` boxes, keep both at preprocess time and let manual review resolve the conflict.
- Manual review must allow converting one `ships` box into two `ship` boxes, or merging two `ship` boxes into one `ships` box.

## Pre-Processing Pipeline

A script (`scripts/preprocess_sentinel_tiles.py`) that:

1. **Discovers tiles per date** from filenames: parse `{date}_tile_{row}_{col}.tiff`, group by date
2. **Builds a per-date render recipe**:
   - Parse row/col indices from filenames
   - Build grid dynamically (do not hardcode 2x2 - future datasets may have different tile counts)
    - Record source tile paths, tile layout, and review image dimensions in a manifest
3. **Renders a browser-ready review image**:
    - Use PNG, not GeoTIFF, for the web UI
    - Store rendered review images under `workspace/cache/sentinel_stitched/` only as a rebuildable cache
    - If the input is tiled, regenerate the cached PNG from the manifest recipe
    - If the input is already a standalone review image, keep or copy it into the cache without forcing a stitched workflow
4. **Runs both YOLO models** on each cached review image
5. **Merges detections** from both models with class-aware NMS (IoU threshold ~0.5, configurable)
6. **Saves results**:
   - Reproducibility metadata to `workspace/manifests/visual_preprocess_manifest.json`
   - Initial per-image review JSON stubs under `workspace/reviews/`

The preprocessing step should not write initial YOLO label files alongside cached images. Those are regenerable and should be exported only after human review.

### Preprocess Manifest: `workspace/manifests/visual_preprocess_manifest.json`

Store reproducibility metadata outside the YOLO label files. At minimum include:

- source tile paths and tile grid layout per date when applicable
- original input image filename when the source is already a standalone image
- source tile mtimes or hashes
- GeoTIFF -> PNG render settings when applicable
- review image dimensions
- model weight paths
- inference thresholds
- class-aware NMS settings

### Review Files

Persist one review JSON file per image under `workspace/reviews/<reviewer>/<image_key>.json`.

These files store:

- image identity and filename-based `image_key`
- review progress state (`review_status`, `detail_cursor`)
- detections and human review labels
- preprocessing fingerprint for merge validation

The full file schema and collaboration rules are defined in `docs/visual_review_file_storage_plan.md`.

## Review Workflow

The reviewer processes one image at a time, in this order:

### Future: Overlap Classification Helper

Many duplicate detections are the same ship or ship group detected as both `ship` and `ships`. A later UI pass should automatically check for overlapping bounding boxes with conflicting classes and show a small side pop-up for convenient classification selection. This helper should make it easy to choose the correct classification for the overlapping boxes without interfering with the existing label buttons, hotkeys, or normal review flow.

### Step 1: Overview Mode (first pass)

Show the full review image with all current bounding boxes overlaid, including `neither` boxes rendered in muted gray so false-positive locations remain visible.

Purpose: quick scan to see if YOLO missed anything obvious.

Controls:
- Zoom and pan freely
- Add new bounding boxes (for missed detections)
- Mark boxes as `neither` to record that the model produced a false positive at that location (removes them from the active queue; retained in the review file as per-location false-positive signal for later dataset curation)
- Click "Start Detail Review" to proceed

### Step 2: Detail Mode (per-detection)

Show one active detection at a time, zoomed in and centered on the bounding box.

For each detection:
- Display the detection zoomed in with surrounding context
- Allow move and resize of the bounding box
- Allow split/replace workflow when one detection should become two boxes or a different class
- Label buttons: `Ships (STS)`, `Ship (non-STS)`, `Neither`
  - `Ships (STS)` -> class 1 (`ships`)
  - `Ship (non-STS)` -> class 0 (`ship`)
  - `Neither` -> label this detection `neither`; excluded from exported YOLO labels, but kept in the review file as per-location false-positive signal for later dataset curation
- Auto-advance to next detection after labeling
- Back button to revisit previous detection
- Use a stable per-image detection order so interrupted reviews can resume exactly where they left off

### Step 3: Overview Mode (second pass)

Return to full image view with updated bounding boxes.

Purpose: final check for anything missed during detail review.

Controls:
- Same as first overview pass
- Any newly added boxes go into the detail queue for labeling
- Click "Complete" to finalize this image and move to the next

If new boxes are added during the second overview pass, the image should return to detail mode for those boxes before it can be marked `completed`.

### Navigation

- Progress indicator: `3 / 175 images reviewed`
- Per-image detection count: `12 detections (8 reviewed)`
- A bbox is considered reviewed after a review decision is saved: `Ships (STS)`, `Ship (non-STS)`, `Neither`, permanent delete, accepting the current displayed class with `Space` / Right arrow / `Accept / Next`, or saving a geometry edit.
- Geometry edits count as a review decision and save the current class for that bbox.
- A manually drawn bbox is saved as reviewed using the currently selected class because drawing it is an explicit human addition.
- An image is considered completed automatically when every active bbox has been reviewed; image navigation does not itself complete an image.
- Bottom-center per-image detection review progress bar with discrete segments, where each segment represents one box in the active review queue for the current image (`neither` boxes are not counted in this bar)
- The per-image bar should be compact and flat, with the `reviewed/total` count such as `5/10` shown on the right side of the bar
- During keyboard review flow, pressing `Space` should accept the selected/current bbox's displayed class when it is still unreviewed; otherwise it advances through detections and updates the per-image progress bar
- The per-image progress bar should be reconstructed from saved review state when the page reloads, so progress does not reset on refresh
- Next/Back to move between images
- Re-opening the UI should resume each image at its saved review stage and detail cursor

## Review Labels

- `ships_sts` - two vessels in STS configuration (YOLO class 1: `ships`)
- `ship_non_sts` - single vessel, not STS (YOLO class 0: `ship`)
- `neither` - the model produced a false positive at this location. `neither` is not a YOLO class and is excluded from exported labels, but the detection and its location are kept in the review file so the image's false-positive profile can inform dataset curation decisions — which images to include in, or exclude from, the training set. Dataset curation consumes per-image review JSONs directly; no separate sidecar report is emitted by the export step.

## Output Format

### YOLO Training Data

After review, export corrected annotations in standard YOLO format:

```
workspace/datasets/sentinel_reviewed/
  images/
    2020-01-04_tile_1_0.png
    houston_pass_014.png
    ...
  labels/
    2020-01-04_tile_1_0.txt
    houston_pass_014.txt
    ...
```

Each `.txt` line: `class_id x_center y_center width height` (normalized 0-1)

Only include detections labeled `ships_sts` or `ship_non_sts`. Detections labeled `neither` are excluded.

Keep the exported label files strictly standard YOLO format. Do not embed provenance, review notes, model metadata, or any other extra fields in the `.txt` labels.

Store all non-YOLO metadata separately in per-image review files and the preprocess manifest.

### data.yaml

Generate a `data.yaml` pointing to the reviewed dataset so it can be merged with the existing training set.

## UI Requirements

### Technology

- Backend: Python web server (same pattern as AIS candidate reviewer)
- Frontend: HTML + JavaScript
- Image viewing: Leaflet with `L.CRS.Simple` for pixel-coordinate image overlay
- Bounding box editing: Leaflet Draw plugin for add/delete/move/resize rectangles
- Serve cached review images as static files

### Overview Mode Layout

- Full-width image viewer with all current bounding boxes shown, with `neither`-labeled boxes rendered in muted gray
- Color-coded boxes: green = `ships` (STS), blue = `ship`, red = unreviewed, muted gray = `neither`
- Toolbar: Add Box, Mark Neither, Start Detail Review, Complete
- Detection count summary
- Mouse wheel over the image should zoom only the image viewer; the page itself should not vertically scroll
- Vertical scrolling should be confined to the sidebar when the cursor is over the sidebar

### Detail Mode Layout

- Image viewer zoomed to current detection with padding for context
- Current bounding box highlighted and editable (move + resize)
- Label buttons: `Ships (STS)`, `Ship (non-STS)`, `Neither`
- Navigation: Back, Next
- Detection progress: `4 / 12`
- Static bottom-center discrete progress bar mirroring per-image detection review progress, with the count label on the right side of the bar

## Dependencies

- `ultralytics` - YOLO inference (needs to be installed: `pip install ultralytics`)
- `Pillow` - image stitching / PNG export
- `torch` - required by ultralytics
- `rasterio` or equivalent GeoTIFF reader for robust source tile loading

## Implementation Order

1. Pre-processing script: manifest + cached PNG render + YOLO + write initial review JSON stubs
2. Review UI: overview mode with bbox add/delete
3. Review UI: detail mode with bbox edit + labeling + resume support
4. Export script: review JSON files -> YOLO training format

## File Naming

- `scripts/preprocess_sentinel_tiles.py` - tile stitching + PNG cache render + YOLO inference
- `scripts/review_visual_detections_ui.py` - review web UI
- `scripts/export_reviewed_labels.py` - export to YOLO format
