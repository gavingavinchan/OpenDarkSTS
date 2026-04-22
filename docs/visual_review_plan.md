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

## Storage Rules

- The source folder in `/home/gavin/Downloads/AIS data/` is read-only. This workflow must never modify source tiles.
- All generated outputs must live under `workspace/`.
- Do not store regenerable intermediates by default. If a stitched review image or merged detection set can be rebuilt from source tiles plus recorded settings, treat it as cache.
- Persist only the review database, final YOLO exports, and enough provenance metadata to reproduce cached artifacts.

## YOLO Models

Run both available models on each stitched image:

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
   - Record source tile paths, tile layout, and stitched image dimensions in a manifest
3. **Renders a stitched PNG review image** for browser use:
   - Use PNG, not GeoTIFF, for the web UI
   - Store rendered review images under `workspace/cache/sentinel_stitched/` only as a rebuildable cache
   - If a cached PNG is missing, regenerate it from the manifest recipe
4. **Runs both YOLO models** on each stitched PNG review image
5. **Merges detections** from both models with class-aware NMS (IoU threshold ~0.5, configurable)
6. **Saves results**:
   - Detection metadata to `workspace/databases/visual_detections.sqlite`
   - Reproducibility metadata to `workspace/manifests/visual_preprocess_manifest.json`

The preprocessing step should not write initial YOLO label files alongside cached images. Those are regenerable and should be exported only after human review.

### Database Schema: `visual_detections.sqlite`

#### `images` table

- `image_id` (primary key, auto-increment)
- `date` (e.g., `2020-01-04`)
- `width`, `height` (pixels)
- `tile_count` (number of source tiles)
- `cache_path` (nullable path to cached stitched PNG review image)
- `review_status` (`pending`, `overview_1`, `detail`, `overview_2`, `completed`)
- `detail_cursor` (integer index of current detection in detail mode, nullable)
- `created_at`
- `updated_at`

#### `image_sources` table

- `image_source_id` (primary key, auto-increment)
- `image_id` (foreign key)
- `tile_path` (source GeoTIFF path)
- `tile_row`, `tile_col`
- `tile_mtime`
- `tile_size_bytes`
- `created_at`

#### `detections` table

- `detection_id` (primary key, auto-increment)
- `image_id` (foreign key)
- `source_type` (`raw_model`, `merged`, `manual`)
- `source_model` (`yolov8n`, `yolo26n`, null)
- `parent_detection_id` (nullable, for manual edits derived from another detection)
- `class_id` (0 = ship, 1 = ships)
- `class_name` (`ship` or `ships`)
- `confidence` (YOLO confidence, null for manual additions)
- `x_center`, `y_center`, `width`, `height` (YOLO normalized format, 0-1)
- `review_label` (`ships_sts`, `ship_non_sts`, `neither`, null)
- `is_active` (boolean, default true)
- `reviewed` (boolean, default false)
- `review_order` (integer, stable per-image detail review order)
- `reviewed_at`
- `created_at`

#### `detection_sources` table

- `detection_source_id` (primary key, auto-increment)
- `detection_id` (foreign key to merged/manual detection)
- `raw_detection_id` (foreign key to raw model detection)
- `source_model` (`yolov8n` or `yolo26n`)

### Preprocess Manifest: `workspace/manifests/visual_preprocess_manifest.json`

Store reproducibility metadata outside the YOLO label files. At minimum include:

- source tile paths and tile grid layout per date
- source tile mtimes or hashes
- GeoTIFF -> PNG render settings
- stitched image dimensions
- model weight paths
- inference thresholds
- class-aware NMS settings

## Review Workflow

The reviewer processes one image (date) at a time, in this order:

### Step 1: Overview Mode (first pass)

Show the full stitched image with all active bounding boxes overlaid.

Purpose: quick scan to see if YOLO missed anything obvious.

Controls:
- Zoom and pan freely
- Add new bounding boxes (for missed detections)
- Mark boxes as `neither` or hide them from the active queue
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
  - `Neither` -> mark this detection inactive and exclude it from export
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
- Bottom-center per-image detection review progress bar with discrete segments, where each segment represents one active bounding box in the current image
- The per-image bar should be compact and flat, with the `reviewed/total` count such as `5/10` shown on the right side of the bar
- During keyboard review flow, pressing `Space` to advance through detections should also advance the per-image progress bar by one segment until all detections for that image have been traversed
- The per-image progress bar should be reconstructed from saved review state when the page reloads, so progress does not reset on refresh
- Next/Back to move between images
- Re-opening the UI should resume each image at its saved review stage and detail cursor

## Review Labels

- `ships_sts` - two vessels in STS configuration (YOLO class 1: `ships`)
- `ship_non_sts` - single vessel, not STS (YOLO class 0: `ship`)
- `neither` - false positive, no vessel present (kept in the database for history, excluded from export)

## Output Format

### YOLO Training Data

After review, export corrected annotations in standard YOLO format:

```
workspace/datasets/sentinel_reviewed/
  images/
    2020-01-04.png
    2020-01-19.png
    ...
  labels/
    2020-01-04.txt
    2020-01-19.txt
    ...
```

Each `.txt` line: `class_id x_center y_center width height` (normalized 0-1)

Only include detections labeled `ships_sts` or `ship_non_sts`. Detections labeled `neither` are excluded.

Keep the exported label files strictly standard YOLO format. Do not embed provenance, review notes, model metadata, or any other extra fields in the `.txt` labels.

Store all non-YOLO metadata separately in the review database and preprocess manifest.

### data.yaml

Generate a `data.yaml` pointing to the reviewed dataset so it can be merged with the existing training set.

## UI Requirements

### Technology

- Backend: Python web server (same pattern as AIS candidate reviewer)
- Frontend: HTML + JavaScript
- Image viewing: Leaflet with `L.CRS.Simple` for pixel-coordinate image overlay
- Bounding box editing: Leaflet Draw plugin for add/delete/move/resize rectangles
- Serve cached stitched PNG review images as static files

### Overview Mode Layout

- Full-width image viewer with all active bounding boxes shown
- Color-coded boxes: green = `ships` (STS), blue = `ship`, red = unreviewed
- Toolbar: Add Box, Hide/Mark Neither, Start Detail Review, Complete
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

1. Pre-processing script: manifest + cached PNG render + YOLO + save to database
2. Review UI: overview mode with bbox add/delete
3. Review UI: detail mode with bbox edit + labeling + resume support
4. Export script: database -> YOLO training format

## File Naming

- `scripts/preprocess_sentinel_tiles.py` - tile stitching + PNG cache render + YOLO inference
- `scripts/review_visual_detections_ui.py` - review web UI
- `scripts/export_reviewed_labels.py` - export to YOLO format
