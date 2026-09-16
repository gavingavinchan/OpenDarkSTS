# Progress Tracking

## Current Dataset Workspace

- Source copied from read-only folder: `/home/gavin/Downloads/AIS data/Madhulika/data science dark fleet.v7i.yolov8 (1).zip`
- Working copy stored at `workspace/datasets/raw/data_science_dark_fleet_v7i_yolov8.zip`
- Extracted dataset stored at `workspace/datasets/extracted/madhulika_yolov8`
- Rectangular bbox copy stored at `workspace/datasets/derived/madhulika_yolov8_rectbbox`
- Dataset split sizes: 336 train, 27 valid, 18 test
- Current classes in `workspace/datasets/extracted/madhulika_yolov8/data.yaml`: `ship`, `ships`

## Environment

- Local Python environment created at `.venv/`
- Setup notes stored in `docs/local-venv.md`
- Comparison script stored in `scripts/compare_yolo_models.py`

## YOLO Benchmark: 50 Epochs

- Report: `workspace/reports/yolo_compare_summary.json`
- YOLOv8 test metrics: precision 0.7475, recall 0.6987, mAP50 0.7951, mAP50-95 0.4934
- YOLO26 test metrics: precision 0.7295, recall 0.7575, mAP50 0.7902, mAP50-95 0.4933
- Initial conclusion: results were very close; YOLOv8 was slightly better overall and faster, while YOLO26 had slightly better recall

## YOLO Benchmark: 150 Epochs Max (Mixed Labels)

- Report: `workspace/reports/yolo_compare_summary_150.json`
- YOLOv8 run directory: `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960`
- YOLO26 run directory: `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960`

### Early Stopping Behavior

- YOLOv8 stopped early after epoch 112; best validation mAP50-95 was at epoch 92
- YOLO26 stopped early after epoch 87; best validation mAP50-95 was at epoch 67
- Conclusion: both models plateaued before 150 epochs under `patience=20`

### Best Validation Scores

- YOLOv8: best val mAP50 0.9042, best val mAP50-95 0.5556
- YOLO26: best val mAP50 0.8758, best val mAP50-95 0.5568

### Test Set Results From Best Weights

- YOLOv8: precision 0.8753, recall 0.6421, mAP50 0.8079, mAP50-95 0.4970
- YOLO26: precision 0.8601, recall 0.7020, mAP50 0.8263, mAP50-95 0.5251

### Conclusion For Mixed-Label Run

- With longer training, YOLO26 is the stronger model on the current test split
- YOLOv8 keeps slightly higher precision
- YOLO26 now has better recall and better mAP, which is promising for candidate STS event detection

## Mixed-Label Dataset Issue

- Ultralytics warned that the dataset appears to mix boxes and segments
- Training proceeded using boxes only
- This dataset should be cleaned or standardized before treating current metrics as final

## Rectangular BBox Dataset Conversion

- Conversion script: `scripts/copy_yolo_dataset_as_bboxes.py`
- Copied dataset: `workspace/datasets/extracted/madhulika_yolov8` -> `workspace/datasets/derived/madhulika_yolov8_rectbbox`
- Conversion result: `381` label files scanned, `377` files changed, `932` polygon annotations converted, `19` existing bbox annotations kept
- Verification result: `0` polygon lines remain in `train`, `valid`, or `test`

## YOLO Benchmark: 150 Epochs Max (Rectangular BBoxes)

- Report: `workspace/reports/yolo_compare_rectbbox_summary_150.json`
- YOLOv8 run directory: `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960`
- YOLO26 run directory: `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960`

### Early Stopping Behavior

- YOLOv8 stopped early after epoch `84`; best validation mAP50-95 was at epoch `64`
- YOLO26 stopped early after epoch `82`; best validation mAP50-95 was at epoch `62`
- Conclusion: both models still plateaued before the 150-epoch cap under `patience=20`

### Best Validation Scores

- YOLOv8: best val mAP50 `0.8892`, best val mAP50-95 `0.5229`
- YOLO26: best val mAP50 `0.8853`, best val mAP50-95 `0.5520`

### Test Set Results From Best Weights

- YOLOv8: precision `0.8624`, recall `0.6447`, mAP50 `0.8000`, mAP50-95 `0.4775`
- YOLO26: precision `0.7556`, recall `0.7144`, mAP50 `0.8303`, mAP50-95 `0.5301`

### Current Conclusion

- The bbox-only retrain removes the mixed polygon/box ambiguity from the training set
- YOLO26 remains the stronger overall model on the test split
- YOLOv8 keeps higher test precision, but YOLO26 has better recall and better mAP50/mAP50-95

## Named Weight Copies

- Archived mixed-label YOLOv8 weights: `workspace/model_weights/yolov8n_madhulika_mixed_labels_e150_img960_2026-03-19_best.pt`
- Archived mixed-label YOLO26 weights: `workspace/model_weights/yolo26n_madhulika_mixed_labels_e150_img960_2026-03-19_best.pt`
- Current rectbbox YOLOv8 weights: `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`
- Current rectbbox YOLO26 weights: `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`

## Current Best Weights

- YOLOv8 best weights: `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`
- YOLO26 best weights: `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`

## Real Pipeline Smoke Test

- End-to-end pipeline run completed on `workspace/inputs/2020_01_14/2020-01-14 lat 28.40449 long -94.50870 cropped_small.jpg`
- Output files: `workspace/reports/2020_01_14_pipeline.json` and `workspace/reports/2020_01_14_pipeline.csv`
- The crop bbox was inferred from `lat 28.40449 long -94.50870.txt` as `[-94.50870, 28.40449, -94.31704, 28.61105]`
- Result 1: `FULLY_DARK` at `(28.44329114, -94.50266333)` with no AIS vessels found within 1000 m
- Result 2: `PARTIALLY_DARK` at `(28.58931499, -94.40030482)` with one AIS match: `APL HOLLAND` (`MMSI 563754000`) at `696.364 m`

### Spatial Match Warning

- The source image is visually a daytime scene, but the nearest matched AIS ping was `2020-01-14T07:44:43`
- If that AIS timestamp is interpreted as UTC and converted to Texas local time, it becomes `2020-01-14 01:44:43 CST`, which conflicts with the daytime appearance of the image
- Because the satellite filename only encodes the date and daily range (`00_00` to `23_59`), it should currently be treated as a full-day composite or day-level asset, not an exact capture timestamp
- More importantly, this mismatch is a warning that the current georeferencing or spatial match may be wrong, not just that the timestamp is ambiguous
- The current run used a bbox inferred from `lat 28.40449 long -94.50870.txt`; that inferred crop extent may not align exactly with `cropped_small.jpg`, so the computed lat/lon may be offset enough to create a misleading AIS match
- Therefore the current pipeline must not treat this AIS hit as a trusted spatial confirmation until the crop georeferencing is validated against real image metadata or a known bbox for the exact crop
- Follow-up needed: validate the exact crop bbox, verify the timezone semantics of `BaseDateTime`, recover actual Sentinel acquisition/overpass time if available, and add temporal filtering before treating AIS matches as contemporaneous with the image

## GeoTIFF Timing Check

- GeoTIFF tested: `2020-01-14-00_00_2020-01-14-23_59_Sentinel-2_L2A_True_color.tiff`
- Rasterio-confirmed bounds: `left=-94.52739715576173`, `bottom=28.391796786663782`, `right=-94.48362350463867`, `top=28.41682429748743`
- Pipeline result: one `ships` detection centered at `(28.40532401, -94.50881605)` with `AIS_PRESENT`
- Nearest same-day AIS pair near that detection was `EAGLE KLANG` (`MMSI 564329000`) and `MANIFA` (`MMSI 403560000`)
- Their closest matched pings in the current full-day correlation were at `2020-01-14T17:01:19` and `2020-01-14T17:03:23`, roughly `11:01-11:03 AM CST`, and only `91.293 m` / `115.271 m` from the TIFF detection center
- If Sentinel-2 local acquisition is assumed to be about `10:30 AM CST`, the nearest AIS pings to that time were `2020-01-14T16:30:29` and `2020-01-14T16:29:43`
- At `~10:30 AM CST`, those two AIS vessels were still `1221.649 m` and `1299.552 m` from the TIFF detection center, and both were outside the georeferenced YOLO bbox footprint
- This is a notable inconsistency: a supposedly correct Sentinel-2 daytime image does not align tightly with the AIS pair at the expected `~10:30 AM` time, while it does align much better about 30 minutes later
- Working interpretation: either the assumed local acquisition time is wrong for this asset, the AIS timestamps need timezone/latency validation, or there is still residual spatial/georeferencing error even in the GeoTIFF workflow

## Suggested Next Steps

- Inspect false positives and false negatives from the bbox-trained YOLO26 model
- Compare old mixed-label and new rectbbox weights on the same held-out imagery
- Decide whether to force a full 150-epoch run with early stopping disabled for curve inspection

## Denmark AIS Test Scan

- Scanner script added at `scripts/scan_ais_sts_candidates.py`
- Criteria note stored at `docs/ais_denmark_sts_criteria.md`
- Test source scanned: `/home/gavin/Downloads/AIS data/AIS Denmark/AIS DK 2026/aisdk-2026-01-01.zip`
- Output database written to `workspace/databases/ais_denmark_candidates.sqlite`
- Scan summary: `14,516,337` rows read, `12,883,565` usable `Class A` rows, `288` five-minute bins processed
- Candidate output inserted with criteria version `ais_denmark_v1`: `37` events total (`25` `HIGH`, `12` `MEDIUM`) and `2,148` supporting point rows
- Many candidates carry `anchor_status_present`, which suggests the next precision improvement should be an anchorage/port-distance filter before bulk ingesting more Denmark dates

## Clean Evaluation v1 (2026-09-15)

### Why the old numbers could not be trusted

An audit of the evaluation path found leakage at every level:

- `workspace/datasets/derived/madhulika_yolov8_rectbbox` is a pre-augmented Roboflow export. `381` files come from only `88` distinct scene base names; each source appears `3`-`6` times as random crops, flips, rotations and brightness variants. `17` of those `88` scene base names have copies in more than one of the Roboflow `train`/`valid`/`test` folders, so the same image was both trained on and tested on.
- The same set mixes `True_color` and `False_color` renders of the same scene on the same date (`7` such duplicates), which the old split treated as unrelated images.
- The balanced `v1`-`v4` datasets built by `scripts/build_madhulika_manual_sts_balanced_dataset.py` copy the Roboflow splits verbatim, so they inherit that leak, and their manual val/test splits held only `2` and `3` images.
- The `v4` test split is `21` images / `30` `ships` boxes, so the 11-model suite ranking published from it is noise.
- A leak that had not been noticed before: `5` scene dates (`2020-01-24`, `2020-02-08`, `2020-02-28`, `2020-05-18`, `2021-07-22`) appear in *both* the Madhulika Roboflow set and the manual review corpus. They are the same Gulf of Mexico observations reaching the dataset through two different source collections, so splitting per source collection would still have leaked.

### What replaced it

- `scripts/dataset_splits.py` assigns whole `<region>|<date>` scene groups to a single split, deterministically for a given seed. Region is resolved from raster bounds (`texas_gulf`, `gibraltar`, `suez`, `singapore`), with documented fallbacks. The scene token is parsed and recorded but deliberately not part of the key.
- `scripts/build_clean_sts_dataset.py` builds `workspace/datasets/derived/sts_clean_v1/` from the deduplicated Madhulika scenes (`88` images) plus all `474` completed manual reviews, including ship-only images and the `3` pure negatives. No augmentation is baked into the dataset; Ultralytics online augmentation is used at train time.

### sts_clean_v1 split

`562` images across `290` scene groups, split with `seed=42`, target `70/15/15`, and hard floors of `100` test images and `60` test `ships` boxes.

| Split | Images | Groups | `ship` boxes | `ships` boxes | Negatives | Madhulika / review images |
|---|---|---|---|---|---|---|
| train | 350 | 173 | 1486 | 353 | 0 | 49 / 301 |
| val | 102 | 65 | 313 | 96 | 0 | 27 / 75 |
| test | 110 | 52 | 547 | 69 | 3 | 12 / 98 |

Leak check: `PASS` - zero of the `290` groups appear in more than one split. No two images in different splits are even byte-identical.

Actual ratios are about `62/18/20` by image count rather than exactly `70/15/15`. That is deliberate: the group allocator balances image count and both class counts at once, and the test floor pulls extra groups into `test`. The test split is the honest one that matters, and it reached `110` images and `69` `ships` boxes, above the `100`/`60` targets.

### First honest test-split comparison

All three weights evaluated with `yolo detect val split=test` on the same `sts_clean_v1` test split (`110` images, `547` `ship` boxes, `69` `ships` boxes), `imgsz=960`, `batch=8`, `device=0`.

| Weight | all P | all R | all mAP50 | all mAP50-95 | ship mAP50 | ship mAP50-95 | ships mAP50 | ships mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| `workspace/model_weights/yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt` | 0.841 | 0.842 | 0.868 | 0.583 | 0.897 | 0.560 | 0.839 | 0.607 |
| `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` (old promoted) | 0.661 | 0.613 | 0.639 | 0.385 | 0.765 | 0.439 | 0.513 | 0.331 |
| `workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/yolo26n_aug_e300_img960/weights/best.pt` (v4) | 0.809 | 0.858 | 0.880 | 0.602 | 0.875 | 0.494 | 0.886 | 0.711 |

Reading of these numbers:

- The new clean model is a large, real improvement over the previously promoted weight: `ships` mAP50 `0.513` -> `0.839`, overall mAP50 `0.639` -> `0.868`.
- The v4 row must not be used as a ranking. That model was trained on the balanced v4 dataset, which contains most of the images in this test split, so its `ships` mAP50-95 of `0.711` is memorisation, not generalisation. It is reported only to show how much apparent skill contamination buys.
- Even so, the clean model beats v4 on `ship` mAP50-95 (`0.560` vs `0.494`) despite v4 having seen the test images, which suggests the clean dataset - with ship-only images and negatives included - teaches the common class better than the artificially class-balanced v4 dataset did.

### Training run

`yolo26n.pt` -> 300 epochs, `imgsz=960`, `batch=8`, `patience=60`, `seed=42`, `device=0`, `workers=4`, v4 suite online augmentation (`degrees=180 fliplr=0.5 flipud=0.5 hsv_h=0.015 hsv_s=0.35 hsv_v=0.25 mixup=0.05 mosaic=1.0 scale=0.4 translate=0.1 copy_paste=0`). All 300 epochs ran in `0.288` hours; early stopping never triggered, and best validation fitness landed at epoch `296`, so the run was still improving and a longer schedule is worth trying.

Run directory: `workspace/runs/sts_clean_v1_yolo26n_aug_e300_img960/yolo26n_aug_e300_img960`. Log: `workspace/reports/train_sts_clean_v1_yolo26n.log`.

### Promotion

- Promoted weight: `workspace/model_weights/yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt`
- `scripts/preprocess_sentinel_tiles.py` now defaults `--weights-yolo26n` to that file.
- Recreation recipe: `docs/yolo_weight_recreation.md`, section "Clean STS Dataset Weights (sts_clean_v1)".

### Follow-ups

- Re-run the multi-model suite on `sts_clean_v1` before making any architecture claims; the v3/v4 suite rankings are not evidence.
- The YOLOv8n default in `scripts/preprocess_sentinel_tiles.py` still points at a v2 balanced run and is still contaminated; retrain it on `sts_clean_v1` or drop it from the review prefill.
- Consider a longer schedule (the run had not converged) and revisit the Madhulika deduplication if the lost crops turn out to matter.
