# Progress Tracking

## Current Dataset Workspace

- Source copied from read-only folder: `/home/gavin/Downloads/AIS data/Madhulika/data science dark fleet.v7i.yolov8 (1).zip`
- Working copy stored at `workspace/datasets/raw/data_science_dark_fleet_v7i_yolov8.zip`
- Extracted dataset stored at `workspace/datasets/extracted/madhulika_yolov8`
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

## YOLO Benchmark: 150 Epochs Max

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

### Current Conclusion

- With longer training, YOLO26 is the stronger model on the current test split
- YOLOv8 keeps slightly higher precision
- YOLO26 now has better recall and better mAP, which is promising for candidate STS event detection

## Known Dataset Issue

- Ultralytics warned that the dataset appears to mix boxes and segments
- Training proceeded using boxes only
- This dataset should be cleaned or standardized before treating current metrics as final

## Current Best Weights

- YOLOv8 best weights: `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/best.pt`
- YOLO26 best weights: `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/best.pt`

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

- Inspect false positives and false negatives from the best YOLO26 model
- Clean the mixed box/segment dataset issue and rerun the benchmark
- Decide whether to force a full 150-epoch run with early stopping disabled for curve inspection

## Denmark AIS Test Scan

- Scanner script added at `scripts/scan_ais_sts_candidates.py`
- Criteria note stored at `docs/ais_denmark_sts_criteria.md`
- Test source scanned: `/home/gavin/Downloads/AIS data/AIS Denmark/AIS DK 2026/aisdk-2026-01-01.zip`
- Output database written to `workspace/databases/ais_denmark_candidates.sqlite`
- Scan summary: `14,516,337` rows read, `12,883,565` usable `Class A` rows, `288` five-minute bins processed
- Candidate output inserted with criteria version `ais_denmark_v1`: `37` events total (`25` `HIGH`, `12` `MEDIUM`) and `2,148` supporting point rows
- Many candidates carry `anchor_status_present`, which suggests the next precision improvement should be an anchorage/port-distance filter before bulk ingesting more Denmark dates
