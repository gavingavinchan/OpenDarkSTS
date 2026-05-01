# YOLO Weight Recreation

This document is the repository registry for YOLO `*.pt` weight files. Every YOLO weight kept in the repository must be listed here or documented by an adjacent weight-specific markdown file.

Current inventory date: 2026-05-01.

## Reproducibility Standard

For each weight, record enough information to recreate or re-obtain it:

- source dataset or upstream source
- starting weights or model architecture
- training or acquisition command when known
- key training parameters
- output path and whether the file is a run-local checkpoint or promoted/archive copy
- known gaps that prevent byte-identical reproduction

Ultralytics training can be deterministic for a fixed environment, seed, and device, but byte-identical checkpoints are not guaranteed across package, CUDA, PyTorch, driver, or hardware changes.

## Root Base Weights

These are upstream/base weights used as starting points for project training runs. They are not trained on OpenDarkSTS project data.

| Weight | Source | Embedded metadata | Re-obtain or recreate notes | Known gaps |
|---|---|---|---|---|
| `yolov8n.pt` | Official Ultralytics YOLOv8n COCO-pretrained checkpoint | `model=yolov8n.yaml`, `data=coco.yaml`, `epochs=500`, `batch=16`, `imgsz=640`, `seed=0`, Ultralytics `8.0.0.dev0`, COCO class names | Re-obtain with Ultralytics by resolving `yolov8n.pt`, for example `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` | Exact upstream artifact/version should be preserved if byte identity matters |
| `yolo26n.pt` | External YOLO26n COCO-pretrained checkpoint | `model=yolo26n-objv1-150.pt`, `data=/home/lq/codes/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=245`, `patience=100`, `batch=128`, `imgsz=640`, `device=5`, `seed=0`, Ultralytics `8.3.222`, COCO class names | Re-obtain from the same YOLO26/Ultralytics upstream used to provide this file, then place it at repo root as `yolo26n.pt` | Upstream source URL and custom `yolo26n-objv1-150.pt` starting artifact are not recorded in this repo, so exact recreation is incomplete |
| `yolo12n.pt` | External YOLO12n COCO-pretrained checkpoint | Checkpoint contains a `DetectionModel` with COCO class names; no `train_args`, `date`, or `version` metadata were present | Re-obtain from the same YOLO12/Ultralytics upstream used to provide this file, then place it at repo root as `yolo12n.pt` | Upstream source URL, acquisition command, training command, and package version are not recorded in this repo |

## Madhulika Roboflow Dataset Weights

Dataset source for both mixed-label and rectbbox runs:

- Original source copied from read-only folder: `/home/gavin/Downloads/AIS data/Madhulika/data science dark fleet.v7i.yolov8 (1).zip`
- Working copy: `workspace/datasets/raw/data_science_dark_fleet_v7i_yolov8.zip`
- Extracted dataset: `workspace/datasets/extracted/madhulika_yolov8`
- Roboflow metadata: workspace `fleetdatascience`, project `data-science-dark-fleet`, version `7`, license `Public Domain`, URL `https://app.roboflow.com/fleetdatascience/data-science-dark-fleet/7`
- Classes: `0: ship`, `1: ships`
- Training script: `scripts/compare_yolo_models.py`
- Project training package metadata in checkpoints: Ultralytics `8.4.23`

Create the rectbbox dataset from the extracted Roboflow dataset with:

```bash
.venv/bin/python scripts/copy_yolo_dataset_as_bboxes.py \
  workspace/datasets/extracted/madhulika_yolov8 \
  workspace/datasets/derived/madhulika_yolov8_rectbbox
```

Conversion result recorded in `docs/progress.md`: `381` label files scanned, `377` changed, `932` polygon annotations converted, `19` existing bbox annotations kept, and `0` polygon lines remaining.

### 50-Epoch Mixed-Label Comparison

Source dataset: `workspace/datasets/extracted/madhulika_yolov8/data.yaml`.

Recreate with:

```bash
.venv/bin/python scripts/compare_yolo_models.py \
  --data workspace/datasets/extracted/madhulika_yolov8/data.yaml \
  --output-dir workspace/runs/model_compare \
  --report workspace/reports/yolo_compare_summary.json \
  --epochs 50 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --patience 20 \
  --seed 42
```

Covered weights:

| Weight | Starting weights | Run metadata |
|---|---|---|
| `runs/detect/workspace/runs/model_compare/yolov8n_e50_img960/weights/best.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare/yolov8n_e50_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare/yolov8n_e50_img960/weights/last.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare/yolov8n_e50_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare/yolo26n_e50_img960/weights/best.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare/yolo26n_e50_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare/yolo26n_e50_img960/weights/last.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare/yolo26n_e50_img960/args.yaml` |

Metrics report: `workspace/reports/yolo_compare_summary.json`.

### YOLOv8 Continued 150-Epoch Mixed-Label Run

This is a continuation-style run from the 50-epoch YOLOv8 last checkpoint.

Recreate with:

```bash
.venv/bin/yolo detect train \
  model=runs/detect/workspace/runs/model_compare/yolov8n_e50_img960/weights/last.pt \
  data=workspace/datasets/extracted/madhulika_yolov8/data.yaml \
  imgsz=960 \
  epochs=150 \
  batch=8 \
  device=0 \
  seed=42 \
  project=workspace/runs/model_compare \
  name=yolov8n_lastpt_e150 \
  patience=150 \
  workers=8
```

Covered weights:

| Weight | Run metadata |
|---|---|
| `runs/detect/workspace/runs/model_compare/yolov8n_lastpt_e150/weights/best.pt` | `runs/detect/workspace/runs/model_compare/yolov8n_lastpt_e150/args.yaml` |
| `runs/detect/workspace/runs/model_compare/yolov8n_lastpt_e150/weights/last.pt` | `runs/detect/workspace/runs/model_compare/yolov8n_lastpt_e150/args.yaml` |

Known gap: this run depends on the previous `yolov8n_e50_img960/weights/last.pt` checkpoint, so recreate that checkpoint first.

### 150-Epoch Mixed-Label Comparison

Source dataset: `workspace/datasets/extracted/madhulika_yolov8/data.yaml`.

Recreate with:

```bash
.venv/bin/python scripts/compare_yolo_models.py \
  --data workspace/datasets/extracted/madhulika_yolov8/data.yaml \
  --output-dir workspace/runs/model_compare_150 \
  --report workspace/reports/yolo_compare_summary_150.json \
  --epochs 150 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --patience 20 \
  --seed 42
```

Covered weights:

| Weight | Starting weights | Run metadata |
|---|---|---|
| `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/best.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/last.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/best.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/last.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/args.yaml` |

Archive copies of the best checkpoints:

| Archive weight | Source checkpoint |
|---|---|
| `workspace/model_weights/yolov8n_madhulika_mixed_labels_e150_img960_2026-03-19_best.pt` | `runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/best.pt` |
| `workspace/model_weights/yolo26n_madhulika_mixed_labels_e150_img960_2026-03-19_best.pt` | `runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/best.pt` |

Metrics report: `workspace/reports/yolo_compare_summary_150.json`.

Known gap: the source dataset intentionally contains mixed polygon and bbox labels. Ultralytics trained with boxes only and emitted a mixed-label warning, so these weights are archived rather than current best.

### 150-Epoch Rectbbox Comparison

Source dataset: `workspace/datasets/derived/madhulika_yolov8_rectbbox/data.yaml`.

Recreate with:

```bash
.venv/bin/python scripts/compare_yolo_models.py \
  --data workspace/datasets/derived/madhulika_yolov8_rectbbox/data.yaml \
  --output-dir workspace/runs/model_compare_rectbbox_150 \
  --report workspace/reports/yolo_compare_rectbbox_summary_150.json \
  --epochs 150 \
  --imgsz 960 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --patience 20 \
  --seed 42
```

Covered weights:

| Weight | Starting weights | Run metadata |
|---|---|---|
| `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960/weights/best.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960/weights/last.pt` | `yolov8n.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960/weights/best.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960/args.yaml` |
| `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960/weights/last.pt` | `yolo26n.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960/args.yaml` |

Promoted/current copies of the best checkpoints:

| Promoted weight | Source checkpoint |
|---|---|
| `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolov8n_e150_img960/weights/best.pt` |
| `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `runs/detect/workspace/runs/model_compare_rectbbox_150/yolo26n_e150_img960/weights/best.pt` |

Metrics report: `workspace/reports/yolo_compare_rectbbox_summary_150.json`.

Current baseline note: `docs/progress.md` records the rectbbox YOLOv8 and YOLO26 weights as current best, with YOLO26 stronger overall on the test split.

## Manual-Reviewed Gavin Dataset Weights

These weights use datasets exported by the training pipeline from reviewed imagery.

Common classes: `0: ship`, `1: ships`.

Common package metadata in checkpoints: Ultralytics `8.4.23`.

### Smoke Run

Dataset: `workspace/datasets/manual_reviewed_gavin_v1/data.yaml`.

Dataset manifest: `workspace/datasets/manual_reviewed_gavin_v1/dataset_manifest.json`.

Dataset selection manifest: `workspace/dataset_selections/manual_reviewed_gavin_v1.json`.

Training config: `workspace/training_configs/manual_reviewed_gavin_v1_yolov8n_smoke.json`.

Starting weights: `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`.

Recreate with:

```bash
.venv/bin/yolo detect train \
  model=workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt \
  data=/home/gavin/OpenDarkSTS/workspace/datasets/manual_reviewed_gavin_v1/data.yaml \
  imgsz=960 \
  epochs=5 \
  batch=4 \
  device=cpu \
  seed=42 \
  project=/home/gavin/OpenDarkSTS/workspace/runs/manual_reviewed_gavin_v1_smoke \
  name=yolov8n_rectbbox_e5_cpu2 \
  patience=2 \
  workers=2
```

Covered weights:

| Weight | Metadata |
|---|---|
| `workspace/runs/manual_reviewed_gavin_v1_smoke/yolov8n_rectbbox_e5_cpu2/weights/best.pt` | `workspace/runs/manual_reviewed_gavin_v1_smoke/yolov8n_rectbbox_e5_cpu2/training_metadata.json`, `args.yaml` |
| `workspace/runs/manual_reviewed_gavin_v1_smoke/yolov8n_rectbbox_e5_cpu2/weights/last.pt` | `workspace/runs/manual_reviewed_gavin_v1_smoke/yolov8n_rectbbox_e5_cpu2/training_metadata.json`, `args.yaml` |

Evaluation report: `workspace/reports/manual_reviewed_gavin_v1_smoke_evaluation.md`.

Known gap: this is a CPU smoke run, not a promoted model. The evaluation report says it stopped early after 3 epochs due to `patience=2`, and metrics are not robust because the validation split has only 2 images.

### Balanced 1:1 No-Augmentation Runs

Dataset: `workspace/datasets/manual_reviewed_gavin_balanced_1to1_v1/data.yaml`.

Dataset manifest: `workspace/datasets/manual_reviewed_gavin_balanced_1to1_v1/dataset_manifest.json`.

Dataset selection manifest: `workspace/dataset_selections/manual_reviewed_gavin_balanced_1to1_v1.json`.

Command script: `workspace/reports/run_balanced_1to1_training.sh`.

Common parameters: `epochs=150`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=20`, `workers=4`.

Recreate the sequence with:

```bash
bash workspace/reports/run_balanced_1to1_training.sh
```

Equivalent individual commands are recorded in the script and each run's `args.yaml`.

Covered weights:

| Weight | Starting weights | Training config | Run metadata |
|---|---|---|---|
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolov8n_e150_img960/weights/best.pt` | `yolov8n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolov8n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolov8n_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolov8n_e150_img960/weights/last.pt` | `yolov8n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolov8n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolov8n_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo12n_e150_img960/weights/best.pt` | `yolo12n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo12n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo12n_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo12n_e150_img960/weights/last.pt` | `yolo12n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo12n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo12n_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo26n_e150_img960/weights/best.pt` | `yolo26n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo26n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo26n_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo26n_e150_img960/weights/last.pt` | `yolo26n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo26n_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960/yolo26n_e150_img960/args.yaml` |

Logs:

- `workspace/reports/train_manual_reviewed_balanced_1to1_yolov8n.log`
- `workspace/reports/train_manual_reviewed_balanced_1to1_yolo12n.log`
- `workspace/reports/train_manual_reviewed_balanced_1to1_yolo26n.log`

### Balanced 1:1 Augmentation Runs

Dataset: `workspace/datasets/manual_reviewed_gavin_balanced_1to1_v1/data.yaml`.

Dataset manifest: `workspace/datasets/manual_reviewed_gavin_balanced_1to1_v1/dataset_manifest.json`.

Dataset selection manifest: `workspace/dataset_selections/manual_reviewed_gavin_balanced_1to1_v1.json`.

Command script: `workspace/reports/run_balanced_1to1_aug_training.sh`.

Common base parameters: `epochs=150`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=20`, `workers=4`.

Augmentation parameters: `copy_paste=0.0`, `degrees=180.0`, `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `mixup=0.1`, `mosaic=1.0`, `scale=0.5`, `translate=0.1`.

Recreate the sequence with:

```bash
bash workspace/reports/run_balanced_1to1_aug_training.sh
```

Equivalent individual commands are recorded in the script and each run's `args.yaml`.

Covered weights:

| Weight | Starting weights | Training config | Run metadata |
|---|---|---|---|
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolov8n_aug_e150_img960/weights/best.pt` | `yolov8n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolov8n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolov8n_aug_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolov8n_aug_e150_img960/weights/last.pt` | `yolov8n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolov8n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolov8n_aug_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo12n_aug_e150_img960/weights/best.pt` | `yolo12n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo12n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo12n_aug_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo12n_aug_e150_img960/weights/last.pt` | `yolo12n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo12n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo12n_aug_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo26n_aug_e150_img960/weights/best.pt` | `yolo26n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo26n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo26n_aug_e150_img960/args.yaml` |
| `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo26n_aug_e150_img960/weights/last.pt` | `yolo26n.pt` | `workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_yolo26n_aug_e150_img960.json` | `workspace/runs/manual_reviewed_gavin_balanced_1to1_v1_aug_e150_img960/yolo26n_aug_e150_img960/args.yaml` |

Logs:

- `workspace/reports/train_manual_reviewed_balanced_1to1_yolov8n_aug.log`
- `workspace/reports/train_manual_reviewed_balanced_1to1_yolo12n_aug.log`
- `workspace/reports/train_manual_reviewed_balanced_1to1_yolo26n_aug.log`

## Madhulika + Manual STS Balanced Dataset Weights

These weights combine the old Madhulika rectbbox dataset used for the current promoted rectbbox weights with Gavin's completed manual-reviewed STS examples. The dataset recipe and user intent are documented in `docs/madhulika_manual_sts_balanced_dataset.md`.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/data.yaml`.

Dataset manifest: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/dataset_manifest.json`.

Builder command:

```bash
.venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py --overwrite
```

Dataset intent:

- keep the old `426` Madhulika `ships` boxes as-is because they are already augmented
- use completed manual review images containing `ships_sts` labels
- augment only manual train `ships_sts` as focused no-blur crops
- hold out manual ship-only reviews from this first balanced run so the effective class ratio stays about `1:1`

Generated dataset counts: `689 ship`, `689 ships` overall.

Training command script: `workspace/reports/run_madhulika_manual_sts_balanced_training.sh`.

Recreate the sequence with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_training.sh
```

Common parameters: `epochs=150`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=30`, `workers=4`.

Online augmentation parameters intentionally disabled: `copy_paste=0.0`, `degrees=0.0`, `fliplr=0.0`, `flipud=0.0`, `hsv_h=0.0`, `hsv_s=0.0`, `hsv_v=0.0`, `mixup=0.0`, `mosaic=0.0`, `scale=0.0`, `translate=0.0`.

Covered weights:

| Weight | Starting weights | Training config | Run metadata |
|---|---|---|---|
| `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolov8n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v1_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolov8n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v1_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolo26n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v1_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolo26n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v1_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v1_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |

Logs:

- `workspace/reports/train_madhulika_manual_sts_balanced_yolov8n_rectbbox.log`
- `workspace/reports/train_madhulika_manual_sts_balanced_yolo26n_rectbbox.log`
- `workspace/reports/run_madhulika_manual_sts_balanced_training.log`

Run outcomes:

- YOLOv8n stopped early after `48` epochs; best results were observed at epoch `18` according to the training log, and the best checkpoint was saved as `best.pt`.
- YOLO26n stopped early after `49` epochs; best results were observed at epoch `19` according to the training log, and the best checkpoint was saved as `best.pt`.

Training durations:

| Run | Sequence wall time | Ultralytics reported train time |
|---|---:|---:|
| `yolov8n_rectbbox_e150_img960` | `2m 55s` (`2026-05-01T03:48:45-05:00` to `2026-05-01T03:51:40-05:00`) | `0.047 hours` (`2m 49s`) |
| `yolo26n_rectbbox_e150_img960` | `4m 16s` (`2026-05-01T03:51:40-05:00` to `2026-05-01T03:55:56-05:00`) | `0.069 hours` (`4m 08s`) |

Known gaps: these are run-local checkpoints, not promoted/current-best copies. Further evaluation against old Madhulika test behavior and raw manual STS recall is still needed before promotion.

## Updating This Registry

When adding, moving, renaming, promoting, or deleting a YOLO `*.pt` file, update this document in the same change. If a weight has an adjacent doc instead, that doc must include the same reproduction fields and this registry should still point to it.

If exact recreation is impossible, state the reason explicitly. Acceptable gaps include missing upstream URLs, external proprietary artifacts, unavailable hardware details, or unknown package versions, but the gap must be documented rather than left implicit.
