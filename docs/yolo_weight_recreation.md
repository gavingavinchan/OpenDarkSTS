# YOLO Weight Recreation

This document is the repository registry for YOLO `*.pt` weight files. Every YOLO weight kept in the repository must be listed here or documented by an adjacent weight-specific markdown file.

Current inventory date: 2026-05-07.

## Reproducibility Standard

For each weight, record enough information to recreate or re-obtain it:

- source dataset or upstream source
- starting weights or model architecture
- training or acquisition command when known
- key training parameters
- output path and whether the file is a run-local checkpoint or promoted/archive copy
- known gaps that prevent byte-identical reproduction

Ultralytics training can be deterministic for a fixed environment, seed, and device, but byte-identical checkpoints are not guaranteed across package, CUDA, PyTorch, driver, or hardware changes.

Portable collaborator bundles are not an exception to this rule. Any future `export-bundle` command that includes `*.pt` files must also include the matching recreation documentation from this file or an adjacent weight-specific markdown file. Any future `import-bundle` command must reject, skip, or quarantine bundled weights whose recreation metadata is missing. Annotation-only bundles should omit weights by default.

## Root Base Weights

These are upstream/base weights used as starting points for project training runs. They are not trained on OpenDarkSTS project data.

| Weight | Source | Embedded metadata | Re-obtain or recreate notes | Known gaps |
|---|---|---|---|---|
| `yolov8n.pt` | Official Ultralytics YOLOv8n COCO-pretrained checkpoint | `model=yolov8n.yaml`, `data=coco.yaml`, `epochs=500`, `batch=16`, `imgsz=640`, `seed=0`, Ultralytics `8.0.0.dev0`, COCO class names | Re-obtain with Ultralytics by resolving `yolov8n.pt`, for example `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` | Exact upstream artifact/version should be preserved if byte identity matters |
| `yolov8s.pt` | Official Ultralytics YOLOv8s COCO-pretrained checkpoint | `model=yolov8s.yaml`, `data=coco.yaml`, `epochs=500`, `batch=16`, `imgsz=640`, `seed=0`, Ultralytics `8.0.0.dev0`, COCO class names | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters |
| `yolov9s.pt` | Official Ultralytics YOLOv9s COCO-pretrained checkpoint | `model=yolov9s.yaml`, `data=/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=500`, `batch=128`, `imgsz=640`, `seed=0`, Ultralytics `8.2.31`, COCO class names | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov9s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters; `yolov9n.pt` was not resolvable in this environment |
| `yolov10n.pt` | Official Ultralytics YOLOv10n COCO-pretrained checkpoint | Checkpoint metadata records Ultralytics `8.2.30`, `epochs=100`, `batch=16`, `imgsz=640`, `seed=0`; `model` and `data` fields are `None` in embedded `train_args` | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov10n.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters; embedded training metadata is incomplete |
| `yolov10s.pt` | Official Ultralytics YOLOv10s COCO-pretrained checkpoint | Checkpoint metadata records Ultralytics `8.2.30`, `epochs=100`, `batch=16`, `imgsz=640`, `seed=0`; `model` and `data` fields are `None` in embedded `train_args` | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolov10s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters; embedded training metadata is incomplete |
| `yolo11n.pt` | Official Ultralytics YOLO11n COCO-pretrained checkpoint | `model=yolo11n.yaml`, `data=/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=600`, `batch=128`, `imgsz=640`, `seed=0`, Ultralytics `8.2.100`, COCO class names | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters |
| `yolo11s.pt` | Official Ultralytics YOLO11s COCO-pretrained checkpoint | `model=yolo11s.yaml`, `data=/usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=600`, `batch=128`, `imgsz=640`, `seed=0`, Ultralytics `8.2.100`, COCO class names | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Exact upstream artifact/version should be preserved if byte identity matters |
| `yolo26n.pt` | External YOLO26n COCO-pretrained checkpoint | `model=yolo26n-objv1-150.pt`, `data=/home/lq/codes/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=245`, `patience=100`, `batch=128`, `imgsz=640`, `device=5`, `seed=0`, Ultralytics `8.3.222`, COCO class names | Re-obtain from the same YOLO26/Ultralytics upstream used to provide this file, then place it at repo root as `yolo26n.pt` | Upstream source URL and custom `yolo26n-objv1-150.pt` starting artifact are not recorded in this repo, so exact recreation is incomplete |
| `yolo12n.pt` | External YOLO12n COCO-pretrained checkpoint | Checkpoint contains a `DetectionModel` with COCO class names; no `train_args`, `date`, or `version` metadata were present | Re-obtain from the same YOLO12/Ultralytics upstream used to provide this file, then place it at repo root as `yolo12n.pt` | Upstream source URL, acquisition command, training command, and package version are not recorded in this repo |
| `yolo12s.pt` | External/Ultralytics-resolved YOLO12s COCO-pretrained checkpoint | Checkpoint contains a `DetectionModel` with COCO class names; no `train_args`, `date`, or `version` metadata were present | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo12s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Upstream source URL, training command, and package version are not recorded in the checkpoint |
| `yolo26s.pt` | External YOLO26s COCO-pretrained checkpoint resolved by Ultralytics | `model=yolo26s-objv1-150.pt`, `data=/home/lq/codes/ultralytics/ultralytics/cfg/datasets/coco.yaml`, `epochs=70`, `patience=100`, `batch=128`, `imgsz=640`, `device=3`, `seed=0`, Ultralytics `8.3.222`, COCO class names | Re-obtain with `.venv/bin/python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"`; initially downloaded from Ultralytics assets during 2026-05-05 model-suite preflight | Upstream source URL and custom `yolo26s-objv1-150.pt` starting artifact are not recorded in this repo, so exact recreation is incomplete |

## Madhulika Roboflow Dataset Weights

Dataset source for both mixed-label and rectbbox runs:

- Original source copied from read-only folder: `<READ_ONLY_SOURCE_DIR>/Madhulika/data science dark fleet.v7i.yolov8 (1).zip`
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
  data=workspace/datasets/manual_reviewed_gavin_v1/data.yaml \
  imgsz=960 \
  epochs=5 \
  batch=4 \
  device=cpu \
  seed=42 \
  project=workspace/runs/manual_reviewed_gavin_v1_smoke \
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

### Expanded Manual STS Balanced v2 Runs

These runs repeat the v1 setup above after more manual `ships_sts` labels were completed. The intent is a fair comparison against v1 by keeping the starting weights and training parameters unchanged while changing only the derived dataset version.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v2/data.yaml`.

Dataset manifest: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v2/dataset_manifest.json`.

Builder command:

```bash
.venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py \
  --output-dir workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v2 \
  --dataset-version 2 \
  --overwrite
```

Generated dataset counts: `1086 ship`, `1086 ships` overall.

Manual `ships_sts` source counts: `106` raw manual `ships` labels plus `554` train-only augmented manual `ships` labels, for `660` manual `ships` labels after augmentation. The dataset also keeps the old `426` Madhulika rectbbox `ships` labels.

Training command script: `workspace/reports/run_madhulika_manual_sts_balanced_v2_training.sh`.

Recreate the sequence with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_v2_training.sh
```

Common parameters: `epochs=150`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=30`, `workers=4`.

Online augmentation parameters intentionally disabled: `copy_paste=0.0`, `degrees=0.0`, `fliplr=0.0`, `flipud=0.0`, `hsv_h=0.0`, `hsv_s=0.0`, `hsv_v=0.0`, `mixup=0.0`, `mosaic=0.0`, `scale=0.0`, `translate=0.0`.

Covered weights:

| Weight | Starting weights | Training config | Run metadata |
|---|---|---|---|
| `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolov8n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v2_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolov8n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v2_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolo26n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v2_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolo26n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v2_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v2_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |

Logs:

- `workspace/reports/train_madhulika_manual_sts_balanced_v2_yolov8n_rectbbox.log`
- `workspace/reports/train_madhulika_manual_sts_balanced_v2_yolo26n_rectbbox.log`
- `workspace/reports/run_madhulika_manual_sts_balanced_v2_training.log`

Run outcomes:

- YOLOv8n stopped early after `34` epochs; best results were observed at epoch `4` according to the training log, and the best checkpoint was saved as `best.pt`.
- YOLO26n stopped early after `31` epochs; best results were observed at epoch `1` according to the training log, and the best checkpoint was saved as `best.pt`.

Training durations:

| Run | Sequence wall time | Ultralytics reported train time |
|---|---:|---:|
| `yolov8n_rectbbox_e150_img960` | `2m 39s` (`2026-05-05T04:56:54-05:00` to `2026-05-05T04:59:33-05:00`) | `0.040 hours` (`2m 24s`) |
| `yolo26n_rectbbox_e150_img960` | `3m 17s` (`2026-05-05T04:59:33-05:00` to `2026-05-05T05:02:50-05:00`) | `0.053 hours` (`3m 11s`) |

Promotion note: at user request on 2026-05-05, the YOLOv8n `best.pt` from this run became the default YOLOv8n weight for `scripts/preprocess_sentinel_tiles.py` so new visual-review preprocessing uses the expanded manual STS training. The checkpoint remains in its run-local path rather than being copied to `workspace/model_weights/`.

Known gaps: further evaluation against the old v1 validation/test behavior, old Madhulika test behavior, and raw manual STS recall is still needed before treating these as fully validated current-best copies.

### Expanded Manual STS Balanced v3 Fair Runs

These runs repeat the v2 setup above after additional manual `ships_sts` labels were completed. The fair-comparison intent is unchanged: keep the starting weights and training parameters fixed while changing only the derived dataset version.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v3/data.yaml`.

Dataset manifest: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v3/dataset_manifest.json`.

Builder command:

```bash
.venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py \
  --output-dir workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v3 \
  --dataset-version 3 \
  --overwrite
```

Generated dataset counts: `1364 ship`, `1364 ships` overall.

Manual `ships_sts` source counts: `196` raw manual `ships` labels plus `742` train-only augmented manual `ships` labels, for `938` manual `ships` labels after augmentation. The dataset also keeps the old `426` Madhulika rectbbox `ships` labels.

Training command script: `workspace/reports/run_madhulika_manual_sts_balanced_v3_training.sh`.

Recreate the sequence with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_v3_training.sh
```

Common parameters: `epochs=150`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=30`, `workers=4`.

Online augmentation parameters intentionally disabled: `copy_paste=0.0`, `degrees=0.0`, `fliplr=0.0`, `flipud=0.0`, `hsv_h=0.0`, `hsv_s=0.0`, `hsv_v=0.0`, `mixup=0.0`, `mosaic=0.0`, `scale=0.0`, `translate=0.0`.

Covered weights:

| Weight | Starting weights | Training config | Run metadata |
|---|---|---|---|
| `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolov8n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v3_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolov8n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v3_yolov8n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolov8n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolo26n_rectbbox_e150_img960/weights/best.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v3_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolo26n_rectbbox_e150_img960/weights/last.pt` | `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` | `workspace/training_configs/madhulika_manual_sts_balanced_v3_yolo26n_rectbbox_e150_img960.json` | `workspace/runs/madhulika_manual_sts_balanced_v3_e150_img960/yolo26n_rectbbox_e150_img960/args.yaml` |

Logs:

- `workspace/reports/train_madhulika_manual_sts_balanced_v3_yolov8n_rectbbox.log`
- `workspace/reports/train_madhulika_manual_sts_balanced_v3_yolo26n_rectbbox.log`
- `workspace/reports/run_madhulika_manual_sts_balanced_v3_training.log`
- `workspace/reports/test_madhulika_manual_sts_balanced_v3_yolov8n_rectbbox.log`
- `workspace/reports/test_madhulika_manual_sts_balanced_v3_yolo26n_rectbbox.log`

Run outcomes:

- YOLOv8n stopped early after `66` epochs; test split `all` metrics were `P=0.855`, `R=0.697`, `mAP50=0.787`, `mAP50-95=0.485`; test `ships` metrics were `P=0.918`, `R=0.750`, `mAP50=0.841`, `mAP50-95=0.538`.
- YOLO26n stopped early after `63` epochs; test split `all` metrics were `P=0.833`, `R=0.714`, `mAP50=0.784`, `mAP50-95=0.509`; test `ships` metrics were `P=0.936`, `R=0.767`, `mAP50=0.832`, `mAP50-95=0.535`.

Training durations:

| Run | Sequence wall time |
|---|---:|
| `yolov8n_rectbbox_e150_img960` | `5m 27s` (`2026-05-05T20:11:59-05:00` to `2026-05-05T20:17:26-05:00`) |
| `yolo26n_rectbbox_e150_img960` | `7m 22s` (`2026-05-05T20:17:26-05:00` to `2026-05-05T20:24:48-05:00`) |

Known gaps: these are run-local checkpoints, not promoted/current-best copies. Broader evaluation against v2, old Madhulika behavior, and raw manual STS recall is still needed before promotion.

### Expanded Manual STS Balanced v3 YOLO Version Suite

These runs use the v3 balanced dataset but intentionally are not a fair comparison to the no-augmentation fair runs above. The goal is to compare available YOLO nano and small variants using stronger satellite-appropriate online augmentation while keeping the dataset class ratio at `1:1`.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v3/data.yaml`.

Weight preflight report: `workspace/reports/yolo_version_suite_weight_preflight_v3.json`.

Training suite config: `workspace/training_configs/madhulika_manual_sts_balanced_v3_yolo_version_suite_aug_e300_img960.json`.

Training command script: `workspace/reports/run_madhulika_manual_sts_balanced_v3_yolo_suite_aug_training.sh`.

Recreate the sequence with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_v3_yolo_suite_aug_training.sh
```

Common parameters: `epochs=300`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=60`, `workers=4`.

Online augmentation parameters: `copy_paste=0.0`, `degrees=180.0`, `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.35`, `hsv_v=0.25`, `mixup=0.05`, `mosaic=1.0`, `scale=0.4`, `translate=0.1`.

Available starting weights selected by preflight: `yolov8n.pt`, `yolov8s.pt`, `yolov9s.pt`, `yolov10n.pt`, `yolov10s.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo12n.pt`, `yolo12s.pt`, `yolo26n.pt`, and `yolo26s.pt`.

Unavailable in this environment during preflight: `yolov9n.pt`/`yolo9n.pt` and all tested nano/small weights for versions `13` through `25` using `yolo{version}{size}.pt` and `yolov{version}{size}.pt` naming.

Covered weights, once each run completes:

| Weight pattern | Starting weights | Run metadata |
|---|---|---|
| `workspace/runs/madhulika_manual_sts_balanced_v3_yolo_suite_aug_e300_img960/<run_name>/weights/best.pt` | one selected root base weight listed above | `workspace/runs/madhulika_manual_sts_balanced_v3_yolo_suite_aug_e300_img960/<run_name>/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v3_yolo_suite_aug_e300_img960/<run_name>/weights/last.pt` | one selected root base weight listed above | `workspace/runs/madhulika_manual_sts_balanced_v3_yolo_suite_aug_e300_img960/<run_name>/args.yaml` |

Known gaps: this suite was started on 2026-05-05 and may still be running. Update this section with per-model outcomes, test metrics, and logs after all models complete.

### Expanded Manual STS Balanced v4 YOLO Version Suite

These runs repeat the v3 YOLO version suite after additional manual annotations were completed. The suite is intentionally not a fair comparison to the no-augmentation fair runs; it uses the same stronger satellite-appropriate online augmentation and selected available nano/small YOLO weights as the v3 suite, while keeping the dataset class ratio at `1:1`.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4/data.yaml`.

Dataset manifest: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4/dataset_manifest.json`.

Builder command:

```bash
.venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py \
  --output-dir workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4 \
  --dataset-version 4 \
  --overwrite
```

Generated dataset counts: `1991 ship`, `1991 ships` overall.

Manual `ships_sts` source counts: `404` raw manual `ships` labels plus `1161` train-only augmented manual `ships` labels, for `1565` manual `ships` labels after augmentation. The dataset also keeps the old `426` Madhulika rectbbox `ships` labels.

Weight preflight report: `workspace/reports/yolo_version_suite_weight_preflight_v4.json`.

Training suite config: `workspace/training_configs/madhulika_manual_sts_balanced_v4_yolo_version_suite_aug_e300_img960.json`.

Training command script: `workspace/reports/run_madhulika_manual_sts_balanced_v4_yolo_suite_aug_training.sh`.

Recreate the sequence with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_v4_yolo_suite_aug_training.sh
```

Common parameters: `epochs=300`, `imgsz=960`, `batch=8`, `device=0`, `seed=42`, `patience=60`, `workers=4`.

Online augmentation parameters: `copy_paste=0.0`, `degrees=180.0`, `fliplr=0.5`, `flipud=0.5`, `hsv_h=0.015`, `hsv_s=0.35`, `hsv_v=0.25`, `mixup=0.05`, `mosaic=1.0`, `scale=0.4`, `translate=0.1`.

Available starting weights selected by preflight: `yolov8n.pt`, `yolov8s.pt`, `yolov9s.pt`, `yolov10n.pt`, `yolov10s.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo12n.pt`, `yolo12s.pt`, `yolo26n.pt`, and `yolo26s.pt`.

Unavailable in this environment during preflight: `yolov9n.pt`/`yolo9n.pt` and all tested nano/small weights for versions `13` through `25` using `yolo{version}{size}.pt` and `yolov{version}{size}.pt` naming.

Covered weights, once each run completes:

| Weight pattern | Starting weights | Run metadata |
|---|---|---|
| `workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/<run_name>/weights/best.pt` | one selected root base weight listed above | `workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/<run_name>/args.yaml` |
| `workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/<run_name>/weights/last.pt` | one selected root base weight listed above | `workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/<run_name>/args.yaml` |

Logs, once each run starts or completes, follow these patterns:

- `workspace/reports/run_madhulika_manual_sts_balanced_v4_yolo_suite_aug_training.log`
- `workspace/reports/train_madhulika_manual_sts_balanced_v4_suite_<run_name>.log`
- `workspace/reports/test_madhulika_manual_sts_balanced_v4_suite_<run_name>.log`

Run duration: the full 11-model train-and-test sequence took about `8h 2m`, from `2026-05-07T21:48:06-05:00` to `2026-05-08T05:50:13-05:00`.

Known gaps: update this section with per-model outcomes, test metrics, and logs after the final comparison summary is written.

## Clean STS Dataset Weights (sts_clean_v1)

This is the first weight trained on a dataset with a verified leak-free split. It supersedes the Madhulika-derived weights for evaluation purposes; every earlier number in this registry was measured on a split that shared scenes with training.

Dataset: `workspace/datasets/derived/sts_clean_v1/data.yaml`.

Dataset manifest: `workspace/datasets/derived/sts_clean_v1/dataset_manifest.json` (per-item provenance, region, scene-date group and split assignment).

Dataset builder:

```bash
.venv/bin/python scripts/build_clean_sts_dataset.py --overwrite
```

Dataset inputs:

- `workspace/datasets/derived/madhulika_yolov8_rectbbox` deduplicated from `381` Roboflow files to `88` unique scenes (`286` pre-augmented duplicates and `7` `False_color` duplicates of a `True_color` scene dropped).
- `workspace/reviews/gavin`, all `474` reviews with `review_status: completed`, including ship-only images and the `3` pure negatives. `117` `neither` boxes are not exported.

Split rule: whole `<region>|<date>` scene groups are assigned to exactly one split by `scripts/dataset_splits.py::assign_groups_to_splits` with `seed=42` and target ratios `0.70 / 0.15 / 0.15`, with hard floors of `100` test images and `60` test `ships` boxes. Regions come from raster bounds in `workspace/manifests/training_import_manifest.json`, falling back to the read-only source directory then the album majority.

Resulting dataset: `562` images / `290` groups.

| Split | Images | Groups | `ship` boxes | `ships` boxes | Negatives |
|---|---|---|---|---|---|
| train | 350 | 173 | 1486 | 353 | 0 |
| val | 102 | 65 | 313 | 96 | 0 |
| test | 110 | 52 | 547 | 69 | 3 |

Leak check: `PASS`, zero groups appear in more than one split.

Starting weights: root base weight `yolo26n.pt`.

Training command script: `workspace/reports/run_sts_clean_v1_training.sh`.

Recreate with:

```bash
bash workspace/reports/run_sts_clean_v1_training.sh
```

which runs:

```bash
.venv/bin/yolo detect train \
  model=yolo26n.pt \
  data=workspace/datasets/derived/sts_clean_v1/data.yaml \
  imgsz=960 batch=8 epochs=300 patience=60 seed=42 device=0 workers=4 \
  degrees=180 fliplr=0.5 flipud=0.5 hsv_h=0.015 hsv_s=0.35 hsv_v=0.25 \
  mixup=0.05 mosaic=1.0 scale=0.4 translate=0.1 copy_paste=0 \
  project=workspace/runs/sts_clean_v1_yolo26n_aug_e300_img960 \
  name=yolo26n_aug_e300_img960
```

Classes: `0: ship`, `1: ships`. Package metadata: Ultralytics `8.4.23`, Python `3.12.2`, torch `2.10.0+cu128`, CUDA device `0` (`NVIDIA GeForce RTX 5070 Ti`).

The dataset contains no baked-in augmentation; all augmentation is Ultralytics online augmentation applied at train time with the parameters above.

Run outcome: all `300` epochs completed in `0.288` hours (`patience=60` never triggered). Best validation fitness at epoch `296`, val `mAP50-95 = 0.5931`.

Covered weights:

| Weight | Kind | Starting weights | Run metadata |
|---|---|---|---|
| `workspace/model_weights/yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt` | promoted copy (`sha256:080179e28767af1c10f5da297aa667ba2ee1f5846ca76c6765f6f577228433c9`) | `yolo26n.pt` | copy of the run `best.pt` below |
| `workspace/runs/sts_clean_v1_yolo26n_aug_e300_img960/yolo26n_aug_e300_img960/weights/best.pt` | run-local checkpoint | `yolo26n.pt` | `.../args.yaml`, `.../results.csv` |
| `workspace/runs/sts_clean_v1_yolo26n_aug_e300_img960/yolo26n_aug_e300_img960/weights/last.pt` | run-local checkpoint | `yolo26n.pt` | `.../args.yaml`, `.../results.csv` |

Test-split evaluation (`yolo detect val split=test` on the `sts_clean_v1` test split, `imgsz=960`, `batch=8`, `device=0`):

| Weight | all P | all R | all mAP50 | all mAP50-95 | ship mAP50 | ship mAP50-95 | ships mAP50 | ships mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| `yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt` (clean) | 0.841 | 0.842 | 0.868 | 0.583 | 0.897 | 0.560 | 0.839 | 0.607 |
| `yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt` (old promoted) | 0.661 | 0.613 | 0.639 | 0.385 | 0.765 | 0.439 | 0.513 | 0.331 |
| v4 suite `yolo26n_aug_e300_img960/weights/best.pt` (contaminated) | 0.809 | 0.858 | 0.880 | 0.602 | 0.875 | 0.494 | 0.886 | 0.711 |

The v4 row is **not** a fair comparison: that model was trained on the balanced v4 dataset, which contains most of the images in this test split. Treat it as an upper bound of memorisation, not as generalisation.

Evaluation scripts and logs:

- `workspace/reports/eval_sts_clean_v1_baselines.sh` / `workspace/reports/eval_sts_clean_v1_baselines.log`
- `workspace/reports/eval_sts_clean_v1_new_weight.sh` / `workspace/reports/eval_sts_clean_v1_new_weight.log`
- `workspace/reports/train_sts_clean_v1_yolo26n.log`
- Ultralytics val outputs under `workspace/runs/sts_clean_v1_eval/`

Known gaps:

- Byte-identical recreation is not guaranteed across Ultralytics, torch, CUDA, driver or GPU changes.
- The dataset build depends on the read-only source imagery under `<READ_ONLY_SOURCE_DIR>` and on `workspace/reviews/gavin`; adding or completing reviews changes the group set and therefore the split.
- `2` reviewed images carry no georeferenced bounds and are placed by the source-path/album fallbacks rather than by raster bounds.
- The Madhulika deduplication keeps one representative per scene base name. Where Roboflow produced several genuinely different crops of one scene under the same base name, only the most-labelled crop survives; the discarded crops are of the same scene, so no information about a new observation is lost, but some labelled boxes are.

## Updating This Registry

When adding, moving, renaming, promoting, or deleting a YOLO `*.pt` file, update this document in the same change. If a weight has an adjacent doc instead, that doc must include the same reproduction fields and this registry should still point to it.

If exact recreation is impossible, state the reason explicitly. Acceptable gaps include missing upstream URLs, external proprietary artifacts, unavailable hardware details, or unknown package versions, but the gap must be documented rather than left implicit.

## Installed application copies (release preparation)

`opendarksts init` copies or downloads `opendarksts_yolo11s_v4_best.pt` into
`<workspace>/model_weights/`, verifies its recorded SHA-256, and writes accompanying
`model_card.md` and `weight_recreation.md`. These are packaged under
`opendarksts/assets/`; `default_model.json` specifies the proposed GitHub release
asset. The asset is not published yet. Local review workspaces use byte-identical
copies of the v4 YOLO11s run checkpoint and the same documentation. Initialization
upgrades a recorded automatically installed former default, while preserving an
explicitly promoted current model. The former clean-split YOLO26n remains an
alternative. No weight file is added to git.

## Isolated GUI acceptance weights (18 September 2026)

The test run `workspace/reports/gui_acceptance/20260918_independent/workspace/runs/gui_acceptance/cpu_smoke_fixed/`
keeps `weights/best.pt` and `weights/last.pt`. Its adjacent
[`weight_recreation.md`](../workspace/reports/gui_acceptance/20260918_independent/workspace/runs/gui_acceptance/cpu_smoke_fixed/weight_recreation.md)
records source images, acceptance labels, starting checkpoint, exact command,
environment, parameters, hashes and reproducibility limitations. The GUI promotion
target is `gui_acceptance_cpu_smoke_2026-09-18_best.pt` within that isolated
workspace's `model_weights/`. This is a one-epoch CPU workflow test with reused
images and model-assisted labels; it provides no independent performance estimate.
It is not a replacement release model. No weight file is added to git.

## SNAME 2026 release default: YOLO11s v4

The published filename is `opendarksts_yolo11s_v4_best.pt`, installed under
`<workspace>/model_weights/`. SHA-256:
`cf11d54de2aa2ba3fa74e23d1118482a281b62cd29a95b2b1e58a06e92b261ea`.
It is a byte-identical copy of
`workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/yolo11s_aug_e300_img960/weights/best.pt`.
The proposed release asset is
https://huggingface.co/gavingavinchan/OpenDarkSTS-YOLO11s/resolve/main/opendarksts_yolo11s_v4_best.pt;
no asset has been published as part of this change.

The [packaged acquisition and recreation recipe](../opendarksts/assets/weight_recreation.md)
records the v4 dataset builder, original inputs, COCO-pretrained `yolo11s.pt`
starting checkpoint, exact training command, 300 epochs, image size 960, batch 8,
seed 42, patience 60 and augmentation parameters. The v4 suite section above
records the original run script and logs. Ultralytics checkpoint metadata is
8.4.23. Exact recreation requires the original dataset/review snapshot and upstream
starting artifact; byte identity across hardware and dependency changes is not
guaranteed. The upstream starting asset URL was not recorded.

The [model card](../opendarksts/assets/model_card.md) reproduces the paper's
held-out test results: overall P 0.830, R 0.820, mAP50 0.895, mAP50–95 0.544;
`ship` mAP50 0.869, `ships` mAP50 0.920. The 21-image test split is small, and
the pre-augmented v4 source set causes some source-scene overlap with training.
These are the paper's reported results, not an independent-scene estimate or a
replacement for the separate clean-split evaluation recorded above. The default
selection aligns the application with the SNAME 2026 paper and presentation.

Local validation copies also reside in this worktree's gitignored
`workspace/model_weights/` and isolated installed-example workspaces under
`workspace/reports/`. All are byte-identical to the source checkpoint and covered
by this recipe. No `.pt` file is committed.

## User workspace promotion records

Improve and CLI promotion preserve `<target-name>.recreation.json` adjacent to each promoted workspace checkpoint. These records preserve the recorded command, configuration, dataset hashes, source revision, environment, weight hash, and known reproducibility gaps. User-trained checkpoints do not require edits to this tracked document. The bundled default filename is reserved so restoration remains possible. Legacy runs without training metadata require an explicit existing recreation recipe, which is copied into the adjacent record.
