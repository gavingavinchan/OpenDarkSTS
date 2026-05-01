# Madhulika + Manual STS Balanced Dataset

This document records the reproducible recipe and intent for the first combined dataset that uses the old Madhulika data behind the promoted rectbbox weights plus Gavin's completed manual-reviewed STS labels.

## Intent

- Combine the old dataset used to create the current promoted rectbbox weights with the manually reviewed STS boxes.
- Treat the old Madhulika `ships` examples as already augmented; do not multiply them again offline.
- Treat the new manual `ships` examples as high-value raw reviewed labels that need controlled train-only augmentation.
- Avoid blur in augmentation, per Gavin's preference.
- Keep the effective `ship` and `ships` ratio about `1:1` for the first candidate training dataset.
- Do not include all manual ship-only review images in this first balanced run because they would overwhelm the `ships` class.

## Source Inputs

Old rectbbox dataset:

```text
workspace/datasets/derived/madhulika_yolov8_rectbbox
```

This is the bbox-converted version of the Roboflow Madhulika dataset documented in `docs/yolo_weight_recreation.md`. It contains:

| Split | `ship` | `ships` |
|---|---:|---:|
| train | 442 | 371 |
| valid | 32 | 29 |
| test | 51 | 26 |
| total | 525 | 426 |

Manual review source:

```text
workspace/reviews/gavin
```

Completed manual reviews currently contain `1072 ship` and `35 ships` positive boxes. For this first balanced dataset, only completed review images containing at least one `ships_sts` label are selected. Those selected manual STS images contain `154 ship` and `35 ships` boxes. Completed ship-only manual review images are intentionally held out for a later full-combined or hard-negative/ship-robustness experiment.

## Dataset Output

The generated YOLO dataset path is:

```text
workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1
```

Recreate it with:

```bash
.venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py --overwrite
```

The command writes:

```text
workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/images/{train,val,test}/
workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/labels/{train,val,test}/
workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/data.yaml
workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1/dataset_manifest.json
```

Current generated counts:

| Split | `ship` | `ships` |
|---|---:|---:|
| train | 557 | 626 |
| val | 62 | 33 |
| test | 70 | 30 |
| total | 689 | 689 |

Counts by source:

| Source | `ship` | `ships` | Intent |
|---|---:|---:|---|
| Madhulika rectbbox | 525 | 426 | Preserve the old training source as-is. |
| Raw manual STS images | 154 | 35 | Include every positive label on completed manual images containing `ships_sts`. |
| Manual augmented STS crops | 10 | 228 | Add no-blur train-only focused STS examples until the overall ratio reaches `1:1`. |

## Split Policy

- Old Madhulika split ownership is preserved: `train -> train`, `valid -> val`, `test -> test`.
- Manual STS images are split before augmentation with seed `42`.
- Manual `val` and `test` stay raw only; no augmented derivative of a manual image crosses split boundaries.
- Manual train `ships_sts` boxes are the only source for offline augmentation.

## Augmentation Policy

The dataset builder emits focused train-only crops around manual train `ships_sts` boxes until the overall `ships` count reaches the overall `ship` count, or until candidates are exhausted.

Allowed variants:

- crop-only focused STS crop
- horizontal flip
- vertical flip
- 90 degree rotation
- 180 degree rotation
- 270 degree rotation
- brightness `0.85`
- brightness `1.15`
- contrast `0.90`
- contrast `1.10`

Explicitly disabled:

- blur
- mosaic
- mixup
- copy-paste

The focused crop policy uses a `256 px` minimum crop, `512 px` maximum crop, and a `10x` bbox-size scale. A positive box is included in a crop label file only if at least `50%` of its original bbox area is inside the crop. Candidate crops whose `ship` count exceeds `ships` count are skipped because they do not help the intended class balance.

## Training Intent

Train from the old promoted rectbbox weights rather than COCO base weights so the new run behaves like a continuation/refinement of the existing detector:

```text
workspace/model_weights/yolov8n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt
workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt
```

Recommended first training settings:

| Setting | Value | Intent |
|---|---:|---|
| `imgsz` | 960 | Match the old promoted rectbbox training resolution. |
| `epochs` | 150 | Match the old baseline training budget. |
| `batch` | 8 | Match prior GPU runs. |
| `seed` | 42 | Stable split and training reproducibility. |
| `patience` | 30 | Slightly more tolerant than prior `20` because the dataset now mixes old full images and focused manual crops. |
| `workers` | 4 | Existing manual-reviewed runs used this safely. |
| online mosaic/mixup/copy-paste | 0 | Avoid adding synthetic geometry on top of already-augmented old `ships` and explicit offline manual STS augmentation. |

The first run should evaluate class-specific `ships` recall on raw manual test images and ensure old Madhulika test performance does not regress severely.

Training configs prepared for this dataset:

```text
workspace/training_configs/madhulika_manual_sts_balanced_v1_yolov8n_rectbbox_e150_img960.json
workspace/training_configs/madhulika_manual_sts_balanced_v1_yolo26n_rectbbox_e150_img960.json
```

Run both trainings sequentially with:

```bash
bash workspace/reports/run_madhulika_manual_sts_balanced_training.sh
```

For an LLM/session-safe detached run, use:

```bash
nohup bash workspace/reports/run_madhulika_manual_sts_balanced_training.sh > workspace/reports/run_madhulika_manual_sts_balanced_training.log 2>&1 < /dev/null &
```
