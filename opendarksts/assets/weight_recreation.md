# Default checkpoint recreation / acquisition

File: `yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt`.
SHA-256: `080179e28767af1c10f5da297aa667ba2ee1f5846ca76c6765f6f577228433c9`.

Acquire after publication with `opendarksts init`; its pinned URL and checksum
are in the package's `assets/default_model.json`. Before publication use
`opendarksts init --weights PATH_TO_THIS_CHECKPOINT`. Weights are excluded from git.

Dataset: `sts_clean_v1`, 562 images / 290 whole region/date groups, from 88
Roboflow version-7 source scenes after deduplication and 474 completed manual
reviews. Build using `python -m scripts.build_clean_sts_dataset --overwrite` with
the inputs described in repository `docs/yolo_weight_recreation.md`, section
"Clean STS Dataset Weights (sts_clean_v1)". Publication of the annotation and
split bundle awaits a source-rights review.

Starting weights: `yolo26n.pt` (the historical upstream URL was not recorded).
Training command, from the workspace's parent folder:

```
yolo detect train model=yolo26n.pt data=workspace/datasets/derived/sts_clean_v1/data.yaml imgsz=960 batch=8 epochs=300 patience=60 seed=42 device=0 workers=4 degrees=180 fliplr=0.5 flipud=0.5 hsv_h=0.015 hsv_s=0.35 hsv_v=0.25 mixup=0.05 mosaic=1.0 scale=0.4 translate=0.1 copy_paste=0 project=workspace/runs/sts_clean_v1_yolo26n_aug_e300_img960 name=yolo26n_aug_e300_img960
```

Promoted copy: run-local `weights/best.pt`, best validation fitness at epoch 296.
Training metadata: Ultralytics 8.4.23, Python 3.12.2, torch 2.10.0+cu128,
NVIDIA RTX 5070 Ti. The full registry records all settings, sources, split
counts and previous weights. Byte identity depends on original inputs, upstream
checkpoint, packages, drivers and hardware, and is not guaranteed by retraining.
