# Default checkpoint recreation / acquisition

File: `opendarksts_yolo11s_v4_best.pt`.
SHA-256: `cf11d54de2aa2ba3fa74e23d1118482a281b62cd29a95b2b1e58a06e92b261ea`.

Acquire with `opendarksts init`. The pinned release asset is
https://huggingface.co/gavingavinchan/OpenDarkSTS-YOLO11s/resolve/main/opendarksts_yolo11s_v4_best.pt
and its identity is recorded in `assets/default_model.json`.
A local copy can be installed with `opendarksts init --weights PATH_TO_THIS_CHECKPOINT`.
Weights are excluded from git.

The release file is a byte-identical copy of:
`workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960/yolo11s_aug_e300_img960/weights/best.pt`.
Classes: `0: ship`, `1: ships`. This is the YOLO11s detector reported in the
SNAME 2026 paper, not the later clean-split YOLO26n detector.

Dataset: `workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4/data.yaml`.
Inputs: the pre-augmented Madhulika rectbbox source dataset
(`workspace/datasets/derived/madhulika_yolov8_rectbbox`) and completed manual reviews
under `workspace/reviews/gavin`. The balanced dataset has 1,991 instances of each
class, including train-only manual STS-crop augmentation. Rebuild from the original
inputs using the repository command:

```
python -m scripts.build_madhulika_manual_sts_balanced_dataset --output-dir workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4 --dataset-version 4 --overwrite
```

Starting weights: official Ultralytics COCO-pretrained `yolo11s.pt`, acquired with
`python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"`. The historical exact
upstream asset URL was not recorded. Training command from the repository root,
with `yolo` from the active Python environment (the same arguments apply on Windows):

```
yolo detect train model=yolo11s.pt data=workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v4/data.yaml imgsz=960 batch=8 epochs=300 patience=60 seed=42 device=0 workers=4 degrees=180 fliplr=0.5 flipud=0.5 hsv_h=0.015 hsv_s=0.35 hsv_v=0.25 mixup=0.05 mosaic=1.0 scale=0.4 translate=0.1 copy_paste=0 project=workspace/runs/madhulika_manual_sts_balanced_v4_yolo_suite_aug_e300_img960 name=yolo11s_aug_e300_img960
```

The run's `args.yaml` records all training settings. Checkpoint metadata records
Ultralytics 8.4.23 and creation time `2026-05-08T02:16:42.655264`. Select the run's
`weights/best.pt`, which is chosen by validation fitness, rather than `last.pt`.
The repository's `docs/yolo_weight_recreation.md` records the full v4 suite and
source dataset derivation.

Known gaps: original images, review snapshots and starting weights are required;
changes to these inputs change the rebuilt dataset. Publication of the annotation
and split bundle requires a source-rights review. The 21-image held-out test split
is small, and some test images share source scenes with training images because
the source set was pre-augmented. Retraining does not guarantee byte identity
across package versions, drivers or hardware; acquisition plus the pinned SHA-256
is the method for recovering the exact published checkpoint.
