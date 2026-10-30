# Default model: SNAME 2026 YOLO11s

This detector identifies candidate pairs of ships in approximately 10 m/pixel,
true-colour satellite imagery for human review. Classes: `0: ship`, `1: ships`.
A visible pair does not establish a transfer, vessel identity, cargo, transponder
state, or legality. Initialize with `opendarksts init` after asset publication,
or `opendarksts init --weights PATH_TO_THIS_CHECKPOINT` before publication.

Checkpoint: `opendarksts_yolo11s_v4_best.pt`
SHA-256: `cf11d54de2aa2ba3fa74e23d1118482a281b62cd29a95b2b1e58a06e92b261ea`
Release asset (publication pending):
https://github.com/gavingavinchan/OpenDarkSTS/releases/download/v0.1/opendarksts_yolo11s_v4_best.pt

The checkpoint is the YOLO11s headline detector presented in the SNAME 2026 paper,
trained on `madhulika_rectbbox_plus_manual_sts_balanced_v4`. This balanced dataset
contains 1,991 instances of each class, combining the pre-augmented Madhulika
rectbbox source set with manually reviewed Sentinel-2 images and train-only
manual STS-crop augmentation. Source metadata identifies Roboflow project
`fleetdatascience/data-science-dark-fleet`, version 7, as Public Domain; source
and annotation redistribution rights still require contributor review.

Held-out test results at image size 960, reproduced exactly from the paper's
"Headline detector --- YOLO11s per-class results" section and cross-architecture
table (21 test images):

| Class | Instances | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| `ship` | 62 | 0.812 | 0.806 | 0.869 | 0.512 |
| `ships` | 30 | 0.849 | 0.833 | 0.920 | 0.575 |
| all | 92 | 0.830 | 0.820 | 0.895 | 0.544 |

YOLO11s ranked highest by overall mAP50 among the eleven compared architectures;
YOLOv12n had the highest mAP50–95 (0.572). These are recorded test metrics, not a
fresh independent benchmark. Operational inference uses the checkpoint's
training image size (960) and a default confidence of 0.25; these metrics do not
measure the accuracy of the whole operational pipeline or of the dark-status labels.

Limitations: the test split contains only 21 images. The v4 dataset derives from
a pre-augmented source set, so some test images share source scenes with training
images. These results therefore do not establish generalization to independent
scenes. Small or occluded hulls, cloud, wakes, ports, coarse imagery and unfamiliar
regions can produce misses or false positives. Overlapping boxes or tiles can
produce repeated candidates. AIS comparison uses any valid position within the
radius on the same UTC day, with no acquisition-time window, track association
or hull identification. Missing or invalid AIS remains unknown. Coverage is US AIS.

Training: COCO-pretrained `yolo11s.pt`; 300 epochs, image size 960, batch 8,
seed 42, patience 60, Ultralytics 8.4.23. See `weight_recreation.md` for the command,
source checkpoint and reproducibility gaps. The clean-split YOLO26n checkpoint
remains an alternative documented in `docs/yolo_weight_recreation.md`; its scores
use a different dataset and must not be substituted for the paper's YOLO11s results.
The release asset and proposed project licence remain pending publication review.
