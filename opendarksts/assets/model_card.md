# Default model: clean STS YOLO26n

Purpose: flag candidate pairs of ships in approximately 10 m/pixel, true-colour
satellite imagery for human review. Classes: `0: ship`, `1: ships`. This is not a
model of transponder state, vessel identity, cargo exchange or legality.

Checkpoint: `yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt`
SHA-256: `080179e28767af1c10f5da297aa667ba2ee1f5846ca76c6765f6f577228433c9`

Dataset `sts_clean_v1`: 562 images, 290 region/date groups. Whole groups were
assigned to train (350 images), validation (102), and test (110), with no group
shared between splits. Inputs are 88 deduplicated Roboflow source scenes and
474 completed manual reviews. Roboflow metadata identifies project
`fleetdatascience/data-science-dark-fleet`, version 7, as Public Domain; source
and annotation redistribution rights still need contributor review.

Clean test results at image size 960: overall precision 0.841, recall 0.842,
mAP50 0.868, mAP50–95 0.583; `ships` mAP50 0.839 and mAP50–95 0.607.
Test data: 547 `ship` boxes and 69 `ships` boxes. The three negative images are all
in test, so negative-background generalization is weakly measured. Metrics are
from the recorded evaluation, not a fresh independent benchmark. Default
operational inference uses Ultralytics' image size 640 and confidence 0.25;
these test metrics should not be presented as the operational pipeline's accuracy.

Training: YOLO26n starting checkpoint; 300 epochs, image size 960, batch 8,
seed 42; Ultralytics 8.4.23, Python 3.12.2, torch 2.10.0+cu128. See
`weight_recreation.md` and the repository's `docs/yolo_weight_recreation.md`.

Limits: small or occluded hulls, cloud, wakes, ports, coarse imagery and unfamiliar
regions can produce misses or false positives. Duplicate boxes and overlapping
tiles can produce repeated candidates. A visible pair does not prove a transfer.
AIS matching uses any valid position within the radius on the same UTC day;
there is no acquisition-time window, track association or hull identification.
Missing/invalid AIS is unknown, not an absence. US AIS coverage only.

Earlier contaminated-split headline metrics must not replace the clean test
figures above. Byte-identical training recreation is not guaranteed across
hardware or dependency changes; the upstream starting artifact URL was not
recorded. The release model asset and proposed AGPL licence await approval.
