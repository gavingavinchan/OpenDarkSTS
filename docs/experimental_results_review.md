# Candidate results review checklist

The paper-alignment release exposes this existing feature at Detect → Results /
STS Events and from completed detection jobs. The environment switch is no longer
required. Software acceptance evidence is recorded in `paper_release_alignment.md`;
the checklist below remains applicable to scientific and operator review.

A completed operational output has an immutable `dark_sts.json` and a separate
`decisions.json`. Table, map, crop, nearby AIS records, and CSV/JSON/GeoJSON exports
use those records. The map distinguishes satellite detections from same-day AIS
positions. The latter are not vessel identities inferred from the image.

Only an explicit human verdict supplies a training label. `real_sts` becomes
`ships_sts`. For `not_sts`, the reviewer must also specify `ship_non_sts` or `neither`;
rejecting an STS detection alone cannot distinguish a single ship from background.
`unclear` cannot be transferred. Transfer imports the tight detection crop, records
its original observation and pixel bounds, and completes the training review. It
preserves the parent's geographic region and date for split grouping. Repeated
transfer returns the original record. Edit a transferred label in Improve.

Backend/CLI checks use `python -m scripts.application_cli results DIRECTORY` with
`--detection ID --verdict real_sts`, `--send-to-improve`, or
`--format geojson --output WORKSPACE/exports/candidates.geojson`. The switch controlled
the HTTP surface in earlier releases; backend helpers remain directly testable.

Review a positive, rejected and unclear candidate; verify crop framing, map context,
exports and the resulting completed review in Improve. Do not interpret sample
verdicts entered during automated UI checks as scientific ground truth.
