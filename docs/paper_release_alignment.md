# Paper-to-release interface alignment

This audit compares the user-facing claims in `sname_2026_paper.tex` with release
base `f4004b0`. All thirteen main pages were opened in Chromium before changes;
initial evidence is in `workspace/reports/paper_alignment/before/`. The separate
`origin/darkfleet-clean` dashboard was inspected as a vocabulary reference only.
Its source and weights were not imported. Implementation differences that do not
change the operator's task are documented rather than restored.

## Gap analysis recorded before implementation

“Matches” denotes an existing feature; “Elsewhere” denotes a different name or
location; “Different” denotes different behavior. Estimates are engineering effort,
including validation, rather than elapsed promises.

| Paper item | Before | Action or remaining work |
|---|---|---|
| Two tools; six training steps; operational GUI | Elsewhere: one server, Detect and Improve; seven training steps | Expose paper names and explain six steps plus evaluation extension. |
| Training GUI/CLI parity; classes `ship` and `ships` | Matches | Preserve shared backend and CLI. |
| Step 1 RGB import, albums, group selection, stable content identifiers, re-import preservation, referenced album membership and import audit | Matches; heading “Import imagery” | Make paper step name visible; preserve behavior. |
| Step 2 auto proposals; accept, resize, relabel, reject, delete and draw boxes; overview/detail modes; keyboard shortcuts | Matches; “Manual Review” | Add annotation terminology alongside existing name. |
| Three review labels; retained negative regions; reviews survive regeneration | Matches | Preserve behavior. Negative boxes are excluded from positive labels; their pixels remain in training images. |
| Competing completed reviews block export | Matches | Preserve conflict warnings and export block. Dedicated GUI reconciliation remains absent (1–2 days); existing reviews can be reopened and corrected. |
| Step 3 cropped grid, class/warning/search filters, sorting, thumbnail size, metadata, editor deep links; conflict exclusion/count | Matches; “BBox Overview” | Replace abbreviated heading with reviewed bounding-box terminology. |
| Step 4 album/label/search selection, default 80/10/10, save then export, frozen splits, incomplete/conflict blocking | Matches | Preserve behavior. |
| Step 4 assigns splits per image | Different: defaults to whole observation groups to prevent scene leakage; image mode remains available | Retain safer default and document difference. |
| Step 5 dataset/starting weight/settings, named config, readiness and overwrite checks, online augmentation | Matches, under Advanced | Expose augmentation in step heading. |
| Four augmentation presets: none/light/default/heavy | Missing: fill-default and clear controls only | Add shared presets through backend/API/CLI and GUI; retain custom edits. |
| Multiple starting weights expand one config to a run group | Missing: one starting weight per configuration | Not planned tonight. Save one config per model with a shared group; automatic suite creation/launch needs 1–2 days. |
| Step 6 detached training, live framework metrics/log, stop, retained weights/metrics/plots | Matches; “Train YOLO” | Use “Train detector”; preserve process management. |
| Launch provenance: command, config and dataset hashes, source revision/dirty state, environment versions/device | Matches | Preserve records. |
| Eight-record reproducibility contract and evaluation report | Elsewhere: records exist; report/promotion in Step 7 | Add visible contract and explain Step 7 as explicit evaluation after the paper's six steps. |
| Runtime stage 1 candidate detection, configurable confidence | Matches; “Run detection” | Show “Scan Images” and three runtime stages. Operational output retains only `ships` pairs; single-vessel annotation remains in Improve. |
| Low=.10 / Medium=.25 / High=.45 presets | Missing | Add named threshold presets and explain lower threshold includes weaker predictions. |
| Reject boxes occupying more than half the tile | Missing | Add shared detector safety filter and record fixed area threshold in run provenance. |
| Runtime stage 2 WGS84 center, coarse bounds plus haversine AIS matching; default radius 1,000 m | Matches | Show “AIS search radius” and default; retain same-UTC-day matching without time window. |
| FULLY_DARK / PARTIALLY_DARK / AIS_PRESENT | Matches in exports; plain AIS-count text in GUI | Show status codes alongside count-based interpretation. No hull identity or legality inference. |
| DETECTED_ONLY / AIS_ERROR | Different: AIS_MISSING / AIS_INVALID preserve diagnostic detail | Show paper-name equivalents alongside existing codes without changing stored/exported schema. |
| NO_STS | Different: empty detection array and zero-count run summary | Display NO_STS for completed zero-candidate runs; do not insert a fake detection. |
| User Mode / Intelligence HQ, cumulative statistics | Elsewhere: Home has model/data/recent work, no cumulative detection totals | Add paper heading, workflow links and cumulative totals over completed detection jobs. |
| Scan Images accepts one/multiple uploaded tiles, sensitivity, optional AIS source | Elsewhere: dated existing-folder inputs on Area & dates; settings on Run; AIS on Data. Operational upload absent | Add paper vocabulary and navigation. Direct operational file upload deferred (1 day, including validation/containment); existing folder supports one or multiple tiles. |
| STS Events cards with annotated image, AIS vessel data and coordinates | Exists but gated: experimental Results table and selected crop | Validate and expose existing Results; retain table/detail layout and show GPS coordinates. Full card layout unnecessary. |
| Live Map: OSM detections, clickable popups | Exists but gated; selected candidate only; tooltips | Expose Results map, show all run candidates and clickable candidate/AIS popups. |
| Export Report: CSV and JSON | Elsewhere: Jobs downloads; gated Results exports add GeoJSON | Expose paper heading and exports in Results and Jobs. |
| AIS Manager: date, range, month, year | Elsewhere: Data downloads date range from Area & dates; month/year shortcuts absent | Add date-period shortcuts using existing shared project/date-range behavior. |
| AIS Manager: five world-region presets and custom bbox | Different: four US presets; custom coordinates supported | Add Gulf of Mexico US preset; retain US coverage warning. Other four world regions need non-US AIS providers (at least 1–2 weeks plus licensing/access), not misleading NOAA choices. |
| AIS Manager: bbox-filtered local files | Different: full US day cached; optional bbox filtering in acquisition backend/CLI | Preserve full-day reuse. Dedicated filtered-download GUI choice deferred (1–2 days); matching is geographically filtered. |
| AIS Manager: automatic zst decompression and one-click apply | Matches: zst for newer days, zip for older days; successful acquisition selects source automatically | Name AIS Manager visibly and preserve behavior. No live full-day download planned for this audit. |
| Scan history persists as JSON and restores on restart | Matches: Jobs and immutable run outputs; gated Results lists runs | Show “Scan history”; verify persistence after restart. |
| Clear all history | Missing | Defer destructive history deletion (1 day for contained helper, API/CLI, explicit GUI confirmation and tests). |
| Developer Mode: single-image validation, lightweight retraining, reference tab | Elsewhere/partial: one-file Detect folder; Improve steps 5–7; no separate reference tab | Link Improve as Training Tool; add operator reference text. Separate daemon-thread trainer or second mode would duplicate stable workflows (2–3 days if required). |
| Flask, Chart.js, Windows base-path update | Different implementation: shared portable standard-library server, HTML summaries, Linux/Windows support | Preserve portable server; do not merge historical dashboard. |
| Data: RGB WGS84 GeoTIFFs, 640px/10m tiles, cloud exclusion >35% | Different: acquisition cloud default 30%; runtime rejects non-north-up/non-WGS84 inputs; training accepts RGB beyond Sentinel-2 | Set new project/imagery CLI cloud default to 35%; preserve existing saved choices and geometry validation. |
| Data/Results: reviewed positives and retained negatives; train-only augmentation; held-out evaluation | Matches workflow; published historical datasets/results are not recreated by opening the app | Preserve scene-safe splitting and separate Step 7 evaluation. Historical v4 offline crop generator is a specialized script, not a GUI action; a general offline-augmentation GUI would need 2–4 days. |
| Eleven-architecture comparison, 300 epochs/60 patience, batch 8, 960px, seed 42 | Different: editable setup defaults 150/30, batch 8, 960px, seed 42 | Retain general defaults; paper experiment settings can be entered in Advanced. No retraining or historical accuracy claim in this task. |
| Polygon-to-box conversion and preview utility | Matches: standalone scripts | Preserve utilities; no new GUI required by the paper. |

## Paper names and application locations

To be finalized after implementation and verification. README is intentionally
outside the scope of this change.

## Validation

Pending implementation. The test workspace is isolated at
`workspace/paper_alignment`; all input/model copies and derived files remain there.
The initialized weight copy includes the existing adjacent recreation documentation.
