# Paper-to-release interface alignment

This audit compares the user-facing claims in `sname_2026_paper.tex` with release
base `f4004b0`. All thirteen main pages were opened in Chromium before changes;
initial evidence is in `workspace/reports/paper_alignment/before/`. The separate
`origin/darkfleet-clean` dashboard was inspected as a vocabulary reference only.
Its source and weights were not imported. Implementation differences that do not
change the operator's task are documented rather than restored.

Branch: `t3/paper-workflow-alignment`, based on `f4004b0`. Commits:

- `3825659`: initial gap analysis, committed before implementation.
- `affa1dc`: detector safeguards, optional-AIS correction, shared presets and tests.
- `e7fc974`: paper names, accessible Results, map/popups, date controls and UI tests.
- A final documentation commit records this verification report. No commits were
  pushed; no AI co-author lines were added.

## Gap table: initial state and final disposition

“Matches” denotes an existing feature; “Elsewhere” denotes a different name or
location; “Different” denotes different behavior. Estimates are engineering effort,
including validation, rather than elapsed promises.
The initial analysis was committed before implementation; the final column below
records completed work and explicit deferrals.

| Paper item | Before | Action or remaining work |
|---|---|---|
| Two tools; six training steps; operational GUI | Elsewhere: one server, Detect and Improve; seven training steps | Completed: paper names are visible, with six training steps plus the evaluation extension explained. |
| Training GUI/CLI parity; classes `ship` and `ships` | Matches | Preserved shared backend and CLI. New augmentation presets are exposed through the same helper, HTTP defaults and training CLI. |
| Step 1 RGB import, albums, group selection, stable content identifiers, re-import preservation, referenced album membership and import audit | Matches; heading “Import imagery” | Completed: “Import imagery & albums”; behavior preserved. |
| Step 1 identifies images by pixel content | Different: source-file byte hash plus render profile, rather than decoded pixel hash | Not changed: renamed byte-identical files remain recognized; re-encoding identical pixels can create a new identity. Pixel-based identity and migration of existing review references require 2–3 days. |
| Step 2 auto proposals; accept, resize, relabel, reject, delete and draw boxes; overview/detail modes; keyboard shortcuts | Matches; “Manual Review” | Completed: annotation terminology appears alongside Manual Review. |
| Three review labels; retained negative regions; reviews survive regeneration | Matches | Preserved behavior. Negative boxes are excluded from positive labels; their pixels remain in training images. |
| Competing completed reviews block export | Matches | Preserved conflict warnings and export block. Dedicated GUI reconciliation remains absent (1–2 days); existing reviews can be reopened and corrected. |
| Step 3 cropped grid, class/warning/search filters, sorting, thumbnail size, metadata, editor deep links; conflict exclusion/count | Matches; “BBox Overview” | Completed: the heading uses reviewed bounding-box terminology. |
| Step 4 album/label/search selection, default 80/10/10, save then export, frozen splits, incomplete/conflict blocking | Matches | Preserved behavior. |
| Step 4 assigns splits per image | Different: defaults to whole observation groups to prevent scene leakage; image mode remains available | Retained safer default; difference documented here. |
| Step 5 dataset/starting weight/settings, named config, readiness and overwrite checks, online augmentation | Matches, under Advanced | Completed: augmentation appears in the step heading. |
| Four augmentation presets: none/light/default/heavy | Missing: fill-default and clear controls only | Completed: shared backend/API/CLI and GUI presets, with custom edits retained. |
| Multiple starting weights expand one config to a run group | Missing: one starting weight per configuration | Not done tonight. Save one config per model with a shared group; automatic suite creation/launch needs 1–2 days. |
| Step 6 detached training, live framework metrics/log, stop, retained weights/metrics/plots | Matches; “Train YOLO” | Completed: “Train detector”; existing process management preserved. |
| Launch provenance: command, config and dataset hashes, source revision/dirty state, environment versions/device | Matches | Preserved records. |
| Eight-record reproducibility contract and evaluation report | Elsewhere: records exist; report/promotion in Step 7 | Completed: visible eight-record contract and explicit Step 7 relationship. |
| Runtime stage 1 candidate detection, configurable confidence | Matches; “Run detection” | Completed: “Scan Images” and the three runtime stages are visible. Operational output retains only `ships` pairs; single-vessel annotation remains in Improve. |
| Low=.10 / Medium=.25 / High=.45 presets | Missing | Completed: named threshold presets and explanation of weaker predictions at lower thresholds. |
| Reject boxes occupying more than half the tile | Missing | Completed: shared safety filter and fixed area threshold in run provenance. |
| Runtime stage 2 WGS84 center, coarse bounds plus haversine AIS matching; default radius 1,000 m | Matches | Completed: “AIS search radius” and default are visible; same-UTC-day matching retained without a time window. |
| FULLY_DARK / PARTIALLY_DARK / AIS_PRESENT | Matches in exports; plain AIS-count text in GUI | Completed: status codes appear alongside count-based interpretation. No hull identity or legality inference. |
| DETECTED_ONLY / AIS_ERROR | Different: AIS_MISSING / AIS_INVALID preserve diagnostic detail | Completed: Results shows DETECTED_ONLY / AIS_MISSING when no AIS was supplied and AIS_ERROR / AIS_INVALID or AIS_MISSING when verification fails. Empty AIS input no longer falls back to cached files or searches the current directory. Stored/exported status codes remain unchanged. |
| NO_STS | Different: empty detection array and zero-count run summary | Completed: NO_STS appears for completed zero-candidate scans; detection exports remain empty. |
| User Mode / Intelligence HQ, cumulative statistics | Elsewhere: Home has model/data/recent work, no cumulative detection totals | Completed: paper heading, workflow links and cumulative completed-scan totals. |
| Scan Images accepts one/multiple uploaded tiles, sensitivity, optional AIS source | Elsewhere: dated existing-folder inputs on Area & dates; settings on Run; AIS on Data. Operational upload absent | Completed: paper vocabulary and navigation. Direct operational file upload deferred (1 day, including validation/containment); existing folder supports one or multiple tiles. |
| STS Events cards with annotated image, AIS vessel data and coordinates | Exists but gated: experimental Results table and selected crop | Completed: validated and exposed existing Results with table/detail layout and GPS coordinates. A full card-layout rewrite was unnecessary. |
| Live Map: OSM detections, clickable popups | Exists but gated; selected candidate only; tooltips | Completed: Results map shows all run candidates and clickable candidate/AIS popups. |
| Export Report: CSV and JSON | Elsewhere: Jobs downloads; gated Results exports add GeoJSON | Completed: paper heading and exports in Results and Jobs. |
| AIS Manager: date, range, month, year | Elsewhere: Data downloads date range from Area & dates; month/year shortcuts absent | Completed: date-period shortcuts use existing shared project/date-range behavior. |
| AIS Manager: five world-region presets and custom bbox | Different: four US presets; custom coordinates supported | Completed: Gulf of Mexico US preset and explicit US coverage warning. Other four world regions need non-US AIS providers (at least 1–2 weeks plus licensing/access), not misleading NOAA choices. |
| AIS Manager: bbox-filtered local files | Different: full US day cached; optional bbox filtering in acquisition backend/CLI | Preserved full-day reuse. Dedicated filtered-download GUI choice not done (1–2 days); matching is geographically filtered. |
| AIS Manager: automatic zst decompression and one-click apply | Matches: zst for newer days, zip for older days; successful acquisition selects source automatically | Completed: AIS Manager is visible; existing behavior preserved. No live full-day download was performed. |
| Scan history persists as JSON and restores on restart | Matches: Jobs and immutable run outputs; gated Results lists runs | Completed: “Scan history” is visible; saved runs were recovered after restart. |
| Clear all history | Missing | Not done: destructive history deletion (1 day for contained helper, API/CLI, explicit GUI confirmation and tests). |
| Developer Mode: single-image validation, lightweight retraining, reference tab | Elsewhere/partial: one-file Detect folder; Improve steps 5–7; no separate reference tab | Completed: Improve is linked as Training Tool, with operator reference text. Separate daemon-thread trainer or second mode would duplicate stable workflows (2–3 days if required). |
| Flask, Chart.js, Windows base-path update | Different implementation: shared portable standard-library server, HTML summaries, Linux/Windows support | Preserved portable server; do not merge historical dashboard. |
| Data: RGB WGS84 GeoTIFFs, 640px/10m tiles, cloud exclusion >35% | Different: acquisition cloud default 30%; runtime rejects non-north-up/non-WGS84 inputs; training accepts RGB beyond Sentinel-2 | Completed: new project and imagery acquisition defaults are 35%; existing saved choices and geometry validation are preserved. |
| Data/Results: reviewed positives and retained negatives; train-only augmentation; held-out evaluation | Matches workflow; published historical datasets/results are not recreated by opening the app | Preserved scene-safe splitting and separate Step 7 evaluation. Historical v4 offline crop generator is a specialized script, not a GUI action; a general offline-augmentation GUI would need 2–4 days. |
| Eleven-architecture comparison, 300 epochs/60 patience, batch 8, 960px, seed 42 | Different: editable setup defaults 150/30, batch 8, 960px, seed 42 | Retained general defaults; paper experiment settings can be entered in Advanced. No retraining or historical accuracy claim in this task. |
| Polygon-to-box conversion and preview utility | Matches: standalone scripts | Preserved utilities; no new GUI required by the paper. |

## Paper names and application locations

This mapping is suitable for the README. README itself was not edited.

| Paper name | Released application location |
|---|---|
| User Mode / Operational Tool / AIS Verification | Detect: Area & dates → Data → Run → Results; Home provides links to each paper-named function. |
| Intelligence HQ | Home (`/`): model/data inventory, cumulative completed-scan totals, recent work. |
| Scan Images | Detect → Run (`/run`): sensitivity and radius under “Sensitivity, AIS search radius & model”. Select the imagery folder and dates on Area & dates (`/detect`); select AIS on Data. |
| STS Events | Detect → Results (`/results`): candidate table, annotated image, GPS center, AIS evidence and human verdict. Completed Jobs link directly to their run. |
| Live Map | Results → Live Map: all candidates in the selected run; click a marker for its popup and evidence. The selected candidate adds same-day AIS positions and the search radius. |
| Export Report | Results → Export Report (CSV/JSON/GeoJSON), or completed Jobs → Export Report & results (original CSV/JSON). |
| AIS Manager | Detect → Data (`/data`); dates/range/month/year and US region selection on Area & dates. Custom coordinates are available there; NOAA coverage is US-only. |
| Scan history | Jobs (`/jobs`), plus Results' saved-run selector. |
| Training Tool / Six-Step Pipeline | Improve (`/import`), with the original six stages and an additional explicit evaluation stage. |
| Step 1: Import and album-based provenance | Improve → 1 Import & albums (`/import`). |
| Step 2: Auto-assisted bounding-box annotation | Improve → 2 Annotation (`/manual-review/`), headed “Auto-assisted annotation”, with the existing Manual Review controls. |
| Step 3: Reviewed bounding-box overview | Improve → 3 Reviewed boxes (`/bbox-overview`). |
| Step 4: Dataset selection | Improve → 4 Dataset Selection (`/datasets`). |
| Step 5: Augmentation and training setup | Improve → 5 Training Setup (`/training-setup`); augmentation presets and editable training parameters are in Advanced. |
| Step 6: Train detector | Improve → 6 Train detector (`/training`). |
| Evaluation report / Reproducibility contract | Improve → 7 Evaluate (`/evaluation`), headed “Evaluate & Promote”; the eight-record contract is expandable above the evaluation controls. |
| Developer Mode | Existing tasks are distributed across single-tile Detect and Improve; Home → Operator reference & Training Tool explains this relationship. There is no separate mode switch or second trainer. |

## Validation

The isolated application ran on Linux at `http://127.0.0.1:8778`, with workspace
`workspace/paper_alignment` and a detached server. It uses the existing default
checkpoint copied by `init --weights`, including the existing adjacent recreation
documentation. The model choice, model documentation, README and historical
dashboard source were not changed. The read-only AIS source directory was not used.

- `/home/gavin/OpenDarkSTS/.venv/bin/python -m pytest -q`: **362 passed**, no skips,
  in 33.07 s. The worktree workspace was initialized with the documented existing
  weight; the bundled tile was copied into the legacy integration-test input
  directory so both real-inference tests executed. No new trained weights were made.
- Chromium acceptance: **44 checks** across all thirteen main pages, including the
  four sensitivity/custom-value cases, leap-year month/year shortcuts, four editable
  augmentation presets, downloads, persistent review, and clickable map evidence.
  Every main page fit 1366 × 900 and 390 × 844 viewports without horizontal page
  overflow. No page JavaScript errors were recorded.
- A real GUI scan of the bundled example completed with **two AIS_PRESENT candidates**
  at confidence 0.25 and radius 1,000 m. Both original CSV/JSON downloads and reviewed
  CSV/JSON/GeoJSON downloads succeeded. These are software checks, not new scientific
  validation labels or model accuracy measurements.
- Six supplementary browser checks covered real no-source, invalid-AIS and
  zero-candidate runs: DETECTED_ONLY / AIS_MISSING, AIS_ERROR / AIS_INVALID, and
  NO_STS respectively. A blocked map-viewer script left coordinates, images and AIS
  evidence usable. A further valid scan restored the ordinary example settings.
- Restarted-server acceptance checked the saved scan history and new controls.
  CLI `training-defaults` reported the same augmentation values as the GUI;
  `training save-training-config --aug-preset light --aug degrees=15` saved explicit
  values successfully. Unit tests verify overrides, disabled transforms, the strict
  half-tile boundary and explicit empty-AIS preservation.
- Production changes use the existing Python/browser interfaces, with no new
  dependencies or platform-specific process handling. Windows execution, live CDSE
  authentication, full NOAA downloads and long training were not revalidated.

Local evidence: `workspace/reports/paper_alignment/pytest.log`, `gui_results.json`,
`status_results.json`, the two `verify_*.py` browser scripts, download copies,
server log and CLI JSON records. Initial screenshots and extracted page text are
in `workspace/reports/paper_alignment/before/`. Mobile screenshots are in the same
report directory. All test data are inside this repository; the requested screenshot
copies are in the main checkout's gitignored workspace.

## Screenshots

All PNGs are under `/home/gavin/OpenDarkSTS/workspace/release_screens/`:

| Page | Filename |
|---|---|
| Home / Intelligence HQ | `home.png` |
| Detect: Area & dates | `detect_area_dates.png` |
| Detect: Data / AIS Manager | `detect_data.png` |
| Detect: Run / Scan Images | `detect_run.png` |
| Detect: Results / STS Events / Live Map / Export Report | `detect_results.png` |
| Improve 1 | `improve_1_import.png` |
| Improve 2 | `improve_2_annotation.png` |
| Improve 3 | `improve_3_reviewed_boxes.png` |
| Improve 4 | `improve_4_dataset_selection.png` |
| Improve 5 | `improve_5_training_setup.png` |
| Improve 6 | `improve_6_train_detector.png` |
| Improve 7 | `improve_7_evaluation.png` |
| Jobs / scan history | `jobs.png` |
| Supplementary controls and outcomes | `improve_5_augmentation_presets.png`, `detect_results_detected_only.png`, `detect_results_ais_error.png`, `detect_results_no_sts.png` |

Training screenshots show the isolated workspace, rather than the original reviewed
training collection. Its import/review/dataset screens are initially empty.
