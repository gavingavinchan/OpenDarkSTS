# OpenDarkSTS

OpenDarkSTS finds candidate ship-to-ship (STS) transfers in Sentinel-2 satellite
imagery and compares each candidate with nearby US AIS vessel records from the same
day. A candidate with few or no AIS vessels nearby is a possible *dark* transfer and
is flagged for human review. Results are candidates for review, not evidence of
wrongdoing.

This is the tool presented in the SNAME Maritime Convention 2026 paper. It runs
locally on Linux or Windows as one web application with two parts:

- **Detect** (the paper's *Operational Tool: Flask GUI and AIS Verification*): scan
  images with the trained detector, match candidates with AIS, review and export.
- **Improve** (the paper's *Training Tool: Six-Step Pipeline*): import imagery,
  annotate, review, build a dataset, train, and evaluate a new detector.

The default detector is the paper's headline **YOLO11s** model (held-out test
mAP50 0.895; see [Default model](#default-model)). The annotated image datasets are
published on Hugging Face (see [Datasets](#datasets)).

## Install and start

Use Python 3.11–3.13 (3.12 recommended). A GPU is optional for detection; the
PyTorch download can take several minutes.

```bash
git clone https://github.com/gavingavinchan/OpenDarkSTS.git
cd OpenDarkSTS
python -m venv .venv
# Linux
source .venv/bin/activate
# Windows PowerShell, instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

From the folder where the workspace should live:

```bash
opendarksts init
opendarksts serve
```

`init` creates the workspace, copies a small example, and downloads the default
model from the [v0.1 release](https://github.com/gavingavinchan/OpenDarkSTS/releases/tag/v0.1),
checking its SHA-256. If you already have the checkpoint file, use
`opendarksts init --weights PATH_TO_CHECKPOINT` instead.

Open the URL printed by `serve` (normally http://127.0.0.1:8766). The server runs in
the background and writes its log to `workspace/reports/server/`; running `serve`
again returns the existing server. Every command accepts `--workspace PATH` before
its subcommand, or reads `OPENDARKSTS_WORKSPACE`.

## Try the bundled example

No credentials or downloads are needed after `init`:

```bash
opendarksts run --images workspace/inputs/example/2020-01-24_example.tiff --ais-dir workspace/inputs/example/AIS_2020_01_24.csv
```

The command prints its output folder, which holds `dark_sts.csv`, `dark_sts.json`,
`run.json` and annotated `previews/`. The example produces one candidate near
28.6564 N, 94.3228 W with two AIS vessels within 1,000 m that day (`AIS_PRESENT`).
The example's [provenance and limits](opendarksts/example/README.md) are copied
into the workspace.

In the web application: on **Detect → Area & dates**, enter the example imagery
folder `workspace/inputs/example` and the date **2020-01-24**; on **Data / AIS
Manager**, select the example AIS CSV; then start the scan on **Run / Scan Images**.
Candidates appear on **Results / STS Events**, with the Live Map and Export Report.

## Where the paper's features are

| Paper | Application |
|---|---|
| Intelligence HQ (cumulative statistics) | **Home** |
| Scan Images (sensitivity Low 0.10 / Medium 0.25 / High 0.45, AIS search radius) | **Detect → Run / Scan Images**; choose imagery and dates on **Area & dates** |
| AIS Manager (NOAA Marine Cadastre download, region presets, custom box) | **Detect → Data / AIS Manager**, with dates and region on **Area & dates** |
| STS Events (annotated image, GPS position, nearby vessels) | **Detect → Results / STS Events** |
| Live Map | **Results → Live Map** (click a marker for its evidence) |
| Export Report (CSV / JSON) | **Results → Export Report** (also GeoJSON), or **Jobs** |
| Scan history | **Jobs**, and the saved-run selector on **Results** |
| Developer Mode | **Improve**; **Home → Operator reference & Training Tool** explains the link |
| Training Step 1: Import and album-based provenance | **Improve → 1 Import & albums** |
| Step 2: Auto-assisted bounding-box annotation | **Improve → 2 Annotation** |
| Step 3: Reviewed bounding-box overview | **Improve → 3 Reviewed boxes** |
| Step 4: Dataset selection | **Improve → 4 Dataset Selection** |
| Step 5: Augmentation and training setup | **Improve → 5 Training Setup** (augmentation presets under Advanced) |
| Step 6: Train detector | **Improve → 6 Train detector** |
| Evaluation report and reproducibility contract | **Improve → 7 Evaluate & Promote** |

Since the paper, the two tools have been merged into one application that runs on
Linux and Windows, and the evaluation that the paper reports separately is now a
seventh step in Improve. A detailed comparison is in
[docs/paper_release_alignment.md](docs/paper_release_alignment.md).

## Use your own area and dates

On **Detect → Area & dates**, choose a US preset, draw a box on the map, or enter
an existing image folder. **Data / AIS Manager** downloads Sentinel-2 imagery and
whole NOAA AIS days in the background. Sentinel-2 downloads need a free Copernicus
Data Space account: set `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in the server's
environment before starting it. NOAA downloads need no credentials, but a full AIS
day is large. See [acquisition](docs/acquisition.md).

The same operations are available from the command line:

```bash
opendarksts acquire imagery --start 2020-01-24 --bbox -94.73 29.07 -94.57 29.205 --background
opendarksts acquire ais --start 2020-01-24 --background
opendarksts jobs
opendarksts run --images workspace/inputs/imagery --ais-dir workspace/inputs/ais/full --background
```

`opendarksts project --check-ready` checks the saved Detect settings and reports
the same image count, blockers and AIS warnings as the Run page.

Input images must be north-up EPSG:4326 (WGS84). Plain RGB images need a matching
`IMAGE.json` containing `{"bbox": [west, south, east, north]}`. File names must
include the date as `YYYY-MM-DD`. Input folders are only read; all derived files stay
in the workspace. If several AIS files match a date, select the intended day's CSV.

## Read the results

Each scan runs the three stages described in the paper: (1) the detector finds STS
candidates (boxes covering more than half of a tile are discarded as implausible);
(2) each candidate's centre is converted to latitude and longitude, and AIS
positions within the search radius (default 1,000 m) on the same UTC day are
counted; (3) the candidate receives a dark-status label.

| Status | Meaning |
|---|---|
| `FULLY_DARK` | No AIS vessel within the radius that UTC day |
| `PARTIALLY_DARK` | One AIS vessel within the radius that UTC day |
| `AIS_PRESENT` | Two or more AIS vessels within the radius that UTC day |
| `DETECTED_ONLY` (stored as `AIS_MISSING`) | Candidate found, but no AIS data were supplied |
| `AIS_ERROR` (stored as `AIS_INVALID`) | Candidate found, but the AIS file could not be used |
| `NO_STS` | The scan found no candidate |

The CSV lists the two closest AIS vessels; the JSON keeps all matches. Each run
records the model checksum, inputs and settings.

The AIS comparison has **no time window** and no track association: an AIS position
may be hours away from the satellite pass and need not belong to either visible
hull. AIS coverage is US only (NOAA Marine Cadastre). Cloud, image resolution,
duplicate detections and detector errors also limit what can be concluded.

## Default model

`opendarksts_yolo11s_v4_best.pt`, the YOLO11s detector reported in the paper,
trained on the v4 balanced dataset (classes `ship` and `ships`). Held-out test
results (21 images, 92 instances):

| Class | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| `ship` (single vessel) | 0.812 | 0.806 | 0.869 | 0.512 |
| `ships` (STS pair) | 0.849 | 0.833 | 0.920 | 0.575 |
| all | 0.830 | 0.820 | 0.895 | 0.544 |

The test split is small, and some test images share source scenes with training
images. The [model card](opendarksts/assets/model_card.md) gives the SHA-256,
training settings and limitations. A model trained in **Improve** can be promoted
on step 7 to replace the default for both Detect and annotation prefill.

## Improve the model

**Improve** follows the paper's training pipeline: import imagery into albums,
correct automatically proposed boxes, check all reviewed boxes in one grid, select
a dataset with fixed training/validation/test splits, set up augmentation and
training, train, then evaluate and promote. Images from the same region and date
are kept in the same split. Every step is also available from the command line:

```bash
opendarksts training --pretty import-data --source PATH_TO_IMAGES --auto-detect
opendarksts training --help
```

See the [training CLI reference](docs/training_pipeline_cli.md). CLI responses are
JSON; input errors return exit code 2, execution failures 1, and training launch
conflicts 3.

## Datasets

The annotated Sentinel-2 images are published on Hugging Face in YOLO format:

- **OpenDarkSTS annotation dataset**: the paper's 474 reviewed images with 2,636
  reviewed boxes, including the rejected regions kept as hard negatives.
- **OpenDarkSTS v4 balanced dataset**: the training, validation and test data
  of the default YOLO11s model.

Links: _to be added on publication._

## Development

```bash
python -m pytest -q
python -m build
```

Tests mock network downloads. The real-inference tests run when
`OPENDARKSTS_TEST_WEIGHTS` points at the default checkpoint. Windows checks are
tracked in [docs/windows_validation.md](docs/windows_validation.md).

## Licence and attribution

OpenDarkSTS, including the default model checkpoint, is licensed under
[AGPL-3.0-only](LICENSE). It uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is also AGPL-3.0. The annotated datasets are licensed under CC BY 4.0.
Third-party notices are in [docs/licensing.md](docs/licensing.md).

Contains modified Copernicus Sentinel data 2020–2025. AIS data: NOAA Office for
Coastal Management and US Coast Guard (Marine Cadastre).

## Citation

If you use OpenDarkSTS, please cite the SNAME Maritime Convention 2026 paper
(full reference to be added).
