# OpenDarkSTS

Find candidate ship-to-ship transfers in satellite imagery and compare them with
nearby US AIS vessel records from the same day. OpenDarkSTS runs locally: one web
application provides data acquisition, detection, training, and job logs.
Results are candidates for human review, not evidence of wrongdoing.

This branch prepares version 0.1. The release checkpoint has not yet been published,
and contributor approval of the proposed licence is pending. See the
[release status](docs/release_progress.md).

## Install and start

Use Python 3.11–3.13 (3.12 recommended), on Linux or Windows. Create one environment
for the application, training and tests. A GPU is optional for inference; downloading
PyTorch can take several minutes.

```bash
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

Open the URL printed by `serve` (normally http://127.0.0.1:8766). The server runs in
the background and writes its log to `workspace/reports/server/`. Repeating `serve`
returns the existing server. Every command accepts `--workspace PATH` before its
subcommand, or reads `OPENDARKSTS_WORKSPACE`.

`init` downloads and checks the default model from the GitHub release. **Before that
asset is published**, use `opendarksts init --weights PATH_TO_THE_DEFAULT_CHECKPOINT`.
The expected file and SHA-256 are in the [model card](opendarksts/assets/model_card.md).
`init --no-model` prepares the folders and example without downloading weights.

## Try the bundled example

No CDSE credentials or additional data download are needed after initialization:

```bash
opendarksts run --images workspace/inputs/example/2020-01-24_example.tiff --ais-dir workspace/inputs/example/AIS_2020_01_24.csv
```

The command prints the output directory. Inspect `dark_sts.csv`, `dark_sts.json`,
`run.json`, and `previews/`. The example should produce a candidate near
28.6564 N, 94.3228 W with two nearby AIS vessels that day. Its
[provenance and limits](opendarksts/example/README.md) are included in the workspace.

For the GUI, set the example imagery folder and date **2020-01-24** in Detect, set
its AIS CSV on Data, then open Run. Jobs shows the log and output downloads.

## Use your own area and dates

In Detect, choose a US preset, draw a box, or enter an existing image folder. Data
can download Sentinel-2 imagery and complete NOAA AIS days in the background.
Sentinel downloads require `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in the server's
environment; NOAA downloads do not require credentials. Set credentials before
starting the server. Full AIS days can be large. See [acquisition](docs/acquisition.md).

The same operations are available from the CLI:

```bash
opendarksts acquire imagery --start 2020-01-24 --bbox -94.73 29.07 -94.57 29.205 --background
opendarksts acquire ais --start 2020-01-24 --background
opendarksts jobs
opendarksts run --images workspace/inputs/imagery --ais-dir workspace/inputs/ais/full --background
```

`opendarksts project --check-ready` checks the saved Detect settings and returns
the same date-matched image count, launch blockers, and AIS-source warnings used
by the Run page. A successful command can return `ready: false`; inspect the
`blockers` before launching. Missing AIS remains a warning because detection can
produce candidates with unknown vessel presence.

Input images must be north-up EPSG:4326. Plain RGB images require a matching
`IMAGE.json` containing `{"bbox": [west, south, east, north]}`. Filenames must include
`YYYY-MM-DD`. External input folders are read only; derived files stay in the
workspace. If several AIS files match a date, select the intended day CSV explicitly.

## Read the output

The default confidence is 0.25; the AIS radius is 1000 metres (`--conf`, `--radius`).

| Stored status | Meaning |
|---|---|
| `FULLY_DARK` | No AIS vessels within the radius that UTC day |
| `PARTIALLY_DARK` | One AIS vessel within the radius that UTC day |
| `AIS_PRESENT` | Two or more AIS vessels within the radius that UTC day |
| `AIS_MISSING` / `AIS_INVALID` | Vessel presence cannot be assessed; count is unknown |

CSV includes the two closest AIS vessels; JSON retains all matches. Each run records
its model checksum, selected inputs and settings. Previews show detections and AIS
context for manual inspection.

This comparison has **no time window** or track association. An AIS position can be
hours away from the satellite observation and need not belong to either visible hull.
New downloads retain acquisition metadata, but the matcher still uses the entire day.
US coverage, cloud, image resolution, duplicate detections and model errors limit what
can be inferred. The [model card](opendarksts/assets/model_card.md) describes the clean
split metrics and known limits.

## Improve the model

Improve provides seven steps: import, visual review, reviewed overview, dataset
selection, training setup, training run, and evaluation/promotion. Whole region/date
groups stay together across dataset splits. Promoting a documented checkpoint makes
it the current model for both Detect and review prefill.

```bash
opendarksts training --pretty import-data --source PATH_TO_IMAGES --auto-detect
opendarksts training --help
```

See the [training CLI reference](docs/training_pipeline_cli.md) and
[training plan](docs/training_pipeline_plan.md). CLI responses are JSON; input errors
return exit code 2, execution failures 1, and training launch conflicts 3.

## Development and data terms

```bash
python -m pytest -q
python -m build
```

Tests mock network acquisition. A real inference test uses the documented checkpoint
when available; set `OPENDARKSTS_TEST_WEIGHTS` to its path. Actual Windows validation
is tracked in [the Windows checklist](docs/windows_validation.md).

Contains modified Copernicus Sentinel data 2020. AIS example: NOAA Office for Coastal
Management / US Coast Guard. Third-party notices, the proposed code licence and
training-data rights are tracked in [licensing](docs/licensing.md). The
[release plan](docs/release_plan.md) is the plan of record.
