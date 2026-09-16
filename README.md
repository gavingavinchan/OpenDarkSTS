# OpenDarkSTS

Detect dark ship-to-ship (STS) transfers. A YOLO detector finds STS pairs in Sentinel-2
imagery; each detection is then cross-checked against US AIS records for the same day and
location. Two visible hulls with no AIS behind them are a dark STS candidate.

- System overview, as built: [`docs/grand_plan_v3.md`](docs/grand_plan_v3.md)
- Training pipeline plan: [`docs/training_pipeline_plan.md`](docs/training_pipeline_plan.md)
- Training pipeline CLI reference: [`docs/training_pipeline_cli.md`](docs/training_pipeline_cli.md)
- Running log of results and decisions: [`docs/progress.md`](docs/progress.md)

## Environment

Two interpreters, on purpose:

| Interpreter | Use it for | Why |
|---|---|---|
| `.venv/bin/python` | anything touching `ultralytics`, `torch`, `rasterio` — inference, training, the operational pipeline, the training GUI | the project venv (see [`docs/local-venv.md`](docs/local-venv.md)) |
| `/home/gavin/anaconda3/bin/python3` | `pytest` | the venv has no `pytest`; anaconda has no `ultralytics`/`rasterio` |

The venv's `sqlite3` module is broken — do not use sqlite from `.venv/bin/python`.

```bash
# tests
/home/gavin/anaconda3/bin/python3 -m pytest -q
```

Tests that need `rasterio` or `ultralytics` skip themselves under anaconda, so the suite
runs green without the heavy dependencies.

## Operational pipeline (one command)

Given a folder of Sentinel-2 GeoTIFF tiles for a date, detect STS pairs, georeference
them, and cross-check the matching US AIS day file:

```bash
.venv/bin/python scripts/pipeline.py \
  --images workspace/inputs/2020_01_14 \
  --ais-dir "/home/gavin/Downloads/AIS data" \
  --date 2020-01-14 \
  --output workspace/reports/operational/2020-01-14/dark_sts.csv \
  --also-write workspace/reports/operational/2020-01-14/dark_sts.json
```

- `--weights` defaults to the promoted weight
  `workspace/model_weights/yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt`.
- `--date` is optional; without it the date is inferred per file from the filename.
- `--radius` (default `1000` m) is the AIS search radius; `--conf` (default `0.25`) is the
  YOLO confidence threshold.
- Zero-byte and unreadable rasters are skipped with a warning.
- Non-GeoTIFF inputs need a sidecar `<image>.json` of
  `{"bbox": [lon_min, lat_min, lon_max, lat_max]}`.

### Where outputs land

`workspace/reports/operational/<date>/`:

- `dark_sts.csv` — one row per detection, flattened, with the two closest AIS vessels
  (`MMSI`, name, matched `BaseDateTime`, `VesselType`, `SOG`, `COG`, `Length`, distance).
- `dark_sts.json` — the same detections with **every** AIS match, not just the closest two.
- `previews/<detection_id>.png` — the crop around each detection with the box drawn in red
  and the AIS matches printed underneath, for human adjudication.

Each detection is labelled `FULLY_DARK` (no AIS vessel within the radius),
`PARTIALLY_DARK` (one), or `AIS_PRESENT` (two or more).

Individual stages can also be run on their own: `scripts/detect.py`,
`scripts/georeference.py`, `scripts/correlate_ais.py`.

## Training pipeline

GUI (steps 1-7: import, visual review, reviewed bboxes, dataset selection, training
setup, training run, evaluation). Start it detached so the session stays usable:

```bash
nohup .venv/bin/python scripts/training_pipeline_ui.py --port 8766 \
  > workspace/reports/training_pipeline_ui.log 2>&1 < /dev/null &
# then open http://localhost:8766
```

Everything the GUI does is also available from the CLI, which returns JSON:

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty import-data --source /path/to/images --auto-detect
.venv/bin/python scripts/training_pipeline_cli.py --pretty export-dataset --help
.venv/bin/python scripts/training_pipeline_cli.py --pretty check-training-ready
.venv/bin/python scripts/training_pipeline_cli.py --pretty start-training
.venv/bin/python scripts/training_pipeline_cli.py --pretty training-status
.venv/bin/python scripts/training_pipeline_cli.py --pretty run-test-split
.venv/bin/python scripts/training_pipeline_cli.py --pretty promote-weight
```

`--human` prints a terse summary instead of JSON. Full reference and exit-code contract:
[`docs/training_pipeline_cli.md`](docs/training_pipeline_cli.md).

Training artifacts land in `workspace/`: `datasets/` (exported YOLO datasets),
`runs/` (training runs), `model_weights/` (promoted weights), `reports/` (logs and
evaluation reports), `reviews/` (human review JSON).

## Data rules

`/home/gavin/Downloads/AIS data` is **read-only**. Copy the inputs you need into
`workspace/inputs/<date>/` and work there. See [`AGENTS.md`](AGENTS.md) for the full set
of repository rules.

## Known limitations

- AIS matching has **no time window**: any ping from the same UTC day within the radius
  counts. The matched `BaseDateTime` is recorded so hits can be judged by eye.
- Sentinel-2 tiles carry no acquisition time, so there is nothing to centre a window on.
- US AIS (NOAA MarineCadastre) only — outside US waters everything would look dark.
- One zoom level (10 m/pixel true colour).
