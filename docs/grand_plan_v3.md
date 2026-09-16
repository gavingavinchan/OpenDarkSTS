# OpenDarkSTS — Grand Plan v3 (as-built)

Detect dark ship-to-ship (STS) transfers: find STS pairs in Sentinel-2 imagery with a
YOLO detector, then cross-check each detection against US AIS records for the same day
and location. Vessels visible in the imagery but missing from AIS are dark.

This document describes the system **as it actually exists today**. It supersedes
`docs/grand_plan_v2.md`, which described a two-zoom-level design with a separate
"dot detector" that was never built and is no longer planned.

---

## What changed from v2

| v2 design | v3 reality |
|---|---|
| Two YOLO models (dot detector + STS detector) | One model: the STS detector |
| Two zoom levels (~55 m/px wide + ~18 m/px close) | One zoom level: Sentinel-2 10 m/px tiles |
| False-colour NIR tier for wide-area ship spotting | True colour only |
| Global AIS ambitions | US AIS (NOAA MarineCadastre) only |
| AIS time-window matching | Same-day, radius-only matching (no time window) |

---

## Data

### Imagery
- Sentinel-2 L2A true colour, **10 m/pixel**, 3-band RGB GeoTIFF, EPSG:4326.
- Tiles are named `YYYY-MM-DD_tile_X_Y.tiff`, with a `YYYY-MM-DD_scene1_tile_X_Y.tiff`
  variant when a date has more than one scene. The date in the filename is the only
  source of acquisition date; there is **no acquisition time** in the files.
- Source of truth for the corner coordinates is the GeoTIFF itself (read via `rasterio`
  bounds). Non-GeoTIFF inputs (JPEG/PNG) need a sidecar `<image>.json` of the form
  `{"bbox": [lon_min, lat_min, lon_max, lat_max]}`.
- Working set on this machine: `/home/gavin/Downloads/AIS data/Sentinel 2A/` (read-only).
  `sentinel2/` covers Galveston / the Texas Gulf, 2020-2024.

### AIS
- NOAA MarineCadastre daily CSVs, one file per date: `AIS_YYYY_MM_DD.csv`.
- Columns used: `MMSI, BaseDateTime, LAT, LON, SOG, COG, VesselName, VesselType, Length`.
- `BaseDateTime` is **UTC** (verified).
- AIS is a cross-reference, never ground truth: it can be switched off, spoofed, or
  badly calibrated. The imagery is the spatial ground truth.

### Model
- Promoted weight: `workspace/model_weights/yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt`
- Classes: `ship` (class 0, a single vessel) and `ships` (class 1, an active STS pair,
  one box around both hulls). The operational pipeline keeps only `ships`.
- First leak-free evaluation (`sts_clean_v1` test split): overall mAP50 `0.868`,
  `ships` mAP50 `0.839`. See `docs/progress.md`, section "Clean Evaluation v1".
- Recreation recipe: `docs/yolo_weight_recreation.md`.

---

## Pipeline A — Training (steps 1-7, GUI + CLI)

Purpose: curate reviewed imagery into YOLO datasets, train, evaluate, promote a weight.

```
1 Import          local imagery -> import manifest + browser-ready PNG cache
                  (optional YOLO auto-detect to prefill boxes)
2 Visual review   human confirms/edits boxes per image (ship / ships / reject)
3 Reviewed bboxes overview of everything reviewed so far
4 Dataset select  choose images -> group-aware train/val/test split -> YOLO export
5 Training setup  starting weight + augmentation + hyperparameters -> training config
6 Training run    pipeline-managed YOLO run, live status, stop/resume
7 Evaluation      test-split metrics per run group, compare weights, promote one
```

- GUI: `scripts/training_pipeline_ui.py` (default port 8766).
- CLI parity: `scripts/training_pipeline_cli.py` exposes a JSON command for every GUI
  action (`import-data`, `export-dataset`, `check-training-ready`, `start-training`,
  `training-status`, `run-test-split`, `promote-weight`, ...). Full reference in
  `docs/training_pipeline_cli.md`.
- Splits are **group-aware**: tiles from the same date/scene never straddle a split, so
  test metrics are not inflated by near-duplicate crops. See `docs/progress.md`.

---

## Pipeline B — Operational (satellite-first, one command)

```
[folder of GeoTIFF tiles for a date]
      |
      | scripts/pipeline.py
      |
      |-- select_images: resolve dirs/globs, skip zero-byte and unreadable rasters,
      |                  parse the date from the filename, optionally filter to --date
      v
[detect.py]  YOLO inference with the promoted weight; keep class `ships` only
      v
[georeference.py]  pixel box centre -> lat/lon via rasterio bounds (or sidecar bbox)
      v
[correlate_ais.py]  stream AIS_YYYY_MM_DD.csv, keep pings within --radius (default
                    1000 m) of the detection centre, one row per vessel (closest ping)
      v
[output]  dark_sts.csv + dark_sts.json + previews/<detection_id>.png
```

Dark status per detection:

| AIS vessels within radius | Status | Reading |
|---|---|---|
| 0 | `FULLY_DARK` | Two hulls visible, nobody transmitting |
| 1 | `PARTIALLY_DARK` | One side transmitting, the other dark |
| 2+ | `AIS_PRESENT` | Both sides accounted for; check type/size/speed for inconsistencies |

Each output row carries `MMSI`, `VesselName`, `BaseDateTime` (the matched ping, UTC),
`VesselType`, `SOG`, `COG`, `Length` and `distance_m` for the two closest vessels; the
JSON carries every match. Every detection also gets an annotated PNG (crop around the
box, box drawn in red, AIS matches printed underneath) so a human can adjudicate.

---

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| `--weights` | promoted `yolo26n_sts_clean_v1_...best.pt` | Override to compare models |
| `--conf` | `0.25` | YOLO confidence threshold |
| `--radius` | `1000` m | AIS search radius from the STS centre |
| `--date` | inferred per file | Restrict a mixed folder to one acquisition date |
| `--preview-dir` | `<output parent>/previews` | `none` disables preview rendering |

A fixed radius rather than the box footprint is deliberate: AIS position and image
capture are minutes apart, so a tight radius would invent dark vessels.

---

## Known limitations

- **No time window.** A detection matches any ping from the same UTC day within the
  radius. A vessel that passed the spot twelve hours earlier still counts as "present".
  The matched `BaseDateTime` is recorded so a reader can judge each hit by eye. Adding a
  window is deliberately deferred.
- **No acquisition time.** Sentinel-2 capture time is not stored in the tiles, so there
  is nothing to centre a time window on yet.
- **US AIS only.** NOAA MarineCadastre coverage is US waters. Outside it, everything
  would look dark, so the pipeline must not be pointed at non-US imagery as-is.
- **One zoom level.** Detection quality is bounded by 10 m/pixel; small craft alongside
  a tanker may not resolve as two hulls.
- **Zero-byte rasters.** About 4,900 files under
  `Sentinel 2A/2020/ship_tiles new Apr 29/2025/` are zero bytes. The pipeline skips them
  with a warning rather than crashing.
- **No automated AIS-vs-imagery consistency check.** Vessel type/length/speed are
  reported, but nothing flags "AIS says fishing boat, image shows a 300 m tanker" yet.

---

## Next steps

1. Run the operational pipeline over more of the 59 dates where a Sentinel-2 tile set
   and a US AIS day file both exist, and triage the previews.
2. Feed adjudicated detections (true and false positives) back into step 1 of the
   training pipeline to grow `sts_clean_v1`.
3. Retrain on a longer schedule — the promoted run had not converged at 300 epochs.
4. Only then consider a time window, acquisition-time recovery, or non-US AIS.
