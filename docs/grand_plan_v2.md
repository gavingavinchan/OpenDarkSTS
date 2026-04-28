# OpenDarkSTS — Grand Plan v2 (AIS-First Pipeline)

Detect dark ship-to-ship (STS) transfers using a two-stage satellite imagery approach
combined with AIS cross-referencing. The system operates in two modes: a training data
generation pipeline (AIS-first) and an operational dark STS detection pipeline
(satellite-first).

---

## Overview

There are two YOLO models in this system:

1. **Dot-detector YOLO** — operates on zoomed-out imagery (~137 km x 89 km at ~55 m/pixel).
   Ships appear as small dots on the ocean. Detects ship presence, not STS events.
   Classes: `ship` (a dot). Needs to be trained from scratch.

2. **STS YOLO** — operates on zoomed-in imagery (~1 km x 0.7 km at ~18 m/pixel). Ships
   fill the frame. Detects whether two ships are performing an STS transfer.
   Classes: `ship` (single vessel), `ships` (active STS pair, bbox around both).
   Already trained to acceptable performance (mAP50 ~0.83).

There are two pipelines:

- **Pipeline A (AIS-first)** — for generating training data. Scans AIS for candidate STS
  events, downloads imagery to confirm, builds training sets for both YOLO models.

- **Pipeline B (satellite-first)** — the operational dark STS detection pipeline. Scans
  satellite imagery of regions of interest, detects ships, confirms STS events, then
  cross-references AIS to identify dark vessels.

---

## Data Sources

### Satellite Imagery
- **Source:** Sentinel Hub API (automated GeoTIFF downloads — handled on the `images` branch)
- **Format:** GeoTIFF with embedded georeferencing (EPSG:4326), read via `rasterio`
- **Bands:** True color for zoomed-in STS detection; false color (NIR) for zoomed-out
  dot detection (ships have a distinct spectral signature against water in NIR)
- **Zoom levels:**

| Level | Coverage | Resolution | Pixel size | Purpose |
|-------|----------|------------|------------|---------|
| Zoomed-out | ~137 km x 89 km | ~55 m/pixel | 2500 x 1625 | Dot detector — find ship locations |
| Zoomed-in | ~1 km x 0.7 km | ~18 m/pixel | 61 x 39 | STS confirmation — are two ships side-by-side? |

### AIS Data
- **Format:** CSV, one file per date: `AIS_YYYY_MM_DD.csv`
- **Columns:** `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO,
  CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass`
- **Scale:** ~7.4M records per day (US-wide); loaded once per image date, filtered
  spatially on the fly
- **Trust level:** AIS is NOT ground truth. It can be tampered with, poorly calibrated,
  or switched off entirely. Satellite imagery is the spatial ground truth. AIS is used
  as a cross-reference, not as authoritative data.

### YOLO Models
- **STS YOLO** (existing): YOLO26n / YOLOv8n trained on `data-science-dark-fleet` dataset
  - Classes: `ship` (class 0), `ships` (class 1 — active STS, bbox around both vessels)
  - Best weights: `workspace/model_weights/yolo26n_madhulika_rectbbox_e150_img960_2026-03-21_best.pt`
- **Dot-detector YOLO** (to be trained): YOLO26n / YOLOv8n on zoomed-out false-color imagery
  - Classes: `ship` (a dot on the ocean)
  - Training data: generated via Pipeline A + manual annotation

---

## Pipeline A: AIS-First (Training Data Generation)

Purpose: generate labelled training data for both YOLO models by using AIS to find
candidate STS events, then downloading and reviewing satellite imagery.

The detailed training-side GUI and artifact plan lives in `docs/training_pipeline_plan.md`.
That document is authoritative for import normalization, manual visual review, reviewed
bbox overview, dataset selection, augmentation/training setup, training metadata, and
evaluation. This section only explains how the training pipeline fits into the full
OpenDarkSTS system.

```
[AIS Dataset]
      |
      | scan for candidate STS pairs:
      |   - two vessels within proximity threshold
      |   - matching speed (SOG)
      |   - other params (heading, vessel type) tuned via manual review
      v
[Candidate STS Events]  — list of (lat, lon, date, MMSI_1, MMSI_2, SOG, ...)
      |
      | download via Sentinel Hub API (images branch)
      v
[Zoomed-Out GeoTIFF]  — false color, ~137x89 km
      |
      | find where the ships actually are
      | (AIS location may be offset by ~1-2 km due to
      |  time mismatch between AIS ping and image capture)
      v
[Dot-Detector YOLO / manual inspection]
      |
      | download zoomed-in image at detected ship location
      v
[Zoomed-In GeoTIFF]  — true color, ~1x0.7 km
      |
      | confirm STS event
      v
[STS YOLO inference]
      |
      v
[Manual Review]  ←── REQUIRED before adding to any training set
      |
      ├── confirmed STS → add zoomed-in image to STS YOLO training set
      ├── confirmed ship dot → add zoomed-out image annotation to dot-detector training set
      └── false positive → discard or add as negative example
```

### AIS STS Candidate Pairing Criteria (initial, to be tuned)

| Parameter | Initial rule | Notes |
|-----------|-------------|-------|
| Proximity | Two vessels within configurable radius | Start loose, tighten after review |
| Speed | Matching SOG between the two vessels | Ships may be at any speed initially |
| Heading | Not filtered initially | Add later based on review |
| Vessel type | Not filtered initially | Add later based on review |
| Temporal | AIS pings within same time window | Same date, close timestamps |

These criteria are intentionally loose. The manual review step catches false positives.
After reviewing the first batch of results, we tighten the parameters.

### Training the Dot-Detector YOLO

The dot-detector YOLO needs to be trained first (the STS YOLO is already acceptable).

Training data generation for the dot detector:
1. Download zoomed-out **false color** GeoTIFFs via Sentinel Hub
2. Use AIS positions as initial annotation hints — overlay AIS positions on the image
   to suggest where ships might be
3. Human reviewer verifies each AIS-suggested dot AND scans the full image for any
   ships that AIS missed (AIS is unreliable — ships can be dark, spoofed, or miscalibrated)
4. Final annotations are the verified ship positions
5. Train both YOLOv8 and YOLO26 on this dataset

Why AIS cannot be used as automatic ground truth:
- AIS can be tampered with (position spoofing)
- AIS GPS can be poorly calibrated
- AIS can be switched off entirely (the whole point of "dark" ships)
- Satellite imagery is the only reliable spatial ground truth

---

## Pipeline B: Satellite-First (Operational Dark STS Detection)

Purpose: detect dark STS events in regions of interest. This is the operational pipeline.

```
[Region of Interest]
      |
      | user-designated area or historic dark STS hotspot
      v
[Download Zoomed-Out GeoTIFF]  — false color, ~137x89 km
      |
      v
[Dot-Detector YOLO]
      |
      | detect all ship dots in the image
      v
[Ship Locations]  — list of (lat, lon) per detected dot
      |
      | for each ship dot (or cluster of nearby dots),
      | download zoomed-in image at that location
      v
[Download Zoomed-In GeoTIFF]  — true color, ~1x0.7 km
      |
      v
[STS YOLO]
      |
      | detect if the ships are performing STS
      | filter: class "ships" (class_id=1) only
      v
[Confirmed STS Detections]  — (lat, lon, date, confidence)
      |
      v
[AIS Cross-Reference]
      |
      | load AIS for image date
      | search within fixed radius R (default 1000m) of STS center
      | check ALL vessel types — not just tankers
      v
[AIS Analysis]
      |
      ├── No AIS vessels found
      |     → FULLY DARK STS
      |
      ├── 1 AIS vessel found
      |     → PARTIALLY DARK STS
      |
      └── 2+ AIS vessels found
            → check if AIS data makes sense vs satellite:
              - does vessel type match what's visible? (e.g. AIS says
                fishing boat but image shows tanker-sized vessel = suspicious)
              - does speed/heading match the imagery?
              - log all AIS data for further analysis
      |
      v
[Output]  — per STS event:
    date, lat, lon, confidence, dark_status,
    vessel_1_mmsi, vessel_1_name, vessel_1_type, vessel_1_sog, vessel_1_dist_m,
    vessel_2_mmsi, vessel_2_name, vessel_2_type, vessel_2_sog, vessel_2_dist_m,
    notes (any AIS vs imagery inconsistencies)
```

---

## Key Design Decisions

### Why two zoom levels?
A zoomed-in image downloaded directly at the AIS-reported location may miss the ship
entirely. Smoke tests showed ships can be ~1.2 km offset from AIS position due to
time mismatch between the AIS ping and the actual satellite image capture time.
The zoomed-out view catches ships regardless of this offset, then the zoomed-in
view confirms the STS event at the correct location.

### Why false color for zoomed-out?
Ships (metal structures) have a very different spectral signature from ocean water in
the near-infrared bands. False color makes ships pop as bright dots against dark water,
making detection easier for both humans and the YOLO model at low resolution.

### Why fixed search radius for AIS?
Using the YOLO bounding box dimensions directly would be too tight and increase false
positive dark STS detections (missing AIS matches due to positioning errors). A fixed
radius of 1000m with margin is more forgiving and reduces false positives. This is a
configurable parameter.

### AIS is a cross-reference, not ground truth
The pipeline does NOT trust AIS as authoritative. AIS data is checked against visual
evidence from satellite imagery. Inconsistencies between AIS and imagery (wrong vessel
type, impossible speed, suspicious position) are logged as potential indicators of
AIS manipulation.

---

## Implementation Order

### Phase 1: Train the dot-detector YOLO (this branch: `ais-first-pipeline`)
1. Build the AIS STS candidate scanner (scan AIS CSVs for vessel pairs)
2. Download zoomed-out false-color GeoTIFFs for candidate locations (depends on `images` branch)
3. Run the training pipeline in `docs/training_pipeline_plan.md` to review, curate, train, and evaluate the dot-detector YOLO candidates

### Phase 2: AIS-first training data pipeline
4. Build the full Pipeline A: AIS scan → zoomed-out → dot detection → zoomed-in → STS YOLO → manual review
5. Use the training pipeline to generate additional STS YOLO datasets
6. Retrain and evaluate STS YOLO with expanded reviewed datasets

### Phase 3: Operational dark STS detection
7. Build Pipeline B: region of interest → zoomed-out → dot detection → zoomed-in → STS YOLO → AIS cross-reference
8. Run on historic hotspot areas
9. Tune parameters based on results

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--conf` | `0.25` | YOLO confidence threshold (both models) |
| `--radius` | `1000 m` | AIS search radius from STS center |
| Zoomed-out coverage | ~137 x 89 km | Dot-detector operating area |
| Zoomed-in coverage | ~1 x 0.7 km | STS confirmation area |
| Zoomed-out resolution | ~55 m/pixel | False color, ships are dots |
| Zoomed-in resolution | ~18 m/pixel | True color, ships fill frame |
| AIS pairing: proximity | TBD | Start loose, tune after manual review |
| AIS pairing: speed match | SOG must match | Start loose, tune after manual review |

---

## Known Issues & Open Questions

- **Image acquisition time:** Sentinel-2 actual capture time varies and cannot be assumed
  to be ~10:30 local. Smoke tests showed ~30min offset from expected overpass time,
  causing ~1.2km spatial mismatch. The zoomed-out stage mitigates this but the exact
  acquisition time should be verified per image when possible.
- **AIS timestamp semantics:** `BaseDateTime` timezone (UTC vs local) needs verification.
  A smoke test showed a daytime image matching an AIS ping at what would be 01:44 CST
  if interpreted as UTC.
- **Dot-detector training data:** No training data exists yet for the zoomed-out
  dot-detector. This is the first thing to build.
- **Speed matching threshold:** "Matching SOG" needs a tolerance definition (e.g. within
  ±2 knots). STS at full speed is physically implausible but we start loose and tighten
  after manual review.
- **Sentinel Hub API integration:** Image downloading is handled on the `images` branch.
  The pipeline assumes GeoTIFFs are available locally.
