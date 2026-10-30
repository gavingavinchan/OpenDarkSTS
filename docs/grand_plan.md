# OpenDarkSTS — Grand Plan

This is the original operational pipeline plan. For the current two-pipeline hierarchy, use `docs/grand_plan_v2.md` as the project-level plan. The training-side pipeline is defined in `docs/training_pipeline_plan.md`.

Detect dark ship-to-ship (STS) transfers by running YOLO on satellite imagery to find
active STS events, then cross-referencing those detections against AIS records to
determine whether either vessel was broadcasting. If one or both vessels are absent from
AIS at that location and date, the event is flagged as a dark (or partially dark) STS.

---

## Data Sources

### Satellite Imagery
- **Format:** JPEG (Sentinel Hub EO Browser export) or GeoTIFF
- **Sensor:** Sentinel-2 L2A True Color
- **Filename convention:** `YYYY-MM-DD-00_00_YYYY-MM-DD-23_59_Sentinel-2_L2A_True_color.jpg`
  - The date is embedded in the filename and used to match the correct AIS file.
- **Coverage:** One full-day composite per file. Any geographic area.
- **Georeferencing:**
  - GeoTIFF: bounding box read directly via `rasterio`
  - JPEG: requires a sidecar `<image_stem>.json` file containing:
    ```json
    { "bbox": [lon_min, lat_min, lon_max, lat_max] }
    ```

### AIS Data
- **Format:** CSV, one file per date: `AIS_YYYY_MM_DD.csv`
- **Columns:** `MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO,
  CallSign, VesselType, Status, Length, Width, Draft, Cargo, TransceiverClass`
- **Scale:** ~7.4M records per day (US-wide); loaded once per image, filtered spatially
  on the fly.

### YOLO Model
- **Architecture:** YOLO26n (preferred) or YOLOv8n, trained on the `data-science-dark-fleet`
  Roboflow dataset.
- **Classes:**
  - `ship` (class 0) — single vessel. Not an STS event. Ignored by the AIS correlation step.
  - `ships` (class 1) — two vessels side-by-side, actively mid-transfer. This IS the STS
    detection. Bounding box drawn tightly around both ships together.
- **Best weights (trained so far):**
  - `runs/detect/workspace/runs/model_compare/yolo26n_e50_img960/weights/best.pt`
  - `runs/detect/workspace/runs/model_compare/yolov8n_lastpt_e150/weights/best.pt`

---

## Pipeline Overview

```
[Satellite Image]
      |
      | parse date from filename
      v
[YOLO Inference]  ←── model weights (.pt)
      |
      | filter: keep class "ships" (class_id=1) only
      v
[STS Detections]  — list of (cx, cy, w, h, confidence) in pixels
      |
      | pixel → lat/lon using image bbox
      | (GeoTIFF via rasterio  OR  sidecar .json)
      v
[Georeferenced STS Events]  — (center_lat, center_lon, confidence, date)
      |
      | load AIS_YYYY_MM_DD.csv for image date
      | spatial search: all AIS records within radius R of center
      v
[AIS Correlation]
      |
      ├── 0 vessels found  →  FULLY DARK STS
      ├── 1 vessel  found  →  PARTIALLY DARK STS  (one ship broadcasting)
      └── 2+ vessels found →  AIS PRESENT  (both ships visible to AIS)
      |
      v
[Output]  — results CSV / JSON
```

---

## Module Breakdown

### 1. `scripts/detect.py`
Run YOLO inference on one or more satellite images. Output only `ships` detections.

**Inputs:**
- `--image` — path to satellite image (JPEG or GeoTIFF)
- `--weights` — path to `.pt` model weights
- `--conf` — confidence threshold (default: `0.25`)
- `--output` — path to write detections JSON

**Output JSON schema (one entry per `ships` detection):**
```json
[
  {
    "image_path": "...",
    "date": "2024-03-23",
    "cx_px": 1388.5,
    "cy_px": 1305.5,
    "w_px": 15,
    "h_px": 23,
    "confidence": 0.063
  }
]
```

**Notes:**
- Uses `from ultralytics import YOLO`
- Skips `ship` (class 0) detections entirely
- Image dimensions extracted from the loaded image for use in georeferencing

---

### 2. `scripts/georeference.py`
Convert pixel bounding box center to geographic coordinates.

**Inputs:**
- Image path (to determine pixel dimensions)
- Detection entry with `(cx_px, cy_px)`
- Bounding box source: GeoTIFF metadata (via `rasterio`) or sidecar JSON

**Georeferencing transform (for JPEG with sidecar):**
```
lon = lon_min + (cx_px / img_width)  * (lon_max - lon_min)
lat = lat_max - (cy_px / img_height) * (lat_max - lat_min)
```
(y=0 is top of image → lat_max; y increases downward)

**Output:** adds `center_lat`, `center_lon` fields to each detection entry.

**Sidecar JSON format:**
```json
{ "bbox": [lon_min, lat_min, lon_max, lat_max] }
```

---

### 3. `scripts/correlate_ais.py`
Load AIS data for the image date and spatially match vessels to each STS detection.

**Inputs:**
- `--detections` — georeferenced detections JSON (output of step 2)
- `--ais-dir` — directory containing `AIS_YYYY_MM_DD.csv` files
- `--radius` — search radius in metres (default: `1000`)

**AIS loading:**
- Parse date from detection `date` field → find `AIS_YYYY_MM_DD.csv`
- Load full CSV into a pandas DataFrame (once per unique date)
- Filter to a coarse lat/lon bounding box first for speed, then apply precise
  haversine distance filter

**Haversine distance filter:**
For each STS detection center `(lat0, lon0)`, keep AIS rows where:
```
haversine(lat0, lon0, row.LAT, row.LON) <= radius_m
```

**AIS match result per detection:**
```json
{
  "ais_vessels": [
    { "MMSI": 123456789, "VesselName": "TANKER A", "distance_m": 342.1 },
    { "MMSI": 987654321, "VesselName": "TANKER B", "distance_m": 498.7 }
  ],
  "ais_count": 2,
  "dark_status": "AIS_PRESENT"
}
```

**Dark status rules:**
| `ais_count` | `dark_status`      |
|-------------|--------------------|
| 0           | `FULLY_DARK`       |
| 1           | `PARTIALLY_DARK`   |
| 2+          | `AIS_PRESENT`      |

---

### 4. `scripts/pipeline.py`
Orchestrates the full pipeline end-to-end for one or more images.

**Inputs:**
- `--images` — glob pattern or directory of satellite images
- `--weights` — YOLO model weights path
- `--ais-dir` — directory of AIS CSV files
- `--radius` — AIS search radius in metres (default: `1000`)
- `--conf` — YOLO confidence threshold (default: `0.25`)
- `--output` — output file path (CSV or JSON)

**Output CSV columns:**
```
date, image_file, detection_id, center_lat, center_lon, confidence,
ais_count, dark_status, vessel_1_mmsi, vessel_1_name, vessel_1_dist_m,
vessel_2_mmsi, vessel_2_name, vessel_2_dist_m
```

**Execution flow:**
```python
for image in images:
    detections = detect(image, weights, conf)          # step 1
    detections = georeference(detections, image)       # step 2
    results    = correlate_ais(detections, ais_dir, R) # step 3
    append results to output
```

---

## Directory Structure

```
OpenDarkSTS/
├── CLAUDE.md
├── docs/
│   └── grand_plan.md          ← this file
├── scripts/
│   ├── detect.py              ← step 1: YOLO inference
│   ├── georeference.py        ← step 2: pixel → lat/lon
│   ├── correlate_ais.py       ← step 3: AIS spatial search
│   ├── pipeline.py            ← orchestrator
│   └── compare_yolo_models.py ← existing model benchmarking
├── workspace/
│   ├── datasets/
│   ├── reports/
│   └── runs/
├── runs/                      ← YOLO training outputs
├── skills/
│   └── yoloV26_skill.md
├── yolo26n.pt                 ← base weights
└── yolov8n.pt
```

---

## Key Parameters

| Parameter         | Default  | Notes                                              |
|-------------------|----------|----------------------------------------------------|
| `--conf`          | `0.25`   | YOLO confidence threshold for `ships` detections   |
| `--radius`        | `1000 m` | AIS search radius from STS bbox center             |
| YOLO class filter | class 1  | Only `ships` detections passed to AIS correlation  |

---

## Open Questions / Future Work

- **Georeferencing source:** JPEG images need a sidecar bbox JSON. A future improvement
  would be to ingest GeoTIFFs natively via `rasterio` so no manual metadata is needed.
- **AIS temporal filtering:** Currently uses all AIS records for the full day. A future
  improvement would be to use the Sentinel-2 overpass time (typically ~10:30 local) to
  narrow AIS matches to a ±1h window, reducing false positives in busy shipping lanes.
- **Timestamp semantics:** A real smoke test produced a daytime-looking image whose nearest
  AIS ping would convert to `01:44 CST` if `BaseDateTime` is UTC. This means the pipeline
  must not infer the image's local capture time from AIS alone; the timezone basis of AIS
  timestamps and the actual satellite acquisition time both need to be verified.
- **Spatial validation:** That same smoke test also suggests the current spatial match may be
  wrong, because the crop bbox was inferred rather than read from exact image metadata. The
  pipeline needs a validated bbox or native georeferencing for each image before AIS matches
  can be treated as reliable spatial evidence.
- **Overpass-time mismatch:** A GeoTIFF test aligned well with AIS around `11:01-11:03 AM CST`,
  but the same vessel pair sat about `1.2-1.3 km` away at `~10:30 AM CST`, which is close to
  the commonly cited Sentinel-2 local overpass time. The pipeline therefore needs explicit
  verification of actual acquisition time per image instead of assuming `10:30 local`.
- **Confidence threshold tuning:** The 0.25 default may produce low-quality `ships`
  detections. Should be evaluated against ground truth.
- **Multiple AIS matches:** When `ais_count >= 2`, the two closest vessels are reported.
  There is no guarantee these are the actual STS pair vs. unrelated nearby traffic.
- **Model retraining:** Current best mAP50 is ~0.83 (YOLO26, 150 epochs). Dataset has
  mixed box/segment annotations; cleaning may improve performance.
