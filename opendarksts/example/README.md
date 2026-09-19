# Galveston offshore example, 24 January 2020

Contains modified Copernicus Sentinel data 2020. AIS: NOAA Office for Coastal
Management, derived from US Coast Guard records. The 1024 × 1024 north-up EPSG:4326
GeoTIFF is a pixel crop of `2020-01-24_scene1_tile_3_3.tiff`. No scaling or additional
colour rendering was applied. The source product ID and exact acquisition time
were not retained in the historical download; the date comes from its filename.

The 1,653 AIS rows contain **every position in the documented bounding box that
day**, copied from the original NOAA CSV. The box includes a margin around the
image. Its coverage and SHA-256 are recorded in `AIS_2020_01_24.csv.json`; larger
search radii or other areas must use a full-day file. The original AIS file remains
unchanged. See `provenance.json` for source and output hashes.

Run with the default model at confidence 0.25 and radius 1000 m. At least one STS
candidate near 28.6564 N, 94.3228 W should be detected. Same-day AIS includes two
nearby vessels. Detections may overlap; their count is not a count of distinct
transfers. This example is a software smoke test, not an independent evaluation
or a human-confirmed event label. It may overlap historical training observations.

Recreation (from the repository):

```
python -m scripts.build_release_example --image SOURCE/2020-01-24_scene1_tile_3_3.tiff --ais SOURCE/AIS_2020_01_24.csv --output workspace/exports/release_example
```

The current NOAA download is at
https://noaaocm.blob.core.windows.net/ais/csv2/csv2020/ais-2020-01-24.csv.zst;
normalize it using `acquire ais`. Serialization and source revisions may change
byte hashes; `provenance.json` records the historical inputs actually used.

Data terms:
- Copernicus: https://dataspace.copernicus.eu/terms-and-conditions
- Sentinel legal notice: https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice
- NOAA metadata: https://www.fisheries.noaa.gov/inport/item/77594
