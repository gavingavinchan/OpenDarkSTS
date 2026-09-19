# Acquisition backend

`python -m scripts.application_cli acquire ais --start 2024-01-01 --end 2024-01-02`
downloads complete NOAA day files into `workspace/inputs/ais/full`. Optional
`--bbox WEST SOUTH EAST NORTH` creates a separate region cache keyed by its bounds;
the full day remains available. CSV columns are normalized to the pipeline's schema.
Archives and decompressed CSVs are streamed. Only validated, checksummed files with
completion records are reused. A failed transfer cannot become a completed cache entry.

The old NOAA URLs in the contributed downloader returned 404 during validation.
The current [NOAA distribution record](https://www.fisheries.noaa.gov/inport/item/77594)
points to `noaaocm.blob.core.windows.net/ais/csv2/csvYYYY/ais-YYYY-MM-DD.csv.zst`.
The [2024 archive](https://noaaocm.blob.core.windows.net/ais/csv2/csv2024/index.html)
also uses Zstandard and lower-case column names. The implementation supports both
Zstandard and ZIP payloads, but requests the current archive for all supported years.
This is a documented correction to the Phase 2 plan's year-based ZIP assumption.

Set `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in the environment, then use:

```bash
python -m scripts.application_cli acquire imagery --start 2024-01-01 --end 2024-01-02 \
  --bbox -94.73 29.07 -94.57 29.205 --max-cloud 30
```

Imagery uses the [CDSE Catalog](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog/Examples.html)
and [Processing API metadata](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/Metadata.html).
The RGB rendering and tiled 10 m requests derive from the imagery branch. The previous
whole-day request per scene is replaced by one request per acquisition timestamp and
tile. Scene IDs with the same timestamp are grouped; the response records the actual
contributing tiles and explicitly describes its pixel composition. It is not presented
as one catalog scene. The requested time interval is two seconds, centered on the
catalog timestamp, and is unrelated to the same-day AIS matching policy.

Each `.tiff.json` sidecar records the catalog IDs, timestamp, bounds, size, rendering,
cloud threshold, contributing-source metadata, and file checksum. Credentials are
neither written into these records nor included in logs. Tests mock network responses;
the authenticated CDSE workflow still requires a user's own account for live validation.
