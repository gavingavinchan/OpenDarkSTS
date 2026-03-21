"""
Download ALL Sentinel-2A true color images (.tiff) for the Galveston offshore
region using the Copernicus Data Space Ecosystem (CDSE) API.

- Queries the CDSE Catalog for every acquisition in the date range
- Discards scenes with cloud cover > MAX_CLOUD_PCT
- Saves each scene as a georeferenced GeoTIFF named YYYY-MM-DD_tile_X_Y.tiff
- All files land in ONE flat output folder (no subfolders)
- Auto-tiles large areas to stay within the 2500-px API limit

Register free at:   https://dataspace.copernicus.eu
Get credentials at: https://shapps.dataspace.copernicus.eu/dashboard

Requirements:
    pip install sentinelhub rasterio
"""

import argparse
import os

import numpy as np
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

MAX_PX = 2500  # CDSE hard pixel limit per side

# ---------------------------------------------------------------------------
# Galveston offshore bounding box  (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------
GALVESTON_BBOX = BBox(
    bbox=[-94.54, 28.37, -94.15, 28.68],
    crs=CRS.WGS84,
)

RESOLUTION    = 10   # metres per pixel
MAX_CLOUD_PCT = 90   # discard scenes above this %

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input:  [{ bands: ["B02", "B03", "B04"], units: "REFLECTANCE" }],
    output: { bands: 3, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [
    Math.min(255, Math.round(sample.B04 * 3.5 * 255)),
    Math.min(255, Math.round(sample.B03 * 3.5 * 255)),
    Math.min(255, Math.round(sample.B02 * 3.5 * 255))
  ];
}
"""

# SCL evalscript to measure actual cloud cover inside the bbox
EVALSCRIPT_SCL = """
//VERSION=3
function setup() {
  return {
    input:  [{ bands: ["SCL"], units: "DN" }],
    output: { bands: 1, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [sample.SCL];
}
"""

# Register CDSE collection once at module level (avoids "name already taken" error)
_CDSE_S2L2A = DataCollection.define(
    name="cdse_sentinel2_l2a",
    api_id="sentinel-2-l2a",
    service_url="https://sh.dataspace.copernicus.eu",
)


def build_config(client_id: str, client_secret: str) -> SHConfig:
    config = SHConfig()
    config.sh_client_id     = client_id
    config.sh_client_secret = client_secret
    config.sh_base_url      = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url     = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    return config


def list_scenes(
    config: SHConfig,
    bbox: BBox,
    start: str,
    end: str,
    max_cloud_pct: int,
) -> list[dict]:
    catalog = SentinelHubCatalog(config=config)
    search_iterator = catalog.search(
        collection=_CDSE_S2L2A,
        bbox=bbox,
        time=(start, end),
        filter=f"eo:cloud_cover <= {max_cloud_pct}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
    )
    return list(search_iterator)


def bbox_cloud_cover(config: SHConfig, bbox: BBox, date: str) -> float:
    """
    Compute actual cloud cover % inside the bbox using the SCL band.
    SCL cloud values: 3=shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus.
    Uses 60m resolution for speed — sufficient for cloud detection.
    """
    time_interval = (f"{date}T00:00:00Z", f"{date}T23:59:59Z")
    size = bbox_to_dimensions(bbox, resolution=60)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_SCL,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=_CDSE_S2L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    result = request.get_data()
    if not result or result[0] is None:
        return 100.0  # assume fully cloudy if no data returned

    scl = result[0].squeeze()
    cloud_pixels = np.isin(scl, [3, 8, 9, 10]).sum()
    return float(cloud_pixels / scl.size * 100)


def save_geotiff(
    data: np.ndarray,
    tile_bbox: BBox,
    out_path: str,
) -> None:
    """
    Write a georeferenced GeoTIFF (WGS84 / EPSG:4326).
    data shape: (height, width, bands) — as returned by sentinelhub
    Every pixel can be converted to lon/lat using the embedded transform.
    """
    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    height, width, bands = data.shape

    # Affine transform: maps pixel (row, col) → (lon, lat)
    transform = from_bounds(
        tile_bbox.min_x,  # left
        tile_bbox.min_y,  # bottom
        tile_bbox.max_x,  # right
        tile_bbox.max_y,  # top
        width,
        height,
    )

    with rasterio.open(
        out_path,
        "w",
        driver    = "GTiff",
        height    = height,
        width     = width,
        count     = bands,
        dtype     = data.dtype,
        crs       = RasterioCRS.from_epsg(4326),   # WGS84 lon/lat
        transform = transform,
        compress  = "deflate",                      # lossless compression
    ) as dst:
        for b in range(bands):
            dst.write(data[:, :, b], b + 1)


def download_scene(
    config: SHConfig,
    bbox: BBox,
    date: str,
    output_dir: str,
    scene_idx: int = 0,
) -> None:
    """Download all tiles for one acquisition date, save as flat named GeoTIFFs."""
    size     = bbox_to_dimensions(bbox, resolution=RESOLUTION)
    splits_x = max(1, -(-size[0] // MAX_PX))
    splits_y = max(1, -(-size[1] // MAX_PX))

    lon_step = (bbox.max_x - bbox.min_x) / splits_x
    lat_step = (bbox.max_y - bbox.min_y) / splits_y

    time_interval = (f"{date}T00:00:00Z", f"{date}T23:59:59Z")

    for ix in range(splits_x):
        for iy in range(splits_y):
            tile_bbox = BBox(
                bbox=[
                    bbox.min_x + ix * lon_step,
                    bbox.min_y + iy * lat_step,
                    bbox.min_x + (ix + 1) * lon_step,
                    bbox.min_y + (iy + 1) * lat_step,
                ],
                crs=CRS.WGS84,
            )
            tile_size = bbox_to_dimensions(tile_bbox, resolution=RESOLUTION)

            request = SentinelHubRequest(
                evalscript=EVALSCRIPT,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=_CDSE_S2L2A,
                        time_interval=time_interval,
                        other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD_PCT}},
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=tile_bbox,
                size=tile_size,
                config=config,
            )

            result = request.get_data()   # returns list of numpy arrays
            if not result or result[0] is None:
                print(f"    No data for tile ({ix},{iy}) on {date} — skipping.")
                continue

            # Include scene index so multiple scenes on the same date don't overwrite each other
            if splits_x == 1 and splits_y == 1:
                filename = f"{date}_scene{scene_idx}.tiff"
            else:
                filename = f"{date}_scene{scene_idx}_tile_{ix}_{iy}.tiff"

            out_path = os.path.join(output_dir, filename)
            save_geotiff(result[0], tile_bbox, out_path)
            print(f"    Saved: {filename}")


# ---------------------------------------------------------------------------
# FILL IN YOUR CREDENTIALS HERE
# ---------------------------------------------------------------------------
HARDCODED_CLIENT_ID     = ""   # fill in your CDSE client ID here
HARDCODED_CLIENT_SECRET = ""   # fill in your CDSE client secret here
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all Sentinel-2 true colour GeoTIFFs via CDSE"
    )
    parser.add_argument("--client_id",    default=HARDCODED_CLIENT_ID)
    parser.add_argument("--client_secret",default=HARDCODED_CLIENT_SECRET)
    parser.add_argument("--start",     default="2020-01-14", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       default="2020-01-14", help="End date   YYYY-MM-DD")
    parser.add_argument("--output",    default="data/sentinel2")
    parser.add_argument("--max_cloud", type=int, default=MAX_CLOUD_PCT,
                        help="Max cloud cover %% (default 20)")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="Custom bounding box (overrides Galveston default)")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        parser.error("Credentials missing. Fill in HARDCODED_CLIENT_ID/SECRET at the top.")

    config = build_config(args.client_id, args.client_secret)
    bbox   = BBox(bbox=args.bbox, crs=CRS.WGS84) if args.bbox else GALVESTON_BBOX

    os.makedirs(args.output, exist_ok=True)

    print(f"Region     : {bbox.min_x},{bbox.min_y} -> {bbox.max_x},{bbox.max_y}")
    print(f"Date range : {args.start} – {args.end}")
    print(f"Cloud cover: <= {args.max_cloud}%")
    print("\nQuerying CDSE catalog ...")

    scenes = list_scenes(config, bbox, args.start, args.end, args.max_cloud)

    if not scenes:
        print("No scenes found matching your criteria.")
        return

    print(f"Found {len(scenes)} qualifying scene(s):\n")
    for s in scenes:
        props = s.get("properties", {})
        dt    = props.get("datetime", "unknown")[:10]
        cc    = props.get("eo:cloud_cover", "?")
        print(f"  {dt}   cloud cover = {cc:.1f}%")

    print(f"\nChecking actual cloud cover inside your bbox ...\n")
    accepted = []
    for i, scene in enumerate(scenes, 1):
        date     = scene["properties"]["datetime"][:10]
        tile_cc  = scene["properties"].get("eo:cloud_cover", 100)
        bbox_cc  = bbox_cloud_cover(config, bbox, date)
        status   = "OK" if bbox_cc <= args.max_cloud else "SKIP"
        print(f"  [{i}/{len(scenes)}] {date}  tile={tile_cc:.1f}%  bbox={bbox_cc:.1f}%  [{status}]")
        if bbox_cc <= args.max_cloud:
            accepted.append((i, date, bbox_cc))

    if not accepted:
        print("\nNo scenes passed the bbox cloud cover check. Try raising --max_cloud or widening the date range.")
        return

    print(f"\nDownloading {len(accepted)} scene(s) into '{args.output}/' ...")
    for i, date, bbox_cc in accepted:
        print(f"  {date}  (bbox cloud={bbox_cc:.1f}%)")
        download_scene(config, bbox, date, args.output, scene_idx=i)

    print(f"\nAll done. Files saved in: {args.output}/")


if __name__ == "__main__":
    main()
