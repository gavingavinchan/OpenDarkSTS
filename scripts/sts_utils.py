from __future__ import annotations

import csv
import glob
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")
AIS_MATCH_FIELDS = ("MMSI", "VesselName", "BaseDateTime", "VesselType", "SOG", "COG", "Length")
PIPELINE_CSV_FIELDS = [
    "date",
    "image_file",
    "image_path",
    "detection_id",
    "center_lat",
    "center_lon",
    "confidence",
    "ais_count",
    "dark_status",
    "preview_png",
    "vessel_1_mmsi",
    "vessel_1_name",
    "vessel_1_time",
    "vessel_1_type",
    "vessel_1_sog",
    "vessel_1_cog",
    "vessel_1_length",
    "vessel_1_dist_m",
    "vessel_2_mmsi",
    "vessel_2_name",
    "vessel_2_time",
    "vessel_2_type",
    "vessel_2_sog",
    "vessel_2_cog",
    "vessel_2_length",
    "vessel_2_dist_m",
]


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: Any) -> None:
    destination = Path(path)
    ensure_parent_dir(destination)
    destination.write_text(json.dumps(data, indent=2))


def write_csv_rows(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    destination = Path(path)
    ensure_parent_dir(destination)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_image_date(image_path: str | Path) -> str:
    image_name = Path(image_path).name
    match = DATE_PATTERN.search(image_name)
    if not match:
        raise ValueError(f"Could not parse date from image filename: {image_name}")
    return match.group(1)


def load_image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def load_bbox_for_image(image_path: str | Path, bbox_path: str | Path | None = None) -> tuple[float, float, float, float]:
    image_path = Path(image_path)
    if image_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            rasterio = importlib.import_module("rasterio")
        except ModuleNotFoundError:
            rasterio = None

        if rasterio is not None:
            with rasterio.open(image_path) as dataset:
                bounds = dataset.bounds
            return float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)

        with Image.open(image_path) as image:
            tags = getattr(image, "tag_v2", None)
            if tags is None:
                raise RuntimeError("GeoTIFF georeferencing tags are not available from this TIFF reader")
            tiepoint = tags.get(33922)
            pixel_scale = tags.get(33550)
            if tiepoint is None or pixel_scale is None:
                raise RuntimeError(
                    "GeoTIFF georeferencing requires rasterio or GeoTIFF ModelTiepoint/ModelPixelScale tags."
                )

            width, height = image.size
            model_x = float(tiepoint[3])
            model_y = float(tiepoint[4])
            scale_x = float(pixel_scale[0])
            scale_y = float(pixel_scale[1])

        lon_min = model_x
        lat_max = model_y
        lon_max = lon_min + (width * scale_x)
        lat_min = lat_max - (height * scale_y)
        return lon_min, lat_min, lon_max, lat_max

    sidecar_path = Path(bbox_path) if bbox_path is not None else image_path.with_suffix(".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"No bbox metadata found for {image_path}. Expected sidecar JSON at {sidecar_path}."
        )

    payload = read_json(sidecar_path)
    bbox = payload.get("bbox") if isinstance(payload, dict) else None
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox metadata in {sidecar_path}; expected {{\"bbox\": [lon_min, lat_min, lon_max, lat_max]}}")
    lon_min, lat_min, lon_max, lat_max = [float(value) for value in bbox]
    return lon_min, lat_min, lon_max, lat_max


def pixel_to_lat_lon(
    cx_px: float,
    cy_px: float,
    img_width: int,
    img_height: int,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")

    lon_min, lat_min, lon_max, lat_max = bbox
    lon = lon_min + (cx_px / img_width) * (lon_max - lon_min)
    lat = lat_max - (cy_px / img_height) * (lat_max - lat_min)
    return lat, lon


def haversine_m(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    radius_m = 6_371_000.0
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)

    dlat = lat1_rad - lat0_rad
    dlon = lon1_rad - lon0_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat0_rad) * math.cos(lat1_rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


def coarse_degree_window(lat: float, radius_m: float) -> tuple[float, float]:
    lat_delta = radius_m / 111_320.0
    cos_lat = max(abs(math.cos(math.radians(lat))), 1e-6)
    lon_delta = radius_m / (111_320.0 * cos_lat)
    return lat_delta, lon_delta


def classify_dark_status(ais_count: int) -> str:
    if ais_count == 0:
        return "FULLY_DARK"
    if ais_count == 1:
        return "PARTIALLY_DARK"
    return "AIS_PRESENT"


def flatten_pipeline_result(result: dict[str, Any]) -> dict[str, Any]:
    vessels = result.get("ais_vessels", [])
    vessel_1 = vessels[0] if len(vessels) >= 1 else {}
    vessel_2 = vessels[1] if len(vessels) >= 2 else {}
    row = {
        "date": result.get("date", ""),
        "image_file": Path(result.get("image_path", "")).name,
        "image_path": result.get("image_path", ""),
        "detection_id": result.get("detection_id", ""),
        "center_lat": result.get("center_lat", ""),
        "center_lon": result.get("center_lon", ""),
        "confidence": result.get("confidence", ""),
        "ais_count": result.get("ais_count", 0),
        "dark_status": result.get("dark_status", ""),
        "preview_png": result.get("preview_png", ""),
    }
    for index, vessel in ((1, vessel_1), (2, vessel_2)):
        row[f"vessel_{index}_mmsi"] = vessel.get("MMSI", "")
        row[f"vessel_{index}_name"] = vessel.get("VesselName", "")
        row[f"vessel_{index}_time"] = vessel.get("BaseDateTime", "")
        row[f"vessel_{index}_type"] = vessel.get("VesselType", "")
        row[f"vessel_{index}_sog"] = vessel.get("SOG", "")
        row[f"vessel_{index}_cog"] = vessel.get("COG", "")
        row[f"vessel_{index}_length"] = vessel.get("Length", "")
        row[f"vessel_{index}_dist_m"] = vessel.get("distance_m", "")
    return row


def resolve_image_inputs(inputs: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    resolved.append(candidate)
            continue

        if any(token in item for token in "*?[]"):
            for match in sorted(glob.glob(item, recursive=True)):
                candidate = Path(match)
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    resolved.append(candidate)
            continue

        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(path)

    unique_paths = sorted({candidate.resolve() for candidate in resolved})
    return unique_paths


def filter_readable_images(paths: Sequence[Path], warn: Any = None) -> list[Path]:
    """Drop zero-byte or otherwise unreadable rasters instead of crashing the pipeline."""
    emit = warn if warn is not None else (lambda message: print(f"WARNING: {message}"))
    readable: list[Path] = []
    for path in paths:
        candidate = Path(path)
        try:
            if candidate.stat().st_size == 0:
                emit(f"Skipping zero-byte image: {candidate}")
                continue
        except OSError as error:
            emit(f"Skipping unreadable image {candidate}: {error}")
            continue

        try:
            with Image.open(candidate) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 - any decoder failure means "skip it"
            emit(f"Skipping unreadable image {candidate}: {error}")
            continue

        readable.append(candidate)
    return readable


def build_vessel_key(row: dict[str, str]) -> str:
    mmsi = row.get("MMSI", "").strip()
    if mmsi:
        return f"MMSI:{mmsi}"

    vessel_name = row.get("VesselName", "").strip()
    imo = row.get("IMO", "").strip()
    call_sign = row.get("CallSign", "").strip()
    return f"FALLBACK:{vessel_name}|{imo}|{call_sign}"


def resolve_ais_file(ais_dir: str | Path, date_str: str) -> Path:
    ais_path = Path(ais_dir)
    expected_name = f"AIS_{date_str.replace('-', '_')}.csv"

    if ais_path.is_file():
        if ais_path.name != expected_name:
            raise FileNotFoundError(f"AIS file {ais_path} does not match expected date file {expected_name}")
        return ais_path

    matches = sorted(ais_path.rglob(expected_name))
    if not matches:
        raise FileNotFoundError(f"Could not find {expected_name} under {ais_path}")
    return matches[0]
