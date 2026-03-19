from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .sts_utils import load_bbox_for_image, load_image_size, pixel_to_lat_lon, read_json, write_json
except ImportError:
    from sts_utils import load_bbox_for_image, load_image_size, pixel_to_lat_lon, read_json, write_json


def georeference_detection(detection: dict[str, Any], bbox_path: str | Path | None = None) -> dict[str, Any]:
    image_path = Path(detection["image_path"]).resolve()
    image_width = int(detection.get("image_width_px") or 0)
    image_height = int(detection.get("image_height_px") or 0)
    if image_width <= 0 or image_height <= 0:
        image_width, image_height = load_image_size(image_path)

    bbox = load_bbox_for_image(image_path=image_path, bbox_path=bbox_path)
    center_lat, center_lon = pixel_to_lat_lon(
        cx_px=float(detection["cx_px"]),
        cy_px=float(detection["cy_px"]),
        img_width=image_width,
        img_height=image_height,
        bbox=bbox,
    )

    enriched = dict(detection)
    enriched["image_width_px"] = image_width
    enriched["image_height_px"] = image_height
    enriched["center_lat"] = round(center_lat, 8)
    enriched["center_lon"] = round(center_lon, 8)
    return enriched


def georeference_detections(
    detections: list[dict[str, Any]],
    bbox_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if bbox_path is not None:
        image_paths = {Path(detection["image_path"]).resolve() for detection in detections}
        if len(image_paths) > 1:
            raise ValueError("--bbox-json can only be used when all detections come from the same image")

    return [georeference_detection(detection, bbox_path=bbox_path) for detection in detections]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pixel detections into lat/lon coordinates")
    parser.add_argument("--detections", required=True, help="Path to detections JSON from detect.py")
    parser.add_argument("--output", required=True, help="Path to write georeferenced detections JSON")
    parser.add_argument("--bbox-json", help="Optional override path for a single sidecar bbox JSON")
    args = parser.parse_args()

    detections = read_json(args.detections)
    if not isinstance(detections, list):
        raise ValueError("Detections JSON must contain a list")

    georeferenced = georeference_detections(detections, bbox_path=args.bbox_json)
    write_json(args.output, georeferenced)
    print(f"Wrote {len(georeferenced)} georeferenced detections to {args.output}")


if __name__ == "__main__":
    main()
