from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .correlate_ais import correlate_detections
    from .detect import detect_images
    from .georeference import georeference_detections
    from .sts_utils import PIPELINE_CSV_FIELDS, flatten_pipeline_result, resolve_image_inputs, write_csv_rows, write_json
except ImportError:
    from correlate_ais import correlate_detections
    from detect import detect_images
    from georeference import georeference_detections
    from sts_utils import PIPELINE_CSV_FIELDS, flatten_pipeline_result, resolve_image_inputs, write_csv_rows, write_json


def run_pipeline(
    image_inputs: list[str],
    weights: str | Path,
    ais_dir: str | Path,
    conf: float = 0.25,
    radius_m: float = 1000.0,
    bbox_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    image_paths = resolve_image_inputs(image_inputs)
    if not image_paths:
        raise FileNotFoundError("No input images were found")

    detections = detect_images(image_paths=image_paths, weights=weights, conf=conf)
    if not detections:
        return []

    georeferenced = georeference_detections(detections=detections, bbox_path=bbox_json)
    return correlate_detections(detections=georeferenced, ais_dir=ais_dir, radius_m=radius_m)


def write_pipeline_output(output_path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        rows = [flatten_pipeline_result(result) for result in results]
        write_csv_rows(output_path, rows, PIPELINE_CSV_FIELDS)
        return

    write_json(output_path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection, georeferencing, and AIS correlation")
    parser.add_argument("--images", nargs="+", required=True, help="Image path(s), directories, or glob patterns")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights")
    parser.add_argument("--ais-dir", required=True, help="AIS directory or specific AIS CSV file")
    parser.add_argument("--radius", type=float, default=1000.0, help="AIS search radius in meters")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--bbox-json", help="Optional override path for a single sidecar bbox JSON")
    parser.add_argument("--output", required=True, help="Output path (.json or .csv)")
    args = parser.parse_args()

    results = run_pipeline(
        image_inputs=args.images,
        weights=args.weights,
        ais_dir=args.ais_dir,
        conf=args.conf,
        radius_m=args.radius,
        bbox_json=args.bbox_json,
    )
    write_pipeline_output(args.output, results)
    print(f"Wrote {len(results)} pipeline results to {args.output}")


if __name__ == "__main__":
    main()
