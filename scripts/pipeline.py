"""One-command operational pipeline: detect STS pairs, georeference, cross-check AIS.

Given a folder of Sentinel-2 GeoTIFF tiles (and optionally a single date), this runs
the promoted STS YOLO weight, converts pixel boxes to lat/lon, correlates each
detection against the matching US AIS day file, and writes a CSV/JSON of dark STS
candidates plus one annotated PNG per detection so a human can check every hit.
"""

from __future__ import annotations

import argparse
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageDraw

try:
    from .workspace import current_model, atomic_json, sha256, WORKSPACE_DIR, DEFAULT_WEIGHT_NAME
    from .correlate_ais import correlate_detections
    from .detect import detect_images, MAX_BOX_AREA_FRACTION
    from .georeference import georeference_detections
    from .sts_utils import (
        PIPELINE_CSV_FIELDS,
        filter_readable_images,
        flatten_pipeline_result,
        parse_image_date,
        load_bbox_for_image,
        resolve_image_inputs,
        write_csv_rows,
        write_json,
    )
except ImportError:
    from workspace import current_model, atomic_json, sha256, WORKSPACE_DIR, DEFAULT_WEIGHT_NAME
    from correlate_ais import correlate_detections
    from detect import detect_images, MAX_BOX_AREA_FRACTION
    from georeference import georeference_detections
    from sts_utils import (
        PIPELINE_CSV_FIELDS,
        filter_readable_images,
        flatten_pipeline_result,
        parse_image_date,
        load_bbox_for_image,
        resolve_image_inputs,
        write_csv_rows,
        write_json,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = WORKSPACE_DIR / "model_weights" / DEFAULT_WEIGHT_NAME
PREVIEW_MARGIN_PX = 120
PREVIEW_MIN_SIDE_PX = 480
PREVIEW_CAPTION_LINE_PX = 14


def _default_warn(message: str) -> None:
    print(f"WARNING: {message}")


def select_images(
    image_inputs: Sequence[str],
    date: str | None = None,
    warn: Callable[[str], None] | None = None,
) -> list[Path]:
    """Resolve inputs, drop unreadable/zero-byte rasters, and optionally keep one date."""
    emit = warn or _default_warn
    image_paths = resolve_image_inputs(list(image_inputs))
    image_paths = filter_readable_images(image_paths, warn=emit)

    kept: list[Path] = []
    for path in image_paths:
        try:
            image_date = parse_image_date(path)
        except ValueError as error:
            emit(f"Skipping image with no parsable date: {error}")
            continue
        if date is not None and image_date != date:
            continue
        kept.append(path)
    return kept


def _preview_name(detection: dict[str, Any]) -> str:
    detection_id = str(detection.get("detection_id", "detection"))
    safe = "".join(character if character.isalnum() or character in "-_." else "_" for character in detection_id)
    return f"{safe}.png"


def _caption_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        f"{result.get('detection_id', '')}  conf={result.get('confidence', '')}",
        f"{result.get('date', '')}  lat={result.get('center_lat', '')}  lon={result.get('center_lon', '')}",
        str(result.get("ais_status_text") or f"AIS vessels within search radius that day: {result.get('ais_count', 0)}"),
    ]
    vessels = result.get("ais_vessels") or []
    if result.get("ais_error"):
        lines.append(str(result["ais_error"]))
    for vessel in vessels[:6]:
        lines.append(
            "MMSI {mmsi} {name} t={time} type={vtype} SOG={sog} COG={cog} len={length} d={dist}m".format(
                mmsi=vessel.get("MMSI", ""),
                name=vessel.get("VesselName", "") or "-",
                time=vessel.get("BaseDateTime", ""),
                vtype=vessel.get("VesselType", "") or "-",
                sog=vessel.get("SOG", "") or "-",
                cog=vessel.get("COG", "") or "-",
                length=vessel.get("Length", "") or "-",
                dist=vessel.get("distance_m", ""),
            )
        )
    if len(vessels) > 6:
        lines.append(f"... and {len(vessels) - 6} more AIS vessels (see JSON)")
    return lines


def write_detection_preview(result: dict[str, Any], preview_dir: str | Path) -> Path:
    """Write one annotated PNG: the crop around the detection with the box and AIS matches."""
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    destination = preview_dir / _preview_name(result)

    with Image.open(result["image_path"]) as source:
        image = source.convert("RGB")

    cx = float(result["cx_px"])
    cy = float(result["cy_px"])
    half_w = float(result["w_px"]) / 2.0
    half_h = float(result["h_px"]) / 2.0
    box = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    left = max(0, int(box[0]) - PREVIEW_MARGIN_PX)
    top = max(0, int(box[1]) - PREVIEW_MARGIN_PX)
    right = min(image.width, int(box[2]) + PREVIEW_MARGIN_PX)
    bottom = min(image.height, int(box[3]) + PREVIEW_MARGIN_PX)
    if right <= left:
        left, right = 0, image.width
    if bottom <= top:
        top, bottom = 0, image.height

    crop = image.crop((left, top, right, bottom))
    scale = max(1.0, PREVIEW_MIN_SIDE_PX / max(1, min(crop.width, crop.height)))
    if scale > 1.0:
        crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.NEAREST)

    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [
            (box[0] - left) * scale,
            (box[1] - top) * scale,
            (box[2] - left) * scale,
            (box[3] - top) * scale,
        ],
        outline=(255, 0, 0),
        width=max(2, int(scale)),
    )

    lines = _caption_lines(result)
    caption_height = PREVIEW_CAPTION_LINE_PX * len(lines) + 8
    caption_width = 6 * max((len(line) for line in lines), default=0) + 12
    canvas = Image.new("RGB", (max(crop.width, caption_width), crop.height + caption_height), (12, 12, 12))
    canvas.paste(crop, (0, 0))
    caption = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        caption.text((6, crop.height + 4 + index * PREVIEW_CAPTION_LINE_PX), line, fill=(235, 235, 235))

    canvas.save(destination)
    return destination


def write_detection_previews(results: list[dict[str, Any]], preview_dir: str | Path) -> list[dict[str, Any]]:
    """Render a preview per result and record its path on the result dict."""
    for result in results:
        result["preview_png"] = str(write_detection_preview(result, preview_dir))
    return results


def run_pipeline(
    image_inputs: list[str],
    weights: str | Path | None = None,
    ais_dir: str | Path = "",
    conf: float = 0.25,
    radius_m: float = 1000.0,
    bbox_json: str | Path | None = None,
    date: str | None = None,
    preview_dir: str | Path | None = None,
    warn: Callable[[str], None] | None = None,
    record_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    image_paths = select_images(image_inputs, date=date, warn=warn)
    if not image_paths:
        raise FileNotFoundError("No readable input images were found")

    if bbox_json is not None and len(image_paths) > 1:
        raise ValueError("A bbox override can only be used with one image")
    for image_path in image_paths:
        load_bbox_for_image(image_path, bbox_json)
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("Confidence must be between zero and one")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("AIS radius must be positive and finite")
    weight_path = Path(weights or current_model()).resolve()
    run_id = uuid.uuid4().hex
    record_path = Path(record_path or WORKSPACE_DIR / "reports" / "operational" / run_id / "run.json")
    record = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "weight": str(weight_path), "weight_sha256": sha256(weight_path), "confidence": conf,
              "ais_radius_m": radius_m, "ais_source": str(ais_dir), "date": date,
              "max_box_area_fraction": MAX_BOX_AREA_FRACTION,
              "inputs": [str(path) for path in image_paths],
              "input_sha256": {str(path): sha256(path) for path in image_paths}, "bbox_override": str(bbox_json) if bbox_json else None,
              "matching_policy": "same UTC day, radius only", "status": "running"}
    atomic_json(record_path, record)
    try:
        detections = detect_images(image_paths=image_paths, weights=weight_path, conf=conf)
        georeferenced = georeference_detections(detections=detections, bbox_path=bbox_json)
        results = correlate_detections(detections=georeferenced, ais_dir=ais_dir, radius_m=radius_m)
        for item in results:
            item.update(run_id=run_id, run_record=str(record_path))
        if preview_dir is not None:
            write_detection_previews(results, preview_dir)
        record.update(status="completed", detection_count=len(results),
                      ais_files=sorted({item["ais_file"] for item in results if item.get("ais_file")}))
        return results
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(record_path, record)



def write_pipeline_output(output_path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        rows = [flatten_pipeline_result(result) for result in results]
        write_csv_rows(output_path, rows, PIPELINE_CSV_FIELDS)
        return

    write_json(output_path, results)


def summarise_results(results: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"detections": len(results), "FULLY_DARK": 0, "PARTIALLY_DARK": 0, "AIS_PRESENT": 0}
    for result in results:
        status = result.get("dark_status", "")
        if status:
            summary[status] = summary.get(status, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection, georeferencing, and AIS correlation")
    parser.add_argument("--images", nargs="+", required=True, help="Image path(s), directories, or glob patterns")
    parser.add_argument(
        "--weights",
        default=None,
        help=f"Path to YOLO .pt weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument("--ais-dir", required=True, help="AIS directory or specific AIS CSV file")
    parser.add_argument("--date", help="Only process tiles for this YYYY-MM-DD (default: infer per file)")
    parser.add_argument("--radius", type=float, default=1000.0, help="AIS search radius in meters")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--bbox-json", help="Optional override path for a single sidecar bbox JSON")
    parser.add_argument("--output", required=True, help="Output path (.json or .csv)")
    parser.add_argument(
        "--also-write",
        help="Optional second output path in the other format (.json or .csv)",
    )
    parser.add_argument(
        "--preview-dir",
        help="Directory for annotated PNGs (default: <output parent>/previews; use 'none' to disable)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.preview_dir is None:
        preview_dir: Path | None = output_path.parent / "previews"
    elif args.preview_dir.lower() == "none":
        preview_dir = None
    else:
        preview_dir = Path(args.preview_dir)

    results = run_pipeline(
        image_inputs=args.images,
        weights=args.weights,
        ais_dir=args.ais_dir,
        conf=args.conf,
        radius_m=args.radius,
        bbox_json=args.bbox_json,
        date=args.date,
        preview_dir=preview_dir,
        record_path=output_path.parent / "run.json",
    )
    write_pipeline_output(output_path, results)
    if args.also_write:
        write_pipeline_output(args.also_write, results)

    summary = summarise_results(results)
    print(f"Wrote {len(results)} pipeline results to {output_path}")
    if args.also_write:
        print(f"Wrote {len(results)} pipeline results to {args.also_write}")
    if preview_dir is not None and results:
        print(f"Wrote {len(results)} annotated previews to {preview_dir}")
    print(f"AIS vessels within {args.radius:g} metres that day: no={summary['FULLY_DARK']} one={summary['PARTIALLY_DARK']} two-or-more={summary['AIS_PRESENT']}; missing/invalid={summary.get('AIS_MISSING', 0) + summary.get('AIS_INVALID', 0)}")


if __name__ == "__main__":
    main()
