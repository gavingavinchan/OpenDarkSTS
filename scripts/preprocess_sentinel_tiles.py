#!/usr/bin/python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .visual_review_lib import (
        DEFAULT_CACHE_DIR,
        DEFAULT_DATABASE_PATH,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SOURCE_DIR,
        delete_image_by_date,
        discover_tiles,
        get_db,
        image_exists,
        initialize_database,
        insert_preprocessed_image,
        load_yolo_models,
        render_stitched_png,
        update_manifest_entry,
        build_stitch_recipe,
        merge_detections_class_aware,
        run_model_predictions,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_CACHE_DIR,
        DEFAULT_DATABASE_PATH,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_SOURCE_DIR,
        delete_image_by_date,
        discover_tiles,
        get_db,
        image_exists,
        initialize_database,
        insert_preprocessed_image,
        load_yolo_models,
        render_stitched_png,
        update_manifest_entry,
        build_stitch_recipe,
        merge_detections_class_aware,
        run_model_predictions,
    )


DEFAULT_YOLOV8 = Path("runs/detect/workspace/runs/model_compare_150/yolov8n_e150_img960/weights/best.pt")
DEFAULT_YOLO26 = Path("runs/detect/workspace/runs/model_compare_150/yolo26n_e150_img960/weights/best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Sentinel tiles for visual review")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Read-only directory containing Sentinel GeoTIFF tiles")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database for review state")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="JSON manifest for reproducibility")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory for stitched PNG review images")
    parser.add_argument("--weights-yolov8n", default=str(DEFAULT_YOLOV8), help="YOLOv8n weights path")
    parser.add_argument("--weights-yolo26n", default=str(DEFAULT_YOLO26), help="YOLO26n weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for model inference")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for class-aware NMS")
    parser.add_argument("--date", action="append", dest="dates", help="Only preprocess specific date(s), repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Only preprocess the first N dates after filtering")
    parser.add_argument("--force", action="store_true", help="Replace existing database rows for matching dates")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_tiles = discover_tiles(args.source_dir)
    if not grouped_tiles:
        raise FileNotFoundError(f"No matching Sentinel tiles found under {args.source_dir}")

    requested_dates = set(args.dates or [])
    dates = [date_value for date_value in grouped_tiles if not requested_dates or date_value in requested_dates]
    if args.limit and args.limit > 0:
        dates = dates[: args.limit]
    if not dates:
        raise ValueError("No dates matched the requested filters")

    conn = get_db(args.db)
    initialize_database(conn)

    model_paths = {
        "yolov8n": args.weights_yolov8n,
        "yolo26n": args.weights_yolo26n,
    }
    models = load_yolo_models(model_paths)

    processed = 0
    skipped = 0
    for date_value in dates:
        already_exists = image_exists(conn, date_value)
        if already_exists and not args.force:
            print(f"Skipping {date_value}: already present in database (use --force to replace)")
            skipped += 1
            continue
        if already_exists and args.force:
            delete_image_by_date(conn, date_value)

        tiles = grouped_tiles[date_value]
        recipe = build_stitch_recipe(tiles)
        cache_path = Path(args.cache_dir) / f"{date_value}.png"
        render_info = render_stitched_png(recipe, cache_path)
        raw_detections = run_model_predictions(cache_path, models=models, conf_threshold=args.conf)
        merged_detections = merge_detections_class_aware(raw_detections, iou_threshold=args.iou)

        insert_preprocessed_image(
            conn,
            date_value=date_value,
            cache_path=cache_path,
            width=int(render_info["width"]),
            height=int(render_info["height"]),
            tiles=tiles,
            raw_detections=raw_detections,
            merged_detections=merged_detections,
        )

        update_manifest_entry(
            args.manifest,
            recipe=recipe,
            render_info=render_info,
            model_info={
                "weights": {name: str(path) for name, path in model_paths.items()},
                "confidence_threshold": args.conf,
            },
            merge_info={
                "strategy": "class_aware_nms_keep_best_box",
                "iou_threshold": args.iou,
                "raw_detection_count": len(raw_detections),
                "merged_detection_count": len(merged_detections),
                "class_conflicts": "preserve overlapping ship and ships detections for manual review",
            },
        )

        processed += 1
        print(
            f"Processed {date_value}: {len(tiles)} tiles, {len(raw_detections)} raw detections, "
            f"{len(merged_detections)} merged detections"
        )

    conn.close()
    print(f"Done. processed={processed} skipped={skipped} total_requested={len(dates)}")


if __name__ == "__main__":
    main()
