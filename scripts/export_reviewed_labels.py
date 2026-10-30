#!/usr/bin/python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .visual_review_lib import (
        DEFAULT_EXPORT_DIR,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_REVIEWS_DIR,
        collect_review_documents,
        ensure_cache_for_image_key,
        exportable_detections,
        repo_resolve,
        write_data_yaml,
        write_yolo_label_file,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_EXPORT_DIR,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_REVIEWS_DIR,
        collect_review_documents,
        ensure_cache_for_image_key,
        exportable_detections,
        repo_resolve,
        write_data_yaml,
        write_yolo_label_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed visual detections in standard YOLO format")
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR), help="Root directory containing per-image review JSON files")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Manifest used to regenerate missing PNG cache files")
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPORT_DIR), help="Destination dataset directory")
    parser.add_argument("--include-incomplete", action="store_true", help="Also export images that are not marked completed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = repo_resolve(args.output_dir)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    review_docs = collect_review_documents(args.reviews_dir)
    if not args.include_incomplete:
        review_docs = [review for review in review_docs if review.get("review_status") == "completed"]

    exported = 0
    for review in review_docs:
        image_key = review["image_key"]
        image_filename = review["image_filename"]
        tile_stem = Path(image_filename).stem
        source_png = ensure_cache_for_image_key(image_key, args.manifest)
        target_png = images_dir / image_filename
        shutil.copy2(source_png, target_png)

        detections = exportable_detections(review)
        label_path = labels_dir / f"{tile_stem}.txt"
        write_yolo_label_file(label_path, detections)
        exported += 1
        print(f"Exported {tile_stem}: {len(detections)} labels")

    write_data_yaml(output_dir)
    print(f"Wrote YOLO dataset to {output_dir} ({exported} images)")


if __name__ == "__main__":
    main()
