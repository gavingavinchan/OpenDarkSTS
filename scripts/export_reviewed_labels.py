#!/usr/bin/python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .visual_review_lib import (
        DEFAULT_DATABASE_PATH,
        DEFAULT_EXPORT_DIR,
        DEFAULT_MANIFEST_PATH,
        ensure_cache_for_date,
        exportable_detections,
        get_db,
        initialize_database,
        repo_resolve,
        write_data_yaml,
        write_yolo_label_file,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_DATABASE_PATH,
        DEFAULT_EXPORT_DIR,
        DEFAULT_MANIFEST_PATH,
        ensure_cache_for_date,
        exportable_detections,
        get_db,
        initialize_database,
        repo_resolve,
        write_data_yaml,
        write_yolo_label_file,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reviewed visual detections in standard YOLO format")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database for visual review state")
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

    conn = get_db(args.db)
    initialize_database(conn)
    if args.include_incomplete:
        image_rows = conn.execute(
            "SELECT image_id, date FROM images ORDER BY date ASC"
        ).fetchall()
    else:
        image_rows = conn.execute(
            "SELECT image_id, date FROM images WHERE review_status = 'completed' ORDER BY date ASC"
        ).fetchall()

    exported = 0
    for row in image_rows:
        date_value = row["date"]
        image_id = int(row["image_id"])
        source_png = ensure_cache_for_date(date_value, manifest_path=args.manifest)
        target_png = images_dir / f"{date_value}.png"
        shutil.copy2(source_png, target_png)

        detections = exportable_detections(conn, image_id)
        label_path = labels_dir / f"{date_value}.txt"
        write_yolo_label_file(label_path, detections)
        exported += 1
        print(f"Exported {date_value}: {len(detections)} labels")

    write_data_yaml(output_dir)
    conn.close()
    print(f"Wrote YOLO dataset to {output_dir} ({exported} images)")


if __name__ == "__main__":
    main()
