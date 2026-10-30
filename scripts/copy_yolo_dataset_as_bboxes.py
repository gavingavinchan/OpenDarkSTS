#!/usr/bin/python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionStats:
    label_files: int = 0
    changed_files: int = 0
    bbox_lines: int = 0
    polygon_lines: int = 0
    other_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a YOLO dataset and convert polygon labels to rectangular YOLO boxes."
    )
    parser.add_argument("source_dir", help="Source YOLO dataset directory")
    parser.add_argument("output_dir", help="Destination for the copied dataset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists",
    )
    return parser.parse_args()


def iter_label_files(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in dataset_dir.rglob("*.txt") if path.parent.name == "labels")


def convert_line(line: str) -> tuple[str, bool, str]:
    parts = line.strip().split()
    if not parts:
        return "", False, "empty"

    class_id = parts[0]
    coords = [float(value) for value in parts[1:]]

    if len(coords) == 4:
        return (
            " ".join([class_id, *(f"{value:.8f}" for value in coords)]),
            False,
            "bbox",
        )

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        return (
            f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}",
            True,
            "polygon",
        )

    return line.strip(), False, "other"


def convert_dataset(output_dir: Path) -> ConversionStats:
    stats = ConversionStats()
    for label_path in iter_label_files(output_dir):
        stats.label_files += 1
        original_lines = label_path.read_text().splitlines()
        new_lines: list[str] = []
        changed = False

        for line in original_lines:
            converted, line_changed, kind = convert_line(line)
            if converted:
                new_lines.append(converted)
            if line_changed:
                changed = True
            if kind == "bbox":
                stats.bbox_lines += 1
            elif kind == "polygon":
                stats.polygon_lines += 1
            elif kind == "other":
                stats.other_lines += 1

        label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
        if changed:
            stats.changed_files += 1

    return stats


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source dataset is not a directory: {source_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, ignore=shutil.ignore_patterns("labels.cache"))

    stats = convert_dataset(output_dir)
    print(f"Copied dataset: {source_dir} -> {output_dir}")
    print(f"Label files scanned: {stats.label_files}")
    print(f"Label files changed: {stats.changed_files}")
    print(f"Polygon lines converted: {stats.polygon_lines}")
    print(f"BBox lines kept: {stats.bbox_lines}")
    print(f"Other lines kept: {stats.other_lines}")


if __name__ == "__main__":
    main()
