from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OLD_DATASET = ROOT_DIR / "workspace/datasets/derived/madhulika_yolov8_rectbbox"
DEFAULT_REVIEWS_DIR = ROOT_DIR / "workspace/reviews/gavin"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "workspace/datasets/derived/madhulika_rectbbox_plus_manual_sts_balanced_v1"
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}
CLASS_NAMES = {0: "ship", 1: "ships"}
OLD_SPLITS = {"train": "train", "valid": "val", "test": "test"}
TARGET_MANUAL_SPLIT_SHIPS = {"val": 4, "test": 4}
VARIANTS = (
    "crop",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "bright085",
    "bright115",
    "contrast090",
    "contrast110",
)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to rebuild it.")
    allowed_root = (ROOT_DIR / "workspace/datasets/derived").resolve()
    resolved = output_dir.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to remove output outside {allowed_root}: {resolved}")
    shutil.rmtree(resolved)


def ensure_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def label_counts_from_text(text: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        counts[int(line.split()[0])] += 1
    return counts


def add_counts(counts: dict[str, Any], split: str, source: str, label_counts: Counter[int]) -> None:
    for class_id, amount in label_counts.items():
        class_name = CLASS_NAMES[class_id]
        counts["by_split"][split][class_name] += amount
        counts["by_source"][source][class_name] += amount
        counts["by_split_and_source"][split][source][class_name] += amount
        counts["overall"][class_name] += amount


def record_file(output_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(output_dir)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def copy_old_dataset(old_dataset: Path, output_dir: Path, counts: dict[str, Any], files: list[dict[str, Any]]) -> None:
    for old_split, out_split in OLD_SPLITS.items():
        image_dir = old_dataset / old_split / "images"
        label_dir = old_dataset / old_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing old dataset split: {image_dir} / {label_dir}")
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            output_name = f"madhulika__{image_path.name}"
            target_image = output_dir / "images" / out_split / output_name
            target_label = output_dir / "labels" / out_split / f"{Path(output_name).stem}.txt"
            source_label = label_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, target_image)
            if source_label.exists():
                label_text = source_label.read_text()
            else:
                label_text = ""
            target_label.write_text(label_text)

            add_counts(counts, out_split, "madhulika_rectbbox", label_counts_from_text(label_text))
            files.append(record_file(output_dir, target_image))
            files.append(record_file(output_dir, target_label))


def positive_detections(review: dict[str, Any]) -> list[dict[str, Any]]:
    return [d for d in review.get("detections", []) if d.get("review_label") in REVIEW_LABEL_TO_CLASS]


def write_yolo_labels(path: Path, detections: list[dict[str, Any]]) -> Counter[int]:
    lines: list[str] = []
    counts: Counter[int] = Counter()
    for detection in detections:
        class_id = REVIEW_LABEL_TO_CLASS[str(detection["review_label"])]
        counts[class_id] += 1
        lines.append(
            " ".join(
                [
                    str(class_id),
                    f"{float(detection['x_center']):.8f}",
                    f"{float(detection['y_center']):.8f}",
                    f"{float(detection['width']):.8f}",
                    f"{float(detection['height']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return counts


def load_completed_reviews_with_sts(reviews_dir: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    for path in sorted(reviews_dir.glob("**/*.json")):
        review = read_json(path)
        if review.get("review_status") != "completed":
            continue
        counts = Counter(d.get("review_label") for d in review.get("detections", []))
        if counts["ships_sts"]:
            records.append({"path": path, "review": review, "ship": counts["ship_non_sts"], "ships": counts["ships_sts"]})
        elif counts["ship_non_sts"]:
            excluded_counts["manual_ship_only_images"] += 1
            excluded_counts["manual_ship_only_ship_boxes"] += counts["ship_non_sts"]
    return records, excluded_counts


def split_manual_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    split_records = {"train": [], "val": [], "test": []}
    split_ships = Counter({"val": 0, "test": 0})
    for record in shuffled:
        if split_ships["val"] < TARGET_MANUAL_SPLIT_SHIPS["val"]:
            split = "val"
        elif split_ships["test"] < TARGET_MANUAL_SPLIT_SHIPS["test"]:
            split = "test"
        else:
            split = "train"
        split_records[split].append(record)
        split_ships[split] += int(record["ships"])
    return split_records


def copy_manual_raw(
    output_dir: Path,
    split_records: dict[str, list[dict[str, Any]]],
    counts: dict[str, Any],
    files: list[dict[str, Any]],
) -> None:
    for split, records in split_records.items():
        for record in records:
            review = record["review"]
            source_image = repo_path(str(review["cache_path"]))
            suffix = source_image.suffix or ".png"
            image_key = str(review["image_key"])
            output_name = f"gavin_raw__{image_key}{suffix}"
            target_image = output_dir / "images" / split / output_name
            target_label = output_dir / "labels" / split / f"{Path(output_name).stem}.txt"

            shutil.copy2(source_image, target_image)
            label_counts = write_yolo_labels(target_label, positive_detections(review))
            add_counts(counts, split, "manual_raw_sts_images", label_counts)
            files.append(record_file(output_dir, target_image))
            files.append(record_file(output_dir, target_label))


def detection_box_pixels(detection: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    xc = float(detection["x_center"]) * width
    yc = float(detection["y_center"]) * height
    box_w = float(detection["width"]) * width
    box_h = float(detection["height"]) * height
    return (xc - box_w / 2, yc - box_h / 2, xc + box_w / 2, yc + box_h / 2)


def crop_window_for_detection(
    detection: dict[str, Any], image_width: int, image_height: int, crop_min_size: int, crop_max_size: int, crop_scale: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = detection_box_pixels(detection, image_width, image_height)
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    box_size = max(x2 - x1, y2 - y1)
    size = int(round(max(crop_min_size, box_size * crop_scale)))
    size = min(size, crop_max_size, image_width, image_height)
    left = int(round(xc - size / 2))
    top = int(round(yc - size / 2))
    left = max(0, min(left, image_width - size))
    top = max(0, min(top, image_height - size))
    return left, top, left + size, top + size


def labels_in_crop(
    review: dict[str, Any], crop_box: tuple[int, int, int, int], image_width: int, image_height: int
) -> list[dict[str, Any]]:
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    labels: list[dict[str, Any]] = []
    for detection in positive_detections(review):
        x1, y1, x2, y2 = detection_box_pixels(detection, image_width, image_height)
        ix1 = max(x1, crop_left)
        iy1 = max(y1, crop_top)
        ix2 = min(x2, crop_right)
        iy2 = min(y2, crop_bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        original_area = max(1e-9, (x2 - x1) * (y2 - y1))
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        if intersection_area / original_area < 0.5:
            continue
        labels.append(
            {
                "review_label": detection["review_label"],
                "x_center": ((ix1 + ix2) / 2 - crop_left) / crop_width,
                "y_center": ((iy1 + iy2) / 2 - crop_top) / crop_height,
                "width": (ix2 - ix1) / crop_width,
                "height": (iy2 - iy1) / crop_height,
            }
        )
    return labels


def transform_labels(labels: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for label in labels:
        x = float(label["x_center"])
        y = float(label["y_center"])
        w = float(label["width"])
        h = float(label["height"])
        if variant == "hflip":
            x = 1.0 - x
        elif variant == "vflip":
            y = 1.0 - y
        elif variant == "rot90":
            x, y, w, h = y, 1.0 - x, h, w
        elif variant == "rot180":
            x, y = 1.0 - x, 1.0 - y
        elif variant == "rot270":
            x, y, w, h = 1.0 - y, x, h, w
        transformed.append({**label, "x_center": x, "y_center": y, "width": w, "height": h})
    return transformed


def transform_image(image: Image.Image, variant: str) -> Image.Image:
    if variant == "hflip":
        return ImageOps.mirror(image)
    if variant == "vflip":
        return ImageOps.flip(image)
    if variant == "rot90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if variant == "rot180":
        return image.transpose(Image.Transpose.ROTATE_180)
    if variant == "rot270":
        return image.transpose(Image.Transpose.ROTATE_270)
    if variant == "bright085":
        return ImageEnhance.Brightness(image).enhance(0.85)
    if variant == "bright115":
        return ImageEnhance.Brightness(image).enhance(1.15)
    if variant == "contrast090":
        return ImageEnhance.Contrast(image).enhance(0.90)
    if variant == "contrast110":
        return ImageEnhance.Contrast(image).enhance(1.10)
    return image.copy()


def write_augmented_label_file(path: Path, labels: list[dict[str, Any]]) -> Counter[int]:
    lines: list[str] = []
    counts: Counter[int] = Counter()
    for label in labels:
        class_id = REVIEW_LABEL_TO_CLASS[str(label["review_label"])]
        counts[class_id] += 1
        lines.append(
            " ".join(
                [
                    str(class_id),
                    f"{float(label['x_center']):.8f}",
                    f"{float(label['y_center']):.8f}",
                    f"{float(label['width']):.8f}",
                    f"{float(label['height']):.8f}",
                ]
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return counts


def make_augmented_sts_crops(
    *,
    output_dir: Path,
    train_records: list[dict[str, Any]],
    counts: dict[str, Any],
    files: list[dict[str, Any]],
    crop_min_size: int,
    crop_max_size: int,
    crop_scale: float,
) -> dict[str, Any]:
    generated = 0
    skipped_unhelpful = 0
    candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for record in train_records:
        review = record["review"]
        for detection in review.get("detections", []):
            if detection.get("review_label") == "ships_sts":
                for variant in VARIANTS:
                    candidates.append((record, detection, variant))

    for record, target_detection, variant in candidates:
        if counts["overall"]["ships"] >= counts["overall"]["ship"]:
            break
        review = record["review"]
        source_image = repo_path(str(review["cache_path"]))
        with Image.open(source_image) as image:
            image = image.convert("RGB")
            crop_box = crop_window_for_detection(
                target_detection, image.width, image.height, crop_min_size, crop_max_size, crop_scale
            )
            crop = image.crop(crop_box)
            labels = labels_in_crop(review, crop_box, image.width, image.height)
            labels = transform_labels(labels, variant)
            label_counts = Counter(REVIEW_LABEL_TO_CLASS[str(label["review_label"])] for label in labels)
            if label_counts[1] == 0 or label_counts[0] > label_counts[1]:
                skipped_unhelpful += 1
                continue
            augmented = transform_image(crop, variant)
            image_key = str(review["image_key"])
            detection_id = str(target_detection["detection_id"])
            output_name = f"gavin_augships__{image_key}__{detection_id}__{variant}.png"
            target_image = output_dir / "images" / "train" / output_name
            target_label = output_dir / "labels" / "train" / f"{Path(output_name).stem}.txt"
            augmented.save(target_image)
            written_counts = write_augmented_label_file(target_label, labels)

        add_counts(counts, "train", "manual_augmented_sts_crops", written_counts)
        files.append(record_file(output_dir, target_image))
        files.append(record_file(output_dir, target_label))
        generated += 1

    return {
        "generated_images": generated,
        "skipped_unhelpful_candidates": skipped_unhelpful,
        "candidate_images": len(candidates),
        "stopped_when_overall_ships_reached_ship_count": counts["overall"]["ships"] >= counts["overall"]["ship"],
    }


def write_data_yaml(output_dir: Path) -> Path:
    path = output_dir / "data.yaml"
    path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ship\n"
        "  1: ships\n"
    )
    return path


def plain_counts(counts: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, defaultdict):
            return {key: convert(child) for key, child in value.items()}
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, dict):
            return {key: convert(child) for key, child in value.items()}
        return value

    return convert(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced Madhulika + manual-reviewed STS YOLO dataset.")
    parser.add_argument("--old-dataset", default=str(DEFAULT_OLD_DATASET))
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-min-size", type=int, default=256)
    parser.add_argument("--crop-max-size", type=int, default=512)
    parser.add_argument("--crop-scale", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    old_dataset = repo_path(args.old_dataset)
    reviews_dir = repo_path(args.reviews_dir)
    output_dir = repo_path(args.output_dir)
    clean_output_dir(output_dir, args.overwrite)
    ensure_dirs(output_dir)

    counts: dict[str, Any] = {
        "overall": Counter(),
        "by_split": defaultdict(Counter),
        "by_source": defaultdict(Counter),
        "by_split_and_source": defaultdict(lambda: defaultdict(Counter)),
    }
    exported_files: list[dict[str, Any]] = []

    copy_old_dataset(old_dataset, output_dir, counts, exported_files)
    manual_records, excluded_counts = load_completed_reviews_with_sts(reviews_dir)
    split_records = split_manual_records(manual_records, args.seed)
    copy_manual_raw(output_dir, split_records, counts, exported_files)
    augmentation_summary = make_augmented_sts_crops(
        output_dir=output_dir,
        train_records=split_records["train"],
        counts=counts,
        files=exported_files,
        crop_min_size=args.crop_min_size,
        crop_max_size=args.crop_max_size,
        crop_scale=args.crop_scale,
    )
    data_yaml = write_data_yaml(output_dir)
    exported_files.append(record_file(output_dir, data_yaml))

    manifest = {
        "schema_version": 1,
        "dataset_name": "madhulika_rectbbox_plus_manual_sts_balanced",
        "dataset_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "generator": str(Path(__file__).relative_to(ROOT_DIR)),
        "command_recreate": (
            ".venv/bin/python scripts/build_madhulika_manual_sts_balanced_dataset.py "
            "--overwrite"
        ),
        "user_intent": (
            "Combine the old Madhulika rectbbox dataset used for the old promoted weights with the completed "
            "manual-reviewed STS examples. The old 426 `ships` boxes are treated as already augmented; the new "
            "35 manual `ships` boxes are augmented with targeted train-only crops and no blur. Ship-only manual "
            "reviews are intentionally excluded from this first balanced run so the class ratio stays about 1:1."
        ),
        "classes": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "sources": {
            "old_dataset": str(old_dataset.relative_to(ROOT_DIR) if old_dataset.is_relative_to(ROOT_DIR) else old_dataset),
            "reviews_dir": str(reviews_dir.relative_to(ROOT_DIR) if reviews_dir.is_relative_to(ROOT_DIR) else reviews_dir),
        },
        "manual_selection_policy": {
            "include": "completed review files with at least one ships_sts label; export all positive labels on those images",
            "exclude_first_balanced_run": "completed ship-only manual review files",
            "excluded_counts": dict(excluded_counts),
            "split_seed": args.seed,
            "manual_val_target_ships": TARGET_MANUAL_SPLIT_SHIPS["val"],
            "manual_test_target_ships": TARGET_MANUAL_SPLIT_SHIPS["test"],
            "manual_records_by_split": {
                split: [str(record["review"].get("image_key")) for record in records]
                for split, records in split_records.items()
            },
        },
        "augmentation_policy": {
            "applies_to": "manual train split ships_sts boxes only, emitted as focused crop images",
            "target": "generate helpful crops until overall ships count reaches overall ship count or candidates are exhausted",
            "blur": "disabled by user intent",
            "variants": list(VARIANTS),
            "crop_min_size_px": args.crop_min_size,
            "crop_max_size_px": args.crop_max_size,
            "crop_scale": args.crop_scale,
            "included_crop_labels": "positive boxes with at least 50 percent of original bbox area inside the crop",
            "candidate_filter": "skip crops whose ship count exceeds ships count",
            **augmentation_summary,
        },
        "counts": plain_counts(counts),
        "exported_files": exported_files,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "counts": manifest["counts"], "augmentation": augmentation_summary}, indent=2))


if __name__ == "__main__":
    main()
