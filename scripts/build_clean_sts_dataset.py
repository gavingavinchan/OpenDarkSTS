#!/usr/bin/env python3
"""Build a leak-free STS detection dataset (``sts_clean_v1``).

Inputs
------
1. ``workspace/datasets/derived/madhulika_yolov8_rectbbox`` -- the Roboflow export.
   It is pre-augmented: each source image appears 3-6 times (random crop, flip,
   rotation, brightness) under a shared base name, copies of one scene are spread
   across train/valid/test, and several scenes ship both a ``True_color`` and a
   ``False_color`` render. Only one representative per scene base name is kept
   (preferring ``True_color``, then the most labelled boxes, then the largest
   image), which removes both the augmented duplicates and the False_color
   duplicate of a True_color scene.
2. ``workspace/reviews/gavin`` -- every completed manual review, including
   ship-only images and pure negatives. ``ship_non_sts`` -> class 0 ``ship``,
   ``ships_sts`` -> class 1 ``ships``; ``neither`` is dropped (the box stays in the
   review file for reviewer context only).

No augmentation is baked in. Use Ultralytics online augmentation at train time.

Splitting is delegated to :mod:`scripts.dataset_splits`: whole
``<region>|<date>`` groups go to exactly one split. This matters because the
Madhulika scenes, the ``sentinel2`` review tiles and the
``ship_tiles_new_apr29_2025`` review tiles are all Gulf-of-Mexico imagery and
share several acquisition dates -- keying on the source collection would have
left the same scene on both sides of the split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset_splits import (  # type: ignore
        DEFAULT_RATIOS,
        GroupStats,
        assign_groups_to_splits,
        check_no_group_overlap,
        group_key,
        parse_source_name,
        region_for_bounds,
    )
else:  # pragma: no cover - package import path
    from .dataset_splits import (
        DEFAULT_RATIOS,
        GroupStats,
        assign_groups_to_splits,
        check_no_group_overlap,
        group_key,
        parse_source_name,
        region_for_bounds,
    )

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MADHULIKA_DIR = ROOT_DIR / "workspace/datasets/derived/madhulika_yolov8_rectbbox"
DEFAULT_REVIEWS_DIR = ROOT_DIR / "workspace/reviews/gavin"
DEFAULT_IMPORT_MANIFEST = ROOT_DIR / "workspace/manifests/training_import_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "workspace/datasets/derived/sts_clean_v1"

CLASS_NAMES = {0: "ship", 1: "ships"}
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}
MADHULIKA_SPLIT_DIRS = ("train", "valid", "test")

# Review cache collections whose source rasters are not in the import manifest.
# Both were cut from Gulf of Mexico Sentinel-2 scenes:
#   workspace/cache/sentinel_tiles            <- "AIS data/Sentinel 2A/sentinel2"       bounds ~ (-94.73, 29.07, -94.57, 29.21)
#   workspace/cache/ship_tiles_new_apr29_2025 <- "AIS data/Sentinel 2A/2020/ship_tiles new Apr 29" bounds ~ (-94.30, 28.57, -94.24, 28.63)
CACHE_DIR_REGIONS = {
    "sentinel_tiles": "texas_gulf",
    "ship_tiles_new_apr29_2025": "texas_gulf",
}
# Fallback when an imported image carries no georeferenced bounds: the read-only
# source directory it was imported from already names the region.
SOURCE_PATH_REGIONS = (
    ("/Sentinel 2A/2020/ship_tiles", "texas_gulf"),
    ("/Sentinel 2A/sentinel2", "texas_gulf"),
    ("/Sentinel 2A/Gibraltar", "gibraltar"),
    ("/Sentinel 2A/Suez", "suez"),
    ("/Sentinel 2A/Singapore", "singapore"),
)
# The Madhulika Roboflow scenes are Gulf of Mexico too (e.g. the 2020-01-14 source
# GeoTIFF in the repository root has bounds -94.53/28.39 .. -94.48/28.42).
MADHULIKA_REGION = "texas_gulf"


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def rel_to_root(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT_DIR)) if resolved.is_relative_to(ROOT_DIR) else str(resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def parse_label_text(text: str) -> tuple[list[str], Counter]:
    lines: list[str] = []
    counts: Counter = Counter()
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id not in CLASS_NAMES:
            continue
        values = [float(value) for value in parts[1:5]]
        counts[CLASS_NAMES[class_id]] += 1
        lines.append(" ".join([str(class_id), *(f"{value:.8f}" for value in values)]))
    return lines, counts


class Item:
    """One image plus its labels, provenance and group assignment."""

    def __init__(
        self,
        *,
        item_id: str,
        source: str,
        image_path: Path,
        original_name: str,
        region: str,
        label_lines: list[str],
        class_counts: Counter,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.item_id = item_id
        self.source = source
        self.image_path = image_path
        self.original_name = original_name
        self.region = region
        self.label_lines = label_lines
        self.class_counts = class_counts
        self.parsed = parse_source_name(original_name)
        self.group = group_key(self.parsed, region)
        self.extra = extra or {}
        self.split: str | None = None
        self.output_name: str | None = None


# --------------------------------------------------------------------------- #
# Madhulika
# --------------------------------------------------------------------------- #
def collect_madhulika_items(madhulika_dir: Path) -> tuple[list[Item], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_files = 0
    for split_dir in MADHULIKA_SPLIT_DIRS:
        image_dir = madhulika_dir / split_dir / "images"
        label_dir = madhulika_dir / split_dir / "labels"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing Madhulika split: {image_dir}")
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            total_files += 1
            label_path = label_dir / f"{image_path.stem}.txt"
            text = label_path.read_text() if label_path.exists() else ""
            lines, counts = parse_label_text(text)
            with Image.open(image_path) as image:
                width, height = image.size
            parsed = parse_source_name(image_path.name)
            candidates[parsed.base].append(
                {
                    "image_path": image_path,
                    "original_split": split_dir,
                    "lines": lines,
                    "counts": counts,
                    "pixels": width * height,
                    "render_profile": parsed.render_profile,
                    "parsed": parsed,
                }
            )

    items: list[Item] = []
    dropped_augmented = 0
    dropped_false_color = 0
    dedup_records: list[dict[str, Any]] = []
    for base, entries in sorted(candidates.items()):
        has_true_color = any(entry["render_profile"] == "true_color" for entry in entries)
        pool = [entry for entry in entries if entry["render_profile"] == "true_color"] if has_true_color else entries
        dropped_false_color += len(entries) - len(pool) if has_true_color else 0
        best = max(
            pool,
            key=lambda entry: (
                sum(entry["counts"].values()),
                entry["counts"].get("ships", 0),
                entry["pixels"],
                entry["image_path"].name,
            ),
        )
        dropped_augmented += len(pool) - 1
        image_path = best["image_path"]
        items.append(
            Item(
                item_id=f"madhulika__{image_path.stem}",
                source="madhulika_rectbbox",
                image_path=image_path,
                original_name=image_path.name,
                region=MADHULIKA_REGION,
                label_lines=best["lines"],
                class_counts=best["counts"],
                extra={
                    "scene_base": base,
                    "roboflow_original_split": best["original_split"],
                    "render_profile": best["render_profile"],
                    "duplicate_files_dropped": len(entries) - 1,
                },
            )
        )
        dedup_records.append(
            {
                "scene_base": base,
                "files_seen": len(entries),
                "kept": image_path.name,
                "render_profiles_seen": sorted({str(entry["render_profile"]) for entry in entries}),
                "roboflow_splits_seen": sorted({entry["original_split"] for entry in entries}),
            }
        )

    summary = {
        "files_seen": total_files,
        "scene_bases": len(candidates),
        "images_kept": len(items),
        "augmented_duplicates_dropped": dropped_augmented,
        "false_color_duplicates_dropped": dropped_false_color,
        "multi_split_scene_bases": sorted(
            record["scene_base"]
            for record in dedup_records
            if len(record["roboflow_splits_seen"]) > 1
        ),
        "dedup_records": dedup_records,
    }
    return items, summary


# --------------------------------------------------------------------------- #
# Manual reviews
# --------------------------------------------------------------------------- #
def load_import_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("images", {}) if isinstance(payload, dict) else {}


def album_region_index(import_images: dict[str, Any]) -> dict[str, str]:
    """Majority region per album id, used when an image has no bounds of its own."""
    tally: dict[str, Counter] = defaultdict(Counter)
    for entry in import_images.values():
        region = region_for_bounds((entry.get("render") or {}).get("metadata", {}).get("bounds"))
        if region == "unknown":
            continue
        for album_id in entry.get("album_ids") or []:
            tally[str(album_id)][region] += 1
    return {album_id: counter.most_common(1)[0][0] for album_id, counter in tally.items()}


def region_for_import_entry(entry: dict[str, Any], album_regions: dict[str, str]) -> str:
    """Resolve a region for an imported image: bounds, then source path, then album."""
    region = region_for_bounds((entry.get("render") or {}).get("metadata", {}).get("bounds"))
    if region != "unknown":
        return region
    source_path = str(entry.get("source_path") or "")
    for marker, marker_region in SOURCE_PATH_REGIONS:
        if marker in source_path:
            return marker_region
    for album_id in entry.get("album_ids") or []:
        if str(album_id) in album_regions:
            return album_regions[str(album_id)]
    return "unknown"


def review_origin(
    review: dict[str, Any], import_images: dict[str, Any], album_regions: dict[str, str]
) -> tuple[str, str]:
    """Return ``(original_name, region)`` for a review document."""
    image_key = str(review.get("image_key", ""))
    entry = import_images.get(image_key)
    if entry:
        original = entry.get("browser_relative_path") or entry.get("original_filename") or entry.get("image_filename")
        return str(original), region_for_import_entry(entry, album_regions)
    cache_path = str(review.get("cache_path") or "")
    cache_dir = Path(cache_path).parent.name
    region = CACHE_DIR_REGIONS.get(cache_dir, "unknown")
    original = str(review.get("image_filename") or image_key)
    return original, region


def collect_review_items(reviews_dir: Path, import_manifest: Path) -> tuple[list[Item], dict[str, Any]]:
    import_images = load_import_manifest(import_manifest)
    album_regions = album_region_index(import_images)
    items: list[Item] = []
    skipped: Counter = Counter()
    dropped_neither = 0
    for path in sorted(reviews_dir.glob("**/*.json")):
        if any(part.startswith("_") for part in path.relative_to(reviews_dir).parts[:-1]):
            continue
        review = json.loads(path.read_text())
        if review.get("review_status") != "completed":
            skipped["not_completed"] += 1
            continue
        image_path = repo_path(str(review["cache_path"]))
        if not image_path.exists():
            skipped["missing_cache_image"] += 1
            continue
        lines: list[str] = []
        counts: Counter = Counter()
        for detection in review.get("detections", []):
            label = detection.get("review_label")
            if label not in REVIEW_LABEL_TO_CLASS:
                dropped_neither += 1
                continue
            class_id = REVIEW_LABEL_TO_CLASS[label]
            counts[CLASS_NAMES[class_id]] += 1
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
        original_name, region = review_origin(review, import_images, album_regions)
        items.append(
            Item(
                item_id=f"review__{review['image_key']}",
                source="manual_review",
                image_path=image_path,
                original_name=original_name,
                region=region,
                label_lines=lines,
                class_counts=counts,
                extra={
                    "image_key": review["image_key"],
                    "review_file": rel_to_root(path),
                    "cache_collection": Path(str(review.get("cache_path"))).parent.name,
                    "is_negative": not lines,
                },
            )
        )
    summary = {
        "completed_reviews_used": len(items),
        "skipped": dict(skipped),
        "neither_boxes_dropped": dropped_neither,
        "pure_negative_images": sum(1 for item in items if not item.label_lines),
        "region_counts": dict(Counter(item.region for item in items)),
    }
    return items, summary


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_group_stats(items: list[Item]) -> list[GroupStats]:
    by_key: dict[str, GroupStats] = {}
    for item in items:
        stats = by_key.setdefault(item.group, GroupStats(key=item.group))
        stats.add(item.class_counts)
    return list(by_key.values())


def write_data_yaml(output_dir: Path) -> Path:
    path = output_dir / "data.yaml"
    path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: ship\n"
        "  1: ships\n"
    )
    return path


def clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --overwrite to rebuild it.")
    allowed_root = (ROOT_DIR / "workspace/datasets/derived").resolve()
    resolved = output_dir.resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise ValueError(f"Refusing to remove output outside {allowed_root}: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the leak-free sts_clean_v1 YOLO dataset.")
    parser.add_argument("--madhulika-dir", default=str(DEFAULT_MADHULIKA_DIR))
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR))
    parser.add_argument("--import-manifest", default=str(DEFAULT_IMPORT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-name", default="sts_clean_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument("--min-test-images", type=int, default=100)
    parser.add_argument("--min-test-ships", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    madhulika_dir = repo_path(args.madhulika_dir)
    reviews_dir = repo_path(args.reviews_dir)
    output_dir = repo_path(args.output_dir)

    madhulika_items, madhulika_summary = collect_madhulika_items(madhulika_dir)
    review_items, review_summary = collect_review_items(reviews_dir, repo_path(args.import_manifest))
    items = madhulika_items + review_items

    groups = build_group_stats(items)
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    min_requirements = {"test": {"images": args.min_test_images, "ships": args.min_test_ships}}
    assignment = assign_groups_to_splits(
        groups, ratios=ratios, seed=args.seed, min_requirements=min_requirements
    )
    for item in items:
        item.split = assignment[item.group]

    clean_output_dir(output_dir, args.overwrite)
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts: dict[str, Counter] = {split: Counter() for split in ("train", "val", "test")}
    image_counts: Counter = Counter()
    negative_counts: Counter = Counter()
    source_counts: dict[str, Counter] = defaultdict(Counter)
    item_records: list[dict[str, Any]] = []

    for item in items:
        suffix = item.image_path.suffix.lower() or ".png"
        output_name = f"{item.item_id}{suffix}"
        item.output_name = output_name
        target_image = output_dir / "images" / item.split / output_name
        target_label = output_dir / "labels" / item.split / f"{Path(output_name).stem}.txt"
        shutil.copy2(item.image_path, target_image)
        target_label.write_text("\n".join(item.label_lines) + ("\n" if item.label_lines else ""))

        counts[item.split].update(item.class_counts)
        image_counts[item.split] += 1
        if not item.label_lines:
            negative_counts[item.split] += 1
        source_counts[item.split][item.source] += 1
        item_records.append(
            {
                "item_id": item.item_id,
                "source": item.source,
                "split": item.split,
                "group": item.group,
                "region": item.region,
                "date": item.parsed.date,
                "scene": item.parsed.scene,
                "original_name": item.original_name,
                "source_path": rel_to_root(item.image_path),
                "output_image": f"images/{item.split}/{output_name}",
                "output_label": f"labels/{item.split}/{Path(output_name).stem}.txt",
                "class_counts": dict(item.class_counts),
                "sha256": sha256_file(target_image),
                **item.extra,
            }
        )

    leak_report = check_no_group_overlap({item.item_id: (item.split, item.group) for item in items})
    data_yaml = write_data_yaml(output_dir)

    group_records = sorted(
        (
            {
                "group": stats.key,
                "split": assignment[stats.key],
                "images": stats.images,
                "boxes": dict(stats.boxes),
            }
            for stats in groups
        ),
        key=lambda record: record["group"],
    )

    summary = {
        "images_by_split": dict(image_counts),
        "boxes_by_split": {split: dict(values) for split, values in counts.items()},
        "negative_images_by_split": dict(negative_counts),
        "sources_by_split": {split: dict(values) for split, values in source_counts.items()},
        "groups_by_split": leak_report["group_counts_by_split"],
        "leak_check": {
            "ok": leak_report["ok"],
            "group_count": leak_report["group_count"],
            "overlapping_groups": leak_report["overlapping_groups"],
        },
    }

    manifest = {
        "schema_version": 1,
        "dataset_name": args.dataset_name,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "generator": rel_to_root(Path(__file__)),
        "command_recreate": (
            ".venv/bin/python scripts/build_clean_sts_dataset.py --overwrite"
            + (f" --seed {args.seed}" if args.seed != 42 else "")
        ),
        "user_intent": (
            "Produce the first trustworthy STS detection dataset: deduplicate the pre-augmented Madhulika "
            "Roboflow export to one image per scene, add every completed manual review (ship-only images and "
            "pure negatives included), and split by whole (region, date) scene groups so no observation can "
            "appear in more than one split. No augmentation is baked in; Ultralytics online augmentation is "
            "used at training time instead."
        ),
        "classes": {str(class_id): name for class_id, name in CLASS_NAMES.items()},
        "split_rule": {
            "group_key": "<region>|<date>",
            "region_source": "raster bounds from the import manifest, else a documented per-collection constant",
            "scene_token": "parsed and recorded but deliberately not part of the key (see scripts/dataset_splits.py)",
            "ratios": ratios,
            "seed": args.seed,
            "min_requirements": min_requirements,
            "helper": "scripts/dataset_splits.py::assign_groups_to_splits",
        },
        "sources": {
            "madhulika_dir": rel_to_root(madhulika_dir),
            "reviews_dir": rel_to_root(reviews_dir),
            "import_manifest": rel_to_root(repo_path(args.import_manifest)),
        },
        "madhulika_dedup": madhulika_summary,
        "manual_reviews": review_summary,
        "summary": summary,
        "groups": group_records,
        "items": item_records,
        "data_yaml": rel_to_root(data_yaml),
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"output_dir": rel_to_root(output_dir), **summary}, indent=2, sort_keys=True))
    print()
    print("LEAK CHECK:", "PASS - no group appears in more than one split" if leak_report["ok"] else "FAIL")
    if not leak_report["ok"]:
        raise SystemExit(f"Group leak detected: {leak_report['overlapping_groups']}")


if __name__ == "__main__":
    main()
