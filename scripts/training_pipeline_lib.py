from __future__ import annotations

import hashlib
import csv
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from . import visual_review_lib as review_lib
except ImportError:
    import visual_review_lib as review_lib


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
SUPPORTED_IMPORT_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_IMPORT_MANIFEST_PATH = WORKSPACE_DIR / "manifests" / "training_import_manifest.json"
DEFAULT_IMPORT_CACHE_DIR = WORKSPACE_DIR / "cache" / "imported_images"
RGB_ORIGINAL_RENDER_PROFILE = {"kind": "rgb_original"}
TRUE_COLOR_RENDER_PROFILE = {
    "kind": "true_color",
    "bands": [1, 2, 3],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB",
}
FALSE_COLOR_RENDER_PROFILE = {
    "kind": "false_color",
    "bands": [4, 3, 2],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB",
}
DOCUMENTED_IMPORT_STATUSES = {"imported", "already_imported", "regenerated_cache", "failed"}
REQUIRED_IMPORT_ENTRY_FIELDS = {
    "schema_version",
    "image_key",
    "original_filename",
    "source_path",
    "browser_relative_path",
    "source_collection",
    "source_size_bytes",
    "source_mtime",
    "source_content_hash",
    "width",
    "height",
    "cache_path",
    "cache_width",
    "cache_height",
    "cache_content_hash",
    "render_profile",
    "loader",
    "import_status",
    "imported_at",
    "updated_at",
    "warnings",
}
REVIEW_LABEL_TO_CLASS = {"ship_non_sts": 0, "ships_sts": 1}
CLASS_ID_TO_NAME = {0: "ship", 1: "ships"}
RECOGNIZED_RESULTS_COLUMNS = {
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
}
REQUIRED_TRAINING_CONFIG_FIELDS = {
    "dataset_name": "dataset name/version",
    "dataset_version": "dataset name/version",
    "model": "model or starting weights",
    "image_size": "image size",
    "epochs": "epochs",
    "batch": "batch",
    "device": "device",
    "seed": "seed",
    "run_name": "output run name",
    "augmentation": "augmentation settings",
}


def repo_resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def repo_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(ROOT_DIR))
    except ValueError:
        return str(candidate)


def load_json(path: str | Path) -> dict[str, Any]:
    target = repo_resolve(path)
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {target}")
    return payload


def load_json_or_dict(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return load_json(value)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = repo_resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_import_manifest(manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH) -> dict[str, Any]:
    target = repo_resolve(manifest_path)
    now = utcnow_iso_for_training()
    if not target.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "training_import",
            "created_at": now,
            "updated_at": now,
            "cache_dir": repo_relative(DEFAULT_IMPORT_CACHE_DIR),
            "images": {},
        }
    if not target.is_file():
        raise ValueError(f"Import manifest path is not a file: {target}")
    manifest = json.loads(target.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Import manifest must be a JSON object: {target}")
    if int(manifest.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError("Import manifest must use schema_version 1")
    images = manifest.setdefault("images", {})
    if not isinstance(images, dict):
        raise ValueError("Import manifest images field must be a JSON object")
    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest.setdefault("manifest_type", "training_import")
    manifest.setdefault("created_at", now)
    manifest.setdefault("updated_at", now)
    manifest.setdefault("cache_dir", repo_relative(DEFAULT_IMPORT_CACHE_DIR))
    return manifest


def save_import_manifest(manifest_path: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utcnow_iso_for_training()
    write_json(manifest_path, manifest)


def default_render_profile(kind: str) -> dict[str, Any]:
    profiles = {
        "rgb_original": RGB_ORIGINAL_RENDER_PROFILE,
        "true_color": TRUE_COLOR_RENDER_PROFILE,
        "false_color": FALSE_COLOR_RENDER_PROFILE,
    }
    if kind not in profiles:
        raise ValueError(f"Unknown render profile kind: {kind!r}")
    return copy.deepcopy(profiles[kind])


def resolve_import_render_profile(render_profile_value: str, source_path: str | Path) -> dict[str, Any]:
    value = str(render_profile_value or "auto")
    source_suffix = Path(source_path).suffix.lower()
    if value == "auto":
        if source_suffix in {".png", ".jpg", ".jpeg"}:
            profile = default_render_profile("rgb_original")
        elif source_suffix in {".tif", ".tiff"}:
            profile = default_render_profile("true_color")
        else:
            raise ValueError(f"Unsupported image suffix for auto render profile: {source_suffix}")
    else:
        profile_path = repo_resolve(value)
        if profile_path.is_file():
            profile = load_json(profile_path)
        else:
            profile = default_render_profile(value)
    review_lib.validate_render_profile(profile)
    return profile


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def scan_import_source(source: str | Path, *, recursive: bool) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    source_path = repo_resolve(source)
    if not source_path.exists():
        return [], {
            "image_key": None,
            "source": str(source_path),
            "cache_path": None,
            "render_profile_kind": None,
            "status": "failed",
            "reason": "Source path not found",
        }
    if source_path.is_file():
        if source_path.name.startswith(".") or source_path.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            return [], None
        return [{"source_path": source_path, "source_collection": str(source_path)}], None
    if not source_path.is_dir():
        return [], {
            "image_key": None,
            "source": str(source_path),
            "cache_path": None,
            "render_profile_kind": None,
            "status": "failed",
            "reason": "Source path is neither a file nor a directory",
        }

    pattern = "**/*" if recursive else "*"
    files: list[dict[str, Any]] = []
    for candidate in sorted(source_path.glob(pattern)):
        try:
            relative = candidate.relative_to(source_path)
        except ValueError:
            relative = Path(candidate.name)
        if is_hidden_path(relative):
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_IMPORT_SUFFIXES:
            continue
        files.append({"source_path": candidate, "source_collection": str(source_path)})
    return files, None


def discover_import_files(sources: list[str | Path], *, recursive: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source in sources:
        discovered, failure = scan_import_source(source, recursive=recursive)
        if failure is not None:
            failed.append(failure)
        for candidate in discovered:
            path = repo_resolve(candidate["source_path"])
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append({"source_path": resolved, "source_collection": candidate["source_collection"]})
    return files, failed


def render_profile_bands(render_profile: dict[str, Any]) -> list[int]:
    bands = render_profile.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("render_profile bands must be a non-empty list")
    parsed = [int(value) for value in bands[:3]]
    while len(parsed) < 3:
        parsed.append(parsed[-1])
    return parsed


def load_import_image(source_path: str | Path, render_profile: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    path = repo_resolve(source_path)
    kind = str(render_profile.get("kind"))
    if kind == "rgb_original":
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return rgb.copy(), {"loader": "pillow", "mode": image.mode, "width": rgb.width, "height": rgb.height}

    rasterio = review_lib.optional_rasterio()
    if rasterio is not None:
        with rasterio.open(path) as dataset:
            band_count = max(int(dataset.count), 1)
            indexes = [max(1, min(index, band_count)) for index in render_profile_bands(render_profile)]
            data = dataset.read(indexes)
            import numpy as np

            if data.ndim == 2:
                data = np.stack([data, data, data], axis=0)
            normalized = np.stack([review_lib.normalize_band_to_uint8(data[index]) for index in range(3)], axis=-1)
            image = Image.fromarray(normalized, mode="RGB")
            metadata = {
                "loader": "rasterio",
                "dtype": str(data.dtype),
                "band_count": band_count,
                "bands": indexes,
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bounds": {
                    "left": float(dataset.bounds.left),
                    "bottom": float(dataset.bounds.bottom),
                    "right": float(dataset.bounds.right),
                    "top": float(dataset.bounds.top),
                },
            }
            return image, metadata

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.copy(), {
            "loader": "pillow",
            "mode": image.mode,
            "width": rgb.width,
            "height": rgb.height,
            "render_profile_approximated": kind,
        }


def render_import_cache(source_path: str | Path, cache_path: str | Path, render_profile: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    output = repo_resolve(cache_path)
    image, metadata = load_import_image(source_path, render_profile)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
    return {
        "path": repo_relative(output),
        "width": int(image.width),
        "height": int(image.height),
        "cache_content_hash": None if dry_run else file_hash(output),
        "loader": metadata.get("loader"),
        "metadata": metadata,
        "rendered_at": utcnow_iso_for_training(),
    }


def import_manifest_entry(
    source_path: Path,
    cache_path: Path,
    render_profile: dict[str, Any],
    render_info: dict[str, Any],
    *,
    source_collection: str,
    import_status: str,
    previous_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if import_status not in DOCUMENTED_IMPORT_STATUSES:
        raise ValueError(f"Unknown import status: {import_status}")
    now = utcnow_iso_for_training()
    source_hash = review_lib.source_content_hash(source_path)
    image_key = review_lib.image_key_for_identity_payload(
        {
            "source_content_hash": source_hash,
            "render_profile": render_profile,
        }
    )
    stat = source_path.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "image_filename": f"{image_key}.png",
        "original_filename": source_path.name,
        "display_filename": source_path.name,
        "source_path": str(source_path),
        "browser_relative_path": None,
        "source_collection": source_collection,
        "source_kind": "imported_image",
        "date": None,
        "tile_row": None,
        "tile_col": None,
        "source_content_hash": source_hash,
        "source_size_bytes": stat.st_size,
        "source_mtime": stat.st_mtime,
        "width": int(render_info["width"]),
        "height": int(render_info["height"]),
        "cache_path": repo_relative(cache_path),
        "cache_width": int(render_info["width"]),
        "cache_height": int(render_info["height"]),
        "cache_content_hash": render_info.get("cache_content_hash"),
        "render_profile": render_profile,
        "loader": render_info.get("loader"),
        "import_status": import_status,
        "imported_at": (previous_entry or {}).get("imported_at") or now,
        "updated_at": now,
        "warnings": [],
        "render": {
            "loader": render_info.get("loader"),
            "rendered_at": render_info.get("rendered_at"),
            "output_format": "png",
            "metadata": render_info.get("metadata") or {},
        },
    }


def import_result_item(entry: dict[str, Any], status: str) -> dict[str, Any]:
    if status not in DOCUMENTED_IMPORT_STATUSES:
        raise ValueError(f"Unknown import status: {status}")
    return {
        "image_key": entry["image_key"],
        "source": entry.get("source_path") or entry.get("source"),
        "cache_path": entry["cache_path"],
        "render_profile_kind": entry.get("render_profile", {}).get("kind"),
        "status": status,
    }


def import_entry_for_review_stub(entry: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(entry)
    adapted.setdefault("image_filename", f"{entry['image_key']}.png")
    adapted.setdefault("display_filename", entry.get("original_filename") or adapted["image_filename"])
    adapted.setdefault("date", None)
    adapted.setdefault("tile_row", None)
    adapted.setdefault("tile_col", None)
    return adapted


def attach_review_stub_result(
    item: dict[str, Any],
    entry: dict[str, Any],
    *,
    reviewer: str,
    reviews_dir: str | Path,
    skip: bool,
    dry_run: bool,
) -> dict[str, Any]:
    review_path = review_lib.review_file_path(entry["image_key"], reviewer, reviews_dir)
    item["review_stub_path"] = repo_relative(review_path)
    if skip or dry_run:
        item["review_stub_status"] = "skipped"
        return item
    if review_path.exists():
        item["review_stub_status"] = "existing"
        return item
    try:
        review_stub = review_lib.build_review_stub(import_entry_for_review_stub(entry), reviewer, merged_detections=[])
        review_lib.write_review_stub(review_doc=review_stub, reviewer=reviewer, reviews_dir=reviews_dir, overwrite=False)
        item["review_stub_status"] = "created"
    except Exception as exc:
        item["review_stub_status"] = "failed"
        item["review_stub_error"] = str(exc)
    return item


def review_stub_failure_item(item: dict[str, Any]) -> dict[str, Any]:
    failed_item = dict(item)
    failed_item["status"] = "failed"
    failed_item["reason"] = f"Review stub failed: {item.get('review_stub_error') or 'unknown error'}"
    return failed_item


def import_entry_contract_complete(entry: dict[str, Any]) -> bool:
    if not REQUIRED_IMPORT_ENTRY_FIELDS.issubset(entry):
        return False
    if entry.get("import_status") not in DOCUMENTED_IMPORT_STATUSES - {"failed"}:
        return False
    if not entry.get("cache_content_hash"):
        return False
    return True


def import_training_data(
    sources: list[str | Path],
    *,
    manifest_path: str | Path = DEFAULT_IMPORT_MANIFEST_PATH,
    cache_dir: str | Path = DEFAULT_IMPORT_CACHE_DIR,
    render_profile_value: str = "auto",
    reviews_dir: str | Path = review_lib.DEFAULT_REVIEWS_DIR,
    reviewer: str = review_lib.DEFAULT_REVIEWER,
    create_review_stubs: bool = True,
    recursive: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one --source is required")
    manifest_file = repo_resolve(manifest_path)
    cache_root = repo_resolve(cache_dir)
    if manifest_file.exists() and not manifest_file.is_file():
        raise ValueError(f"Import manifest path is not a file: {manifest_file}")
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError(f"Import cache path is not a directory: {cache_root}")

    manifest = load_import_manifest(manifest_file)
    manifest["manifest_type"] = "training_import"
    manifest["cache_dir"] = repo_relative(cache_root)
    files, failed = discover_import_files(sources, recursive=recursive)
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    review_stub_counts = {"created": 0, "existing": 0, "skipped": 0}

    for candidate in files:
        source_path = repo_resolve(candidate["source_path"])
        source_collection = str(candidate["source_collection"])
        try:
            render_profile = resolve_import_render_profile(render_profile_value, source_path)
            image_key = review_lib.image_key_for_source(source_path, render_profile)
            cache_path = cache_root / f"{image_key}.png"
            existing = manifest.get("images", {}).get(image_key)
            cache_exists = cache_path.exists()
            existing_complete = bool(existing and import_entry_contract_complete(existing))
            if existing and cache_exists and existing_complete and not force:
                item = import_result_item(existing, "already_imported")
                item = attach_review_stub_result(
                    item,
                    existing,
                    reviewer=reviewer,
                    reviews_dir=reviews_dir,
                    skip=not create_review_stubs,
                    dry_run=dry_run,
                )
                if item["review_stub_status"] == "failed":
                    failed.append(review_stub_failure_item(item))
                    continue
                review_stub_counts[item["review_stub_status"]] += 1
                skipped.append(item)
                continue

            status = "regenerated_cache" if existing else "imported"
            render_info = render_import_cache(source_path, cache_path, render_profile, dry_run=dry_run)
            entry = import_manifest_entry(
                source_path,
                cache_path,
                render_profile,
                render_info,
                source_collection=source_collection,
                import_status=status,
                previous_entry=existing,
            )
            item = import_result_item(entry, status)
            item = attach_review_stub_result(
                item,
                entry,
                reviewer=reviewer,
                reviews_dir=reviews_dir,
                skip=not create_review_stubs,
                dry_run=dry_run,
            )
            if item["review_stub_status"] == "failed":
                failed.append(review_stub_failure_item(item))
                continue
            review_stub_counts[item["review_stub_status"]] += 1
            imported.append(item)
            if not dry_run:
                manifest.setdefault("images", {})[image_key] = entry
        except Exception as exc:
            render_profile_kind = None
            try:
                render_profile_kind = resolve_import_render_profile(render_profile_value, source_path).get("kind")
            except Exception:
                pass
            failed.append(
                {
                    "image_key": None,
                    "source": str(source_path),
                    "cache_path": None,
                    "render_profile_kind": render_profile_kind,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    if not dry_run and imported:
        save_import_manifest(manifest_file, manifest)

    return {
        "manifest_path": repo_relative(manifest_file),
        "cache_dir": repo_relative(cache_root),
        "processed_count": len(imported) + len(skipped) + len(failed),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "review_stubs_created_count": review_stub_counts["created"],
        "review_stubs_existing_count": review_stub_counts["existing"],
        "review_stubs_skipped_count": review_stub_counts["skipped"],
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
    }


def yaml_defines_required_classes(data_yaml: Path) -> bool:
    if not data_yaml.exists():
        return False
    content = data_yaml.read_text()
    return "0: ship" in content and "1: ships" in content


def split_has_images(dataset_dir: Path, split: str) -> bool:
    image_dir = dataset_dir / "images" / split
    if not image_dir.is_dir():
        return False
    return any(path.is_file() for path in image_dir.iterdir())


def config_missing_items(config: dict[str, Any]) -> list[str]:
    missing = []
    emitted = set()
    for field, label in REQUIRED_TRAINING_CONFIG_FIELDS.items():
        if field in config and config[field] not in (None, ""):
            continue
        if label not in emitted:
            missing.append(f"Missing required training config field: {label}")
            emitted.add(label)
    return missing


def check_training_readiness(selection_path: str | Path, training_config: str | Path | dict[str, Any]) -> dict[str, Any]:
    selection = load_json(selection_path)
    config = load_json_or_dict(training_config)
    output_dir = selection.get("export", {}).get("output_dir")
    dataset_dir = repo_resolve(output_dir) if output_dir else None
    missing: list[str] = []
    warnings: list[str] = []

    if dataset_dir is None or not (dataset_dir / "dataset_manifest.json").is_file():
        missing.append("Selected dataset has not been exported dataset to YOLO format")
    if dataset_dir is None or not (dataset_dir / "data.yaml").is_file():
        missing.append("Exported dataset is missing data.yaml")
    elif not yaml_defines_required_classes(dataset_dir / "data.yaml"):
        missing.append("data.yaml must define class 0: ship and class 1: ships")

    for split in ("train", "val"):
        if dataset_dir is None or not (dataset_dir / "images" / split).is_dir():
            missing.append(f"Exported dataset is missing images/{split}")
        if dataset_dir is None or not (dataset_dir / "labels" / split).is_dir():
            missing.append(f"Exported dataset is missing labels/{split}")
        if dataset_dir is None or not split_has_images(dataset_dir, split):
            missing.append(f"{split} split must contain at least one image")

    if dataset_dir is not None and not split_has_images(dataset_dir, "test"):
        warnings.append("test split has no images; final test metrics will be unavailable")

    missing.extend(config_missing_items(config))

    run_name = config.get("run_name")
    run_group = config.get("run_group") or "default"
    output_root = config.get("output_dir") or ROOT_DIR / "workspace" / "runs"
    if run_name:
        run_dir = repo_resolve(output_root) / str(run_group) / str(run_name)
        if run_dir.exists() and not bool(config.get("overwrite")):
            missing.append("output run name would overwrite an existing run; enable overwrite to proceed")

    return {"ready": not missing, "missing": missing, "warnings": warnings}


def config_value(config: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = config.get(key, fallback)
    if value is None:
        return fallback
    return value


def build_yolo_train_command(training_config: dict[str, Any], dataset_dir: str | Path) -> list[str]:
    dataset_path = repo_resolve(dataset_dir)
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    run_group = str(config_value(training_config, "run_group", "default"))
    run_name = str(training_config["run_name"])
    output_root = repo_resolve(config_value(training_config, "output_dir", WORKSPACE_DIR / "runs"))
    command = [
        "yolo",
        "detect",
        "train",
        f"model={training_config['model']}",
        f"data={data_yaml}",
        f"imgsz={training_config['image_size']}",
        f"epochs={training_config['epochs']}",
        f"batch={training_config['batch']}",
        f"device={training_config['device']}",
        f"seed={training_config['seed']}",
        f"project={output_root / run_group}",
        f"name={run_name}",
    ]
    for optional_key in ("patience", "workers"):
        if optional_key in training_config and training_config[optional_key] is not None:
            command.append(f"{optional_key}={training_config[optional_key]}")
    for key, value in sorted((training_config.get("augmentation") or {}).items()):
        command.append(f"{key}={value}")
    if bool(training_config.get("overwrite")):
        command.append("exist_ok=True")
    return command


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_epoch(value: Any, fallback: int) -> int:
    parsed = parse_float(value)
    if parsed is None:
        return fallback
    return int(parsed)


def load_training_results_series(results_csv: str | Path) -> dict[str, Any]:
    target = repo_resolve(results_csv)
    if not target.exists():
        return {"status": "waiting", "message": "Waiting for first epoch", "series": {}}

    with target.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        recognized = [field for field in fieldnames if field in RECOGNIZED_RESULTS_COLUMNS]
        series: dict[str, list[list[float | int]]] = {field: [] for field in recognized}
        for fallback_epoch, row in enumerate(reader, start=1):
            normalized_row = {str(key).strip(): value for key, value in row.items()}
            epoch = parse_epoch(normalized_row.get("epoch"), fallback_epoch)
            for field in recognized:
                value = parse_float(normalized_row.get(field))
                if value is not None:
                    series[field].append([epoch, value])

    return {"status": "ready", "message": None, "series": series}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def optional_package_version(package_name: str) -> str | None:
    try:
        module = __import__(package_name)
    except ModuleNotFoundError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package_name in ("ultralytics", "torch"):
        version = optional_package_version(package_name)
        if version is not None:
            versions[package_name] = version
    return versions


def git_info() -> dict[str, Any]:
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return info
    info["commit"] = commit
    info["dirty"] = bool(status)
    return info


def find_review_for_image(image_key: str, reviews_dir: str | Path, *, include_incomplete: bool = False) -> tuple[dict[str, Any], Path]:
    root = repo_resolve(reviews_dir)
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(root.glob("**/*.json")):
        review = load_json(path)
        if str(review.get("image_key") or "") != image_key:
            continue
        if include_incomplete or review.get("review_status") == "completed":
            candidates.append((review, path))

    if not candidates:
        raise ValueError(f"No completed review found for image_key={image_key}")
    if len(candidates) > 1:
        paths = ", ".join(str(path) for _, path in candidates)
        raise ValueError(f"Reconciliation conflict for image_key={image_key}: {paths}")
    return candidates[0]


def exportable_detections(review: dict[str, Any], item: dict[str, Any]) -> list[dict[str, Any]]:
    # Reviewed positive labels are ground truth regardless of auto/manual origin.
    selected_ids = item.get("selected_detection_ids")
    if item.get("include_labels") == "selected_only":
        if not isinstance(selected_ids, list):
            raise ValueError(f"selected_only requires selected_detection_ids for image_key={item.get('image_key')}")
        selected = {str(detection_id) for detection_id in selected_ids}
    else:
        selected = None

    detections = []
    for detection in review.get("detections", []):
        if detection.get("review_label") not in REVIEW_LABEL_TO_CLASS:
            continue
        if selected is not None and str(detection.get("detection_id")) not in selected:
            continue
        detections.append(detection)
    return detections


def write_yolo_label_file(path: Path, detections: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for detection in detections:
        class_id = REVIEW_LABEL_TO_CLASS[str(detection["review_label"])]
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


def write_data_yaml(dataset_dir: Path) -> None:
    content = (
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ship\n"
        "  1: ships\n"
    )
    (dataset_dir / "data.yaml").write_text(content)


def relative_export_path(dataset_dir: Path, path: Path) -> str:
    return str(path.relative_to(dataset_dir))


def record_exported_file(dataset_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_export_path(dataset_dir, path),
        "sha256": file_hash(path),
        "size_bytes": path.stat().st_size,
    }


def background_negative_limit(positive_count: int, requested_count: int, max_fraction: float | None) -> int:
    if max_fraction is None:
        return requested_count
    fraction = float(max_fraction)
    if fraction <= 0.0 or positive_count <= 0:
        return 0
    if fraction >= 1.0:
        return requested_count
    return min(requested_count, math.floor((fraction * positive_count) / (1.0 - fraction)))


def ensure_dataset_dirs(dataset_dir: Path) -> None:
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_review_image(review: dict[str, Any], image_target: Path) -> None:
    source = repo_resolve(str(review["cache_path"]))
    if not source.exists():
        raise FileNotFoundError(f"Review cache image missing: {source}")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, image_target)


def export_review_item(
    *,
    dataset_dir: Path,
    review: dict[str, Any],
    item: dict[str, Any],
    detections: list[dict[str, Any]],
) -> list[Path]:
    split = str(item.get("split") or "train")
    if split not in SPLITS:
        raise ValueError(f"Invalid split for image_key={item.get('image_key')}: {split}")

    image_filename = str(review.get("image_filename") or f"{review['image_key']}.png")
    image_stem = Path(image_filename).stem
    image_target = dataset_dir / "images" / split / image_filename
    label_target = dataset_dir / "labels" / split / f"{image_stem}.txt"
    copy_review_image(review, image_target)
    write_yolo_label_file(label_target, detections)
    return [image_target, label_target]


def export_dataset_selection(
    selection_path: str | Path,
    *,
    output_dir_override: str | Path | None = None,
    include_incomplete_override: bool | None = None,
) -> dict[str, Any]:
    selection_file = repo_resolve(selection_path)
    selection = load_json(selection_file)
    if int(selection.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Dataset selection manifest must use schema_version 1")
    if output_dir_override is not None:
        selection.setdefault("export", {})["output_dir"] = str(output_dir_override)
    if include_incomplete_override is not None:
        selection.setdefault("export", {})["include_incomplete_reviews"] = bool(include_incomplete_override)

    reviews_dir = selection.get("sources", {}).get("reviews_dir")
    if not reviews_dir:
        raise ValueError("Dataset selection manifest is missing sources.reviews_dir")
    output_dir = selection.get("export", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Dataset selection manifest is missing export.output_dir")

    dataset_dir = repo_resolve(output_dir)
    include_incomplete = bool(selection.get("export", {}).get("include_incomplete_reviews", False))
    ensure_dataset_dirs(dataset_dir)

    exported_files: list[dict[str, Any]] = []
    exported_image_keys: list[str] = []
    warnings: list[str] = []
    item_count = 0

    for item in selection.get("items", []):
        image_key = str(item.get("image_key") or "")
        if not image_key:
            raise ValueError("Dataset selection item is missing image_key")
        review, _ = find_review_for_image(image_key, reviews_dir, include_incomplete=include_incomplete)
        detections = exportable_detections(review, item)
        for path in export_review_item(dataset_dir=dataset_dir, review=review, item=item, detections=detections):
            exported_files.append(record_exported_file(dataset_dir, path))
        exported_image_keys.append(image_key)
        item_count += 1

    background = selection.get("background_negatives") or {}
    if background.get("enabled"):
        if background.get("mode") != "whole_image_empty_label":
            raise ValueError(f"Unsupported background negative mode: {background.get('mode')}")
        requested_keys = [str(image_key) for image_key in background.get("image_keys") or []]
        limit = background_negative_limit(item_count, len(requested_keys), background.get("max_fraction"))
        if limit < len(requested_keys):
            warnings.append(
                f"Truncated background negatives from {len(requested_keys)} to {limit} because of max_fraction={background.get('max_fraction')}"
            )
        for image_key in requested_keys[:limit]:
            review, _ = find_review_for_image(image_key, reviews_dir, include_incomplete=include_incomplete)
            item = {"image_key": image_key, "split": "train", "include_labels": "selected_only", "selected_detection_ids": []}
            for path in export_review_item(dataset_dir=dataset_dir, review=review, item=item, detections=[]):
                exported_files.append(record_exported_file(dataset_dir, path))
            exported_image_keys.append(image_key)

    write_data_yaml(dataset_dir)
    exported_files.append(record_exported_file(dataset_dir, dataset_dir / "data.yaml"))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": selection.get("dataset_name"),
        "dataset_version": selection.get("dataset_version"),
        "dataset_selection_manifest": str(selection_file),
        "dataset_selection_manifest_sha256": file_hash(selection_file),
        "classes": {str(class_id): class_name for class_id, class_name in CLASS_ID_TO_NAME.items()},
        "exported_image_keys": exported_image_keys,
        "exported_files": exported_files,
        "warnings": warnings,
    }
    write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def write_training_metadata(
    *,
    run_dir: str | Path,
    training_config_path: str | Path,
    dataset_manifest_path: str | Path,
    command_line: list[str],
    status: str,
    started_at: str,
    ended_at: str | None = None,
) -> Path:
    output_run_dir = repo_resolve(run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    config_file = repo_resolve(training_config_path)
    dataset_manifest_file = repo_resolve(dataset_manifest_path)
    data_yaml = dataset_manifest_file.parent / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found next to dataset manifest: {data_yaml}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "training_config_path": str(config_file),
        "training_config_sha256": file_hash(config_file),
        "training_config": load_json(config_file),
        "dataset_manifest_path": str(dataset_manifest_file),
        "dataset_manifest_sha256": file_hash(dataset_manifest_file),
        "data_yaml_path": str(data_yaml),
        "data_yaml_sha256": file_hash(data_yaml),
        "command_line": list(command_line),
        "started_at": started_at,
        "ended_at": ended_at,
        "output_run_dir": str(output_run_dir),
        "package_versions": package_versions(),
        "git": git_info(),
    }
    metadata_path = output_run_dir / "training_metadata.json"
    write_json(metadata_path, metadata)
    return metadata_path


def metric_value(metrics: dict[str, Any] | None, name: str) -> Any:
    if not isinstance(metrics, dict):
        return ""
    return metrics.get(name, "")


def evaluation_markdown(report: dict[str, Any]) -> str:
    promotion = report["promotion"]
    promoted_text = promotion["run_name"] if promotion["promoted"] else "none"
    lines = [
        f"# Evaluation: {report['run_group']}",
        "",
        f"Generated: {report['generated_at']}",
        f"Promoted: {promoted_text}",
        f"Reason: {promotion['reason']}",
        "",
        "## Runs",
        "",
        "| Run | Val mAP50 | Test mAP50 | Test Precision | Test Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report.get("runs", []):
        validation = run.get("validation_metrics") or {}
        test = run.get("test_metrics") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("run_name", "")),
                    str(metric_value(validation, "mAP50")),
                    str(metric_value(test, "mAP50")),
                    str(metric_value(test, "precision")),
                    str(metric_value(test, "recall")),
                ]
            )
            + " |"
        )
    if not report.get("runs"):
        lines.append("| none |  |  |  |  |")
    if report.get("baseline") is not None:
        lines.extend(["", "## Baseline", "", f"`{json.dumps(report['baseline'], sort_keys=True)}`"])
    return "\n".join(lines) + "\n"


def write_evaluation_report(
    *,
    run_group: str,
    runs: list[dict[str, Any]],
    reports_dir: str | Path = WORKSPACE_DIR / "reports",
    baseline: dict[str, Any] | None = None,
    promoted_run_name: str | None,
    promotion_reason: str,
    generated_at: str | None = None,
) -> dict[str, Path]:
    output_dir = repo_resolve(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_group": run_group,
        "generated_at": generated_at or utcnow_iso_for_training(),
        "runs": runs,
        "baseline": baseline,
        "promotion": {
            "promoted": promoted_run_name is not None,
            "run_name": promoted_run_name,
            "reason": promotion_reason,
        },
    }
    json_path = output_dir / f"{run_group}_evaluation.json"
    markdown_path = output_dir / f"{run_group}_evaluation.md"
    write_json(json_path, report)
    markdown_path.write_text(evaluation_markdown(report))
    return {"json_path": json_path, "markdown_path": markdown_path}


def utcnow_iso_for_training() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
