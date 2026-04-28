from __future__ import annotations

import hashlib
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
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
