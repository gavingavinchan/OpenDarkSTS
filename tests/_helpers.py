from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from PIL import Image


def load_training_lib():
    try:
        return importlib.import_module("scripts.training_pipeline_lib")
    except ModuleNotFoundError:  # pragma: no cover - this is the first TDD red state.
        pytest.fail("scripts.training_pipeline_lib must provide training-pipeline export helpers")


def load_review_lib():
    try:
        return importlib.import_module("scripts.visual_review_lib")
    except ModuleNotFoundError:
        pytest.fail("scripts.visual_review_lib must provide visual review helpers")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_image(path: Path, color: tuple[int, int, int] = (20, 80, 140)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=color).save(path)


def detection(
    detection_id: str,
    review_label: str,
    *,
    x_center: float = 0.5,
    y_center: float = 0.5,
    width: float = 0.25,
    height: float = 0.2,
) -> dict:
    class_by_label = {"ship_non_sts": (0, "ship"), "ships_sts": (1, "ships"), "neither": (0, "ship")}
    class_id, class_name = class_by_label[review_label]
    return {
        "detection_id": detection_id,
        "source_type": "auto",
        "parent_detection_id": None,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": 0.9,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "review_label": review_label,
        "reviewed": True,
        "reviewed_at": "2026-04-28T00:00:00Z",
        "source_models": ["fixture"],
    }


def write_review(
    reviews_dir: Path,
    cache_dir: Path,
    *,
    reviewer: str,
    image_key: str,
    image_filename: str | None = None,
    detections: list[dict] | None = None,
    review_status: str = "completed",
) -> Path:
    filename = image_filename or f"{image_key}.png"
    image_path = cache_dir / filename
    write_image(image_path)
    review_path = reviews_dir / reviewer / f"{image_key}.json"
    write_json(
        review_path,
        {
            "schema_version": 1,
            "image_key": image_key,
            "image_filename": filename,
            "reviewer": reviewer,
            "preprocess_fingerprint": f"sha256:{image_key}",
            "width": 32,
            "height": 24,
            "cache_path": str(image_path),
            "review_status": review_status,
            "detail_cursor": None,
            "updated_at": "2026-04-28T00:00:00Z",
            "detections": detections or [],
        },
    )
    return review_path


def write_selection(
    path: Path,
    *,
    reviews_dir: Path,
    output_dir: Path,
    items: list[dict],
    background_negatives: dict | None = None,
) -> Path:
    write_json(
        path,
        {
            "schema_version": 1,
            "dataset_name": "fixture_dataset",
            "dataset_version": 1,
            "created_at": "2026-04-28T00:00:00Z",
            "updated_at": "2026-04-28T00:00:00Z",
            "created_by": "pytest",
            "selection_mode": "image",
            "classes": {"0": "ship", "1": "ships"},
            "sources": {"reviews_dir": str(reviews_dir), "manifest": None},
            "split_policy": {
                "mode": "auto_with_manual_override",
                "default_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
                "seed": 42,
            },
            "items": items,
            "background_negatives": background_negatives or {"enabled": False},
            "export": {"output_dir": str(output_dir), "include_incomplete_reviews": False},
        },
    )
    return path


def write_training_config(path: Path, *, output_dir: Path, overwrite: bool = False) -> Path:
    write_json(
        path,
        {
            "schema_version": 1,
            "dataset_name": "fixture_dataset",
            "dataset_version": 1,
            "model": "yolov8n.pt",
            "image_size": 960,
            "epochs": 2,
            "batch": 4,
            "device": "cpu",
            "seed": 42,
            "run_group": "fixture_group",
            "run_name": "fixture_run",
            "output_dir": str(output_dir),
            "overwrite": overwrite,
            "augmentation": {},
        },
    )
    return path
