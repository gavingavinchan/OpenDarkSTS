from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from PIL import Image


def load_training_lib():
    try:
        return importlib.import_module("scripts.training_pipeline_lib")
    except ModuleNotFoundError as exc:  # pragma: no cover - this is the first TDD red state.
        pytest.fail("scripts.training_pipeline_lib must provide training-pipeline export helpers")


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


def test_export_dataset_selection_writes_split_yolo_dataset_and_manifest(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"

    write_review(
        reviews_dir,
        cache_dir,
        reviewer="gavin",
        image_key="img_train",
        detections=[detection("auto-0001", "ship_non_sts"), detection("auto-0002", "neither")],
    )
    write_review(
        reviews_dir,
        cache_dir,
        reviewer="gavin",
        image_key="img_val",
        detections=[detection("auto-0001", "ships_sts")],
    )
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[
            {"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_val", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )

    training_lib.export_dataset_selection(selection_path)

    assert (output_dir / "images" / "train" / "img_train.png").is_file()
    assert (output_dir / "images" / "val" / "img_val.png").is_file()
    assert (output_dir / "images" / "test").is_dir()
    assert (output_dir / "labels" / "train" / "img_train.txt").read_text() == "0 0.50000000 0.50000000 0.25000000 0.20000000\n"
    assert (output_dir / "labels" / "val" / "img_val.txt").read_text() == "1 0.50000000 0.50000000 0.25000000 0.20000000\n"

    data_yaml = (output_dir / "data.yaml").read_text()
    assert "train: images/train" in data_yaml
    assert "val: images/val" in data_yaml
    assert "test: images/test" in data_yaml
    assert "0: ship" in data_yaml
    assert "1: ships" in data_yaml

    manifest = read_json(output_dir / "dataset_manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["dataset_selection_manifest"] == str(selection_path)
    exported_paths = {entry["path"] for entry in manifest["exported_files"]}
    assert "images/train/img_train.png" in exported_paths
    assert "labels/train/img_train.txt" in exported_paths
    assert "images/val/img_val.png" in exported_paths
    assert "labels/val/img_val.txt" in exported_paths


def test_export_dataset_selection_blocks_reconciliation_conflicts(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"

    write_review(reviews_dir, cache_dir, reviewer="alice", image_key="img_conflict", detections=[detection("auto-0001", "ship_non_sts")])
    write_review(reviews_dir, cache_dir, reviewer="bob", image_key="img_conflict", detections=[detection("auto-0001", "ship_non_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[{"image_key": "img_conflict", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
    )

    with pytest.raises(ValueError, match="(?i)reconciliation|conflict"):
        training_lib.export_dataset_selection(selection_path)


def test_export_dataset_selection_caps_background_negatives_and_warns(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"

    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_positive", detections=[detection("auto-0001", "ship_non_sts")])
    for index in range(3):
        write_review(reviews_dir, cache_dir, reviewer="gavin", image_key=f"img_bg_{index}", detections=[detection("auto-0001", "neither")])

    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[{"image_key": "img_positive", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
        background_negatives={
            "enabled": True,
            "mode": "whole_image_empty_label",
            "max_fraction": 0.5,
            "image_keys": ["img_bg_0", "img_bg_1", "img_bg_2"],
        },
    )

    training_lib.export_dataset_selection(selection_path)

    background_labels = sorted((output_dir / "labels" / "train").glob("img_bg_*.txt"))
    assert len(background_labels) == 1
    assert background_labels[0].read_text() == ""

    manifest = read_json(output_dir / "dataset_manifest.json")
    assert any("background" in warning.lower() and "max_fraction" in warning for warning in manifest["warnings"])
