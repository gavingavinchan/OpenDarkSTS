from __future__ import annotations

from pathlib import Path

import pytest
from _helpers import detection, load_training_lib, read_json, write_review, write_selection


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


def test_export_dataset_selection_selected_only_filters_detection_ids(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"

    write_review(
        reviews_dir,
        cache_dir,
        reviewer="gavin",
        image_key="img_train",
        detections=[
            detection("auto-0001", "ship_non_sts"),
            detection("auto-0002", "ships_sts", x_center=0.25),
        ],
    )
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[
            {
                "image_key": "img_train",
                "split": "train",
                "include_labels": "selected_only",
                "selected_detection_ids": ["auto-0002"],
            }
        ],
    )

    training_lib.export_dataset_selection(selection_path)

    assert (output_dir / "labels" / "train" / "img_train.txt").read_text() == "1 0.25000000 0.50000000 0.25000000 0.20000000\n"


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
