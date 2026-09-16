"""Parity guard for the shared export helpers.

`exportable_detections`, `write_yolo_label_file`, and `write_data_yaml` used to exist as
diverged copies in `visual_review_lib` and `training_pipeline_lib`. They now have a single
canonical implementation in `visual_review_lib`. The fixtures below are real completed
review JSONs plus the label files that the pre-consolidation export path actually wrote
into `workspace/datasets/`, so this test fails if the shared helpers ever drift again.
"""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import load_review_lib, load_training_lib, write_image, write_json

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "export_parity"
REVIEWS_FIXTURE_DIR = FIXTURE_DIR / "reviews"
IMAGE_KEYS = ("2020-01-24_scene1_tile_0_3", "2020-01-24_scene1_tile_1_3")


def stage_reviews(tmp_path: Path) -> Path:
    """Copy the fixture reviews into tmp_path and point them at generated cache images."""
    reviews_dir = tmp_path / "reviews" / "gavin"
    cache_dir = tmp_path / "cache"
    for image_key in IMAGE_KEYS:
        review = json.loads((REVIEWS_FIXTURE_DIR / f"{image_key}.json").read_text())
        image_path = cache_dir / review["image_filename"]
        write_image(image_path)
        review["cache_path"] = str(image_path)
        write_json(reviews_dir / f"{image_key}.json", review)
    return tmp_path / "reviews"


def export_with_items(tmp_path: Path, items: list[dict], output_name: str) -> Path:
    training_lib = load_training_lib()
    reviews_dir = stage_reviews(tmp_path)
    output_dir = tmp_path / "datasets" / output_name
    selection_path = tmp_path / "dataset_selections" / f"{output_name}.json"
    write_json(
        selection_path,
        {
            "schema_version": 1,
            "dataset_name": output_name,
            "dataset_version": "v1",
            "sources": {"reviews_dir": str(reviews_dir)},
            "items": items,
            "background_negatives": {"enabled": False},
            "export": {"output_dir": str(output_dir), "include_incomplete_reviews": False},
        },
    )
    training_lib.export_dataset_selection(selection_path)
    return output_dir


def assert_labels_match(output_dir: Path, expected_dir: Path, items: list[dict]) -> None:
    for item in items:
        filename = f"{item['image_key']}.txt"
        actual = (output_dir / "labels" / item["split"] / filename).read_text()
        expected = (expected_dir / filename).read_text()
        assert actual == expected, f"label drift for {filename}"


def test_export_all_exportable_labels_match_pre_consolidation_output(tmp_path: Path) -> None:
    items = [
        {"image_key": "2020-01-24_scene1_tile_0_3", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        {"image_key": "2020-01-24_scene1_tile_1_3", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
    ]

    output_dir = export_with_items(tmp_path, items, "all_exportable_fixture")

    assert_labels_match(output_dir, FIXTURE_DIR / "expected_labels_all_exportable", items)


def test_export_selected_only_labels_match_pre_consolidation_output(tmp_path: Path) -> None:
    items = [
        {"image_key": "2020-01-24_scene1_tile_0_3", "split": "val", "include_labels": "selected_only", "selected_detection_ids": ["auto-0001", "auto-0002"]},
        {"image_key": "2020-01-24_scene1_tile_1_3", "split": "train", "include_labels": "selected_only", "selected_detection_ids": ["auto-0004"]},
    ]

    output_dir = export_with_items(tmp_path, items, "selected_only_fixture")

    assert_labels_match(output_dir, FIXTURE_DIR / "expected_labels_selected_only", items)


def test_shared_helpers_have_one_canonical_implementation() -> None:
    training_lib = load_training_lib()
    review_lib = load_review_lib()

    assert training_lib.exportable_detections is review_lib.exportable_detections
    assert training_lib.write_yolo_label_file is review_lib.write_yolo_label_file


def test_selected_only_requires_selected_detection_ids() -> None:
    training_lib = load_training_lib()
    review = {"detections": [{"detection_id": "auto-0001", "review_label": "ship_non_sts"}]}

    try:
        training_lib.exportable_detections(review, {"image_key": "img_x", "include_labels": "selected_only"})
    except ValueError as exc:
        assert "selected_detection_ids" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("selected_only without ids must raise")


def test_data_yaml_layouts_stay_distinct(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    review_lib = load_review_lib()
    split_dir = tmp_path / "split"
    flat_dir = tmp_path / "flat"
    split_dir.mkdir()
    flat_dir.mkdir()

    training_lib.write_data_yaml(split_dir)
    review_lib.write_data_yaml(flat_dir)

    split_yaml = (split_dir / "data.yaml").read_text()
    flat_yaml = (flat_dir / "data.yaml").read_text()
    assert "train: images/train" in split_yaml and "test: images/test" in split_yaml
    assert "train: images\n" in flat_yaml and "test:" not in flat_yaml
    assert "0: ship" in split_yaml and "1: ships" in split_yaml
    assert "0: ship" in flat_yaml and "1: ships" in flat_yaml
