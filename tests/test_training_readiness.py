from __future__ import annotations

from pathlib import Path

from _helpers import detection, load_training_lib, write_review, write_selection, write_training_config


def missing_text(result: dict) -> str:
    return "\n".join(result["missing"])


def test_training_readiness_reports_missing_export_and_config_fields(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )

    result = training_lib.check_training_readiness(selection_path, {})

    assert result["ready"] is False
    missing = missing_text(result)
    assert "exported dataset" in missing
    assert "data.yaml" in missing
    assert "images/train" in missing
    assert "images/val" in missing
    assert "dataset name/version" in missing
    assert "model or starting weights" in missing
    assert "output run name" in missing


def test_training_readiness_passes_for_exported_dataset_and_valid_config(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"

    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_val", detections=[detection("auto-0001", "ships_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=dataset_dir,
        items=[
            {"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_val", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )
    training_lib.export_dataset_selection(selection_path)
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=tmp_path / "runs")

    result = training_lib.check_training_readiness(selection_path, config_path)

    assert result["ready"] is True
    assert result["missing"] == []
    assert result["warnings"] == [
        "test split has no images; final test metrics will be unavailable",
        "Class names may be singular/plural duplicates: 'ship' and 'ships'; allowed as intentional project classes",
    ]


def test_training_readiness_blocks_existing_run_without_overwrite(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    runs_dir = tmp_path / "runs"

    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_val", detections=[detection("auto-0001", "ships_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=dataset_dir,
        items=[
            {"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_val", "split": "val", "include_labels": "all_exportable", "selected_detection_ids": None},
        ],
    )
    training_lib.export_dataset_selection(selection_path)
    (runs_dir / "fixture_group" / "fixture_run").mkdir(parents=True)
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=runs_dir)

    result = training_lib.check_training_readiness(selection_path, config_path)

    assert result["ready"] is False
    assert any("overwrite" in item for item in result["missing"])

    overwrite_config_path = write_training_config(tmp_path / "training_configs" / "fixture_group_overwrite.json", output_dir=runs_dir, overwrite=True)
    assert training_lib.check_training_readiness(selection_path, overwrite_config_path)["ready"] is True
