from __future__ import annotations

import json
from pathlib import Path

from _helpers import detection, load_training_lib, read_json, write_review, write_selection, write_training_config


def test_write_training_metadata_records_reproducibility_inputs(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    run_dir = tmp_path / "runs" / "fixture_group" / "fixture_run"

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

    metadata_path = training_lib.write_training_metadata(
        run_dir=run_dir,
        training_config_path=config_path,
        dataset_manifest_path=dataset_dir / "dataset_manifest.json",
        command_line=["yolo", "detect", "train", "data=data.yaml"],
        status="completed",
        started_at="2026-04-28T00:00:00Z",
        ended_at="2026-04-28T00:01:00Z",
    )

    assert metadata_path == run_dir / "training_metadata.json"
    metadata = read_json(metadata_path)
    training_config = json.loads(config_path.read_text())
    dataset_manifest = read_json(dataset_dir / "dataset_manifest.json")

    assert metadata["schema_version"] == 1
    assert metadata["status"] == "completed"
    assert metadata["training_config"] == training_config
    assert metadata["training_config_path"] == str(config_path)
    assert metadata["dataset_manifest_path"] == str(dataset_dir / "dataset_manifest.json")
    assert dataset_manifest["schema_version"] == 1
    assert metadata["dataset_manifest_sha256"].startswith("sha256:")
    assert metadata["data_yaml_path"] == str(dataset_dir / "data.yaml")
    assert metadata["data_yaml_sha256"].startswith("sha256:")
    assert metadata["command_line"] == ["yolo", "detect", "train", "data=data.yaml"]
    assert metadata["started_at"] == "2026-04-28T00:00:00Z"
    assert metadata["ended_at"] == "2026-04-28T00:01:00Z"
    assert metadata["output_run_dir"] == str(run_dir)
    assert "python" in metadata["package_versions"]
    assert "git" in metadata
