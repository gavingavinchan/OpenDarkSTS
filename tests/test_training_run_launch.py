from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from _helpers import (
    detection,
    load_training_lib,
    write_json,
    write_review,
    write_selection,
    write_training_config,
)


def build_ready_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (selection_path, config_path, run_dir) for a launchable fixture run."""
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
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=tmp_path / "runs", overwrite=True)
    run_dir = tmp_path / "runs" / "fixture_group" / "fixture_run"
    return selection_path, config_path, run_dir


def fake_yolo(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return path


def wait_until_dead(training_lib, pid: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not training_lib.process_is_alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"pid {pid} did not exit within {timeout}s")


def test_start_training_run_refuses_when_a_process_is_still_alive(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    selection_path, config_path, run_dir = build_ready_run(tmp_path)
    sleeper = fake_yolo(tmp_path / "fake_yolo_sleep", "exec sleep 60")

    first = training_lib.start_training_run(selection_path, config_path, yolo_executable=sleeper)
    try:
        with pytest.raises(training_lib.TrainingRunActiveError) as excinfo:
            training_lib.start_training_run(selection_path, config_path, yolo_executable=sleeper)
        assert str(first["pid"]) in str(excinfo.value)
        assert excinfo.value.pid == first["pid"]
        # The original process record must survive the refused launch.
        record = training_lib.load_json(run_dir / training_lib.PROCESS_RECORD_FILENAME)
        assert record["pid"] == first["pid"]
    finally:
        training_lib.stop_training_run(run_dir, grace_seconds=2.0)
    assert not training_lib.process_is_alive(first["pid"])


def test_start_training_run_allows_relaunch_after_the_previous_process_exits(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    selection_path, config_path, run_dir = build_ready_run(tmp_path)
    quick = fake_yolo(tmp_path / "fake_yolo_quick", "exit 1")

    first = training_lib.start_training_run(selection_path, config_path, yolo_executable=quick)
    wait_until_dead(training_lib, first["pid"])
    second = training_lib.start_training_run(selection_path, config_path, yolo_executable=quick)
    wait_until_dead(training_lib, second["pid"])
    assert second["launch_id"] != first["launch_id"]


def test_failed_relaunch_is_not_reported_completed_from_the_previous_launch(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    selection_path, config_path, run_dir = build_ready_run(tmp_path)
    # Pretend a previous launch completed here: completion marker, results, test metrics, weights.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / training_lib.TRAINING_LOG_FILENAME).write_text(
        "Epoch 2/2\n2 epochs completed in 0.001 hours.\nResults saved to runs/fixture_group/fixture_run\n"
    )
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.5,0.50,0.30\n2,0.9,0.9,0.90,0.60\n"
    )
    write_json(run_dir / training_lib.TEST_METRICS_FILENAME, {"metrics": {"mAP50": 0.9}})
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"old-best")

    failing = fake_yolo(tmp_path / "fake_yolo_fail", "exit 1")
    launch = training_lib.start_training_run(selection_path, config_path, yolo_executable=failing)
    wait_until_dead(training_lib, launch["pid"])

    metadata = training_lib.load_json(run_dir / training_lib.TRAINING_METADATA_FILENAME)
    assert metadata["status"] == "running"  # reset for this launch, not the previous one

    status = training_lib.training_status(run_dir)
    assert status["status"] == "failed"
    assert status["reason"] == "process_missing"
    # The previous launch's artifacts were rotated aside, not inherited.
    assert "epochs completed in" not in "\n".join(status["log_tail"])
    assert not (run_dir / training_lib.TEST_METRICS_FILENAME).is_file()
    assert not (run_dir / "results.csv").is_file()
    assert launch["rotated_previous_artifacts"]
    assert training_lib.run_evaluation_summary(run_dir)["test_metrics"] is None


def test_cli_start_training_exits_3_when_the_run_is_already_active(tmp_path: Path) -> None:
    import importlib

    cli = importlib.import_module("scripts.training_pipeline_cli")
    training_lib = load_training_lib()
    selection_path, config_path, run_dir = build_ready_run(tmp_path)
    sleeper = fake_yolo(tmp_path / "fake_yolo_sleep", "exec sleep 60")
    first = training_lib.start_training_run(selection_path, config_path, yolo_executable=sleeper)
    try:
        code = cli.main([
            "start-training",
            "--selection", str(selection_path),
            "--config", str(config_path),
            "--yolo-executable", str(sleeper),
        ])
        assert code == 3
    finally:
        training_lib.stop_training_run(run_dir, grace_seconds=2.0)
