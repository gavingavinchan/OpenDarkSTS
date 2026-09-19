from __future__ import annotations

from pathlib import Path

from _helpers import load_training_lib


def test_build_yolo_train_command_from_config(tmp_path: Path, monkeypatch) -> None:
    training_lib = load_training_lib()
    monkeypatch.setattr(training_lib, 'ROOT_DIR', tmp_path)
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    runs_dir = tmp_path / "runs"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("names:\n  0: ship\n  1: ships\n")

    command = training_lib.build_yolo_train_command(
        {
            "model": "yolov8n.pt",
            "image_size": 960,
            "epochs": 10,
            "batch": 4,
            "device": "cpu",
            "seed": 42,
            "workers": 2,
            "patience": 5,
            "run_group": "fixture_group",
            "run_name": "fixture_run",
            "output_dir": str(runs_dir),
            "augmentation": {"degrees": 3.5, "fliplr": 0.25},
        },
        dataset_dir,
    )

    assert command == [
        "yolo",
        "detect",
        "train",
        "model=yolov8n.pt",
        f"data={dataset_dir / 'data.yaml'}",
        "imgsz=960",
        "epochs=10",
        "batch=4",
        "device=cpu",
        "seed=42",
        "project=" + str(runs_dir / "fixture_group"),
        "name=fixture_run",
        "patience=5",
        "workers=2",
        "degrees=3.5",
        "fliplr=0.25",
    ]


def test_build_yolo_train_command_adds_exist_ok_only_for_overwrite(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    dataset_dir = tmp_path / "datasets" / "fixture_dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("names:\n  0: ship\n  1: ships\n")
    config = {
        "model": "yolov8n.pt",
        "image_size": 960,
        "epochs": 10,
        "batch": 4,
        "device": "cpu",
        "seed": 42,
        "run_name": "fixture_run",
        "augmentation": {},
        "overwrite": True,
    }

    command = training_lib.build_yolo_train_command(config, dataset_dir)

    assert "exist_ok=True" in command
