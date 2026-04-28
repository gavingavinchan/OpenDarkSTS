from __future__ import annotations

from pathlib import Path

from _helpers import load_training_lib


def test_load_training_results_series_waits_when_results_csv_missing(tmp_path: Path) -> None:
    training_lib = load_training_lib()

    assert training_lib.load_training_results_series(tmp_path / "results.csv") == {
        "status": "waiting",
        "message": "Waiting for first epoch",
        "series": {},
    }


def test_load_training_results_series_returns_recognized_columns(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),unknown/new_metric\n"
        "1,0.9,0.8,0.4,123\n"
        "2,0.7,0.6,0.5,456\n"
    )

    result = training_lib.load_training_results_series(results_csv)

    assert result == {
        "status": "ready",
        "message": None,
        "series": {
            "train/box_loss": [[1, 0.9], [2, 0.7]],
            "train/cls_loss": [[1, 0.8], [2, 0.6]],
            "metrics/mAP50(B)": [[1, 0.4], [2, 0.5]],
        },
    }


def test_load_training_results_series_skips_blank_or_non_numeric_values(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    results_csv = tmp_path / "results.csv"
    results_csv.write_text("epoch,val/box_loss,metrics/recall(B)\n1,,bad\n2,0.3,0.5\n")

    result = training_lib.load_training_results_series(results_csv)

    assert result["series"] == {
        "val/box_loss": [[2, 0.3]],
        "metrics/recall(B)": [[2, 0.5]],
    }
