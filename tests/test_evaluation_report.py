from __future__ import annotations

from pathlib import Path

from _helpers import load_training_lib, read_json


def test_write_evaluation_report_outputs_json_and_markdown(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    reports_dir = tmp_path / "reports"
    runs = [
        {
            "run_name": "yolov8n_fixture",
            "weights_path": "workspace/runs/fixture/yolov8n/weights/best.pt",
            "validation_metrics": {"precision": 0.82, "recall": 0.71, "mAP50": 0.78, "mAP50-95": 0.52},
            "test_metrics": {"precision": 0.8, "recall": 0.7, "mAP50": 0.76, "mAP50-95": 0.5},
            "class_metrics": {"ship": {"mAP50": 0.8}, "ships": {"mAP50": 0.72}},
        },
        {
            "run_name": "yolo26n_fixture",
            "weights_path": "workspace/runs/fixture/yolo26n/weights/best.pt",
            "validation_metrics": {"precision": 0.86, "recall": 0.75, "mAP50": 0.83, "mAP50-95": 0.58},
            "test_metrics": {"precision": 0.84, "recall": 0.73, "mAP50": 0.81, "mAP50-95": 0.55},
            "class_metrics": {"ship": {"mAP50": 0.84}, "ships": {"mAP50": 0.78}},
        },
    ]
    baseline = {"name": "current_best", "weights_path": "workspace/model_weights/current.pt", "test_metrics": {"mAP50": 0.79}}

    result = training_lib.write_evaluation_report(
        run_group="fixture_group",
        runs=runs,
        reports_dir=reports_dir,
        baseline=baseline,
        promoted_run_name="yolo26n_fixture",
        promotion_reason="Highest test mAP50 with better STS-class performance.",
        generated_at="2026-04-28T00:00:00Z",
    )

    json_path = reports_dir / "fixture_group_evaluation.json"
    md_path = reports_dir / "fixture_group_evaluation.md"
    assert result == {"json_path": json_path, "markdown_path": md_path}

    report = read_json(json_path)
    assert report["schema_version"] == 1
    assert report["run_group"] == "fixture_group"
    assert report["runs"] == runs
    assert report["baseline"] == baseline
    assert report["promotion"] == {
        "promoted": True,
        "run_name": "yolo26n_fixture",
        "reason": "Highest test mAP50 with better STS-class performance.",
    }

    markdown = md_path.read_text()
    assert "# Evaluation: fixture_group" in markdown
    assert "Promoted: yolo26n_fixture" in markdown
    assert "Highest test mAP50" in markdown
    assert "yolov8n_fixture" in markdown
    assert "yolo26n_fixture" in markdown
    assert "0.81" in markdown


def test_write_evaluation_report_records_non_promotion_reason(tmp_path: Path) -> None:
    training_lib = load_training_lib()

    result = training_lib.write_evaluation_report(
        run_group="fixture_group",
        runs=[],
        reports_dir=tmp_path / "reports",
        baseline=None,
        promoted_run_name=None,
        promotion_reason="No completed runs were available for evaluation.",
        generated_at="2026-04-28T00:00:00Z",
    )

    report = read_json(result["json_path"])
    assert report["promotion"] == {
        "promoted": False,
        "run_name": None,
        "reason": "No completed runs were available for evaluation.",
    }
    assert "Promoted: none" in result["markdown_path"].read_text()
