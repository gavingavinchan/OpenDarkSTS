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
        "promoted": False,
        "run_name": None,
        "reason": "No current promotion from this run group",
    }

    markdown = md_path.read_text()
    assert "# Evaluation: fixture_group" in markdown
    assert "Recommended/selected: yolo26n_fixture" in markdown
    assert "Promoted: none" in markdown
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
        "reason": "No current promotion from this run group",
    }
    assert "Promoted: none" in result["markdown_path"].read_text()


def write_results_csv(run_dir: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)"]
    for epoch, precision, recall, map50, map5095 in rows:
        lines.append(f"{epoch},{precision},{recall},{map50},{map5095}")
    (run_dir / "results.csv").write_text("\n".join(lines) + "\n")


def test_run_evaluation_summary_reports_the_best_epoch_not_the_last(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    run_dir = tmp_path / "runs" / "group" / "peaks_early"
    # A run that peaks at epoch 2 and collapses at epoch 3; best.pt is the epoch 2 checkpoint.
    write_results_csv(run_dir, [(1, 0.4, 0.4, 0.40, 0.20), (2, 0.9, 0.9, 0.90, 0.60), (3, 0.1, 0.1, 0.10, 0.05)])

    summary = training_lib.run_evaluation_summary(run_dir)

    assert summary["validation_metrics"]["mAP50"] == 0.90
    assert summary["validation_metrics"]["mAP50-95"] == 0.60
    assert summary["best_validation_epoch"] == 2
    assert summary["final_validation_metrics"]["mAP50"] == 0.10


def test_best_run_by_metric_ranks_on_the_best_checkpoint(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    group_dir = tmp_path / "runs" / "group"
    write_results_csv(group_dir / "peaks_early", [(1, 0.9, 0.9, 0.90, 0.60), (2, 0.1, 0.1, 0.10, 0.05)])
    write_results_csv(group_dir / "steady", [(1, 0.5, 0.5, 0.50, 0.30), (2, 0.5, 0.5, 0.50, 0.30)])

    runs = training_lib.collect_run_group_summaries("group", tmp_path / "runs")
    best_name, reason = training_lib.best_run_by_metric(runs)

    assert best_name == "peaks_early"
    assert "0.9000" in reason


def test_evaluation_markdown_shows_best_and_final_validation(tmp_path: Path) -> None:
    training_lib = load_training_lib()
    paths = training_lib.write_evaluation_report(
        run_group="fixture_group",
        runs=[{
            "run_name": "peaks_early",
            "best_validation_epoch": 2,
            "validation_metrics": {"mAP50": 0.9},
            "final_validation_metrics": {"mAP50": 0.1},
            "test_metrics": {},
        }],
        reports_dir=tmp_path / "reports",
        promoted_run_name="peaks_early",
        promotion_reason="fixture",
        generated_at="2026-04-28T00:00:00Z",
    )

    markdown = paths["markdown_path"].read_text()
    assert "Best epoch" in markdown and "Final val mAP50" in markdown
    assert "| peaks_early | 2 | 0.9 | 0.1 |" in markdown
