from pathlib import Path

import pytest

from scripts import training_pipeline_lib as pipeline
from scripts import training_pipeline_cli as cli


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "WORKSPACE_DIR", tmp_path)
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(pipeline, "DEFAULT_RUNS_DIR", root)
    return root


@pytest.mark.parametrize("entry", [
    pipeline.training_status, pipeline.stop_training_run, pipeline.active_run_process,
    pipeline.run_evaluation_summary, pipeline.run_test_split, pipeline.list_training_runs,
    pipeline.list_run_groups,
])
@pytest.mark.parametrize("escape", ["absolute", "parent", "symlink"])
def test_run_entry_points_reject_escaped_paths(runs, entry, escape):
    outside = runs.parent / "outside"
    outside.mkdir()
    target = outside
    if escape == "parent":
        target = runs / ".." / "outside"
    elif escape == "symlink":
        target = runs / "linked"
        target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="run_dir must be inside"):
        entry(target)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("group,name", [("../outside", "run"), ("group", "../../outside"), ("group", "/outside"), ("..\\outside", "run")])
def test_run_references_reject_path_segments(runs, group, name):
    with pytest.raises(ValueError, match="single path segment"):
        pipeline.resolve_training_run_ref(runs, group, name)
    with pytest.raises(ValueError, match="single path segment"):
        pipeline.run_dir_for_config({"output_dir": str(runs), "run_group": group, "run_name": name})
    with pytest.raises(ValueError, match="single path segment"):
        pipeline.promote_weight(run_group=group, run_name=name, target_name="test", recreation_doc_updated=True, runs_dir=runs)


def test_evaluation_and_weight_listing_reject_outside_roots(runs):
    outside = runs.parent / "outside"
    with pytest.raises(ValueError, match="run_dir must be inside"):
        pipeline.collect_run_group_summaries("group", outside)
    with pytest.raises(ValueError, match="run_dir must be inside"):
        pipeline.list_starting_weights(root_dir=outside, model_weights_dir=outside, runs_dir=outside)
    with pytest.raises(ValueError, match="single path segment"):
        pipeline.build_run_group_evaluation_report(run_group="../outside", runs_dir=runs)


def test_run_listings_skip_escaped_symlinks(runs):
    outside = runs.parent / "outside"
    outside.mkdir()
    pipeline.write_json(outside / "process.json", {"status": "running", "pid": 0})
    (outside / "results.csv").write_text("epoch\n1\n")
    group = runs / "group"
    group.mkdir()
    (group / "escape").symlink_to(outside, target_is_directory=True)
    safe = group / "safe"
    safe.mkdir()
    pipeline.write_json(safe / "process.json", {"status": "completed", "pid": None})
    (safe / "results.csv").write_text("epoch\n1\n")
    assert [item["run_name"] for item in pipeline.list_training_runs(runs)] == ["safe"]
    assert pipeline.list_run_groups(runs)[0]["run_names"] == ["safe"]
    assert [item["run_name"] for item in pipeline.collect_run_group_summaries("group", runs)] == ["safe"]
    assert pipeline.load_json(outside / "process.json")["status"] == "running"


def test_cli_run_reference_rejects_traversal(runs):
    args = cli.build_parser().parse_args(["training-status", "--runs-dir", str(runs), "--run-group", "../outside", "--run-name", "run"])
    with pytest.raises(ValueError, match="single path segment"):
        cli.resolved_run_dir(args)


def test_hand_edited_launch_config_rejected_before_artifacts(runs):
    config_path = runs.parent / "config.json"
    pipeline.write_json(config_path, {"output_dir": str(runs.parent / "outside"), "run_group": "group", "run_name": "run"})
    with pytest.raises(ValueError, match="run_dir must be inside"):
        pipeline.start_training_run(runs.parent / "missing_selection.json", config_path)
    assert not (runs.parent / "outside").exists()


def test_evaluation_report_rejects_group_path_before_writing(runs):
    reports = runs.parent / "reports"
    with pytest.raises(ValueError, match="single path segment"):
        pipeline.write_evaluation_report(run_group="../escape", runs=[], reports_dir=reports, promoted_run_name=None, promotion_reason="test")
    assert not reports.exists()
