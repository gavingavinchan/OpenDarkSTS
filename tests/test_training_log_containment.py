from pathlib import Path

import pytest

from scripts import training_pipeline_lib as pipeline


@pytest.mark.parametrize("kind", ["absolute", "relative", "symlink"])
def test_status_rejects_outside_recorded_log(tmp_path: Path, kind):
    run = tmp_path / "group" / "run"
    run.mkdir(parents=True)
    outside = tmp_path / "outside.log"
    outside.write_text("secret outside log\nResults saved to outside\n")
    log_path = str(outside)
    if kind == "relative":
        log_path = pipeline.repo_relative(outside)
    elif kind == "symlink":
        linked = run / "linked.log"
        linked.symlink_to(outside)
        log_path = str(linked)
    pipeline.write_json(run / "process.json", {"status": "running", "pid": 0, "log_path": log_path})
    (run / "train.log").write_text("local log\n")
    status = pipeline.training_status(run)
    assert status["log_tail"] == ["local log"]
    assert status["status"] == "failed"
    assert "process log_path is outside" in status["warnings"][0]
    assert pipeline.load_json(run / "process.json")["status"] == "failed"


def test_status_does_not_follow_default_log_symlink(tmp_path: Path):
    run = tmp_path / "run"
    pipeline.write_json(run / "process.json", {"status": "running", "pid": 0})
    outside = tmp_path / "outside.log"
    outside.write_text("Results saved to secret\n")
    (run / "train.log").symlink_to(outside)
    status = pipeline.training_status(run)
    assert status["log_tail"] == []
    assert status["log_path"] is None
    assert status["status"] == "failed"
    assert "log unavailable" in status["warnings"][0]


def test_status_reads_contained_recorded_log(tmp_path: Path):
    run = tmp_path / "run"
    alternate = run / "logs" / "training.log"
    alternate.parent.mkdir(parents=True)
    alternate.write_text("first\nsecond\n")
    pipeline.write_json(run / "process.json", {"status": "completed", "log_path": pipeline.repo_relative(alternate)})
    status = pipeline.training_status(run, log_lines=1)
    assert status["log_tail"] == ["second"]
    assert status["warnings"] == []


def test_test_split_rejects_outside_log_before_launch(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"fixture")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: [ship, ships]\n")
    outside = tmp_path / "outside.log"
    outside.write_text("unchanged\n")
    (run / "test_eval.log").symlink_to(outside)
    calls = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="test_eval.log must be inside"):
        pipeline.run_test_split(run, dataset_dir=dataset)
    assert calls == []
    assert outside.read_text() == "unchanged\n"
