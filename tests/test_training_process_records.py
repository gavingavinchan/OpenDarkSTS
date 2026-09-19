from pathlib import Path

import pytest

from scripts import training_pipeline_lib as pipeline


@pytest.mark.parametrize("pid", [0, -1, -1234, "0", "-1", "invalid", None, {}, 10**100])
def test_invalid_pid_is_never_signalled(pid, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.os, "kill", lambda *args: calls.append(args))
    assert pipeline.process_is_alive(pid) is False
    assert calls == []


@pytest.mark.parametrize("payload", ['{"status": "runn', '[]', 'null'])
@pytest.mark.parametrize("completed", [False, True])
def test_corrupt_process_status_uses_artifacts_without_rewriting(tmp_path: Path, payload, completed):
    run = tmp_path / "runs" / "group" / "run"
    run.mkdir(parents=True)
    process = run / "process.json"
    process.write_text(payload)
    if completed:
        (run / "train.log").write_text("1 epochs completed in 0.01 hours\n")
    status = pipeline.training_status(run)
    assert status["status"] == ("completed" if completed else "failed")
    assert status["alive"] is False
    assert status["terminal"] is True
    assert "process.json could not be parsed" in status["warnings"][0]
    assert process.read_text() == payload
    assert pipeline.list_training_runs(tmp_path / "runs")[0]["status"] == status["status"]


@pytest.mark.parametrize("payload", ['{"status":', '[]'])
def test_corrupt_process_stop_refuses_to_signal(tmp_path: Path, monkeypatch, payload):
    run = tmp_path / "run"
    run.mkdir()
    (run / "process.json").write_text(payload)
    calls = []
    monkeypatch.setattr(pipeline.os, "kill", lambda *args: calls.append(args))
    monkeypatch.setattr(pipeline.os, "killpg", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="no process can safely be stopped"):
        pipeline.stop_training_run(run)
    assert calls == []
    assert (run / "process.json").read_text() == payload


@pytest.mark.parametrize("pid", [0, -1, "bad"])
def test_stop_invalid_pid_does_not_signal_group(tmp_path: Path, monkeypatch, pid):
    run = tmp_path / "run"
    pipeline.write_json(run / "process.json", {"status": "running", "pid": pid})
    calls = []
    monkeypatch.setattr(pipeline.os, "kill", lambda *args: calls.append(args))
    monkeypatch.setattr(pipeline.os, "killpg", lambda *args: calls.append(args))
    assert pipeline.stop_training_run(run)["stopped"] is False
    assert calls == []


def test_invalid_start_ticks_does_not_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline.os, "kill", lambda *args: calls.append(args))
    assert pipeline.process_is_alive(123, "bad") is False
    assert calls == []
