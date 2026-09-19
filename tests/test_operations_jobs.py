import json
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from scripts import jobs, operations
from test_training_pipeline_ui import UiServer, get_json, post_json_capture


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from scripts import training_pipeline_ui as ui
    monkeypatch.setattr(ui, "TRAINING_UI_INTERACTION_LOG_PATH", tmp_path / "interactions.jsonl")
    monkeypatch.setenv("OPENDARKSTS_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(jobs, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(operations, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(operations, "PROJECT_PATH", tmp_path / "project.json")
    return tmp_path


def test_job_failure_persists_and_is_readable_from_cli(isolated):
    record = jobs.submit_job("run", {"images": [str(isolated / "missing.tiff")]})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = jobs.read_job(record["job_id"])
        if record["status"] in jobs.TERMINAL:
            break
        time.sleep(.05)
    assert record["status"] == "failed"
    assert "No readable" in record["error"]
    completed = subprocess.run([sys.executable, "-m", "scripts.application_cli", "jobs", record["job_id"]], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["result"]["status"] == "failed"


def test_duplicate_job_submission_is_refused(isolated, monkeypatch):
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **kw: type("Process", (), {"pid": 12345})())
    monkeypatch.setattr(jobs, "process_is_alive", lambda *args: True)
    options = {"start": "2024-01-01"}
    jobs.submit_job("ais", options)
    with pytest.raises(ValueError, match="already running"):
        jobs.submit_job("ais", options)


def test_job_options_reject_arbitrary_commands_and_outputs(isolated):
    with pytest.raises(ValueError):
        jobs.submit_job("shell", {"command": "echo test"})
    with pytest.raises(ValueError):
        jobs.submit_job("ais", {"start": "2024-01-01", "client_secret": "not-allowed"})


def test_operation_pages_and_project_api_share_settings(isolated):
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    with UiServer(TrainingPipelineHandler) as server:
        for path in ["/", "/detect", "/data", "/run", "/jobs", "/import"]:
            with urlopen(server.base_url + path) as response:
                html = response.read().decode()
            assert "app-navigation" in html
            assert 'href="/results"' not in html
        status, result = post_json_capture(server.base_url + "/api/operations/project", {"start": "2024-01-01", "end": "2024-01-02"})
        assert status == 200
        assert result["start"] == "2024-01-01"
        assert get_json(server.base_url + "/api/operations/project")["end"] == "2024-01-02"
        status, error = post_json_capture(server.base_url + "/api/operations/project", {"bbox": [2, 1, 1, 2]})
        assert status == 400


def test_project_cli_updates_same_record(isolated):
    completed = subprocess.run([sys.executable, "-m", "scripts.application_cli", "project", "--set", '{"radius": 1500}'], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["result"]["radius"] == 1500
    assert json.loads((isolated / "detection_project.json").read_text())["radius"] == 1500
