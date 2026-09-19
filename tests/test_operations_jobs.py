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


def test_failed_launch_does_not_leave_a_queued_job(isolated, monkeypatch):
    def fail(*a, **kw):
        raise OSError('launch failed')
    monkeypatch.setattr(jobs.subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='launch failed'):
        jobs.submit_job('ais', {'start':'2020-01-24'})
    record = jobs.list_jobs()[0]
    assert record['status'] == 'failed'


def test_detection_output_refuses_to_overwrite_reviewed_run(isolated):
    root = isolated/'output'
    root.mkdir()
    (root/'run.json').write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match='must be empty'):
        jobs.execute_job('run', {'images':['does-not-matter.tiff'], 'output_dir':str(root)})
    assert (root/'run.json').read_text() == '{"status":"completed"}'


@pytest.mark.parametrize("kind", ["imagery", "ais"])
@pytest.mark.parametrize("outcome", ["success", "empty", "failure", "edited"])
def test_download_selects_inputs_only_after_success(isolated, monkeypatch, kind, outcome):
    operations.save_project({"images": str(isolated / "custom_images"), "ais_dir": str(isolated / "custom_ais")})
    original = operations.project()
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **kw: type("Process", (), {"pid": 12345})())
    record = operations.submit_project_job(kind)
    assert operations.project() == original
    if outcome == "edited":
        operations.save_project({"images": str(isolated / "new_choice")})
    expected = operations.project()
    def execute(*args, **kwargs):
        if outcome == "failure":
            raise ValueError("Acquisition failed")
        return {"files": [] if outcome == "empty" else [{"path": "acquired"}]}
    monkeypatch.setattr(jobs, "execute_job", execute)
    jobs.worker(record["job_id"])
    if outcome == "success":
        key = "images" if kind == "imagery" else "ais_dir"
        assert operations.project()[key] == record["project_update"]["updates"][key]
    else:
        assert operations.project() == expected
    assert jobs.read_job(record["job_id"])["status"] == ("failed" if outcome == "failure" else "completed")


def test_gui_download_areas_have_distinct_destinations(isolated, monkeypatch):
    submitted = []
    def submit(kind, options, **kwargs):
        submitted.append(options)
        return {"job_id": "fixture"}
    monkeypatch.setattr(jobs, "submit_job", submit)
    operations.submit_project_job("imagery")
    operations.save_project({"bbox": [-95, 28, -94.9, 28.1]})
    operations.submit_project_job("imagery")
    assert submitted[0]["output_dir"] != submitted[1]["output_dir"]
