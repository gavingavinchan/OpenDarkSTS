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
    monkeypatch.setenv("CDSE_CLIENT_ID", "test-client")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "test-secret")
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
            if path != '/import':
                assert 'href="/results"' in html
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


def test_readiness_and_submission_use_the_same_dates(isolated, monkeypatch):
    images = isolated / 'imagery'
    images.mkdir()
    for name in ['2020-01-24_scene.tiff', '2020-01-25_scene.tiff', 'undated.png']:
        (images / name).write_bytes(b'fixture')
    model = isolated / 'model.pt'
    model.write_bytes(b'fixture')
    monkeypatch.setattr(operations, 'current_model', lambda: model)
    operations.save_project({'images': str(images), 'start': '2020-01-24', 'end': '2020-01-24'})
    readiness = operations.detection_readiness()
    assert readiness['ready'] and readiness['image_count'] == 1
    assert readiness['warnings']  # Missing AIS is explicitly unknown, but not a launch blocker.
    submitted = []
    monkeypatch.setattr(jobs, 'submit_job', lambda kind, options, **kw: submitted.append(options) or {'job_id': 'fixture'})
    operations.submit_project_job('run')
    assert submitted[0]['images'] == [str(images / '2020-01-24_scene.tiff')]
    operations.save_project({'start': '2021-01-01', 'end': '2021-01-01'})
    with pytest.raises(ValueError, match='No image filenames match'):
        operations.submit_project_job('run')
    assert len(submitted) == 1


def test_dashboard_distinguishes_training_data_from_detection_inputs(isolated, monkeypatch):
    manifest = isolated / 'manifests' / 'training_import_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'images': {'imported': {}}}))
    reviews = isolated / 'reviews' / 'tester'
    reviews.mkdir(parents=True)
    (reviews / 'imported.json').write_text(json.dumps({'image_key': 'imported', 'review_status': 'completed'}))
    (reviews / 'legacy.json').write_text(json.dumps({'image_key': 'legacy', 'review_status': 'overview_1'}))
    (isolated / 'albums').mkdir()
    (isolated / 'albums' / 'alb_fixture.json').write_text('{}')
    dashboard = operations.dashboard()
    assert dashboard['inventory']['images'] == 0
    assert dashboard['training_inventory'] == {'images': 2, 'completed': 1, 'albums': 1}
    assert not dashboard['readiness']['ready']


def test_dashboard_skips_nonobject_legacy_records(isolated):
    manifest = isolated / 'manifests' / 'training_import_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('[]')
    reviews = isolated / 'reviews' / 'tester'
    reviews.mkdir(parents=True)
    (reviews / 'invalid.json').write_text('[]')
    dashboard = operations.dashboard()
    assert dashboard['training_inventory']['images'] == 0
    assert not dashboard['readiness']['ready']


@pytest.mark.parametrize('kind', ['ais', 'imagery'])
def test_readiness_blocks_only_downloads_of_selected_inputs(isolated, monkeypatch, kind):
    model = isolated / 'model.pt'
    model.touch()
    monkeypatch.setattr(operations, 'current_model', lambda: model)
    images = isolated / 'images'
    images.mkdir()
    (images / '2020-01-24_scene.tiff').touch()
    operations.save_project({'images': str(images), 'start': '2020-01-24', 'end': '2020-01-24'})
    selected = isolated / 'ais' if kind == 'ais' else images
    operations.save_project({'ais_dir': str(isolated / 'ais')})
    active = {'kind': kind, 'status': 'running', 'options': {'start': '2020-01-24', 'output_dir': str(isolated / 'unrelated')}}
    monkeypatch.setattr(jobs, 'list_jobs', lambda: [active])
    assert operations.detection_readiness()['ready']
    active['options']['output_dir'] = str(selected)
    assert not operations.detection_readiness()['ready']
    active['options']['start'] = '2020-01-25'
    assert operations.detection_readiness()['ready']
    active['options']['start'] = '2020-01-24'
    active['status'] = 'cancelled'
    assert operations.detection_readiness()['ready']


def test_explicit_ais_preflight_uses_contents_and_rechecks_changes(isolated, monkeypatch):
    model = isolated / 'model.pt'
    model.touch()
    monkeypatch.setattr(operations, 'current_model', lambda: model)
    image = isolated / '2020-01-24_scene.tiff'
    image.touch()
    csv = isolated / 'my-vessel-records.csv'
    csv.write_text('MMSI,BaseDateTime,LAT,LON\n1,2020-01-24T01:00:00,28,-94\n')
    operations.save_project({'images': str(image), 'ais_dir': str(csv), 'start': '2020-01-24', 'end': '2020-01-24'})
    assert operations.detection_readiness()['ready']
    csv.write_text('MMSI,BaseDateTime,LAT,LON\n1,unknown,28,-94\n')
    assert 'Cannot establish' in ' '.join(operations.detection_readiness()['blockers'])
    csv.write_text('MMSI,BaseDateTime,LAT,LON\n1,2020-01-25T01:00:00,28,-94\n')
    assert 'no records for image date' in ' '.join(operations.detection_readiness()['blockers'])


def test_foreground_run_updates_jobs_and_dashboard(isolated, monkeypatch):
    summary = {'detections': 1, 'AIS_PRESENT': 1}
    monkeypatch.setattr(jobs, 'execute_job', lambda *a, **kw: {'output_dir': str(isolated / 'output'), 'summary': summary})
    result = jobs.run_foreground('run', {'images': ['2020-01-24_scene.tiff'], 'conf': .1, 'radius': 70})
    record = jobs.list_jobs()[0]
    assert record['status'] == 'completed' and record['result'] == result
    assert all(word in record['description'] for word in ['2020-01-24_scene.tiff', 'threshold 0.1', 'radius 70 m', 'UTC'])
    assert operations.dashboard()['detection_summary']['runs'] == 1
    assert operations.dashboard()['detection_summary']['AIS_PRESENT'] == 1


def test_foreground_failure_is_recorded(isolated, monkeypatch):
    def fail(*a, **kw):
        raise ValueError('bad input')
    monkeypatch.setattr(jobs, 'execute_job', fail)
    with pytest.raises(ValueError, match='bad input'):
        jobs.run_foreground('run', {'images': ['missing.tiff']})
    assert jobs.list_jobs()[0]['status'] == 'failed'


@pytest.mark.parametrize('kind', ['ais', 'imagery'])
def test_cancel_download_via_http_and_cli_is_cooperative(isolated, monkeypatch, kind):
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    original_popen = subprocess.Popen
    monkeypatch.setattr(jobs.subprocess, 'Popen', lambda *a, **kw: type('Process', (), {'pid': 12345})())
    monkeypatch.setattr(jobs, 'process_is_alive', lambda *a: True)
    record = operations.submit_project_job(kind)
    monkeypatch.setattr(jobs.subprocess, 'Popen', original_popen)
    with UiServer(TrainingPipelineHandler) as server:
        status, cancelled = post_json_capture(server.base_url + '/api/operations/jobs/' + record['job_id'] + '/cancel', {})
    assert status == 202 and cancelled['cancel_requested']
    jobs.worker(record['job_id'])
    assert jobs.read_job(record['job_id'])['status'] == 'cancelled'
    completed = subprocess.run([sys.executable, '-m', 'scripts.application_cli', 'jobs', record['job_id'], '--cancel'], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)['result']['status'] == 'cancelled'
