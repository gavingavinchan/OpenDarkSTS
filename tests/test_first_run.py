"""First-visitor setup, model information, and safe server shutdown regressions."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import operations, workspace
from test_training_pipeline_ui import UiServer, get_json, post_json_capture

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(operations, 'WORKSPACE_DIR', tmp_path)
    monkeypatch.setattr(operations, 'PROJECT_PATH', tmp_path / 'project.json')
    return tmp_path


def test_cli_example_is_independent_of_start_directory(tmp_path):
    target = tmp_path / 'visitor workspace'
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    def cli(*args):
        result = subprocess.run([sys.executable, '-m', 'opendarksts', '--workspace', str(target), *args],
                                cwd=tmp_path, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr + result.stdout
        return json.loads(result.stdout)['result']
    cli('init', '--no-model')
    settings = cli('project', '--use-example')
    assert settings['start'] == settings['end'] == '2020-01-24'
    assert Path(settings['images']) == target / 'inputs' / 'example'
    assert Path(settings['ais_dir']).is_file()
    west, south, east, north = settings['bbox']
    assert west < -94.3228 < east and south < 28.6564 < north
    assert cli('project') == settings
    selected = cli('project', '--set', '{"images":"inputs/example","ais_dir":"inputs/example/AIS_2020_01_24.csv"}')
    assert selected == settings


def test_model_scores_and_card_follow_selected_model(isolated, monkeypatch):
    path = isolated / 'model_weights' / 'default.pt'
    path.parent.mkdir()
    path.touch()
    monkeypatch.setattr(operations, 'current_model', lambda: path)
    manifest = workspace.default_manifest()
    record = path.parent / 'current.json'
    record.write_text(json.dumps({'sha256': manifest['sha256'], 'metrics': manifest['metrics']}))
    model = operations.model_information()['model']
    assert model['name'] == 'YOLO11s'
    assert model['model_card_url'] == '/model-card'
    assert model['metrics']['ships_mAP50'] == .920
    record.write_text(json.dumps({'sha256': 'custom', 'metrics': {'mAP50': .5}}))
    model = operations.model_information()['model']
    assert model['name'] == path.stem
    assert model['metrics'] == {'mAP50': .5}
    assert model['model_card_url'] is None  # Never label an improved model with default scores/card.


def test_missing_credentials_prevents_job_creation(isolated, monkeypatch):
    monkeypatch.delenv('CDSE_CLIENT_ID', raising=False)
    monkeypatch.delenv('CDSE_CLIENT_SECRET', raising=False)
    monkeypatch.setattr(operations.jobs, 'submit_job', lambda *a, **kw: pytest.fail('Must not launch'))
    assert not operations.imagery_credentials_ready()
    with pytest.raises(ValueError, match='restart the server'):
        operations.submit_project_job('imagery')
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    with UiServer(TrainingPipelineHandler) as server:
        status, payload = post_json_capture(server.base_url + '/api/operations/jobs', {'kind': 'imagery'})
    assert status == 400 and 'Copernicus' in payload['error']
    monkeypatch.setenv('CDSE_CLIENT_ID', 'client')
    assert not operations.imagery_credentials_ready()
    monkeypatch.setenv('CDSE_CLIENT_SECRET', 'secret')
    assert operations.imagery_credentials_ready()


@pytest.mark.parametrize("prefix", ["TIFF_Warning:", "TIFF_Warning TIFFReadDirectory:"])
def test_job_log_hides_only_known_geotiff_metadata_warning(isolated, monkeypatch, prefix):
    from scripts import jobs
    monkeypatch.setattr(jobs, 'JOB_ROOT', isolated / 'jobs')
    job_id = 'a' * 32
    root = jobs.job_dir(job_id)
    root.mkdir(parents=True)
    workspace.atomic_json(root / 'job.json', {'job_id': job_id, 'kind': 'run', 'status': 'completed',
        'options': {}, 'created_at': '2026-10-30T00:00:00+00:00'})
    warning = f'[ WARN:0] grfmt_tiff.cpp: {prefix} Unknown field with tag 33550 (0x830e) encountered'
    error = 'TIFF_Error: cannot read raster'
    unknown = 'TIFF_Warning: Unknown field with tag 9999 (0x270f) encountered'
    (root / 'job.log').write_text('\n'.join([warning, error, unknown, 'Detection completed']))
    assert jobs.read_job(job_id)['log_tail'] == [error, unknown, 'Detection completed']
    assert warning in (root / 'job.log').read_text()  # Raw diagnostic record remains available.
