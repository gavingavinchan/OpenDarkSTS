import json
import time
from urllib.request import Request, urlopen

import pytest

from scripts import training_pipeline_ui as ui
from test_training_pipeline_ui import UiServer, post_json


@pytest.mark.parametrize('fail', [False, True])
def test_upload_and_second_evaluation_after_test_split(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(ui, 'JOBS', {})
    monkeypatch.setattr(ui, 'TRAINING_UI_INTERACTION_LOG_PATH', tmp_path / 'interactions.jsonl')
    monkeypatch.setattr(ui.pipeline, 'DEFAULT_RUNS_DIR', tmp_path / 'runs')
    def evaluate(run_dir, **kwargs):
        if fail:
            raise ValueError('test evaluation failure')
        return {'run_dir': str(run_dir)}
    monkeypatch.setattr(ui.pipeline, 'run_test_split', evaluate)
    monkeypatch.setattr(ui, 'start_import_job', lambda *args, **kwargs: 'upload-fixture')
    with UiServer(ui.TrainingPipelineHandler) as server:
        result = post_json(server.base_url + '/api/training/runs/group/run/test-split', {'background': True})
        job_id = result['job_id']
        deadline = time.monotonic() + 5
        while ui.JOBS[job_id]['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(.01)
        assert ui.JOBS[job_id]['status'] == ('failed' if fail else 'completed')
        assert ui.JOBS[job_id]['finished_epoch'] > 0
        body = b'--fixture\r\nContent-Disposition: form-data; name="files"; filename="image.png"\r\n\r\nimage\r\n--fixture--\r\n'
        request = Request(server.base_url + '/api/import/upload', data=body,
                          headers={'Content-Type': 'multipart/form-data; boundary=fixture'})
        with urlopen(request) as response:
            assert json.load(response)['job_id'] == 'upload-fixture'
        assert job_id in ui.JOBS  # Recent results remain available to polling clients.
        second = post_json(server.base_url + '/api/training/runs/group/run/test-split', {'background': True})
        assert second['job_id'] != job_id


def test_job_retention_handles_both_import_and_evaluation_records(monkeypatch):
    old = time.time() - ui.JOB_RETENTION_SECONDS - 1
    records = {
        'import_old': {'state': 'done', 'finished_epoch': old},
        'eval_old': {'status': 'completed', 'finished_epoch': old},
        'eval_failed': {'status': 'failed', 'finished_epoch': old},
        'eval_running': {'status': 'running'},
        'import_running': {'state': 'running', 'finished_epoch': 0},
        'eval_recent': {'status': 'completed', 'finished_epoch': time.time()},
    }
    monkeypatch.setattr(ui, 'JOBS', records)
    ui.reap_old_jobs()
    assert set(records) == {'eval_running', 'import_running', 'eval_recent'}
