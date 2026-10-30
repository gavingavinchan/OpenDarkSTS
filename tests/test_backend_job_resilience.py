"""A malformed old job must not make the current Jobs view unusable."""
import json
import subprocess
import sys

import pytest

from scripts import jobs
from test_operations_jobs import isolated
from test_training_pipeline_ui import UiServer, get_json


def write_job(root, job_id, payload):
    directory = root / 'jobs' / job_id
    directory.mkdir(parents=True)
    (directory / 'job.json').write_text(json.dumps(payload))
    return directory


def completed_record(job_id):
    return {'job_id': job_id, 'kind': 'run', 'options': {'images': ['legacy.tiff']},
            'status': 'completed', 'created_at': '2020-01-04T00:00:00+00:00'}


@pytest.mark.parametrize('bad', [[], None, {}, {'status': 'running'},
                                {'created_at': 3}, {'created_at': 'not_a_date'},
                                {'created_at': '2020-01-04T00:00:00'}, {'kind': []}, {'status': {}}])
def test_malformed_job_does_not_hide_good_jobs(isolated, bad):
    good_id, bad_id = 'a' * 32, 'b' * 32
    write_job(isolated, good_id, completed_record(good_id))
    updates_record = isinstance(bad, dict) and bad != {'status': 'running'} and any(key in bad for key in ['created_at', 'kind', 'status'])
    payload = {**completed_record(bad_id), **bad} if updates_record else bad
    write_job(isolated, bad_id, payload)
    assert [job['job_id'] for job in jobs.list_jobs()] == [good_id]
    with pytest.raises(ValueError, match='Invalid job'):
        jobs.read_job(bad_id)
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    with UiServer(TrainingPipelineHandler) as server:
        assert [job['job_id'] for job in get_json(server.base_url + '/api/operations/jobs')['jobs']] == [good_id]
        assert get_json(server.base_url + '/api/operations/dashboard')['jobs'][0]['job_id'] == good_id


@pytest.mark.parametrize('launch', [[], {'pid': -1}, {'pid': 42, 'status': []}, {'pid': '42'}])
def test_malformed_launch_record_is_cli_error_and_does_not_break_listing(isolated, launch):
    job_id = 'c' * 32
    directory = write_job(isolated, job_id, completed_record(job_id))
    (directory / 'launch.json').write_text(json.dumps(launch))
    assert jobs.list_jobs() == []
    process = subprocess.run([sys.executable, '-m', 'scripts.application_cli', 'jobs', job_id],
                             text=True, capture_output=True)
    assert process.returncode == 2
    assert 'Invalid job launch record' in json.loads(process.stdout)['errors'][0]
    assert 'Traceback' not in process.stderr
