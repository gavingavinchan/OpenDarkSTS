import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from PIL import Image

from scripts import results, training_pipeline_lib as training
from test_training_pipeline_ui import UiServer, get_json, post_json_capture


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(results, 'WORKSPACE_DIR', tmp_path)
    image = tmp_path / '2020-01-04_tile_0_0.png'
    Image.new('RGB', (100, 100), '#30485f').save(image)
    image.with_suffix('.json').write_text(json.dumps({'bbox': [-94.7, 29.1, -94.6, 29.2]}))
    root = tmp_path / 'reports' / 'operational' / 'example'
    root.mkdir(parents=True)
    (root / 'run.json').write_text(json.dumps({'inputs': [str(image)], 'status': 'completed'}))
    row = {'detection_id': 'one', 'date': '2020-01-04', 'image_path': str(image), 'cx_px': 50, 'cy_px': 50,
           'w_px': 20, 'h_px': 10, 'confidence': .8, 'center_lat': 29.15, 'center_lon': -94.65,
           'dark_status': 'FULLY_DARK', 'ais_status_text': 'no AIS vessels within 1000 metres that day', 'ais_vessels': [], 'ais_count': 0}
    (root / 'dark_sts.json').write_text(json.dumps([row]))
    return root


def test_no_ais_outcome_automatically_enters_training(run):
    with pytest.raises(ValueError, match='human training label'):
        results.send_to_improve(run, 'one')
    results.review_detection(run, 'one', 'unclear')
    with pytest.raises(ValueError, match='human training label'):
        results.send_to_improve(run, 'one')
    results.review_detection(run, 'one', 'not_sts')
    with pytest.raises(ValueError, match='human training label'):
        results.send_to_improve(run, 'one')


@pytest.mark.parametrize('verdict,label', [('real_sts', None), ('not_sts', 'ship_non_sts'), ('not_sts', 'neither')])
def test_transfer_retains_human_review_and_scene_group(run, verdict, label):
    results.review_detection(run, 'one', verdict, training_label=label)
    transfer = results.send_to_improve(run, 'one')
    doc = json.loads(open(transfer['review_path']).read())
    assert doc['review_status'] == 'completed'
    assert doc['detections'][0]['review_label'] == (label or 'ships_sts')
    assert results.send_to_improve(run, 'one') == transfer
    with pytest.raises(ValueError, match='Already sent'):
        results.review_detection(run, 'one', 'unclear')
    key = transfer['image_key']
    groups = training.resolve_selection_group_keys([key], manifest_path=results.WORKSPACE_DIR/'manifests'/'training_import_manifest.json', reviews_dir=results.WORKSPACE_DIR/'reviews')
    assert groups[key] == 'texas_gulf|2020-01-04'


def test_exports_include_human_decision_and_coordinates(run):
    results.review_detection(run, 'one', 'real_sts')
    geo = json.loads(results.export_results(run, 'geojson'))
    assert geo['features'][0]['geometry']['coordinates'] == [-94.65, 29.15]
    assert geo['features'][0]['properties']['human_review']['verdict'] == 'real_sts'
    assert b'real_sts' in results.export_results(run, 'csv')
    assert results.crop_detection(run, 'one')[0].size == (20,10)
    with pytest.raises(ValueError):
        results.run_directory(results.WORKSPACE_DIR.parent)


def test_jobs_exports_match_current_results_exports(run, monkeypatch):
    from scripts import jobs, training_pipeline_ui as ui
    monkeypatch.setattr(jobs, 'read_job', lambda value: {'kind': 'run', 'status': 'completed', 'result': {'output_dir': str(run)}})
    with UiServer(ui.TrainingPipelineHandler) as server:
        for verdict in ['real_sts', 'not_sts']:
            results.review_detection(run, 'one', verdict, training_label='ship_non_sts' if verdict == 'not_sts' else None)
            for extension in ['json', 'csv']:
                with urlopen(server.base_url + '/api/operations/jobs/example/output?format=' + extension) as response:
                    assert response.read() == results.export_results(run, extension)
                    assert 'reviewed_candidates' in response.headers['Content-Disposition']


def test_run_description_includes_input_dates_and_settings(run):
    record = {'inputs': [str(run / '2020-01-04_tile.tiff')], 'confidence': .45, 'ais_radius_m': 500,
              'started_at': '2026-10-30T12:34:56+00:00'}
    (run / 'run.json').write_text(json.dumps(record))
    description = results.list_runs()[0]['description']
    assert all(word in description for word in ['2020-01-04_tile.tiff', 'threshold 0.45', 'radius 500 m', '12:34:56 UTC'])


def test_results_http_available_by_default_and_review(run, monkeypatch):
    from scripts import training_pipeline_ui as ui
    monkeypatch.setattr(ui, 'TRAINING_UI_INTERACTION_LOG_PATH', results.WORKSPACE_DIR/'interactions.jsonl')
    monkeypatch.delenv('OPENDARKSTS_EXPERIMENTAL_RESULTS', raising=False)
    with UiServer(ui.TrainingPipelineHandler) as server:
        assert urlopen(server.base_url+'/results').status == 200
        status, decision = post_json_capture(server.base_url+'/api/results/review', {'directory':str(run), 'detection_id':'one', 'verdict':'real_sts'})
        assert status == 200
        assert decision['training_label'] == 'ships_sts'


def test_review_refuses_changed_source_image(run):
    from scripts.workspace import sha256
    record = json.loads((run/'run.json').read_text())
    source = record['inputs'][0]
    record['input_sha256'] = {source:sha256(source)}
    (run/'run.json').write_text(json.dumps(record))
    Image.new('RGB',(100,100),'red').save(source)
    with pytest.raises(ValueError, match='changed since detection'):
        results.crop_detection(run, 'one')


def test_same_crop_from_another_run_reuses_review_and_rejects_conflicting_label(run):
    import shutil
    results.review_detection(run, 'one', 'real_sts')
    first = results.send_to_improve(run, 'one')
    second = run.with_name('second')
    second.mkdir()
    for name in ['run.json','dark_sts.json']:
        shutil.copyfile(run/name, second/name)
    results.review_detection(second, 'one', 'real_sts')
    assert results.send_to_improve(second, 'one')['image_key'] == first['image_key']
    third = run.with_name('third')
    third.mkdir()
    for name in ['run.json','dark_sts.json']:
        shutil.copyfile(run/name, third/name)
    results.review_detection(third, 'one', 'not_sts', training_label='neither')
    with pytest.raises(ValueError, match='different completed label'):
        results.send_to_improve(third, 'one')
    doc = json.loads(open(first['review_path']).read())
    assert doc['detections'][0]['review_label'] == 'ships_sts'
    assert doc['operational_source']['directory'] == str(run)
