"""Behavioral checks for the small paper-to-release compatibility additions."""
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import detect, operations, training_pipeline_lib as training, training_pipeline_cli as cli


@pytest.mark.parametrize("width,height,kept", [(100, 100, False), (100, 50, True), (100, 51, False), (20, 20, True)])
def test_detector_safety_filter_uses_original_tile_area(tmp_path, width, height, kept):
    image = tmp_path / '2020-01-24_test.png'
    Image.new('RGB', (100, 100)).save(image)
    box = SimpleNamespace(cls=SimpleNamespace(item=lambda: 1), conf=SimpleNamespace(item=lambda: .8),
                          xywh=[SimpleNamespace(tolist=lambda: [50, 50, width, height])])
    model = SimpleNamespace(names={0: 'ship', 1: 'ships'},
                            predict=lambda **kw: [SimpleNamespace(names={1: 'ships'}, boxes=[box])])
    assert bool(detect.detect_image(image, model)) is kept


def test_augmentation_presets_disable_all_transforms_for_none():
    presets = training.augmentation_presets()
    assert set(presets) == {'none', 'light', 'default', 'heavy'}
    for preset in presets.values():
        assert set(preset) == set(training.ALLOWED_AUGMENTATION_KEYS)
        assert training.normalize_augmentation(preset) == preset
    assert presets['none']['crop_fraction'] == 1
    assert all(value == 0 for key, value in presets['none'].items() if key != 'crop_fraction')
    presets['heavy']['degrees'] = 0
    assert training.augmentation_presets()['heavy']['degrees'] == 180


def test_no_ais_source_never_searches_the_current_directory(tmp_path, monkeypatch):
    from scripts import correlate_ais
    monkeypatch.setattr(correlate_ais, 'resolve_ais_file', lambda **kw: pytest.fail('No source must not search cwd'))
    rows = correlate_ais.correlate_detections([{'detection_id': 'one', 'date': '2020-01-24'}], '')
    assert rows[0]['dark_status'] == 'AIS_MISSING'
    assert rows[0]['ais_count'] is None
    assert rows[0]['ais_error'] == 'No AIS source supplied'
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'AIS_2020_01_24.csv').write_text('MMSI,BaseDateTime,LAT,LON\n')
    monkeypatch.setattr(operations.jobs, 'list_jobs', lambda: [])
    assert operations.detection_readiness({**operations.project(), 'ais_dir': ''})['warnings']


@pytest.mark.parametrize('source', ['', '/selected/daily.csv'])
def test_job_preserves_explicit_ais_source(tmp_path, monkeypatch, source):
    from scripts import jobs
    received = []
    monkeypatch.setattr(jobs.pipeline, 'run_pipeline', lambda *a, **kw: received.append(kw['ais_dir']) or [])
    jobs._execute_pipeline({'images': ['example.tiff'], 'ais_dir': source}, tmp_path, lambda text: None)
    assert received == [source]


def test_cli_preset_is_saved_with_explicit_overrides(tmp_path, capsys):
    code = cli.main(['save-training-config', '--dataset-name', 'fixture', '--model', 'fixture.yaml',
                     '--run-group', 'fixture', '--run-name', 'preset', '--device', 'cpu',
                     '--configs-dir', str(tmp_path / 'configs'), '--output-dir', str(tmp_path / 'runs'),
                     '--aug-preset', 'heavy', '--aug', 'degrees=90'])
    response = json.loads(capsys.readouterr().out)
    assert code == 0, response
    config = response['result']['config']
    assert config['augmentation'] == {**training.augmentation_presets()['heavy'], 'degrees': 90}


def test_dashboard_counts_completed_scans_and_preserves_custom_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(operations, 'WORKSPACE_DIR', tmp_path)
    monkeypatch.setattr(operations, 'PROJECT_PATH', tmp_path / 'project.json')
    monkeypatch.setattr(operations.jobs, 'WORKSPACE_DIR', tmp_path)
    monkeypatch.setattr(operations, 'detection_readiness', lambda settings: {'ready': True})
    monkeypatch.setattr(operations.jobs, 'list_jobs', lambda **kw: [
        {'kind': 'run', 'status': 'completed', 'result': {'summary': {'detections': 2, 'AIS_PRESENT': 2}}},
        {'kind': 'run', 'status': 'completed', 'result': {'summary': {'detections': 1, 'AIS_INVALID': 1}}},
        {'kind': 'run', 'status': 'failed'}, {'kind': 'ais', 'status': 'completed'}])
    assert operations.project()['max_cloud'] == 35
    assert operations.project()['radius'] == 1000
    saved = operations.save_project({'confidence': .32, 'radius': 800, 'max_cloud': 18})
    dashboard = operations.dashboard()
    assert dashboard['project'] == saved
    assert dashboard['sensitivity_presets'] == {'Low': .1, 'Medium': .25, 'High': .45}
    assert dashboard['detection_summary'] == {'runs': 2, 'detections': 3, 'FULLY_DARK': 0,
        'PARTIALLY_DARK': 0, 'AIS_PRESENT': 2, 'AIS_MISSING': 0, 'AIS_INVALID': 1}
