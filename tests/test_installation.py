import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import initialization
from scripts.workspace import sha256


def test_model_defaults_match_the_published_manifest(tmp_path, monkeypatch):
    from scripts import pipeline, training_pipeline_lib, workspace

    manifest = initialization.default_manifest()
    assert manifest['filename'] == 'opendarksts_yolo11s_v4_best.pt'
    assert manifest['sha256'] == 'cf11d54de2aa2ba3fa74e23d1118482a281b62cd29a95b2b1e58a06e92b261ea'
    assert manifest['url'] == 'https://github.com/gavingavinchan/OpenDarkSTS/releases/download/v0.1/' + manifest['filename']
    assert workspace.DEFAULT_WEIGHT_NAME == pipeline.DEFAULT_WEIGHTS.name == manifest['filename']
    assert workspace.current_model(tmp_path).name == manifest['filename']
    monkeypatch.setattr(training_pipeline_lib, 'WORKSPACE_DIR', tmp_path)
    assert training_pipeline_lib.normalize_auto_detect_model_paths() == {'current': tmp_path / 'model_weights' / manifest['filename']}


def test_direct_script_uses_its_adjacent_model_manifest(tmp_path):
    root = Path(__file__).resolve().parent.parent
    # Isolate sys.path from the working directory, as direct script entry points do.
    code = "import runpy, sys; print(runpy.run_path(sys.argv[1])['DEFAULT_WEIGHT_NAME'])"
    completed = subprocess.run([sys.executable, '-I', '-c', code, str(root / 'scripts' / 'workspace.py')],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == 'opendarksts_yolo11s_v4_best.pt'


@pytest.mark.parametrize('promoted', [False, True])
def test_init_upgrades_only_automatically_installed_former_default(tmp_path, monkeypatch, promoted):
    monkeypatch.setattr(initialization, 'WORKSPACE_DIR', tmp_path)
    model_root = tmp_path / 'model_weights'
    model_root.mkdir()
    old_weight = model_root / 'old.pt'
    old_weight.write_bytes(b'former checkpoint')
    source = tmp_path / 'new.pt'
    source.write_bytes(b'new checkpoint')
    previous = {'filename': old_weight.name, 'sha256': sha256(old_weight), 'url': 'https://example.invalid/old.pt'}
    record = {'filename': old_weight.name, 'sha256': previous['sha256'], 'source_url': previous['url'], 'metrics': {}}
    if promoted:
        record.update(run_group='manual', run_name='explicit promotion')
    (model_root / 'current.json').write_text(json.dumps(record))
    monkeypatch.setattr(initialization, 'default_manifest', lambda: {
        'filename': source.name, 'sha256': sha256(source), 'url': 'https://example.invalid/new.pt',
        'metrics': {}, 'previous_defaults': [previous],
    })
    result = initialization.initialize(weights=source)
    assert Path(result['model']).name == ('old.pt' if promoted else 'new.pt')
    assert old_weight.read_bytes() == b'former checkpoint'
    # Repeated initialization preserves the selected checkpoint.
    assert initialization.initialize(weights=source)['model'] == result['model']


@pytest.mark.parametrize('flag', [None, '--weights', '--weights-yolo26n'])
def test_preprocess_uses_current_model_and_accepts_legacy_override(tmp_path, monkeypatch, flag):
    from scripts import preprocess_sentinel_tiles as preprocess

    selected = {}
    default = tmp_path / 'opendarksts_yolo11s_v4_best.pt'
    override = tmp_path / 'alternative.pt'
    monkeypatch.setattr(sys, 'argv', ['preprocess'] + ([flag, str(override)] if flag else []))
    monkeypatch.setattr(preprocess, 'current_model', lambda: default)
    monkeypatch.setattr(preprocess, 'discover_tiles', lambda _: {'2020-01-24': []})
    monkeypatch.setattr(preprocess, 'load_yolo_models', lambda paths: selected.update(paths) or {})
    preprocess.main()
    assert selected == {'current': str(override if flag else default)}


def test_init_checksum_is_verified_before_replacing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(initialization, 'WORKSPACE_DIR', tmp_path)
    source = tmp_path / 'source.pt'
    source.write_bytes(b'documented fixture')
    monkeypatch.setattr(initialization, 'default_manifest', lambda: {'filename':'model.pt', 'sha256':sha256(source), 'url':'https://example.invalid/model.pt', 'metrics':{}})
    initialized = initialization.initialize(weights=source)
    assert Path(initialized['model']).read_bytes() == b'documented fixture'
    assert (tmp_path/'inputs'/'example'/'2020-01-24_example.tiff').is_file()
    (tmp_path/'model_weights'/'model.pt').write_bytes(b'existing wrong checkpoint')
    source.write_bytes(b'changed source')
    monkeypatch.setattr(initialization, 'default_manifest', lambda: {'filename':'model.pt', 'sha256':'0'*64, 'url':'unused', 'metrics':{}})
    with pytest.raises(ValueError, match='checksum mismatch'):
        initialization.initialize(weights=source)
    assert (tmp_path/'model_weights'/'model.pt').read_bytes() == b'existing wrong checkpoint'


def test_module_cli_creates_isolated_workspace(tmp_path):
    completed = subprocess.run([sys.executable, '-m', 'opendarksts', '--workspace', str(tmp_path/'space in name'), 'init', '--no-model'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)['result']
    assert Path(result['example']).is_dir()
    assert result['model'] is None


@pytest.mark.parametrize('arguments', [['--help'], ['--pretty', 'list-albums']])
def test_application_cli_forwards_training_flags(tmp_path, arguments):
    completed = subprocess.run([sys.executable, '-m', 'opendarksts', '--workspace', str(tmp_path),
                                'training', *arguments], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    if '--help' in arguments:
        assert 'import-data' in completed.stdout
    else:
        assert json.loads(completed.stdout)['ok']


@pytest.mark.integration
def test_real_bundled_example_pipeline(tmp_path):
    pytest.importorskip('ultralytics')
    pytest.importorskip('rasterio')
    root = Path(__file__).resolve().parent.parent
    weight = Path(os.environ.get('OPENDARKSTS_TEST_WEIGHTS', root/'workspace'/'model_weights'/'opendarksts_yolo11s_v4_best.pt')).expanduser().resolve()
    if not weight.is_file():
        pytest.skip('Set OPENDARKSTS_TEST_WEIGHTS to the documented release checkpoint')
    assert sha256(weight) == initialization.default_manifest()['sha256']
    env = {**os.environ, 'OPENDARKSTS_WORKSPACE':str(tmp_path), 'PYTHONPATH':str(root)}
    init = subprocess.run([sys.executable, '-m', 'opendarksts', 'init', '--weights', str(weight)], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert init.returncode == 0, init.stderr
    example = tmp_path/'inputs'/'example'
    run = subprocess.run([sys.executable, '-m', 'opendarksts', 'run', '--images', str(example/'2020-01-24_example.tiff'), '--ais-dir', str(example/'AIS_2020_01_24.csv')], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    output = Path(json.loads(run.stdout)['result']['output_dir'])
    rows = json.loads((output/'dark_sts.json').read_text())
    candidates = [r for r in rows if abs(r['center_lat']-28.6564)<.002 and abs(r['center_lon']+94.3228)<.002]
    assert candidates
    assert all(r['dark_status']=='AIS_PRESENT' and r['ais_count']>=2 for r in candidates)
    assert (output/'dark_sts.csv').is_file()
    assert list((output/'previews').glob('*.png'))
    assert json.loads((output/'run.json').read_text())['weight_sha256'] == sha256(weight)
