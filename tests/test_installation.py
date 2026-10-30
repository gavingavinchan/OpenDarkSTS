import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import initialization
from scripts.workspace import sha256


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
    weight = Path(os.environ.get('OPENDARKSTS_TEST_WEIGHTS', root/'workspace'/'model_weights'/'yolo26n_sts_clean_v1_e300_img960_2026-09-15_best.pt'))
    if not weight.is_file():
        pytest.skip('Set OPENDARKSTS_TEST_WEIGHTS to the documented release checkpoint')
    assert sha256(weight) == initialization.default_manifest()['sha256']
    env = {**os.environ, 'OPENDARKSTS_WORKSPACE':str(tmp_path)}
    init = subprocess.run([sys.executable, '-m', 'opendarksts', 'init', '--weights', str(weight)], env=env, capture_output=True, text=True)
    assert init.returncode == 0, init.stderr
    example = tmp_path/'inputs'/'example'
    run = subprocess.run([sys.executable, '-m', 'opendarksts', 'run', '--images', str(example/'2020-01-24_example.tiff'), '--ais-dir', str(example/'AIS_2020_01_24.csv')], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    output = Path(json.loads(run.stdout)['result']['output_dir'])
    rows = json.loads((output/'dark_sts.json').read_text())
    candidates = [r for r in rows if abs(r['center_lat']-28.6564)<.002 and abs(r['center_lon']+94.3228)<.002]
    assert candidates
    assert all(r['dark_status']=='AIS_PRESENT' and r['ais_count']>=2 for r in candidates)
    assert (output/'dark_sts.csv').is_file()
    assert list((output/'previews').glob('*.png'))
    assert json.loads((output/'run.json').read_text())['weight_sha256'] == sha256(weight)
