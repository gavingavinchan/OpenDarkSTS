"""Model lifecycle regressions for public Improve findings F1, F2, and F7."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import training_pipeline_lib as pipeline, workspace

ROOT = Path(__file__).resolve().parents[1]


def recorded_run(tmp_path):
    run = tmp_path / 'runs' / 'public' / 'cpu'
    (run / 'weights').mkdir(parents=True)
    (run / 'weights' / 'best.pt').write_bytes(b'trained checkpoint')
    config = tmp_path / 'config.json'
    starting = tmp_path / 'starting.pt'
    starting.write_bytes(b'starting checkpoint')
    config.write_text(json.dumps({'model': str(starting), 'epochs': 1, 'image_size': 128, 'seed': 42, 'device': 'cpu'}))
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'data.yaml').write_text('names: {0: ship, 1: ships}\n')
    manifest = dataset / 'dataset_manifest.json'
    manifest.write_text('{"exported_files": [{"image_sha256": "test hash"}]}')
    pipeline.write_training_metadata(run_dir=run, training_config_path=config, dataset_manifest_path=manifest,
                                     command_line=['yolo', 'detect', 'train', 'epochs=1', 'imgsz=128', 'device=cpu'],
                                     status='completed', started_at='2026-10-30T00:00:00Z')
    return run


def test_cli_promotion_preserves_workspace_recipe_without_source_edit(tmp_path):
    run = recorded_run(tmp_path)
    tracked_doc = ROOT / 'docs/yolo_weight_recreation.md'
    original = tracked_doc.read_bytes()
    args = [sys.executable, str(ROOT / 'scripts/training_pipeline_cli.py'), 'promote-weight',
            '--run-group', 'public', '--run-name', 'cpu', '--target-name', 'public_best',
            '--runs-dir', str(tmp_path / 'runs'), '--model-weights-dir', str(tmp_path / 'model_weights')]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)['result']
    record = json.loads((ROOT / payload['recreation_path']).read_text())
    assert record['sha256'] == pipeline.file_hash(run / 'weights' / 'best.pt')
    assert record['training_metadata']['command_line'][-1] == 'device=cpu'
    assert record['training_metadata']['training_config']['epochs'] == 1
    assert record['training_metadata']['starting_weight_sha256'] == pipeline.file_hash(tmp_path / 'starting.pt')
    assert record['training_metadata']['dataset_manifest_sha256']
    assert record['training_metadata']['git']
    assert record['training_metadata']['package_versions']['python']
    assert 'exported_files' in record['dataset_manifest']
    assert record['reproducibility_gaps']
    selected = workspace.current_model(tmp_path / 'model_weights')
    assert selected.name == 'public_best.pt'
    assert pipeline.weight_is_documented(selected, doc_text='')
    assert tracked_doc.read_bytes() == original
    selected.write_bytes(b'modified')
    assert not pipeline.weight_is_documented(selected, doc_text='')


def test_incomplete_recipe_does_not_change_current_model(tmp_path):
    run = recorded_run(tmp_path)
    original = tmp_path / 'model_weights' / 'previous.pt'
    original.parent.mkdir()
    original.write_bytes(b'previous')
    workspace.set_current_model(original)
    metadata = json.loads((run / 'training_metadata.json').read_text())
    del metadata['command_line']
    (run / 'training_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='incomplete: command_line'):
        pipeline.promote_weight(run_group='public', run_name='cpu', target_name='new', runs_dir=tmp_path / 'runs', model_weights_dir=original.parent)
    assert workspace.current_model(original.parent) == original
    assert not (original.parent / 'new.pt').exists()


def test_report_recommendation_does_not_claim_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'WORKSPACE_DIR', tmp_path)
    run = recorded_run(tmp_path)
    kwargs = dict(run_group='public', runs=[{'run_name': 'cpu'}], reports_dir=tmp_path / 'reports',
                  promoted_run_name='cpu', promotion_reason='selected')
    paths = pipeline.write_evaluation_report(**kwargs)
    report = json.loads(paths['json_path'].read_text())
    assert report['recommendation']['run_name'] == 'cpu'
    assert report['promotion']['promoted'] is False
    assert 'Promoted: none' in paths['markdown_path'].read_text()
    pipeline.promote_weight(run_group='public', run_name='cpu', target_name='new', runs_dir=tmp_path / 'runs', model_weights_dir=tmp_path / 'model_weights')
    report = json.loads(pipeline.write_evaluation_report(**kwargs)['json_path'].read_text())
    assert report['promotion']['promoted'] is True
    assert report['promotion']['run_name'] == 'cpu'


def test_restore_default_verifies_checksum_and_preserves_improved_weight(tmp_path, monkeypatch):
    model_root = tmp_path / 'model_weights'
    model_root.mkdir()
    default = model_root / 'bundled.pt'
    default.write_bytes(b'bundled')
    improved = model_root / 'improved.pt'
    improved.write_bytes(b'improved')
    manifest = {'filename': default.name, 'sha256': workspace.sha256(default), 'url': 'upstream', 'metrics': {}}
    monkeypatch.setattr(workspace, 'default_manifest', lambda: manifest)
    workspace.set_current_model(improved)
    assert workspace.restore_default_model(model_root)['is_default']
    assert workspace.current_model(model_root) == default
    assert improved.read_bytes() == b'improved'
    workspace.set_current_model(improved)
    default.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        workspace.restore_default_model(model_root)
    assert workspace.current_model(model_root) == improved
    # Exercise the CLI failure without downloads or changing the selected model.
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/training_pipeline_cli.py'), 'restore-default-model',
                             '--model-weights-dir', str(model_root)], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'opendarksts init' in json.loads(result.stdout)['errors'][0]


@pytest.mark.parametrize("uppercase", [False, True])
def test_promotion_cannot_replace_restorable_bundled_checkpoint(tmp_path, uppercase):
    with pytest.raises(ValueError, match='reserved'):
        pipeline.promote_weight(run_group='public', run_name='cpu', target_name=workspace.DEFAULT_WEIGHT_NAME.upper() if uppercase else workspace.DEFAULT_WEIGHT_NAME, overwrite=True)
