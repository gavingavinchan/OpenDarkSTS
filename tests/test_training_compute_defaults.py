"""Hardware selection must be shared, GPU-aware, and reproducible after saving."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import training_pipeline_lib as training, training_pipeline_cli as cli
from test_training_pipeline_ui import UiServer, get_json, post_json


def fake_torch(monkeypatch, *, cuda=False, mps=False):
    torch = SimpleNamespace(__version__='test-torch',
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda index: 'Test NVIDIA GPU'),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, 'torch', torch)


@pytest.mark.parametrize('cuda,mps,device,backend', [(True,True,0,'cuda'), (False,True,'mps','mps'), (False,False,'cpu','cpu')])
def test_auto_selects_available_accelerator(monkeypatch, cuda, mps, device, backend):
    fake_torch(monkeypatch,cuda=cuda,mps=mps)
    compute = training.training_compute()
    assert compute['device'] == device
    assert compute['backend'] == backend
    assert compute['python_executable'] == sys.executable
    assert training.resolve_training_device('auto') == device


def test_explicit_cpu_does_not_probe_or_change_device(monkeypatch, tmp_path):
    monkeypatch.setattr(training, 'training_compute', lambda *a,**kw: pytest.fail('Explicit CPU must not trigger automatic selection'))
    saved = training.save_training_config(dataset_name='fixture',dataset_version='v1',model='fixture.yaml',
        run_group='fixture',run_name='cpu',device='cpu',configs_dir=tmp_path/'configs')
    config = training.load_json(saved['config_path'])
    assert config['device'] == config['device_request'] == 'cpu'


def test_auto_saved_configuration_records_concrete_gpu(monkeypatch, tmp_path):
    fake_torch(monkeypatch,cuda=True)
    saved = training.save_training_config(dataset_name='fixture',dataset_version='v1',model='fixture.yaml',
        run_group='fixture',run_name='gpu',configs_dir=tmp_path/'configs')
    config = training.load_json(saved['config_path'])
    assert config['device'] == 0
    assert config['device_request'] == 'auto'
    assert config['training_python'] == sys.executable
    # A later hardware change does not reinterpret the saved concrete choice.
    fake_torch(monkeypatch,cuda=False)
    dataset=tmp_path/'dataset'; dataset.mkdir(); (dataset/'data.yaml').write_text('names: {0: ship, 1: ships}\n')
    assert 'device=0' in training.build_yolo_train_command(config,dataset)


def test_other_selected_python_is_used_for_hardware_probe(monkeypatch):
    seen=[]
    def run(argv,**kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0,stdout=json.dumps({'device':'cpu','backend':'cpu','label':'CPU','python_executable':'/chosen/python'}),stderr='')
    monkeypatch.setattr(training.subprocess,'run',run)
    assert training.resolve_training_device('auto',python_executable='/chosen/python') == 'cpu'
    assert seen[0][0] == '/chosen/python'


def test_cli_and_http_defaults_match_shared_helper(monkeypatch, capsys):
    fake_torch(monkeypatch,cuda=True)
    expected=training.training_defaults()
    assert cli.main(['training-defaults']) == 0
    assert json.loads(capsys.readouterr().out)['result'] == expected
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    with UiServer(TrainingPipelineHandler) as server:
        assert get_json(server.base_url+'/api/training/defaults') == expected
    assert cli.build_parser().parse_args(['save-training-config','--dataset-name','fixture','--model','model.yaml','--run-group','fixture','--run-name','run']).device == 'auto'


def test_http_save_uses_auto_when_device_omitted(monkeypatch,tmp_path):
    fake_torch(monkeypatch,cuda=True)
    monkeypatch.setattr(training,'DEFAULT_TRAINING_CONFIGS_DIR',tmp_path/'configs')
    from scripts import training_pipeline_ui as ui
    monkeypatch.setattr(ui,'TRAINING_UI_INTERACTION_LOG_PATH',tmp_path/'interactions.jsonl')
    with UiServer(ui.TrainingPipelineHandler) as server:
        saved=post_json(server.base_url+'/api/training/configs',{'dataset_name':'fixture','model':'fixture.yaml','run_group':'fixture','run_name':'http'})
    assert saved['config']['device'] == 0
    assert saved['config']['device_request'] == 'auto'
