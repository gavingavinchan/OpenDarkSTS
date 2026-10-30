"""Release diagnostics must never write into the installation directory."""
import json
from pathlib import Path

import pytest

from scripts import review_visual_detections_ui as annotation, training_pipeline_ui as ui


def test_annotation_has_no_temporary_logger():
    source = Path(annotation.__file__).read_text()
    for removed in ('DELETE BEFORE SHIP', 'debug-interaction', 'logDebugInteraction', 'console.debug'):
        assert removed not in source
    assert not hasattr(annotation, 'append_debug_interaction_event')


def test_training_diagnostics_write_only_in_selected_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / 'user workspace'
    monkeypatch.setattr(ui, 'WORKSPACE_DIR', workspace)
    log = workspace / 'reports' / 'interactions.jsonl'
    monkeypatch.setattr(ui, 'TRAINING_UI_INTERACTION_LOG_PATH', log)
    ui.append_training_ui_interaction({'action': 'test'})
    assert json.loads(log.read_text())['event'] == {'action': 'test'}
    outside = tmp_path / 'install' / 'interactions.jsonl'
    monkeypatch.setattr(ui, 'TRAINING_UI_INTERACTION_LOG_PATH', outside)
    with pytest.raises(ValueError, match='inside'):
        ui.append_training_ui_interaction({'action': 'test'})
    assert not outside.parent.exists()


def test_training_diagnostics_reject_symlink_escape(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    outside = tmp_path / 'install'
    outside.mkdir()
    try:
        (workspace / 'reports').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Symlinks unavailable')
    monkeypatch.setattr(ui, 'WORKSPACE_DIR', workspace)
    monkeypatch.setattr(ui, 'TRAINING_UI_INTERACTION_LOG_PATH', workspace / 'reports' / 'interactions.jsonl')
    with pytest.raises(ValueError, match='inside'):
        ui.append_training_ui_interaction({'action': 'test'})
    assert list(outside.iterdir()) == []
