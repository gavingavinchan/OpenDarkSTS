from pathlib import Path

import pytest

from scripts import training_pipeline_lib as pipeline


def test_write_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "record.json"
    pipeline.write_json(target, {"old": True})
    pipeline.write_json(target, {"new": True})
    assert target.read_text() == '{\n  "new": true\n}\n'
    assert list(target.parent.iterdir()) == [target]


def test_write_json_failed_replace_preserves_original(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "record.json"
    pipeline.write_json(target, {"old": True})
    original = target.read_bytes()

    def fail_replace(source, destination):
        assert source.parent == target.parent
        assert pipeline.load_json(source) == {"new": True}
        assert destination == target
        assert target.read_bytes() == original
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        pipeline.write_json(target, {"new": True})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


def test_write_json_serialization_failure_preserves_original(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    pipeline.write_json(target, {"old": True})
    with pytest.raises(TypeError):
        pipeline.write_json(target, {"invalid": object()})
    assert pipeline.load_json(target) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]
