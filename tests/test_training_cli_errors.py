from pathlib import Path

import pytest

from test_training_pipeline_cli import ROOT_DIR, parse_envelope, run_cli


@pytest.mark.parametrize("command,options", [
    ("training-status", ["--run-dir"]),
    ("stop-training", ["--run-dir"]),
    ("run-test-split", ["--run-dir"]),
    ("list-active-runs", ["--runs-dir"]),
    ("list-run-groups", ["--runs-dir"]),
    ("show-run-group", ["--run-group", "fixture", "--runs-dir"]),
    ("write-evaluation-report", ["--run-group", "fixture", "--runs-dir"]),
    ("promote-weight", ["--run-group", "fixture", "--run-name", "run", "--target-name", "fixture.pt", "--recreation-doc-updated", "--runs-dir"]),
])
def test_cli_invalid_run_path_is_validation_error(command, options):
    process = run_cli(command, *options, str(ROOT_DIR))
    assert process.returncode == 2
    payload = parse_envelope(process)
    assert payload["command"] == command
    assert payload["ok"] is False
    assert "run_dir must be inside" in payload["errors"][0]
    assert "Traceback" not in process.stderr


def test_cli_invalid_run_reference_is_validation_error():
    process = run_cli("training-status", "--run-group", "../outside", "--run-name", "run")
    assert process.returncode == 2
    assert "single path segment" in parse_envelope(process)["errors"][0]


def test_cli_corrupt_process_reports_warnings_and_refuses_stop(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "process.json").write_text("{")
    process = run_cli("training-status", "--run-dir", str(run))
    assert process.returncode == 4
    payload = parse_envelope(process)
    assert payload["result"]["status"] == "failed"
    assert "process.json could not be parsed" in payload["warnings"][0]
    stopped = run_cli("stop-training", "--run-dir", str(run))
    assert stopped.returncode == 2
    assert "no process can safely be stopped" in parse_envelope(stopped)["errors"][0]
    assert (run / "process.json").read_text() == "{"


def test_cli_runtime_file_error_keeps_exit_code_one(tmp_path: Path):
    process = run_cli("check-training-ready", "--selection", str(tmp_path / "missing.json"), "--config", str(tmp_path / "config.json"))
    assert process.returncode == 1
    assert parse_envelope(process)["ok"] is False


def test_cli_malformed_selection_is_validation_error(tmp_path: Path):
    selection = tmp_path / "selection.json"
    selection.write_text("[]")
    process = run_cli("check-training-ready", "--selection", str(selection), "--config", str(tmp_path / "config.json"))
    assert process.returncode == 2
    assert "Expected JSON object" in parse_envelope(process)["errors"][0]


def test_cli_explicit_value_error_handler_keeps_exit_code_two(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    created = run_cli("create-album", "--album-name", "fixture", "--manifest", str(manifest))
    assert created.returncode == 0
    album_id = parse_envelope(created)["result"]["album_id"]
    renamed = run_cli("rename-album", "--album-id", album_id, "--album-name", "", "--manifest", str(manifest))
    assert renamed.returncode == 2
    assert parse_envelope(renamed)["ok"] is False
