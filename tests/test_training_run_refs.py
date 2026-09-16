from __future__ import annotations

from pathlib import Path

from test_training_pipeline_ui import (
    UiServer,
    get_json,
    get_json_capture,
    post_json_capture,
    prepare_training_workspace,
    save_and_export_selection,
)
from _helpers import write_json

import urllib.parse


def test_training_run_routes_percent_decode_path_segments(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    # A non-ASCII run name is percent-encoded by the browser; the handler must decode it.
    run_group = "grup\u00e9"
    run_name = "run\u00e9"
    run_dir = tmp_path / "runs" / run_group / run_name
    run_dir.mkdir(parents=True)
    write_json(run_dir / "process.json", {
        "schema_version": 1, "status": "completed", "pid": None, "start_ticks": None,
        "started_at": "2026-05-19T00:00:00Z", "ended_at": "2026-05-19T00:01:00Z",
        "expected_epochs": 1, "command_line": [],
    })
    monkeypatch.setattr(ui.pipeline, "DEFAULT_RUNS_DIR", tmp_path / "runs")

    with UiServer(ui.TrainingPipelineHandler) as server:
        encoded = f"{urllib.parse.quote(run_group)}/{urllib.parse.quote(run_name)}"
        status = get_json(f"{server.base_url}/api/training/runs/{encoded}/status")
        missing = f"{urllib.parse.quote(run_group)}/not_a_run"
        raw_status, _ = get_json_capture(f"{server.base_url}/api/training/runs/{missing}/status")

    assert status["run_group"] == run_group
    assert status["run_name"] == run_name
    assert status["status"] == "completed"
    assert raw_status == 404


def test_run_group_route_percent_decodes_the_group_name(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    run_group = "grup\u00e9"
    run_dir = tmp_path / "runs" / run_group / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.6,0.7,0.4\n"
    )
    monkeypatch.setattr(ui.pipeline, "DEFAULT_RUNS_DIR", tmp_path / "runs")

    with UiServer(ui.TrainingPipelineHandler) as server:
        detail = get_json(f"{server.base_url}/api/training/run-groups/{urllib.parse.quote(run_group)}")

    assert detail["run_group"] == run_group
    assert [run["run_name"] for run in detail["runs"]] == ["run_a"]


def test_save_training_config_rejects_whitespace_in_run_names(tmp_path: Path, monkeypatch) -> None:
    ui, _ = prepare_training_workspace(tmp_path, monkeypatch)

    with UiServer(ui.TrainingPipelineHandler) as server:
        saved_selection = save_and_export_selection(server.base_url)
        base = {
            "dataset_name": "API Training Fixture", "dataset_version": "v1", "model": "yolov8n.pt",
            "image_size": 320, "epochs": 1, "batch": 2, "device": "cpu",
            "selection_path": saved_selection["selection_path"],
        }
        space_name_status, space_name = post_json_capture(
            server.base_url + "/api/training/configs", {**base, "run_group": "group_ok", "run_name": "run one"}
        )
        space_group_status, space_group = post_json_capture(
            server.base_url + "/api/training/configs", {**base, "run_group": "group one", "run_name": "run_ok"}
        )
        ok_status, _ = post_json_capture(
            server.base_url + "/api/training/configs", {**base, "run_group": "group_ok", "run_name": "run_ok"}
        )

    assert space_name_status == 400
    assert "run_name must not contain whitespace" in space_name["error"]
    assert space_group_status == 400
    assert "run_group must not contain whitespace" in space_group["error"]
    assert ok_status == 201
