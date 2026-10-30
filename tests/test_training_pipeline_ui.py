from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from _helpers import detection, write_json, write_review


class UiServer:
    def __init__(self, handler) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_import_summary_includes_review_only_queue_images(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_manifest", review_status="overview_1")
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_reviewed", review_status="completed")
    archived_path = reviews_dir / "_archived_unreviewed" / "2026-05-04T00-00-00-000Z" / "tester" / "img_archived.json"
    write_json(archived_path, {"image_key": "img_archived", "review_status": "overview_1", "detections": []})
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "training_import",
            "created_at": "2026-05-04T00:00:00Z",
            "updated_at": "2026-05-04T00:00:00Z",
            "cache_dir": str(cache_dir),
            "images": {
                "img_manifest": {
                    "schema_version": 1,
                    "image_key": "img_manifest",
                    "original_filename": "img_manifest.png",
                    "source_path": str(tmp_path / "source.png"),
                    "browser_relative_path": None,
                    "source_collection": str(tmp_path),
                    "source_display_name": str(tmp_path),
                    "source_size_bytes": 1,
                    "source_mtime": 0,
                    "source_content_hash": "sha256:fixture",
                    "width": 32,
                    "height": 24,
                    "cache_path": str(cache_dir / "img_manifest.png"),
                    "cache_width": 32,
                    "cache_height": 24,
                    "cache_content_hash": "sha256:cache",
                    "render_profile": {"kind": "rgb_original"},
                    "loader": "pillow",
                    "import_status": "imported",
                    "imported_at": "2026-05-04T00:00:00Z",
                    "updated_at": "2026-05-04T00:00:00Z",
                    "warnings": [],
                },
                "img_archived": {
                    "schema_version": 1,
                    "image_key": "img_archived",
                    "original_filename": "img_archived.png",
                    "source_path": str(tmp_path / "archived.png"),
                    "browser_relative_path": None,
                    "source_collection": str(tmp_path),
                    "source_display_name": str(tmp_path),
                    "source_size_bytes": 1,
                    "source_mtime": 0,
                    "source_content_hash": "sha256:archived",
                    "width": 32,
                    "height": 24,
                    "cache_path": str(cache_dir / "img_archived.png"),
                    "cache_width": 32,
                    "cache_height": 24,
                    "cache_content_hash": "sha256:cache-archived",
                    "render_profile": {"kind": "rgb_original"},
                    "loader": "pillow",
                    "import_status": "imported",
                    "imported_at": "2026-05-04T00:00:00Z",
                    "updated_at": "2026-05-04T00:00:00Z",
                    "warnings": [],
                },
            },
        },
    )

    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ui, "REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(ui, "IMPORT_REVIEWER", "tester")

    summary = ui.manifest_summary()

    assert summary["image_count"] == 3
    assert summary["status_counts"]["imported"] == 2
    assert summary["status_counts"]["review_only"] == 1
    filters = {item["image_key"]: item["review_queue_filter"] for item in summary["images"]}
    assert filters == {"img_manifest": "unreviewed", "img_archived": "missing_review", "img_reviewed": "reviewed"}


def test_step4_api_lists_and_saves_dataset_selection(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_api", detections=[detection("auto-0001", "ship_non_sts")])
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "training_import",
            "created_at": "2026-05-19T00:00:00Z",
            "updated_at": "2026-05-19T00:00:00Z",
            "cache_dir": str(cache_dir),
            "images": {
                "img_api": {"image_key": "img_api", "original_filename": "img_api.png", "cache_path": str(cache_dir / "img_api.png"), "album_ids": []},
            },
        },
    )
    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ui, "REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(ui, "IMPORT_REVIEWER", "tester")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASET_SELECTIONS_DIR", tmp_path / "dataset_selections")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASETS_DIR", tmp_path / "datasets")

    with UiServer(ui.TrainingPipelineHandler) as server:
        candidates = get_json(server.base_url + "/api/dataset-candidates")
        saved = post_json(
            server.base_url + "/api/dataset-selections",
            {"dataset_name": "API Fixture", "dataset_version": "v1", "image_keys": ["img_api"]},
        )
        listed = get_json(server.base_url + "/api/dataset-selections")

    assert candidates["candidate_count"] == 1
    assert candidates["candidates"][0]["image_key"] == "img_api"
    assert saved["selection"]["dataset_name"] == "API Fixture"
    assert Path(saved["selection_path"]).is_file()
    assert listed["selections"][0]["dataset_name"] == "API Fixture"


def test_step4_api_split_preview_reports_scene_groups(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    images = {}
    keys = []
    for date in ("2020-01-14", "2020-02-15", "2020-03-16"):
        for tile in range(3):
            key = f"img_{date}_{tile}"
            name = f"sentinel2__S2_{date}_tile_{tile}_0.png"
            keys.append(key)
            write_review(reviews_dir, cache_dir, reviewer="tester", image_key=key, image_filename=name)
            images[key] = {
                "image_key": key,
                "original_filename": name,
                "browser_relative_path": name,
                "source_path": "/Sentinel 2A/sentinel2/" + name,
                "album_ids": [],
            }
    write_json(manifest_path, {"schema_version": 1, "manifest_type": "training_import", "cache_dir": str(cache_dir), "images": images})
    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ui, "REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(ui, "IMPORT_REVIEWER", "tester")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASET_SELECTIONS_DIR", tmp_path / "dataset_selections")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASETS_DIR", tmp_path / "datasets")

    with UiServer(ui.TrainingPipelineHandler) as server:
        preview = post_json(server.base_url + "/api/dataset-selections/split-preview", {"image_keys": keys})
        legacy = post_json(
            server.base_url + "/api/dataset-selections/split-preview",
            {"image_keys": keys, "split_mode": "image"},
        )
        saved = post_json(
            server.base_url + "/api/dataset-selections",
            {"dataset_name": "Scene Fixture", "dataset_version": "v1", "image_keys": keys},
        )
        zero_preview = post_json(
            server.base_url + "/api/dataset-selections/split-preview",
            {"image_keys": keys, "seed": 0},
        )
        zero_saved = post_json(
            server.base_url + "/api/dataset-selections",
            {"dataset_name": "Seed Zero Fixture", "image_keys": keys, "seed": 0},
        )

    assert preview["split_mode"] == "group"
    assert preview["group_count"] == 3
    assert preview["groups_spanning_splits"] == 0
    assert legacy["groups_spanning_splits"] > 0
    assert saved["split_summary"]["groups_spanning_splits"] == 0
    assert saved["split_summary"]["group_count"] == 3
    assert zero_preview["seed"] == 0
    assert zero_saved["selection"]["split_policy"]["seed"] == 0
    assert {item["image_key"]: item["split"] for item in zero_preview["items"]} == {
        item["image_key"]: item["split"] for item in zero_saved["selection"]["items"]
    }


def test_step4_api_exports_dataset_selection(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "api_fixture_v1"
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_export", detections=[detection("auto-0001", "ship_non_sts")])
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "training_import",
            "created_at": "2026-05-19T00:00:00Z",
            "updated_at": "2026-05-19T00:00:00Z",
            "cache_dir": str(cache_dir),
            "images": {
                "img_export": {"image_key": "img_export", "original_filename": "img_export.png", "cache_path": str(cache_dir / "img_export.png"), "album_ids": []},
            },
        },
    )
    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ui, "REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(ui, "IMPORT_REVIEWER", "tester")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASET_SELECTIONS_DIR", tmp_path / "dataset_selections")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASETS_DIR", tmp_path / "datasets")

    with UiServer(ui.TrainingPipelineHandler) as server:
        saved = post_json(
            server.base_url + "/api/dataset-selections",
            {"dataset_name": "API Fixture", "dataset_version": "v1", "image_keys": ["img_export"]},
        )
        exported = post_json(server.base_url + "/api/dataset-selections/export", {"selection_path": saved["selection_path"]})

    assert exported["dataset_dir"] == str(output_dir)
    assert exported["exported_image_count"] == 1
    assert (output_dir / "dataset_manifest.json").is_file()
    assert (output_dir / "images" / "train" / "img_export.png").is_file()
    assert (output_dir / "labels" / "train" / "img_export.txt").is_file()


# ---------------------------------------------------------------------------
# Steps 5-7 API smoke tests
# ---------------------------------------------------------------------------


def post_json_capture(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def get_json_capture(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def prepare_training_workspace(tmp_path: Path, monkeypatch) -> tuple[object, Path]:
    from scripts import training_pipeline_ui as ui

    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    images = {}
    for image_key in ("img_train", "img_val"):
        write_review(reviews_dir, cache_dir, reviewer="tester", image_key=image_key, detections=[detection("auto-0001", "ship_non_sts")])
        images[image_key] = {"image_key": image_key, "original_filename": f"{image_key}.png", "cache_path": str(cache_dir / f"{image_key}.png"), "album_ids": []}
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "training_import",
            "created_at": "2026-05-19T00:00:00Z",
            "updated_at": "2026-05-19T00:00:00Z",
            "cache_dir": str(cache_dir),
            "images": images,
        },
    )
    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ui, "REVIEWS_DIR", reviews_dir)
    monkeypatch.setattr(ui, "IMPORT_REVIEWER", "tester")
    monkeypatch.setattr(ui, "TRAINING_UI_INTERACTION_LOG_PATH", tmp_path / "reports" / "interactions.jsonl")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASET_SELECTIONS_DIR", tmp_path / "dataset_selections")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_DATASETS_DIR", tmp_path / "datasets")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_TRAINING_CONFIGS_DIR", tmp_path / "training_configs")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(ui.pipeline, "DEFAULT_MODEL_WEIGHTS_DIR", tmp_path / "model_weights")
    return ui, reviews_dir


def save_and_export_selection(base_url: str) -> dict:
    saved = post_json(
        base_url + "/api/dataset-selections",
        {"dataset_name": "API Training Fixture", "dataset_version": "v1", "image_keys": ["img_train", "img_val"], "split_ratios": {"train": 0.5, "val": 0.5, "test": 0.0}},
    )
    post_json(base_url + "/api/dataset-selections/export", {"selection_path": saved["selection_path"]})
    return saved


def test_step5_api_saves_training_config_and_lists_setup_inputs(tmp_path: Path, monkeypatch) -> None:
    ui, _ = prepare_training_workspace(tmp_path, monkeypatch)

    with UiServer(ui.TrainingPipelineHandler) as server:
        saved_selection = save_and_export_selection(server.base_url)
        status, saved_config = post_json_capture(
            server.base_url + "/api/training/configs",
            {
                "dataset_name": "API Training Fixture",
                "dataset_version": "v1",
                "model": "yolov8n.pt",
                "run_group": "api_fixture_group",
                "run_name": "api_fixture_run",
                "image_size": 320,
                "epochs": 1,
                "batch": 2,
                "seed": 0,
                "device": "cpu",
                "augmentation": {"fliplr": 0.5},
                "selection_path": saved_selection["selection_path"],
            },
        )
        configs = get_json(server.base_url + "/api/training/configs")
        datasets = get_json(server.base_url + "/api/training/exported-datasets")
        weights = get_json(server.base_url + "/api/training/starting-weights")
        readiness = post_json(
            server.base_url + "/api/training/readiness",
            {"selection_path": saved_selection["selection_path"], "config_path": saved_config["config_path"]},
        )
        preview = post_json(
            server.base_url + "/api/training/command-preview",
            {"selection_path": saved_selection["selection_path"], "config_path": saved_config["config_path"]},
        )
        bad_aug_status, bad_aug = post_json_capture(
            server.base_url + "/api/training/configs",
            {"dataset_name": "x", "model": "yolov8n.pt", "run_group": "g", "run_name": "r", "augmentation": {"not_real": 1}},
        )

    assert status == 201
    assert saved_config["config"]["augmentation"] == {"fliplr": 0.5}
    assert saved_config["config"]["seed"] == 0
    assert configs["configs"][0]["run_name"] == "api_fixture_run"
    assert datasets["datasets"][0]["split_counts"]["train"] == 1
    assert isinstance(weights["weights"], list)
    assert readiness["ready"] is True
    assert preview["command_line"][:3] == ["yolo", "detect", "train"]
    assert "imgsz=320" in preview["command_line"]
    assert "seed=0" in preview["command_line"]
    assert bad_aug_status == 400
    assert "not_real" in bad_aug["error"]
    interactions = (tmp_path / "reports" / "interactions.jsonl").read_text()
    assert "save_training_config" in interactions


def test_step6_api_starts_polls_and_stops_training_run(tmp_path: Path, monkeypatch) -> None:
    ui, _ = prepare_training_workspace(tmp_path, monkeypatch)
    fake_yolo = tmp_path / "bin" / "yolo.py"
    fake_yolo.parent.mkdir(parents=True)
    fake_yolo.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "from pathlib import Path\n"
        "args = dict(part.split('=', 1) for part in sys.argv[1:] if '=' in part)\n"
        "run_dir = Path(args['project']) / args['name']\n"
        "run_dir.mkdir(parents=True, exist_ok=True)\n"
        "print('starting')\n"
        "sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    fake_yolo.chmod(0o755)
    monkeypatch.setattr(ui.pipeline, "resolve_yolo_executable", lambda executable=None: str(fake_yolo))

    with UiServer(ui.TrainingPipelineHandler) as server:
        saved_selection = save_and_export_selection(server.base_url)
        _, saved_config = post_json_capture(
            server.base_url + "/api/training/configs",
            {
                "dataset_name": "API Training Fixture", "dataset_version": "v1", "model": "yolov8n.pt",
                "run_group": "api_fixture_group", "run_name": "api_fixture_run", "image_size": 320,
                "epochs": 1, "batch": 2, "device": "cpu",
                "selection_path": saved_selection["selection_path"],
            },
        )
        start_status, started = post_json_capture(
            server.base_url + "/api/training/runs/start",
            {"selection_path": saved_selection["selection_path"], "config_path": saved_config["config_path"]},
        )
        active = get_json(server.base_url + "/api/training/runs/active")
        run_status = get_json(server.base_url + "/api/training/runs/api_fixture_group/api_fixture_run/status")
        _, stopped = post_json_capture(server.base_url + "/api/training/runs/api_fixture_group/api_fixture_run/stop", {})
        after_stop = get_json(server.base_url + "/api/training/runs/active")
        missing_status, missing = post_json_capture(server.base_url + "/api/training/runs/no_group/no_run/stop", {})

    assert start_status == 202
    assert started["status"] == "running"
    assert active["runs"][0]["run_name"] == "api_fixture_run"
    assert run_status["status"] == "running"
    assert run_status["alive"] is True
    assert stopped["status"] == "cancelled"
    assert after_stop["runs"] == []
    assert missing_status == 404
    assert "No pipeline-managed training run" in missing["error"]
    interactions = (tmp_path / "reports" / "interactions.jsonl").read_text()
    assert "start_training" in interactions and "stop_training" in interactions


def test_step6_api_refuses_a_second_start_into_a_live_run_dir(tmp_path: Path, monkeypatch) -> None:
    ui, _ = prepare_training_workspace(tmp_path, monkeypatch)
    fake_yolo = tmp_path / "bin" / "yolo.py"
    fake_yolo.parent.mkdir(parents=True)
    fake_yolo.write_text("#!/bin/sh\nexec sleep 120\n")
    fake_yolo.chmod(0o755)
    monkeypatch.setattr(ui.pipeline, "resolve_yolo_executable", lambda executable=None: str(fake_yolo))

    with UiServer(ui.TrainingPipelineHandler) as server:
        saved_selection = save_and_export_selection(server.base_url)
        _, saved_config = post_json_capture(
            server.base_url + "/api/training/configs",
            {
                "dataset_name": "API Training Fixture", "dataset_version": "v1", "model": "yolov8n.pt",
                "run_group": "api_dup_group", "run_name": "api_dup_run", "image_size": 320,
                "epochs": 1, "batch": 2, "device": "cpu", "overwrite": True,
                "selection_path": saved_selection["selection_path"],
            },
        )
        body = {"selection_path": saved_selection["selection_path"], "config_path": saved_config["config_path"]}
        first_status, first = post_json_capture(server.base_url + "/api/training/runs/start", body)
        second_status, second = post_json_capture(server.base_url + "/api/training/runs/start", body)
        record = json.loads((tmp_path / "runs" / "api_dup_group" / "api_dup_run" / "process.json").read_text())
        post_json_capture(server.base_url + "/api/training/runs/api_dup_group/api_dup_run/stop", {})

    assert first_status == 202
    assert second_status == 409
    assert second["already_running"] is True
    assert second["pid"] == first["pid"]
    # The refused launch must not have replaced the live process record.
    assert record["pid"] == first["pid"]


def test_step7_api_writes_evaluation_report_and_refuses_undocumented_promotion(tmp_path: Path, monkeypatch) -> None:
    ui, _ = prepare_training_workspace(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "api_eval_group" / "api_eval_run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"fake-best")
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.6,0.7,0.4\n"
    )
    monkeypatch.setattr(ui.pipeline, "WORKSPACE_DIR", tmp_path)

    with UiServer(ui.TrainingPipelineHandler) as server:
        report_status, report = post_json_capture(server.base_url + "/api/training/evaluation-report", {"run_group": "api_eval_group"})
        missing_status, missing = post_json_capture(server.base_url + "/api/training/evaluation-report", {"run_group": "no_such_group"})
        refuse_status, refused = post_json_capture(
            server.base_url + "/api/training/promote-weight",
            {"run_group": "api_eval_group", "run_name": "api_eval_run", "target_name": "definitely_not_documented_fixture.pt"},
        )

    assert report_status == 201
    assert report["run_count"] == 1
    assert report["selected_run_name"] == "api_eval_run"
    assert report["promoted_run_name"] is None
    assert Path(report["report_path"]).name == "api_eval_group_evaluation.json"
    assert missing_status == 400
    assert "Run group not found" in missing["error"]
    assert refuse_status == 400
    assert "recreation_doc_updated" in refused["error"]
    assert not (tmp_path / "model_weights").exists()
    interactions = (tmp_path / "reports" / "interactions.jsonl").read_text()
    assert "promote_weight_refused" in interactions


def get_text(url: str) -> tuple[int, str, str]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read().decode("utf-8")


def test_template_loader_reads_every_page_template() -> None:
    from scripts import training_pipeline_ui as ui

    templates = ui.load_templates()
    for name in ("bbox_overview", "import", "datasets", "training_setup", "training", "evaluation"):
        assert name in templates, f"missing template {name}"
        assert templates[name].lstrip().startswith("<!DOCTYPE html>")
    assert ui.template_text("datasets") is templates["datasets"]


def test_render_template_substitutes_placeholders() -> None:
    from scripts import training_pipeline_ui as ui

    html = ui.render_template("datasets", stage_nav="<nav id='probe'></nav>")
    assert "__STAGE_NAV__" not in html
    assert "<nav id='probe'></nav>" in html


def test_stage_nav_covers_all_seven_stages_and_marks_current() -> None:
    from scripts import training_pipeline_ui as ui

    nav = ui.render_stage_nav(4)
    for href in ("/import", "/bbox-overview", "/datasets", "/training-setup", "/training", "/evaluation"):
        assert f'href="{href}"' in nav
    assert 'class="stage active" href="/datasets"' in nav
    assert nav.count("data-stage=") == 7


def test_every_pipeline_page_renders_200_with_nav(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    monkeypatch.setattr(ui, "REVIEWS_DIR", tmp_path / "reviews")
    monkeypatch.setattr(ui, "IMPORT_MANIFEST_PATH", tmp_path / "manifests" / "training_import_manifest.json")

    pages = {
        "/import": "Intake",
        "/bbox-overview": "Reviewed bounding-box overview",
        "/datasets": "Dataset Selection",
        "/training-setup": "Training Setup",
        "/training": "Train detector",
        "/evaluation": "Evaluate",
    }
    with UiServer(ui.TrainingPipelineHandler) as server:
        for path, marker in pages.items():
            status, content_type, body = get_text(f"{server.base_url}{path}")
            assert status == 200, path
            assert "text/html" in content_type, path
            assert marker in body, path
            assert 'aria-label="Training pipeline stages"' in body, path
            assert 'href="/evaluation"' in body, path
            assert "__STAGE_NAV__" not in body, path


def test_new_pages_serve_shared_static_assets(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    with UiServer(ui.TrainingPipelineHandler) as server:
        status, content_type, css = get_text(f"{server.base_url}/static/pipeline.css")
        assert status == 200 and "css" in content_type
        assert ".stage.active" in css
        status, _, js = get_text(f"{server.base_url}/static/pipeline.js")
        assert status == 200
        assert "training-ui-interaction" in js
        try:
            get_text(f"{server.base_url}/static/../training_pipeline_ui.py")
            raise AssertionError("path traversal should not be served")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404


def test_page_aliases_match_the_plan_route_names(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_ui as ui

    with UiServer(ui.TrainingPipelineHandler) as server:
        for alias, marker in (("/dataset-selection", "Dataset Selection"), ("/train-yolo", "Train detector"), ("/evaluate", "Evaluate")):
            status, _, body = get_text(f"{server.base_url}{alias}")
            assert status == 200 and marker in body


def test_run_group_routes_back_the_evaluation_page(tmp_path: Path, monkeypatch) -> None:
    from scripts import training_pipeline_lib as pipeline
    from scripts import training_pipeline_ui as ui

    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "group_a" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,train/box_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.8,0.7,0.6,0.3\n"
    )
    monkeypatch.setattr(pipeline, "DEFAULT_RUNS_DIR", runs_dir)

    with UiServer(ui.TrainingPipelineHandler) as server:
        groups = get_json(f"{server.base_url}/api/training/run-groups")
        assert [item["run_group"] for item in groups["run_groups"]] == ["group_a"]
        detail = get_json(f"{server.base_url}/api/training/run-groups/group_a")
        assert detail["run_count"] == 1
        assert detail["suggested_run_name"] == "run_1"
        assert detail["runs"][0]["validation_metrics"]["mAP50"] == 0.6
        report = get_json(f"{server.base_url}/api/training/evaluation-report?run_group=group_a")
        assert report["exists"] is False


def test_unknown_job_id_returns_404(tmp_path: Path) -> None:
    from scripts import training_pipeline_ui as ui

    with UiServer(ui.TrainingPipelineHandler) as server:
        try:
            get_json(f"{server.base_url}/api/jobs/does-not-exist")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
