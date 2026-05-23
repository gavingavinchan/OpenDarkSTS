from __future__ import annotations

import json
import threading
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
    monkeypatch.setattr(ui, "cache_url_for", lambda path: f"/cache/{Path(path).name}" if path else None)

    with UiServer(ui.TrainingPipelineHandler) as server:
        candidates = get_json(server.base_url + "/api/dataset-candidates")
        saved = post_json(
            server.base_url + "/api/dataset-selections",
            {"dataset_name": "API Fixture", "dataset_version": "v1", "image_keys": ["img_api"]},
        )
        listed = get_json(server.base_url + "/api/dataset-selections")

    assert candidates["candidate_count"] == 1
    assert candidates["candidates"][0]["image_key"] == "img_api"
    assert candidates["candidates"][0]["cache_url"] == "/cache/img_api.png"
    assert saved["selection"]["dataset_name"] == "API Fixture"
    assert Path(saved["selection_path"]).is_file()
    assert listed["selections"][0]["dataset_name"] == "API Fixture"


def test_step4_dataset_selection_page_route() -> None:
    from scripts import training_pipeline_ui as ui

    with UiServer(ui.TrainingPipelineHandler) as server:
        with urllib.request.urlopen(server.base_url + "/dataset-selection", timeout=5) as response:
            html = response.read().decode("utf-8")

    assert response.status == 200
    assert "Dataset Selection" in html
    assert "/api/dataset-candidates" in html
    assert "/api/dataset-selections" in html
    assert "training-ui-interaction" in html


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
