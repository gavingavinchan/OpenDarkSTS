from __future__ import annotations

from pathlib import Path

from _helpers import write_json, write_review


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
