from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from _helpers import detection, load_training_lib, write_json, write_review, write_selection, write_training_config


ROOT_DIR = Path(__file__).resolve().parent.parent
CLI = ROOT_DIR / "scripts" / "training_pipeline_cli.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_envelope(process: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(process.stdout)


def assert_envelope(envelope: dict, *, command: str, ok: bool) -> None:
    assert envelope["ok"] is ok
    assert envelope["command"] == command
    assert envelope["schema_version"] == 1
    assert isinstance(envelope["result"], dict)
    assert isinstance(envelope["warnings"], list)
    assert isinstance(envelope["errors"], list)


def write_png(path: Path, *, size: tuple[int, int] = (8, 6)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def test_cli_import_data_writes_manifest_and_browser_cache(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"

    process = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
    )

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=True)
    result = envelope["result"]
    assert result["processed_count"] == 1
    assert result["imported_count"] == 1
    assert result["skipped_count"] == 0
    assert result["review_stubs_created_count"] == 1
    assert result["review_stubs_existing_count"] == 0
    assert result["review_stubs_skipped_count"] == 0
    assert result["import_job_id"].startswith("imp_")
    assert result["album"]["album_id"].startswith("alb_")
    imported = result["imported"][0]
    assert imported["image_key"].startswith("img_")
    assert imported["render_profile_kind"] == "rgb_original"
    assert imported["album_id"] == result["album"]["album_id"]
    assert imported["member_status"] == "added_imported"
    assert Path(imported["cache_path"]).is_file()
    assert imported["review_stub_status"] == "created"
    assert imported["review_stub_path"] == str(reviews_dir / "tester" / f"{imported['image_key']}.json")
    assert Path(result["import_log_path"]).is_file()
    assert Path(result["import_summary_path"]).is_file()

    album_path = Path(result["album"]["album_path"])
    album = json.loads(album_path.read_text())
    assert album["album_id"] == result["album"]["album_id"]
    assert album["album_name"] == "source.png"
    assert album["source_display_name"] == str(source)
    assert album["counts"]["members_added_imported"] == 1
    assert album["images"][0]["image_key"] == imported["image_key"]
    assert album["images"][0]["member_status"] == "added_imported"
    album_index = json.loads(Path(result["album"]["album_index_path"]).read_text())
    assert album_index["albums"][0]["album_id"] == album["album_id"]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["manifest_type"] == "training_import"
    assert manifest["created_at"]
    assert manifest["updated_at"]
    assert manifest["cache_dir"] == str(cache_dir)
    entry = manifest["images"][imported["image_key"]]
    assert entry["schema_version"] == 1
    assert entry["image_filename"] == f"{imported['image_key']}.png"
    assert entry["original_filename"] == "source.png"
    assert entry["display_filename"] == "source.png"
    assert entry["source_path"] == str(source)
    assert entry["browser_relative_path"] is None
    assert entry["source_collection"] == str(source)
    assert entry["source_display_name"] == str(source)
    assert entry["album_ids"] == [album["album_id"]]
    assert "album_names" not in entry
    assert entry["cache_path"] == imported["cache_path"]
    assert entry["width"] == 8
    assert entry["height"] == 6
    assert entry["cache_width"] == 8
    assert entry["cache_height"] == 6
    assert entry["cache_content_hash"].startswith("sha256:")
    assert entry["loader"] == "pillow"
    assert entry["import_status"] == "imported"
    assert entry["imported_at"]
    assert entry["updated_at"]
    assert entry["warnings"] == []

    review = json.loads((reviews_dir / "tester" / f"{imported['image_key']}.json").read_text())
    assert review["image_key"] == imported["image_key"]
    assert review["image_filename"] == f"{imported['image_key']}.png"
    assert review["original_filename"] == "source.png"
    assert review["display_filename"] == "source.png"
    assert review["reviewer"] == "tester"
    assert review["cache_path"] == imported["cache_path"]
    assert review["width"] == 8
    assert review["height"] == 6
    assert review["review_status"] == "overview_1"
    assert review["detections"] == []
    assert "album_ids" not in review


def test_import_data_auto_detect_writes_initial_review_bboxes(tmp_path: Path, monkeypatch) -> None:
    pipeline = load_training_lib()
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"

    monkeypatch.setattr(pipeline.review_lib, "load_yolo_models", lambda model_paths: {"yolov8n": object()})
    monkeypatch.setattr(
        pipeline.review_lib,
        "run_model_predictions",
        lambda image_path, models, conf_threshold: [
            {
                "temp_id": "yolov8n:1",
                "source_model": "yolov8n",
                "class_id": 0,
                "class_name": "ship",
                "confidence": 0.91,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.25,
                "height": 0.2,
            }
        ],
    )

    result = pipeline.import_training_data(
        [source],
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        reviews_dir=reviews_dir,
        reviewer="tester",
        auto_detect=True,
    )

    imported = result["imported"][0]
    assert imported["auto_detect_status"] == "created_review"
    assert imported["auto_detection_count"] == 1
    assert result["auto_detect_created_review_count"] == 1
    review = json.loads((reviews_dir / "tester" / f"{imported['image_key']}.json").read_text())
    assert len(review["detections"]) == 1
    assert review["detections"][0]["detection_id"] == "auto-0001"
    assert review["detections"][0]["source_models"] == ["yolov8n"]
    assert review["detections"][0]["reviewed"] is False


def test_cli_import_data_skips_existing_identical_image(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    first = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--reviews-dir", str(reviews_dir))
    assert first.returncode == 0

    second = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--reviews-dir", str(reviews_dir))

    assert second.returncode == 0
    envelope = parse_envelope(second)
    assert_envelope(envelope, command="import-data", ok=True)
    assert envelope["result"]["imported_count"] == 0
    assert envelope["result"]["skipped_count"] == 1
    assert envelope["result"]["skipped"][0]["status"] == "already_imported"


def test_cli_import_data_adds_existing_image_to_new_album(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    first = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--album-name",
        "First album",
    )
    assert first.returncode == 0
    first_result = parse_envelope(first)["result"]

    second = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--album-name",
        "Second album",
        "--album-description",
        "Known image reuse",
    )

    assert second.returncode == 0
    result = parse_envelope(second)["result"]
    assert result["imported_count"] == 0
    assert result["skipped_count"] == 1
    skipped = result["skipped"][0]
    assert skipped["member_status"] == "added_existing"
    assert skipped["album_id"] != first_result["album"]["album_id"]
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["images"][skipped["image_key"]]
    assert entry["album_ids"] == [first_result["album"]["album_id"], result["album"]["album_id"]]
    album = json.loads(Path(result["album"]["album_path"]).read_text())
    assert album["album_name"] == "Second album"
    assert album["album_description"] == "Known image reuse"
    assert album["counts"]["members_added_existing"] == 1


def test_cli_import_data_appends_to_existing_album_by_id(tmp_path: Path) -> None:
    source_a = write_png(tmp_path / "inputs" / "a.png")
    source_b = write_png(tmp_path / "inputs" / "b.png", size=(9, 7))
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    first = run_cli(
        "import-data",
        "--source",
        str(source_a),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--album-name",
        "Append target",
    )
    assert first.returncode == 0
    album_id = parse_envelope(first)["result"]["album"]["album_id"]

    second = run_cli(
        "import-data",
        "--source",
        str(source_b),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--append-to-album-id",
        album_id,
    )

    assert second.returncode == 0
    result = parse_envelope(second)["result"]
    assert result["album"]["album_id"] == album_id
    album = json.loads(Path(result["album"]["album_path"]).read_text())
    assert album["album_name"] == "Append target"
    assert len([image for image in album["images"] if image.get("image_key")]) == 2
    assert len(album["import_job_ids"]) == 2


def test_cli_add_to_album_allows_partial_failures(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    imported = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
    )
    assert imported.returncode == 0
    image_key = parse_envelope(imported)["result"]["imported"][0]["image_key"]
    created = run_cli(
        "create-album",
        "--album-name",
        "Partial add target",
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
    )
    assert created.returncode == 0
    album_id = parse_envelope(created)["result"]["album_id"]

    process = run_cli(
        "add-to-album",
        "--album-id",
        album_id,
        "--image-key",
        image_key,
        "--image-key",
        "img_missing",
        "--manifest",
        str(manifest_path),
    )

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="add-to-album", ok=True)
    assert envelope["warnings"] == ["1 image_key(s) could not be added; see result.failed"]
    result = envelope["result"]
    assert result["members_added_existing"] == 1
    assert result["members_failed"] == 1
    assert result["failed"] == [{"image_key": "img_missing", "reason": "not in import manifest"}]


def test_cli_add_to_album_fails_when_every_key_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    created = run_cli(
        "create-album",
        "--album-name",
        "All failed target",
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
    )
    assert created.returncode == 0
    album_id = parse_envelope(created)["result"]["album_id"]

    process = run_cli(
        "add-to-album",
        "--album-id",
        album_id,
        "--image-key",
        "img_missing_a",
        "--image-key",
        "img_missing_b",
        "--manifest",
        str(manifest_path),
    )

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="add-to-album", ok=False)
    assert envelope["errors"] == ["No image_key could be added to album"]
    assert envelope["warnings"] == ["2 image_key(s) could not be added; see result.failed"]
    result = envelope["result"]
    assert result["requested_count"] == 2
    assert result["members_added_existing"] == 0
    assert result["members_failed"] == 2


def test_cli_import_data_creates_missing_stub_for_already_imported_image(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    first = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--no-review-stubs",
    )
    assert first.returncode == 0

    second = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
    )

    assert second.returncode == 0
    envelope = parse_envelope(second)
    assert_envelope(envelope, command="import-data", ok=True)
    result = envelope["result"]
    assert result["imported_count"] == 0
    assert result["skipped_count"] == 1
    assert result["review_stubs_created_count"] == 1
    skipped = result["skipped"][0]
    assert skipped["status"] == "already_imported"
    assert skipped["review_stub_status"] == "created"
    assert Path(skipped["review_stub_path"]).is_file()


def test_cli_import_data_treats_review_stub_write_failure_as_failed_item(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    reviews_dir.write_text("not a directory\n")

    process = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
    )

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=False)
    assert envelope["result"]["imported_count"] == 0
    assert envelope["result"]["failed_count"] == 1
    failed = envelope["result"]["failed"][0]
    assert failed["review_stub_status"] == "failed"
    assert "Review stub failed" in failed["reason"]


def test_cli_import_data_dry_run_does_not_write_manifest_or_cache(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"

    process = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--dry-run")

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=True)
    assert envelope["result"]["imported"][0]["status"] == "imported"
    assert not manifest_path.exists()
    assert not cache_dir.exists()


def test_cli_import_data_regenerates_missing_cache_with_documented_status(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    first = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--reviews-dir", str(reviews_dir))
    assert first.returncode == 0
    first_item = parse_envelope(first)["result"]["imported"][0]
    Path(first_item["cache_path"]).unlink()

    second = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--reviews-dir", str(reviews_dir))

    assert second.returncode == 0
    envelope = parse_envelope(second)
    assert_envelope(envelope, command="import-data", ok=True)
    assert envelope["result"]["imported_count"] == 1
    assert envelope["result"]["imported"][0]["status"] == "regenerated_cache"
    entry = json.loads(manifest_path.read_text())["images"][first_item["image_key"]]
    assert entry["import_status"] == "regenerated_cache"
    assert Path(entry["cache_path"]).is_file()


def test_cli_import_data_folder_scan_skips_hidden_and_unsupported_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    write_png(input_dir / "visible.png")
    write_png(input_dir / ".hidden.png")
    (input_dir / "notes.txt").write_text("not an image\n")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"

    process = run_cli("import-data", "--source", str(input_dir), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir), "--reviews-dir", str(reviews_dir))

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=True)
    assert envelope["result"]["processed_count"] == 1
    assert envelope["result"]["imported_count"] == 1
    item = envelope["result"]["imported"][0]
    entry = json.loads(manifest_path.read_text())["images"][item["image_key"]]
    assert entry["source_collection"] == str(input_dir)


def test_cli_import_data_missing_sources_exit_1(tmp_path: Path) -> None:
    process = run_cli("import-data", "--source", str(tmp_path / "missing.png"), "--manifest", str(tmp_path / "manifest.json"), "--cache-dir", str(tmp_path / "cache"))

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=False)
    assert envelope["result"]["failed_count"] == 1
    assert "No input source" in envelope["errors"][0]


def test_cli_prune_review_queue_dry_run_protects_reviewed_and_ambiguous_files(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    safe = write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_safe", review_status="overview_1")
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_done", review_status="completed")
    write_review(
        reviews_dir,
        cache_dir,
        reviewer="tester",
        image_key="img_manual",
        review_status="overview_1",
        detections=[{**detection("manual-0001", "ship_non_sts"), "source_type": "manual", "reviewed": False, "review_label": None}],
    )
    corrupt = reviews_dir / "tester" / "img_corrupt.json"
    corrupt.write_text("{not-json\n")

    process = run_cli(
        "prune-review-queue",
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--image-key",
        "img_safe",
        "--image-key",
        "img_done",
        "--image-key",
        "img_manual",
        "--image-key",
        "img_corrupt",
        "--image-key",
        "img_missing",
        "--dry-run",
    )

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="prune-review-queue", ok=True)
    result = envelope["result"]
    assert result["requested_count"] == 5
    assert result["archivable_count"] == 1
    assert result["protected_count"] == 4
    assert result["archivable"][0]["image_key"] == "img_safe"
    reasons = {item["image_key"]: item["reason"] for item in result["protected"]}
    assert "review_status" in reasons["img_done"]
    assert "manual" in reasons["img_manual"]
    assert "unreadable" in reasons["img_corrupt"]
    assert "missing" in reasons["img_missing"]
    assert safe.is_file()


def test_cli_prune_review_queue_archives_only_safe_unreviewed_stubs(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    reports_dir = tmp_path / "reports" / "review_queue_prune"
    cache_dir = tmp_path / "cache"
    safe = write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_safe", review_status="overview_1")
    protected = write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_done", review_status="completed")

    process = run_cli(
        "prune-review-queue",
        "--reviews-dir",
        str(reviews_dir),
        "--reports-dir",
        str(reports_dir),
        "--reviewer",
        "tester",
        "--image-key",
        "img_safe",
        "--image-key",
        "img_done",
    )

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="prune-review-queue", ok=True)
    result = envelope["result"]
    assert result["archived_count"] == 1
    assert not safe.exists()
    assert protected.is_file()
    archived_path = Path(result["archived"][0]["archive_path"])
    assert archived_path.is_file()
    assert "_archived_unreviewed" in archived_path.parts
    report_path = Path(result["report_path"])
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["archived"][0]["image_key"] == "img_safe"
    assert report["protected"][0]["image_key"] == "img_done"


def test_cli_prune_review_queue_filter_unreviewed_intersects_explicit_keys(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_safe", review_status="overview_1")
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_done", review_status="completed")

    process = run_cli(
        "prune-review-queue",
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--filter",
        "unreviewed",
        "--image-key",
        "img_safe",
        "--image-key",
        "img_done",
        "--dry-run",
    )

    assert process.returncode == 0
    result = parse_envelope(process)["result"]
    assert [item["image_key"] for item in result["archivable"]] == ["img_safe"]
    assert [item["image_key"] for item in result["protected"]] == ["img_done"]


def test_review_queue_filter_summary_classifies_step1_filters(tmp_path: Path) -> None:
    pipeline = load_training_lib()
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_unreviewed", review_status="overview_1")
    write_review(reviews_dir, cache_dir, reviewer="tester", image_key="img_reviewed", review_status="completed")
    write_review(
        reviews_dir,
        cache_dir,
        reviewer="tester",
        image_key="img_partial",
        review_status="overview_1",
        detections=[detection("auto-0001", "ship_non_sts")],
    )
    archived_path = reviews_dir / "_archived_unreviewed" / "2026-05-04T00-00-00-000Z" / "tester" / "img_archived.json"
    write_json(archived_path, {"image_key": "img_archived", "review_status": "overview_1", "detections": []})

    assert pipeline.review_queue_filter_summary("img_unreviewed", "tester", reviews_dir)["review_queue_filter"] == "unreviewed"
    assert pipeline.review_queue_filter_summary("img_unreviewed", "tester", reviews_dir)["unreviewed_archivable"] is True
    assert pipeline.review_queue_filter_summary("img_reviewed", "tester", reviews_dir)["review_queue_filter"] == "reviewed"
    assert pipeline.review_queue_filter_summary("img_partial", "tester", reviews_dir)["review_queue_filter"] == "partially_reviewed"
    assert pipeline.review_queue_filter_summary("img_archived", "tester", reviews_dir)["review_queue_filter"] == "missing_review"
    assert pipeline.review_queue_filter_summary("img_missing", "tester", reviews_dir)["review_queue_filter"] == "missing_review"


def test_cli_remove_images_removes_manifest_album_cache_and_reviews(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    reviews_dir = tmp_path / "reviews"
    reports_dir = tmp_path / "reports" / "image_removals"
    import_process = run_cli(
        "import-data",
        "--source",
        str(source),
        "--manifest",
        str(manifest_path),
        "--cache-dir",
        str(cache_dir),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--album-name",
        "Removal fixture",
    )
    assert import_process.returncode == 0
    import_result = parse_envelope(import_process)["result"]
    image_key = import_result["imported"][0]["image_key"]
    cache_path = Path(import_result["imported"][0]["cache_path"])
    review_path = reviews_dir / "tester" / f"{image_key}.json"
    album_path = Path(import_result["album"]["album_path"])
    review_doc = json.loads(review_path.read_text())
    review_doc["review_status"] = "completed"
    write_json(review_path, review_doc)

    dry_run = run_cli(
        "remove-images",
        "--manifest",
        str(manifest_path),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--reports-dir",
        str(reports_dir),
        "--image-key",
        image_key,
        "--dry-run",
    )
    assert dry_run.returncode == 0
    assert parse_envelope(dry_run)["result"]["removable_count"] == 1
    assert cache_path.is_file()
    assert review_path.is_file()

    process = run_cli(
        "remove-images",
        "--manifest",
        str(manifest_path),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--reports-dir",
        str(reports_dir),
        "--image-key",
        image_key,
    )

    assert process.returncode == 0
    result = parse_envelope(process)["result"]
    assert result["removed_count"] == 1
    assert not cache_path.exists()
    assert not review_path.exists()
    assert image_key not in json.loads(manifest_path.read_text())["images"]
    assert all(image.get("image_key") != image_key for image in json.loads(album_path.read_text())["images"])
    assert Path(result["report_path"]).is_file()


def test_cli_remove_images_filter_returns_zero_when_no_inactive_reviews(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    reviews_dir = tmp_path / "reviews"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "manifest_type": "training_import",
            "created_at": "2026-05-05T00:00:00Z",
            "updated_at": "2026-05-05T00:00:00Z",
            "cache_dir": str(tmp_path / "cache"),
            "images": {},
        },
    )

    process = run_cli(
        "remove-images",
        "--manifest",
        str(manifest_path),
        "--reviews-dir",
        str(reviews_dir),
        "--reviewer",
        "tester",
        "--filter",
        "inactive-review",
        "--dry-run",
    )

    assert process.returncode == 0
    result = parse_envelope(process)["result"]
    assert result["requested_count"] == 0
    assert result["removable_count"] == 0


def test_cli_check_training_ready_exits_3_with_blockers(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 3
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="check-training-ready", ok=True)
    assert envelope["result"]["ready"] is False
    assert envelope["result"]["missing"]


def test_cli_export_dataset_writes_yolo_layout_and_returns_paths_json(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[{"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
    )

    process = run_cli("export-dataset", "--selection", str(selection_path))

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="export-dataset", ok=True)
    assert envelope["result"]["dataset_dir"] == str(output_dir)
    assert envelope["result"]["dataset_manifest"] == str(output_dir / "dataset_manifest.json")
    assert (output_dir / "images" / "train" / "img_train.png").is_file()
    assert (output_dir / "labels" / "train" / "img_train.txt").is_file()


def test_cli_build_train_command_prints_command_list_json(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[{"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
    )
    output_dir.mkdir(parents=True)
    (output_dir / "data.yaml").write_text("names:\n  0: ship\n  1: ships\n")
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=tmp_path / "runs")

    process = run_cli("build-train-command", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="build-train-command", ok=True)
    assert envelope["result"]["command_line"][:3] == ["yolo", "detect", "train"]
    assert envelope["result"]["data_yaml"] == str(output_dir / "data.yaml")


def test_cli_unknown_command_exits_2_with_json_error_envelope() -> None:
    process = run_cli("not-a-real-command")

    assert process.returncode == 2
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="not-a-real-command", ok=False)
    assert envelope["errors"]


def test_cli_help_does_not_emit_json_error_envelope() -> None:
    process = run_cli("--help")

    assert process.returncode == 0
    assert "Training pipeline automation CLI" in process.stdout
    assert '"ok"' not in process.stdout
    assert process.stderr == ""


def test_cli_envelope_has_no_nested_result_warnings(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    envelope = parse_envelope(process)
    assert "warnings" in envelope
    assert "warnings" not in envelope["result"]


def test_cli_export_dry_run_with_overrides_does_not_mutate_selection(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    before = selection_path.read_bytes()

    process = run_cli(
        "export-dataset",
        "--selection",
        str(selection_path),
        "--output-dir",
        str(tmp_path / "datasets" / "override_dataset"),
        "--include-incomplete",
        "--dry-run",
    )

    assert process.returncode == 0
    assert selection_path.read_bytes() == before
    envelope = parse_envelope(process)
    assert envelope["result"]["dataset_dir"] == str(tmp_path / "datasets" / "override_dataset")
    assert envelope["result"]["exported_image_count"] == 0


def test_cli_export_dry_run_reports_selected_image_count(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[
            {"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None},
            {"image_key": "img_val", "split": "val", "include_labels": "selected_only", "selected_detection_ids": ["auto-0001", "auto-0002"]},
        ],
    )

    process = run_cli("export-dataset", "--selection", str(selection_path), "--dry-run")

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert envelope["result"]["exported_image_count"] == 2
    assert envelope["result"]["exported_label_count"] == 2


def test_cli_export_with_output_override_does_not_mutate_selection(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    original_output_dir = tmp_path / "datasets" / "fixture_dataset"
    override_output_dir = tmp_path / "datasets" / "override_dataset"
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=original_output_dir,
        items=[{"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
    )
    before = selection_path.read_bytes()

    process = run_cli("export-dataset", "--selection", str(selection_path), "--output-dir", str(override_output_dir))

    assert process.returncode == 0
    assert selection_path.read_bytes() == before
    assert (override_output_dir / "dataset_manifest.json").is_file()
    assert not (original_output_dir / "dataset_manifest.json").exists()


def test_cli_build_train_command_missing_data_yaml_exits_1(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = write_training_config(tmp_path / "training_configs" / "fixture_group.json", output_dir=tmp_path / "runs")

    process = run_cli("build-train-command", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="build-train-command", ok=False)
    assert "data.yaml" in envelope["errors"][0]


def test_cli_build_train_command_missing_run_name_has_friendly_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "datasets" / "fixture_dataset"
    output_dir.mkdir(parents=True)
    (output_dir / "data.yaml").write_text("names:\n  0: ship\n  1: ships\n")
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=output_dir,
        items=[],
    )
    config_path = tmp_path / "training_configs" / "missing_run_name.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "dataset_name": "fixture_dataset",
            "dataset_version": 1,
            "model": "yolov8n.pt",
            "image_size": 960,
            "epochs": 2,
            "batch": 4,
            "device": "cpu",
            "seed": 42,
            "augmentation": {},
        },
    )

    process = run_cli("build-train-command", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="build-train-command", ok=False)
    assert "output run name" in envelope["errors"][0]


def test_cli_pretty_prints_indented_json(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("--pretty", "check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 3
    assert "\n  \"command\": \"check-training-ready\"" in process.stdout
    assert parse_envelope(process)["command"] == "check-training-ready"


def test_cli_human_outputs_non_json_text(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("--human", "check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 3
    assert process.stdout.startswith("check-training-ready: ok")
    assert '"schema_version"' not in process.stdout


def test_cli_human_export_includes_counts(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "datasets" / "fixture_dataset"
    write_review(reviews_dir, cache_dir, reviewer="gavin", image_key="img_train", detections=[detection("auto-0001", "ship_non_sts")])
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=reviews_dir,
        output_dir=output_dir,
        items=[{"image_key": "img_train", "split": "train", "include_labels": "all_exportable", "selected_detection_ids": None}],
    )

    process = run_cli("--human", "export-dataset", "--selection", str(selection_path))

    assert process.returncode == 0
    assert "dataset_dir: " + str(output_dir) in process.stdout
    assert "exported_image_count: 1" in process.stdout


def test_cli_verbose_flag_is_accepted(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("--verbose", "check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 3
    envelope = parse_envelope(process)
    assert envelope["command"] == "check-training-ready"
    assert envelope["result"]["diagnostics"]["helper"] == "check_training_readiness"
    assert envelope["result"]["diagnostics"]["selection_path"] == str(selection_path.resolve())


def test_cli_non_verbose_omits_diagnostics(tmp_path: Path) -> None:
    selection_path = write_selection(
        tmp_path / "dataset_selections" / "fixture_dataset.json",
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "datasets" / "fixture_dataset",
        items=[],
    )
    config_path = tmp_path / "training_configs" / "empty.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")

    process = run_cli("check-training-ready", "--selection", str(selection_path), "--config", str(config_path))

    assert process.returncode == 3
    assert "diagnostics" not in parse_envelope(process)["result"]


def test_cli_image_key_returns_stable_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"stable image bytes")
    profile = tmp_path / "render_profile.json"
    write_json(profile, {"kind": "rgb_original"})

    process = run_cli("image-key", "--source", str(source), "--render-profile", str(profile))

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="image-key", ok=True)
    assert envelope["result"]["source"] == str(source)
    assert envelope["result"]["source_content_hash"].startswith("sha256:")
    assert envelope["result"]["render_profile"] == {"kind": "rgb_original"}
    assert envelope["result"]["image_key"].startswith("img_")


def test_cli_image_key_missing_source_exits_1(tmp_path: Path) -> None:
    profile = tmp_path / "render_profile.json"
    write_json(profile, {"kind": "rgb_original"})

    process = run_cli("image-key", "--source", str(tmp_path / "missing.png"), "--render-profile", str(profile))

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="image-key", ok=False)
    assert "Source image not found" in envelope["errors"][0]


def test_cli_image_key_rejects_unknown_render_profile_kind(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"stable image bytes")
    profile = tmp_path / "render_profile.json"
    write_json(profile, {"kind": "rgb-orignal"})

    process = run_cli("image-key", "--source", str(source), "--render-profile", str(profile))

    assert process.returncode == 1
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="image-key", ok=False)
    assert "Unknown render profile kind" in envelope["errors"][0]
