from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from _helpers import detection, write_json, write_review, write_selection, write_training_config


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

    process = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))

    assert process.returncode == 0
    envelope = parse_envelope(process)
    assert_envelope(envelope, command="import-data", ok=True)
    result = envelope["result"]
    assert result["processed_count"] == 1
    assert result["imported_count"] == 1
    assert result["skipped_count"] == 0
    imported = result["imported"][0]
    assert imported["image_key"].startswith("img_")
    assert imported["render_profile_kind"] == "rgb_original"
    assert Path(imported["cache_path"]).is_file()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["manifest_type"] == "training_import"
    assert manifest["created_at"]
    assert manifest["updated_at"]
    assert manifest["cache_dir"] == str(cache_dir)
    entry = manifest["images"][imported["image_key"]]
    assert entry["schema_version"] == 1
    assert entry["original_filename"] == "source.png"
    assert entry["source_path"] == str(source)
    assert entry["browser_relative_path"] is None
    assert entry["source_collection"] == str(source)
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


def test_cli_import_data_skips_existing_identical_image(tmp_path: Path) -> None:
    source = write_png(tmp_path / "inputs" / "source.png")
    manifest_path = tmp_path / "manifests" / "training_import_manifest.json"
    cache_dir = tmp_path / "cache" / "imported_images"
    first = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))
    assert first.returncode == 0

    second = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))

    assert second.returncode == 0
    envelope = parse_envelope(second)
    assert_envelope(envelope, command="import-data", ok=True)
    assert envelope["result"]["imported_count"] == 0
    assert envelope["result"]["skipped_count"] == 1
    assert envelope["result"]["skipped"][0]["status"] == "already_imported"


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
    first = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))
    assert first.returncode == 0
    first_item = parse_envelope(first)["result"]["imported"][0]
    Path(first_item["cache_path"]).unlink()

    second = run_cli("import-data", "--source", str(source), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))

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

    process = run_cli("import-data", "--source", str(input_dir), "--manifest", str(manifest_path), "--cache-dir", str(cache_dir))

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
