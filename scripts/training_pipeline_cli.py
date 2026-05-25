#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from . import training_pipeline_lib as pipeline
    from . import visual_review_lib as review_lib
except ImportError:
    import training_pipeline_lib as pipeline
    import visual_review_lib as review_lib


SCHEMA_VERSION = 1


def envelope(command: str, *, ok: bool, result: dict[str, Any] | None = None, warnings: list[str] | None = None, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "ok": ok,
        "command": command,
        "schema_version": SCHEMA_VERSION,
        "result": result or {},
        "warnings": warnings or [],
        "errors": errors or [],
    }


def print_json(payload: dict[str, Any], *, pretty: bool = False) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=pretty))


def print_human(payload: dict[str, Any]) -> None:
    status = "ok" if payload.get("ok") else "error"
    print(f"{payload.get('command')}: {status}")
    for warning in payload.get("warnings") or []:
        print(f"warning: {warning}")
    for error in payload.get("errors") or []:
        print(f"error: {error}")
    result = payload.get("result") or {}
    if "ready" in result:
        print(f"ready: {result['ready']}")
    for item in result.get("missing") or []:
        print(f"missing: {item}")
    for key in (
        "manifest_path",
        "cache_dir",
        "processed_count",
        "imported_count",
        "skipped_count",
        "failed_count",
        "review_stubs_created_count",
        "review_stubs_existing_count",
        "review_stubs_skipped_count",
        "auto_detect_created_review_count",
        "auto_detect_updated_existing_review_count",
        "auto_detect_protected_existing_review_count",
        "auto_detect_skipped_count",
        "created_review_count",
        "updated_existing_review_count",
        "protected_existing_review_count",
        "dataset_dir",
        "dataset_manifest",
        "data_yaml",
        "exported_image_count",
        "exported_label_count",
        "run_dir",
        "image_key",
        "source_content_hash",
        "requested_count",
        "removable_count",
        "removed_count",
        "archivable_count",
        "protected_count",
        "archived_count",
        "report_path",
        "state",
        "pid",
        "log_path",
        "process_metadata",
        "training_metadata",
    ):
        if key in result:
            print(f"{key}: {result[key]}")
    if "command_line" in result:
        print("command_line: " + " ".join(str(part) for part in result["command_line"]))


def print_response(payload: dict[str, Any], *, pretty: bool = False, human: bool = False) -> None:
    if human:
        print_human(payload)
    else:
        print_json(payload, pretty=pretty)


def selection_dataset_dir(selection_path: str | Path) -> Path:
    selection = pipeline.load_json(selection_path)
    output_dir = selection.get("export", {}).get("output_dir")
    if not output_dir:
        raise ValueError("Dataset selection manifest is missing export.output_dir")
    return pipeline.repo_resolve(output_dir)


def load_config_argument(value: str) -> dict[str, Any]:
    if value.lstrip().startswith("{"):
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("--config must be a JSON object or a path to a JSON object")
        return payload
    candidate = Path(value)
    if candidate.exists() or value.endswith(".json"):
        return pipeline.load_json(candidate)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--config must be a JSON object or a path to a JSON object")
    return payload


def parse_roots_option(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_selection_for_config(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    selection = getattr(args, "selection", None) or config.get("selection_path")
    if not selection:
        raise ValueError("Dataset selection is required; pass --selection or include selection_path in the training config")
    return pipeline.repo_resolve(selection)


def add_diagnostics(result: dict[str, Any], args: argparse.Namespace, helper: str, **paths: str | Path) -> None:
    if not getattr(args, "verbose", False):
        return
    diagnostics: dict[str, Any] = {"helper": helper}
    for key, value in paths.items():
        diagnostics[key] = str(pipeline.repo_resolve(value))
    result["diagnostics"] = diagnostics


def parse_model_path_args(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Model weights must be provided as name=path")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("Model weights must be provided as name=path")
        parsed[name] = path
    return parsed


def parse_split_ratios(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    ratios: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError("split ratios must be comma-separated key=value pairs")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split ratio key: {key}")
        ratios[key] = float(raw_value.strip())
    return ratios


def command_export_dataset(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    selection = pipeline.load_json(args.selection)
    if args.output_dir:
        selection.setdefault("export", {})["output_dir"] = args.output_dir
    if args.include_incomplete:
        selection.setdefault("export", {})["include_incomplete_reviews"] = True

    if args.dry_run:
        dataset_dir = pipeline.repo_resolve(selection.get("export", {}).get("output_dir"))
        selected_items = selection.get("items") or []
        selected_only_label_count = sum(
            len(item.get("selected_detection_ids") or [])
            for item in selected_items
            if item.get("include_labels") == "selected_only"
        )
        result = {
            "dataset_dir": str(dataset_dir),
            "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
            "data_yaml": str(dataset_dir / "data.yaml"),
            "exported_image_count": len(selected_items),
            "exported_label_count": selected_only_label_count,
        }
        add_diagnostics(result, args, "export_dataset_selection", selection_path=args.selection, dataset_dir=dataset_dir)
        return 0, envelope("export-dataset", ok=True, result=result)

    manifest = pipeline.export_dataset_selection(
        args.selection,
        output_dir_override=args.output_dir,
        include_incomplete_override=True if args.include_incomplete else None,
    )
    dataset_dir = pipeline.repo_resolve(selection.get("export", {}).get("output_dir"))
    exported_files = manifest.get("exported_files") or []
    result = {
        "dataset_dir": str(dataset_dir),
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "data_yaml": str(dataset_dir / "data.yaml"),
        "exported_image_count": len(manifest.get("exported_image_keys") or []),
        "exported_label_count": sum(1 for item in exported_files if Path(str(item.get("path", ""))).parts[:1] == ("labels",) and str(item.get("path", "")).endswith(".txt")),
    }
    add_diagnostics(result, args, "export_dataset_selection", selection_path=args.selection, dataset_dir=dataset_dir)
    return 0, envelope("export-dataset", ok=True, result=result, warnings=list(manifest.get("warnings") or []))


def command_list_dataset_candidates(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.list_dataset_candidates(
        reviews_dir=args.reviews_dir,
        manifest_path=args.manifest,
        albums_dir=args.albums_dir,
        reviewer=args.reviewer,
        album_ids=list(args.album_id or []),
        required_labels=list(args.label or []),
        completed_only=not bool(args.include_incomplete),
    )
    add_diagnostics(result, args, "list_dataset_candidates", reviews_dir=args.reviews_dir, manifest_path=args.manifest)
    return 0, envelope("list-dataset-candidates", ok=True, result=result)


def command_save_dataset_selection(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.save_dataset_selection(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        description=args.description,
        image_keys=list(args.image_key or []),
        selections_dir=args.selections_dir,
        datasets_dir=args.datasets_dir,
        reviews_dir=args.reviews_dir,
        manifest_path=args.manifest,
        reviewer=args.reviewer,
        album_ids=list(args.album_id or []),
        seed=int(args.seed),
        split_ratios=parse_split_ratios(args.split_ratios),
        created_by=args.created_by,
    )
    payload = {
        "selection_path": result["selection_path"],
        "dataset_name": result["selection"]["dataset_name"],
        "dataset_version": result["selection"]["dataset_version"],
        "item_count": len(result["selection"].get("items") or []),
        "export_output_dir": result["selection"].get("export", {}).get("output_dir"),
    }
    add_diagnostics(payload, args, "save_dataset_selection", selections_dir=args.selections_dir)
    return 0, envelope("save-dataset-selection", ok=True, result=payload)


def command_list_dataset_selections(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = {"selections": pipeline.list_dataset_selections(args.selections_dir)}
    add_diagnostics(result, args, "list_dataset_selections", selections_dir=args.selections_dir)
    return 0, envelope("list-dataset-selections", ok=True, result=result)


def command_list_exported_datasets(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = {"datasets": pipeline.list_exported_datasets(args.selections_dir)}
    add_diagnostics(result, args, "list_exported_datasets", selections_dir=args.selections_dir)
    return 0, envelope("list-exported-datasets", ok=True, result=result)


def command_list_starting_weights(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = {"weights": pipeline.list_starting_weights(parse_roots_option(args.weights_roots), parse_roots_option(args.docs_roots))}
    add_diagnostics(result, args, "list_starting_weights")
    return 0, envelope("list-starting-weights", ok=True, result=result)


def command_list_training_configs(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = {"configs": pipeline.list_training_configs(args.configs_dir)}
    add_diagnostics(result, args, "list_training_configs", configs_dir=args.configs_dir)
    return 0, envelope("list-training-configs", ok=True, result=result)


def command_save_training_config(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = load_config_argument(args.config)
    saved = pipeline.save_training_config(
        config,
        configs_dir=args.configs_dir,
        weights_roots=parse_roots_option(args.weights_roots),
        docs_roots=parse_roots_option(args.docs_roots),
    )
    result = {
        "config_path": saved["config_path"],
        "config": saved["config"],
        "provenance_warnings": saved.get("provenance_warnings") or [],
    }
    add_diagnostics(result, args, "save_training_config", configs_dir=args.configs_dir)
    return 0, envelope("save-training-config", ok=True, result=result, warnings=list(saved.get("provenance_warnings") or []))


def command_device_info(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.probe_training_devices()
    add_diagnostics(result, args, "probe_training_devices")
    return 0, envelope("device-info", ok=True, result=result)


def command_import_data(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.import_training_data(
        list(args.source),
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        albums_dir=args.albums_dir,
        import_jobs_dir=args.import_jobs_dir,
        album_name=args.album_name,
        album_description=args.album_description,
        source_display_name=args.source_display_name,
        append_to_album_id=args.append_to_album_id,
        resume_job_id=args.resume_job,
        render_profile_value=args.render_profile,
        reviews_dir=args.reviews_dir,
        reviewer=args.reviewer,
        create_review_stubs=not bool(args.no_review_stubs),
        auto_detect=bool(args.auto_detect),
        auto_detect_model_paths=parse_model_path_args(args.auto_detect_weight),
        auto_detect_confidence=float(args.auto_detect_conf),
        auto_detect_iou=float(args.auto_detect_iou),
        recursive=bool(args.recursive),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    warnings = [f"Failed to import {item.get('source')}: {item.get('reason')}" for item in result.get("failed") or []]
    add_diagnostics(result, args, "import_training_data", manifest_path=args.manifest, cache_dir=args.cache_dir)
    if int(result.get("imported_count") or 0) + int(result.get("skipped_count") or 0) == 0:
        return 1, envelope("import-data", ok=False, result=result, warnings=warnings, errors=["No input source could be processed successfully"])
    return 0, envelope("import-data", ok=True, result=result, warnings=warnings)


def command_run_auto_detect(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.run_auto_detect_on_imported_reviews(
        manifest_path=args.manifest,
        reviews_dir=args.reviews_dir,
        reviewer=args.reviewer,
        image_keys=list(args.image_key or []),
        limit=int(args.limit or 0),
        model_paths=parse_model_path_args(args.weight),
        confidence=float(args.conf),
        iou=float(args.iou),
        dry_run=bool(args.dry_run),
    )
    add_diagnostics(result, args, "run_auto_detect_on_imported_reviews", manifest_path=args.manifest, reviews_dir=args.reviews_dir)
    if int(result.get("failed_count") or 0) > 0:
        return 1, envelope("run-auto-detect", ok=False, result=result, errors=["One or more images failed auto detection"])
    return 0, envelope("run-auto-detect", ok=True, result=result)


def command_check_training_ready(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    readiness = pipeline.check_training_readiness(args.selection, args.config)
    result = {"ready": readiness["ready"], "missing": readiness["missing"]}
    if "runs" in readiness:
        result["runs"] = readiness["runs"]
    add_diagnostics(result, args, "check_training_readiness", selection_path=args.selection or "", config_path=args.config)
    response = envelope("check-training-ready", ok=True, result=result, warnings=list(readiness.get("warnings") or []))
    return (0 if readiness["ready"] else 3), response


def command_build_train_command(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = pipeline.load_json(args.config)
    missing = pipeline.config_missing_items(config)
    if missing:
        raise ValueError("; ".join(missing))
    selection_path = resolve_selection_for_config(args, config)
    dataset_dir = selection_dataset_dir(selection_path)
    built = pipeline.build_yolo_train_command(config, dataset_dir)
    commands = built if built and isinstance(built[0], list) else [built]
    run_group = str(config.get("run_group") or "default")
    output_root = pipeline.repo_resolve(config.get("output_dir") or pipeline.WORKSPACE_DIR / "runs")
    result = {
        "commands": commands,
        "run_dirs": [str(output_root / run_group / str(run["run_name"])) for run in pipeline.expand_training_config_runs(config)],
        "data_yaml": str(dataset_dir / "data.yaml"),
    }
    if len(commands) == 1:
        result["command_line"] = commands[0]
        result["run_dir"] = result["run_dirs"][0]
    add_diagnostics(result, args, "build_yolo_train_command", selection_path=selection_path, config_path=args.config, dataset_dir=dataset_dir)
    return 0, envelope("build-train-command", ok=True, result=result)


def command_start_training(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    try:
        result = pipeline.start_training_run(
            args.selection,
            args.config,
            run_name=args.run_name,
            dry_run=bool(args.dry_run),
            foreground=bool(args.foreground),
        )
    except pipeline.TrainingRunSelectionError as exc:
        return 2, envelope(
            "start-training",
            ok=False,
            result={"available_run_names": exc.available_run_names},
            errors=[str(exc)],
        )
    add_diagnostics(result, args, "start_training_run", selection_path=args.selection or "", config_path=args.config)
    warnings = list(result.get("warnings") or [])
    if result.get("state") == "blocked":
        return 3, envelope("start-training", ok=True, result=result, warnings=warnings)
    return 0, envelope("start-training", ok=True, result=result, warnings=warnings)


def command_training_status(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.training_status(args.run_dir, log_lines=int(args.log_lines))
    add_diagnostics(result, args, "training_status", run_dir=args.run_dir)
    return 0, envelope("training-status", ok=True, result=result, warnings=list(result.get("warnings") or []))


def command_stop_training(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    try:
        result = pipeline.stop_training_run(args.run_dir)
    except pipeline.TrainingRunStateError as exc:
        return 4, envelope("stop-training", ok=False, errors=[str(exc)])
    add_diagnostics(result, args, "stop_training_run", run_dir=args.run_dir)
    return 0, envelope("stop-training", ok=True, result=result, warnings=list(result.get("warnings") or []))


def command_image_key(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    render_profile = pipeline.load_json(args.render_profile)
    source_path = pipeline.repo_resolve(args.source)
    if not source_path.is_file():
        raise ValueError(f"Source image not found: {source_path}")
    review_lib.validate_render_profile(render_profile)
    result = {
        "source": str(source_path),
        "source_content_hash": review_lib.source_content_hash(source_path),
        "render_profile": render_profile,
        "image_key": review_lib.image_key_for_source(source_path, render_profile),
    }
    add_diagnostics(result, args, "image_key_for_source", source_path=source_path, render_profile_path=args.render_profile)
    return 0, envelope("image-key", ok=True, result=result)


def command_prune_review_queue(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.prune_review_queue(
        reviewer=args.reviewer,
        reviews_dir=args.reviews_dir,
        image_keys=list(args.image_key or []),
        filter_mode=args.filter,
        dry_run=bool(args.dry_run),
        reports_dir=args.reports_dir,
    )
    add_diagnostics(result, args, "prune_review_queue", reviews_dir=args.reviews_dir)
    return 0, envelope("prune-review-queue", ok=True, result=result)


def resolved_albums_dir(args: argparse.Namespace) -> str:
    return str(args.albums_dir or pipeline.default_albums_dir_for_manifest(args.manifest))


def command_list_albums(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    summaries = pipeline.list_albums_summary(
        albums_dir,
        manifest_path=args.manifest,
        reviews_dir=args.reviews_dir,
        reviewer=args.reviewer,
    )
    result = {
        "albums_dir": pipeline.repo_relative(albums_dir),
        "album_count": len(summaries),
        "albums": summaries,
    }
    add_diagnostics(result, args, "list_albums_summary", albums_dir=albums_dir)
    return 0, envelope("list-albums", ok=True, result=result)


def command_show_album(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    detail = pipeline.load_album_detail(
        albums_dir,
        args.album_id,
        manifest_path=args.manifest,
        reviews_dir=args.reviews_dir,
        reviewer=args.reviewer,
    )
    add_diagnostics(detail, args, "load_album_detail", albums_dir=albums_dir)
    return 0, envelope("show-album", ok=True, result=detail)


def command_create_album(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    record = pipeline.create_empty_album(
        albums_dir=albums_dir,
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        reviews_dir=args.reviews_dir,
        album_name=args.album_name,
        album_description=args.album_description,
        source_display_name=args.source_display_name,
        reviewer=args.reviewer,
    )
    result = {
        "album_id": record["album_id"],
        "album_name": record["album_name"],
        "album_description": record["album_description"],
        "source_kind": record["source_kind"],
        "album_path": pipeline.repo_relative(pipeline.repo_resolve(albums_dir) / f"{record['album_id']}.json"),
        "image_count": 0,
    }
    add_diagnostics(result, args, "create_empty_album", albums_dir=albums_dir)
    return 0, envelope("create-album", ok=True, result=result)


def command_add_to_album(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    image_keys = list(args.image_key or [])
    if not image_keys:
        return 2, envelope("add-to-album", ok=False, errors=["At least one --image-key is required"])
    result = pipeline.add_existing_image_keys_to_album(
        albums_dir=albums_dir,
        manifest_path=args.manifest,
        album_id=args.album_id,
        image_keys=image_keys,
        reviewer=args.reviewer,
    )
    warnings = []
    if result.get("members_failed"):
        warnings.append(f"{result['members_failed']} image_key(s) could not be added; see result.failed")
    add_diagnostics(result, args, "add_existing_image_keys_to_album", albums_dir=albums_dir, manifest_path=args.manifest)
    if int(result.get("members_failed") or 0) == int(result.get("requested_count") or 0):
        return 1, envelope("add-to-album", ok=False, result=result, warnings=warnings, errors=["No image_key could be added to album"])
    return 0, envelope("add-to-album", ok=True, result=result, warnings=warnings)


def command_rename_album(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    if args.album_name is None and args.album_description is None:
        return 2, envelope("rename-album", ok=False, errors=["At least one of --album-name or --album-description is required"])
    try:
        record = pipeline.rename_album(
            albums_dir=albums_dir,
            manifest_path=args.manifest,
            album_id=args.album_id,
            album_name=args.album_name,
            album_description=args.album_description,
        )
    except ValueError as exc:
        return 1, envelope("rename-album", ok=False, errors=[str(exc)])
    result = {
        "album_id": record["album_id"],
        "album_name": record["album_name"],
        "album_description": record["album_description"],
        "updated_at": record["updated_at"],
        "created_at": record["created_at"],
    }
    add_diagnostics(result, args, "rename_album", albums_dir=albums_dir, manifest_path=args.manifest)
    return 0, envelope("rename-album", ok=True, result=result)


def command_remove_from_album(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    albums_dir = resolved_albums_dir(args)
    image_keys = list(args.image_key or [])
    if not image_keys:
        return 2, envelope("remove-from-album", ok=False, errors=["At least one --image-key is required"])
    result = pipeline.remove_image_keys_from_album(
        albums_dir=albums_dir,
        manifest_path=args.manifest,
        album_id=args.album_id,
        image_keys=image_keys,
    )
    add_diagnostics(result, args, "remove_image_keys_from_album", albums_dir=albums_dir, manifest_path=args.manifest)
    return 0, envelope("remove-from-album", ok=True, result=result)


def command_remove_images(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.remove_images(
        reviewer=args.reviewer,
        reviews_dir=args.reviews_dir,
        manifest_path=args.manifest,
        albums_dir=args.albums_dir,
        image_keys=list(args.image_key or []),
        filter_mode=args.filter,
        dry_run=bool(args.dry_run),
        reports_dir=args.reports_dir,
    )
    add_diagnostics(result, args, "remove_images", manifest_path=args.manifest, reviews_dir=args.reviews_dir)
    return 0, envelope("remove-images", ok=True, result=result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training pipeline automation CLI")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--human", action="store_true", help="Print concise human-readable output")
    parser.add_argument("--verbose", action="store_true", help="Include additional diagnostics when available")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import-data")
    import_parser.add_argument("--source", action="append", required=True)
    import_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    import_parser.add_argument("--cache-dir", default=str(pipeline.DEFAULT_IMPORT_CACHE_DIR))
    import_parser.add_argument("--albums-dir")
    import_parser.add_argument("--import-jobs-dir")
    import_parser.add_argument("--album-name")
    import_parser.add_argument("--album-description")
    import_parser.add_argument("--source-display-name")
    import_parser.add_argument("--append-to-album-id")
    import_parser.add_argument("--resume-job")
    import_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    import_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    import_parser.add_argument("--render-profile", default="auto")
    import_parser.add_argument("--auto-detect", action="store_true", help="Run YOLO and create/enrich safe review stubs with auto bboxes")
    import_parser.add_argument("--auto-detect-weight", action="append", help="YOLO weight as name=path; repeatable. Defaults to the current YOLOv8n auto-detect weight")
    import_parser.add_argument("--auto-detect-conf", type=float, default=pipeline.DEFAULT_AUTO_DETECT_CONFIDENCE)
    import_parser.add_argument("--auto-detect-iou", type=float, default=pipeline.DEFAULT_AUTO_DETECT_IOU)
    import_parser.add_argument("--recursive", action="store_true")
    import_parser.add_argument("--force", action="store_true")
    import_parser.add_argument("--no-review-stubs", action="store_true")
    import_parser.add_argument("--dry-run", action="store_true")
    import_parser.set_defaults(handler=command_import_data)

    auto_detect_parser = subparsers.add_parser("run-auto-detect")
    auto_detect_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    auto_detect_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    auto_detect_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    auto_detect_parser.add_argument("--image-key", action="append")
    auto_detect_parser.add_argument("--limit", type=int, default=0)
    auto_detect_parser.add_argument("--weight", action="append", help="YOLO weight as name=path; repeatable. Defaults to the current YOLOv8n auto-detect weight")
    auto_detect_parser.add_argument("--conf", type=float, default=pipeline.DEFAULT_AUTO_DETECT_CONFIDENCE)
    auto_detect_parser.add_argument("--iou", type=float, default=pipeline.DEFAULT_AUTO_DETECT_IOU)
    auto_detect_parser.add_argument("--dry-run", action="store_true")
    auto_detect_parser.set_defaults(handler=command_run_auto_detect)

    export_parser = subparsers.add_parser("export-dataset")
    export_parser.add_argument("--selection", required=True)
    export_parser.add_argument("--output-dir")
    export_parser.add_argument("--include-incomplete", action="store_true")
    export_parser.add_argument("--dry-run", action="store_true")
    export_parser.set_defaults(handler=command_export_dataset)

    candidates_parser = subparsers.add_parser("list-dataset-candidates")
    candidates_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    candidates_parser.add_argument("--albums-dir")
    candidates_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    candidates_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    candidates_parser.add_argument("--album-id", action="append")
    candidates_parser.add_argument("--label", action="append", choices=["ship", "ships", "neither", "exportable"])
    candidates_parser.add_argument("--include-incomplete", action="store_true")
    candidates_parser.set_defaults(handler=command_list_dataset_candidates)

    save_selection_parser = subparsers.add_parser("save-dataset-selection")
    save_selection_parser.add_argument("--dataset-name", required=True)
    save_selection_parser.add_argument("--dataset-version", default="v1")
    save_selection_parser.add_argument("--description")
    save_selection_parser.add_argument("--image-key", action="append", required=True)
    save_selection_parser.add_argument("--album-id", action="append")
    save_selection_parser.add_argument("--seed", type=int, default=42)
    save_selection_parser.add_argument("--split-ratios", help="Comma-separated train=0.8,val=0.1,test=0.1")
    save_selection_parser.add_argument("--selections-dir", default=str(pipeline.DEFAULT_DATASET_SELECTIONS_DIR))
    save_selection_parser.add_argument("--datasets-dir", default=str(pipeline.DEFAULT_DATASETS_DIR))
    save_selection_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    save_selection_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    save_selection_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    save_selection_parser.add_argument("--created-by")
    save_selection_parser.set_defaults(handler=command_save_dataset_selection)

    list_selections_parser = subparsers.add_parser("list-dataset-selections")
    list_selections_parser.add_argument("--selections-dir", default=str(pipeline.DEFAULT_DATASET_SELECTIONS_DIR))
    list_selections_parser.set_defaults(handler=command_list_dataset_selections)

    list_exported_parser = subparsers.add_parser("list-exported-datasets")
    list_exported_parser.add_argument("--selections-dir", default=str(pipeline.DEFAULT_DATASET_SELECTIONS_DIR))
    list_exported_parser.set_defaults(handler=command_list_exported_datasets)

    list_weights_parser = subparsers.add_parser("list-starting-weights")
    list_weights_parser.add_argument("--weights-roots")
    list_weights_parser.add_argument("--docs-roots")
    list_weights_parser.set_defaults(handler=command_list_starting_weights)

    list_configs_parser = subparsers.add_parser("list-training-configs")
    list_configs_parser.add_argument("--configs-dir", default=str(pipeline.DEFAULT_TRAINING_CONFIGS_DIR))
    list_configs_parser.set_defaults(handler=command_list_training_configs)

    save_config_parser = subparsers.add_parser("save-training-config")
    save_config_parser.add_argument("--config", required=True)
    save_config_parser.add_argument("--configs-dir", default=str(pipeline.DEFAULT_TRAINING_CONFIGS_DIR))
    save_config_parser.add_argument("--weights-roots")
    save_config_parser.add_argument("--docs-roots")
    save_config_parser.set_defaults(handler=command_save_training_config)

    device_parser = subparsers.add_parser("device-info")
    device_parser.set_defaults(handler=command_device_info)

    ready_parser = subparsers.add_parser("check-training-ready")
    ready_parser.add_argument("--selection")
    ready_parser.add_argument("--config", required=True)
    ready_parser.set_defaults(handler=command_check_training_ready)

    command_parser = subparsers.add_parser("build-train-command")
    command_parser.add_argument("--selection")
    command_parser.add_argument("--config", required=True)
    command_parser.set_defaults(handler=command_build_train_command)

    start_training_parser = subparsers.add_parser("start-training")
    start_training_parser.add_argument("--selection")
    start_training_parser.add_argument("--config", required=True)
    start_training_parser.add_argument("--run-name")
    start_training_parser.add_argument("--foreground", action="store_true")
    start_training_parser.add_argument("--dry-run", action="store_true")
    start_training_parser.set_defaults(handler=command_start_training)

    training_status_parser = subparsers.add_parser("training-status")
    training_status_parser.add_argument("--run-dir", required=True)
    training_status_parser.add_argument("--log-lines", type=int, default=80)
    training_status_parser.set_defaults(handler=command_training_status)

    stop_training_parser = subparsers.add_parser("stop-training")
    stop_training_parser.add_argument("--run-dir", required=True)
    stop_training_parser.set_defaults(handler=command_stop_training)

    image_key_parser = subparsers.add_parser("image-key")
    image_key_parser.add_argument("--source", required=True)
    image_key_parser.add_argument("--render-profile", required=True)
    image_key_parser.set_defaults(handler=command_image_key)

    prune_parser = subparsers.add_parser("prune-review-queue")
    prune_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    prune_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    prune_parser.add_argument("--reports-dir", default=str(pipeline.DEFAULT_REVIEW_QUEUE_PRUNE_REPORTS_DIR))
    prune_parser.add_argument("--image-key", action="append")
    prune_parser.add_argument("--filter", choices=["unreviewed"])
    prune_parser.add_argument("--dry-run", action="store_true")
    prune_parser.set_defaults(handler=command_prune_review_queue)

    list_albums_parser = subparsers.add_parser("list-albums")
    list_albums_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    list_albums_parser.add_argument("--albums-dir")
    list_albums_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    list_albums_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    list_albums_parser.set_defaults(handler=command_list_albums)

    show_album_parser = subparsers.add_parser("show-album")
    show_album_parser.add_argument("--album-id", required=True)
    show_album_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    show_album_parser.add_argument("--albums-dir")
    show_album_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    show_album_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    show_album_parser.set_defaults(handler=command_show_album)

    create_album_parser = subparsers.add_parser("create-album")
    create_album_parser.add_argument("--album-name", required=True)
    create_album_parser.add_argument("--album-description")
    create_album_parser.add_argument("--source-display-name")
    create_album_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    create_album_parser.add_argument("--cache-dir", default=str(pipeline.DEFAULT_IMPORT_CACHE_DIR))
    create_album_parser.add_argument("--albums-dir")
    create_album_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    create_album_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    create_album_parser.set_defaults(handler=command_create_album)

    add_to_album_parser = subparsers.add_parser("add-to-album")
    add_to_album_parser.add_argument("--album-id", required=True)
    add_to_album_parser.add_argument("--image-key", action="append", required=True)
    add_to_album_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    add_to_album_parser.add_argument("--albums-dir")
    add_to_album_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    add_to_album_parser.set_defaults(handler=command_add_to_album)

    rename_album_parser = subparsers.add_parser("rename-album")
    rename_album_parser.add_argument("--album-id", required=True)
    rename_album_parser.add_argument("--album-name")
    rename_album_parser.add_argument("--album-description")
    rename_album_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    rename_album_parser.add_argument("--albums-dir")
    rename_album_parser.set_defaults(handler=command_rename_album)

    remove_from_album_parser = subparsers.add_parser("remove-from-album")
    remove_from_album_parser.add_argument("--album-id", required=True)
    remove_from_album_parser.add_argument("--image-key", action="append", required=True)
    remove_from_album_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    remove_from_album_parser.add_argument("--albums-dir")
    remove_from_album_parser.set_defaults(handler=command_remove_from_album)

    remove_parser = subparsers.add_parser("remove-images")
    remove_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    remove_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    remove_parser.add_argument("--manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH))
    remove_parser.add_argument("--albums-dir")
    remove_parser.add_argument("--reports-dir", default=str(pipeline.DEFAULT_IMAGE_REMOVAL_REPORTS_DIR))
    remove_parser.add_argument("--image-key", action="append")
    remove_parser.add_argument("--filter", choices=["inactive-review"])
    remove_parser.add_argument("--dry-run", action="store_true")
    remove_parser.set_defaults(handler=command_remove_images)

    return parser


def display_command_name(argv: list[str]) -> str:
    global_flags_without_values = {"--pretty", "--human", "--verbose"}
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in global_flags_without_values:
            continue
        if token.startswith("-"):
            continue
        return token
    return ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_name = display_command_name(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        if code != 0:
            print_json(envelope(command_name, ok=False, errors=["Invalid command-line arguments"]), pretty="--pretty" in argv)
        return code

    if not getattr(args, "command", None):
        print_response(envelope("", ok=False, errors=["Command is required"]), pretty=bool(args.pretty), human=bool(args.human))
        return 2

    try:
        status, response = args.handler(args)
    except Exception as exc:
        print_response(envelope(args.command, ok=False, errors=[str(exc)]), pretty=bool(args.pretty), human=bool(args.human))
        return 1
    print_response(response, pretty=bool(args.pretty), human=bool(args.human))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
