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
        "dataset_dir",
        "dataset_manifest",
        "data_yaml",
        "exported_image_count",
        "exported_label_count",
        "run_dir",
        "image_key",
        "source_content_hash",
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


def add_diagnostics(result: dict[str, Any], args: argparse.Namespace, helper: str, **paths: str | Path) -> None:
    if not getattr(args, "verbose", False):
        return
    diagnostics: dict[str, Any] = {"helper": helper}
    for key, value in paths.items():
        diagnostics[key] = str(pipeline.repo_resolve(value))
    result["diagnostics"] = diagnostics


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


def command_import_data(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    result = pipeline.import_training_data(
        list(args.source),
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        render_profile_value=args.render_profile,
        reviews_dir=args.reviews_dir,
        reviewer=args.reviewer,
        create_review_stubs=not bool(args.no_review_stubs),
        recursive=bool(args.recursive),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    warnings = [f"Failed to import {item.get('source')}: {item.get('reason')}" for item in result.get("failed") or []]
    add_diagnostics(result, args, "import_training_data", manifest_path=args.manifest, cache_dir=args.cache_dir)
    if int(result.get("imported_count") or 0) + int(result.get("skipped_count") or 0) == 0:
        return 1, envelope("import-data", ok=False, result=result, warnings=warnings, errors=["No input source could be processed successfully"])
    return 0, envelope("import-data", ok=True, result=result, warnings=warnings)


def command_check_training_ready(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    readiness = pipeline.check_training_readiness(args.selection, args.config)
    result = {"ready": readiness["ready"], "missing": readiness["missing"]}
    add_diagnostics(result, args, "check_training_readiness", selection_path=args.selection, config_path=args.config)
    response = envelope("check-training-ready", ok=True, result=result, warnings=list(readiness.get("warnings") or []))
    return (0 if readiness["ready"] else 3), response


def command_build_train_command(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    config = pipeline.load_json(args.config)
    missing = pipeline.config_missing_items(config)
    if missing:
        raise ValueError("; ".join(missing))
    dataset_dir = selection_dataset_dir(args.selection)
    command_line = pipeline.build_yolo_train_command(config, dataset_dir)
    run_group = str(config.get("run_group") or "default")
    run_name = str(config.get("run_name") or "")
    output_root = pipeline.repo_resolve(config.get("output_dir") or pipeline.WORKSPACE_DIR / "runs")
    result = {
        "command_line": command_line,
        "run_dir": str(output_root / run_group / run_name),
        "data_yaml": str(dataset_dir / "data.yaml"),
    }
    add_diagnostics(result, args, "build_yolo_train_command", selection_path=args.selection, config_path=args.config, dataset_dir=dataset_dir)
    return 0, envelope("build-train-command", ok=True, result=result)


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
    import_parser.add_argument("--reviews-dir", default=str(review_lib.DEFAULT_REVIEWS_DIR))
    import_parser.add_argument("--reviewer", default=review_lib.DEFAULT_REVIEWER)
    import_parser.add_argument("--render-profile", default="auto")
    import_parser.add_argument("--recursive", action="store_true")
    import_parser.add_argument("--force", action="store_true")
    import_parser.add_argument("--no-review-stubs", action="store_true")
    import_parser.add_argument("--dry-run", action="store_true")
    import_parser.set_defaults(handler=command_import_data)

    export_parser = subparsers.add_parser("export-dataset")
    export_parser.add_argument("--selection", required=True)
    export_parser.add_argument("--output-dir")
    export_parser.add_argument("--include-incomplete", action="store_true")
    export_parser.add_argument("--dry-run", action="store_true")
    export_parser.set_defaults(handler=command_export_dataset)

    ready_parser = subparsers.add_parser("check-training-ready")
    ready_parser.add_argument("--selection", required=True)
    ready_parser.add_argument("--config", required=True)
    ready_parser.set_defaults(handler=command_check_training_ready)

    command_parser = subparsers.add_parser("build-train-command")
    command_parser.add_argument("--selection", required=True)
    command_parser.add_argument("--config", required=True)
    command_parser.set_defaults(handler=command_build_train_command)

    image_key_parser = subparsers.add_parser("image-key")
    image_key_parser.add_argument("--source", required=True)
    image_key_parser.add_argument("--render-profile", required=True)
    image_key_parser.set_defaults(handler=command_image_key)

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
