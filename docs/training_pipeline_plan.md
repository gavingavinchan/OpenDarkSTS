# Training Pipeline Plan

## Hierarchy

This document is the authoritative plan for the training side of OpenDarkSTS.

The concrete implementation sequence for Step 1 album management plus GUI-backed Steps 4-7 is tracked in `docs/training_pipeline_step_4-7_gui_training_plan.md`. That plan should be used when turning the backend/CLI primitives in this document into end-user GUI workflows and CLI retraining checks.

Planning hierarchy:

1. `docs/grand_plan_v2.md` describes the whole project and the split between the training pipeline and the operational detection pipeline.
2. This document describes the training pipeline end to end.
3. `docs/visual_review_plan.md` describes step 2 of this training pipeline in detail.
4. `docs/visual_review_file_storage_plan.md` describes the file-backed review storage used by step 2 and consumed by later training stages.
5. `docs/review_queue_selection_removal_plan.md` describes shared Step 1/Step 3 selection tools and local workspace image-removal behavior.

The operational/deployment pipeline is separate. It consumes promoted trained weights and model metadata produced by this training pipeline, but it does not own import, review, curation, augmentation, training, or model evaluation.

## Goal

Build a reproducible GUI-driven pipeline for creating, curating, training, and evaluating YOLO datasets for ship and STS detection.

The GUI is the primary human workflow, but it must not be the only interface. Pipeline behavior should be implemented in reusable backend helpers with command-line/API entry points so automated jobs, tests, diagnostics, and LLM-assisted development can exercise the same pipeline without relying on browser interaction.

The intended flow is:

```text
Import / Download Data
  -> Auto Detect + Manual Review
  -> Reviewed BBox Overview
  -> Dataset Selection
  -> Augmentation + Training Setup
  -> Train YOLO(s)
  -> Evaluate YOLO(s)
```

The GUI should show a clickable top-center progress bar for these stages. Clicking a stage returns the user to that stage while preserving the previous UI state when possible.

## Automation And Backend Interfaces

The training pipeline should be backend-first. The browser GUI is one client of the pipeline, not the owner of the pipeline logic.

Purpose:

- allow repeatable automated runs without manual clicking
- allow tests to validate pipeline behavior directly
- allow an LLM or developer to run diagnostics against backend stages when the GUI cannot be inspected reliably
- support batch processing, scheduled training, remote execution, and future integration with other tools
- keep the GUI thin enough that backend behavior and UI behavior can be debugged separately

Rules:

- Core stage behavior must live in importable Python helpers, primarily `scripts/visual_review_lib.py` for review/crop helpers and `scripts/training_pipeline_lib.py` for dataset selection, export, training setup, training metadata, results parsing, and evaluation helpers.
- The GUI should call these helpers rather than duplicating pipeline logic in request handlers or frontend JavaScript.
- CLI/API commands should call the same helpers as the GUI.
- Every CLI/API command should read and write the same workspace artifacts described in this plan.
- Commands must write machine-readable JSON results by default. A human-readable mode may be added with `--human`, but tests and automation should use the default JSON mode.
- Commands should return non-zero exit codes for hard failures and include actionable error messages.
- Commands must never write to read-only source data folders.
- Long-running commands, especially training, should support detached/background execution and write logs under `workspace/runs/` or `workspace/reports/`.

Command-line entry points should eventually cover the pipeline backend without requiring the GUI:

```text
scripts/training_pipeline_cli.py import-data ...
scripts/training_pipeline_cli.py remove-images ...
scripts/training_pipeline_cli.py run-auto-detect ...
scripts/training_pipeline_cli.py generate-crops ...
scripts/training_pipeline_cli.py export-dataset --selection workspace/dataset_selections/<dataset_name>.json
scripts/training_pipeline_cli.py check-training-ready --selection ... --config ...
scripts/training_pipeline_cli.py build-train-command --selection ... --config ...
scripts/training_pipeline_cli.py start-training --selection ... --config ...
scripts/training_pipeline_cli.py training-status --run-dir workspace/runs/<run_group>/<run_name>
scripts/training_pipeline_cli.py stop-training --run-dir workspace/runs/<run_group>/<run_name>
scripts/training_pipeline_cli.py write-evaluation-report --run-group <run_group> ...
scripts/training_pipeline_cli.py image-key --source <path> --render-profile <profile.json>
scripts/training_pipeline_cli.py export-bundle --output workspace/bundles/<bundle_name>.zip ...
scripts/training_pipeline_cli.py import-bundle --bundle workspace/bundles/<bundle_name>.zip ...
```

The exact flags may evolve, but these capabilities should exist outside the GUI. `import-data` is specified as the next backend-first step 1 command. `run-auto-detect` should follow once the imported-image manifest helpers are stable.

Recommended CLI behavior:

- `import-data` creates or updates the import manifest and cache artifacts.
- `remove-images` removes selected images from the local training workspace records/cache/review files without modifying original external source imagery.
- `run-auto-detect` creates review stubs from configured model weights.
- `generate-crops` creates or refreshes review crop PNGs and sidecar JSON files for completed or selected reviews.
- `export-dataset` consumes a dataset selection manifest and writes a standard YOLO dataset plus `dataset_manifest.json`.
- `check-training-ready` returns a readiness checklist equivalent to the GUI's disabled/enabled `Run Training` state.
- `build-train-command` prints the exact command that would be used for YOLO training.
- `start-training` starts training in the background, records process/log metadata, and writes initial `training_metadata.json`.
- `training-status` reads run metadata, process state, log tail, and `results.csv` series for live diagnostics.
- `write-evaluation-report` writes the JSON and Markdown evaluation report and records any promotion decision.
- `export-bundle` writes a portable collaborator ZIP containing enough workspace artifacts to continue annotation, dataset export, training, and weight evaluation on another machine.
- `import-bundle` validates and merges a collaborator bundle into the local workspace without overwriting unrelated local work by default.

### CLI Contract

The first CLI should be implemented as one argparse-based entry point:

```text
python scripts/training_pipeline_cli.py <command> [options]
```

It may later be wrapped by a shell command, but tests and docs should use the Python invocation first.

Global CLI options:

- JSON output is the default and should be printed to stdout.
- `--pretty` pretty-prints JSON with indentation for human debugging.
- `--human` prints concise human-readable output instead of the JSON envelope for commands where that is useful.
- Workspace-root override is not part of the first CLI. If isolated workspace roots become necessary, prefer an environment variable such as `OPENDARKSTS_WORKSPACE` over threading a workspace argument through every helper.
- `--dry-run` validates inputs and prints the actions that would be taken without writing artifacts or launching processes. Commands that cannot support dry-run should reject it explicitly.
- `--verbose` includes additional diagnostics such as resolved paths and helper function names.

Exit codes:

- `0` success.
- `1` hard pipeline failure, such as missing input files, invalid manifests, reconciliation conflicts, failed export, or failed process start.
- `2` command-line usage error, such as missing required flags or invalid enum values.
- `3` readiness/check command completed but found blockers. This is not an exception; it means the command produced a valid checklist with `ready: false`.
- `4` long-running run/process state mismatch, such as asking to stop an already-completed run or a run with no active process.

All JSON command responses should share this envelope:

```json
{
  "ok": true,
  "command": "export-dataset",
  "schema_version": 1,
  "result": {},
  "warnings": [],
  "errors": []
}
```

For failed commands, `ok` is `false`, `errors` contains actionable strings, and the process exits non-zero. Commands should avoid printing non-JSON text to stdout in default JSON mode; logs and progress text should go to stderr or log files.

Envelope `ok` means the command executed and produced a valid response without an internal CLI/pipeline error. It does not mean a domain predicate is positive. For example, `check-training-ready` should return `ok: true`, exit `3`, and `result.ready: false` when it successfully finds readiness blockers.

First-version command details:

`import-data`

```text
python scripts/training_pipeline_cli.py import-data \
  --source <image-or-folder> \
  [--source <image-or-folder> ...] \
  [--manifest workspace/manifests/training_import_manifest.json] \
  [--cache-dir workspace/cache/imported_images] \
  [--reviews-dir workspace/reviews] \
  [--reviewer <reviewer>] \
  [--render-profile auto|rgb_original|true_color|false_color|<profile.json>] \
  [--recursive] \
  [--force] \
  [--no-review-stubs] \
  [--dry-run] \
  [--human]
```

Behavior:

- Imports user-provided local image files and/or folders into the step 1 import manifest.
- Supports `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff` in the first version.
- Does not modify, rename, move, or delete source files.
- Writes browser-ready RGB cache images to `workspace/cache/imported_images/<image_key>.png` unless `--dry-run` is set.
- Writes or updates `workspace/manifests/training_import_manifest.json` unless `--dry-run` is set.
- Creates missing per-image review stubs under `workspace/reviews/<reviewer>/<image_key>.json` unless `--dry-run` or `--no-review-stubs` is set.
- Populates or updates the step 2 manual-review queue automatically. The integrated GUI upload path should run YOLO auto detection before creating or enriching safe review stubs so the user starts from model-proposed boxes.
- Leaves existing reviewed, labeled, completed, or manual-bbox review JSON files untouched by default so manual labels, manual detections, and review progress are not overwritten by re-importing, regenerating cache files, or running auto detection.
- Defaults `--reviewer` to `visual_review_lib.DEFAULT_REVIEWER`, which is `$USER` or `default` when the environment variable is unset.
- Uses the same image identity and render-profile validation helpers as the GUI.
- Skips already-imported identical `image_key` entries by default and reports them in `skipped_count`.
- Re-renders and refreshes an existing manifest entry when `--force` is set, or when the manifest entry exists but the cache PNG is missing.
- Continues past per-file import failures by default, records those failures as warnings, and exits `0` if at least one source was imported or skipped successfully.
- Exits `1` when no input source can be processed successfully, when every requested source path is missing, or when the manifest/cache paths are invalid.

Folder scan rules:

- File inputs are imported directly.
- Folder inputs are scanned non-recursively by default.
- `--recursive` scans nested folders.
- Hidden files and hidden directories are skipped by default.
- Symlinked files may be imported, but symlinked directories are not traversed in the first version.
- Files with unsupported suffixes are ignored, not warnings.

Render-profile resolution:

- `--render-profile auto` is the default.
- `auto` maps `.png`, `.jpg`, and `.jpeg` to `{"kind": "rgb_original"}`.
- `auto` maps `.tif` and `.tiff` to the current true-color default: `{"kind": "true_color", "bands": [1, 2, 3], "normalization": "current_loader_default", "output_colorspace": "RGB"}`.
- A bare kind string uses that kind's default profile when one exists.
- A JSON file path loads the full render profile from disk and validates it before import.

JSON `result` should include:

```json
{
  "manifest_path": "workspace/manifests/training_import_manifest.json",
  "cache_dir": "workspace/cache/imported_images",
  "processed_count": 5,
  "imported_count": 4,
  "skipped_count": 1,
  "failed_count": 0,
  "review_stubs_created_count": 4,
  "review_stubs_existing_count": 1,
  "review_stubs_skipped_count": 0,
  "imported": [
    {
      "image_key": "img_abc123def4567890",
      "source": "/abs/path/to/source.tiff",
      "cache_path": "workspace/cache/imported_images/img_abc123def4567890.png",
      "render_profile_kind": "true_color",
      "status": "imported",
      "review_stub_path": "workspace/reviews/gavin/img_abc123def4567890.json",
      "review_stub_status": "created"
    }
  ],
  "skipped": [
    {
      "image_key": "img_def456abc1237890",
      "source": "/abs/path/to/previously_imported.png",
      "cache_path": "workspace/cache/imported_images/img_def456abc1237890.png",
      "render_profile_kind": "rgb_original",
      "status": "already_imported",
      "review_stub_path": "workspace/reviews/gavin/img_def456abc1237890.json",
      "review_stub_status": "existing"
    }
  ],
  "failed": [
    {
      "image_key": null,
      "source": "/abs/path/to/corrupt.tiff",
      "cache_path": null,
      "render_profile_kind": "true_color",
      "status": "failed",
      "reason": "rasterio failed to open file"
    }
  ]
}
```

Per-file failures should be listed in `result.failed` for structured automation and echoed as concise top-level envelope `warnings` for human diagnostics. Other warnings, including skipped existing entries when useful for diagnostics, belong in the top-level envelope `warnings` array rather than inside `result`.

Review-stub behavior:

- Step 1 owns image availability for step 2. Step 2 is manual review and should not have to discover imported images independently.
- A review stub is a review document for an imported image, with `review_status: "overview_1"`, imported image dimensions, cache path, and either model-proposed auto detections or `detections: []` when auto detection is intentionally skipped or produces no boxes.
- `import-data` should create a missing stub for every successfully imported, skipped, or regenerated image unless `--no-review-stubs` is set. GUI imports should enable YOLO auto detection by default; CLI imports may require an explicit `--auto-detect` flag for fast automation.
- Existing review documents with completed, reviewed, labeled, or manual detections must not be overwritten by default. Safe unreviewed auto-only or empty stubs may be enriched/replaced by auto detection.
- `run-auto-detect` may later enrich safe stubs with model detections, but an empty stub is still valid when auto detection is intentionally skipped or unavailable.
- CLI/API results should include review-stub and auto-detect counts or per-item statuses so automation can tell whether step 2 was newly populated, enriched, preserved, or protected.
- `review_stub_status` values are `created`, `existing`, `skipped`, and `failed`.
- `created` means a new review JSON was written; `existing` means an existing review JSON was preserved; `skipped` means `--no-review-stubs` or `--dry-run` prevented writing a stub; `failed` means the review stub could not be written.
- Review-stub write failures are per-image failures unless stub creation was intentionally skipped. The source file should be listed in `failed[]` and warned about, because step 1's contract includes populating the manual-review queue.
- The canonical helper is `visual_review_lib.build_review_stub(entry, reviewer, merged_detections=[])`, with a small adapter if needed so new import-manifest entries provide the fields expected by the review schema. Empty stubs do not require review-status state-machine changes; `review_status: "overview_1"` and `detections: []` are valid.

Empty review-stub shape:

```json
{
  "schema_version": 1,
  "image_key": "img_abc123def4567890",
  "image_filename": "img_abc123def4567890.png",
  "reviewer": "gavin",
  "preprocess_fingerprint": "sha256:0123456789abcdef...",
  "date": null,
  "tile_row": null,
  "tile_col": null,
  "width": 1576,
  "height": 1474,
  "cache_path": "workspace/cache/imported_images/img_abc123def4567890.png",
  "review_status": "overview_1",
  "detail_cursor": 0,
  "updated_at": "2026-05-02T00:00:00Z",
  "detections": []
}
```

For arbitrary local imports, `date`, `tile_row`, and `tile_col` are `null`. They are populated only when the import source or upstream preprocessing provides reliable Sentinel tile metadata.

`export-dataset`

```text
python scripts/training_pipeline_cli.py export-dataset \
  --selection workspace/dataset_selections/<dataset_name>.json \
  [--output-dir workspace/datasets/<dataset_name>] \
  [--include-incomplete] \
  [--dry-run] \
  [--human]
```

Behavior:

- Loads the dataset selection manifest.
- Optionally overrides `export.output_dir` for this run.
- Blocks on reconciliation conflicts. This is a hard export failure and exits `1`, not readiness exit `3`.
- Writes YOLO dataset folders, `data.yaml`, and `dataset_manifest.json` unless `--dry-run` is set.
- Uses `scripts.training_pipeline_lib.export_dataset_selection`.

JSON `result` should include:

```json
{
  "dataset_dir": "workspace/datasets/example",
  "dataset_manifest": "workspace/datasets/example/dataset_manifest.json",
  "data_yaml": "workspace/datasets/example/data.yaml",
  "exported_image_count": 10,
  "exported_label_count": 24
}
```

Warnings, including background-negative truncation warnings, belong in the top-level envelope `warnings` array rather than inside `result`.

`check-training-ready`

```text
python scripts/training_pipeline_cli.py check-training-ready \
  --selection workspace/dataset_selections/<dataset_name>.json \
  --config workspace/training_configs/<run_group>.json \
  [--human]
```

Behavior:

- Runs the same readiness gate as the GUI's `Run Training` button.
- Does not modify files.
- Exits `0` when ready.
- Exits `3` when the checklist is valid but contains blockers.
- Uses `scripts.training_pipeline_lib.check_training_readiness`.

JSON `result` should include:

```json
{
  "ready": false,
  "missing": ["Exported dataset is missing data.yaml"]
}
```

Readiness warnings, such as a missing test split, belong in the top-level envelope `warnings` array.

`build-train-command`

```text
python scripts/training_pipeline_cli.py build-train-command \
  --selection workspace/dataset_selections/<dataset_name>.json \
  --config workspace/training_configs/<run_group>.json \
  [--human]
```

Behavior:

- Resolves the exported dataset directory from the selection manifest.
- Validates `data.yaml` exists.
- Prints the exact non-shell command list for YOLO training.
- Does not start training.
- Uses `scripts.training_pipeline_lib.build_yolo_train_command`.

JSON `result` should include:

```json
{
  "command_line": ["yolo", "detect", "train", "model=yolov8n.pt", "data=workspace/datasets/example/data.yaml"],
  "run_dir": "workspace/runs/<run_group>/<run_name>",
  "data_yaml": "workspace/datasets/example/data.yaml"
}
```

`start-training`

```text
python scripts/training_pipeline_cli.py start-training \
  --selection workspace/dataset_selections/<dataset_name>.json \
  --config workspace/training_configs/<run_group>.json \
  [--foreground] \
  [--dry-run] \
  [--human]
```

Behavior:

- Runs `check-training-ready` first and blocks unless ready.
- Builds the YOLO command from the config.
- Starts detached/background training by default so the terminal, GUI, and LLM session remain usable. `--foreground` is only for deliberate local debugging and should not be used by the GUI or LLM automation.
- Writes logs to `workspace/runs/<run_group>/<run_name>/train.log`.
- Writes a process metadata file immediately after the process launch returns a PID:

```text
workspace/runs/<run_group>/<run_name>/process.json
```

Process metadata shape:

```json
{
  "schema_version": 1,
  "run_group": "example_group",
  "run_name": "yolov8n_example",
  "pid": 12345,
  "state": "running",
  "command_line": ["yolo", "detect", "train"],
  "log_path": "workspace/runs/example_group/yolov8n_example/train.log",
  "started_at": "2026-04-28T00:00:00Z",
  "ended_at": null,
  "return_code": null
}
```

- Writes initial `training_metadata.json` with status `running`; updates it to `completed`, `failed`, or `cancelled` when a watcher or later status command observes terminal state.
- If `--dry-run` is set, returns the command line, run dir, log path, and metadata paths without launching a process or writing files.

JSON `result` should include:

```json
{
  "state": "running",
  "pid": 12345,
  "run_dir": "workspace/runs/example_group/yolov8n_example",
  "log_path": "workspace/runs/example_group/yolov8n_example/train.log",
  "process_metadata": "workspace/runs/example_group/yolov8n_example/process.json",
  "training_metadata": "workspace/runs/example_group/yolov8n_example/training_metadata.json"
}
```

`training-status`

```text
python scripts/training_pipeline_cli.py training-status \
  --run-dir workspace/runs/<run_group>/<run_name> \
  [--log-lines 80] \
  [--human]
```

Behavior:

- Reads `process.json` and `training_metadata.json` when present.
- Checks whether the PID is still running when a PID is recorded. PID liveness checks should guard against PID reuse by comparing the recorded process start time or another stable process attribute when available, not PID existence alone.
- Reads the last N log lines from `train.log`. If the log file does not exist, return `log_tail: []` and add a top-level warning rather than failing the whole status check.
- Reads live metric series from `results.csv` using the recognized-column parser.
- Does not require the GUI or an active browser session.
- Uses `scripts.training_pipeline_lib.load_training_results_series` plus process/log helpers.

JSON `result` should include:

```json
{
  "state": "running",
  "pid": 12345,
  "run_dir": "workspace/runs/example_group/yolov8n_example",
  "latest_metrics": {},
  "series": {},
  "log_tail": ["..."]
}
```

`stop-training`

```text
python scripts/training_pipeline_cli.py stop-training \
  --run-dir workspace/runs/<run_group>/<run_name> \
  [--human]
```

Behavior:

- Reads `process.json`.
- Attempts best-effort graceful termination first.
- Updates `process.json` state to `cancelled` when termination is requested.
- Leaves partial run files in place for debugging.
- Exits `4` if the run is already completed, failed, cancelled, or has no active process.

`generate-crops`

```text
python scripts/training_pipeline_cli.py generate-crops \
  --reviews-dir workspace/reviews \
  [--image-key img_abc123] \
  [--reviewer gavin] \
  [--crop-root workspace/cache/review_crops] \
  [--padding-px 48] \
  [--only-completed] \
  [--dry-run] \
  [--human]
```

Behavior:

- Finds matching review files.
- Generates crop PNGs and JSON sidecars for matching detections unless `--dry-run` is set.
- Skips deleted manual detections because they are not present in review JSON.
- Keeps `neither` crop metadata with `class_id: null` and `class_name: null`.
- Uses `scripts.visual_review_lib.generate_review_crop`.

JSON `result` should include:

```json
{
  "review_count": 5,
  "crop_count": 120,
  "crop_root": "workspace/cache/review_crops",
  "generated": [
    {
      "image_key": "img_abc123",
      "detection_id": "auto-0001",
      "crop_path": "workspace/cache/review_crops/img_abc123/auto-0001.png",
      "sidecar_path": "workspace/cache/review_crops/img_abc123/auto-0001.json"
    }
  ]
}
```

`image-key`

```text
python scripts/training_pipeline_cli.py image-key \
  --source path/to/source-image \
  --render-profile path/to/render-profile.json \
  [--human]
```

Behavior:

- Computes the stable content-hash-based `image_key` for a source image and render profile.
- Does not write files.
- Helps collaborators diagnose why two machines produced different image identities.
- Uses `scripts.visual_review_lib.image_key_for_source`.

JSON `result` should include:

```json
{
  "source": "path/to/source-image",
  "source_content_hash": "sha256:...",
  "render_profile": {"kind": "rgb_original"},
  "image_key": "img_abc123"
}
```

`export-bundle`

```text
python scripts/training_pipeline_cli.py export-bundle \
  --output workspace/bundles/<bundle_name>.zip \
  [--include-imports] \
  [--include-reviews] \
  [--include-dataset-selections] \
  [--include-datasets] \
  [--include-training-configs] \
  [--include-runs] \
  [--include-weights] \
  [--all] \
  [--reviewer <reviewer>] \
  [--selection workspace/dataset_selections/<selection>.json] \
  [--run-group <run_group>] \
  [--dry-run]
```

Purpose:

- Create a single ZIP file that can be handed to a collaborator so they can continue annotation, dataset curation, YOLO training, and evaluation without needing access to the original local source folder.
- Preserve stable `image_key` identity across machines by packaging the import manifest and browser-ready cache images, not by asking the collaborator to rediscover files by local path.
- Package enough provenance and model metadata that a collaborator can understand how annotations, datasets, runs, and weights were produced.

First-version bundle contents:

- `bundle_manifest.json` at the ZIP root, with `schema_version`, `bundle_type: "training_collaboration_bundle"`, creation metadata, exporter git commit when available, included sections, file hashes, and compatibility notes.
- `workspace/manifests/training_import_manifest.json` when imports are included.
- `workspace/cache/imported_images/<image_key>.png` for every bundled imported image, so manual review and dataset export work even if original source paths are unavailable on the collaborator's machine.
- `workspace/reviews/<reviewer>/<image_key>.json` for selected reviewers or all reviewers requested by the export command.
- `workspace/cache/review_crops/` may be included as a rebuildable cache, but importers may safely regenerate it from review files and image caches.
- `workspace/dataset_selections/*.json` for dataset curation state.
- `workspace/datasets/<dataset>/` for already-exported YOLO datasets when requested. These are derived artifacts and can be omitted if dataset selections plus reviews are included.
- `workspace/training_configs/*.json` for reproducible training setup.
- `workspace/runs/<run_group>/...` and `workspace/reports/*` for training/evaluation history when requested.
- `workspace/model_weights/*.pt` only when explicitly requested. Bundles containing weights must also include matching recreation documentation from `docs/yolo_weight_recreation.md` or adjacent weight-specific docs.

Export rules:

- Default export should be conservative and explicit; `--all` is allowed, but otherwise the caller should choose sections.
- The bundle must never include files from read-only external source folders such as `/home/gavin/Downloads/AIS data`; it should include workspace cache/rendered copies and metadata instead.
- Every file included in the ZIP should be listed in `bundle_manifest.json` with path, size, and `sha256` hash.
- Bundle paths should be repository-relative workspace paths so the archive can be imported on a different machine.
- The export command should warn when a review references an image cache or manifest entry that is not included.
- The export command should fail if `--include-weights` is used and weight recreation metadata is missing for any bundled `*.pt` file.

JSON `result` should include:

```json
{
  "bundle_path": "workspace/bundles/example.zip",
  "bundle_manifest_path": "bundle_manifest.json",
  "included_file_count": 1234,
  "included_image_count": 100,
  "included_review_count": 100,
  "included_weight_count": 2,
  "warnings": []
}
```

`import-bundle`

```text
python scripts/training_pipeline_cli.py import-bundle \
  --bundle workspace/bundles/<bundle_name>.zip \
  [--target-workspace workspace] \
  [--reviewer-prefix <prefix>] \
  [--merge-reviews] \
  [--skip-derived-caches] \
  [--allow-weight-import] \
  [--dry-run]
```

Purpose:

- Validate and import a collaborator ZIP into the local workspace so annotation, dataset export, training, and evaluation can continue.
- Merge imported images, review files, dataset selections, configs, runs, and optional weights without clobbering unrelated local work.

Import rules:

- Always read and validate `bundle_manifest.json` before extracting files.
- Validate ZIP entries against path traversal; entries must resolve inside the target workspace/repository.
- Verify each included file hash before installing it.
- Import should be non-destructive by default. If a target file exists with identical hash, skip it. If it differs, report a conflict and leave the local file untouched unless an explicit future conflict-resolution option is provided.
- Review JSON conflicts should be handled as annotation conflicts, not blind overwrites. First-version behavior should either import conflicting reviews under a reviewer-prefixed namespace or fail with a clear conflict report.
- Imported `training_import_manifest.json` entries should merge by `image_key`. Identical entries are skipped; differing entries for the same `image_key` are conflicts unless the difference is limited to non-authoritative local path metadata such as `source_path`.
- Source paths from the exporter are provenance only on the importer machine. The imported cache image is the portable review/dataset source.
- Derived caches such as review crops may be skipped and regenerated.
- Weight import must be opt-in with `--allow-weight-import`, and imported weights must satisfy the YOLO weight reproducibility rule.

JSON `result` should include:

```json
{
  "bundle_path": "workspace/bundles/example.zip",
  "imported_file_count": 1200,
  "skipped_identical_count": 34,
  "conflict_count": 0,
  "imported_image_count": 100,
  "imported_review_count": 100,
  "imported_weight_count": 0,
  "conflicts": []
}
```

`write-evaluation-report`

```text
python scripts/training_pipeline_cli.py write-evaluation-report \
  --run-group <run_group> \
  --runs-dir workspace/runs/<run_group> \
  --reports-dir workspace/reports \
  [--baseline workspace/model_weights/current.pt] \
  [--promote-run <run_name>] \
  --decision-reason "..." \
  [--human]
```

Behavior:

- Collects available metrics from completed runs when possible.
- Writes `<run_group>_evaluation.json` and `<run_group>_evaluation.md`.
- Records whether a model was promoted and why, or why no model was promoted. `--decision-reason` is always required.
- Does not copy promoted weights unless an explicit later `promote-weights` command is added or a `--copy-promoted-weights` option is specified.
- Uses `scripts.training_pipeline_lib.write_evaluation_report`.

JSON `result` should include:

```json
{
  "evaluation_json": "workspace/reports/example_group_evaluation.json",
  "evaluation_markdown": "workspace/reports/example_group_evaluation.md",
  "promoted": true,
  "promoted_run_name": "yolo26n_example"
}
```

Later CLI commands can add import/download, auto-detect orchestration, and portable collaborator bundles once those backend helpers are stable. First implementation should prioritize `export-dataset`, `check-training-ready`, `build-train-command`, `start-training`, `training-status`, `stop-training`, `generate-crops`, `image-key`, and `write-evaluation-report`, because those commands validate the pipeline backend and make LLM/developer diagnostics useful immediately. `export-bundle` and `import-bundle` should be implemented before multi-person annotation or training handoff becomes a routine workflow.

Initial CLI-to-helper map:

| CLI command | Backend helper |
|---|---|
| `export-dataset` | `scripts.training_pipeline_lib.export_dataset_selection` |
| `check-training-ready` | `scripts.training_pipeline_lib.check_training_readiness` |
| `build-train-command` | `scripts.training_pipeline_lib.build_yolo_train_command` |
| `start-training` | new process-launch helper in `scripts.training_pipeline_lib` |
| `training-status` | `scripts.training_pipeline_lib.load_training_results_series` plus new process/log helpers |
| `stop-training` | new process-stop helper in `scripts.training_pipeline_lib` |
| `generate-crops` | `scripts.visual_review_lib.generate_review_crop` |
| `image-key` | `scripts.visual_review_lib.image_key_for_source` |
| `write-evaluation-report` | `scripts.training_pipeline_lib.write_evaluation_report` |
| `export-bundle` | new bundle export helper in `scripts.training_pipeline_lib` |
| `import-bundle` | new bundle import/merge helper in `scripts.training_pipeline_lib` |

Deferred HTTP/API behavior:

The local HTTP/API surface is deferred until the CLI is stable. The CLI should be the first automation contract because it is easier to test, easier for LLM/developer diagnostics, and avoids prematurely designing authentication or network exposure rules.

When added later:

- The local GUI server may expose JSON endpoints mirroring the same backend helpers for interactive use.
- API endpoints should remain local-only by default unless explicit authentication and network-safety rules are added later.
- Endpoints should be stage-oriented and artifact-oriented, for example `/api/datasets/export`, `/api/training/readiness`, `/api/training/status`, and `/api/evaluation/report`.
- API responses should include enough file paths, warnings, and validation messages for automated diagnostics.

Diagnostic contract:

- Each stage should have a cheap validation/check function that can run without launching a full GUI.
- Validation functions should distinguish hard blockers from warnings.
- Validation output should be serializable to JSON and suitable for CLI, API, tests, and GUI checklists.
- Backend functions should be deterministic where possible so repeated LLM/developer diagnostics produce stable results.
- Tests should prefer backend helpers and CLI/API interfaces, with separate browser-level tests added only where UI behavior itself must be verified.

## Artifact Layout

New training-pipeline artifacts should live under `workspace/` and follow this layout:

```text
workspace/
  albums/
    index.json
    <album_id>.json

  cache/
    review_crops/
      <image_key>/
        <detection_id>.png
        <detection_id>.json

  dataset_selections/
    <dataset_name>.json

  datasets/
    <dataset_name>/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      data.yaml
      dataset_manifest.json

  training_configs/
    <run_group>.json

  runs/
    <run_group>/
      <run_name>/
        training_metadata.json
        ...

  reports/
    import_jobs/
      <job_id>.jsonl
      <job_id>.summary.json

    <run_group>_evaluation.json
    <run_group>_evaluation.md

  bundles/
    <bundle_name>.zip

  model_weights/
    <promoted_weights>.pt
```

The read-only source data folders remain external inputs and must not be modified.

Every JSON artifact created by the training pipeline should include `schema_version: 1` unless it is a strict third-party format such as YOLO `data.yaml`. This applies to import manifests, crop sidecars, dataset selection manifests, exported dataset manifests, training configs, training metadata, evaluation reports, and bundle manifests.

## Label Vocabulary

The training pipeline uses three related label vocabularies. Implementations must use this mapping explicitly instead of comparing display strings ad hoc.

| Internal review label | YOLO class id | YOLO class name | Export behavior |
|---|---:|---|---|
| `ship_non_sts` | `0` | `ship` | Positive YOLO label |
| `ships_sts` | `1` | `ships` | Positive YOLO label |
| `neither` | none | none | Excluded from positive labels; optional background-negative signal |

Keep the existing YOLO names `ship` and `ships` for compatibility with current datasets, weights, docs, and scripts.

## Global GUI Navigation

The training GUI should have one global pipeline progress bar centered at the top of the screen.

Stages:

1. `Import Data`
2. `Manual Review`
3. `BBox Overview`
4. `Dataset Selection`
5. `Training Setup`
6. `Train YOLO`
7. `Evaluate`

Rules:

- The progress bar is clickable.
- The current stage is visually emphasized.
- Prior stages may be visually marked as already visited, but the UI does not need strict stage-complete semantics.
- Stage actions are gated by prerequisites. For example, `Run Training` stays grayed out until a dataset and valid training config are selected, then turns blue when runnable.
- Disabled stages or actions are allowed when prerequisites are missing.
- Returning to an earlier stage must not discard later-stage drafts unless the user explicitly chooses to regenerate them.
- The existing bottom-center per-image bbox review progress bar remains scoped only to step 2 manual review.

The top progress bar tracks the whole training pipeline. The bottom review bar tracks bbox progress inside one image.

## Step 1. Import / Download Data

Purpose: prepare just enough source imagery metadata/cache so step 2 manual review can run reliably without modifying original files.

This is not intended to become a large standalone ingestion platform in the first training-pipeline version. Step 2 is already mostly implemented, so step 1 exists mainly to feed step 2 with stable image identity, browser-viewable cache files, and reproducibility metadata.

The new `workspace/manifests/training_import_manifest.json` is additive. Existing legacy Sentinel preprocessing can continue to use `workspace/manifests/visual_preprocess_manifest.json` until it is explicitly migrated or retargeted. Step 1 implementations must not require rewriting legacy review/preprocess manifests before local-image import works.

Inputs may come from:

- files selected through the GUI's native file picker
- folders selected through the GUI when browser support is available
- folder paths entered in the GUI for backend-side scanning
- one or more file or folder paths passed through the CLI
- existing GeoTIFF, TIFF, PNG, or JPEG image collections
- future image collections with arbitrary unique filenames

The GUI should model imported image groups as albums, similar to a phone photo app. An album is an in-software image collection created from a file selection, folder selection, typed folder path, bundle import, or later manual grouping. It is separate from the source filesystem folder: the same source image may appear in more than one album, and importing an album never modifies the source folder.

Sentinel Hub download management is intentionally out of scope for the first version of step 1. The first version imports user-provided local imagery only.

Original images must remain untouched. The import step never edits, renames, moves, or deletes source files.

Normalization means creating workspace artifacts and metadata, not changing source data.

Required minimal normalization:

- assign a stable portable `image_key` based on source content and render profile, not filename alone
- record original source path, source collection, original filename, size, mtime, content hash, image dimensions, and loader settings
- render a browser-friendly RGB image under `workspace/cache/` for review UI use
- preserve the source image's visual geometry: no stretching, rotation, crop, or arbitrary rescale during import
- record band/render choices for GeoTIFFs, such as true color or false color band mapping
- standardize internal bbox coordinates as normalized YOLO-style values: `x_center`, `y_center`, `width`, `height`, all in `0..1`
- store reproducibility metadata in a manifest under `workspace/manifests/`

First-version GUI behavior:

- if no images have been imported yet, the step 1 page shows a large centered `Import Images` button
- clicking `Import Images` opens the operating system's native file picker via a browser file input and allows selecting many images at once
- a secondary `Import Folder` control may use browser folder-picking support when available
- a power-user folder path input should also be available for backend-side folder scanning, because browser file pickers do not expose arbitrary absolute paths to the server
- after images exist, the import controls move to a smaller toolbar near the top of the page
- imported images can be browsed in a grid view similar to the step 3 bbox overview contact sheet
- imported images can also be browsed in a list/table view similar to a file explorer
- the step 1 page should preserve view mode, search, sort, and scroll position when practical

The browser-native picker flow uploads selected image bytes to the local backend. For security reasons, browsers generally provide filenames and relative folder paths, not full local source paths. Backend-side folder scanning through a typed path or the CLI should preserve full source paths when available.

First-version step 1 browse fields:

- thumbnail or preview
- original filename
- browser relative path or backend source path when known
- dimensions
- file size
- imported timestamp
- import status or warning
- stable `image_key`

First-version imported image suffixes:

- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`

First-version folder scan behavior:

- GUI browser uploads import exactly the files selected by the user.
- GUI backend-side folder paths and CLI folder paths scan only the immediate folder by default.
- Recursive scanning requires an explicit `recursive` option in the GUI or `--recursive` in the CLI.
- Hidden files and hidden directories are skipped by default.
- Symlinked files may be imported, but symlinked directories are not traversed.
- Unsupported suffixes are ignored.

Image identity rule:

- canonical `image_key` should be deterministic across machines and collaborators
- use a content hash as the stable identity base, with enough render-profile information to distinguish different review images made from the same source file, such as true color versus false color renders
- keep human-readable source collection, original filename, and original path as metadata, not as the primary join key
- review JSONs, crop caches, dataset selections, and training manifests should all join on this stable `image_key`

Exact first-version formula:

```text
image_key = img_<sha256(canonical_identity_payload)[:16]>
```

Where `canonical_identity_payload` is canonical JSON containing:

```json
{
  "source_content_hash": "sha256:<hash of original source file bytes>",
  "render_profile": {
    "kind": "true_color",
    "bands": [1, 2, 3],
    "normalization": "current_loader_default",
    "output_colorspace": "RGB"
  }
}
```

Canonical JSON encoding for the identity payload is:

- sorted object keys
- no whitespace, using separators `(',', ':')`
- UTF-8 bytes
- no trailing newline

Python reference implementation:

```python
identity_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
image_key = "img_" + hashlib.sha256(identity_bytes).hexdigest()[:16]
```

For normal RGB PNG/JPEG inputs, the render profile can be `rgb_original`. For GeoTIFF/Sentinel inputs, the render profile distinguishes true color, false color, band choices, and normalization choices.

First-version imports use `auto` render-profile resolution by default. `auto` maps `.png`, `.jpg`, and `.jpeg` to `rgb_original`, and maps `.tif` and `.tiff` to the current true-color default profile. GUI and CLI imports may override this with a bare render-profile kind or a full render-profile JSON file.

First-version `render_profile.kind` values and required fields are:

| `kind` | Required fields | Use |
|---|---|---|
| `rgb_original` | `kind` | RGB PNG/JPEG-style inputs used as-is |
| `true_color` | `kind`, `bands`, `normalization`, `output_colorspace` | Sentinel/GeoTIFF visible-band render |
| `false_color` | `kind`, `bands`, `normalization`, `output_colorspace` | Sentinel/GeoTIFF non-visible or analysis-band render |

Shared import/review helpers and CLI commands should reject unknown `render_profile.kind` values instead of silently generating a different `image_key` for a typo. The validation source of truth should live in the backend helper library so the GUI, CLI, tests, and future API all enforce the same identity rules. First-version validation should require the fields listed above. Extra metadata fields may be allowed, but they intentionally become part of the canonical identity payload and therefore affect the generated `image_key`.

RGB PNG/JPEG payload example:

```json
{
  "source_content_hash": "sha256:<hash of original source file bytes>",
  "render_profile": {
    "kind": "rgb_original"
  }
}
```

Allowed workspace transformations:

- GeoTIFF/TIFF to PNG render for browser viewing
- RGB conversion for UI compatibility
- cached thumbnails or preview images
- deterministic tile/image grouping metadata

Disallowed transformations at import:

- editing source files
- overwriting source files
- destructive image cleanup
- lossy resizing as the only retained training source
- baking augmentation into the source dataset

Output artifacts:

- `workspace/manifests/training_import_manifest.json`
- `workspace/albums/<album_id>.json`
- `workspace/albums/index.json`
- `workspace/cache/imported_images/<image_key>.png`
- initial per-image review files under `workspace/reviews/<reviewer>/<image_key>.json`
- import audit logs under `workspace/reports/import_jobs/<job_id>.jsonl` plus a final summary JSON

Step 1 must create or preserve these review files so step 2's queue is populated as soon as import finishes. Auto detection may add detections later, but manual review image visibility must not depend on auto detection.

The cache filename convention is fixed as `<image_key>.png`. Original filenames are metadata only and must not be used as cache identity because filenames can collide across folders, machines, and collaborators.

Queue cleanup, bulk selection, and safe archiving of unreviewed review stubs are specified in `docs/review_queue_selection_removal_plan.md`. Step 1 should use that shared selection/removal model instead of adding one-off delete buttons.

### Albums, Provenance, And Audit Logging

Every GUI or CLI import invocation should create or append to a durable album record. An album is the user-facing image collection inside OpenDarkSTS, analogous to a Samsung/phone photo album. It is not a duplicate image store; it is membership, provenance, and workflow grouping for one user intent such as importing `/home/gavin/Downloads/AIS data/Sentinel 2A/Singapore/2025` as the album `Singapore 2025`.

Why this is required:

- flat queues mix newly imported unreviewed images with older intentionally unreviewed images
- browser uploads currently stage files under `workspace/cache/import_staging/<job_id>`, so the selected source folder name can be lost after upload
- the server log currently captures process startup, but not enough import-job detail to reconstruct what was uploaded, which files succeeded, or which review stubs were created
- later review, overview, dataset selection, and training should be able to include, exclude, sort, or audit albums without relying on timestamps alone

First-version album scope:

- every GUI or CLI import creates exactly one new album record unless the user explicitly appends to an existing album id
- album name defaults to the CLI source basename, browser folder top-level `webkitRelativePath` segment, or a synthetic timestamped name for loose browser file uploads
- the user can override `album_name` before import starts and can later rename it; renaming updates only album metadata, never `album_id`, manifest identity, cache files, review files, or source files
- the user can optionally provide or later edit `album_description`; descriptions are human notes only and are not used for identity or matching
- appending to an existing album is by immutable `album_id`; names are mutable and are not valid script identity by themselves
- first version records album membership, import provenance, import audit logs, and a compact album index
- first version assumes a single concurrent writer for the import manifest, album record, and album index; future versions may add file locking for simultaneous imports
- first version does not implement remove-from-album, move-to-album, empty/manual albums, derived albums, step 2 album-scoped queues, or step 3/4 album editing UI; those are future-version features described below

Required first-version album fields:

- `album_id`: stable opaque id, for example `alb_a1b2c3d4e5`; generated once and never changed
- `album_name`: mutable human label supplied by the GUI/CLI, defaulting from the selected source when available
- `album_description`: optional mutable human description
- `source_display_name`: safe non-authoritative provenance label such as `singapore/2025` or browser top-level folder name; it does not automatically change when the album is renamed
- `source_kind`: `browser_upload`, `local_folder`, `local_file`, or future `bundle_import`
- `created_at`, `updated_at`, `created_by`, `reviewer`, `manifest_path`, `cache_dir`, `reviews_dir`
- import options: recursive, force, render profile, create-review-stubs flag
- aggregate counts: uploaded, discovered, imported, skipped, failed, review stubs created/existing/skipped, members added from imported images, members added from already-known images
- image membership: ordered list of attempted image records with image key when known, original filename, browser relative path, source path when known, cache path, review stub path, and final import/member status
- optional `import_job_ids`: all import jobs that contributed to the album

Canonicality rules:

- album records are canonical for membership
- manifest `album_ids` are a denormalized lookup index for faster filtering and can be rebuilt from album records
- if album records and manifest `album_ids` conflict, album records win
- manifest entries must not store `album_names`; names are mutable and should be joined from album records or `workspace/albums/index.json` for display
- review stubs must not copy album metadata; step 2 should join to album records or the album index when album filtering is implemented

Album list index:

- `workspace/albums/index.json` should contain compact summaries for cheap album listing
- each index row should include `album_id`, `album_name`, optional `album_description`, `source_display_name`, `source_kind`, image count, counts by status/member status, created/updated timestamps, and latest import job id
- the index is derived convenience data; individual album records remain canonical

Browser upload rules:

- the browser cannot reliably provide the user's absolute source path, so the backend must not guess it
- preserve every raw uploaded filename or `webkitRelativePath` before sanitization
- if a browser upload selection contains duplicate staging filenames, the GUI should stop before upload and show a Windows Explorer-style conflict dialog with the image preview when available, filename, and `Skip`, `Replace`, and `Duplicate` choices; `Duplicate` stages the incoming browser file with ` (1)`, ` (2)`, etc. appended before the extension
- browser duplicate filename conflict resolution is a client-side staging concern, not a content-identity override; content-based `image_key` duplicate detection still decides whether an image is already known after staging
- infer the top-level browser folder only from uploaded relative paths when present
- allow the user to provide or edit `album_name` and `source_display_name` before upload starts
- for browser uploads, manifest `source_path` should point at the staging file only if the staging copy is intentionally retained as the workspace source; otherwise it should be `null`
- for browser uploads, manifest `browser_relative_path`, `source_display_name`, `album_ids`, and album membership are the primary provenance fields
- for CLI or local-folder imports, default `source_display_name` to the resolved source path unless the user overrides it
- for browser folder imports, default `source_display_name` to the top-level `webkitRelativePath` segment
- for loose browser file imports with no folder, default `source_display_name` to `null` and ask for an album label before import starts
- browser uploads are intended for small or moderate batches until streaming upload is implemented; large folders should use the CLI path to avoid whole-body buffering and browser/server timeout failures

Album membership rules:

- an image can belong to zero, one, or many albums
- album membership is by `image_key`, not by filename
- re-importing an already-known image into a new album should add membership to that album while preserving existing review status and annotations
- copying an image to another album means adding album membership and must not remove it from existing albums
- moving an image to another album means adding membership to the destination album and removing membership from one or more explicitly chosen source albums; it must never delete the image, cache, review JSON, labels, or source file
- the default UI action should be `Add to Album` / `Copy to Album`, because it is non-destructive and matches the common “save to album” mental model
- destructive membership removal should require an explicit `Remove from Album` or `Move from Album A to Album B` action with clear source/destination labels
- deleting an album should remove album membership only; it must not delete cached images, manifest entries, review files, or source files without a separate explicit cleanup workflow
- first-version albums can be created from import operations; later versions may support manual album creation from selected existing images

Future album edit operations should be available wherever the user can meaningfully select images:

- Step 1 can import into a new album, append imported files to an existing album, or create an empty/manual album for later use
- Step 2 can add the current image, selected queue rows, or all visible filtered queue images to another album without leaving review
- Step 3 can add source images for selected bbox tiles, all visible filtered bbox tiles, or all tiles matching current filters to a new or existing album
- Step 4 can create an album from the currently filtered/selected dataset candidates before or after assigning them to a dataset

Future derived albums are allowed and should record their derivation filters. For example, from Step 3 the user may filter album `Singapore 2025` to class `ships`, select all visible results, and create a new album named `Singapore 2025 ships`. That new album should contain the source image keys for those selected bbox tiles, not crop-only pseudo-images, unless the user explicitly chooses a future crop-derived dataset workflow.

Audit logging requirements:

- write `workspace/reports/import_jobs/<job_id>.jsonl` as append-only structured events for every import job
- write a final `workspace/reports/import_jobs/<job_id>.summary.json` with the same result returned to the GUI/API plus album metadata
- write lightweight GUI interaction diagnostics under `workspace/reports/` for clicks, selections, upload starts, duplicate-conflict choices, destructive confirmations, job ids, and visible filter/sort context; these logs must avoid binary payloads and secrets
- if resume support is implemented, resuming an import job must reuse the original `album_id`, append to the same job JSONL, preserve existing successful/skipped records, and add newly successful retry items to the original album record instead of forking a duplicate album
- log events for queued, upload-staged, discovered, per-file processing, per-file imported/skipped/failed, manifest-saved, review-stub-created/existing/failed, done, and failed
- include `job_id`, `album_id`, current filename, raw browser relative path, sanitized staging path, image key when known, status, and failure reason when applicable
- keep one-line server stdout messages for human diagnosis, but treat JSONL and summary JSON as the source of truth
- never log source file contents or binary payloads
- never write logs into read-only source folders

Planned album record shape:

```json
{
  "schema_version": 1,
  "album_id": "alb_a1b2c3d4e5",
  "album_name": "Singapore 2025",
  "album_description": "Sentinel 2A Singapore imagery imported for STS review",
  "source_display_name": "2025",
  "source_kind": "browser_upload",
  "import_job_ids": ["imp_9cefdae20292"],
  "created_at": "2026-05-03T23:05:40Z",
  "updated_at": "2026-05-03T23:05:53Z",
  "created_by": "gavin",
  "reviewer": "gavin",
  "manifest_path": "workspace/manifests/training_import_manifest.json",
  "cache_dir": "workspace/cache/imported_images",
  "reviews_dir": "workspace/reviews",
  "options": {
    "recursive": true,
    "force": false,
    "render_profile": "auto",
    "create_review_stubs": true
  },
  "counts": {
    "uploaded": 210,
    "discovered": 210,
    "imported": 210,
    "skipped": 0,
    "failed": 0,
    "members_added_imported": 210,
    "members_added_existing": 0,
    "review_stubs_created": 210,
    "review_stubs_existing": 0
  },
  "images": [
    {
      "image_key": "img_abc123def4567890",
      "original_filename": "2025-05-24_tile_3_2.tiff",
      "browser_relative_path": "2025/2025-05-24_tile_3_2.tiff",
      "source_path": null,
      "cache_path": "workspace/cache/imported_images/img_abc123def4567890.png",
      "review_stub_path": "workspace/reviews/gavin/img_abc123def4567890.json",
      "import_status": "imported",
      "member_status": "added_imported"
    }
  ]
}
```

Manifest entries created or refreshed by an import into an album should include:

```json
{
  "album_ids": ["alb_a1b2c3d4e5"],
  "source_display_name": "2025",
  "browser_relative_path": "2025/2025-05-24_tile_3_2.tiff"
}
```

Existing manifest entries without `album_ids` remain valid and are treated as untagged. A later repair/migration helper may create legacy albums from reliable `source_collection` groupings, but first-version albums do not need to backfill the existing manifest automatically.

Import status values are shared by manifest `import_status` fields and CLI/API result item `status` fields:

| Value | Meaning |
|---|---|
| `imported` | New manifest entry written and cache PNG written |
| `already_imported` | Manifest entry and cache PNG already present; no work done |
| `regenerated_cache` | Manifest entry already present, but cache was missing or force refresh was requested; cache PNG rewritten and entry refreshed |
| `failed` | Source could not be processed; counted in `failed_count`, listed in `failed`, and reported in top-level warnings |

`training_import_manifest.json` shape:

```json
{
  "schema_version": 1,
  "manifest_type": "training_import",
  "created_at": "2026-05-02T00:00:00Z",
  "updated_at": "2026-05-02T00:00:00Z",
  "cache_dir": "workspace/cache/imported_images",
  "images": {
    "img_abc123def4567890": {
      "schema_version": 1,
      "image_key": "img_abc123def4567890",
      "original_filename": "source_tile.tiff",
      "source_path": "/abs/path/to/source_tile.tiff",
      "browser_relative_path": null,
      "source_collection": "/abs/path/to",
      "source_size_bytes": 3145728,
      "source_mtime": 1760000000.0,
      "source_content_hash": "sha256:0123456789abcdef...",
      "width": 1576,
      "height": 1474,
      "cache_path": "workspace/cache/imported_images/img_abc123def4567890.png",
      "cache_width": 1576,
      "cache_height": 1474,
      "cache_content_hash": "sha256:fedcba9876543210...",
      "render_profile": {
        "kind": "true_color",
        "bands": [1, 2, 3],
        "normalization": "current_loader_default",
        "output_colorspace": "RGB"
      },
      "loader": "rasterio",
      "import_status": "imported",
      "imported_at": "2026-05-02T00:00:00Z",
      "updated_at": "2026-05-02T00:00:00Z",
      "warnings": []
    }
  }
}
```

For CLI and backend-side folder imports, `source_path` should be the resolved source path and `browser_relative_path` should be `null`. For browser uploads, `source_path` should be `null` unless the backend genuinely knows the original path, and `browser_relative_path` should store the browser-provided filename or relative folder path. Source fields that are not available for browser uploads, such as `source_mtime` or `source_collection`, should be `null` rather than guessed.

For CLI and backend-side folder imports, `source_collection` is the original `--source` argument or GUI source-input value that produced this entry. For files passed individually, this equals `source_path`. For folder scans, this is the scanned folder root, not each file's immediate parent directory.

Import idempotency:

- If the computed `image_key` already exists in the manifest and the cache PNG exists, import skips the image by default.
- If the computed `image_key` already exists but the cache PNG is missing, import regenerates the cache and refreshes the entry.
- If `--force` or the GUI's force option is set, import regenerates the cache and refreshes the manifest entry.
- Re-importing unchanged files should be safe and should not duplicate manifest entries.
- Review stubs are idempotent independently from manifest/cache artifacts. Missing stubs are created; existing stubs are preserved.

Expected mixed-state behavior:

| Manifest entry | Cache PNG | Review stub for reviewer | Expected behavior |
|---|---|---|---|
| present | present | present | Skip manifest/cache work; preserve stub; report `already_imported` and `review_stub_status: "existing"` |
| present | present | missing | Skip manifest/cache work; create stub; report `already_imported` and `review_stub_status: "created"` |
| present | missing | present | Regenerate cache and refresh manifest; preserve stub; report `regenerated_cache` and `review_stub_status: "existing"` |
| missing | present | present | Re-import from source and refresh/reuse cache path as needed; preserve stub; report `imported` and `review_stub_status: "existing"` |
| missing | missing | present | Full import from source; preserve stub; report `imported` and `review_stub_status: "existing"` |
| present | present | present for another reviewer only | Create a stub for the requested/current reviewer; preserve other reviewers' stubs |

If the source file is unavailable, import cannot reconstruct a missing manifest entry from cache alone in the first version because image identity depends on the source content hash and render profile.

Per-file failure behavior:

- A corrupt or unreadable supported image should not abort a larger batch by default.
- The failed file should be counted in `failed_count`, listed in `failed`, and reported in the top-level CLI/API warnings.
- A batch exits successfully when at least one image was imported or skipped successfully.
- A batch fails when no requested source can be processed successfully.

First-version scope:

- support the image formats and source layouts needed by the existing visual reviewer
- avoid adding download-management complexity unless the current review workflow needs it
- extend the existing preprocess/review helpers rather than replacing them with a separate import subsystem

Recommended first-version CLI shape:

```text
python scripts/training_pipeline_cli.py import-data \
  --source <image-or-folder> \
  [--source <image-or-folder> ...] \
  [--reviews-dir workspace/reviews] \
  [--reviewer <reviewer>] \
  [--render-profile auto|rgb_original|true_color|false_color|<profile.json>] \
  [--recursive] \
  [--force] \
  [--no-review-stubs] \
  [--dry-run]
```

## Step 2. Auto Detect + Manual Review

Purpose: run one or more YOLO models to create initial boxes, then have a human correct each image.

This is the workflow currently implemented by the visual detection reviewer and documented in `docs/visual_review_plan.md`.

Responsibilities:

- run configured YOLO model(s) on imported review images
- let the user choose which YOLO weight file or files should be used for auto detection
- merge duplicate detections using class-aware rules
- enrich existing step 1 review stubs with auto detections without clobbering manual review work
- let the reviewer add, delete, resize, relabel, and reject boxes
- keep `neither` boxes as false-positive signal in review JSONs
- preserve review progress so interrupted work can resume

The reviewer remains the source of truth for corrected bbox labels. Exported YOLO labels are generated later from review JSONs.

First-version auto-detect GUI behavior:

- consume the step 1 import manifest rather than discovering source images independently
- read the image queue from the review stubs created by step 1
- list available weights from `workspace/model_weights/`
- allow selecting one or more YOLO weights for inference
- allow entering an explicit weight path for local experimentation when needed
- expose confidence and NMS IoU settings with documented defaults
- update review stubs under `workspace/reviews/<reviewer>/` with auto detections when auto detection is run
- then route the user into manual review

Recommended first-version CLI shape:

```text
python scripts/training_pipeline_cli.py run-auto-detect \
  --manifest workspace/manifests/training_import_manifest.json \
  --weights <name>=<path-to-weights.pt> \
  [--weights <name>=<path-to-weights.pt> ...] \
  --reviewer <reviewer> \
  [--conf 0.25] \
  [--iou 0.5] \
  [--force]
```

`detection_id` must be unique within one `image_key` across all auto and manual detections. Auto detections from multiple configured models should be merged first, then assigned monotonically increasing ids such as `auto-0001` through `auto-NNNN`; manual detections use their own monotonic sequence such as `manual-0001`.

### Background Crop Generation

Every time the reviewer finishes reviewing or updating a bbox, the app should enqueue a background crop job for that bbox.

Crop purpose:

- power the step 3 bbox overview grid
- avoid repeatedly cropping the full source image during grid browsing
- give the user fast review of individual corrected boxes

Crop rules:

- crops are generated from the cached review image, not by modifying the original image
- crops are stored under `workspace/cache/review_crops/<image_key>/<detection_id>.png`
- each crop has a sidecar metadata file at `workspace/cache/review_crops/<image_key>/<detection_id>.json`
- crop metadata is stored separately, including source image key, detection id, bbox geometry, crop padding, source image dimensions, and generation timestamp
- stale crop pixels are regenerated when bbox geometry or source image fingerprint changes
- class changes refresh crop metadata and UI badges, but do not require regenerating crop pixels
- Step 2 class changes, including `Tab` hotkey changes, must be saved immediately and must not be overwritten by later geometry-only autosaves or navigation actions. `Tab` is class-only and must not advance the selected bbox or detail cursor.
- crop jobs must be safe to rerun

The crop should show the reviewed bbox with enough padding to provide visual context. The exact padding should be configurable.

Crop metadata sidecar shape:

```json
{
  "schema_version": 1,
  "image_key": "img_abc123",
  "detection_id": "auto-0007",
  "review_file": "workspace/reviews/gavin/img_abc123.json",
  "crop_path": "workspace/cache/review_crops/img_abc123/auto-0007.png",
  "source_cache_path": "workspace/cache/sentinel_tiles/example.png",
  "source_image_width": 1576,
  "source_image_height": 1476,
  "bbox": {
    "x_center": 0.51,
    "y_center": 0.44,
    "width": 0.03,
    "height": 0.02
  },
  "crop_pixels": {
    "x1": 740,
    "y1": 620,
    "x2": 850,
    "y2": 710,
    "padding_px": 48
  },
  "review": {
    "review_label": "ship_non_sts",
    "class_id": 0,
    "class_name": "ship",
    "source_type": "auto",
    "reviewed": true
  },
  "fingerprints": {
    "source_image": "sha256:...",
    "bbox_geometry": "sha256:...",
    "crop_recipe": "sha256:..."
  },
  "generated_at": "2026-04-28T00:00:00Z"
}
```

### Reviewer Reconciliation Before Overview

Step 3 should show one combined set of reviewed boxes without making reviewer identity central in the UI.

First-version reconciliation rules:

- if an `image_key` has exactly one completed review file, use it directly
- if multiple reviewers completed the same `image_key`, flag the image as a reconciliation conflict instead of silently merging boxes
- conflicted images can appear in step 3 under a `Needs Reconciliation` filter, but should not be exportable into a training dataset until resolved
- later versions may add IoU-based merge tools for duplicate boxes, but the first version should avoid silent label corruption

### Future Album-Aware Manual Review Queues

Album-scoped step 2 navigation is not part of first-version album implementation. In a later version, step 2 must not expose only one undifferentiated review queue. The all-images queue may remain available for legacy browsing, but the primary training workflow should let the user select one or more albums from a dropdown or multi-select control. The queue should then contain only images from those albums, analogous to reviewing images from selected phone photo albums.

Required queue scopes:

- `Selected Albums`: one or more chosen album ids; default immediately after a successful step 1 GUI import
- `All Unreviewed`: every review file whose status is not complete
- `All Images`: legacy full queue for diagnostics and broad browsing
- `By Collection`: choose a `source_display_name` or legacy `source_collection`
- `Completed In Selected Albums`: for spot-checking recently reviewed imported images

Queue rows should display enough provenance to avoid ambiguity:

- queue position within the active scope and total count for that scope
- global queue position only as secondary/debug metadata
- image filename and image key
- review status
- album name or source display name
- imported-at timestamp
- counts of detections and reviewed detections when cheap to compute

Queue sorting should be user-selectable. Future sort modes should include:

- album, then imported order
- album, then filename/date tile order
- review status, then album
- newest imported first
- oldest imported first
- filename/date tile order
- image key for diagnostics

Queue behavior after import:

- when step 1 import finishes, the UI should route to `/manual-review?album_id=<album_id>` or otherwise set the active album selection to the new album
- `Next Image` and `Previous Image` should navigate inside the active album/sort/filter scope, not the global all-image queue
- explicitly switching to `All Images` should be required before older intentionally unreviewed images are mixed into navigation
- active album ids, queue scope, filters, and sort mode should be encoded in the URL and browser local storage so refreshes preserve context
- status updates should update the current scope counts without dropping the selected scope

Album actions in step 2:

- current image menu: `Add to Album`, `Move to Album`, `Remove from Album`, and `Create New Album From Current Image`
- queue selection menu: same actions for selected queue rows
- visible-filter action: `Create Album From Visible Queue` for workflows such as all unreviewed Singapore images after applying filters
- every move action must show the source album(s) that will lose membership and should default to copy/add when there is any ambiguity

Backend/API behavior:

- `/api/state` should accept optional `album_id` (repeatable or comma-separated), `source_display_name`, `status`, `queue_scope`, and `sort` parameters
- response payload should continue to return a small queue window around the selected image, not the entire full queue
- backend should cache queue metadata but invalidate it after imports, review saves, or review-stub writes
- queue filtering should use album records or `workspace/albums/index.json`; it should not require album metadata to be copied into review stubs
- review files created before album metadata existed should appear under a legacy/unknown album group rather than being silently hidden

Migration behavior:

- existing review JSONs without `album_ids` remain valid
- existing manifest entries without `album_ids` remain valid
- a future repair command may backfill album metadata from import logs, album records, source collection, source paths, or timestamp ranges when reliable
- if no reliable album can be inferred, leave the album as `legacy_unknown` rather than guessing

## Step 3. Reviewed BBox Overview

Purpose: let the human do one more pass over all reviewed bboxes before dataset selection.

This stage shows a grid view similar to Windows Explorer or macOS Finder icon/grid mode.

Step 3 should be implemented inside the same local UI/server family as the existing visual reviewer, not as a separate product. The first full-training GUI entry point should be `scripts/training_pipeline_ui.py`, with the existing reviewer reused for step 2 and shared logic kept in helper libraries.

Grid behavior:

- the grid shows bbox crop images, not full source images
- each tile represents one reviewed bbox crop
- every reviewed positive bbox is ground truth regardless of whether it began as an auto detection or a manually drawn bbox; auto/manual origin may be shown as provenance but must not split label counts or export eligibility
- tiles show minimal visual status: class, dataset membership if any, review status, and warning badges
- tile size is adjustable, similar to Windows Explorer icon size controls
- large grids must use lazy loading or virtualization so many thousands of bbox crops remain usable
- filters and sorting should apply before tile rendering when possible
- single-click selects one tile and updates the right sidebar
- double-click opens the original image in step 2 manual review at the corresponding bbox
- a right-sidebar `Edit In Manual Review` button performs the same action as double-click

Shared checkbox/range selection and safe queue-removal behavior for Step 3 are specified in `docs/review_queue_selection_removal_plan.md`.

Album filtering and sorting:

- step 3 should have an album multi-select filter just like step 2
- selected albums constrain which reviewed bbox crops are visible
- existing class filters such as `ship`, `ships`, `neither`, warnings, and conflicts remain available and combine with album filters
- tile badges should show album membership when space allows; the sidebar should always show album names for the source image
- sort modes should include album-first, class-first, image/date order, bbox area, confidence, and review/update time
- double-clicking a crop should preserve album scope when returning to step 2, for example `/manual-review?album_id=<album_id>&image_id=<image_key>&detection_id=<detection_id>`

Album actions in step 3:

- selected crop tiles can add their source images to a new or existing album
- all visible filtered crop tiles can add their source images to a new or existing album
- the UI should deduplicate by `image_key` when multiple selected bboxes come from the same source image
- derived album creation should record the source album filter, class filter, warning/conflict filter, sort mode when relevant, creation timestamp, and selected detection ids as provenance
- creating an album from `ships` bboxes in `Singapore 2025` should not remove those images from `Singapore 2025`; it should create another album containing the same source images unless the user explicitly chooses a move operation
- if the user asks to move from one album to another in step 3, the UI must be clear that membership changes are image-level, not bbox-level, unless a future bbox-album feature is explicitly introduced

Right sidebar information:

- source image filename and `image_key`
- detection id
- reviewed class: `ship`, `ships`, or `neither`
- reviewer, derived from the review file's `reviewer` field or parent `workspace/reviews/<reviewer>/` directory
- confidence and source model provenance when available
- bbox dimensions and normalized coordinates
- manual vs auto source type as provenance only, not a ground-truth quality distinction
- current dataset memberships
- warnings such as stale crop, incomplete review, missing source cache, or unusual bbox size

Returning from manual review:

- when the user jumps from step 3 to step 2, the GUI stores the step 3 state
- stored state includes filters, sort order, scroll position, selected crop ids, current image, and sidebar selection
- clicking `BBox Overview` in the top progress bar returns to step 3 at the same grid state
- any edited bboxes should regenerate their crop thumbnails in the background

This stage does not decide final training membership by itself. It is a visual quality-control pass before dataset selection.

Training-mode clarification:

- bbox crops shown here are review UI artifacts by default
- standard YOLO training uses whole review images plus per-image YOLO label files
- selecting or inspecting a crop should therefore carry its source image context forward to step 4
- crop-level training images are allowed only as explicit derived-dataset options, mainly for hard-negative mining

## Step 4. Dataset Selection

Purpose: choose which reviewed images or bboxes belong to which training dataset versions.

The user may create multiple datasets. A dataset is a named selection plus export settings, not just a folder of copied files.

The default dataset name should be `dataset_1` until the user changes it.

Selection behavior should follow familiar desktop and mobile conventions:

- single-click selects one item
- `Ctrl`/`Cmd` click toggles individual items
- `Shift` click selects a range
- `Ctrl+A`/`Cmd+A` selects all visible filtered items
- toolbar buttons provide `Select All`, `Clear Selection`, and `Invert Selection`
- touch/mobile mode supports long-press to enter selection mode, tap to toggle, and a visible `Select All` action

Dataset assignment behavior:

- the first filter should be an album multi-select so the user can build datasets from one or more albums, such as `Singapore 2025` plus two other reviewed albums
- selected items can be added to one or more datasets
- selected items can be removed from one or more datasets
- the UI should make it clear whether assignment is by whole image or by individual bbox
- the first version should prefer image-level assignment, because standard YOLO datasets include one label file per whole image
- bbox-level selection can be represented as an advanced label-filtering option within selected whole images
- step 3 crop selections should carry forward into this stage as candidate source-image selections, not as an automatic switch to crop training
- train/val/test split ownership lives in this stage; later training stages reference the selected dataset and split rather than redefining them

Album-aware dataset selection:

- album selection is an inclusion filter, not the dataset itself
- other filters still apply inside selected albums, including reviewer, review completion, class labels, warnings, conflicts, date/tile ranges, and positive/negative label rules
- common workflows should be one click or a few clicks: “include all completed images from these three albums”, “include only `ships_sts` bboxes from selected albums”, or “include manually reviewed positives from selected albums plus background negatives from another album”
- the dataset selection manifest should record both the album filter used to create the dataset and the concrete selected `image_key` list so exports remain reproducible even if album membership changes later
- first-version dataset selection supports only snapshot album filters; dynamic album-backed datasets are deferred to a future version and must be explicit if added later
- UI counts should show selected image count, selected bbox count by class, and count by album before export

Album actions in step 4:

- selected dataset candidates can be added to a new or existing album before export
- all visible filtered candidates can be saved as a new album, which is useful for reusable curation groups such as `Singapore 2025 reviewed ships`
- album creation from step 4 should not imply dataset export; albums are reusable organization, while datasets are training/export manifests
- moving images between albums from step 4 follows the same non-destructive default: copy/add first, remove from source album only when explicitly requested
- dataset selection manifests should record whether an album action was used to create the selected image set, but the training export should still depend on concrete `image_key` entries by default

`neither` handling:

- `neither` boxes are excluded from normal YOLO positive label files by default
- step 4 controls how much false-positive signal enters training
- first-version hard-negative support should prefer image-level background negatives: whole images with no exported `ship` or `ships` labels can be included with an empty label file
- crop-level `neither` hard negatives may be added later as an explicit derived-dataset option, where each selected `neither` crop is emitted as a labelless training image

Dataset records should be stored as reproducible selection manifests, for example:

```text
workspace/dataset_selections/<dataset_name>.json
```

Each dataset selection manifest should record:

- dataset name and version
- selected album ids used as filters, if any, plus optional album display-name snapshots for audit only
- selected image keys
- optional selected detection ids if bbox-level filtering is enabled
- review manifest fingerprint or per-image preprocess fingerprints
- selection filters used to create the dataset
- reviewer and timestamp
- class inclusion rules
- train/val/test split policy

`include_labels` enum values for the first version:

- `all_exportable` means every reviewed positive box on this image is exported
- `selected_only` means only detection ids listed in `selected_detection_ids` are exported

Reviewed positive boxes are source-agnostic ground truth. Dataset counts, selected bbox totals, and YOLO exports must include reviewed `ship_non_sts` and `ships_sts` labels the same way for both auto-origin and manually drawn bboxes.

Reconciliation conflicts are a hard export blocker. Images marked `Needs Reconciliation` must not be exported into any training dataset until resolved; this is not a per-dataset option.

First-version split rule:

- default to automatic split assignment with `80%` train, `10%` val, and `10%` test
- allow manual override so the user can move images between splits
- persist the final explicit split per image in the dataset selection manifest
- re-exporting a dataset uses the persisted split, not a fresh random split
- when new images are added to an existing dataset, auto-assign only the new images according to the default split ratios and preserve all existing image split assignments and manual overrides
- the UI should note how many new images were auto-split when this happens

Example selection manifest:

```json
{
  "schema_version": 1,
  "dataset_name": "sts_reviewed_v1",
  "dataset_version": 1,
  "created_at": "2026-04-28T00:00:00Z",
  "updated_at": "2026-04-28T00:00:00Z",
  "created_by": "gavin",
  "description": "First reviewed Sentinel STS training set",
  "selection_mode": "image",
  "album_filters": {
    "mode": "snapshot",
    "album_ids": ["alb_a1b2c3d4e5"],
    "album_name_snapshots": {"alb_a1b2c3d4e5": "Singapore 2025"}
  },
  "classes": {
    "0": "ship",
    "1": "ships"
  },
  "sources": {
    "reviews_dir": "workspace/reviews",
    "manifest": "workspace/manifests/training_import_manifest.json"
  },
  "split_policy": {
    "mode": "auto_with_manual_override",
    "default_ratios": {
      "train": 0.8,
      "val": 0.1,
      "test": 0.1
    },
    "seed": 42
  },
  "items": [
    {
      "image_key": "img_abc123",
      "split": "train",
      "include_labels": "all_exportable",
      "selected_detection_ids": null
    }
  ],
  "background_negatives": {
    "enabled": true,
    "mode": "whole_image_empty_label",
    "max_fraction": 0.1,
    "image_keys": ["img_bg001"]
  },
  "export": {
    "output_dir": "workspace/datasets/sts_reviewed_v1",
    "include_incomplete_reviews": false
  }
}
```

Exporting a dataset should produce a standard YOLO dataset under:

```text
workspace/datasets/<dataset_name>/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
  dataset_manifest.json
```

The YOLO `images/`, `labels/`, and `data.yaml` remain standard. Reproducibility metadata goes in `dataset_manifest.json`.

`dataset_manifest.json` should reference the dataset selection manifest and record the concrete exported files and hashes. It should not become a second independent source of selection truth.

First-version export rules:

- export whole review images as YOLO detection images
- write one YOLO label file per exported image
- `ship_non_sts` exports as class `0`
- `ships_sts` exports as class `1`
- `neither` never exports as a positive label
- selected background-negative images export as whole images with empty label files
- if `background_negatives.image_keys` and `max_fraction` are both set, include selected image keys up to the fraction cap where `max_fraction` is the maximum share of exported images that may be background negatives; truncate any oversubscription and record the warning in `dataset_manifest.json`
- bbox crop images are not training inputs unless a later explicit derived-dataset option is enabled

## Step 5. Augmentation + Training Setup

Purpose: choose which dataset and training settings will be used for one or more YOLO runs.

The default user route requires an exported reviewed dataset and permits an optional run name. The current documented detector is recommended as the starting model. A concise summary identifies the available compute device and the principal training settings. Saving checks readiness and continues to the training page when the configuration is ready; training starts separately on that page.

A collapsed Advanced section preserves the complete controls for:

- dataset name/version
- model architecture or starting weights
- image size
- epochs
- batch size
- patience
- seed
- device
- workers
- augmentation settings
- output run name

The selected dataset already owns the train/val/test split. Step 5 may display the split and validate that it exists, but should not define a conflicting split policy.

The current defaults are 960-pixel images, 150 epochs, batches of 8, patience 30, seed 42, and four data-loading workers. These values remain editable in Advanced. Automatic compute selection uses the active training Python environment and prefers NVIDIA CUDA device 0, then an available Apple GPU, then CPU. The configuration stores the concrete device, original device request, and Python interpreter; explicit CPU selection and the devices in previously saved configurations are preserved. The shared `training_defaults` helper is exposed by the `training-defaults` CLI command and `GET /api/training/defaults`.

Augmentation should be stored as configuration, not pre-applied to source images. YOLO/Ultralytics can apply augmentation during training. The UI should expose all supported augmentation parameters, with sensible defaults and an advanced section for less common knobs. If offline augmentation is later needed, it should write derived images into a new derived dataset with its own manifest.

Training setup output:

```text
workspace/training_configs/<run_group>.json
```

The config should include every parameter needed to reproduce the training command.

`run_group` rule:

- one `run_group` corresponds to one training config
- a `run_group` may contain multiple `run_name` entries, for example one each for YOLOv8n and YOLO26n
- default `run_group` should be derived from the dataset name and timestamp, unless the user edits it
- `run_name` identifies one concrete model run inside that group

### Run Training Readiness Gate

The `Run Training` button should stay grayed out until the run is ready. When disabled, the UI must show a clear checklist of what is still missing or invalid.

Required before `Run Training` is enabled:

- a dataset selection manifest exists
- the selected dataset has been exported to YOLO format
- exported dataset contains `data.yaml`
- exported dataset contains `images/train/`, `images/val/`, `labels/train/`, and `labels/val/`
- train split has at least one image
- val split has at least one image
- `data.yaml` defines class `0: ship` and class `1: ships`
- no unresolved `Needs Reconciliation` images are included in the selected dataset
- required training config fields are present: dataset name/version, model or starting weights, image size, epochs, batch, device, seed, output run name, and augmentation settings
- output run name is safe and will not overwrite an existing run unless the user explicitly enables overwrite

The test split is recommended but not required to start training. If no test split exists, step 7 evaluation should warn that final test metrics are unavailable.

## Step 6. Train YOLO(s)

Purpose: run one or more training jobs from the selected dataset and training config.

Training may include multiple models, for example YOLOv8n and YOLO26n.

When training starts, it should run as a background process so the browser UI and LLM/terminal session remain usable. The UI should create an active run card below the top progress bar, aligned to the right side of the main content area, that remains visible even if the user navigates to another stage.

Run states:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

While a run is active, the UI should show:

- run name
- model or starting weights
- dataset name/version
- current epoch when available
- elapsed time
- latest metrics when available
- log tail
- live performance/loss plots over epoch

Live plots should be read from the run directory's Ultralytics `results.csv` when available and refreshed every few seconds while the run state is `running`.

Preferred plotted series when available:

- `train/box_loss`
- `train/cls_loss`
- `train/dfl_loss`
- `val/box_loss`
- `val/cls_loss`
- `val/dfl_loss`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`

If `results.csv` does not exist yet, the plot area should show `Waiting for first epoch`. If columns differ across YOLO versions, plot whichever recognized columns are available.

Running constraints:

- do not allow editing the dataset/config attached to the active run
- allow safe navigation to other stages while training continues
- write logs under the run directory, such as `workspace/runs/<run_group>/<run_name>/train.log`
- `Stop Training` should attempt best-effort termination of the background process, mark run metadata as `cancelled`, and leave partial run files in place for debugging; the final checkpoint may be partial or unusable
- when training completes, enable the evaluation stage and show a `Go To Evaluation` action

Required reproducibility metadata:

- exact training config JSON
- dataset manifest path and hash
- data.yaml path and hash
- starting weights path and hash when available
- model architecture name
- package versions, including `ultralytics`, `torch`, Python, CUDA if available
- git commit hash and dirty-worktree flag
- command line used to start training
- start time, end time, and host/device summary
- output run directory

Training outputs should live under `workspace/runs/` or another configured workspace directory, never in the read-only source data folder.

Each completed training run should produce a run metadata file, for example:

```text
workspace/runs/<run_group>/<run_name>/training_metadata.json
```

## Step 7. Evaluate YOLO(s)

Purpose: compare trained models and decide whether any weights should be promoted for downstream use.

Evaluation should include:

- validation metrics
- test metrics
- class-level metrics for `ship` and `ships`
- precision, recall, mAP50, and mAP50-95
- confusion matrix when available
- representative prediction images
- false-positive and false-negative review notes when available
- comparison against current baseline weights in `workspace/model_weights/`

Evaluation outputs:

```text
workspace/reports/<run_group>_evaluation.json
workspace/reports/<run_group>_evaluation.md
```

Promoted weights should be copied or referenced under:

```text
workspace/model_weights/
```

The evaluation report must state why a model was or was not promoted.

Baseline weights:

- current baseline weights live under `workspace/model_weights/`
- `docs/progress.md` is the human-readable record of current best weights per architecture
- every promoted-weights decision should update `docs/progress.md` so the human-readable baseline record does not drift
- a machine-readable baseline registry can be added later if the GUI needs automatic baseline selection

## Reproducibility Contract

A training result is reproducible only if these artifacts are present:

- import manifest
- per-image review JSONs used for the dataset
- dataset selection manifest
- exported dataset manifest
- training config
- training metadata
- evaluation report
- final/promoted weight path

Source images remain external inputs and are never modified by the pipeline.

## Relationship To Existing Code

Current code already covers much of step 2 and part of dataset export:

- `scripts/preprocess_sentinel_tiles.py` supports import, render cache, YOLO inference, and review stub creation.
- `scripts/review_visual_detections_ui.py` supports manual bbox review.
- `scripts/visual_review_lib.py` stores review JSON state and provides export helpers.
- `scripts/export_reviewed_labels.py` exports completed review JSONs to YOLO format.
- `scripts/compare_yolo_models.py` trains and evaluates YOLOv8n and YOLO26n on whole-image YOLO detection datasets, but should eventually consume step 5 training configs and emit the step 6 metadata described here.

Planned first implementation structure:

- `scripts/training_pipeline_ui.py` owns the top progress bar and routes between training stages.
- `scripts/training_pipeline_cli.py` owns non-GUI automation and diagnostic commands for the same stages.
- `scripts/review_visual_detections_ui.py` remains useful for focused step 2 manual-review testing, and the mostly-working step 2 reviewer should be left alone except for integration hooks needed by the training pipeline.
- `scripts/visual_review_lib.py` keeps shared review, bbox, crop, and review-file helpers.
- `scripts/training_pipeline_lib.py` should hold new dataset-selection, export, training-config, training-metadata, and evaluation helpers.
- GUI routes, CLI commands, and tests should all use the shared helper libraries rather than each implementing independent pipeline logic.

The implementation can attempt the full training flow in one pass, but it should still be built as modular helper functions and stage-specific UI paths so failures are debuggable.

Missing higher-level pieces:

- a unified training GUI shell with the top progress bar
- a command-line/API automation surface for backend stages and diagnostics
- background reviewed-bbox crop generation
- step 3 bbox overview grid
- step 4 dataset selection manifests and multi-dataset assignment UI
- step 5 training setup UI and config writer
- step 6 reproducibility metadata writer
- step 7 evaluation report UI/report generator
