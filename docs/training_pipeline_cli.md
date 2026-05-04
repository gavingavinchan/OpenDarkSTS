# Training Pipeline CLI

`scripts/training_pipeline_cli.py` is the machine-readable command-line interface for the training-side pipeline. It calls the same backend helpers used by the reviewer and training pipeline code, so prefer it for automation, smoke checks, and LLM/developer diagnostics instead of reimplementing pipeline logic in shell scripts.

Review-queue selection, pruning, and safe archive semantics are planned in `docs/review_queue_selection_removal_plan.md`.

Run commands from the repository root:

```bash
.venv/bin/python scripts/training_pipeline_cli.py <command> [options]
```

The system Python also works when dependencies are available, but `.venv/bin/python` is the local project environment documented in `docs/local-venv.md`.

## Output Contract

Default output is JSON on stdout:

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

Global flags:

- `--pretty`: pretty-print the JSON envelope.
- `--human`: print concise human-readable output instead of JSON.
- `--verbose`: include extra diagnostics such as resolved paths and helper names when supported.

Exit codes:

- `0`: command succeeded.
- `1`: pipeline failure, such as missing files or invalid manifests.
- `2`: command-line usage error.
- `3`: readiness command completed successfully but found blockers. In this case `ok` is still `true` and `result.ready` is `false`.

`ok` means the CLI command ran and returned a valid response. It does not always mean the domain condition is positive. For example, `check-training-ready` exits `3` with `ok: true` when it successfully reports training blockers.

## Commands

### `import-data`

Imports local imagery into the step 1 training import manifest and browser-ready PNG cache.

```bash
.venv/bin/python scripts/training_pipeline_cli.py import-data \
  --source /path/to/image-or-folder
```

Options:

- `--source <image-or-folder>`: required, repeatable. Supports `.png`, `.jpg`, `.jpeg`, `.tif`, and `.tiff`.
- `--manifest <path>`: import manifest path. Defaults to `workspace/manifests/training_import_manifest.json`.
- `--cache-dir <path>`: rendered PNG cache directory. Defaults to `workspace/cache/imported_images`.
- `--reviews-dir <path>`: review stub root. Defaults to `workspace/reviews`.
- `--reviewer <name>`: reviewer folder for review stubs. Defaults to `$USER`, or `default` when unset.
- `--render-profile auto|rgb_original|true_color|false_color|<profile.json>`: rendering profile. Defaults to `auto`.
- `--recursive`: scan folders recursively. Hidden files and directories are skipped.
- `--force`: refresh an existing manifest/cache entry.
- `--no-review-stubs`: skip creating missing step 2 review JSON stubs.
- `--dry-run`: validate inputs and report actions without writing the manifest or cache.

The command never modifies source files. It writes or updates the import manifest, cache, and missing review stubs unless `--dry-run` is used. Already-imported identical image/profile pairs are skipped and reported in `skipped_count`; their missing review stubs are still created unless `--no-review-stubs` is set.

Planned album/provenance options:

- `--album-name <name>`: human-readable OpenDarkSTS album name, such as `Singapore 2025`; defaults to the source basename for first-version imports.
- `--album-description <text>`: optional mutable human description for notes about the album.
- `--source-display-name <name>`: provenance label shown in the GUI, such as `singapore/2025`.
- `--append-to-album-id <id>`: add imported or already-known images to an existing album by immutable id.
- `--resume-job <id>`: continue a failed or interrupted import job by reusing the original album id and appending to the original job JSONL instead of creating a duplicate album.

Planned album/import artifacts:

- `workspace/albums/<album_id>.json`: durable album ledger with image membership and aggregate counts.
- `workspace/albums/index.json`: compact album summaries for listing albums without loading every membership ledger.
- `workspace/reports/import_jobs/<job_id>.jsonl`: structured event log for every import job.
- `workspace/reports/import_jobs/<job_id>.summary.json`: final import result plus album metadata.

The GUI upload path should use the same backend fields. Browser uploads must preserve raw browser relative paths and user-provided album/source labels because the browser cannot reliably provide the user's original absolute folder path.

Future step 2 review should support opening directly into an album-scoped queue, for example `/manual-review?album_id=<album_id>`, so new imports are not mixed with older intentionally unreviewed images unless the user chooses the all-images queue.

First-version CLI album support should create one album per import or append to `--append-to-album-id`. Future CLI helpers should include `list-albums`, `show-album`, `rename-album`, `update-album-description`, `create-album`, `add-to-album`, `remove-from-album`, and `move-to-album` so automation can use the same album model as the GUI. `add-to-album` should be the default non-destructive operation; `move-to-album` must explicitly name the source album membership to remove.

### `export-dataset`

Exports a dataset selection manifest into YOLO layout.

```bash
.venv/bin/python scripts/training_pipeline_cli.py export-dataset \
  --selection workspace/dataset_selections/manual_reviewed_gavin_v1.json
```

Options:

- `--selection <path>`: required dataset selection manifest.
- `--output-dir <path>`: override `export.output_dir` for this run without mutating the selection file.
- `--include-incomplete`: allow incomplete reviews if the selection/export logic supports them.
- `--dry-run`: report expected output paths and counts without writing artifacts.

Writes, unless `--dry-run` is used:

- `workspace/datasets/<dataset>/images/{train,val,test}/...`
- `workspace/datasets/<dataset>/labels/{train,val,test}/...`
- `workspace/datasets/<dataset>/data.yaml`
- `workspace/datasets/<dataset>/dataset_manifest.json`

Example dry run:

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty export-dataset \
  --selection workspace/dataset_selections/manual_reviewed_gavin_v1.json \
  --dry-run
```

### `check-training-ready`

Checks whether an exported dataset and training config are sufficient for YOLO training.

```bash
.venv/bin/python scripts/training_pipeline_cli.py check-training-ready \
  --selection workspace/dataset_selections/manual_reviewed_gavin_v1.json \
  --config workspace/training_configs/manual_reviewed_gavin_v1_yolov8n_smoke.json
```

Checks include:

- dataset export exists and contains `dataset_manifest.json`
- `data.yaml` exists and defines `0: ship` and `1: ships`
- `train` and `val` image/label folders exist and contain images
- required training config fields are present
- target run directory will not overwrite an existing run unless `overwrite` is enabled

Required training config fields:

- `dataset_name`
- `dataset_version`
- `model`
- `image_size`
- `epochs`
- `batch`
- `device`
- `seed`
- `run_name`
- `augmentation`

Optional but commonly used config fields include `run_group`, `output_dir`, `workers`, `patience`, and `overwrite`.

### `build-train-command`

Builds the exact Ultralytics YOLO training command for a selection/config pair. It does not launch training.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty build-train-command \
  --selection workspace/dataset_selections/manual_reviewed_gavin_v1.json \
  --config workspace/training_configs/manual_reviewed_gavin_v1_yolov8n_smoke.json
```

The JSON result includes:

- `command_line`: array form of the `yolo detect train ...` command
- `run_dir`: expected output directory
- `data_yaml`: dataset `data.yaml` path

To run the command manually, use the project virtualenv's `yolo` executable or preserve the generated arguments with `.venv/bin/yolo`:

```bash
.venv/bin/yolo detect train model=yolov8n.pt data=workspace/datasets/manual_reviewed_gavin_v1/data.yaml imgsz=960 epochs=5 batch=2 device=cpu
```

For long training runs, start them detached and log under `workspace/runs/` or `workspace/reports/` so the LLM session remains usable.

### `image-key`

Computes the stable image key for a source image and render profile.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty image-key \
  --source workspace/cache/sentinel_tiles/2020-04-23_tile_1_0.png \
  --render-profile workspace/render_profiles/sentinel_tile_v1.json
```

The command validates the render profile and returns:

- `source`: resolved source image path
- `source_content_hash`: content hash for the source image
- `render_profile`: parsed render profile object
- `image_key`: stable key derived from source content and render profile

Use this when diagnosing identity mismatches between preprocessing, review files, crops, and exported datasets.

### Planned: `export-bundle`

Not implemented yet. This planned command should create a portable collaborator ZIP for handing annotation/training state to another machine.

Expected shape:

```bash
.venv/bin/python scripts/training_pipeline_cli.py export-bundle \
  --output workspace/bundles/my_annotation_handoff.zip \
  --include-imports \
  --include-reviews \
  --include-dataset-selections \
  --include-training-configs
```

Expected bundle responsibilities:

- Include `bundle_manifest.json` at the ZIP root with schema version, exporter metadata, included sections, and per-file hashes.
- Include import manifests and `workspace/cache/imported_images/<image_key>.png` so the receiver does not need the sender's original source paths.
- Include review JSONs under `workspace/reviews/<reviewer>/` so annotation can continue.
- Optionally include dataset selections, exported YOLO datasets, training configs, runs, reports, and model weights.
- Include YOLO weights only when explicitly requested, and only when the required weight recreation documentation is bundled too.
- Never include files directly from read-only external source folders such as `/home/gavin/Downloads/AIS data`.

### Planned: `import-bundle`

Not implemented yet. This planned command should validate and merge a collaborator ZIP into the local workspace.

Expected shape:

```bash
.venv/bin/python scripts/training_pipeline_cli.py import-bundle \
  --bundle workspace/bundles/my_annotation_handoff.zip \
  --merge-reviews \
  --dry-run
```

Expected import responsibilities:

- Validate `bundle_manifest.json` and verify file hashes before extracting.
- Reject unsafe ZIP paths that would write outside the workspace.
- Merge import manifest entries by stable `image_key`.
- Skip existing identical files.
- Report conflicts without overwriting local work by default.
- Treat exported source paths as provenance only; imported cache images are the portable review/dataset source.
- Require explicit opt-in before importing bundled `*.pt` weights.

## Selection Manifest Shape

Selection manifests are JSON files under `workspace/dataset_selections/`. A typical shape is:

```json
{
  "schema_version": 1,
  "dataset_name": "manual_reviewed_gavin",
  "dataset_version": 1,
  "reviews_dir": "workspace/reviews",
  "export": {
    "output_dir": "workspace/datasets/manual_reviewed_gavin_v1",
    "include_incomplete_reviews": false
  },
  "items": [
    {
      "image_key": "2020-04-23_tile_1_0",
      "split": "train",
      "include_labels": "all_exportable",
      "selected_detection_ids": null
    }
  ],
  "background_negatives": {
    "enabled": false,
    "mode": "whole_image_empty_label",
    "max_fraction": 0.5,
    "image_keys": []
  }
}
```

`include_labels` supports the current export modes used by the backend:

- `all_exportable`: export all `ship_non_sts` and `ships_sts` labels for the image.
- `selected_only`: export only IDs listed in `selected_detection_ids`.

Labels marked `neither` are not exported as YOLO labels. Background negatives can be exported as empty label files when enabled.

## Safety Notes

- Run from the repository root so relative workspace paths resolve correctly.
- Do not write to `/home/gavin/Downloads/AIS data`; it is read-only project input.
- Prefer JSON output for tests, scripts, and agent diagnostics.
- Use `--dry-run` before exports when checking path overrides or manifest changes.
- The CLI currently implements backend preparation and diagnostics. It does not yet provide `start-training`, `training-status`, or `stop-training`; those are planned in `docs/training_pipeline_plan.md`.
