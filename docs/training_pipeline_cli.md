# Training Pipeline CLI

`scripts/training_pipeline_cli.py` is the machine-readable command-line interface for the training-side pipeline. It calls the same backend helpers used by the reviewer and training pipeline code, so prefer it for automation, smoke checks, and LLM/developer diagnostics instead of reimplementing pipeline logic in shell scripts.

The implementation plan for album management and GUI-driven training is in `docs/training_pipeline_step_4-7_gui_training_plan.md`. That plan defines the expected CLI parity for album commands, dataset selection commands, training config creation, and pipeline-managed YOLO retraining.

Selection and workspace image-removal semantics are specified in `docs/review_queue_selection_removal_plan.md`.

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
- `1`: pipeline or runtime failure, such as an unreadable input file.
- `2`: command-line usage or backend validation error (`ValueError`), including invalid run paths and malformed JSON objects. The JSON error envelope is preserved.
- `3`: readiness command completed successfully but found blockers. In this case `ok` is still `true` and `result.ready` is `false`. `start-training` also exits `3` with `ok: false` when readiness blocks the launch, or when a live process already owns the target run directory.
- `4`: `training-status` ran successfully and observed a `failed` run.

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
- `--auto-detect`: run YOLO inference and create or enrich safe review stubs with automatic bboxes during import.
- `--auto-detect-weight <name=path>`: YOLO weight to use for import-time auto detection, repeatable. Defaults to the current YOLOv8n auto-detect weight when `--auto-detect` is set.
- `--auto-detect-conf <float>`: YOLO confidence threshold. Defaults to `0.25`.
- `--auto-detect-iou <float>`: class-aware merge IoU threshold. Defaults to `0.5`.
- `--recursive`: scan folders recursively. Hidden files and directories are skipped.
- `--force`: refresh an existing manifest/cache entry.
- `--no-review-stubs`: skip creating missing step 2 review JSON stubs.
- `--dry-run`: validate inputs and report actions without writing the manifest or cache.

The command never modifies source files. It writes or updates the import manifest, cache, and missing review stubs unless `--dry-run` is used. Already-imported identical image/profile pairs are skipped and reported in `skipped_count`; their missing review stubs are still created unless `--no-review-stubs` is set.

The integrated GUI upload path enables auto detection by default. The CLI keeps it explicit through `--auto-detect` so automation can choose between fast manifest/cache import and YOLO-backed import.

### `run-auto-detect`

Runs YOLO inference for images already present in the import manifest and creates or enriches safe step 2 review stubs. Existing completed, reviewed, labeled, or manual-bbox review files are protected and not overwritten.

```bash
.venv/bin/python scripts/training_pipeline_cli.py run-auto-detect \
  --reviewer gavin
```

Options:

- `--manifest <path>`: import manifest path. Defaults to `workspace/manifests/training_import_manifest.json`.
- `--reviews-dir <path>`: review root. Defaults to `workspace/reviews`.
- `--reviewer <name>`: reviewer folder to update.
- `--image-key <key>`: restrict to one image key, repeatable.
- `--limit <n>`: only process the first `n` requested/imported images.
- `--weight <name=path>`: YOLO weight to use, repeatable. Defaults to the current YOLOv8n auto-detect weight.
- `--conf <float>`: YOLO confidence threshold. Defaults to `0.25`.
- `--iou <float>`: class-aware merge IoU threshold. Defaults to `0.5`.
- `--dry-run`: report which review files would be created, updated, or protected without loading YOLO or writing files.

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

In both modes `exported_image_count` and `exported_label_count` count the same thing: one image and one YOLO label file per exported item, including enabled background negatives. `--dry-run` does not read review JSONs, so it cannot report per-bbox label row counts.

An export is a replacement, not a merge: every `images/<split>` and `labels/<split>`
directory is emptied before the selection is written, so the exported dataset always equals
the selection exactly. Re-exporting after dropping an image, or after moving an image
between splits, leaves no stale copy behind for YOLO to train on. The written
`dataset_manifest.json` records `dataset_selection_manifest_sha256`, which
`check-training-ready` uses to reject a stale export.

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

### `list-dataset-candidates`

Lists completed review images that can be saved into a Step 4 dataset selection. Candidates can be filtered by album and reviewed label class.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-dataset-candidates \
  --reviewer gavin \
  --album-id alb_abc123 \
  --label exportable
```

Options:

- `--manifest <path>`: import manifest. Defaults to `workspace/manifests/training_import_manifest.json`.
- `--albums-dir <path>`: album ledger root. Defaults from the manifest path.
- `--reviews-dir <path>`: review root. Defaults to `workspace/reviews`.
- `--reviewer <name>`: reviewer folder. Defaults to `$USER`, or `default` when unset.
- `--album-id <id>`: restrict to album membership, repeatable.
- `--label ship|ships|neither|exportable`: restrict to candidates containing that reviewed label, repeatable.
- `--include-incomplete`: include non-completed reviews. The default is completed reviews only.

### `save-dataset-selection`

Saves a Step 4 dataset selection manifest with deterministic train/val/test splits.

Splits are **group-aware by default** (`--split-mode group`): every image is keyed to a
`<region>|<date>` scene group using the same rule as `scripts/build_clean_sts_dataset.py`
(region from the import manifest's raster bounds, then source path, then album majority;
date parsed from the most original filename), and a whole group always lands in exactly
one split. This is what stops adjacent tiles of the same Sentinel-2 observation appearing
in both train and val/test. `--split-mode image` restores the old per-image shuffle and is
only an escape hatch; it will normally report a non-zero `groups_spanning_splits`.

The command result reports `split_mode`, `group_key_rule`, `split_counts`, `group_count`
and `groups_spanning_splits`, and each selection item carries its `group_key`.

```bash
.venv/bin/python scripts/training_pipeline_cli.py save-dataset-selection \
  --dataset-name manual_reviewed_gavin \
  --dataset-version v1 \
  --image-key img_abc \
  --image-key img_def
```

Options:

- `--dataset-name <name>`: required human dataset name.
- `--dataset-version <version>`: defaults to `v1`.
- `--description <text>`: optional description stored in the selection.
- `--image-key <key>`: selected whole image, repeatable and required.
- `--album-id <id>`: source album metadata, repeatable.
- `--seed <int>`: deterministic split seed. Defaults to `42`.
- `--split-ratios train=0.8,val=0.1,test=0.1`: optional split ratios.
- `--split-mode group|image`: defaults to `group` (scene-safe). `image` is the legacy leaky shuffle.
- `--selections-dir <path>`: defaults to `workspace/dataset_selections`.
- `--datasets-dir <path>`: default export root is `workspace/datasets`.
- `--manifest`, `--reviews-dir`, `--reviewer`: source metadata stored in the selection.

### `preview-dataset-split`

Computes the exact splits `save-dataset-selection` would write, without saving anything.
The step 4 GUI Split Preview calls the same helper through `POST /api/dataset-selections/split-preview`.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty preview-dataset-split \
  --image-key img_abc --image-key img_def
```

Options mirror `save-dataset-selection`: `--seed`, `--split-ratios`, `--split-mode`,
`--manifest`, `--reviews-dir`, `--reviewer`. The result reports `split_counts`,
`group_count`, `groups_spanning_splits`, `group_key_rule` and the per-image `items`.

### `list-dataset-selections`

Lists saved Step 4 dataset selection manifests and reports item and split counts.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-dataset-selections
```

Each row also reports `split_mode`, `group_count`, `groups_spanning_splits`, `exported`
(an export directory exists) and `export_is_current` (that export was built from the
selection file exactly as it stands now).

### `check-training-ready`

Checks whether an exported dataset and training config are sufficient for YOLO training.

```bash
.venv/bin/python scripts/training_pipeline_cli.py check-training-ready \
  --selection workspace/dataset_selections/manual_reviewed_gavin_v1.json \
  --config workspace/training_configs/manual_reviewed_gavin_v1_yolov8n_smoke.json
```

Checks include:

- dataset export exists and contains `dataset_manifest.json`
- `data.yaml` exists and its exported `names` block defines exactly `0: ship` and `1: ships`
- `train` and `val` image/label folders exist and contain images
- the export is current: `dataset_manifest.json` records the sha256 of the selection
  manifest it was built from, and readiness fails when the selection has changed since
  (re-export in step 4)
- required training config fields are present
- target run directory will not overwrite an existing run unless `overwrite` is enabled

Readiness distinguishes a dataset that has not been exported from an existing export
directory missing its manifest. Class-name mismatches block training. Duplicate or
singular/plural names produce warnings; the intentional `ship` / `ships` pair is allowed
and its warning is also recorded in exported dataset manifests.

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

### `save-training-config`

Writes a Step 5 training config to `workspace/training_configs/<run-group>.json` and reports readiness.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty save-training-config \
  --dataset-name manual_reviewed_gavin_balanced_1to1 \
  --dataset-version v1 \
  --model yolov8n.pt \
  --run-group manual_reviewed_gavin_balanced_1to1_v1_e150_img960 \
  --run-name yolov8n_e150_img960 \
  --image-size 960 --epochs 150 --batch 8 --patience 30 --seed 42 \
  --device 0 --workers 4 \
  --aug fliplr=0.5 --aug mosaic=1.0 \
  --selection workspace/dataset_selections/manual_reviewed_gavin_balanced_1to1_v1.json
```

Options:

- `--config-name`: override the saved filename stem (defaults to the run group).
- `--configs-dir`, `--output-dir`: override the config directory and the run output root.
- `--selection`: validate the config against an exported dataset with `check_training_readiness`.
- `--overwrite`: allow the run directory to be reused.
- `--aug key=value`: repeatable augmentation setting.

Only these Ultralytics augmentation args are accepted; anything else is rejected with exit `1`:

`hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `mosaic`, `mixup`, `copy_paste`, `erasing`, `crop_fraction`.

The result includes `config_path`, `ready`, `missing`, and the saved `config`. Warnings call out starting weights that are not mentioned in `docs/yolo_weight_recreation.md`.

### `list-training-configs`

Lists saved training configs with run group/name, model, key hyperparameters, and any missing required fields.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-training-configs
```

### `list-exported-datasets`

Lists exported YOLO datasets under `workspace/datasets/` with per-split image counts and `data.yaml` presence.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-exported-datasets
```

### `list-starting-weights`

Lists usable starting weights from the repository root, `workspace/model_weights/`, and `workspace/runs/**/weights/best.pt`.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-starting-weights
```

Each entry reports `path`, `source` (`repo_root`, `model_weights`, `run_best`), `documented`, and `provenance_warning`. A weight counts as documented when `docs/yolo_weight_recreation.md` mentions its repo-relative path, its filename, or the matching `<run_name>` run-group row. Undocumented weights are also surfaced in the envelope `warnings`.

### `start-training`

Launches the exact `build-train-command` command as a detached process. This is the supported way to retrain; do not run `yolo detect train ...` by hand.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty start-training \
  --selection workspace/dataset_selections/manual_reviewed_gavin_balanced_1to1_v1.json \
  --config workspace/training_configs/manual_reviewed_gavin_balanced_1to1_v1_e150_img960.json
```

A launch is refused when `process.json` in the target run directory still records a live
process, even with `overwrite` enabled: the CLI exits `3` with `ok: false` and
`result.already_running: true`, and `POST /api/training/runs/start` answers `409`. Stop the
existing run first, otherwise two YOLO processes write into one directory and only the
second can be stopped.

Readiness is checked first; blockers exit `3` with `ok: false` and nothing is launched.

Each launch gets a `launch_id`. Relaunching into an existing run directory rotates the
previous launch's regenerable artifacts aside (`train.log`, `results.csv`,
`test_metrics.json`, `test_eval.log` become `<stem>.<previous launch_id><suffix>`) and
rewrites `training_metadata.json` with `status: running`, so a failed relaunch can never be
reported as `completed` from the previous launch's log, metrics, or checkpoint.

On success the run directory receives:

- `train.log`: combined stdout/stderr of the YOLO process.
- `process.json`: pid, process-group id, `start_ticks` from `/proc` for PID-reuse protection, `launch_id`, `rotated_previous_artifacts`, command line, argv, and launch time.
- `training_metadata.json`: written by `write_training_metadata` with `status: running`.

`--yolo-executable` overrides the resolved `yolo` binary (default `.venv/bin/yolo`, then `PATH`).

### `training-status`

Reports live status for a pipeline-managed run and lazily reconciles terminal state; there is no watcher daemon.

Run paths, configured output roots, and run references must stay under `workspace/runs`,
including their resolved symlink targets. Group and run names must be single path
segments. These checks also apply to stopping, listing, evaluation, and promotion.
Tests use uniquely allocated temporary directories under that root.

A corrupt `process.json` produces a status warning and an inferred terminal status
without rewriting the damaged record. Stopping such a run is refused. Recorded logs
are read only within the run directory; an escaped log falls back to a contained
`train.log`, or is omitted if that path also escapes.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty training-status \
  --run-dir workspace/runs/<run_group>/<run_name>
```

Accepts `--run-dir`, or `--run-group` plus `--run-name` (with optional `--runs-dir`). Other options: `--log-lines` (default 40) and `--include-series` to include the full `results.csv` series from `load_training_results_series`.

The result includes `status`, `terminal`, `alive`, `pid`, `reason`, `current_epoch`, `latest_metrics`, `best_metrics`, `log_tail`, `best_weights_path`, and `last_weights_path`.

State transitions happen on read. When `process.json` says `running` but the recorded process is gone, the run becomes `completed` if the log shows a completion marker (or `best.pt` plus the configured epoch count is present), and otherwise `failed` with `reason: process_missing`. Terminal state is written back to both `process.json` and `training_metadata.json`. A `failed` run exits `4` with `ok: false`.

### `stop-training`

Terminates a run's process group (SIGTERM, then SIGKILL after `--grace-seconds`, default 10) and marks it `cancelled`. Partial artifacts are left in place for debugging.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty stop-training \
  --run-dir workspace/runs/<run_group>/<run_name> --grace-seconds 10
```

### `list-active-runs`

Lists pipeline-managed runs under `workspace/runs/<run_group>/<run_name>/process.json`, reconciling each one's state.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-active-runs
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-active-runs --all
```

By default only non-terminal runs are returned; `--all` includes completed, failed, and cancelled runs.

### `list-run-groups`

Lists every run group under `workspace/runs/` that holds at least one run with a `results.csv`, including
legacy runs that predate `process.json`. This is what the step 7 GUI page uses to populate its run-group picker.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty list-run-groups
```

### `show-run-group`

Returns the per-run validation/test metric summaries for one run group plus the suggested promotion target,
without writing anything. The step 7 results table is this command's output rendered as a table.
`validation_metrics` describes the `best.pt` epoch; `final_validation_metrics` and
`best_validation_epoch` are reported alongside it.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty show-run-group --run-group manual_reviewed_gavin_v1_smoke
```

### `write-evaluation-report`

Writes `workspace/reports/<run_group>_evaluation.json` and `.md` for every run in a run group that has a `results.csv`.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty write-evaluation-report \
  --run-group manual_reviewed_gavin_balanced_1to1_v1_e150_img960
```

Validation metrics come from the **best** `results.csv` row: the epoch with the highest
mAP50-95, which is the fitness Ultralytics uses to decide when to save `best.pt`. That is the
checkpoint offered for promotion, so ranking and display use it; the last row is still
reported separately as `final_validation_metrics`, and `best_validation_epoch` names the
epoch. Using the final row instead made a run that peaked early and collapsed look as bad as
its last epoch and could recommend a worse checkpoint from another run. Test metrics come from `test_metrics.json` in the run directory when `run-test-split` has been used. The promoted run defaults to the highest test mAP50, falling back to validation mAP50; override with `--promote-run-name` and `--promotion-reason`. `--baseline` accepts a JSON file of baseline metrics, and `--include-runs` embeds the full per-run detail in the envelope.

### `run-test-split`

Evaluates a run's `best.pt` against the dataset `test` split and stores `test_metrics.json` next to the run.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty run-test-split \
  --run-group <run_group> --run-name <run_name> --device cpu
```

The dataset and image size are read from the run's `training_metadata.json` unless `--dataset-dir` and `--image-size` are given. Ultralytics output is appended to `test_eval.log` in the run directory. This command runs synchronously and can take a while on CPU.

### `promote-weight`

Copies a run's `best.pt` to `workspace/model_weights/<target-name>.pt`.

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty promote-weight \
  --run-group <run_group> --run-name <run_name> \
  --target-name yolov8n_example_2026-05-20_best.pt \
  --recreation-doc-updated
```

The command enforces the repository YOLO weight reproducibility rule and exits `2` without copying anything when either check fails:

- `--recreation-doc-updated` must be passed, acknowledging that `docs/yolo_weight_recreation.md` is already updated.
- `--target-name` must already be mentioned in `docs/yolo_weight_recreation.md` as a whole
  filename token. A substring match is not enough: promoting to `26n.pt` is refused even
  though the document mentions `yolo26n.pt`, because the promoted destination has no recipe.

Document the weight first, then promote. `--overwrite` is required to replace an existing promoted weight.

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

### `remove-images`

Removes selected images from the local training workspace. This deletes import manifest entries, album membership, active review JSONs for the selected reviewer, legacy archived review JSONs for the same reviewer/key, workspace cache files, and browser-upload staging files under `workspace/cache/import_staging`. It does not modify original source imagery outside the repository/workspace.

```bash
.venv/bin/python scripts/training_pipeline_cli.py remove-images \
  --reviewer gavin \
  --image-key img_abc \
  --image-key img_def
```

Options:

- `--reviewer <name>`: reviewer folder to clean. Defaults to `$USER`, or `default` when unset.
- `--reviews-dir <path>`: review root. Defaults to `workspace/reviews`.
- `--manifest <path>`: import manifest path. Defaults to `workspace/manifests/training_import_manifest.json`.
- `--albums-dir <path>`: album ledger root. Defaults from the manifest path.
- `--reports-dir <path>`: removal report destination. Defaults to `workspace/reports/image_removals`.
- `--image-key <key>`: image key to remove, repeatable. Explicit keys are removed regardless of reviewed/unreviewed status.
- `--filter inactive-review`: expand to manifest-backed images that no longer have an active review JSON for the selected reviewer.
- `--dry-run`: preview removable/missing counts and affected paths without deleting files or updating records.

Report path:

```text
workspace/reports/image_removals/<timestamp_ms>.json
```

Use `remove-images --dry-run` before bulk removal. Example for cleaning old manifest rows whose active review JSONs are already absent:

```bash
.venv/bin/python scripts/training_pipeline_cli.py --pretty remove-images \
  --reviewer gavin \
  --filter inactive-review \
  --dry-run
```

### Legacy: `prune-review-queue`

Archives safe unreviewed review stubs without deleting source images, cache PNGs, manifest entries, or album records.

```bash
.venv/bin/python scripts/training_pipeline_cli.py prune-review-queue \
  --reviewer gavin \
  --image-key img_abc \
  --image-key img_def
```

Options:

- `--reviewer <name>`: reviewer folder to inspect. Defaults to `$USER`, or `default` when unset.
- `--reviews-dir <path>`: review root. Defaults to `workspace/reviews`.
- `--reports-dir <path>`: prune report destination. Defaults to `workspace/reports/review_queue_prune`.
- `--image-key <key>`: image key to evaluate, repeatable.
- `--filter unreviewed`: expand to safe unreviewed stubs; when combined with explicit keys, reports which explicit keys match the unreviewed/archive-safe rule.
- `--dry-run`: preview archivable/protected counts without moving files.

Protection rules are conservative. Only initial `overview_1` review files with no reviewed detections, no review labels, and no manual detections are archived. Corrupt, missing, completed, partially reviewed, manually labeled, and ambiguous review files are protected and reported.

Archive path:

```text
workspace/reviews/_archived_unreviewed/<timestamp_ms>/<reviewer>/<image_key>.json
```

Report path:

```text
workspace/reports/review_queue_prune/<timestamp_ms>.json
```

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

## Retraining Runbook

The full pipeline-managed retraining loop, with no hand-written YOLO shell commands:

```bash
.venv/bin/python scripts/training_pipeline_cli.py export-dataset --selection workspace/dataset_selections/<dataset>.json
.venv/bin/python scripts/training_pipeline_cli.py save-training-config --dataset-name <name> --model yolov8n.pt --run-group <group> --run-name <run> --selection workspace/dataset_selections/<dataset>.json
.venv/bin/python scripts/training_pipeline_cli.py check-training-ready --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<group>.json
.venv/bin/python scripts/training_pipeline_cli.py build-train-command --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<group>.json
.venv/bin/python scripts/training_pipeline_cli.py start-training --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<group>.json
.venv/bin/python scripts/training_pipeline_cli.py training-status --run-group <group> --run-name <run>
.venv/bin/python scripts/training_pipeline_cli.py run-test-split --run-group <group> --run-name <run> --device cpu
.venv/bin/python scripts/training_pipeline_cli.py write-evaluation-report --run-group <group>
```

## Training API Routes

`scripts/training_pipeline_ui.py` exposes the same behavior over HTTP for the Step 4-7 GUI pages:

| Method | Route | Backend helper |
|---|---|---|
| GET | `/api/training/exported-datasets` | `list_exported_datasets` |
| GET | `/api/training/configs` | `list_training_configs` |
| POST | `/api/training/configs` | `save_training_config` |
| GET | `/api/training/starting-weights` | `list_starting_weights` |
| POST | `/api/training/readiness` | `check_training_readiness` |
| POST | `/api/training/command-preview` | `build_yolo_train_command` |
| POST | `/api/training/runs/start` | `start_training_run` |
| GET | `/api/training/runs/active` | `list_training_runs` |
| GET | `/api/training/runs/<run_group>/<run_name>/status` | `training_status` |
| POST | `/api/training/runs/<run_group>/<run_name>/stop` | `stop_training_run` |
| POST | `/api/training/runs/<run_group>/<run_name>/test-split` | `run_test_split` |
| POST | `/api/training/evaluation-report` | `build_run_group_evaluation_report` |
| POST | `/api/training/promote-weight` | `promote_weight` |
| GET | `/api/training/run-groups` | `list_run_groups` |
| GET | `/api/training/run-groups/<run_group>` | `collect_run_group_summaries` + `best_run_by_metric` |
| GET | `/api/training/evaluation-report?run_group=<run_group>` | reads `workspace/reports/<run_group>_evaluation.md` |
| GET | `/api/jobs/<job_id>` | status of a detached GUI job (currently test-split) |

Mutating training routes log to `workspace/reports/training_pipeline_ui_interactions.jsonl` through the existing interaction-logging path.

### GUI Pages And Their CLI Equivalents

| GUI page | Stage | CLI equivalents |
|---|---|---|
| `/import` (alias `/`) | 1 | `import-data`, `list-albums`, `create-album`, `add-to-album`, `rename-album`, `remove-from-album`, `remove-images` |
| `/manual-review` | 2 | `run-auto-detect`, `image-key` |
| `/bbox-overview` | 3 | `list-dataset-candidates` |
| `/datasets` (alias `/dataset-selection`) | 4 | `list-dataset-candidates`, `preview-dataset-split`, `save-dataset-selection`, `list-dataset-selections`, `export-dataset` |
| `/training-setup` | 5 | `list-exported-datasets`, `list-starting-weights`, `save-training-config`, `list-training-configs`, `check-training-ready`, `build-train-command` |
| `/training` (alias `/train-yolo`) | 6 | `start-training`, `training-status`, `stop-training`, `list-active-runs` |
| `/evaluation` (alias `/evaluate`) | 7 | `list-run-groups`, `show-run-group`, `run-test-split`, `write-evaluation-report`, `promote-weight` |

Known GUI/CLI parity gaps, tracked here rather than solved with GUI-only behavior:

- Test-split evaluation runs synchronously in the CLI (`run-test-split`) but as a detached polled job in the
  GUI (`POST .../test-split` with `{"background": true}` then `GET /api/jobs/<job_id>`). The work performed is
  identical; only the wait is different.
- The GUI reads the generated markdown report back through `GET /api/training/evaluation-report?run_group=...`.
  The CLI equivalent is reading `workspace/reports/<run_group>_evaluation.md` directly; no command wraps that.
- Step 4's split ratio boxes preview counts client-side before saving. The authoritative split assignment is
  always the one returned by `save-dataset-selection` / `POST /api/dataset-selections`.

`scripts/verify_training_gui.py` drives steps 1-7 in a headless browser against the real workspace and prints a
pass/fail summary; it is the browser-side counterpart to the CLI retraining runbook.

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
