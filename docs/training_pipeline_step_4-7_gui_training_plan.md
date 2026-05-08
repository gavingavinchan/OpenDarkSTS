# Training Pipeline GUI Training Plan

This document is the implementation plan for making the GUI able to reproduce the existing LLM-assisted YOLO training workflow through the OpenDarkSTS training pipeline interfaces.

Authoritative context remains in `docs/training_pipeline_plan.md`. The whole-project pipeline split remains in `docs/grand_plan_v2.md`. CLI contracts and examples live in `docs/training_pipeline_cli.md`. Weight provenance and recreation requirements live in `docs/yolo_weight_recreation.md`.

## Goal

The user should be able to import imagery into albums, review detections, choose reviewed images for a dataset, configure YOLO training, and launch/monitor training without asking an LLM to run raw `yolo detect train ...` commands directly.

Manual terminal YOLO training was previously successful, but the repeatable path should become:

```text
GUI action or training_pipeline_cli command
  -> backend helper in scripts/training_pipeline_lib.py
  -> workspace artifact written with schema_version: 1
  -> detached YOLO process launched by the pipeline when training starts
```

The final validation target is that an agent can retrain from the already reviewed datasets and existing starting weights using only `scripts/training_pipeline_cli.py` commands, not direct ad hoc YOLO shell commands.

## Scope Order

Do album management first, then steps 4-6.

```text
Step 1 album management fixes
  -> Step 4 Dataset Selection
  -> Step 5 Training Setup
  -> Step 6 Train YOLO
  -> Step 7 Evaluation later
```

Step 7 remains deferred except for recording the run metadata needed by future evaluation.

## Prerequisite: Make Albums Work In Step 1

Current state: backend album-aware import exists, and the import UI has some album fields, but the workflow is not clear enough. The user does not see an obvious way to add photos to an album or manage albums.

Existing backend pieces should be reused rather than rewritten. `scripts/training_pipeline_lib.py` already has album ids, album ledgers, album index writes, import-job ledgers, image-key membership merging, manifest `album_ids`, and `import_training_data(..., append_to_album_id=...)`. The missing work is clearer GUI affordance, read/list APIs, empty-album creation, existing-image album membership management, and CLI parity.

### Required Step 1 GUI Behavior

Add explicit album controls to `/import`:

- Primary action: `Add Photos To Album`.
- Secondary action: `Create New Album`.
- Album picker: select an existing album before upload.
- Album creation fields: name and optional description.
- Import target summary: show whether selected files will create a new album or append to an existing album.
- Album grid/list: show existing albums with name, description, image count, reviewed count, imported count, last updated time, and actions.
- Album detail view/filter: clicking an album shows only images in that album.
- Add-to-album flow: importing additional photos into an existing album must call the same backend path as CLI `import-data --append-to-album-id`.
- Manual-review handoff: album detail actions should include `Review This Album`, routing to `/manual-review?album_id=<album_id>` so the user can review only the newly imported batch when desired.
- Album sidebar: clicking the body of an album card opens an album detail panel in the right sidebar. The panel shows `album_id`, `album_name`, `album_description`, `source_kind`, `source_display_name`, `created_at`, `updated_at`, `latest_import_job_id`, `image_count`, review completed/total, and the album `counts` breakdown. The same panel hosts the rename form and per-album actions (`Add Photos`, `Review`, `Filter`, `Remove From Album`). Image-detail mode in the sidebar still works for a selected image; album-detail mode takes priority when both are set, and the panel includes a clear-selection control.
- Rename action: each album supports renaming the human `album_name` and editing `album_description`. The rename UI lives inside the album sidebar as an inline form (no `window.prompt`). `album_id` remains immutable; `created_at` is preserved; `updated_at` advances on rename.
- Immediate album naming on import: starting a new album from the import page must accept the album name inline before any OS file dialog opens. A styled in-page naming card replaces `window.prompt` whenever the user invokes Pick Images / Pick Folder without a target album and without a name already set, and the card is also the explicit `+ Import to new album` entry point in both the empty and populated states.
- No source modification: importing or adding to albums never modifies source folders or source files.
- Preserve existing image selection/removal behavior, but make destructive scope explicit. `Remove From Album` only removes album membership. `Remove From Workspace` removes import records, active review files, archived review files, cache/staging copies, and album membership, while leaving original external source files untouched.

### Required Step 1 Backend/API Behavior

Reuse or add helpers in `scripts/training_pipeline_lib.py`:

- Reuse existing `new_album_id`, `load_album_record`, `write_album_index`, `merge_album_images`, `add_album_id_to_entry`, and `import_training_data(..., append_to_album_id=...)` helpers.
- Add list helpers that read `workspace/albums/index.json` and enrich summaries with review counts.
- Add a helper to create a new empty album without requiring immediate image upload.
- Add a helper to append existing manifest-backed image keys to an album without re-importing image bytes.
- Add a rename helper `rename_album(albums_dir, album_id, *, album_name=None, album_description=None)` that updates the album record and `index.json` summary, refreshes `updated_at`, preserves the immutable `album_id` and `created_at`, and supports updating either field independently.
- Return album summaries with image counts, reviewed counts, imported counts, source metadata, and latest import job.
- Preserve immutable `album_id`; allow mutable album name/description.

Empty album creation should write `workspace/albums/<album_id>.json` and update `workspace/albums/index.json` using the current album schema. The empty album ledger should include:

```json
{
  "schema_version": 1,
  "album_id": "alb_...",
  "album_name": "Singapore 2025",
  "album_description": null,
  "source_display_name": null,
  "source_kind": "manual_album",
  "import_job_ids": [],
  "created_at": "2026-05-07T00:00:00Z",
  "updated_at": "2026-05-07T00:00:00Z",
  "created_by": "gavin",
  "reviewer": "gavin",
  "manifest_path": "workspace/manifests/training_import_manifest.json",
  "cache_dir": "workspace/cache/imported_images",
  "reviews_dir": "workspace/reviews",
  "options": {},
  "counts": {
    "uploaded": 0,
    "discovered": 0,
    "imported": 0,
    "skipped": 0,
    "failed": 0,
    "members_added_imported": 0,
    "members_added_existing": 0,
    "members_already_present": 0,
    "members_failed": 0,
    "review_stubs_created": 0,
    "review_stubs_existing": 0,
    "review_stubs_skipped": 0,
    "auto_detect_created_review": 0,
    "auto_detect_updated_existing_review": 0,
    "auto_detect_protected_existing_review": 0,
    "auto_detect_skipped": 0
  },
  "images": []
}
```

The index summary for an empty album should set `image_count: 0`, preserve `counts`, and set `latest_import_job_id: null`.

Album list, filter, and CLI output code must treat `source_kind` as an open string enum and support `manual_album` in addition to existing values such as `browser_upload`, `local_file`, and `local_folder`.

Add GUI API routes in `scripts/training_pipeline_ui.py`:

- `GET /api/albums`
- `GET /api/albums/<album_id>`
- `POST /api/albums`
- `POST /api/albums/<album_id>/add-images`
- `POST /api/albums/<album_id>/remove-images`
- `PATCH /api/albums/<album_id>` body `{album_name?, album_description?}` for rename / description edits.
- Existing upload route should accept either `album_name` for new album or `append_to_album_id` for existing album.
- `GET /api/runs/active` is not needed for albums, but later Step 6 should use an equivalent active-run API for persistent run cards.

Album UI actions should log through the existing `/api/training-ui-interaction` endpoint. Do not add a second interaction logging mechanism.

Add CLI parity if missing:

- `list-albums`
- `show-album`
- `create-album`
- `add-to-album`
- `remove-from-album`
- `rename-album --album-id <id> [--album-name <name>] [--album-description <desc>]`

`add-to-album` should support adding existing manifest-backed `--image-key` values. Adding new local files to an album should continue to use `import-data --append-to-album-id`. `rename-album` accepts either `--album-name` or `--album-description` independently and rejects calls that touch neither.

These commands should use JSON envelopes consistent with `docs/training_pipeline_cli.md`.

### Step 1 Acceptance Checks

- User can create an empty album in the GUI.
- User can upload files into a new album from the GUI.
- User can add another upload batch to an existing album from the GUI.
- Album list counts update after import.
- Filtering by album shows the expected images.
- CLI can list/show/create/add to albums using the same backend helpers.
- UI interaction logs record album creation, album selection, upload target, and add-to-album actions in `workspace/reports/training_pipeline_ui_interactions.jsonl`.
- `/manual-review?album_id=<album_id>` opens an album-scoped review queue.

## Step 4: Dataset Selection

Purpose: choose reviewed whole images from one or more albums and save a reproducible dataset selection manifest.

### Backend

Add helpers in `scripts/training_pipeline_lib.py`:

- List dataset candidates from completed review JSONs.
- Join candidates to import manifest entries and album metadata.
- Filter candidates by album, reviewer, review completion, and label class.
- Count candidate images and bboxes by album, split, and YOLO class.
- Save `workspace/dataset_selections/<dataset_name>.json`.
- Auto-assign `train`/`val`/`test` splits with default `80/10/10` and seed `42`.
- Preserve existing split assignments when editing an existing selection.
- Bbox-level selection is explicitly out of scope for the first implementation. Step 4 selects whole images; `selected_only` remains supported by backend manifests for existing/export diagnostics but should not be exposed as the default GUI workflow yet.
- Reconciliation conflict handling remains the existing export-time hard blocker in `find_review_for_image`: if more than one completed review JSON matches one image key, export fails. Do not add a new reconciliation-state filter until the review schema defines one.

Use existing `export_dataset_selection` for YOLO export.

### GUI

Add `/dataset-selection`:

- Album multi-select is the first filter.
- Image-level selection is the first-version default.
- Show selected image count and reviewed bbox counts for `ship` and `ships`.
- Let the user choose dataset name, version, and description.
- Let the user review/adjust splits.
- Provide `Save Selection`, `Export YOLO Dataset`, and `Continue To Training Setup` actions.

### CLI

Add commands:

- `list-dataset-candidates`
- `save-dataset-selection`

Keep existing:

- `export-dataset`

## Step 5: Training Setup

Purpose: configure one or more YOLO runs against an exported dataset.

### Backend

Add helpers in `scripts/training_pipeline_lib.py`:

- List dataset selections and exported datasets.
- Save `workspace/training_configs/<run_group>.json`.
- Validate config fields using existing `check_training_readiness`.
- Build command preview using existing `build_yolo_train_command`.

Training config must include:

- dataset name/version
- model or starting weights
- image size
- epochs
- batch size
- patience
- seed
- device
- workers
- augmentation settings
- output run group and run name
- overwrite policy

### GUI

Add `/training-setup`:

- Select exported dataset.
- Select model/start weights, including existing repo root weights and promoted weights under `workspace/model_weights/`.
- Surface starting-weight provenance. For weights inside the repository, show whether `docs/yolo_weight_recreation.md` or an adjacent weight-specific doc documents the weight. Missing provenance should warn in the GUI and CLI config save/check output. Promoting or adding weights without documentation remains forbidden.
- Expose explicit fields for image size, epochs, batch, patience, seed, device, workers, run group, and run name.
- Store augmentation settings in config, not as pre-rendered images.
- Show readiness checklist.
- Show exact command preview.
- Keep `Run Training` disabled until backend readiness is true.

### CLI

Add:

- `save-training-config`

Keep existing:

- `check-training-ready`
- `build-train-command`

Minimum augmentation fields for first implementation should be a whitelist of Ultralytics train args passed through by `build_yolo_train_command`:

- `hsv_h`
- `hsv_s`
- `hsv_v`
- `degrees`
- `translate`
- `scale`
- `shear`
- `perspective`
- `flipud`
- `fliplr`
- `mosaic`
- `mixup`
- `copy_paste`
- `erasing`
- `crop_fraction`

The backend should reject unknown augmentation keys unless an explicit advanced pass-through mode is added later. The GUI may show common fields first and hide the rest in an advanced section.

## Step 6: Train YOLO

Purpose: launch and monitor training jobs from pipeline-managed configs.

### Backend

Add helpers in `scripts/training_pipeline_lib.py`:

- `start_training_run(...)`
- `training_status(...)`
- `stop_training_run(...)`

`start_training_run` must:

- run readiness checks first
- build the YOLO command from the config
- start the process detached by default
- write `workspace/runs/<run_group>/<run_name>/train.log`
- write `workspace/runs/<run_group>/<run_name>/process.json`
- write initial `training_metadata.json`
- record enough process identity to guard against PID reuse when feasible, such as PID, command line, launch time, and platform-specific process start time when available.

`training_status` must:

- read `process.json` and `training_metadata.json`
- check process state safely with PID-reuse protection where available
- read log tail
- parse live `results.csv` using `load_training_results_series`
- reconcile state lazily on every read, because there is no separate watcher daemon in the first implementation
- update terminal state when a process has completed or failed
- if `process.json` says `running` but the recorded process is gone and no return code can be recovered, mark the process and training metadata as `failed` with a diagnostic reason such as `process_missing` rather than leaving it permanently running

`stop_training_run` must:

- attempt graceful termination
- mark the run `cancelled`
- leave partial files in place for debugging

### GUI

Add `/train-yolo`:

- Start training from the selected dataset/config.
- Show an active run card that survives stage navigation.
- Poll status every few seconds.
- Show state, PID, elapsed time, latest metrics, log tail, and plots from `results.csv`.
- Provide `Stop Training`.
- On completion, show `best.pt`, `last.pt`, run directory, and metadata paths.

Persistent active run cards need an API mechanism rather than page-local state only. Add run APIs such as:

- `GET /api/runs/active`
- `GET /api/runs/<run_group>/<run_name>/status`
- `POST /api/runs/<run_group>/<run_name>/stop`

The exact route shape may vary, but the GUI must be able to discover active runs after navigation or page refresh.

### CLI

Add:

- `start-training`
- `training-status`
- `stop-training`

The CLI is the required final retraining interface. Agents should retrain by invoking these commands, for example:

```bash
.venv/bin/python scripts/training_pipeline_cli.py export-dataset --selection workspace/dataset_selections/<dataset>.json
.venv/bin/python scripts/training_pipeline_cli.py check-training-ready --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<run_group>.json
.venv/bin/python scripts/training_pipeline_cli.py build-train-command --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<run_group>.json
.venv/bin/python scripts/training_pipeline_cli.py start-training --selection workspace/dataset_selections/<dataset>.json --config workspace/training_configs/<run_group>.json
.venv/bin/python scripts/training_pipeline_cli.py training-status --run-dir workspace/runs/<run_group>/<run_name>
```

The agent may inspect the generated command, but should not bypass the pipeline by launching raw `yolo detect train ...` directly.

The final smoke check should poll `training-status` until a terminal state is observed, then verify the run directory contains `training_metadata.json`, `train.log`, and the expected Ultralytics output artifacts or a clear failure diagnostic.

## Step 7 Deferred

Do not build full evaluation yet. Step 6 should only prepare clean artifacts for later evaluation:

- complete `training_metadata.json`
- command line
- dataset manifest path/hash
- starting weights path/hash when available
- package versions
- git commit and dirty flag
- best/last checkpoint paths

The `/evaluate` stage may remain disabled or show a stub until evaluation is implemented.

## Final Validation Target

Before considering steps 1 album fixes and steps 4-6 complete:

- GUI can create and populate albums.
- CLI can create/list/show/add to albums.
- GUI can create a dataset selection from album-filtered reviewed images.
- CLI can save/export the same dataset selection.
- GUI can save a training config and preview readiness.
- CLI can run `check-training-ready` and `build-train-command` for that config.
- CLI can launch training through `start-training` and monitor it through `training-status`.
- A smoke retraining run can complete using existing reviewed data and existing starting weights, invoked only through `scripts/training_pipeline_cli.py`.
- Any new or promoted `*.pt` files kept in the repo are documented according to `docs/yolo_weight_recreation.md`.
