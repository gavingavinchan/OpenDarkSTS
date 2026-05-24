# Training Pipeline Step 5 Plan: Training Setup

This document is the focused implementation plan for Step 5 of the GUI training pipeline. It complements but does not replace the higher-level plan in `docs/training_pipeline_step_4-7_gui_training_plan.md`, which remains authoritative on cross-step contracts. Step 5 specifics that differ from or refine the higher-level doc are recorded here.

Related docs:

- `docs/training_pipeline_step_4-7_gui_training_plan.md` — overarching Step 1, 4-7 plan.
- `docs/training_pipeline_plan.md` — overall pipeline shape.
- `docs/training_pipeline_cli.md` — CLI envelope conventions.
- `docs/yolo_weight_recreation.md` — weight provenance registry.

Council coordination notes that fed this plan:

- `council/step5_training_setup_readiness_2026-05-24.md`
- `council/step5_suite_handling_opinion_2026-05-24.md`

## Goal

Allow the user to assemble a reproducible YOLO training configuration through a single GUI page (`/training-setup`) backed by helpers in `scripts/training_pipeline_lib.py` and CLI parity in `scripts/training_pipeline_cli.py`. The saved config is the input artifact that Step 6 ("Train YOLO") will consume to launch and monitor runs.

Step 5 does not launch training. Step 6 does.

## Scope

In scope for the first Step 5 slice:

- Single-model and multi-model "suite" configs share one on-disk schema with a `models[]` array.
- Suite-aware backend helpers via a normalization layer that expands a config into one or more run specs.
- GUI page `/training-setup` with dataset picker, model checklist, hyperparameter grid, augmentation block, run identity, live readiness checklist, and footer actions.
- CLI parity for save/list/inspect, plus device probing.
- Augmentation-key whitelist enforcement.
- Starting-weight provenance lookup against `docs/yolo_weight_recreation.md` (warn-but-allow at save time).

Out of scope for Step 5 (deferred to Step 6 or later):

- Launching training (`start_training_run`, `training_status`, `stop_training_run`).
- Live results.csv plots and metrics in the GUI.
- Per-model hyperparameter overrides inside a suite (one shared hyperparameter block plus `models[]` only).
- Bbox-level selection editing (still a Step 4 concern; out of scope here too).
- Promotion workflow for trained weights into `workspace/model_weights/`. Provenance is checked at Step 5 save, not enforced at promotion.

## Decisions Locked In

- **Suite handling: Option A.** One config file with a `models[]` array. Helpers expand the config into a list of run specs at runtime via a `expand_training_config_runs(config)` normalization layer.
- **Single vs. suite UI: not a toggle.** The page shows one flat checklist of weights. Selecting one means single-run; selecting many means suite. The user never sees a mode switch.
- **Run name derivation: weight stem + full group.** For a selected weight `yolov8n.pt` and group `manual_reviewed_gavin_balanced_1to1_v1_e150_img960`, the derived run name is `yolov8n_manual_reviewed_gavin_balanced_1to1_v1_e150_img960`. Long but self-describing if a run directory is inspected in isolation.
- **Legacy configs: load-only, save migrates.** Existing hand-authored configs with singular `model` + `run_name` continue to load and continue to produce valid YOLO commands. Pressing Save in the GUI writes the canonical `models[]` shape (one-way migration). The migrated config's expanded YOLO command is equivalent to the legacy command for the same selection, but the literal `run_name` string changes from e.g. `yolov8n_e150_img960` to the derived `yolov8n_<run_group>`. Users who need to keep the old run-name string should keep the legacy file on disk and not press Save.
- **Provenance policy at save: warn but allow.** Selecting an undocumented weight emits a readiness warning and a `provenance_warnings` array in the saved config. Save is not blocked. Blocking is reserved for the future promotion workflow.
- **Continue button: rendered disabled stub.** Mirrors the Step 4 "Continue to Training Setup" stub pattern. Tooltip says "Available in Step 6".
- **Execution mode: sequential only.** No `execution_mode` field in the schema for now. Step 6 will iterate suite runs sequentially on the user's single-GPU laptop (RTX 4070, 8GB). Parallel can be added later if the field is needed.
- **No command preview in the GUI.** The CLI still exposes `build-train-command` for manual inspection. The GUI does not surface the raw `yolo detect train ...` string because it would clutter the page without adding user value. The higher-level plan (`docs/training_pipeline_step_4-7_gui_training_plan.md`) is updated in this commit to drop its "show exact command preview" requirement; this plan is the authoritative source for that decision.
- **Dataset identity in APIs/URLs: `selection_path`.** API payloads and the `/training-setup?selection_path=<path>` query parameter use the dataset selection file path (relative to repo root), not a display name. Selection names are not guaranteed unique; the file path is.

## Schema

Saved configs continue to live at `workspace/training_configs/<group>.json` with `schema_version: 1`. Both single and suite shapes are valid; the only difference is whether `models[]` has one entry or many. Pre-existing hand-authored configs in `workspace/training_configs/` remain valid.

Canonical fields:

```json
{
  "schema_version": 1,
  "created_at": "<ISO8601>",
  "selection_path": "workspace/dataset_selections/<name>.json",
  "dataset_name": "<str>",
  "dataset_version": <int>,
  "models": ["yolov8n.pt", "yolo11n.pt"],
  "image_size": 960,
  "epochs": 150,
  "batch": 8,
  "patience": 20,
  "seed": 42,
  "device": "0",
  "workers": 4,
  "augmentation": {
    "mosaic": 1.0,
    "fliplr": 0.5
  },
  "output_dir": "workspace/runs",
  "run_group": "<group>",
  "overwrite": false,
  "provenance_warnings": []
}
```

The `selection_path` field is the canonical, stable identifier for the dataset selection a config trains against. It is repo-root-relative and must point at a JSON file whose `export.output_dir` exists and passes `list_exported_datasets` availability. Including it in the config makes the file self-contained for Step 6 launch and for future retraining diagnostics.

Selection resolution contract for `check_training_readiness`, `build_yolo_train_command`, and their CLI wrappers (`check-training-ready`, `build-train-command`):

- If `--selection <path>` is provided on the CLI (or `selection_path` is passed explicitly to the helper), use it.
- Else, if the loaded config has a `selection_path` field, use it.
- If both are provided, the explicit CLI/argument `--selection` wins. This preserves existing behavior in tests and gives a clean override for ad hoc retraining against a different selection.
- If neither is available, return the existing friendly missing-selection error and usage hint; do not silently fall back to any default.

Backwards compatibility:

- Legacy configs with `model` (singular) + `run_name` continue to load. Legacy configs typically lack `selection_path`; readiness and command helpers fall back to requiring `--selection` on the CLI in that case (existing behavior). The normalization layer treats `model` as `models: [model]` and uses the stored `run_name` for that single entry, so existing legacy configs keep producing the exact YOLO command they produce today (no behavior change for read-only / CLI consumers).
- Saving a loaded legacy config in the GUI writes the canonical `models[]` shape and drops the explicit `run_name` field. The new derived run name is `<weight_stem>_<run_group>`. This is a one-way migration: the saved file becomes canonical-shape, and the YOLO command for the migrated config differs only in the `name=` argument (the rest of the command — model, data, hyperparameters, augmentation — is byte-identical).
- Unknown fields in legacy configs are preserved on read but the GUI only edits the canonical fields. Save writes only canonical fields plus `provenance_warnings`.
- Run-name collision: when two selected weights share a basename stem (e.g. two `yolov8n.pt` files in different roots), `expand_training_config_runs` raises with a clear message naming the conflicting weights. Save and readiness fail until the user removes one or renames a weight file.

Run-name derivation:

- A run spec produced by `expand_training_config_runs(config)` has the shape `{model, run_name, run_group, image_size, epochs, batch, patience, seed, device, workers, augmentation, output_dir, overwrite}`.
- `run_name = f"{Path(model).stem}_{run_group}"`.
- The output run directory is `workspace/runs/<run_group>/<run_name>/`.

## Backend (`scripts/training_pipeline_lib.py`)

New helpers:

- `list_starting_weights() -> list[WeightEntry]` — discovers `*.pt` files in the repository root and in `workspace/model_weights/`, returns each with `path`, `basename`, `size_bytes`, `mtime`, and `provenance: {documented: bool, source_doc: str | None}`.
- `weight_provenance(basename: str) -> ProvenanceRecord` — looks up a weight's basename in `docs/yolo_weight_recreation.md` and any adjacent `docs/yolo_weight_*.md` files. Returns `{documented, source_doc, notes}`. Lookup is a literal-substring scan of the markdown files for the weight's basename; this is enough for the first slice and avoids a brittle table parser.
- `expand_training_config_runs(config: dict) -> list[RunSpec]` — central normalization. Handles both legacy `model`/`run_name` configs and new `models[]` configs. Raises `ValueError` for shapes that satisfy neither, and raises if two expanded run specs would share the same `run_name` after derivation (duplicate stem case).
- `list_exported_datasets() -> list[DatasetSummary]` — returns **every** dataset selection found in `workspace/dataset_selections/`, each annotated with an `available: bool` flag plus an optional `unavailable_reason: str`. A selection is `available: true` when its `export.output_dir` exists and passes the same dataset-side requirements that `check_training_readiness` enforces: `dataset_manifest.json` present, `data.yaml` present with classes `0: ship` and `1: ships`, and both `images/train`/`labels/train` and `images/val`/`labels/val` populated. Selections that fail any of these are returned with `available: false` and an `unavailable_reason` string so the GUI can render them disabled with a tooltip rather than drop them silently. The GUI dataset picker uses this; the CLI surfaces the same list.
- `list_training_configs() -> list[TrainingConfigSummary]` — returns existing configs in `workspace/training_configs/` with basic metadata (run_group, dataset_name, model count, created_at). Used by the GUI "Load Saved Config" picker and the new CLI `list-training-configs`.
- `save_training_config(config: dict, *, allow_unknown_augmentation: bool = False) -> dict` — validates required fields, applies the augmentation whitelist, applies provenance lookup per selected weight, writes `workspace/training_configs/<run_group>.json` atomically, and returns the saved config (with `provenance_warnings` populated).
- `probe_training_devices() -> DeviceInfo` — returns `{cuda_available, gpus: [{index, name, memory_gb}], recommended_device}`. Uses `torch.cuda.*` when available; falls back to `{cuda_available: false, recommended_device: "cpu"}` otherwise. Tolerant of `torch` import failure (returns CPU recommendation).

Augmentation whitelist (already specified in the higher-level plan):

```
hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective,
flipud, fliplr, mosaic, mixup, copy_paste, erasing, crop_fraction
```

`save_training_config` and `build_yolo_train_command` both enforce this list. Unknown keys raise unless `allow_unknown_augmentation=True` (reserved for a future advanced pass-through mode; not exposed via the GUI or CLI yet).

Suite-aware updates to existing helpers:

- `check_training_readiness(selection_path, training_config)` — when `expand_training_config_runs` yields multiple run specs, returns `{ready, missing, warnings, runs: [{model, run_name, ready, missing, warnings}]}`. Top-level `ready` is the AND of all per-run `ready`. Top-level `missing`/`warnings` are deduplicated across runs and prefixed with `[<run_name>] ` when the issue is per-run.
- `build_yolo_train_command(training_config, dataset_dir)` — returns `list[list[str]]` (one command list per run spec) when `expand_training_config_runs` yields multiple. Returns `list[str]` (the historical single command) when there is exactly one run spec. Callers must handle both shapes.

The shape change for `build_yolo_train_command` is a breaking change for downstream callers. Audit: only the CLI handler and tests call this helper today, so the impact is contained. The CLI envelope updates below absorb the change.

## CLI (`scripts/training_pipeline_cli.py`)

CLI envelope conventions follow `docs/training_pipeline_cli.md`: `ok` indicates the command executed successfully (parsed args, ran without crashing), while domain outcomes such as readiness live under `result`. Readiness blockers return exit code `3` with `ok: true` and `result.ready: false`. New Step 5 subcommands follow the same pattern.

New subcommands:

- `save-training-config --config <json_or_path> [--configs-dir <path>] [--weights-roots <path,path,...>] [--docs-roots <path,path,...>]` — accepts an inline JSON blob or a path to a JSON file with the canonical shape above. Overrides exist for tests: `--configs-dir` redirects the write target, `--weights-roots` redirects starting-weight discovery, `--docs-roots` redirects provenance lookup. On success prints the saved config envelope including `provenance_warnings`.
- `list-training-configs [--configs-dir <path>]` — JSON envelope listing summaries from `list_training_configs()`.
- `list-exported-datasets [--selections-dir <path>]` — JSON envelope listing entries from `list_exported_datasets()`. Each entry includes `available` and (when false) `unavailable_reason`, giving CLI/API parity with the GUI dataset picker.
- `list-starting-weights [--weights-roots <path,path,...>] [--docs-roots <path,path,...>]` — JSON envelope listing entries from `list_starting_weights()`.
- `device-info` — JSON envelope from `probe_training_devices()`. Useful for headless reproducibility checks. Tests must monkeypatch `torch` and the probe helper; the command does not require a real GPU.

Updated subcommands (suite-aware):

- `check-training-ready` — when the config has multiple models, `result` gains a `runs` array, one entry per expanded run spec with its own `ready`/`missing`/`warnings`. Top-level `result.ready` is the AND of per-run `ready`. The CLI envelope's `ok` field is **not** tied to readiness: it stays `true` when the command executed cleanly, even if `result.ready` is `false`. Exit code is `3` when `result.ready` is `false`, preserving existing behavior in `tests/test_training_pipeline_cli.py:796-813`.
- `build-train-command` — when the config has multiple models, `result.commands` is `list[list[str]]`, one command per expanded run spec, in the same order as `models[]`. When the config has exactly one model, the historical `result.command_line` (`list[str]`) is preserved for backward compatibility with existing tests in `tests/test_training_pipeline_cli.py:972-993`, and `result.commands` (`list[list[str]]`, length 1) is added alongside so future callers can adopt the uniform shape without breaking the old one.

Unchanged subcommands relevant here: `list-dataset-selections`, `export-dataset`.

## GUI (`scripts/training_pipeline_ui.py`)

New route: `GET /training-setup`. Layout, top to bottom:

1. **Dataset block.** Dropdown of exported datasets from `GET /api/exported-datasets`. Shows dataset name, version, train/val/test image counts, and `ship`/`ships` bbox counts. Entries with `available: false` are rendered disabled with `unavailable_reason` as a tooltip. If the user arrived from Step 4's "Continue to Training Setup" with a `?selection_path=<path>` query (URL-encoded, repo-root-relative), pre-selects that dataset.
2. **Model checklist.** Pulled from `GET /api/starting-weights`. One flat list with checkboxes. Each row shows weight basename, size, and a provenance badge: green check when documented, yellow warning when not. Hover/focus reveals the source doc filename or warning text.
3. **Hyperparameter grid.** Seven inputs in a two-column grid: `image_size`, `epochs`, `batch`, `patience`, `seed`, `device`, `workers`. Each has an inline help icon revealing a one-line "what it does" plus a rule-of-thumb tooltip. Defaults: 960, 150, 8, 20, 42, auto-detected, 4. `device` is filled from `GET /api/system/device-info` on page load.
4. **Augmentation block.** Preset row of four chips: `None`, `Light`, `Default`, `Heavy`. Selecting a preset overwrites the fields below. Common fields visible by default: `mosaic`, `mixup`, `fliplr`, `flipud`, `degrees`, `scale`, `translate`. Advanced fields collapsed: `hsv_h`, `hsv_s`, `hsv_v`, `shear`, `perspective`, `copy_paste`, `erasing`, `crop_fraction`. The `Default` preset mirrors the v4 suite config values (`mosaic=1.0`, `mixup=0.05`, `fliplr=0.5`, `flipud=0.5`, `degrees=180`, `scale=0.4`, `translate=0.1`, `hsv_s=0.35`, `hsv_v=0.25`).
5. **Run identity block.** Text input for `run_group`. A read-only auto-derived run name list (one entry per selected weight) below the input, updating live. Checkbox for "Overwrite existing run directories".
6. **Readiness checklist.** Live readout from `POST /api/training-config/check-ready` (debounced as fields change). When multiple weights are selected, expandable per-weight sub-list. Blocker items render in red, warnings in yellow, passes in muted green.

Footer (sticky):

- **Save Config** — POST to `/api/training-configs`. Disabled while the form is invalid (missing required fields). Provenance warnings do not disable Save.
- **Load Saved Config** — opens a picker (modal or sidebar) listing existing configs; selecting one populates the form.
- **Reset to Defaults** — clears the form back to schema defaults.
- **Continue to Train YOLO** — rendered disabled. Tooltip "Available in Step 6". Will route to `/train-yolo?config=<run_group>` once Step 6 lands.

New API endpoints in `scripts/training_pipeline_ui.py`:

- `GET /api/exported-datasets` — wraps `list_exported_datasets()`.
- `GET /api/starting-weights` — wraps `list_starting_weights()`.
- `GET /api/training-configs` — wraps `list_training_configs()`.
- `GET /api/training-configs/<run_group>` — load a saved config for editing.
- `POST /api/training-configs` — save a config (single canonical write endpoint; PATCH not added until the user needs partial edits).
- `POST /api/training-config/check-ready` — wraps `check_training_readiness` against a pending (not-yet-saved) config payload plus a `selection_path` (repo-root-relative).
- `GET /api/system/device-info` — wraps `probe_training_devices()`.

All Step 5 GUI actions log through the existing `/api/training-ui-interaction` endpoint. Events to log at minimum: page open, dataset selected, weight checked/unchecked, augmentation preset selected, save attempted, save succeeded, save failed (with reason), config loaded, reset clicked, continue clicked (records that the stub was hit).

Design system: match the Import / Dataset Selection page CSS (Fraunces + IBM Plex, paper-grain background, stamp-colored italic accents). No new design language.

Stage rail: add `Training Setup` between `Dataset Selection` and `Train YOLO`. Wire the Step 4 Continue button to navigate here with `?selection_path=<selection_path>` (URL-encoded, repo-root-relative).

## Acceptance Checks

Backend / CLI (automated, no real GPU required):

- `list-starting-weights` lists all repo-root `.pt` files and any `workspace/model_weights/*.pt` files with correct provenance badges for the 11 currently documented root weights. Test uses `--weights-roots` / `--docs-roots` overrides to point at a fixture directory; does not depend on the live repo state.
- `device-info` is exercised under two mocked scenarios: (a) `torch.cuda.is_available()` true with one fake device → `recommended_device == "0"`; (b) `torch.cuda.is_available()` false → `recommended_device == "cpu"`.
- `save-training-config` round-trips a **canonical** (`models[]`) config — write, read, write again — with byte-identical output (ignoring `created_at`).
- `save-training-config` migrates a legacy `model` + `run_name` config from `workspace/training_configs/` into canonical shape on Save. The post-migration `build-train-command` output differs from the pre-migration output **only** in the `name=` argument; all other arguments are identical.
- `save-training-config` rejects unknown augmentation keys with a clear error message.
- `check-training-ready` against a suite config returns per-run `runs[]` and an aggregate `ready`. When `ready` is false the CLI exits with code `3` while keeping envelope `ok: true`.
- `build-train-command` against a single-model config returns both `result.command_line` (historical) and `result.commands` (new). Against a suite config returns `result.commands` with one entry per model in declared order; `result.command_line` is absent or null.
- `expand_training_config_runs` raises with a clear collision message when two selected weights share a basename stem.
- `list-exported-datasets` returns every selection in `workspace/dataset_selections/`, with `available: false` and a populated `unavailable_reason` for any selection missing `dataset_manifest.json`, `labels/train`, `labels/val`, or the required class labels.
- Provenance warnings appear in the saved config when an undocumented `workspace/model_weights/<x>.pt` is selected, but save still succeeds.

Backend / CLI (manual smoke on the user's laptop, hardware-specific):

- `device-info` on the RTX 4070 reports `cuda_available: true`, GPU index 0 with the right model name, and ~8GB memory.

GUI:

- `/training-setup` renders, loads existing configs, lists exported datasets, lists starting weights with provenance badges, and applies the recommended device.
- Selecting one weight produces a single auto-derived run name; selecting three produces three auto-derived run names that update live as the run group is edited.
- Selecting an undocumented weight surfaces a yellow warning in the readiness panel and a yellow badge on the weight row, but Save remains enabled.
- Saving an empty or invalid config disables Save and shows the responsible field's validation hint.
- The Step 4 "Continue to Training Setup" button navigates to `/training-setup?selection_path=<path>` and pre-selects that dataset.
- The footer Continue button is visually disabled and shows the "Available in Step 6" tooltip on hover.
- All listed interaction events appear in `workspace/reports/training_pipeline_ui_interactions.jsonl`.

Final integration check (cross-step):

- Saving a config in the GUI then running `scripts/training_pipeline_cli.py check-training-ready --selection ... --config workspace/training_configs/<group>.json` returns `result.ready: true` with envelope `ok: true` for a dataset that is fully exported.
- For a config loaded from the legacy `manual_reviewed_gavin_balanced_1to1_v1_yolov8n_e150_img960.json` file and not yet migrated, `build-train-command` produces command lines byte-identical to today's output (no regression on legacy configs).
- For the same selection migrated to canonical shape via Save, `build-train-command` produces a command line identical to the legacy command **except** for the `name=` argument (which now reflects the derived `<weight_stem>_<run_group>`).

## Implementation Order

1. Backend helpers: `list_starting_weights`, `weight_provenance`, `probe_training_devices`, `list_exported_datasets`, `list_training_configs`, `expand_training_config_runs`.
2. Backend helpers: augmentation whitelist enforcement in `save_training_config` and `build_yolo_train_command`.
3. Suite-aware updates to `check_training_readiness` and `build_yolo_train_command`.
4. CLI subcommands and tests.
5. API endpoints and tests in `scripts/training_pipeline_ui.py`.
6. GUI page, route, stage-rail link, interaction logging.
7. Wire Step 4's Continue stub to `/training-setup?selection_path=<path>` (URL-encoded, repo-root-relative).
8. Acceptance pass against the checks above, including a round-trip of one existing single-model config and one existing suite config.

## Risks and Open Questions

- **Provenance lookup precision.** A literal-substring scan of `docs/yolo_weight_recreation.md` for `yolov8n.pt` will also match strings containing that basename in other contexts. This is acceptable for the first slice because false-positives bias the badge toward "documented", which is the lenient direction. If false-positives become a real problem we can switch to a structured registry file.
- **CLI envelope shape change for `build-train-command`.** Single-model configs keep the historical `result.command_line` (`list[str]`) and gain a new `result.commands` (`list[list[str]]`, length 1) alongside it. Suite configs return only `result.commands`. Existing tests on the historical field stay green; new tests cover the suite shape and the dual presence on single configs.
- **One-way legacy migration on Save.** Once a user presses Save in the GUI on a legacy config, the file becomes canonical shape and the literal `run_name` string is replaced by the derived name. The migrated YOLO command is equivalent (same hyperparameters, same data, same model) but the `name=` argument differs. Users who want to keep the original `run_name` literal should avoid pressing Save on legacy files, or restore from git. This is acceptable because run-directory contents are reproducible from the config and dataset; the literal name string is not load-bearing.
- **Beforeunload warning.** Will be added in Step 6 when training is actually launching. Modern Chrome restricts the prompt text, so the user gets a generic browser warning, not a custom message. Set expectations in the Step 6 plan.
- **Device probing import cost.** Importing `torch` just to probe devices adds a few hundred ms to the API response. Cache the probe result for the lifetime of the server process; invalidate only if the user explicitly requests a refresh.
