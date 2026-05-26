# Backend Bug-Hunt Test Plan

Status: merged with Opus' independent bug-class strategy in `council/notes/backend_test_strategy_opus.md`.

## Goal

Find and remove backend bugs across completed training-pipeline Steps 1-6, excluding full Step 7 evaluation. Treat the CLI and backend helpers as the primary contract, with API route tests only where the GUI depends on backend behavior.

The test effort should prove this path works without browser interaction:

```text
import-data / album commands
  -> review artifacts already completed or fixture completed reviews
  -> list/save dataset selection
  -> export YOLO dataset
  -> save/check training config
  -> build train command
  -> start/status/stop training through managed run artifacts
```

## Core Principles

- Prefer black-box CLI tests for pipeline behavior because agents and automation will use `scripts/training_pipeline_cli.py`.
- Keep helper-level unit tests for edge cases that are hard, slow, or unsafe to exercise through subprocess CLI.
- Use isolated `tmp_path` fixtures for all generated manifests, albums, reviews, datasets, configs, reports, and runs.
- Never read from or write to `/home/gavin/Downloads/AIS data`.
- Avoid real YOLO training in normal tests; simulate training with monkeypatched `yolo_executable()` or `subprocess` behavior, then keep one manually-invoked smoke test for actual CPU/GPU training.
- Assert artifacts, not just `ok: true`: inspect JSON files, counts, hashes, labels, log tails, process metadata, and terminal states.

## Real-Data Strategy

The project has a lot of real workspace data. We should use it aggressively to find bugs, while preserving a small fixture suite for fast regression checks.

Use two complementary tracks:

- Fixture regression tests: tiny generated `tmp_path` cases that run quickly and pin exact bug fixes. These make failures easy to reproduce.
- Real-data diagnostics: read the existing workspace manifests, albums, reviews, selections, exported datasets, training configs, run ledgers, and model-weight docs to find integration bugs that fixtures will miss.

Real-data diagnostics should be thorough by default for workspace artifacts:

- scan all workspace import manifests, album ledgers, album indexes, review JSONs, archived review JSONs, dataset selections, exported dataset manifests, training configs, run `process.json` files, and `training_metadata.json` files
- validate every cross-reference between manifests, albums, reviews, datasets, configs, and runs
- verify every referenced file path exists or is intentionally missing with a documented reason
- check every JSON artifact is parseable, has expected schema fields, and has internally consistent counts
- inspect every exported YOLO label file for class ids, normalized bbox ranges, malformed rows, empty-label behavior, and image/label pairing
- compute hashes only when needed; do not hash every large image unless a diagnostic specifically needs it
- read image dimensions for real cache/export images where practical, but do not copy or rewrite image bytes
- write machine-readable and human-readable reports under `workspace/reports/`

Rules for real data use:

- Real workspace artifacts may be read exhaustively.
- Existing workspace artifacts should not be mutated by diagnostics unless the command is explicitly a repair command and the user approves it.
- External source imagery and `/home/gavin/Downloads/AIS data` must never be modified.
- If testing import/export/training behavior against real data requires writes, copy only the needed subset into an isolated `workspace/test_runs/<timestamp>/` or `tmp_path` area and run there.
- Full real-data diagnostics should be runnable separately from `pytest tests -q`, for example through a future `scripts/training_pipeline_cli.py validate-workspace` or `scripts/diagnose_training_workspace.py` command.
- Reports should include counts, failures, warnings, sampled examples, and recommended next commands; they should not stop at the first error.

Recommended real-data diagnostics:

- `validate-workspace-artifacts`: parse every workspace JSON artifact and report schema/count/path issues.
- `validate-review-corpus`: scan every review file for duplicate image keys, conflicting completed reviews, invalid bboxes, invalid labels, missing cache images, stale archive records, and reviewer/status distributions.
- `validate-albums`: verify album indexes, album ledgers, image membership, manifest `album_ids`, counts, latest import job ids, and orphaned records.
- `validate-dataset-selections`: verify every selection references existing reviews/cache files, has valid splits, and can be dry-run exported.
- `validate-exported-datasets`: inspect every exported dataset folder, `data.yaml`, image/label pairing, class ids, bbox ranges, and dataset manifest hashes/paths.
- `validate-training-configs`: verify every config resolves a selection/export, expands runs, passes/fails readiness for the expected reasons, and builds commands without launching training.
- `validate-runs`: scan all run ledgers/metadata/logs/results files for terminal-state consistency, corrupt JSON, missing logs, missing checkpoints, stale running processes, and unsafe paths.
- `validate-weight-provenance`: scan all repo/workspace `*.pt` files and verify recreation documentation exists.

## Phase 0: Baseline And Test Harness

1. Run the current backend test suite once and capture a failure inventory under `workspace/reports/`, for example `workspace/reports/backend_test_baseline_<date>.md`.
2. Classify every failure as command-contract, artifact-schema, state-transition, source-safety, filesystem cleanup, API parity, or test-fixture issue.
3. Add or standardize test helpers for tiny RGB PNGs, completed review JSONs, import manifests, album ledgers, exported datasets, training configs, fake weights, fake docs, fake YOLO executable behavior, and CLI envelope parsing.
4. Make sure every CLI subprocess test asserts stdout is parseable JSON unless `--human` or `--help` is intentionally used, and stderr is either empty or explicitly expected.
5. Add a checked-in `tests/fixtures/process_v1.json` and a schema/compatibility test so future process-ledger changes must intentionally migrate or update the fixture.

## Phase 1: CLI Contract Sweep

Every backend command should have a minimal success test, a representative failure test, and an output-contract test.

Commands to cover:

- `import-data`
- `run-auto-detect`
- `image-key`
- `prune-review-queue`
- `remove-images`
- `list-albums`
- `show-album`
- `create-album`
- `add-to-album`
- `remove-from-album`
- `rename-album`
- `list-dataset-candidates`
- `save-dataset-selection`
- `list-dataset-selections`
- `export-dataset`
- `list-exported-datasets`
- `list-starting-weights`
- `save-training-config`
- `list-training-configs`
- `device-info`
- `check-training-ready`
- `build-train-command`
- `start-training`
- `training-status`
- `stop-training`

Contract assertions:

- default stdout is a single JSON envelope with `ok`, `command`, `schema_version`, `result`, `warnings`, and `errors`
- usage errors exit `2`
- hard failures exit `1`
- readiness blockers exit `3` while `ok: true`
- process state mismatches exit `4`
- no progress logs or traceback text leak to stdout in JSON mode
- `--pretty`, `--human`, and `--verbose` do not change domain behavior

## Phase 2: Step 1 Import And Album Invariants

High-value tests:

- Import one PNG into a new album and assert manifest entry, cache PNG, review stub, album ledger, album index, import-job JSONL, and source file immutability.
- Import a folder non-recursively and recursively; assert hidden files/dirs and unsupported suffixes are skipped correctly.
- Re-import identical image into the same album and a different album; assert `skipped_count`, `members_added_existing`, and deduplicated `album_ids`.
- Append a new image to an existing album with `--append-to-album-id`; assert immutable album id and preserved existing images.
- Create empty album, rename it, edit description, add existing image keys, remove image keys; assert album index stays in sync.
- Attempt invalid album ids, duplicate image keys, missing image keys, and all-failed add-to-album; assert partial-failure semantics.
- Force import when cache is missing or stale; assert review labels are not overwritten.
- Dry-run every write-capable import/album/removal path; assert no files are created or mutated.

Bug classes to target:

- album counts drifting from ledger images
- manifest `album_ids` mismatching album ledgers
- review stubs overwritten after manual review
- source path mutation or accidental cleanup outside workspace
- partial failures returning the wrong exit code

## Phase 3: Review Queue And Removal Safety

High-value tests:

- `prune-review-queue --dry-run` classifies unreviewed, completed, archived, missing, and ambiguous reviews without writes.
- Actual prune archives only safe unreviewed stubs and writes a report.
- `remove-images --dry-run` reports manifest/cache/review/album impact without writes.
- Actual workspace removal deletes only manifest records, cache/staging copies, active review files, archived review files, and album membership.
- Protected cases: completed reviews, ambiguous duplicate reviews, missing cache files, and image keys present in multiple albums.

Bug classes to target:

- deleting reviewed labels accidentally
- leaving album ledgers stale after workspace removal
- removing archived files but not active files, or vice versa
- report contents not matching actual changes

## Phase 4: Step 4 Dataset Selection And Export

High-value tests:

- `list-dataset-candidates` filters by reviewer, album, completed-only, label class, and `--include-incomplete`.
- `save-dataset-selection` writes deterministic sorted image keys and split assignments for fixed seeds.
- Split ratio edge cases: tiny datasets, zero test split, invalid ratios, repeated image keys, and nonstandard dataset names that need slugging.
- `export-dataset` writes images, labels, `data.yaml`, and `dataset_manifest.json` with correct classes and hashes.
- Export selected-only labels, all-exportable labels, empty-label background negatives, and max-fraction truncation.
- Export fails clearly on missing completed review, duplicate completed review conflict, invalid split, missing cache image, and bad selection schema.
- `--output-dir` and `--dry-run` do not mutate the selection manifest.

Bug classes to target:

- incorrect YOLO normalized bbox output
- empty labels omitted when they should be written
- split folders missing but readiness passing
- stale files left in dataset directories causing false readiness
- nondeterministic selection/export output

## Phase 5: Step 5 Training Setup And Readiness

High-value tests:

- `list-exported-datasets` distinguishes available, missing, and malformed exports.
- `list-starting-weights` reports provenance for documented and undocumented weights using fixture roots.
- `save-training-config` accepts valid single-model and multi-model configs and rejects unknown augmentation keys.
- Derived run names are stable, unique, and friendly for path-like model names.
- `check-training-ready` catches missing `dataset_manifest.json`, bad `data.yaml` classes, empty train/val images, missing required fields, existing run dirs without overwrite, and multi-run per-run blockers.
- `build-train-command` emits exact command arrays for single-run and multi-run configs with augmentation, workers, patience, output dir, and overwrite handling.

Bug classes to target:

- config accepted by save but rejected later by readiness or command build
- undocumented weights only warned, not silently ignored
- readiness false positives from incomplete exported dataset folders
- command shape drifting from what `start-training` actually launches

## Phase 6: Step 6 Managed Training Process Tests

Normal automated tests should use fake processes rather than real YOLO training.

High-value tests:

- `start-training --dry-run` writes nothing and returns the command, run dir, log path, process path, and metadata path.
- Helper-level start in foreground with a fake executable that exits `0`; assert `process.json`, `training_metadata.json`, `train.log`, and terminal `completed` state.
- Helper-level start with fake executable that exits nonzero; assert terminal `failed` state and useful log tail.
- Detached fake long-running process; assert initial `running`, `training-status` reads log tail/results series, `stop-training` moves to `cancelled`, and active-run listing drops terminal runs.
- Running process disappears without return code; assert lazy reconciliation marks `failed` with `process_missing` unless artifacts prove completion.
- Detached training should eventually use a wrapper/reaper that records the exact YOLO process return code after the launching CLI exits. Until that exists, lazy reconciliation must mark artifact-inferred completed runs with `return_code_unavailable` so the missing return code is explicit rather than ambiguous.
- Add a workspace-root override such as `OPENDARKSTS_WORKSPACE_DIR` for CLI/backend tests so subprocess-based tests can point at `tmp_path` instead of writing temporary run artifacts under the real `workspace/runs/`.
- Existing historical run with `weights/best.pt`, `weights/last.pt`, or `results.csv` but no `process.json`; assert status inference and warning.
- PID reuse protection: mismatched `process_start_jiffies` means not live.
- Multi-model configs require `--run-name`; valid chosen run starts only that run.

Bug classes to target:

- stale `running` state that never becomes terminal
- metadata status disagreeing with process status
- `start-training` creating artifacts before readiness succeeds
- log handles left open or missing logs reported badly
- stop killing the wrong process group

### Opus Bug-Class Checklist For Step 6

Opus' independent review is deeper on failure modes than this phase list. Add these as targeted tests while building Step 6 coverage:

- real Python subprocess lifecycle tests, including dead process reconciliation and daemon watcher recovery
- `process_is_live` rejects `pid=0`, negative pids, and PID-reuse cases with mismatched start jiffies
- `stop_training_run` handles already-dead processes without leaving permanently stale `running` ledgers, or the required `training_status` reconciliation retry is documented and tested
- watcher-thread and lazy-reconcile double writes leave parseable, consistent `process.json` and `training_metadata.json`
- corrupt or half-written `process.json` does not crash `training_status`, `list_active_training_runs`, or `stop_training_run`
- `write_json` should become atomic or have tests proving partial-write failures preserve the previous valid file
- concurrent `training_status` calls and concurrent same-run `start_training_run` calls cannot corrupt ledgers or orphan duplicate jobs
- untrusted ledger fields such as `log_path`, `run_dir`, `run_group`, and `run_name` cannot read outside the intended workspace/run root or poison API responses
- malformed selection/config paths return structured errors rather than raw `TypeError`, `FileNotFoundError`, or tracebacks
- stale or future `schema_version` values in process ledgers are warned about or rejected intentionally
- huge logs are tailed without loading the entire file into memory
- symlinked run artifacts or run dirs are skipped or resolved safely
- UI/API surfaces reject unsafe run paths and do not expose `--foreground` style behavior unless explicitly intended

## Phase 7: API Parity Tests For GUI-Backed Backend Behavior

Keep these lighter than CLI tests but ensure every GUI action has an API/backend path.

Routes to cover:

- album list/detail/create/rename/add/remove APIs
- dataset candidate/list/save/export APIs
- training dataset/weight/config/readiness/device APIs
- run active/status/start/stop APIs
- `/api/training-ui-interaction` logging path

Assertions:

- route responses use stable `ok`/`error` style expected by frontend code
- route calls use the same backend helper behavior as CLI tests
- route errors are JSON and do not crash the server thread
- UI interaction logs include timestamp, page/action, selected ids/counts, and error messages where applicable

## Phase 8: End-To-End Backend Smoke Scenarios

Add a small number of slower integration tests or manual smoke scripts. These should run against generated fixture images, not external datasets.

Scenario A: no real YOLO

```text
create-album
import-data two generated PNGs into album
write completed fixture reviews for those image keys
list-dataset-candidates --album-id ...
save-dataset-selection
export-dataset
save-training-config using fake documented weight
check-training-ready
build-train-command
start-training --dry-run
```

Scenario B: fake managed training

```text
same through exported dataset/config
helper-level start_training_run with fake yolo executable
poll training_status until completed
assert metadata/process/log/results artifacts
```

Scenario C: optional real YOLO smoke, not default CI

```text
use an intentionally tiny fixture dataset
epochs=1, batch=1, imgsz small, device=cpu unless GPU explicitly chosen
start only through training_pipeline_cli.py start-training
poll training-status until terminal
do not promote or keep new .pt files unless provenance docs are updated
```

Scenario D: sampled existing-workspace smoke, explicit diagnostic

```text
choose one or a few existing workspace dataset selections
verify referenced reviews/cache images exist without exporting everything
run export-dataset --dry-run where possible
run check-training-ready/build-train-command against one known config
write a summary report under workspace/reports/
do not mutate existing selections, datasets, runs, weights, or source imagery
```

## Suggested Execution Order

1. Run existing `pytest tests -q` and fix red tests first.
2. Fix the highest-risk Step 6 bug classes first: atomic `write_json` plus corruption recovery, untrusted ledger/path reads, watcher/lazy-reconcile races, concurrent `training_status`, then process schema fixture coverage.
3. Add missing Step 6 managed-process lifecycle tests because that is the newest and riskiest area.
4. Add real-data workspace validators for manifests, albums, reviews, dataset selections, exported datasets, training configs, runs, and weight provenance.
5. Run the real-data validators and triage every report finding into bug, data cleanup, or acceptable legacy artifact.
6. Add full CLI workflow smoke using generated fixtures.
7. Expand import/album/removal safety tests because those can destroy user trust if wrong.
8. Expand dataset export/readiness edge cases.
9. Add API parity tests only after backend/CLI behavior is stable.
10. Run optional real YOLO smoke after fake lifecycle tests pass.

## Minimum Exit Criteria Before Calling Backend Stable

- Full `pytest tests -q` passes.
- Every CLI command listed above has success and failure coverage.
- At least one CLI-only end-to-end fixture workflow passes from album/import through `start-training --dry-run`.
- At least one fake managed-training lifecycle reaches `completed`, one reaches `failed`, and one reaches `cancelled`.
- Removal/prune dry-run and actual-run tests prove no external source file changes.
- API route tests cover the GUI-backed actions for Steps 1, 4, 5, and 6.
- Sampled existing-workspace diagnostics can inspect real manifests/selections without mutating them or scanning the full corpus by default.
- Any real training smoke writes only workspace run artifacts and does not add undocumented `*.pt` files to the repo.
