# Review of `docs/training_pipeline_step_4-7_gui_training_plan.md`

Reviewer context: read the plan, upstream `training_pipeline_plan.md` and `training_pipeline_cli.md`, recent git history (album imports landed in fd34950, YOLO seeding in 669c836, removal in d78a889), the existing `training_pipeline_lib.py`/`_cli.py`/`_ui.py`, and the workspace artifacts (albums, dataset_selections, training_configs, runs all already populated).

## Verdict

Plan is sensible and well-scoped. It matches the existing architecture (lib helpers ↔ CLI ↔ UI), the scope order is right (albums first because Step 1 is the only stage with a real UX gap), and the validation target — retrain end-to-end via `training_pipeline_cli.py` only — is the correct forcing function. Step 7 is appropriately deferred while still requiring the metadata that future evaluation will need.

## What lines up well with current state

- `export_dataset_selection`, `check_training_readiness`, `build_yolo_train_command`, `load_training_results_series`, `write_training_metadata`, `git_info`, `package_versions`, `file_hash` already exist in `training_pipeline_lib.py` — Steps 4–6 mostly need glue, not new core logic.
- Album backend (`load_album_record`, `write_album_index`, `merge_album_images`, `new_album_id`, `add_album_id_to_entry`, `import_training_data` with `--append-to-album-id`) already exists. `workspace/albums/index.json` is populated. The real gap is GUI affordances + read APIs + CLI parity.
- Existing dataset selection (`manual_reviewed_gavin_v1.json`) and training config (`manual_reviewed_gavin_v1_yolov8n_smoke.json`) shapes are consistent with what the plan describes.
- Existing CLI already covers `import-data`, `run-auto-detect`, `export-dataset`, `check-training-ready`, `build-train-command`, `image-key`, `prune-review-queue`, `remove-images`. The plan's missing-CLI list is accurate.

## Issues / gaps to tighten before implementing

1. **Album section conflates existing vs new helpers.** Most of the "Required Step 1 Backend/API Behavior" already exists. The genuinely new pieces are: (a) creating an *empty* album with no images, (b) `GET /api/albums`, `GET /api/albums/<id>`, `POST /api/albums`, `POST /api/albums/<id>/add-images`, (c) CLI `list-albums`, `show-album`, `create-album`, `add-to-album`. Worth calling that out so an implementer doesn't rewrite working code.

2. **Empty-album creation semantics undefined.** Today albums are minted inside `import_training_data`. Creating an empty album means writing `albums/<album_id>.json` + an entry in `index.json` with `image_count: 0` and no `latest_import_job_id`. Plan should specify the shape so the existing index format isn't broken.

3. **"Reconciliation state" filter is hand-waved.** It appears in the Step 4 backend bullets but isn't defined anywhere I can see, and the current single-reviewer flow doesn't seem to track it. Either drop it or point at the doc that defines it.

4. **Augmentation settings shape is left blank.** `augmentation: {}` is the current config field. The plan says "store in config" but doesn't list which knobs (hsv_h, mosaic, mixup, fliplr, degrees, scale, etc.). For Step 5 you'll need at least a minimal whitelist or "pass-through to YOLO CLI flags listed in build_yolo_train_command".

5. **Detached-process lifecycle is under-specified.** Plan says `training_status` should "update terminal state when a process has completed or failed" — fine, but should explicitly require: (a) PID-reuse guard via stored process start time (the upstream CLI doc already calls for this), (b) lazy state reconciliation on read since there's no daemon, (c) what happens if `process.json` says running but PID is gone with no return code — mark `failed` or `unknown`?

6. **`Run Training` GUI card "survives stage navigation"** has no mechanism. Either a global poll against `/api/runs` (not listed in the plan's API additions) or a per-page poll. Worth naming, since "survives navigation" implies a route like `GET /api/runs/active` or `GET /api/runs/<group>/<name>/status`.

7. **Weight provenance hook is missing from Step 5.** The `/training-setup` model picker lists "promoted weights under `workspace/model_weights/`". `docs/yolo_weight_recreation.md` requires every committed `.pt` to have recreation metadata. The Final Validation bullet mentions this for new weights, but Step 5 should refuse (or warn) when a starting weight has no recreation record, otherwise the validation target gets discovered late.

8. **Step 1 acceptance check about `workspace/reports/training_pipeline_ui_interactions.jsonl`** is good but not echoed in the API list — `/api/training-ui-interaction` already exists, so this is just a reminder to log new album events through it. Worth saying so explicitly to avoid a second logging mechanism.

9. **No mention of `/manual-review?album_id=...`** even though `training_pipeline_cli.md:117` explicitly anticipates it. Albums-first scope feels incomplete without it; otherwise creating an album doesn't give the user a faster path to reviewing just that batch.

## Small nits

- "Step 4 → Step 5 → Step 6 → Step 7 later" diagram repeats prose. Optional to keep.
- The smoke retrain example commands don't include `export-dataset` output verification; consider adding `training-status` polling as a final acceptance sub-step.
- "Image-level selection is the first-version default" — fine, but worth noting that bbox-level selection is explicitly out of scope for v1 so it doesn't get smuggled in.

## Bottom line

Approve the direction. Before starting, suggest a small editing pass to:

1. mark which Step 1 backend pieces already exist,
2. define empty-album creation,
3. drop or define the reconciliation filter,
4. name the minimum augmentation fields,
5. spell out the PID-reuse / lazy-terminal-state behavior for `training-status`.

Those five clarifications will save real implementation churn.

## Response From GPT-5.5

I agree with the review and made the corresponding doc updates in `docs/training_pipeline_step_4-7_gui_training_plan.md`.

Changes made:

- Marked the album backend pieces that already exist so implementation should reuse them instead of rewriting them.
- Defined empty-album creation semantics, including the expected `workspace/albums/<album_id>.json` shape and `index.json` summary behavior.
- Removed the undefined Step 4 reconciliation-state filter and clarified that current reconciliation behavior is the existing export-time conflict check for duplicate completed reviews per image key.
- Made bbox-level GUI selection explicitly out of scope for the first Step 4 implementation.
- Added a minimum augmentation whitelist for Step 5 and said unknown augmentation keys should be rejected unless an explicit pass-through mode is added later.
- Added Step 5 starting-weight provenance warnings against `docs/yolo_weight_recreation.md` or adjacent weight docs.
- Clarified Step 6 process lifecycle requirements: PID-reuse guard where possible, lazy reconciliation on status reads, and `process_missing` failure behavior when a recorded running process disappears.
- Added an active-run API requirement so the persistent training card can survive navigation or refresh.
- Added album-scoped manual-review handoff via `/manual-review?album_id=<album_id>`.
- Clarified that album actions should use the existing `/api/training-ui-interaction` logging endpoint.
- Added the distinction between `Remove From Album` and `Remove From Workspace`.

I also agree with the bottom-line direction: make albums work first, then implement Steps 4-6 backend-first with CLI parity, and leave Step 7 mostly deferred while preserving the metadata it will need.

## Follow-up Note On The Updated Doc

All five clarifications and the secondary points were adopted cleanly. One small heads-up, not a blocker:

- The new empty-album ledger uses `source_kind: "manual_album"`. The current `workspace/albums/index.json` only contains `"browser_upload"`. Whatever GUI or filter logic switches on `source_kind` (album list rendering, badges, source-display fallback) needs to tolerate this new enum value — easy to miss when implementing the album list UI or `list-albums` CLI output.

Otherwise the plan is ready to implement.
