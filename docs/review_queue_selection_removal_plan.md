# Review Queue Selection And Removal Plan

## Purpose

Add safe, general-purpose image selection and review-queue removal tools to the training pipeline GUI. The immediate use case is clearing a long queue of unreviewed review stubs before re-importing already-known imagery to test album behavior, without risking reviewed work or source data.

This plan covers Step 1 import/intake and Step 3 reviewed bbox overview. It intentionally treats selection as a reusable UI primitive, not a delete-only feature.

## Non-Goals

- Do not delete source imagery.
- Do not delete cache PNGs.
- Do not remove import manifest entries.
- Do not remove album records or album membership.
- Do not overwrite reviewed or partially reviewed review JSON files.
- Do not implement import multithreading until duplicate/re-import correctness and queue removal safety are tested.

## Selection Model

Selection identity:

- Step 1 selection is by `image_key`.
- Step 3 tile selection is by crop/detection id plus source `image_key`.
- Any review-queue removal action in Step 3 deduplicates selected tiles by source `image_key` before acting.

Basic selection behavior:

- each Step 1 row/card has a checkbox
- each Step 3 crop/tile has a checkbox
- clicking a checkbox toggles that item
- `Shift+click` selects the full range between the last selected item and the clicked item, following Windows Explorer-style range selection
- `Ctrl+click`/`Cmd+click` toggles individual items without clearing the rest of the selection when selection is also supported on the row/tile body
- toolbar shows `N selected`
- `Clear selection` clears all selected items in the current stage

No visible-only bulk selection:

- do not add `Select visible`
- do not add a visible-only inverse or visible-only removal action
- selecting many images should be handled by range selection and `Select all matching filter`

Selection persistence:

- selection should persist across sort changes when selected items still exist in the active result set
- when filters change, selected items that no longer match may remain selected internally, but the UI must make the count clear, for example `142 selected, 37 shown`
- removal confirmation must act on all selected items, not only currently shown items
- selection may be persisted in browser local storage, but only up to a soft cap such as 50,000 selected ids; above the cap, drop persistence for that selection and warn the user rather than writing a very large local storage value

## Filtering

Step 1 filter dropdown should include:

- `All images`
- `Unreviewed images`
- `Reviewed images`
- `Partially reviewed images`
- `Selected album`
- `No album`
- later: import status, source label, date range, cache/render status

Step 3 filter dropdown should include existing bbox filters and later album/source filters:

- class label filters
- warning/conflict filters
- reviewed/incomplete filters where useful
- album/source filters when album-aware Step 3 work is implemented

`Unreviewed images` in Step 1 means images whose review JSON is safe to archive under the protection rules below. It is not just a manifest import status.

Album filters such as `Selected album` and `No album` should be hidden or disabled until first-version album metadata is available in the Step 1 listing payload. The UI must not ship inert filter options.

For performance, Step 1's listing API should expose a server-computed `unreviewed_archivable` boolean per image. Computing this requires reading review JSONs and applying the protection rules; it should happen server-side so client-side filtering remains fast for queues with thousands of images. The preview endpoint remains the confirmation source, not the primary way to populate the filter.

## Select Dropdown

The select dropdown should include:

- `Select all matching filter`
- `Clear selection`
- `Invert current filtered selection`

`Select all matching filter` selects every item matching the active filter/search/sort scope, not only rows currently rendered by pagination, virtualization, or lazy loading.

## Removal Semantics

Removal from the queue means archiving review JSON files, not deleting training data.

When the user presses `Remove`:

- show a confirmation dialog with the selected count, for example `Remove 143 selected images from the review queue?`
- explain that source files, cache files, import manifest entries, and album records are untouched
- explain that review files are archived, not permanently deleted
- always show how many selected items are expected to be removable and how many are protected
- if preview takes more than about one second, show a spinner and wait for the preview before opening the confirmation dialog; never show a removal dialog without protected counts
- after success, show a toast such as `Archived 140 reviews` with the report path
- in Step 3, make source-image deduplication explicit, for example `Remove 12 images from the review queue (deduplicated from 47 selected bbox tiles)?`; show an affected-image list when the count is small enough

Archive destination:

```text
workspace/reviews/_archived_unreviewed/<timestamp_ms>/<reviewer>/<image_key>.json
```

Use a millisecond-precision UTC timestamp such as `YYYY-MM-DDTHH-MM-SS-fffZ` for archive directories. First version assumes a single writer per reviewer, matching the training import album plan's single-writer assumption.

Report destination:

```text
workspace/reports/review_queue_prune/<timestamp>.json
```

The report should include:

- timestamp
- reviewer
- requested image keys
- archived image keys and paths
- protected/skipped image keys and reasons
- active filters, if removal came from a filtered selection workflow

After a successful archive operation:

- Step 2 queue caches must observe review-directory filesystem mtimes and invalidate automatically when review JSONs are moved; the training UI should not rely on reaching into another server process's memory
- refresh Step 1 counts and filters
- keep the import manifest and album records unchanged

Step 2 reviewer enumeration must skip review-root directories whose names begin with `_`, including `_archived_unreviewed`, so archived review files do not appear as a pseudo-reviewer.

Archiving a review JSON removes it from the active queue, but the import manifest still exists. Re-importing the same image can recreate a fresh unreviewed stub if no active review JSON exists. Permanent removal from queue regeneration requires manifest cleanup and is a separate future feature.

## Protection Rules

Only archive review JSONs that are clearly unreviewed initial stubs.

Safe to archive:

- `review_status` is an initial/uncompleted status such as `overview_1`
- no detection has `reviewed: true`
- no detection has a non-null `review_label`
- no detection has `source_type: manual`
- no image-level completed/reviewed status is present

Never archive:

- completed images
- partially reviewed images
- review files with any manually reviewed detection
- review files with any manual label
- review files with manually drawn detections
- corrupt or ambiguous review JSON files
- missing review files, except they may be reported as skipped/missing

If in doubt, protect the file and report the reason.

## API Plan

Preview endpoint:

```text
POST /api/review-queue/prune-unreviewed/preview
```

Input:

```json
{
  "reviewer": "gavin",
  "image_keys": ["img_..."],
  "filters": {}
}
```

Output:

```json
{
  "requested_count": 143,
  "archivable_count": 140,
  "protected_count": 3,
  "archivable": [{"image_key": "img_...", "review_path": "..."}],
  "protected": [{"image_key": "img_...", "reason": "has reviewed detection"}]
}
```

Confirm endpoint:

```text
POST /api/review-queue/prune-unreviewed
```

Input:

```json
{
  "reviewer": "gavin",
  "image_keys": ["img_..."],
  "confirmed_archivable_count": 140,
  "filters": {}
}
```

The confirm endpoint should recompute protection rules server-side instead of trusting the preview payload. `confirmed_archivable_count` is only a guard against accidental stale confirmations.

Reviewer scope:

- preview and confirm consider only review JSONs under `<reviews-dir>/<reviewer>/`
- other reviewers' review files for the same `image_key` are never archived by this operation
- cross-reviewer cleanup must be an explicit future operation

## CLI Plan

Future CLI helper:

```bash
.venv/bin/python scripts/training_pipeline_cli.py prune-review-queue \
  --reviewer gavin \
  --image-key img_abc \
  --image-key img_def
```

Useful options:

- `--reviewer <name>`
- `--reviews-dir <path>`
- `--image-key <key>` repeatable
- `--filter unreviewed`
- `--album-id <id>` later, once album-aware queue filtering is implemented
- `--dry-run`

The CLI must use the same backend helper as the GUI endpoints.

CLI semantics:

- if both `--image-key` and `--filter unreviewed` are provided, the filter narrows the explicit image-key list
- if only `--filter unreviewed` is provided, expand to all matching keys for that reviewer
- if only `--image-key` values are provided, evaluate those keys without extra filter expansion
- if neither `--image-key` nor `--filter` is provided, return a usage error
- `--dry-run` should return the same JSON envelope shape as the preview endpoint, not a separate free-form summary
- successful non-dry-run output should include the archive report path and archived/protected counts

## Re-Import Safety

Duplicate detection must remain content-based:

- duplicate identity is `source_content_hash + render_profile`
- same filename is not enough to be a duplicate
- same content under a different filename should map to the same `image_key`
- same filename with different content should map to a different `image_key`

Review overwrite rules:

- existing review JSON files are never overwritten during import
- reviewed or partially reviewed JSON files remain untouched
- if an unreviewed stub was archived, re-import may create a new fresh stub for that image
- import may add album membership and manifest `album_ids` for already-known images without changing review labels, detections, or review status

Tests before queue removal ships:

- same content under two filenames produces the same `image_key`
- same filename with different content produces a different `image_key`
- re-import of a reviewed image does not overwrite the review JSON
- pruning archives only safe unreviewed stubs
- pruning protects reviewed, partially reviewed, manually labeled, manually drawn, corrupt, and ambiguous review files
- Step 3 removal deduplicates selected bbox tiles by source `image_key`
- Step 2 reviewer enumeration skips `_archived_unreviewed` and other underscore-prefixed directories
- Step 2 queue cache invalidates when review-directory mtimes change
- CLI `--filter unreviewed` intersects correctly with explicit `--image-key` values

## Import Multithreading Later

Parallel import should happen only after selection/removal and duplicate safety are tested.

Parallelizable work:

- source hashing
- image loading
- render/cache generation

Serialized work:

- manifest updates and save
- album record updates and save
- album index updates and save
- review stub creation decisions and writes
- import job JSONL finalization and summary writes

Requirements:

- use a bounded worker pool
- keep discovered file order deterministic
- merge worker results in discovery order
- keep duplicate detection content-based, not filename-based
- preserve `overwrite=False` review-stub behavior
