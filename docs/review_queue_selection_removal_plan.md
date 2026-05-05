# Review Queue Selection And Removal Plan

## Purpose

Add safe, general-purpose image selection and workspace removal tools to the training pipeline GUI. The immediate use case is clearing unwanted imported imagery and review queue entries from the local training workspace without modifying read-only/original source imagery.

This plan covers Step 1 import/intake and Step 3 reviewed bbox overview. It intentionally treats selection as a reusable UI primitive, not a delete-only feature.

## Non-Goals

- Do not delete source imagery outside the repository/workspace.
- Do not modify read-only source folders.
- Do not overwrite reviewed or partially reviewed review JSON files during import.
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

`Unreviewed images` in Step 1 means images with an active review JSON under `workspace/reviews/<reviewer>/` whose review file is still an initial unreviewed stub. It is not a manifest-only status and it does not include removed/archived review files.

Album filters such as `Selected album` and `No album` should be hidden or disabled until first-version album metadata is available in the Step 1 listing payload. The UI must not ship inert filter options.

For performance, Step 1's listing API should expose a server-computed `unreviewed_archivable` boolean per image for the active unreviewed filter. Computing this requires reading active review JSONs; it should happen server-side so client-side filtering remains fast for queues with thousands of images. The remove preview endpoint remains the confirmation source for destructive removal.

Implemented note:

- Step 1 now merges active review JSONs that are not present in the import manifest into the listing as `review_only` rows, so `Reviewed images` and `Partially reviewed images` filters reflect the actual manual review queue rather than only manifest-backed imports.
- The Step 1 filter dropdown labels include `Filter:` for discoverability.
- Step 1 no longer exposes an `archived_unreviewed` filter. Images intentionally removed from the workspace should be removed from the manifest/album listing instead of lingering as archived queue entries.

## Select Dropdown

The select dropdown should include:

- `Select all matching filter`
- `Clear selection`
- `Invert current filtered selection`

`Select all matching filter` selects every item matching the active filter/search/sort scope, not only rows currently rendered by pagination, virtualization, or lazy loading.

## Removal Semantics

Step 1 removal means removing selected images from the local training workspace. This removes import manifest entries, album membership, active review JSONs, archived review JSONs for the same reviewer/key, workspace cache PNGs, and browser-upload staging copies when they live under `workspace/cache/import_staging`.

Removal does not modify original source imagery outside the repository/workspace. If an imported row points at an external source path, that path is left untouched.

When the user presses `Remove`:

- show a confirmation dialog with the selected count, for example `Remove 143 selected images from this workspace?`
- explain that import records, album membership, active review files, archived review files, workspace cache files, and browser-upload staging files will be removed
- explain that original external source files are untouched
- always show how many selected items are expected to be removable and how many requested keys are missing
- if preview takes more than about one second, show a spinner and wait for the preview before opening the confirmation dialog; never show a removal dialog without removable/missing counts
- after success, show a toast such as `Removed 140 images` with the report path
- in Step 3, make source-image deduplication explicit, for example `Remove 12 images from the review queue (deduplicated from 47 selected bbox tiles)?`; show an affected-image list when the count is small enough

Report destination:

```text
workspace/reports/image_removals/<timestamp>.json
```

Use a millisecond-precision UTC timestamp such as `YYYY-MM-DDTHH-MM-SS-fffZ` for removal reports.

The report should include:

- timestamp
- reviewer
- requested image keys
- removed image keys and deleted workspace paths
- missing/skipped image keys and reasons
- active filters, if removal came from a filtered selection workflow

After a successful remove operation:

- Step 2 queue caches must observe review-directory filesystem mtimes or be explicitly invalidated by the integrated training UI
- refresh Step 1 counts and filters
- removed images must no longer appear in Step 1 or Step 2

## Step 1 Review Handoff

Every image in Step 1 should provide an `Edit in manual review` action from the detail sidebar. The action routes to Step 2 for that exact image with `/manual-review?image_id=<image_key>`, matching the Step 3 pattern of returning to manual review with source-image context.

Implemented note:

- Step 1 shows `Edit in manual review` for every selected row, including unreviewed, partially reviewed, reviewed, review-only, and manifest-backed images.

Step 2 reviewer enumeration must skip review-root directories whose names begin with `_`, including legacy `_archived_unreviewed`, so old archive folders do not appear as pseudo-reviewers.

Re-importing the same external source after removal is allowed and will create a new manifest entry, cache PNG, album membership, and review stub for the same stable `image_key`.

## Protection Rules

The legacy `prune-review-queue` helper only archives review JSONs that are clearly unreviewed initial stubs. Step 1's primary `Remove selected` action does not use these protection rules; it is an explicit workspace removal action and may remove reviewed, partially reviewed, or unreviewed selected images.

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
POST /api/images/remove/preview
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
  "removable_count": 140,
  "missing_count": 3,
  "removable": [{"image_key": "img_...", "cache_paths": ["..."]}],
  "missing": [{"image_key": "img_...", "reason": "not found"}]
}
```

Confirm endpoint:

```text
POST /api/images/remove
```

Input:

```json
{
  "reviewer": "gavin",
  "image_keys": ["img_..."],
  "confirmed_removable_count": 140,
  "filters": {}
}
```

The confirm endpoint should recompute the removal preview server-side instead of trusting the preview payload. `confirmed_removable_count` is only a guard against accidental stale confirmations.

Reviewer scope:

- preview and confirm remove only review JSONs under `<reviews-dir>/<reviewer>/` and legacy archives for that reviewer/key
- other reviewers' review files for the same `image_key` are not removed by this operation
- cross-reviewer cleanup must be an explicit future operation

## CLI Plan

First-version CLI helper:

```bash
.venv/bin/python scripts/training_pipeline_cli.py remove-images \
  --reviewer gavin \
  --image-key img_abc \
  --image-key img_def
```

Useful options:

- `--reviewer <name>`
- `--reviews-dir <path>`
- `--reports-dir <path>`
- `--image-key <key>` repeatable
- `--filter inactive-review` to remove manifest-backed images that no longer have an active review JSON
- `--album-id <id>` later, once album-aware queue filtering is implemented
- `--dry-run`

The CLI must use the same backend helper as the GUI endpoints.

CLI semantics:

- explicit `--image-key` values are removed regardless of review status
- if only `--filter inactive-review` is provided, expand to manifest-backed images without an active review JSON for that reviewer
- if only `--image-key` values are provided, evaluate those keys without extra filter expansion
- if neither `--image-key` nor `--filter` is provided, return a usage error
- `--dry-run` should return the same JSON envelope shape as the preview endpoint, not a separate free-form summary
- successful non-dry-run output should include the removal report path, removed count, and deleted workspace paths

## Re-Import Safety

Duplicate detection must remain content-based:

- duplicate identity is `source_content_hash + render_profile`
- same filename is not enough to be a duplicate
- same content under a different filename should map to the same `image_key`
- same filename with different content should map to a different `image_key`

Review overwrite rules:

- existing review JSON files are never overwritten during import
- reviewed or partially reviewed JSON files remain untouched
- if an image was removed, re-import may create a new fresh manifest/cache/review entry for that image
- import may add album membership and manifest `album_ids` for already-known images without changing review labels, detections, or review status

Tests before queue removal ships:

- same content under two filenames produces the same `image_key`
- same filename with different content produces a different `image_key`
- re-import of a reviewed image does not overwrite the review JSON
- `remove-images --image-key` removes reviewed, partially reviewed, and unreviewed selected images from the local workspace
- `remove-images --filter inactive-review` removes manifest rows whose active review JSON is absent
- Step 3 removal deduplicates selected bbox tiles by source `image_key`
- Step 2 reviewer enumeration skips `_archived_unreviewed` and other underscore-prefixed directories
- Step 2 queue cache invalidates when review-directory mtimes change
- CLI `remove-images --dry-run` previews affected manifest entries, review paths, cache paths, and album membership

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
