# Step 1 GUI Fix — Implementation Summary

Implemented per `docs/training_pipeline_step_4-7_gui_training_plan.md` (album-management prerequisite). Steps 4–7 deferred.

## Files changed

```
scripts/training_pipeline_lib.py        +292
scripts/training_pipeline_cli.py        +133
scripts/training_pipeline_ui.py         +489
scripts/review_visual_detections_ui.py   +84
```

No changes outside these four scripts. `workspace/manifests/training_import_manifest.json` and `workspace/albums/index.json` were used during smoke testing and confirmed byte-identical to baseline at the end (`git diff --stat` empty for both).

## Backend helpers (`training_pipeline_lib.py`)

Five new functions, plus an `EMPTY_ALBUM_COUNTS` constant matching the existing `import_training_data` count keys:

- `list_albums_summary(albums_dir, *, manifest_path, reviews_dir, reviewer)` — reads `albums/index.json`, joins per-album review status from `list_review_documents`, returns enriched summaries (image_count, counts, review_counts, latest_import_job_id, updated_at).
- `create_empty_album(*, albums_dir?, manifest_path, cache_dir, reviews_dir, album_name, album_description?, source_display_name?, reviewer)` — writes `albums/<id>.json` and updates `index.json` using the schema in plan lines 69–110, with `source_kind: "manual_album"` and `image_count: 0` / `latest_import_job_id: null`.
- `add_existing_image_keys_to_album(*, albums_dir?, manifest_path, album_id, image_keys, reviewer)` — appends manifest-backed keys without re-importing; updates `entry.album_ids`, album `images`, and album `counts`. Returns `members_added_existing`/`members_already_present`/`members_failed` plus a `failed: [{image_key, reason}]` list for keys not in the manifest.
- `remove_image_keys_from_album(*, albums_dir?, manifest_path, album_id, image_keys)` — membership-only removal: drops `album_id` from each entry's `album_ids` and removes from `album.images`. Leaves all cache, review files, manifest entries intact.
- `load_album_detail(albums_dir, album_id, *, manifest_path, reviews_dir, reviewer)` — full ledger plus `image_keys` and `review_counts`.

Reuse: existing `new_album_id`, `load_album_record`, `write_album_index`, `merge_album_images`, `add_album_id_to_entry`, `import_training_data(append_to_album_id=...)`. Nothing pre-existing was rewritten.

## CLI commands (`training_pipeline_cli.py`)

JSON-envelope subcommands matching `docs/training_pipeline_cli.md` shape:

- `list-albums`
- `show-album --album-id <id>`
- `create-album --album-name <name> [--album-description ...] [--source-display-name ...]`
- `add-to-album --album-id <id> --image-key <key> [--image-key ...]`
- `remove-from-album --album-id <id> --image-key <key> [--image-key ...]`

All accept `--manifest`, `--albums-dir`, `--reviews-dir`, `--reviewer` overrides.

## HTTP API (`training_pipeline_ui.py`)

- `GET /api/albums` → `{albums: [...summaries]}`
- `GET /api/albums/<album_id>` → full ledger + `image_keys` + `review_counts`
- `POST /api/albums` body `{album_name, album_description?, source_display_name?}`
- `POST /api/albums/<album_id>/add-images` body `{image_keys: [...]}`
- `POST /api/albums/<album_id>/remove-images` body `{image_keys: [...]}`
- `POST /api/import/upload` now accepts `append_to_album_id` query param (threaded through `start_import_job` → `run_import_job` → `import_training_data(append_to_album_id=...)`).

All album-related GUI events log through the existing `/api/training-ui-interaction` endpoint; no second logging mechanism was added.

## `/import` UI

- Albums section visible in both empty and populated states. Grid of album cards plus a leading "+ New empty album" tile.
- Each album card shows name, description, image count, reviewed count, last-updated date, album_id.
- Per-album actions: `Add Photos` (sets target album, opens file picker), `Review` (link to `/manual-review?album_id=<id>`), `Filter` (toggles the existing filter dropdown to `album:<id>`).
- Target bar above the pick buttons shows whether the next upload creates a new album (red/stamp tint) or appends to a selected album (green tint) with a "New album instead" clear button.
- `+ New empty album` prompts for name + optional description, calls `POST /api/albums`, refreshes the album list, and auto-targets the new album for the next upload.
- After successful import, `state.targetAlbumId` is cleared and both albums and manifest reload.
- Removal buttons distinguish scope per plan line 56:
  - `Remove From Workspace` (renamed from "Remove selected") — full purge via existing `/api/images/remove`.
  - `Remove From Album` — membership-only, only enabled when filtered to a specific album, calls `POST /api/albums/<id>/remove-images`.

## `/manual-review?album_id=<id>` album-scoped queue

- New `album_image_key_set(album_id)` and `filter_queue_by_album(queue_rows, album_id)` helpers in `review_visual_detections_ui.py` apply album membership at queue time.
- `build_state(image_id, album_id=None)` and the integrated `/manual-review` and `/api/state` GET handlers in `training_pipeline_ui.py` accept `album_id`.
- `ReviewHandler.do_POST` (`/api/start-detail`, `/api/save-detection`, `/api/label-detection`, `/api/set-detection-class`, `/api/mark-neither`, `/api/delete-detection`, `/api/mark-image-reviewed`, `/api/complete-image`, `/api/reopen-image`, `/api/navigate-detail`) reads `album_id` from query and threads it into `build_state(...)`.
- `render_app_html` now injects `initialAlbumId` plus an `albumFilterId` JS const captured from `?album_id=` on page load.
- `setImageIdInUrl`, `fetchState`, and `postJson` all preserve `album_id` across navigation, so the album scope sticks once the user enters via `/manual-review?album_id=<id>`.

## Smoke test (run before/after, then cleaned up)

```
HTTP 200 on /import and /manual-review?album_id=alb_731c09f02c
GET /api/albums  → 4 albums (baseline)
POST /api/albums {SMOKE A}            → alb_bf61bfe490 (manual_album, 0 images)
CLI create-album SMOKE B              → alb_0427cb07a5
POST /api/albums/A1/add-images real_key → image_count=1, added=1, failed=0
CLI add-to-album A2 real_key            → image_count=1, added=1
manifest entry album_ids = [A1, A2]
GET /api/state?album_id=A1 → total_images=1 (full queue 712)
POST /api/start-detail?album_id=A1     → total_images=1 (filter sticks across POST)
POST /api/albums/A1/remove-images real_key → removed=1, image_count=0
manifest entry preserved, album_ids = [A2] (A1 dropped, A2 retained)
cleanup: deleted both smoke album files, removed both smoke entries from index.json
git diff --stat workspace/{manifests,albums}: empty
```

## Acceptance bullets (plan §"Step 1 Acceptance Checks")

- Create empty album in GUI ✓ (`+ New empty album` tile → `POST /api/albums`)
- Upload files into a new album ✓ (existing flow, preserved)
- Add another upload batch to an existing album ✓ (target-album path → `append_to_album_id`)
- Album list counts update after import ✓ (`loadAlbums()` re-runs as part of `loadManifest`)
- Filtering by album shows expected images ✓ (existing filter dropdown + `Filter` button on cards)
- CLI list/show/create/add-to-album ✓
- UI interaction logs record album events ✓ (`album_create_pressed`, `album_create_succeeded`, `album_add_photos_pressed`, `album_target_set`, `album_target_cleared`, `album_filter_pressed`, `album_review_pressed`, `remove_from_album_pressed`/`_confirmation`)
- `/manual-review?album_id=<id>` opens an album-scoped review queue ✓

## Not in this change

- Steps 4–7 remain deferred per plan.
- The plan reservations like `/api/runs/active` for Step 6's persistent run card are not built yet; only the album / manual-review surface is.
- `add-to-album` accepts existing manifest-backed `--image-key` only, as specified; "add new local files" continues to use `import-data --append-to-album-id`.

## GPT-5.5 Follow-up On `add-to-album` Failure Semantics

I tightened the one automation concern noted after reviewing the album implementation.

Changed behavior:

- `add-to-album` still succeeds with exit code `0` when at least one requested image key is added or is otherwise a valid album member, even if other requested keys fail. It returns warnings and the structured `result.failed` list for the failed keys.
- `add-to-album` now exits nonzero when every requested key fails, returning `ok: false`, `errors: ["No image_key could be added to album"]`, warnings, and the structured failure result.
- `POST /api/albums/<album_id>/add-images` mirrors this by returning HTTP `400` with the same structured result when every requested key fails.

Files changed:

- `scripts/training_pipeline_cli.py`
- `scripts/training_pipeline_ui.py`
- `tests/test_training_pipeline_cli.py`
- `scratchpad.md`

Tests added:

- partial add failure remains successful with warnings
- all requested keys missing causes `add-to-album` to fail nonzero

Verification:

```text
python -m pytest tests/test_training_pipeline_cli.py tests/test_training_pipeline_ui.py tests/test_visual_review_labels.py
44 passed

python -m py_compile scripts/training_pipeline_lib.py scripts/training_pipeline_cli.py scripts/training_pipeline_ui.py scripts/review_visual_detections_ui.py
passed
```
