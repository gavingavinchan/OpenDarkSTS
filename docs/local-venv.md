# Local Python Environment

This repo uses a local virtual environment at `.venv/` for model training and evaluation.

## Why

- Keeps PyTorch, Ultralytics, and related packages isolated from the system Python
- Makes future LLM sessions use the same interpreter and package set
- Avoids mixing training dependencies with unrelated global packages

## Use It

Run tools from the repo root with the executables inside `.venv`:

```bash
.venv/bin/python --version
.venv/bin/pip list
.venv/bin/python scripts/compare_yolo_models.py --help
```

If you want an interactive shell session to use the venv first:

```bash
source .venv/bin/activate
python --version
```

## Installed Core Packages

- `torch`
- `torchvision`
- `torchaudio`
- `ultralytics`

## Notes For Future Sessions

- Prefer `.venv/bin/python` and `.venv/bin/pip` in commands for reproducibility
- Store training outputs under `workspace/` so the read-only AIS source is never touched
- If dependencies drift later, update this file with the new setup steps

## Visual Reviewer Runtime Note

The current Python environment has a `sqlite3` loading issue unless `LD_PRELOAD` is set to the newer Conda SQLite library.

Use this exact prefix when running the visual review tools:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 .venv/bin/python ...
```

Without that preload, imports may fail with an error about `_sqlite3` and `sqlite3_deserialize`.

## Visual Reviewer Commands

Preprocess a small batch first:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 \
  .venv/bin/python scripts/preprocess_sentinel_tiles.py --limit 5
```

Start the review UI:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 \
  .venv/bin/python scripts/review_visual_detections_ui.py --port 8766
```

Then open:

```text
http://localhost:8766
```

Export reviewed labels:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 \
  .venv/bin/python scripts/export_reviewed_labels.py
```

Useful generated paths:

- Database: `workspace/databases/visual_detections.sqlite`
- Manifest: `workspace/manifests/visual_preprocess_manifest.json`
- Cache: `workspace/cache/sentinel_stitched/`
- Exported dataset: `workspace/datasets/sentinel_reviewed/`
- UI log if started with `nohup`: `workspace/reports/visual_reviewer.log`
