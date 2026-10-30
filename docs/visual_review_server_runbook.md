# Training Pipeline UI Runbook

## Purpose

These are the exact steps used to start the integrated training pipeline UI in the background for manual testing.

For normal training-pipeline GUI work, start one server with `scripts/training_pipeline_ui.py`. That server owns `/import`, `/manual-review`, and `/bbox-overview`, and defaults to port `8766`.

Use the standalone visual reviewer command at the end of this document only when testing the focused step 2 reviewer by itself.

## Environment Prefix

The local `.venv` Python can require the newer Conda SQLite library for reviewer imports. Use this prefix for the integrated training UI and standalone reviewer:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1
```

## 1. Stop Any Existing Server

```bash
pkill -f "training_pipeline_ui.py --port 8766"
```

If you are intentionally running only the standalone step 2 reviewer, stop that separately:

```bash
pkill -f "review_visual_detections_ui.py --port 8766"
```

## 2. Start The Integrated Server In The Background

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 \
nohup .venv/bin/python -u scripts/training_pipeline_ui.py \
  --port 8766 \
  --reviews-dir workspace/reviews \
  --reviewer gavin \
  > workspace/reports/training_pipeline_ui_8766.log 2>&1 < /dev/null &
```

Notes:

- `nohup ... &` keeps the session usable.
- `-u` enables unbuffered logging so crashes show up immediately in the log.
- The log file is `workspace/reports/training_pipeline_ui_8766.log`.
- Do not leave the server attached to an LLM or automation terminal session.

## 3. Verify The Process Started

```bash
pgrep -af "training_pipeline_ui.py --port 8766"
```

You should see a Python process running with the same command line.

## 4. Verify The APIs Respond

Import stage:

```bash
curl -sS http://127.0.0.1:8766/api/import
```

Manual review stage:

```bash
curl -sS http://127.0.0.1:8766/api/state
```

BBox overview stage:

```bash
curl -sS http://127.0.0.1:8766/api/overview
```

These should return JSON. Useful quick checks are:

- `/api/import`: `image_count` and `status_counts`
- `/api/state`: `queue.total_images`, `queue.completed_images`, and current image metadata
- `/api/overview`: `tiles` and `conflicts`

## 5. Open The UI

Open:

```text
http://localhost:8766
```

## 6. Check Logs If Something Breaks

Read:

```text
workspace/reports/training_pipeline_ui_8766.log
```

## 7. Shut It Down

```bash
pkill -f "training_pipeline_ui.py --port 8766"
```

## Standalone Step 2 Reviewer

Use this only when testing the focused manual-review UI by itself. The integrated training UI above is preferred for normal pipeline work.

Start standalone reviewer:

```bash
LD_PRELOAD=/home/gavin/anaconda3/lib/libsqlite3.so.3.50.1 \
nohup .venv/bin/python -u scripts/review_visual_detections_ui.py \
  --port 8766 \
  --manifest workspace/manifests/visual_preprocess_manifest.json \
  --reviews-dir workspace/reviews \
  --reviewer gavin \
  > workspace/reports/visual_review_ui.log 2>&1 < /dev/null &
```

Verify standalone reviewer:

```bash
pgrep -af "review_visual_detections_ui.py --port 8766"
curl -sS http://127.0.0.1:8766/api/state
```

Stop standalone reviewer:

```bash
pkill -f "review_visual_detections_ui.py --port 8766"
```
