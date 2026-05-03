# Visual Review Server Runbook

## Purpose

These are the exact steps used to start the visual review UI in the background for manual testing.

For normal training-pipeline GUI work, start one server with `scripts/training_pipeline_ui.py` instead. That server owns `/import`, `/manual-review`, and `/bbox-overview`, and defaults to port `8766`. Use this runbook only when testing the focused step 2 reviewer by itself.

## 1. Stop Any Existing Server

```bash
pkill -f "review_visual_detections_ui.py --port 8766"
```

## 2. Start The Server In The Background

```bash
nohup python -u scripts/review_visual_detections_ui.py \
  --port 8766 \
  --manifest workspace/manifests/visual_preprocess_manifest.json \
  --reviews-dir workspace/reviews \
  --reviewer gavin \
  > workspace/reports/visual_review_ui.log 2>&1 < /dev/null &
```

Notes:

- `nohup ... &` keeps the session usable.
- `-u` enables unbuffered logging so crashes show up immediately in the log.
- The log file is `workspace/reports/visual_review_ui.log`.

## 3. Verify The Process Started

```bash
pgrep -af "review_visual_detections_ui.py --port 8766"
```

You should see a Python process running with the same command line.

## 4. Verify The API Responds

```bash
curl -sS http://127.0.0.1:8766/api/state
```

This should return JSON with queue, image, detections, and counts.

## 5. Open The UI

Open:

```text
http://localhost:8766
```

## 6. Check Logs If Something Breaks

Read:

```text
workspace/reports/visual_review_ui.log
```

## 7. Shut It Down

```bash
pkill -f "review_visual_detections_ui.py --port 8766"
```
