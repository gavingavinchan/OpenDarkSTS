# Project Goal

Detect dark ship-to-ship (STS) transfers by first analyzing satellite imagery to identify likely STS events with a YOLO-based model, then cross-checking those detections against AIS records for matching time and location.

# Branch Context

The current `ais-first-pipeline` branch is focused on the training pipeline: reviewing, curating, training, and evaluating YOLO datasets for the dot-detector and later STS models. See `docs/grand_plan_v2.md` implementation order and `docs/training_pipeline_plan.md` for the authoritative training-side plan. For training pipeline CLI usage and automation diagnostics, refer to `docs/training_pipeline_cli.md`.

# Data Handling Rule

The folder `/home/gavin/Downloads/AIS data` is read-only. Agents must never modify it in any way, including writing files, changing files, moving files, renaming files, deleting files, or running code or commands that alter anything in that folder.

# Workspace Rule

Agents must do all working steps in their own workspace inside this repository and copy only the inputs they need from the read-only source when appropriate.

# YOLO Weight Reproducibility Rule

Every YOLO weight file (`*.pt`) kept in this repository must have accompanying documentation that explains how to recreate or re-obtain it. The documentation must include the source dataset or upstream source, starting weights or architecture, training command or acquisition command when known, key training parameters, and any known reproducibility gaps. Update `docs/yolo_weight_recreation.md` or an adjacent weight-specific doc whenever adding, moving, renaming, promoting, or deleting YOLO weights.

# Server Rule

When running a local development server or long-lived process, agents must start it as a detached background process so the LLM session remains usable. Prefer `nohup ... > workspace/reports/<name>.log 2>&1 < /dev/null &` and then verify it with `pgrep` or a targeted health check instead of leaving it attached to the current session. See the [visual review server runbook](docs/visual_review_server_runbook.md) for an example background server workflow.
