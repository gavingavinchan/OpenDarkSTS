# Project Goal

Detect dark ship-to-ship (STS) transfers by first analyzing satellite imagery to identify likely STS events with a YOLO-based model, then cross-checking those detections against AIS records for matching time and location.

# Branch Context

The current `ais-first-pipeline` branch is focused on the training pipeline: reviewing, curating, training, and evaluating YOLO datasets for the dot-detector and later STS models. See `docs/grand_plan_v2.md` implementation order and `docs/training_pipeline_plan.md` for the authoritative training-side plan. For training pipeline CLI usage and automation diagnostics, refer to `docs/training_pipeline_cli.md`.

# Academic Writing Tone Rule

Unless explicitly told otherwise, treat all prose writing (the paper, draft sections, abstracts, captions, and similar) as text for the academic paper and write it in a formal, academic register. Use precise, impersonal phrasing; avoid contractions, colloquialisms, casual asides, marketing language, and analogies aimed at a general audience. This rule governs tone and wording only; it does not change the factual content the user asks for.

# Paper Audience Rule

Most readers of the paper want to learn how to use the tool; only a minority read it to understand how the tool is implemented. Write accordingly: lead with the user-facing overview (what the tool does and how it is operated), and place implementation detail after it or in clearly secondary positions. When a section opens, prefer an overview-first ordering, with whole-tool implementation notes following the user-perspective material.

The readership is the SNAME maritime and ocean-engineering community, not computer scientists. Use technical terms that an engineer from any background will understand, and avoid computer-science jargon: prefer "dataset" or "the reviewed images" over "corpus", and "a grid of image thumbnails" over "contact sheet". If a specialized term is unavoidable, explain it plainly on first use.

# Data Handling Rule

The folder `/home/gavin/Downloads/AIS data` is read-only. Agents must never modify it in any way, including writing files, changing files, moving files, renaming files, deleting files, or running code or commands that alter anything in that folder.

# Workspace Rule

Agents must do all working steps in their own workspace inside this repository and copy only the inputs they need from the read-only source when appropriate.

# YOLO Weight Reproducibility Rule

Every YOLO weight file (`*.pt`) kept in this repository must have accompanying documentation that explains how to recreate or re-obtain it. The documentation must include the source dataset or upstream source, starting weights or architecture, training command or acquisition command when known, key training parameters, and any known reproducibility gaps. Update `docs/yolo_weight_recreation.md` or an adjacent weight-specific doc whenever adding, moving, renaming, promoting, or deleting YOLO weights.

# Server Rule

When running a local development server or long-lived process, agents must start it as a detached background process so the LLM session remains usable. Prefer `nohup ... > workspace/reports/<name>.log 2>&1 < /dev/null &` and then verify it with `pgrep` or a targeted health check instead of leaving it attached to the current session. See the [visual review server runbook](docs/visual_review_server_runbook.md) for an example background server workflow.

# GUI/CLI Parity Rule

Every training-pipeline GUI function should have a corresponding backend helper and CLI/API command so agents and automated checks can exercise the same behavior without relying on browser interaction. When implementing or changing GUI behavior, update or add the matching CLI/backend path and verify the backend through the CLI whenever feasible. If no CLI exists yet for a GUI action, either add one or document the gap in the relevant plan before treating the feature as complete.

# GUI Interaction Logging Rule

Training-pipeline GUI actions should emit lightweight debug logs for user presses, clicks, selections, uploads, and destructive confirmations so UI problems can be diagnosed from `workspace/reports/` without relying only on browser inspection. Logs must avoid binary payloads and secrets, but should include timestamps, action names, selected ids/counts, current filters, job ids, and error messages where useful. When changing GUI behavior, update or preserve this logging path.
