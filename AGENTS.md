# Project Goal

Detect dark ship-to-ship (STS) transfers by first analyzing satellite imagery to identify likely STS events with a YOLO-based model, then cross-checking those detections against AIS records for matching time and location.

# Data Handling Rule

The folder `/home/gavin/Downloads/AIS data` is read-only. Agents must never modify it in any way, including writing files, changing files, moving files, renaming files, deleting files, or running code or commands that alter anything in that folder.

# Workspace Rule

Agents must do all working steps in their own workspace inside this repository and copy only the inputs they need from the read-only source when appropriate.

# Server Rule

When running a local development server or long-lived process, agents must start it as a detached background process so the LLM session remains usable. Prefer `nohup ... > workspace/reports/<name>.log 2>&1 < /dev/null &` and then verify it with `pgrep` or a targeted health check instead of leaving it attached to the current session.
