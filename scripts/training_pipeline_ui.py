#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import secrets
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .visual_review_lib import (
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )
    from . import training_pipeline_lib as pipeline
    from . import review_visual_detections_ui as review_ui
except ImportError:
    from visual_review_lib import (
        DEFAULT_REVIEWER,
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )
    import training_pipeline_lib as pipeline
    import review_visual_detections_ui as review_ui


REVIEWS_DIR = DEFAULT_REVIEWS_DIR
CROP_ROOT = WORKSPACE_DIR / "cache" / "review_crops"
PADDING_PX = 48
INCLUDE_INCOMPLETE = False
GENERATE_CROPS = True
MANUAL_REVIEW_URL = "/manual-review"

IMPORT_MANIFEST_PATH = pipeline.DEFAULT_IMPORT_MANIFEST_PATH
IMPORT_CACHE_DIR = pipeline.DEFAULT_IMPORT_CACHE_DIR
IMPORT_STAGING_ROOT = WORKSPACE_DIR / "cache" / "import_staging"
IMPORT_REVIEWER = DEFAULT_REVIEWER

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_RETENTION_SECONDS = 60 * 30
REVIEW_POST_PATHS = {
    "/api/debug-interaction",
    "/api/start-detail",
    "/api/navigate-detail",
    "/api/save-detection",
    "/api/label-detection",
    "/api/mark-neither",
    "/api/delete-detection",
    "/api/mark-image-reviewed",
    "/api/complete-image",
    "/api/reopen-image",
}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def query_bool(parsed, name: str, default: bool) -> bool:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return default
    return values[0].strip().lower() not in {"0", "false", "no", "off"}


def parse_multipart_files(body: bytes, content_type: str) -> list[tuple[str, bytes]]:
    boundary_match = re.search(r"boundary=([^;]+)", content_type or "")
    if not boundary_match:
        raise ValueError("multipart upload missing boundary")
    boundary = b"--" + boundary_match.group(1).strip().strip('"').encode("latin-1")
    chunks = body.split(boundary)
    files: list[tuple[str, bytes]] = []
    for chunk in chunks[1:-1]:
        chunk = chunk.lstrip(b"\r\n")
        header_end = chunk.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_blob = chunk[:header_end].decode("latin-1", errors="replace")
        payload = chunk[header_end + 4:]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        filename_match = re.search(r'filename="([^"]*)"', header_blob)
        if not filename_match:
            continue
        filename = filename_match.group(1)
        if not filename:
            continue
        files.append((filename, payload))
    return files


def safe_relative_path(raw: str) -> Path:
    cleaned = raw.replace("\\", "/").lstrip("/")
    parts: list[str] = []
    for segment in cleaned.split("/"):
        if not segment or segment in {".", ".."}:
            continue
        parts.append(segment)
    if not parts:
        raise ValueError(f"refusing unsafe upload path: {raw!r}")
    return Path(*parts)


def new_job_id() -> str:
    return "imp_" + secrets.token_hex(6)


def reap_old_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_SECONDS
    with JOBS_LOCK:
        stale = [
            job_id for job_id, job in JOBS.items()
            if job["state"] in {"done", "failed"} and job.get("finished_epoch", 0) < cutoff
        ]
        for job_id in stale:
            JOBS.pop(job_id, None)


def update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)


def progress_caption(event: dict) -> str:
    phase = event.get("phase")
    if phase == "discovered":
        total = event.get("total", 0)
        skipped_unsupported = event.get("discovered_failed", 0)
        if total == 0:
            return "No supported images found in the upload."
        suffix = f" ({skipped_unsupported} unsupported skipped)" if skipped_unsupported else ""
        return f"Discovered {total} files{suffix}. Beginning import."
    if phase == "processing":
        index = event.get("index", 0) + 1
        total = event.get("total", 0)
        current = event.get("current", "")
        return f"Importing {index} of {total}: {current}"
    if phase == "saving":
        return "Writing manifest..."
    if phase == "done":
        return (
            f"Done. {event.get('imported_count', 0)} imported, "
            f"{event.get('skipped_count', 0)} already imported, "
            f"{event.get('failed_count', 0)} failed."
        )
    return ""


def query_text(parsed, name: str) -> str | None:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def run_import_job(
    job_id: str,
    staging_dir: Path,
    *,
    force: bool,
    recursive: bool,
    album_name: str | None,
    album_description: str | None,
    source_display_name: str | None,
) -> None:
    def callback(event: dict) -> None:
        phase = event.get("phase")
        fields = {
            "phase": phase,
            "current_step": progress_caption(event),
            "total_count": event.get("total", 0),
        }
        if phase == "processing":
            fields["processed_count"] = event.get("index", 0)
            fields["current_filename"] = event.get("current", "")
            fields["imported_count"] = event.get("imported_count", 0)
            fields["skipped_count"] = event.get("skipped_count", 0)
            fields["failed_count"] = event.get("failed_count", 0)
        elif phase == "done":
            fields["processed_count"] = event.get("total", 0)
            fields["imported_count"] = event.get("imported_count", 0)
            fields["skipped_count"] = event.get("skipped_count", 0)
            fields["failed_count"] = event.get("failed_count", 0)
        update_job(job_id, **fields)

    update_job(job_id, state="importing", phase="starting", current_step="Scanning uploaded files...")
    try:
        result = pipeline.import_training_data(
            sources=[str(staging_dir)],
            manifest_path=IMPORT_MANIFEST_PATH,
            cache_dir=IMPORT_CACHE_DIR,
            album_name=album_name,
            album_description=album_description,
            source_display_name=source_display_name,
            source_kind="browser_upload",
            import_job_id=job_id,
            reviews_dir=REVIEWS_DIR,
            reviewer=IMPORT_REVIEWER,
            recursive=recursive,
            force=force,
            progress_callback=callback,
        )
        review_ui.invalidate_review_queue_cache()
        update_job(
            job_id,
            state="done",
            phase="done",
            current_step=progress_caption({
                "phase": "done",
                "imported_count": result["imported_count"],
                "skipped_count": result["skipped_count"],
                "failed_count": result["failed_count"],
            }),
            result=result,
            album=result.get("album"),
            finished_at=utcnow_iso(),
            finished_epoch=time.time(),
        )
    except Exception as exc:
        update_job(
            job_id,
            state="failed",
            phase="failed",
            current_step=f"Import failed: {exc}",
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            finished_at=utcnow_iso(),
            finished_epoch=time.time(),
        )


def start_import_job(
    uploads: list[tuple[str, bytes]],
    *,
    force: bool,
    album_name: str | None,
    album_description: str | None,
    source_display_name: str | None,
) -> str:
    job_id = new_job_id()
    staging_dir = IMPORT_STAGING_ROOT / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for raw_name, blob in uploads:
        try:
            relative = safe_relative_path(raw_name)
        except ValueError:
            continue
        target = staging_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        saved += 1
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "state": "queued",
            "phase": "queued",
            "current_step": f"Queued {saved} uploaded files for import.",
            "total_count": saved,
            "processed_count": 0,
            "imported_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "current_filename": "",
            "started_at": utcnow_iso(),
            "finished_at": None,
            "finished_epoch": 0,
            "error": None,
            "result": None,
            "album": None,
            "staging_dir": str(staging_dir),
            "force": bool(force),
            "album_name": album_name,
            "album_description": album_description,
            "source_display_name": source_display_name,
        }
    thread = threading.Thread(
        target=run_import_job,
        args=(job_id, staging_dir),
        kwargs={
            "force": force,
            "recursive": True,
            "album_name": album_name,
            "album_description": album_description,
            "source_display_name": source_display_name,
        },
        daemon=True,
        name=f"import-job-{job_id}",
    )
    thread.start()
    return job_id


def manifest_summary() -> dict:
    manifest_file = repo_resolve(IMPORT_MANIFEST_PATH)
    if not manifest_file.exists():
        return {
            "manifest_exists": False,
            "manifest_path": str(manifest_file),
            "image_count": 0,
            "images": [],
            "status_counts": {},
            "updated_at": None,
        }
    manifest = pipeline.load_import_manifest(manifest_file)
    images_map = manifest.get("images", {}) or {}
    images: list[dict] = []
    status_counts: dict[str, int] = {}
    for image_key, entry in images_map.items():
        status = entry.get("import_status", "imported")
        status_counts[status] = status_counts.get(status, 0) + 1
        cache_path = entry.get("cache_path")
        cache_url = "/cache/" + cache_path[len("workspace/"):] if cache_path and cache_path.startswith("workspace/") else None
        images.append({
            "image_key": image_key,
            "original_filename": entry.get("original_filename"),
            "browser_relative_path": entry.get("browser_relative_path"),
            "source_path": entry.get("source_path"),
            "source_collection": entry.get("source_collection"),
            "source_display_name": entry.get("source_display_name"),
            "album_ids": entry.get("album_ids") or [],
            "source_size_bytes": entry.get("source_size_bytes"),
            "width": entry.get("width"),
            "height": entry.get("height"),
            "import_status": status,
            "imported_at": entry.get("imported_at"),
            "updated_at": entry.get("updated_at"),
            "render_profile_kind": (entry.get("render_profile") or {}).get("kind"),
            "loader": entry.get("loader"),
            "warnings": entry.get("warnings") or [],
            "cache_url": cache_url,
            "cache_path": cache_path,
        })
    images.sort(key=lambda item: item.get("imported_at") or "", reverse=True)
    return {
        "manifest_exists": True,
        "manifest_path": str(manifest_file),
        "image_count": len(images),
        "images": images,
        "status_counts": status_counts,
        "updated_at": manifest.get("updated_at"),
    }


def render_bbox_overview_html() -> str:
    return (
        BBOX_OVERVIEW_HTML
        .replace("__MANUAL_REVIEW_URL__", json.dumps(MANUAL_REVIEW_URL))
        .replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")
    )


def render_app_html() -> str:
    return render_bbox_overview_html()


def render_import_html() -> str:
    return IMPORT_HTML.replace("__MANUAL_REVIEW_HREF__", MANUAL_REVIEW_URL.rstrip("/") + "/")


BBOX_OVERVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / BBox Overview</title>
  <style>
    :root {
      --ink: #111713;
      --muted: #667164;
      --paper: #ece7da;
      --card: #f8f4e8;
      --line: rgba(17, 23, 19, 0.16);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --slate: #44515a;
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }
    .shell { min-height: 100vh; display: grid; grid-template-rows: auto auto 1fr; }
    header {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.78);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: clamp(28px, 3vw, 44px);
      letter-spacing: -0.03em;
    }
    .subtitle { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.34);
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .controls {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) repeat(3, auto);
      gap: 10px;
      align-items: center;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(236, 231, 218, 0.72);
      position: sticky;
      top: 92px;
      z-index: 3;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: rgba(248, 244, 232, 0.92);
      color: var(--ink);
    }
    button { cursor: pointer; font-weight: 800; }
    button.primary { color: #fffaf0; background: var(--green); border-color: var(--green); }
    .size-control { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
    .main { display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: 0; }
    .contact-sheet { padding: 24px 28px 36px; min-width: 0; }
    .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .stat {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      background: rgba(248, 244, 232, 0.8);
      font-size: 12px;
      color: var(--muted);
    }
    .stat strong { color: var(--ink); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--tile-size, 170px), 1fr));
      gap: 14px;
      align-items: start;
    }
    .tile {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(248, 244, 232, 0.92);
      box-shadow: 0 10px 26px rgba(17, 23, 19, 0.08);
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .tile:hover, .tile.selected { transform: translateY(-2px); border-color: rgba(15, 107, 79, 0.55); box-shadow: var(--shadow); }
    .thumb {
      aspect-ratio: 1 / 0.72;
      display: grid;
      place-items: center;
      background: #17211b;
      color: rgba(255, 250, 240, 0.78);
      overflow: hidden;
    }
    .thumb img { width: 100%; height: 100%; object-fit: contain; display: block; image-rendering: auto; }
    .tile-meta { padding: 10px 11px 12px; display: grid; gap: 7px; }
    .tile-title { display: flex; justify-content: space-between; gap: 8px; align-items: center; font-size: 13px; font-weight: 900; }
    .tile-sub { color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .badge-row { display: flex; gap: 5px; flex-wrap: wrap; }
    .badge { border-radius: 999px; padding: 4px 7px; font-size: 11px; font-weight: 900; }
    .badge.ship { background: rgba(36, 88, 166, 0.14); color: var(--blue); }
    .badge.ships { background: rgba(168, 50, 38, 0.14); color: var(--red); }
    .badge.neither { background: rgba(68, 81, 90, 0.14); color: var(--slate); }
    .badge.warn { background: rgba(183, 121, 31, 0.18); color: #7a4d0e; }
    aside {
      border-left: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.86);
      padding: 22px;
      position: sticky;
      top: 154px;
      height: calc(100vh - 154px);
      overflow: auto;
    }
    .panel { display: grid; gap: 16px; }
    .panel h2 { margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: 25px; }
    .kv { display: grid; grid-template-columns: 110px 1fr; gap: 7px 10px; font-size: 13px; }
    .kv span:nth-child(odd) { color: var(--muted); }
    .mono { font-family: "Cascadia Mono", "IBM Plex Mono", monospace; font-size: 12px; overflow-wrap: anywhere; }
    .empty { padding: 56px 20px; text-align: center; color: var(--muted); }
    .conflicts { margin-bottom: 16px; display: grid; gap: 8px; }
    .conflict { border: 1px solid rgba(168, 50, 38, 0.28); border-radius: 14px; padding: 11px; background: rgba(168, 50, 38, 0.08); }
    @media (max-width: 980px) {
      header, .controls { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .main { grid-template-columns: 1fr; }
      aside { position: static; height: auto; border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>BBox Overview</h1>
        <p class="subtitle">Reviewed crop contact sheet for final visual QA before dataset selection.</p>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage active" href="/bbox-overview">BBox Overview</a>
        <span class="stage">Dataset Selection</span>
        <span class="stage">Training Setup</span>
        <span class="stage">Train YOLO</span>
        <span class="stage">Evaluate</span>
      </nav>
    </header>
    <section class="controls">
      <input id="search" placeholder="Search image, detection, reviewer, model" oninput="renderAndPersist()">
      <select id="classFilter" onchange="renderAndPersist()">
        <option value="all">All reviewed bboxes</option>
        <option value="ship_non_sts">Ship</option>
        <option value="ships_sts">Ships</option>
        <option value="neither">Neither</option>
        <option value="warnings">Warnings</option>
        <option value="conflicts">Needs Reconciliation</option>
      </select>
      <select id="sortMode" onchange="renderAndPersist()">
        <option value="image">Sort by image</option>
        <option value="class">Sort by class</option>
        <option value="area_desc">Largest boxes first</option>
        <option value="confidence_desc">Highest confidence first</option>
      </select>
      <label class="size-control">Tile <input id="tileSize" type="range" min="120" max="280" value="170" oninput="renderAndPersist()"></label>
      <button class="primary" onclick="reloadOverview(true)">Refresh Crops</button>
    </section>
    <main class="main">
      <section class="contact-sheet">
        <div class="stats" id="stats"></div>
        <div class="conflicts" id="conflicts"></div>
        <div class="grid" id="grid"></div>
      </section>
      <aside><div class="panel" id="sidebar"></div></aside>
    </main>
  </div>
  <script>
    const manualReviewUrl = __MANUAL_REVIEW_URL__;
    const overviewStateKey = 'opendarksts.bboxOverview.state.v1';
    let overview = { counts: {}, tiles: [], conflicts: [] };
    let selectedTileId = null;
    let pendingScrollY = null;
    let scrollSaveTimer = null;

    function readStoredOverviewState() {
      try {
        return JSON.parse(localStorage.getItem(overviewStateKey) || sessionStorage.getItem(overviewStateKey) || '{}');
      } catch (error) {
        console.warn('Could not read overview state:', error);
        return {};
      }
    }

    function restoreOverviewState() {
      const stored = readStoredOverviewState();
      if (typeof stored.search === 'string') document.getElementById('search').value = stored.search;
      if (typeof stored.classFilter === 'string') document.getElementById('classFilter').value = stored.classFilter;
      if (typeof stored.sortMode === 'string') document.getElementById('sortMode').value = stored.sortMode;
      if (typeof stored.tileSize === 'string') document.getElementById('tileSize').value = stored.tileSize;
      if (typeof stored.selectedTileId === 'string') selectedTileId = stored.selectedTileId;
      if (Number.isFinite(Number(stored.scrollY))) pendingScrollY = Number(stored.scrollY);
    }

    function currentOverviewState() {
      return {
        search: document.getElementById('search').value,
        classFilter: document.getElementById('classFilter').value,
        sortMode: document.getElementById('sortMode').value,
        tileSize: document.getElementById('tileSize').value,
        selectedTileId,
        scrollY: window.scrollY,
      };
    }

    function persistOverviewState() {
      const serialized = JSON.stringify(currentOverviewState());
      localStorage.setItem(overviewStateKey, serialized);
      sessionStorage.setItem(overviewStateKey, serialized);
    }

    function renderAndPersist() {
      render();
      persistOverviewState();
    }

    function restoreScrollIfNeeded() {
      if (pendingScrollY === null) return;
      const target = pendingScrollY;
      pendingScrollY = null;
      requestAnimationFrame(() => {
        window.scrollTo(0, target);
        persistOverviewState();
      });
    }

    async function reloadOverview(forceGenerate=false) {
      const response = await fetch('/api/overview' + (forceGenerate ? '?generate=1' : ''));
      overview = await response.json();
      if (!overview.tiles.some(tile => tile.tile_id === selectedTileId)) selectedTileId = null;
      render();
      restoreScrollIfNeeded();
    }

    function labelClass(tile) {
      if (tile.review_label === 'ships_sts') return 'ships';
      if (tile.review_label === 'neither') return 'neither';
      return 'ship';
    }

    function labelText(tile) {
      if (tile.review_label === 'ships_sts') return 'ships';
      if (tile.review_label === 'neither') return 'neither';
      return 'ship';
    }

    function filteredTiles() {
      const filter = document.getElementById('classFilter').value;
      if (filter === 'conflicts') return [];
      const query = document.getElementById('search').value.trim().toLowerCase();
      let tiles = overview.tiles.filter(tile => {
        if (filter === 'warnings' && (!tile.warnings || tile.warnings.length === 0)) return false;
        if (filter !== 'all' && filter !== 'warnings' && tile.review_label !== filter) return false;
        if (!query) return true;
        const haystack = [tile.image_key, tile.image_filename, tile.detection_id, tile.reviewer, tile.source_type, ...(tile.source_models || [])].join(' ').toLowerCase();
        return haystack.includes(query);
      });
      const sort = document.getElementById('sortMode').value;
      tiles.sort((left, right) => {
        if (sort === 'class') return String(left.review_label).localeCompare(String(right.review_label)) || String(left.image_key).localeCompare(String(right.image_key));
        if (sort === 'area_desc') return Number(right.bbox_area || 0) - Number(left.bbox_area || 0);
        if (sort === 'confidence_desc') return Number(right.confidence || -1) - Number(left.confidence || -1);
        return String(left.image_key).localeCompare(String(right.image_key)) || String(left.detection_id).localeCompare(String(right.detection_id));
      });
      return tiles;
    }

    function renderStats(tiles) {
      const counts = overview.counts || {};
      document.getElementById('stats').innerHTML = `
        <span class="stat"><strong>${tiles.length}</strong> visible</span>
        <span class="stat"><strong>${counts.reviewed_bbox_count || 0}</strong> reviewed bboxes</span>
        <span class="stat"><strong>${counts.reviewed_image_count || 0}</strong> images</span>
        <span class="stat"><strong>${counts.reviewed_ship_bboxes || 0}</strong> ship</span>
        <span class="stat"><strong>${counts.reviewed_ships_bboxes || 0}</strong> ships</span>
        <span class="stat"><strong>${counts.neither_bboxes || 0}</strong> neither</span>
        <span class="stat"><strong>${counts.warning_tile_count || 0}</strong> warning tiles</span>
        <span class="stat"><strong>${counts.conflict_image_count || 0}</strong> conflicts</span>
      `;
    }

    function renderConflicts() {
      const host = document.getElementById('conflicts');
      const showOnly = document.getElementById('classFilter').value === 'conflicts';
      const conflicts = overview.conflicts || [];
      if (!conflicts.length || !showOnly) {
        host.innerHTML = '';
        return;
      }
      host.innerHTML = conflicts.map(conflict => `
        <div class="conflict">
          <strong>${conflict.image_key}</strong><br>
          <span class="mono">${(conflict.review_files || []).join('<br>')}</span>
        </div>
      `).join('');
    }

    function renderGrid(tiles) {
      const grid = document.getElementById('grid');
      const tileSize = document.getElementById('tileSize').value;
      grid.style.setProperty('--tile-size', `${tileSize}px`);
      if (document.getElementById('classFilter').value === 'conflicts') {
        grid.innerHTML = '';
        return;
      }
      if (!tiles.length) {
        grid.innerHTML = '<div class="empty">No reviewed bbox crops match the current filters.</div>';
        return;
      }
      grid.innerHTML = tiles.map(tile => `
        <article class="tile ${tile.tile_id === selectedTileId ? 'selected' : ''}" onclick="selectTile('${tile.tile_id}')" ondblclick="openManualReview('${tile.image_key}', '${tile.detection_id}')">
          <div class="thumb">${tile.crop_url ? `<img loading="lazy" src="${tile.crop_url}" alt="${tile.image_key} ${tile.detection_id}">` : 'No crop'}</div>
          <div class="tile-meta">
            <div class="tile-title"><span>${tile.detection_id}</span><span class="badge ${labelClass(tile)}">${labelText(tile)}</span></div>
            <div class="tile-sub">${tile.image_filename || tile.image_key}</div>
            <div class="badge-row">
              <span class="badge">${tile.source_type || 'unknown'}</span>
              ${(tile.warnings || []).map(warning => `<span class="badge warn">${warning}</span>`).join('')}
            </div>
          </div>
        </article>
      `).join('');
    }

    function selectedTile() {
      return overview.tiles.find(tile => tile.tile_id === selectedTileId) || null;
    }

    function renderSidebar() {
      const tile = selectedTile();
      const host = document.getElementById('sidebar');
      if (!tile) {
        host.innerHTML = '<h2>No Tile Selected</h2><p class="subtitle">Single-click a crop to inspect metadata. Double-click a crop to open the source image in manual review.</p>';
        return;
      }
      host.innerHTML = `
        <h2>${labelText(tile)} <span class="badge ${labelClass(tile)}">${tile.detection_id}</span></h2>
        <button class="primary" onclick="openManualReview('${tile.image_key}', '${tile.detection_id}')">Edit In Manual Review</button>
        <div class="kv">
          <span>Image</span><span class="mono">${tile.image_filename || tile.image_key}</span>
          <span>Image Key</span><span class="mono">${tile.image_key}</span>
          <span>Reviewer</span><span>${tile.reviewer || 'unknown'}</span>
          <span>Source</span><span>${tile.source_type || 'unknown'}</span>
          <span>Confidence</span><span>${tile.confidence === null || tile.confidence === undefined ? '-' : tile.confidence}</span>
          <span>Models</span><span>${(tile.source_models || []).join(', ') || '-'}</span>
          <span>Dataset</span><span>not assigned yet</span>
          <span>Warnings</span><span>${(tile.warnings || []).join(', ') || 'none'}</span>
          <span>BBox</span><span class="mono">x=${tile.bbox.x_center.toFixed(6)} y=${tile.bbox.y_center.toFixed(6)} w=${tile.bbox.width.toFixed(6)} h=${tile.bbox.height.toFixed(6)}</span>
          <span>Review File</span><span class="mono">${tile.review_file}</span>
        </div>
      `;
    }

    function selectTile(tileId) {
      selectedTileId = tileId;
      renderAndPersist();
    }

    function openManualReview(imageKey, detectionId) {
      persistOverviewState();
      const url = `${manualReviewUrl.replace(/[/]$/, '')}/?image_id=${encodeURIComponent(imageKey)}&detection_id=${encodeURIComponent(detectionId)}`;
      window.location.href = url;
    }

    function render() {
      const tiles = filteredTiles();
      renderStats(tiles);
      renderConflicts();
      renderGrid(tiles);
      renderSidebar();
    }

    window.addEventListener('scroll', () => {
      if (scrollSaveTimer) window.clearTimeout(scrollSaveTimer);
      scrollSaveTimer = window.setTimeout(persistOverviewState, 120);
    }, { passive: true });
    window.addEventListener('pagehide', persistOverviewState);
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') persistOverviewState();
    });

    restoreOverviewState();
    reloadOverview(false);
  </script>
</body>
</html>"""


IMPORT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS / Intake</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT@9..144,300..900,0..100&family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #111713;
      --ink-soft: #2a3530;
      --muted: #6f7a6e;
      --paper: #ece7da;
      --paper-warm: #e6dfca;
      --paper-cool: #d9e1d7;
      --card: #f8f4e8;
      --card-deep: #efe7d2;
      --line: rgba(17, 23, 19, 0.18);
      --line-strong: rgba(17, 23, 19, 0.42);
      --green: #0f6b4f;
      --red: #a83226;
      --blue: #2458a6;
      --amber: #b7791f;
      --stamp: #7a1f12;
      --stamp-deep: #4d1109;
      --rule: rgba(17, 23, 19, 0.12);
      --shadow: 0 18px 48px rgba(17, 23, 19, 0.16);
      --serif: "Fraunces", "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --sans: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", "Cascadia Mono", monospace;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 12% -10%, rgba(122, 31, 18, 0.06), transparent 60%),
        radial-gradient(900px 700px at 110% 110%, rgba(15, 107, 79, 0.07), transparent 60%),
        linear-gradient(90deg, rgba(17, 23, 19, 0.045) 1px, transparent 1px) 0 0 / 36px 36px,
        linear-gradient(180deg, #ded7c5 0%, #f2eddf 48%, #d9e1d7 100%);
      font-family: var(--sans);
      font-feature-settings: "ss01", "cv11";
    }
    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.06 0'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
      mix-blend-mode: multiply;
      opacity: 0.55;
    }
    .shell { position: relative; z-index: 1; min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }

    header {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 28px;
      align-items: end;
      padding: 22px 36px 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.72);
      backdrop-filter: blur(14px);
      position: sticky; top: 0; z-index: 6;
    }
    .titleblock { display: grid; grid-template-columns: auto 1fr; gap: 16px; align-items: center; }
    .filenum {
      font-family: var(--serif);
      font-weight: 300;
      font-style: italic;
      font-size: clamp(56px, 6vw, 96px);
      line-height: 0.85;
      color: var(--stamp);
      letter-spacing: -0.04em;
      transform: translateY(4px);
    }
    .filenum sup { font-size: 0.42em; vertical-align: super; color: var(--muted); font-style: normal; letter-spacing: 0.18em; font-family: var(--sans); font-weight: 700; padding-left: 4px; }
    h1 {
      margin: 0;
      font-family: var(--serif);
      font-weight: 500;
      font-size: clamp(30px, 3.4vw, 50px);
      letter-spacing: -0.035em;
      line-height: 1.0;
    }
    h1 em { font-style: italic; font-weight: 400; color: var(--ink-soft); }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      max-width: 56ch;
      letter-spacing: 0.01em;
    }
    .rail { display: flex; gap: 7px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .stage {
      display: inline-flex; align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.36);
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      text-decoration: none;
    }
    .stage.active { background: var(--ink); color: var(--card); border-color: var(--ink); }
    .stage:hover:not(.active):not(span) { background: rgba(255, 255, 255, 0.8); color: var(--ink); }

    main { padding: 0; min-height: 0; }

    /* ==================== empty state: intake card ==================== */
    .intake {
      display: grid;
      grid-template-columns: minmax(280px, 580px);
      justify-content: center;
      padding: 64px 28px 80px;
    }
    .intake-card {
      position: relative;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 56px 56px 44px;
      box-shadow:
        0 1px 0 rgba(255, 255, 255, 0.6) inset,
        0 28px 64px rgba(17, 23, 19, 0.18),
        0 80px 120px rgba(17, 23, 19, 0.06);
      overflow: hidden;
    }
    .intake-card::before {
      content: "";
      position: absolute; inset: 12px;
      border: 1px solid var(--rule);
      pointer-events: none;
      border-radius: 1px;
    }
    .intake-card::after {
      content: "FORM 01-A — TRAINING INTAKE";
      position: absolute; top: 22px; left: 50%;
      transform: translateX(-50%);
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.32em;
      color: var(--muted);
    }
    .intake-stamp {
      position: absolute;
      top: 36px; right: 28px;
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.22em;
      color: var(--stamp);
      border: 1.5px solid var(--stamp);
      padding: 6px 10px;
      transform: rotate(6deg);
      opacity: 0.78;
      border-radius: 3px;
    }
    .intake-eyebrow {
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.32em;
      color: var(--muted);
      text-transform: uppercase;
      margin: 20px 0 6px;
    }
    .intake-title {
      font-family: var(--serif);
      font-weight: 400;
      font-size: clamp(36px, 4.8vw, 60px);
      line-height: 0.95;
      letter-spacing: -0.035em;
      margin: 0 0 8px;
    }
    .intake-title em {
      font-style: italic;
      color: var(--stamp);
    }
    .intake-lede {
      color: var(--muted);
      font-size: 15px;
      line-height: 1.55;
      margin: 0 0 28px;
      max-width: 46ch;
    }
    .intake-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-bottom: 28px;
    }
    .pick-btn {
      position: relative;
      border: 1.5px solid var(--ink);
      background: var(--ink);
      color: var(--card);
      font-family: var(--sans);
      font-weight: 700;
      letter-spacing: 0.04em;
      padding: 18px 18px 16px;
      border-radius: 2px;
      cursor: pointer;
      text-align: left;
      display: grid;
      gap: 4px;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .pick-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 22px rgba(17, 23, 19, 0.22); }
    .pick-btn:active { transform: translateY(0); }
    .pick-btn .glyph { font-family: var(--serif); font-style: italic; font-size: 22px; font-weight: 300; opacity: 0.9; }
    .pick-btn .label { font-size: 13px; }
    .pick-btn .hint { font-size: 11px; opacity: 0.7; font-family: var(--mono); letter-spacing: 0.06em; }
    .pick-btn.secondary { background: transparent; color: var(--ink); }
    .pick-btn.secondary:hover { background: rgba(17, 23, 19, 0.06); }
    .intake-options {
      display: flex; align-items: center; gap: 14px;
      font-size: 12px; color: var(--muted);
      padding-top: 16px; border-top: 1px dashed var(--rule);
    }
    .intake-options label { display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
    .intake-options input { accent-color: var(--stamp); }
    .album-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 18px 0; }
    .album-fields label { display: grid; gap: 5px; color: var(--muted); font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; }
    .album-fields input { width: 100%; border: 1px solid var(--line); border-radius: 3px; background: rgba(255,255,255,0.52); padding: 9px 10px; color: var(--ink); font: 13px var(--mono); }
    .album-fields .wide { grid-column: 1 / -1; }
    .intake-footer {
      margin-top: 22px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.04em;
      color: var(--muted);
      line-height: 1.6;
    }
    .intake-footer code { color: var(--ink); background: rgba(17, 23, 19, 0.06); padding: 1px 5px; border-radius: 2px; }

    /* ==================== progress block ==================== */
    .progress {
      margin-top: 28px;
      padding: 22px 22px 18px;
      border: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.85);
      border-radius: 3px;
      display: none;
      animation: rise 240ms ease-out;
    }
    .progress.active { display: block; }
    .progress.failed { border-color: rgba(168, 50, 38, 0.45); background: rgba(168, 50, 38, 0.06); }
    @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .progress-head {
      display: flex; align-items: baseline; justify-content: space-between;
      gap: 12px; margin-bottom: 12px;
    }
    .progress-label {
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em;
      text-transform: uppercase; color: var(--stamp); font-weight: 700;
    }
    .progress-pct {
      font-family: var(--serif); font-style: italic; font-weight: 400;
      font-size: 22px; color: var(--ink); line-height: 1;
    }
    .progress-bar {
      position: relative; height: 12px;
      background:
        repeating-linear-gradient(45deg, rgba(17,23,19,0.05) 0 6px, transparent 6px 12px),
        var(--card-deep);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
    }
    .progress-fill {
      position: absolute; inset: 0 auto 0 0;
      width: 0%;
      background: linear-gradient(90deg, var(--green), #137a5a);
      transition: width 220ms ease;
    }
    .progress-fill.indeterminate {
      width: 30% !important;
      animation: slide 1.4s ease-in-out infinite;
    }
    @keyframes slide { 0% { transform: translateX(-110%); } 100% { transform: translateX(360%); } }
    .progress-caption {
      margin-top: 10px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink-soft);
      letter-spacing: 0.01em;
      min-height: 1.2em;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .progress-counts {
      margin-top: 10px;
      display: flex; gap: 14px;
      font-family: var(--mono); font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.04em;
    }
    .progress-counts strong { color: var(--ink); font-weight: 700; }
    .progress-counts .fail strong { color: var(--stamp); }

    /* ==================== populated state ==================== */
    .ledger { display: grid; grid-template-rows: auto auto 1fr; min-height: calc(100vh - 110px); }

    .compact-toolbar {
      display: grid;
      grid-template-columns: auto auto auto 1fr auto auto auto;
      gap: 10px; align-items: center;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(236, 231, 218, 0.72);
      position: sticky; top: 102px; z-index: 4;
    }
    .compact-toolbar .pill {
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
      text-transform: uppercase; color: var(--muted);
      padding: 8px 12px; border: 1px dashed var(--rule); border-radius: 2px;
    }
    .compact-toolbar .pill strong { color: var(--ink); font-weight: 700; }
    .toolbar-btn {
      border: 1px solid var(--ink);
      background: var(--ink);
      color: var(--card);
      font-family: var(--sans);
      font-weight: 700;
      font-size: 12px;
      padding: 10px 14px;
      border-radius: 2px;
      cursor: pointer;
      letter-spacing: 0.04em;
    }
    .toolbar-btn.ghost { background: transparent; color: var(--ink); }
    .toolbar-btn.ghost:hover { background: rgba(17, 23, 19, 0.06); }
    .toolbar-search {
      border: 1px solid var(--line);
      background: var(--card);
      padding: 10px 12px;
      border-radius: 2px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }
    .toolbar-search:focus { outline: 2px solid var(--stamp); outline-offset: 1px; }
    .view-toggle { display: inline-flex; border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }
    .view-toggle button {
      border: 0; background: transparent;
      font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 10px 12px; cursor: pointer; color: var(--muted);
    }
    .view-toggle button.active { background: var(--ink); color: var(--card); }
    .sort-select {
      border: 1px solid var(--line);
      background: var(--card);
      padding: 10px 12px;
      border-radius: 2px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }

    .layout {
      display: grid; grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 0;
    }
    .canvas { padding: 24px 28px 56px; min-width: 0; }
    .canvas .progress { margin-bottom: 22px; margin-top: 0; max-width: 720px; }

    /* ledger view */
    .ledger-table {
      width: 100%; border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
      font-family: var(--mono);
      font-size: 12px;
    }
    .ledger-table thead th {
      text-align: left;
      font-family: var(--mono);
      font-weight: 700;
      font-size: 10px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      padding: 12px 12px;
      border-bottom: 1.5px solid var(--ink);
      background: var(--card-deep);
    }
    .ledger-table tbody td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--rule);
      vertical-align: middle;
      color: var(--ink-soft);
    }
    .ledger-table tbody tr {
      cursor: pointer;
      transition: background 100ms ease;
    }
    .ledger-table tbody tr:hover { background: rgba(122, 31, 18, 0.04); }
    .ledger-table tbody tr.selected { background: rgba(15, 107, 79, 0.08); }
    .ledger-table tbody tr.selected td:first-child { box-shadow: inset 3px 0 0 var(--green); }
    .row-num { width: 56px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .row-thumb { width: 64px; }
    .row-thumb img {
      width: 56px; height: 36px; object-fit: cover;
      border: 1px solid var(--rule);
      background: #17211b;
      display: block;
    }
    .row-name { font-family: var(--sans); font-weight: 500; color: var(--ink); }
    .row-status { width: 130px; }
    .status-tag {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 1px;
      font-size: 10px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .status-tag.imported { background: rgba(15, 107, 79, 0.14); color: var(--green); border: 1px solid rgba(15, 107, 79, 0.32); }
    .status-tag.regenerated_cache { background: rgba(36, 88, 166, 0.14); color: var(--blue); border: 1px solid rgba(36, 88, 166, 0.32); }
    .status-tag.already_imported { background: rgba(68, 81, 90, 0.14); color: var(--ink-soft); border: 1px solid var(--rule); }
    .status-tag.failed { background: rgba(168, 50, 38, 0.14); color: var(--red); border: 1px solid rgba(168, 50, 38, 0.32); }

    /* contact sheet view */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 14px;
    }
    .tile {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 2px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }
    .tile:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(17, 23, 19, 0.12); border-color: rgba(15, 107, 79, 0.4); }
    .tile.selected { border-color: var(--green); box-shadow: 0 14px 28px rgba(15, 107, 79, 0.2); }
    .tile-img {
      aspect-ratio: 1.2 / 1;
      background: #17211b;
      display: grid; place-items: center;
      color: rgba(255, 250, 240, 0.62);
      overflow: hidden;
    }
    .tile-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .tile-meta { padding: 10px 12px 12px; display: grid; gap: 4px; }
    .tile-name {
      font-family: var(--sans); font-weight: 600; font-size: 12px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .tile-sub {
      font-family: var(--mono); font-size: 10px; color: var(--muted);
      letter-spacing: 0.04em;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }

    /* sidebar */
    aside.detail {
      border-left: 1px solid var(--line);
      background: rgba(248, 244, 232, 0.86);
      padding: 26px 24px;
      position: sticky; top: 168px;
      height: calc(100vh - 168px);
      overflow: auto;
    }
    .detail-empty {
      font-family: var(--serif); font-style: italic; font-weight: 300;
      color: var(--muted); font-size: 18px; line-height: 1.4;
      padding: 30px 0;
    }
    .detail-empty::before {
      content: "—";
      display: block; font-size: 36px; color: var(--rule);
      margin-bottom: 14px;
    }
    .detail-eyebrow {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.28em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
    }
    .detail-title {
      font-family: var(--serif); font-weight: 500; font-size: 22px;
      letter-spacing: -0.02em; margin: 0 0 4px; word-break: break-word;
    }
    .detail-key {
      font-family: var(--mono); font-size: 11px; color: var(--stamp);
      letter-spacing: 0.04em; margin-bottom: 18px;
    }
    .detail-thumb {
      width: 100%;
      aspect-ratio: 1.2 / 1;
      background: #17211b;
      border: 1px solid var(--line);
      margin-bottom: 18px;
      display: grid; place-items: center;
    }
    .detail-thumb img { width: 100%; height: 100%; object-fit: contain; }
    .kv {
      display: grid; grid-template-columns: 110px 1fr;
      gap: 8px 12px; font-size: 12px;
    }
    .kv dt {
      font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em;
      text-transform: uppercase; color: var(--muted); margin: 0;
    }
    .kv dd {
      margin: 0; color: var(--ink-soft);
      font-family: var(--mono); font-size: 11px;
      word-break: break-word;
    }
    .kv dd.value-strong { font-family: var(--sans); font-size: 13px; color: var(--ink); font-weight: 500; }

    .toast {
      position: fixed; bottom: 24px; left: 50%;
      transform: translateX(-50%) translateY(20px);
      background: var(--ink); color: var(--card);
      font-family: var(--mono); font-size: 12px;
      padding: 10px 16px; border-radius: 2px;
      box-shadow: 0 10px 24px rgba(17, 23, 19, 0.3);
      opacity: 0; transition: all 200ms ease;
      pointer-events: none; z-index: 50;
      letter-spacing: 0.04em;
    }
    .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
    .toast.error { background: var(--stamp); }

    @media (max-width: 980px) {
      header, .compact-toolbar { position: static; grid-template-columns: 1fr; }
      .rail { justify-content: flex-start; }
      .layout { grid-template-columns: 1fr; }
      aside.detail { position: static; height: auto; border-left: 0; border-top: 1px solid var(--line); }
      .intake-card { padding: 36px 28px 28px; }
      .intake-actions { grid-template-columns: 1fr; }
      .album-fields { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <input type="file" id="fileInput" multiple accept=".png,.jpg,.jpeg,.tif,.tiff" style="display:none">
  <input type="file" id="folderInput" webkitdirectory directory multiple style="display:none">

  <div class="shell">
    <header>
      <div class="titleblock">
        <div class="filenum">01<sup>/07</sup></div>
        <div>
          <h1>Intake <em>&amp;</em> Cache</h1>
          <p class="subtitle">Catalog source imagery, render browser-ready RGB cache, and seed the manual review queue. Source files are never modified.</p>
        </div>
      </div>
      <nav class="rail" aria-label="Training pipeline stages">
        <a class="stage active" href="/import">Import Data</a>
        <a class="stage" href="__MANUAL_REVIEW_HREF__">Manual Review</a>
        <a class="stage" href="/bbox-overview">BBox Overview</a>
        <span class="stage">Dataset Selection</span>
        <span class="stage">Training Setup</span>
        <span class="stage">Train YOLO</span>
        <span class="stage">Evaluate</span>
      </nav>
    </header>

    <main id="main"></main>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const stateKey = 'opendarksts.import.state.v1';
    const POLL_MS = 700;

    const state = {
      manifest: null,
      job: null,
      pollTimer: null,
      view: 'ledger',
      sort: 'imported_desc',
      search: '',
      selectedKey: null,
      forceRefresh: false,
      albumName: '',
      albumDescription: '',
      sourceDisplayName: '',
    };

    function restoreState() {
      try {
        const raw = JSON.parse(localStorage.getItem(stateKey) || '{}');
        if (raw.view === 'grid' || raw.view === 'ledger') state.view = raw.view;
        if (typeof raw.sort === 'string') state.sort = raw.sort;
        if (typeof raw.search === 'string') state.search = raw.search;
        if (typeof raw.selectedKey === 'string') state.selectedKey = raw.selectedKey;
        if (typeof raw.forceRefresh === 'boolean') state.forceRefresh = raw.forceRefresh;
        if (typeof raw.albumName === 'string') state.albumName = raw.albumName;
        if (typeof raw.albumDescription === 'string') state.albumDescription = raw.albumDescription;
        if (typeof raw.sourceDisplayName === 'string') state.sourceDisplayName = raw.sourceDisplayName;
      } catch (e) {}
    }
    function persistState() {
      localStorage.setItem(stateKey, JSON.stringify({
        view: state.view, sort: state.sort, search: state.search,
        selectedKey: state.selectedKey, forceRefresh: state.forceRefresh,
        albumName: state.albumName, albumDescription: state.albumDescription,
        sourceDisplayName: state.sourceDisplayName,
      }));
    }

    function escape(value) {
      return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[ch]);
    }
    function formatBytes(n) {
      if (!n && n !== 0) return '-';
      const units = ['B', 'KB', 'MB', 'GB'];
      let i = 0; let v = n;
      while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
      return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + ' ' + units[i];
    }
    function formatTime(iso) {
      if (!iso) return '-';
      try {
        const d = new Date(iso);
        return d.toISOString().replace('T', ' ').replace('Z', 'Z').slice(0, 19) + 'Z';
      } catch (e) { return iso; }
    }
    function showToast(message, kind) {
      const toast = document.getElementById('toast');
      toast.textContent = message;
      toast.classList.toggle('error', kind === 'error');
      toast.classList.add('show');
      clearTimeout(showToast._t);
      showToast._t = setTimeout(() => toast.classList.remove('show'), 2400);
    }

    async function loadManifest() {
      const resp = await fetch('/api/import');
      state.manifest = await resp.json();
      render();
    }

    function render() {
      const main = document.getElementById('main');
      const empty = !state.manifest || state.manifest.image_count === 0;
      const jobActive = state.job && (state.job.state === 'queued' || state.job.state === 'importing');
      if (empty && !jobActive && !(state.job && state.job.state === 'failed')) {
        main.innerHTML = renderEmptyState();
        wireIntakeControls();
      } else if (empty) {
        main.innerHTML = renderEmptyState();
        wireIntakeControls();
        renderProgressInto(document.querySelector('.intake-card'));
      } else {
        main.innerHTML = renderPopulatedState();
        wireToolbarControls();
        renderProgressInto(document.querySelector('.canvas'), { atTop: true });
      }
    }

    function renderEmptyState() {
      return `
        <div class="intake">
          <div class="intake-card">
            <div class="intake-stamp">UNFILED</div>
            <p class="intake-eyebrow">Step 01 / Training Pipeline</p>
            <h2 class="intake-title">Bring your <em>imagery</em> into the workspace.</h2>
            <p class="intake-lede">Pick image files or a folder. We will compute a stable <code>image_key</code>, render a browser-friendly cache, and create review stubs so manual review is ready immediately.</p>
            <div class="intake-actions">
              <button class="pick-btn" id="pickFiles">
                <span class="glyph">⌥</span>
                <span class="label">Pick Images</span>
                <span class="hint">.png .jpg .tif &mdash; one or many</span>
              </button>
              <button class="pick-btn secondary" id="pickFolder">
                <span class="glyph">▢</span>
                <span class="label">Pick Folder</span>
                <span class="hint">recursive scan, hidden skipped</span>
              </button>
            </div>
            ${renderAlbumFields()}
            <div class="intake-options">
              <label><input type="checkbox" id="forceRefresh" ${state.forceRefresh ? 'checked' : ''}> Force re-render existing cache</label>
            </div>
            <p class="intake-footer">For very large batches, prefer the CLI:<br><code>.venv/bin/python scripts/training_pipeline_cli.py import-data --source &lt;folder&gt; --recursive</code></p>
          </div>
        </div>
      `;
    }

    function renderPopulatedState() {
      const m = state.manifest;
      const sc = m.status_counts || {};
      return `
        <div class="ledger">
          <div class="compact-toolbar">
            <button class="toolbar-btn" id="pickFiles">+ Pick Images</button>
            <button class="toolbar-btn ghost" id="pickFolder">+ Pick Folder</button>
            <span class="pill"><strong>${m.image_count}</strong> imported · ${(sc.imported || 0)}+${(sc.regenerated_cache || 0)}+${(sc.already_imported || 0)}</span>
            <input class="toolbar-search" id="albumName" placeholder="album name" value="${escape(state.albumName)}">
            <input class="toolbar-search" id="search" placeholder="search filename, key, status" value="${escape(state.search)}">
            <select class="sort-select" id="sort">
              <option value="imported_desc" ${state.sort === 'imported_desc' ? 'selected' : ''}>Newest first</option>
              <option value="imported_asc" ${state.sort === 'imported_asc' ? 'selected' : ''}>Oldest first</option>
              <option value="name_asc" ${state.sort === 'name_asc' ? 'selected' : ''}>Filename A→Z</option>
              <option value="size_desc" ${state.sort === 'size_desc' ? 'selected' : ''}>Largest first</option>
            </select>
            <div class="view-toggle">
              <button class="${state.view === 'ledger' ? 'active' : ''}" id="viewLedger">Ledger</button>
              <button class="${state.view === 'grid' ? 'active' : ''}" id="viewGrid">Sheet</button>
            </div>
            <label style="font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:0.06em;display:inline-flex;gap:6px;align-items:center;">
              <input type="checkbox" id="forceRefresh" ${state.forceRefresh ? 'checked' : ''}> force
            </label>
          </div>
          <div class="layout">
            <section class="canvas" id="canvas">
              ${state.view === 'ledger' ? renderLedger() : renderGrid()}
            </section>
            <aside class="detail" id="detail">${renderDetail()}</aside>
          </div>
        </div>
      `;
    }

    function renderAlbumFields() {
      return `
        <div class="album-fields">
          <label>Album name<input id="albumName" value="${escape(state.albumName)}" placeholder="defaults from selected folder"></label>
          <label>Source label<input id="sourceDisplayName" value="${escape(state.sourceDisplayName)}" placeholder="e.g. Singapore / 2025"></label>
          <label class="wide">Description<input id="albumDescription" value="${escape(state.albumDescription)}" placeholder="optional notes for this album"></label>
        </div>
      `;
    }

    function filteredImages() {
      const images = (state.manifest?.images || []).slice();
      const q = state.search.trim().toLowerCase();
      const filtered = q ? images.filter(im => {
        const hay = [im.image_key, im.original_filename, im.import_status, im.source_collection, im.source_path].join(' ').toLowerCase();
        return hay.includes(q);
      }) : images;
      const sortKey = state.sort;
      filtered.sort((a, b) => {
        if (sortKey === 'imported_asc') return String(a.imported_at || '').localeCompare(String(b.imported_at || ''));
        if (sortKey === 'name_asc') return String(a.original_filename || '').localeCompare(String(b.original_filename || ''));
        if (sortKey === 'size_desc') return Number(b.source_size_bytes || 0) - Number(a.source_size_bytes || 0);
        return String(b.imported_at || '').localeCompare(String(a.imported_at || ''));
      });
      return filtered;
    }

    function renderLedger() {
      const images = filteredImages();
      if (!images.length) return '<p style="font-family:var(--serif);font-style:italic;color:var(--muted);padding:40px 0;">No imported images match the current search.</p>';
      const rows = images.map((im, idx) => `
        <tr class="${im.image_key === state.selectedKey ? 'selected' : ''}" onclick="selectImage('${escape(im.image_key)}')">
          <td class="row-num">${String(idx + 1).padStart(4, '0')}</td>
          <td class="row-thumb">${im.cache_url ? `<img loading="lazy" src="${escape(im.cache_url)}" alt="">` : ''}</td>
          <td class="row-name">${escape(im.original_filename || im.image_key)}</td>
          <td>${im.width || '?'}×${im.height || '?'}</td>
          <td>${formatBytes(im.source_size_bytes)}</td>
          <td>${escape(im.render_profile_kind || '-')}</td>
          <td class="row-status"><span class="status-tag ${escape(im.import_status)}">${escape(im.import_status)}</span></td>
          <td>${formatTime(im.imported_at).slice(0, 19)}</td>
        </tr>
      `).join('');
      return `
        <table class="ledger-table">
          <thead>
            <tr>
              <th class="row-num">№</th>
              <th></th>
              <th>Filename</th>
              <th>Dims</th>
              <th>Size</th>
              <th>Profile</th>
              <th>Status</th>
              <th>Imported</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    function renderGrid() {
      const images = filteredImages();
      if (!images.length) return '<p style="font-family:var(--serif);font-style:italic;color:var(--muted);padding:40px 0;">No imported images match the current search.</p>';
      return '<div class="grid">' + images.map(im => `
        <article class="tile ${im.image_key === state.selectedKey ? 'selected' : ''}" onclick="selectImage('${escape(im.image_key)}')">
          <div class="tile-img">${im.cache_url ? `<img loading="lazy" src="${escape(im.cache_url)}" alt="">` : 'no cache'}</div>
          <div class="tile-meta">
            <div class="tile-name">${escape(im.original_filename || im.image_key)}</div>
            <div class="tile-sub">${im.width || '?'}×${im.height || '?'} · ${escape(im.import_status)}</div>
          </div>
        </article>
      `).join('') + '</div>';
    }

    function renderDetail() {
      const im = (state.manifest?.images || []).find(x => x.image_key === state.selectedKey);
      if (!im) return '<div class="detail-empty">Select an entry to inspect its manifest record, render profile, and cache path.</div>';
      return `
        <p class="detail-eyebrow">Manifest record</p>
        <h3 class="detail-title">${escape(im.original_filename || im.image_key)}</h3>
        <div class="detail-key">${escape(im.image_key)}</div>
        <div class="detail-thumb">${im.cache_url ? `<img src="${escape(im.cache_url)}" alt="">` : ''}</div>
        <dl class="kv">
          <dt>Status</dt><dd class="value-strong"><span class="status-tag ${escape(im.import_status)}">${escape(im.import_status)}</span></dd>
          <dt>Dimensions</dt><dd>${im.width || '?'} × ${im.height || '?'}</dd>
          <dt>Source size</dt><dd>${formatBytes(im.source_size_bytes)}</dd>
          <dt>Render profile</dt><dd>${escape(im.render_profile_kind || '-')}</dd>
          <dt>Loader</dt><dd>${escape(im.loader || '-')}</dd>
          <dt>Imported at</dt><dd>${formatTime(im.imported_at)}</dd>
          <dt>Updated at</dt><dd>${formatTime(im.updated_at)}</dd>
          <dt>Album ids</dt><dd>${(im.album_ids || []).map(escape).join(', ') || '-'}</dd>
          <dt>Source label</dt><dd>${escape(im.source_display_name || '-')}</dd>
          <dt>Source path</dt><dd>${escape(im.source_path || im.browser_relative_path || '-')}</dd>
          <dt>Source collection</dt><dd>${escape(im.source_collection || '-')}</dd>
          <dt>Cache path</dt><dd>${escape(im.cache_path || '-')}</dd>
          <dt>Warnings</dt><dd>${(im.warnings || []).map(escape).join(', ') || 'none'}</dd>
        </dl>
      `;
    }

    function renderProgressInto(host, opts) {
      if (!host || !state.job) return;
      const job = state.job;
      const pct = job.total_count > 0
        ? Math.min(100, Math.round((job.processed_count / job.total_count) * 100))
        : (job.state === 'done' ? 100 : 0);
      const indeterminate = job.state === 'queued' || job.state === 'starting' || (job.total_count === 0 && job.state !== 'done' && job.state !== 'failed');
      const failed = job.state === 'failed';
      const block = document.createElement('div');
      block.className = 'progress active' + (failed ? ' failed' : '');
      block.innerHTML = `
        <div class="progress-head">
          <span class="progress-label">${failed ? 'Import failed' : (job.state === 'done' ? 'Import complete' : 'Importing')}</span>
          <span class="progress-pct">${pct}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill ${indeterminate ? 'indeterminate' : ''}" style="width:${pct}%"></div>
        </div>
        <div class="progress-caption">${escape(job.current_step || '')}</div>
        <div class="progress-counts">
          <span><strong>${job.imported_count || 0}</strong> imported</span>
          <span><strong>${job.skipped_count || 0}</strong> already imported</span>
          <span class="fail"><strong>${job.failed_count || 0}</strong> failed</span>
          <span style="margin-left:auto;">${job.processed_count || 0} / ${job.total_count || '?'} files</span>
        </div>
      `;
      if (opts && opts.atTop) host.prepend(block); else host.appendChild(block);
    }

    function wireIntakeControls() {
      const fileBtn = document.getElementById('pickFiles');
      const folderBtn = document.getElementById('pickFolder');
      const force = document.getElementById('forceRefresh');
      if (fileBtn) fileBtn.onclick = () => document.getElementById('fileInput').click();
      if (folderBtn) folderBtn.onclick = () => document.getElementById('folderInput').click();
      if (force) force.onchange = (e) => { state.forceRefresh = e.target.checked; persistState(); };
      const albumName = document.getElementById('albumName');
      const albumDescription = document.getElementById('albumDescription');
      const sourceDisplayName = document.getElementById('sourceDisplayName');
      if (albumName) albumName.oninput = (e) => { state.albumName = e.target.value; persistState(); };
      if (albumDescription) albumDescription.oninput = (e) => { state.albumDescription = e.target.value; persistState(); };
      if (sourceDisplayName) sourceDisplayName.oninput = (e) => { state.sourceDisplayName = e.target.value; persistState(); };
    }

    function wireToolbarControls() {
      wireIntakeControls();
      const search = document.getElementById('search');
      if (search) {
        search.oninput = (e) => { state.search = e.target.value; persistState(); rerenderCanvas(); };
      }
      const sort = document.getElementById('sort');
      if (sort) {
        sort.onchange = (e) => { state.sort = e.target.value; persistState(); rerenderCanvas(); };
      }
      const vL = document.getElementById('viewLedger');
      const vG = document.getElementById('viewGrid');
      if (vL) vL.onclick = () => { state.view = 'ledger'; persistState(); rerenderCanvas(true); };
      if (vG) vG.onclick = () => { state.view = 'grid'; persistState(); rerenderCanvas(true); };
    }

    function rerenderCanvas(includeToggle) {
      const canvas = document.getElementById('canvas');
      const detail = document.getElementById('detail');
      if (canvas) canvas.innerHTML = state.view === 'ledger' ? renderLedger() : renderGrid();
      if (detail) detail.innerHTML = renderDetail();
      if (includeToggle) {
        document.getElementById('viewLedger')?.classList.toggle('active', state.view === 'ledger');
        document.getElementById('viewGrid')?.classList.toggle('active', state.view === 'grid');
      }
    }

    function selectImage(key) {
      state.selectedKey = key;
      persistState();
      document.querySelectorAll('.ledger-table tbody tr, .tile').forEach(el => el.classList.remove('selected'));
      const detail = document.getElementById('detail');
      if (detail) detail.innerHTML = renderDetail();
      rerenderCanvas();
    }

    async function uploadFiles(fileList) {
      if (!fileList || fileList.length === 0) return;
      const formData = new FormData();
      let topLevel = '';
      for (const f of fileList) {
        const name = f.webkitRelativePath || f.name;
        if (!topLevel && f.webkitRelativePath && f.webkitRelativePath.includes('/')) {
          topLevel = f.webkitRelativePath.split('/')[0];
        }
        formData.append('files', f, name);
      }
      let albumName = state.albumName.trim();
      const sourceDisplayName = state.sourceDisplayName.trim() || topLevel;
      if (!albumName && !sourceDisplayName) {
        albumName = window.prompt('Album name for these files?', '') || '';
        if (!albumName.trim()) return;
        state.albumName = albumName.trim();
        persistState();
      }
      const params = new URLSearchParams();
      if (state.forceRefresh) params.set('force', '1');
      if (albumName.trim()) params.set('album_name', albumName.trim());
      if (state.albumDescription.trim()) params.set('album_description', state.albumDescription.trim());
      if (sourceDisplayName) params.set('source_display_name', sourceDisplayName);
      const url = '/api/import/upload' + (params.toString() ? '?' + params.toString() : '');
      try {
        const resp = await fetch(url, { method: 'POST', body: formData });
        if (!resp.ok) {
          const text = await resp.text();
          showToast('Upload failed: ' + text.slice(0, 120), 'error');
          return;
        }
        const data = await resp.json();
        state.job = { state: 'queued', current_step: 'Uploaded. Queued for import.', total_count: 0, processed_count: 0 };
        render();
        startPolling(data.job_id);
      } catch (err) {
        showToast('Upload error: ' + err.message, 'error');
      }
    }

    function startPolling(jobId) {
      stopPolling();
      const tick = async () => {
        try {
          const resp = await fetch('/api/import/jobs/' + encodeURIComponent(jobId));
          if (!resp.ok) { stopPolling(); return; }
          const job = await resp.json();
          state.job = job;
          render();
          if (job.state === 'done' || job.state === 'failed') {
            stopPolling();
            if (job.state === 'done') {
              showToast(`Imported ${job.imported_count}, skipped ${job.skipped_count}, failed ${job.failed_count}`);
              await loadManifest();
              setTimeout(() => { state.job = null; render(); }, 4000);
            }
          }
        } catch (err) {
          stopPolling();
        }
      };
      tick();
      state.pollTimer = setInterval(tick, POLL_MS);
    }
    function stopPolling() {
      if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
    }

    document.getElementById('fileInput').addEventListener('change', (e) => uploadFiles(e.target.files));
    document.getElementById('folderInput').addEventListener('change', (e) => uploadFiles(e.target.files));

    window.selectImage = selectImage;

    restoreState();
    loadManifest();
  </script>
</body>
</html>"""


class TrainingPipelineHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", ""}:
            self._respond_html(render_import_html())
            return
        if path == "/bbox-overview":
            self._respond_html(render_bbox_overview_html())
            return
        if path == "/import":
            self._respond_html(render_import_html())
            return
        if path in {"/manual-review", "/manual-review/"}:
            queue_rows = review_ui.synced_review_queue()
            default_image_id = review_ui.pick_image_id(queue_rows, self._query_image_id(parsed))
            self._respond_html(review_ui.render_app_html(default_image_id, self._query_detection_id(parsed)))
            return
        if path == "/api/overview":
            generate = query_bool(parsed, "generate", GENERATE_CROPS)
            payload = build_review_bbox_overview(
                REVIEWS_DIR,
                crop_root=CROP_ROOT,
                padding_px=PADDING_PX,
                include_incomplete=INCLUDE_INCOMPLETE,
                generate_crops=generate,
            )
            self._respond_json(payload)
            return
        if path == "/api/state":
            self._respond_json(review_ui.build_state(self._query_image_id(parsed)))
            return
        if path == "/api/import":
            self._respond_json(manifest_summary())
            return
        if path.startswith("/api/import/jobs/"):
            job_id = path[len("/api/import/jobs/"):]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                payload = dict(job) if job else None
            if payload is None:
                self._respond_json({"error": "job not found"}, status=404)
            else:
                self._respond_json(payload)
            return
        if path.startswith("/cache/"):
            self._serve_cache_file(path[len("/cache/"):])
            return
        self._respond_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/import/upload":
            self._handle_upload(parsed)
            return
        if parsed.path in REVIEW_POST_PATHS:
            review_ui.ReviewHandler.do_POST(self)
            return
        self._respond_json({"error": "not found"}, status=404)

    def _handle_upload(self, parsed) -> None:
        reap_old_jobs()
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._respond_json({"error": "empty upload"}, status=400)
            return
        body = self.rfile.read(length)
        try:
            files = parse_multipart_files(body, content_type)
        except ValueError as exc:
            self._respond_json({"error": str(exc)}, status=400)
            return
        if not files:
            self._respond_json({"error": "no files in upload"}, status=400)
            return
        force = query_bool(parsed, "force", False)
        album_name = query_text(parsed, "album_name")
        album_description = query_text(parsed, "album_description")
        source_display_name = query_text(parsed, "source_display_name")
        try:
            job_id = start_import_job(
                files,
                force=force,
                album_name=album_name,
                album_description=album_description,
                source_display_name=source_display_name,
            )
        except Exception as exc:
            self._respond_json({"error": f"could not stage upload: {exc}"}, status=500)
            return
        self._respond_json({"job_id": job_id, "queued_files": len(files)}, status=202)

    def _serve_cache_file(self, relative_path: str) -> None:
        cleaned = relative_path.lstrip("/")
        candidates = [repo_resolve(cleaned)] if cleaned.startswith("workspace/") else [repo_resolve("workspace") / cleaned]
        cache_root = repo_resolve("workspace/cache").resolve()
        target_resolved = None
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.exists() and (resolved == cache_root or cache_root in resolved.parents):
                target_resolved = resolved
                break
        if target_resolved is None:
            self._respond_json({"error": "cache file not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target_resolved))[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(target_resolved.read_bytes())

    def _query_image_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("image_id")
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _query_detection_id(self, parsed) -> str | None:
        values = parse_qs(parsed.query).get("detection_id")
        if not values:
            return None
        value = values[0].strip()
        return value or None

    def _respond_html(self, html: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _respond_json(self, payload: dict, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenDarkSTS training pipeline UI")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve on")
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR), help="Root directory containing review JSON files")
    parser.add_argument("--crop-root", default=str(WORKSPACE_DIR / "cache" / "review_crops"), help="Root directory for bbox crop thumbnails")
    parser.add_argument("--padding-px", type=int, default=48, help="Padding around generated bbox crop thumbnails")
    parser.add_argument("--manual-review-url", default="/manual-review", help="Base URL for the step 2 manual reviewer")
    parser.add_argument("--include-incomplete", action="store_true", help="Include reviewed bboxes from incomplete review files")
    parser.add_argument("--no-generate-crops", action="store_true", help="Do not generate missing or stale overview crop files")
    parser.add_argument("--import-manifest", default=str(pipeline.DEFAULT_IMPORT_MANIFEST_PATH), help="Path to the training import manifest")
    parser.add_argument("--import-cache-dir", default=str(pipeline.DEFAULT_IMPORT_CACHE_DIR), help="Directory holding browser-ready cache PNGs")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="Reviewer used when seeding new review stubs")
    args = parser.parse_args()

    global REVIEWS_DIR, CROP_ROOT, PADDING_PX, INCLUDE_INCOMPLETE, GENERATE_CROPS, MANUAL_REVIEW_URL
    global IMPORT_MANIFEST_PATH, IMPORT_CACHE_DIR, IMPORT_REVIEWER
    REVIEWS_DIR = Path(args.reviews_dir)
    CROP_ROOT = Path(args.crop_root)
    PADDING_PX = args.padding_px
    INCLUDE_INCOMPLETE = args.include_incomplete
    GENERATE_CROPS = not args.no_generate_crops
    MANUAL_REVIEW_URL = args.manual_review_url
    IMPORT_MANIFEST_PATH = Path(args.import_manifest)
    IMPORT_CACHE_DIR = Path(args.import_cache_dir)
    IMPORT_REVIEWER = args.reviewer
    IMPORT_STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    review_ui.MANIFEST_PATH = IMPORT_MANIFEST_PATH
    review_ui.REVIEWS_DIR = REVIEWS_DIR
    review_ui.REVIEWER = IMPORT_REVIEWER

    print("OpenDarkSTS Training Pipeline UI")
    print(f"  Import manifest: {repo_resolve(IMPORT_MANIFEST_PATH)}")
    print(f"  Import cache:    {repo_resolve(IMPORT_CACHE_DIR)}")
    print(f"  Reviews:         {repo_resolve(REVIEWS_DIR)}")
    print(f"  Reviewer:        {IMPORT_REVIEWER}")
    print(f"  Open http://localhost:{args.port}")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), TrainingPipelineHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
