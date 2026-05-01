#!/usr/bin/python3
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .visual_review_lib import (
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_REVIEWS_DIR,
        WORKSPACE_DIR,
        build_review_bbox_overview,
        repo_resolve,
    )


REVIEWS_DIR = DEFAULT_REVIEWS_DIR
CROP_ROOT = WORKSPACE_DIR / "cache" / "review_crops"
PADDING_PX = 48
INCLUDE_INCOMPLETE = False
GENERATE_CROPS = True
MANUAL_REVIEW_URL = "http://localhost:8766"


def query_bool(parsed, name: str, default: bool) -> bool:
    values = parse_qs(parsed.query).get(name)
    if not values:
        return default
    return values[0].strip().lower() not in {"0", "false", "no", "off"}


def render_app_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenDarkSTS Training Pipeline</title>
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
        <span class="stage">Import Data</span>
        <a class="stage" href="/">Manual Review</a>
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
</html>""".replace("__MANUAL_REVIEW_URL__", json.dumps(MANUAL_REVIEW_URL))


class TrainingPipelineHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", ""}:
            self._respond_html(render_app_html())
            return
        if parsed.path == "/api/overview":
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
        if parsed.path.startswith("/cache/"):
            self._serve_cache_file(parsed.path[len("/cache/") :])
            return
        self._respond_json({"error": "not found"}, status=404)

    def _serve_cache_file(self, relative_path: str) -> None:
        target = repo_resolve(relative_path)
        cache_root = repo_resolve("workspace/cache")
        if not target.exists() or cache_root not in target.parents:
            self._respond_json({"error": "cache file not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(target.read_bytes())

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
    parser.add_argument("--port", type=int, default=8767, help="Port to serve on")
    parser.add_argument("--reviews-dir", default=str(DEFAULT_REVIEWS_DIR), help="Root directory containing review JSON files")
    parser.add_argument("--crop-root", default=str(WORKSPACE_DIR / "cache" / "review_crops"), help="Root directory for bbox crop thumbnails")
    parser.add_argument("--padding-px", type=int, default=48, help="Padding around generated bbox crop thumbnails")
    parser.add_argument("--manual-review-url", default="http://localhost:8766", help="Base URL for the step 2 manual reviewer")
    parser.add_argument("--include-incomplete", action="store_true", help="Include reviewed bboxes from incomplete review files")
    parser.add_argument("--no-generate-crops", action="store_true", help="Do not generate missing or stale overview crop files")
    args = parser.parse_args()

    global REVIEWS_DIR, CROP_ROOT, PADDING_PX, INCLUDE_INCOMPLETE, GENERATE_CROPS, MANUAL_REVIEW_URL
    REVIEWS_DIR = Path(args.reviews_dir)
    CROP_ROOT = Path(args.crop_root)
    PADDING_PX = args.padding_px
    INCLUDE_INCOMPLETE = args.include_incomplete
    GENERATE_CROPS = not args.no_generate_crops
    MANUAL_REVIEW_URL = args.manual_review_url

    print("OpenDarkSTS Training Pipeline UI")
    print("  Stage: BBox Overview")
    print(f"  Reviews: {repo_resolve(REVIEWS_DIR)}")
    print(f"  Crop root: {repo_resolve(CROP_ROOT)}")
    print(f"  Open http://localhost:{args.port}")

    server = HTTPServer(("0.0.0.0", args.port), TrainingPipelineHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
