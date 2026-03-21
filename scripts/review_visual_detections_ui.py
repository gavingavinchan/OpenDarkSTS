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
        DEFAULT_DATABASE_PATH,
        DEFAULT_MANIFEST_PATH,
        CLASS_ID_TO_NAME,
        REVIEW_LABEL_TO_CLASS,
        complete_image_review,
        create_manual_detection,
        detection_payload,
        ensure_cache_for_date,
        get_active_detection_queue,
        get_db,
        get_image_queue,
        get_image_row,
        get_reviewable_detections,
        initialize_database,
        label_detection,
        mark_detection_neither,
        navigate_detail_cursor,
        queue_progress,
        repo_relative,
        repo_resolve,
        review_counts,
        set_image_status,
        start_detail_mode,
        sync_image_review_status,
        update_detection_geometry,
    )
except ImportError:
    from visual_review_lib import (
        DEFAULT_DATABASE_PATH,
        DEFAULT_MANIFEST_PATH,
        CLASS_ID_TO_NAME,
        REVIEW_LABEL_TO_CLASS,
        complete_image_review,
        create_manual_detection,
        detection_payload,
        ensure_cache_for_date,
        get_active_detection_queue,
        get_db,
        get_image_queue,
        get_image_row,
        get_reviewable_detections,
        initialize_database,
        label_detection,
        mark_detection_neither,
        navigate_detail_cursor,
        queue_progress,
        repo_relative,
        repo_resolve,
        review_counts,
        set_image_status,
        start_detail_mode,
        sync_image_review_status,
        update_detection_geometry,
    )


DB_PATH = DEFAULT_DATABASE_PATH
MANIFEST_PATH = DEFAULT_MANIFEST_PATH


def json_body(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode("utf-8")) if raw else {}


def pick_image_id(queue_rows, requested_image_id: int | None) -> int | None:
    if not queue_rows:
        return None
    queue_ids = [int(row["image_id"]) for row in queue_rows]
    if requested_image_id in queue_ids:
        return requested_image_id
    for row in queue_rows:
        if row["review_status"] != "completed":
            return int(row["image_id"])
    return int(queue_rows[0]["image_id"])


def build_state(conn, image_id: int | None) -> dict:
    queue_rows = get_image_queue(conn)
    progress = queue_progress(conn)
    selected_image_id = pick_image_id(queue_rows, image_id)

    if selected_image_id is None:
        return {
            "queue": {
                "images": [],
                "current_index": None,
                "prev_image_id": None,
                "next_image_id": None,
                **progress,
            },
            "image": None,
            "detections": [],
            "detail_queue": [],
            "current_detection_id": None,
            "counts": {
                "total_detections": 0,
                "active_detections": 0,
                "reviewed_active_detections": 0,
                "pending_active_detections": 0,
                "neither_detections": 0,
            },
        }

    sync_image_review_status(conn, selected_image_id)
    queue_rows = get_image_queue(conn)
    queue_payload = []
    for index, row in enumerate(queue_rows):
        queue_payload.append(
            {
                "image_id": int(row["image_id"]),
                "date": row["date"],
                "review_status": row["review_status"],
                "queue_index": index,
            }
        )
    image = get_image_row(conn, selected_image_id)
    if image is None:
        raise KeyError(f"Image {selected_image_id} not found")

    cache_path = ensure_cache_for_date(image["date"], manifest_path=MANIFEST_PATH)
    if repo_relative(cache_path) != image["cache_path"]:
        conn.execute(
            "UPDATE images SET cache_path = ?, updated_at = CURRENT_TIMESTAMP WHERE image_id = ?",
            (repo_relative(cache_path), selected_image_id),
        )
        conn.commit()
        image = get_image_row(conn, selected_image_id)
        if image is None:
            raise KeyError(f"Image {selected_image_id} not found after cache update")

    detections = [detection_payload(row, image["width"], image["height"]) for row in get_reviewable_detections(conn, selected_image_id)]
    detail_rows = get_active_detection_queue(conn, selected_image_id)
    detail_payload = [detection_payload(row, image["width"], image["height"]) for row in detail_rows]

    current_index = next(index for index, row in enumerate(queue_rows) if int(row["image_id"]) == selected_image_id)
    prev_image_id = int(queue_rows[current_index - 1]["image_id"]) if current_index > 0 else None
    next_image_id = int(queue_rows[current_index + 1]["image_id"]) if current_index < len(queue_rows) - 1 else None

    current_detection_id = None
    detail_cursor = image["detail_cursor"]
    if image["review_status"] == "detail" and detail_cursor is not None and 0 <= detail_cursor < len(detail_payload):
        current_detection_id = detail_payload[detail_cursor]["detection_id"]

    return {
        "queue": {
            "images": queue_payload,
            "current_index": current_index,
            "prev_image_id": prev_image_id,
            "next_image_id": next_image_id,
            **progress,
        },
        "image": {
            "image_id": int(image["image_id"]),
            "date": image["date"],
            "width": int(image["width"]),
            "height": int(image["height"]),
            "tile_count": int(image["tile_count"]),
            "review_status": image["review_status"],
            "detail_cursor": image["detail_cursor"],
            "cache_url": "/cache/" + repo_relative(cache_path),
        },
        "detections": detections,
        "detail_queue": detail_payload,
        "current_detection_id": current_detection_id,
        "counts": review_counts(conn, selected_image_id),
    }


def parse_image_id(payload: dict) -> int:
    image_id = payload.get("image_id")
    if image_id is None:
        raise ValueError("image_id is required")
    return int(image_id)


def parse_detection_geometry(payload: dict) -> tuple[float, float, float, float]:
    return (
        float(payload["x_center"]),
        float(payload["y_center"]),
        float(payload["width"]),
        float(payload["height"]),
    )


def reopen_if_completed(conn, image_id: int) -> None:
    image = get_image_row(conn, image_id)
    if image is not None and image["review_status"] == "completed":
        set_image_status(conn, image_id, "overview_2", None)


def render_app_html(default_image_id: int | None) -> str:
    initial_image = "null" if default_image_id is None else str(default_image_id)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visual Detection Reviewer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
  <style>
    :root {{
      --ink: #102330;
      --muted: #5e7380;
      --paper: #f4efe5;
      --paper-strong: #fffaf2;
      --panel: rgba(255, 250, 242, 0.9);
      --line: rgba(16, 35, 48, 0.14);
      --accent: #0f766e;
      --accent-strong: #115e59;
      --ship: #2563eb;
      --ships: #16803c;
      --pending: #b93824;
      --neither: #7c6f64;
      --shadow: 0 18px 48px rgba(16, 35, 48, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.16), transparent 34%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.12), transparent 28%),
        linear-gradient(180deg, #d9e8eb 0%, #f1ebdd 52%, #efe6d5 100%);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      min-height: 100vh;
    }}
    .shell {{ display: grid; grid-template-rows: auto 1fr; min-height: 100vh; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 250, 242, 0.82);
      backdrop-filter: blur(10px);
    }}
    .brand h1 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-size: 25px;
      letter-spacing: 0.01em;
    }}
    .brand p {{ margin: 4px 0 0; color: var(--muted); font-size: 13px; }}
    .queue-meta {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 12px;
      background: rgba(16, 35, 48, 0.07);
      color: var(--ink);
    }}
    .layout {{ display: grid; grid-template-columns: 390px 1fr; min-height: 0; }}
    .sidebar {{
      padding: 20px;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,250,242,0.88), rgba(250,244,232,0.78));
      overflow: auto;
    }}
    .viewer {{ position: relative; min-height: 520px; }}
    #map {{ width: 100%; height: calc(100vh - 88px); }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(16, 35, 48, 0.08);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 16px;
    }}
    .card h2, .card h3 {{
      margin: 0 0 10px;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      font-weight: 700;
    }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .meta-block {{ padding: 10px 12px; border-radius: 14px; background: rgba(16, 35, 48, 0.04); }}
    .meta-block label {{ display: block; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
    .meta-block strong {{ font-size: 15px; }}
    .controls {{ display: grid; gap: 10px; }}
    .button-row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    button, select {{
      border: 0;
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: var(--ink);
      color: white;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    button.secondary {{ background: rgba(16, 35, 48, 0.09); color: var(--ink); }}
    button.accent {{ background: var(--accent); }}
    button.ship {{ background: var(--ship); }}
    button.ships {{ background: var(--ships); }}
    button.warn {{ background: var(--pending); }}
    button.flat {{ background: rgba(16, 35, 48, 0.08); color: var(--ink); }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    .status-line {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
    .badge {{ border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .badge.overview_1 {{ background: rgba(185, 56, 36, 0.12); color: var(--pending); }}
    .badge.detail {{ background: rgba(15, 118, 110, 0.12); color: var(--accent-strong); }}
    .badge.overview_2 {{ background: rgba(37, 99, 235, 0.12); color: var(--ship); }}
    .badge.completed {{ background: rgba(22, 128, 60, 0.12); color: var(--ships); }}
    .legend {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; font-size: 13px; }}
    .legend div {{ display: flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .selection {{ font-size: 13px; color: var(--muted); line-height: 1.45; }}
    .selection strong {{ color: var(--ink); }}
    .empty {{ padding: 40px 20px; text-align: center; color: var(--muted); }}
    .footer-nav {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .queue-list {{ display: grid; gap: 6px; max-height: 220px; overflow: auto; }}
    .queue-item {{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid transparent;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      cursor: pointer;
      background: rgba(16, 35, 48, 0.04);
    }}
    .queue-item.active {{ border-color: rgba(15, 118, 110, 0.24); background: rgba(15, 118, 110, 0.08); }}
    .queue-item small {{ color: var(--muted); }}
    .leaflet-container {{ background: #0f172a; }}
    .viewer-hint {{
      position: absolute;
      right: 16px;
      bottom: 16px;
      z-index: 500;
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid rgba(16, 35, 48, 0.08);
      border-radius: 14px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
      max-width: 320px;
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      #map {{ height: 60vh; }}
      .topbar {{ flex-direction: column; align-items: flex-start; }}
      .queue-meta {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <h1>Visual Detection Reviewer</h1>
        <p>Review stitched Sentinel imagery, correct detections, and export clean YOLO labels.</p>
      </div>
      <div class="queue-meta" id="queueMeta"></div>
    </div>
    <div class="layout">
      <aside class="sidebar" id="sidebar"></aside>
      <main class="viewer">
        <div id="map"></div>
        <div class="viewer-hint" id="viewerHint">Load an image to review detections. Draw new boxes for misses, select a box to edit it, and use detail mode for labeling.</div>
      </main>
    </div>
  </div>

  <script>
    const initialImageId = {initial_image};
    const statusLabels = {{ pending: 'Pending', overview_1: 'Overview 1', detail: 'Detail', overview_2: 'Overview 2', completed: 'Completed' }};
    const defaultClassOptions = [{{value: 0, label: 'Ship'}}, {{value: 1, label: 'Ships'}}];

    let state = null;
    let map = null;
    let overlay = null;
    let detectionGroup = null;
    let selectedDetectionId = null;
    let drawHandler = null;
    let isEditing = false;

    function colorForDetection(detection) {{
      if (detection.review_label === 'neither' || !detection.is_active) return '#7c6f64';
      if (!detection.reviewed) return '#b93824';
      if (detection.class_id === 1) return '#16803c';
      return '#2563eb';
    }}

    function detectionName(detection) {{
      if (detection.review_label === 'ships_sts') return 'Ships (STS)';
      if (detection.review_label === 'ship_non_sts') return 'Ship (non-STS)';
      if (detection.review_label === 'neither') return 'Neither';
      return detection.class_id === 1 ? 'Ships' : 'Ship';
    }}

    function getImageIdFromUrl() {{
      const params = new URLSearchParams(window.location.search);
      const raw = params.get('image_id');
      if (!raw) return initialImageId;
      const parsed = parseInt(raw, 10);
      return Number.isFinite(parsed) ? parsed : initialImageId;
    }}

    function setImageIdInUrl(imageId) {{
      const params = new URLSearchParams(window.location.search);
      if (imageId === null || imageId === undefined) params.delete('image_id');
      else params.set('image_id', imageId);
      const next = `${{window.location.pathname}}?${{params.toString()}}`;
      window.history.replaceState({{}}, '', next);
    }}

    async function fetchState(imageId) {{
      const query = imageId ? `?image_id=${{encodeURIComponent(imageId)}}` : '';
      const response = await fetch('/api/state' + query);
      state = await response.json();
      if (state.image) setImageIdInUrl(state.image.image_id);
      renderAll();
    }}

    function renderAll() {{
      renderTopbar();
      renderSidebar();
      renderMap();
    }}

    function renderTopbar() {{
      const host = document.getElementById('queueMeta');
      if (!state || !state.queue) {{
        host.innerHTML = '';
        return;
      }}
      const currentIndex = state.queue.current_index === null ? '-' : state.queue.current_index + 1;
      host.innerHTML = `
        <span class="pill">${{state.queue.completed_images}} / ${{state.queue.total_images}} completed</span>
        <span class="pill">Queue position ${{currentIndex}}</span>
        <span class="pill">${{state.image ? state.image.date : 'No images loaded'}}</span>
      `;
    }}

    function renderSidebar() {{
      const host = document.getElementById('sidebar');
      if (!state.image) {{
        host.innerHTML = '<div class="empty">No preprocessed images found. Run <code>scripts/preprocess_sentinel_tiles.py</code> first.</div>';
        return;
      }}

      const current = state.image;
      const counts = state.counts;
      const currentDetection = state.detail_queue.find(item => item.detection_id === state.current_detection_id) || null;
      const selectedDetection = state.detections.find(item => item.detection_id === selectedDetectionId) || null;
      const defaultClassValue = currentDetection ? currentDetection.class_id : 0;

      host.innerHTML = `
        <div class="card">
          <div class="status-line">
            <span class="badge ${{current.review_status}}">${{statusLabels[current.review_status] || current.review_status}}</span>
            <span>${{current.date}}</span>
          </div>
          <h2 style="margin-top:12px">Image ${{current.date}}</h2>
          <div class="meta-grid">
            <div class="meta-block"><label>Tiles</label><strong>${{current.tile_count}}</strong></div>
            <div class="meta-block"><label>Canvas</label><strong>${{current.width}} x ${{current.height}}</strong></div>
            <div class="meta-block"><label>Active</label><strong>${{counts.active_detections}}</strong></div>
            <div class="meta-block"><label>Reviewed</label><strong>${{counts.reviewed_active_detections}}</strong></div>
          </div>
        </div>

        <div class="card">
          <h3>Legend</h3>
          <div class="legend">
            <div><span class="swatch" style="background:#16803c"></span> Reviewed STS</div>
            <div><span class="swatch" style="background:#2563eb"></span> Reviewed ship</div>
            <div><span class="swatch" style="background:#b93824"></span> Needs review</div>
            <div><span class="swatch" style="background:#7c6f64"></span> Neither / inactive</div>
          </div>
          <p class="note" style="margin-top:12px">New boxes are saved as manual detections and added to the detail queue. The final export keeps only standard YOLO labels.</p>
        </div>

        <div class="card controls">
          <h3>Overview Controls</h3>
          <div class="button-row">
            <button class="accent" onclick="startDraw()">Draw Box</button>
            <button class="secondary" onclick="toggleEditSelected()">${{isEditing ? 'Finish Edit' : 'Edit Selected'}}</button>
          </div>
          <div class="button-row">
            <select id="newClassSelect">${{defaultClassOptions.map(opt => `<option value="${{opt.value}}" ${{opt.value === defaultClassValue ? 'selected' : ''}}>${{opt.label}}</option>`).join('')}}</select>
            <button class="flat" onclick="saveSelectedGeometry()">Save Geometry</button>
          </div>
          <div class="button-row">
            <button class="warn" onclick="markSelectedNeither()">Mark Neither</button>
            <button class="secondary" onclick="clearSelection()">Clear Selection</button>
          </div>
          <div class="button-row">
            <button class="accent" onclick="startDetail()">Start Detail Review</button>
            <button class="secondary" onclick="completeImage()">Complete Image</button>
          </div>
          <p class="selection">${{selectedDetection ? `Selected <strong>#${{selectedDetection.detection_id}}</strong> - ${{detectionName(selectedDetection)}}` : 'No detection selected. Click a box to inspect or edit it.'}}</p>
        </div>

        <div class="card controls">
          <h3>Detail Controls</h3>
          <div class="status-line">
            <span>${{currentDetection ? `Detection ${{(current.detail_cursor || 0) + 1}} / ${{state.detail_queue.length}}` : 'No active detail target'}}</span>
            <span>${{counts.pending_active_detections}} pending</span>
          </div>
          <div class="button-row">
            <button class="ships" onclick="labelCurrent('ships_sts')" ${{currentDetection ? '' : 'disabled'}}>Ships (STS)</button>
            <button class="ship" onclick="labelCurrent('ship_non_sts')" ${{currentDetection ? '' : 'disabled'}}>Ship (non-STS)</button>
          </div>
          <div class="button-row">
            <button class="warn" onclick="labelCurrent('neither')" ${{currentDetection ? '' : 'disabled'}}>Neither</button>
            <button class="secondary" onclick="saveSelectedGeometry()">Save Current Box</button>
          </div>
          <div class="button-row">
            <button class="secondary" onclick="navigateDetail('prev')" ${{current.review_status === 'detail' ? '' : 'disabled'}}>Back</button>
            <button class="secondary" onclick="navigateDetail('next')" ${{current.review_status === 'detail' ? '' : 'disabled'}}>Next</button>
          </div>
          <p class="note">Need to split one detection into two? Draw replacement boxes, then mark the old one as neither.</p>
        </div>

        <div class="card">
          <h3>Queue</h3>
          <div class="queue-list">
            ${{state.queue.images.map(item => `
              <div class="queue-item ${{item.image_id === current.image_id ? 'active' : ''}}" onclick="openImage(${{item.image_id}})">
                <div>
                  <strong>${{item.date}}</strong><br>
                  <small>${{statusLabels[item.review_status] || item.review_status}}</small>
                </div>
                <small>#${{item.queue_index + 1}}</small>
              </div>
            `).join('')}}
          </div>
        </div>

        <div class="footer-nav">
          <button class="secondary" onclick="openImage(${{state.queue.prev_image_id}})" ${{state.queue.prev_image_id ? '' : 'disabled'}}>Previous Image</button>
          <button class="secondary" onclick="openImage(${{state.queue.next_image_id}})" ${{state.queue.next_image_id ? '' : 'disabled'}}>Next Image</button>
        </div>
      `;
    }}

    function renderMap() {{
      if (!map) {{
        map = L.map('map', {{ crs: L.CRS.Simple, minZoom: -4, zoomSnap: 0.25, attributionControl: false }});
        detectionGroup = L.featureGroup().addTo(map);
        map.on('draw:created', onDrawCreated);
      }}
      if (!state.image) return;

      if (overlay) overlay.remove();
      detectionGroup.clearLayers();
      const bounds = [[0, 0], [state.image.height, state.image.width]];
      overlay = L.imageOverlay(state.image.cache_url, bounds).addTo(map);
      map.fitBounds(bounds, {{ padding: [24, 24] }});

      state.detections.forEach(detection => {{
        const layer = L.rectangle(
          [[detection.y1_px, detection.x1_px], [detection.y2_px, detection.x2_px]],
          {{ color: colorForDetection(detection), weight: detection.detection_id === selectedDetectionId ? 4 : 2, fillOpacity: 0.03 }}
        );
        layer.detection = detection;
        layer.on('click', () => selectLayer(layer));
        detectionGroup.addLayer(layer);
        if (detection.detection_id === selectedDetectionId) {{
          layer.setStyle({{ weight: 4 }});
        }}
      }});

      if (state.image.review_status === 'detail' && state.current_detection_id) {{
        const currentLayer = findLayerByDetectionId(state.current_detection_id);
        if (currentLayer) {{
          selectLayer(currentLayer, false);
          focusLayer(currentLayer);
        }}
      }}
    }}

    function selectLayer(layer, rerender=true) {{
      selectedDetectionId = layer.detection.detection_id;
      detectionGroup.eachLayer(item => item.setStyle({{ weight: item === layer ? 4 : 2 }}));
      document.getElementById('viewerHint').textContent = `Selected detection #${{layer.detection.detection_id}}. Edit geometry, mark neither, or label it in detail mode.`;
      if (rerender) renderSidebar();
    }}

    function clearSelection() {{
      selectedDetectionId = null;
      isEditing = false;
      detectionGroup.eachLayer(item => item.setStyle({{ weight: 2 }}));
      renderSidebar();
    }}

    function findLayerByDetectionId(detectionId) {{
      let match = null;
      detectionGroup.eachLayer(layer => {{ if (layer.detection && layer.detection.detection_id === detectionId) match = layer; }});
      return match;
    }}

    function focusLayer(layer) {{
      map.fitBounds(layer.getBounds(), {{ padding: [80, 80], maxZoom: 2 }});
    }}

    function startDraw() {{
      if (!map || !state.image) return;
      drawHandler = new L.Draw.Rectangle(map, {{ shapeOptions: {{ color: '#0f766e', weight: 2 }} }});
      drawHandler.enable();
    }}

    function toggleEditSelected() {{
      const layer = selectedDetectionId ? findLayerByDetectionId(selectedDetectionId) : null;
      if (!layer || !layer.editing) return;
      if (isEditing) {{
        layer.editing.disable();
        isEditing = false;
      }} else {{
        layer.editing.enable();
        isEditing = true;
      }}
      renderSidebar();
    }}

    function boundsToGeometry(bounds) {{
      const west = bounds.getWest();
      const east = bounds.getEast();
      const north = bounds.getNorth();
      const south = bounds.getSouth();
      return {{
        x_center: ((west + east) / 2) / state.image.width,
        y_center: ((north + south) / 2) / state.image.height,
        width: (east - west) / state.image.width,
        height: (south - north) / state.image.height,
      }};
    }}

    async function onDrawCreated(event) {{
      const geometry = boundsToGeometry(event.layer.getBounds());
      const classId = parseInt(document.getElementById('newClassSelect')?.value || '0', 10);
      await postJson('/api/save-detection', {{
        image_id: state.image.image_id,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
        parent_detection_id: state.current_detection_id,
      }});
      await fetchState(state.image.image_id);
    }}

    async function saveSelectedGeometry() {{
      if (!selectedDetectionId) return;
      const layer = findLayerByDetectionId(selectedDetectionId);
      if (!layer) return;
      const geometry = boundsToGeometry(layer.getBounds());
      const classId = parseInt(document.getElementById('newClassSelect')?.value || String(layer.detection.class_id), 10);
      await postJson('/api/save-detection', {{
        image_id: state.image.image_id,
        detection_id: selectedDetectionId,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
      }});
      isEditing = false;
      await fetchState(state.image.image_id);
    }}

    async function markSelectedNeither() {{
      if (!selectedDetectionId) return;
      await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: selectedDetectionId }});
      selectedDetectionId = null;
      await fetchState(state.image.image_id);
    }}

    async function startDetail() {{
      await postJson('/api/start-detail', {{ image_id: state.image.image_id }});
      await fetchState(state.image.image_id);
    }}

    async function navigateDetail(direction) {{
      await postJson('/api/navigate-detail', {{ image_id: state.image.image_id, direction }});
      await fetchState(state.image.image_id);
    }}

    async function labelCurrent(reviewLabel) {{
      const targetId = selectedDetectionId || state.current_detection_id;
      if (!targetId) return;
      if (reviewLabel === 'neither') {{
        await postJson('/api/mark-neither', {{ image_id: state.image.image_id, detection_id: targetId }});
        selectedDetectionId = null;
        await fetchState(state.image.image_id);
        return;
      }}

      const layer = findLayerByDetectionId(targetId);
      const geometry = layer ? boundsToGeometry(layer.getBounds()) : state.detail_queue.find(item => item.detection_id === targetId);
      const classId = reviewLabel === 'ships_sts' ? 1 : 0;
      await postJson('/api/label-detection', {{
        image_id: state.image.image_id,
        detection_id: targetId,
        review_label: reviewLabel,
        class_id: classId,
        x_center: geometry.x_center,
        y_center: geometry.y_center,
        width: geometry.width,
        height: geometry.height,
      }});
      selectedDetectionId = null;
      await fetchState(state.image.image_id);
    }}

    async function completeImage() {{
      try {{
        await postJson('/api/complete-image', {{ image_id: state.image.image_id }});
        await fetchState(state.image.image_id);
      }} catch (error) {{
        alert(error.message);
      }}
    }}

    function openImage(imageId) {{
      if (!imageId) return;
      selectedDetectionId = null;
      isEditing = false;
      fetchState(imageId);
    }}

    async function postJson(url, payload) {{
      const response = await fetch(url, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
      }});
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || 'Request failed');
      return data;
    }}

    fetchState(getImageIdFromUrl());
  </script>
</body>
</html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            conn = get_db(DB_PATH)
            initialize_database(conn)
            queue_rows = get_image_queue(conn)
            default_image_id = pick_image_id(queue_rows, self._query_image_id(parsed))
            conn.close()
            self._respond_html(render_app_html(default_image_id))
            return

        if path == "/api/state":
            conn = get_db(DB_PATH)
            initialize_database(conn)
            payload = build_state(conn, self._query_image_id(parsed))
            conn.close()
            self._respond_json(payload)
            return

        if path.startswith("/cache/"):
            self._serve_cache_file(path[len("/cache/") :])
            return

        self._respond_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = json_body(self)
        conn = get_db(DB_PATH)
        initialize_database(conn)

        try:
            if parsed.path == "/api/start-detail":
                image_id = parse_image_id(payload)
                start_detail_mode(conn, image_id)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/navigate-detail":
                image_id = parse_image_id(payload)
                direction = payload.get("direction", "next")
                navigate_detail_cursor(conn, image_id, direction)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/save-detection":
                image_id = parse_image_id(payload)
                reopen_if_completed(conn, image_id)
                x_center, y_center, width, height = parse_detection_geometry(payload)
                detection_id = payload.get("detection_id")
                class_id = int(payload.get("class_id", 0))
                if detection_id is None:
                    create_manual_detection(
                        conn,
                        image_id=image_id,
                        class_id=class_id,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        parent_detection_id=payload.get("parent_detection_id"),
                    )
                else:
                    update_detection_geometry(conn, int(detection_id), x_center, y_center, width, height, class_id=class_id)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/label-detection":
                image_id = parse_image_id(payload)
                reopen_if_completed(conn, image_id)
                detection_id = int(payload["detection_id"])
                review_label = payload["review_label"]
                class_id = int(payload.get("class_id", REVIEW_LABEL_TO_CLASS[review_label]))
                x_center, y_center, width, height = parse_detection_geometry(payload)
                label_detection(conn, image_id, detection_id, review_label, class_id, x_center, y_center, width, height)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/mark-neither":
                image_id = parse_image_id(payload)
                reopen_if_completed(conn, image_id)
                detection_id = int(payload["detection_id"])
                mark_detection_neither(conn, image_id, detection_id)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/complete-image":
                image_id = parse_image_id(payload)
                complete_image_review(conn, image_id)
                self._respond_json(build_state(conn, image_id))
            elif parsed.path == "/api/reopen-image":
                image_id = parse_image_id(payload)
                set_image_status(conn, image_id, "overview_2", None)
                self._respond_json(build_state(conn, image_id))
            else:
                self._respond_json({"error": "not found"}, status=404)
        except Exception as exc:
            self._respond_json({"error": str(exc)}, status=400)
        finally:
            conn.close()

    def _query_image_id(self, parsed) -> int | None:
        values = parse_qs(parsed.query).get("image_id")
        if not values:
            return None
        try:
            return int(values[0])
        except ValueError:
            return None

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
    parser = argparse.ArgumentParser(description="Visual detection reviewer web UI")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve on")
    parser.add_argument("--db", default=str(DEFAULT_DATABASE_PATH), help="SQLite database path")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Preprocess manifest path")
    args = parser.parse_args()

    global DB_PATH, MANIFEST_PATH
    DB_PATH = args.db
    MANIFEST_PATH = args.manifest

    conn = get_db(DB_PATH)
    initialize_database(conn)
    progress = queue_progress(conn)
    conn.close()

    print("Visual Detection Reviewer")
    print(f"  Database: {repo_resolve(DB_PATH)}")
    print(f"  Manifest: {repo_resolve(MANIFEST_PATH)}")
    print(f"  Queue: {progress['completed_images']} / {progress['total_images']} completed")
    print(f"  Open http://localhost:{args.port}")

    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
