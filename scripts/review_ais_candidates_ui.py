#!/usr/bin/python3
"""
AIS Candidate Reviewer — manual review web UI for AIS-based STS candidates.

Reads from workspace/databases/ais_denmark_candidates.sqlite, presents each
candidate event with a human-readable card and embedded OpenStreetMap view,
and saves review verdicts to an ais_sts_reviews table.

Usage:
    python scripts/review_ais_candidates_ui.py [--port 8765]
"""

import argparse
import json
import sqlite3
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DB_PATH = Path(__file__).resolve().parent.parent / "workspace" / "databases" / "ais_denmark_candidates.sqlite"

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def ensure_reviews_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ais_sts_reviews (
            event_id TEXT PRIMARY KEY,
            review_status TEXT NOT NULL DEFAULT 'pending_imagery',
            review_notes TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT,
            reviewer TEXT NOT NULL DEFAULT 'manual',
            duplicate_of_event_id TEXT,
            FOREIGN KEY (event_id) REFERENCES ais_sts_candidates(event_id) ON DELETE CASCADE
        );
    """)
    # Seed rows for any candidates not yet in the reviews table.
    conn.execute("""
        INSERT OR IGNORE INTO ais_sts_reviews (event_id, review_status, review_notes)
        SELECT event_id, review_status, review_notes
        FROM ais_sts_candidates;
    """)
    conn.commit()


def get_review_queue(conn):
    """Return ordered list of event_ids for the review queue.

    Order: HIGH before MEDIUM, then longer duration, then smaller separation.
    """
    rows = conn.execute("""
        SELECT c.event_id
        FROM ais_sts_candidates c
        LEFT JOIN ais_sts_reviews r ON c.event_id = r.event_id
        ORDER BY
            CASE c.confidence_tier WHEN 'HIGH' THEN 0 ELSE 1 END,
            c.duration_min DESC,
            c.median_separation_m ASC
    """).fetchall()
    return [r["event_id"] for r in rows]


def get_candidate(conn, event_id):
    row = conn.execute("""
        SELECT c.*, r.review_status AS current_review_status,
               r.review_notes AS current_review_notes,
               r.reviewed_at, r.duplicate_of_event_id
        FROM ais_sts_candidates c
        LEFT JOIN ais_sts_reviews r ON c.event_id = r.event_id
        WHERE c.event_id = ?
    """, (event_id,)).fetchone()
    return dict(row) if row else None


def get_progress(conn):
    total = conn.execute("SELECT COUNT(*) FROM ais_sts_candidates").fetchone()[0]
    reviewed = conn.execute(
        "SELECT COUNT(*) FROM ais_sts_reviews WHERE review_status != 'pending_imagery'"
    ).fetchone()[0]
    return reviewed, total


def save_review(conn, event_id, status, notes):
    now = datetime.utcnow().isoformat() + "Z"
    conn.execute("""
        INSERT INTO ais_sts_reviews (event_id, review_status, review_notes, reviewed_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            review_status = excluded.review_status,
            review_notes = excluded.review_notes,
            reviewed_at = excluded.reviewed_at
    """, (event_id, status, notes, now))
    conn.commit()


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def format_duration(minutes):
    if minutes < 60:
        return f"{minutes:.0f} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m"


def format_time(iso_str):
    """Extract HH:MM UTC from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M UTC")
    except Exception:
        return iso_str


def format_date(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso_str


def render_page(candidate, queue, queue_index, reviewed, total):
    c = candidate
    lat = c["center_lat"]
    lon = c["center_lon"]
    osm_url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=13/{lat}/{lon}"

    review_status = c.get("current_review_status", "pending_imagery")
    review_notes = c.get("current_review_notes", "")

    # Status badge color
    status_colors = {
        "pending_imagery": "#6b7280",
        "confirmed_sts": "#16a34a",
        "rejected_no_sts": "#dc2626",
        "unclear_imagery": "#d97706",
        "duplicate_event": "#7c3aed",
    }
    status_color = status_colors.get(review_status, "#6b7280")
    status_label = review_status.replace("_", " ").title()

    # Navigation
    prev_id = queue[queue_index - 1] if queue_index > 0 else None
    next_id = queue[queue_index + 1] if queue_index < len(queue) - 1 else None

    # Vessel speed descriptions
    def speed_desc(sog):
        if sog < 0.5:
            return "stationary"
        elif sog < 2:
            return "nearly stationary"
        else:
            return f"{sog:.1f} kn"

    v1_speed = speed_desc(c["vessel_1_mean_sog_kn"])
    v2_speed = speed_desc(c["vessel_2_mean_sog_kn"])

    # Confidence tier badge
    tier = c["confidence_tier"]
    tier_color = "#dc2626" if tier == "HIGH" else "#d97706"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIS Candidate Review — {c['event_id'][:12]}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; }}
  .top-bar {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 20px; background: #1e293b; border-bottom: 1px solid #334155;
  }}
  .top-bar .progress {{ color: #94a3b8; font-size: 14px; }}
  .main {{ display: flex; height: calc(100vh - 50px); }}
  .left {{ width: 420px; min-width: 380px; overflow-y: auto; padding: 20px;
           border-right: 1px solid #334155; background: #1e293b; }}
  .right {{ flex: 1; }}
  #map {{ width: 100%; height: 100%; }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600; color: #fff;
  }}
  .event-header {{
    display: flex; gap: 10px; align-items: center;
    margin-bottom: 16px; flex-wrap: wrap;
  }}
  .event-id {{ font-size: 13px; color: #64748b; font-family: monospace; }}
  .section {{ margin-bottom: 18px; }}
  .section h3 {{ font-size: 13px; color: #94a3b8; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 6px; }}
  .section p {{ font-size: 15px; line-height: 1.5; }}
  .vessel-block {{ margin-bottom: 10px; }}
  .vessel-name {{ font-weight: 600; color: #f1f5f9; }}
  .vessel-detail {{ font-size: 13px; color: #94a3b8; }}
  .separation {{ font-size: 15px; }}
  .sep-value {{ font-weight: 600; color: #38bdf8; }}
  .review-section {{ margin-top: 20px; padding-top: 16px; border-top: 1px solid #334155; }}
  .nav-row {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .nav-btn {{
    padding: 8px 20px; border-radius: 6px; font-size: 14px; font-weight: 600;
    text-decoration: none; display: inline-block; text-align: center; flex: 1;
    background: #334155; color: #e2e8f0; cursor: pointer;
  }}
  .nav-btn:hover {{ background: #475569; }}
  .nav-btn.disabled {{ background: #1e293b; color: #475569; cursor: default; }}
  .review-buttons {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
  .review-buttons button {{
    padding: 8px 16px; border: none; border-radius: 6px;
    font-size: 14px; font-weight: 600; cursor: pointer; color: #fff;
  }}
  .btn-confirm {{ background: #16a34a; }}
  .btn-confirm:hover {{ background: #15803d; }}
  .btn-reject {{ background: #dc2626; }}
  .btn-reject:hover {{ background: #b91c1c; }}
  .btn-unclear {{ background: #d97706; }}
  .btn-unclear:hover {{ background: #b45309; }}
  .btn-duplicate {{ background: #7c3aed; }}
  .btn-duplicate:hover {{ background: #6d28d9; }}
  .btn-pending {{ background: #6b7280; }}
  .btn-pending:hover {{ background: #4b5563; }}
  textarea {{
    width: 100%; height: 70px; padding: 8px; border-radius: 6px;
    border: 1px solid #475569; background: #0f172a; color: #e2e8f0;
    font-family: inherit; font-size: 14px; resize: vertical;
  }}
  .save-row {{ margin-top: 8px; display: flex; gap: 8px; align-items: center; }}
  .save-row button {{
    padding: 6px 16px; border-radius: 6px; border: none;
    background: #2563eb; color: #fff; font-size: 14px; cursor: pointer;
  }}
  .save-row button:hover {{ background: #1d4ed8; }}
  .save-msg {{ font-size: 13px; color: #22c55e; display: none; }}
  .osm-link {{ display: inline-block; margin-top: 8px; font-size: 13px; color: #38bdf8; }}
  .coords-box {{
    background: #0f172a; padding: 10px 14px; border-radius: 8px;
    font-size: 18px; font-weight: 700; color: #f8fafc; text-align: center;
    letter-spacing: 0.02em; margin-bottom: 4px;
  }}
  .date-box {{
    background: #0f172a; padding: 10px 14px; border-radius: 8px;
    font-size: 18px; font-weight: 700; color: #fbbf24; text-align: center;
    margin-bottom: 4px;
  }}
</style>
</head>
<body>

<div class="top-bar">
  <div class="progress">{reviewed} / {total} reviewed &mdash; #{queue_index + 1} in queue</div>
</div>

<div class="main">
  <div class="left">
    <div class="event-header">
      <span class="badge" style="background:{tier_color}">{tier}</span>
      <span class="badge" style="background:{status_color}">{status_label}</span>
    </div>
    <div class="event-id">{c['event_id']}</div>

    <div class="section" style="margin-top:14px">
      <h3>Date</h3>
      <div class="date-box">{format_date(c['start_time'])}</div>
    </div>

    <div class="section">
      <h3>Location</h3>
      <div class="coords-box">{lat:.4f}, {lon:.4f}</div>
      <a class="osm-link" href="{osm_url}" target="_blank">Open in OpenStreetMap &nearr;</a>
    </div>

    <div class="section">
      <h3>Time Window</h3>
      <p>{format_time(c['start_time'])} &mdash; {format_time(c['end_time'])} ({format_duration(c['duration_min'])})</p>
    </div>

    <div class="section">
      <h3>Vessel 1</h3>
      <div class="vessel-block">
        <div class="vessel-name">{c['vessel_1_name'] or 'Unknown'}</div>
        <div class="vessel-detail">{c['vessel_1_ship_type'] or '?'} &middot; MMSI {c['vessel_1_mmsi']} &middot; {v1_speed}</div>
      </div>
      <h3>Vessel 2</h3>
      <div class="vessel-block">
        <div class="vessel-name">{c['vessel_2_name'] or 'Unknown'}</div>
        <div class="vessel-detail">{c['vessel_2_ship_type'] or '?'} &middot; MMSI {c['vessel_2_mmsi']} &middot; {v2_speed}</div>
      </div>
    </div>

    <div class="section">
      <h3>Separation</h3>
      <p class="separation">
        Min <span class="sep-value">{c['min_separation_m']:.0f} m</span> &middot;
        Median <span class="sep-value">{c['median_separation_m']:.0f} m</span> &middot;
        Max <span class="sep-value">{c['max_separation_m']:.0f} m</span>
      </p>
    </div>

    <div class="section">
      <h3>AIS Observations</h3>
      <p>{c['observation_count']} points</p>
    </div>

    <div class="review-section">
      <h3 style="font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px">Review</h3>
      <div class="review-buttons">
        <button class="btn-confirm" onclick="submitReview('confirmed_sts')">Confirm STS</button>
        <button class="btn-reject" onclick="submitReview('rejected_no_sts')">Reject</button>
        <button class="btn-unclear" onclick="submitReview('unclear_imagery')">Unclear</button>
        <button class="btn-duplicate" onclick="submitReview('duplicate_event')">Duplicate</button>
        <button class="btn-pending" onclick="submitReview('pending_imagery')">Reset</button>
      </div>
      <div class="nav-row">
        {'<a class="nav-btn" href="/event/' + prev_id + '">&#8592; Back</a>' if prev_id else '<span class="nav-btn disabled">&#8592; Back</span>'}
        {'<a class="nav-btn" href="/event/' + next_id + '">Next &#8594;</a>' if next_id else '<span class="nav-btn disabled">Next &#8594;</span>'}
      </div>
      <textarea id="notes" placeholder="Review notes...">{review_notes}</textarea>
      <div class="save-row">
        <button onclick="saveNotes()">Save Notes</button>
        <span class="save-msg" id="saveMsg">Saved</span>
      </div>
    </div>
  </div>

  <div class="right">
    <div id="map"></div>
  </div>
</div>

<script>
  var map = L.map('map').setView([{lat}, {lon}], 13);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap'
  }}).addTo(map);
  L.marker([{lat}, {lon}]).addTo(map)
    .bindPopup('STS candidate center').openPopup();

  function submitReview(status) {{
    var notes = document.getElementById('notes').value;
    fetch('/api/review', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        event_id: '{c["event_id"]}',
        review_status: status,
        review_notes: notes
      }})
    }}).then(r => r.json()).then(data => {{
      if (data.ok) {{
        // Update badge
        var badges = document.querySelectorAll('.badge');
        if (badges.length > 1) {{
          var colors = {{
            'pending_imagery': '#6b7280',
            'confirmed_sts': '#16a34a',
            'rejected_no_sts': '#dc2626',
            'unclear_imagery': '#d97706',
            'duplicate_event': '#7c3aed'
          }};
          badges[1].style.background = colors[status] || '#6b7280';
          badges[1].textContent = status.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        }}
        // Update progress
        var prog = document.querySelector('.progress');
        if (data.reviewed !== undefined && data.total !== undefined) {{
          prog.textContent = data.reviewed + ' / ' + data.total + ' reviewed — #' + {queue_index + 1} + ' in queue';
        }}
        // Auto-advance to next candidate
        var nextUrl = {'"/event/' + next_id + '"' if next_id else 'null'};
        if (nextUrl) {{
          window.location.href = nextUrl;
        }} else {{
          showSaveMsg();
        }}
      }}
    }});
  }}

  function saveNotes() {{
    var notes = document.getElementById('notes').value;
    fetch('/api/save_notes', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{
        event_id: '{c["event_id"]}',
        review_notes: notes
      }})
    }}).then(r => r.json()).then(data => {{
      if (data.ok) showSaveMsg();
    }});
  }}

  function showSaveMsg() {{
    var el = document.getElementById('saveMsg');
    el.style.display = 'inline';
    setTimeout(function() {{ el.style.display = 'none'; }}, 2000);
  }}
</script>
</body>
</html>"""


def render_index(queue, conn):
    reviewed, total = get_progress(conn)
    rows_html = []
    for i, eid in enumerate(queue):
        c = get_candidate(conn, eid)
        if not c:
            continue
        status = c.get("current_review_status", "pending_imagery")
        status_colors = {
            "pending_imagery": "#6b7280",
            "confirmed_sts": "#16a34a",
            "rejected_no_sts": "#dc2626",
            "unclear_imagery": "#d97706",
            "duplicate_event": "#7c3aed",
        }
        sc = status_colors.get(status, "#6b7280")
        tier = c["confidence_tier"]
        tc = "#dc2626" if tier == "HIGH" else "#d97706"
        rows_html.append(f"""
        <tr onclick="window.location='/event/{eid}'" style="cursor:pointer">
          <td style="padding:8px;font-family:monospace;font-size:12px">{i+1}</td>
          <td style="padding:8px"><span class="badge" style="background:{tc}">{tier}</span></td>
          <td style="padding:8px"><span class="badge" style="background:{sc}">{status.replace('_',' ').title()}</span></td>
          <td style="padding:8px">{c['vessel_1_name'] or '?'}</td>
          <td style="padding:8px">{c['vessel_2_name'] or '?'}</td>
          <td style="padding:8px">{format_date(c['start_time'])}</td>
          <td style="padding:8px">{format_duration(c['duration_min'])}</td>
          <td style="padding:8px">{c['median_separation_m']:.0f} m</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIS Candidate Review Queue</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 30px; }}
  h1 {{ margin-bottom: 8px; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 15px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 10px 8px; color: #94a3b8; font-size: 12px;
       text-transform: uppercase; letter-spacing: 0.05em;
       border-bottom: 1px solid #334155; }}
  tr:hover {{ background: #1e293b; }}
  td {{ border-bottom: 1px solid #1e293b; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 12px; font-weight: 600; color: #fff; }}
</style>
</head>
<body>
<h1>AIS Candidate Review Queue</h1>
<p class="subtitle">{reviewed} / {total} reviewed</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>Tier</th><th>Status</th><th>Vessel 1</th><th>Vessel 2</th>
      <th>Date</th><th>Duration</th><th>Med. Sep.</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        conn = get_db()
        ensure_reviews_table(conn)
        queue = get_review_queue(conn)

        if path == "" or path == "/":
            html = render_index(queue, conn)
            self._respond_html(html)
        elif path.startswith("/event/"):
            event_id = path[len("/event/"):]
            candidate = get_candidate(conn, event_id)
            if not candidate:
                self._respond_html("<h1>Event not found</h1>", status=404)
                return
            try:
                idx = queue.index(event_id)
            except ValueError:
                idx = 0
            reviewed, total = get_progress(conn)
            html = render_page(candidate, queue, idx, reviewed, total)
            self._respond_html(html)
        else:
            self._respond_html("<h1>Not found</h1>", status=404)

        conn.close()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        conn = get_db()
        ensure_reviews_table(conn)

        if path == "/api/review":
            event_id = body.get("event_id", "")
            status = body.get("review_status", "pending_imagery")
            notes = body.get("review_notes", "")
            save_review(conn, event_id, status, notes)
            reviewed, total = get_progress(conn)
            self._respond_json({"ok": True, "reviewed": reviewed, "total": total})
        elif path == "/api/save_notes":
            event_id = body.get("event_id", "")
            notes = body.get("review_notes", "")
            conn.execute(
                "UPDATE ais_sts_reviews SET review_notes = ? WHERE event_id = ?",
                (notes, event_id),
            )
            conn.commit()
            self._respond_json({"ok": True})
        else:
            self._respond_json({"error": "not found"}, status=404)

        conn.close()

    def _respond_html(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _respond_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Quieter logging: just method and path
        pass


def main():
    parser = argparse.ArgumentParser(description="STS candidate review UI")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve on (default 8765)")
    args = parser.parse_args()

    # Ensure reviews table exists on startup
    conn = get_db()
    ensure_reviews_table(conn)
    reviewed, total = get_progress(conn)
    conn.close()

    print(f"AIS Candidate Reviewer")
    print(f"  Database: {DB_PATH}")
    print(f"  Progress: {reviewed} / {total} reviewed")
    print(f"  Open http://localhost:{args.port}")
    print()

    server = HTTPServer(("0.0.0.0", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
