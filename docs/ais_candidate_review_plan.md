# AIS Candidate Review Plan

This is the AIS candidate reviewer — a manual review tool for STS candidates generated from AIS data. It provides a fast review loop where each candidate event opens at its location in OpenStreetMap, then the reviewer checks Sentinel-2 imagery in Copernicus Browser to confirm or reject STS.

A separate visual reviewer (based on satellite imagery detection) will be built later.

## Goal

Turn the AIS candidate database into a practical review queue.

For each candidate event, the reviewer should be able to:

1. Open the event location in OpenStreetMap
2. See the key AIS context for that event
3. Go to Copernicus Browser and inspect Sentinel-2 imagery for the event date/location
4. Mark the event as `confirmed`, `rejected`, or `unclear`
5. Save review notes back into the database

## Review Unit

The review unit should be one candidate event row from `ais_sts_candidates`.

Each review item should display at least:

- `event_id`
- `confidence_tier`
- `start_time`
- `end_time`
- `duration_min`
- `center_lat`
- `center_lon`
- `vessel_1_name`, `vessel_1_ship_type`, `vessel_1_mmsi`
- `vessel_2_name`, `vessel_2_ship_type`, `vessel_2_mmsi`
- `min_separation_m`, `median_separation_m`, `max_separation_m`
- `review_notes`
- OpenStreetMap URL

## Map View

The review tool should embed an OpenStreetMap view directly in the UI instead of only printing a link.

Recommended map behavior:

- center on `center_lat` / `center_lon`
- show a marker for the candidate event center
- start at a practical zoom such as `12` to `14`
- allow zoom and pan for context
- still provide an external OpenStreetMap link as a fallback

Example fallback URL:

`https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=13/{lat}/{lon}`

## Manual Confirmation Source

The actual confirmation step is done in Copernicus Browser.

Manual review should use:

- event date from `start_time`
- event center from `center_lat` / `center_lon`
- Sentinel-2 imagery around the event time window

The reviewer checks whether the imagery shows:

- two ships side-by-side in a plausible transfer configuration
- no visible STS
- unclear imagery because of cloud, resolution, timing mismatch, or coverage issues

## Review Outcomes

Use these statuses in the database:

- `pending_imagery` - not reviewed yet
- `confirmed_sts` - manually confirmed in imagery
- `rejected_no_sts` - imagery does not support STS
- `unclear_imagery` - imagery unavailable or inconclusive
- `duplicate_event` - duplicate of another review item

## Recommended Database Additions

The current table already has `review_status` and `review_notes`. Store all review metadata in a separate `ais_sts_reviews` table keyed by `event_id`. This keeps the base candidates table clean and allows re-review without mutating original data.

### `ais_sts_reviews` table

- `event_id` (foreign key to `ais_sts_candidates`)
- `review_status` (`pending_imagery`, `confirmed_sts`, `rejected_no_sts`, `unclear_imagery`, `duplicate_event`) — this is the final manual verdict
- `review_notes`
- `reviewed_at`
- `reviewer`
- `duplicate_of_event_id`

### Fields for future automated imagery pipeline

These are not needed now but reserved for when satellite imagery is downloaded automatically:

- `imagery_source` default `copernicus_browser`
- `imagery_checked_date`
- `imagery_result`
- `confirmed_event_lat`
- `confirmed_event_lon`

## UI Requirements

The first review interface should be a small local web app, not a terminal-only flow.

### Main Layout

The page should have two main panels:

- left panel: human-readable event details
- right panel: embedded OpenStreetMap view

### Human-Readable Event Card

Show the AIS record in plain language, not raw database-style columns.

Recommended presentation:

- event ID and confidence tier
- date and time window
- duration in minutes or hours
- latitude and longitude
- vessel 1: name, type, MMSI, average speed
- vessel 2: name, type, MMSI, average speed
- separation summary: min / median / max meters
- review flags such as anchor or unknown-status warnings

Example wording:

- `Tanker AMAK SWAN (219355000) and tanker TANKSKAR (265550210)`
- `Observed near 57.6907, 11.8896 from 08:19 to 23:59 UTC`
- `Median separation 63 m; both vessels nearly stationary`

### Review Controls

The UI should include explicit action buttons:

- `Confirm STS`
- `Reject`
- `Unclear`
- `Duplicate`

Also include:

- `Next`
- `Back`
- optional free-text notes field
- optional link button: `Open in OSM`

### Navigation Behavior

- `Next` moves to the next item in the current review queue
- `Back` returns to the previous item without losing saved review data
- when a review action is clicked, save immediately to the database
- after save, optionally auto-advance to the next event

### Progress Indicator

Show a simple progress counter such as `14 / 87 reviewed` so the reviewer can gauge remaining effort.

### Copernicus Browser Step

The UI does not need to embed Copernicus Browser yet.

The event card should present the date and location prominently so the reviewer can quickly punch them into Copernicus Browser manually.

For now, the reviewer will:

1. inspect the event details and map in the local review UI
2. manually open Copernicus Browser separately, using the date and coordinates shown in the event card
3. check Sentinel-2 imagery for the same date/location
4. return to the review UI and click the result button

## Reviewer Workflow

The first usable workflow should be:

1. Load the next unreviewed event in the web UI
2. Read the human-readable AIS summary
3. Inspect the embedded OpenStreetMap view
4. Manually check Copernicus Browser for Sentinel-2 imagery
5. Click one result: `Confirm STS`, `Reject`, `Unclear`, or `Duplicate`
6. Optionally enter a note
7. Move with `Next` or `Back`

## Territorial Water Exclusion

Candidates in territorial waters (within 12 nautical miles of the coast) should be excluded from the review queue. STS transfers in territorial waters are overwhelmingly normal port operations — bunkering, lightering, crew transfers — not the dark STS events this project targets.

The initial Denmark AIS scan produced 37 candidates, all within territorial waters near Frederikshavn/Skagerrak and southern Denmark. None warranted further review.

### Implementation

The AIS scan script should compute distance-to-shore for each candidate pair's center point and reject events closer than 12 nm. This requires a coastline dataset such as:

- GSHHS (Global Self-consistent Hierarchical High-resolution Geography) shoreline data
- Natural Earth coastline vectors

The distance check should run during candidate generation, not during review. Events that fail the check should not be inserted into `ais_sts_candidates` at all.

### Configurable threshold

Use 12 nm (22.2 km) as the default exclusion distance, matching the standard territorial waters boundary. Allow override via a parameter in the scan script for testing.

## Recommended Processing Order

To make review efficient, sort the queue by:

1. `confidence_tier` (`HIGH` before `MEDIUM`)
2. longer `duration_min`
3. smaller `median_separation_m`

Also add a duplicate-reduction pass before review.

## Duplicate Reduction

Many AIS candidates are likely repeated views of the same real-world situation.

Before presenting the queue, merge or mark likely duplicates when all are true:

- same vessel pair
- same day
- centers within `500 m`
- time gap between events less than `2 hours`

This should reduce review fatigue a lot.

## First Implementation Target

Build a small local web review app, for example `scripts/review_ais_candidates_ui.py` serving a simple HTML UI, that:

- reads from `workspace/databases/ais_denmark_candidates.sqlite`
- selects one pending event at a time
- renders a human-readable event card
- embeds an OpenStreetMap map centered on the event
- provides `Confirm STS`, `Reject`, `Unclear`, `Duplicate`, `Next`, and `Back` buttons
- saves review status and notes back into the database

Recommended implementation shape:

- backend: lightweight Python web server
- frontend: simple server-rendered HTML with a little JavaScript
- map: Leaflet with OpenStreetMap tiles

That gives us a useful human-in-the-loop workflow now, before automating satellite downloads.

## Later Automation

Once the review loop is working, the next step is automatic imagery support:

1. derive the best Sentinel-2 search window from the event date/time
2. query/download imagery automatically
3. save imagery paths and metadata in the database
4. present imagery directly in the review tool

At that point, Copernicus Browser becomes optional instead of required.

## Recommended Next Step

Implement the local review web UI first. That is the smallest useful layer on top of the AIS candidate database and matches the review workflow needed before automated imagery download is added.
