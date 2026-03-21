# AIS Denmark STS Candidate Criteria

This note defines the first-pass AIS-only rules for generating potential STS events from the Denmark dataset. These are candidate events for later satellite review, not confirmed STS cases.

## Dataset Notes

- Source file inspected: `/home/gavin/Downloads/AIS data/AIS Denmark/AIS DK 2026/aisdk-2026-01-01.zip`
- The zip contains one CSV: `aisdk-2026-01-01.csv`
- Column names differ from the earlier U.S. AIS files and will need normalization before scanning:
  - `# Timestamp`
  - `Type of mobile`
  - `MMSI`
  - `Latitude`
  - `Longitude`
  - `Navigational status`
  - `SOG`
  - `COG`
  - `Heading`
  - `IMO`
  - `Callsign`
  - `Name`
  - `Ship type`
  - `Cargo type`
  - `Width`
  - `Length`
  - `Draught`
  - `Destination`
  - `ETA`

## Why The Filter Needs To Be Strict

Quick sampling of the Denmark file shows a lot of non-STS clutter:

- First 200,000 rows: mostly `Class A`, but also many `Base Station`, `Class B`, and `AtoN` records
- The most common ship types in that sample are `Fishing`, `Cargo`, `Passenger`, and `Tanker`
- A loose close-pair search on the first 10 minutes produced thousands of pairs within 250 m when small craft and port traffic were included
- The same search dropped to a single pair within 250 m when limited to slow `Class A` cargo/tanker vessels

That means the scanner should favor precision over recall on the first pass, then expand only after manual review of results.

## Recommended Candidate Definition

Treat an AIS pair as a potential STS event when all of the following are true.

### 1. Valid Vessel Records

- `Type of mobile` is `Class A`
- `MMSI`, timestamp, latitude, and longitude are present and parse cleanly
- The two vessels have different MMSIs
- Exclude `Base Station` and `AtoN` entirely

### 2. Target Vessel Heuristic

Use these vessel filters to avoid flooding the review queue with ferries, fishing boats, and harbor traffic:

- Keep pairs where at least one vessel is `Tanker`
- Allow `Cargo`-`Cargo` or `Cargo`-`Tanker` pairs
- Allow `Undefined` or `Other` only as a fallback when the vessel looks commercially sized, for example:
  - `Length >= 60 m`, or
  - `IMO` is present
- Exclude `Fishing`, `Passenger`, `Pleasure`, `SAR`, `Pilot`, and small-service traffic by default
- Exclude pairs where both vessels are `Tug`

### 3. Temporal Match

- Compare vessel positions only when timestamps are within `10 minutes`
- Build continuous pair tracks using `5-minute` bins or nearest-in-time matches
- Require at least `3` matched observations spanning at least `20 minutes`

This removes one-off crossings and brief port maneuvers.

### 4. Spatial Proximity

- Default pairing radius: `150 m`
- Keep a lower-confidence overflow tier up to `250 m`
- Use the event centroid as the mean lat/lon across the matched observations
- Store min, median, and max pair separation for later review

`150 m` is a good first-pass default because side-by-side commercial vessels should usually be well under that threshold, while Denmark port clutter rises quickly when the radius is looser.

### 5. Speed Behavior

- Mean SOG for each vessel should be `<= 2.0 kn`
- Maximum SOG during the event should be `<= 5.0 kn`
- Mean absolute SOG difference between the two vessels should be `<= 1.5 kn`

Practical rule: reject pairs where both vessels are moving fast. True STS transfer at normal transit speed is unlikely.

### 6. Heading Behavior

- If both vessels have mean SOG above `1.0 kn`, require heading or course difference `<= 20 deg`
- If one or both vessels are nearly stationary, heading agreement is optional and should be treated as advisory only

This keeps drifting or slow-steaming side-by-side vessels while filtering opposite-direction crossings.

### 7. Navigational Status

- Reject when both vessels are `Moored`
- Allow `Under way using engine`
- Allow `At anchor`, but mark those events with a review flag
- Allow `Restricted maneuverability` as a lower-confidence case
- Ignore `Unknown value` as a blocker, but lower the confidence tier if too many fields are unknown

## Confidence Tiers

Assign each event a tier so we can review the strongest candidates first.

### High

- Separation `<= 100 m`
- Duration `>= 30 minutes`
- Mean SOG for both vessels `<= 1.0 kn`
- One vessel is `Tanker` and the other is `Tanker` or `Cargo`
- Heading difference `<= 15 deg` when moving

### Medium

- Separation `<= 150 m`
- Duration `>= 20 minutes`
- Mean SOG for both vessels `<= 2.0 kn`
- Vessel types pass the target-vessel heuristic

### Low

- Separation `<= 250 m`
- Duration `>= 15 minutes`
- Speed and type rules are close but not ideal
- Keep only for manual review, not model training without visual confirmation

## Database Shape For Candidate Events

Use a candidate-events table plus an optional evidence table.

### `ais_sts_candidates`

- `event_id`
- `source_dataset`
- `source_file`
- `criteria_version`
- `vessel_1_mmsi`
- `vessel_2_mmsi`
- `vessel_1_name`
- `vessel_2_name`
- `vessel_1_ship_type`
- `vessel_2_ship_type`
- `start_time`
- `end_time`
- `duration_min`
- `center_lat`
- `center_lon`
- `min_separation_m`
- `median_separation_m`
- `max_separation_m`
- `vessel_1_mean_sog_kn`
- `vessel_2_mean_sog_kn`
- `mean_heading_delta_deg`
- `confidence_tier`
- `review_status` default `pending_imagery`
- `review_notes`
- `created_at`

### `ais_sts_candidate_points`

- `event_id`
- `timestamp`
- `vessel_1_lat`
- `vessel_1_lon`
- `vessel_2_lat`
- `vessel_2_lon`
- `separation_m`
- `vessel_1_sog_kn`
- `vessel_2_sog_kn`
- `vessel_1_cog_deg`
- `vessel_2_cog_deg`

The events table is what the imagery pipeline consumes. The points table preserves the AIS evidence used to create the event.

## Recommended First Implementation

1. Normalize the Denmark CSV columns into the canonical field names already used by the repo
2. Build pair tracks from `Class A` records only
3. Apply the `Medium` tier rules as the default insert threshold
4. Insert `High` and `Medium` candidates into the database
5. Optionally store `Low` candidates in a separate review queue table if the volume is manageable

### 8. Distance From Shore

- Reject candidates whose center point is within `12 nautical miles` (22.2 km) of the nearest coastline
- This matches the standard territorial waters boundary
- Use a coastline dataset such as GSHHS or Natural Earth coastline vectors for the distance check
- The threshold should be configurable via a parameter in the scan script, defaulting to `12 nm`

The first Denmark scan produced 37 candidates, all within territorial waters near Frederikshavn/Skagerrak and southern Denmark. Manual review confirmed these were normal port operations (bunkering, lightering), not suspicious STS. This filter eliminates that entire class of false positives at scan time.

## Known Open Questions

- The timestamp timezone semantics still need verification
- If the first batch misses obvious events, relax vessel-type rules before relaxing the distance rule
