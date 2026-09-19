# Release implementation progress

The plan of record remains `docs/release_plan.md`. Work proceeds on `release`; review
is deferred until implementation is ready. Nothing is pushed or published from this
session. The experimental Results page remains hidden pending review.

- Phase 0: local hardening complete; 250 tests passed, 2 skipped. Remote branch
  retirement remains pending.
- Phase 1: model class validation, unambiguous AIS selection, explicit missing/invalid
  AIS outcomes, north-up EPSG:4326 validation, path-qualified detection IDs, shared
  current-model record, and run provenance implemented. The class count remains a
  same-day radius comparison, not a statement about either visible hull's identity.
  Validation: 260 passed, 2 skipped.
- Phase 2: streaming acquisition helpers and CLI implemented; cache identities include
  bounds and source checksums, and imagery retains acquisition metadata. Current NOAA
  endpoint/schema migration is documented in `docs/acquisition.md`. Validation: 266
  passed, 2 skipped; current NOAA endpoint and header checked without downloading a day.
- Phases 3–7: in progress.

Final external checks will include contributor licence agreement, data redistribution
rights, Windows execution, review of the experimental Results page, and publication.
These do not prevent preparing and testing the implementation locally.
