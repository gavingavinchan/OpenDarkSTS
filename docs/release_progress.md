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
- Phase 3: shared Home, Area & dates, Data, Run, and Jobs pages implemented on the
  existing server. Acquisition/detection jobs persist status and logs; training jobs
  are included. CLI uses the same project and job helpers. Validation: 271 passed,
  2 skipped; six pages checked in Chromium with no JavaScript errors.
- Phase 4: experimental table/map/crop review, explicit human verdicts, reviewed
  CSV/JSON/GeoJSON exports, and idempotent transfer to Improve implemented. Parent
  observation groups are preserved. HTTP remains hidden behind the feature switch.
  Validation: 277 passed, 2 skipped. Review checklist: `experimental_results_review.md`.
- Phase 5: Windows process identity and descendant termination implemented; training
  launch locking added; launch fixtures use Python on both platforms; active environment
  supplies YOLO and test evaluation. Validation: 280 passed, 2 skipped on Linux, including
  mocked Windows process checks. Actual Windows run remains pending (see
  `windows_validation.md`).
- Phase 6: installable package, `opendarksts init/serve` and shared CLI families,
  checksum-verified default weight setup, packaged model documentation and a small
  Sentinel/NOAA example implemented. The installed wheel was exercised outside the
  repository root. Validation: 283 passed, 3 skipped with the requested interpreter;
  286 passed with all inference dependencies, including a genuine example run.
  Default weight publication remains pending; local initialization accepts its exact
  existing checkpoint. Area-filtered AIS now requires coverage of the whole radius.
- Phase 7: in progress.

Final external checks will include contributor licence agreement, data redistribution
rights, Windows execution, review of the experimental Results page, and publication.
These do not prevent preparing and testing the implementation locally.
