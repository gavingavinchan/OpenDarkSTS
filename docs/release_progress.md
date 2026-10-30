# Release implementation progress

The plan of record remains `docs/release_plan.md`. The `release` branch is ready
for coauthor review through a draft pull request into `main`. Public release
remains deferred. The experimental Results page remains hidden pending review.

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
- Phase 7: local server boundaries hardened; public README, model/data notices,
  proposed licence text, annotation review bundle and Linux/Windows CI prepared.
  Final validation: 297 passed, 3 skipped with the requested interpreter; 300 passed
  in the inference environment. The installed wheel passed eight page checks,
  repeated review transfer and mobile layout checks, with no JavaScript errors.
  Local implementation is ready for review; publishing and external validations
  remain pending. See `release_review.md` for the concrete review packet.

Independent GUI acceptance subsequently exercised detection, import/review, dataset
export, real CPU training, cancellation, evaluation and promotion. Six defects were
fixed and retested. Latest local validation: 310 passed, 3 skipped with the requested
interpreter; 313 passed in the model-enabled environment. The portable coverage
summary is in `gui_acceptance_report.md`.

Final external checks will include contributor licence agreement, data redistribution
rights, Windows execution, review of the experimental Results page, and publication.
These do not prevent preparing and testing the implementation locally.

The SNAME 2026 release default is now the paper's fine-tuned YOLO11s v4 checkpoint,
`opendarksts_yolo11s_v4_best.pt`. Initialization, operational inference, training
auto detection and preprocessing share the manifest-backed default. Initialization
upgrades a recorded automatic installation of the former default while preserving
explicit promotions. Model documentation records the paper's exact results and
the 21-image, partially scene-overlapping v4 test split. The clean-split YOLO26n
remains an alternative.

Local validation with `/home/gavin/OpenDarkSTS/.venv/bin/python -m pytest -q`:
359 passed. The built wheel was installed and exercised from an isolated working
directory outside the repository root, with spaces in its path.
`opendarksts init --weights` accepted the exact source YOLO11s checkpoint; the
bundled example produced one STS candidate, two AIS matches, CSV/JSON outputs
and one annotated preview. Relative checkpoint paths used by the Linux/Windows
example workflow were also verified. Actual Windows execution remains pending.
No release, upload or push was performed, and no checkpoint was committed.
