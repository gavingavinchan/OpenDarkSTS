# Release review packet

Local implementation follows `release_plan.md` through Phase 7 preparation.
The working branch is `release`. No remote branch, pull request, tag or release
has been changed. The experimental Results interface remains hidden.

## Prepared for review

- Shared Detect/Improve server, background job records and CLI; validated model,
  imagery and AIS inputs; run provenance and human-readable same-day AIS wording.
- Streaming NOAA/CDSE acquisition, completion/hash checks, geographic cache
  separation, acquisition metadata, and radius coverage checks for filtered AIS.
- Experimental candidate map/table/crops, explicit human verdicts, exports and
  transfer into completed training reviews with observation groups preserved.
- Windows process path, portable launch tests, packaging and checksum-verified
  initialization, a genuine satellite/AIS example, and Linux/Windows CI workflows.
- Local server checks for loopback Host/Origin, request types/sizes, uploaded image
  paths/content, export/deletion boundaries and cache containment.

## Concrete artifacts

The build and UI evidence are under `workspace/reports/release_review/`: package
archives, build log, browser screenshots, and an isolated installed application
workspace. Its automated example review is a test action, not scientific ground truth.

`workspace/reports/release_candidate/sts_clean_v1_annotations_review.zip` contains
562 label files and a split manifest with image/label hashes, groups and source
categories. It contains no imagery or weights, strips local machine paths from
public-facing records, and is explicitly marked **not cleared for publication**.
The adjacent JSON records its hash. Recreate with
`python -m scripts.prepare_annotation_release --dataset workspace/datasets/derived/sts_clean_v1 --output workspace/reports/release_candidate/NEW_NAME.zip`.

The model card and recreation instructions are packaged in `opendarksts/assets/`.
The proposed licence text is `docs/AGPL-3.0.proposed.txt`; it has not been applied as
an agreed project licence. Data terms and unresolved rights are in `licensing.md`.

## Final local validation

- Requested `/home/gavin/anaconda3/bin/python3 -m pytest -q`: **310 passed, 3 skipped**.
- Full `.venv/bin/python -m pytest -q`: **313 passed**, including actual YOLO inference
  over the bundled satellite image and region-complete AIS sample.
- Wheel and source distribution built successfully. A wheel installation under an
  isolated directory initialized its workspace, launched a background inference job
  and served all eight checked pages. Browser checks found no JavaScript errors;
  Home, Area and Results fit a 390-pixel viewport. Repeated transfers reuse completed
  reviews and reject conflicting human labels.
- `git diff --check` passed. The protected AIS folder was only read;
  `scratchpad.md` was not modified by this work.

The isolated review server is available at `http://127.0.0.1:8773`; `/results` is
available there through the experimental switch. It uses sample data and test
verdicts. Its PID and log path are in the isolated workspace's
`reports/server/service.json`.

## Independent GUI acceptance

A separate agent with fresh context exercised the current source through a real
Chromium browser in an isolated workspace, including genuine detection, image
upload and review, dataset export, one-epoch CPU training, evaluation, promotion
and training cancellation. Test images and model-assisted acceptance labels do
not constitute an independent scientific evaluation.

The acceptance work identified and corrected failed-download input selection,
relative checkpoint resolution, original-filename search, mixed evaluation/import
job cleanup, custom-workspace crop URLs, and narrow-screen layouts. Regression tests cover the behavioral
fixes. The detailed feature checklist, limitations and browser evidence are in
`council/notes/gui-acceptance-report.md` and
`workspace/reports/gui_acceptance/20260918_independent/`. Updated wheel and source
archives are under `workspace/reports/gui_acceptance/dist/`; the earlier installed
review server on port 8773 predates these fixes.

## Remaining external checks

1. Contributor approval of the proposed code licence and redistribution rights for
   model, annotations and source data; verify the Roboflow metadata against upstream
   rights. Approve only the parts of the prepared bundle that may be published.
2. Actual Windows CI and a manual Detect/training launch/stop run; the Linux session
   cannot certify Windows execution. See `windows_validation.md`.
3. Authenticated CDSE acquisition against the live service. Network tests are mocked;
   NOAA's current endpoint/schema were checked, and the example uses historical NOAA
   records. No CDSE credentials were requested or copied into source or logs.
4. Review the hidden Results interface using `experimental_results_review.md`.
   Keep it out of navigation and public usage instructions until approved.
5. Publish the documented weight asset, run the real-example workflow with that URL,
   then merge/tag/publish and close the superseded PRs. These are publication actions
   and remain deferred. `init --weights` supports local review before publication.

The SNAME paper remains a separate historical draft. Its contaminated-split headline
metrics, six-step description, missing figures and bibliography still need revision;
those are not represented as publication-ready by this software work.
