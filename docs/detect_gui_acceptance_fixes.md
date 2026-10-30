# Detect acceptance regressions

The public Detect workflow has regression coverage for findings F01–F09 from the
release-candidate browser acceptance report. README.md and finding F10 are outside
this change.

- Download jobs support cooperative cancellation through Jobs, the HTTP endpoint
  `POST /api/operations/jobs/<job_id>/cancel`, and
  `opendarksts jobs <job_id> --cancel`. Cancellation removes temporary transfer,
  extraction, and filtering files while retaining completed cached inputs. Transfer
  logs show received bytes, provider-reported totals when available, percentage,
  rate, and elapsed time. A pending network request can take up to its network
  timeout to acknowledge cancellation. Only downloads affecting the selected
  source and dates prevent detection.
- Explicit AIS CSV selection uses timestamp contents to establish applicable
  dates. Readiness reports missing or unidentifiable dates before submission.
  The foreground CLI performs the same explicit-file preflight before inference.
- Foreground CLI jobs retain status, logs, output location, and summaries in the
  workspace job history. Home therefore includes foreground detections.
- Jobs and Results exports use the same current review-aware export helper.
- Jobs and Results show input names, acquisition dates, confidence thresholds,
  AIS radii, and UTC timestamps including seconds.
- Results run selection is stored in the URL and restored on refresh and browser
  history navigation. AIS inventory, unsaved radius summaries, and transfer
  feedback update immediately after the corresponding action.

## Automated checks

Backend regressions are in `tests/test_acquisition.py`,
`tests/test_operations_jobs.py`, `tests/test_operational_pipeline.py`, and
`tests/test_candidate_results.py`. `tests/test_detect_frontend_state.py` executes
the actual JavaScript controllers through Node to test URL history, inventory,
unsaved radius summaries, and correction of transfer errors. These checks run
with the normal pytest suite; frontend checks are skipped if Node is unavailable.

## Real-browser acceptance

`tests/browser/detect_acceptance.py` drives Chromium through Playwright. Install
Playwright and Chromium in a disposable environment separate from the application
environment. Prepare a fresh workspace using the documented release checkpoint:

```bash
opendarksts --workspace <fresh-workspace> init --weights <checkpoint>
opendarksts --workspace <fresh-workspace> run --images <fresh-workspace>/inputs/example --ais-dir <fresh-workspace>/inputs/example/AIS_2020_01_24.csv --date 2020-01-24
opendarksts --workspace <fresh-workspace> serve --port 8794
<browser-python> tests/browser/detect_acceptance.py --workspace <fresh-workspace> --evidence <evidence-directory>
```

The test verifies the original reproduction steps, downloads both reviewed export
formats from both GUI routes, checks the bundled candidate near 28.6564 N,
94.3228 W, and starts a real NOAA transfer for 2020-01-24. It verifies changing
progress, the selected-source dependency guard, a successful independent local
scan during the transfer, cancellation, and partial-file cleanup. It saves
screenshots, page text, downloads, and a machine-readable outcome. The optional
`--noaa-only` argument repeats the network-dependent section in an already tested
workspace. Stop the server after verification.
