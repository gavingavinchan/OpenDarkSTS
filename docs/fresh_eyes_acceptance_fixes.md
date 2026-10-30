# First-visitor GUI acceptance fixes

The first-visitor workflow now provides model information, selection of the bundled example without path entry, plain explanations of candidate results, and a workspace-specific server stop command. Detection and AIS matching algorithms are unchanged. README.md was intentionally left unchanged.

Branch: `t3/improve-first-run-gui`, based on `release` at `c1b0dfe`.

Implementation commits:

- `5453345` — Safe Linux/Windows server shutdown and readable command help.
- `608d973` — Bundled example selection, current-model scores, and imagery credential guidance.
- `399cd32` — Plain result explanations, consistent training-page names, responsive layouts, and browser acceptance coverage.

## Findings

| Finding | Outcome and verification |
|---|---|
| 1 — Model information | Fixed. Intelligence HQ and Step 7 display YOLO11s, the recorded all-class and per-class instance counts, precision, recall, mAP50, and mAP50–95. Definitions and dataset limitations accompany the table. The model card opens in a browser tab. Improved models retain their own recorded metrics and are not labelled with the bundled model's card or scores. Verified in Chromium and backend tests. |
| 2 — Promotion documentation requirement [DONE] | Existing automatic recreation-record and default-restoration fixes confirmed by the model lifecycle regression tests. The remaining YOLO26n placeholder is corrected to `yolo11s_myrun_2026-01-01_best.pt`; verified in Chromium. No tracked weight was added or changed. |
| 3 — Manual example paths | Fixed. Detect and Data provide **Use the bundled example**. The shared backend selects the installed imagery folder, example AIS CSV, 2020-01-24, matching bounds, confidence 0.25, and a 1,000 m radius. The imagery-folder input is visible. Saved input paths are absolute; newly entered relative GUI paths resolve from the workspace. CLI parity: `project --use-example`; API parity: `POST /api/operations/example`. The GUI example was run without entering any paths. |
| 4 — Stop command | Fixed. `opendarksts stop` stops the server recorded for the selected workspace, using existing process identity and Windows process-tree helpers. Invalid identities and PID reuse are guarded against. Linux shutdown and an unreachable health endpoint were verified; Windows dispatch and process-tree handling are covered by tests. Independent detached jobs continue. |
| 5 — Stale saved AIS count [DONE] | Existing fix confirmed by the frontend regression test and the selected example's visible `1 AIS file` count. |
| 6 — Defaults and dates | Fixed. New projects start on 2020-01-24 with bounds containing the example candidate. Dates use explicit YYYY-MM-DD text fields with format validation and server-side date checks. The folder guidance explains that the drawn area controls downloads; existing tiles are selected by date and are not cropped by the drawn area. |
| 7 — Result interpretation | Fixed. Each candidate shows detector confidence and a sentence explaining its AIS classification. AIS vessel types have names; missing measurements show **Not reported**, without an attached unit, while genuine zero values remain visible. Candidate headings and verdict names use readable labels. Existing descriptive run names are retained. The explanation beside **Sent to Improve** states why the detection verdict is locked and directs corrections to Improve. Actual review, transfer, and CSV download were verified in Chromium. |
| 8 — Live Map | Fixed. A green unfilled ring marks the candidate; smaller solid orange dots mark AIS positions. The legend explains the distinction. Candidate popups identify the candidate and explain its AIS reading. Candidate-ring and AIS-dot counts, appearance, and popup interaction were verified. |
| 9 — Job warnings and grammar | Fixed. Known harmless GeoTIFF metadata-tag warnings are omitted from the displayed/API log tail. Raw diagnostic logs remain unchanged, and other warnings and errors remain visible. Empty successful logs say that the job completed successfully. Singular/plural forms are corrected for candidate detections, completed scans, and the import image count. The actual example log format is covered by a regression test. |
| 10 — Missing Copernicus credentials | Fixed. Imagery download is disabled when either credential is absent. Data explains this before launch, provides Linux bash and Windows PowerShell commands, and requires a server restart from the same terminal. The GUI job backend rejects launch before creating a job when credentials are missing. Credentials themselves are not returned by the dashboard or stored in project settings. The credential-free bundled example remains available. |
| 11 — Jargon | Addressed in the reported areas. AIS labels have direct definitions without duplicate missing-record categories. Detection thresholds, image preparation, album details/references, original colour, training batches, augmentation controls, model selection, and box location error use plain explanations. Scientific metric names and configuration keys remain as secondary detail where needed for reproducibility. |
| 12 — Names and training defaults | Fixed. Home navigation uses **Intelligence HQ**. The seven training-page headings, navigation labels, and tab titles agree: **Import imagery & albums**, **Annotation**, **Reviewed bounding-box overview**, **Dataset Selection**, **Training Setup**, **Train detector**, and **Evaluate & Promote**. Detect uses **Area & dates**, **Data & AIS Manager**, **Scan Images**, and **STS Events**. Training Setup explains that its editable starting settings are 150 epochs / patience 30, whereas the paper model used 300 / 60. Defaults themselves were preserved. |
| 13 — Phone layout | Fixed. Results display labelled, stacked candidate fields at narrow widths, including Image verdict. All seven training steps remain visible in a two-column navigation grid. Import row numbers remain on one line; import and dataset tables scroll within their containers, and filenames no longer wrap one character at a time. All pages passed viewport overflow checks; screenshots were visually inspected. |
| 14 — Cosmetic consistency | Low-risk changes completed: STS Events uses the operational heading and four-step bar, sans-serif title, common content width, and closely grouped verdict buttons. Result cards do not stretch to match the entire evidence panel. Import selection glyphs are replaced with simple plus signs. A broader redesign was not undertaken. |
| 15 — CLI help | Fixed. Top-level commands have descriptions; the `==SUPPRESS==` artefact is removed. `run --ais-dir` explicitly documents support for a folder or one daily CSV. Existing argument names remain compatible. |

## Browser verification

A fresh workspace was initialized under a temporary directory inside this worktree whose path contains spaces:

```text
workspace/temp/new visitor fclmjdwf
```

The application was initialized using the supplied documented checkpoint and served as a detached process on port 8796. Browser packages were installed in the separate throwaway environment `workspace/browser env`; the project Python environment was not changed. The browser was Chromium 156.0.8078.4.

The four newcomer tasks were completed:

1. **Run the example:** Detect → Use the bundled example → Data → Continue to Scan Images → Start detection → Open STS Events & Live Map. No input paths were typed. The result was exactly one candidate, `AIS_PRESENT`, at **28.65640007 N, 94.32280446 W**, with **two** nearby AIS vessels and confidence **0.731051** (displayed as **0.73**). A reviewed crop was sent to Improve, and an actual reviewed CSV was downloaded. The export includes absolute source paths.
2. **Identify the model and its performance:** YOLO11s, its per-class table, the metric definitions, and the browser model card were checked on Intelligence HQ and Step 7.
3. **Inspect the pages:** 14 pages, including all seven training steps and the model card, were checked at **1366×900**, **1920×1080**, and **390×844**. All 42 page/viewport combinations had no document overflow. There were **zero browser console errors and zero uncaught page errors**. The final Import and Training Setup wording was additionally rechecked at all three sizes.
4. **Stop the server:** `opendarksts --workspace "…/new visitor fclmjdwf" stop` returned `stopped: true`. The recorded process was no longer alive and the health endpoint was unreachable. The final verification server is stopped.

The reproducible browser script is `tests/browser/fresh_eyes_acceptance.py`. The original tester screenshots are preserved as the **before** evidence, rather than being presented as newly captured baseline screenshots. Final screenshots, page text, reviewed CSV, browser results, the final text recheck, and shutdown evidence are saved under:

```text
/home/gavin/OpenDarkSTS/workspace/gui_test/fresh_eyes_fixes/
  before/
  after/
```

## Tests

The required command passes:

```bash
OPENDARKSTS_TEST_WEIGHTS=/home/gavin/OpenDarkSTS/workspace/model_weights/opendarksts_yolo11s_v4_best.pt /home/gavin/OpenDarkSTS/.venv/bin/python -m pytest -q
```

Result: **411 passed**. Coverage includes safe process identity checks, a real detached Linux process stop, Windows abstraction dispatch, example selection from a different working directory, absolute path persistence, model metadata, credential preflight, narrow warning filtering, and result presentation. The existing optional real-tile test now honours `OPENDARKSTS_TEST_WEIGHTS` and uses the bundled tile when the historical local imagery directory is absent, allowing it to run in a fresh checkout.

## Facts for the README rewrite

- `opendarksts init` installs the default model and example inputs; `init --weights PATH` installs the documented local checkpoint.
- `opendarksts serve --port PORT` starts the detached local GUI; the default port remains 8766.
- `opendarksts stop` stops this workspace's recorded GUI server. Independent background jobs continue; `opendarksts jobs` inspects them.
- For a custom workspace, use `opendarksts --workspace "path with spaces" init`, `serve`, and `stop`, or set `OPENDARKSTS_WORKSPACE` consistently. The workspace option precedes the subcommand.
- The GUI example flow is **Detect → Use the bundled example → Data & AIS Manager → Continue to Scan Images → Start detection → STS Events**. The same example button is also available on Data.
- The example button selects imagery, the matching AIS CSV, **2020-01-24**, matching Gulf of Mexico bounds, confidence **0.25**, and AIS radius **1,000 m**. It requires neither downloads nor Copernicus credentials.
- CLI equivalent: `opendarksts project --use-example`, followed by `opendarksts project --start-job run`. `project --check-ready` checks the saved inputs.
- The example yields one **AIS_PRESENT** candidate near **28.6564 N, 94.3228 W**, confidence approximately **0.73**, with two nearby AIS vessels. This does not identify either visible hull or establish a transfer or legality.
- GUI input paths are stored absolutely. Relative paths entered in Detect/Data resolve from the workspace. The imagery-folder field is visible on Area & dates.
- Dates are entered as **YYYY-MM-DD**. New-project date and area defaults match the example; existing saved settings are preserved.
- The selected area controls imagery downloads. Existing imagery is selected by filename date and is not cropped by the drawn area.
- **Intelligence HQ** and **Evaluate & Promote** display the current model and recorded metrics. The bundled model is **YOLO11s**, with a browser model-card link and the all-class/per-class table from its metadata. The table concerns the 21-image held-out test dataset, not whole-pipeline dark-status accuracy or independent-scene generalization.
- Page names are those listed in finding 12 above. **STS Events** contains the results, Live Map, human verdicts, and Export Report links. **Jobs & scan history** contains progress and job output.
- Results show detector confidence and a plain AIS interpretation. Green rings represent satellite candidates; orange dots represent same-day AIS positions.
- After a reviewed crop enters Improve, its detection verdict is locked; use **Open in Improve** to correct the training review.
- Copernicus imagery downloads require **both** `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`. Set them in bash on Linux or PowerShell on Windows, then stop and restart the server from that terminal for the same workspace. Missing credentials disable the GUI download before a job is launched. Data contains the platform-specific instructions.
- `run --ais-dir` accepts an AIS folder **or one daily CSV**, despite the legacy option name.
- Training Setup retains editable starting defaults of **150 epochs / patience 30**; reproducing the paper model requires **300 / 60**, together with the other model-card parameters.
- Lightweight GUI interaction diagnostics remain local in the selected workspace's reports directory. No external analytics endpoint was introduced.

## Limits and intentionally retained behaviour

A native Windows host and physical phones were not available. Windows process identity and tree termination retain the existing psutil abstraction and are tested through its Windows branch; layouts were tested through Chromium viewport emulation. Training or model promotion was not rerun from scratch; the pre-existing promotion/default-restoration regression coverage passed. Copernicus downloads were not attempted without credentials, and a live authenticated download was outside this task. AIS matching rules, confidence thresholds, training defaults, stored review/provenance identifiers, and the post-transfer verdict lock were retained. Technical names remain in advanced configuration and provenance details where removing them would impair reproducibility.

No files in `/home/gavin/Downloads/AIS data` were modified. No AI co-author was added. No changes were pushed.
