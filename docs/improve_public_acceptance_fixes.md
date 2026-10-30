# Public Improve loop acceptance verification

Verification date: 30 October 2026. Base: `release` at `feaf3ae`.

The Improve workflow completed through the GUI in Chromium, using a fresh workspace initialized with the published YOLO11s checkpoint. Playwright was installed in a separate disposable environment under this worktree's `workspace/gui_test/improve_fixes/`; the project environment was used only for application execution and pytest.

## Findings and verification

| Finding | Change | Browser verification |
|---|---|---|
| F1 | Promotion preserves an adjacent workspace recreation record before selecting the weight. GUI and CLI share the helper. Recorded command, configuration, starting checkpoint hash, dataset hashes, source revision, environment, available input snapshots, and reproducibility gaps are retained. Legacy recipes remain supported. | A one-epoch CPU run was promoted through Evaluation without editing or acknowledging a source document. Home and Detect displayed the new weight; its scan record contained the promoted weight hash. |
| F2 | Evaluation recommendations and the current operational promotion are recorded separately. Report generation never changes the selected model. | Before promotion, the report displayed the recommended run and `Promoted: none`. After actual promotion, a newly generated report identified the promoted run. |
| F3 | Import states that byte-identical copies are recognized and re-encoded copies are new images. Image identity is unchanged. | The note was visible on Import. Existing identity tests remained unchanged and passed. |
| F4 | Album-naming and upload-conflict dialogs have bounded height and internal scrolling. | Both New empty album and the New Album toolbar entry were exercised at 390 × 600 and 780 × 437. Save and Cancel remained reachable; empty albums were saved. |
| F5 | Results provides an Open in Improve link for transferred crops. Successful transfer invalidates the Annotation queue cache. The JavaScript change is confined to candidate rendering. | A reviewed example crop was transferred and its link opened the exact completed image in Annotation. The review queue showed 21/21 completed images. |
| F6 | Duplicate launches identify the active setup, refresh readiness and runs, and link to the selected progress panel. Readiness also reports the active conflict. | Two tabs loaded the same ready setup before launch. The second tab reported that it was already running, removed the ready message, and offered View its progress. Only one trainer ran. |
| F7 | Evaluation shows the current model and restores the checksummed bundled default. CLI: `training restore-default-model`. The default filename is reserved during promotion, including case variants for Windows. | GUI restoration changed Home and Detect back to YOLO11s. The bundled example then produced one AIS_PRESENT candidate at 28.65640007 N, 94.32280446 W, with two nearby AIS vessels. The CLI restoration command also succeeded. |
| F8 | Escape dismisses keyboard help before other shortcuts or input-focus handling. | Keyboard help opened with `?` and closed with Escape. |

## Workflow and evidence

Twenty images were copied from the released annotation images into the local test input folder. Import generated proposals; the GUI review controls completed all 20 reviews. The dataset contained 16 training, two validation, and two test images, with 20 scene groups and zero groups spanning splits. Training used YOLO11s for one epoch on CPU, image size 128, batch size 2, workers 0, and the no-augmentation preset. Held-out evaluation, report generation, promotion, operational scanning, and default restoration were performed through the GUI. These short-run metrics assess workflow execution, not model quality; the promoted smoke-test model produced no STS candidates in its example scan.

Before/after screenshots, browser evidence, scripts, test logs, training metadata, the promotion recreation record, and scan records are saved in `/home/gavin/OpenDarkSTS/workspace/gui_test/improve_fixes/`. Original acceptance screenshots are prefixed `before-`; new screenshots are prefixed `after-` or numbered by workflow stage. The verification also detected and corrected a stale Annotation queue after crop transfer.

The supplied checkpoint was passed through `OPENDARKSTS_TEST_WEIGHTS` for the full pytest suite: **379 passed, 1 skipped**. The final command was `/home/gavin/OpenDarkSTS/.venv/bin/python -m pytest -q`. Regression coverage includes automatic CLI promotion, provenance refusal and checksum checks, report semantics, restoration, Windows-style paths, dialog bounds, transfer links and cache invalidation, duplicate readiness, and Escape handling. The local server and test browsers were stopped after verification. Windows execution was not available; existing portability tests and Windows-path regressions passed. No application source edit was needed to promote the test model. No files in the prohibited AIS source directory were accessed or changed. No push was performed.
