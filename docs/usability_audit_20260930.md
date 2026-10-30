# Application usability and reliability audit

The audit evaluated the release-branch application on Linux, with particular emphasis on users without prior machine-learning experience. The application provides a functional route from imagery import and review to dataset export, training, and evaluation. Detection and same-day AIS comparison also completed with the bundled example. Several misleading displays and two data-handling defects were identified and corrected. The remaining priorities concern guidance, data selection, and inspection of detection results.

## User-facing changes

- Home distinguishes imagery selected for detection from images and albums used for training. An empty detection folder no longer conceals the presence of the training collection.
- Run displays the number of image filenames that match the selected dates, explains launch blockers, and provides links to change imagery or AIS settings. Missing or empty AIS sources produce an explicit warning.
- Jobs reports the numbers of candidates with zero, one, or at least two nearby AIS vessels. Candidates with missing or invalid AIS receive an explicit unknown-presence message.
- The image import page reports numerical totals rather than unevaluated additions, identifies the number of images matching the filter, and uses “Image list”, “Thumbnails”, and “Refresh existing previews”.
- Image review uses “Single vessel” and “STS pair” in the principal controls and bounding-box labels. Long internal image identifiers wrap within their panels. The empty review queue links to image import instead of requiring a preprocessing command.
- Detect provides map-loading feedback, a failure message, and retry. Presets, coordinates, and dates remain usable when map tiles or the viewer are unavailable. Image review loads its viewer and drawing tools locally, including the required images and licences in the distributable package.
- Older exported datasets report their actual image counts from the split folders.
- Dataset export requires the current selection and settings to be saved. Subsequent changes disable export and continuation until the selection is saved and exported again. Exporting an unrelated historical selection cannot enable continuation for the current form. A seed of zero is preserved in dataset splitting and training configuration rather than replaced by the default value.

The training interface also preserves album controls when changing image views, displays the selected image after inspecting an album, and retains search focus while results update. Filter totals and selection checkboxes reflect the displayed candidates. Loading failures in the bounding-box overview provide a retry instruction, and wrapped navigation no longer compresses page titles. Evidence is recorded under `workspace/reports/usability_audit_20260930/training_gui/`.

## Backend corrections

Dataset re-export now builds the replacement in a temporary sibling directory before changing the existing export. Validation failures preserve the previous dataset. Publication is serialized, and ordinary publication exceptions restore the previous managed entries. Unrelated dataset notes remain present. This does not provide crash-atomic publication or isolation from concurrent readers: interruption between component renames remains a known limitation.

Malformed persisted job records are rejected with structured validation errors and skipped during job listing. A malformed historical record therefore no longer prevents the valid jobs, detection dashboard, or readiness checks from loading.

The application CLI now forwards leading training options, including the documented `opendarksts training --help` and `opendarksts training --pretty ...` forms. `opendarksts project --check-ready` exposes the same saved-project readiness check used by the GUI. Its successful response can contain `ready: false`; callers must inspect the blockers.

## Design follow-up

Following direct user feedback, all three agents were explicitly instructed to apply the frontend-design skill to the visual revision. Detect now presents a coherent sequence for area and dates, data acquisition, and execution, with a clear primary action on each page. Coordinates, existing-folder paths, and detection tuning occupy secondary expandable sections. The interface uses a restrained navy and teal palette, a plain background, and locally served IBM Plex Sans fonts.

All seven training screens use the same shared header and navigation layout. Browser measurements at 1280 × 800 and 1366 × 768 found identical positions and widths for every stage link, with no horizontal page overflow. Manual-review progress occupies a stable toolbar beneath the image viewer, and queue statistics remain in the sidebar. Bounding-box cards have equal image-region heights and centered labels; detection metadata is collapsed by default.

Training Setup initially exposes only the exported dataset and an optional run name. The current documented detector is selected automatically. The recommended training settings are summarized, while model alternatives, augmentation, device overrides, commands, and saved configurations are available through Advanced. The Advanced control remains visible within the checked desktop viewports. The training screen selects the dataset associated with a prepared configuration and places process metadata and the training log in expandable sections.

The shared backend and CLI now support automatic hardware selection: CUDA device 0 is preferred, followed by an available Apple GPU, with CPU used otherwise. Explicit overrides are preserved, and saved configurations record the concrete resolved device for reproducibility. The training interpreter on this Linux computer reported an NVIDIA GeForce RTX 5070 Ti with PyTorch 2.10.0+cu128; automatic selection resolved to device 0. Hardware-selection tests cover GPU, Apple GPU, CPU, explicit CPU, and interpreter selection. This revision did not launch a long GPU training run.

The combined automated suite after the design and hardware changes passed 352 tests. Wheel and source builds succeeded, and the wheel was checked for all four font weights and the font licence. The restarted application was verified with the reviewed 37-image dataset, current detector, automatic GPU selection, and collapsed Advanced controls. Training Setup was opened on the Linux display. Evidence for this revision is under `workspace/reports/design_revision_20260930/`, with additional agent records under the original audit directories.

## Existing data inventory

The original training collection contains 2,580 review-queue images and five albums. There are 474 completed image reviews and 2,106 incomplete reviews, comprising 2,091 unreviewed and 15 partially reviewed images. The image-cache inspection found no missing cached images.

Seven exported datasets are discoverable, with 4,012 image files in their split folders. No manifest-listed exported file was missing during the backend inventory. The five derived datasets do not declare a dataset-selection manifest; the two reviewed exports reference existing selections. These records should not be interpreted as seven equally recommended training datasets.

The main Detect configuration initially selected empty imagery and AIS folders. This was a selection issue, rather than loss of the training collection. The original datasets were not reorganized during the audit, and `/home/gavin/Downloads/AIS data` was not modified.

## Workflow verification

The initial automated suite passed 313 tests; the final combined suite passed 344 tests. The wheel and source package built successfully, including the locally served viewer assets. Focused backend validation passed 179 relevant tests; map and reviewer validation passed 26 tests. Training-interface verification passed 18 focused browser checks across the seven screens and 18 relevant automated tests. Final operational browser verification passed ten checks after the application service was restarted with the completed changes.

The isolated GUI acceptance workspace is `workspace/reports/usability_audit_20260930/test_workspace`. Nine existing image copies were uploaded through the GUI with automatic detection, without import failures. Eight reviews were populated from existing reference annotations through the backend helpers, and the remaining image was reviewed through the GUI. This procedure tests application behavior; it does not establish independent scientific annotations.

The GUI selected and exported the nine images with whole observation groups assigned to the training, validation, and test splits. A one-epoch CPU training run completed through the GUI; the CLI reported the same completed state, metrics, log, and checkpoint paths. Test-split evaluation was launched through the GUI, and the evaluation report was written through the CLI after the collaborative browser host disconnected. The completed report was subsequently verified in a headless browser. The smoke checkpoints have an adjacent recreation record. Their accuracy scores are not evidence of an improved model, and they were not installed as the operational model.

Two operational GUI runs produced two candidates each. The run with the selected bundled AIS CSV reported `AIS_PRESENT` for both candidates. The run with missing AIS reported `AIS_MISSING`, with unknown vessel counts. CSV and JSON downloads were checked. Headless browser verification confirmed the revised Home, Run, Jobs, and evaluation displays without page JavaScript errors.

Map failure, missing viewer assets, and retry recovery were tested explicitly. Image review was also exercised with external requests blocked. The user subsequently clarified that the Linux-display error was a localhost connection failure, rather than providing an OpenStreetMap tile error. The application service was verified at `http://127.0.0.1:8766`. Provider-specific failure on the user's display was not established.

Live CDSE authentication, fresh imagery downloads, full NOAA day downloads, Windows execution, and long-duration training were not validated in this audit. Network-acquisition tests use mocked services. The same-day AIS comparison retains its documented lack of a time window and vessel-track association.

## Priorities for the next usability iteration

1. **Extend guidance across the complete training route.** Training Setup now provides recommended defaults and secondary advanced controls. Further onboarding should explain the review and dataset stages and indicate the next action consistently across the workflow. The roles of training, validation, and testing should remain clear before users change split percentages.
2. **Identify the recommended dataset and its provenance.** A unified dataset overview should distinguish imported albums, reviewed images, exported training datasets, and historical experiments. Display source, review completion, split method, file health, and whether an export still matches its selection. A default recommendation should be an explicit designation rather than a filename inference.
3. **Make detection results directly inspectable.** The current ordinary Jobs flow provides CSV/JSON downloads and summaries. Users should be able to inspect each annotated candidate image and nearby AIS evidence within the application, with a clear review decision and a route back to Improve. The existing experimental results view requires further acceptance work before inclusion in the normal navigation.
4. **Reduce path entry and improve first use.** Provide a bundled-example action and appropriate input selection controls. Explain which computer owns a selected path when the GUI is viewed remotely. Display the active workspace and a stable local URL so an obsolete port is distinguishable from an empty dataset.
5. **Complete terminology and layout consistency.** Replace remaining internal labels and dense parameter descriptions with engineering-oriented explanations. Further review of the manual-review footer, keyboard navigation, and smaller displays remains warranted.

## Evidence

Detailed logs, JSON checks, screenshots, test fixtures, checkpoint recreation records, and reproduction scripts are under `workspace/reports/usability_audit_20260930/`. The map report is `map_frontend/audit.md`; backend findings and inventory are `backend/findings.md` and `backend/dataset_inventory.json`; operational browser results are `operations_browser_checks.json`.
