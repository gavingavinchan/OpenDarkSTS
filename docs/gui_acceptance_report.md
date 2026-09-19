# GUI acceptance summary

The local Detect and Improve workflows were exercised in Chromium on Linux on
18 September 2026. Six defects were corrected and successfully retested. This
summary is included in the review branch so reviewers do not require access to
the local browser evidence directory.

## Exercised workflows

| Workflow | Observed result |
|---|---|
| Home, navigation, area and dates | Pages loaded; presets, map selection, coordinates, dates and input selection worked; reversed dates were rejected. |
| Acquisition failure and recovery | Missing imagery credentials produced an actionable failed job without replacing the selected input folder. |
| Detection and Jobs | Real inference on the bundled satellite/AIS example produced two candidates; logs and CSV/JSON downloads worked. |
| Experimental Results | Crops, maps, AIS context, decisions, CSV/JSON/GeoJSON exports and transfer into Improve worked. Positive, background-negative and single-ship transfers were exercised. Contradictory completed labels were protected. |
| Import and albums | PNG, JPEG and TIFF uploads, folder import, album creation/editing, selection, duplicate Replace/Duplicate handling and force regeneration worked. Regeneration preserved completed reviews. |
| Removal | Album-only removal, workspace removal, cancellation and confirmation worked; source images remained unchanged. |
| Manual review | Box creation, class changes, Neither, deletion and geometry editing worked; saved geometry persisted across reloads. Nine image reviews were completed. |
| Reviewed overview | Filters, sorting, crop refresh and editor links worked. All 26 displayed crop images loaded in a custom workspace. Conflicting completed reviews were flagged and excluded from the ordinary grid. |
| Dataset creation | Original-filename filtering, group-aware split preview, save and YOLO export worked. Nine observations were split 5/2/2 without groups crossing splits. Background-only export produced an empty label file. Image-mode splitting displayed its group crossing. |
| Training setup and execution | Configuration save/restore, augmentation controls, readiness and command preview worked. A genuine one-epoch CPU run completed. A second run was cancelled and its process terminated. |
| Evaluation and promotion | Test evaluation, report generation/reload, documentation checks, promotion and explicit overwrite worked. Subsequent detection recorded the promoted checkpoint's exact checksum. |
| Cross-workflow reuse | Evaluation followed by another upload succeeded in the same server process. |
| Narrow-screen layout | Home, Detect, Results, Import, datasets, setup, training and evaluation fit a 390-pixel viewport; wide tables scrolled internally. |

## Corrections verified

- Failed or empty downloads preserve selected inputs (`163205d`).
- Training workers resolve local checkpoint paths correctly (`b03fe07`).
- Dataset selection displays and searches original filenames (`84aa2ce`).
- Import and training pages fit narrow viewports (`f68ae6d`).
- Evaluation jobs no longer break subsequent uploads or evaluations (`0b203da`).
- Review crop URLs work in a custom workspace (`043c1ff`).

The requested local test interpreter completed **310 passed, 3 skipped**. The
model-enabled environment completed **313 passed**, including actual inference.
These results are local validation; GitHub CI results are reported separately.

## Limits and reviewer priorities

- Authenticated CDSE acquisition and full-day NOAA downloads were not exercised
  through the GUI. Windows, GPU training, long training, simultaneous launches,
  server-loss recovery and all failure-boundary combinations remain unverified
  by this browser session.
- A dedicated GUI action to reconcile competing completed reviews does not
  exist. Conflict detection and display were tested; a complete GUI resolution
  workflow cannot be certified. Background-negative ratio/cap settings are
  available through backend manifests, not a dedicated GUI selector.
- Some keyboard, filter and validation combinations remain untested. Successful
  checks apply to the stated cases, not every possible input or interaction.
- Training used reused genuine imagery with model-assisted acceptance labels.
  Its metrics are software-test results, not independent scientific performance
  estimates. The isolated application was restored to the documented default
  model after promotion testing.
- The Results interface remains experimental. Review its scientific wording
  and workflow using [the Results checklist](experimental_results_review.md).

For installation and model prerequisites, follow the [README](../README.md).
The default checkpoint asset is not published yet: inference requires an
authorized local copy supplied to `opendarksts init --weights PATH`. The
[model card](../opendarksts/assets/model_card.md) identifies the required checksum.
`init --no-model` permits workspace and interface inspection without that asset.

Detailed browser scripts, screenshots, raw assertions and their reconciliations
remain local under `workspace/reports/gui_acceptance/20260918_independent/`;
they are not included in the repository or required to read this summary.
