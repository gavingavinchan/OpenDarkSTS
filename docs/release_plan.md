# OpenDarkSTS — Release Plan

Written 2026-09-18. This is the top-level plan for turning OpenDarkSTS into one coherent
program and releasing it as open source. It supersedes `docs/grand_plan_v3.md` as the
plan of record; v3 remains the accurate description of what is built today.

Inputs to this plan: a whole-repo survey, an audit of PR #4, an audit of the May/September
split of `ais-first-pipeline`, and an independent review by a second model. The working
notes are in `council/notes/` (not tracked).

---

## 1. Who it is for

The main user wants to **find** candidate dark ship-to-ship transfers. They do not train
models and may not know how. They should be able to go from "an area and some dates" to
"a reviewed list of candidates" without touching a terminal more than once.

The shipped model is good but not perfect. So the training tool ships too, as the way to
improve the model, and it has to stay usable by someone who is not an ML engineer.

Supported platforms: **Linux and Windows**. macOS is a possible later addition.

Name: **OpenDarkSTS**.

---

## 2. The finished program

One program, one local web app on one port, one command-line tool, one workspace folder,
one current model.

```
[Home]   [Detect: Area & dates · Data · Run · Results]   [Improve: steps 1-7]   [Jobs]
```

### Detect (the front door)

| Page | What the user does | Built from |
|---|---|---|
| Area & dates | Draw a box on a map or pick a US preset; pick dates. Or point at a folder of tiles they already have. | New page; region presets from Madhulika's dashboard |
| Data | Download Sentinel-2 tiles and the matching NOAA AIS day files, as background jobs with a visible log. | Sohail's `download_sentinel2.py` (PR #3); Madhulika's AIS Manager (PR #4) |
| Run | Detection -> georeference -> AIS matching over the chosen tiles. | Today's `scripts/pipeline.py` |
| Results | Table + map of what the run found. Per detection: the crop, nearby AIS vessels, a plainly worded AIS status, and buttons *real STS / not STS / unclear*. Export CSV / JSON / GeoJSON. Reviewed detections can be sent to Improve as training data. | New page, reusing the step 2 review screen; design reference is Madhulika's dashboard |

**The Results page is experimental until Gavin has checked it.** It is built behind a
switch, is not linked from the navigation, and is not mentioned in user-facing docs until
he confirms it. Until then the official output of a run stays `dark_sts.csv`,
`dark_sts.json` and the preview PNGs.

### Improve (the existing training tool, steps 1-7)

Import -> visual review -> reviewed overview -> dataset selection -> training setup ->
training run -> evaluation and promotion. Unchanged in shape. Promoting a weight makes it
the current model for the whole program.

### Home and Jobs

Home shows the current model with its clean-split metrics, recent runs, and anything
waiting for review. Jobs lists running and finished downloads, runs, and trainings with
their logs.

### Underneath

- **One server.** The existing standard-library server in `scripts/training_pipeline_ui.py`
  gains the Detect section. No second web app, no second port. (PR #4's dashboard is a
  separate web app built with the Flask library; it is used as a design reference and a
  source of parts, not merged. See section 4.)
- **One set of functions** called by both the GUI and the CLI, so they cannot disagree.
  CLI families: `acquire imagery`, `acquire ais`, `run`, `results`, plus today's training
  commands. JSON output and exit codes as today.
- **One workspace folder** holds every download and every derived file. Outside folders
  (such as `/home/gavin/Downloads/AIS data`) are only ever read.
- **One current model.** Every feature that runs the detector uses the promoted weight by
  default, including step 1 auto-detect (which today still points at an old v2 weight,
  `training_pipeline_lib.py:49`).
- **Every run records** the weight's checksum, the confidence threshold, the AIS radius
  and the input list, so a result can always be traced.

---

## 3. Order of work

Core functions and GUI first. Install and first-run setup later, but before release.

### Phase 0 — One branch

`release` is the working branch (created 2026-09-18 from local `ais-first-pipeline`,
`3d92e2a`). The name `ais-first-pipeline` no longer describes the product.

1. Record the May-June side of `origin/ais-first-pipeline` (17 commits, tip `763d9e0`) as
   merged while keeping the September tree: `git merge -s ours`. No force-push needed.
2. Bring across from the May side, as files: `docs/sname_2026_paper.tex`,
   `docs/training_pipeline_step_5_plan.md`, the backend bug-hunt test plan (into `docs/`),
   and the writing-tone guidance blocks for `AGENTS.md` / `CLAUDE.md`.
3. Port the May hardening onto the September code, one commit each, with its tests
   (from `17648db`, `06c6bad`): atomic `write_json`; run-directory containment check;
   pid <= 0 and corrupt `process.json` handling; log path containment; duplicate
   class-name readiness warnings; CLI `ValueError` -> exit code 2.
4. Push `release`. Retire `ais-first-pipeline` once `release` is on the remote.

Done when: one branch holds everything, all tests pass, nothing of value exists only on
the old remote branch.

### Phase 1 — Core correctness

Small fixes that stop the pipeline from giving confidently wrong answers.

- **Refuse a model that has no `ships` class.** `scripts/detect.py:27` currently falls back
  to class id 1, so a stock COCO checkpoint would report bicycles as STS pairs.
- **Refuse an AIS file with the wrong columns.** Today a wrong-schema CSV produces zero
  matches, which reads as `FULLY_DARK`. Report "AIS missing / invalid" as its own outcome.
- **Refuse imagery that is not north-up EPSG:4326** with a clear message, instead of
  treating its bounds as lon/lat.
- Make the AIS file choice explicit when more than one file matches a date.
- Detection ids that cannot collide when two folders contain the same filename.
- One current model everywhere (see section 2).
- Run record with weight checksum and settings.
- Wording: statuses stay `FULLY_DARK` / `PARTIALLY_DARK` / `AIS_PRESENT` in data files,
  but every screen and document describes them as "no / one / two or more AIS vessels
  within R metres that day". No "confirmed", no "likely legitimate".

### Phase 2 — Getting the data

- **AIS download** from Madhulika's AIS Manager (`app.py` ~593-820 on
  `origin/darkfleet-clean`): NOAA MarineCadastre, zip for 2024 and earlier, zst for 2025
  onward, retries, output named `AIS_YYYY_MM_DD.csv` as the pipeline expects. Fixes
  needed: an area-filtered file must not be reused for a different area (record the
  bounds, keep the full-day file distinct); stream instead of loading whole files into
  memory; only treat a file as present once it is complete and valid.
- **Imagery download** from Sohail's `scripts/download_sentinel2.py` (`origin/Images`).
  Check first: the second review reports that it requests a whole day per catalog scene,
  so `sceneN` files may be duplicates or mosaics. Keep the scene's identity and
  acquisition time in a sidecar file. Credentials from environment variables or an
  untracked config file, never from constants or logged.
- Both as functions + CLI commands + tests with the network mocked.

### Phase 3 — Detect in the GUI

Area & dates, Data, Run, Jobs pages on the one server, following the existing template
and navigation conventions in `scripts/templates/`. Long tasks run as managed background
jobs with on-disk records, using the same launch-locking approach as training runs.

### Phase 4 — Results page (experimental)

Built as described in section 2, hidden until Gavin signs off. Includes the
"send to Improve" link so a reviewed detection becomes training data without being
reviewed twice. An AIS outcome never becomes a label by itself; only the human's verdict
on the image does.

### Phase 5 — Windows

- Training run management uses `/proc` and `os.killpg`; give it a Windows path.
- No hardcoded `/home/gavin` paths (`scripts/visual_review_lib.py` has one).
- Run the test suite and one full Detect run on Windows. Madhulika works on Windows and is
  the natural tester.

### Phase 6 — Install and first run

- `pyproject.toml`, one supported environment, an `opendarksts` command
  (`opendarksts serve`, `opendarksts init`, and the CLI families above).
- `init` creates the workspace and fetches the default weight from a GitHub release,
  verifying its checksum. Weights do not live in git.
- A small bundled example (one tile, a cut-down AIS file) so a newcomer sees a result in
  minutes without a CDSE account, and so CI can run a real end-to-end test.

### Phase 7 — Release

- **Licence.** AGPL-3.0 is the default proposal because the program imports Ultralytics
  (AGPL-3.0) and ships weights trained with it. Needs agreement from all contributors.
- README written for a stranger; model card for the shipped weight (what it was trained
  on, clean-split metrics, known limits); attribution for Sentinel-2 and NOAA data.
- Training data provenance: record the source and licence of the Roboflow dataset; publish
  the annotations and split manifests we are allowed to publish.
- Safety pass on the local server: listens on loopback by default, file access and
  deletion contained to the workspace, uploads validated.
- CI on Linux and Windows.
- Merge `release` to `main`, tag `v0.1`. Close PR #2 as superseded; close PR #3 and PR #4
  with a note that their work was carried into `release`.

### Alongside — the SNAME 2026 paper

`docs/sname_2026_paper.tex` describes the May tool: six steps, per-image splits, and a
headline mAP50 of 0.895 measured on a test split that shared scenes with training. Before
submission it needs seven steps, scene-safe splits, and results regenerated on the clean
split (current honest figure: 0.868 mAP50, `yolo26n`, `sts_clean_v1`). Its figures and
`references.bib` are not committed anywhere yet. This is tracked here but is not a
blocker for the software release.

---

## 4. Where each piece comes from

| Source | What it brings | How it enters |
|---|---|---|
| Gavin | Training tool steps 1-7, leak-free evaluation, operational pipeline, tests | The base (`release`) |
| Gavin (May-June) | Run-status hardening, paper draft, test plans | Phase 0 |
| Madhulika | NOAA AIS downloader; dashboard design (results table, map, export, region presets); `compare_yolo_models.py` portability fix | Phase 2, Phase 3-4 design reference |
| Sohail | Sentinel-2 CDSE downloader | Phase 2 |

We work as one team, so commits are not attributed per person; the table only says where
to find the source material. PR #4 is not merged as a branch, for these reasons: its `weights/best.pt` is
byte-identical to the stock Ultralytics `yolo11s.pt` COCO checkpoint (SHA-256
`85a76fe8...`), so it detects people and bicycles, and merging would keep the 19 MB file
in history for good; its `.gitignore` rewrite drops our ignores for `workspace/`, secrets
and `.venv/`; its `app.py` runs inference twice per scan, keeps a Windows-only base path,
and has routes that can delete or read files outside its folders. Madhulika should be told
about the weight file so she does not rely on it. Her own trained weight is **not needed**
for the release: the shipped model is our promoted clean-split `yolo26n`.

---

## 5. Not in the release

- Dot detector, false-colour / zoomed-out tier, non-US AIS: dropped for good.
- The AIS-first scanner and its Denmark review UI (`scan_ais_sts_candidates.py`,
  `review_ais_candidates_ui.py`): left out; 0 of 37 candidates confirmed and it depends on
  non-US AIS.
- The training notebook as a second training path.
- Time-window AIS matching, acquisition-time recovery, SAR, multi-user hosting, databases,
  any automatic judgement about legality.
- Stricter NMS / box-size cutoffs from PR #4 (`0020cda`) unless they prove out on the
  clean test split.

---

## 6. How the work gets done

Codex (Astra) does the implementation, phase by phase, working in the `release` working
tree. Fable plans each phase, reviews Astra's changes against this plan, runs the tests,
and commits. Occasionally the roles swap so each model's work gets a second pair of eyes.
Gavin checks anything user-facing before it becomes official.

---

## 7. Still to decide

| Question | Default if nobody objects |
|---|---|
| Bring back the May-only features (one config -> several runs, one per starting weight; device probing)? The paper's Step 5 text claims the first. | Defer; revise the paper text instead |
| Licence | AGPL-3.0, after asking Madhulika, Sohail and Prof. Koola |
| Can the Roboflow training images be published? | Publish annotations and manifests only until the licence is known |
| When does the Results page become official? | When Gavin has checked it |
