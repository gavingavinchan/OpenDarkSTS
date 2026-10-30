#!/usr/bin/env python3
"""End-to-end Playwright verification of the training pipeline GUI, steps 1-7.

Starts the training pipeline UI detached on a free port, drives every page with a
headless browser, clicks every control, asserts the DOM/API effect, saves a PNG per
state under workspace/reports/gui_verification/, and prints a pass/fail summary.

Reads run against the real workspace. Writes go to a scratch dataset selection,
scratch export, and a 1-epoch CPU training run, all removed again unless --keep is
passed. No GPU training is ever launched from here.

Usage:
    .venv/bin/python scripts/verify_training_gui.py [--keep] [--skip-training]
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOT_DIR = ROOT / "workspace" / "reports" / "gui_verification"
SERVER_LOG = ROOT / "workspace" / "reports" / "verify_training_gui_server.log"

SCRATCH_DATASET_NAME = "gui_verify_scratch"
SCRATCH_DATASET_VERSION = "v1"
SCRATCH_RUN_GROUP = "gui_verify_scratch_group"
SCRATCH_RUN_NAME = "gui_verify_cpu_smoke"
CANDIDATE_SEARCH = "2020-0"

CHECKS: list[tuple[str, bool, str]] = []
CONSOLE_ERRORS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return bool(ok)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/import", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def start_server(port: int) -> subprocess.Popen:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).is_file():
        python = sys.executable
    handle = SERVER_LOG.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [python, str(ROOT / "scripts" / "training_pipeline_ui.py"), "--port", str(port), "--no-generate-crops"],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process


def shot(page, name: str) -> Path:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return path


def api_json(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def text_of(page, selector: str) -> str:
    node = page.query_selector(selector)
    return (node.inner_text() if node else "").strip()


def verify_nav(page, base_url: str) -> None:
    """Every page in the pipeline must carry the same clickable 7-stage progress bar."""
    for path, label in (
        ("/import", "step1_import"),
        ("/manual-review", "step2_manual_review"),
        ("/bbox-overview", "step3_bbox_overview"),
    ):
        page.goto(f"{base_url}{path}", wait_until="networkidle")
        page.wait_for_timeout(1200)
        stages = page.query_selector_all('[aria-label="Training pipeline stages"] a')
        check(f"{label}: 7 clickable stages in the nav", len(stages) == 7, f"found {len(stages)}")
        shot(page, label)


def verify_step4(page, base_url: str) -> dict:
    page.goto(f"{base_url}/datasets", wait_until="networkidle")
    page.wait_for_selector("#candidateBody tr", timeout=30000)
    shot(page, "step4_01_loaded")

    rows = page.query_selector_all("#candidateBody tr")
    check("step4: candidate table is populated from the real workspace", len(rows) > 1, f"{len(rows)} rows")
    check("step4: album filter has options", len(page.query_selector_all("#albumFilter option")) > 0)

    page.fill("#searchFilter", CANDIDATE_SEARCH)
    page.wait_for_timeout(600)
    narrowed = page.query_selector_all("#candidateBody tr")
    check("step4: search filter narrows the candidate list", 0 < len(narrowed) < len(rows), f"{len(narrowed)} rows")

    page.click("#selectAll")
    page.wait_for_timeout(300)
    checked = page.query_selector_all("#candidateBody input.pick:checked")
    check("step4: Select All Visible selects every visible row", len(checked) == len(narrowed), f"{len(checked)} selected")
    shot(page, "step4_02_selected")

    page.click("#invertSelection")
    page.wait_for_timeout(200)
    check("step4: Invert Selection clears an all-selected page",
          len(page.query_selector_all("#candidateBody input.pick:checked")) == 0)
    page.click("#selectAll")
    page.wait_for_timeout(200)
    page.click("#clearSelection")
    page.wait_for_timeout(200)
    check("step4: Clear Selection empties the selection",
          len(page.query_selector_all("#candidateBody input.pick:checked")) == 0)
    page.click("#selectAll")
    page.wait_for_timeout(200)

    page.click("#previewSplits")
    page.wait_for_function("document.getElementById('splitStatus').textContent.indexOf('Previewing') === -1", timeout=60000)
    page.wait_for_timeout(300)
    preview = text_of(page, "#splitPreview").lower()
    check("step4: Preview Splits renders train/val/test counts",
          "train" in preview and "val" in preview and "test" in preview, preview.replace("\n", " ")[:90])
    group_count = text_of(page, "#splitGroupCount")
    spanning = text_of(page, "#splitSpanning")
    check("step4: Preview Splits reports scene groups and zero groups spanning splits",
          group_count.isdigit() and int(group_count) > 0 and spanning == "0",
          f"{group_count} groups, {spanning} spanning")
    shot(page, "step4_03_split_preview")

    page.fill("#datasetName", SCRATCH_DATASET_NAME)
    page.fill("#datasetVersion", SCRATCH_DATASET_VERSION)
    page.fill("#datasetDescription", "scratch selection written by scripts/verify_training_gui.py")
    page.click("#saveSelection")
    page.wait_for_function("document.getElementById('saveStatus').textContent.indexOf('Saving') === -1", timeout=60000)
    save_status = text_of(page, "#saveStatus")
    check("step4: Save Selection writes a selection manifest", save_status.startswith("Saved "), save_status[:120])
    shot(page, "step4_04_saved")

    page.click("#exportSelection")
    page.wait_for_function("document.getElementById('saveStatus').textContent.indexOf('Exporting') === -1", timeout=300000)
    export_status = text_of(page, "#saveStatus")
    check("step4: Export YOLO Dataset exports the selection", export_status.startswith("Exported "), export_status[:160])
    shot(page, "step4_05_exported")

    selections = api_json(base_url, "/api/dataset-selections")["selections"]
    scratch = [item for item in selections if item["dataset_name"] == SCRATCH_DATASET_NAME]
    check("step4: saved-selection table row reports exported=true", bool(scratch) and scratch[0]["exported"],
          json.dumps(scratch[0], sort_keys=True)[:160] if scratch else "not found")

    page.click("#continueToSetup")
    page.wait_for_load_state("networkidle")
    check("step4: Continue To Training Setup navigates to step 5", page.url.endswith("/training-setup"), page.url)
    return scratch[0] if scratch else {}


def verify_step5(page, base_url: str, selection: dict) -> dict:
    page.goto(f"{base_url}/training-setup", wait_until="networkidle")
    page.wait_for_selector("#datasetSelect option", timeout=30000, state="attached")
    shot(page, "step5_01_loaded")

    options = page.query_selector_all("#datasetSelect option")
    check("step5: exported datasets are listed", len(options) > 0, f"{len(options)} datasets")
    index = None
    for option in options:
        if SCRATCH_DATASET_NAME in option.inner_text():
            index = option.get_attribute("value")
    check("step5: the scratch export appears in the dataset picker", index is not None)
    if index is not None:
        page.select_option("#datasetSelect", index)
        page.wait_for_timeout(300)
    check("step5: dataset hint resolves the matching selection manifest",
          "selection " in text_of(page, "#datasetHint"), text_of(page, "#datasetHint")[:140])

    weights = page.query_selector_all("#weightSelect option")
    check("step5: starting weights are listed", len(weights) > 0, f"{len(weights)} weights")
    undocumented = [option.get_attribute("value") for option in weights if "[undocumented]" in option.inner_text()]
    documented = [option.get_attribute("value") for option in weights if "[undocumented]" not in option.inner_text()]
    if undocumented:
        page.select_option("#weightSelect", undocumented[0])
        page.wait_for_timeout(200)
        check("step5: undocumented weight shows a provenance warning but stays selectable",
              text_of(page, "#weightHint").startswith("WARNING:"), text_of(page, "#weightHint")[:140])
        shot(page, "step5_02_provenance_warning")
    else:
        check("step5: undocumented weight shows a provenance warning but stays selectable", True, "no undocumented weights present")
    if documented:
        page.select_option("#weightSelect", documented[0])
        page.wait_for_timeout(200)

    page.click("#augDefaults")
    page.wait_for_timeout(150)
    filled = page.eval_on_selector_all("input.aug", "nodes => nodes.filter(n => n.value !== '').length")
    check("step5: Fill Default Augmentation populates augmentation fields", filled > 0, f"{filled} fields")
    page.click("#augClear")
    page.wait_for_timeout(150)
    cleared = page.eval_on_selector_all("input.aug", "nodes => nodes.filter(n => n.value !== '').length")
    check("step5: Clear Augmentation empties augmentation fields", cleared == 0)
    page.click("#augDefaults")
    page.wait_for_timeout(150)

    page.fill("#runGroup", SCRATCH_RUN_GROUP)
    page.fill("#runName", SCRATCH_RUN_NAME)
    page.fill("#imageSize", "320")
    page.fill("#epochs", "1")
    page.fill("#batch", "2")
    page.fill("#patience", "5")
    page.fill("#device", "cpu")
    page.fill("#workers", "0")
    page.fill("#configName", "")
    page.check("#overwrite")
    shot(page, "step5_03_fields")

    page.click("#saveConfig")
    page.wait_for_function("document.getElementById('setupStatus').textContent.indexOf('Saving') === -1", timeout=60000)
    setup_status = text_of(page, "#setupStatus")
    check("step5: Save Training Config writes the config", setup_status.startswith("Saved "), setup_status[:160])
    expected_name = f"{SCRATCH_RUN_GROUP}_{SCRATCH_RUN_NAME}.json"
    check("step5: config filename defaults to <run_group>_<run_name>.json",
          expected_name in setup_status, setup_status[:160])

    page.click("#checkReadiness")
    page.wait_for_timeout(1500)
    readiness = text_of(page, "#readinessSummary")
    ready = "Ready to train" in readiness
    check("step5: readiness checklist reports a backend verdict", bool(readiness), readiness[:140])
    check("step5: Go To Train YOLO is gated by readiness",
          page.query_selector("#goTrain").is_disabled() != ready, f"ready={ready}")
    shot(page, "step5_04_readiness")

    page.click("#previewCommand")
    page.wait_for_timeout(1500)
    command = text_of(page, "#commandPreview")
    check("step5: Preview Command shows the exact yolo command",
          "yolo" in command and "detect" in command and "train" in command, command.replace("\n", " ")[:160])
    shot(page, "step5_05_command_preview")

    config_path = None
    for row in page.query_selector_all("#configBody tr"):
        if expected_name in row.inner_text():
            config_path = row.query_selector("button.use-config").get_attribute("data-path")
    check("step5: saved config appears in the saved-config table", config_path is not None, str(config_path))
    if config_path:
        # Dirty every field first: Use must replace them with the config's own values,
        # not leave a different dataset/device/epoch count visible next to config A.
        page.fill("#runGroup", "dirty_group")
        page.fill("#runName", "dirty_run")
        page.fill("#epochs", "77")
        page.fill("#batch", "9")
        page.fill("#imageSize", "1280")
        page.fill("#device", "cuda:0")
        page.fill("#workers", "8")
        page.uncheck("#overwrite")
        page.click("#augClear")
        page.wait_for_timeout(150)

        page.click(f'button.use-config[data-path="{config_path}"]')
        page.wait_for_timeout(1500)
        check("step5: Use re-selects a saved config", "Using config" in text_of(page, "#setupStatus") or
              "Readiness" in text_of(page, "#setupStatus"), text_of(page, "#setupStatus")[:120])

        stored = next((item for item in api_json(base_url, "/api/training/configs")["configs"]
                       if item["config_path"] == config_path), {})
        form = page.evaluate(
            "() => ({"
            " dataset: document.getElementById('datasetSelect').selectedOptions[0].textContent,"
            " model: document.getElementById('weightSelect').selectedOptions[0].textContent,"
            " run_group: document.getElementById('runGroup').value,"
            " run_name: document.getElementById('runName').value,"
            " image_size: document.getElementById('imageSize').value,"
            " epochs: document.getElementById('epochs').value,"
            " batch: document.getElementById('batch').value,"
            " device: document.getElementById('device').value,"
            " workers: document.getElementById('workers').value,"
            " patience: document.getElementById('patience').value,"
            " seed: document.getElementById('seed').value,"
            " overwrite: document.getElementById('overwrite').checked,"
            " augmentation: Array.from(document.querySelectorAll('input.aug'))"
            "   .filter(n => n.value !== '').map(n => n.dataset.key).sort().join(',')"
            "})"
        )
        expected_aug = ",".join(sorted(stored.get("augmentation") or {}))
        mismatches = []
        for field in ("run_group", "run_name", "image_size", "epochs", "batch", "device", "workers", "patience", "seed"):
            want = "" if stored.get(field) is None else str(stored.get(field))
            if form[field] != want:
                mismatches.append(f"{field}: form={form[field]!r} config={want!r}")
        if str(stored.get("dataset_name") or "") not in form["dataset"]:
            mismatches.append(f"dataset: form={form['dataset']!r} config={stored.get('dataset_name')!r}")
        if str(stored.get("model") or "") not in form["model"]:
            mismatches.append(f"model: form={form['model']!r} config={stored.get('model')!r}")
        if form["overwrite"] != bool(stored.get("overwrite")):
            mismatches.append(f"overwrite: form={form['overwrite']} config={stored.get('overwrite')}")
        if form["augmentation"] != expected_aug:
            mismatches.append(f"augmentation: form={form['augmentation']!r} config={expected_aug!r}")
        check("step5: Use loads every field of the chosen config into the form",
              not mismatches, "; ".join(mismatches[:4]))
        shot(page, "step5_06_use_config")

    return {"config_path": config_path, "selection_path": selection.get("selection_path"), "ready": ready}


def verify_step6(page, base_url: str, setup: dict, skip_training: bool) -> bool:
    url = f"{base_url}/training"
    if setup.get("config_path") and setup.get("selection_path"):
        url += f"?config_path={setup['config_path']}&selection_path={setup['selection_path']}"
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(1500)
    shot(page, "step6_01_loaded")

    check("step6: training configs are listed", len(page.query_selector_all("#configSelect option")) > 0)
    check("step6: exported dataset selections are listed", len(page.query_selector_all("#selectionSelect option")) > 0)

    page.click("#refreshReadiness")
    page.wait_for_timeout(2000)
    start_status = text_of(page, "#startStatus")
    check("step6: Check Readiness returns a backend verdict", start_status in {"Ready.", "Not ready."}, start_status[:140])
    shot(page, "step6_02_readiness")

    page.click("#refreshRuns")
    page.wait_for_timeout(1500)
    page.uncheck("#showAll")
    page.wait_for_timeout(1200)
    page.check("#showAll")
    page.wait_for_timeout(1200)
    check("step6: Refresh Runs / Show finished runs toggle re-renders the run list",
          page.query_selector("#runCards") is not None)

    if skip_training or start_status != "Ready.":
        check("step6: CPU training run launched from the GUI", False,
              "skipped" if skip_training else f"readiness={start_status}")
        return False

    page.click("#startRun")
    page.wait_for_function("document.getElementById('startStatus').textContent.indexOf('Starting') === -1", timeout=120000)
    started = text_of(page, "#startStatus")
    check("step6: Start Training launches a detached run", started.startswith("Started "), started[:160])
    shot(page, "step6_03_started")

    deadline = time.time() + 900
    terminal = False
    while time.time() < deadline:
        try:
            status = api_json(base_url, f"/api/training/runs/{SCRATCH_RUN_GROUP}/{SCRATCH_RUN_NAME}/status")
        except Exception:
            time.sleep(4)
            continue
        if status.get("terminal"):
            terminal = status
            break
        time.sleep(5)
    page.wait_for_timeout(5000)
    shot(page, "step6_04_live_run")

    check("step6: the run reaches a terminal state", bool(terminal),
          json.dumps({k: terminal.get(k) for k in ("status", "reason", "current_epoch")}) if terminal else "timed out")
    if terminal:
        check("step6: a run that ended without completing is labelled, not shown as a crash code",
              terminal.get("status") in {"completed", "failed", "cancelled"}, str(terminal.get("status")))
        check("step6: best.pt is reported for a completed run",
              terminal.get("status") != "completed" or bool(terminal.get("best_weights_path")),
              str(terminal.get("best_weights_path")))
    detail = text_of(page, "#detailMeta")
    check("step6: selected-run detail shows pid, run dir, and command", "pid" in detail and "run dir" in detail,
          detail.replace("\n", " ")[:120])
    log_tail = text_of(page, "#logTail")
    check("step6: log tail is rendered", len(log_tail) > 0, log_tail[-80:].replace("\n", " "))
    metrics = text_of(page, "#metricsBody")
    check("step6: metrics table renders results.csv rows or an explicit empty state", len(metrics) > 0,
          metrics.replace("\n", " ")[:100])
    shot(page, "step6_05_completed")

    stop_disabled = page.query_selector("#stopRun").is_disabled()
    check("step6: Stop Training is disabled once a run is terminal", stop_disabled)
    eval_enabled = not page.query_selector("#gotoEval").is_disabled()
    check("step6: a completed run links through to step 7", eval_enabled)
    if eval_enabled:
        page.click("#gotoEval")
        page.wait_for_load_state("networkidle")
        check("step6: Go To Evaluation navigates to step 7", "/evaluation" in page.url, page.url)
    return bool(terminal and terminal.get("status") == "completed")


def verify_step7(page, base_url: str, trained: bool) -> None:
    page.goto(f"{base_url}/evaluation", wait_until="networkidle")
    page.wait_for_selector("#groupSelect option", timeout=30000, state="attached")
    shot(page, "step7_01_loaded")

    groups = [option.get_attribute("value") for option in page.query_selector_all("#groupSelect option")]
    check("step7: run groups are listed from the real workspace", len([g for g in groups if g]) > 0, f"{len(groups)} groups")

    # Every write on this page (evaluation report, promotion) must land in the scratch run
    # group this script owns. There is deliberately no fallback: overwriting a real group's
    # report would destroy its baseline and promotion rationale, so an unavailable scratch
    # run is a FAIL, not a reason to pick someone else's run group.
    scratch_available = bool(trained) and SCRATCH_RUN_GROUP in groups
    check("step7: the scratch run group is available to evaluate", scratch_available,
          f"trained={trained}, scratch group present={SCRATCH_RUN_GROUP in groups}")
    target = SCRATCH_RUN_GROUP if scratch_available else None
    read_only_target = target or next((g for g in groups if g), None)
    check("step7: a run group can be selected for evaluation", read_only_target is not None, str(read_only_target))
    if read_only_target is None:
        return

    other = next((group for group in groups if group and group != groups[0]), None)
    if other:
        page.select_option("#groupSelect", other)
        page.click("#refreshGroups")
        page.wait_for_function("document.getElementById('groupStatus').textContent.indexOf('available') !== -1", timeout=30000)
        page.wait_for_timeout(300)
        kept = page.eval_on_selector("#groupSelect", "node => node.value")
        check("step7: Refresh Groups keeps the currently selected run group", kept == other,
              f"selected {other!r}, after refresh {kept!r}")
    else:
        check("step7: Refresh Groups keeps the currently selected run group", False,
              "skipped: only one run group is available")

    page.select_option("#groupSelect", read_only_target)
    page.click("#loadGroup")
    page.wait_for_function("document.getElementById('groupStatus').textContent.indexOf('Loading') === -1", timeout=60000)
    page.wait_for_timeout(500)
    rows = page.query_selector_all("#resultsBody tr")
    body_text = text_of(page, "#resultsBody")
    check("step7: the run-group results table has real metric rows",
          len(rows) > 0 and "No report" not in body_text and "empty" not in (rows[0].get_attribute("class") or ""),
          f"{len(rows)} rows: {body_text.replace(chr(10), ' ')[:110]}")
    shot(page, "step7_02_results")

    if not scratch_available:
        skipped = "skipped: refusing to write a report into a non-scratch run group"
        check("step7: Write Evaluation Report writes the markdown report", False, skipped)
        check("step7: the generated markdown report is rendered on the page", False, skipped)
        check("step7: Reload Report re-reads the report from disk", False, skipped)
        check("step7: Run Test Split completes as a background job", False, skipped)
        check("step7: completing a test split keeps the selected run group", False, skipped)
        check("step7: Promote stays disabled until a run and target name are chosen", False, skipped)
        check("step7: promotion is refused and the backend refusal text is shown", False, skipped)
        check("step7: the refusal explains the two-step doc-first flow", False, skipped)
        return

    page.click("#writeReport")
    page.wait_for_function("document.getElementById('reportStatus').textContent.indexOf('Writing') === -1", timeout=120000)
    report_status = text_of(page, "#reportStatus")
    check("step7: Write Evaluation Report writes the markdown report", report_status.startswith("Wrote "), report_status[:160])
    page.wait_for_timeout(800)
    report_view = text_of(page, "#reportView")
    check("step7: the generated markdown report is rendered on the page",
          len(report_view) > 40 and "No report" not in report_view, report_view.replace("\n", " ")[:120])
    shot(page, "step7_03_report")

    page.click("#loadReport")
    page.wait_for_timeout(1500)
    check("step7: Reload Report re-reads the report from disk",
          text_of(page, "#reportStatus").startswith("Loaded "), text_of(page, "#reportStatus")[:120])

    test_buttons = page.query_selector_all("#resultsBody button.test-split")
    if test_buttons:
        test_buttons[0].click()
        page.wait_for_function(
            "document.getElementById('testStatus').textContent.indexOf('running') === -1 && "
            "document.getElementById('testStatus').textContent.indexOf('Starting') === -1",
            timeout=900000,
        )
        test_status = text_of(page, "#testStatus")
        check("step7: Run Test Split completes as a background job",
              test_status.startswith("Test split complete"), test_status[:200])
        page.wait_for_timeout(1500)
        after_test = page.eval_on_selector("#groupSelect", "node => node.value")
        check("step7: completing a test split keeps the selected run group",
              after_test == target, f"selected {target!r}, after test split {after_test!r}")
        shot(page, "step7_04_test_split")
    else:
        check("step7: Run Test Split completes as a background job", False,
              "skipped: no scratch trained run to evaluate on CPU")

    page.fill("#targetName", "gui_verify_undocumented_weight.pt")
    page.wait_for_timeout(200)
    check("step7: Promote stays disabled until a run and target name are chosen",
          not page.query_selector("#promoteWeight").is_disabled())
    page.click("#promoteWeight")
    page.wait_for_function("document.getElementById('promoteStatus').textContent.indexOf('Promoting') === -1", timeout=60000)
    promote_status = text_of(page, "#promoteStatus")
    promote_detail = text_of(page, "#promoteDetail")
    check("step7: promotion is refused and the backend refusal text is shown",
          "refused" in promote_status.lower() and "yolo_weight_recreation.md" in promote_detail,
          promote_status[:120])
    check("step7: the refusal explains the two-step doc-first flow",
          "Step 1:" in promote_detail and "Step 2:" in promote_detail)
    shot(page, "step7_05_promotion_refused")


def cleanup(keep: bool) -> None:
    if keep:
        print("\n--keep given: scratch selection, dataset, config, and run left in place.")
        return
    targets = [
        ROOT / "workspace" / "dataset_selections" / f"{SCRATCH_DATASET_NAME}_{SCRATCH_DATASET_VERSION}.json",
        ROOT / "workspace" / "datasets" / f"{SCRATCH_DATASET_NAME}_{SCRATCH_DATASET_VERSION}",
        ROOT / "workspace" / "training_configs" / f"{SCRATCH_RUN_GROUP}_{SCRATCH_RUN_NAME}.json",
        ROOT / "workspace" / "runs" / SCRATCH_RUN_GROUP,
        ROOT / "workspace" / "reports" / f"{SCRATCH_RUN_GROUP}_evaluation.md",
        ROOT / "workspace" / "reports" / f"{SCRATCH_RUN_GROUP}_evaluation.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.is_file():
            target.unlink()
    print("\nScratch selection, dataset, config, run, and report removed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the scratch selection/export/run afterwards")
    parser.add_argument("--skip-training", action="store_true", help="Skip the live CPU training run in step 6")
    parser.add_argument("--port", type=int, default=0, help="Serve on a specific port instead of a free one")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = start_server(port)
    try:
        if not wait_for_server(base_url):
            print(f"Server did not come up on {base_url}; see {SERVER_LOG}")
            return 2
        print(f"Server up on {base_url} (pid {server.pid})\n")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_context(viewport={"width": 1500, "height": 1000}).new_page()
            # The promotion probe deliberately triggers the backend's 400 refusal, so the
            # browser's "failed to load resource" note for that route is expected.
            expected = "/api/training/promote-weight"

            def on_console(message) -> None:
                url = message.location.get("url") or page.url
                if message.type == "error" and expected not in url:
                    CONSOLE_ERRORS.append(f"{url} :: {message.text}")

            page.on("console", on_console)
            page.on("pageerror", lambda error: CONSOLE_ERRORS.append(f"{page.url} :: {error}"))

            verify_nav(page, base_url)
            selection = verify_step4(page, base_url)
            setup = verify_step5(page, base_url, selection)
            trained = verify_step6(page, base_url, setup, args.skip_training)
            verify_step7(page, base_url, trained)

            check("no browser console errors on any page", not CONSOLE_ERRORS,
                  "; ".join(CONSOLE_ERRORS[:3]) if CONSOLE_ERRORS else "")
            browser.close()
    finally:
        cleanup(args.keep)
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    failed = [(name, detail) for name, ok, detail in CHECKS if not ok]
    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed}/{len(CHECKS)} checks passed")
    for name, detail in failed:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
    shots = sorted(path.name for path in SHOT_DIR.glob("*.png"))
    print(f"Screenshots ({len(shots)}) in {SHOT_DIR.relative_to(ROOT)}:")
    for name in shots:
        print(f"  {name}")
    print("=" * 72)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
