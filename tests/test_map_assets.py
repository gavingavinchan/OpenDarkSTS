from __future__ import annotations

import json
import shutil
import subprocess
import threading
import tomllib
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["review_visual_detections_ui", "training_pipeline_ui"])
def test_review_map_assets_are_served_locally_and_cannot_escape_static_root(module):
    from importlib import import_module

    ui = import_module("scripts." + module)
    handler = ui.ReviewHandler if module == "review_visual_detections_ui" else ui.TrainingPipelineHandler
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for asset in ("leaflet/leaflet.js", "leaflet-draw/leaflet.draw.js", "leaflet-draw/leaflet.draw.css", "leaflet/images/marker-icon.png", "leaflet-draw/images/spritesheet.svg"):
            with urllib.request.urlopen(base + "/static/vendor/" + asset, timeout=5) as response:
                assert response.status == 200
                assert response.read() == (ROOT / "scripts/templates/static/vendor" / asset).read_bytes()
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(base + "/static/../../../pyproject.toml", timeout=5)
        assert denied.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_review_renderer_uses_local_dependencies_and_package_contains_all_vendor_assets():
    from scripts import review_visual_detections_ui as ui

    html = ui.render_app_html(None)
    assert 'src="/static/vendor/leaflet/leaflet.js"' in html
    assert 'src="/static/vendor/leaflet-draw/leaflet.draw.js"' in html
    assert "https://unpkg.com" not in html
    assert "https://cdnjs.cloudflare.com" not in html
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = package["tool"]["setuptools"]["package-data"]["scripts"]
    script_root = ROOT / "scripts"
    packaged = {path for pattern in patterns for path in script_root.glob(pattern) if path.is_file()}
    assert set((script_root / "templates/static/vendor").rglob("*")) - packaged == {
        path for path in (script_root / "templates/static/vendor").rglob("*") if path.is_dir()
    }


@pytest.mark.parametrize("viewer_available", [True, False])
def test_area_controls_survive_map_failure_and_retry_recovers(viewer_available):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the frontend regression test")
    # Execute the real page controller with a failed tile provider, rather than
    # requiring an internet outage or changing the user's browser settings.
    harness = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const viewerAvailable = JSON.parse(process.argv[2]);
const elements = {}, logs = [], callbacks = {}, tileImages = [{complete:true,naturalWidth:0}];
const element = id => elements[id] ||= {value:'', hidden:false, disabled:false, textContent:'', classList:{toggle(){}}, add(){}, addEventListener(){}, querySelectorAll(){return tileImages}};
const project = {bbox:[-94.73,29.07,-94.57,29.205],start:'2020-04-01',end:'2020-04-01',images:'/test/images'};
const map = {setView(){return this}, fitBounds(){}, on(){}};
const tiles = {on(name,handler){callbacks[name]=handler;return this},addTo(){callbacks.loading();return this},redraw(){callbacks.loading();return this}};
const L = {map(){return map},rectangle(){return {addTo(){return this},setBounds(){}}},latLngBounds(){return {pad(){return this}}},tileLayer(){return tiles}};
const context = {window:{L:viewerAvailable ? L : undefined,Pipeline:{api:async()=>({project,presets:{Test:[-95,28,-94,29]}}),setStatus(){},logInteraction:(name,payload)=>logs.push({name,payload})}},
 document:{body:{dataset:{page:'area'}},getElementById:element,querySelectorAll:()=>[]},
 location:{search:'',reload(){}},URLSearchParams,Option:function(){},setTimeout:()=>1,clearTimeout(){},L};
(async()=>{
 await vm.runInNewContext(source,context);
 assert.equal(element('start').value,'2020-04-01');
 assert.equal(typeof element('areaForm').onsubmit,'function');
 element('start').value='2024-02-10';element('datePeriod').value='month';element('datePeriod').onchange();
 assert.equal(element('start').value,'2024-02-01');assert.equal(element('end').value,'2024-02-29');
 element('datePeriod').value='year';element('datePeriod').onchange();
 assert.equal(element('start').value,'2024-01-01');assert.equal(element('end').value,'2024-12-31');
 element('datePeriod').value='date';element('datePeriod').onchange();
 assert.equal(element('start').value,element('end').value);
 element('preset').value='Test';element('preset').onchange();
 assert.equal(element('west').value,'-95.00000');
 if (!viewerAvailable) {
   assert.equal(element('drawArea').disabled,true);
   assert.match(element('mapStatus').textContent,/enter coordinates/);
 } else {
   callbacks.tileerror();callbacks.load();
   assert.match(element('mapStatus').textContent,/background unavailable/);
   assert.equal(element('retryMap').hidden,false);
   element('retryMap').onclick();
   tileImages[0].naturalWidth=256;callbacks.load();
   assert.equal(element('retryMap').hidden,true);
   assert.match(element('mapStatus').textContent,/OpenStreetMap/);
   assert.ok(logs.some(item=>item.name==='area_map_tile_error'));
   assert.ok(logs.some(item=>item.name==='retry_area_map'));
 }
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run([node, "-e", harness, str(ROOT / "scripts/templates/static/operations.js"), json.dumps(viewer_available)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
