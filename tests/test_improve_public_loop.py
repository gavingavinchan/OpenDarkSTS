"""GUI regressions for public Improve findings F3–F6 and F8."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_import_explains_byte_identity_and_bounds_both_dialogs():
    html = (ROOT / 'scripts/templates/import.html').read_text()
    assert 'Byte-identical copies are recognised; re-encoded copies are treated as new images.' in html
    for selector in ('.naming-card {', '.conflict-modal {'):
        css = html.split(selector, 1)[1].split('}', 1)[0]
        assert 'max-height: calc(100dvh - 48px)' in css
        assert 'overflow-y: auto' in css


def node_check(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for GUI keyboard/launch regressions')
    result = subprocess.run([node, '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_escape_closes_keyboard_help_even_when_input_has_focus():
    from scripts.review_visual_detections_ui import render_app_html
    html = render_app_html(None)
    handler = html.split("document.addEventListener('keydown', async event => {", 1)[1].split('// Space: advance', 1)[0]
    node_check('''const assert = require('assert');
let visible=true, prevented=false, cleared=false;
const overlay={classList:{contains:()=>visible,remove:()=>visible=false}};
const document={activeElement:{tagName:'INPUT'},getElementById:()=>overlay};
const event={key:'Escape',preventDefault:()=>prevented=true};
(async()=>{''' + handler + '''})();
assert(!visible); assert(prevented);
''')


def test_duplicate_launch_refreshes_readiness_and_links_progress():
    html = (ROOT / 'scripts/templates/training.html').read_text()
    handler = html.split('    async function startRun() {', 1)[1].split('    function statusPill', 1)[0].rsplit('}', 1)[0]
    node_check('''const assert=require('assert');
const nodes={};const document={getElementById:id=>nodes[id]||(nodes[id]={append(link){this.link=link}}),createElement:()=>({})};
const pickedPaths=()=>({});const state={};let readiness=0,runs=0,detail=0;
const checkReadiness=async()=>{readiness++};const loadRuns=async()=>{runs++};const refreshDetail=async()=>{detail++};
const P={setStatus:(id,message)=>document.getElementById(id).message=message,api:async()=>{throw Object.assign(new Error('already running'),{payload:{already_running:true,run_dir:'C:\\\\workspace\\\\runs\\\\group\\\\run'}})}};
(async()=>{''' + handler + '''})().then(()=>{
assert.equal(readiness,1);assert.equal(runs,1);assert.equal(detail,1);
assert.equal(state.selectedRun.run_group,'group');assert.equal(state.selectedRun.run_name,'run');
assert(nodes.startStatus.message.includes('already running'));assert.equal(nodes.startStatus.link.href,'#selectedRunProgress');assert(nodes.startRun.disabled);
}).catch(e=>{console.error(e);process.exitCode=1});''')


def test_results_transferred_crop_link_uses_encoded_image_identity():
    js = (ROOT / 'scripts/templates/static/results.js').read_text()
    code = js.split("  let improveLink=$('openInImprove');", 1)[1].split('  document.querySelectorAll', 1)[0]
    node_check('''const assert=require('assert');let improveLink=null;const review={transfer:{image_key:'crop & one'}};
const $=()=>({after:link=>{}});const document={createElement:()=>({style:{}})};
''' + code + '''
assert.equal(improveLink.textContent,'Open in Improve');assert(!improveLink.hidden);
assert.equal(improveLink.href,'/manual-review?image_id=crop%20%26%20one');
''')
