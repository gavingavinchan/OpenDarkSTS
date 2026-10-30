"""Regression coverage for the public evaluation, setup, and saved-selection findings."""
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import training_pipeline_lib as training, workspace
from test_improve_model_lifecycle import recorded_run

ROOT = Path(__file__).resolve().parents[1]


def run_js(source):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for controller regressions')
    result = subprocess.run([node, '-e', source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def controller(template):
    source = (ROOT/'scripts/templates'/template).read_text().split('<script>')[-1].split('</script>')[0]
    return source.split('\n    wire();')[0]


DOM = r'''
const assert=require('assert/strict');
const elements={};
const el=id=>elements[id]||=({_value:'',get value(){return this._value},set value(v){this._value=String(v)},textContent:'',innerHTML:'',dataset:{},style:{},disabled:false,
 querySelectorAll:()=>[],addEventListener(){}});
const document={getElementById:el,querySelector:el};
const window={Pipeline:{esc:String,fmt:x=>Number(x).toFixed(3),shortTime:String,logInteraction(){},autoLogControls(){},setStatus:(id,text)=>el(id).textContent=text},location:{search:''}};
'''


def test_promotion_warns_before_request_and_explains_required_filename():
    run_js(DOM + controller('evaluation.html') + r'''
state.runs=[{run_name:'short',promotion_warning:'far below current model'}];
el('promoteRun').value='short';
updatePromoteButton();
assert.equal(el('promoteWeight').disabled,true);
assert.match(el('promoteHelp').textContent,/filename/);
el('targetName').value='public_best.pt';
updatePromoteButton();
assert.equal(el('promoteWeight').disabled,false);
let asked=0,calls=0;
window.confirm=message=>{asked++;assert.match(message,/far below/);return false};
P.api=()=>{calls++;throw Error('must not promote')};
(async()=>{await promoteWeight();assert.equal(asked,1);assert.equal(calls,0)})().catch(e=>{console.error(e);process.exitCode=1});
'''.replace('const params = new URLSearchParams(window.location.search);', ''))


def test_low_test_score_warning_and_promoted_record_use_test_metrics(tmp_path):
    run = recorded_run(tmp_path)
    (run/'results.csv').write_text('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.8,0.6\n')
    metrics = {'split':'test', 'images':7, 'mAP50':.11, 'mAP50-95':.07, 'precision':.2, 'recall':.1}
    (run/'test_metrics.json').write_text(json.dumps({'metrics':metrics}))
    root = tmp_path/'model_weights'
    root.mkdir()
    old = root/'old.pt'
    old.write_bytes(b'old')
    workspace.set_current_model(old, metrics={'split':'test','mAP50':.895})
    warnings=[]
    result = training.promote_weight(run_group='public',run_name='cpu',target_name='improved',runs_dir=tmp_path/'runs',model_weights_dir=root,warn=warnings.append)
    selected=json.loads((root/'current.json').read_text())
    assert selected['metrics'] == metrics
    assert warnings == result['warnings']
    assert '0.110' in warnings[0] and '0.895' in warnings[0]
    assert training.promotion_warning({'mAP50':.85}, {'mAP50':.895,'split':'test'}) is None
    assert training.promotion_warning({'mAP50':.1}, {'mAP50':.895,'split':'val'}) is None


def test_legacy_test_count_recovered_from_matching_manifest(tmp_path):
    run=recorded_run(tmp_path)
    metadata=json.loads((run/'training_metadata.json').read_text())
    manifest=Path(metadata['dataset_manifest_path'])
    manifest.write_text(json.dumps({'exported_files':[{'path':'images/test/one.png'},{'path':'images/test/two.png'},{'path':'images/train/three.png'}]}))
    metadata['dataset_manifest_sha256']=training.file_hash(manifest)
    (run/'training_metadata.json').write_text(json.dumps(metadata))
    (run/'test_metrics.json').write_text(json.dumps({'metrics':{'mAP50':.2,'mAP50-95':.1}}))
    assert training.run_evaluation_summary(run)['test_metrics']['images'] == 2


def test_evaluation_report_rounds_metrics_without_changing_json():
    report={'run_group':'round','generated_at':'today','recommendation':{'run_name':'one','reason':'test'},
            'promotion':{'promoted':False},'runs':[{'run_name':'one','test_metrics':{'mAP50':.123456789,'precision':0.,'recall':1.}}]}
    markdown=training.evaluation_markdown(report)
    assert '0.123 | 0.000 | 1.000' in markdown
    assert '.123456789' not in markdown
    assert report['runs'][0]['test_metrics']['mAP50'] == .123456789


def test_saved_selection_restores_export_and_invalidates_on_edit():
    run_js(DOM + controller('datasets.html') + r'''
const item={selection_path:'/saved.json',export_is_current:true,export_output_dir:'/dataset',selection:{
 dataset_name:'saved',dataset_version:'v2',items:[{image_key:'one',split:'train'}],split_policy:{seed:42,split_mode:'group',default_ratios:{train:.8,val:.1,test:.1}}}};
el('candidateSummary').textContent='1 candidate';
restoreSelection(item);
assert.equal(el('continueToSetup').disabled,false);
assert.equal(el('datasetName').value,'saved');
el('datasetName').value='changed';updateSelectionSummary();
assert.equal(el('continueToSetup').disabled,true);
assert.equal(el('exportSelection').disabled,true);
restoreSelection({...item,export_is_current:false});
assert.equal(el('continueToSetup').disabled,true);
''')


def test_command_preview_quotes_paths_on_linux_and_windows():
    args=['yolo','detect','train','model=/folder with space/ship.pt',"data=C:\\User's folder\\data.yaml",'name=$unsafe`echo`']
    assert shlex.split(training.display_shell_command(args,windows=False)) == args
    ps=training.display_shell_command(args,windows=True)
    assert ps.startswith("& 'yolo' 'detect' 'train'")
    assert "'model=/folder with space/ship.pt'" in ps
    assert "'data=C:\\User''s folder\\data.yaml'" in ps
    assert "'name=$unsafe`echo`'" in ps


def test_default_weight_hint_names_yolo11s_recipe(tmp_path):
    path=tmp_path/workspace.DEFAULT_WEIGHT_NAME
    path.write_bytes(b'fixture')
    assert training.starting_weight_entry(path,'workspace','')['recreation_document'] == 'opendarksts/assets/weight_recreation.md'


def test_markdown_paragraphs_code_front_matter_and_metric_aliases():
    source=(ROOT/'scripts/templates/static/pipeline.js').read_text()
    run_js("const assert=require('assert/strict'); const window={}; const document={body:{dataset:{}}};\n" + source + r'''
const html=window.Pipeline.renderMarkdown('---\nlicense: agpl\n---\n# Model\n\nA wrapped\nparagraph with `inline code` and <script>.\n\n| Value | Score |\n|---|---|\n| `ships` | 0.544 |');
assert.ok(!html.includes('license:'));
assert.ok(!html.includes('<script>'));
assert.ok(html.includes('<p>A wrapped paragraph with <code>inline code</code> and &lt;script&gt;.</p>'));
assert.ok(html.includes('<td><code>ships</code></td>'));
const node={};
window.Pipeline.renderModel(node,{filename:'new.pt',metrics:{split:'test',images:7,'mAP50-95':.12345,mAP50:.2}});
assert.match(node.innerHTML,/Held-out test results: 7 images/);
assert.match(node.innerHTML,/<td>0.123<\/td>/);
''')


def test_import_tip_uses_public_command():
    html=(ROOT/'scripts/templates/import.html').read_text()
    assert 'opendarksts training import-data --source' in html
    assert '.venv/bin/python scripts/training_pipeline_cli.py import-data' not in html


def test_single_epoch_summary():
    run_js(DOM + controller('training_setup.html') + r'''
['epochs','imageSize','batch'].forEach(id=>el(id).value='1');
updateSettingsSummary();
assert.equal(el('epochSummary').textContent,'1 epoch');
el('epochs').value='2';updateSettingsSummary();
assert.equal(el('epochSummary').textContent,'2 epochs');
''')
