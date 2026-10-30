"""Small regressions for the final public-release findings."""
import json
import os
import subprocess
import sys
from urllib.request import urlopen

import pytest

from scripts import acquisition, jobs, pipeline, training_pipeline_lib as training, training_pipeline_ui as ui, workspace
from test_improve_model_lifecycle import recorded_run
from test_release_training_fixes import ROOT, DOM, controller, run_js
from test_training_pipeline_ui import UiServer


@pytest.mark.parametrize('value', ['', ' ', None])
def test_blank_coordinate_rejected_by_backend(value):
    with pytest.raises(ValueError, match='all four area coordinates'):
        acquisition.validate_bbox([-95, 28, value, 29])


@pytest.mark.parametrize('background', [False, True])
def test_missing_ais_cli_fails_before_job_or_inference(tmp_path, background):
    from PIL import Image
    image = tmp_path/'2020-01-24_scene.png'
    Image.new('RGB', (20,20)).save(image)
    target = tmp_path/'user workspace'
    result = subprocess.run([sys.executable,'-m','opendarksts','--workspace',str(target),
        'run','--images',str(image),'--ais-dir',str(target/'nonexistent'),*(['--background'] if background else [])],
        env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
    assert result.returncode != 0
    assert 'AIS path does not exist' in json.loads(result.stdout)['errors'][0]
    assert not (target/'jobs').exists()
    with pytest.raises(ValueError, match='AIS path does not exist'):
        pipeline.run_pipeline([str(image)], ais_dir=target/'nonexistent')


def test_status_labels_do_not_depend_on_whether_missing_source_was_configured():
    source=(ROOT/'scripts/templates/static/results.js').read_text()
    function=source.split(' function statusText(r){',1)[1].split(' const verdictLabel',1)[0]
    run_js("const assert=require('assert/strict');function statusText(r){"+function+"""
for(const ais_source of ['', '/missing']) {
 assert.equal(statusText({dark_status:'AIS_MISSING',ais_status_text:'unknown'}),'DETECTED_ONLY (stored as AIS_MISSING) — unknown');
 assert.equal(statusText({dark_status:'AIS_INVALID',ais_status_text:'unknown'}),'AIS_ERROR (stored as AIS_INVALID) — unknown');
}
""")
    assert 'DETECTED_ONLY (stored as AIS_MISSING)' in (ROOT/'scripts/templates/static/operations.js').read_text()


def test_setup_error_is_next_to_clicked_button_and_field_hint():
    run_js(DOM+controller('training_setup.html')+r'''
state.datasets=[{dataset_name:'reviewed',selection_path:'/selection'}];
state.weights=[{path:'/model.pt'}];el('datasetSelect').value=0;el('weightSelect').value=0;
// currentDataset selects by the index of the exported dataset.
el('runName').value='bad run name';
(async()=>{
 await saveConfig(false);
 assert.match(el('advancedStatus').textContent,/must not contain spaces/);
 assert.equal(el('setupStatus').textContent,'');
 assert.equal(el('runName').ariaInvalid,'true');
 await saveConfig(true);
 assert.match(el('setupStatus').textContent,/must not contain spaces/);
})().catch(e=>{console.error(e);process.exitCode=1});
''')
    assert 'id="runNameHint"' in (ROOT/'scripts/templates/training_setup.html').read_text()


def test_recipe_route_is_browsable_and_does_not_read_arbitrary_paths(tmp_path, monkeypatch):
    weight = tmp_path/workspace.DEFAULT_WEIGHT_NAME
    weight.write_bytes(b'fixture')
    monkeypatch.setattr(training,'list_starting_weights',lambda:[{'path':str(weight)}])
    with UiServer(ui.TrainingPipelineHandler) as server:
        from urllib.parse import urlencode
        with urlopen(server.base_url+'/training-recipe?'+urlencode({'weight':str(weight)})) as response:
            html=response.read().decode()
        assert 'Training recipe' in html
        assert 'yolo11s.pt' in html
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            urlopen(server.base_url+'/training-recipe?weight=/etc/passwd')
        assert error.value.code == 400
        for route in ['/detect','/results','/training-setup']:
            html=urlopen(server.base_url+route).read().decode()
            assert html.count('href="/static/operations.css"') == 1


def test_promotion_regenerates_report_and_clears_stale_comparison():
    run_js(DOM+controller('evaluation.html')+r'''
state.group='public';state.runs=[{run_name:'cpu',promotion_warning:'far below current model'}];
el('promoteRun').value='cpu';el('targetName').value='weak.pt';el('promotionWarning').textContent='far below current model';
window.confirm=()=>true;
let requests=[];
P.api=async(path,options)=>{
 requests.push(path);
 if(path.includes('promote-weight'))return {promoted_path:'weak.pt',recreation_path:'weak.recreation.json'};
 if(path==='/api/operations/model')return {};
 if(path==='/api/training/model')return {};
 if(options&&options.method==='POST')return {runs:[{run_name:'cpu'}],selected_run_name:'cpu',report_markdown_path:'report.md',run_count:1};
 return {exists:true,markdown:'Promoted: cpu'};
};
P.renderModel=()=>{};P.renderMarkdown=x=>x;renderResults=()=>{};
(async()=>{
 await promoteWeight();
 assert(requests.includes('/api/training/evaluation-report'));
 assert.equal(el('promotionWarning').textContent,'');
 assert.equal(el('reportView').innerHTML,'Promoted: cpu');
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_split_seed_and_plain_class_warning(tmp_path):
    saved=training.save_dataset_selection(dataset_name='seed',image_keys=['one'],seed=123,
        selections_dir=tmp_path/'selections',datasets_dir=tmp_path/'datasets',reviews_dir=tmp_path/'reviews',manifest_path=tmp_path/'manifest.json')
    assert saved['split_summary']['seed'] == 123
    assert saved['selection']['split_summary']['seed'] == 123
    assert 'similar' in training.likely_duplicate_class_name_warnings(['ship','ships'])[0]
    description=jobs.scan_description({'images':['2020-01-24_scene.png'],'weights':r'C:\models\detector.pt'},None)
    assert 'model detector.pt' in description


def test_import_tip_help_model_card_and_public_paths():
    from scripts.training_pipeline_cli import build_parser
    help_result=subprocess.run([sys.executable,'-m','scripts.training_pipeline_cli','import-data','--help'],capture_output=True,text=True)
    assert 'YOLO11s' in help_result.stdout and 'YOLOv8n' not in help_result.stdout
    assert '--recursive --auto-detect</code>' in (ROOT/'scripts/templates/import.html').read_text()
    card=(ROOT/'opendarksts/assets/model_card.md').read_text()
    assert ' --- ' not in card and '[Download checkpoint](https://' in card
    assert 'Edit in Annotation' in (ROOT/'scripts/templates/bbox_overview.html').read_text()
    for file in ['training_pipeline_cli.md','paper_release_alignment.md','yolo_weight_recreation.md']:
        assert '/home/gavin/' not in (ROOT/'docs'/file).read_text()
