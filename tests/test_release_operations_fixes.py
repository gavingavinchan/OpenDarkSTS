"""Submission preflight, exports, area settings, TIFF decoding, and lock cleanup."""
import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import acquisition, detect, jobs, operations, results, workspace
from test_candidate_results import run
from test_release_training_fixes import ROOT, run_js
from test_training_pipeline_ui import UiServer


@pytest.mark.parametrize('background',[False,True])
@pytest.mark.parametrize('kind',['imagery','run'])
def test_cli_rejects_missing_inputs_before_creating_job(tmp_path,kind,background):
    target=tmp_path/'cli workspace'
    env={**os.environ,'PYTHONPATH':str(ROOT)}
    env.pop('CDSE_CLIENT_ID',None)
    env.pop('CDSE_CLIENT_SECRET',None)
    command=['acquire','imagery','--start','2020-01-24','--bbox','-95','28','-94','29'] if kind=='imagery' else ['run','--images',str(target/'missing folder')]
    result=subprocess.run([sys.executable,'-m','opendarksts','--workspace',str(target),*command,*(['--background'] if background else [])],env=env,capture_output=True,text=True)
    assert result.returncode != 0
    payload=json.loads(result.stdout)
    assert not payload['ok']
    assert ('CDSE_CLIENT_ID' if kind=='imagery' else 'No imagery files found') in payload['errors'][0]
    assert not (target/'jobs').exists()


def test_area_validation_is_plain_and_date_mode_roundtrips(tmp_path,monkeypatch):
    monkeypatch.setattr(operations,'PROJECT_PATH',tmp_path/'project.json')
    for bounds,text in [([2,1,1,2],'West must be less than East'),([1,2,2,1],'South must be less than North')]:
        with pytest.raises(ValueError,match=text):
            acquisition.validate_bbox(bounds)
    saved=operations.save_project({'date_period':'date','start':'2020-01-24','end':'2020-02-01'})
    assert saved['end']=='2020-01-24'
    assert operations.project()['date_period']=='date'
    saved=operations.save_project({'date_period':'range','end':'2020-01-25'})
    assert saved['end']=='2020-01-25'
    assert operations.project()['date_period']=='range'
    html=(ROOT/'scripts/templates/operations.html').read_text()
    assert html.index('id="areaError"') < html.index('Save & continue')
    assert 'DETECTED_ONLY (stored as AIS_MISSING)' in html
    assert 'AIS_ERROR (stored as AIS_INVALID)' in html


def test_area_controller_remembers_mode_disables_end_and_shows_local_error():
    source=(ROOT/'scripts/templates/static/operations.js').read_text()
    run_js(r'''
const assert=require('assert/strict'),elements={};
const el=id=>elements[id]||=({value:'',textContent:'',disabled:false,hidden:true,add(){},addEventListener(){},classList:{toggle(){}}});
const settings={bbox:[-95,28,-94,29],date_period:'date',start:'2020-01-24',end:'2020-01-24',images:'/example'};
const document={body:{dataset:{page:'area'}},getElementById:el,querySelectorAll:()=>[]};
let saved;
const window={Pipeline:{api:async(path,options)=>{
 if(options.body){saved=options.body;throw Error('West must be less than East. Check the area coordinates.')}
 return {project:settings,presets:{}};
},setStatus:(node,text)=>node.textContent=text,logInteraction(){}}};
const location={search:''},Option=function(){};
(async()=>{
''' + 'await '+source.rstrip().rstrip(';') + r''';
assert.equal(el('datePeriod').value,'date');assert.equal(el('end').disabled,true);
el('datePeriod').value='range';el('datePeriod').onchange();assert.equal(el('end').disabled,false);
el('datePeriod').value='date';el('datePeriod').onchange();
await el('areaForm').onsubmit({preventDefault(){}});
assert.equal(saved.date_period,'date');assert.equal(saved.end,saved.start);
assert.match(el('areaError').textContent,/West must be less than East/);
assert.equal(el('areaCoordinates').open,true);
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_csv_review_columns_and_run_specific_download_names(run):
    empty=list(csv.DictReader(io.StringIO(results.export_results(run,'csv').decode())))[0]
    assert all(empty[k]=='' for k in ['review_verdict','review_label','reviewer','review_time'])
    assert 'human_review' not in empty
    results.review_detection(run,'one','not_sts',training_label='ship_non_sts',reviewer='tester')
    row=list(csv.DictReader(io.StringIO(results.export_results(run,'csv').decode())))[0]
    assert row['review_verdict']=='not_sts' and row['review_label']=='ship_non_sts' and row['reviewer']=='tester'
    assert row['review_time']
    record=json.loads((run/'run.json').read_text())
    record['started_at']='2026-10-30T12:34:56+00:00'
    (run/'run.json').write_text(json.dumps(record))
    filename=results.export_filename(run,'csv')
    assert '2026-10-30T12-34-56' in filename and filename.endswith('.csv')
    from scripts.training_pipeline_ui import TrainingPipelineHandler
    with UiServer(TrainingPipelineHandler) as server:
        from urllib.request import urlopen
        from urllib.parse import urlencode
        with urlopen(server.base_url+'/api/results/export?'+urlencode({'directory':str(run),'format':'csv'})) as response:
            assert filename in response.headers['Content-Disposition']


def test_download_history_has_dates_and_old_failed_scans_avoid_date_unknown(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'JOB_ROOT',tmp_path/'jobs')
    ident='a'*32
    root=jobs.job_dir(ident)
    root.mkdir(parents=True)
    workspace.atomic_json(root/'job.json',{'job_id':ident,'kind':'ais','status':'completed','options':{'start':'2020-01-24','end':'2020-01-25'},'created_at':'2026-10-30T00:00:00+00:00'})
    assert jobs.read_job(ident)['description']=='AIS download / 2020-01-24 to 2020-01-25'
    description=jobs.scan_description({'images':['/missing/imagery']},'2026-10-30T00:00:00+00:00')
    assert 'date unknown' not in description
    assert 'no dated imagery found' in description


def test_progress_log_scrolls_to_latest():
    # Execute the shipped controller, with a growing log in the selected job.
    source=(ROOT/'scripts/templates/static/operations.js').read_text()
    run_js(r'''
const assert=require('assert/strict'),elements={};
const el=id=>elements[id]||=({value:'',textContent:'',dataset:{},scrollTop:0,scrollHeight:800,addEventListener(){},querySelectorAll:()=>[],replaceChildren(){}});
const document={body:{dataset:{page:'jobs'}},getElementById:el,querySelectorAll:()=>[]};
const job={job_id:'one',kind:'ais',status:'completed',description:'AIS download / 2020-01-24',created_at:'2026-10-30T00:00:00Z',log_tail:['first','latest']};
const location={search:''},setInterval=()=>{};
const window={Pipeline:{esc:String,logInteraction(){},setStatus(){},api:async path=>path.endsWith('dashboard')?{project:{}}:path.endsWith('/jobs')?{jobs:[job]}:job}};
(async()=>{
'''+'await '+source.rstrip().rstrip(';')+r''';
assert.match(el('jobTitle').textContent,/2020-01-24/);
assert.equal(el('jobLog').scrollTop,800);assert.equal(el('jobLog').textContent,'first\nlatest');
})().catch(e=>{console.error(e);process.exitCode=1});
''')


def test_rgb_geotiff_uses_identical_pixels_without_opencv_tiff_reader(tmp_path):
    image=tmp_path/'2020-01-24_tile.tiff'
    Image.new('RGB',(30,20),(12,34,56)).save(image,tiffinfo={33550:(.01,.01,0.)})
    sources=[]
    def predict(**kwargs):
        sources.append(kwargs['source'])
        return []
    model=SimpleNamespace(names={1:'ships'},predict=predict)
    detect.detect_image(image,model)
    assert isinstance(sources[0],Image.Image)
    assert sources[0].getpixel((0,0))==(12,34,56)


@pytest.mark.parametrize('fail',[False,True])
def test_operation_lock_cleans_legacy_markers_and_keeps_stable_anchor(tmp_path,fail):
    marker=tmp_path/'output'/'.run.lock'
    marker.parent.mkdir()
    marker.touch()
    try:
        with workspace.operation_lock(marker):
            with pytest.raises(BlockingIOError):
                with workspace.operation_lock(marker):
                    pytest.fail('must serialize operations')
            if fail:
                raise RuntimeError('failed run')
    except RuntimeError:
        assert fail
    assert not marker.exists()
    with workspace.operation_lock(marker):
        assert not marker.exists()


def test_initialization_removes_stale_init_marker(tmp_path):
    target=tmp_path/'fresh workspace'
    target.mkdir()
    (target/'.init.lock').touch()
    completed=subprocess.run([sys.executable,'-m','opendarksts','--workspace',str(target),'init','--no-model','--no-example'],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    assert not (target/'.init.lock').exists()
