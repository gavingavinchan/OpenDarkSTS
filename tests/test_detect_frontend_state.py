"""Execute the shipped controllers to regress F02/F07/F08/F09 without a browser dependency."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_js(harness, controller, *arguments):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for frontend controller tests')
    result = subprocess.run([node, '-e', harness, str(ROOT / 'scripts/templates/static' / controller), *arguments], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('page', ['data', 'run'])
def test_saved_inventory_and_unsaved_radius_update_immediately(page):
    run_js(r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements={},page=process.argv[2];
const element=id=>elements[id]||=( {value:'',textContent:'',dataset:{},hidden:false,
 add(){},addEventListener(name,fn){this[name]=fn},closest(){return {dataset:{}}},append(){}} );
let count=0;
const project={start:'2020-01-24',end:'2020-01-24',images:'/example',ais_dir:'/empty',confidence:.25,radius:1000};
const dashboard=()=>({project,inventory:{images:1,ais_dir:count},readiness:{ready:true,image_count:1,warnings:[],blockers:[]},model:{filename:'model.pt'},sensitivity_presets:{Low:.1,Medium:.25,High:.45}});
const P={api:async(path,options)=>{
 if(path.endsWith('/project')){Object.assign(project,options.body);count=1;return project}
 return dashboard();
},setStatus:(el,message)=>el.textContent=message,logInteraction(){},fmt:n=>n.toLocaleString('en-US')};
const context={window:{Pipeline:P},document:{body:{dataset:{page}},getElementById:element,querySelectorAll:()=>[]},location:{search:''},URLSearchParams,Option:function(){}};
(async()=>{
 await vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
 if(page==='data'){
  assert.equal(element('aisCount').textContent,'0 AIS files');
  element('aisPath').value='/records.csv';await element('saveAis').onclick();
  assert.equal(element('aisCount').textContent,'1 AIS file');
  assert.equal(element('notice').textContent,'AIS source saved.');
 }else{
  assert.match(element('runInputs').textContent,/1,000 metres/);
  element('radius').value='500';element('radius').input();
  assert.match(element('runInputs').textContent,/500 metres/);
  assert.equal(project.radius,1000);
 }
})().catch(error=>{console.error(error);process.exitCode=1});
""", 'operations.js', page)


def test_results_history_and_corrected_transfer_notice():
    run_js(r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const elements={},listeners={},entries=[];
const element=id=>elements[id]||=({value:'',textContent:'',hidden:false,
 add(option){if(!this.value)this.value=option.value},querySelectorAll:()=>[]});
const row={detection_id:'one',date:'2020-01-24',dark_status:'AIS_PRESENT',ais_status_text:'two',
 confidence:.73,ais_vessels:[{MMSI:1,VesselType:80,Length:null,SOG:0}],
 human_review:{verdict:'not_sts',training_label:null,reviewer:'tester'}};
const runs=[{directory:'/new',description:'radius 500 m'},{directory:'/old',description:'radius 70 m'}];
const location={href:'http://localhost/results?directory=%2Fold',search:'?directory=%2Fold'};
function move(url){location.href=String(url);location.search=new URL(url).search}
const P={logInteraction(){},esc:String,fmt:(value,digits)=>Number(value).toFixed(digits),setStatus:(el,message)=>el.textContent=message,
 api:async(path,options)=>{
  if(path==='/api/results')return {runs};
  if(path.startsWith('/api/results?'))return {detections:[row]};
  if(path.endsWith('/review')){row.human_review.training_label=options.body.training_label;return row.human_review}
  if(path.endsWith('/improve')){
   if(!row.human_review.training_label)throw Error('A human training label is required');
   row.human_review.transfer={image_key:'one'};return row.human_review.transfer;
  }
  throw Error(path);
 }};
const context={window:{Pipeline:P,addEventListener:(name,fn)=>listeners[name]=fn},document:{getElementById:element,querySelectorAll:()=>[]},location,URLSearchParams,URL,
 history:{pushState(a,b,url){entries.push(String(url));move(url)},replaceState(a,b,url){move(url)}},Option:function(label,value){this.value=value}};
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 await vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);
 assert.equal(element('runs').value,'/old');
 assert.match(element('vessels').textContent,/Tanker \(AIS code 80\)/);
 assert.match(element('vessels').textContent,/Length: Not reported/);
 assert.match(element('vessels').textContent,/Speed: 0 knots/);
 assert.match(element('confidenceValue').textContent,/0.73/);
 assert.match(element('resultReading').textContent,/not classified as dark/);
 element('runs').value='/new';element('runs').onchange();await tick();
 assert.equal(new URL(location.href).searchParams.get('directory'),'/new');
 assert.equal(entries.length,1);
 move('http://localhost/results?directory=%2Fold');listeners.popstate();await tick();
 assert.equal(element('runs').value,'/old');
 element('negativeLabel').value='';await element('improve').onclick();
 assert.match(element('notice').textContent,/human training label is required/);
 element('negativeLabel').value='ship_non_sts';await element('improve').onclick();
 assert.equal(element('notice').textContent,'Reviewed crop sent to Improve.');
 assert.equal(element('improve').textContent,'Sent to Improve');
})().catch(error=>{console.error(error);process.exitCode=1});
""", 'results.js')
