(async function(){
 'use strict';
 const P=window.Pipeline, $=id=>document.getElementById(id);
 const api=(path,body)=>P.api(path,body===undefined?{}:{method:'POST',body});
 const error=e=>{P.setStatus($('notice'),e.message,'error');P.logInteraction('results_error',{message:e.message});};
 let rows=[], selected=null, directory=null, runs=[];
 let map, candidates, evidence;
 if(window.L){
  map=L.map('resultMap',{scrollWheelZoom:false}).setView([29,-95],5);
  candidates=L.layerGroup().addTo(map); evidence=L.layerGroup().addTo(map);
  const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
  tiles.on('tileerror',()=>{$('resultMapStatus').textContent='Map background unavailable. Candidate coordinates, markers and evidence remain available.';});
 }else $('resultMapStatus').textContent='Map viewer unavailable. Candidate coordinates and evidence remain available below.';
 const query=()=>new URLSearchParams({directory,detection_id:selected||''});
 function statusText(r){
  const paper=r.dark_status==='AIS_INVALID'?'AIS_ERROR':r.dark_status==='AIS_MISSING'?'DETECTED_ONLY':r.dark_status;
  return paper+(paper!==r.dark_status?' (stored as '+r.dark_status+')':'')+' — '+r.ais_status_text;
 }
 const verdictLabel = value => ({real_sts:'Real STS',not_sts:'Not STS',unclear:'Unclear'}[value] || 'Unreviewed');
 // USCG AIS Encoding Guide: https://www.navcen.uscg.gov/sites/default/files/pdf/AIS/AISGuide.pdf
 function vesselType(value){
  const code=Number(value);
  if(!Number.isInteger(code)||code<=0)return 'Not reported';
  const names={30:'Fishing',31:'Towing',32:'Towing (large tow)',33:'Dredging or underwater operations',34:'Diving operations',35:'Military operations',36:'Sailing',37:'Pleasure craft',50:'Pilot vessel',51:'Search and rescue',52:'Tug',53:'Port tender',54:'Anti-pollution vessel',55:'Law enforcement',58:'Medical transport',59:'Special transport'};
  const family=code>=20&&code<=29?'Wing-in-ground craft':code>=40&&code<=49?'High-speed craft':code>=60&&code<=69?'Passenger ship':code>=70&&code<=79?'Cargo ship':code>=80&&code<=89?'Tanker':code>=90&&code<=99?'Other vessel':null;
  return (names[code]||family||'Unclassified vessel')+' (AIS code '+code+')';
 }
 function measurement(value,unit,positive=false){
  if(value===null||value===undefined||String(value).trim()===''||!Number.isFinite(Number(value))||(positive&&Number(value)<=0))return 'Not reported';
  return value+' '+unit;
 }
 function reading(r){
  if(r.dark_status==='AIS_PRESENT')return 'AIS shows two or more vessels nearby on the same day — this candidate is not classified as dark.';
  if(r.dark_status==='PARTIALLY_DARK')return 'AIS shows one vessel nearby on the same day — this candidate is classified as partially dark.';
  if(r.dark_status==='FULLY_DARK')return 'AIS shows no vessels nearby on the same day — this candidate is classified as fully dark.';
  return 'AIS records are unavailable or could not be verified — vessel presence is unknown.';
 }
 function select(id,focus=true){
  selected=id; const r=rows.find(x=>x.detection_id===id); if(!r)return;
  P.logInteraction('select_candidate',{detection_id:id});
  $('detail').hidden=false; $('candidateName').textContent='Candidate '+(rows.indexOf(r)+1)+' / '+r.date;
  $('crop').src='/api/results/crop?'+query(); $('aisStatus').textContent=statusText(r);
  $('coordinates').textContent='GPS center: '+r.center_lat+' latitude, '+r.center_lon+' longitude (WGS84).';
  $('resultReading').textContent=reading(r);
  $('confidenceValue').textContent='Detector confidence: '+P.fmt(r.confidence,2)+' (model score from 0 to 1; not a probability of an actual transfer).';
  $('vessels').textContent=(r.ais_vessels||[]).map(v=>['MMSI: '+v.MMSI,'Name: '+(v.VesselName||'Not reported'),'Position: '+v.LAT+', '+v.LON,'Distance: '+measurement(v.distance_m,'m'),'Time (UTC): '+(v.BaseDateTime||'Not reported'),'Vessel type: '+vesselType(v.VesselType),'Speed: '+measurement(v.SOG,'knots'),'Course: '+measurement(v.COG,'degrees'),'Length: '+measurement(v.Length,'m',true)].join('\n')).join('\n\n')||r.ais_error||'No matching AIS vessels.';
  const review=r.human_review||{};
  $('verdictStatus').textContent=review.verdict?verdictLabel(review.verdict)+' — '+review.reviewer:'No human verdict yet.';
  $('negativeField').hidden=review.verdict!=='not_sts'; $('negativeLabel').value=review.training_label||'';
  $('improve').disabled=!!review.transfer||!['real_sts','not_sts'].includes(review.verdict);
  $('improve').textContent=review.transfer?'Sent to Improve':'Send reviewed crop to Improve';
  $('reviewHelp').textContent=review.transfer?'This verdict is locked because the crop has entered the training dataset. Use Open in Improve to correct its training review.':'Only an explicit human label enters training. Images from the same observation stay together when splitting datasets.';
  let improveLink=$('openInImprove');
  if(!improveLink){improveLink=document.createElement('a');improveLink.id='openInImprove';improveLink.textContent='Open in Improve';improveLink.style.marginLeft='12px';$('improve').after(improveLink);}
  improveLink.hidden=!review.transfer;
  if(review.transfer)improveLink.href='/manual-review?image_id='+encodeURIComponent(review.transfer.image_key);
  document.querySelectorAll('[data-verdict]').forEach(b=>b.disabled=!!review.transfer);
  if(!map)return;
  evidence.clearLayers();
  L.circle([r.center_lat,r.center_lon],{radius:r.ais_radius_m||1000,color:'#155b46',weight:1,fillOpacity:.04,interactive:false}).addTo(evidence);
  (r.ais_vessels||[]).forEach(v=>{
   const popup=document.createElement('p');popup.textContent='Same-day AIS / MMSI '+v.MMSI+' / '+(v.VesselName||'unknown name')+' / '+v.BaseDateTime+' / '+v.distance_m+' m from candidate';
   L.circleMarker([v.LAT,v.LON],{color:'#9b4a00',fillColor:'#ffac45',fillOpacity:1,weight:2,radius:5,className:'ais-marker'}).addTo(evidence).bindPopup(popup);
  });
  candidates.eachLayer(marker=>marker.bringToFront());
  if(focus)map.fitBounds(L.latLng(r.center_lat,r.center_lon).toBounds((r.ais_radius_m||1000)*3));
 }
 async function refresh(){
  directory=$('runs').value;if(!directory)return;
  rows=(await api('/api/results?'+new URLSearchParams({directory}))).detections;
  $('candidates').innerHTML=rows.map((r,i)=>'<tr><td data-label="Candidate"><button data-index="'+i+'">'+P.esc(r.date+' / '+(i+1))+'</button></td><td data-label="AIS within radius that day">'+P.esc(statusText(r))+'</td><td data-label="Image verdict">'+P.esc(verdictLabel((r.human_review||{}).verdict))+'</td></tr>').join('')||'<tr><td colspan="3">NO_STS — no STS candidates detected in this run.</td></tr>';
  $('candidates').querySelectorAll('button').forEach(b=>b.onclick=()=>select(rows[Number(b.dataset.index)].detection_id));
  $('exports').innerHTML=['csv','json','geojson'].map(format=>'<a style="margin-right:16px" href="/api/results/export?'+new URLSearchParams({directory,format})+'">'+format.toUpperCase()+'</a>').join('');
  if(map){
   candidates.clearLayers();evidence.clearLayers();
   rows.forEach((r,i)=>{
    const popup=document.createElement('div'),text=document.createElement('p'),button=document.createElement('button');
    text.textContent='Satellite candidate '+(i+1)+' / '+r.date+' / '+r.center_lat+', '+r.center_lon;
    button.textContent='Inspect candidate '+(i+1);text.textContent+=' / '+reading(r);button.onclick=()=>select(r.detection_id);popup.append(text,button);
    L.circleMarker([r.center_lat,r.center_lon],{color:'#155b46',fill:false,weight:3,radius:13,className:'candidate-marker'}).addTo(candidates).bindPopup(popup).on('click',()=>select(r.detection_id,false));
   });
   if(rows.length)map.fitBounds(L.latLngBounds(rows.map(r=>[r.center_lat,r.center_lon])).pad(.2),{maxZoom:13});
  }
  if(rows.length)select(rows.some(r=>r.detection_id===selected)?selected:rows[0].detection_id,false);else $('detail').hidden=true;
 }
 document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>api('/api/results/review',{directory,detection_id:selected,verdict:b.dataset.verdict}).then(refresh).catch(error));
 $('improve').onclick=async()=>{try{
  const row=rows.find(r=>r.detection_id===selected);
  if(row.human_review.verdict==='not_sts')await api('/api/results/review',{directory,detection_id:selected,verdict:'not_sts',training_label:$('negativeLabel').value||null});
  await api('/api/results/improve',{directory,detection_id:selected});await refresh();
  P.setStatus($('notice'),'Reviewed crop sent to Improve.','success');
 }catch(e){error(e);}};
 function rememberRun(replace=false){
  const url=new URL(location.href);url.searchParams.set('directory',$('runs').value);
  history[replace?'replaceState':'pushState'](null,'',url);
 }
 $('runs').onchange=()=>{selected=null;rememberRun();P.logInteraction('select_result_run',{directory:$('runs').value});refresh().catch(error);};
 window.addEventListener('popstate',()=>{
  const requested=new URLSearchParams(location.search).get('directory');
  if(runs.some(r=>r.directory===requested)){$('runs').value=requested;selected=null;refresh().catch(error);}
 });
 try{
  runs=(await api('/api/results')).runs;
  runs.forEach(r=>$('runs').add(new Option(r.description||P.shortTime(r.started_at)||'Detection run',r.directory)));
  const requested=new URLSearchParams(location.search).get('directory');
  if(runs.some(r=>r.directory===requested))$('runs').value=requested;
  if(runs.length)rememberRun(true);
  await refresh();if(!runs.length)P.setStatus($('notice'),'No completed detection runs in this workspace. Start with Scan Images.');
 }catch(e){error(e);}
})();
