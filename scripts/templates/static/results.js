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
  const source=(runs.find(run=>run.directory===directory)||{}).ais_source;
  const paper=r.dark_status==='AIS_INVALID'?'AIS_ERROR':r.dark_status==='AIS_MISSING'?(source?'AIS_ERROR':'DETECTED_ONLY'):r.dark_status;
  return paper+(paper!==r.dark_status?' / '+r.dark_status:'')+' — '+r.ais_status_text;
 }
 function select(id,focus=true){
  selected=id; const r=rows.find(x=>x.detection_id===id); if(!r)return;
  P.logInteraction('select_candidate',{detection_id:id});
  $('detail').hidden=false; $('candidateName').textContent=id;
  $('crop').src='/api/results/crop?'+query(); $('aisStatus').textContent=statusText(r);
  $('coordinates').textContent='GPS center: '+r.center_lat+' latitude, '+r.center_lon+' longitude (WGS84).';
  $('vessels').textContent=(r.ais_vessels||[]).map(v=>['MMSI: '+v.MMSI,'Name: '+(v.VesselName||'unknown'),'Position: '+v.LAT+', '+v.LON,'Distance: '+v.distance_m+' m','Time: '+v.BaseDateTime,'Type: '+(v.VesselType||'unknown'),'Speed: '+(v.SOG||'unknown')+' knots','Course: '+(v.COG||'unknown')+' degrees','Length: '+(v.Length||'unknown')+' m'].join('\n')).join('\n\n')||r.ais_error||'No matching AIS vessels.';
  const review=r.human_review||{};
  $('verdictStatus').textContent=review.verdict?review.verdict.replaceAll('_',' ')+' — '+review.reviewer:'No human verdict yet.';
  $('negativeField').hidden=review.verdict!=='not_sts'; $('negativeLabel').value=review.training_label||'';
  $('improve').disabled=!!review.transfer||!['real_sts','not_sts'].includes(review.verdict);
  $('improve').textContent=review.transfer?'Sent to Improve':'Send reviewed crop to Improve';
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
   L.circleMarker([v.LAT,v.LON],{color:'#bb701e',radius:5}).addTo(evidence).bindPopup(popup);
  });
  candidates.eachLayer(marker=>marker.bringToFront());
  if(focus)map.fitBounds(L.latLng(r.center_lat,r.center_lon).toBounds((r.ais_radius_m||1000)*3));
 }
 async function refresh(){
  directory=$('runs').value;if(!directory)return;
  rows=(await api('/api/results?'+new URLSearchParams({directory}))).detections;
  $('candidates').innerHTML=rows.map((r,i)=>'<tr><td><button data-index="'+i+'">'+P.esc(r.date+' / '+(i+1))+'</button></td><td>'+P.esc(statusText(r))+'</td><td>'+P.esc((r.human_review||{}).verdict||'Unreviewed')+'</td></tr>').join('')||'<tr><td colspan="3">NO_STS — no STS candidates detected in this run.</td></tr>';
  $('candidates').querySelectorAll('button').forEach(b=>b.onclick=()=>select(rows[Number(b.dataset.index)].detection_id));
  $('exports').innerHTML=['csv','json','geojson'].map(format=>'<a style="margin-right:16px" href="/api/results/export?'+new URLSearchParams({directory,format})+'">'+format.toUpperCase()+'</a>').join('');
  if(map){
   candidates.clearLayers();evidence.clearLayers();
   rows.forEach((r,i)=>{
    const popup=document.createElement('div'),text=document.createElement('p'),button=document.createElement('button');
    text.textContent='Satellite candidate '+(i+1)+' / '+r.date+' / '+r.center_lat+', '+r.center_lon;
    button.textContent='Inspect STS event';button.onclick=()=>select(r.detection_id);popup.append(text,button);
    L.circleMarker([r.center_lat,r.center_lon],{color:'#155b46',radius:8,className:'candidate-marker'}).addTo(candidates).bindPopup(popup).on('click',()=>select(r.detection_id,false));
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
 }catch(e){error(e);}};
 $('runs').onchange=()=>{selected=null;P.logInteraction('select_result_run',{directory:$('runs').value});refresh().catch(error);};
 try{
  runs=(await api('/api/results')).runs;
  runs.forEach(r=>$('runs').add(new Option((r.started_at||r.run_id)+' / '+r.directory.split(/[\\/]/).pop(),r.directory)));
  const requested=new URLSearchParams(location.search).get('directory');
  if(runs.some(r=>r.directory===requested))$('runs').value=requested;
  await refresh();if(!runs.length)P.setStatus($('notice'),'No completed detection runs in this workspace. Start with Scan Images.');
 }catch(e){error(e);}
})();
