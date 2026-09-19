(async function(){
 'use strict';
 const P=window.Pipeline, $=id=>document.getElementById(id);
 const api=(path,body)=>P.api(path,body===undefined?{}:{method:'POST',body});
 const error=e=>{P.setStatus($('notice'),e.message,'error');P.logInteraction('results_error',{message:e.message});};
 let rows=[], selected=null, directory=null;
 const map=L.map('resultMap',{scrollWheelZoom:false}).setView([29,-95],5), markers=L.layerGroup().addTo(map);
 L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
 const query=()=>new URLSearchParams({directory,detection_id:selected||''});
 function select(id){
  selected=id; const r=rows.find(x=>x.detection_id===id); if(!r)return;
  P.logInteraction('select_candidate',{detection_id:id});
  $('detail').hidden=false; $('candidateName').textContent=id;
  $('crop').src='/api/results/crop?'+query(); $('aisStatus').textContent=r.ais_status_text;
  $('vessels').textContent=(r.ais_vessels||[]).map(v=>[v.MMSI,v.VesselName, v.distance_m+' m',v.BaseDateTime].filter(Boolean).join(' / ')).join('\n')||r.ais_error||'No matching AIS vessels.';
  const review=r.human_review||{};
  $('verdictStatus').textContent=review.verdict?review.verdict.replaceAll('_',' ')+' — '+review.reviewer:'No human verdict yet.';
  $('negativeField').hidden=review.verdict!=='not_sts'; $('negativeLabel').value=review.training_label||'';
  $('improve').disabled=!!review.transfer||!['real_sts','not_sts'].includes(review.verdict);
  $('improve').textContent=review.transfer?'Sent to Improve':'Send reviewed crop to Improve';
  document.querySelectorAll('[data-verdict]').forEach(b=>b.disabled=!!review.transfer);
  markers.clearLayers();
  L.circleMarker([r.center_lat,r.center_lon],{color:'#155b46',radius:8}).addTo(markers).bindTooltip('Satellite candidate');
  L.circle([r.center_lat,r.center_lon],{radius:r.ais_radius_m||1000,color:'#155b46',weight:1,fillOpacity:.04}).addTo(markers);
  (r.ais_vessels||[]).forEach(v=>L.circleMarker([v.LAT,v.LON],{color:'#bb701e',radius:5}).addTo(markers).bindTooltip(String(v.MMSI)));
  map.fitBounds(L.latLng(r.center_lat,r.center_lon).toBounds((r.ais_radius_m||1000)*3));
 }
 async function refresh(){
  directory=$('runs').value;if(!directory)return;
  rows=(await api('/api/results?'+new URLSearchParams({directory}))).detections;
  $('candidates').innerHTML=rows.map((r,i)=>'<tr><td><button data-index="'+i+'">'+P.esc(r.date+' / '+(i+1))+'</button></td><td>'+P.esc(r.ais_status_text)+'</td><td>'+P.esc((r.human_review||{}).verdict||'Unreviewed')+'</td></tr>').join('')||'<tr><td colspan="3">No detections in this run.</td></tr>';
  $('candidates').querySelectorAll('button').forEach(b=>b.onclick=()=>select(rows[Number(b.dataset.index)].detection_id));
  $('exports').innerHTML=['csv','json','geojson'].map(format=>'<a style="margin-left:16px" href="/api/results/export?'+new URLSearchParams({directory,format})+'">'+format.toUpperCase()+'</a>').join('');
  if(rows.length)select(rows.some(r=>r.detection_id===selected)?selected:rows[0].detection_id);else{$('detail').hidden=true;markers.clearLayers();}
 }
 document.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>api('/api/results/review',{directory,detection_id:selected,verdict:b.dataset.verdict}).then(refresh).catch(error));
 $('improve').onclick=async()=>{try{
  const row=rows.find(r=>r.detection_id===selected);
  if(row.human_review.verdict==='not_sts')await api('/api/results/review',{directory,detection_id:selected,verdict:'not_sts',training_label:$('negativeLabel').value||null});
  await api('/api/results/improve',{directory,detection_id:selected});await refresh();
 }catch(e){error(e);}};
 $('runs').onchange=()=>{selected=null;refresh().catch(error);};
 try{const runs=(await api('/api/results')).runs;runs.forEach(r=>$('runs').add(new Option((r.started_at||r.run_id)+' / '+r.directory.split(/[\\/]/).pop(),r.directory)));await refresh();if(!runs.length)P.setStatus($('notice'),'No completed detection runs in this workspace.');}catch(e){error(e);}
})();
