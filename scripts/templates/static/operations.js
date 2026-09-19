/* Detect workflow; all writes share the CLI's backend helpers. */
(async function () {
  'use strict';
  const P = window.Pipeline, page = document.body.dataset.page;
  const $ = id => document.getElementById(id);
  let dashboard, selectedJob = new URLSearchParams(location.search).get('job'), map, rectangle, corners = null;
  const notify = (message, kind) => P.setStatus($('notice'), message, kind);
  const request = async (path, body) => P.api(path, body === undefined ? {} : {method: 'POST', body});
  function error(err) { notify(err.message, 'error'); P.logInteraction('operation_error', {message: err.message}); }
  async function save(values) { const result = await request('/api/operations/project', values); dashboard.project = result; return result; }
  async function launch(kind) {
    P.logInteraction('start_' + kind, {start: dashboard.project.start, end: dashboard.project.end});
    const job = await request('/api/operations/jobs', {kind});
    location.href = '/jobs?job=' + encodeURIComponent(job.job_id);
  }
  function drawBounds(bbox, fit) {
    const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
    if (rectangle) rectangle.setBounds(bounds); else rectangle = L.rectangle(bounds, {color: '#155b46', weight: 2, fillOpacity: .12}).addTo(map);
    if (fit) map.fitBounds(L.latLngBounds(bounds).pad(.7), {padding: [28,28]});
    ['west','south','east','north'].forEach((key, i) => $(key).value = Number(bbox[i]).toFixed(5));
  }
  async function refreshJobs() {
    const data = await request('/api/operations/jobs');
    $('jobRows').innerHTML = data.jobs.map(job => '<tr><td><button class="job-select" data-id="'+P.esc(job.job_id)+'">'+P.esc(job.kind)+'</button></td><td>'+P.esc(P.shortTime(job.created_at))+'</td><td>'+P.esc(job.status)+'</td></tr>').join('') || '<tr><td colspan="3">No jobs yet. Start with an area and some dates.</td></tr>';
    $('jobRows').querySelectorAll('button').forEach(button => button.onclick = () => { selectedJob = button.dataset.id; P.logInteraction('select_job', {job_id: selectedJob}); refreshJobs().catch(error); });
    if (!selectedJob && data.jobs.length) selectedJob = data.jobs[0].job_id;
    if (!selectedJob) return;
    const job = await request('/api/operations/jobs/' + encodeURIComponent(selectedJob));
    $('jobTitle').textContent = job.kind + ' / ' + job.status;
    $('jobLog').textContent = (job.log_tail || []).join('\n') || job.error || 'Waiting for output…';
    $('jobOutput').replaceChildren();
    if (job.error) { const text = document.createElement('p'); text.textContent = job.error; $('jobOutput').append(text); }
    if (job.kind === 'run' && job.status === 'completed') {
      for (const extension of ['csv', 'json']) {
        const link = document.createElement('a'); link.href = '/api/operations/jobs/' + job.job_id + '/output?format=' + extension; link.textContent = 'Download ' + extension.toUpperCase(); $('jobOutput').append(link);
      }
      const text = document.createElement('p'); text.textContent = job.result.summary.detections + ' candidate detections'; $('jobOutput').append(text);
    }
  }
  try {
    dashboard = await request('/api/operations/dashboard');
    const settings = dashboard.project;
    $(page + 'Panel').hidden = false;
    if (page === 'home') {
      $('modelName').textContent = dashboard.model ? dashboard.model.filename : 'No model selected';
      $('modelInfo').textContent = dashboard.model_error || (dashboard.model.metrics && dashboard.model.metrics.mAP50 ? 'Validation mAP50: ' + P.fmt(dashboard.model.metrics.mAP50, 3) : 'Used by Detect and Improve.');
      $('inventorySummary').textContent = dashboard.inventory.images + ' images / ' + dashboard.inventory.ais_dir + ' AIS files';
      $('pendingReviews').textContent = dashboard.pending_reviews + ' images waiting for review in Improve.';
      $('recentRuns').innerHTML = dashboard.jobs.slice(0,3).map(j => '<li><a href="/jobs?job='+encodeURIComponent(j.job_id)+'">'+P.esc(j.kind+' / '+j.status)+'</a></li>').join('');
      $('jobsSummary').textContent = dashboard.jobs.filter(j => !['completed','failed','cancelled'].includes(j.status)).length + ' active jobs';
    }
    if (page === 'area') {
      map = L.map('areaMap', {scrollWheelZoom: false}).setView([29.14,-94.65], 10);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(map);
      drawBounds(settings.bbox, true);
      Object.entries(dashboard.presets).forEach(([name]) => $('preset').add(new Option(name, name)));
      $('preset').onchange = () => { if (dashboard.presets[$('preset').value]) drawBounds(dashboard.presets[$('preset').value], true); P.logInteraction('choose_area_preset', {preset: $('preset').value}); };
      ['west','south','east','north'].forEach(id => $(id).onchange = () => drawBounds(['west','south','east','north'].map(key => Number($(key).value)), false));
      ['start','end','images'].forEach(key => $(key).value = settings[key]);
      $('drawArea').onclick = () => { corners = []; $('mapHint').textContent = 'Click the first corner, then the opposite corner.'; P.logInteraction('draw_area_started', {}); };
      map.on('click', e => { if (corners === null) return; corners.push(e.latlng); if (corners.length === 2) { const b = L.latLngBounds(corners); drawBounds([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], false); corners = null; $('mapHint').textContent = 'Area selected. Save to continue.'; P.logInteraction('draw_area_done', {}); } });
      $('mapExtent').onclick = () => { const b = map.getBounds(); drawBounds([b.getWest(),b.getSouth(),b.getEast(),b.getNorth()], false); P.logInteraction('use_map_extent', {}); };
      $('areaForm').onsubmit = async e => { e.preventDefault(); try { await save({bbox: ['west','south','east','north'].map(k => Number($(k).value)), start: $('start').value, end: $('end').value, images: $('images').value}); P.logInteraction('save_area', dashboard.project); location.href = '/data'; } catch (err) {error(err);} };
    }
    if (page === 'data') {
      $('imageCount').textContent = dashboard.inventory.images + ' image files'; $('imagePath').textContent = settings.images;
      $('aisCount').textContent = dashboard.inventory.ais_dir + ' AIS files'; $('aisPath').value = settings.ais_dir; $('maxCloud').value = settings.max_cloud;
      $('dateSummary').textContent = settings.start + ' → ' + settings.end;
      $('saveAis').onclick = () => save({ais_dir: $('aisPath').value}).then(() => notify('AIS source saved.')).catch(error);
      $('getAis').onclick = () => launch('ais').catch(error);
      $('getImagery').onclick = () => save({max_cloud: Number($('maxCloud').value)}).then(() => launch('imagery')).catch(error);
    }
    if (page === 'run') {
      $('runModel').textContent = dashboard.model ? 'Current model: ' + dashboard.model.filename : dashboard.model_error;
      $('runButton').disabled = !dashboard.model || !dashboard.inventory.images;
      $('confidence').value = settings.confidence; $('radius').value = settings.radius;
      $('runInputs').textContent = dashboard.inventory.images + ' image files selected, ' + settings.start + ' to ' + settings.end + '. AIS source: ' + settings.ais_dir;
      $('runForm').onsubmit = async e => { e.preventDefault(); $('runButton').disabled = true; try { await save({confidence: Number($('confidence').value), radius: Number($('radius').value)}); await launch('run'); } catch (err) { $('runButton').disabled = false; error(err); } };
    }
    if (page === 'jobs') { await refreshJobs(); $('refreshJobs').onclick = () => refreshJobs().catch(error); setInterval(() => refreshJobs().catch(error), 2500); }
    document.querySelectorAll('input').forEach(input => input.addEventListener('change', () => P.logInteraction('change_setting', {field: input.id, value: input.value})));
  } catch (err) { error(err); }
})();
