/* Detect workflow; all writes share the CLI's backend helpers. */
(async function () {
  'use strict';
  const P = window.Pipeline, page = document.body.dataset.page;
  const $ = id => document.getElementById(id);
  const subtitles = {
    home: 'One workspace for satellite detections, vessel records, and reviewed training images.',
    area: 'Select a search area, then choose the dates to investigate.',
    data: 'Prepare imagery and vessel records for your saved search.',
    run: 'Confirm your inputs and start the satellite and AIS comparison.',
    jobs: 'Follow background work and download completed results.'
  };
  if ($('pageSubtitle')) $('pageSubtitle').textContent = subtitles[page] || '';
  if ($('pageContext') && ['home', 'jobs'].includes(page)) $('pageContext').textContent = 'Workspace / ' + (page === 'home' ? 'Overview' : 'Background work');
  document.querySelectorAll('.detect-steps a').forEach(link => {
    if (link.dataset.step === page) link.setAttribute('aria-current', 'step');
  });
  document.querySelectorAll('.app-navigation a').forEach(link => {
    const activePath = page === 'home' ? '/' : page === 'jobs' ? '/jobs' : '/detect';
    if (link.getAttribute('href') === activePath && !link.classList.contains('app-brand')) link.setAttribute('aria-current', 'page');
  });
  let dashboard, selectedJob = new URLSearchParams(location.search).get('job'), displayedJob, map, rectangle, corners = null;
  const jobNames = {run: 'Detection', ais: 'AIS download', imagery: 'Imagery download', training: 'Model training', evaluation: 'Evaluation'};
  const countLabel = (count, noun) => count + ' ' + noun + (count === 1 ? '' : 's');
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
    if (bbox.length !== 4 || !bbox.every(Number.isFinite) || bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) return;
    const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
    if (map) {
      if (rectangle) rectangle.setBounds(bounds); else rectangle = L.rectangle(bounds, {color: '#155b46', weight: 2, fillOpacity: .12}).addTo(map);
      if (fit) map.fitBounds(L.latLngBounds(bounds).pad(.7), {padding: [28,28]});
    }
    ['west','south','east','north'].forEach((key, i) => $(key).value = Number(bbox[i]).toFixed(5));
    if ($('areaSummary')) $('areaSummary').textContent = 'Area selected. Use the map or edit coordinates to adjust it.';
  }
  function initializeAreaMap() {
    const status = $('mapStatus'), retry = $('retryMap');
    function backgroundStatus(message, unavailable) {
      status.textContent = message;
      status.classList.toggle('map-warning', unavailable);
      retry.hidden = !unavailable;
    }
    if (!window.L) {
      backgroundStatus('The map viewer could not load. Choose a US area preset or enter coordinates to continue.', true);
      retry.textContent = 'Reload map viewer';
      retry.onclick = () => location.reload();
      $('drawArea').disabled = $('mapExtent').disabled = true;
      $('mapHint').textContent = 'Presets and coordinates remain available.';
      P.logInteraction('area_map_unavailable', {reason: 'viewer_assets'});
      return;
    }
    map = L.map('areaMap', {scrollWheelZoom: false}).setView([29.14,-94.65], 10);
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18, referrerPolicy: 'strict-origin-when-cross-origin',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    });
    let failures = 0, timeout;
    const unavailable = () => backgroundStatus('Map background unavailable. Check the internet connection, then retry. You can still choose a preset or enter coordinates.', true);
    tiles.on('loading', () => {
      clearTimeout(timeout);
      timeout = setTimeout(unavailable, 10000);
      if (!failures) backgroundStatus('Loading map background…', false);
    });
    tiles.on('tileerror', () => {
      if (!failures++) P.logInteraction('area_map_tile_error', {provider: 'OpenStreetMap'});
      unavailable();
    });
    tiles.on('load', () => {
      clearTimeout(timeout);
      const missing = Array.from($('areaMap').querySelectorAll('.leaflet-tile')).some(tile => !tile.complete || !tile.naturalWidth);
      if (missing) unavailable();
      else { failures = 0; backgroundStatus('Map background: OpenStreetMap. Internet connection required.', false); }
    });
    retry.onclick = () => {
      failures = 0;
      P.logInteraction('retry_area_map', {});
      tiles.redraw();
    };
    tiles.addTo(map);
  }
  async function refreshJobs() {
    const data = await request('/api/operations/jobs');
    if (!selectedJob && data.jobs.length) selectedJob = data.jobs[0].job_id;
    $('jobRows').innerHTML = data.jobs.map(job => '<tr><td><button class="job-select" data-id="'+P.esc(job.job_id)+'"'+(job.job_id === selectedJob ? ' aria-current="true"' : '')+'>'+P.esc(job.description || jobNames[job.kind] || job.kind)+'</button></td><td>'+P.esc(job.created_at.replace('T', ' ').slice(0,19))+'</td><td><span class="job-state job-state-'+P.esc(job.status)+'">'+P.esc(job.status)+'</span></td></tr>').join('') || '<tr><td colspan="3">No jobs yet. Start with an area and some dates.</td></tr>';
    $('jobRows').querySelectorAll('button').forEach(button => button.onclick = () => { selectedJob = button.dataset.id; P.logInteraction('select_job', {job_id: selectedJob}); refreshJobs().catch(error); });
    if (!selectedJob) return;
    const job = await request('/api/operations/jobs/' + encodeURIComponent(selectedJob));
    $('jobTitle').textContent = (job.description || jobNames[job.kind] || job.kind) + ' · ' + job.status;
    if ($('jobDescription')) $('jobDescription').textContent = job.status === 'completed' ? 'This job completed successfully.' : job.status === 'failed' ? 'This job stopped with an error. Check the details below.' : job.status === 'cancelled' ? 'This job was cancelled.' : 'This job continues in the background. Progress updates automatically.';
    if ($('jobDetails') && displayedJob !== selectedJob) $('jobDetails').open = !['completed', 'failed', 'cancelled'].includes(job.status);
    displayedJob = selectedJob;
    $('jobLog').textContent = (job.log_tail || []).join('\n') || job.error || (job.status === 'completed' ? 'Completed successfully. No additional log messages.' : 'Waiting for output…');
    $('jobLog').scrollTop = $('jobLog').scrollHeight;
    $('jobOutput').replaceChildren();
    if (['ais', 'imagery'].includes(job.kind) && !['completed', 'failed', 'cancelled'].includes(job.status)) {
      const cancel = document.createElement('button'); cancel.type = 'button';
      cancel.textContent = job.cancel_requested ? 'Cancelling…' : 'Cancel download'; cancel.disabled = job.cancel_requested;
      cancel.onclick = async () => { try {
        cancel.disabled = true;
        await request('/api/operations/jobs/' + job.job_id + '/cancel', {});
        P.logInteraction('cancel_download', {job_id: job.job_id});
        await refreshJobs();
      } catch (err) { cancel.disabled = false; error(err); } };
      $('jobOutput').append(cancel);
      const progress = document.createElement('p');
      progress.textContent = (job.log_tail || []).filter(line => line.startsWith('Download progress:')).pop() || 'Connecting to the data provider…';
      $('jobOutput').append(progress);
    }
    if (job.error) { const text = document.createElement('p'); text.textContent = job.error; $('jobOutput').append(text); }
    if (job.kind === 'run' && job.status === 'completed') {
      for (const extension of ['csv', 'json']) {
        const link = document.createElement('a'); link.href = '/api/operations/jobs/' + job.job_id + '/output?format=' + extension; link.textContent = 'Download ' + extension.toUpperCase(); $('jobOutput').append(link);
      }
      const summary = job.result.summary;
      const text = document.createElement('p'); text.textContent = summary.detections ? countLabel(summary.detections, 'candidate detection') : 'NO_STS — no STS candidates detected.'; $('jobOutput').append(text);
      const resultsLink = document.createElement('a'); resultsLink.href = '/results?' + new URLSearchParams({directory: job.result.output_dir}); resultsLink.textContent = 'Open STS Events & Live Map'; $('jobOutput').append(resultsLink);
      const presence = document.createElement('p');
      presence.textContent = 'Same-day AIS: FULLY_DARK ' + (summary.FULLY_DARK || 0) + ' (no nearby vessels); PARTIALLY_DARK ' + (summary.PARTIALLY_DARK || 0) + ' (one); AIS_PRESENT ' + (summary.AIS_PRESENT || 0) + ' (two or more).';
      $('jobOutput').append(presence);
      const unknown = (summary.AIS_MISSING || 0) + (summary.AIS_INVALID || 0);
      if (unknown) { const warning = document.createElement('p'); warning.className = 'status error'; warning.textContent = 'AIS presence is unknown for ' + unknown + ' candidates because records are missing or invalid. Check the AIS source on Data.'; $('jobOutput').append(warning); }
    }
  }
  try {
    dashboard = await request('/api/operations/dashboard');
    const settings = dashboard.project;
    document.querySelectorAll('.use-example').forEach(button => button.onclick = async () => {
      button.disabled = true;
      try {
        await request('/api/operations/example', {});
        P.logInteraction('use_bundled_example', {date: '2020-01-24'});
        location.href = '/data';
      } catch (err) { button.disabled = false; error(err); }
    });
    $(page + 'Panel').hidden = false;
    if (page === 'home') {
      const totals = dashboard.detection_summary;
      $('detectionTotals').textContent = countLabel(totals.runs, 'completed scan') + ' / ' + countLabel(totals.detections, 'candidate detection') + '. FULLY_DARK: ' + totals.FULLY_DARK + '; PARTIALLY_DARK: ' + totals.PARTIALLY_DARK + '; AIS_PRESENT: ' + totals.AIS_PRESENT + '; DETECTED_ONLY (stored as AIS_MISSING): ' + totals.AIS_MISSING + '; AIS_ERROR (stored as AIS_INVALID): ' + totals.AIS_INVALID + '. Counts describe the selected workspace history.';
      P.renderModel($('modelDetails'), dashboard.model, dashboard.model_error);
      $('inventorySummary').textContent = countLabel(dashboard.inventory.images, 'image') + ' / ' + countLabel(dashboard.inventory.ais_dir, 'AIS file');
      const training = dashboard.training_inventory;
      $('trainingInventory').textContent = countLabel(training.images, 'training image') + ' in ' + countLabel(training.albums, 'album') + '; ' + training.completed + ' reviewed.';
      $('pendingReviews').textContent = countLabel(dashboard.pending_reviews, 'image') + ' waiting for review in Improve.';
      $('recentRuns').innerHTML = dashboard.jobs.slice(0,3).map(j => '<li><a href="/jobs?job='+encodeURIComponent(j.job_id)+'">'+P.esc((jobNames[j.kind] || j.kind)+' · '+j.status)+'</a></li>').join('');
      $('jobsSummary').textContent = countLabel(dashboard.jobs.filter(j => !['completed','failed','cancelled'].includes(j.status)).length, 'active job');
    }
    if (page === 'area') {
      initializeAreaMap();
      drawBounds(settings.bbox, true);
      Object.entries(dashboard.presets).forEach(([name]) => $('preset').add(new Option(name, name)));
      const currentPreset = Object.entries(dashboard.presets).find(([, bbox]) => bbox.every((value, i) => Math.abs(value - settings.bbox[i]) < .00001));
      if (currentPreset) $('preset').value = currentPreset[0];
      $('preset').onchange = () => { if (dashboard.presets[$('preset').value]) drawBounds(dashboard.presets[$('preset').value], true); P.logInteraction('choose_area_preset', {preset: $('preset').value}); };
      ['west','south','east','north'].forEach(id => $(id).onchange = () => {
        $('preset').value = '';
        const values = ['west','south','east','north'].map(key => $(key).value.trim());
        if (values.some(value => value === '')) return;
        drawBounds(values.map(Number), false);
      });
      ['start','end','images'].forEach(key => $(key).value = settings[key]);
      $('datePeriod').value = settings.date_period || (settings.start === settings.end ? 'date' : 'range');
      function applyDatePeriod() {
        const date = $('start').value, period = $('datePeriod').value;
        $('end').disabled = period === 'date';
        if (!date || period === 'range') return;
        const [year, month] = date.split('-').map(Number);
        if (period === 'date') $('end').value = date;
        if (period === 'month') {
          $('start').value = date.slice(0, 7) + '-01';
          $('end').value = date.slice(0, 7) + '-' + new Date(Date.UTC(year, month, 0)).getUTCDate();
        }
        if (period === 'year') { $('start').value = year + '-01-01'; $('end').value = year + '-12-31'; }
        P.logInteraction('choose_date_period', {period, start: $('start').value, end: $('end').value});
      }
      $('datePeriod').onchange = applyDatePeriod;
      applyDatePeriod();
      $('start').addEventListener('change', applyDatePeriod);
      $('end').addEventListener('change', () => { $('datePeriod').value = 'range'; });
      $('drawArea').onclick = () => { corners = []; $('mapHint').textContent = 'Click the first corner, then the opposite corner.'; P.logInteraction('draw_area_started', {}); };
      if (map) map.on('click', e => { if (corners === null) return; corners.push(e.latlng); if (corners.length === 2) { const b = L.latLngBounds(corners); $('preset').value = ''; drawBounds([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], false); corners = null; $('mapHint').textContent = 'Area selected. Save to continue.'; P.logInteraction('draw_area_done', {}); } });
      $('mapExtent').onclick = () => { const b = map.getBounds(); $('preset').value = ''; drawBounds([b.getWest(),b.getSouth(),b.getEast(),b.getNorth()], false); P.logInteraction('use_map_extent', {}); };
      $('areaForm').onsubmit = async e => {
        e.preventDefault(); P.setStatus($('areaError'), '');
        try {
          applyDatePeriod();
          const bbox = ['west','south','east','north'].map(k => {
            const value = $(k).value.trim();
            if (!value) throw new Error('Enter a coordinate for ' + k + '.');
            return Number(value);
          });
          await save({bbox, start: $('start').value, end: $('end').value, date_period: $('datePeriod').value, images: $('images').value});
          P.logInteraction('save_area', dashboard.project); location.href = '/data';
        } catch (err) {
          P.setStatus($('areaError'), err.message, 'err');
          $('areaCoordinates').open = true;
          P.logInteraction('save_area_error', {message: err.message});
        }
      };
    }
    if (page === 'data') {
      $('imageCount').textContent = countLabel(dashboard.inventory.images, 'image file'); $('imagePath').textContent = settings.images;
      $('aisCount').textContent = countLabel(dashboard.inventory.ais_dir, 'AIS file'); $('aisPath').value = settings.ais_dir; $('maxCloud').value = settings.max_cloud;
      $('dateSummary').textContent = settings.start + ' → ' + settings.end;
      $('saveAis').onclick = async () => { try {
        await save({ais_dir: $('aisPath').value});
        dashboard = await request('/api/operations/dashboard');
        $('aisCount').textContent = countLabel(dashboard.inventory.ais_dir, 'AIS file');
        notify('AIS source saved.');
      } catch (err) { error(err); } };
      $('getAis').onclick = () => launch('ais').catch(error);
      $('getImagery').disabled = !dashboard.imagery_credentials_ready;
      $('credentialsStatus').textContent = dashboard.imagery_credentials_ready ? 'Copernicus credentials are available to this server.' : 'Imagery download is unavailable until credentials are set and the server is restarted. Use the bundled example to continue now.';
      $('getImagery').onclick = () => save({max_cloud: Number($('maxCloud').value)}).then(() => launch('imagery')).catch(error);
    }
    if (page === 'run') {
      $('runModel').textContent = dashboard.model ? 'Current model: ' + dashboard.model.name + ' (' + dashboard.model.filename + ')' : dashboard.model_error;
      $('runButton').disabled = !dashboard.readiness.ready;
      $('confidence').value = settings.confidence; $('radius').value = settings.radius;
      Object.entries(dashboard.sensitivity_presets).forEach(([name, value]) => $('sensitivity').add(new Option(name + ' / ' + value.toFixed(2), String(value))));
      function syncSensitivity() {
        const value = Number($('confidence').value);
        $('sensitivity').value = Object.values(dashboard.sensitivity_presets).includes(value) ? String(value) : '';
      }
      syncSensitivity();
      $('confidence').addEventListener('input', syncSensitivity);
      $('sensitivity').onchange = () => { if ($('sensitivity').value) $('confidence').value = $('sensitivity').value; P.logInteraction('choose_sensitivity', {confidence: Number($('confidence').value)}); };
      function syncRunInputs() {
        $('runInputs').textContent = countLabel(dashboard.readiness.image_count, 'image file') + (dashboard.readiness.image_count === 1 ? ' matches ' : ' match ') + settings.start + ' to ' + settings.end + '. Same-day AIS records will be checked within ' + P.fmt(Number($('radius').value), 0) + ' metres.';
      }
      syncRunInputs();
      $('radius').addEventListener('input', syncRunInputs);
      const readinessBox = $('runReadiness').closest('.run-readiness-box');
      if (readinessBox) readinessBox.dataset.ready = String(dashboard.readiness.ready);
      for (const message of [...dashboard.readiness.blockers, ...dashboard.readiness.warnings]) { const item = document.createElement('p'); item.textContent = message; $('runReadiness').append(item); }
      if (dashboard.readiness.ready && !dashboard.readiness.warnings.length) $('runReadiness').textContent = 'Ready to start detection.';
      P.logInteraction('detection_readiness', dashboard.readiness);
      $('runForm').onsubmit = async e => { e.preventDefault(); $('runButton').disabled = true; try { await save({confidence: Number($('confidence').value), radius: Number($('radius').value)}); await launch('run'); } catch (err) { $('runButton').disabled = false; error(err); } };
    }
    if (page === 'jobs') { await refreshJobs(); $('refreshJobs').onclick = () => refreshJobs().catch(error); setInterval(() => refreshJobs().catch(error), 2500); }
    document.querySelectorAll('input').forEach(input => {
      input.addEventListener('change', () => P.logInteraction('change_setting', {field: input.id, value: input.value}));
      input.addEventListener('invalid', () => {
        if (['west','south','east','north'].includes(input.id) && !input.value.trim()) {
          P.setStatus($('areaError'), 'Enter a coordinate for ' + input.id + '.', 'err');
        }
        const detail = input.closest('.ops-details');
        if (detail) detail.open = true;
        P.logInteraction('invalid_operation_setting', {field: input.id});
      });
    });
    document.querySelectorAll('.ops-details').forEach((detail, index) => detail.addEventListener('toggle', () => P.logInteraction('toggle_operation_details', {page, section: index, open: detail.open})));
  } catch (err) { error(err); }
})();
