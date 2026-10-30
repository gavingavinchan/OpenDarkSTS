/* Shared helpers for the step 4-7 training pipeline pages. */
(function (global) {
  'use strict';

  const PAGE = document.body ? (document.body.dataset.page || 'unknown') : 'unknown';

  function logInteraction(action, details) {
    const payload = {
      source: 'training_ui',
      action: action,
      page: document.body.dataset.page || PAGE,
      timestamp: new Date().toISOString(),
      details: details || {},
    };
    const body = JSON.stringify(payload);
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon('/api/training-ui-interaction', new Blob([body], { type: 'application/json' }));
        return;
      }
    } catch (e) { /* fall through to fetch */ }
    fetch('/api/training-ui-interaction', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
    }).catch(function () {});
  }

  async function api(path, options) {
    const opts = Object.assign({ method: 'GET' }, options || {});
    if (opts.body !== undefined && typeof opts.body !== 'string') {
      opts.headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
      opts.body = JSON.stringify(opts.body);
    }
    const response = await fetch(path, opts);
    let payload = null;
    const text = await response.text();
    if (text) {
      try { payload = JSON.parse(text); } catch (e) { payload = { error: text.slice(0, 400) }; }
    }
    if (!response.ok) {
      const message = (payload && (payload.error || (payload.missing || []).join('; '))) || ('HTTP ' + response.status);
      const error = new Error(message);
      error.payload = payload;
      error.status = response.status;
      throw error;
    }
    return payload || {};
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function setStatus(node, message, kind) {
    if (typeof node === 'string') { node = document.getElementById(node); }
    if (!node) { return; }
    node.textContent = message || '';
    node.className = 'status' + (kind ? ' ' + kind : '');
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || value === '') { return '—'; }
    const number = Number(value);
    if (Number.isNaN(number)) { return String(value); }
    return number.toFixed(digits === undefined ? 4 : digits);
  }

  /* Ultralytics writes ANSI colour/erase codes into train.log; strip them for the tail. */
  function stripAnsi(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/\u001b\[[0-9;]*[A-Za-z]/g, '')
      .replace(/\u001b\][^\u0007]*\u0007/g, '');
  }

  function shortTime(value) {
    if (!value) { return '—'; }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) { return String(value); }
    return date.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  }

  /* Wire every button/select/input on the page so each control logs once. */
  function autoLogControls(root) {
    const scope = root || document;
    scope.querySelectorAll('button').forEach(function (button) {
      if (button.dataset.logged === '1') { return; }
      button.dataset.logged = '1';
      button.addEventListener('click', function () {
        logInteraction('click', { control: button.id || button.textContent.trim().slice(0, 48) });
      });
    });
    scope.querySelectorAll('details').forEach(function (node) {
      if (node.dataset.logged === '1') { return; }
      node.dataset.logged = '1';
      node.addEventListener('toggle', function () {
        logInteraction('toggle_details', { control: node.id || node.querySelector('summary').textContent.trim().slice(0, 64), open: node.open });
      });
    });
    scope.querySelectorAll('select, input[type="checkbox"], input[type="radio"]').forEach(function (node) {
      if (node.dataset.logged === '1') { return; }
      node.dataset.logged = '1';
      node.addEventListener('change', function () {
        let value = node.value;
        if (node.multiple) {
          value = Array.from(node.selectedOptions).map(function (option) { return option.value; });
        } else if (node.type === 'checkbox') {
          value = node.checked;
        }
        logInteraction('change', { control: node.id || node.name, value: value });
      });
    });
  }

  function captureConsoleErrors() {
    global.addEventListener('error', function (event) {
      logInteraction('page_error', { message: String(event.message || ''), source: String(event.filename || '') });
    });
  }

  /* Render local model cards and generated reports, escaping all raw HTML. */
  function renderMarkdown(text) {
    const source = String(text || '').replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
    function inline(value) {
      return value.split(/(`[^`]+`)/g).map(part => {
        if (part.startsWith('`') && part.endsWith('`')) return '<code>' + esc(part.slice(1, -1)) + '</code>';
        return esc(part)
          .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|\/(?!\/)[^\s)]+)\)/g, '<a href="$2" rel="noopener">$1</a>')
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      }).join('');
    }
    const out = [], paragraph = [], code = [];
    let inTable = false, inCode = false, list = null;
    const flushParagraph = () => { if (paragraph.length) { out.push('<p>' + inline(paragraph.join(' ')) + '</p>'); paragraph.length = 0; } };
    const closeBlocks = () => {
      flushParagraph();
      if (inTable) { out.push('</tbody></table></div>'); inTable = false; }
      if (list) { out.push('</' + list + '>'); list = null; }
    };
    source.split(/\r?\n/).forEach(line => {
      const trimmed = line.trim();
      if (trimmed.startsWith('```')) {
        closeBlocks();
        if (inCode) { out.push('<pre><code>' + esc(code.join('\n')) + '</code></pre>'); code.length = 0; }
        inCode = !inCode;
        return;
      }
      if (inCode) { code.push(line); return; }
      const isRow = trimmed.startsWith('|') && trimmed.endsWith('|');
      if (isRow) {
        flushParagraph();
        const cells = trimmed.slice(1, -1).split('|').map(cell => cell.trim());
        if (cells.every(cell => /^:?-{2,}:?$/.test(cell))) return;
        if (!inTable) { if (list) { out.push('</'+list+'>'); list = null; } out.push('<div class="scroll"><table><tbody>'); inTable = true; }
        out.push('<tr>' + cells.map(cell => '<td>' + inline(cell) + '</td>').join('') + '</tr>');
        return;
      }
      if (inTable) { out.push('</tbody></table></div>'); inTable = false; }
      const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
      const bullet = /^([-*]|\d+\.)\s+(.*)$/.exec(trimmed);
      if (heading) {
        closeBlocks();
        const level = Math.min(heading[1].length + 1, 6);
        out.push('<h' + level + '>' + inline(heading[2]) + '</h' + level + '>');
      } else if (bullet) {
        flushParagraph();
        const kind = /^\d/.test(bullet[1]) ? 'ol' : 'ul';
        if (list !== kind) { if (list) out.push('</'+list+'>'); out.push('<'+kind+'>'); list = kind; }
        out.push('<li>' + inline(bullet[2]) + '</li>');
      } else if (/^---+$/.test(trimmed)) {
        closeBlocks(); out.push('<hr>');
      } else if (!trimmed) {
        closeBlocks();
      } else {
        if (list) { out.push('</'+list+'>'); list = null; }
        paragraph.push(trimmed);
      }
    });
    closeBlocks();
    if (inCode) out.push('<pre><code>' + esc(code.join('\n')) + '</code></pre>');
    return out.join('\n');
  }

  function renderModel(node, model, error) {
    if (!model) { node.textContent = error || 'No detection model selected.'; return; }
    const metrics = model.metrics || {};
    const rows = [['All classes', '', 'instances'], ['Single ship (ship)', 'ship_', 'ship_instances'], ['STS pair (ships)', 'ships_', 'ships_instances']];
    node.innerHTML = '<h3>' + esc(model.name || model.filename) + '</h3><p>' +
      (metrics.split === 'test' ? 'Held-out test results: ' : 'Recorded evaluation results: ') +
      esc(metrics.images === undefined ? 'image count not recorded' : metrics.images + ' images') +
      '. These scores describe this dataset; performance in other areas can differ.</p>' +
      (Object.keys(metrics).length ? '<div class="scroll"><table aria-label="Model performance"><thead><tr><th>Class</th><th>Objects</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50–95</th></tr></thead><tbody>' +
      rows.map(([label, prefix, count]) => '<tr><td>' + esc(label) + '</td><td>' + esc(metrics[count] ?? '—') + '</td>' +
        ['precision', 'recall', 'mAP50', 'mAP50_95'].map(key => '<td>' + fmt(metrics[prefix + key] ?? (key === 'mAP50_95' ? metrics[prefix + 'mAP50-95'] : undefined), 3) + '</td>').join('') + '</tr>').join('') + '</tbody></table></div>' : '<p>No evaluation scores were recorded for this model.</p>') +
      '<p class="hint">Precision measures how often predicted objects are correct; recall measures how many labelled objects are found. Average precision (mAP) combines detection and location accuracy: mAP50 uses 50% box overlap; mAP50–95 averages stricter overlap thresholds. Scores range from 0 to 1; higher is better.</p>' +
      (model.model_card_url ? '<a href="' + esc(model.model_card_url) + '" target="_blank" rel="noopener">Open YOLO11s model card</a>' : '') +
      '<details><summary>Model file</summary><p class="path">' + esc(model.filename) + '</p></details>';
  }

  global.Pipeline = {
    api: api,
    renderModel: renderModel,
    esc: esc,
    fmt: fmt,
    logInteraction: logInteraction,
    renderMarkdown: renderMarkdown,
    setStatus: setStatus,
    shortTime: shortTime,
    stripAnsi: stripAnsi,
    autoLogControls: autoLogControls,
    captureConsoleErrors: captureConsoleErrors,
  };
})(window);
