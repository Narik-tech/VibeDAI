const componentLabels = {
  material: 'Material', activity: 'Activity', kingSafety: 'King safety',
  temporal: 'Temporal pressure', timelines: 'Timelines', travel: 'Travel opportunities',
};
const storageKey = 'vibe-d-ai.classical-heuristics.v1';
const searchKeys = new Set(['quiescenceDepth', 'aspirationWindow', 'temporalMovePenalty', 'quietCentralization', 'killerBonus', 'historyBonus']);
const $ = id => document.getElementById(id);
function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}
function cp(value) {
  if (!Number.isFinite(value)) return '—';
  const rounded = Math.round(value * 10) / 10;
  return `${rounded > 0 ? '+' : rounded < 0 ? '−' : ''}${Math.abs(rounded).toLocaleString('en', {maximumFractionDigits: 1})}`;
}
function scoreColor(element, value) {
  element.classList.toggle('positive', Number.isFinite(value) && value > 0);
  element.classList.toggle('negative', Number.isFinite(value) && value < 0);
}
function groupLabel(group) {
  if (typeof group === 'object') return group.label || group.id || 'Evaluation';
  return componentLabels[group] || String(group || 'Evaluation').replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, c => c.toUpperCase());
}

export function createHeuristicsPanel({api, onChange, onReady, boardLabel}) {
  let settings = [], defaults = {}, values = {}, loaded = false;
  let revision = null, engine = 'classical', busy = false;
  let evaluationToken = 0, evaluationTimer = null;
  let settingsSignature = '[]', evaluationCleared = false;
  const controls = new Map(), componentNodes = new Map();

  const signature = () => settingsSignature;
  const refreshSignature = () => { settingsSignature = JSON.stringify(settings.map(setting => values[setting.key])); };
  const configuration = () => ({...values});
  function status(message, error = false) {
    if ($('heuristics-status').textContent !== message) $('heuristics-status').textContent = message;
    $('heuristics-status').classList.toggle('error', error);
  }
  function cleanValue(setting, value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return defaults[setting.key];
    const min = Number(setting.min), max = Number(setting.max), step = Number(setting.step) || 1;
    const bounded = Math.max(min, Math.min(max, value));
    return Number((min + Math.round((bounded - min) / step) * step).toFixed(8));
  }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(values)); } catch { /* In-memory settings still work. */ }
  }
  function syncControl(setting, editingNumber = document.activeElement) {
    const control = controls.get(setting.key), value = values[setting.key], text = String(value);
    if (control.range.value !== text) control.range.value = text;
    if (control.number !== editingNumber && control.number.value !== text) control.number.value = text;
    const modified = value !== defaults[setting.key];
    if (control.modified !== modified) {
      control.row.classList.toggle('modified', modified);
      control.modified = modified;
    }
    const disabled = busy || engine !== 'classical';
    if (control.range.disabled !== disabled) control.range.disabled = disabled;
    if (control.number.disabled !== disabled) control.number.disabled = disabled;
    if (control.renderedValue !== value) {
      control.range.setAttribute('aria-valuetext', `${value}${setting.unit ? ` ${setting.unit}` : ''}`);
      control.renderedValue = value;
    }
  }
  function syncProfile() {
    const changed = settings.reduce((count, setting) => count + Number(values[setting.key] !== defaults[setting.key]), 0);
    const label = changed ? `${changed} CUSTOM ${changed === 1 ? 'VALUE' : 'VALUES'}` : 'DEFAULT WEIGHTS';
    if ($('heuristics-profile').textContent !== label) $('heuristics-profile').textContent = label;
    const disabled = !loaded || !changed || busy || engine !== 'classical';
    if ($('heuristics-reset').disabled !== disabled) $('heuristics-reset').disabled = disabled;
  }
  function syncControls(editingNumber = document.activeElement) {
    for (const setting of settings) syncControl(setting, editingNumber);
    syncProfile();
  }
  function clearEvaluation() {
    if (evaluationCleared) return;
    evaluationCleared = true;
    $('heuristics-total').textContent = '—';
    scoreColor($('heuristics-total'), null);
    for (const value of componentNodes.values()) { value.textContent = '—'; scoreColor(value, null); }
    for (const control of controls.values()) {
      control.contribution.textContent = searchKeys.has(control.key) ? 'Search only' : '— cp';
      scoreColor(control.contribution, null);
    }
    $('heuristics-features').replaceChildren(node('p', 'heuristics-empty', 'Waiting for the current position…'));
    $('heuristics-boards').replaceChildren();
  }
  function change(setting, raw, input) {
    if (busy || engine !== 'classical') return;
    if (String(raw).trim() === '' || !Number.isFinite(Number(raw))) {
      input.setCustomValidity('Enter a finite number.');
      return;
    }
    input.setCustomValidity('');
    const next = cleanValue(setting, Number(raw));
    if (values[setting.key] === next) return;
    values = {...values, [setting.key]: next};
    refreshSignature();
    syncControl(setting, input.type === 'number' ? input : null); syncProfile(); persist();
    onChange?.();
    queueEvaluation(220, 'Weights changed · updating the current position. Run analysis for a new continuation.');
  }
  function buildControls() {
    const groups = new Map();
    for (const setting of settings) {
      const label = groupLabel(setting.group);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(setting);
    }
    const fragment = document.createDocumentFragment();
    let opened = false;
    for (const [label, group] of groups) {
      const details = node('details', 'heuristics-detail heuristics-group');
      const summary = node('summary', '', label);
      summary.append(node('span', '', `${group.length} ${group.length === 1 ? 'setting' : 'settings'}`));
      details.append(summary);
      if (!opened && group.length <= 8) { details.open = true; opened = true; }
      const list = node('div', 'heuristics-control-list');
      for (const setting of group) {
        const id = `heuristic-${setting.key}`, descriptionId = `${id}-description`;
        const row = node('div', 'heuristic-control'); row.dataset.heuristic = setting.key;
        const heading = node('div', 'heuristic-control-heading');
        const labelNode = node('label', '', setting.label); labelNode.htmlFor = `${id}-number`;
        const contribution = node('span', 'heuristic-contribution', searchKeys.has(setting.key) ? 'Search only' : '— cp');
        contribution.title = 'Net current-position contribution for White. Shared effects are listed in Feature contributions.';
        heading.append(labelNode, contribution);
        const inputs = node('div', 'heuristic-inputs');
        const range = node('input'); range.type = 'range'; range.id = id;
        const number = node('input'); number.type = 'number'; number.id = `${id}-number`; number.inputMode = 'decimal';
        for (const input of [range, number]) {
          input.min = setting.min; input.max = setting.max; input.step = setting.step;
          input.setAttribute('aria-label', `${setting.label}${input === range ? ' slider' : ' value'}`);
          input.setAttribute('aria-describedby', descriptionId);
          input.addEventListener('input', () => change(setting, input.value, input));
          input.addEventListener('change', () => { input.setCustomValidity(''); syncControl(setting, null); });
        }
        inputs.append(range, number, node('span', 'heuristic-unit', setting.unit || 'weight'));
        const description = node('p', 'heuristic-description', setting.description); description.id = descriptionId;
        description.append(node('span', 'heuristic-default', ` Default: ${defaults[setting.key]}${setting.unit ? ` ${setting.unit}` : ''}.`));
        row.append(heading, inputs, description); list.append(row);
        controls.set(setting.key, {key: setting.key, row, range, number, contribution});
      }
      details.append(list); fragment.append(details);
    }
    $('heuristics-groups').replaceChildren(fragment);
    $('heuristics-components').replaceChildren(...Object.entries(componentLabels).map(([key, label]) => {
      const card = node('div', 'heuristic-component'); card.dataset.component = key;
      const value = node('strong', '', '—'); componentNodes.set(key, value);
      card.append(node('span', '', label), value);
      return card;
    }));
    syncControls();
  }
  function showEvaluation(evaluation) {
    evaluationCleared = false;
    $('heuristics-total').textContent = `${cp(evaluation.total)} cp`;
    scoreColor($('heuristics-total'), evaluation.total);
    for (const [key, value] of componentNodes) {
      value.textContent = `${cp(evaluation[key])} cp`; scoreColor(value, evaluation[key]);
    }
    const features = Array.isArray(evaluation.features) ? evaluation.features : [];
    const featureMap = new Map(features.map(feature => [feature.key, feature]));
    for (const [key, control] of controls) {
      const feature = featureMap.get(key);
      const component = key.endsWith('Weight') ? key.slice(0, -6) : null;
      const value = Object.hasOwn(componentLabels, component) ? evaluation[component] : feature?.value;
      control.contribution.textContent = searchKeys.has(key) ? 'Search only' : Number.isFinite(value) ? `${cp(value)} cp` : 'Shared effect';
      scoreColor(control.contribution, value);
    }
    const groups = new Map();
    for (const feature of features) {
      const component = feature.component || 'activity';
      if (!groups.has(component)) groups.set(component, []);
      groups.get(component).push(feature);
    }
    $('heuristics-features').replaceChildren(...[...groups].map(([component, entries]) => {
      const group = node('section', 'heuristic-feature-group');
      group.append(node('h4', '', componentLabels[component] || groupLabel(component)));
      for (const feature of entries) {
        const row = node('div', 'heuristic-feature'); row.dataset.feature = feature.key;
        const heading = node('div'); const value = node('strong', '', `${cp(feature.value)} cp`);
        scoreColor(value, feature.value);
        heading.append(node('span', '', feature.label), value); row.append(heading);
        if (feature.description) row.append(node('p', '', feature.description));
        group.append(row);
      }
      return group;
    }));
    if (!features.length) $('heuristics-features').append(node('p', 'heuristics-empty', 'No feature breakdown is available for this position.'));
    const table = node('table', 'heuristics-board-table'), head = node('thead'), headings = node('tr');
    for (const label of ['Board', 'Weight', 'Material', 'Activity', 'King safety']) headings.append(node('th', '', label));
    head.append(headings); table.append(head);
    const body = node('tbody');
    for (const board of evaluation.boards || []) {
      const row = node('tr');
      row.append(node('td', '', `${boardLabel?.(board) || `${board.timeline}L · ${board.turn}T`} · ${board.active ? 'active' : 'inactive'}`), node('td', '', String(board.weight)));
      for (const key of ['material', 'activity', 'kingSafety']) row.append(node('td', '', cp(board[key])));
      body.append(row);
    }
    table.append(body); $('heuristics-boards').replaceChildren(table);
    status('Current position · changes apply to Classical analysis and engine play. Feature and component values are rounded for display.');
  }
  function queueEvaluation(delay = 0, message = 'Updating the current position…') {
    clearTimeout(evaluationTimer);
    const token = ++evaluationToken;
    clearEvaluation();
    if (!loaded || revision === null || engine !== 'classical') return;
    const requestedRevision = revision, requestedSignature = signature(), requestedValues = configuration();
    status(message);
    evaluationTimer = setTimeout(async () => {
      try {
        const data = await api('/api/evaluate', {revision: requestedRevision, heuristics: requestedValues});
        if (token !== evaluationToken || revision !== requestedRevision || signature() !== requestedSignature || engine !== 'classical' || data.revision !== revision) return;
        if (data.heuristics && settings.some(setting => data.heuristics[setting.key] !== values[setting.key])) {
          status('The evaluation used different settings. Refresh the page to reload the available controls.', true); return;
        }
        showEvaluation(data.evaluation);
      } catch (error) {
        if (token === evaluationToken && revision === requestedRevision && engine === 'classical') status(`Evaluation unavailable: ${error.message}`, true);
      }
    }, delay);
  }
  $('heuristics-reset').addEventListener('click', () => {
    if (!loaded || busy || engine !== 'classical') return;
    values = {...defaults}; refreshSignature(); syncControls(); persist(); onChange?.();
    queueEvaluation(0, 'Default weights restored · updating the current position.');
  });
  return {
    configuration, signature,
    ready: () => loaded,
    async load() {
      try {
        const schema = await api('/api/heuristics');
        settings = schema.settings; defaults = {...schema.defaults}; values = {...defaults};
        try {
          const saved = JSON.parse(localStorage.getItem(storageKey));
          for (const setting of settings) if (saved && Object.hasOwn(saved, setting.key)) values[setting.key] = cleanValue(setting, saved[setting.key]);
        } catch { /* Ignore unavailable storage and obsolete saved formats. */ }
        refreshSignature(); loaded = true; buildControls(); queueEvaluation(); onReady?.();
      } catch (error) { status(`Heuristic controls unavailable: ${error.message}`, true); onReady?.(); }
    },
    setContext(context) {
      const nextRevision = context.revision ?? null, nextEngine = context.engine || 'classical', nextBusy = Boolean(context.busy);
      const engineChanged = engine !== nextEngine, changed = revision !== nextRevision || engineChanged;
      const controlsChanged = busy !== nextBusy || engineChanged;
      // Analysis polls repeat this context; leave the entire panel untouched.
      if (!changed && !controlsChanged) return;
      revision = nextRevision; engine = nextEngine; busy = nextBusy;
      if (engineChanged) {
        $('heuristics-engine-note').hidden = engine === 'classical';
        $('heuristics-content').hidden = engine !== 'classical';
      }
      if (controlsChanged) syncControls();
      if (changed) queueEvaluation();
    },
  };
}
