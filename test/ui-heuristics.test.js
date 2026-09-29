import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// Count UI mutations without running a browser or a competing search workload.
function panelHarness() {
  const nodes = new Map(), allNodes = [], timers = new Map(), requests = [], storage = new Map();
  let writes = 0, serializations = 0, nextTimer = 0, changes = 0;
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.attributes = new Map();
      this.listeners = new Map(); this.writes = 0; this._value = ''; this._disabled = false; this._hidden = false; this._text = '';
      this.classList = {toggle: () => this.record()};
      allNodes.push(this);
    }
    record() { this.writes++; writes++; }
    set id(value) { this._id = value; nodes.set(value, this); }
    get id() { return this._id; }
    set value(value) { this.record(); this._value = String(value); }
    get value() { return this._value; }
    set disabled(value) { this.record(); this._disabled = value; }
    get disabled() { return this._disabled; }
    set hidden(value) { this.record(); this._hidden = value; }
    get hidden() { return this._hidden; }
    set textContent(value) { this.record(); this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
    append(...children) { this.record(); this.children.push(...children); }
    replaceChildren(...children) { this.record(); this._text = ''; this.children = children; }
    setAttribute(key, value) { this.record(); this.attributes.set(key, value); }
    addEventListener(type, callback) { this.listeners.set(type, callback); }
    setCustomValidity(message) { this.validityMessage = message; }
    fire(type) { return this.listeners.get(type)?.(); }
  }
  const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
  for (const match of html.matchAll(/\bid="(heuristics-[^"]+)"/g)) { const element = new Element(); element.id = match[1]; }
  const document = {activeElement: null, getElementById: id => nodes.get(id), createElement: tag => new Element(tag), createDocumentFragment: () => new Element()};
  const schema = {
    settings: [
      {key: 'materialWeight', label: 'Material', description: 'Material multiplier.', group: 'Components', default: 1, min: 0, max: 4, step: 0.05, unit: '×'},
      {key: 'queenValue', label: 'Queen', description: 'Queen material value.', group: 'Piece values', default: 1150, min: 0, max: 5000, step: 5, unit: 'cp'},
    ],
    defaults: {materialWeight: 1, queenValue: 1150},
  };
  const context = vm.createContext({
    document,
    localStorage: {getItem: key => storage.get(key) || null, setItem: (key, value) => storage.set(key, value)},
    setTimeout: callback => { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout: id => timers.delete(id),
    JSON: {parse: JSON.parse, stringify: value => { serializations++; return JSON.stringify(value); }},
  });
  const source = readFileSync(new URL('../public/heuristics.js', import.meta.url), 'utf8').replace('export function createHeuristicsPanel', 'function createHeuristicsPanel');
  vm.runInContext(source, context);
  const createPanel = vm.runInContext('createHeuristicsPanel', context);
  let currentContext = {revision: 1, engine: 'classical', busy: false};
  const panel = createPanel({
    api: async (path, body) => {
      if (path === '/api/heuristics') return schema;
      return new Promise(resolve => requests.push({body, resolve}));
    },
    onChange: () => {
      changes++;
      // Analysis invalidation currently renders twice. Neither render should
      // rewrite inputs or overwrite a numeric value still being typed.
      panel.setContext(currentContext); panel.setContext(currentContext);
    },
  });
  return {
    panel, nodes, document, timers, requests,
    get writes() { return writes; }, get serializations() { return serializations; }, get changes() { return changes; },
    clearCounts() { writes = 0; serializations = 0; for (const element of allNodes) element.writes = 0; },
    setContext(value) { currentContext = value; panel.setContext(value); },
    async load() { panel.setContext(currentContext); await panel.load(); },
    runTimer() { const [id, callback] = timers.entries().next().value; timers.delete(id); return callback(); },
    reply(index, total) {
      const request = requests[index];
      request.resolve({revision: request.body.revision, heuristics: request.body.heuristics, evaluation: {
        total, material: total, activity: 0, kingSafety: 0, temporal: 0, timelines: 0, travel: 0, features: [], boards: [],
      }});
    },
  };
}

test('unchanged analysis contexts and signature checks do no panel work', async () => {
  const app = panelHarness();
  await app.load();
  const signature = app.panel.signature(), timerCount = app.timers.size;
  app.clearCounts();
  for (let poll = 0; poll < 100; poll++) {
    app.setContext({revision: 1, engine: 'classical', busy: 0});
    assert.equal(app.panel.signature(), signature);
  }
  assert.equal(app.writes, 0, 'Repeated polls leave controls and evaluation DOM untouched.');
  assert.equal(app.serializations, 0, 'Unchanged signatures are cached.');
  assert.equal(app.timers.size, timerCount, 'Polls do not queue static evaluations.');
  assert.equal(app.requests.length, 0);
  const unloaded = panelHarness();
  unloaded.clearCounts();
  unloaded.setContext({}); unloaded.setContext({revision: null, engine: 'classical', busy: false});
  assert.equal(unloaded.writes, 0, 'Missing context values normalize before comparison.');
});

test('tuning updates only the changed control and preserves numeric typing and busy state', async () => {
  const app = panelHarness(); await app.load();
  const number = app.nodes.get('heuristic-queenValue-number'), range = app.nodes.get('heuristic-queenValue');
  const otherNumber = app.nodes.get('heuristic-materialWeight-number'), otherRange = app.nodes.get('heuristic-materialWeight');
  const before = app.panel.signature();
  app.document.activeElement = number; number.value = '1161'; app.clearCounts(); number.fire('input');
  assert.equal(number.value, '1161', 'Intermediate exact text survives callbacks.');
  assert.equal(range.value, '1160');
  assert.equal(app.panel.configuration().queenValue, 1160);
  assert.notEqual(app.panel.signature(), before);
  assert.equal(app.changes, 1);
  assert.equal(otherNumber.writes + otherRange.writes, 0, 'Unchanged controls are not rewritten.');
  assert.equal(app.nodes.get('heuristics-features').writes, 0, 'An already-pending breakdown is not cleared again.');
  number.fire('change'); assert.equal(number.value, '1160', 'Commit synchronizes the snapped value.');
  const timerCount = app.timers.size;
  app.setContext({revision: 1, engine: 'classical', busy: true});
  assert.equal(number.disabled, true); assert.equal(otherRange.disabled, true);
  number.value = '1500'; number.fire('input');
  assert.equal(app.panel.configuration().queenValue, 1160, 'Busy transactions reject tuning.');
  assert.equal(app.timers.size, timerCount, 'Busy changes alone do not reevaluate.');
  app.setContext({revision: 1, engine: 'classical', busy: false});
  assert.equal(number.disabled, false); assert.equal(otherRange.disabled, false);
});

test('cached signatures still reject stale evaluations after tuning and engine changes', async () => {
  const app = panelHarness(); await app.load();
  const original = app.runTimer();
  const range = app.nodes.get('heuristic-queenValue'); range.value = '1500'; range.fire('input');
  const tuned = app.runTimer();
  app.reply(1, 50); await tuned;
  assert.equal(app.nodes.get('heuristics-total').textContent, '+50 cp');
  app.reply(0, 9999); await original;
  assert.equal(app.nodes.get('heuristics-total').textContent, '+50 cp', 'Old profiles cannot replace the current result.');
  app.setContext({revision: 2, engine: 'classical', busy: false});
  const nextPosition = app.runTimer();
  app.setContext({revision: 2, engine: 'leela', busy: false});
  app.reply(2, 123); await nextPosition;
  assert.equal(app.nodes.get('heuristics-total').textContent, '—');
  assert.equal(app.nodes.get('heuristics-content').hidden, true);
  assert.equal(range.disabled, true);
  app.setContext({revision: 2, engine: 'classical', busy: false});
  assert.equal(range.disabled, false); assert.equal(app.timers.size, 1);
  const tunedSignature = app.panel.signature();
  app.nodes.get('heuristics-reset').fire('click');
  assert.notEqual(app.panel.signature(), tunedSignature);
  assert.equal(app.panel.configuration().queenValue, 1150);
});
