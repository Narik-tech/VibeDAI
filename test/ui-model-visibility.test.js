import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

// A small DOM harness exercises the shipped UI handlers without starting Python
// or requiring a browser. Layout is checked separately in the live app.
function ui(page) {
  const nodes = new Map(), storage = new Map(), timers = [];
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
      this.style = {}; this.value = ''; this.listeners = new Map(); this.validity = {valid:true};
      this.classList = {toggle(){},add(){},remove(){}};
    }
    set id(value) { this._id = value; nodes.set(value, this); }
    get id() { return this._id; }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
    get options() { return this.children; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) {
      const remove = node => { if (node.id) nodes.delete(node.id); node.children.forEach(remove); };
      this.children.forEach(remove); this.text = ''; this.children = children;
      const register = node => { if (node.id) nodes.set(node.id, node); node.children.forEach(register); };
      children.forEach(register);
    }
    addEventListener(type, handler) { this.listeners.set(type, handler); }
    setAttribute(key, value) { this[key] = value; }
    removeAttribute(key) { delete this[key]; }
    setCustomValidity(message) { this.validity.valid = !message; }
    reportValidity() { return this.validity.valid; }
    focus() {}
    closest() { return null; }
  }
  const html = readFileSync(new URL(`../public/${page === 'app' ? 'index' : page}.html`, import.meta.url), 'utf8');
  for (const match of html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const node = new Element(match[1]); node.id = match[3];
    for (const key of ['value','min','max']) node[key] = match[2].match(new RegExp(`\\b${key}="([^"]*)"`))?.[1] || '';
    node.hidden = /\bhidden\b/.test(match[2]); node.disabled = /\bdisabled\b/.test(match[2]);
  }
  for (const match of html.matchAll(/<select\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
    const select = nodes.get(match[1]);
    for (const optionMatch of match[2].matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)) {
      const option = new Element('option'); option.value = optionMatch[1].match(/value="([^"]*)"/)?.[1] || optionMatch[2]; option.textContent = optionMatch[2];
      select.append(option);
      if (!select.value || /\bselected\b/.test(optionMatch[1])) select.value = option.value;
    }
  }
  const pipeline = new Element('ol');
  const context = vm.createContext({
    document:{getElementById:id => nodes.get(id) || null, createElement:tag => new Element(tag), createTextNode:text => { const node = new Element(); node.textContent = text; return node; },
      querySelector:selector => selector === '.pipeline' ? pipeline : null, querySelectorAll:selector => selector === '.pipeline li' ? pipeline.children : [], addEventListener(){}},
    window:{addEventListener(){},innerWidth:1200},
    localStorage:{getItem:key => storage.get(key) || null,setItem:(key,value) => storage.set(key,value)},
    setTimeout:callback => { timers.push(callback); return timers.length; },clearTimeout(){},
    performance:{now:() => 0}, console,
  });
  let source = readFileSync(new URL(`../public/${page}.js`, import.meta.url), 'utf8');
  source = page === 'app' ? source.slice(0, source.indexOf("$('orientation').addEventListener")) : source.replace(/void refreshTraining\(\);\s*$/, '');
  vm.runInContext(source, context);
  return {nodes,storage,timers,run:code => vm.runInContext(code, context)};
}

function readyTraining(app) {
  app.run(`defaults = Object.fromEntries(fields.map(field => [field.key, field.choices ? field.choices[0][0] : field.key === 'learningRate' ? .0001 : field.key === 'promotionScore' ? .55 : field.min || 1]));
    defaults.batchSize = 16; defaults.maxTokens = 512;
    snapshot = {defaults, status:{state:'idle'}, availability:{available:true}, freshAvailability:{available:true},
      freshDefaults:{samples:32,teacherNodes:100,teacherTimeMs:100,device:'auto',steps:8,batchSize:1,maxTokens:512,learningRate:.0001,seed:3},
      leelaDefaults:{...defaults,batchSize:4,model:'leela'}, leelaAvailability:{available:true},
      leelaModel:{available:true,checkpoint:'artifacts/lc0/best.pt'}, iterations:[]}; online = true;`);
}

test('analysis offers only Classical and the 800k transformer, with archived settings falling back to Classical', () => {
  const app = ui('app');
  assert.deepEqual(app.nodes.get('engine-select').options.map(option => option.value), ['classical','transformer']);
  assert.match(app.nodes.get('engine-select').options[1].textContent, /800k/);
  assert.equal(app.nodes.has('leela-setup'), false);
  app.storage.set('vibe-d-ai.search-settings.v1', JSON.stringify({'engine-select':'leela','search-depth':'0'}));
  app.run(`restoreSearchSettings(); renderEngine(); saveSearchSettings();`);
  assert.equal(app.nodes.get('engine-select').value, 'classical');
  assert.equal(app.nodes.get('search-depth').value, '4');
  assert.equal(app.nodes.get('cache-memory').disabled, false);
  assert.equal(JSON.parse(app.storage.get('vibe-d-ai.search-settings.v1'))['engine-select'], 'classical');
});

test('800k selection persists and uses neural depth and resource controls', () => {
  const app = ui('app');
  app.storage.set('vibe-d-ai.search-settings.v1', JSON.stringify({'engine-select':'transformer','search-depth':'0'}));
  app.run(`restoreSearchSettings(); engines.transformer = {available:true,status:'unloaded',checkpoint:'artifacts/transformer/model.pt'}; renderEngine(); saveSearchSettings();`);
  assert.equal(app.nodes.get('engine-select').value, 'transformer');
  assert.equal(app.nodes.get('search-depth').value, '0');
  assert.equal(app.nodes.get('cache-memory').disabled, true);
  assert.equal(app.nodes.get('search-threads').disabled, true);
  assert.match(app.nodes.get('engine-description').textContent, /800k transformer/);
  assert.match(app.nodes.get('opponent-engine').textContent, /Transformer · 800k/);
  assert.match(app.nodes.get('engine-checkpoint').textContent, /artifacts\/transformer\/model.pt/);
  assert.equal(JSON.parse(app.storage.get('vibe-d-ai.search-settings.v1'))['engine-select'], 'transformer');
  app.run(`$('engine-select').value = 'classical'; renderEngine();`);
  assert.equal(app.nodes.get('search-depth').value, '4');
  assert.equal(app.nodes.get('cache-memory').disabled, false);
});

test('engine refresh ignores archived engines and preserves transformer availability and setup', async () => {
  const app = ui('app');
  app.run(`$('engine-select').value = 'transformer'; api = async () => ({engines:[{id:'transformer',available:true,status:'unloaded'},{id:'leela',available:true,status:'unloaded'}]});`);
  await app.run('refreshEngines()');
  assert.equal(app.run('engineAvailable()'), true);
  assert.equal(app.run("Object.hasOwn(engines, 'leela')"), false);
  app.run(`api = async () => { throw new Error('Disconnected'); };`);
  await app.run('refreshEngines()');
  assert.equal(app.run('engineAvailable()'), false);
  assert.equal(app.nodes.get('transformer-setup').hidden, false);
});

test('automatic opponent sends transformer identity with neural resource budgets', async () => {
  const app = ui('app');
  app.run(`let request; api = async (path,body) => { request = {path,body}; return {jobId:'transformer-job'}; }; renderAnalysis = () => {};
    $('engine-select').value = 'transformer'; $('ai-side').value = 'black'; engines.transformer.available = true;
    game = {revision:5,position:{action:1},pending:[]}; scheduleOpponent();`);
  assert.equal(app.timers.length, 1);
  await app.timers[0]();
  const request = JSON.parse(app.run('JSON.stringify(request)'));
  assert.equal(request.path, '/api/analyze');
  assert.equal(request.body.engine, 'transformer');
  assert.equal(request.body.cacheMemoryMb, 0);
  assert.equal(request.body.threads, 1);
  assert.equal(app.run('search.autoPlay'), true);
});

test('training offers only the 800k model and restores its saved parameters', () => {
  const app = ui('training'); readyTraining(app);
  assert.deepEqual(app.nodes.get('training-mode').options.map(option => option.value), ['current']);
  assert.equal(app.nodes.has('leela-model-heading'), false);
  assert.equal(app.nodes.has('fresh-model-heading'), false);
  app.storage.set('vibe-d-ai.training-settings.v1', JSON.stringify({batchSize:8}));
  app.storage.set('vibe-d-ai.training-leela-settings.v1', JSON.stringify({batchSize:2}));
  app.storage.set('vibe-d-ai.training-20m-settings.v1', JSON.stringify({batchSize:1}));
  app.run(`selectMode('leela');`);
  assert.equal(app.nodes.get('training-mode').value, 'current');
  assert.equal(app.nodes.get('param-batchSize').value, '8');
  assert.equal(app.nodes.get('start-training').disabled, false);
  app.run(`$('param-batchSize').value = '4'; saveParameters(); selectMode('fresh20m');`);
  assert.equal(app.nodes.get('param-batchSize').value, '4');
  assert.equal(app.nodes.get('training-mode').value, 'current');
  assert.equal(JSON.parse(app.storage.get('vibe-d-ai.training-leela-settings.v1')).batchSize, 2);
  app.run(`snapshot.availability = {available:false,reason:'800k checkpoint missing'}; updateControls();`);
  assert.equal(app.nodes.get('start-training').disabled, true);
  assert.equal(app.nodes.get('run-hint').textContent, '800k checkpoint missing');
});

test('training submits only the current 800k model even after a retired mode is requested', async () => {
  const app = ui('training'); readyTraining(app);
  app.run(`let requests = []; api = async (path,body) => { requests.push({path,body}); return {}; }; refreshTraining = async () => {}; selectMode('leela');`);
  await app.nodes.get('training-form').listeners.get('submit')({preventDefault(){}});
  const request = JSON.parse(app.run('JSON.stringify(requests.at(-1))'));
  assert.equal(request.path, '/api/training/start');
  assert.equal(request.body.options.model, 'current');
  assert.equal(request.body.options.batchSize, 16);
});

test('retained iteration choices omit archived models and parameter reuse stays on the 800k model', () => {
  const app = ui('training'); readyTraining(app);
  app.run(`snapshot.iterations = [
    {id:'leela__iteration-00000003',model:'leela',iteration:3,status:'evaluated'},
    {id:'fresh-test__iteration-00000002',model:'20m',iteration:2,status:'evaluated'},
    {id:'iteration-00000001',model:'current',iteration:1,status:'evaluated'},
    {id:'iteration-00000000',iteration:0,status:'evaluated'}];
    selectedIteration = 'leela__iteration-00000003'; renderIterationChoices();
    iterationData = {report:{options:{...defaults,model:'current',batchSize:2}}};`);
  assert.deepEqual(app.nodes.get('iteration-select').options.map(option => option.value), ['iteration-00000001','iteration-00000000']);
  assert.equal(app.run('selectedIteration'), 'iteration-00000001');
  assert.match(app.nodes.get('iteration-select').options[0].textContent, /800k/);
  app.nodes.get('reuse-parameters').listeners.get('click')();
  assert.equal(app.nodes.get('training-mode').value, 'current');
  assert.equal(app.nodes.get('param-batchSize').value, '2');
  app.run(`snapshot.status = {state:'running',mode:'selfplay',model:'current',phase:'training'}; renderMonitor();`);
  assert.match(app.nodes.get('run-title').textContent, /800k self-play/);
});

test('older archived run status keeps stop controls without being labeled as the 800k model', () => {
  const app = ui('training'); readyTraining(app);
  for (const status of [
    {state:'running',mode:'selfplay',model:'leela'},
    {state:'running',mode:'selfplay',model:'20m'},
    {state:'running',mode:'fresh20m'},
    {state:'running',mode:'selfplay',options:{model:'leela'}},
  ]) {
    app.run(`snapshot.status = ${JSON.stringify(status)}; renderMonitor(); updateControls();`);
    assert.equal(app.nodes.get('run-title').textContent, 'Training · training in progress');
    assert.equal(app.nodes.get('stop-training').disabled, false);
    assert.equal(app.nodes.get('start-training').disabled, true);
  }
});
