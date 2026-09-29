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

test('Leela selection persists and uses neural depth and resource controls', () => {
  const app = ui('app');
  app.storage.set('vibe-d-ai.search-settings.v1', JSON.stringify({'engine-select':'leela','search-depth':'0'}));
  app.run(`restoreSearchSettings(); engines.leela = {available:true,status:'unloaded',checkpoint:'artifacts/lc0/best.pt'}; renderEngine(); saveSearchSettings();`);
  assert.equal(app.nodes.get('engine-select').value, 'leela');
  assert.equal(app.nodes.get('search-depth').value, '0');
  assert.equal(app.nodes.get('cache-memory').disabled, true);
  assert.equal(app.nodes.get('search-threads').disabled, true);
  assert.match(app.nodes.get('engine-description').textContent, /Leela in a 5D Trenchcoat/);
  assert.match(app.nodes.get('opponent-engine').textContent, /Leela in a 5D Trenchcoat/);
  assert.match(app.nodes.get('engine-checkpoint').textContent, /artifacts\/lc0\/best.pt/);
  assert.equal(JSON.parse(app.storage.get('vibe-d-ai.search-settings.v1'))['engine-select'], 'leela');
  app.run(`$('engine-select').value = 'classical'; renderEngine();`);
  assert.equal(app.nodes.get('search-depth').value, '4');
  assert.equal(app.nodes.get('cache-memory').disabled, false);
});

test('Leela availability is refreshed independently and reports its own setup', async () => {
  const app = ui('app');
  app.run(`$('engine-select').value = 'leela'; api = async () => ({engines:[{id:'leela',available:true,status:'unloaded'}]});`);
  await app.run('refreshEngines()');
  assert.equal(app.run('engineAvailable()'), true);
  app.run(`api = async () => { throw new Error('Disconnected'); };`);
  await app.run('refreshEngines()');
  assert.equal(app.run('engineAvailable()'), false);
  assert.equal(app.nodes.get('leela-setup').hidden, false);
  assert.equal(app.nodes.get('transformer-setup').hidden, true);
});

test('automatic opponent sends Leela identity with neural resource budgets', async () => {
  const app = ui('app');
  app.run(`let request; api = async (path,body) => { request = {path,body}; return {jobId:'leela-job'}; }; renderAnalysis = () => {};
    $('engine-select').value = 'leela'; $('ai-side').value = 'black'; engines.leela.available = true;
    game = {revision:5,position:{action:1},pending:[]}; scheduleOpponent();`);
  assert.equal(app.timers.length, 1);
  await app.timers[0]();
  const request = JSON.parse(app.run('JSON.stringify(request)'));
  assert.equal(request.path, '/api/analyze');
  assert.equal(request.body.engine, 'leela');
  assert.equal(request.body.cacheMemoryMb, 0);
  assert.equal(request.body.threads, 1);
  assert.equal(app.run('search.autoPlay'), true);
});

test('Leela training has independent defaults, saved parameters and checkpoint readiness', () => {
  const app = ui('training'); readyTraining(app);
  app.run(`selectMode('leela'); renderLeelaModel();`);
  assert.equal(app.nodes.get('param-batchSize').value, '4');
  assert.equal(app.nodes.get('start-training').disabled, false);
  assert.match(app.nodes.get('model-summary').textContent, /Leela in a 5D Trenchcoat/);
  assert.equal(app.nodes.get('leela-checkpoint').textContent, 'artifacts/lc0/best.pt');
  app.run(`$('param-batchSize').value = '2'; saveParameters(); selectMode('current');`);
  assert.equal(app.nodes.get('param-batchSize').value, '16');
  app.run(`selectMode('leela');`);
  assert.equal(app.nodes.get('param-batchSize').value, '2');
  app.run(`snapshot.leelaAvailability = {available:false,reason:'LCZero checkpoint missing'}; updateControls();`);
  assert.equal(app.nodes.get('start-training').disabled, true);
  assert.equal(app.nodes.get('run-hint').textContent, 'LCZero checkpoint missing');
  app.run(`selectMode('fresh20m');`);
  assert.equal(app.nodes.get('start-training').disabled, false);
});

test('Leela training submits a resume run while fresh mode retains its own endpoint', async () => {
  const app = ui('training'); readyTraining(app);
  app.run(`let requests = []; api = async (path,body) => { requests.push({path,body}); return {}; }; refreshTraining = async () => {}; selectMode('leela');`);
  await app.nodes.get('training-form').listeners.get('submit')({preventDefault(){}});
  let request = JSON.parse(app.run('JSON.stringify(requests.at(-1))'));
  assert.equal(request.path, '/api/training/start');
  assert.equal(request.body.options.model, 'leela');
  assert.equal(request.body.options.batchSize, 4);
  app.run(`selectMode('fresh20m');`);
  await app.nodes.get('training-form').listeners.get('submit')({preventDefault(){}});
  request = JSON.parse(app.run('JSON.stringify(requests.at(-1))'));
  assert.equal(request.path, '/api/training/fresh/start');
  assert.equal(request.body.options.model, undefined);
});

test('retained Leela iteration restores its model and displays its full name', () => {
  const app = ui('training'); readyTraining(app);
  app.run(`snapshot.iterations = [{id:'leela__iteration-00000001',model:'leela',iteration:1,status:'evaluated'}]; renderIterationChoices();
    iterationData = {report:{options:{...snapshot.leelaDefaults,batchSize:2}}};`);
  assert.match(app.nodes.get('iteration-select').options[0].textContent, /Leela in a 5D Trenchcoat/);
  app.nodes.get('reuse-parameters').listeners.get('click')();
  assert.equal(app.nodes.get('training-mode').value, 'leela');
  assert.equal(app.nodes.get('param-batchSize').value, '2');
  app.run(`snapshot.status = {state:'running',mode:'selfplay',model:'leela',phase:'training'}; renderMonitor();`);
  assert.match(app.nodes.get('run-title').textContent, /Leela in a 5D Trenchcoat/);
});
