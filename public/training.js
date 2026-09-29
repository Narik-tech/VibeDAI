const $ = id => document.getElementById(id);
const storageKey = 'vibe-d-ai.training-settings.v1';
const modeStorageKeys = {current:storageKey,selfplay20m:'vibe-d-ai.training-20m-settings.v1',fresh20m:'vibe-d-ai.fresh-20m-settings.v1'};
const fields = [
  { key:'iterations', label:'Iterations', min:0, max:1000000, help:'0 runs continuously', group:'Run length' },
  { key:'device', label:'Device', choices:[['auto','Automatic'],['cuda','CUDA / GPU'],['cpu','CPU']], group:'Run length' },
  { key:'games', label:'Games / iteration', min:1, max:128, group:'Self-play' },
  { key:'gameConcurrency', label:'Concurrent games', min:1, max:8, help:'Self-play games share inference; higher values use more CPU', group:'Self-play' },
  { key:'maxPlies', label:'Turns / game', min:1, max:256, group:'Self-play' },
  { key:'exploration', label:'Exploration chance', min:0, max:1, step:'any', help:'0–1 · early random turns', group:'Self-play' },
  { key:'explorationPlies', label:'Exploration turns', min:0, max:256, help:'At the start of each game', group:'Self-play' },
  { key:'steps', label:'Training updates', min:1, max:1000000, group:'Learning' },
  { key:'batchSize', label:'Batch size', min:1, max:128, group:'Learning' },
  { key:'maxTokens', label:'Context tokens', min:16, max:4096, help:'Shorter contexts use less memory', group:'Learning' },
  { key:'learningRate', label:'Learning rate', min:0, exclusiveMin:true, max:.1, step:'any', group:'Learning' },
  { key:'replaySize', label:'Replay positions', min:1, max:100000, help:'Maximum retained samples', group:'Learning' },
  { key:'outcomeWeight', label:'Outcome weight', min:0, max:1, step:'any', help:'Finished result vs search target', group:'More learning options', advanced:true },
  { key:'seed', label:'Random seed', min:0, max:4294967295, group:'More learning options', advanced:true },
  { key:'keepIterations', label:'Retain iterations', min:1, max:100, help:'Older cycle folders are pruned', group:'More learning options', advanced:true },
  { key:'maxNodes', label:'Nodes / turn', min:1, max:10000000, group:'Search budgets', advanced:true },
  { key:'maxDepth', label:'Search depth', min:0, max:64, help:'Maximum full turns; 0 grows after the top 20 moves at every depth have True Evaluations', group:'Search budgets', advanced:true },
  { key:'timeMs', label:'Time / turn (ms)', min:1, max:60000, group:'Search budgets', advanced:true },
  { key:'terminalWork', label:'Terminal work', min:1, max:10000000, help:'Verify checkmate or stalemate', group:'Search budgets', advanced:true },
  { key:'arenaPairs', label:'Arena pairs', min:1, max:128, help:'Each start played both colors', group:'Arena & promotion', advanced:true },
  { key:'arenaConcurrency', label:'Concurrent arena games', min:1, max:8, help:'Games share candidate and incumbent models; higher values use more CPU and memory', group:'Arena & promotion', advanced:true },
  { key:'minPairs', label:'Minimum pairs', min:1, max:128, help:'Complete pairs for promotion', group:'Arena & promotion', advanced:true },
  { key:'arenaPlies', label:'Arena turn cap', min:1, max:256, group:'Arena & promotion', advanced:true },
  { key:'promotionScore', label:'Promotion score', min:.5, exclusiveMin:true, max:1, step:'any', help:'Greater than 0.5; up to 1', group:'Arena & promotion', advanced:true },
];
const freshFields = [
  { key:'samples', label:'Labeled positions', min:1, max:100000, group:'Training data' },
  { key:'teacherNodes', label:'Teacher nodes', min:10, max:1000000, help:'Search budget per position', group:'Training data' },
  { key:'teacherTimeMs', label:'Teacher time (ms)', min:1, max:60000, help:'Time budget per position', group:'Training data' },
  { key:'device', label:'Device', choices:[['auto','Automatic'],['cuda','CUDA / GPU'],['cpu','CPU']], group:'Learning' },
  { key:'steps', label:'Training updates', min:1, max:1000000, group:'Learning' },
  { key:'batchSize', label:'Batch size', min:1, max:128, help:'Start with 1 to limit GPU memory', group:'Learning' },
  { key:'maxTokens', label:'Context tokens', min:16, max:4096, help:'512 is the starting memory budget', group:'Learning' },
  { key:'learningRate', label:'Learning rate', min:0, exclusiveMin:true, max:.1, step:'any', group:'Learning' },
  { key:'seed', label:'Random seed', min:0, max:4294967295, group:'Learning' },
];
const glyphs = {1:'♟',2:'♟',3:'♝',4:'♝',5:'♞',6:'♞',7:'♜',8:'♜',9:'♛',10:'♛',11:'♚',12:'♚',13:'S',14:'S',15:'W',16:'W',17:'C',18:'C',19:'Y',20:'Y',21:'U',22:'U',23:'D',24:'D'};
const pieceNames = {1:'pawn',3:'bishop',5:'knight',7:'rook',9:'queen',11:'king',13:'princess',15:'brawn',17:'common king',19:'royal queen',21:'unicorn',23:'dragon'};
const results = {WHITE_WIN:'White win',BLACK_WIN:'Black win',A_WIN:'Candidate win',B_WIN:'Incumbent win',DRAW:'Draw',UNFINISHED:'Unfinished'};
let snapshot = null, defaults = null, online = false, mutation = false, refreshing = false;
let trainingMode = 'fresh20m';
let selectedIteration = null, iterationData = null, selectedGame = null, replay = null;
let detailRequest = 0, replayRequest = 0, pendingReplay = false, toastTimer, pollTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}
function number(value) { return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString() : '—'; }
function percent(value) { return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—'; }
function human(value) { return String(value || '').replaceAll('-', ' ').replaceAll('_', ' '); }
function dateLabel(value) { const date = new Date(value); return Number.isFinite(date.getTime()) ? date.toLocaleString([], {month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}) : ''; }
function resultLabel(game) { return game?.valid === false ? 'Invalid' : results[game?.result] || human(game?.result) || 'Pending'; }
function resultClass(game) { return game?.valid === false ? 'invalid' : game?.result === 'UNFINISHED' ? 'unfinished' : ''; }
function activeRun() { return ['running','stopping','external'].includes(snapshot?.status?.state); }
function activeFields() { return trainingMode === 'fresh20m' ? freshFields : fields; }
function modeDefaults() { return trainingMode === 'fresh20m' ? snapshot?.freshDefaults : trainingMode === 'selfplay20m' ? {...defaults,batchSize:1,maxTokens:512} : defaults; }
function modeAvailability() {
  if (trainingMode === 'fresh20m') return snapshot?.freshAvailability;
  if (trainingMode === 'selfplay20m') return snapshot?.training20mAvailability || (snapshot?.model20m?.available ? snapshot?.freshAvailability : {available:false,reason:'Complete a fresh 20M run before starting self-play with it.'});
  return snapshot?.availability;
}
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $('training-toast').textContent = message;
  $('training-toast').classList.toggle('error', error);
  $('training-toast').hidden = false;
  toastTimer = setTimeout(() => { $('training-toast').hidden = true; }, error ? 7000 : 3500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache:'no-store'} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { throw new Error(`The local server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(data.error || data.message || `Request failed (${response.status}).`);
  return data;
}

function validField(field, value) {
  if (field.choices) return field.choices.some(([key]) => key === value);
  return typeof value === 'number' && Number.isFinite(value) && (field.step || Number.isSafeInteger(value))
    && (field.exclusiveMin ? value > field.min : value >= field.min) && value <= field.max;
}
function createParameters(values) {
  values ||= {};
  const groups = new Map();
  for (const field of activeFields()) {
    if (!Object.hasOwn(values, field.key)) continue;
    if (!groups.has(field.group)) {
      const group = element(field.advanced ? 'details' : 'section', field.advanced ? 'parameter-advanced' : 'parameter-group');
      group.append(element(field.advanced ? 'summary' : 'h3', '', field.group));
      const grid = element('div','parameter-grid');
      group.append(grid);
      groups.set(field.group, {group,grid});
    }
    const label = element('label','parameter-field');
    label.htmlFor = `param-${field.key}`;
    label.append(element('span','',field.label));
    const input = element(field.choices ? 'select' : 'input');
    input.id = `param-${field.key}`;
    input.name = field.key;
    if (field.choices) {
      for (const [value, title] of field.choices) { const option = element('option','',title); option.value = value; input.append(option); }
    } else {
      input.type = 'number'; input.min = String(field.min); input.max = String(field.max); input.step = String(field.step || 1); input.required = true;
      input.inputMode = field.step ? 'decimal' : 'numeric';
    }
    input.value = String(values[field.key]);
    label.append(input);
    if (field.help) { const help = element('small','',field.help); help.id = `${input.id}-help`; input.setAttribute('aria-describedby',help.id); label.append(help); }
    groups.get(field.group).grid.append(label);
  }
  $('parameter-groups').replaceChildren(...[...groups.values()].map(item => item.group));
  try {
    const saved = JSON.parse(localStorage.getItem(modeStorageKeys[trainingMode]));
    applyParameters(saved);
  } catch { /* Storage may be unavailable; the server defaults still apply. */ }
  validateParameters();
}
function applyParameters(values) {
  for (const field of activeFields()) {
    const input = $(`param-${field.key}`);
    if (input && validField(field, values?.[field.key])) input.value = String(values[field.key]);
  }
}
function validateParameters() {
  let problem = null;
  const options = {};
  for (const field of activeFields()) {
    const input = $(`param-${field.key}`);
    if (!input) continue;
    input.setCustomValidity('');
    const value = field.choices ? input.value : input.value.trim() === '' ? NaN : Number(input.value);
    options[field.key] = value;
    if (!validField(field, value)) {
      const message = field.choices ? `Choose a valid ${field.label.toLowerCase()}.` : `${field.label} must be ${field.step ? 'a number' : 'a whole number'} ${field.exclusiveMin ? 'greater than' : 'from'} ${field.min} ${field.exclusiveMin ? 'and no more than' : 'to'} ${field.max}.`;
      input.setCustomValidity(message); problem ||= {input,message};
    }
  }
  if (!problem && trainingMode !== 'fresh20m' && options.minPairs > options.arenaPairs) {
    const input = $('param-minPairs'), message = 'Minimum completed pairs cannot exceed the number of arena pairs.';
    input.setCustomValidity(message); problem = {input,message};
  }
  $('parameter-error').textContent = problem?.message || '';
  $('parameter-error').hidden = !problem;
  return {options,problem};
}
function saveParameters() {
  const {options,problem} = validateParameters();
  if (!problem) { try { localStorage.setItem(modeStorageKeys[trainingMode], JSON.stringify(options)); } catch { /* Session settings still work. */ } }
}
function selectMode(mode) {
  trainingMode = mode;
  $('training-mode').value = mode;
  createParameters(modeDefaults());
  renderMode(); renderMonitor(); updateControls();
}
function renderMode() {
  const fresh = trainingMode === 'fresh20m', current = trainingMode === 'current';
  $('model-summary').hidden = current;
  $('mode-description').textContent = fresh ? 'Generate labeled positions, then train a new model from scratch.' : current ? 'Continue self-play with the configured current checkpoint.' : 'Continue self-play with the latest completed 20M model.';
  $('parameter-note').textContent = fresh ? 'Each fresh run starts with new weights and saves its own checkpoint. Completed 20M models are available for self-play.' : 'Each cycle generates games, trains a candidate, then evaluates both colors. Passing the arena gate promotes the candidate for this model.';
}
function updateControls() {
  const state = snapshot?.status?.state;
  const availability = modeAvailability(), ready = Boolean(modeDefaults());
  $('parameter-fields').disabled = !ready || mutation || activeRun();
  $('training-mode').disabled = !defaults || mutation || activeRun();
  $('start-training').disabled = !online || !ready || mutation || activeRun() || availability?.available !== true;
  $('stop-training').disabled = !online || mutation || state !== 'running';
  $('stop-training').textContent = state === 'stopping' ? 'Stopping…' : 'Stop run';
  $('start-training').textContent = mutation && state !== 'running' && state !== 'stopping' ? 'Starting…' : trainingMode === 'fresh20m' ? '▷  Start fresh 20M' : '▷  Start self-play';
  $('reset-parameters').disabled = !ready || mutation || activeRun();
  $('reuse-parameters').disabled = !iterationData?.report?.options || mutation || activeRun();
  $('refresh-training').disabled = refreshing;
  if (!online) $('run-hint').textContent = 'Waiting for the local server. Use Refresh to reconnect.';
  else if (state === 'external') $('run-hint').textContent = 'A training process started outside this page is active. Stop it in its original terminal before starting a new run.';
  else if (state === 'stopping') $('run-hint').textContent = 'Stopping owned processes and retaining completed results…';
  else if (state === 'running') $('run-hint').textContent = 'Training is running locally. You can review saved games while it works.';
  else if (!availability?.available) $('run-hint').textContent = availability?.reason || 'Training is not available. Check the local transformer Python environment.';
  else $('run-hint').textContent = 'Ready to train. Settings are saved in this browser.';
}

function renderMonitor() {
  const status = snapshot?.status || {state:'idle'};
  const state = status.state || 'idle';
  const phase = status.phase || '';
  const fresh = status.mode ? status.mode === 'fresh20m' : trainingMode === 'fresh20m';
  const events = Array.isArray(status.events) ? status.events : [];
  const trainingProgress = [...events].reverse().find(event => event && typeof event === 'object' && (event.step !== undefined || event.trainedSteps !== undefined));
  const dataProgress = [...events].reverse().find(event => event?.event === 'data-progress');
  const step = trainingProgress?.step ?? trainingProgress?.trainedSteps;
  const label = {idle:'Ready when you are',running:'Training in progress',stopping:'Stopping this run',completed:'Run completed',interrupted:'Run interrupted',failed:'Run needs attention',external:'External run in progress'}[state] || human(state);
  $('run-state').textContent = state.toUpperCase();
  $('run-state').classList.toggle('searching', state === 'running');
  $('run-dot').style.background = state === 'failed' ? '#e39e91' : activeRun() ? 'var(--accent)' : '#657c8e';
  $('run-title').textContent = status.mode && state !== 'idle' ? `${fresh ? '20M model' : 'Self-play'} · ${label.toLowerCase()}` : label;
  const progressText = fresh ? /data|generat/.test(phase) && dataProgress ? ` · ${number(dataProgress.samples)}${dataProgress.total ? ` / ${number(dataProgress.total)}` : ''} positions` : step !== undefined ? ` · update ${number(step)}${typeof trainingProgress.loss === 'number' ? ` · loss ${trainingProgress.loss.toFixed(4)}` : ''}` : '' : '';
  const description = {idle:trainingMode === 'fresh20m' ? 'Ready to generate labeled data and train a fresh 20M model.' : 'Choose your parameters and start a self-play training cycle.',running:`${human(phase) || 'Starting local training'}${progressText}${status.startedAt ? ` · started ${dateLabel(status.startedAt)}` : ''}`,stopping:'Stopping local training and keeping saved results.',completed:fresh ? 'The new checkpoint is saved below. Select Self-play · 20M model to continue training.' : 'The cycle results and recorded games are ready to review.',interrupted:fresh ? 'Saved data and partial checkpoints were retained. A new fresh run starts from scratch.' : 'Completed results were retained. Start another run to continue training.',failed:'Review the error and training log before starting another run.',external:'A separate local process owns the training run. Its retained games appear below.'}[state];
  $('run-detail').textContent = description || human(phase);
  $('run-counter-label').textContent = fresh ? 'UPDATE' : 'ITERATION';
  $('current-iteration').textContent = number(fresh ? step : status.iteration);
  const pipeline = document.querySelector('.pipeline');
  const pipelineMode = fresh ? 'fresh20m' : 'selfplay';
  if (pipeline.dataset.mode !== pipelineMode) {
    const stages = fresh ? [['data','Training data','Teacher policy + value'],['training','Train 20M','Fresh weights'],['complete','Save model','Ready for self-play']] : [['selfplay','Self-play','Generate experience'],['training','Train','Update candidate'],['arena','Arena','Evaluate both colors'],['complete','Promotion','Apply the score gate']];
    pipeline.replaceChildren(...stages.map(([key,title,detail],index) => {
      const item = element('li'); item.dataset.phase = key;
      const text = element('div'); text.append(element('strong','',title),element('small','',detail));
      item.append(element('span','',String(index+1).padStart(2,'0')),text); return item;
    }));
    pipeline.dataset.mode = pipelineMode;
  }
  const phaseIndex = fresh ? /data|generat/.test(phase) ? 0 : phase.startsWith('train') ? 1 : /complete|sav/.test(phase) ? 2 : -1 : phase.startsWith('selfplay') ? 0 : phase.startsWith('train') ? 1 : phase.startsWith('arena') ? 2 : /promot|complete|evaluat/.test(phase) ? 3 : -1;
  [...document.querySelectorAll('.pipeline li')].forEach((item,index) => {
    item.classList.toggle('active', activeRun() && index === phaseIndex);
    item.classList.toggle('done', state === 'completed' || (activeRun() && index < phaseIndex));
    if (activeRun() && index === phaseIndex) item.setAttribute('aria-current','step'); else item.removeAttribute('aria-current');
  });
  $('run-error').hidden = !status.error;
  $('run-error').textContent = status.error || '';
  $('event-count').textContent = `${events.length} event${events.length === 1 ? '' : 's'}`;
  const eventText = events.length ? events.map(event => typeof event === 'string' ? event : JSON.stringify(event)).join('\n') : 'No events for this server session.';
  if ($('event-log').textContent !== eventText) {
    const follow = $('event-log').scrollHeight - $('event-log').scrollTop - $('event-log').clientHeight < 35;
    $('event-log').textContent = eventText;
    if (follow) $('event-log').scrollTop = $('event-log').scrollHeight;
  }
}
function renderFreshRuns() {
  const model = snapshot?.model20m;
  $('model20m-state').textContent = model?.available ? 'READY FOR SELF-PLAY' : 'NO MODEL YET';
  $('model20m-detail').textContent = model?.available ? `Self-play uses the latest completed 20M model (${number(model.parameters)} parameters · ${number(model.trainedSteps)} updates). ${model.checkpoint || ''}` : 'Complete a fresh run to make its checkpoint available for self-play. Each run keeps its own data and checkpoint.';
  const runs = snapshot?.freshRuns || [];
  $('fresh-runs').replaceChildren(...(runs.length ? runs.map(run => {
    const item = element('article','fresh-run');
    const heading = element('div','fresh-run-heading');
    heading.append(element('strong','',dateLabel(run.startedAt) || run.id),element('span',`result-badge${run.status === 'failed' ? ' invalid' : run.status === 'interrupted' ? ' unfinished' : ''}`,human(run.status) || 'Starting'));
    const details = [run.parameters ? `${number(run.parameters)} parameters` : '20M architecture',run.trainedSteps !== undefined ? `${number(run.trainedSteps)} updates` : '',typeof run.loss === 'number' ? `loss ${run.loss.toFixed(4)}` : ''].filter(Boolean).join(' · ');
    item.append(heading,element('p','',details));
    if (run.checkpoint) item.append(element('code','checkpoint-path',run.checkpoint));
    if (run.error) item.append(element('p','form-error',run.error));
    return item;
  }) : [element('p','library-empty','No fresh runs yet. Start a fresh 20M model from the run configuration.')]));
}
function renderIterationChoices() {
  const iterations = snapshot?.iterations || [];
  const select = $('iteration-select');
  const previous = selectedIteration;
  if (!iterations.some(item => item.id === selectedIteration)) selectedIteration = iterations[0]?.id || null;
  const signature = JSON.stringify(iterations.map(item => [item.id,item.iteration,item.status,item.promoted,item.model,item.modelRunId]));
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...(iterations.length ? iterations.map(item => {
      const modelLabel = item.model === '20m' ? `20M${item.modelRunId ? ` · ${String(item.modelRunId).replace(/^fresh-/, '').slice(0,8)}` : ''}` : item.model === 'current' ? 'Current' : '';
      const option = element('option','',`${modelLabel ? `${modelLabel} · ` : ''}Iteration ${item.iteration} · ${item.promoted ? 'promoted' : human(item.status) || 'in progress'}`);
      option.value = item.id; return option;
    }) : [element('option','','No retained iterations')]));
    select.dataset.signature = signature;
  }
  if (selectedIteration) select.value = selectedIteration;
  select.disabled = !iterations.length;
  if (!selectedIteration) {
    iterationData = null;
    $('iteration-results').hidden = true;
    $('iteration-empty').hidden = false;
    $('iteration-empty').replaceChildren(element('span','empty-symbol','↗'),element('p','','No iterations yet'),element('small','','Start a run to collect games, training metrics, and arena results.'));
    clearReview('Choose an iteration to explore its games.');
  }
  return previous !== selectedIteration;
}
async function refreshTraining({manual = false} = {}) {
  if (refreshing) return;
  refreshing = true; updateControls();
  try {
    const previousState = snapshot?.status?.state;
    snapshot = await api('/api/training');
    online = true;
    if (!defaults) { defaults = snapshot.defaults || {}; createParameters(modeDefaults()); renderMode(); }
    $('training-connection').classList.add('online');
    $('training-connection').replaceChildren(element('span','status-dot'),document.createTextNode('Local training'));
    renderMonitor();
    renderFreshRuns();
    const changed = renderIterationChoices();
    if (selectedIteration && (changed || manual || activeRun() || ['running','stopping','external'].includes(previousState) || ['running','evaluated'].includes(iterationData?.report?.status))) await loadIteration(selectedIteration, !changed);
  } catch (error) {
    online = false;
    $('training-connection').classList.remove('online');
    $('training-connection').replaceChildren(element('span','status-dot'),document.createTextNode('Disconnected'));
    $('run-title').textContent = 'Training connection unavailable';
    $('run-detail').textContent = error.message;
    if (manual) toast(error.message, true);
  } finally {
    refreshing = false; updateControls();
    clearTimeout(pollTimer);
    pollTimer = setTimeout(() => { if (document.hidden) scheduleHiddenPoll(); else void refreshTraining(); }, activeRun() ? 2500 : 5000);
  }
}
function scheduleHiddenPoll() { clearTimeout(pollTimer); pollTimer = setTimeout(() => { if (document.hidden) scheduleHiddenPoll(); else void refreshTraining(); }, 5000); }

function metric(label, value, detail) {
  const node = element('div','training-metric');
  node.append(element('span','',label),element('strong','',value),element('small','',detail));
  return node;
}
function renderIteration() {
  const report = iterationData?.report || {};
  const games = iterationData?.games || [];
  const selfplay = report.selfplay;
  const arena = report.arena?.decision;
  const stopped = ['failed','interrupted','incomplete'].includes(report.status);
  const selfplayGames = games.filter(game => game.kind === 'selfplay');
  const gameCount = selfplay?.games ?? selfplayGames.length;
  const finishedCount = selfplay?.finished ?? selfplayGames.filter(game => game.valid !== false && game.result && game.result !== 'UNFINISHED').length;
  const unfinishedCount = selfplay?.unfinished ?? selfplayGames.filter(game => game.result === 'UNFINISHED').length;
  const samples = selfplay?.samples ?? (selfplayGames.length ? selfplayGames.reduce((total,game) => total + (game.samples || 0), 0) : undefined);
  $('iteration-empty').hidden = true;
  $('iteration-results').hidden = false;
  $('iteration-description').textContent = `${human(report.status) || 'In progress'}${report.startedAt ? ` · ${dateLabel(report.startedAt)}` : ''}${report.seed !== undefined ? ` · seed ${report.seed}` : ''}`;
  $('iteration-metrics').replaceChildren(
    metric('Self-play games',number(gameCount),`${number(finishedCount)} finished · ${number(unfinishedCount)} unfinished`),
    metric('New samples',number(samples),selfplay ? `${number(selfplay.outcomeSamples)} outcome · ${number(selfplay.bootstrapSamples)} search` : 'From completed search targets'),
    metric('Replay buffer',number(report.replay?.samples),report.replay ? 'Unique retained positions' : stopped ? 'Training did not finish' : 'Waiting for training'),
    metric('Candidate score',percent(arena?.candidateScore),arena ? `${number(arena.eligiblePairs)} eligible pairs` : stopped ? 'No arena result saved' : 'Waiting for arena results'),
  );
  const promotion = $('promotion-notice');
  promotion.classList.toggle('promoted', report.promoted === true);
  const lead = stopped ? `Iteration ${human(report.status)}. ` : report.promoted ? 'Candidate promoted. ' : arena ? 'Current model retained. ' : 'Promotion pending. ';
  const decision = arena ? `${human(arena.reason)}. Required score: ${percent(arena.promotionScore)} with ${number(arena.minPairs)} complete, distinct pairs.` : '';
  const reason = [report.error, report.promoted && stopped ? 'The candidate was promoted before this iteration stopped.' : '', decision].filter(Boolean).join(' ')
    || (stopped ? 'Saved games and completed results remain available. Start another run to continue training.' : 'The candidate is evaluated after self-play and training complete.');
  promotion.replaceChildren(element('strong','',lead),document.createTextNode(reason));
  const outcomes = element('div'); outcomes.append(element('strong','','Self-play outcomes'));
  if (selfplay) {
    outcomes.append(element('p','',`${number(selfplay.whiteWins)} White wins · ${number(selfplay.blackWins)} Black wins · ${number(selfplay.draws)} draws`),element('p','',`${number(selfplay.unfinished)} unfinished · ${number(selfplay.invalid)} invalid · ${number(selfplay.discardedSamples)} discarded samples`));
    const reasons = Object.entries(selfplay.unfinishedReasons || {}).map(([reason,count]) => `${human(reason)}: ${count}`).join(' · ');
    if (reasons) outcomes.append(element('p','',reasons));
  } else outcomes.append(element('p','','A full outcome summary is saved when self-play finishes.'));
  const arenaDetails = element('div'); arenaDetails.append(element('strong','','Arena gate'));
  if (arena) {
    arenaDetails.append(element('p','',`Candidate ${number(arena.candidatePoints)} points · current model ${number(arena.incumbentPoints)} points`),element('p','',`${number(arena.eligibleGames)} eligible games · ${number(arena.invalidGames)} invalid · ${number(arena.excludedPairs)} excluded pairs`));
    if (arena.completion) arenaDetails.append(element('p','',`Game completion ${percent(arena.completion.gameCompletionRate)} · pair completion ${percent(arena.completion.pairCompletionRate)}`));
  } else arenaDetails.append(element('p','','No arena decision saved for this iteration.'));
  $('outcome-details').replaceChildren(outcomes,arenaDetails);
  $('training-log').textContent = iterationData.log || 'No training log saved for this iteration.';
  renderGameList(); updateControls();
}
async function loadIteration(id, preserve = false) {
  const request = ++detailRequest;
  if (!preserve) {
    iterationData = null; clearReview('Loading saved games…');
    $('iteration-results').hidden = true;
    $('iteration-empty').hidden = false;
    $('iteration-empty').replaceChildren(element('span','empty-symbol','↗'),element('p','','Loading iteration…'));
  }
  try {
    const data = await api(`/api/training/iterations/${encodeURIComponent(id)}`);
    if (request !== detailRequest || id !== selectedIteration) return;
    iterationData = data; renderIteration();
    if (selectedGame && !data.games?.some(game => game.id === selectedGame)) clearReview('The selected game is no longer retained. Choose another game.');
    if (!selectedGame) {
      const first = filteredGames()[0];
      if (first) await loadGame(first.id, 0);
    }
  } catch (error) {
    if (request !== detailRequest || id !== selectedIteration) return;
    $('iteration-empty').hidden = false;
    $('iteration-empty').replaceChildren(element('p','','Could not load this iteration'),element('small','',error.message));
    if (!preserve) $('iteration-results').hidden = true;
    $('review-error').hidden = false; $('review-error').textContent = error.message;
  }
}
function filteredGames() {
  const kind = $('game-kind').value, outcome = $('game-outcome').value, query = $('game-search').value.trim().toLowerCase();
  return (iterationData?.games || []).filter(game => (kind === 'all' || game.kind === kind)
    && (outcome === 'all' || (outcome === 'invalid' ? game.valid === false : outcome === 'unfinished' ? game.result === 'UNFINISHED' && game.valid !== false : game.valid !== false && game.result && game.result !== 'UNFINISHED'))
    && (!query || `${game.id} ${game.label} ${game.reason} ${resultLabel(game)}`.toLowerCase().includes(query)));
}
function renderGameList() {
  const games = filteredGames(), total = iterationData?.games?.length || 0;
  $('game-count').textContent = `${games.length === total ? total : `${games.length} of ${total}`} game${total === 1 ? '' : 's'}`;
  const signature = JSON.stringify([selectedGame,games]);
  if ($('game-list').dataset.signature === signature) return;
  $('game-list').dataset.signature = signature;
  $('game-list').replaceChildren(...(games.length ? games.map(game => {
    const button = element('button','game-item'); button.type = 'button'; button.dataset.gameId = game.id;
    button.setAttribute('aria-pressed',String(selectedGame === game.id));
    const top = element('span','game-item-type'); top.append(element('span','',game.kind === 'arena' ? 'Arena' : 'Self-play'),element('span','',`#${game.id.split('-').at(-1).replace(/^0+(?=\d)/,'')}`));
    const bottom = element('span','game-item-bottom'); bottom.append(element('span',`result-badge ${resultClass(game)}`,resultLabel(game)),element('span','',`${game.plies ?? 0} turns`));
    button.append(top,element('strong','',game.label || game.id),bottom);
    button.addEventListener('click',() => void loadGame(game.id, 0));
    return button;
  }) : [element('p','library-empty',total ? 'No games match these filters.' : 'Games appear here as each game finishes.')]));
}
function clearReview(message) {
  selectedGame = null; replay = null; replayRequest++; pendingReplay = false;
  $('review-content').hidden = true; $('review-empty').hidden = false; $('review-error').hidden = true;
  $('review-viewport').removeAttribute('aria-busy');
  if (message) $('game-list').replaceChildren(element('p','library-empty',message));
  delete $('game-list').dataset.signature;
  $('game-count').textContent = '0 games';
}
async function loadGame(id, ply) {
  if (!selectedIteration) return;
  const iteration = selectedIteration, request = ++replayRequest;
  const changed = selectedGame !== id;
  selectedGame = id; pendingReplay = true; renderGameList();
  $('review-error').hidden = true;
  if (changed) { replay = null; $('review-content').hidden = true; $('review-empty').hidden = false; $('review-empty').querySelector('h3').textContent = 'Loading recorded position…'; }
  $('review-viewport').setAttribute('aria-busy','true'); updatePlayback();
  try {
    const data = await api(`/api/training/iterations/${encodeURIComponent(iteration)}/games/${encodeURIComponent(id)}?ply=${ply}`);
    if (request !== replayRequest || iteration !== selectedIteration || id !== selectedGame) return;
    replay = data;
    $('review-empty').hidden = true; $('review-content').hidden = false;
    renderReplay();
  } catch (error) {
    if (request !== replayRequest) return;
    $('review-error').textContent = error.message; $('review-error').hidden = false;
    if (!replay) { $('review-empty').hidden = true; $('review-content').hidden = true; }
  } finally {
    if (request === replayRequest) { pendingReplay = false; $('review-viewport').removeAttribute('aria-busy'); updatePlayback(); }
  }
}
function updatePlayback() {
  const ply = replay?.ply ?? 0, count = replay?.totalPlies ?? 0;
  $('first-ply').disabled = !replay || pendingReplay || ply === 0;
  $('previous-ply').disabled = !replay || pendingReplay || ply === 0;
  $('next-ply').disabled = !replay || pendingReplay || ply >= count;
  $('last-ply').disabled = !replay || pendingReplay || ply >= count;
  $('review-ply').disabled = !replay || pendingReplay || count === 0;
  $('review-ply').max = String(count); $('review-ply').value = String(ply);
  $('ply-label').textContent = `${ply} / ${count}`;
}
function moveNotation(move) {
  if (Array.isArray(move?.notation)) return move.notation.join(' / ');
  if (move?.notation) return String(move.notation);
  if (Array.isArray(move?.action)) return move.action.map(raw => Array.isArray(raw) && raw.length > 1 ? `${coordinate(raw[0])} → ${coordinate(raw[1])}` : 'Recorded move').join(' / ');
  return 'Recorded turn';
}
function renderReplay() {
  const game = replay.game, ply = replay.ply, move = game.moves?.[ply - 1];
  const listing = iterationData?.games?.find(item => item.id === selectedGame);
  const title = listing?.label || game.startId || game.caseId || selectedGame;
  $('review-game-title').textContent = title;
  const extra = listing?.kind === 'arena' ? `Candidate plays ${game.aColor === 0 ? 'White' : 'Black'}` : `${number(game.samples)} samples`;
  $('review-game-detail').textContent = `${listing?.kind === 'arena' ? 'Arena' : 'Self-play'} · ${extra} · ${human(game.reason) || 'No termination reason'}`;
  $('review-game-result').textContent = resultLabel(game); $('review-game-result').className = `result-badge ${resultClass(game)}`;
  $('review-position-label').textContent = `${ply === 0 ? 'Initial position' : `After turn ${ply}`} · ${replay.position.action % 2 === 0 ? 'White' : 'Black'} to play`;
  $('review-moves').replaceChildren(...[null,...(game.moves || [])].map((turn,index) => {
    const li = element('li'), button = element('button'); button.type = 'button';
    button.append(element('span','move-number',index === 0 ? '0' : `${Math.floor((turn.ply ?? index - 1) / 2) + 1}${turn.color === 1 ? '…' : '.'}`),element('span','',index === 0 ? 'Initial position' : moveNotation(turn)));
    if (turn?.exploration?.explored) { const mark = element('span','move-exploration','◇'); mark.title = 'Exploration turn'; mark.setAttribute('aria-label','Exploration turn'); button.append(mark); }
    if (index === ply) button.setAttribute('aria-current','step');
    button.addEventListener('click',() => void seek(index)); li.append(button); return li;
  }));
  renderSearch(move);
  renderBoards(); updatePlayback();
  const current = $('review-moves').querySelector('[aria-current=step]');
  if (current) { const list = $('review-moves'); if (current.offsetTop < list.scrollTop || current.offsetTop + current.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = Math.max(0,current.offsetTop - list.offsetTop - list.clientHeight / 2); }
  requestAnimationFrame(scrollBoardsToPresent);
}
function renderSearch(move) {
  const search = move?.search;
  const score = typeof search?.score === 'number' && Number.isFinite(search.score) ? `${search.score > 0 ? '+' : ''}${(search.score / 100).toFixed(2)}` : '—';
  $('turn-search-metrics').replaceChildren(...[['Score · White',score],['Depth',number(search?.depth)],['Nodes',number(search?.nodes)]].map(([label,value]) => {
    const item = element('div'); item.append(element('span','',label),element('strong','',value)); return item;
  }));
  const exploration = move?.exploration;
  $('exploration-badge').hidden = !exploration?.explored;
  const details = search ? [search.completed ? 'Search completed' : 'Search incomplete',typeof search.elapsedMs === 'number' ? `${number(Math.round(search.elapsedMs))} ms` : null,search.stoppedReason ? human(search.stoppedReason) : null].filter(Boolean).join(' · ') : 'Choose a recorded turn to inspect its search.';
  $('turn-search-note').textContent = exploration?.explored ? `${details}. An exploration action was played; the score describes the searched action.` : `${details}${search ? '. Positive scores favor White.' : ''}`;
  $('raw-search-details').hidden = !search;
  $('raw-search').textContent = search ? JSON.stringify({search, ...(exploration ? {exploration} : {}), playedAction:move.action, ...(move.searchedAction ? {searchedAction:move.searchedAction} : {})}, null, 2) : '';
}
function timelineNumber(index) { return index === 0 ? 0 : index % 2 === 0 ? index / 2 : -(index + 1) / 2; }
function timelineLabel(index) { const value = timelineNumber(index); return replay?.isEvenTimeline && value !== 0 ? `${value > 0 ? '+' : '-'}${Math.abs(value) - 1}` : `${value > 0 ? '+' : ''}${value}`; }
function timeLabel(time) { return `T${Math.floor(time / 2) + (replay?.isTurnZero ? 0 : 1)}`; }
function coordinate(value) { if (!Array.isArray(value) || value.length < 4) return '?'; const [l,t,r,f] = value; return `(${timelineLabel(l)}L ${timeLabel(t)}) ${'abcdefghijklmnopqrstuvwxyz'[f] || f + 1}${r + 1}`; }
function latestTime(timeline) { for (let index = timeline.length - 1; index >= 0; index--) if (Array.isArray(timeline[index])) return index; return -1; }
function renderBoards() {
  if (!replay?.position) return;
  const viewport = $('review-viewport'), grid = $('review-grid');
  const scroll = {left:viewport.scrollLeft, top:viewport.scrollTop};
  const timelines = (replay.position.board || []).map((timeline,index) => ({timeline,index})).filter(({timeline}) => Array.isArray(timeline) && timeline.some(Array.isArray)).sort((a,b) => timelineNumber(b.index) - timelineNumber(a.index));
  const active = new Set(replay.active || []), history = $('review-history').checked, black = $('review-orientation').value === 'black';
  const times = [...new Set(timelines.flatMap(({timeline}) => timeline.map((squares,time) => Array.isArray(squares) && (history || time === latestTime(timeline)) ? time : null).filter(time => time !== null)))].sort((a,b) => a-b);
  const columns = new Map(times.map((time,index) => [time,index+2]));
  const activeTimes = timelines.filter(({index}) => active.has(index)).map(({timeline}) => latestTime(timeline));
  const presentTime = activeTimes.length ? Math.min(...activeTimes) : Math.min(...times);
  const highlights = new Set((replay.game.moves?.[replay.ply - 1]?.action || []).flatMap(move => Array.isArray(move) ? move.filter(Array.isArray).map(coord => coord.slice(0,4).join(',')) : []));
  const fragment = document.createDocumentFragment();
  grid.style.gridTemplateColumns = `68px repeat(${times.length},max-content)`;
  grid.style.setProperty('--square',`${$('review-board-size').value}px`);
  if (!timelines.length) { grid.replaceChildren(element('div','empty-universe','No boards in this recorded position.')); return; }
  const corner = element('div','grid-corner'); corner.style.gridColumn = '1'; corner.style.gridRow = '1'; fragment.append(corner);
  for (const time of times) {
    const head = element('div',`time-heading${time === presentTime ? ' present-heading' : ''}`,timeLabel(time));
    head.append(element('span','',time % 2 === 0 ? 'WHITE' : 'BLACK'));
    head.style.gridColumn = columns.get(time); head.style.gridRow = '1'; fragment.append(head);
  }
  for (const [row,{timeline,index:l}] of timelines.entries()) {
    const latest = latestTime(timeline), label = element('div',`timeline-label ${active.has(l) ? 'active' : 'inactive'}`);
    label.style.gridRow = row + 2;
    label.append(element('strong','',`${timelineLabel(l)} L`),element('span','timeline-kind',l === 0 ? 'Origin' : active.has(l) ? 'Active' : 'Inactive'));
    if (l !== 0) label.append(element('span','branch-line'));
    fragment.append(label);
    for (let time = 0; time < timeline.length; time++) {
      const squares = timeline[time];
      if (!Array.isArray(squares) || (!history && time !== latest)) continue;
      const cell = element('div','board-cell'); cell.style.gridColumn = columns.get(time); cell.style.gridRow = row + 2; cell.dataset.time = time;
      const current = time === latest && active.has(l);
      cell.dataset.current = String(current);
      const card = element('article',`board-card${current ? ' playable' : ''}${time < latest ? ' historical' : ''}`);
      card.setAttribute('aria-label',`Timeline ${timelineLabel(l)}, ${timeLabel(time)}, ${time % 2 === 0 ? 'White' : 'Black'} board`);
      const header = element('header'); header.append(element('span','',`${timelineLabel(l)}L · ${timeLabel(time)}`),element('span','board-state',time < latest ? 'Past' : current ? 'Active' : 'Inactive'));
      const board = element('div','board-squares');
      const ranks = squares.length, files = Math.max(0,...squares.map(rank => rank?.length || 0));
      board.style.setProperty('--ranks',ranks); board.style.setProperty('--files',files);
      const rankOrder = Array.from({length:ranks},(_,i) => black ? i : ranks-i-1), fileOrder = Array.from({length:files},(_,i) => black ? files-i-1 : i);
      for (const [ri,r] of rankOrder.entries()) for (const [fi,f] of fileOrder.entries()) {
        const piece = Math.abs(squares[r]?.[f] || 0), color = piece % 2 === 0 ? 'white' : 'black';
        const square = element('div',`square${(r+f)%2 === 0 ? ' dark' : ''}${highlights.has([l,time,r,f].join(',')) ? ' last-move' : ''}`);
        square.setAttribute('role','img');
        const description = `${coordinate([l,time,r,f])}, ${piece ? `${color} ${pieceNames[piece % 2 === 0 ? piece-1 : piece] || 'piece'}` : 'empty'}`;
        square.setAttribute('aria-label',description); square.title = description;
        if (piece) square.append(element('span',`piece ${color}${piece > 12 ? ' fairy' : ''}`,glyphs[piece] || '?'));
        if (fi === 0) square.append(element('span','rank-label',r+1));
        if (ri === ranks-1) square.append(element('span','file-label','abcdefghijklmnopqrstuvwxyz'[f] || f+1));
        board.append(square);
      }
      card.append(header,board); cell.append(card); fragment.append(cell);
    }
  }
  const line = element('div','present-line'); line.dataset.presentTime = presentTime; fragment.append(line);
  grid.replaceChildren(fragment); viewport.scrollLeft = scroll.left; viewport.scrollTop = scroll.top;
  positionPresentLine();
}
function positionPresentLine() {
  const line = $('review-grid').querySelector('.present-line');
  if (!line) return;
  const cells = [...$('review-grid').querySelectorAll('.board-cell')], time = Number(line.dataset.presentTime);
  const cell = cells.find(item => Number(item.dataset.time) === time) || cells.find(item => Number(item.dataset.time) > time);
  line.style.left = `${cell ? cell.offsetLeft + 3 : 68}px`; line.hidden = !cell;
}
function scrollBoardsToPresent() {
  const grid = $('review-grid'), viewport = $('review-viewport');
  const present = grid.querySelector('.present-line')?.dataset.presentTime;
  const cells = [...grid.querySelectorAll('.board-cell')];
  const target = cells.find(cell => cell.dataset.time === present && cell.dataset.current === 'true')
    || cells.find(cell => cell.dataset.time === present) || cells[0];
  if (!target) return;
  viewport.scrollLeft = Math.max(0,target.offsetLeft - 78);
  viewport.scrollTop = Math.max(0,target.offsetTop - 45);
}
function seek(ply) { if (!replay || !selectedGame || pendingReplay) return; const next = Math.max(0,Math.min(replay.totalPlies,ply)); if (next !== replay.ply) return loadGame(selectedGame,next); }

$('training-form').addEventListener('submit',async event => {
  event.preventDefault();
  const {options,problem} = validateParameters();
  if (problem) { problem.input.closest('details')?.setAttribute('open',''); problem.input.focus(); problem.input.reportValidity(); return; }
  if ($('start-training').disabled) return;
  saveParameters(); mutation = true; updateControls();
  try {
    const fresh = trainingMode === 'fresh20m';
    await api(fresh ? '/api/training/fresh/start' : '/api/training/start',{options:fresh ? options : {...options,model:trainingMode === 'selfplay20m' ? '20m' : 'current'}});
    toast(fresh ? 'Fresh 20M run started. Generating training data.' : 'Self-play training started.');
    await refreshTraining({manual:true});
  }
  catch (error) { $('run-error').textContent = error.message; $('run-error').hidden = false; toast(error.message,true); }
  finally { mutation = false; updateControls(); }
});
$('stop-training').addEventListener('click',async () => {
  if ($('stop-training').disabled) return;
  mutation = true; updateControls();
  try { await api('/api/training/stop',{}); toast('Stop requested. Completed results will be retained.'); await refreshTraining({manual:true}); }
  catch (error) { $('run-error').textContent = error.message; $('run-error').hidden = false; toast(error.message,true); }
  finally { mutation = false; updateControls(); }
});
$('parameter-fields').addEventListener('input',saveParameters);
$('parameter-fields').addEventListener('change',saveParameters);
$('training-mode').addEventListener('change',() => selectMode($('training-mode').value));
$('reset-parameters').addEventListener('click',() => { applyParameters(modeDefaults()); saveParameters(); toast('Restored default training parameters.'); });
$('reuse-parameters').addEventListener('click',() => { selectMode(iterationData?.report?.options?.model === '20m' ? 'selfplay20m' : 'current'); applyParameters(iterationData?.report?.options); saveParameters(); toast('Iteration parameters copied to the next run.'); $('param-iterations')?.focus(); });
$('refresh-training').addEventListener('click',() => void refreshTraining({manual:true}));
$('iteration-select').addEventListener('change',() => { selectedIteration = $('iteration-select').value; void loadIteration(selectedIteration); });
for (const id of ['game-kind','game-outcome']) $(id).addEventListener('change',renderGameList);
$('game-search').addEventListener('input',renderGameList);
$('first-ply').addEventListener('click',() => void seek(0));
$('previous-ply').addEventListener('click',() => void seek((replay?.ply || 0)-1));
$('next-ply').addEventListener('click',() => void seek((replay?.ply || 0)+1));
$('last-ply').addEventListener('click',() => void seek(replay?.totalPlies || 0));
$('review-ply').addEventListener('input',() => { $('ply-label').textContent = `${$('review-ply').value} / ${replay?.totalPlies || 0}`; });
$('review-ply').addEventListener('change',() => void seek(Number($('review-ply').value)));
for (const id of ['review-orientation','review-history']) $(id).addEventListener('change',() => { renderBoards(); scrollBoardsToPresent(); });
$('review-board-size').addEventListener('input',renderBoards);
$('game-review').addEventListener('keydown',event => {
  if (event.target.closest('input,select,textarea') || event.ctrlKey || event.altKey || event.metaKey) return;
  if (['ArrowLeft','ArrowRight','Home','End'].includes(event.key) && replay) {
    event.preventDefault();
    void seek(event.key === 'Home' ? 0 : event.key === 'End' ? replay.totalPlies : replay.ply + (event.key === 'ArrowLeft' ? -1 : 1));
  }
});
document.addEventListener('visibilitychange',() => { if (!document.hidden) void refreshTraining(); });
window.addEventListener('resize',positionPresentLine);
void refreshTraining();
