const $ = (id) => document.getElementById(id);
const files = 'abcdefghijklmnopqrstuvwxyz';
const glyphs = { 1:'♟', 2:'♟', 3:'♝', 4:'♝', 5:'♞', 6:'♞', 7:'♜', 8:'♜', 9:'♛', 10:'♛', 11:'♚', 12:'♚', 13:'S', 14:'S', 15:'W', 16:'W', 17:'C', 18:'C', 19:'Y', 20:'Y', 21:'U', 22:'U', 23:'D', 24:'D' };
const pieceNames = {1:'pawn',3:'bishop',5:'knight',7:'rook',9:'queen',11:'king',13:'princess',15:'brawn',17:'common king',19:'royal queen',21:'unicorn',23:'dragon'};
let game = null;
let selected = null;
let busy = false;
let search = null;
let pollTimer = null;
let toastTimer = null;
let autoTimer = null;
let autoRevision = null;
let analysisGeneration = 0;
let heuristicsPanel = null;
let initialPosition = true;
let rankingJob = null, selectedRankingDepth = null, visibleRankings = [];
let recommendationSignature = '', variationSignature = '', continuationAnimation;
const rankingTabs = new Map(), rankingRows = new Map();
let boardSize = Number($('board-size').value);
const searchSettingsKey = 'vibe-d-ai.search-settings.v1';
const transformerName = 'Transformer · 800k';
const resourceSettingIds = ['time-budget', 'search-depth', 'node-budget', 'cache-memory', 'search-threads'];
const searchSettingIds = ['engine-select', ...resourceSettingIds];
let refreshingEngines = false;
let engines = {
  classical: {id:'classical', available:true},
  transformer: {id:'transformer', available:false, status:'checking'},
};
function selectedEngine() { return $('engine-select').value; }
function neuralEngine(id = selectedEngine()) { return id === 'transformer'; }
function engineAvailable() { return engines[selectedEngine()]?.available === true; }
function currentHeuristicsSignature() { return selectedEngine() === 'classical' ? heuristicsPanel?.signature() || '' : ''; }
function searchMatchesPosition(current = search) {
  return Boolean(current && current.revision === game?.revision && current.engine === selectedEngine()
    && (current.heuristicsSignature || '') === currentHeuristicsSignature());
}
function engineIdleStatus() {
  const info = engines[selectedEngine()];
  return info?.status === 'checking' ? 'CHECKING' : info?.status === 'error' ? 'ERROR' : engineAvailable() ? 'READY' : 'SETUP';
}

function renderEngine() {
  const neural = neuralEngine();
  for (const option of $('search-depth').options) option.hidden = option.disabled = !neural && (Number(option.value) === 0 || Number(option.value) > 16);
  if (!neural && Number($('search-depth').value) === 0) $('search-depth').value = '4';
  if (!neural && Number($('search-depth').value) > 16) $('search-depth').value = '16';
  const info = engines[selectedEngine()] || {};
  const name = transformerName;
  const device = typeof info.device === 'string' ? ` · ${info.device}` : '';
  $('engine-readiness').textContent = neural
    ? info.status === 'error' ? info.error || `${name} could not load; check setup`
      : info.available ? info.status === 'unloaded' ? 'Checkpoint ready · loads on first analysis' : `${name} ready${device}`
        : info.status === 'checking' ? `Checking ${name} availability…` : info.status === 'starting' ? `${name} starting…` : info.error || `${name} setup required`
    : 'Classical search ready · CPU';
  $('engine-readiness').classList.toggle('unavailable', neural && !info.available);
  $('engine-description').textContent = neural
    ? '800k transformer with learned evaluation, bounded historical context and candidate turns. Playing strength is unmeasured; legal moves use the full rules.'
    : 'Full-turn search with a handcrafted position evaluation.';
  $('transformer-setup').hidden = !neural || (info.available && info.status !== 'error') || info.status === 'checking';
  $('engine-checkpoint').hidden = !neural || !info.checkpoint;
  $('engine-checkpoint').textContent = neural && info.checkpoint ? `Checkpoint: ${info.checkpoint}` : '';
  $('opponent-engine').textContent = `Engine plays uses ${neural ? transformerName : 'Classical search'}.`;
  $('cache-memory-help').textContent = neural
    ? 'Neural engines use their own bounded model memory. Search cache (RAM) applies to Classical search only.'
    : 'Cache budget estimates RAM for saved search results; total RAM is higher. This CPU engine does not use GPU VRAM.';
  $('search-threads-help').textContent = neural
    ? 'Search threads apply to Classical search only.'
    : 'Threads search turns in parallel. Time, node, and cache budgets apply to the whole search. More threads may help deeper searches; each thread also uses extra RAM.';
  $('refresh-engines').disabled = refreshingEngines;
  if (!search) $('search-status').textContent = engineIdleStatus();
  updateControls();
}
async function refreshEngines() {
  if (refreshingEngines) return;
  refreshingEngines = true;
  renderEngine();
  try {
    const data = await api('/api/engines');
    for (const info of data.engines || []) if (Object.hasOwn(engines, info.id)) engines[info.id] = info;
  } catch (error) {
    engines.transformer = {...engines.transformer, available:false, status:'error', error:`Availability check failed: ${error.message}`};
  } finally {
    refreshingEngines = false;
    renderEngine();
    scheduleOpponent();
  }
}

function restoreSearchSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(searchSettingsKey));
    for (const id of searchSettingIds) {
      const input = $(id), value = saved?.[id];
      if (value === undefined || value === null) continue;
      if (input.tagName === 'SELECT') {
        if ([...input.options].some(option => option.value === String(value))) input.value = String(value);
      } else if ((typeof value === 'number' || (typeof value === 'string' && value.trim() !== '')) && Number.isInteger(Number(value)) && Number(value) >= Number(input.min) && Number(value) <= Number(input.max)) {
        input.value = String(Number(value));
      }
    }
  } catch { /* Unavailable storage or old settings must not prevent play. */ }
}
function saveSearchSettings() {
  if (!$('node-budget').validity.valid) return;
  try {
    localStorage.setItem(searchSettingsKey, JSON.stringify(Object.fromEntries(searchSettingIds.map(id => [id, $(id).value]))));
  } catch { /* Settings still apply to this session if storage is unavailable. */ }
}

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function coordKey(coord) { return coord?.slice(0, 4).join(','); }
function timelineNumber(index) { return index === 0 ? 0 : index % 2 === 0 ? index / 2 : -(index + 1) / 2; }
function timelineLabel(index) {
  const n = timelineNumber(index);
  if (game?.isEvenTimeline && n !== 0) return `${n > 0 ? '+' : '-'}${Math.abs(n)-1}`;
  return `${n > 0 ? '+' : ''}${n}`;
}
function timeLabel(t) { return `T${Math.floor(t / 2) + (game?.isTurnZero ? 0 : 1)}`; }
function sideLabel(action = game?.position?.action ?? 0) { return action % 2 === 0 ? 'White' : 'Black'; }
function coordinateLabel([l,t,r,f]) { return `(${timelineLabel(l)}L ${timeLabel(t)}) ${files[f] || f + 1}${r + 1}`; }
function latestTime(timeline) { for (let t = timeline.length - 1; t >= 0; t--) if (Array.isArray(timeline[t])) return t; return -1; }
function notation(value) { if (typeof value === 'string') return value; if (Array.isArray(value)) return value.map(notation).join(' / '); return value?.notation || ''; }
function readableMove(move) { return notation(move) || (move?.raw?.length >= 2 ? `${coordinateLabel(move.raw[0])} → ${coordinateLabel(move.raw[1])}` : 'Move'); }
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.toggle('error', error);
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 8000 : 3500);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  let data;
  try { data = await response.json(); } catch { throw new Error(`The local engine returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(data.error || data.message || `Engine request failed (${response.status}).`);
  return data;
}
function setBusy(value) { busy = value; updateControls(); }
function updateControls() {
  const hasGame = Boolean(game);
  const running = search?.status === 'running';
  $('submit-button').disabled = !hasGame || busy || !game.canSubmit;
  $('undo-button').disabled = !hasGame || busy || (!(game.pending?.length) && !(game.history?.length));
  $('new-button').disabled = busy || !hasGame;
  $('analyze-button').disabled = !hasGame || busy || (!running && (!engineAvailable() || (selectedEngine() === 'classical' && heuristicsPanel && !heuristicsPanel.ready())));
  $('analyze-label').textContent = running ? 'Stop analysis' : 'Analyze position';
  $('analyze-button').classList.toggle('running', running);
  $('play-button').disabled = busy || !search?.result?.bestAction || !searchMatchesPosition() || running;
  $('import-pgn').disabled = busy;
  $('variant').disabled = busy;
  $('ai-side').disabled = busy;
  $('engine-select').disabled = busy;
  for (const id of resourceSettingIds) $(id).disabled = busy || running || (['cache-memory', 'search-threads'].includes(id) && neuralEngine());
  heuristicsPanel?.setContext({revision: game?.revision, engine: selectedEngine(), busy});
}
function receiveGame(next) {
  game = next;
  selected = null;
  $('loading-state').hidden = true;
  $('connection').classList.add('online');
  $('connection').replaceChildren(element('span','status-dot'), document.createTextNode('Local engine'));
  renderGame();
  if (initialPosition) { initialPosition = false; requestAnimationFrame(scrollToPresent); }
}
function renderGame() {
  if (!game) return;
  const action = game.position?.action ?? 0;
  const pending = game.pending || [];
  const timelineCount = (game.position?.board || []).filter(Array.isArray).length;
  $('turn-piece').textContent = action % 2 === 0 ? '♔' : '♚';
  $('turn-label').textContent = `${sideLabel()} to play`;
  $('turn-detail').textContent = game.canSubmit ? 'Ready to submit this turn' : `${(game.present || []).length} present board${game.present?.length === 1 ? '' : 's'} · ${game.active?.length || 0} active timeline${game.active?.length === 1 ? '' : 's'}`;
  $('check-badge').hidden = !game.inCheck;
  $('timeline-count').textContent = `${timelineCount} TIMELINE${timelineCount === 1 ? '' : 'S'}`;
  $('pending-summary').textContent = pending.length ? `${pending.length} move${pending.length === 1 ? '' : 's'} played · ${game.canSubmit ? 'ready to submit' : game.inCheck ? 'resolve check before submitting' : 'continue playing the present'}` : 'No moves played yet';
  $('pending-moves').replaceChildren(...pending.map(move => element('span','move-chip',readableMove(move))));
  const history = game.history || [];
  $('history-count').textContent = `${history.length} turn${history.length === 1 ? '' : 's'}`;
  $('move-history').replaceChildren(...(history.length ? history.map((turn,i) => {
    const li = element('li');
    li.append(element('span','history-number',`${Math.floor(i / 2) + 1}${i % 2 ? '…' : '.'}`),element('span','history-notation',readableMove(turn)));
    return li;
  }) : [element('li','empty-history','Your story begins on this board.')]));
  $('move-history').scrollTop = $('move-history').scrollHeight;
  if (game.variants?.length && !$('variant').dataset.loaded) {
    $('variant').replaceChildren(...game.variants.map(variant => {
      const option = element('option','',variant.name || variant.shortName);
      option.value = variant.shortName || variant.name;
      return option;
    }));
    const standard = [...$('variant').options].find(option => /^standard$/i.test(option.textContent) || /^standard$/i.test(option.value));
    if (standard) $('variant').value = standard.value;
    $('variant').dataset.loaded = 'true';
  }
  renderBoards();
  updateControls();
}
function renderBoards() {
  if (!game) return;
  const viewport = $('multiverse-viewport');
  const oldScroll = {left:viewport.scrollLeft,top:viewport.scrollTop};
  const grid = $('timeline-grid');
  const board = game.position?.board || [];
  const timelines = board.map((timeline,index) => ({index,timeline})).filter(({timeline}) => Array.isArray(timeline) && timeline.some(Array.isArray)).sort((a,b) => timelineNumber(b.index) - timelineNumber(a.index));
  const active = new Set(game.active || []);
  const sourceKeys = new Set((game.moves || []).map(move => coordKey(move.raw?.[0])));
  const playableBoards = new Set((game.moves || []).map(move => move.raw?.[0]?.slice(0,2).join(',')));
  const choices = selected ? (game.moves || []).filter(move => coordKey(move.raw?.[0]) === coordKey(selected)) : [];
  const destinations = new Set(choices.map(move => coordKey(move.raw?.[1])));
  const destinationBoards = new Set(choices.map(move => move.raw?.[1]?.slice(0,2).join(',')));
  const lastMove = game.pending?.at(-1)?.raw;
  const lastCoords = new Set(lastMove?.slice(0,2).map(coordKey) || []);
  const showHistory = $('show-history').checked;
  const blackView = $('orientation').value === 'black';
  const allTimes = new Set();
  for (const {index:l,timeline} of timelines) {
    const latest = latestTime(timeline);
    for (let t = 0; t < timeline.length; t++) if (Array.isArray(timeline[t]) && (showHistory || t === latest || destinationBoards.has(`${l},${t}`))) allTimes.add(t);
  }
  const times = [...allTimes].sort((a,b) => a-b);
  const timeColumns = new Map(times.map((t,i) => [t,i+2]));
  const fragment = document.createDocumentFragment();
  grid.style.gridTemplateColumns = `${window.innerWidth <= 570 ? 70 : 94}px repeat(${times.length},max-content)`;
  grid.style.setProperty('--square',`${boardSize}px`);
  if (!timelines.length) { grid.replaceChildren(element('div','empty-universe','There are no boards in this position.')); return; }
  const currentTimes = timelines.filter(({index}) => active.has(index)).map(({timeline}) => latestTime(timeline));
  const presentTime = currentTimes.length ? Math.min(...currentTimes) : Math.min(...times);
  const corner = element('div','grid-corner'); corner.style.gridColumn = '1'; corner.style.gridRow = '1'; fragment.append(corner);
  for (const t of times) {
    const heading = element('div',`time-heading${t === presentTime ? ' present-heading' : ''}`,timeLabel(t));
    heading.append(element('span','',t % 2 === 0 ? 'WHITE' : 'BLACK'));
    heading.style.gridColumn = timeColumns.get(t); heading.style.gridRow = '1'; fragment.append(heading);
  }
  for (const [row,{index:l,timeline}] of timelines.entries()) {
    const latest = latestTime(timeline);
    const label = element('div',`timeline-label ${active.has(l) ? 'active' : 'inactive'}`);
    label.style.gridRow = row + 2;
    label.append(element('strong','',`${timelineLabel(l)} L`),element('span','timeline-kind',l === 0 ? 'Origin' : active.has(l) ? 'Active' : 'Inactive'));
    if (l !== 0) label.append(element('span','branch-line'));
    fragment.append(label);
    for (let t = 0; t < timeline.length; t++) {
      const squares = timeline[t];
      if (!Array.isArray(squares) || (!showHistory && t !== latest && !destinationBoards.has(`${l},${t}`))) continue;
      const cell = element('div','board-cell');
      cell.style.gridColumn = timeColumns.get(t); cell.style.gridRow = row + 2;
      cell.dataset.time = t;
      const playable = playableBoards.has(`${l},${t}`);
      const card = element('article',`board-card${playable ? ' playable' : ''}${t < latest ? ' historical' : ''}${selected?.[0] === l && selected?.[1] === t ? ' selected-board' : ''}${destinationBoards.has(`${l},${t}`) ? ' has-destination' : ''}${t > presentTime ? ' future-board' : ''}`);
      card.setAttribute('aria-label',`Timeline ${timelineLabel(l)}, ${timeLabel(t)}, ${t % 2 === 0 ? 'White' : 'Black'} board`);
      const head = element('header');
      head.append(element('span','',`${timelineLabel(l)}L · ${timeLabel(t)}`),element('span','board-state',playable ? 'Play' : t < latest ? 'Past' : t > presentTime ? 'Ahead' : 'Waiting'));
      const squaresEl = element('div','board-squares');
      const ranks = squares.length;
      const fileCount = Math.max(0,...squares.map(rank => rank?.length || 0));
      squaresEl.style.setProperty('--ranks',ranks); squaresEl.style.setProperty('--files',fileCount);
      const rankOrder = Array.from({length:ranks},(_,i) => blackView ? i : ranks-i-1);
      const fileOrder = Array.from({length:fileCount},(_,i) => blackView ? fileCount-i-1 : i);
      for (const [rankIndex,r] of rankOrder.entries()) for (const [fileIndex,f] of fileOrder.entries()) {
        const value = squares[r]?.[f] || 0;
        const absolutePiece = Math.abs(value);
        const coord = [l,t,r,f];
        const key = coordKey(coord);
        const interactive = sourceKeys.has(key) || destinations.has(key);
        const button = element('button',`square${(r+f)%2 === 0 ? ' dark' : ''}${value ? ' occupied' : ''}${interactive ? ' interactive' : ''}${destinations.has(key) ? ' destination' : ''}${coordKey(selected) === key ? ' selected' : ''}${lastCoords.has(key) ? ' last-move' : ''}`);
        button.type = 'button';
        const color = absolutePiece % 2 === 0 ? 'white' : 'black';
        const pieceName = value ? `${color} ${pieceNames[absolutePiece % 2 === 0 ? absolutePiece-1 : absolutePiece] || 'piece'}` : 'empty';
        button.setAttribute('aria-label',`${coordinateLabel(coord)}, ${pieceName}${destinations.has(key) ? ', available destination' : ''}`);
        button.title = `${coordinateLabel(coord)} · ${pieceName}`;
        button.tabIndex = interactive ? 0 : -1;
        if (value) button.append(element('span',`piece ${color}${absolutePiece > 12 ? ' fairy' : ''}`,glyphs[absolutePiece] || '?'));
        if (fileIndex === 0) button.append(element('span','rank-label',String(r+1)));
        if (rankIndex === ranks-1) button.append(element('span','file-label',files[f] || String(f+1)));
        button.addEventListener('click',() => selectSquare(coord));
        squaresEl.append(button);
      }
      card.append(head,squaresEl); cell.append(card); fragment.append(cell);
    }
  }
  const line = element('div','present-line');
  line.dataset.presentTime = presentTime;
  fragment.append(line);
  grid.replaceChildren(fragment);
  viewport.scrollLeft = oldScroll.left; viewport.scrollTop = oldScroll.top;
  positionPresentLine();
  $('board-hint').textContent = selected ? `${coordinateLabel(selected)} · ${choices.length} available move${choices.length === 1 ? '' : 's'}` : 'Click a piece to explore its moves.';
}
function positionPresentLine() {
  const line = $('timeline-grid').querySelector('.present-line');
  if (!line) return;
  const cells = [...$('timeline-grid').querySelectorAll('.board-cell')];
  const cell = cells.find(c => Number(c.dataset.time) === Number(line.dataset.presentTime)) || cells.find(c => Number(c.dataset.time) > Number(line.dataset.presentTime));
  line.style.left = `${cell ? cell.offsetLeft + 3 : 94}px`;
  line.hidden = !cell;
}
function scrollToPresent() {
  const line = $('timeline-grid').querySelector('.present-line');
  if (line) $('multiverse-viewport').scrollTo({left:Math.max(0,parseFloat(line.style.left)-110),behavior:'smooth'});
}
async function selectSquare(coord) {
  if (busy || !game) return;
  const key = coordKey(coord);
  if (selected) {
    const choices = game.moves.filter(move => coordKey(move.raw?.[0]) === coordKey(selected) && coordKey(move.raw?.[1]) === key);
    if (choices.length === 1) { await playMove(choices[0]); return; }
    if (choices.length > 1) {
      $('promotion-choices').replaceChildren(...choices.map(move => {
        const button = element('button','button quiet',readableMove(move));
        button.addEventListener('click',() => { $('promotion-dialog').close(); playMove(move); });
        return button;
      }));
      $('promotion-dialog').showModal();
      return;
    }
  }
  selected = coordKey(selected) === key ? null : game.moves.some(move => coordKey(move.raw?.[0]) === key) ? coord : null;
  renderBoards();
}
async function invalidateSearch() {
  analysisGeneration++;
  clearTimeout(pollTimer); clearTimeout(autoTimer);
  const current = search;
  search = null;
  renderAnalysis();
  if (current?.status === 'running') {
    try { await api(`/api/analysis/${encodeURIComponent(current.id)}/stop`,{}); } catch { /* A completed or expired search no longer needs cancellation. */ }
  }
  renderAnalysis();
}
async function changeGame(path, payload, after) {
  if (busy) return;
  setBusy(true);
  try {
    await invalidateSearch();
    receiveGame(await api(path,payload));
    if (after) after();
  } catch (error) { toast(error.message,true); }
  finally { setBusy(false); scheduleOpponent(); }
}
async function playMove(move) { await changeGame('/api/move',{move:move.raw,revision:game.revision}); }
function compactNumber(value) { return Number.isFinite(value) ? Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(value) : '—'; }
function scoreLabel(value) {
  if (!Number.isFinite(value?.score)) return '—';
  const prefix = value.score > 0 ? '+' : value.score < 0 ? '−' : '';
  return value.scoreType === 'mate' ? `${value.score >= 0 ? '+' : '−'}M${Number.isFinite(value.mateIn) ? Math.abs(value.mateIn) : ''}` : `${prefix}${(Math.abs(value.score) / 100).toFixed(2)}`;
}
function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
function reconcileChildren(parent, children) {
  children.forEach((child, index) => { if (parent.children[index] !== child) parent.insertBefore(child, parent.children[index] || null); });
  while (parent.children.length > children.length) parent.lastElementChild.remove();
}
function renderRankings(result) {
  if (rankingJob !== search?.id) {
    rankingJob = search?.id;
    selectedRankingDepth = null;
    rankingTabs.clear(); rankingRows.clear();
    $('ranking-tabs').replaceChildren(); $('ranking-list').replaceChildren();
    recommendationSignature = ''; variationSignature = '';
    continuationAnimation?.cancel();
  }
  visibleRankings = Array.isArray(result?.rankings) ? result.rankings.filter(level => Number.isInteger(level.depth) && Array.isArray(level.entries)).toSorted((a, b) => a.depth - b.depth) : [];
  $('continuation-rankings').hidden = !visibleRankings.length;
  if (!visibleRankings.length) return;
  const restoreTabFocus = $('ranking-tabs').contains(document.activeElement) && !visibleRankings.some(level => level.depth === Number(document.activeElement.dataset.depth));
  if (!visibleRankings.some(level => level.depth === selectedRankingDepth)) selectedRankingDepth = visibleRankings[0].depth;
  const wantedTabs = visibleRankings.map(level => {
    let tab = rankingTabs.get(level.depth);
    if (!tab) {
      tab = element('button', 'ranking-tab', `Depth ${level.depth}`);
      tab.type = 'button'; tab.id = `ranking-tab-${level.depth}`; tab.dataset.depth = String(level.depth);
      tab.setAttribute('role', 'tab'); tab.setAttribute('aria-controls', 'ranking-panel');
      rankingTabs.set(level.depth, tab);
    }
    const active = level.depth === selectedRankingDepth;
    tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
    return tab;
  });
  reconcileChildren($('ranking-tabs'), wantedTabs);
  for (const [depth] of rankingTabs) if (!visibleRankings.some(level => level.depth === depth)) rankingTabs.delete(depth);
  if (restoreTabFocus) rankingTabs.get(selectedRankingDepth)?.focus({ preventScroll: true });
  $('ranking-panel').setAttribute('aria-labelledby', `ranking-tab-${selectedRankingDepth}`);
  const level = visibleRankings.find(item => item.depth === selectedRankingDepth), entries = level.entries.slice(0, 10);
  const side = level.side === 'black' ? 'Black' : 'White';
  setText($('ranking-summary'), `Showing ${entries.length} of ${level.total ?? level.entries.length} · ${side} to move · scores for White${level.depth > 1 ? ' · Higher-ranked parent moves first' : ''}`);
  const wantedRows = entries.map(entry => {
    const key = `${level.depth}:${entry.id}`;
    let row = rankingRows.get(key);
    if (!row) {
      row = element('li', 'ranking-entry'); row.dataset.entryId = String(entry.id);
      const heading = element('div', 'ranking-row-heading');
      heading.append(element('span', 'ranking-rank'), element('span', 'evaluation-badge'), element('strong', 'ranking-score'));
      const line = element('details', 'ranking-line');
      line.append(element('summary', ''), element('ol', ''));
      row.append(heading, element('p', 'ranking-move'), line);
      rankingRows.set(key, row);
    }
    setText(row.querySelector('.ranking-rank'), `#${entry.rank}`);
    const badge = row.querySelector('.evaluation-badge'), type = entry.evaluationType === 'true' ? 'true' : 'candidate';
    badge.dataset.evaluationType = type; setText(badge, type === 'true' ? 'True' : 'Candidate');
    setText(row.querySelector('.ranking-score'), scoreLabel(entry));
    setText(row.querySelector('.ranking-move'), notation(entry.notation) || 'Submit the current turn');
    const line = Array.isArray(entry.line) ? entry.line.map(notation) : [], signature = JSON.stringify(line), details = row.querySelector('details');
    details.hidden = !line.length;
    setText(details.querySelector('summary'), `Full continuation · ${line.length} ${line.length === 1 ? 'turn' : 'turns'}`);
    if (details.dataset.signature !== signature) {
      details.querySelector('ol').replaceChildren(...line.map(move => element('li', '', move || 'Submit the current turn')));
      details.dataset.signature = signature;
    }
    return row;
  });
  reconcileChildren($('ranking-list'), wantedRows);
  const retained = new Set(visibleRankings.flatMap(item => item.entries.slice(0, 10).map(entry => `${item.depth}:${entry.id}`)));
  for (const [key] of rankingRows) if (!retained.has(key)) rankingRows.delete(key);
}
function selectRankingDepth(depth, focus = false) {
  selectedRankingDepth = depth;
  renderRankings(search?.result || search?.progress);
  if (focus) rankingTabs.get(depth)?.focus({ preventScroll: true });
  rankingTabs.get(depth)?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}
function renderResourceStats(result) {
  const maxNodes = result?.limits?.maxNodes ?? Number($('node-budget').value);
  const cacheMb = result?.limits?.cacheMemoryMb ?? Number($('cache-memory').value);
  const usedMb = Number.isFinite(result?.cacheMemoryBytes) ? (result.cacheMemoryBytes / 1048576).toFixed(1) : null;
  $('stat-node-budget').textContent = `Max nodes: ${compactNumber(maxNodes)}`;
  if (neuralEngine(search?.engine || selectedEngine())) {
    $('stat-cache').textContent = result?.model?.device ? `Model: ${result.model.device}` : 'Model: 800k transformer';
    $('stat-cache').title = 'Neural inference device. GPU memory is bounded by the model configuration.';
    return;
  }
  $('stat-cache').textContent = cacheMb === 0 ? 'Cache: off' : `Cache: ${usedMb === null ? '—' : `≈${usedMb}`} / ${cacheMb} MiB`;
  $('stat-cache').title = 'Estimated retained search-cache memory; not total process RAM or GPU VRAM.';
}
function renderAnalysis() {
  const result = search?.result || search?.progress;
  const running = search?.status === 'running';
  const adaptiveDepth = result?.searchPolicy === 'transformer-adaptive-depth';
  const rankedDepth = adaptiveDepth || result?.searchPolicy === 'transformer-ranked-depth';
  renderRankings(result);
  const liveLeader = running && rankedDepth ? result.rankings?.find(level => level.depth === 1)?.entries?.[0] : null;
  const displayedEvaluation = liveLeader || result;
  const score = displayedEvaluation?.score;
  $('search-status').textContent = running ? 'SEARCHING' : result?.stoppedReason === 'cancelled' || search?.status === 'cancelled' ? 'STOPPED' : search?.status === 'done' ? result?.completed ? 'COMPLETE' : 'PARTIAL' : search?.status === 'error' ? 'ERROR' : engineIdleStatus();
  $('search-status').classList.toggle('searching',running);
  const isMate = displayedEvaluation?.scoreType === 'mate';
  setText($('eval-score'), scoreLabel(displayedEvaluation));
  $('eval-label').textContent = Number.isFinite(score) ? result?.status === 'checkmate' ? `${sideLabel()} is checkmated` : result?.status === 'stalemate' ? 'Stalemate' : isMate ? `${score >= 0 ? 'White' : 'Black'} has a mating line` : Math.abs(score) < 20 ? 'Approximately equal' : `${score > 0 ? 'White' : 'Black'} is favored` : running ? 'Exploring possible continuations…' : search?.result ? 'Evaluation unavailable' : 'Run an analysis to evaluate';
  $('evaluation-fill').style.height = `${Number.isFinite(score) ? 50 + 47 * Math.tanh(score / 600) : 50}%`;
  $('stat-depth').textContent = result?.depth ?? '—';
  $('stat-depth').title = result ? rankedDepth
    ? `Deepest true evaluation: ${result.depth ?? 0} turns. Current best line: ${result.pvDepth ?? result.pv?.length ?? 0} turns. Deepest generated or probed turn: ${result.selectiveDepth ?? result.depth ?? 0}.`
    : `Completed full-turn depth: ${result.depth ?? 0}. Deepest visited turn: ${result.selectiveDepth ?? result.depth ?? 0}. Capture extension depth: ${result.effectiveQuiescenceDepth ?? 0}.`
    : neuralEngine() ? 'Deepest true evaluation in complete turns' : 'Deepest fully completed full-turn search';
  if (result?.depthMode === 'dynamic') $('stat-depth').title += ` Dynamic mode; current depth ceiling: ${result.currentMaxDepth ?? 1} turns.`;
  $('stat-nodes').textContent = compactNumber(result?.nodes);
  $('stat-nodes').title = result ? `${(result.nodes ?? 0).toLocaleString()} search and generation work nodes` : 'Search and generation work nodes';
  $('stat-nps').textContent = compactNumber(result?.nps);
  $('stat-time').textContent = Number.isFinite(result?.elapsedMs) ? `${(result.elapsedMs / 1000).toFixed(1)}s` : '—';
  renderResourceStats(result);
  let note = running && result ? `Searching depth ${result.searchingDepth ?? result.depth} · ${result.rootActionsSearched ?? 0} root turns compared` : !running && result?.stoppedReason === 'policy' ? 'Legal turns exist, but none satisfy the search restriction on optional boards. Play a turn manually.' : !running && result?.stoppedReason === 'nodes' ? 'Node limit reached. Increase Max nodes to search further within your think time.' : !running && search?.result && !result.completed ? Number.isFinite(score) ? 'Partial search; no full depth completed. Allow more think time for a deeper comparison.' : result.bestAction ? 'A legal fallback is available. Allow more think time to evaluate alternatives.' : 'No recommendation is available within the search limits.' : '';
  if (rankedDepth && !running && !result.completed && result.stoppedReason !== 'nodes') {
    note = result.bestAction ? 'A legal fallback is available; no root turn has a true evaluation yet. Allow more think time to compare continuations.' : 'No recommendation is available within the search limits.';
  }
  if (neuralEngine(search?.engine) && result) {
    const name = transformerName;
    const candidateLimit = result.candidateLimit ?? result.limits?.candidateLimit;
    const innerCandidateLimit = result.innerCandidateLimit ?? result.limits?.innerCandidateLimit;
    const alphaBeta = result.searchPolicy === 'transformer-bounded-alpha-beta';
    const legacyBeamWidth = alphaBeta || rankedDepth ? null : result.beamWidth ?? result.limits?.beamWidth;
    const tokenLimit = result.model?.config?.max_tokens;
    const boardLimit = result.model?.config?.max_boards;
    let candidateScope = 'candidate turns are capped';
    if (Number.isFinite(candidateLimit)) {
      candidateScope = Number.isFinite(innerCandidateLimit)
        ? candidateLimit === innerCandidateLimit ? `up to ${candidateLimit} candidate turns per position` : `up to ${candidateLimit} root / ${innerCandidateLimit} reply candidate turns`
        : `up to ${candidateLimit} root candidate turns`;
    }
    const details = [`${name}: ${adaptiveDepth ? 'adaptive depth search' : rankedDepth ? 'ranked depth search' : alphaBeta ? 'alpha-beta search' : 'selective search'}; ${candidateScope}${Number.isFinite(legacyBeamWidth) ? `; best ${legacyBeamWidth} deepened` : ''}.`];
    if (rankedDepth) {
      if (result.depthMode === 'dynamic') details.push(adaptiveDepth
        ? `Dynamic depth; current ceiling: ${result.currentMaxDepth ?? 1} turns. Leading alternatives and their strongest replies are checked before deepening.`
        : `Dynamic depth; current ceiling: ${result.currentMaxDepth ?? 1} turns. The top ${result.dynamicDepthThreshold ?? 20} ranks at every searched depth must be True before the ceiling increases (all ranks if fewer exist).`);
      if (adaptiveDepth) {
        details.push('Promising branches widen gradually. Optional moves are limited to temporal moves.');
        if (result.effectiveQuiescenceDepth) details.push(`Forcing lines extend up to ${result.effectiveQuiescenceDepth} turns beyond the current ceiling.`);
        const leader = result.rankings?.find(level => level.depth === 1)?.entries?.[0];
        if (leader?.generatedReplies) details.push(`Leading move: ${leader.searchedReplies} of ${leader.generatedReplies} generated replies evaluated${leader.repliesExhaustive ? '; all legal replies generated' : '; additional legal replies may exist'}.`);
        details.push(result.model?.policyAvailable ? 'Trained component policy enabled.' : 'Value-guided component ordering.');
      } else if (Number.isFinite(result.expansionRank)) details.push(`Shared evaluated ranks: ${result.expansionRank}.`);
      const depths = (result.depthStats || []).filter(item => item.candidates > 0 || item.trueEvaluations > 0);
      if (depths.length) details.push(`Searched moves by depth: ${depths.map(item => `${item.depth}: ${item.searchedMoves === null ? 'no candidates' : item.searchedMoves}`).join(' · ')}.`);
    }
    if (Number.isFinite(tokenLimit)) details.push(`Model context: ${tokenLimit} tokens.`);
    if (Number.isFinite(boardLimit)) details.push(`Spatial context: up to ${boardLimit} boards.`);
    if (result.contextTruncated) details.push('Historical context was truncated for the model. Full history still determines legality.');
    if (result.frontierTruncated) details.push('The position exceeds model context; some current-board features were omitted.');
    note = [note, ...details].filter(Boolean).join(' ');
  }
  $('analysis-note').textContent = note;
  $('analysis-note').hidden = !note;
  const bestNotation = liveLeader ? notation(liveLeader.notation) || notation(liveLeader.line?.[0]) || 'Submit the current turn' : notation(result?.notation) || (Array.isArray(result?.bestAction) ? result.bestAction.length ? result.bestAction.map(raw => readableMove({raw})).join(' / ') : 'Submit the current turn' : '');
  setText($('continuation-heading'), liveLeader ? 'LEADING CONTINUATION' : 'BEST CONTINUATION');
  setText($('best-move'), bestNotation);
  $('best-move').hidden = !bestNotation;
  $('recommendation-empty').hidden = Boolean(bestNotation);
  $('recommendation-empty').textContent = running ? 'Searching for a complete turn…' : result?.status === 'checkmate' ? 'No safe turn is available.' : result?.status === 'stalemate' ? 'No playable turn is available.' : result?.stoppedReason === 'policy' ? 'No turn satisfies the optional-board search restriction.' : search?.status === 'cancelled' ? 'Analysis stopped.' : 'The next possibility is waiting.';
  const pv = liveLeader ? liveLeader.line || [liveLeader.notation] : result?.pvNotation || [];
  const evaluationType = liveLeader?.evaluationType || (rankedDepth && bestNotation ? result.completed ? 'true' : 'candidate' : null);
  $('best-evaluation-meta').hidden = !evaluationType;
  if (evaluationType) {
    $('best-evaluation').dataset.evaluationType = evaluationType;
    setText($('best-evaluation'), evaluationType === 'true' ? 'True' : 'Candidate');
    setText($('best-evaluation-score'), scoreLabel(displayedEvaluation));
  }
  const nextVariation = JSON.stringify(pv.slice(1).map(notation));
  if (nextVariation !== variationSignature) {
    $('principal-variation').replaceChildren(...pv.slice(1).map(move => element('li','',notation(move))));
    variationSignature = nextVariation;
  }
  const nextRecommendation = JSON.stringify([liveLeader?.id, bestNotation, evaluationType, nextVariation]);
  if (recommendationSignature && nextRecommendation !== recommendationSignature && bestNotation && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    continuationAnimation?.cancel();
    continuationAnimation = $('best-move').animate([{ backgroundColor: '#9af2d72b' }, { backgroundColor: 'transparent' }], { duration: 650 });
  }
  recommendationSignature = nextRecommendation;
  updateControls();
}
async function startAnalysis(autoPlay = false) {
  if (busy || !game || search?.status === 'running' || !engineAvailable()) return;
  if (selectedEngine() === 'classical' && heuristicsPanel && !heuristicsPanel.ready()) return;
  if (!$('node-budget').reportValidity()) { if (autoPlay) autoRevision = null; return; }
  const options = {
    engine: selectedEngine(),
    timeMs: Number($('time-budget').value), maxDepth: Number($('search-depth').value),
    maxNodes: Number($('node-budget').value), cacheMemoryMb: neuralEngine() ? 0 : Number($('cache-memory').value),
    threads: neuralEngine() ? 1 : Number($('search-threads').value),
    ...(selectedEngine() === 'classical' && heuristicsPanel ? {heuristics: heuristicsPanel.configuration()} : {}),
  };
  saveSearchSettings();
  const revision = game.revision;
  const generation = ++analysisGeneration, heuristicsSignature = currentHeuristicsSignature();
  setBusy(true);
  try {
    const data = await api('/api/analyze', options);
    if (generation !== analysisGeneration || revision !== game?.revision || options.engine !== selectedEngine() || heuristicsSignature !== currentHeuristicsSignature()) {
      try { await api(`/api/analysis/${encodeURIComponent(data.jobId)}/stop`,{}); } catch { /* Discarded jobs cannot update this position. */ }
      return;
    }
    search = {id:data.jobId,revision,engine:options.engine,heuristicsSignature,status:'running',result:null,progress:null,autoPlay};
    renderAnalysis();
    pollTimer = setTimeout(pollAnalysis,100);
  } catch (error) { toast(error.message,true); }
  finally { setBusy(false); }
}
async function pollAnalysis() {
  const current = search;
  if (!current || current.status !== 'running' || current.polling) return;
  current.polling = true;
  const pollStarted = performance.now();
  try {
    const data = await api(`/api/analysis/${encodeURIComponent(current.id)}`);
    if (search !== current || !searchMatchesPosition(current)) return;
    current.status = data.status;
    if (data.progress) current.progress = data.progress;
    if (data.result) current.result = data.result;
    renderAnalysis();
    if (data.status === 'error') { toast(data.error || 'The engine could not complete this analysis.',true); }
    else if (data.status === 'done' && current.autoPlay && searchMatchesPosition(current) && current.result?.bestAction && $('ai-side').value === sideLabel().toLowerCase()) await playBest();
  } catch (error) {
    if (search === current) { current.status = 'error'; renderAnalysis(); toast(error.message,true); }
  } finally {
    current.polling = false;
    if (search === current && searchMatchesPosition(current) && current.status === 'running') pollTimer = setTimeout(pollAnalysis, Math.max(0, 100 - (performance.now() - pollStarted)));
  }
}
async function stopAnalysis() {
  if (!search || search.status !== 'running') return;
  const current = search;
  current.autoPlay = false;
  try { await api(`/api/analysis/${encodeURIComponent(current.id)}/stop`,{}); if (search === current && !current.polling) { clearTimeout(pollTimer); pollTimer = setTimeout(pollAnalysis,100); } }
  catch (error) { toast(error.message,true); }
}
async function playBest() {
  if (busy || search?.status === 'running' || !search?.result?.bestAction || !searchMatchesPosition()) return;
  setBusy(true);
  try {
    const next = await api('/api/play',{jobId:search.id,revision:game.revision});
    search = null; renderAnalysis(); receiveGame(next); scrollToPresent();
  } catch (error) { toast(error.message,true); }
  finally { setBusy(false); scheduleOpponent(); }
}
function scheduleOpponent() {
  clearTimeout(autoTimer);
  if (!game || busy || !engineAvailable() || (selectedEngine() === 'classical' && heuristicsPanel && !heuristicsPanel.ready()) || search?.status === 'running' || game.pending?.length || $('ai-side').value !== sideLabel().toLowerCase() || autoRevision === game.revision) return;
  autoRevision = game.revision;
  autoTimer = setTimeout(() => startAnalysis(true),250);
}

$('orientation').addEventListener('change',renderBoards);
$('ranking-tabs').addEventListener('click', event => {
  const tab = event.target.closest('[role="tab"]');
  if (tab) selectRankingDepth(Number(tab.dataset.depth));
});
$('ranking-tabs').addEventListener('keydown', event => {
  const tab = event.target.closest('[role="tab"]');
  if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const index = visibleRankings.findIndex(level => level.depth === Number(tab.dataset.depth));
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? visibleRankings.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + visibleRankings.length) % visibleRankings.length;
  selectRankingDepth(visibleRankings[next].depth, true);
});
for (const id of resourceSettingIds) $(id).addEventListener('change', () => {
  saveSearchSettings();
  renderResourceStats(search?.result || search?.progress);
  scheduleOpponent();
});
$('engine-select').addEventListener('change', async () => {
  autoRevision = null;
  setBusy(true);
  renderEngine();
  saveSearchSettings();
  try { await invalidateSearch(); }
  finally { setBusy(false); scheduleOpponent(); }
});
$('refresh-engines').addEventListener('click', refreshEngines);
$('show-history').addEventListener('change',() => { renderBoards(); scrollToPresent(); });
$('board-size').addEventListener('input',() => { boardSize = Number($('board-size').value); renderBoards(); });
$('submit-button').addEventListener('click',() => { if (game?.canSubmit) changeGame('/api/submit',{revision:game.revision},scrollToPresent); });
$('undo-button').addEventListener('click',async () => { $('ai-side').value = 'off'; autoRevision = null; await changeGame('/api/undo',{},scrollToPresent); });
$('new-button').addEventListener('click',() => { autoRevision = null; changeGame('/api/new',{variant:$('variant').value},scrollToPresent); });
$('analyze-button').addEventListener('click',() => search?.status === 'running' ? stopAnalysis() : startAnalysis());
$('play-button').addEventListener('click',playBest);
$('ai-side').addEventListener('change',() => { autoRevision = null; if (search?.autoPlay) search.autoPlay = false; scheduleOpponent(); });
$('help-button').addEventListener('click',() => $('help-dialog').showModal());
$('notation-button').addEventListener('click',() => { $('pgn-input').value = game?.pgn || ''; $('notation-dialog').showModal(); });
$('copy-pgn').addEventListener('click',async () => {
  try { await navigator.clipboard.writeText($('pgn-input').value); toast('Game notation copied.'); }
  catch { $('pgn-input').focus(); $('pgn-input').select(); toast('Select and copy the notation from the text field.'); }
});
$('download-pgn').addEventListener('click',() => {
  const url = URL.createObjectURL(new Blob([$('pgn-input').value],{type:'text/plain;charset=utf-8'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'vibe-d-ai.pgn'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});
$('import-pgn').addEventListener('click',() => {
  const pgn = $('pgn-input').value.trim();
  if (!pgn) { toast('Paste a game in PGN format first.',true); return; }
  autoRevision = null;
  changeGame('/api/import',{pgn},() => { $('notation-dialog').close(); scrollToPresent(); toast('Game loaded.'); });
});
document.addEventListener('keydown',(event) => {
  if (event.key === 'Escape' && selected) { selected = null; renderBoards(); }
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !document.querySelector('dialog[open]') && game?.canSubmit && !busy) { event.preventDefault(); changeGame('/api/submit',{revision:game.revision},scrollToPresent); }
});
window.addEventListener('resize',positionPresentLine);
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click',(event) => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } });
if (window.innerWidth < 570) { boardSize = 34; $('board-size').value = '34'; }
const {createHeuristicsPanel} = await import('/heuristics.js');
heuristicsPanel = createHeuristicsPanel({
  api,
  boardLabel: board => `${timelineLabel(board.timeline)}L · ${timeLabel(board.turn)} ${board.turn % 2 ? 'Black' : 'White'}`,
  onChange: () => {
    // Keep a late poll or queued automatic turn from applying an old profile.
    autoRevision = game?.revision ?? null;
    void invalidateSearch();
  },
  onReady: () => { updateControls(); scheduleOpponent(); },
});
restoreSearchSettings();
renderEngine();
renderResourceStats();
void heuristicsPanel.load();
refreshEngines();
try { receiveGame(await api('/api/game')); }
catch (error) {
  $('connection').textContent = 'Engine unavailable';
  $('turn-label').textContent = 'Connection unavailable';
  $('turn-detail').textContent = 'Start the local server, then reload this page.';
  $('loading-state').replaceChildren(element('p','',error.message));
  toast(error.message,true);
}
